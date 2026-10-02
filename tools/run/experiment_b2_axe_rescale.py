"""Lane Q8 addendum-2 runner: rescale-P / axers-P arms (accumulator-aware per-channel scale; exploratory, post hoc).

    GPU_LANE=Q8 artifacts/agent_orchestration/gpu_run.sh --min-free-mib 5000 \
        .venv-b/bin/python -m tools.run.experiment_b2_axe_rescale run --case resnet18-int8 --arms rescale-P22,axers-P22
    exit 0: all requested arms recorded; 3: stopped by --budget (run again; finished arms are skipped)

Records (write-once, sealed): artifacts/experiment_b2_axe/evals-a2/<case>/<arm>.{json,npz}.
Protocol: public/experiments/configs/breadth-study/b2-axe-protocol-v1-addendum-2.json (amends b2-axe-protocol-v1).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import fcntl
import json
import sys
import time

from pathlib import Path

from tools.experiment_b.common import ROOT, digest, file_hash, seal
from tools.run.experiment_b2_axe import OWN_SOURCES, READ_ONLY_SOURCES

BASE = ROOT / "artifacts/experiment_b2_axe"
PROTOCOL = ROOT / "public/experiments/configs/breadth-study/b2-axe-protocol-v1.json"
ADDENDUM = ROOT / "public/experiments/configs/breadth-study/b2-axe-protocol-v1-addendum-2.json"
NEW_SOURCES = ("tools/experiment_b2_axe/rescale.py", "tools/run/experiment_b2_axe_rescale.py",
               "tools/experiment_b2_axe/lean.py")


def parse_arm(arm):
    method, p = arm.split("-P")
    if method not in ("rescale", "axers"):
        raise ValueError(arm)
    return method, int(p)


def sources():
    return {path: file_hash(ROOT / path) for path in OWN_SOURCES + NEW_SOURCES + READ_ONLY_SOURCES}


def run(args):
    import torch
    from tools.experiment_b2.common import PROTOCOL as B2_PROTOCOL, source_identity
    from tools.experiment_b2_recon.evaluate import write_arrays
    from tools.experiment_b2_axe import axe
    from tools.experiment_b2_axe.build import CALIBRATION_IMAGES, CHECK_IMAGES, codes_digest
    from tools.experiment_b2_axe.check import engine_certificate, prefix_stats
    from tools.experiment_b2_axe import build, lean
    from tools.experiment_b2_axe.rescale import ScaledSetup, factor_summary, factors_digest

    build.bias_correct = lean.bias_correct_lean  # this process only: bit-identical, activations parked on the host

    started = time.monotonic()
    folder = (Path(args.out) if args.out else BASE / "evals-a2") / args.case
    folder.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    for arm in arms:
        parse_arm(arm)
    missing = [a for a in arms if not (folder / f"{a}.json").exists()]
    if not missing:
        print(json.dumps({"case": args.case, "status": "all arms recorded"}), flush=True)
        return 0
    setup = ScaledSetup(args.case, args.device)
    spec = setup.spec
    if args.images:
        if not args.out:
            raise SystemExit("--images is for smoke tests and needs --out")
        setup.inputs, setup.rows = setup.inputs[:args.images], setup.rows[:args.images]
    batch = B2_PROTOCOL["inference_batch_size"]
    common = {"protocol": "b2-axe-protocol-v1-addendum-2", "protocol_sha256": file_hash(PROTOCOL),
              "addendum_sha256": file_hash(ADDENDUM), "case": args.case, "case_spec": spec,
              "b2_source_sha256": source_identity(), "sources": sources(), "runtime": setup.setup["env"],
              "calibration_images": {"list": "imagenet_calibration_2k", "first": CALIBRATION_IMAGES,
                                     "rows_sha256": digest(setup.calib_rows)},
              "activation_scales_sha256": digest(setup.act_scales),
              "base_weight_scales_sha256": digest(setup.scales), "evaluation_list": "imagenet_screen_1k",
              "rows_sha256": digest(setup.rows), "images": len(setup.rows),
              "evidence": "development_evidence_screen1k", "status": "exploratory, post hoc (addendum-2)",
              "bias_correction_path": "lean (tools/experiment_b2_axe/lean.py, bit-identical to B2's)"}
    for arm in missing:
        if time.monotonic() - started > args.budget:
            print(json.dumps({"case": args.case, "status": "budget reached"}), flush=True)
            return 3
        # one writer per arm even if two chains of this lane overlap: a non-blocking lock, then re-check the record
        lock = open(folder / f".{arm}.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            print(json.dumps({"case": args.case, "arm": arm, "status": "another process holds it; skipped"}), flush=True)
            continue
        if (folder / f"{arm}.json").exists():
            lock.close()
            continue
        tick = time.monotonic()
        method, P = parse_arm(arm)
        factors, codes = setup.rescale(P)
        infos = {}
        if method == "axers":
            setup.use_factors(factors)
            codes, infos = setup.fit("axe", P)
        else:
            setup.use_factors(factors)
        fit_seconds = time.monotonic() - tick
        record = {**common, "arm": arm, "method": method, "P": P, "weight_codes_sha256": codes_digest(codes),
                  "factors_sha256": factors_digest(factors),
                  "factors": {name: factor_summary(k) for name, k in factors.items()},
                  "layers": infos, "fit_seconds": fit_seconds}
        betas = {name: [int(x) for x in axe.sign_sums(q)[0].cpu()] for name, q in codes.items()}
        negs = {name: [int(x) for x in axe.sign_sums(q)[1].cpu()] for name, q in codes.items()}
        record["max_beta"] = max(max(v) for v in betas.values())
        record["max_neg"] = max(max(v) for v in negs.values())
        record["sign_sums_sha256"] = digest({"beta": betas, "neg": negs})
        record["integer_limit"] = axe.integer_limit(P, spec["N"])
        record["bound_holds"] = all(axe.bound_holds(q, P, spec["N"]) for q in codes.values())
        if not record["bound_holds"]:
            raise AssertionError(f"{arm}: bound violated")
        if spec["l8_case"]:
            certificate, _ = engine_certificate(spec["l8_case"], codes)
            record["certificate"] = certificate
        result, bias_report = setup.deploy(codes)
        ties, per_image, _ = setup.screen(result, batch)
        record["readout"] = ties
        record["bias_correction_max_abs"] = (max(v["max_abs_correction"] for v in bias_report.values())
                                             if bias_report else None)
        if not args.no_check:
            record["prefix_check"] = prefix_stats(setup, result, codes, CHECK_IMAGES)
            record["prefix_check"]["holds_at_P"] = (record["prefix_check"]["prefix_bits"] <= P
                                                    and record["prefix_check"]["orderfree_bits"] <= P)
        setup.reset()
        record["seconds"] = time.monotonic() - tick
        write_arrays(folder / f"{arm}.npz", per_image)
        seal(folder / f"{arm}.json", record)
        (folder / f".{arm}.lock").unlink(missing_ok=True)
        lock.close()
        print(json.dumps({"case": args.case, "arm": arm, "expected": round(ties["top1_percent_expected"], 2),
                          "lowest": round(ties["top1_percent_lowest_index"], 2),
                          "cert_struct": record.get("certificate", {}).get("signed_bits_structural"),
                          "prefix_bits": record.get("prefix_check", {}).get("prefix_bits"),
                          "max_k": max(v["max_k"] for v in record["factors"].values()),
                          "seconds": round(record["seconds"], 1)}), flush=True)
        del result
        torch.cuda.empty_cache() if args.device == "cuda" else None
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--case", required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--budget", type=float, default=780.0)
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--out", default="", help="smoke tests only: records go to this folder instead of the lane's")
    p.add_argument("--images", type=int, default=0, help="smoke tests only: evaluate the first N screen images")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
