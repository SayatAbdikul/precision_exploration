"""Lane Q8 runner: AXE / naive / OPTQ / RTN arms on the B2 simulator (development evidence, 1k screen).

    GPU_LANE=Q8 artifacts/agent_orchestration/gpu_run.sh --min-free-mib 4000 \
        .venv-b/bin/python -m tools.run.experiment_b2_axe run --case resnet18-int8 --arms rtn,optq,axe-P20,naive-P20
    exit 0: all requested arms recorded; 3: stopped by --budget (run again; finished arms are skipped)

Records (write-once, sealed): artifacts/experiment_b2_axe/evals-v2/<case>/<arm>.{json,npz}. The first launch's folder
evals/ (12:05-12:15, 2026-10-02) is superseded: its naive arms rounded a float64 weight quotient, which differs from
B2's float32 nearest rounding on near-ties, so naive at lambda=0 was not exactly the B2 default.
Protocol: public/experiments/configs/breadth-study/b2-axe-protocol-v1.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import sys
import time

from pathlib import Path

import numpy as np

from tools.experiment_b.common import ROOT, digest, file_hash, seal, unseal

BASE = ROOT / "artifacts/experiment_b2_axe"
PROTOCOL = ROOT / "public/experiments/configs/breadth-study/b2-axe-protocol-v1.json"
OWN_SOURCES = ("tools/experiment_b2_axe/__init__.py", "tools/experiment_b2_axe/axe.py",
               "tools/experiment_b2_axe/build.py", "tools/experiment_b2_axe/check.py", "tools/run/experiment_b2_axe.py")
READ_ONLY_SOURCES = ("tools/experiment_b2_recon/engine.py", "tools/experiment_b2_recon/evaluate.py",
                     "tools/scaled_bridge_v2/certificates.py", "tools/scaled_bridge_v2/export.py")


def parse_arm(arm):
    if arm in ("rtn", "optq"):
        return arm, None
    method, p = arm.split("-P")
    if method not in ("axe", "naive"):
        raise ValueError(arm)
    return method, int(p)


def sources():
    return {path: file_hash(ROOT / path) for path in OWN_SOURCES + READ_ONLY_SOURCES}


def run(args):
    import torch
    from tools.experiment_b2.common import PROTOCOL as B2_PROTOCOL, source_identity
    from tools.experiment_b2_recon.evaluate import sealed_b2_predictions, write_arrays
    from tools.experiment_b2_axe import axe
    from tools.experiment_b2_axe.build import CALIBRATION_IMAGES, CHECK_IMAGES, CASES, Setup, codes_digest
    from tools.experiment_b2_axe.check import engine_certificate, prefix_stats

    started = time.monotonic()
    folder = (Path(args.out) if args.out else BASE / "evals-v2") / args.case
    folder.mkdir(parents=True, exist_ok=True)
    arms = [a for a in args.arms.split(",") if a]
    missing = [a for a in arms if not (folder / f"{a}.json").exists()]
    if not missing:
        print(json.dumps({"case": args.case, "status": "all arms recorded"}), flush=True)
        return 0
    spec = CASES[args.case]
    setup = Setup(args.case, args.device)
    if args.images:
        if not args.out:
            raise SystemExit("--images is for smoke tests and needs --out")
        setup.inputs, setup.rows = setup.inputs[:args.images], setup.rows[:args.images]
    batch = B2_PROTOCOL["inference_batch_size"]
    protocol_sha = file_hash(PROTOCOL)
    common = {"protocol": "b2-axe-protocol-v1", "protocol_sha256": protocol_sha, "case": args.case, "case_spec": spec,
              "b2_source_sha256": source_identity(), "sources": sources(), "runtime": setup.setup["env"],
              "calibration_images": {"list": "imagenet_calibration_2k", "first": CALIBRATION_IMAGES,
                                     "rows_sha256": digest(setup.calib_rows)},
              "activation_scales_sha256": digest(setup.act_scales),
              "weight_scales_sha256": digest(setup.scales), "evaluation_list": "imagenet_screen_1k",
              "rows_sha256": digest(setup.rows), "images": len(setup.rows),
              "evidence": "development_evidence_screen1k"}
    save = set(a for a in (args.save_codes or "").split(",") if a)
    for arm in missing:
        if time.monotonic() - started > args.budget:
            print(json.dumps({"case": args.case, "status": "budget reached"}), flush=True)
            return 3
        tick = time.monotonic()
        method, P = parse_arm(arm)
        if method == "rtn":
            codes, infos = setup.rtn_codes(), {}
        else:
            codes, infos = setup.fit(method, P)
        fit_seconds = time.monotonic() - tick
        identity = codes_digest(codes)
        record = {**common, "arm": arm, "method": method, "P": P, "weight_codes_sha256": identity,
                  "layers": infos, "fit_seconds": fit_seconds}
        betas = {name: [int(x) for x in axe.sign_sums(q)[0].cpu()] for name, q in codes.items()}
        negs = {name: [int(x) for x in axe.sign_sums(q)[1].cpu()] for name, q in codes.items()}
        record["max_beta"] = max(max(v) for v in betas.values())
        record["max_neg"] = max(max(v) for v in negs.values())
        record["sign_sums_sha256"] = digest({"beta": betas, "neg": negs})
        if P is not None:
            record["integer_limit"] = axe.integer_limit(P, spec["N"])
            record["bound_holds"] = all(axe.bound_holds(q, P, spec["N"]) for q in codes.values())
        if spec["l8_case"]:
            certificate, original = engine_certificate(spec["l8_case"], codes)
            record["certificate"] = certificate
            if method == "rtn":
                record["rtn_equals_adapted_export_weight_units"] = all(
                    np.array_equal(codes[k].cpu().numpy().reshape(v.shape), v) for k, v in original.items())
        reference = {"axe": "optq", "naive": "rtn"}.get(method)
        if reference and (folder / f"{reference}.json").exists():
            ref = unseal(folder / f"{reference}.json")
            if ref["weight_codes_sha256"] == identity:
                record.update({"identical_to": reference, "readout": ref["readout"],
                               "prefix_check": ref.get("prefix_check"), "seconds": time.monotonic() - tick})
                seal(folder / f"{arm}.json", record)
                print(json.dumps({"case": args.case, "arm": arm, "identical_to": reference}), flush=True)
                continue
        result, bias_report = setup.deploy(codes)
        ties, per_image, top5 = setup.screen(result, batch)
        record["readout"] = ties
        record["bias_correction_max_abs"] = (max(v["max_abs_correction"] for v in bias_report.values())
                                             if bias_report else None)
        if method == "rtn":
            sealed_id, sealed = sealed_b2_predictions(spec["model"], spec["weight_format"], spec["recipe"], setup.rows)
            record["sealed_b2_check"] = {"b2_configuration_sha256": sealed_id,
                                         "top5_lists_reproduced": None if sealed is None
                                         else sum(a == b for a, b in zip(sealed, top5))}
        if not args.no_check:
            record["prefix_check"] = prefix_stats(setup, result, codes, CHECK_IMAGES)
            if P is not None:
                record["prefix_check"]["holds_at_P"] = (record["prefix_check"]["prefix_bits"] <= P
                                                        and record["prefix_check"]["orderfree_bits"] <= P)
        if arm in save:
            from tools.experiment_b2_recon.evaluate import write_arrays as write
            dtype = np.int8
            write(folder / f"{arm}.codes.npz", {k: v.cpu().numpy().astype(dtype) for k, v in codes.items()})
            record["codes_file"] = f"{arm}.codes.npz"
        record["seconds"] = time.monotonic() - tick
        write_arrays(folder / f"{arm}.npz", per_image)
        seal(folder / f"{arm}.json", record)
        print(json.dumps({"case": args.case, "arm": arm, "expected": round(ties["top1_percent_expected"], 2),
                          "lowest": round(ties["top1_percent_lowest_index"], 2),
                          "cert_struct": record.get("certificate", {}).get("signed_bits_structural"),
                          "prefix_bits": record.get("prefix_check", {}).get("prefix_bits"),
                          "sealed": record.get("sealed_b2_check"), "seconds": round(record["seconds"], 1)}), flush=True)
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
    p.add_argument("--save-codes", default="")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--out", default="", help="smoke tests only: records go to this folder instead of the lane's")
    p.add_argument("--images", type=int, default=0, help="smoke tests only: evaluate the first N screen images")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
