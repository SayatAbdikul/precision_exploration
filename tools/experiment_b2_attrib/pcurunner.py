"""``pcu`` per-channel arms of lane Q4 (agent r6, protocol addendum 8) through the low-memory path.

    python -m tools.run.experiment_b2_attrib_pcu arms --model M --format F --stage S --arms a,b [--audit]
           [--combo] [--regression-combo-from MODEL/FORMAT]
    python -m tools.run.experiment_b2_attrib_pcu budget NEED_SECONDS

``pcu:TARGET`` is ``pcf:TARGET`` (addendum 2, ``lmrunner.parse_lm``) with the per-channel scales of
``pcunbiased.py`` (an unbiased random sample of search values) instead of ``pcfull.py``.  ``pcu`` and ``pcf`` terms
cannot be mixed.  Records go to the lane's arm folder under their own names (``pcu~...``); the ``pcf`` records stay.

``--combo`` runs, after the listed arms, the addendum-2 combination re-applied to ``pcu``: ``affine:signed`` plus the
best of the four ``pcu`` arms of this (model, format) (highest top-1 expected on the screen, ties by lower mean KL).
``--regression-combo-from mobilenet_v3_large/int8`` runs ``affine:signed+pcu:all`` on this model when ``pcu:all`` is
the best of the four MobileNetV3 INT8 ``pcu`` arms (the addendum-2 regression rule); otherwise it is skipped.

Everything else (gate, arm identity, no-overwrite writes, STOP switch, time cap) follows ``lmrunner.run_arms``;
the job ledger is ``artifacts/experiment_b2_attrib/pcu/logs/jobs.jsonl``; the stop switch
``artifacts/experiment_b2_attrib/pcu/STOP``.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import time

import numpy as np

from tools.experiment_b.common import ROOT, digest, file_hash, unseal
from tools.experiment_b2 import data, readout
from tools.experiment_b2.common import MODELS, source_identity

from . import lmrunner, lowmem, pcunbiased, runner
from .lmrunner import npz_bytes, peak_mib, sealed_bytes, verified_record, write_new

PCU = runner.OUT / "pcu"
STOP = PCU / "STOP"
LEDGER = PCU / "logs" / "jobs.jsonl"
ADDENDUM = ROOT / "public/experiments/configs/breadth-study/b2-attrib-protocol-v1-addendum-8.json"
CAP_SECONDS = 3600.0
FOUR = ("pcu:role.dw_conv", "pcu:role.se_product", "pcu:kind.hardswish", "pcu:all")
SOURCES = ("pcunbiased.py", "pcurunner.py")
START = time.monotonic()


def pcu_sources():
    return {f"tools/experiment_b2_attrib/{n}": file_hash(ROOT / "tools/experiment_b2_attrib" / n) for n in SOURCES}


def parse_pcu(name, plan, groups):
    """``(arm, pcu_nodes)``: a ``pcu`` arm parsed as its ``pcf`` twin and renamed; other names as ``parse_lm``."""
    if "pcu:" not in name:
        return lmrunner.parse_lm(name, plan, groups)
    if "pcf:" in name or name.endswith("+fixbc"):
        raise ValueError(f"arm {name}: pcu cannot be mixed with pcf or fixbc")
    twin, nodes = lmrunner.parse_lm(name.replace("pcu:", "pcf:"), plan, groups)
    return dataclasses.replace(twin, name=name, note="pcu"), nodes


def best_of_four(model, fmt):
    """Best of the four pcu arms on the screen (top-1 expected, ties by lower mean KL), or None if one is missing."""
    found = []
    for name in FOUR:
        path, _ = runner.arm_paths(model, fmt, name)
        if not path.exists():
            return None
        record = unseal(path)
        found.append((record["readout"]["top1_expected_percent"], -record["kl_nats_mean"], name))
    return max(found)[2]


def recorded_seconds():
    if not LEDGER.exists():
        return 0.0
    return sum(float(json.loads(line).get("job_wall_seconds", 0.0)) for line in LEDGER.read_text().splitlines() if line.strip())


def run_arms(model, format_name, names, device, with_audit, stage, combo=False, regression_from=None, max_seconds=780.0):
    from .engine import build_arm
    from .groups import partition
    verified = verified_record(model, format_name)
    tick = time.monotonic()
    peak_mib(reset=True)
    setup, shared, bias_inputs, info = runner.shared_state(model, format_name, device)
    if shared.fingerprint() != verified["fingerprint"]:
        raise ValueError("shared state differs from the verified one")
    groups = partition(model, shared.fp_graph, shared.plan)
    reference = runner.fp32_logits(setup["graph"], setup["inputs"], device)
    setup_seconds, setup_peak = time.monotonic() - tick, peak_mib(reset=True)
    protocol = {"path": str(runner.PROTOCOL_FILE.relative_to(ROOT)), "sha256": file_hash(runner.PROTOCOL_FILE),
                "addendum_8": {"path": str(ADDENDUM.relative_to(ROOT)), "sha256": file_hash(ADDENDUM)}}
    results, arm_seconds, pcu_cache, v1_cache = [], [], {}, shared.channel_cache
    queue, selections = list(names), {}

    def extend():
        """Arms chosen by a selection rule are appended once their inputs exist (after the listed arms)."""
        if combo and "combo" not in selections:
            best = best_of_four(model, format_name)
            selections["combo"] = {"rule": "affine:signed + best of the four pcu arms (top-1, ties by lower KL)",
                                   "best": best}
            if best is not None:
                queue.append(f"affine:signed+{best}")
        if regression_from and "regression" not in selections:
            source_model, source_format = regression_from.split("/")
            best = best_of_four(source_model, source_format)
            selections["regression"] = {"rule": "affine:signed+pcu:all when pcu:all is the best of the four pcu arms "
                                                "of " + regression_from, "best_there": best}
            if best == "pcu:all":
                queue.append("affine:signed+pcu:all")

    i = 0
    while True:
        if i == len(queue):
            extend()
            if i == len(queue):
                break
        name = queue[i]
        i += 1
        if STOP.exists():
            results.append({"arm": name, "status": "stopped (STOP file)"})
            continue
        elapsed = time.monotonic() - START
        if arm_seconds and elapsed + 1.3 * max(arm_seconds) > max_seconds:
            results.append({"arm": name, "status": "deferred (time cap)"})
            continue
        arm, pcu_nodes = parse_pcu(name, shared.plan, groups)
        if arm.note == "fixbc":
            raise ValueError("fixbc arms run through lmrunner only")
        json_path, npz_path = runner.arm_paths(model, format_name, name)
        key = {"base_configuration": verified["configuration_sha256"], "fingerprint": verified["fingerprint"],
               "arm": arm.spec(), "own_sources": runner.own_sources(), "audit": with_audit}
        if pcu_nodes and arm.note == "pcu":
            key["pcu"] = {"spec": pcunbiased.SPEC,
                          "source_sha256": file_hash(ROOT / "tools/experiment_b2_attrib/pcunbiased.py")}
        elif pcu_nodes:
            raise ValueError(f"{name}: pcf arms run through lmrunner only")
        identity = digest(key)
        if json_path.exists():
            same = unseal(json_path)["arm_identity"] == identity
            results.append({"arm": name, "status": "exists" if same else "exists (different identity; not overwritten)"})
            print(json.dumps(results[-1]), flush=True)
            continue
        start = time.monotonic()
        peak_mib(reset=True)
        if pcu_nodes:
            missing = [n for n in pcu_nodes if n not in pcu_cache]
            if missing:
                pcu_cache.update(pcunbiased.channel_cache(shared, missing, bias_inputs))
            shared.channel_cache = pcu_cache
        else:
            shared.channel_cache = v1_cache
        engine, meta = build_arm(shared, arm, bias_inputs)
        build_seconds, build_peak = time.monotonic() - start, peak_mib(reset=True)
        arrays, sha, kl, seconds = runner.evaluate(engine, setup["inputs"], setup["rows"], device, reference)
        occupancy = runner.audit(engine, setup["inputs"], device) if with_audit else None
        evaluation_peak = peak_mib(reset=True)
        del engine
        if json_path.exists() or npz_path.exists():
            results.append({"arm": name, "status": "written meanwhile by another job; mine discarded"})
            continue
        if not write_new(npz_path, npz_bytes(arrays, kl)):
            results.append({"arm": name, "status": "readout written meanwhile by another job; mine discarded"})
            continue
        summary = readout.summary(arrays)
        record = {"model": model, "format": format_name, "base_recipe": "default", "arm_name": name,
                  "arm_identity": identity, "arm": meta["arm"], "arm_kind": arm.kind, "meta": meta,
                  "base_configuration_sha256": verified["configuration_sha256"], "fingerprint": verified["fingerprint"],
                  "readout": summary, "readout_file": str(npz_path.relative_to(ROOT)),
                  "readout_file_sha256": file_hash(npz_path), "logits_sha256": sha,
                  "kl_nats_mean": float(kl.mean()), "kl_nats_median": float(np.median(kl)),
                  "rows_sha256": digest(setup["rows"]), "occupancy": occupancy, "stage": stage,
                  "evidence": "development: imagenet_screen_1k (frozen 1k screen); audit on its first 128 images",
                  "protocol": protocol, "source_sha256": source_identity(), "own_sources": runner.own_sources(),
                  "pcu": key.get("pcu"), "selections": selections or None,
                  "lowmem": {"sources": lmrunner.lm_sources(), "pcu_sources": pcu_sources(), "offload": lowmem.OFFLOAD,
                             "verification": "old path" if runner.verification_path(model, format_name).exists() and
                             unseal(runner.verification_path(model, format_name))["passed"] else "lowmem gate",
                             "bias_correction_peak_mib": build_peak, "evaluation_peak_mib": evaluation_peak},
                  "recorded_at": runner.now(), "cost": {"build_seconds": build_seconds, "evaluation_seconds": seconds,
                                                        "arm_wall_seconds": time.monotonic() - start, "device": device,
                                                        "note": "shared GPU slot; not a throughput measurement"}, **info}
        if not write_new(json_path, sealed_bytes(record)):
            results.append({"arm": name, "status": "record written meanwhile by another job; mine discarded"})
            continue
        arm_seconds.append(time.monotonic() - start)
        per_channel = meta.get("per_channel") or {}
        results.append({"arm": name, "status": "computed", "top1_expected": round(summary["top1_expected_percent"], 2),
                        "kl": round(float(kl.mean()), 4), "seconds": round(arm_seconds[-1], 1),
                        "peak_mib": max(build_peak or 0, evaluation_peak or 0),
                        "min_ratio_to_maxabs_scale": min((v.get("min_ratio_to_maxabs_scale", 1.0) for v in per_channel.values()),
                                                         default=None)})
        print(json.dumps(results[-1]), flush=True)
    return {"model": model, "format": format_name, "setup_seconds": setup_seconds, "setup_peak_mib": setup_peak,
            "arms": results, "selections": selections, "job_wall_seconds": time.monotonic() - START,
            "lowmem_stats": dict(lowmem.STATS)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("arms")
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--format", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--arms", default="")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--combo", action="store_true")
    p.add_argument("--regression-combo-from", default=None)
    p.add_argument("--stage", required=True)
    p.add_argument("--max-seconds", type=float, default=780.0)
    b = sub.add_parser("budget")
    b.add_argument("need", type=float)
    args = parser.parse_args(argv)
    if args.command == "budget":
        used = recorded_seconds()
        print(f"addendum-8 jobs: recorded {used:.0f} s of {CAP_SECONDS:.0f} s; next job estimated {args.need:.0f} s")
        return 0 if used + args.need <= CAP_SECONDS else 1
    if not ADDENDUM.exists():
        raise SystemExit("protocol addendum 8 must be on disk before any pcu arm is measured")
    from tools.experiment_b2 import frozen  # noqa: F401  (registers the frozen recipes)
    data.cached_inputs = functools.partial(data.cached_inputs, build=False)
    lowmem.install()
    names = [n for n in args.arms.split(",") if n]
    if not names and not (args.combo or args.regression_combo_from):
        raise SystemExit("no arms given")
    result = run_arms(args.model, args.format, names, args.device, args.audit, args.stage, args.combo,
                      args.regression_combo_from, args.max_seconds)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as stream:
        stream.write(json.dumps({"command": "arms", "stage": args.stage, "recorded_at": runner.now(), **result},
                                sort_keys=True, default=str) + "\n")
    done = sum(r["status"] in ("computed", "exists") for r in result["arms"])
    print(json.dumps({"done_or_existing": done, "of": len(result["arms"]), "selections": result["selections"],
                      "job_wall_seconds": round(result["job_wall_seconds"], 1)}), flush=True)
    return 0 if done == len(result["arms"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
