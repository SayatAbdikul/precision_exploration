"""GPU commands of the detector breadth study (lane Q6).  Run through the shared queue:

    GPU_LANE=Q6 artifacts/agent_orchestration/gpu_run.sh --min-free-mib 5000 \\
        .venv-b/bin/python -m tools.run.experiment_b2_det_breadth gate
    ... formats --formats int5 int4            (part A: default, then conformant / nearest unless at chance)
    ... seeds --formats int8 --subsets 0 1 2 3 4 --recipes default conformant   (part B)
    ... groups --format int6 --groups stem neck (part C, one group wide at a time)
    ... fine --format int6 --group neck --units model_12 model_15             (part C, finer stage)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from tools.experiment_b.common import digest, file_hash, seal
from tools.experiment_b2_det import runner
from tools.experiment_b2_det.stats import annotations

from . import core

L7_INT8_DEFAULT = "1834351b1f740296e04f455a7853e654fdca14c92cd9810b574c96fb7a348a7f"


def gate(args):
    """Reproduce L7's INT8 `default` through the three wrapper paths before any new measurement."""
    session, truth = core.BreadthSession(args.device), annotations()
    target = core.BASE / "reproduction" / "int8-default-gate.json"
    _, configuration, plain = session.build_arm("int8", "default")
    session.use_subset("all", list(range(250)))
    full_samples_equal = (set(session.samples) == set(session.full_samples)
                          and all(np.array_equal(session.samples[k], session.full_samples[k]) for k in session.samples)
                          and session.maxima == session.full_maxima)
    engine_subset, _, via_subset = session.build_arm("int8", "default")
    session.use_subset(None)
    with core.patched_analyze(session.graph, frozenset()):
        _, _, via_patch = session.build_arm("int8", "default")
    results, rows, seconds = session.evaluate(engine_subset, core.IMAGES)
    stored = runner.load_detections(runner.detection_file(L7_INT8_DEFAULT, core.IMAGES))
    identical = all(np.array_equal(stored[k], results["index"][k]) for k in stored) and set(stored) == set(results["index"])
    metrics = runner.point_metrics(results["index"], rows, truth)
    record = {"l7_configuration": L7_INT8_DEFAULT,
              "identity_plain_path": plain, "identity_all_batches_subset_path": via_subset,
              "identity_empty_attribution_path": via_patch,
              "all_identities_equal_l7": plain == via_subset == via_patch == L7_INT8_DEFAULT,
              "all_batch_samples_and_maxima_equal_full_set": bool(full_samples_equal),
              "detections_bit_identical_to_l7_screen1k": bool(identical),
              "l7_detection_file_sha256": file_hash(runner.detection_file(L7_INT8_DEFAULT, core.IMAGES)),
              "map50_95": metrics[0], "evaluation_seconds": seconds, "runtime": session.runtime,
              "finished_at": runner.now(), "protocol": core.PROTOCOL}
    record["passed"] = record["all_identities_equal_l7"] and record["all_batch_samples_and_maxima_equal_full_set"] and identical
    if not target.exists():
        seal(target, record)
    print(json.dumps({k: v for k, v in record.items() if k != "runtime"}), flush=True)
    return 0 if record["passed"] else 1


def formats(args):
    """Part A: `default` (six tie orders unless at chance), then the conformant arm (or nearest) with ties."""
    session, truth = core.BreadthSession(args.device), annotations()
    for fmt in args.formats:
        if fmt not in core.PART_A_FORMATS:
            raise SystemExit(f"not a part-A format: {fmt}")
        base = core.measure(session, truth, fmt, "default", "default", ties="unless_chance", part="A")
        if base["at_chance"]:
            print(json.dumps({"format": fmt, "stop_rule": "default below 1.0 mAP50-95: conformant and ties skipped"}), flush=True)
            continue
        arm = core.conformant_arm(fmt)
        core.measure(session, truth, fmt, arm, arm, ties="always", part="A",
                     extra={"rule_conformance": "conformant" if arm == "conformant" else "nearest_available_joins_not_stored"})
    return 0


def seeds(args):
    """Part B: one configuration per format, recipe and calibration subset (fixed tie rule)."""
    session, truth = core.BreadthSession(args.device), annotations()
    partition = core.subset_batches()
    for k in args.subsets:
        name = f"s{k}"
        session.use_subset(name, partition[k])
        for fmt in args.formats:
            for recipe in args.recipes:
                arm = core.conformant_arm(fmt) if recipe == "conformant" else recipe
                core.measure(session, truth, fmt, arm, f"{arm}@{name}", part="B",
                             extra={"subset": name, "subset_images": 8 * len(partition[k]),
                                    "calibration_subset_digest": digest(partition[k])})
    session.use_subset(None)
    return 0


def groups(args):
    """Part C: one protocol group kept wide at a time around `default` (diagnostic only)."""
    session, truth = core.BreadthSession(args.device), annotations()
    for group in args.groups:
        if group in ("weights_only", "activations_only"):
            core.measure(session, truth, args.format, f"default_{group}", f"default_{group}", part="C",
                         extra={"attribution_group": group})
            continue
        wide = core.group_nodes(session.graph, group)
        core.measure(session, truth, args.format, "default", f"default+wide:{group}", wide=wide, group=group, part="C",
                     extra={"attribution_group": group})
    return 0


def fine(args):
    """Part C, finer stage: one unit of the worst group kept wide at a time."""
    session, truth = core.BreadthSession(args.device), annotations()
    units = core.fine_units(session.graph, args.group)
    for unit in args.units:
        core.measure(session, truth, args.format, "default", f"default+wide:{args.group}/{unit}", wide=units[unit],
                     group=f"{args.group}/{unit}", part="C-fine", extra={"attribution_group": args.group, "unit": unit})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    job = commands.add_parser("gate")
    job.add_argument("--device", default="cuda")
    job = commands.add_parser("formats")
    job.add_argument("--formats", nargs="+", required=True)
    job.add_argument("--device", default="cuda")
    job = commands.add_parser("seeds")
    job.add_argument("--formats", nargs="+", required=True)
    job.add_argument("--subsets", nargs="+", type=int, required=True)
    job.add_argument("--recipes", nargs="+", default=["default", "conformant"], choices=("default", "conformant"))
    job.add_argument("--device", default="cuda")
    job = commands.add_parser("groups")
    job.add_argument("--format", required=True, choices=core.ATTRIBUTION_FORMATS)
    job.add_argument("--groups", nargs="+", required=True,
                     choices=core.GROUP_ORDER + ("weights_only", "activations_only"))
    job.add_argument("--device", default="cuda")
    job = commands.add_parser("fine")
    job.add_argument("--format", required=True, choices=core.ATTRIBUTION_FORMATS)
    job.add_argument("--group", required=True, choices=core.GROUP_ORDER)
    job.add_argument("--units", nargs="+", required=True)
    job.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    return {"gate": gate, "formats": formats, "seeds": seeds, "groups": groups, "fine": fine}[args.command](args)
