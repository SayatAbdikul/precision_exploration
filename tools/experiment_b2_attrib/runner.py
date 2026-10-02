"""Jobs of lane Q4: reproduce the sealed B2 default cell, then run attribution and repair arms.

Every output goes under ``artifacts/experiment_b2_attrib/``; nothing of ``artifacts/experiment_b2/`` is
written (the input cache is used with ``build=False``).  Evidence is development evidence on the frozen
ImageNet 1k screen; calibration uses the frozen v1 observations and the first 256 calibration images only.

    python -m tools.run.experiment_b2_attrib verify --model M --format F
    python -m tools.run.experiment_b2_attrib arms --model M --format F --arms A1,A2,... [--audit] [--stage S]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import hashlib
import json
import time
from types import SimpleNamespace

import numpy as np

from tools.experiment_b.common import ROOT, digest, file_hash, seal, unseal
from tools.experiment_b2 import data, readout
from tools.experiment_b2.common import BIAS_CORRECTION_IMAGES, MODELS, PROTOCOL, source_identity

from .arms import parse_arm

OUT = ROOT / "artifacts/experiment_b2_attrib"
PROTOCOL_FILE = ROOT / "public/experiments/configs/breadth-study/b2-attrib-protocol-v1.json"
MATRIX = ROOT / "artifacts/experiment_b2/matrix"
IMAGES = 1000
AUDIT_IMAGES = 128
OWN = ("__init__.py", "groups.py", "quant.py", "engine.py", "arms.py", "runner.py")
START = time.monotonic()


def now():
    return datetime.now(timezone.utc).isoformat()


def own_sources():
    return {f"tools/experiment_b2_attrib/{n}": file_hash(ROOT / "tools/experiment_b2_attrib" / n) for n in OWN}


def sealed_cell(model, format_name, recipe_name="default"):
    paths = sorted((MATRIX / "cells").glob(f"{model}--{format_name}--{recipe_name}--{IMAGES}--*.json"))
    if len(paths) != 1:
        raise ValueError(f"expected one sealed matrix cell for {model}/{format_name}/{recipe_name}, found {len(paths)}")
    return paths[0], unseal(paths[0])


def safe(name):
    return name.replace(":", "~").replace("/", "_")


def fp32_logits(graph, inputs, device):
    import torch
    batch = PROTOCOL["inference_batch_size"]
    parts = []
    with torch.inference_mode():
        for start in range(0, IMAGES, batch):
            tensor = torch.from_numpy(np.array(inputs[start:start + batch], dtype=np.float32)).to(device)
            parts.append(graph(tensor).float())
    return torch.cat(parts)


def evaluate(engine, inputs, rows, device, reference_logits):
    """Batch-8 evaluation: readout arrays, logits digest, per-image KL(FP32 || arm) in nats, seconds."""
    import torch
    batch = PROTOCOL["inference_batch_size"]
    labels = torch.tensor([int(row["label"]) for row in rows[:IMAGES]], device=device)
    parts, kl, sha, tick = [], [], hashlib.sha256(), time.monotonic()
    for start in range(0, IMAGES, batch):
        tensor = torch.from_numpy(np.array(inputs[start:start + batch], dtype=np.float32)).to(device)
        with torch.inference_mode():
            output = engine.run(tensor) if hasattr(engine, "run") else engine(tensor)
            if tuple(output.shape) != (len(tensor), 1000) or output.dtype != torch.float32:
                raise ValueError("invalid classifier output")
            parts.append(readout.batch_readout(output, labels[start:start + len(tensor)]))
            sha.update(output.cpu().numpy().tobytes())
            if reference_logits is not None:
                p = torch.log_softmax(reference_logits[start:start + len(tensor)].double(), dim=1)
                q = torch.log_softmax(output.double(), dim=1)
                kl.append(((p.exp() * (p - q)).sum(dim=1)).cpu().numpy())
    arrays = readout.concatenate(parts)
    return arrays, sha.hexdigest(), (np.concatenate(kl) if kl else None), time.monotonic() - tick


def context(model, device):
    """Frozen model, inputs, v1 observations and the bias-correction images (no cache is built)."""
    import torch
    from tools.experiment_b2 import runner
    from tools.experiment_b2.guards import check_graph
    setup = runner.model_setup(model, device)
    check_graph(setup["graph"])
    arrays, maxima, calibration = data.v1_calibration(model)
    if calibration["graph"] != str(setup["graph"].graph):
        raise ValueError("v1 calibration graph differs from the loaded graph")
    bias_inputs, bias_rows = data.cached_inputs("imagenet_calibration_2k", BIAS_CORRECTION_IMAGES, setup["transform"],
                                                build=False)
    probe = torch.from_numpy(np.array(setup["inputs"][:1], dtype=np.float32)).to(device)
    return setup, arrays, maxima, calibration, bias_inputs, {"images": len(bias_rows), "rows_sha256": digest(bias_rows)}, probe


def shared_state(model, format_name, device):
    from tools.experiment_b2.recipe import named
    from .engine import prepare_shared
    setup, arrays, maxima, calibration, bias_inputs, bias_identity, probe = context(model, device)
    tick = time.monotonic()
    shared = prepare_shared(model, setup["graph"], format_name, named("default"), arrays, maxima, device, probe)
    return setup, shared, bias_inputs, {"calibration": calibration["identity"], "bias_inputs": bias_identity,
                                        "preparation_seconds": time.monotonic() - tick}


def verification_path(model, format_name):
    return OUT / "verification" / f"{model}--{format_name}--default.json"


def verify(model, format_name, device):
    """Reproduce the sealed B2 default matrix cell twice: through ``runner.build`` and through the arm engine."""
    import torch
    from tools.experiment_b2 import runner
    from .engine import Arm, build_arm
    path = verification_path(model, format_name)
    if path.exists():
        return {**unseal(path), "status": "exists"}
    start = time.monotonic()
    cell_path, cell = sealed_cell(model, format_name)
    sealed_arrays = readout.load(ROOT / cell["readout_file"])
    args = SimpleNamespace(model=model, format=format_name, recipe="default", device=device, images=IMAGES)
    _, _, reference, meta, _, identity = runner.build(args, seal_configuration=False)
    setup, shared, bias_inputs, info = shared_state(model, format_name, device)
    ref_arrays, ref_sha, _, _ = evaluate(reference, setup["inputs"], setup["rows"], device, None)
    engine, arm_meta = build_arm(shared, Arm("ref_default", kind="reference"), bias_inputs)
    arrays, sha, _, _ = evaluate(engine, setup["inputs"], setup["rows"], device, None)
    params_equal = all(torch.equal(a, b) for a, b in zip(reference.module.state_dict().values(),
                                                          engine.module.state_dict().values()))
    checks = {
        "configuration_identity_reproduced": identity == cell["configuration_sha256"],
        "runner_build_readout_equals_sealed": all(np.array_equal(ref_arrays[k], sealed_arrays[k]) for k in readout.DTYPES),
        "runner_build_logits_sha256_equals_sealed": ref_sha == cell["logits_sha256"],
        "activation_scales_equal": shared.activation_scales == meta["activation_scales"],
        "weight_scales_equal": shared.weight_scales == meta["weight_scales"],
        "parameters_bit_identical_after_bias_correction": bool(params_equal) and
            list(reference.module.state_dict()) == list(engine.module.state_dict()),
        "arm_engine_readout_equals_sealed": all(np.array_equal(arrays[k], sealed_arrays[k]) for k in readout.DTYPES),
        "arm_engine_logits_sha256_equals_sealed": sha == cell["logits_sha256"],
    }
    record = {"model": model, "format": format_name, "recipe_name": "default", "images": IMAGES,
              "sealed_cell": str(cell_path.relative_to(ROOT)), "configuration_sha256": cell["configuration_sha256"],
              "sealed_readout_file": cell["readout_file"], "sealed_logits_sha256": cell["logits_sha256"],
              "checks": checks, "passed": all(checks.values()), "fingerprint": shared.fingerprint(),
              "readout": readout.summary(arrays), "source_sha256": source_identity(), "own_sources": own_sources(),
              "protocol": {"path": str(PROTOCOL_FILE.relative_to(ROOT)), "sha256": file_hash(PROTOCOL_FILE)},
              "wall_seconds": time.monotonic() - start, "device": device, "recorded_at": now(), **info}
    seal(path, record)
    return {**record, "status": "computed"}


def arm_paths(model, format_name, name):
    folder = OUT / "arms" / f"{model}--{format_name}"
    return folder / f"{safe(name)}.json", folder / f"{safe(name)}.npz"


def save_arrays(path, arrays, kl):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, kl_nats=kl.astype(np.float32), **{k: arrays[k] for k in readout.DTYPES})
    temporary.replace(path)


def load_arm(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def audit(engine, inputs, device):
    import torch
    from tools.experiment_b2.engine import B2Interpreter, audit_summary
    from tools.experiment_b2.matrix import occupancy_summary
    from tools.experiment_b2.matrix_finite import finite_rows
    probe = B2Interpreter(engine.module, engine.quantizers, engine.plan,
                          {k: float(v) for k, v in engine.scales.items()}, audit=True)
    with torch.inference_mode():
        for start in range(0, AUDIT_IMAGES, 8):
            probe.run(torch.from_numpy(np.array(inputs[start:start + 8], dtype=np.float32)).to(device))
    rows = finite_rows(audit_summary(probe))
    return {"images": AUDIT_IMAGES, "summary": occupancy_summary(rows),
            "sqnr_db": {k: v.get("sqnr_db") for k, v in rows.items()}}


def run_arms(model, format_name, names, device, with_audit, stage, audit_arms=()):
    from .engine import build_arm
    from .groups import partition
    check = verification_path(model, format_name)
    if not check.exists() or not unseal(check)["passed"]:
        raise SystemExit(f"run the verify job for {model}/{format_name} first (it must pass)")
    verified = unseal(check)
    tick = time.monotonic()
    setup, shared, bias_inputs, info = shared_state(model, format_name, device)
    if shared.fingerprint() != verified["fingerprint"]:
        raise ValueError("shared state differs from the verified one")
    groups = partition(model, shared.fp_graph, shared.plan)
    reference = fp32_logits(setup["graph"], setup["inputs"], device)
    setup_seconds = time.monotonic() - tick
    protocol = {"path": str(PROTOCOL_FILE.relative_to(ROOT)), "sha256": file_hash(PROTOCOL_FILE)}
    results = []
    for name in names:
        arm = parse_arm(name, shared.plan, groups)
        with_audit_arm = with_audit or name in audit_arms
        json_path, npz_path = arm_paths(model, format_name, name)
        identity = digest({"base_configuration": verified["configuration_sha256"], "fingerprint": verified["fingerprint"],
                           "arm": arm.spec(), "own_sources": own_sources(), "audit": with_audit_arm})
        if json_path.exists() and unseal(json_path)["arm_identity"] == identity:
            results.append({"arm": name, "status": "exists"})
            continue
        start = time.monotonic()
        engine, meta = build_arm(shared, arm, bias_inputs)
        build_seconds = time.monotonic() - start
        arrays, sha, kl, seconds = evaluate(engine, setup["inputs"], setup["rows"], device, reference)
        occupancy = audit(engine, setup["inputs"], device) if with_audit_arm else None
        save_arrays(npz_path, arrays, kl)
        summary = readout.summary(arrays)
        record = {"model": model, "format": format_name, "base_recipe": "default", "arm_name": name,
                  "arm_identity": identity, "arm": meta["arm"], "arm_kind": arm.kind, "meta": meta,
                  "base_configuration_sha256": verified["configuration_sha256"], "fingerprint": verified["fingerprint"],
                  "readout": summary, "readout_file": str(npz_path.relative_to(ROOT)),
                  "readout_file_sha256": file_hash(npz_path), "logits_sha256": sha,
                  "kl_nats_mean": float(kl.mean()), "kl_nats_median": float(np.median(kl)),
                  "rows_sha256": digest(setup["rows"]), "occupancy": occupancy, "stage": stage,
                  "evidence": "development: imagenet_screen_1k (frozen 1k screen); audit on its first 128 images",
                  "protocol": protocol, "source_sha256": source_identity(), "own_sources": own_sources(),
                  "recorded_at": now(), "cost": {"build_seconds": build_seconds, "evaluation_seconds": seconds,
                                                 "arm_wall_seconds": time.monotonic() - start, "device": device,
                                                 "note": "shared GPU slot; not a throughput measurement"}, **info}
        seal(json_path, record)
        results.append({"arm": name, "status": "computed", "top1_expected": round(summary["top1_expected_percent"], 2),
                        "kl": round(float(kl.mean()), 4), "seconds": round(time.monotonic() - start, 1)})
        print(json.dumps(results[-1]), flush=True)
    return {"model": model, "format": format_name, "setup_seconds": setup_seconds, "arms": results,
            "job_wall_seconds": time.monotonic() - START}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "arms"):
        p = sub.add_parser(command)
        p.add_argument("--model", choices=MODELS, required=True)
        p.add_argument("--format", required=True)
        p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        if command == "arms":
            p.add_argument("--arms", default="", help="comma-separated arm names (grammar in arms.py)")
            p.add_argument("--arms-file", default=None, help="file with one arm name per line")
            p.add_argument("--audit", action="store_true", help="occupancy/SQNR audit for every arm")
            p.add_argument("--audit-arms", default="", help="comma-separated arms that get the audit")
            p.add_argument("--stage", required=True)
    args = parser.parse_args(argv)
    from tools.experiment_b2 import frozen  # noqa: F401  (registers the frozen recipes)
    # Never enlarge the shared B2 input cache (rule: build=False).
    data.cached_inputs = functools.partial(data.cached_inputs, build=False)
    if not PROTOCOL_FILE.exists():
        raise SystemExit("the attribution protocol must be on disk before anything is measured")
    if args.command == "verify":
        record = verify(args.model, args.format, args.device)
        print(json.dumps({k: record[k] for k in ("model", "format", "passed", "checks", "status")}), flush=True)
        return 0 if record["passed"] else 1
    names = [n for n in args.arms.split(",") if n]
    if args.arms_file:
        names += [line.strip() for line in open(args.arms_file) if line.strip() and not line.startswith("#")]
    if not names:
        raise SystemExit("no arms given")
    result = run_arms(args.model, args.format, names, args.device, args.audit, args.stage,
                      tuple(n for n in args.audit_arms.split(",") if n))
    log = OUT / "logs" / "jobs.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(json.dumps({"stage": args.stage, "recorded_at": now(), **{k: v for k, v in result.items()}},
                                sort_keys=True) + "\n")
    print(json.dumps({"done": len(result["arms"]), "job_wall_seconds": round(result["job_wall_seconds"], 1)}), flush=True)
    return 0
