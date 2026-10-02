"""Low-memory jobs of lane Q4 (agent r3): the gate and the arms of ``runner.py`` through ``lowmem.bias_correct``.

    python -m tools.run.experiment_b2_attrib_lm gate --model M --format F
    python -m tools.run.experiment_b2_attrib_lm arms --model M --format F --stage S --arms a,b [--audit] [--audit-arms a]

``gate`` reproduces the sealed B2 default matrix cell through the low-memory path without ``runner.build``: the
sealed configuration file (its digest is the cell's configuration identity) must give the same activation and
weight scales and the same bias-correction report as the ``ref_default`` arm, and that arm's readout arrays and
logits sha256 must equal the sealed ones.  Records go to ``artifacts/experiment_b2_attrib/lowmem/verification/``.

``arms`` writes the very records of ``runner.run_arms`` (same arm identity, so either path skips what the other
wrote), plus a ``lowmem`` block.  Differences from ``runner.run_arms``: an arm that an old-path job of this lane
is computing at that moment is deferred; the existence of a record is checked right before each arm and right
before writing; records are written to a temporary file and linked into place (never overwriting); the job stops
before the next arm when ``artifacts/experiment_b2_attrib/lowmem/STOP`` exists or ``--max-seconds`` would be
exceeded; the GPU peak memory of every arm is logged.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import time

import numpy as np

from tools.experiment_b.common import ROOT, digest, file_hash, unseal
from tools.experiment_b2 import data, readout
from tools.experiment_b2.common import BASE, MODELS, source_identity

from . import lowmem, pcfull, runner
from .arms import parse_arm

LM = runner.OUT / "lowmem"
STOP = LM / "STOP"
LM_SOURCES = ("lowmem.py", "lmrunner.py", "pcfull.py")
START = time.monotonic()


def lm_sources():
    return {f"tools/experiment_b2_attrib/{n}": file_hash(ROOT / "tools/experiment_b2_attrib" / n) for n in LM_SOURCES}


def peak_mib(reset=False):
    import torch
    if not torch.cuda.is_available():
        return None
    value = torch.cuda.max_memory_allocated() / 2 ** 20
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return round(value, 1)


def write_new(path, payload_bytes):
    """Write ``payload_bytes`` to ``path`` unless it exists (temporary file + hard link; never overwrites)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.lmtmp")
    temporary.write_bytes(payload_bytes)
    try:
        os.link(temporary, path)
        return True
    except FileExistsError:
        return False
    finally:
        temporary.unlink()


def sealed_bytes(payload):
    return json.dumps({"payload": payload, "sha256": digest(payload)}, indent=1, sort_keys=True).encode()


def npz_bytes(arrays, kl):
    import io
    stream = io.BytesIO()
    np.savez_compressed(stream, kl_nats=kl.astype(np.float32), **{k: arrays[k] for k in readout.DTYPES})
    return stream.getvalue()


def gate_path(model, format_name):
    return LM / "verification" / f"{model}--{format_name}--default.json"


def verified_record(model, format_name):
    """The passed verification of the old path or of the low-memory gate (same configuration and fingerprint)."""
    found = []
    for path in (runner.verification_path(model, format_name), gate_path(model, format_name)):
        if path.exists():
            record = unseal(path)
            if record["passed"]:
                found.append(record)
    if not found:
        raise SystemExit(f"no passed verification for {model}/{format_name}: run the gate job first")
    keys = {(r["configuration_sha256"], r["fingerprint"]) for r in found}
    if len(keys) != 1:
        raise ValueError("old-path and low-memory verifications disagree")
    return found[0]


def gate(model, format_name, device):
    import torch
    from .engine import Arm, build_arm
    path = gate_path(model, format_name)
    if path.exists():
        return {**unseal(path), "status": "exists"}
    start = time.monotonic()
    peak_mib(reset=True)
    cell_path, cell = runner.sealed_cell(model, format_name)
    configuration = unseal(BASE / "configurations" / f"{cell['configuration_sha256']}.json")
    sealed_arrays = readout.load(ROOT / cell["readout_file"])
    setup, shared, bias_inputs, info = runner.shared_state(model, format_name, device)
    setup_peak = peak_mib(reset=True)
    tick = time.monotonic()
    engine, meta = build_arm(shared, Arm("ref_default", kind="reference"), bias_inputs)
    build_seconds, build_peak = time.monotonic() - tick, peak_mib(reset=True)
    arrays, sha, _, evaluation_seconds = runner.evaluate(engine, setup["inputs"], setup["rows"], device, None)
    scales = configuration["scales"]
    checks = {
        "sealed_configuration_digest_is_cell_identity": digest(configuration) == cell["configuration_sha256"],
        "activation_scales_equal_sealed": shared.activation_scales == scales["activation_scales"],
        "weight_scales_equal_sealed": shared.weight_scales == scales["weight_scales"],
        "bias_correction_report_equals_sealed": meta["bias_correction_digest"] == digest(scales["bias_correction"]),
        "lowmem_readout_equals_sealed": all(np.array_equal(arrays[k], sealed_arrays[k]) for k in readout.DTYPES),
        "lowmem_logits_sha256_equals_sealed": sha == cell["logits_sha256"],
    }
    record = {"model": model, "format": format_name, "recipe_name": "default", "images": runner.IMAGES,
              "path": "lowmem (tools/experiment_b2_attrib/lowmem.py)", "sealed_cell": str(cell_path.relative_to(ROOT)),
              "configuration_sha256": cell["configuration_sha256"], "sealed_readout_file": cell["readout_file"],
              "sealed_logits_sha256": cell["logits_sha256"], "checks": checks, "passed": all(checks.values()),
              "fingerprint": shared.fingerprint(), "readout": readout.summary(arrays),
              "source_sha256": source_identity(), "own_sources": runner.own_sources(), "lowmem_sources": lm_sources(),
              "protocol": {"path": str(runner.PROTOCOL_FILE.relative_to(ROOT)), "sha256": file_hash(runner.PROTOCOL_FILE)},
              "memory": {"offload": lowmem.OFFLOAD, "setup_peak_mib": setup_peak, "bias_correction_peak_mib": build_peak,
                         "evaluation_peak_mib": peak_mib(), "torch_max_memory_allocated_note":
                         "torch.cuda.max_memory_allocated per phase; the CUDA context comes on top"},
              "cost": {"build_seconds": build_seconds, "evaluation_seconds": evaluation_seconds, "lowmem": dict(lowmem.STATS)},
              "wall_seconds": time.monotonic() - start, "device": device, "recorded_at": runner.now(), **info}
    write_new(path, sealed_bytes(record))
    return {**record, "status": "computed"}


def old_path_jobs():
    """(model, format, arms) of every running old-path arms job of this lane (python processes only)."""
    jobs = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if "-m" not in argv or not argv[0].endswith("python"):
            continue
        i = argv.index("-m")
        if argv[i + 1:i + 3] != ["tools.run.experiment_b2_attrib", "arms"]:
            continue
        rest = argv[i + 3:]
        value = {rest[j]: rest[j + 1] for j in range(len(rest) - 1) if rest[j].startswith("--")}
        names = [n for n in value.get("--arms", "").split(",") if n]
        if value.get("--arms-file"):
            try:
                names += [s.strip() for s in open(ROOT / value["--arms-file"]) if s.strip() and not s.startswith("#")]
            except OSError:
                pass
        jobs.append((value.get("--model"), value.get("--format"), set(names)))
    return jobs


def busy_elsewhere(model, format_name, name):
    return any(m == model and f == format_name and name in arms for m, f, arms in old_path_jobs())


def parse_lm(name, plan, groups):
    """``parse_arm`` plus the ``pcf:TARGET`` term (addendum 2): per-channel scales fitted by ``pcfull``.

    A boundary named by a ``pcf`` term is removed from the arm's affine set (``pcf`` wins on overlap).
    Returns ``(arm, pcf_nodes)``; ``pcf_nodes`` is empty for every v1 arm name.
    """
    import dataclasses
    from .arms import resolve
    if name.endswith("+fixbc"):
        arm, pcf = parse_lm(name[:-len("+fixbc")], plan, groups)
        if pcf or arm.affine or arm.per_channel:
            raise ValueError("fixbc is defined for wide/only/reference arms only")
        return dataclasses.replace(arm, name=name, kind="diagnostic", note="fixbc"), ()
    if "pcf:" not in name:
        return parse_arm(name, plan, groups), ()
    terms = name.split("+")
    pcf = set()
    for term in terms:
        if term.startswith("pcf:"):
            pcf |= {n for n in resolve(term[4:], plan, groups) if plan[n]["kind"] not in ("input", "linear")}
    if not pcf:
        raise ValueError(f"arm {name}: no eligible boundary in the pcf target")
    rest = [t for t in terms if not t.startswith("pcf:")]
    base = parse_arm("+".join(rest), plan, groups) if rest else None
    wide = set(base.wide) if base else set()
    affine = (set(base.affine) if base else set()) - pcf
    if base is not None and base.per_channel:
        raise ValueError(f"arm {name}: pc and pcf terms cannot be mixed")
    if wide & pcf:
        raise ValueError(f"arm {name} changes the code of a boundary it keeps wide")
    from .engine import Arm
    arm = Arm(name, wide=tuple(sorted(wide)), affine=tuple(sorted(affine)), per_channel=tuple(sorted(pcf)),
              kind="diagnostic" if wide else "recipe", note="pcf")
    return arm, tuple(sorted(pcf))


FIXBC = {"version": "fixbc_v1", "rule": "conv/linear biases are the default arm's corrected biases (bias correction "
         "fitted once for the frozen default plan) instead of a refit for the arm's own plan"}


def build_fixbc(shared, arm, bias_inputs, fixed):
    """Arm engine whose biases are the default arm's corrected biases (addendum 3 diagnostic)."""
    from . import engine as attrib_engine
    from .engine import Arm, build_arm
    if not fixed:
        reference, meta = build_arm(shared, Arm("ref_default", kind="reference"), bias_inputs)
        fixed["biases"] = {name: m.bias.detach().clone() for name, m in reference.module.named_modules()
                           if getattr(m, "bias", None) is not None and hasattr(m, "weight")}
        fixed["report"] = meta["bias_correction_digest"]
        del reference

    def copy_biases(fp_graph, q_graph, plan, quantizers, activation_scales, inputs, chunk=32):
        report = {}
        for name, module in q_graph.named_modules():
            if name in fixed["biases"]:
                module.bias.copy_(fixed["biases"][name])
                report[name] = {"max_abs_correction": 0.0, "fixbc": True}
        return report

    original = attrib_engine.bias_correct
    attrib_engine.bias_correct = copy_biases
    try:
        engine, meta = build_arm(shared, arm, bias_inputs)
    finally:
        attrib_engine.bias_correct = original
    meta["fixbc"] = {**FIXBC, "default_bias_correction_digest": fixed["report"]}
    return engine, meta


def run_arms(model, format_name, names, device, with_audit, stage, audit_arms=(), max_seconds=780.0):
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
    protocol = {"path": str(runner.PROTOCOL_FILE.relative_to(ROOT)), "sha256": file_hash(runner.PROTOCOL_FILE)}
    results, arm_seconds = [], []
    v1_cache, pcf_cache, fixed = shared.channel_cache, {}, {}
    for name in names:
        if STOP.exists():
            results.append({"arm": name, "status": "stopped (STOP file)"})
            continue
        elapsed = time.monotonic() - START
        if arm_seconds and elapsed + 1.3 * max(arm_seconds) > max_seconds:
            results.append({"arm": name, "status": "deferred (time cap)"})
            continue
        arm, pcf_nodes = parse_lm(name, shared.plan, groups)
        with_audit_arm = with_audit or name in audit_arms
        json_path, npz_path = runner.arm_paths(model, format_name, name)
        key = {"base_configuration": verified["configuration_sha256"], "fingerprint": verified["fingerprint"],
               "arm": arm.spec(), "own_sources": runner.own_sources(), "audit": with_audit_arm}
        if pcf_nodes:
            key["pcf"] = {"spec": pcfull.SPEC, "source_sha256": file_hash(ROOT / "tools/experiment_b2_attrib/pcfull.py")}
        if arm.note == "fixbc":
            key["fixbc"] = FIXBC
        identity = digest(key)
        if json_path.exists():
            same = unseal(json_path)["arm_identity"] == identity
            results.append({"arm": name, "status": "exists" if same else "exists (different identity; not overwritten)"})
            continue
        if busy_elsewhere(model, format_name, name):
            results.append({"arm": name, "status": "deferred (an old-path job is computing it)"})
            print(json.dumps(results[-1]), flush=True)
            continue
        start = time.monotonic()
        peak_mib(reset=True)
        if pcf_nodes:
            missing = [n for n in pcf_nodes if n not in pcf_cache]
            if missing:
                pcf_cache.update(pcfull.channel_cache(shared, missing, bias_inputs))
            shared.channel_cache = pcf_cache
        else:
            shared.channel_cache = v1_cache
        if arm.note == "fixbc":
            engine, meta = build_fixbc(shared, arm, bias_inputs, fixed)
        else:
            engine, meta = build_arm(shared, arm, bias_inputs)
        build_seconds, build_peak = time.monotonic() - start, peak_mib(reset=True)
        arrays, sha, kl, seconds = runner.evaluate(engine, setup["inputs"], setup["rows"], device, reference)
        occupancy = runner.audit(engine, setup["inputs"], device) if with_audit_arm else None
        evaluation_peak = peak_mib(reset=True)
        del engine
        if json_path.exists() or npz_path.exists():
            results.append({"arm": name, "status": "written meanwhile by another job; mine discarded"})
            print(json.dumps(results[-1]), flush=True)
            continue
        blob = npz_bytes(arrays, kl)
        if not write_new(npz_path, blob):
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
                  "pcf": key.get("pcf"),
                  "lowmem": {"sources": lm_sources(), "offload": lowmem.OFFLOAD,
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
        results.append({"arm": name, "status": "computed", "top1_expected": round(summary["top1_expected_percent"], 2),
                        "kl": round(float(kl.mean()), 4), "seconds": round(arm_seconds[-1], 1),
                        "peak_mib": max(build_peak or 0, evaluation_peak or 0)})
        print(json.dumps(results[-1]), flush=True)
    return {"model": model, "format": format_name, "setup_seconds": setup_seconds, "setup_peak_mib": setup_peak,
            "arms": results, "job_wall_seconds": time.monotonic() - START, "lowmem_stats": dict(lowmem.STATS)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("gate", "arms"):
        p = sub.add_parser(command)
        p.add_argument("--model", choices=MODELS, required=True)
        p.add_argument("--format", required=True)
        p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        if command == "arms":
            p.add_argument("--arms", default="")
            p.add_argument("--arms-file", default=None)
            p.add_argument("--audit", action="store_true")
            p.add_argument("--audit-arms", default="")
            p.add_argument("--stage", required=True)
            p.add_argument("--max-seconds", type=float, default=780.0)
    args = parser.parse_args(argv)
    from tools.experiment_b2 import frozen  # noqa: F401  (registers the frozen recipes)
    data.cached_inputs = functools.partial(data.cached_inputs, build=False)
    if not runner.PROTOCOL_FILE.exists():
        raise SystemExit("the attribution protocol must be on disk before anything is measured")
    lowmem.install()
    log = LM / "logs" / "jobs.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "gate":
        record = gate(args.model, args.format, args.device)
        line = {k: record[k] for k in ("model", "format", "passed", "checks", "status")}
        line.update(memory=record.get("memory"), wall_seconds=record.get("wall_seconds"))
        print(json.dumps(line), flush=True)
        with log.open("a") as stream:
            stream.write(json.dumps({"command": "gate", "recorded_at": runner.now(), **line}, sort_keys=True) + "\n")
        return 0 if record["passed"] else 1
    names = [n for n in args.arms.split(",") if n]
    if args.arms_file:
        names += [line.strip() for line in open(args.arms_file) if line.strip() and not line.startswith("#")]
    if not names:
        raise SystemExit("no arms given")
    result = run_arms(args.model, args.format, names, args.device, args.audit, args.stage,
                      tuple(n for n in args.audit_arms.split(",") if n), args.max_seconds)
    with log.open("a") as stream:
        stream.write(json.dumps({"command": "arms", "stage": args.stage, "recorded_at": runner.now(), **result},
                                sort_keys=True) + "\n")
    done = sum(r["status"] in ("computed", "exists") for r in result["arms"])
    print(json.dumps({"done_or_existing": done, "of": len(names), "job_wall_seconds": round(result["job_wall_seconds"], 1)}),
          flush=True)
    return 0 if done == len(names) else 3
