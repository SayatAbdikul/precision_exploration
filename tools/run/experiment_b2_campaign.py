"""Queue B2 jobs under the shared GPU lock (resumable; skips recorded runs).

Every configuration runs in its own process.  One lock acquisition covers a batch of such
processes capped by --budget seconds of started work (default 240 s), because single INT8
arms take about 8 s and per-arm lock handovers behind multi-minute jobs of other lanes made
the queue wait dominate.  Use --budget 0 for strictly one configuration per acquisition.

Usage: .venv-b/bin/python -m tools.run.experiment_b2_campaign STAGE [--models ...] [--dry-run]
Stages: regress, diagnosis, diagnosis1k, cle, exit, vendor, sentinel, lowbit, audit, export.
"""
import argparse
import json
import subprocess
import sys
import time

from tools.experiment_b.common import ROOT, unseal
from tools.experiment_b2 import frozen
from tools.experiment_b2.common import BASE, MODELS, source_identity
from tools.experiment_b2.recipe import CUMULATIVE, ONE_AT_A_TIME

LOCK = ROOT / "artifacts/agent_orchestration/gpu.lock"
PYTHON = str(ROOT / ".venv-b/bin/python")
SENTINELS = ("int8", "int6", "int4", "fp8_e4m3fn", "fp7_e3m3", "fp6_e2m3", "log8", "posit8_es1", "nf4")
EQUALIZED = tuple(name for name in CUMULATIVE if name.endswith("_cle"))
LADDER = ("v1_maxabs", *(n for n in CUMULATIVE if n not in EQUALIZED), "v1_percentile_99_9", *ONE_AT_A_TIME)


def recorded(stage, model, format_name, recipe, images, *, any_source=False):
    source = source_identity()
    for path in (BASE / "runs" / stage).glob(f"{model}--{format_name}--{recipe}--{images}--*.json"):
        record = unseal(path)
        if any_source or record.get("source_sha256") == source:
            return True
    return False


def jobs(stage, models):
    default = frozen.FROZEN.get("default")
    for model in models:
        if stage == "regress":
            for recipe in ("v1_maxabs", "v1_percentile_99_9"):
                for name in ("int8", "posit8_es1"):
                    done = (BASE / "regression" / f"{model}--{name}--{recipe}--1000.json").exists()
                    yield done, True, ["regress", "--model", model, "--format", name, "--recipe", recipe, "--images", "1000"]
        elif stage in ("diagnosis", "diagnosis1k", "cle"):
            images = "1000" if stage == "diagnosis1k" else "512"
            arms = EQUALIZED if stage == "cle" else LADDER
            for recipe in arms:
                yield (recorded("diagnosis", model, "int8", recipe, images), True,
                       ["run", "--model", model, "--format", "int8", "--recipe", recipe, "--images", images,
                        "--stage", "diagnosis"])
        elif stage == "exit":
            for recipe in frozen.FROZEN:
                yield (recorded("exit", model, "int8", recipe, "1000"), True,
                       ["run", "--model", model, "--format", "int8", "--recipe", recipe, "--images", "1000", "--stage", "exit"])
            for recipe in ("v1_maxabs", "v1_percentile_99_9"):
                yield (recorded("exit", model, "int8", recipe, "1000"), True,
                       ["run", "--model", model, "--format", "int8", "--recipe", recipe, "--images", "1000", "--stage", "exit"])
        elif stage == "vendor":
            for qconfig in ("x86_default", "x86_fullrange", "qnnpack_fullrange"):
                yield (recorded("vendor", model, "int8", f"vendor_{qconfig}", "1000", any_source=True), False,
                       ["vendor", "--model", model, "--qconfig", qconfig, "--images", "1000"])
        elif stage == "sentinel":
            for name in SENTINELS:
                yield (recorded("sentinel", model, name, "default", "1000"), True,
                       ["run", "--model", model, "--format", name, "--recipe", "default", "--images", "1000",
                        "--stage", "sentinel"])
        elif stage == "lowbit":
            for name in ("int6", "int4", "fp6_e2m3", "posit8_es1"):
                for recipe in ("default_signed", "default_no_bias_correction", "default_weight_maxabs", "minimal"):
                    if recipe == "default_signed" and not name.startswith("int"):
                        continue
                    yield (recorded("lowbit", model, name, recipe, "1000"), True,
                           ["run", "--model", model, "--format", name, "--recipe", recipe, "--images", "1000",
                            "--stage", "lowbit"])
        elif stage == "audit":
            for recipe in ("v1_maxabs", "default"):
                for name in ("posit8_es1", "int8", "fp8_e4m3fn"):
                    done = any((BASE / "occupancy").glob(f"{model}--{name}--{recipe}--128--*.json"))
                    yield done, True, ["run", "--model", model, "--format", name, "--recipe", recipe, "--images", "128",
                                       "--stage", "audit", "--audit"]
        elif stage == "export":
            yield False, True, ["export", "--model", model, "--format", "int8", "--recipe", "default"]
        else:
            raise SystemExit(f"unknown stage {stage}")
    if default is None and stage in ("exit", "sentinel", "lowbit", "audit", "export"):
        raise SystemExit("frozen default recipe is not registered yet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inner", action="store_true", help="already under the GPU lock: run pending jobs until the budget")
    parser.add_argument("--budget", type=float, default=240.0,
                        help="seconds of work started per lock acquisition (each arm is its own process)")
    args = parser.parse_args()
    if not args.inner and not args.dry_run and args.stage != "vendor":
        # Outer loop: one short lock hold per batch of pending single-configuration processes.
        failures = 0
        while any(not done for done, gpu, _ in jobs(args.stage, args.models)):
            pending = sum(not done for done, _, _ in jobs(args.stage, args.models))
            result = subprocess.run(["flock", "-w", "3600", str(LOCK), PYTHON, "-m", "tools.run.experiment_b2_campaign",
                                     args.stage, "--inner", "--budget", str(args.budget), "--models", *args.models], cwd=ROOT)
            after = sum(not done for done, _, _ in jobs(args.stage, args.models))
            if result.returncode != 0 or (after >= pending and args.stage != "export"):
                failures += 1
                break
            if args.stage == "export":
                break
        print(f"CAMPAIGN {args.stage} COMPLETE failures={failures}", flush=True)
        return 1 if failures else 0
    log_dir = BASE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ledger = log_dir / f"campaign-{args.stage}.jsonl"
    failures = 0
    started = time.time()
    for done, gpu, command in list(jobs(args.stage, args.models)):
        if done:
            continue
        if args.inner and time.time() - started > args.budget:
            break
        full = [PYTHON, "-m", "tools.run.experiment_b2", *command]
        if args.dry_run:
            print(" ".join(command))
            continue
        tick = time.time()
        with (log_dir / f"campaign-{args.stage}.err").open("a") as err:
            result = subprocess.run(full, cwd=ROOT, stdout=subprocess.PIPE, stderr=err, text=True)
        last = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        entry = {"command": command, "returncode": result.returncode, "launched_unix": tick,
                 "wall_seconds_including_lock_wait": time.time() - tick, "output": last}
        with ledger.open("a") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
        print(json.dumps(entry, sort_keys=True), flush=True)
        failures += result.returncode != 0
    if not args.inner:
        print(f"CAMPAIGN {args.stage} COMPLETE failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.exit(main())
