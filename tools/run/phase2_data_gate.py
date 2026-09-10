"""Finish the long ImageNet data/calibration gate as one resumable process.

This is a one-shot continuation of Phase 2, not a scheduled service. Recovery
reuses verified images. Completed calibration artifacts are verified and reused.
The final report continues to expose external gates and failing regressions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from public.quantization.calibration.artifact import validate
from public.workloads.datasets.identity import verify_payload_record

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "artifacts/dataset_indexes/phase2-data-gate.json"


def main():
    state = {"schema_version": "phase2-data-gate-1.0.0", "pid": os.getpid(),
             "started_at": datetime.now(timezone.utc).isoformat(), "status": "running", "steps": []}
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "PRECISION_TEST_BACKEND": "cpp"}

    def save():
        STATE.parent.mkdir(parents=True, exist_ok=True)
        temporary = STATE.with_suffix(".partial")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(STATE)

    def run(name, arguments, *, allow_failure=False):
        step = {"name": name, "status": "running", "arguments": arguments}
        state["steps"].append(step)
        save()
        print(f"starting {name}", flush=True)
        result = subprocess.run([sys.executable, "-m", *arguments], cwd=ROOT, env=environment)
        step.update(status="completed" if result.returncode == 0 else "failed", returncode=result.returncode)
        save()
        if result.returncode and not allow_failure:
            raise RuntimeError(f"{name} failed with exit code {result.returncode}; rerun to resume")

    save()
    try:
        run("recover_imagenet_training", ["tools.setup.restore_phase2_imagenet_archives", "--split", "train", "--workers", "8"])
        index = json.loads((ROOT / "data/manifests/index.json").read_text())["records"]
        for name in ("imagenet_calibration_2k", "imagenet_screen_1k", "imagenet_evaluation_10k"):
            verify_payload_record(ROOT, index[name])
        for model in ("resnet18", "mobilenet_v2", "mobilenet_v3_large"):
            formats = set()
            for path in (ROOT / "artifacts/calibration").glob(f"{model}-*.json"):
                if "observations" not in path.name:
                    document = validate(json.loads(path.read_text()), ROOT)
                    formats.add(document["format"])
            missing = sorted({"fp6_e3m2", "int8"} - formats)
            if missing:
                run(f"calibrate_{model}", ["tools.run.phase2_calibrate", "--model", model, "--formats", *missing])
        run("phase2_cpu_tests", ["pytest", "tests/unit", "tests/conformance", "tests/integration/test_exact_graph.py",
            "tests/integration/test_workload_resume.py", "-q", "--junitxml=artifacts/conformance/phase2/pytest-phase2-final.xml"], allow_failure=True)
        run("repository_tests", ["pytest", "-q", "--junitxml=artifacts/conformance/phase2/pytest-cpu-final.xml"], allow_failure=True)
        run("decision_evidence", ["tools.analysis.phase2_decision_evidence"])
        run("verification_report", ["tools.analysis.phase2_finalize"])
        report = json.loads((ROOT / "results/summaries/phase2-final-verification.json").read_text())
        state.update(status="data_and_calibration_completed", phase2_status=report["status"], remaining_gates=report["remaining_gates"])
    except Exception as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
