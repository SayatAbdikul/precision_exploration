"""Capture Phase 2 counters with retained failures and separate trace inputs.

Use --sudo only with administrator authorization. It uses noninteractive sudo
and never changes driver policy. Each attempt retains its own logs and inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from public.inference.conformance_job import source_identity
from tools.analysis.phase2_counter_evidence import CAPTURE, METRICS, parse_counters

ROOT = Path(__file__).resolve().parents[2]


def capture(*, sudo=False, ncu="/usr/local/cuda/bin/ncu"):
    work = ROOT / CAPTURE.parent
    work.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="counter-attempt-", dir=work))
    inputs, csv_path, log = (attempt / name for name in ("inputs.json", "counters.csv", "process.log"))
    command = [ncu, "--config-file", "off", "--target-processes", "application-only",
               "--metrics", ",".join(METRICS), "--csv", "--page", "raw",
               "--print-units", "base", "--clock-control", "none",
               "--log-file", str(csv_path), sys.executable, "-m", "tools.run.phase2_profile_inputs",
               "--output", str(inputs)]
    if sudo:
        command = ["sudo", "-n", "env", "OMP_NUM_THREADS=4", "LC_ALL=C", *command]
    report = {"schema_version": "phase2-counter-capture-1.0.0", "source_sha256": source_identity(),
              "started_at": datetime.now(timezone.utc).isoformat(), "command": command,
              "status": "running", "returncode": None,
              "inputs": str(inputs.relative_to(ROOT)), "csv": str(csv_path.relative_to(ROOT))}
    try:
        report["ncu_version"] = subprocess.check_output([ncu, "--version"], text=True).strip()
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "4", "LC_ALL": "C"},
                                    stdout=stream, stderr=subprocess.STDOUT)
        report["returncode"] = result.returncode
        messages = log.read_text() + (csv_path.read_text() if csv_path.exists() else "")
        if "ERR_NVGPUCTRPERM" in messages:
            report["status"] = "ERR_NVGPUCTRPERM"
        elif sudo and any(message in messages for message in ("a password is required", "interactive authentication is required")):
            report["status"] = "administrator_authentication_required"
        elif result.returncode:
            report["status"] = "process_failed"
        else:
            document = json.loads(inputs.read_text())
            reference = json.loads((work / "profile-inputs.json").read_text())
            if document != reference or document["source_sha256"] != report["source_sha256"]:
                raise ValueError("counter inputs do not match retained Systems trace")
            if source_identity() != report["source_sha256"]:
                raise ValueError("engine source changed during profiling")
            parse_counters(csv_path.read_text(), document["launches"])
            report["status"] = "completed"
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report.update(status="capture_failed", error=str(exc))
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["artifacts"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (inputs, csv_path, log) if path.exists()}
        payload = json.dumps(report, indent=2) + "\n"
        (attempt / "capture.json").write_text(payload)
        temporary = (ROOT / CAPTURE).with_suffix(".partial")
        temporary.write_text(payload)
        temporary.replace(ROOT / CAPTURE)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sudo", action="store_true", help="Use already-authorized noninteractive sudo for this process")
    parser.add_argument("--ncu", default="/usr/local/cuda/bin/ncu")
    sys.exit(capture(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
