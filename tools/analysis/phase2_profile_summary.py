"""Normalize Nsight Systems kernel timing without inventing hardware counters."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

from public.inference.conformance_job import source_identity

ROOT = Path(__file__).resolve().parents[2]


def main():
    work = ROOT / "artifacts/benchmarks/phase2"
    inputs = json.loads((work / "profile-inputs.json").read_text())
    if inputs["source_sha256"] != source_identity():
        raise ValueError("profile inputs refer to stale engine sources")
    with (work / "stats_cuda_gpu_trace.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    kernels = [row for row in rows if row["GrdX"]]
    if len(kernels) != len(inputs["launches"]):
        raise ValueError("trace does not have one kernel per declared launch")
    records = []
    for launch, row in zip(inputs["launches"], kernels):
        expected = "gemm_kernel" if launch["strategy"] == "predecoded" else "encoded_kernel"
        if not row["Name"].startswith(expected) or int(row["Duration (ns)"]) <= 0:
            raise ValueError("unexpected profile kernel identity/duration")
        duration = int(row["Duration (ns)"])
        records.append({**launch, "kernel": row["Name"], "duration_ns": duration,
                        "mac_per_second": launch["batch"]*launch["channels"]*launch["k"]*1e9/duration,
                        "registers_per_thread": int(row["Reg/Trd"]), "device": row["Device"]})
    report = {"schema_version": "2.0.0", "source_sha256": inputs["source_sha256"], "records": records,
              "tool": subprocess.check_output(["nsys", "--version"], text=True).strip(),
              "scope": "one traced kernel launch per strategy/family, no statistical latency ranking",
              "counter_status": "unavailable: Nsight Compute ERR_NVGPUCTRPERM; no GPU counter settings changed",
              "missing_metrics": ["achieved_occupancy", "kernel_DRAM_bytes", "kernel_DRAM_bandwidth"],
              "artifacts": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (work/"profile-inputs.json", work/"cuda-profile.nsys-rep",
                                         work/"stats_cuda_gpu_trace.csv", work/"stats_cuda_gpu_mem_time_sum.csv", work/"ncu-attempt.log")}}
    (ROOT / "results/summaries/phase2-cuda-profile.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
