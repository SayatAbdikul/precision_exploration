"""Validate retained Nsight Compute counters against the six frozen launches."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

METRICS = (
    "gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
)
MISSING_METRICS = ["achieved_occupancy", "kernel_DRAM_bytes", "kernel_DRAM_bandwidth"]
CAPTURE = Path("artifacts/benchmarks/phase2/counter-capture.json")


def checked_artifacts(root: Path, artifacts: dict) -> dict:
    for relative, expected in artifacts.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("profiling artifact escapes repository")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"profiling artifact hash mismatch: {relative}")
    return artifacts


def parse_counters(text: str, launches: list[dict]) -> list[dict]:
    """Read raw (wide) or details (long) CSV with explicit metric units."""
    lines = text.splitlines()
    if any("==ERROR==" in line for line in lines):
        raise ValueError("Nsight Compute reported an error")
    header = next((i for i, line in enumerate(lines)
                   if {"ID", "Kernel Name"}
                   <= set(next(csv.reader([line]), []))), None)
    if header is None:
        raise ValueError("Nsight Compute CSV header missing")
    reader = csv.DictReader(io.StringIO("\n".join(line for line in lines[header:]
                                                  if line.strip() and not line.startswith("==PROF=="))))
    rows = list(reader)
    fields = set(reader.fieldnames)
    if not {"Metric Name", "Metric Unit", "Metric Value"} <= fields:
        if not set(METRICS) <= fields or not rows or rows[0]["ID"] != "":
            raise ValueError("required Nsight Compute metrics/units missing")
        units = rows.pop(0)
        rows = [{**row, "Metric Name": name, "Metric Unit": units[name], "Metric Value": row[name]}
                for row in rows for name in METRICS]
    groups = {}
    for row in rows:
        identity = (row.get("Process ID"), row.get("Device"), row["ID"])
        if not row["ID"].isdigit():
            raise ValueError("invalid Nsight Compute launch ID")
        group = groups.setdefault(identity, {"kernel": row["Kernel Name"], "metrics": {}})
        if group["kernel"] != row["Kernel Name"]:
            raise ValueError("inconsistent kernel identity within launch")
        name = row["Metric Name"]
        if name not in METRICS:
            continue
        if name in group["metrics"]:
            raise ValueError("duplicate Nsight Compute metric")
        value_text = row["Metric Value"] or ""
        if not re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?(?:[eE][+-]?\d+)?", value_text):
            raise ValueError("counter must be finite and nonnegative")
        value = float(value_text.replace(",", ""))
        if not math.isfinite(value):
            raise ValueError("counter must be finite and nonnegative")
        unit = row["Metric Unit"]
        if name == METRICS[0]:
            factors = {"second": 1e9, "msecond": 1e6, "usecond": 1e3, "nsecond": 1,
                       "s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1}
            if unit not in factors or value <= 0:
                raise ValueError("duration must have time units and be positive")
            value *= factors[unit]
            if not math.isfinite(value):
                raise ValueError("duration must be finite")
        elif name in METRICS[1:3]:
            if unit != "byte" or not value.is_integer():
                raise ValueError("DRAM counters must be whole bytes")
            value = int(value)
        elif unit != "%" or value > 100:
            raise ValueError("achieved occupancy must be a percentage in [0, 100]")
        group["metrics"][name] = value
    identities = list(groups)
    if (len(groups) != len(launches) or not launches
            or len({key[:2] for key in identities}) != 1
            or [int(key[2]) for key in identities] != list(range(len(launches)))):
        raise ValueError("counter launch coverage/order does not match profile inputs")
    records = []
    for index, (launch, group) in enumerate(zip(launches, groups.values())):
        expected = "gemm_kernel" if launch["strategy"] == "predecoded" else "encoded_kernel"
        kernel = group["kernel"]
        if launch["launch_index"] != index or not re.search(r"\b" + expected + r"\s*(?:<|\()", kernel):
            raise ValueError("counter kernel does not match declared strategy")
        values = group["metrics"]
        if set(values) != set(METRICS):
            raise ValueError("required Nsight Compute metrics missing")
        duration, read, written, occupancy = (values[name] for name in METRICS)
        records.append({**launch, "kernel": kernel, "duration_ns": duration,
                        "dram_read_bytes": read, "dram_write_bytes": written,
                        "dram_bytes": read + written,
                        "dram_bytes_per_second": (read + written) * 1e9 / duration,
                        "achieved_occupancy_percent": occupancy})
    return records


def counter_evidence(root: Path, inputs: dict) -> dict:
    path = root / CAPTURE
    if not path.exists():
        return {"status": "unavailable: no retained counter capture", "records": [], "artifacts": {}}
    capture = json.loads(path.read_text())
    artifacts = {str(CAPTURE): hashlib.sha256(path.read_bytes()).hexdigest(),
                 **checked_artifacts(root, capture["artifacts"])}
    if capture["status"] != "completed" or capture["returncode"] != 0:
        return {"status": "unavailable: " + capture["status"], "records": [], "artifacts": artifacts}
    if capture["source_sha256"] != inputs["source_sha256"]:
        raise ValueError("counter capture uses a different engine source")
    if capture["inputs"] not in artifacts or capture["csv"] not in artifacts:
        raise ValueError("counter inputs/CSV lack artifact hashes")
    if json.loads((root / capture["inputs"]).read_text()) != inputs:
        raise ValueError("counter inputs differ from the Systems trace launches")
    records = parse_counters((root / capture["csv"]).read_text(), inputs["launches"])
    return {"status": "measured", "records": records, "artifacts": artifacts,
            "tool": capture["ncu_version"]}


def profile_errors(root: Path, profile: dict, current_source: str) -> list[str]:
    """Revalidate raw bytes and derived counters before a completion decision."""
    errors = []
    if profile.get("source_sha256") != current_source:
        errors.append("profiling engine source is stale")
    try:
        checked_artifacts(root, profile["artifacts"])
        if not profile["missing_metrics"]:
            inputs_path = "artifacts/benchmarks/phase2/profile-inputs.json"
            if inputs_path not in profile["artifacts"]:
                raise ValueError("profile inputs lack artifact hash")
            inputs = json.loads((root / inputs_path).read_text())
            if inputs["source_sha256"] != current_source:
                raise ValueError("profile input source is stale")
            counters = counter_evidence(root, inputs)
            if (profile["counter_status"] != "measured" or counters["status"] != "measured"
                    or counters["records"] != profile.get("counter_records")):
                raise ValueError("profile summary lacks matching measured counter evidence")
            if any(profile["artifacts"].get(key) != value for key, value in counters["artifacts"].items()):
                raise ValueError("counter evidence is not bound to profile summary")
    except (ValueError, KeyError, OSError) as exc:
        errors.append(str(exc))
    return errors
