"""Counter completion requires matching raw measurements, units, and provenance."""
import copy
import csv
import hashlib
import io
import json
import subprocess

import pytest

from tools.analysis.phase2_counter_evidence import (
    CAPTURE, METRICS, counter_evidence, parse_counters, profile_errors,
)
from tools.run import phase2_profile_counters as collector


@pytest.fixture
def launches():
    return [{"launch_index": index, "format": fmt, "strategy": strategy,
             "batch": 196, "channels": 64, "k": 288, "output_sha256": fmt}
            for index, (fmt, strategy) in enumerate(
                (fmt, strategy) for fmt in ("fp6_e3m2", "int8")
                for strategy in ("predecoded", "algorithmic", "lookup"))]


def metric_rows(launches):
    return [{"ID": str(launch["launch_index"]), "Process ID": "123", "Device": "0",
             "Kernel Name": ("gemm_kernel" if launch["strategy"] == "predecoded" else "encoded_kernel") + "(int *)",
             "Metric Name": name, "Metric Unit": unit, "Metric Value": value}
            for launch in launches
            for name, unit, value in zip(METRICS, ("nsecond", "byte", "byte", "%"),
                                         ("2,000", "32", "64", "50"))]


def to_csv(rows):
    stream = io.StringIO()
    stream.write("==PROF== Connected to process 123\n")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def test_bandwidth_uses_paired_counter_duration_and_bytes(launches):
    records = parse_counters(to_csv(metric_rows(launches)), launches)
    assert len(records) == 6
    assert records[0]["dram_bytes_per_second"] == 48_000_000
    assert records[0]["achieved_occupancy_percent"] == 50
    assert records[0]["dram_bytes"] == 96


def test_raw_wide_csv_matches_installed_ncu_export_layout(launches):
    # Layout and units checked against Nsight Compute 2025.4.1's bundled report.
    rows = [{"ID": "", "Process ID": "", "Device": "", "Kernel Name": "",
             **dict(zip(METRICS, ("ns", "byte", "byte", "%")))}]
    for launch in launches:
        rows.append({"ID": str(launch["launch_index"]), "Process ID": "123", "Device": "0",
                     "Kernel Name": ("gemm_kernel" if launch["strategy"] == "predecoded" else "encoded_kernel") + "(int *)",
                     **dict(zip(METRICS, ("2000", "32", "64", "50")))})
    assert parse_counters(to_csv(rows), launches) == parse_counters(to_csv(metric_rows(launches)), launches)


@pytest.mark.parametrize("metric,field,value", [
    (0, "Metric Value", "0"), (0, "Metric Value", "nan"),
    (0, "Metric Value", "1e999"), (0, "Metric Unit", "byte"),
    (1, "Metric Value", "n/a"), (1, "Metric Value", "-1"),
    (1, "Metric Value", "1.5"), (1, "Metric Unit", "Kbyte"),
    (2, "Metric Value", "1,23"), (3, "Metric Value", "101"),
    (3, "Metric Unit", "ratio"),
])
def test_invalid_counter_values_cannot_close_gate(launches, metric, field, value):
    rows = metric_rows(launches)
    rows[metric][field] = value
    with pytest.raises(ValueError):
        parse_counters(to_csv(rows), launches)


def test_zero_dram_traffic_is_valid(launches):
    rows = metric_rows(launches)
    rows[1]["Metric Value"] = rows[2]["Metric Value"] = "0"
    assert parse_counters(to_csv(rows), launches)[0]["dram_bytes_per_second"] == 0


@pytest.mark.parametrize("mutation", ["missing_metric", "duplicate", "missing_launch", "order", "kernel", "process"])
def test_incomplete_or_mismatched_launches_are_rejected(launches, mutation):
    rows = metric_rows(launches)
    if mutation == "missing_metric":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[-1])
    elif mutation == "missing_launch":
        rows = rows[:-4]
    elif mutation == "order":
        rows = rows[4:] + rows[:4]
    elif mutation == "kernel":
        for row in rows[:4]:
            row["Kernel Name"] = "encoded_kernel(int *)"
    else:
        rows[-1]["Process ID"] = "999"
    with pytest.raises(ValueError):
        parse_counters(to_csv(rows), launches)


@pytest.fixture
def capture_files(tmp_path, launches):
    work = tmp_path / CAPTURE.parent
    work.mkdir(parents=True)
    inputs = {"source_sha256": "current", "launches": launches}
    paths = {"inputs": work / "counter-inputs.json", "csv": work / "counters.csv"}
    paths["inputs"].write_text(json.dumps(inputs))
    paths["csv"].write_text(to_csv(metric_rows(launches)))
    capture = {"status": "completed", "returncode": 0, "source_sha256": "current",
               "ncu_version": "test fixture", **{key: str(path.relative_to(tmp_path)) for key, path in paths.items()},
               "artifacts": {str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in paths.values()}}
    (tmp_path / CAPTURE).write_text(json.dumps(capture))
    return tmp_path, inputs, paths, capture


def test_counter_capture_checks_hashes_and_input_identity(capture_files):
    root, inputs, paths, _ = capture_files
    assert counter_evidence(root, inputs)["status"] == "measured"
    changed = copy.deepcopy(inputs)
    changed["launches"][0]["output_sha256"] = "different output"
    with pytest.raises(ValueError, match="inputs differ"):
        counter_evidence(root, changed)
    paths["csv"].write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        counter_evidence(root, inputs)


def test_failed_process_cannot_reuse_leftover_counters(capture_files):
    root, inputs, _, capture = capture_files
    capture.update(status="ERR_NVGPUCTRPERM", returncode=1)
    (root / CAPTURE).write_text(json.dumps(capture))
    assert counter_evidence(root, inputs)["records"] == []


def test_final_profile_rechecks_derived_values_and_source(capture_files):
    root, inputs, _, _ = capture_files
    reference = root / CAPTURE.parent / "profile-inputs.json"
    reference.write_text(json.dumps(inputs))
    counters = counter_evidence(root, inputs)
    profile = {"source_sha256": "current", "counter_status": "measured", "missing_metrics": [],
               "counter_records": counters["records"],
               "artifacts": {**counters["artifacts"], str(reference.relative_to(root)): hashlib.sha256(reference.read_bytes()).hexdigest()}}
    assert profile_errors(root, profile, "current") == []
    assert profile_errors(root, profile, "new source")
    profile["counter_records"][0]["dram_bytes_per_second"] = 1
    assert profile_errors(root, profile, "current")


def test_empty_missing_metric_list_alone_is_not_profiling_evidence(tmp_path):
    profile = {"source_sha256": "current", "counter_status": "measured", "missing_metrics": [], "artifacts": {}}
    assert profile_errors(tmp_path, profile, "current")


def test_failed_collector_preserves_trace_inputs_and_retains_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "ROOT", tmp_path)
    monkeypatch.setattr(collector, "source_identity", lambda: "current")
    monkeypatch.setattr(collector.subprocess, "check_output", lambda *args, **kwargs: "test ncu")
    def fail(command, **kwargs):
        kwargs["stdout"].write("==ERROR== ERR_NVGPUCTRPERM\n")
        return subprocess.CompletedProcess(command, 1)
    monkeypatch.setattr(collector.subprocess, "run", fail)
    trace = tmp_path / CAPTURE.parent / "profile-inputs.json"
    trace.parent.mkdir(parents=True)
    trace.write_bytes(b"retained Systems trace input")
    assert collector.capture() == 1
    assert collector.capture() == 1
    assert trace.read_bytes() == b"retained Systems trace input"
    assert len(list(trace.parent.glob("counter-attempt-*/capture.json"))) == 2
    report = json.loads((tmp_path / CAPTURE).read_text())
    assert report["status"] == "ERR_NVGPUCTRPERM"
    assert len(report["artifacts"]) == 1


def test_successful_collector_produces_verifiable_capture(tmp_path, monkeypatch, launches):
    monkeypatch.setattr(collector, "ROOT", tmp_path)
    monkeypatch.setattr(collector, "source_identity", lambda: "current")
    monkeypatch.setattr(collector.subprocess, "check_output", lambda *args, **kwargs: "test ncu")
    inputs = {"source_sha256": "current", "launches": launches}
    trace = tmp_path / CAPTURE.parent / "profile-inputs.json"
    trace.parent.mkdir(parents=True)
    trace.write_text(json.dumps(inputs))
    def succeed(command, **kwargs):
        from pathlib import Path
        Path(command[command.index("--output") + 1]).write_text(json.dumps(inputs))
        Path(command[command.index("--log-file") + 1]).write_text(to_csv(metric_rows(launches)))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(collector.subprocess, "run", succeed)
    assert collector.capture() == 0
    assert counter_evidence(tmp_path, inputs)["status"] == "measured"


@pytest.mark.parametrize("message", ["a password is required", "interactive authentication is required"])
def test_collector_retains_administrator_authentication_failure(tmp_path, monkeypatch, message):
    monkeypatch.setattr(collector, "ROOT", tmp_path)
    monkeypatch.setattr(collector, "source_identity", lambda: "current")
    monkeypatch.setattr(collector.subprocess, "check_output", lambda *args, **kwargs: "test ncu")
    def fail(command, **kwargs):
        assert command[:2] == ["sudo", "-n"]
        kwargs["stdout"].write("sudo: " + message + "\n")
        return subprocess.CompletedProcess(command, 1)
    monkeypatch.setattr(collector.subprocess, "run", fail)
    assert collector.capture(sudo=True) == 1
    assert json.loads((tmp_path / CAPTURE).read_text())["status"] == "administrator_authentication_required"


def test_profile_input_cli_saves_all_launches_to_requested_path(tmp_path, monkeypatch):
    from tools.run import phase2_profile_inputs as workload
    output = tmp_path / "attempt/inputs.json"
    monkeypatch.setattr("sys.argv", ["profile-inputs", "--output", str(output)])
    monkeypatch.setattr(workload, "source_identity", lambda: "current")
    monkeypatch.setattr(workload, "prepare_tensor", lambda tensor: tensor)
    class Native:
        def gemm(self, *args, **kwargs):
            assert (kwargs["batch"], kwargs["channels"], kwargs["k"]) == (196, 64, 288)
            return (0,)
        encoded_gemm = gemm
    monkeypatch.setattr(workload, "NativeBackend", lambda backend: Native())
    workload.main()
    document = json.loads(output.read_text())
    assert document["source_sha256"] == "current"
    assert [row["launch_index"] for row in document["launches"]] == list(range(6))
    assert {row["strategy"] for row in document["launches"]} == {"predecoded", "algorithmic", "lookup"}
