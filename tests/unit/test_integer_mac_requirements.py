import json

import pytest

from public.generic_rtl.mac.requirements import ROOT, file_sha256, indexed_resolution, requirements
from tools.hardware.integer_mac_baseline import generate_vectors, run_one


def test_accepted_requirements_keep_unknown_mappings_explicit():
    inventory = requirements()
    rows = inventory["formats"]
    assert len(rows) == 25
    assert len({row["format"] for row in rows}) == 25
    integer = next(row for row in rows if row["format"] == "int4")
    assert integer["product"]["integer_bits_if_raw_integer"] == 8
    assert integer["scale"]["precision_bits"] is None
    assert integer["scale"]["mapped_scale_bits_64"] == "analytical_storage_assumption_only"
    assert integer["hardware_status"]["full_operator"] == "requirements_only"
    bfp = next(row for row in rows if row["format"] == "bfp6")
    assert bfp["scale"]["block"]["block_size"] == 32
    assert bfp["scale"]["precision_bits"] == 8
    assert "maximum_based_shared_scale_in_place_of_mse_search" in bfp["numerical_change_required_for"]
    binary = next(row for row in rows if row["format"] == "binary_pm1")
    assert binary["weight"]["code_mapping"]["code_values"] == [
        {"code": 0, "value": "-1", "classification": "finite"},
        {"code": 1, "value": "1", "classification": "finite"},
    ]
    assert binary["weight"]["code_mapping"]["manifest_zero"] == "not_representable"
    ternary = next(row for row in rows if row["format"] == "ternary")
    assert [row["value"] for row in ternary["weight"]["code_mapping"]["code_values"]] == ["0", "1", "-1", "NaN"]
    assert ternary["weight"]["code_mapping"]["reserved_codes"] == [3]


def test_indexed_resolution_rejects_tampered_or_misidentified_contents(tmp_path):
    index = json.loads((ROOT / "public/experiments/configs/experiment_a/phase3-accumulator-resolutions-v1.json").read_text())
    reference = index["resolutions"]["resnet18/int8"]
    original = ROOT / reference["path"]
    destination = tmp_path / "resolution.json"
    destination.write_bytes(original.read_bytes())
    checked = {"path": destination.name, "sha256": reference["sha256"]}
    assert indexed_resolution(tmp_path, checked, "resnet18", "int8")["accumulator"] == "int64_accumulator"
    changed = json.loads(destination.read_text())
    changed["accumulator"] = "int32_accumulator"
    destination.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="hash mismatch"):
        indexed_resolution(tmp_path, checked, "resnet18", "int8")
    checked["sha256"] = file_sha256(destination)
    with pytest.raises(ValueError, match="identity mismatch"):
        indexed_resolution(tmp_path, checked, "mobilenet_v2", "int8")


def test_oracle_driven_stream_and_wrapper_conformance(tmp_path):
    rows, cases, _ = generate_vectors(4, 4, 32)
    assert cases["exhaustive_pairs"] == 256
    assert cases["resets"] == 2
    assert len(rows) > 256
    result = run_one(4, 4, 32, output_dir=tmp_path)
    assert result["status"] == "pass"
    assert result["result_checks"] > 256
    assert result["vectors_sha256"] == "390d9975559a500a6e8ca5b664403c470b831e654f7e1b932476658c445411bb"
    assert result["harness_simulation_stdout"].startswith("PASS harness")
