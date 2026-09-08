from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from public.formats.oracle.manifest import ManifestError, load_manifest, manifest_sha256, validate_manifest


ROOT = Path(__file__).resolve().parents[2]
FP6 = ROOT / "tests" / "conformance" / "fixtures" / "manifests" / "valid" / "fp6_e3m2_finite.json"


def test_phase0_fp6_manifest_is_valid_and_stable() -> None:
    manifest = load_manifest(FP6, role="weight")
    assert manifest["name"] == "fp6_e3m2_finite"
    assert len(manifest_sha256(manifest)) == 64


@pytest.mark.parametrize(
    "relative",
    [
        "fp6_missing_float_semantics.json",
        "fp6_missing_rounding.json",
        "zero_bit_integer.json",
    ],
)
def test_phase0_invalid_manifests_are_rejected(relative: str) -> None:
    path = ROOT / "tests" / "conformance" / "fixtures" / "manifests" / "invalid" / relative
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_role_width_is_enforced() -> None:
    manifest = json.loads(FP6.read_text())
    manifest["name"] = "fp16_e5m10"
    manifest["bits"] = 16
    manifest["float"] = {**manifest["float"], "exp_bits": 5, "mantissa_bits": 10, "bias": 15}
    validate_manifest(manifest, role="accumulator")
    with pytest.raises(ManifestError, match="role limit"):
        validate_manifest(manifest, role="activation")


def test_float_allocation_must_match_width() -> None:
    manifest = json.loads(FP6.read_text())
    manifest["float"]["mantissa_bits"] = 3
    with pytest.raises(ManifestError, match="allocation"):
        validate_manifest(manifest)


def test_validation_returns_an_independent_copy() -> None:
    source = json.loads(FP6.read_text())
    validated = validate_manifest(source)
    changed = copy.deepcopy(validated)
    changed["name"] = "different"
    assert source["name"] == "fp6_e3m2_finite"


def test_standard_name_rejects_nonstandard_semantics() -> None:
    manifest = json.loads((ROOT / "public/formats/manifests/accepted/fp8_e4m3fn.json").read_text())
    manifest["float"]["bias"] = 8
    with pytest.raises(ManifestError, match="standard name"):
        validate_manifest(manifest)


def test_codebook_values_must_be_unique() -> None:
    manifest = json.loads((ROOT / "public/formats/manifests/accepted/nf4.json").read_text())
    manifest["name"] = "nf4_duplicate_fixture"
    manifest["codebook"]["values"][1] = manifest["codebook"]["values"][0]
    with pytest.raises(ManifestError, match="unique"):
        validate_manifest(manifest)


@pytest.mark.parametrize("name", ["mxfp4_e2m1", "mxfp6_e3m2"])
def test_ocp_mx_names_reject_element_nan(name):
    manifest = json.loads((ROOT / f"public/formats/manifests/accepted/{name}.json").read_text())
    manifest["float"]["nan"] = True
    with pytest.raises(ManifestError, match="OCP MX"):
        validate_manifest(manifest)
