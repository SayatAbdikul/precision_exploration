"""Datatype-manifest loading, canonical identity, and semantic validation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "manifests" / "datatype-manifest.schema.json"
ROLE_LIMITED_TO_EIGHT_BITS = frozenset({"weight", "activation", "output"})
NF4_VALUES = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)


STANDARD_SEMANTICS: dict[str, dict[str, Any]] = {
    "fp8_e4m3fn": {
        "family": "float", "bits": 8, "signed": True, "encoding": "sign_e4m3",
        "rounding": "rne", "overflow": "saturate", "underflow": "subnormal", "zero": "signed_zero",
        "float": {"exp_bits": 4, "mantissa_bits": 3, "bias": 7, "subnormals": True, "nan": True, "infinity": False},
    },
    "fp8_e5m2": {
        "family": "float", "bits": 8, "signed": True, "encoding": "sign_e5m2",
        "rounding": "rne", "overflow": "infinity", "underflow": "subnormal", "zero": "signed_zero",
        "float": {"exp_bits": 5, "mantissa_bits": 2, "bias": 15, "subnormals": True, "nan": True, "infinity": True},
    },
    "posit8_es1": {"family": "posit", "bits": 8, "encoding": "posit_standard", "posit": {"es": 1}},
    "posit6_es1": {"family": "posit", "bits": 6, "encoding": "posit_standard", "posit": {"es": 1}},
    "posit4_es0": {"family": "posit", "bits": 4, "encoding": "posit_standard", "posit": {"es": 0}},
    "nf4": {"family": "codebook", "bits": 4, "encoding": "bitsandbytes_normalfloat4",
            "zero": "code_0111", "codebook": {"values": list(NF4_VALUES)}},
}


class ManifestError(ValueError):
    """Raised when a datatype manifest is structurally or semantically invalid."""


def _load_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(text)
        else:
            value = json.loads(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ManifestError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"manifest root must be an object: {path}")
    return value


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Return the contract-defined canonical bytes for a manifest."""

    value = copy.deepcopy(dict(manifest))
    value.pop("manifest_id", None)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (encoded + "\n").encode("utf-8")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def _error(message: str) -> None:
    raise ManifestError(message)


def validate_manifest(manifest: Mapping[str, Any], *, role: str | None = None) -> dict[str, Any]:
    """Validate schema plus the Phase 0 cross-field semantic contract.

    The returned object is a deep copy so callers cannot mutate the validated
    value held by another subsystem.
    """

    value = copy.deepcopy(dict(manifest))
    validator = Draft202012Validator(_schema())
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.absolute_path))
    if errors:
        details = "; ".join(
            f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors
        )
        _error(f"datatype manifest schema validation failed: {details}")

    bits = value["bits"]
    family = value["family"]
    signed = value["signed"]
    scaling = value["scaling"]

    if not value.get("zero"):
        _error("every manifest requires explicit zero/reserved-value semantics")

    if role is not None and role not in {"weight", "activation", "accumulator", "output"}:
        _error(f"unknown datatype role: {role}")
    if role in ROLE_LIMITED_TO_EIGHT_BITS and bits > 8:
        _error(f"{role} manifest {value['name']} uses {bits} bits; the role limit is 8")

    if scaling["mode"] == "none" and scaling["granularity"] != "none":
        _error("scale mode 'none' requires granularity 'none'")
    if scaling["mode"] == "intrinsic_shared":
        if scaling["granularity"] != "block" or "block" not in value:
            _error("intrinsic shared scaling requires block granularity and block metadata")
    if scaling["mode"] != "none" and scaling["granularity"] == "none":
        _error("a non-none scale mode requires a non-none granularity")

    numeric = value.get("numeric")
    if family in {"integer", "fixed_point", "mx_integer"}:
        if numeric is None:
            _error(f"{family} requires numeric parameters")
        if "integer_bits" not in numeric or "fractional_bits" not in numeric:
            _error(f"{family} requires explicit integer_bits and fractional_bits")
        allocation = numeric["integer_bits"] + numeric["fractional_bits"] + int(signed)
        if allocation > bits:
            _error(f"numeric bit allocation {allocation} exceeds declared width {bits}")
        if family == "integer" and numeric["fractional_bits"] != 0:
            _error("integer manifests require fractional_bits = 0")

    float_params = value.get("float")
    if family in {"float", "mx_float"}:
        if float_params is None:
            _error(f"{family} requires float parameters")
        allocation = float_params["exp_bits"] + float_params["mantissa_bits"] + int(signed)
        if allocation != bits:
            _error(f"float bit allocation {allocation} does not equal declared width {bits}")
        if float_params["infinity"] and value["overflow"] != "infinity":
            _error("an infinity-enabled float requires overflow='infinity'")
        if value["overflow"] == "infinity" and not float_params["infinity"]:
            _error("overflow='infinity' requires an infinity encoding")
        if float_params["subnormals"] != (value["underflow"] == "subnormal"):
            _error("float subnormal and underflow policies disagree")

    block = value.get("block")
    if family in {"bfp", "mx_float", "mx_integer"}:
        if block is None:
            _error(f"{family} requires block metadata")
        if scaling["mode"] != "intrinsic_shared":
            _error(f"{family} requires intrinsic_shared scaling")
        if scaling.get("scale_format") not in {None, block["shared_scale_format"]}:
            _error("scale_format and block.shared_scale_format disagree")
    if family == "bfp" and not any(key in value for key in ("numeric", "float", "codebook")):
        _error("bfp requires explicit element numeric, float, or codebook semantics")

    if family == "posit":
        if not (2 <= bits <= 64):
            _error("posit width must be between 2 and 64 bits")
        if value["overflow"] != "saturate" or value["underflow"] == "subnormal":
            _error("posit uses saturation and has no IEEE subnormal encoding")

    if family in {"logarithmic", "power_of_two"}:
        params = value["logarithmic"]
        allocation = params["integer_bits"] + params["fractional_bits"] + int(signed) + 1
        if allocation > bits:
            _error("logarithmic fields plus the explicit zero code exceed declared width")

    codebook = value.get("codebook")
    if family == "codebook":
        if codebook is None:
            _error("codebook family requires codebook values")
        expected = 1 << bits
        if len(codebook["values"]) != expected:
            _error(f"codebook requires exactly 2^bits={expected} values in code order")
        if len(set(codebook["values"])) != len(codebook["values"]):
            _error("codebook values must be unique")
    if codebook is not None and len(codebook["values"]) > (1 << bits):
        _error("codebook has more entries than encodings")

    if family == "binary" and not (
        bits == 1 and value["encoding"] in {"binary_pm1", "binary_01"}
    ):
        _error("binary requires one bit and encoding binary_pm1 or binary_01")
    if family == "ternary" and not (
        bits == 2 and value["encoding"] == "ternary_zero_pos_neg_reserved"
    ):
        _error("ternary requires two bits with an explicit reserved fourth code")

    if family in {"integer", "fixed_point", "mx_integer"}:
        zero_point = numeric.get("zero_point", 0)
        minimum = -(1 << (bits - 1)) if signed else 0
        maximum = (1 << (bits - int(signed))) - 1
        if not minimum <= zero_point <= maximum:
            _error("zero_point is not representable in the declared encoding")

    expected_standard = STANDARD_SEMANTICS.get(value["name"])
    if expected_standard is not None:
        for key, expected in expected_standard.items():
            if value.get(key) != expected:
                _error(f"standard name {value['name']} requires standard {key} semantics")

    if value["name"].startswith("mxfp"):
        expected_mx = {
            "mxfp8_e4m3": (8, 4, 3, 7),
            "mxfp6_e3m2": (6, 3, 2, 3),
            "mxfp4_e2m1": (4, 2, 1, 1),
        }.get(value["name"])
        if expected_mx is not None:
            width, exponent, mantissa, bias = expected_mx
            if (
                family != "mx_float" or bits != width
                or value.get("float") != {"exp_bits": exponent, "mantissa_bits": mantissa, "bias": bias,
                                             "subnormals": True, "nan": width == 8, "infinity": False}
                or value.get("block") != {"shared_scale_format": "e8m0", "block_size": 32,
                                           "block_axis": "reduction_k",
                                           "incomplete_block_policy": "scale_valid_values_zero_pad"}
                or scaling != {"mode": "intrinsic_shared", "granularity": "block",
                               "scale_format": "e8m0", "scale_bits": 8}
            ):
                _error(f"standard name {value['name']} requires OCP MX block and element semantics")

    return value


def load_manifest(path: str | Path, *, role: str | None = None) -> dict[str, Any]:
    source = Path(path)
    return validate_manifest(_load_document(source), role=role)
