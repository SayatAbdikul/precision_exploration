from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from public.formats.oracle import NumberFormat, load_manifest


ROOT = Path(__file__).resolve().parents[2]


def fp6() -> NumberFormat:
    path = ROOT / "tests" / "conformance" / "fixtures" / "manifests" / "valid" / "fp6_e3m2_finite.json"
    return NumberFormat(load_manifest(path))


def test_fp6_known_codes_round_trip() -> None:
    number_format = fp6()
    for value in (Decimal("0"), Decimal("0.25"), Decimal("0.5"), Decimal("0.75"), Decimal("1.5"), Decimal("2"), Decimal("-0.5")):
        code = number_format.encode(value)
        assert number_format.decode(code) == value


def test_model_c_witness_operand_products_are_exactly_representable() -> None:
    number_format = fp6()
    x0 = number_format.encode("1.5")
    w0 = number_format.encode("0.5")
    x1 = number_format.encode("-0.5")
    w1 = number_format.encode("2")
    assert number_format.decode(x0) * number_format.decode(w0) == Decimal("0.75")
    assert number_format.decode(x1) * number_format.decode(w1) == Decimal("-1.0")


def test_add_and_mul_return_codes() -> None:
    number_format = fp6()
    one = number_format.encode(1)
    half = number_format.encode("0.5")
    assert number_format.decode(number_format.add(one, half)) == Decimal("1.5")
    assert number_format.decode(number_format.mul(one, half)) == Decimal("0.5")


def test_overflow_saturates() -> None:
    number_format = fp6()
    maximum = max(value for value in (number_format.decode(code) for code in range(64)) if value.is_finite())
    assert number_format.decode(number_format.encode(Decimal("1e100"))) == maximum


def test_ieee_like_float_all_nonzero_max_exponent_payloads_are_nan() -> None:
    manifest = {
        "schema_version": "1.0.0", "name": "test_e5m2", "family": "float", "bits": 8,
        "signed": True, "encoding": "sign_e5m2", "rounding": "rne", "overflow": "infinity",
        "underflow": "subnormal", "zero": "signed_zero", "scaling": {"mode": "none", "granularity": "none"},
        "float": {"exp_bits": 5, "mantissa_bits": 2, "bias": 15, "subnormals": True, "nan": True, "infinity": True},
    }
    number_format = NumberFormat(manifest)
    assert number_format.decode(0b01111100).is_infinite()
    assert all(number_format.decode(code).is_nan() for code in (0b01111101, 0b01111110, 0b01111111))


def test_infinity_overflow_policy_encodes_finite_overflow_as_infinity() -> None:
    manifest = {
        "schema_version": "1.0.0", "name": "fp8_e5m2", "family": "float", "bits": 8,
        "signed": True, "encoding": "sign_e5m2", "rounding": "rne", "overflow": "infinity",
        "underflow": "subnormal", "zero": "signed_zero", "scaling": {"mode": "none", "granularity": "none"},
        "float": {"exp_bits": 5, "mantissa_bits": 2, "bias": 15, "subnormals": True, "nan": True, "infinity": True},
    }
    number_format = NumberFormat(manifest)
    assert number_format.decode(number_format.encode("1e100")) == Decimal("Infinity")
    assert number_format.decode(number_format.encode("-1e100")) == Decimal("-Infinity")
