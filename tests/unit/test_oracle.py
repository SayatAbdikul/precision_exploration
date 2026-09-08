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


def accepted(name):
    return NumberFormat(load_manifest(ROOT / f"public/formats/manifests/accepted/{name}.json"))


def test_ocp_mxfp4_all_encodings_and_mxfp6_endpoints():
    # OCP MX v1.0 Tables 4/5: no element NaN/Inf; all codes are finite.
    fmt = accepted("mxfp4_e2m1")
    values = ["0", "0.5", "1", "1.5", "2", "3", "4", "6"]
    for code, value in enumerate(values):
        assert fmt.decode(code) == Decimal(value)
        assert fmt.decode(code + 8) == -Decimal(value)
        assert fmt.encode(value) == code
    fp6 = accepted("mxfp6_e3m2")
    assert fp6.decode(31) == 28 and fp6.decode(63) == -28
    assert all(fp6.decode(code).is_finite() for code in range(64))


def test_fp8_overflow_rounds_before_range_check_and_preserves_signed_zero():
    fmt = accepted("fp8_e5m2")
    for value in ("57345", "60000", "61439"):
        assert fmt.encode(value) == 123
        assert fmt.encode("-" + value) == 251
    assert fmt.encode("61440") == 124
    assert fmt.encode("-61440") == 252
    assert fmt.encode("-0") == 128
    assert fmt.encode("-0.000001") == 128
    assert fmt.decode(128).is_signed()
    assert fmt.mul(128, fmt.encode(1)) == 128
    assert fmt.convert(128, fmt) == 128


def test_oracle_is_independent_of_decimal_context_and_rejects_invalid_mx_scales():
    import pytest
    from decimal import localcontext
    from public.formats.oracle.number_format import OracleError
    original = accepted("log8").decode(31)
    with localcontext() as context:
        context.prec = 3
        assert accepted("log8").decode(31) == original
        assert accepted("fp8_e5m2").encode("60000") == 123
    mx = accepted("mxfp4_e2m1")
    assert mx.decode(7, scale=2) == 12
    assert mx.encode(12, scale=2) == 7
    for scale in (0, -1, 3, "Infinity", "NaN", Decimal(2) ** 128):
        with pytest.raises(OracleError):
            mx.encode(1, scale=scale)
        with pytest.raises(OracleError):
            mx.decode(1, scale=scale)


def test_fp8_against_independent_torch_cast():
    import pytest
    torch = pytest.importorskip("torch")
    fmt = accepted("fp8_e5m2")
    values = [0., -0., 57344., 57345., 60000., 61439., 61440., -60000., -61440., -1e-6, 1e-6]
    # Add midpoints and immediate neighbours across the finite exponent range.
    for code in range(1, 123):
        a, b = float(fmt.decode(code)), float(fmt.decode(code + 1))
        values.extend([a, (a + b) / 2, -(a + b) / 2])
    expected = torch.tensor(values, dtype=torch.float64).to(torch.float8_e5m2).view(torch.uint8).tolist()
    assert [fmt.encode(value) for value in values] == expected
