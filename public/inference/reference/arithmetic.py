"""Exact rational Model C arithmetic with explicit storage-format rounding.

Finite oracle values become rational numbers. For logarithmic formats these
are the oracle's versioned 200-digit approximations, not exact irrational reals.
No host floating-point multiply/add is used for the reduction.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from public.formats.oracle import NumberFormat, load_manifest
from public.formats.oracle.manifest import validate_manifest

ROOT = Path(__file__).resolve().parents[2]
Real = Fraction | Decimal


def real(value) -> Real:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ValueError("boolean is not a numerical value")
    if not isinstance(value, Decimal):
        # Preserve the exact value of checkpoint FP32/FP64 host containers.
        value = Decimal.from_float(value) if isinstance(value, float) else Decimal(value)
    if not value.is_finite() or value.is_zero() and value.is_signed():
        return value
    return Fraction(value)


def decimal(value: Real) -> Decimal:
    if isinstance(value, Decimal):
        return value
    with localcontext() as context:
        context.prec = 1200
        return Decimal(value.numerator) / Decimal(value.denominator)


def binary(left: Real, right: Real, operation: str) -> Real:
    if isinstance(left, Fraction) and isinstance(right, Fraction):
        return left + right if operation == "add" else left * right
    with localcontext() as context:
        context.prec = 1200
        from decimal import InvalidOperation
        context.traps[InvalidOperation] = False
        a, b = decimal(left), decimal(right)
        return real(a + b if operation == "add" else a * b)


def round_integer(value: Fraction, mode: str = "rne") -> int:
    sign = -1 if value < 0 else 1
    quotient, remainder = divmod(abs(value.numerator), value.denominator)
    if mode != "truncate":
        comparison = 2 * remainder - value.denominator
        if comparison > 0 or comparison == 0 and (mode == "nearest" or quotient & 1):
            quotient += 1
    return sign * quotient


def pow2(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


class Accumulator:
    """Constant-memory integer/fixed/IEEE-like float encoder, up to 64 bits."""

    def __init__(self, manifest):
        self.manifest = validate_manifest(manifest, role="accumulator")
        self.name, self.bits = self.manifest["name"], self.manifest["bits"]
        self.family = self.manifest["family"]
        if self.family not in {"integer", "fixed_point", "float"}:
            raise ValueError("accumulator requires an explicit integer, fixed-point or float manifest")
        if self.manifest["scaling"]["mode"] != "none":
            raise ValueError("accumulator manifest must be unscaled; domain scale is explicit")
        if self.family in {"integer", "fixed_point"} and self.manifest["numeric"].get("zero_point", 0) != 0:
            raise ValueError("accumulator mapping requires zero point 0")
        if self.family == "float":
            params = self.manifest["float"]
            if not params["infinity"] or not params["nan"] or not params["subnormals"]:
                raise ValueError("floating accumulator currently requires IEEE-like special values and subnormals")

    def encode(self, value, *, scale=1) -> int:
        value, scale = real(value), real(scale)
        if not isinstance(scale, Fraction) or scale <= 0:
            raise ValueError("accumulator scale must be finite and positive")
        signed = self.manifest["signed"]
        if self.family in {"integer", "fixed_point"}:
            lower = -(1 << (self.bits - 1)) if signed else 0
            upper = (1 << (self.bits - int(signed))) - 1
            if isinstance(value, Decimal) and not value.is_finite():
                if value.is_nan():
                    raise ValueError("integer accumulator cannot encode NaN")
                code = lower if value.is_signed() else upper
            else:
                numeric = self.manifest["numeric"]
                raw = Fraction(value) / scale * pow2(numeric["fractional_bits"])
                code = round_integer(raw, self.manifest["rounding"]) + numeric.get("zero_point", 0)
                code = min(upper, max(lower, code))
            return code & ((1 << self.bits) - 1)
        params = self.manifest["float"]
        mantissa, exponent_bits = params["mantissa_bits"], params["exp_bits"]
        negative = value.is_signed() if isinstance(value, Decimal) else value < 0
        if negative and not signed:
            raise ValueError("negative value in unsigned accumulator")
        sign = int(negative) << (self.bits - 1) if signed else 0
        special = ((1 << exponent_bits) - 1) << mantissa
        if isinstance(value, Decimal) and not value.is_finite():
            return special | 1 if value.is_nan() else sign | special
        magnitude = abs(Fraction(value)) / scale
        if not magnitude:
            return sign
        exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
        if magnitude < pow2(exponent):
            exponent -= 1
        minimum_exp = 1 - params["bias"]
        step = pow2(max(exponent, minimum_exp) - mantissa)
        significand = round_integer(magnitude / step, self.manifest["rounding"])
        if exponent < minimum_exp:
            # Includes rounding from the largest subnormal to the first normal.
            return sign | significand
        if significand == 1 << (mantissa + 1):
            exponent += 1
            significand >>= 1
        encoded_exp = exponent + params["bias"]
        if encoded_exp >= (1 << exponent_bits) - 1:
            return sign | special
        return sign | (encoded_exp << mantissa) | (significand - (1 << mantissa))

    def decode(self, code: int, *, scale=1) -> Real:
        if type(code) is not int or not 0 <= code < 1 << self.bits:
            raise ValueError("accumulator code outside declared width")
        scale = real(scale)
        if not isinstance(scale, Fraction) or scale <= 0:
            raise ValueError("accumulator scale must be finite and positive")
        negative = self.manifest["signed"] and code >> (self.bits - 1)
        if self.family in {"integer", "fixed_point"}:
            raw = code - (1 << self.bits) if negative else code
            numeric = self.manifest["numeric"]
            return Fraction(raw - numeric.get("zero_point", 0)) * pow2(-numeric["fractional_bits"]) * scale
        params = self.manifest["float"]
        mantissa = params["mantissa_bits"]
        exponent = (code >> mantissa) & ((1 << params["exp_bits"]) - 1)
        fraction = code & ((1 << mantissa) - 1)
        if exponent == (1 << params["exp_bits"]) - 1:
            return Decimal("NaN") if fraction else Decimal("-Infinity" if negative else "Infinity")
        if exponent == 0 and fraction == 0:
            return Decimal("-0") if negative else Fraction(0)
        significand = fraction if exponent == 0 else (1 << mantissa) + fraction
        return (-1 if negative else 1) * significand * pow2(max(1, exponent) - params["bias"] - mantissa) * scale

    def rounded(self, value, *, scale=1) -> Real:
        return self.decode(self.encode(value, scale=scale), scale=scale)


@lru_cache(maxsize=64)
def format_named(name: str):
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name):
        raise ValueError("invalid format name")
    base = ROOT / "formats/manifests"
    for folder in ("accepted", "accumulators"):
        path = base / folder / f"{name}.json"
        if path.is_file():
            manifest = load_manifest(path)
            return Accumulator(manifest) if folder == "accumulators" else NumberFormat(manifest)
    raise ValueError(f"unknown format: {name}")


def decode(fmt, code: int, scale=1) -> Real:
    return real(fmt.decode(code, scale=decimal(real(scale))))


def encode(fmt, value, scale=1) -> int:
    return fmt.encode(value if isinstance(fmt, Accumulator) else decimal(real(value)), scale=decimal(real(scale)))


def model_c(pairs: Iterable[tuple[Real, Real]], accumulator: Accumulator, *, bias=None, scale=1) -> int:
    """One round after each exact multiply-add; bias is stored then added once."""
    state = accumulator.encode(0, scale=scale)
    for activation, weight in pairs:
        product = binary(real(activation), real(weight), "mul")
        state = accumulator.encode(binary(accumulator.decode(state, scale=scale), product, "add"), scale=scale)
    if bias is not None:
        stored_bias = accumulator.rounded(bias, scale=scale)
        state = accumulator.encode(binary(accumulator.decode(state, scale=scale), stored_bias, "add"), scale=scale)
    return state
