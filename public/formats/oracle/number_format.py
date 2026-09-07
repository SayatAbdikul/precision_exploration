"""Correctness-first, manifest-driven high-precision datatype oracle."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
from typing import Any, Mapping

from .manifest import validate_manifest


class OracleError(ValueError):
    """Raised for undefined or unsupported oracle operations."""


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise OracleError("booleans are not numeric oracle inputs")
    if isinstance(value, float):
        return Decimal(str(value))
    return Decimal(value)


def _pow2(exponent: int) -> Decimal:
    return Decimal(2) ** exponent


class NumberFormat:
    """A small exact/high-precision reference implementation.

    Codes, never host container widths, define the datatype. Finite binary
    float, fixed-point, integer, and posit values are represented exactly by
    Decimal because all are dyadic rationals. Logarithmic values are evaluated
    with 200 decimal digits, which is sufficient high-precision oracle
    arithmetic for the accepted widths and is deterministic.
    """

    def __init__(self, manifest: Mapping[str, Any]):
        self.manifest = validate_manifest(manifest)
        self.name = self.manifest["name"]
        self.bits = self.manifest["bits"]
        self.family = self.manifest["family"]
        self.code_count = 1 << self.bits
        self._values = tuple(self._decode_code(code) for code in range(self.code_count))

    def _decode_code(self, code: int) -> Decimal:
        family = self.family
        if family in {"integer", "fixed_point", "mx_integer", "bfp"} and "numeric" in self.manifest:
            return self._decode_numeric(code)
        if family in {"float", "mx_float", "bfp"} and "float" in self.manifest:
            return self._decode_float(code)
        if family == "posit":
            return self._decode_posit(code)
        if family in {"logarithmic", "power_of_two"}:
            return self._decode_logarithmic(code)
        if family == "codebook" or "codebook" in self.manifest:
            values = self.manifest["codebook"]["values"]
            if code >= len(values):
                return Decimal("NaN")
            return _decimal(values[code])
        if family == "binary":
            if self.manifest["encoding"] == "binary_pm1":
                return Decimal(-1 if code == 0 else 1)
            return Decimal(code)
        if family == "ternary":
            return (Decimal(0), Decimal(1), Decimal(-1), Decimal("NaN"))[code]
        raise OracleError(f"unsupported family/encoding for {self.name}")

    def _decode_numeric(self, code: int) -> Decimal:
        numeric = self.manifest["numeric"]
        raw = code
        if self.manifest["signed"] and code & (1 << (self.bits - 1)):
            raw = code - (1 << self.bits)
        raw -= numeric.get("zero_point", 0)
        return Decimal(raw) / _pow2(numeric["fractional_bits"])

    def _decode_float(self, code: int) -> Decimal:
        params = self.manifest["float"]
        mantissa_bits = params["mantissa_bits"]
        exponent_bits = params["exp_bits"]
        mantissa_mask = (1 << mantissa_bits) - 1
        exponent_mask = (1 << exponent_bits) - 1
        mantissa = code & mantissa_mask
        exponent = (code >> mantissa_bits) & exponent_mask
        negative = bool(self.manifest["signed"] and code & (1 << (self.bits - 1)))
        sign = Decimal(-1 if negative else 1)

        if exponent == 0:
            if mantissa == 0 or not params["subnormals"]:
                return Decimal("-0") if negative else Decimal(0)
            significand = Decimal(mantissa) / _pow2(mantissa_bits)
            return sign * significand * _pow2(1 - params["bias"])

        if exponent == exponent_mask:
            if params["infinity"] and mantissa == 0:
                return Decimal("-Infinity") if negative else Decimal("Infinity")
            # IEEE-like encodings reserve every non-zero payload at the
            # maximum exponent for NaN. Finite-only minifloats in this project
            # reserve only the all-ones payload so the remaining top-binade
            # encodings stay usable.
            if params["nan"] and (
                (params["infinity"] and mantissa != 0) or mantissa == mantissa_mask
            ):
                return Decimal("NaN")

        significand = Decimal(1) + Decimal(mantissa) / _pow2(mantissa_bits)
        return sign * significand * _pow2(exponent - params["bias"])

    def _decode_posit(self, code: int) -> Decimal:
        if code == 0:
            return Decimal(0)
        nar = 1 << (self.bits - 1)
        if code == nar:
            return Decimal("NaN")

        negative = bool(code & nar)
        magnitude_bits = ((~code + 1) & (self.code_count - 1)) if negative else code
        payload = magnitude_bits & (nar - 1)
        cursor = self.bits - 2
        regime_bit = (payload >> cursor) & 1
        run = 0
        while cursor >= 0 and ((payload >> cursor) & 1) == regime_bit:
            run += 1
            cursor -= 1
        regime = run - 1 if regime_bit else -run
        if cursor >= 0:
            cursor -= 1

        es = self.manifest["posit"]["es"]
        exponent = 0
        for _ in range(es):
            exponent <<= 1
            if cursor >= 0:
                exponent |= (payload >> cursor) & 1
                cursor -= 1

        fraction = Decimal(1)
        place = Decimal("0.5")
        while cursor >= 0:
            if (payload >> cursor) & 1:
                fraction += place
            place /= 2
            cursor -= 1

        scale_exponent = regime * (1 << es) + exponent
        value = fraction * _pow2(scale_exponent)
        return -value if negative else value

    def _decode_logarithmic(self, code: int) -> Decimal:
        if code == 0:
            return Decimal(0)
        params = self.manifest["logarithmic"]
        payload_bits = self.bits - int(self.manifest["signed"])
        payload_mask = (1 << payload_bits) - 1
        payload = code & payload_mask
        negative = bool(self.manifest["signed"] and code & (1 << (self.bits - 1)))
        signed_exponent = payload - 1
        magnitude_bits = params["integer_bits"] + params["fractional_bits"]
        if magnitude_bits and signed_exponent & (1 << (magnitude_bits - 1)):
            signed_exponent -= 1 << magnitude_bits
        exponent = Decimal(signed_exponent) / _pow2(params["fractional_bits"])
        with localcontext() as context:
            context.prec = 200
            magnitude = Decimal(2) ** exponent
        return -magnitude if negative else magnitude

    def decode(self, code: int, *, scale: Any = 1) -> Decimal:
        if not isinstance(code, int) or not 0 <= code < self.code_count:
            raise OracleError(f"code {code!r} is outside {self.bits}-bit range")
        value = self._values[code]
        if value.is_finite():
            return value * _decimal(scale)
        return value

    def _nan_code(self) -> int | None:
        for code, value in enumerate(self._values):
            if value.is_nan():
                return code
        return None

    def _infinity_code(self, negative: bool) -> int | None:
        for code, value in enumerate(self._values):
            if value.is_infinite() and value.is_signed() == negative:
                return code
        return None

    def _finite_candidates(self, *, scale: Decimal) -> list[tuple[int, Decimal]]:
        return [
            (code, value * scale)
            for code, value in enumerate(self._values)
            if value.is_finite()
        ]

    def encode(self, value: Any, *, scale: Any = 1) -> int:
        target = _decimal(value)
        scale_value = _decimal(scale)
        if scale_value == 0 or not scale_value.is_finite():
            raise OracleError("scale must be finite and non-zero")

        if target.is_nan():
            code = self._nan_code()
            if code is None:
                raise OracleError(f"{self.name} has no NaN encoding")
            return code
        if target.is_infinite():
            code = self._infinity_code(target.is_signed())
            if code is not None:
                return code

        candidates = self._finite_candidates(scale=scale_value)
        if not candidates:
            raise OracleError(f"{self.name} has no finite encodings")

        if target.is_infinite():
            return min(candidates, key=lambda item: item[1])[0] if target.is_signed() else max(
                candidates, key=lambda item: item[1]
            )[0]

        minimum_code, minimum_value = min(candidates, key=lambda item: item[1])
        maximum_code, maximum_value = max(candidates, key=lambda item: item[1])
        if target < minimum_value:
            if self.manifest["overflow"] == "infinity":
                infinity = self._infinity_code(True)
                if infinity is not None:
                    return infinity
            return minimum_code
        if target > maximum_value:
            if self.manifest["overflow"] == "infinity":
                infinity = self._infinity_code(False)
                if infinity is not None:
                    return infinity
            return maximum_code

        rounding = self.manifest["rounding"]
        if rounding == "truncate":
            if target >= 0:
                eligible = [item for item in candidates if Decimal(0) <= item[1] <= target]
                return max(eligible or candidates, key=lambda item: item[1])[0]
            eligible = [item for item in candidates if target <= item[1] <= Decimal(0)]
            return min(eligible or candidates, key=lambda item: item[1])[0]

        distances = [(abs(candidate - target), code, candidate) for code, candidate in candidates]
        minimum = min(item[0] for item in distances)
        tied = [item for item in distances if item[0] == minimum]
        if len(tied) == 1:
            return tied[0][1]
        if rounding == "rne":
            even = [item for item in tied if item[1] & 1 == 0]
            if even:
                return min(even, key=lambda item: item[1])[1]
        # "nearest" resolves an exact tie away from zero; RNE falls back to
        # stable code order only for duplicate encodings such as flushed zeros.
        if rounding == "nearest":
            return max(tied, key=lambda item: abs(item[2]))[1]
        return min(tied, key=lambda item: item[1])[1]

    def add(self, left_code: int, right_code: int, *, scale: Any = 1) -> int:
        left = self.decode(left_code, scale=scale)
        right = self.decode(right_code, scale=scale)
        try:
            result = left + right
        except InvalidOperation:
            result = Decimal("NaN")
        return self.encode(result, scale=scale)

    def mul(self, left_code: int, right_code: int, *, scale: Any = 1) -> int:
        left = self.decode(left_code, scale=scale)
        right = self.decode(right_code, scale=scale)
        try:
            result = left * right
        except InvalidOperation:
            result = Decimal("NaN")
        return self.encode(result, scale=scale)

    def requantize(self, value: Any, *, scale: Any = 1) -> int:
        return self.encode(value, scale=scale)

    def convert(self, source_code: int, destination: "NumberFormat", *, source_scale: Any = 1, destination_scale: Any = 1) -> int:
        return destination.encode(self.decode(source_code, scale=source_scale), scale=destination_scale)
