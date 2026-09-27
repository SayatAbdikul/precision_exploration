"""Exact integer output stores with rational thresholds and lazy diagnostics.

This optional execution extension leaves the frozen engine untouched. Production
use requires a per-graph eight-image compatibility certificate. Saturating bias
addition and integer threshold searches preserve sequential MAC results; no
floating-point approximation is used to choose an output code.
"""
from collections.abc import Sequence
from contextlib import contextmanager
from fractions import Fraction
from math import prod

import numpy as np

from public.inference.reference.arithmetic import format_named, real
from public.inference.tensor import Encoding, Tensor, observe_quantization


def supported_scale(scale):
    if not isinstance(scale, Fraction) or scale <= 0 or max(scale.numerator.bit_length(), scale.denominator.bit_length()) > 128:
        return False
    # Finite decimals ensure the frozen 200-digit oracle represents the
    # admitted raw values exactly before division by the destination scale.
    denominator = scale.denominator
    for factor in (2, 5):
        while denominator % factor == 0:
            denominator //= factor
    return denominator == 1


def signed(code, bits):
    return code - (1 << bits) if code >= 1 << (bits - 1) else code


class StoredValues(Sequence):
    """Exactly the reference pre-quantization values, evaluated on observation."""
    def __init__(self, states, shape, scales, activation):
        self.states, self.shape, self.scales, self.activation = states.ravel(), shape, scales, activation
        self.spatial = prod(shape[2:])
    def __len__(self):
        return len(self.states)
    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        channel = (index // self.spatial) % self.shape[1]
        value = int(self.states[index]) * self.scales[channel]
        if self.activation in {"relu", "relu6"}:
            value = max(value, Fraction(0))
        if self.activation == "relu6":
            value = min(value, Fraction(6))
        return value


def integer_store(states, shape, accumulator, scales, bias, activation, output):
    """Return None outside the certified domain, allowing the original store."""
    acc = format_named(accumulator)
    if not isinstance(output, Encoding):
        return None
    fmt = format_named(output.format)
    if (acc.family != "integer" or acc.bits not in (32, 64) or not acc.manifest["signed"] or
            fmt.family != "integer" or not 1 <= fmt.bits <= 8 or not fmt.manifest["signed"] or fmt.manifest["rounding"] != "rne" or
            fmt.manifest["numeric"].get("zero_point", 0) != 0 or fmt.manifest["numeric"]["fractional_bits"] != 0 or
            output.axis is not None or output.block_size is not None or len(shape) < 2 or
            activation not in {"identity", "relu", "relu6"} or len(scales) != shape[1] or
            not all(supported_scale(s) for s in (*scales, *output.scales))):
        return None
    if bias is not None and len(bias) != shape[1]:
        raise ValueError("bias count must equal output channels")
    if len(states) != prod(shape) or any(type(code) is not int or not 0 <= code < 1 << acc.bits for code in states):
        raise ValueError("accumulator code outside declared width/shape")
    output.validate_shape(shape)
    array = np.asarray(states, dtype=np.uint64).view(np.int64).reshape(shape[0], shape[1], -1).copy()
    if acc.bits == 32:
        array[array >= 1 << 31] -= 1 << 32
    lo, hi = -(1 << (acc.bits - 1)), (1 << (acc.bits - 1)) - 1
    minimum, maximum = -(1 << (fmt.bits - 1)), (1 << (fmt.bits - 1)) - 1
    mask, out_scale = (1 << fmt.bits) - 1, output.scales[0]
    codes = np.empty_like(array, dtype=np.uint8)
    for channel, scale in enumerate(scales):
        values = array[:, channel, :]
        if bias is not None:
            stored = signed(acc.encode(real(bias[channel]), scale=scale), acc.bits)
            if stored >= 0:
                # Clamp before adding, so the host int64 addition cannot overflow.
                values[:] = np.minimum(values, hi - stored) + stored
            else:
                values[:] = np.maximum(values, lo - stored) + stored
        cutoffs, skipped = [], 0
        for lower in range(minimum, maximum):
            threshold = Fraction(2 * lower + 1, 2) * out_scale / scale
            floor = threshold.numerator // threshold.denominator
            # Tie picks the even output code: lower odd admits its upper neighbor.
            first_upper = floor + (0 if lower & 1 and threshold.denominator == 1 else 1)
            if first_upper <= lo:
                skipped += 1
            elif first_upper <= hi:
                cutoffs.append(first_upper)
        stored_codes = np.searchsorted(np.asarray(cutoffs, dtype=np.int64), values, side="right") + minimum + skipped
        if activation in {"relu", "relu6"}:
            stored_codes = np.maximum(stored_codes, 0)
        if activation == "relu6":
            # Use the actual frozen oracle for this single channel-independent limit.
            from public.inference.reference.arithmetic import encode
            six = signed(encode(fmt, 6, out_scale), fmt.bits)
            stored_codes = np.minimum(stored_codes, six)
        codes[:, channel, :] = np.bitwise_and(stored_codes, mask).astype(np.uint8)
    tensor = Tensor(tuple(shape), tuple(codes.ravel().tolist()), output)
    return observe_quantization(StoredValues(array, shape, scales, activation), tensor)


@contextmanager
def accelerated_stores():
    from public.inference.operators.dispatch import Operators
    original = Operators._store
    def store(self, states, shape, accumulator, scales, bias, activation, output):
        result = integer_store(states, shape, accumulator, scales, bias, activation, output)
        return result if result is not None else original(self, states, shape, accumulator, scales, bias, activation, output)
    Operators._store = store
    try:
        yield
    finally:
        Operators._store = original
