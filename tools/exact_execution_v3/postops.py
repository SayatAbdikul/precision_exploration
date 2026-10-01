"""Finite FP64 stores and oracle-derived floating residual tables.

These are optional operations for a separately versioned prepared graph.  The
unchanged scalar reference remains the fallback for every unsupported domain.
Quantization observation retains the original three residual events and exact
pre-store values; no FP32 conversion participates in the strict FP64 store.
"""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from math import prod

import numpy as np

from public.inference.reference import operators as ref
from public.inference.reference.arithmetic import binary, decode, encode, format_named, real
from public.inference.tensor import Encoding, Tensor, observe_quantization
from public.quantization.ptq.encoding import direct_float_tensor
from tools.exact_execution_v2.engine import IndexedValues, IntegerResiduals


DIRECT_FORMATS = frozenset({'fp6_e2m3', 'fp6_e3m2', 'fp7_e3m3'})
FP64 = 'fp64_e11m52_accumulator'


class FP64Stores:
    """Per-graph stored-bias cache, with exact reference fallback by caller."""

    def __init__(self):
        self.biases = {}

    def _bias(self, accumulator, bias):
        values = tuple(real(value) for value in bias)
        # Decimal signed zero equals Fraction zero, so value equality alone is
        # insufficient for a cache key.  Preserve its textual sign/kind.
        key = (accumulator, tuple((type(value).__name__, str(value)) for value in values))
        if key not in self.biases:
            acc = format_named(accumulator)
            codes = np.asarray([acc.encode(value) for value in values], dtype=np.uint64)
            codes.flags.writeable = False
            self.biases[key] = codes.view(np.float64)
        return self.biases[key]

    def store(self, states, shape, accumulator, scales, bias, activation, output):
        """Return None unless this is an admitted finite, unscaled FP64 store.

        `states` are the FP64 accumulator bit patterns produced by the native
        MAC.  Bias is independently encoded to FP64 and then added exactly once
        with IEEE RNE, matching Model C's separate bias boundary.
        """
        if (accumulator != FP64 or not isinstance(output, Encoding)
                or output.format not in DIRECT_FORMATS or output.axis is not None
                or output.block_size is not None or len(shape) < 2
                or activation not in {'identity', 'relu', 'relu6'}
                or len(scales) != shape[1] or any(scale != 1 for scale in scales)):
            return None
        output.validate_shape(shape)
        if bias is not None and len(bias) != shape[1]:
            raise ValueError('bias count must equal output channels')
        if len(states) != prod(shape):
            raise ValueError('FP64 accumulator state count does not match output shape')
        bits = np.asarray(states, dtype=np.uint64)
        values = bits.view(np.float64).reshape(shape)
        # Special values follow the scalar oracle.  In particular, NaN payload
        # canonicalization must not be inferred from host NumPy behavior.
        if not np.isfinite(values).all():
            return None
        if bias is not None:
            stored = self._bias(accumulator, bias)
            if not np.isfinite(stored).all():
                return None
            dims = (1, len(bias)) + (1,) * (len(shape) - 2)
            with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                values = np.add(values, stored.reshape(dims), dtype=np.float64)
            if not np.isfinite(values).all():
                return None
        if activation in {'relu', 'relu6'}:
            # The reference clamp chooses its +0 lower bound on either zero
            # sign.  np.where expresses that policy explicitly.
            values = np.where(values <= 0.0, np.float64(0.0), values)
            if activation == 'relu6':
                values = np.where(values >= 6.0, np.float64(6.0), values)
        return direct_float_tensor(values, output)


def _finite(value):
    return not isinstance(value, Decimal) or value.is_finite()


class FloatingResiduals:
    """Oracle-created finite FP6/FP7 alignment and add tables.

    Tables cache codes and exact pre-quantization values.  They are not used if
    any actually-present source or aligned code is nonfinite.  Those cases,
    other arithmetic, and non-target formats execute the original operator.
    """

    def __init__(self):
        self.integers = IntegerResiduals()
        self.alignments = {}
        self.sums = {}

    @staticmethod
    def supported(left, right, operation, output, accumulator, alignment):
        return (operation == 'add' and accumulator == FP64
                and all(isinstance(e, Encoding) for e in (left.encoding, right.encoding, alignment, output))
                and all(e.axis is None and e.block_size is None and e.scales == (Fraction(1),)
                        for e in (left.encoding, right.encoding, alignment, output))
                and len({left.encoding.format, right.encoding.format, alignment.format, output.format}) == 1
                and alignment.format in DIRECT_FORMATS)

    def _alignment(self, encoding, alignment):
        key = encoding, alignment
        if key not in self.alignments:
            source = format_named(encoding.format)
            target = format_named(alignment.format)
            values = tuple(decode(source, code) for code in range(1 << source.bits))
            table = np.asarray([encode(target, value) if _finite(value) else 0 for value in values], dtype=np.uint8)
            table.flags.writeable = False
            self.alignments[key] = values, table
        return self.alignments[key]

    def _sum(self, alignment, accumulator, output):
        key = alignment, accumulator, output
        if key not in self.sums:
            fmt = format_named(alignment.format)
            destination = format_named(output.format)
            acc = format_named(accumulator)
            count = 1 << fmt.bits
            values = tuple(decode(fmt, code) for code in range(count))
            sums, codes = [], []
            for a in values:
                for b in values:
                    if _finite(a) and _finite(b):
                        value = acc.rounded(binary(a, b, 'add'))
                        sums.append(value)
                        codes.append(encode(destination, value))
                    else:
                        sums.append(None)
                        codes.append(0)
            code_table = np.asarray(codes, dtype=np.uint8)
            code_table.flags.writeable = False
            self.sums[key] = tuple(sums), code_table
        return self.sums[key]

    def add(self, left, right, *, operation, output, accumulator=None, alignment=None):
        if not self.supported(left, right, operation, output, accumulator, alignment):
            return self.integers.add(left, right, operation=operation, output=output,
                                     accumulator=accumulator, alignment=alignment)
        if left.shape != right.shape:
            raise ValueError('elementwise shapes must match; broadcasting must be explicit')
        prepared = []
        for tensor in (left, right):
            values, table = self._alignment(tensor.encoding, alignment)
            indices = np.asarray(tensor.codes, dtype=np.uint8)
            if any(not _finite(values[int(code)]) for code in np.unique(indices)):
                return ref.elementwise(left, right, operation=operation, output=output,
                                       accumulator=accumulator, alignment=alignment)
            prepared.append((tensor, indices, values, table))
        source = []
        for tensor, indices, values, table in prepared:
            aligned = Tensor(tensor.shape, tuple(table[indices].tolist()), alignment)
            source.append(observe_quantization(IndexedValues(indices, values), aligned))
        a, b = source
        count = 1 << format_named(alignment.format).bits
        a_codes, b_codes = (np.asarray(tensor.codes, dtype=np.uint8) for tensor in (a, b))
        values, codes = self._sum(alignment, accumulator, output)
        pairs = a_codes.astype(np.uint16) * count + b_codes.astype(np.uint16)
        if any(values[int(index)] is None for index in np.unique(pairs)):
            # This cannot occur for a valid finite target table, but fail
            # closed before publishing an output quantization event.
            raise ValueError('finite residual alignment produced nonfinite code')
        result = Tensor(left.shape, tuple(codes[pairs].tolist()), output)
        return observe_quantization(IndexedValues(pairs, values), result)
