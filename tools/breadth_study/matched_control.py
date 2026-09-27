"""Matched FP32 reduction control on the unchanged exact graph and quantizers.

Only the reduction state changes: sequential correctly rounded FP32 FMA/sums,
in the original reduction order and product domain. Bias encoding, activation,
scale application, patch quantization and output quantizers remain original.
This isolates accumulation precision, not cuDNN layout/order or B calibration.
"""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from math import prod

import numpy as np

from public.inference.native import NativeBackend
from public.inference.reference import operators as ref, detector
from public.inference.reference.arithmetic import binary, format_named, real, model_c
from public.inference.tensor import Tensor, Encoding, SharedEncoding, indices, offset
from tools.exact_execution_v2.engine import PreparedGraph, PreparedOperators, tensor_sha256, thaw

FP32 = 'fp32_e8m23_accumulator'


def native_conformance(backend):
    """Independent rational FP32 oracle, including cancellation and tiny scales."""
    from public.quantization.graph.executable import freeze_graph
    fp = format_named(FP32)
    native = FP32Native(backend)
    cases = [([Fraction(2**24), Fraction(1), Fraction(-2**24)], [Fraction(1)]*3),
             ([Fraction(1, 3), Fraction(-1, 7), Fraction(2, 5)], [Fraction(3, 11), Fraction(5), Fraction(-7)]),
             ([Fraction(1, 2**70), Fraction(-1, 2**69)], [Fraction(1, 2**70)]*2)]
    for accumulator in ('int32_accumulator', 'fp64_e11m52_accumulator'):
        for left, right in cases:
            expected = format_named(accumulator).encode(fp.decode(model_c(zip(left, right), fp)))
            actual = native.flex_gemm(left, right, batch=1, channels=1, k=len(left), accumulator=accumulator)
            if actual != (expected,):
                raise ValueError('matched native FP32 reduction disagrees with rational oracle')
    # Preserve product-domain scaling and original saturating bias/output store.
    x = Tensor((1, 3), (7, 8, 1), Encoding('int4', (Fraction(1, 3),)))
    w = Tensor((1, 3), (7, 7, 15), Encoding('int4', (Fraction(2, 7),), axis=0))
    # Graph JSON freezes decimal scales. Compare the frozen values, not an
    # unrepresentable infinite decimal retained only in the synthetic caller.
    x, w = Tensor.from_document(x.document()), Tensor.from_document(w.document())
    attrs = {'accumulator': 'int32_accumulator', 'output': Encoding('int4', (Fraction(1, 5),)).document(),
             'bias': ['0.123456'], 'activation': 'relu6'}
    graph = freeze_graph(inputs={'x': x.encoding.document()}, constants={'w': w},
                         nodes=[{'name': 'fc', 'op': 'linear', 'inputs': ['x', 'w'], 'attrs': attrs}],
                         outputs=['fc'], provenance={'kind': 'matched_control_unit_conformance'})
    expected = ref.linear(x, w, **{**attrs, 'output': Encoding('int4', (Fraction(1, 5),))})
    actual = MatchedControl(graph, backend=backend).execute({'x': x})['outputs']['fc']
    if actual != expected:
        raise ValueError('matched graph changed integer bias/store policy')
    return {'backend': backend, 'status': 'passed', 'reduction_cases': 6, 'matched_graph_cases': 1}


class FP32Native:
    """Use the certified native FP32 FMA and retain the original store boundary."""
    def __init__(self, backend):
        self.inner = NativeBackend(backend)
        self.last_strategy = None

    @staticmethod
    def cast_states(states, accumulator):
        acc = format_named(accumulator)
        if accumulator == FP32:
            return states
        floats = np.asarray(states, dtype=np.uint32).view(np.float32)
        if acc.family == 'integer' and acc.bits <= 64:
            # A representable integer float can be cast exactly. Use a strict
            # upper bound: INT64_MAX is rounded up when compared as float64.
            if (np.isfinite(floats).all() and np.equal(floats, np.trunc(floats)).all()
                    and (floats >= -(2.0 ** (acc.bits - 1))).all()
                    and (floats < 2.0 ** (acc.bits - 1)).all()):
                signed = floats.astype(np.int64)
                if acc.bits == 64:
                    return tuple(signed.view(np.uint64).tolist())
                return tuple((signed & ((1 << acc.bits) - 1)).tolist())
        fp = format_named(FP32)
        table = {code: acc.encode(fp.decode(code)) for code in set(states)}
        return tuple(table[code] for code in states)

    def flex_gemm(self, inputs, weights, **kwargs):
        original = kwargs['accumulator']
        states = self.inner.flex_gemm(inputs, weights, **{**kwargs, 'accumulator': FP32})
        self.last_strategy = 'matched_fp32_sequential_' + self.inner.last_strategy
        return self.cast_states(states, original)

    def flex_conv2d(self, inputs, weights, *, geometry, accumulator):
        states = self.inner.flex_conv2d(inputs, weights, geometry=geometry, accumulator=FP32)
        self.last_strategy = 'matched_fp32_sequential_' + self.inner.last_strategy
        return self.cast_states(states, accumulator)


def fp32_sum(values, scale=Fraction(1)):
    """Sequential sum in the same integer product/input domain as strict A."""
    fp = format_named(FP32)
    total = Fraction(0)
    for value in values:
        total = fp.rounded(binary(total, binary(value, Fraction(1)/scale, 'mul'), 'add'))
    return binary(total, scale, 'mul')


def residual(left, right, *, operation, output, accumulator=None, alignment=None):
    if operation != 'add':
        return ref.elementwise(left, right, operation=operation, output=output,
                               accumulator=accumulator, alignment=alignment)
    if left.shape != right.shape:
        raise ValueError('residual shapes differ')
    if isinstance(alignment, SharedEncoding):
        a = Tensor.quantize(left.values(), left.shape, alignment)
        b = Tensor.quantize(right.values(), right.shape, alignment)
        alignment = Encoding(alignment.format, tuple(max(x, y) for x, y in zip(a.encoding.scales, b.encoding.scales)),
                             alignment.axis, alignment.block_size)
    left = Tensor.quantize(left.values(), left.shape, alignment)
    right = Tensor.quantize(right.values(), right.shape, alignment)
    acc = format_named(accumulator)
    result = []
    for index in indices(left.shape):
        scale = alignment.scale_at(left.shape, index) if acc.family == 'integer' else Fraction(1)
        result.append(fp32_sum((left.value(index), right.value(index)), scale))
    return Tensor.quantize(result, left.shape, output)


def pool(inputs, *, kind, kernel_size, output, stride=None, padding=0,
         accumulator=None, count_include_pad=True):
    if kind == 'max':
        return ref.pool2d(inputs, kind=kind, kernel_size=kernel_size, output=output,
                          stride=stride, padding=padding, accumulator=accumulator,
                          count_include_pad=count_include_pad)
    if kind != 'average' or len(inputs.shape) != 4:
        raise ValueError('unsupported matched pool')
    kh, kw = ref.pair(kernel_size)
    sh, sw = ref.pair(kernel_size if stride is None else stride)
    ph, pw = ref.pair(padding)
    n, c, h, w = inputs.shape
    oh, ow = (h + 2*ph-kh)//sh+1, (w+2*pw-kw)//sw+1
    acc = format_named(accumulator)
    values = []
    for b, channel, row, col in indices((n, c, oh, ow)):
        window, scales = [], set()
        for kr, kc in indices((kh, kw)):
            y, x = row*sh-ph+kr, col*sw-pw+kc
            if 0 <= y < h and 0 <= x < w:
                address = (b, channel, y, x)
                window.append(inputs.value(address))
                scales.add(inputs.encoding.scale_at(inputs.shape, address))
            elif count_include_pad:
                window.append(Fraction(0))
        if not window or acc.family == 'integer' and len(scales) != 1:
            raise ValueError('invalid matched pool domain')
        scale = next(iter(scales)) if acc.family == 'integer' else Fraction(1)
        # Keep the original division policy after changing only the sum.
        divided = binary(fp32_sum(window, scale), Fraction(1, len(window)), 'mul')
        values.append(divided if acc.family == 'integer' else acc.rounded(divided))
    return Tensor.quantize(values, (n, c, oh, ow), output)


def softmax(inputs, *, axis, accumulator, output):
    acc = format_named(accumulator)
    shape = inputs.shape
    result = [None]*len(inputs.codes)
    for index in indices(shape[:axis]+shape[axis+1:]):
        addresses = [index[:axis]+(i,)+index[axis:] for i in range(shape[axis])]
        values = [inputs.value(a) for a in addresses]
        if any(isinstance(v, Decimal) and not v.is_finite() for v in values):
            raise ValueError('softmax requires finite logits')
        maximum = max(values)
        exponentials = [acc.rounded(detector._exp_difference(acc.rounded(binary(v, -maximum, 'add')))) for v in values]
        total = fp32_sum(exponentials)
        if not total:
            raise ValueError('matched softmax reduction underflow')
        for address, value in zip(addresses, exponentials):
            result[offset(shape, address)] = acc.rounded(Fraction(value)/Fraction(total))
    return Tensor.quantize(result, shape, output)


def dfl(inputs, weights, *, bins, accumulator, output):
    n, _, h, w = inputs.shape
    if inputs.shape[1] != 4*bins or weights.shape not in {(1, bins), (1, bins, 1, 1)}:
        raise ValueError('DFL shape mismatch')
    if inputs.encoding.block_size:
        logits = Tensor.quantize(inputs.values(), (n, 4, bins, h, w), SharedEncoding(inputs.encoding.format, axis=2))
    elif inputs.encoding.axis is not None:
        raise ValueError('DFL requires aligned scalar encoding')
    else:
        logits = Tensor((n, 4, bins, h, w), inputs.codes, inputs.encoding)
    prob_encoding = SharedEncoding(output.format, axis=2) if isinstance(output, SharedEncoding) else output
    probabilities = softmax(logits, axis=2, accumulator=accumulator, output=prob_encoding)
    fp = format_named(FP32)
    result = []
    for b, side, y, x in indices((n, 4, h, w)):
        pairs = ((probabilities.value((b, side, i, y, x)), weights.value((0, i) if len(weights.shape) == 2 else (0, i, 0, 0)))
                 for i in range(bins))
        result.append(fp.decode(model_c(pairs, fp)))
    return Tensor.quantize(result, (n, 4, h, w), output)


class MatchedControl(PreparedGraph):
    def __init__(self, document, *, backend='cpp'):
        super().__init__(document, backend=backend)
        self._operators.native = FP32Native(backend)

    def execute(self, inputs, *, observer=None):
        if set(inputs) != set(self._inputs):
            raise ValueError('graph input names mismatch')
        for name, expected in self._inputs.items():
            actual = inputs[name].encoding
            compatible = (actual.format == expected.format and actual.axis == expected.axis and actual.block_size == expected.block_size
                          if isinstance(expected, SharedEncoding) else actual == expected)
            if not compatible:
                raise ValueError('graph input encoding mismatch')
        values = {**inputs, **self._constants}
        traces, modes = {}, {}
        for name, op, edges, frozen_attrs, frozen_node in self._nodes:
            attrs, node = thaw(frozen_attrs), thaw(frozen_node)
            args = [values[edge] for edge in edges]
            if observer is not None and hasattr(observer, 'begin_node'):
                observer.begin_node(node)
            if op in {'linear', 'conv2d', 'depthwise_conv2d', 'block_conv2d'}:
                value = getattr(self._operators, op)(*args, **attrs)
                modes[name] = self._operators.native.last_strategy
            elif op == 'elementwise':
                # Two <=8-bit integer codes sum exactly in FP32 in their common
                # scale domain. Reuse the certified LUT and its diagnostics.
                if all(self._residuals.supported(e) for e in (args[0].encoding, args[1].encoding, attrs.get('alignment'), attrs['output'])):
                    value = self._residuals.add(*args, **attrs)
                else:
                    value = residual(*args, **attrs)
            elif op == 'pool2d':
                value = pool(*args, **attrs)
            elif op == 'adaptive_average_pool2d':
                if ref.pair(attrs.get('output_size', 1)) != (1, 1):
                    raise ValueError('only global average pool is supported')
                value = pool(args[0], kind='average', kernel_size=args[0].shape[-2:],
                             output=attrs['output'], accumulator=attrs['accumulator'])
            elif op in {'dfl', 'softmax'}:
                value = (dfl if op == 'dfl' else softmax)(*args, **attrs)
            elif op == 'reshape':
                if args[0].encoding.axis is not None or prod(attrs['shape']) != prod(args[0].shape):
                    raise ValueError('invalid reshape')
                value = Tensor(tuple(attrs['shape']), args[0].codes, args[0].encoding)
            elif op == 'lut':
                if args[0].encoding != attrs.pop('input_encoding'):
                    raise ValueError('LUT input mismatch')
                function = attrs.pop('function')
                if tuple(attrs['table']) != ref.nonlinear_lut(args[0].encoding, attrs['output'], function):
                    raise ValueError('LUT contents mismatch')
                value = ref.apply_lut(*args, **attrs)
            else:
                value = getattr(detector if hasattr(detector, op) else ref, op)(*args, **attrs)
            values[name] = value
            traces[name] = {'shape': list(value.shape), 'encoding': value.encoding.document(), 'sha256': tensor_sha256(value)}
            if observer is not None:
                observer(node, value)
        return {'outputs': {name: values[name] for name in self._outputs}, 'layers': traces,
                'backend': self.backend, 'execution_modes': modes, 'graph_sha256': self.graph_sha256}
