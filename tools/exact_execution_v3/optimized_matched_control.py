"""Versioned accelerated matched-FP32 graph with unchanged reduction semantics.

Only the finite FP64 *post*-store and finite unscaled floating residual are
replaced.  MACs remain the frozen sequential FP32 native reduction, cast to the
original accumulator domain before bias/storage.  Pooling and all unsupported
operators retain matched_control_v2 semantics.
"""
from __future__ import annotations

from decimal import Decimal

import numpy as np

from public.inference.reference.arithmetic import binary, decode, encode, format_named, real
from public.inference.tensor import Encoding, QUANTIZATION_OBSERVER, Tensor
from tools.breadth_study import matched_control_v2 as matched
from tools.exact_execution_v2.activation_trace import ActivationTrace
from tools.exact_execution_v2.engine import IntegerResiduals, PreparedOperators
from tools.exact_execution_v3.postops import DIRECT_FORMATS, FP64, FP64Stores, FloatingResiduals


def _finite(value):
    return not isinstance(value, Decimal) or value.is_finite()


class MatchedFloatTables(FloatingResiduals):
    """Same alignment events as v2; sum table generated from the FP32 oracle."""

    def _sum(self, alignment, accumulator, output):
        key = alignment, accumulator, output
        if key not in self.sums:
            fmt = format_named(alignment.format)
            destination = format_named(output.format)
            fp32 = format_named(matched.FP32)
            count = 1 << fmt.bits
            decoded = tuple(decode(fmt, code) for code in range(count))
            values, codes = [], []
            for a in decoded:
                for b in decoded:
                    if _finite(a) and _finite(b):
                        value = fp32.rounded(binary(a, b, 'add'))
                        values.append(value)
                        codes.append(encode(destination, value))
                    else:
                        values.append(None)
                        codes.append(0)
            table = np.asarray(codes, dtype=np.uint8)
            table.flags.writeable = False
            self.sums[key] = tuple(values), table
        return self.sums[key]

    def add(self, left, right, *, operation, output, accumulator=None, alignment=None):
        if not FloatingResiduals.supported(left, right, operation, output, accumulator, alignment):
            return matched.residual(left, right, operation=operation, output=output,
                                    accumulator=accumulator, alignment=alignment)
        for tensor in (left, right):
            decoded = self._alignment(tensor.encoding, alignment)[0]
            if any(not _finite(decoded[int(code)]) for code in set(tensor.codes)):
                return matched.residual(left, right, operation=operation, output=output,
                                        accumulator=accumulator, alignment=alignment)
        # The table is generated from the matched FP32 oracle.  The parent's
        # alignment observer emits exactly the v2 left, right and sum events.
        return super().add(left, right, operation=operation, output=output,
                           accumulator=accumulator, alignment=alignment)


class MatchedResiduals:
    """Adapter for MatchedControl.execute's single-encoding admission check."""

    def __init__(self):
        self.integers = IntegerResiduals()
        self.floats = MatchedFloatTables()

    @staticmethod
    def supported(encoding):
        return (IntegerResiduals.supported(encoding)
                or isinstance(encoding, Encoding) and encoding.format in DIRECT_FORMATS
                and encoding.axis is None and encoding.block_size is None
                and len(encoding.scales) == 1 and encoding.scales[0] == 1)

    def add(self, left, right, *, operation, output, accumulator=None, alignment=None):
        if FloatingResiduals.supported(left, right, operation, output, accumulator, alignment):
            return self.floats.add(left, right, operation=operation, output=output,
                                   accumulator=accumulator, alignment=alignment)
        if (operation == 'add' and accumulator in {'int32_accumulator', 'int64_accumulator'}
                and all(IntegerResiduals.supported(e) for e in (left.encoding, right.encoding, alignment, output))):
            return self.integers.add(left, right, operation=operation, output=output,
                                     accumulator=accumulator, alignment=alignment)
        return matched.residual(left, right, operation=operation, output=output,
                                accumulator=accumulator, alignment=alignment)


class OptimizedControlOperators(PreparedOperators):
    def __init__(self, backend, library, constants):
        super().__init__(backend, library, constants)
        self.fp64_stores = FP64Stores()

    def _store(self, states, shape, accumulator, scales, bias, activation, output):
        result = self.fp64_stores.store(states, shape, accumulator, scales, bias, activation, output)
        return result if result is not None else super()._store(states, shape, accumulator, scales, bias, activation, output)


class OptimizedMatchedControl(matched.MatchedControl):
    """Drop-in versioned matched control; graph/pool/native FP32 path unchanged."""

    def __init__(self, document, *, backend='cpp'):
        super().__init__(document, backend=backend)
        fp32_native = self._operators.native
        self._operators = OptimizedControlOperators(backend, None, self._constants)
        self._operators.native = fp32_native
        self._residuals = MatchedResiduals()


def optimized_control_conformance(backend):
    """Small native graph plus exhaustive finite residual/event parity with v2.

    The frozen v2 `native_conformance` remains the independent reduction gate;
    call both before an eight-image admission.  This helper never accepts an
    image checkpoint and does not substitute for that image gate.
    """
    from public.quantization.graph.executable import freeze_graph
    pair_counts = {}
    for name in sorted(DIRECT_FORMATS):
        encoding = Encoding(name)
        fmt = format_named(name)
        finite = [code for code in range(1 << fmt.bits) if _finite(fmt.decode(code))]
        left = Tensor((1, len(finite)**2), tuple(a for a in finite for _ in finite), encoding)
        right = Tensor(left.shape, tuple(b for _ in finite for b in finite), encoding)
        attrs = dict(operation='add', output=encoding, accumulator=FP64, alignment=encoding)
        def observed(call):
            events = []
            token = QUANTIZATION_OBSERVER.set(lambda values, tensor: events.append(
                (tuple(str(real(value.item() if isinstance(value, np.generic) else value)) for value in values), tensor)))
            try:
                result = call()
            finally:
                QUANTIZATION_OBSERVER.reset(token)
            return result, events
        expected = observed(lambda: matched.residual(left, right, **attrs))
        actual = observed(lambda: MatchedFloatTables().add(left, right, **attrs))
        if actual != expected or len(actual[1]) != 3:
            raise ValueError(f'optimized matched {name} finite residual/event table differs from v2')
        pair_counts[name] = len(finite)**2
        x = Tensor((1, 1, 2, 2), (1, 1 << (fmt.bits-1), 3, 4), encoding)
        w = Tensor((1, 1, 1, 1), (8,), encoding)
        graph = freeze_graph(inputs={'x': encoding.document()}, constants={'w': w}, nodes=[
            {'name': 'conv', 'op': 'conv2d', 'inputs': ['x', 'w'],
             'attrs': {'accumulator': FP64, 'output': encoding.document(), 'bias': ['0.125'], 'activation': 'relu'}},
            {'name': 'add', 'op': 'elementwise', 'inputs': ['conv', 'x'], 'attrs': attrs | {'output': encoding.document(), 'alignment': encoding.document()}},
            {'name': 'pool', 'op': 'pool2d', 'inputs': ['add'],
             'attrs': {'kind': 'average', 'kernel_size': 2, 'output': encoding.document(), 'accumulator': FP64}}],
            outputs=['pool'], provenance={'kind': 'optimized_matched_control_conformance'})
        original = observed(lambda: matched.MatchedControl(graph, backend=backend).execute({'x': x}))
        accelerated = observed(lambda: OptimizedMatchedControl(graph, backend=backend).execute({'x': x}))
        if accelerated != original:
            raise ValueError(f'optimized matched {name} native graph differs from v2')
    return {'status': 'passed', 'backend': backend, 'finite_residual_pairs': pair_counts,
            'native_graphs_with_conv_residual_pool': len(DIRECT_FORMATS),
            'matches_frozen_v2_layers_outputs_and_all_quantizer_events': True}


def compare_first_image(document, inputs, *, backend, references):
    """Bounded full-graph v2/v3 comparison on one image with original diagnostics.

    `references` is the FP32 ReferenceSamples object for this same image.
    Each arm starts with a cold activation cache.  The caller decides whether
    to persist the resulting evidence under a separately sealed run identity.
    """
    from public.analysis.phase3.diagnostics import LayerDiagnostics
    from tools.run.exact_execution import event_signature
    rows = []
    for engine in (matched.MatchedControl(document, backend=backend),
                   OptimizedMatchedControl(document, backend=backend)):
        diagnostics = LayerDiagnostics(references)
        with ActivationTrace():
            token = QUANTIZATION_OBSERVER.set(diagnostics.quantization)
            try:
                result = engine.execute(inputs, observer=diagnostics)
            finally:
                QUANTIZATION_OBSERVER.reset(token)
        rows.append((result, diagnostics.records))
    old, new = rows
    if (old[0]['graph_sha256'] != new[0]['graph_sha256'] or old[0]['layers'] != new[0]['layers']
            or old[0]['execution_modes'] != new[0]['execution_modes']
            or old[0]['outputs'] != new[0]['outputs']
            or old[1] != new[1]
            or event_signature(old[1]) != event_signature(new[1])):
        raise ValueError('optimized matched control differs from frozen v2 on the first image')
    return {'status': 'passed', 'backend': backend, 'graph_sha256': old[0]['graph_sha256'],
            'layers': len(old[0]['layers']), 'diagnostic_signature': event_signature(old[1])}
