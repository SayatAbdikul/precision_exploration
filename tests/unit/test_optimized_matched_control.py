from types import SimpleNamespace

import numpy as np

from public.analysis.phase3.diagnostics import sample_indices
from public.inference.reference.arithmetic import format_named
from public.inference.tensor import Encoding, Tensor, QUANTIZATION_OBSERVER
from public.quantization.graph.executable import freeze_graph
from tools.breadth_study.matched_control_v2 import MatchedControl, residual
from tools.exact_execution_v3.optimized_matched_control import (
    MatchedFloatTables, OptimizedMatchedControl, compare_first_image, optimized_control_conformance)


def graph_and_input(name='fp6_e2m3'):
    encoding = Encoding(name)
    fmt = format_named(name)
    x = Tensor((1, 1, 2, 2), (1, 1 << (fmt.bits - 1), 3, 4), encoding)
    w = Tensor((1, 1, 1, 1), (8,), encoding)
    graph = freeze_graph(inputs={'x': encoding.document()}, constants={'w': w}, nodes=[
        {'name': 'conv', 'op': 'conv2d', 'inputs': ['x', 'w'], 'attrs': {
            'accumulator': 'fp64_e11m52_accumulator', 'output': encoding.document(),
            'bias': ['0.125'], 'activation': 'relu'}},
        {'name': 'add', 'op': 'elementwise', 'inputs': ['conv', 'x'], 'attrs': {
            'operation': 'add', 'accumulator': 'fp64_e11m52_accumulator',
            'alignment': encoding.document(), 'output': encoding.document()}},
        {'name': 'pool', 'op': 'pool2d', 'inputs': ['add'], 'attrs': {
            'kind': 'average', 'kernel_size': 2, 'output': encoding.document(),
            'accumulator': 'fp64_e11m52_accumulator'}}],
        outputs=['pool'], provenance={'kind': 'matched_v3_unit'})
    return graph, {'x': x}


def test_optimized_control_conformance_covers_all_finite_pairs_events_and_pool():
    result = optimized_control_conformance('cpp')
    assert result['status'] == 'passed'
    assert result['native_graphs_with_conv_residual_pool'] == 3
    assert result['finite_residual_pairs'] == {'fp6_e2m3': 62**2, 'fp6_e3m2': 62**2, 'fp7_e3m3': 126**2}


def test_special_residual_uses_original_matched_path_without_extra_event():
    encoding = Encoding('fp6_e2m3')
    fmt = format_named(encoding.format)
    nan = next(code for code in range(1 << fmt.bits) if fmt.decode(code).is_nan())
    left = Tensor((1, 2), (0, 1), encoding)
    right = Tensor((1, 2), (nan, 2), encoding)
    attrs = dict(operation='add', output=encoding, accumulator='fp64_e11m52_accumulator', alignment=encoding)
    def observed(call):
        events = []
        token = QUANTIZATION_OBSERVER.set(lambda values, tensor: events.append((tuple(str(v) for v in values), tensor)))
        try:
            result = call()
        finally:
            QUANTIZATION_OBSERVER.reset(token)
        return result, events
    assert observed(lambda: MatchedFloatTables().add(left, right, **attrs)) == observed(
        lambda: residual(left, right, **attrs))


def test_first_image_comparator_checks_layers_and_original_diagnostic_events():
    graph, inputs = graph_and_input()
    tensors = {}
    class Capture:
        def begin_node(self, node):
            pass
        def __call__(self, node, tensor):
            tensors[node['name']] = tensor
    MatchedControl(graph, backend='cpp').execute(inputs, observer=Capture())
    references = SimpleNamespace(limit=4096, records={})
    for name, tensor in tensors.items():
        selected = sample_indices(len(tensor.codes))
        values = np.asarray([float(tensor.value(tuple(int(v) for v in np.unravel_index(int(i), tensor.shape))))
                             for i in selected], dtype=np.float64)
        references.records[name] = {'shape': tensor.shape, 'indices': selected,
                                    'values': values, 'population': len(tensor.codes)}
    assert compare_first_image(graph, inputs, backend='cpp', references=references)['status'] == 'passed'
    assert OptimizedMatchedControl(graph, backend='cpp').execute(inputs)['outputs'] == MatchedControl(
        graph, backend='cpp').execute(inputs)['outputs']
