from fractions import Fraction

import pytest

from public.inference.reference import operators as ref
from public.inference.tensor import Encoding, Tensor, QUANTIZATION_OBSERVER
from public.quantization.graph.executable import execute, freeze_graph
from tools.exact_execution_v2.engine import IntegerResiduals, PreparedGraph


def observed(call):
    rows = []
    token = QUANTIZATION_OBSERVER.set(lambda values, tensor: rows.append((tuple(values), tensor)))
    try:
        result = call()
    finally:
        QUANTIZATION_OBSERVER.reset(token)
    return result, rows


@pytest.mark.parametrize('bits', [4, 5, 6, 8])
def test_all_integer_residual_code_pairs_and_every_diagnostic_value(bits):
    n = 1 << bits
    left = Tensor((1, n*n), tuple(a for a in range(n) for _ in range(n)), Encoding(f'int{bits}', ('0.7',)))
    right = Tensor(left.shape, tuple(b for _ in range(n) for b in range(n)), Encoding(f'int{bits}', ('0.3',)))
    attrs = dict(operation='add', accumulator='int32_accumulator', alignment=Encoding(f'int{bits}', ('0.5',)),
                 output=Encoding(f'int{bits}', ('0.2',)))
    expected = observed(lambda: ref.elementwise(left, right, **attrs))
    engine = IntegerResiduals()
    assert observed(lambda: engine.add(left, right, **attrs)) == expected
    assert observed(lambda: engine.add(left, right, **attrs)) == expected


@pytest.mark.parametrize('scale', ['1/3', '0.0000000000000001', '10000000000000000'])
def test_residual_scale_extremes_and_nonterminating_rationals(scale):
    left = Tensor((1, 16), tuple(range(16)), Encoding('int4', (Fraction(scale),)))
    right = Tensor(left.shape, left.codes[::-1], Encoding('int4', ('0.5',)))
    attrs = dict(operation='add', accumulator='int64_accumulator', alignment=left.encoding, output=right.encoding)
    assert observed(lambda: IntegerResiduals().add(left, right, **attrs)) == observed(lambda: ref.elementwise(left, right, **attrs))


@pytest.mark.parametrize('format_name,operation', [('fp6_e3m2', 'add'), ('int4', 'mul')])
def test_unadmitted_residuals_use_original_operator(format_name, operation):
    encoding = Encoding(format_name)
    tensor = Tensor((1, 4), (0, 1, 2, 3), encoding)
    attrs = dict(operation=operation, accumulator='fp32_e8m23_accumulator', alignment=encoding, output=encoding)
    assert observed(lambda: IntegerResiduals().add(tensor, tensor, **attrs)) == observed(lambda: ref.elementwise(tensor, tensor, **attrs))


def small_graph():
    encoding = Encoding('int4')
    graph = freeze_graph(inputs={'x': encoding.document()}, constants={'w': Tensor((2, 2), (1, 2, 3, 4), encoding)},
        nodes=[{'name': 'fc', 'op': 'linear', 'inputs': ['x', 'w'],
                'attrs': {'accumulator': 'int32_accumulator', 'output': encoding.document()}},
               {'name': 'add', 'op': 'elementwise', 'inputs': ['fc', 'x'],
                'attrs': {'operation': 'add', 'accumulator': 'int32_accumulator', 'alignment': encoding.document(),
                          'output': encoding.document()}}], outputs=['add'], provenance={'kind': 'synthetic_conformance'})
    return graph, {'x': Tensor((1, 2), (2, 15), encoding)}


def test_compiled_graph_matches_reference_and_owns_immutable_snapshot():
    document, inputs = small_graph()
    expected = observed(lambda: execute(document, inputs))
    graph = PreparedGraph(document)
    assert observed(lambda: graph.execute(inputs)) == expected
    document['constants']['w']['codes'][0] = 7
    document['nodes'][1]['attrs']['operation'] = 'mul'
    document['outputs'][:] = ['fc']
    assert observed(lambda: graph.execute(inputs)) == expected
    changed = {'x': Tensor((1, 2), (3, 4), inputs['x'].encoding)}
    original, _ = small_graph()
    assert graph.execute(changed) == execute(original, changed)


def test_validation_and_input_checks_remain_fail_closed():
    document, inputs = small_graph()
    graph = PreparedGraph(document)
    with pytest.raises(ValueError, match='input names'):
        graph.execute({})
    with pytest.raises(ValueError, match='input encoding'):
        graph.execute({'x': Tensor((1, 2), (1, 2), Encoding('int8'))})
    document['manifest_hashes']['int4'] = 'wrong'
    with pytest.raises(ValueError, match='manifest identity'):
        PreparedGraph(document)


def test_activation_diagnostic_cache_replay_matches_uninterrupted_execution():
    from tools.exact_execution_v2.activation_trace import ActivationTrace
    encoding=Encoding('int4', (Fraction(1,3),))
    first=Tensor((1,3),(0,1,15),encoding)
    second=Tensor((1,3),(1,2,14),encoding)
    def activate(tensor):
        return ref.activation(tensor,function='relu',output=encoding)
    with ActivationTrace() as trace:
        observed(lambda: activate(first))
        history=list(trace.calls)
        uninterrupted=observed(lambda: activate(second))
    with ActivationTrace() as trace:
        trace.replay(history)
        resumed=observed(lambda: activate(second))
    assert resumed==uninterrupted
    with ActivationTrace():
        cold=observed(lambda: activate(second))
    assert cold[0]==resumed[0]
    assert cold[1]!=resumed[1]  # The frozen oracle reports stores only on cache misses.


def test_native_constant_preparation_is_cached_without_caching_runtime_inputs():
    document,inputs=small_graph()
    graph=PreparedGraph(document,backend='cpp')
    assert graph.execute(inputs)==execute(document,inputs,backend='cpp')
    cached=tuple(graph._operators.prepared.values())
    assert len(cached)==1
    assert not cached[0].array.flags.writeable
    changed={'x':Tensor((1,2),(3,4),inputs['x'].encoding)}
    assert graph.execute(changed)==execute(document,changed,backend='cpp')
    assert tuple(graph._operators.prepared.values())==cached


def test_tensor_hash_matches_frozen_canonical_bytes_for_every_accepted_format():
    import hashlib
    from tools.experiment_b.common import formats
    from public.experiments.registry.identity import canonical_json_bytes
    from tools.exact_execution_v2.engine import tensor_sha256
    for fmt in formats():
        shared=fmt['family'] in {'bfp','mx_float'}
        n=1<<fmt['bits']
        size=max(32,n)
        encoding=(Encoding(fmt['name'],('1',)*((size+31)//32),1,32) if shared else Encoding(fmt['name']))
        tensor=Tensor((1,size),tuple(i%n for i in range(size)),encoding)
        assert tensor_sha256(tensor)==hashlib.sha256(canonical_json_bytes(tensor.document())).hexdigest()
