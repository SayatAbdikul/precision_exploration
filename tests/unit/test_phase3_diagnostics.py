import pytest
from public.analysis.phase3.diagnostics import paired_layer
from tools.run.phase2_engine import graph_witness
from public.quantization.graph.executable import execute


def test_identical_layer_samples_are_exact_without_non_json_infinity():
    row = paired_layer([0, 1, -1], [0, 1, -1], population_elements=30, sampled_indices=[0, 14, 29])
    assert row["mse"] == 0 and row["sqnr_db"] is None and row["sqnr_status"] == "exact"
    assert row["event_status"] == "requires_pre_store_instrumentation"


def test_zero_stores_are_not_falsely_labelled_underflow_events():
    row = paired_layer([1, 2], [0, 0], population_elements=2, sampled_indices=[0, 1])
    assert row["nonzero_reference_stored_zero_fraction"] == 1
    assert row["quantizer_event_counts"] is None


def test_graph_observation_preserves_exact_output_and_all_layer_hashes():
    graph, inputs = graph_witness()
    expected = execute(graph, inputs)
    seen = {}
    actual = execute(graph, inputs, observer=lambda node, value: seen.__setitem__(node["name"], value.document()))
    assert actual == expected
    assert set(seen) == set(actual["layers"])


def test_layer_reference_alignment_must_match():
    with pytest.raises(ValueError, match="shape/index"):
        paired_layer([1], [1, 2], population_elements=2, sampled_indices=[0])


def test_actual_quantizer_observation_catches_range_and_underflow_without_changing_codes():
    from fractions import Fraction
    from public.inference.tensor import Tensor, Encoding, QUANTIZATION_OBSERVER
    from public.analysis.phase3.diagnostics import ReferenceSamples, LayerDiagnostics
    values = [Fraction(100), Fraction(1, 1000), Fraction(-1)]
    expected = Tensor.quantize(values, (3,), Encoding("fp6_e3m2"))
    observer = LayerDiagnostics(ReferenceSamples())
    observer.begin_node({"name": "sample"})
    token = QUANTIZATION_OBSERVER.set(observer.quantization)
    try:
        actual = Tensor.quantize(values, (3,), Encoding("fp6_e3m2"))
    finally:
        QUANTIZATION_OBSERVER.reset(token)
    assert actual == expected
    assert observer.events["sample"]["overflow"] == 1
    assert observer.events["sample"]["underflow_to_zero"] == 1
    assert QUANTIZATION_OBSERVER.get() is None
