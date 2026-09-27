import numpy as np
import pytest


def test_detector_graph_matches_folded_model_and_head_observation():
    torch = pytest.importorskip('torch')
    pytest.importorskip('ultralytics')
    from tools.experiment_b.classifier import configure
    from tools.experiment_b_ext.detector import DetectorEngine, DetectorObserver, load_detector
    configure('cpu')
    graph, constants, folded = load_detector('cpu')
    engine = DetectorEngine(graph, constants, 'cpu')
    for value in (0.0, 0.2):
        inputs = torch.full((1, 3, 640, 640), value)
        with torch.inference_mode():
            actual = engine.run(inputs)
            expected = folded(inputs)[0]
        assert actual.shape == (1, 84, 8400)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    observer = DetectorObserver()
    joined = torch.zeros(1, 84, 2, 2)
    joined[:, :4] = 1000
    joined[:, 4:] = .7
    observer.record('head', joined, 0)
    assert observer.maxima['head:boxes'] == 1000
    assert observer.maxima['head:scores'] == pytest.approx(.7)
    assert observer.channels['head:boxes'] == 4
    assert observer.channels['head:scores'] == 80


def test_extension_inventory_is_exact_complement_of_prior_runner():
    from tools.experiment_b.common import inventory as old_inventory
    from tools.experiment_b_ext.runner import inventory as extension_inventory
    old = old_inventory()['configurations']
    extension = extension_inventory()['tasks']
    def key(row):
        return row['model'], row['format'], row['recipe']
    assert {key(row) for row in extension} == {key(row) for row in old if row['blocked_reason']}
    assert len(extension) == 74
