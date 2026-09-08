from __future__ import annotations

from public.workloads.models.identity import classification_metrics, graph_identity


def test_graph_identity_is_order_sensitive_and_key_order_stable() -> None:
    one = graph_identity(architecture="net", graph_version="v1", nodes=[{"op": "conv", "name": "a"}])
    same = graph_identity(architecture="net", graph_version="v1", nodes=[{"name": "a", "op": "conv"}])
    changed = graph_identity(architecture="net", graph_version="v1", nodes=[{"op": "relu", "name": "a"}])
    assert one == same
    assert one != changed


def test_classification_metrics() -> None:
    rows = [
        {"ground_truth": 1, "top5_predictions": [1, 2, 3, 4, 5]},
        {"ground_truth": 3, "top5_predictions": [2, 3, 4, 5, 6]},
    ]
    assert classification_metrics(rows) == {"count": 2, "top1_percent": 50.0, "top5_percent": 100.0}


def test_folded_graph_identifies_connections_parameters_and_tensor_content():
    import pytest
    torch = pytest.importorskip("torch")
    from public.workloads.models.deployment import prepare_deployment, capture_graph
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4), torch.nn.ReLU()).eval()
    folded, nodes, evidence = prepare_deployment(model, [1, 3, 8, 8])
    assert evidence["remaining_batchnorm"] == 0
    assert not any("batch_norm" in row["op"] for row in nodes)
    assert any(row["op"] == "aten::_convolution" for row in nodes)
    folded[0].stride = (2, 2)
    assert capture_graph(folded, torch.zeros(1, 3, 8, 8)) != nodes
    folded[0].stride = (1, 1)
    with torch.no_grad():
        folded[0].weight.add_(0.25)
    assert capture_graph(folded, torch.zeros(1, 3, 8, 8)) != nodes
