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
