"""Readout helpers of the B2 reconstruction report (lane L6)."""
import numpy as np

from tools.experiment_b2_recon.report import credit, default_reference, nearest_counterpart


def test_credit_rules():
    arrays = {"label_among_maxima": np.array([1, 1, 0], dtype=np.uint8), "tie_size": np.array([1, 4, 2], dtype=np.uint16),
              "label_is_lowest_index_maximum": np.array([1, 0, 0], dtype=np.uint8),
              "top5_topk": np.array([[3, 1, 2, 4, 5], [7, 1, 2, 4, 5], [0, 1, 2, 4, 5]], dtype=np.int16)}
    labels = np.array([3, 1, 9])
    assert credit(arrays, labels, "expected").tolist() == [1.0, 0.25, 0.0]
    assert credit(arrays, labels, "lowest_index").tolist() == [1.0, 0.0, 0.0]
    assert credit(arrays, labels, "topk").tolist() == [1.0, 0.0, 0.0]


def test_nearest_counterpart_differs_in_the_rounding_only():
    learned = {"rounding": "learned"}
    assert nearest_counterpart("L-bc--w-int8--a-int8--default--fit-mse_per_channel-fp32in-s0", learned) == \
        "N-default--w-int8--a-int8--default"
    assert nearest_counterpart("L-nobc--w-int4--a-int8--default_no_bias_correction--mse_per_layer"
                               "--fit-mse_per_layer-fp32in-s2", learned) == \
        "N-nobc--w-int4--a-int8--default_no_bias_correction--mse_per_layer"
    assert nearest_counterpart("L-A32--w-int4--a-fp32--mse_per_channel--fit-mse_per_channel-fp32in-s0", learned) == \
        "N-A32--w-int4--a-fp32--mse_per_channel"
    assert nearest_counterpart("N-default--w-int8--a-int8--default", {"rounding": "nearest"}) is None


def test_default_reference_only_for_the_b2_grid():
    assert default_reference({"activation_format": "int8", "weight_scale_rule": "mse_per_channel", "weight_format": "int4"}) == \
        "N-default--w-int4--a-int8--default"
    assert default_reference({"activation_format": None, "weight_scale_rule": "mse_per_channel", "weight_format": "int4"}) is None
    assert default_reference({"activation_format": "int8", "weight_scale_rule": "mse_per_layer", "weight_format": "int4"}) is None
