from public.analysis.phase3.accumulator_probes import hard_activation_probe, uniform_softmax_probe


def domain(scale):
    return {"format": "int8", "scales": [str(scale)]}


def test_wider_integer_still_erases_hard_sigmoid_fraction():
    result = hard_activation_probe(domain("0.125"), domain("0.0078125"), "hard_sigmoid", "int32_accumulator")
    assert result["requires_precision_revision"]
    assert result["variants"]["int64_accumulator"]["mismatching_codes"] > 0
    assert result["variants"]["fp64_e11m52_accumulator"]["mismatching_codes"] == 0


def test_uniform_softmax_separates_accumulator_and_representation_loss():
    result = uniform_softmax_probe(domain("0.125"), domain("0.0625"), 16, "int64_accumulator")
    assert result["requires_precision_revision"]
    assert result["exact_output_codes"] == [1]*16
    assert result["variants"]["int64_accumulator"]["output_codes"] == [0]*16
    assert result["variants"]["fp64_e11m52_accumulator"]["stored_probability_sum"] == "1"
    coarse = uniform_softmax_probe(domain("0.125"), domain("1"), 16, "int64_accumulator")
    assert coarse["output_representation_already_zeros_uniform_probabilities"]
    assert not coarse["requires_precision_revision"]
