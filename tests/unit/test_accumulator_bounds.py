from tools.analysis.phase2_accumulator_bounds import bounds


def test_bounds_distinguish_storage_width_precision_and_non_dyadic_formats():
    integer = bounds("int8", 1024)
    assert integer["maximum_sum_magnitude"] == str(128*128*1024)
    assert integer["signed_grid_accumulator_bits"] == 26
    assert integer["sufficient_float_significand_bits"] == 25
    assert integer["fp32_covers_finite_unscaled_product_grid"] is False
    assert integer["fp64_covers_finite_unscaled_product_grid"] is True
    codebook = bounds("nf4", 9)
    assert codebook["dyadic"] is False
    assert "sufficient_float_significand_bits" not in codebook
