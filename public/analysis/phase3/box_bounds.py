"""Local FP64 box-decoding bounds for scalar-mapped signed integer distances.

Compare every possible stored distance pair and every fixed image-grid anchor
with exact rational box arithmetic. This says nothing about upstream DFL.
"""
from fractions import Fraction

from public.analysis.phase3.fp64_bounds import UNIT_ROUNDOFF, HALF_MIN_SUBNORMAL, MAX_FINITE


def grid_margin(step, origin, lower, upper, output_scale, output_bits):
    """Nearest output-store threshold to a finite affine integer grid."""
    if step <= 0 or output_scale <= 0 or lower > upper or output_bits < 2:
        raise ValueError("invalid box grid")
    margin = None
    for code in range(-(1 << (output_bits-1)), (1 << (output_bits-1))-1):
        threshold = (Fraction(code)+Fraction(1, 2))*output_scale
        position = (threshold-origin)/step
        floor = position.numerator//position.denominator
        for candidate in (floor, floor+1):
            index = max(lower, min(upper, candidate))
            distance = abs(origin+index*step-threshold)
            margin = distance if margin is None else min(margin, distance)
    return margin


def box_rounding_bound(maximum_anchor, maximum_distance, stride):
    """Propagate absolute errors through the actual rounded expression tree.

Each rounding contributes u*(exact magnitude + inherited error) + eta.
The half and stride multiplies include their prescribed rounding even when
they could be exact. Branch errors are added before rounding a sum/difference.
"""
    if maximum_anchor < 0 or maximum_distance < 0 or type(stride) is not int or stride < 1:
        raise ValueError("invalid box arithmetic bounds")
    magnitudes = []

    def rounded(magnitude, inherited):
        error = inherited+UNIT_ROUNDOFF*(magnitude+inherited)+HALF_MIN_SUBNORMAL
        magnitudes.append(magnitude+error)
        return error

    endpoint = maximum_anchor+maximum_distance
    endpoint_error = rounded(endpoint, Fraction(0))
    pair_error = rounded(2*endpoint, 2*endpoint_error)
    center_error = rounded(endpoint, pair_error/2)
    center_error = rounded(stride*endpoint, stride*center_error)
    size_error = rounded(stride*2*maximum_distance, stride*pair_error)
    return {"absolute_error_bound": max(center_error, size_error),
            "overflow_excluded": max(magnitudes) <= MAX_FINITE}


def prove_boxes(input_scale, output_scale, input_bits, output_bits, height, width, stride):
    if (any(type(n) is not int or n < 1 for n in (height, width, stride))
            or input_bits < 2 or output_bits < 2 or input_scale <= 0 or output_scale <= 0):
        raise ValueError("invalid box proof domain")
    lo, hi = -(1 << (input_bits-1)), (1 << (input_bits-1))-1
    anchor_max = Fraction(2*max(height, width)-1, 2)
    bound = box_rounding_bound(anchor_max, -lo*input_scale, stride)
    # Center = stride*(anchor + (right-left)*scale/2).
    # Width/height = stride*(right+left)*scale; the anchor cancels exactly.
    centers = [grid_margin(input_scale*stride/2, Fraction(2*i+1, 2)*stride,
                           lo-hi, hi-lo, output_scale, output_bits)
               for i in range(max(height, width))]
    size = grid_margin(input_scale*stride, Fraction(0), 2*lo, 2*hi, output_scale, output_bits)
    margin = min(min(centers), size)
    safe = bound["overflow_excluded"] and bound["absolute_error_bound"] < margin
    return {"status": "exact_output_codes" if safe else "pending",
            "absolute_error_bound": str(bound["absolute_error_bound"]),
            "boundary_margin": str(margin), "overflow_excluded": bound["overflow_excluded"],
            "anchor_positions_per_axis": [height, width], "stride": stride,
            "distance_code_pairs_per_anchor": (hi-lo+1)**2,
            "reason": "all finite stored distance pairs stay clear of output thresholds under staged FP64 rounding" if safe
                      else "box output thresholds or overflow need further validation"}
