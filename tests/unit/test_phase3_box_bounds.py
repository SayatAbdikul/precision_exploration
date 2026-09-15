from fractions import Fraction

from public.analysis.phase3.box_bounds import grid_margin, box_rounding_bound, prove_boxes
from public.inference.reference.arithmetic import format_named
from public.inference.reference.detector import decode_boxes
from public.inference.tensor import Tensor, Encoding


def test_affine_grid_margin_matches_all_distance_pairs_and_output_thresholds():
    scale, output = Fraction(17, 101), Fraction(139, 157)
    for stride in (1, 8):
        for anchor in (Fraction(1, 2), Fraction(7, 2)):
            for centered in (True, False):
                values = [stride*(anchor+(b-a)*scale/2 if centered else (b+a)*scale)
                          for a in range(-8, 8) for b in range(-8, 8)]
                expected = min(abs(value-(Fraction(code)+Fraction(1, 2))*output)
                               for value in values for code in range(-8, 7))
                actual = grid_margin(scale*stride/(2 if centered else 1), anchor*stride if centered else 0,
                                     -15 if centered else -16, 15 if centered else 14, output, 4)
                assert actual == expected


def test_staged_rounding_error_bound_includes_cancellation_and_stride():
    acc = format_named("fp64_e11m52_accumulator")
    scale = Fraction(170131, 1000007)
    for stride in (1, 8, 32):
        bound = box_rounding_bound(Fraction(159, 2), 128*scale, stride)
        assert bound["overflow_excluded"]
        for anchor in (Fraction(1, 2), Fraction(159, 2)):
            for left, right in ((-128, 127), (127, -128), (0, 0), (-128, -128), (127, 127)):
                x1, x2 = acc.rounded(anchor-left*scale), acc.rounded(anchor+right*scale)
                center = acc.rounded(acc.rounded(acc.rounded(x1+x2)/2)*stride)
                size = acc.rounded(acc.rounded(x2-x1)*stride)
                assert abs(center-stride*(anchor+(right-left)*scale/2)) <= bound["absolute_error_bound"]
                assert abs(size-stride*(right+left)*scale) <= bound["absolute_error_bound"]


def test_box_proof_keeps_ties_pending_and_predicts_reference_codes():
    assert prove_boxes(Fraction(1), Fraction(1), 4, 4, 1, 1, 1)["status"] == "pending"
    source, target = Encoding("int4", (Fraction(17, 101),)), Encoding("int4", (Fraction(139, 157),))
    # One spatial position for every code pair exercises center and size stores,
    # saturation, negative distances, and varying anchors in the actual operator.
    pairs = [(a, b) for a in range(16) for b in range(16)]
    tensor = Tensor((1, 4, 1, len(pairs)), tuple(v for j in (0, 0, 1, 1) for pair in pairs for v in [pair[j]]), source)
    proof = prove_boxes(source.scales[0], target.scales[0], 4, 4, 1, len(pairs), 8)
    assert proof["status"] == "exact_output_codes"
    actual = decode_boxes(tensor, stride=8, accumulator="fp64_e11m52_accumulator", output=target)
    exact = []
    for channel in range(4):
        for i in range(len(pairs)):
            l, r = tensor.value((0, 0, 0, i)), tensor.value((0, 2, 0, i))
            value = ((Fraction(2*i+1, 2) if channel == 0 else Fraction(1, 2))+(r-l)/2) if channel < 2 else r+l
            exact.append(value*8)
    assert actual.codes == Tensor.quantize(exact, tensor.shape, target).codes
