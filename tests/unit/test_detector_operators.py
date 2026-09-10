from fractions import Fraction

from public.inference.tensor import Tensor, Encoding
from public.inference.reference import detector


def test_detector_movement_and_quantized_dfl_decode():
    e = Encoding("fp6_e3m2")
    x = Tensor.quantize([1,2,3,4],(1,2,1,2),e)
    a = detector.channel_slice(x,start=0,stop=1,output=e)
    b = detector.channel_slice(x,start=1,stop=2,output=e)
    assert detector.concatenate(a,b,axis=1,output=e)==x
    up = detector.resize_nearest(a,factor=2,output=e)
    assert up.shape==(1,1,2,4) and list(up.values())==[1,1,2,2,1,1,2,2]
    logits = Tensor.quantize([0]*8,(1,8,1,1),e)
    weights = Tensor.quantize([0,1],(1,2,1,1),e)
    distances = detector.dfl(logits,weights,bins=2,accumulator="fp32_e8m23_accumulator",output=e)
    assert distances.values()==(Fraction(1,2),)*4
    boxes = detector.decode_boxes(distances,stride=2,accumulator="fp32_e8m23_accumulator",output=e)
    assert boxes.values()==(1,1,2,2)


def test_strict_fp6_native_pixel_coordinates_saturate_before_nms():
    """Native-resolution range failure must not trigger a hidden FP32 exception."""
    encoding = Encoding("fp6_e3m2")
    distances = Tensor.quantize([1] * (4 * 80), (1, 4, 1, 80), encoding)
    boxes = detector.decode_boxes(distances, stride=8,
        accumulator="fp32_e8m23_accumulator", output=encoding)
    # Exact centers are 4,12,...,636. The unscaled candidate ends at 24.
    assert [boxes.value((0, 0, 0, i)) for i in range(4)] == [4, 12, 20, 24]
    assert boxes.value((0, 0, 0, 79)) == 24
    assert [boxes.value((0, c, 0, 79)) for c in range(1, 4)] == [4, 16, 16]
    assert sum(boxes.value((0, 0, 0, i)) == 24 for i in range(80)) == 77


def test_strict_fp6_class_probability_underflow_is_explicit():
    from public.inference.reference.operators import nonlinear_lut
    from public.inference.reference.arithmetic import format_named
    encoding = Encoding("fp6_e3m2")
    fmt = format_named(encoding.format)
    sigmoid = nonlinear_lut(encoding, encoding, "sigmoid")
    # sigmoid(-3.5) < half the smallest subnormal; sigmoid(-3) > it.
    assert fmt.decode(sigmoid[fmt.encode(-3.5)]) == 0
    assert fmt.decode(sigmoid[fmt.encode(-3)]) == Fraction(1, 16)
