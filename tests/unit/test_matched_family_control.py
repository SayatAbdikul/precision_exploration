from fractions import Fraction

from public.inference.reference.arithmetic import format_named, encode
from public.inference.tensor import Encoding, Tensor
from tools.breadth_study.matched_control_v2 import residual


def test_residual_cancellation_rounds_the_sum_once_without_casting_operands():
    fmt = format_named('nf4')
    domain = Encoding('nf4', (Fraction(1, 3),))
    left = Tensor((1,), (encode(fmt, Fraction(1)),), domain)
    right = Tensor((1,), (encode(fmt, Fraction(-1)),), domain)
    result = residual(left, right, operation='add', alignment=domain,
                      output=Encoding('int8', (Fraction(1, 2**30),)), accumulator='fp64_e11m52_accumulator')
    assert result.codes == (0,)
