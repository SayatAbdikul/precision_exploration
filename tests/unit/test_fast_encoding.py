import numpy as np
import pytest

from public.inference.reference.arithmetic import format_named
from public.inference.tensor import Encoding, Tensor
from public.quantization.ptq.encoding import direct_float_tensor


@pytest.mark.parametrize("name", ["fp4_e2m1", "fp5_e2m2", "fp6_e3m2", "fp6_e2m3", "fp7_e3m3", "fp8_e4m3fn", "fp8_e5m2"])
def test_every_boundary_and_neighbours_match_reference(name):
    fmt = format_named(name)
    finite = sorted(set(fmt.decode(code) for code in range(1 << fmt.bits) if fmt.decode(code).is_finite()))
    values = [0., -0., float("inf"), -float("inf"), float("nan")]
    for a,b in zip(finite,finite[1:]):
        midpoint = float((a+b)/2)
        values.extend([float(a),midpoint,np.nextafter(midpoint,-np.inf),np.nextafter(midpoint,np.inf)])
    values.extend([1e10,-1e10,60000.,61439.,61440.])
    encoding = Encoding(name)
    array = np.array(values, dtype=np.float64)
    assert direct_float_tensor(array, encoding) == Tensor.quantize(array.tolist(), array.shape, encoding)
