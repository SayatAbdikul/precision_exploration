import numpy as np
import pytest


@pytest.mark.parametrize('name', ('bfp6', 'mxfp8_e4m3', 'mxfp6_e3m2', 'mxfp4_e2m1'))
@pytest.mark.parametrize('recipe', ('maxabs', 'percentile_99_9'))
def test_gpu_style_shared_qdq_matches_independent_numpy_policy(name, recipe):
    torch = pytest.importorskip('torch')
    from tools.experiment_b_ext.shared import BlockQuantizer, numpy_block_qdq
    rng = np.random.default_rng(1203)
    values = rng.normal(size=(9, 67)).astype(np.float32)
    values[0] = 0
    values[1, 31] = 100
    values[2, 34] = -70
    actual = BlockQuantizer(name, recipe, 'cpu')(torch.from_numpy(values)).numpy()
    expected = numpy_block_qdq(values, name, recipe)
    np.testing.assert_array_equal(actual, expected)


def test_block_convolution_requantizes_patches_on_k_axis():
    torch = pytest.importorskip('torch')
    from torch.nn import functional as F
    from tools.experiment_b_ext.shared import BlockQuantizer, block_conv2d
    quantizer = BlockQuantizer('mxfp4_e2m1', 'maxabs', 'cpu')
    module = torch.nn.Conv2d(2, 3, 3, padding=1).eval()
    values = torch.arange(2*5*5, dtype=torch.float32).reshape(1, 2, 5, 5)/37
    weight = quantizer(module.weight.detach().reshape(3, -1)).reshape_as(module.weight)
    actual = block_conv2d(values, module, quantizer, weight)
    patches = F.unfold(values, 3, padding=1).transpose(1, 2)
    expected = torch.matmul(quantizer(patches), weight.reshape(3, -1).T)
    expected = expected.transpose(1, 2).reshape(1, 3, 5, 5) + module.bias.reshape(1, 3, 1, 1)
    torch.testing.assert_close(actual, expected)
