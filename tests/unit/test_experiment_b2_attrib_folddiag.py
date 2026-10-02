"""Lane Q4 r5: helpers of the pcf:all fold diagnostic (protocol addendum 6). CPU only, synthetic tensors."""
import torch

from tools.experiment_b2_attrib.folddiag import _codes, _per_input_channel


def test_codes_divide_by_the_per_output_channel_scale():
    weight = torch.tensor([[[[0.5]], [[-1.0]]], [[[0.25]], [[0.0]]]])  # [out 2, in 2, 1, 1]
    codes = _codes(weight, [0.5, 0.25])
    assert codes.flatten().tolist() == [1.0, -2.0, 1.0, 0.0]


def test_per_input_channel_groups_every_output_channel_of_one_input_channel():
    codes = torch.zeros(3, 4, 2, 2)
    codes[:, 1] = 1  # input channel 1 alive in every output channel
    codes[2, 3, 0, 0] = -1  # input channel 3 alive in one tap of one output channel
    rows = _per_input_channel(codes)
    assert rows.shape == (4, 12)
    silenced = int(((rows != 0).sum(dim=1) == 0).sum())
    assert silenced == 2  # input channels 0 and 2
