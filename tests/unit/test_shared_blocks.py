from fractions import Fraction
import pytest

from public.inference.tensor import Tensor, Encoding, SharedEncoding
from public.inference.reference.blocks import block_conv2d


def test_partial_blocks_have_independent_row_scales():
    tensor = Tensor.quantize([6]*32+[Fraction(1,1024)]+[1]*32+[64],(2,33),SharedEncoding("mxfp4_e2m1",axis=1))
    assert tensor.encoding.scales == (1,Fraction(1,4096),Fraction(1,4),16)
    assert tensor.values() == tuple([6]*32+[Fraction(1,1024)]+[1]*32+[64])


def test_block_conv_keeps_accumulator_live_across_partial_block():
    inputs = Tensor.quantize([1]*32+[Fraction(1,16)],(1,33,1,1),Encoding("fp6_e3m2"))
    weights = Tensor.quantize([1]+[0]*31+[1],(1,33),SharedEncoding("mxfp4_e2m1",axis=1))
    result = block_conv2d(inputs,weights,kernel_size=1,patch_format="mxfp4_e2m1",
                          accumulator="fp32_e8m23_accumulator",output=Encoding("fp6_e3m2"),bias=[Fraction(1,4)])
    assert result.values() == (Fraction(5,4),)


@pytest.mark.parametrize("activation,weight",[("fp6_e3m2","mxfp4_e2m1"),("mxfp4_e2m1","fp6_e3m2"),("mxfp4_e2m1","mxfp4_e2m1")])
def test_lowered_shared_and_mixed_convolution(activation,weight):
    torch = pytest.importorskip("torch")
    from public.quantization.graph.torchvision import lower
    from public.quantization.graph.executable import execute
    model = torch.nn.Sequential(torch.nn.Conv2d(1,1,1,bias=False),torch.nn.ReLU()).eval()
    model[0].weight.data.fill_(1)
    encoding = SharedEncoding(activation) if activation.startswith("mx") else Encoding(activation)
    graph = lower(model,input_encoding=encoding,weight_format=weight,accumulator="fp32_e8m23_accumulator",provenance={"kind":"synthetic_conformance"})
    inputs = {next(iter(graph["inputs"])): Tensor.quantize([1,2],(1,1,1,2),encoding)}
    assert graph["nodes"][0]["op"] == "block_conv2d"
    result = execute(graph,inputs)
    assert next(iter(result["outputs"].values())).values() == (1,2)


def test_binary_activation_padding_remains_mathematical_zero():
    inputs = Tensor.quantize([1],(1,1,1,1),Encoding("binary_pm1"))
    weights = Tensor.quantize([1]*9,(1,9),SharedEncoding("mxfp4_e2m1"))
    result = block_conv2d(inputs,weights,kernel_size=3,padding=1,patch_format="binary_pm1",
                          accumulator="fp32_e8m23_accumulator",output=Encoding("fp6_e3m2"))
    assert result.values() == (1,)
