"""Explicit YOLO tensor movement, quantized DFL and box decoding.

DFL normalizes accumulator-rounded exponentials in index order, stores
probabilities in the candidate encoding, and performs its weighted reduction
with Model C. Box arithmetic rounds in the accumulator and stores candidate
coordinates. Only downstream NMS is outside this graph.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache

from public.inference.tensor import Tensor, Encoding, SharedEncoding, indices, offset
from .arithmetic import binary, real, decimal, model_c
from .operators import _accumulator


def concatenate(*inputs, axis, output):
    if not inputs:
        raise ValueError("concatenation requires inputs")
    rank = len(inputs[0].shape)
    if type(axis) is not int or not 0 <= axis < rank or not inputs:
        raise ValueError("invalid concatenation axis")
    shape = list(inputs[0].shape)
    if any(len(t.shape) != rank or any(s != shape[i] for i,s in enumerate(t.shape) if i != axis) for t in inputs):
        raise ValueError("incompatible concatenation shapes")
    shape[axis] = sum(t.shape[axis] for t in inputs)
    if all(t.encoding == output for t in inputs) and output.axis is None:
        import numpy as np
        codes = np.concatenate([np.asarray(t.codes,dtype=np.uint8).reshape(t.shape) for t in inputs],axis=axis)
        return Tensor(tuple(shape),tuple(codes.ravel().tolist()),output)
    boundaries, total = [], 0
    for tensor in inputs:
        total += tensor.shape[axis]
        boundaries.append(total)
    values = []
    for index in indices(shape):
        selected = next(i for i,end in enumerate(boundaries) if index[axis] < end)
        address = list(index)
        address[axis] -= 0 if selected == 0 else boundaries[selected-1]
        values.append(inputs[selected].value(tuple(address)))
    return Tensor.quantize(values, tuple(shape), output)


def channel_slice(inputs, *, start, stop, output):
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= inputs.shape[1]:
        raise ValueError("invalid channel slice")
    shape = (inputs.shape[0],stop-start)+inputs.shape[2:]
    if inputs.encoding == output and output.axis is None:
        import numpy as np
        codes = np.asarray(inputs.codes,dtype=np.uint8).reshape(inputs.shape)[:,start:stop]
        return Tensor(shape,tuple(codes.ravel().tolist()),output)
    return Tensor.quantize([inputs.value((i[0],i[1]+start)+i[2:]) for i in indices(shape)],shape,output)


def resize_nearest(inputs, *, factor, output):
    if len(inputs.shape) != 4 or type(factor) is not int or factor < 1:
        raise ValueError("nearest resize requires NCHW and positive integer factor")
    n,c,h,w = inputs.shape
    shape = (n,c,h*factor,w*factor)
    if inputs.encoding == output and output.axis is None:
        import numpy as np
        codes = np.asarray(inputs.codes,dtype=np.uint8).reshape(inputs.shape).repeat(factor,axis=2).repeat(factor,axis=3)
        return Tensor(shape,tuple(codes.ravel().tolist()),output)
    return Tensor.quantize([inputs.value((b,ch,y//factor,x//factor)) for b,ch,y,x in indices(shape)],shape,output)


@lru_cache(maxsize=65536)
def _exp_difference(value):
    with localcontext() as context:
        context.prec = 200
        return real(decimal(value).exp())


def softmax(inputs, *, axis, accumulator, output):
    if type(axis) is not int or not 0 <= axis < len(inputs.shape):
        raise ValueError("softmax axis outside tensor rank")
    acc = _accumulator(accumulator)
    shape = inputs.shape
    result = [None]*len(inputs.codes)
    outer = shape[:axis]+shape[axis+1:]
    for index in indices(outer):
        addresses = [index[:axis]+(i,)+index[axis:] for i in range(shape[axis])]
        values = [inputs.value(address) for address in addresses]
        if any(isinstance(v,Decimal) and not v.is_finite() for v in values):
            raise ValueError("DFL softmax requires finite input logits")
        maximum = max(values)
        exponentials = [acc.rounded(_exp_difference(acc.rounded(binary(v,-maximum,"add")))) for v in values]
        total = acc.rounded(0)
        for value in exponentials:
            total = acc.rounded(binary(total,value,"add"))
        if not total:
            raise ValueError("softmax accumulator underflow")
        for address,value in zip(addresses,exponentials):
            result[offset(shape,address)] = acc.rounded(Fraction(value)/Fraction(total))
    return Tensor.quantize(result,shape,output)


def dfl(inputs, weights, *, bins, accumulator, output):
    if len(inputs.shape) != 4 or inputs.shape[1] != 4*bins or type(bins) is not int or bins < 2:
        raise ValueError("DFL requires N x (4*bins) x H x W")
    n,_,h,w = inputs.shape
    if weights.shape not in {(1,bins,1,1),(1,bins)}:
        raise ValueError("DFL projection requires quantized 1 x bins x 1 x 1 weights")
    # Reshape channels into [side,bin]; no arithmetic/rounding at this view.
    if inputs.encoding.block_size:
        # The bin dimension owns the newly formed reduction blocks. This is
        # an explicit conversion, including the final short bins block.
        logits = Tensor.quantize(inputs.values(),(n,4,bins,h,w),SharedEncoding(inputs.encoding.format,axis=2))
    elif inputs.encoding.axis is not None:
        raise ValueError("DFL mapping logits require an aligned scalar domain")
    else:
        logits = Tensor((n,4,bins,h,w),inputs.codes,inputs.encoding)
    probability_encoding = SharedEncoding(output.format,axis=2) if isinstance(output,SharedEncoding) else output
    probabilities = softmax(logits,axis=2,accumulator=accumulator,output=probability_encoding)
    acc = _accumulator(accumulator)
    values = []
    for batch,side,y,x in indices((n,4,h,w)):
        code = model_c(((probabilities.value((batch,side,i,y,x)),weights.value((0,i) if len(weights.shape)==2 else (0,i,0,0))) for i in range(bins)),acc)
        values.append(acc.decode(code))
    return Tensor.quantize(values,(n,4,h,w),output)


def decode_boxes(inputs, *, stride, accumulator, output):
    if len(inputs.shape) != 4 or inputs.shape[1] != 4 or type(stride) is not int or stride < 1:
        raise ValueError("box decode requires four distance channels and integer stride")
    n,_,h,w = inputs.shape
    acc = _accumulator(accumulator)
    values = [None]*len(inputs.codes)
    for b,y,x in indices((n,h,w)):
        l,t,r,d = (inputs.value((b,i,y,x)) for i in range(4))
        x1,y1 = acc.rounded(binary(Fraction(2*x+1,2),-l,"add")),acc.rounded(binary(Fraction(2*y+1,2),-t,"add"))
        x2,y2 = acc.rounded(binary(Fraction(2*x+1,2),r,"add")),acc.rounded(binary(Fraction(2*y+1,2),d,"add"))
        coordinates = (acc.rounded(binary(acc.rounded(binary(x1,x2,"add")),Fraction(1,2),"mul")),
                       acc.rounded(binary(acc.rounded(binary(y1,y2,"add")),Fraction(1,2),"mul")),
                       acc.rounded(binary(x2,-x1,"add")),acc.rounded(binary(y2,-y1,"add")))
        for channel,value in enumerate(coordinates):
            values[offset(inputs.shape,(b,channel,y,x))] = acc.rounded(binary(value,real(stride),"mul"))
    return Tensor.quantize(values,inputs.shape,output)
