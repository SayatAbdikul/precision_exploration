"""Channel-independent stores: retain exact fallback only where it is needed.

No admission bound is widened. Each admitted channel uses the certified integer
store; other channels use the frozen scalar store. Pre-store values are joined
in original NCHW order and observed once, preserving diagnostics exactly.
"""
from collections.abc import Sequence
from math import prod

import numpy as np

from public.inference.reference.arithmetic import format_named
from public.inference.tensor import Encoding, Tensor, QUANTIZATION_OBSERVER, observe_quantization
from tools.phase3.integer_store import integer_store, supported_scale


class ChannelValues(Sequence):
    def __init__(self, values, shape):
        self.values, self.shape = values, shape
        self.spatial = prod(shape[2:])

    def __len__(self):
        return prod(self.shape)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        channel = (index // self.spatial) % self.shape[1]
        batch = index // (self.spatial * self.shape[1])
        return self.values[channel][batch * self.spatial + index % self.spatial]


def mixed_integer_store(states, shape, accumulator, scales, bias, activation, output, *, fallback):
    """None means retain the full original store, without changing any bounds."""
    if (not isinstance(output, Encoding) or len(shape) < 2 or output.axis is not None or output.block_size is not None
            or accumulator not in {'int32_accumulator','int64_accumulator'}
            or format_named(output.format).family != 'integer' or activation not in {'identity','relu','relu6'}
            or len(scales) != shape[1] or not all(supported_scale(s) for s in output.scales)):
        return None
    supported = [supported_scale(s) for s in scales]
    if all(supported) or not any(supported):
        return None
    if bias is not None and len(bias) != shape[1]:
        raise ValueError('bias count must equal output channels')
    output.validate_shape(shape)
    array = np.asarray(states, dtype=np.uint64).reshape(shape[0],shape[1],-1)
    codes = np.empty(array.shape,dtype=np.uint8)
    values=[]
    for channel,scale in enumerate(scales):
        args=(tuple(int(x) for x in array[:,channel,:].ravel()), (shape[0],1,*shape[2:]),
              accumulator,(scale,),None if bias is None else [bias[channel]],activation,output)
        captured=[]
        token=QUANTIZATION_OBSERVER.set(lambda raw,tensor:captured.append(raw))
        try:
            result=integer_store(*args) if supported[channel] else None
            if result is None:
                result=fallback(*args)
        finally:
            QUANTIZATION_OBSERVER.reset(token)
        if len(captured)!=1:
            raise ValueError('channel store did not emit exactly one quantization event')
        values.append(captured[0])
        codes[:,channel,:]=np.asarray(result.codes,dtype=np.uint8).reshape(shape[0],-1)
    tensor=Tensor(tuple(shape),tuple(codes.ravel().tolist()),output)
    return observe_quantization(ChannelValues(values,shape),tensor)
