"""Native reductions plus explicitly shared exact reference post-operations."""
from __future__ import annotations

from fractions import Fraction

from public.inference.native import NativeBackend, prepare_tensor
from public.inference.tensor import Tensor, indices
from public.inference.reference import operators as ref
from public.inference.reference.arithmetic import binary, real


class Operators:
    def __init__(self, backend="reference", library=None):
        if backend not in {"reference", "cpp", "cuda"}:
            raise ValueError("unknown execution backend")
        self.backend = backend
        self.native = None if backend == "reference" else NativeBackend(backend, library)

    def _domains(self, inputs, weights, accumulator):
        acc = ref._accumulator(accumulator)
        def prepare(tensor,unscaled=False):
            try:
                return prepare_tensor(tensor,unscaled=unscaled)
            except ValueError:
                if not unscaled:
                    return tensor.values()
                from public.inference.reference.arithmetic import decode,format_named
                fmt = format_named(tensor.encoding.format)
                return tuple(decode(fmt,code) for code in tensor.codes)
        if acc.family != "integer":
            return prepare(inputs), prepare(weights), (Fraction(1),) * weights.shape[0]
        if inputs.encoding.axis is not None or weights.encoding.axis not in {None, 0} or weights.encoding.block_size:
            raise ValueError("INT32 native reduction requires per-tensor activations and tensor/channel weights")
        left, right = prepare(inputs, unscaled=True), prepare(weights, unscaled=True)
        scales = tuple(inputs.encoding.scales[0] * weights.encoding.scale_at(weights.shape, (c,) + (0,) * (len(weights.shape)-1))
                       for c in range(weights.shape[0]))
        return left, right, scales

    def _store(self, states, shape, accumulator, scales, bias, activation, output):
        acc = ref._accumulator(accumulator)
        if bias is not None and len(bias) != shape[1]:
            raise ValueError("bias count must equal output channels")
        fmt = ref.format_named(output.format)
        if (acc.name in {"fp16_e5m10_accumulator","fp32_e8m23_accumulator"}
                and fmt.family == "float" and fmt.manifest["scaling"]["mode"] == "none"
                and all(scale == 1 for scale in scales)):
            import numpy as np
            from public.quantization.ptq.encoding import direct_float_tensor
            unsigned, floating = (np.uint16,np.float16) if acc.bits == 16 else (np.uint32,np.float32)
            values = np.asarray(states,dtype=unsigned).reshape(shape).view(floating)
            if bias is not None:
                # Bias is encoded exactly once by the rational oracle. The
                # single IEEE add has exactly the declared accumulator dtype;
                # no host float reduction or widened/double-rounded sum occurs.
                stored = np.asarray([acc.encode(real(v)) for v in bias],dtype=unsigned).view(floating)
                with np.errstate(over="ignore",invalid="ignore"):
                    values = np.add(values,stored.reshape((1,len(bias))+(1,)*(len(shape)-2)),dtype=floating)
            if activation in {"relu","relu6"}:
                values = np.maximum(values,floating(0))
                if activation == "relu6":
                    values = np.minimum(values,floating(6))
            elif activation != "identity":
                raise ValueError("unsupported fused activation")
            return direct_float_tensor(values.astype(np.float32),output)
        values = []
        for code, index in zip(states, indices(shape)):
            channel = index[1]
            scale = scales[channel]
            value = acc.decode(code, scale=scale)
            if bias is not None:
                value = acc.rounded(binary(value, acc.rounded(real(bias[channel]), scale=scale), "add"), scale=scale)
            values.append(ref.post_activation(value, activation))
        return Tensor.quantize(values, shape, output)

    def linear(self, inputs, weights, *, accumulator, output, bias=None, activation="identity"):
        if self.native is None:
            return ref.linear(inputs, weights, accumulator=accumulator, output=output, bias=bias, activation=activation)
        if len(inputs.shape) != 2 or len(weights.shape) != 2 or inputs.shape[1] != weights.shape[1]:
            raise ValueError("Linear requires [batch,K] and [outputs,K]")
        left, right, scales = self._domains(inputs, weights, accumulator)
        batch, k = inputs.shape
        channels = weights.shape[0]
        states = self.native.flex_gemm(left, right, batch=batch, channels=channels, k=k, accumulator=accumulator)
        return self._store(states, (batch, channels), accumulator, scales, bias, activation, output)

    def conv2d(self, inputs, weights, *, accumulator, output, bias=None, stride=1, padding=0,
               dilation=1, groups=1, activation="identity"):
        if self.native is None:
            return ref.conv2d(inputs, weights, accumulator=accumulator, output=output, bias=bias,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, activation=activation)
        if len(inputs.shape) != 4 or len(weights.shape) != 4:
            raise ValueError("convolution requires NCHW/OIHW")
        if inputs.encoding.block_size or weights.encoding.block_size:
            raise ValueError("shared-scale convolution requires explicit K-block patch lowering")
        n, ci, h, w = inputs.shape
        co, group_channels, kh, kw = weights.shape
        if type(groups) is not int or groups <= 0 or ci % groups or co % groups or group_channels != ci // groups:
            raise ValueError("invalid convolution groups")
        sh, sw = ref.pair(stride); ph, pw = ref.pair(padding); dh, dw = ref.pair(dilation)
        if min(sh, sw, dh, dw) <= 0 or min(ph, pw) < 0:
            raise ValueError("invalid convolution geometry")
        oh, ow = (h+2*ph-dh*(kh-1)-1)//sh+1, (w+2*pw-dw*(kw-1)-1)//sw+1
        geometry = dict(n=n, ci=ci, h=h, w=w, co=co, kh=kh, kw=kw, oh=oh, ow=ow,
                        sh=sh, sw=sw, ph=ph, pw=pw, dh=dh, dw=dw, groups=groups)
        left, right, scales = self._domains(inputs, weights, accumulator)
        states = self.native.flex_conv2d(left, right, geometry=geometry, accumulator=accumulator)
        return self._store(states, (n, co, oh, ow), accumulator, scales, bias, activation, output)

    def depthwise_conv2d(self, inputs, weights, **kwargs):
        return self.conv2d(inputs, weights, groups=inputs.shape[1], **kwargs)

    def block_conv2d(self, inputs, weights, **kwargs):
        from public.inference.reference.blocks import block_conv2d
        return block_conv2d(inputs,weights,native=self.native,**kwargs)
