"""Deterministic NCHW operators. Reduction order is channel, row, column."""
from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from math import prod
from functools import lru_cache

from public.inference.tensor import Tensor, Encoding, SharedEncoding, indices
from .arithmetic import Accumulator, binary, decimal, format_named, model_c, real


def pair(value, name="parameter"):
    result = (value, value) if type(value) is int else tuple(value)
    if len(result) != 2 or any(type(v) is not int for v in result):
        raise ValueError(f"{name} must be an integer or pair of integers")
    return result


def clamp(value, lower, upper=None):
    if isinstance(value, Decimal) and value.is_nan():
        return value
    return max(lower, value) if upper is None else min(upper, max(lower, value))


def post_activation(value, activation):
    if activation == "identity":
        return value
    if activation == "relu":
        return clamp(value, Fraction(0))
    if activation == "relu6":
        return clamp(value, Fraction(0), Fraction(6))
    raise ValueError("fused activation must be identity, relu or relu6")


def _accumulator(name):
    acc = format_named(name)
    if not isinstance(acc, Accumulator):
        raise ValueError("operator requires an accumulator manifest")
    return acc


def _domain(acc, left, right, addresses):
    if acc.family not in {"integer", "fixed_point"} or acc.manifest["numeric"]["fractional_bits"]:
        return Fraction(1)
    scales = {left.encoding.scale_at(left.shape, a) * right.encoding.scale_at(right.shape, b)
              for a, b in addresses if a is not None}
    if len(scales) > 1:
        raise ValueError("integer dot product requires one product scale per output")
    return next(iter(scales), Fraction(1))


def _dot(left, right, addresses, acc, bias, activation):
    addresses = tuple(addresses)
    scale = _domain(acc, left, right, addresses)
    # Padding is a real zero, not Q(0), which need not be zero for binary_pm1.
    pairs = ((Fraction(0) if a is None else left.value(a), right.value(b)) for a, b in addresses)
    code = model_c(pairs, acc, bias=bias, scale=scale)
    return post_activation(acc.decode(code, scale=scale), activation)


def linear(inputs: Tensor, weights: Tensor, *, accumulator: str, output: Encoding, bias=None, activation="identity"):
    if len(inputs.shape) != 2 or len(weights.shape) != 2 or inputs.shape[1] != weights.shape[1]:
        raise ValueError("Linear requires [batch,K] inputs and [outputs,K] weights")
    batch, k = inputs.shape
    channels = weights.shape[0]
    if bias is not None and len(bias) != channels:
        raise ValueError("bias count must equal output channels")
    acc = _accumulator(accumulator)
    values = [_dot(inputs, weights, [((n, i), (c, i)) for i in range(k)], acc,
                   None if bias is None else real(bias[c]), activation)
              for n in range(batch) for c in range(channels)]
    return Tensor.quantize(values, (batch, channels), output)


def conv2d(inputs: Tensor, weights: Tensor, *, accumulator: str, output: Encoding,
           bias=None, stride=1, padding=0, dilation=1, groups=1, activation="identity"):
    if len(inputs.shape) != 4 or len(weights.shape) != 4:
        raise ValueError("Conv2D requires NCHW inputs and OIHW weights")
    n, channels, height, width = inputs.shape
    outputs, group_channels, kh, kw = weights.shape
    if type(groups) is not int or groups <= 0 or channels % groups or outputs % groups or group_channels != channels // groups:
        raise ValueError("invalid convolution groups")
    if inputs.encoding.block_size or weights.encoding.block_size:
        raise ValueError("shared-scale convolution requires explicit K-block patch lowering; use Linear on lowered patches")
    if bias is not None and len(bias) != outputs:
        raise ValueError("bias count must equal output channels")
    sh, sw = pair(stride, "stride")
    ph, pw = pair(padding, "padding")
    dh, dw = pair(dilation, "dilation")
    if min(sh, sw, dh, dw) <= 0 or min(ph, pw) < 0:
        raise ValueError("invalid convolution geometry")
    oh = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if min(oh, ow) <= 0:
        raise ValueError("empty convolution output")
    acc, values = _accumulator(accumulator), []
    for batch, channel, row, column in indices((n, outputs, oh, ow)):
        group = channel // (outputs // groups)
        addresses = []
        for ic, kr, kc in indices((group_channels, kh, kw)):
            y, x = row * sh - ph + kr * dh, column * sw - pw + kc * dw
            address = (batch, group * group_channels + ic, y, x) if 0 <= y < height and 0 <= x < width else None
            addresses.append((address, (channel, ic, kr, kc)))
        values.append(_dot(inputs, weights, addresses, acc, None if bias is None else real(bias[channel]), activation))
    return Tensor.quantize(values, (n, outputs, oh, ow), output)


def depthwise_conv2d(inputs, weights, **kwargs):
    return conv2d(inputs, weights, groups=inputs.shape[1], **kwargs)


def elementwise(left, right, *, operation, output, accumulator=None, alignment=None):
    if operation not in {"add", "mul"}:
        raise ValueError("unsupported elementwise operation")
    if left.shape != right.shape:
        raise ValueError("elementwise shapes must match; broadcasting must be explicit")
    if operation == "add":
        if accumulator is None or alignment is None:
            raise ValueError("residual add requires explicit accumulator and alignment encoding")
        if isinstance(alignment,SharedEncoding):
            a = Tensor.quantize(left.values(),left.shape,alignment)
            b = Tensor.quantize(right.values(),right.shape,alignment)
            # Shared residuals use one common domain per block. Taking the
            # larger native scale preserves both branches' admitted ranges.
            alignment = Encoding(alignment.format,tuple(max(x,y) for x,y in zip(a.encoding.scales,b.encoding.scales)),alignment.axis,alignment.block_size)
        left = Tensor.quantize(left.values(), left.shape, alignment)
        right = Tensor.quantize(right.values(), right.shape, alignment)
    acc = _accumulator(accumulator) if accumulator else None
    values = []
    for idx in indices(left.shape):
        value = binary(left.value(idx), right.value(idx), operation)
        if acc:
            scale = alignment.scale_at(left.shape, idx) if operation == "add" and acc.family == "integer" else 1
            value = acc.rounded(value, scale=scale)
        values.append(value)
    return Tensor.quantize(values, left.shape, output)


@lru_cache(maxsize=65536)
def _activation_code(code, input_encoding, function, output, accumulator):
    # Evaluate only codes present in the input. An unused NaN/NaR code must
    # not make a finite activation fail when its output cannot represent it.
    return _activation_scalar(Tensor((1,),(code,),input_encoding),function=function,output=output,accumulator=accumulator).codes[0]


def activation(inputs, *, function, output, accumulator=None):
    if inputs.encoding.axis is None and output.axis is None:
        table = {c: _activation_code(c,inputs.encoding,function,output,accumulator) for c in set(inputs.codes)}
        return Tensor(inputs.shape,tuple(table[c] for c in inputs.codes),output)
    return _activation_scalar(inputs,function=function,output=output,accumulator=accumulator)


def _activation_scalar(inputs, *, function, output, accumulator=None):
    if function in {"identity", "relu", "relu6"}:
        values = [post_activation(value, function) for value in inputs.values()]
    elif function in {"hard_swish", "hard_sigmoid"}:
        if accumulator is None:
            raise ValueError("hard-swish requires explicit arithmetic accumulator")
        acc = _accumulator(accumulator)
        values = []
        for value in inputs.values():
            added = acc.rounded(binary(value, Fraction(3), "add"))
            clipped = clamp(added, Fraction(0), Fraction(6))
            multiplied = acc.rounded(binary(value, clipped, "mul")) if function == "hard_swish" else clipped
            values.append(acc.rounded(binary(multiplied, Fraction(1, 6), "mul")))
    else:
        raise ValueError("nonlinear functions require a generated LUT")
    return Tensor.quantize(values, inputs.shape, output)


def adaptive_average_pool2d(inputs, *, output, accumulator, output_size=1):
    size = pair(output_size)
    if size != (1, 1):
        raise ValueError("adaptive average pool currently supports global 1x1 pooling")
    return pool2d(inputs, kind="average", kernel_size=inputs.shape[-2:], accumulator=accumulator, output=output)


def flatten(inputs, *, start_dim=1):
    if type(start_dim) is not int or not 0 <= start_dim < len(inputs.shape):
        raise ValueError("invalid flatten start dimension")
    if inputs.encoding.axis is not None:
        shape = inputs.shape[:start_dim] + (prod(inputs.shape[start_dim:]),)
        if inputs.encoding.block_size:
            return Tensor.quantize(inputs.values(),shape,SharedEncoding(inputs.encoding.format,axis=min(inputs.encoding.axis,start_dim)))
        if inputs.encoding.axis < start_dim or all(s == 1 for s in inputs.shape[inputs.encoding.axis+1:]):
            return Tensor(shape,inputs.codes,inputs.encoding)
        raise ValueError("flatten of channel mapping requires an explicit encoding")
    shape = inputs.shape[:start_dim] + (prod(inputs.shape[start_dim:]),)
    return Tensor(shape, inputs.codes, inputs.encoding)


def broadcast_multiply(left, right, *, output):
    if len(left.shape) != len(right.shape) or any(a != b and a != 1 and b != 1 for a,b in zip(left.shape,right.shape)):
        raise ValueError("broadcast multiply requires equal ranks and compatible dimensions")
    shape = tuple(max(a,b) for a,b in zip(left.shape,right.shape))
    values = [binary(left.value(tuple(0 if s == 1 else i for s,i in zip(left.shape,idx))),
                     right.value(tuple(0 if s == 1 else i for s,i in zip(right.shape,idx))), "mul") for idx in indices(shape)]
    return Tensor.quantize(values, shape, output)


def nonlinear_lut(input_encoding, output_encoding, function):
    if input_encoding.axis is not None or output_encoding.axis is not None:
        raise ValueError("one LUT requires scalar input and output domains")
    fmt = format_named(input_encoding.format)
    values = []
    with localcontext() as context:
        context.prec = 200
        for code in range(1 << fmt.bits):
            value = fmt.decode(code, scale=decimal(input_encoding.scales[0]))
            if value.is_nan():
                result = value
            elif function in {"sigmoid", "silu"}:
                if value == Decimal("Infinity"):
                    result = Decimal(1) if function == "sigmoid" else value
                elif value == Decimal("-Infinity"):
                    result = Decimal(0) if function == "sigmoid" else Decimal("-0")
                else:
                    # Stable on both tails, including very large finite inputs.
                    e = (-abs(value)).exp()
                    sigmoid = 1 / (1 + e) if value >= 0 else e / (1 + e)
                    result = sigmoid if function == "sigmoid" else value * sigmoid
            elif function == "tanh":
                e = (-2 * abs(value)).exp()
                result = (1 - e) / (1 + e) * (-1 if value < 0 else 1)
            else:
                raise ValueError("unsupported LUT function")
            values.append(result)
    return Tensor.quantize(values, (len(values),), output_encoding).codes


def apply_lut(inputs, *, table, output):
    if inputs.encoding.axis is not None or output.axis is not None:
        raise ValueError("LUT requires scalar domains")
    if len(table) != 1 << format_named(inputs.encoding.format).bits:
        raise ValueError("LUT size does not cover input codes")
    return Tensor(inputs.shape, tuple(table[code] for code in inputs.codes), output)


@lru_cache(maxsize=65536)
def _nonlinear_value(format_name,code,scale,function):
    fmt = format_named(format_name)
    value = fmt.decode(code,scale=decimal(scale))
    with localcontext() as context:
        context.prec = 200
        if not value.is_finite():
            if value.is_nan():
                return value
            if function == "tanh":
                return Fraction(-1 if value.is_signed() else 1)
            if function == "sigmoid":
                return Fraction(0 if value.is_signed() else 1)
            return Decimal("-0") if value.is_signed() else value
        e = (-abs(value)).exp()
        if function == "tanh":
            e = (-2*abs(value)).exp()
            result = (1-e)/(1+e)*(-1 if value<0 else 1)
        else:
            result = 1/(1+e) if value>=0 else e/(1+e)
            if function == "silu":
                result *= value
        return real(result)


def scaled_nonlinear_lut(inputs, *, function, output):
    if function not in {"silu","sigmoid","tanh"}:
        raise ValueError("unsupported scaled nonlinear LUT")
    values = [_nonlinear_value(inputs.encoding.format,code,inputs.encoding.scale_at(inputs.shape,index),function)
              for code,index in zip(inputs.codes,indices(inputs.shape))]
    # Resolve the native output block scale before storing each LUT result.
    return Tensor.quantize(values,inputs.shape,output)


def pool2d(inputs, *, kind, kernel_size, output, stride=None, padding=0,
           accumulator=None, count_include_pad=True):
    if len(inputs.shape) != 4 or kind not in {"max", "average"}:
        raise ValueError("pool requires NCHW and max/average kind")
    kh, kw = pair(kernel_size)
    sh, sw = pair(kernel_size if stride is None else stride)
    ph, pw = pair(padding)
    if min(kh, kw, sh, sw) <= 0 or min(ph, pw) < 0:
        raise ValueError("invalid pooling geometry")
    n, channels, height, width = inputs.shape
    oh, ow = (height + 2 * ph - kh) // sh + 1, (width + 2 * pw - kw) // sw + 1
    if min(oh, ow) <= 0:
        raise ValueError("empty pool output")
    if kind == "max" and inputs.encoding.axis is None and output.axis is None:
        import numpy as np
        fmt = format_named(inputs.encoding.format)
        decoded = [real(fmt.decode(code,scale=decimal(inputs.encoding.scales[0]))) for code in range(1<<fmt.bits)]
        finite = sorted(set(v for v in decoded if not isinstance(v,Decimal) or not v.is_nan()))
        ranks = np.array([len(finite) if isinstance(v,Decimal) and v.is_nan() else finite.index(v) for v in decoded]+[-1])
        codes = np.asarray(inputs.codes,dtype=np.int16).reshape(inputs.shape)
        padded = np.pad(codes,((0,0),(0,0),(ph,ph),(pw,pw)),constant_values=len(decoded))
        windows = np.lib.stride_tricks.sliding_window_view(padded,(kh,kw),axis=(2,3))[:,:,::sh,::sw]
        windows = windows.reshape(n,channels,oh,ow,kh*kw)
        choices = np.argmax(ranks[windows],axis=-1)[...,None]
        selected = np.take_along_axis(windows,choices,axis=-1).ravel()
        if np.any(selected == len(decoded)):
            raise ValueError("pool window contains no values")
        table = {int(c): Tensor.quantize([decoded[c]],(1,),output).codes[0] for c in set(selected)}
        return Tensor((n,channels,oh,ow),tuple(table[c] for c in selected),output)
    acc = _accumulator(accumulator) if kind == "average" else None
    values = []
    for batch, channel, row, column in indices((n, channels, oh, ow)):
        window = []
        window_scales = set()
        for kr, kc in indices((kh, kw)):
            y, x = row * sh - ph + kr, column * sw - pw + kc
            if 0 <= y < height and 0 <= x < width:
                address = (batch, channel, y, x)
                window.append(inputs.value(address))
                window_scales.add(inputs.encoding.scale_at(inputs.shape, address))
            elif kind == "average" and count_include_pad:
                window.append(Fraction(0))
        if not window:
            raise ValueError("pool window contains no values")
        if kind == "max":
            value = next((v for v in window if isinstance(v, Decimal) and v.is_nan()), None)
            value = max(window) if value is None else value
        else:
            if acc.family == "integer":
                if len(window_scales) != 1:
                    raise ValueError("integer pool sum requires one input scale per window")
                scale = next(iter(window_scales))
            else:
                scale = Fraction(1)
            value = Fraction(0)
            for item in window:
                value = acc.rounded(binary(value, item, "add"), scale=scale)
            divided = binary(value, Fraction(1, len(window)), "mul")
            # Integer averaging is an exact rational scale/requantization;
            # the sum is integer, but the mean need not be an integer code.
            value = divided if acc.family == "integer" else acc.rounded(divided)
        values.append(value)
    return Tensor.quantize(values, (n, channels, oh, ow), output)
