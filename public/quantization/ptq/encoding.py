"""Vectorized direct-float encoding using exact binary rounding boundaries.

Only unscaled <=8-bit floating formats are admitted. General calibrated
scales and logarithmic/codebook values retain the rational reference path.
"""
from __future__ import annotations

from public.inference.tensor import Tensor
from public.inference.reference.arithmetic import format_named
from public.formats.oracle.number_format import precise


def scalar_float_codes(array, encoding):
    """Bulk finite IEEE input encoding with exact checks near boundaries.

    Float64 only locates a rounding interval. Values within two ULPs of an
    exact rational boundary are encoded by the scalar oracle, including ties.
    """
    import numpy as np
    from fractions import Fraction
    from public.inference.reference.arithmetic import decode,encode,real
    fmt = format_named(encoding.format)
    if encoding.axis is not None or encoding.block_size is not None or fmt.manifest["rounding"] != "rne":
        raise ValueError("bulk encoder requires a scalar RNE encoding")
    source = np.asarray(array)
    if source.dtype not in {np.dtype("float32"),np.dtype("float64")} or not np.isfinite(source).all():
        raise ValueError("bulk encoder requires finite IEEE float32/float64 values")
    flat = source.astype(np.float64).ravel()
    scale = encoding.scales[0]
    decoded = [decode(fmt,c,scale) for c in range(1<<fmt.bits)]
    finite = sorted(set(Fraction(v) for v in decoded if isinstance(v,Fraction) or v.is_finite()))
    bounds = [(a+b)/2 for a,b in zip(finite,finite[1:])]
    boundaries = np.array([float(v) for v in bounds])
    codes = np.array([encode(fmt,v,scale) for v in finite],dtype=np.uint8)
    positions = np.searchsorted(boundaries,flat,side="left")
    result = codes[positions]
    near = np.zeros(len(flat),dtype=bool)
    for adjacent in (positions-1,positions):
        valid = (adjacent>=0)&(adjacent<len(boundaries))
        boundary = boundaries[np.clip(adjacent,0,len(boundaries)-1)]
        near |= valid & (np.abs(flat-boundary)<=2*np.abs(np.spacing(boundary)))
    if fmt.manifest["overflow"] == "infinity":
        threshold = float(finite[-1]+(finite[-1]-finite[-2])/2)
        near |= np.abs(flat)>=np.nextafter(threshold,-np.inf)
    # Preserve signed zero and every format's special underflow convention.
    zero_codes = [c for c in range(1<<fmt.bits) if decoded[c] == 0]
    if fmt.manifest["zero"] == "signed_zero":
        result[np.isin(result,zero_codes)&np.signbit(flat)] = 1 << (fmt.bits-1)
    for i in np.flatnonzero(near):
        result[i] = encode(fmt,real(float(flat[i])),scale)
    return result.reshape(source.shape)


def float_tensor(array, encoding):
    """Quantize finite checkpoint arrays with scalar/per-output-channel scales."""
    import numpy as np
    from public.inference.tensor import Encoding
    source = np.asarray(array)
    encoding.validate_shape(tuple(source.shape))
    if encoding.axis is None:
        result = scalar_float_codes(source,encoding)
    elif encoding.block_size is None:
        result = np.empty(source.shape,dtype=np.uint8)
        for channel,scale in enumerate(encoding.scales):
            index = [slice(None)]*source.ndim
            index[encoding.axis] = channel
            result[tuple(index)] = scalar_float_codes(source[tuple(index)],Encoding(encoding.format,(scale,)))
    else:
        return Tensor.quantize(source.ravel().tolist(),tuple(source.shape),encoding)
    return Tensor(tuple(source.shape),tuple(result.ravel().tolist()),encoding)


@precise
def direct_float_tensor(array, encoding):
    import numpy as np
    fmt = format_named(encoding.format)
    if fmt.family != "float" or fmt.manifest["scaling"]["mode"] != "none" or fmt.manifest["rounding"] != "rne":
        raise ValueError("fast parameter encoder requires a direct RNE float format")
    values = np.asarray(array)
    if values.dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValueError("fast parameter encoder requires IEEE float32/float64 input")
    finite = sorted(set(fmt.decode(code) for code in range(1 << fmt.bits) if fmt.decode(code).is_finite()))
    midpoints = [(a+b)/2 for a,b in zip(finite, finite[1:])]
    boundaries = np.array([float(v) for v in midpoints], dtype=np.float64)
    codes = np.array([fmt.encode(v) for v in finite], dtype=np.uint8)
    ties = np.array([fmt.encode(v) for v in midpoints], dtype=np.uint8)
    flat = values.ravel().astype(np.float64)
    positions = np.searchsorted(boundaries, flat, side="left")
    result = codes[positions]
    clipped = np.minimum(positions, len(boundaries)-1)
    exact_ties = flat == boundaries[clipped]
    result[exact_ties] = ties[clipped[exact_ties]]
    zero_codes = [code for code in range(1 << fmt.bits) if fmt.decode(code).is_zero()]
    if fmt.manifest["zero"] == "signed_zero":
        negative_zero = np.isin(result, zero_codes) & np.signbit(flat)
        result[negative_zero] = 1 << (fmt.bits-1)
    if fmt.manifest["overflow"] == "infinity":
        threshold = float(finite[-1] + (finite[-1]-finite[-2])/2)
        result[flat >= threshold] = fmt.encode("Infinity")
        result[flat <= -threshold] = fmt.encode("-Infinity")
    result[np.isposinf(flat)] = fmt.encode("Infinity")
    result[np.isneginf(flat)] = fmt.encode("-Infinity")
    if np.isnan(flat).any():
        result[np.isnan(flat)] = fmt.encode("NaN")
    return Tensor(tuple(values.shape), tuple(result.tolist()), encoding)
