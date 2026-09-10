"""Pack a certified exact rational grid for the shared native wide kernels."""
import ctypes as C
from decimal import Decimal
from fractions import Fraction
from math import lcm

from public.inference.reference.arithmetic import Accumulator, format_named, real, pow2


class RationalFormat(C.Structure):
    _fields_ = [(name,C.c_int32) for name in "bits mantissa exponent bias integer fractional product_shift size".split()]


def values(sequence):
    from public.inference.native import NativeValues
    if isinstance(sequence,NativeValues):
        special = {1:Decimal("Infinity"),2:Decimal("-Infinity"),3:Decimal("NaN"),4:Decimal("-0")}
        return tuple(special[int(row["kind"])] if row["kind"] else int(row["mantissa"])*pow2(int(row["exponent"])) for row in sequence.array)
    return tuple(real(value) for value in sequence)


def gemm(native, inputs, weights, *, batch, channels, k, accumulator):
    import numpy as np
    if any(type(v) is not int or not 0 < v < 2**31 for v in (batch,channels,k)):
        raise ValueError("rational GEMM requires positive int32 dimensions")
    left,right = values(inputs),values(weights)
    if len(left) != batch*k or len(right) != channels*k:
        raise ValueError("rational GEMM shape mismatch")
    acc = format_named(accumulator)
    if not isinstance(acc,Accumulator) or not acc.manifest["signed"]:
        raise ValueError("rational kernel requires a signed concrete accumulator")
    finite_left = [Fraction(v) for v in left if isinstance(v,Fraction) or v.is_finite()]
    finite_right = [Fraction(v) for v in right if isinstance(v,Fraction) or v.is_finite()]
    dx,dw = lcm(*(v.denominator for v in finite_left)),lcm(*(v.denominator for v in finite_right))
    if acc.family == "float":
        f = acc.manifest["float"]
        m,e,bias,integer,fractional = f["mantissa_bits"],f["exp_bits"],f["bias"],0,0
        min_denominator_bits = max(0,bias+m-1)
        max_state_bits = max(m+1,(1<<e)-2-bias+1)
    else:
        if len(finite_left) != len(left) or len(finite_right) != len(right):
            raise ValueError("integer/fixed accumulator cannot consume nonfinite operands")
        m,e,bias,integer,fractional = 0,0,0,1,acc.manifest["numeric"]["fractional_bits"]
        min_denominator_bits,max_state_bits = fractional,acc.bits
    product_denominator = dx*dw
    denominator = lcm(product_denominator,1<<min_denominator_bits)
    factor = denominator//product_denominator
    if factor & (factor-1):
        raise ValueError("invalid rational product-grid alignment")
    shift = factor.bit_length()-1
    xn = max((abs(v.numerator)*(dx//v.denominator) for v in finite_left),default=0)
    wn = max((abs(v.numerator)*(dw//v.denominator) for v in finite_right),default=0)
    # Include intermediate decoded-state multiplication, signed addition,
    # division guard bit, and normalization shifts, not just stored values.
    required = max(xn.bit_length(),wn.bit_length(),(xn*wn).bit_length()+shift,
                   denominator.bit_length()+max_state_bits,denominator.bit_length()+m+2)+4
    size = next((s for s in (32,64,128,320) if s*32 >= required),None)
    if size is None:
        raise ValueError(f"rational domain requires {required} bits, beyond the certified 10240-bit kernel")

    def limbs(number):
        return [(number>>(32*i))&0xffffffff for i in range(size)]

    def pack(sequence,common_denominator):
        # Reuse the encoding of repeated low-precision values. This is a
        # representation cache; no value or product is approximated.
        packed = np.empty((len(sequence),size+2),dtype=np.uint32)
        cache = {}
        for index,value in enumerate(sequence):
            if isinstance(value,Decimal):
                kind = 3 if value.is_nan() else (2 if value.is_signed() else 1) if value.is_infinite() else 4 if value.is_signed() and not value else 0
                negative = value.is_signed()
            else:
                kind,negative = 0,value<0
            key = (kind,negative,None if kind else value)
            if key not in cache:
                magnitude = 0 if kind else abs(Fraction(value).numerator)*(common_denominator//Fraction(value).denominator)
                cache[key] = [int(negative),kind]+limbs(magnitude)
            packed[index] = cache[key]
        return packed
    x,w,d = pack(left,dx),pack(right,dw),np.asarray(limbs(denominator),dtype=np.uint32)
    result = np.empty(batch*channels,dtype=np.uint64)
    function = native.library.pe_rational_gemm
    function.argtypes = [C.POINTER(C.c_uint32)]*3+[C.POINTER(C.c_uint64),C.c_int,C.c_int,C.c_int,RationalFormat]
    function.restype = C.c_int
    error = function(x.ctypes.data_as(C.POINTER(C.c_uint32)),w.ctypes.data_as(C.POINTER(C.c_uint32)),
                     d.ctypes.data_as(C.POINTER(C.c_uint32)),result.ctypes.data_as(C.POINTER(C.c_uint64)),
                     batch,channels,k,RationalFormat(acc.bits,m,e,bias,integer,fractional,shift,size))
    if error:
        detail = native.library.pe_error(error).decode() if native.backend == "cuda" else str(error)
        raise RuntimeError(f"exact rational {native.backend} execution failed: {detail}")
    native.last_strategy = f"rational_grid_{size*32}"
    return tuple(int(code) for code in result)
