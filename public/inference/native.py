"""Explicit native builds and checked ctypes bindings; no automatic fallback."""
from __future__ import annotations

import ctypes as C
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
from functools import lru_cache

from public.inference.reference.arithmetic import format_named, real

ROOT = Path(__file__).resolve().parents[2]


class Value(C.Structure):
    _fields_ = [("mantissa", C.c_int64), ("exponent", C.c_int32), ("kind", C.c_int32)]


class Format(C.Structure):
    _fields_ = [("mantissa", C.c_int32), ("exponent", C.c_int32), ("bias", C.c_int32), ("integer", C.c_int32)]


class Conv(C.Structure):
    _fields_ = [(name, C.c_int32) for name in "n ci h w co kh kw oh ow sh sw ph pw dh dw groups".split()]


class Operand(C.Structure):
    _fields_ = [(name, C.c_int32) for name in "bits is_signed numeric fractional mantissa exponent bias subnormal nan infinity".split()]


def operand_format(name):
    fmt = format_named(name)
    m = fmt.manifest
    if fmt.bits > 8 or m["scaling"]["mode"] == "intrinsic_shared":
        raise ValueError("encoded pilots require unblocked <=8-bit formats")
    if fmt.family in {"integer", "fixed_point"}:
        if m["numeric"].get("zero_point", 0):
            raise ValueError("encoded pilots require symmetric mapping")
        result = Operand(fmt.bits, m["signed"], 1, m["numeric"]["fractional_bits"], 0, 0, 0, 0, 0, 0)
    elif fmt.family == "float":
        f = m["float"]
        result = Operand(fmt.bits, m["signed"], 0, 0, f["mantissa_bits"], f["exp_bits"], f["bias"], f["subnormals"], f["nan"], f["infinity"])
    else:
        raise ValueError("algorithmic/lookup comparison currently covers integer, fixed-point and float formats")
    table = (Value * (1 << fmt.bits))(*(dyadic(fmt.decode(code)) for code in range(1 << fmt.bits)))
    return result, table


def dyadic(value):
    value = real(value)
    if isinstance(value, Decimal):
        if value.is_nan():
            return Value(0, 0, 3)
        if value.is_infinite():
            return Value(0, 0, 2 if value.is_signed() else 1)
        if value.is_zero():
            return Value(0, 0, 4 if value.is_signed() else 0)
    value = Fraction(value)
    denominator = value.denominator
    if denominator & (denominator - 1):
        raise ValueError("native binary kernel requires exactly dyadic values; this format/scale is unsupported")
    mantissa, exponent = value.numerator, -(denominator.bit_length() - 1)
    while mantissa and mantissa % 2 == 0:
        mantissa //= 2
        exponent += 1
    if abs(mantissa).bit_length() > 30 or not -80 <= exponent <= 60 or exponent + abs(mantissa).bit_length() > 60:
        raise ValueError("value exceeds the certified native binary-kernel range")
    return Value(mantissa, exponent, 0)


def accumulator_format(name):
    if name == "int32_accumulator":
        return Format(0, 0, 0, 1)
    if name not in {"fp16_e5m10_accumulator", "fp32_e8m23_accumulator"}:
        raise ValueError("native kernel currently supports INT32, FP16 and FP32 accumulators")
    params = format_named(name).manifest["float"]
    return Format(params["mantissa_bits"], params["exp_bits"], params["bias"], 0)


class NativeValues:
    """Owned contiguous native records; preparation does not change arithmetic."""
    def __init__(self, array):
        import numpy as np
        expected = np.dtype([("mantissa", "<i8"), ("exponent", "<i4"), ("kind", "<i4")], align=True)
        if not isinstance(array, np.ndarray) or array.dtype != expected or array.ndim != 1 or not array.flags.c_contiguous:
            raise ValueError("native records require a contiguous, one-dimensional Value layout")
        if array.size:
            mantissas, exponents, kinds = array["mantissa"], array["exponent"], array["kind"]
            if np.any(mantissas <= -(1 << 30)) or np.any(mantissas >= 1 << 30) or np.any(exponents < -80) or np.any(exponents > 60):
                raise ValueError("native records exceed the certified range")
            largest_mantissa = max(abs(int(mantissas.min())), abs(int(mantissas.max())))
            if int(exponents.max()) + largest_mantissa.bit_length() > 60 or np.any(kinds < 0) or np.any(kinds > 4):
                raise ValueError("invalid native record magnitude or special value")
            if np.any((kinds != 0) & (mantissas != 0)):
                raise ValueError("special native values cannot have a finite mantissa")
        self.array = array

    def __len__(self):
        return self.array.size

    @property
    def pointer(self):
        return self.array.ctypes.data_as(C.POINTER(Value))

    def has_nonfinite(self):
        import numpy as np
        return bool(np.any((self.array["kind"] >= 1) & (self.array["kind"] <= 3)))


@lru_cache(maxsize=512)
def _decoded_table(name, scale):
    import numpy as np
    from public.inference.reference.arithmetic import decode
    fmt = format_named(name)
    dtype = np.dtype([("mantissa", "<i8"), ("exponent", "<i4"), ("kind", "<i4")], align=True)
    if dtype.itemsize != C.sizeof(Value):
        raise RuntimeError("native Value layout does not match NumPy records")
    rows = [dyadic(decode(fmt, code, scale)) for code in range(1 << fmt.bits)]
    table = np.array([(row.mantissa, row.exponent, row.kind) for row in rows], dtype=dtype)
    table.flags.writeable = False
    return table


def prepare_tensor(tensor, *, unscaled=False):
    import numpy as np
    encoding = tensor.encoding
    codes = np.asarray(tensor.codes, dtype=np.uint8).reshape(tensor.shape)
    if unscaled or encoding.axis is None:
        scale = Fraction(1) if unscaled else encoding.scales[0]
        return NativeValues(_decoded_table(encoding.format, scale)[codes.ravel()])
    if encoding.block_size:
        # Keep block ownership explicit; a generic tensor-axis expansion cannot
        # invent convolution K packing. Linear already receives lowered blocks.
        values = [dyadic(value) for value in tensor.values()]
        dtype = _decoded_table(encoding.format, Fraction(1)).dtype
        return NativeValues(np.array([(v.mantissa,v.exponent,v.kind) for v in values], dtype=dtype))
    output = np.empty(codes.shape, dtype=_decoded_table(encoding.format, encoding.scales[0]).dtype)
    for channel, scale in enumerate(encoding.scales):
        selection = [slice(None)] * len(tensor.shape)
        selection[encoding.axis] = channel
        selection = tuple(selection)
        output[selection] = _decoded_table(encoding.format, scale)[codes[selection]]
    return NativeValues(output.ravel())


def _native_values(values):
    if isinstance(values, NativeValues):
        return values, values.pointer, values.has_nonfinite()
    entries = [dyadic(v) for v in values]
    owner = (Value*len(entries))(*entries)
    return owner, owner, any(v.kind in {1,2,3} for v in entries)


def build(backend="cpp", directory=None):
    if backend not in {"cpp", "cuda"}:
        raise ValueError("backend must be cpp or cuda")
    directory = Path(directory) if directory else ROOT / "build/phase2" / backend
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / f"libprecision_{backend}.so"
    temporary = library.with_suffix(f".{os.getpid()}.tmp.so")
    header = ROOT / "public/inference/cpp/exact_binary.h"
    if backend == "cpp":
        source = ROOT / "public/inference/cpp/backend.cpp"
        command = [os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-fPIC", "-shared", "-fopenmp", str(source), "-o", str(temporary)]
    else:
        source = ROOT / "public/cuda/kernels/exact_gemm.cu"
        command = [os.environ.get("NVCC", "nvcc"), "-std=c++17", "-O2", "-shared", "-Xcompiler", "-fPIC", "-ccbin", os.environ.get("CUDAHOSTCXX", "g++-14"), str(source), "-o", str(temporary)]
    subprocess.run(command, check=True, cwd=ROOT)
    temporary.replace(library)
    dependencies = [source,*sorted(header.parent.glob("*.h"))]
    record = {"backend": backend, "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
              "compiler": subprocess.check_output([command[0], "--version"], text=True),
              "command": command, "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest()}
    library.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    return library


class NativeBackend:
    def __init__(self, backend="cpp", library=None):
        if backend not in {"cpp", "cuda"}:
            raise ValueError("unknown native backend")
        self.backend = backend
        self.last_strategy = None
        path = Path(library) if library else ROOT / "build/phase2" / backend / f"libprecision_{backend}.so"
        if not path.is_file():
            raise RuntimeError(f"native library missing; build the {backend} backend first")
        record = json.loads(path.with_suffix(".json").read_text())
        if record["backend"] != backend or record["library_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError("native library identity mismatch")
        for source, expected in record["sources"].items():
            if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != expected:
                raise RuntimeError("native library is stale; rebuild after source changes")
        self.library = C.CDLL(str(path.resolve()))
        self.library.pe_gemm.argtypes = [C.POINTER(Value), C.POINTER(Value), C.POINTER(C.c_uint32), C.c_int, C.c_int, C.c_int, Format]
        self.library.pe_gemm.restype = C.c_int
        self.library.pe_conv2d.argtypes = [C.POINTER(Value), C.POINTER(Value), C.POINTER(C.c_uint32), Conv, Format]
        self.library.pe_conv2d.restype = C.c_int
        self.library.pe_encoded_gemm.argtypes = [C.POINTER(C.c_uint8), C.POINTER(C.c_uint8), C.POINTER(Value), C.POINTER(Value),
                                                C.POINTER(C.c_uint32), C.c_int, C.c_int, C.c_int, Operand, Operand, Format, C.c_int]
        self.library.pe_encoded_gemm.restype = C.c_int
        if backend == "cuda":
            self.library.pe_error.argtypes = [C.c_int]
            self.library.pe_error.restype = C.c_char_p
            if not self.library.pe_device_count():
                raise RuntimeError("CUDA device is unavailable to this process")

    def gemm(self, inputs, weights, *, batch, channels, k, accumulator):
        if any(type(v) is not int or not 0 < v < 2**31 for v in (batch, channels, k)):
            raise ValueError("native GEMM dimensions must be positive int32 values")
        if len(inputs) != batch * k or len(weights) != channels * k:
            raise ValueError("native GEMM payload does not match dimensions")
        left_owner, left_array, left_nonfinite = _native_values(inputs)
        right_owner, right_array, right_nonfinite = _native_values(weights)
        if accumulator == "int32_accumulator" and (left_nonfinite or right_nonfinite):
            raise ValueError("INT32 does not support nonfinite operands")
        results = (C.c_uint32 * (batch * channels))()
        error = self.library.pe_gemm(left_array, right_array, results, batch, channels, k, accumulator_format(accumulator))
        if error:
            detail = self.library.pe_error(error).decode() if self.backend == "cuda" else str(error)
            raise RuntimeError(f"{self.backend} execution failed: {detail}")
        return tuple(results)

    def rational_gemm(self, inputs, weights, **kwargs):
        from public.inference.rational import gemm
        return gemm(self,inputs,weights,**kwargs)

    def binary_admitted(self, inputs, weights, accumulator):
        if accumulator not in {"fp16_e5m10_accumulator","fp32_e8m23_accumulator","int32_accumulator"}:
            return False
        try:
            _native_values(inputs)
            _native_values(weights)
        except ValueError:
            return False
        return True

    def flex_gemm(self, inputs, weights, **kwargs):
        if self.binary_admitted(inputs,weights,kwargs["accumulator"]):
            self.last_strategy = "binary_predecoded"
            return self.gemm(inputs,weights,**kwargs)
        return self.rational_gemm(inputs,weights,**kwargs)

    def flex_conv2d(self, inputs, weights, *, geometry, accumulator):
        g = Conv(**geometry)
        if (any(type(v) is not int or not 0 <= v < 2**31 for v in geometry.values())
                or min(g.n,g.ci,g.h,g.w,g.co,g.kh,g.kw,g.oh,g.ow,g.sh,g.sw,g.dh,g.dw,g.groups) <= 0):
            raise ValueError("invalid native convolution geometry")
        if g.ci % g.groups or g.co % g.groups:
            raise ValueError("invalid native convolution groups")
        if g.oh != (g.h+2*g.ph-g.dh*(g.kh-1)-1)//g.sh+1 or g.ow != (g.w+2*g.pw-g.dw*(g.kw-1)-1)//g.sw+1:
            raise ValueError("native convolution output geometry mismatch")
        if len(inputs) != g.n*g.ci*g.h*g.w or len(weights) != g.co*(g.ci//g.groups)*g.kh*g.kw:
            raise ValueError("native convolution payload mismatch")
        if self.binary_admitted(inputs,weights,accumulator):
            self.last_strategy = "binary_predecoded"
            return self.conv2d(inputs,weights,geometry=geometry,accumulator=accumulator)
        from public.inference.rational import values
        x,w = values(inputs),values(weights)
        g = geometry
        n,ci,h,width,co,kh,kw,oh,ow = (g[key] for key in "n ci h w co kh kw oh ow".split())
        groups = g["groups"]
        k = ci//groups*kh*kw
        states = [0]*(n*co*oh*ow)
        for group in range(groups):
            patches = []
            for b in range(n):
                for y in range(oh):
                    for xx in range(ow):
                        for channel in range(ci//groups):
                            for kr in range(kh):
                                for kc in range(kw):
                                    iy,ix = y*g["sh"]-g["ph"]+kr*g["dh"],xx*g["sw"]-g["pw"]+kc*g["dw"]
                                    patches.append(x[((b*ci+group*(ci//groups)+channel)*h+iy)*width+ix] if 0 <= iy < h and 0 <= ix < width else Fraction(0))
            first,last = group*(co//groups),(group+1)*(co//groups)
            result = self.rational_gemm(patches,w[first*k:last*k],batch=n*oh*ow,channels=co//groups,k=k,accumulator=accumulator)
            for row in range(n*oh*ow):
                b,position = divmod(row,oh*ow)
                for channel in range(co//groups):
                    states[(b*co+first+channel)*oh*ow+position] = result[row*(co//groups)+channel]
        return tuple(states)

    def conv2d(self, inputs, weights, *, geometry, accumulator):
        g = Conv(**geometry)
        if any(type(v) is not int or not 0 <= v < 2**31 for v in geometry.values()):
            raise ValueError("invalid native convolution geometry")
        if min(g.n, g.ci, g.h, g.w, g.co, g.kh, g.kw, g.oh, g.ow, g.sh, g.sw, g.dh, g.dw, g.groups) <= 0:
            raise ValueError("native convolution requires positive dimensions")
        if g.ci % g.groups or g.co % g.groups:
            raise ValueError("invalid native convolution groups")
        if g.oh != (g.h + 2*g.ph - g.dh*(g.kh-1) - 1)//g.sh+1 or g.ow != (g.w + 2*g.pw - g.dw*(g.kw-1) - 1)//g.sw+1:
            raise ValueError("native convolution output geometry mismatch")
        if len(inputs) != g.n*g.ci*g.h*g.w or len(weights) != g.co*(g.ci//g.groups)*g.kh*g.kw:
            raise ValueError("native convolution payload mismatch")
        left_owner, x, left_nonfinite = _native_values(inputs)
        right_owner, w, right_nonfinite = _native_values(weights)
        if accumulator == "int32_accumulator" and (left_nonfinite or right_nonfinite):
            raise ValueError("INT32 does not support nonfinite operands")
        y = (C.c_uint32 * (g.n*g.co*g.oh*g.ow))()
        error = self.library.pe_conv2d(x, w, y, g, accumulator_format(accumulator))
        if error:
            raise RuntimeError(f"native convolution failed: {error}")
        return tuple(y)

    def encoded_gemm(self, inputs, weights, *, batch, channels, k, activation_format, weight_format, accumulator, strategy):
        if strategy not in {"algorithmic", "lookup"}:
            raise ValueError("encoded strategy must be algorithmic or lookup")
        if any(type(v) is not int or not 0 < v < 2**31 for v in (batch, channels, k)):
            raise ValueError("invalid encoded GEMM dimensions")
        if len(inputs) != batch*k or len(weights) != channels*k:
            raise ValueError("encoded GEMM payload mismatch")
        xf, xt = operand_format(activation_format)
        wf, wt = operand_format(weight_format)
        for codes, width in ((inputs, xf.bits), (weights, wf.bits)):
            if any(type(c) is not int or not 0 <= c < 1 << width for c in codes):
                raise ValueError("operand code outside format width")
        if accumulator == "int32_accumulator" and (any(xt[c].kind in {1,2,3} for c in inputs) or any(wt[c].kind in {1,2,3} for c in weights)):
            raise ValueError("INT32 does not support nonfinite operands")
        x, w = (C.c_uint8*len(inputs))(*inputs), (C.c_uint8*len(weights))(*weights)
        y = (C.c_uint32*(batch*channels))()
        error = self.library.pe_encoded_gemm(x,w,xt,wt,y,batch,channels,k,xf,wf,accumulator_format(accumulator),int(strategy=="lookup"))
        if error:
            raise RuntimeError(f"encoded {self.backend} GEMM failed: {error}")
        return tuple(y)
