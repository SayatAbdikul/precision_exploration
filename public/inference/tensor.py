"""Portable encoded tensors with explicit scale addressing and layout."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import prod
from itertools import product

from public.inference.reference.arithmetic import decode, encode, format_named, real, decimal


@dataclass(frozen=True)
class SharedEncoding:
    """An explicit runtime MSE scale policy; tensors store resolved Encoding."""
    format: str
    axis: int = 1
    block_size: int = 32
    policy: str = "intrinsic_mse_v1"

    def __post_init__(self):
        manifest = format_named(self.format).manifest
        if (self.policy != "intrinsic_mse_v1" or manifest["scaling"]["mode"] != "intrinsic_shared"
                or self.block_size != manifest["block"]["block_size"] or type(self.axis) is not int or self.axis < 0):
            raise ValueError("invalid intrinsic shared encoding policy")

    def document(self):
        return {"format":self.format,"axis":self.axis,"block_size":self.block_size,"policy":self.policy}


def parse_encoding(document):
    return SharedEncoding(**document) if "policy" in document else Encoding(**document)


def quantize_shared(values, shape, policy):
    from public.quantization.calibration.mse import mse_scale
    shape, values = tuple(shape),tuple(values)
    if not shape or policy.axis >= len(shape) or any(type(n) is not int or n <= 0 for n in shape) or len(values) != prod(shape):
        raise ValueError("invalid shared tensor shape")
    scale_shape = list(shape)
    scale_shape[policy.axis] = (shape[policy.axis]+policy.block_size-1)//policy.block_size
    scales = []
    for index in indices(scale_shape):
        start = index[policy.axis]*policy.block_size
        block = []
        for k in range(start,min(start+policy.block_size,shape[policy.axis])):
            address = list(index)
            address[policy.axis] = k
            block.append(values[offset(shape,address)])
        scales.append(mse_scale(block,policy.format)["scale"])
    return Tensor.quantize(values,shape,Encoding(policy.format,tuple(scales),policy.axis,policy.block_size))


def indices(shape):
    return product(*(range(size) for size in shape))


def offset(shape, index):
    if len(shape) != len(index):
        raise ValueError("tensor index rank mismatch")
    result = 0
    for dimension, position in zip(shape, index):
        if type(position) is not int or not 0 <= position < dimension:
            raise ValueError("tensor index outside shape")
        result = result * dimension + position
    return result


@dataclass(frozen=True)
class Encoding:
    format: str
    scales: tuple = ("1",)
    axis: int | None = None
    block_size: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "scales", tuple(real(s) for s in self.scales))
        fmt = format_named(self.format)
        if fmt.bits > 8:
            raise ValueError("tensor storage formats must be at most 8 bits")
        if not self.scales or any(not isinstance(s, Fraction) or s <= 0 for s in self.scales):
            raise ValueError("tensor scales must be finite and positive")
        mode = fmt.manifest["scaling"]["mode"]
        if mode == "none" and (self.scales != (Fraction(1),) or self.axis is not None):
            raise ValueError("Experiment A self-scaling format forbids external scales")
        if mode == "intrinsic_shared":
            if self.axis is None or self.block_size != fmt.manifest["block"]["block_size"]:
                raise ValueError("shared format requires its declared block size and explicit K axis")
            for scale in self.scales:
                fmt.decode(0, scale=decimal(scale))
        elif self.block_size is not None:
            raise ValueError("block metadata requires a shared-scale format")

    def validate_shape(self, shape):
        if not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError("tensor requires positive dimensions")
        if self.axis is None:
            count = 1
        else:
            if type(self.axis) is not int or not 0 <= self.axis < len(shape):
                raise ValueError("scale axis outside tensor rank")
            if self.block_size:
                scale_shape = list(shape)
                scale_shape[self.axis] = (shape[self.axis] + self.block_size - 1) // self.block_size
                count = prod(scale_shape)
            else:
                count = shape[self.axis]
        if len(self.scales) != count:
            raise ValueError(f"scale count {len(self.scales)} does not match shape (expected {count})")

    def scale_at(self, shape, index):
        if self.axis is None:
            return self.scales[0]
        if self.block_size is None:
            return self.scales[index[self.axis]]
        scale_shape, scale_index = list(shape), list(index)
        scale_shape[self.axis] = (shape[self.axis] + self.block_size - 1) // self.block_size
        scale_index[self.axis] //= self.block_size
        return self.scales[offset(scale_shape, scale_index)]

    def document(self):
        return {"format": self.format, "scales": [str(decimal(s)) for s in self.scales],
                "axis": self.axis, "block_size": self.block_size}


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, ...]
    codes: tuple[int, ...]
    encoding: Encoding

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "codes", tuple(self.codes))
        self.encoding.validate_shape(self.shape)
        count = 1 << format_named(self.encoding.format).bits
        if len(self.codes) != prod(self.shape) or any(type(c) is not int or not 0 <= c < count for c in self.codes):
            raise ValueError("encoded payload does not match shape or format width")

    @classmethod
    def quantize(cls, values, shape, encoding):
        if isinstance(encoding,SharedEncoding):
            return quantize_shared(values,shape,encoding)
        shape, values = tuple(shape), tuple(values)
        encoding.validate_shape(shape)
        if len(values) != prod(shape):
            raise ValueError("input value count does not match shape")
        fmt = format_named(encoding.format)
        codes = [encode(fmt, value, encoding.scale_at(shape, idx)) for value, idx in zip(values, indices(shape))]
        return cls(shape, tuple(codes), encoding)

    def value(self, index):
        return decode(format_named(self.encoding.format), self.codes[offset(self.shape, index)],
                      self.encoding.scale_at(self.shape, index))

    def values(self):
        return tuple(self.value(index) for index in indices(self.shape))

    def document(self):
        return {"shape": list(self.shape), "codes": list(self.codes), "encoding": self.encoding.document()}

    @classmethod
    def from_document(cls, document):
        if set(document) != {"shape", "codes", "encoding"}:
            raise ValueError("unexpected tensor fields")
        return cls(tuple(document["shape"]), tuple(document["codes"]), Encoding(**document["encoding"]))
