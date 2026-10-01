"""Versioned, graph-bound exact FP64 MAC acceleration.

The selected ResNet18 FP6/FP7 graphs have finite unscaled operands on a small
dyadic grid. A per-channel actual-weight certificate bounds *every* partial
sum below 2**53 grid units. Thus an integer reduction followed by one exact
power-of-two conversion has the same FP64 state bits as sequential Model C.
Unsupported graphs, domains and special values retain the original rational
backend. Bias, activation, store and layer diagnostics remain its responsibility.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from public.experiments.registry.identity import canonical_json_bytes
from public.inference.native import NativeBackend, NativeValues, ROOT, prepare_tensor
from public.inference.reference.arithmetic import format_named, real
from public.inference.tensor import Tensor, parse_encoding, Encoding
from public.quantization.graph.executable import validate_graph

ACCUMULATOR = "fp64_e11m52_accumulator"
SELECTED_GRAPHS = {
    "c6b632bb01f2f3d90455523265e09dd46d2e7897d70e1fde44f821b2b2d20d52": "fp6_e2m3",
    "fbaeb10ee7599c69537fec7ececcbb0c20ab25c12604030cb25bfd11b1f81c04": "fp6_e3m2",
    "765adddcbb54031e6cc988f48a331347876b80058eb47c0329f5a4365bea645d": "fp7_e3m3",
}
GEOMETRY_NAMES = "n ci h w co kh kw oh ow sh sw ph pw dh dw groups".split()


class GridConv(C.Structure):
    _fields_ = [(name, C.c_int32) for name in GEOMETRY_NAMES]


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _source_files(backend: str) -> tuple[Path, ...]:
    shared = ROOT / "public/inference/native_fp64_grid_v2.py"
    source = ROOT / ("public/inference/cpp/fp64_grid_v2.cpp" if backend == "cpp"
                     else "public/cuda/kernels/fp64_grid_v2.cu")
    return shared, source


def _build_config(backend: str) -> tuple[list[str], str, Path]:
    if backend not in {"cpp", "cuda"}:
        raise ValueError("backend must be cpp or cuda")
    source = _source_files(backend)[1]
    if backend == "cpp":
        command = [os.environ.get("CXX", "g++"), "-std=c++17", "-O3", "-fPIC", "-shared", "-fopenmp",
                   "-fno-fast-math", "-ffp-contract=off", "-frounding-math", str(source)]
    else:
        command = [os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"), "-std=c++17", "-O3", "-shared",
                   "--ftz=false", "--fmad=false", "-Xcompiler", "-fPIC",
                   "-ccbin", os.environ.get("CUDAHOSTCXX", "g++-14"), str(source)]
    compiler = subprocess.check_output([command[0], "--version"], text=True)
    return command, compiler, source


def library_path(backend: str, directory: str | Path | None = None) -> Path:
    command, compiler, _ = _build_config(backend)
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in _source_files(backend)}
    key = hashlib.sha256(canonical_json_bytes({"sources": sources, "compiler": compiler, "command": command})).hexdigest()[:20]
    base = Path(directory) if directory is not None else ROOT / "build/useful_quality_fp64_grid_v2"
    return base / backend / key / f"libprecision_grid_v2_{backend}.so"


def build_grid_v2(backend: str, directory: str | Path | None = None) -> Path:
    """Build an immutable content-addressed library; never replace a live .so."""
    command, compiler, _ = _build_config(backend)
    path = library_path(backend, directory)
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in _source_files(backend)}
    if path.exists():
        verify_library(backend, path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    subprocess.run(command + ["-o", str(temporary)], cwd=ROOT, check=True)
    temporary.replace(path)
    record = {"backend": backend, "sources": sources, "compiler": compiler, "command": command,
              "library_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    return path


def verify_library(backend: str, path: str | Path) -> dict:
    path = Path(path)
    record = json.loads(path.with_suffix(".json").read_text())
    if record["backend"] != backend or record["library_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise RuntimeError("version-2 native library identity mismatch")
    for source, digest in record["sources"].items():
        if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != digest:
            raise RuntimeError("version-2 native library source changed; build a new immutable version")
    return record


def _finite_codebook(name: str) -> tuple[int, np.ndarray, int]:
    fmt = format_named(name)
    if (fmt.family != "float" or fmt.manifest["scaling"]["mode"] != "none"
            or fmt.manifest["rounding"] != "rne"):
        raise ValueError("grid requires unscaled finite RNE float encoding")
    values = []
    for code in range(1 << fmt.bits):
        value = real(fmt.decode(code))
        if isinstance(value, Decimal) and not value.is_finite():
            values.append(None)
        else:
            values.append(Fraction(value))
    finite = [v for v in values if v is not None]
    if not finite or any(v.denominator & (v.denominator - 1) for v in finite):
        raise ValueError("finite codebook must be dyadic")
    shift = max(v.denominator.bit_length() - 1 for v in finite)
    units = np.zeros(len(values), dtype=np.int16)
    for i, value in enumerate(values):
        if value is not None:
            integer = value * (1 << shift)
            if integer.denominator != 1 or abs(integer.numerator) > np.iinfo(np.int16).max:
                raise ValueError("finite codebook exceeds INT16 grid")
            units[i] = integer.numerator
    return shift, units, max(abs(int(v)) for v in units)


def _encoding(document: dict, fmt: str) -> None:
    encoding = parse_encoding(document)
    if not isinstance(encoding, Encoding) or encoding.format != fmt or encoding.axis is not None or not (
            encoding.block_size is None and encoding.scales == (Fraction(1),)):
        raise ValueError("grid certificate requires unscaled per-tensor encoding")


@dataclass(frozen=True)
class MacCertificate:
    node: str
    operation: str
    weight_native_sha256: str
    weight_units_sha256: str
    channels: int
    k: int
    shape: tuple[int, ...]
    attrs: tuple[tuple[str, int], ...]
    per_channel_prefix_units: tuple[int, ...]
    maximum_prefix_units: int

    def document(self) -> dict:
        return {"node": self.node, "operation": self.operation,
                "weight_native_sha256": self.weight_native_sha256,
                "weight_units_sha256": self.weight_units_sha256,
                "channels": self.channels, "k": self.k, "shape": list(self.shape),
                "attrs": dict(self.attrs), "per_channel_prefix_units": list(self.per_channel_prefix_units),
                "maximum_prefix_units": self.maximum_prefix_units}


class GraphGridCertificate:
    def __init__(self, graph: dict):
        validate_graph(graph)
        self.graph_sha256 = hashlib.sha256(canonical_json_bytes(graph)).hexdigest()
        self.format = SELECTED_GRAPHS.get(self.graph_sha256)
        if self.format is None:
            raise ValueError("graph is outside the frozen FP6/FP7 selection")
        self.shift, self.code_units, self.maximum_activation_units = _finite_codebook(self.format)
        allowed = np.zeros(2 * self.maximum_activation_units + 1, dtype=np.bool_)
        fmt = format_named(self.format)
        for code in range(1 << fmt.bits):
            value = real(fmt.decode(code))
            if not isinstance(value, Decimal) or value.is_finite():
                allowed[int(self.code_units[code]) + self.maximum_activation_units] = True
        allowed.flags.writeable = False
        self.allowed = allowed
        if graph["manifest_hashes"].keys() != {self.format, ACCUMULATOR}:
            raise ValueError("graph manifest set differs from the frozen FP64 domain")
        domains = dict(graph["inputs"])
        for encoding in domains.values():
            _encoding(encoding, self.format)
        records = []
        for node in graph["nodes"]:
            attrs, op = node["attrs"], node["op"]
            source = domains[node["inputs"][0]]
            if op in {"conv2d", "depthwise_conv2d", "linear"}:
                _encoding(source, self.format)
                _encoding(attrs["output"], self.format)
                if attrs["accumulator"] != ACCUMULATOR:
                    raise ValueError("grid MAC accumulator differs from FP64")
                weight = graph["constants"][node["inputs"][1]]
                _encoding(weight["encoding"], self.format)
                codes = np.asarray(weight["codes"], dtype=np.uint8)
                if not np.all([not (isinstance(v := real(fmt.decode(int(code))), Decimal) and not v.is_finite())
                               for code in np.unique(codes)]):
                    raise ValueError("frozen weight contains a nonfinite code")
                units = np.ascontiguousarray(self.code_units[codes], dtype=np.int16)
                shape = tuple(weight["shape"])
                channels = shape[0]
                k = int(np.prod(shape[1:]))
                bounds = np.abs(units.reshape(channels, k).astype(np.int32)).sum(axis=1, dtype=np.int64)
                per_channel = tuple(self.maximum_activation_units * int(value) for value in bounds)
                bound = max(per_channel)
                if bound > (1 << 31) - 1 or bound >= (1 << 53):
                    raise ValueError("actual-weight prefix bound exceeds certified INT32/FP64 range")
                native = prepare_tensor(Tensor.from_document(weight))
                local_attrs = {}
                if op != "linear":
                    from public.inference.reference.operators import pair
                    for key, prefix in (("stride", "s"), ("padding", "p"), ("dilation", "d")):
                        a, b = pair(attrs[key])
                        local_attrs[prefix + "h"] = a
                        local_attrs[prefix + "w"] = b
                    local_attrs["groups"] = attrs["groups"] if op == "conv2d" else shape[0]
                records.append(MacCertificate(node["name"], op, _sha(native.array), _sha(units),
                                              channels, k, shape, tuple(sorted(local_attrs.items())), per_channel, bound))
            domains[node["name"]] = attrs.get("output", source)
        if len(records) != 21:
            raise ValueError("selected graph must contain all 21 certified MAC nodes")
        self.records = tuple(records)
        self._by_native_sha = {r.weight_native_sha256: r for r in records}
        if len(self._by_native_sha) != len(records):
            raise ValueError("ambiguous duplicate weight identity")

    def document(self) -> dict:
        return {"version": "native_fp64_grid_v2", "graph_sha256": self.graph_sha256,
                "format": self.format, "accumulator": ACCUMULATOR, "shift": self.shift,
                "maximum_activation_units": self.maximum_activation_units,
                "mac_nodes": len(self.records), "maximum_prefix_units": max(r.maximum_prefix_units for r in self.records),
                "records": [r.document() for r in self.records],
                "scope": "finite unscaled codebook operands; exact integer prefixes; original bias and store"}


def _grid_units(values: NativeValues, cert: GraphGridCertificate) -> np.ndarray | None:
    rows = values.array
    kinds = rows["kind"]
    if np.any((kinds != 0) & (kinds != 4)):
        return None
    # NativeValue mantissas are exactly representable in binary64. Reject
    # anything outside the frozen finite codebook before narrowing to INT16.
    scaled = np.ldexp(rows["mantissa"].astype(np.float64), rows["exponent"] + cert.shift)
    if not np.all(np.isfinite(scaled)) or np.any(np.abs(scaled) > cert.maximum_activation_units):
        return None
    if np.any(scaled != np.trunc(scaled)):
        return None
    result = np.ascontiguousarray(scaled, dtype=np.int16)
    if not np.all(cert.allowed[result.astype(np.int32) + cert.maximum_activation_units]):
        return None
    return result


def _finite_doubles(values: NativeValues) -> np.ndarray | None:
    rows = values.array
    if np.any((rows["kind"] != 0) & (rows["kind"] != 4)):
        return None
    result = np.ldexp(rows["mantissa"].astype(np.float64), rows["exponent"])
    result[rows["kind"] == 4] = -0.0
    return np.ascontiguousarray(result) if np.all(np.isfinite(result)) else None


class CertifiedGridBackend:
    """Drop-in native object for a PreparedGraph's `_operators.native`."""
    def __init__(self, backend: str, graph: dict, *, fallback_library=None, grid_library=None):
        self.backend = backend
        self.certificate = GraphGridCertificate(graph)
        self.fallback = NativeBackend(backend, fallback_library)
        path = Path(grid_library) if grid_library is not None else library_path(backend)
        self.library_record = verify_library(backend, path)
        self.library = C.CDLL(str(path.resolve()))
        i16, f64, u64 = C.POINTER(C.c_int16), C.POINTER(C.c_double), C.POINTER(C.c_uint64)
        self.library.pe_grid_v2_gemm.argtypes = [i16, i16, u64, C.c_int, C.c_int, C.c_int, C.c_int]
        self.library.pe_grid_v2_conv2d.argtypes = [i16, i16, u64, GridConv, C.c_int]
        self.library.pe_seq_v2_gemm.argtypes = [f64, f64, u64, C.c_int, C.c_int, C.c_int]
        self.library.pe_seq_v2_conv2d.argtypes = [f64, f64, u64, GridConv]
        for name in ("pe_grid_v2_gemm", "pe_grid_v2_conv2d", "pe_seq_v2_gemm", "pe_seq_v2_conv2d"):
            getattr(self.library, name).restype = C.c_int
        if backend == "cuda":
            self.library.pe_grid_v2_error.argtypes = [C.c_int]
            self.library.pe_grid_v2_error.restype = C.c_char_p
            if not self.library.pe_grid_v2_device_count():
                raise RuntimeError("CUDA is unavailable for version-2 native kernel")
        self.last_strategy = None
        self._weight_cache: dict[int, tuple[NativeValues, MacCertificate, np.ndarray]] = {}

    def _weight(self, values: NativeValues, attrs: dict) -> tuple[MacCertificate, np.ndarray] | None:
        cached = self._weight_cache.get(id(values))
        if cached is not None and cached[0] is values:
            record, units = cached[1:]
        else:
            # Immutable PreparedOperators constants can be cached. Mutable
            # ad-hoc inputs are checked afresh on every call.
            digest = _sha(values.array)
            record = self.certificate._by_native_sha.get(digest)
            if record is None:
                return None
            units = _grid_units(values, self.certificate)
            if units is None or _sha(units) != record.weight_units_sha256:
                return None
            if not values.array.flags.writeable:
                self._weight_cache[id(values)] = (values, record, units)
        if dict(record.attrs) != attrs:
            return None
        return record, units

    def _failure(self, status: int) -> None:
        if status:
            detail = self.library.pe_grid_v2_error(status).decode() if self.backend == "cuda" else str(status)
            raise RuntimeError(f"version-2 {self.backend} native kernel failed: {detail}")

    def _call_gemm(self, x: np.ndarray, w: np.ndarray, batch: int, channels: int, k: int,
                   *, grid: bool, shift: int = 0) -> tuple[int, ...]:
        out = np.empty(batch * channels, dtype=np.uint64)
        kind = C.c_int16 if grid else C.c_double
        pointer = C.POINTER(kind)
        function = self.library.pe_grid_v2_gemm if grid else self.library.pe_seq_v2_gemm
        args = [x.ctypes.data_as(pointer), w.ctypes.data_as(pointer), out.ctypes.data_as(C.POINTER(C.c_uint64)),
                batch, channels, k]
        self._failure(function(*(args + [shift] if grid else args)))
        return tuple(int(v) for v in out)

    def _call_conv(self, x: np.ndarray, w: np.ndarray, geometry: dict, *, grid: bool,
                   shift: int = 0) -> tuple[int, ...]:
        out = np.empty(geometry["n"] * geometry["co"] * geometry["oh"] * geometry["ow"], dtype=np.uint64)
        kind = C.c_int16 if grid else C.c_double
        pointer = C.POINTER(kind)
        function = self.library.pe_grid_v2_conv2d if grid else self.library.pe_seq_v2_conv2d
        args = [x.ctypes.data_as(pointer), w.ctypes.data_as(pointer), out.ctypes.data_as(C.POINTER(C.c_uint64)),
                GridConv(*(geometry[k] for k in GEOMETRY_NAMES))]
        self._failure(function(*(args + [shift] if grid else args)))
        return tuple(int(v) for v in out)

    def flex_gemm(self, inputs, weights, *, batch, channels, k, accumulator):
        if (any(type(v) is not int or not 0 < v < 2**31 for v in (batch, channels, k))
                or len(inputs) != batch * k or len(weights) != channels * k):
            raise ValueError("version-2 GEMM payload or dimensions are invalid")
        if accumulator == ACCUMULATOR and isinstance(inputs, NativeValues) and isinstance(weights, NativeValues):
            found = self._weight(weights, {})
            if found and found[0].operation == "linear" and (found[0].channels, found[0].k) == (channels, k):
                x = _grid_units(inputs, self.certificate)
                if x is not None:
                    self.last_strategy = "certified_fp64_grid_v2"
                    return self._call_gemm(x, found[1], batch, channels, k, grid=True,
                                           shift=self.certificate.shift)
        result = self.fallback.flex_gemm(inputs, weights, batch=batch, channels=channels, k=k,
                                         accumulator=accumulator)
        self.last_strategy = self.fallback.last_strategy
        return result

    def flex_conv2d(self, inputs, weights, *, geometry, accumulator):
        if (set(geometry) != set(GEOMETRY_NAMES)
                or any(type(value) is not int or not 0 <= value < 2**31 for value in geometry.values())):
            raise ValueError("version-2 convolution geometry is invalid")
        g = geometry
        if (min(g[k] for k in ("n", "ci", "h", "w", "co", "kh", "kw", "oh", "ow", "sh", "sw", "dh", "dw", "groups")) <= 0
                or g["ci"] % g["groups"] or g["co"] % g["groups"]
                or g["oh"] != (g["h"] + 2*g["ph"] - g["dh"]*(g["kh"]-1) - 1)//g["sh"] + 1
                or g["ow"] != (g["w"] + 2*g["pw"] - g["dw"]*(g["kw"]-1) - 1)//g["sw"] + 1
                or len(inputs) != g["n"]*g["ci"]*g["h"]*g["w"]
                or len(weights) != g["co"]*(g["ci"]//g["groups"])*g["kh"]*g["kw"]):
            raise ValueError("version-2 convolution payload does not match geometry")
        if accumulator == ACCUMULATOR and isinstance(inputs, NativeValues) and isinstance(weights, NativeValues):
            attrs = {key: geometry[key] for key in ("sh", "sw", "ph", "pw", "dh", "dw", "groups")}
            found = self._weight(weights, attrs)
            if found and found[0].operation in {"conv2d", "depthwise_conv2d"}:
                record = found[0]
                if (record.channels, record.k, record.shape[2], record.shape[3]) == (
                        geometry["co"], (geometry["ci"] // geometry["groups"]) * geometry["kh"] * geometry["kw"],
                        geometry["kh"], geometry["kw"]):
                    x = _grid_units(inputs, self.certificate)
                    if x is not None:
                        self.last_strategy = "certified_fp64_grid_v2"
                        return self._call_conv(x, found[1], geometry, grid=True,
                                               shift=self.certificate.shift)
        result = self.fallback.flex_conv2d(inputs, weights, geometry=geometry, accumulator=accumulator)
        self.last_strategy = self.fallback.last_strategy
        return result

    def sequential_gemm(self, inputs: NativeValues, weights: NativeValues, *, batch, channels, k):
        """Bounded finite-dyadic sequential FP64 path for oracle comparisons."""
        x, w = _finite_doubles(inputs), _finite_doubles(weights)
        if x is None or w is None:
            raise ValueError("sequential FP64 path requires finite dyadics")
        self.last_strategy = "sequential_fp64_fma_v2"
        return self._call_gemm(x, w, batch, channels, k, grid=False)

    def sequential_conv2d(self, inputs: NativeValues, weights: NativeValues, *, geometry):
        x, w = _finite_doubles(inputs), _finite_doubles(weights)
        if x is None or w is None:
            raise ValueError("sequential FP64 path requires finite dyadics")
        self.last_strategy = "sequential_fp64_fma_v2"
        return self._call_conv(x, w, geometry, grid=False)
