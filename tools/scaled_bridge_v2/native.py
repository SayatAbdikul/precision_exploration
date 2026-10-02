from __future__ import annotations
import ctypes as C
import os
import subprocess
import numpy as np
from .common import ROOT, PACKAGE, LOGICAL, digest, file_hash, reference, immutable, unseal, checked
from .accumulators import resolve

FIELDS = 'n ci h w co kh kw oh ow sh sw ph pw dh dw groups'.split()


class Shape(C.Structure):
    _fields_ = [(k, C.c_int32) for k in FIELDS]


def build(backend):
    source = PACKAGE / ('native.cpp' if backend == 'cpp' else 'native.cu')
    command = (['g++', '-std=c++17', '-O3', '-march=native', '-fPIC', '-shared', '-fopenmp',
                '-fno-fast-math', '-ffp-contract=off', '-frounding-math'] if backend == 'cpp' else
               ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O3', '-shared', '--ftz=false', '--fmad=false',
                '-Xcompiler', '-fPIC', '-ccbin', 'g++-14'])
    compiler = subprocess.check_output([command[0], '--version'], text=True)
    # The identity names sources by their logical (live) names, so an archived
    # copy of the package resolves to the same library as the live sources did.
    identity = {'command': command + [LOGICAL + source.name], 'compiler': compiler,
                'sources': {LOGICAL + p.name: file_hash(p) for p in (source, PACKAGE / 'kernel.h', PACKAGE / 'native.py')}}
    archived = PACKAGE.parent.parent / 'lib' / backend / 'bridge2.so'
    path = ROOT / 'build/scaled_bridge_v2' / backend / digest(identity) / 'bridge2.so'
    if archived.exists():
        path = archived
    elif not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        subprocess.run(command + [str(source), '-o', str(tmp)], check=True)
        tmp.replace(path)
        immutable(path.with_suffix('.json'), {**identity, 'binary_sha256': file_hash(path)})
    saved = unseal(path.with_suffix('.json'))
    if {k: saved[k] for k in identity} != identity or file_hash(path) != saved['binary_sha256']:
        raise ValueError('bridge v2 library drift')
    return path


def operand_width(max_abs_input, max_abs_weight, max_abs_prefix):
    """Kernel operand class for a certified reduction, or None if not admissible."""
    if max(max_abs_input, max_abs_weight) < 2**31 and max_abs_prefix < 2**63:
        return 32
    if max(max_abs_input, max_abs_weight) < 2**32 and max_abs_prefix < 2**127:
        return 64
    return None


def geometry(x, w, attrs):
    n, ci, h, wi = x.shape
    co, cg, kh, kw = w.shape
    sh, sw = attrs.get('stride', [1, 1]); ph, pw = attrs.get('padding', [0, 0])
    dh, dw = attrs.get('dilation', [1, 1]); groups = attrs.get('groups', 1)
    if min(n, ci, h, wi, co, cg, kh, kw, sh, sw, dh, dw, groups) <= 0 or min(ph, pw) < 0 or ci != cg * groups or co % groups:
        raise ValueError('invalid native geometry')
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (wi + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if min(oh, ow) <= 0:
        raise ValueError('empty convolution')
    return Shape(n, ci, h, wi, co, kh, kw, oh, ow, sh, sw, ph, pw, dh, dw, groups)


class Native:
    def __init__(self, backend):
        if backend not in ('cpp', 'cuda'):
            raise ValueError('unsupported backend')
        self.backend, self.path = backend, build(backend)
        self.lib = C.CDLL(str(self.path))
        self.lib.bridge2_conv.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_double), Shape, C.c_int, C.c_int, C.c_int]
        self.lib.bridge2_conv.restype = C.c_int
        self.lib.bridge2_conv_stat.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(C.c_double), C.POINTER(C.c_uint32),
                                               Shape, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_int64)]
        self.lib.bridge2_conv_stat.restype = C.c_int
        if backend == 'cuda' and not self.lib.bridge2_devices():
            raise RuntimeError('CUDA unavailable; no fallback')

    def run(self, x, w, attrs, shift, policy, operand, params=None):
        """Certified call: x and w are contiguous arrays of the operand type.

        The caller has proved the prefix bound and operand ranges. Returns the
        binary64 code-level dots and, for parameterised policies, the event
        word of every output (None otherwise). Only a float accumulator may
        return a non-finite dot; the caller records it.
        """
        policy = resolve(policy)
        kind = np.int32 if operand == 32 else np.int64
        if x.dtype != kind or w.dtype != kind or not x.flags.c_contiguous or not w.flags.c_contiguous:
            raise ValueError('native operands must be contiguous arrays of the certified operand type')
        if x.ndim != 4 or w.ndim != 4 or not 0 <= shift <= 126:
            raise ValueError('invalid bridge native domain')
        g = geometry(x, w, attrs)
        y = np.empty((g.n, g.co, g.oh, g.ow), dtype=np.float64)
        events = None
        if policy.parameterised:
            if params is None:
                params, _ = policy.node(shift)
            params = np.ascontiguousarray(params, dtype=np.int64)
            if params.shape != (4,):
                raise ValueError('invalid policy parameters')
            if policy.kind == 'sat' and not 2 <= int(params[0]) <= 63:
                raise ValueError('saturating width outside the kernel domain')
            if policy.kind == 'float' and not (2 <= int(params[0]) <= 53 and -1000 <= int(params[1]) <= 127
                                               and int(params[1]) <= int(params[2]) <= 1000):
                raise ValueError('float accumulator parameters outside the kernel domain')
            events = np.zeros(y.shape, dtype=np.uint32)
            error = self.lib.bridge2_conv_stat(x.ctypes.data_as(C.c_void_p), w.ctypes.data_as(C.c_void_p),
                                               y.ctypes.data_as(C.POINTER(C.c_double)),
                                               events.ctypes.data_as(C.POINTER(C.c_uint32)), g, shift,
                                               policy.kernel_id, operand, params.ctypes.data_as(C.POINTER(C.c_int64)))
        else:
            error = self.lib.bridge2_conv(x.ctypes.data_as(C.c_void_p), w.ctypes.data_as(C.c_void_p),
                                          y.ctypes.data_as(C.POINTER(C.c_double)), g, shift, policy.kernel_id, operand)
        if error:
            raise RuntimeError(f'bridge v2 kernel error {error}')
        if np.isnan(y).any() or (policy.kind != 'float' and not np.isfinite(y).all()):
            raise ValueError('nonfinite native result')
        return y, events

    def conv(self, x, w, attrs, shift, policy):
        """Self-certifying call for conformance work: dots only."""
        return self.conv_stat(x, w, attrs, shift, policy)[0]

    def conv_stat(self, x, w, attrs, shift, policy):
        """Self-certifying call for conformance work: bounds the reduction itself."""
        policy = resolve(policy)
        x, w = np.asarray(x), np.asarray(w)
        for value in (x, w):
            if value.dtype.kind not in 'iu' or not value.size:
                raise ValueError('native inputs must be grid integers')
        if x.ndim != 4 or w.ndim != 4:
            raise ValueError('invalid bridge native domain')
        max_x = max(abs(int(x.min())), abs(int(x.max())))
        max_w = max(abs(int(w.min())), abs(int(w.max())))
        if max_w >= 2**32 or max_x >= 2**32:
            raise ValueError('grid integers exceed the certified operand range')
        absolute = np.abs(w.astype(np.int64)).reshape(w.shape[0], -1).sum(1)
        bound = max_x * int(absolute.max())
        operand = operand_width(max_x, max_w, bound)
        if operand is None:
            raise ValueError('uncertified native prefix range')
        if policy.needs_float32_exact_operands:
            for value in (x, w):
                if not np.array_equal(value.astype(np.float32).astype(np.int64), value.astype(np.int64)):
                    raise ValueError('operands are not exact binary32 values')
        kind = np.int32 if operand == 32 else np.int64
        return self.run(np.ascontiguousarray(x, dtype=kind), np.ascontiguousarray(w, dtype=kind), attrs, shift, policy, operand)
