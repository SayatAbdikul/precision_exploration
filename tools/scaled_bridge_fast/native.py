"""The archived reduction kernels on device pointers (fastdot.cu includes the archived kernel.h)."""
from __future__ import annotations
import ctypes as C
import os
import subprocess
import torch
from .common import ROOT, PACKAGE, LOGICAL, SPEED, BASE_ARCHIVE, digest, file_hash, unseal, seal

KERNEL_SHA256 = '2fb29b0d13af764dd5b5f7fca3df31e0e8656c80e696246f5e2d739f29ef4cec'   # archives 7c6344af and 1f75c923
FIELDS = 'n ci h w co kh kw oh ow sh sw ph pw dh dw groups'.split()
# The archive's nvcc command (artifacts/.../1f75c923.../lib/cuda/bridge2.json), applied to fastdot.cu.
COMMAND = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O3', '-shared', '--ftz=false', '--fmad=false',
           '-Xcompiler', '-fPIC', '-ccbin', 'g++-14']


class Shape(C.Structure):
    _fields_ = [(k, C.c_int32) for k in FIELDS]


def cache_env():
    """JIT cache of the PTX inside the project (never ~/.nv)."""
    os.environ.setdefault('CUDA_CACHE_PATH', str(SPEED / 'cuda-cache'))


def build():
    if file_hash(PACKAGE / 'kernel.h') != KERNEL_SHA256 or file_hash(BASE_ARCHIVE / 'py/scaled_bridge_v2/kernel.h') != KERNEL_SHA256:
        raise ValueError('kernel.h is not the archived kernel')
    source = PACKAGE / 'fastdot.cu'
    compiler = subprocess.check_output([COMMAND[0], '--version'], text=True)
    ident = {'command': COMMAND + [LOGICAL + source.name], 'compiler': compiler,
             'sources': {LOGICAL + p.name: file_hash(p) for p in (source, PACKAGE / 'kernel.h', PACKAGE / 'native.py')}}
    archived = PACKAGE.parent.parent / 'lib' / 'cuda' / 'fastdot.so'
    path = SPEED / 'build' / digest(ident) / 'fastdot.so'
    if archived.exists():
        path = archived
    elif not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        subprocess.run(COMMAND + [str(source), '-o', str(tmp)], check=True)
        tmp.replace(path)
        seal(path.with_suffix('.json'), {**ident, 'binary_sha256': file_hash(path)})
    saved = unseal(path.with_suffix('.json'))
    if {k: saved[k] for k in ident} != ident or file_hash(path) != saved['binary_sha256']:
        raise ValueError('fast library drift')
    return path


def geometry(xs, ws, attrs):
    n, ci, h, wi = xs
    co, cg, kh, kw = ws
    sh, sw = attrs.get('stride', [1, 1]); ph, pw = attrs.get('padding', [0, 0])
    dh, dw = attrs.get('dilation', [1, 1]); groups = attrs.get('groups', 1)
    if min(n, ci, h, wi, co, cg, kh, kw, sh, sw, dh, dw, groups) <= 0 or min(ph, pw) < 0 or ci != cg * groups or co % groups:
        raise ValueError('invalid native geometry')
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (wi + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if min(oh, ow) <= 0:
        raise ValueError('empty convolution')
    return Shape(n, ci, h, wi, co, kh, kw, oh, ow, sh, sw, ph, pw, dh, dw, groups)


class FastNative:
    def __init__(self):
        cache_env()
        self.path = build()
        self.lib = C.CDLL(str(self.path))
        self.lib.fast2_conv.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, Shape, C.c_int, C.c_int, C.c_int, C.c_void_p]
        self.lib.fast2_conv_stat.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, Shape, C.c_int, C.c_int,
                                             C.c_int, C.POINTER(C.c_int64), C.c_void_p]
        torch.cuda.init()
        if not self.lib.fast2_devices():
            raise RuntimeError('CUDA unavailable; no fallback')

    def run(self, x, w, attrs, shift, policy, operand, params=None):
        """x, w: contiguous CUDA tensors of the certified operand type. Returns (dots float64, events int32|None).

        The same domain checks as the archived Native.run; the result checks (NaN, non-finite for non-float
        policies) are returned as a lazy flag for the caller to raise on (deferred to one synchronisation).
        """
        kind = torch.int32 if operand == 32 else torch.int64
        if x.dtype != kind or w.dtype != kind or not x.is_contiguous() or not w.is_contiguous() or not x.is_cuda:
            raise ValueError('native operands must be contiguous CUDA tensors of the certified operand type')
        if x.dim() != 4 or w.dim() != 4 or not 0 <= shift <= 126:
            raise ValueError('invalid bridge native domain')
        g = geometry(tuple(x.shape), tuple(w.shape), attrs)
        y = torch.empty((g.n, g.co, g.oh, g.ow), dtype=torch.float64, device=x.device)
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        events = None
        if policy.parameterised:
            import numpy as np
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
            events = torch.zeros(y.shape, dtype=torch.int32, device=x.device)    # uint32 bits
            error = self.lib.fast2_conv_stat(x.data_ptr(), w.data_ptr(), y.data_ptr(), events.data_ptr(), g, shift,
                                             policy.kernel_id, operand, params.ctypes.data_as(C.POINTER(C.c_int64)), stream)
        else:
            error = self.lib.fast2_conv(x.data_ptr(), w.data_ptr(), y.data_ptr(), g, shift, policy.kernel_id, operand, stream)
        if error:
            raise RuntimeError(f'fast bridge kernel error {error}')
        bad = torch.isnan(y).any() if policy.kind == 'float' else ~torch.isfinite(y).all()
        return y, events, bad
