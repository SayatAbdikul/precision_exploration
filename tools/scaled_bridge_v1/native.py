from __future__ import annotations
import ctypes as C
import json
import os
import subprocess
import numpy as np
from .common import ROOT, digest, file_hash, reference, immutable

FIELDS='n ci h w co kh kw oh ow sh sw ph pw dh dw groups'.split()
class Shape(C.Structure):
    _fields_=[(k,C.c_int32) for k in FIELDS]

def build(backend):
    folder=ROOT/'tools/scaled_bridge_v1'
    source=folder/('native.cpp' if backend=='cpp' else 'native.cu')
    command=(['g++','-std=c++17','-O3','-march=native','-fPIC','-shared','-fopenmp',
              '-fno-fast-math','-ffp-contract=off','-frounding-math',str(source)] if backend=='cpp' else
             ['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','-shared','--ftz=false','--fmad=false',
              '-Xcompiler','-fPIC','-ccbin','g++-14',str(source)])
    compiler=subprocess.check_output([command[0],'--version'],text=True)
    identity={'command':command,'compiler':compiler,
              'sources':{str(p.relative_to(ROOT)):file_hash(p) for p in (source,folder/'kernel.h',folder/'native.py')}}
    path=ROOT/'build/scaled_bridge_v1'/backend/digest(identity)/'bridge.so'
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(f'.{os.getpid()}.tmp')
        subprocess.run(command+['-o',str(tmp)],check=True)
        tmp.replace(path)
        immutable(path.with_suffix('.json'),{**identity,'binary':reference(path)})
    from .common import unseal, checked
    saved=unseal(path.with_suffix('.json'))
    if {k:saved[k] for k in identity}!=identity or checked(saved['binary'])!=path:
        raise ValueError('bridge library drift')
    return path

class Native:
    def __init__(self,backend):
        if backend not in ('cpp','cuda'):raise ValueError('unsupported backend')
        self.path=build(backend)
        self.lib=C.CDLL(str(self.path))
        self.lib.bridge_conv.argtypes=[C.POINTER(C.c_int16),C.POINTER(C.c_int16),C.POINTER(C.c_double),Shape,C.c_int,C.c_int]
        self.lib.bridge_conv.restype=C.c_int
        if backend=='cuda' and not self.lib.bridge_devices():raise RuntimeError('CUDA unavailable; no fallback')
    def conv(self,x,w,attrs,shift,mode):
        for value in (x,w):
            value=np.asarray(value)
            if value.dtype.kind not in 'iu' or not value.size or np.min(value)<-32768 or np.max(value)>32767:
                raise ValueError('native inputs must be representable int16 grid integers')
        x=np.ascontiguousarray(x,dtype=np.int16);w=np.ascontiguousarray(w,dtype=np.int16)
        if x.ndim!=4 or w.ndim!=4 or mode not in ('wide','control') or not 0<=shift<=15:
            raise ValueError('invalid bridge native domain')
        n,ci,h,wi=x.shape;co,cg,kh,kw=w.shape
        sh,sw=attrs.get('stride',[1,1]);ph,pw=attrs.get('padding',[0,0]);dh,dw=attrs.get('dilation',[1,1]);groups=attrs.get('groups',1)
        if min(n,ci,h,wi,co,cg,kh,kw,sh,sw,dh,dw,groups)<=0 or min(ph,pw)<0 or ci!=cg*groups or co%groups:
            raise ValueError('invalid native geometry')
        oh=(h+2*ph-dh*(kh-1)-1)//sh+1;ow=(wi+2*pw-dw*(kw-1)-1)//sw+1
        if min(oh,ow)<=0:raise ValueError('empty convolution')
        bound=int(np.max(np.abs(x.astype(np.int64))))*np.abs(w.reshape(co,-1).astype(np.int64)).sum(1)
        if np.max(bound)>=2**53:raise ValueError('uncertified native prefix range')
        g=Shape(n,ci,h,wi,co,kh,kw,oh,ow,sh,sw,ph,pw,dh,dw,groups)
        y=np.empty((n,co,oh,ow),dtype=np.float64)
        error=self.lib.bridge_conv(x.ctypes.data_as(C.POINTER(C.c_int16)),w.ctypes.data_as(C.POINTER(C.c_int16)),
                                  y.ctypes.data_as(C.POINTER(C.c_double)),g,shift,mode=='control')
        if error:raise RuntimeError(f'bridge kernel error {error}')
        if not np.isfinite(y).all():raise ValueError('nonfinite native result')
        return y
