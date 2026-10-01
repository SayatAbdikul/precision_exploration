"""Independent rational witnesses, charged by the study controller."""
from fractions import Fraction as Q
import time
import numpy as np
from public.inference.reference.arithmetic import format_named, model_c
from .common import *
from .quantization import codebook, quantize, reconstructed
from .native import Native
from .engine import oracle_dots, maxpool

def require(condition,message):
    if not condition:raise ValueError(message)

def rounded64(value):
    return float(format_named('fp64_e11m52_accumulator').rounded(value))

def primitive_checks():
    start=time.perf_counter();checks=0
    native={b:Native(b) for b in ('cpp','cuda')}
    rng=np.random.default_rng(20260928)
    for name in FORMATS:
        levels,codes,units,bounds,ties,shift=codebook(name)
        # Oracle enumeration comes from the manifest, not the vectorized store.
        fmt=format_named(name);book={}
        for c in range(1<<fmt.bits):
            value=fmt.decode(c)
            if value.is_finite():book[Q(value)]=min(c,book.get(Q(value),c))
        samples=np.concatenate([levels,bounds,np.nextafter(bounds,-np.inf),np.nextafter(bounds,np.inf),[-1e300,1e300,-0.,0.]])
        for scale in (1.,float(np.float32(.0137)),float(np.float32(2.**-126)),float(np.float32(2.**120))):
            for value in samples:
                raw=float(value)
                with np.errstate(over='ignore'):
                    actual_y=np.float64(raw)/np.float64(scale)
                expected_y=rounded64(Q(raw)/Q(scale))
                require(actual_y==expected_y,'binary64 normalization oracle')
                if np.isinf(expected_y):expected=book[max(book) if expected_y>0 else min(book)]
                else:
                    v=Q(expected_y)
                    nearest=min(book,key=lambda k:(abs(k-v),book[k]&1,book[k]))
                    expected=book[nearest]
                actual,_,_=quantize(np.array([raw]),scale,name)
                require(int(actual[0])==expected,'rational scaled store mismatch');checks+=1
        for invalid in (np.nan,np.inf,-np.inf):
            try:quantize([invalid],1.,name)
            except ValueError:checks+=1
            else:raise ValueError('nonfinite store admitted')
        for scale in (0.,-1.,np.inf,np.nan):
            try:quantize([1.],scale,name)
            except ValueError:checks+=1
            else:raise ValueError('invalid scale admitted')
        # Every finite code pair: exact product and one-step FMA witness.
        x=units.reshape(1,1,1,-1);w=units.reshape(-1,1,1,1)
        expected=np.ldexp(units.astype(np.float64)[:,None]*units.astype(np.float64)[None,:],-2*shift).reshape(1,len(units),1,len(units))
        expected[expected==0]=0.
        for backend,kernel in native.items():
            for mode in ('wide','control'):
                got=kernel.conv(x,w,{},shift,mode)
                require(np.array_equal(expected.view('u8'),got.view('u8')),'exhaustive pair product mismatch')
                checks+=got.size
        # Independent small convolution indexing: groups, stride, dilation,
        # padding, and all outputs (not just three graph sample dots).
        for attrs in ({'groups':2,'stride':[2,1],'padding':[1,2],'dilation':[1,2]},
                      {'groups':1,'stride':[1,2],'padding':[0,1],'dilation':[2,1]}):
            g=attrs['groups'];x=rng.choice(units,size=(1,4,6,7));w=rng.choice(units,size=(4,4//g,2,3))
            for mode in ('wide','control'):
                cpu=native['cpp'].conv(x,w,attrs,shift,mode);gpu=native['cuda'].conv(x,w,attrs,shift,mode)
                require(np.array_equal(cpu.view('u8'),gpu.view('u8')),'geometry CPU/CUDA mismatch')
                for oc in range(4):
                    for oy in range(cpu.shape[2]):
                        for ox in range(cpu.shape[3]):
                            pairs=[]
                            for c in range(4//g):
                                for ky in range(2):
                                    for kx in range(3):
                                        iy=oy*attrs['stride'][0]-attrs['padding'][0]+ky*attrs['dilation'][0]
                                        ix=ox*attrs['stride'][1]-attrs['padding'][1]+kx*attrs['dilation'][1]
                                        a=int(x[0,oc//(4//g)*(4//g)+c,iy,ix]) if 0<=iy<6 and 0<=ix<7 else 0
                                        pairs.append((a,int(w[oc,c,ky,kx])))
                            if mode=='wide':expect=np.ldexp(float(sum(a*b for a,b in pairs)),-2*shift)
                            else:
                                f=format_named('fp32_e8m23_accumulator')
                                expect=float(f.decode(model_c(((Q(a,1<<shift),Q(b,1<<shift)) for a,b in pairs),f)))
                            require(np.float64(expect).tobytes()==cpu[0,oc,oy,ox].tobytes(),'independent geometry oracle mismatch');checks+=1
        # Long cancellation and rounding-sensitive FMA sequence in the admitted grid.
        hi=int(max(units));seq=np.array([hi]*2048+[1]*17+[-hi]*2048,dtype=np.int16)
        x=seq.reshape(1,-1,1,1);w=np.full_like(x,hi).reshape(1,-1,1,1)
        for mode in ('wide','control'):
            for kernel in native.values():
                y=kernel.conv(x,w,{},shift,mode)
                checks+=oracle_dots(x,w,{},y,shift,mode)
        # Residual, MAC scale/bias and average operations use an independent
        # rational RNE64 operation sequence before the code store.
        for _ in range(160):
            a,b=rng.choice(units,size=2).astype(int);sa=float(np.float32(rng.uniform(.0001,3)));sb=float(np.float32(rng.uniform(.0001,3)))
            ra=rounded64(Q(int(a),1<<shift)*Q(sa));rb=rounded64(Q(int(b),1<<shift)*Q(sb))
            expect=rounded64(Q(ra)+Q(rb));got=float(reconstructed(a,sa,name)+reconstructed(b,sb,name))
            require(np.float64(expect).tobytes()==np.float64(got).tobytes(),'residual alignment mismatch');checks+=1
            bias=float(np.float32(rng.normal()));dot=float(int(a)*int(b))/(1<<(2*shift))
            expect=rounded64(Q(rounded64(Q(rounded64(Q(dot)*Q(sa)))*Q(sb)))+Q(bias))
            got=np.float64(dot)*np.float64(sa);got=got*np.float64(sb);got=got+np.float64(bias)
            require(np.float64(expect).tobytes()==got.tobytes(),'scale/bias sequence mismatch');checks+=1
            expect=rounded64(Q(rounded64(Q(int(a)+int(b),1<<shift)/49))*Q(sa))
            got=np.ldexp(np.float64(a+b),-shift)/np.float64(49);got=got*np.float64(sa)
            require(np.float64(expect).tobytes()==got.tobytes(),'average sequence mismatch');checks+=1
        p=np.array([[[[-1,-2],[-3,-4]]]],dtype=np.int16)
        require(np.array_equal(maxpool(p,{'kernel_size':[2,2],'stride':[1,1],'padding':[1,1],'dilation':[1,1]}),
            np.array([[[[-1,-1,-2],[-1,-1,-2],[-3,-3,-4]]]],dtype=np.int16)),'negative padded pool mismatch');checks+=1
    for kernel in native.values():
        for invalid in (np.array([[[[32768]]]],dtype=np.int32),np.array([[[[.5]]]])):
            try:kernel.conv(invalid,np.ones((1,1,1,1),dtype=np.int16),{},3,'wide')
            except ValueError:checks+=1
            else:raise ValueError('native invalid grid input accepted')
    record={'status':'pass','checks':checks,'seconds':time.perf_counter()-start,'sources':sources(),
            'native':{b:reference(n.path) for b,n in native.items()},
            'scope':'exhaustive finite code pairs; rational stores/postops; independently indexed small convs; long FMA; special failures'}
    immutable(run_root()/'conformance.json',record)
    print(f'primitive conformance: {checks} checks passed',flush=True)
    return record
