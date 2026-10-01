from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from .common import array_hash,digest
from .quantization import codebook,quantize,reconstructed
from .native import Native

@dataclass
class State:
    codes: np.ndarray
    units: np.ndarray
    scale: float

def maxpool(units,attrs):
    from numpy.lib.stride_tricks import sliding_window_view
    kh,kw=attrs['kernel_size'];sh,sw=attrs['stride'];ph,pw=attrs['padding']
    if attrs['dilation']!=[1,1]:raise ValueError('unsupported maxpool dilation')
    padded=np.pad(units,((0,0),(0,0),(ph,ph),(pw,pw)),constant_values=-32768)
    windows=sliding_window_view(padded,(kh,kw),axis=(-2,-1))
    return windows[:,:,::sh,::sw].max(axis=(-2,-1))

def oracle_dots(x,w,attrs,actual,shift,mode):
    """Independent fixed output dots: exact Python integers / rational Model C."""
    from fractions import Fraction
    from public.inference.reference.arithmetic import model_c,format_named
    n,ci,h,wi=x.shape;co,cg,kh,kw=w.shape;_,_,oh,ow=actual.shape
    sh,sw=attrs.get('stride',[1,1]);ph,pw=attrs.get('padding',[0,0]);dh,dw=attrs.get('dilation',[1,1]);g=attrs.get('groups',1)
    selected=sorted({0,actual.size//2,actual.size-1})
    for flat in selected:
        bn,oc,oy,ox=np.unravel_index(flat,actual.shape);pairs=[]
        for c in range(cg):
            for ky in range(kh):
                for kx in range(kw):
                    iy=oy*sh-ph+ky*dh;ix=ox*sw-pw+kx*dw
                    a=int(x[bn,(oc//(co//g))*cg+c,iy,ix]) if 0<=iy<h and 0<=ix<wi else 0
                    pairs.append((a,int(w[oc,c,ky,kx])))
        if mode=='wide':expected=np.ldexp(float(sum(a*b for a,b in pairs)),-2*shift)
        else:
            fmt=format_named('fp32_e8m23_accumulator')
            bits=model_c(((Fraction(a,1<<shift),Fraction(b,1<<shift)) for a,b in pairs),fmt)
            expected=float(fmt.decode(bits))
        if np.float64(expected).tobytes()!=actual.flat[flat].tobytes():
            raise ValueError(f'independent rational dot mismatch {mode} output {flat}')
    return len(selected)

class Engine:
    def __init__(self,export,arrays,backend,mode):
        self.export,self.arrays,self.name=export,arrays,export['format']
        self.mode=mode;self.native=Native(backend);self.shift=codebook(self.name)[5]
        self.scales=export['scales']['activation_scales']
        self.bounds={x['node']:x for x in export['bounds']}
        for node in export['nodes']:
            if node['op'] in ('conv','linear'):
                w=arrays[node['name']+'_units'];cert=self.bounds[node['name']]
                bound=cert['maximum_activation_units']*np.abs(w.reshape(w.shape[0],-1).astype(np.int64)).sum(1)
                if bound.tolist()!=cert['per_channel_prefix_units'] or np.max(bound)>=2**53:
                    raise ValueError('new scaled-grid prefix certificate changed')
                if array_hash(w)!=cert['weight_units_sha256']:raise ValueError('certified weights changed')
    def run(self,inputs,*,oracle=False):
        values={};trace={};timing={'mac':0.,'nonmac':0.,'quantization_trace':0.,'oracle':0.}
        oracle_count=0;start=time.perf_counter()
        for node in self.export['nodes']:
            key,op=node['name'],node['op'];args=[values[n] for n in node['inputs']]
            mac=None;tick=time.perf_counter()
            if op=='output':
                output=reconstructed(args[0].units,args[0].scale,self.name)
                break
            if op=='input':raw=np.asarray(inputs,dtype=np.float64)
            elif op in ('identity','flatten'):
                s=args[0];shape=(s.units.shape[0],-1) if op=='flatten' else s.units.shape
                state=State(s.codes.reshape(shape),s.units.reshape(shape),s.scale)
                values[key]=state
                trace[key]={'codes':array_hash(state.codes),'state':array_hash(reconstructed(state.units,state.scale,self.name)),
                            'raw':None,'mac':None,'diagnostics':{'skipped':True}}
                timing['nonmac']+=time.perf_counter()-tick
                continue
            elif op in ('conv','linear'):
                s=args[0];x=s.units;w=self.arrays[key+'_units'];attrs=node['attrs']
                if op=='linear':x=x.reshape(x.shape[0],x.shape[1],1,1);w=w[:,:,None,None]
                mstart=time.perf_counter();mac=self.native.conv(x,w,attrs,self.shift,self.mode)
                timing['mac']+=time.perf_counter()-mstart
                if oracle:
                    ostart=time.perf_counter();oracle_count+=oracle_dots(x,w,attrs,mac,self.shift,self.mode)
                    timing['oracle']+=time.perf_counter()-ostart
                raw=mac*np.float64(s.scale)
                ws=np.asarray(self.export['scales']['weight_scales'][key],dtype=np.float64)[None,:,None,None]
                raw=raw*ws
                raw=raw+self.arrays[key+'_bias'].astype(np.float64)[None,:,None,None]
                if op=='linear':raw=raw[:,:,0,0]
            elif op=='add':raw=reconstructed(args[0].units,args[0].scale,self.name)+reconstructed(args[1].units,args[1].scale,self.name)
            elif op=='relu':raw=reconstructed(np.maximum(args[0].units,0),args[0].scale,self.name)
            elif op=='maxpool':raw=reconstructed(maxpool(args[0].units,node['attrs']),args[0].scale,self.name)
            elif op=='avgpool':
                s=args[0];summed=s.units.astype(np.int64).sum(axis=(-2,-1),keepdims=True)
                raw=np.ldexp(summed.astype(np.float64),-self.shift)
                raw=raw/np.float64(s.units.shape[-2]*s.units.shape[-1])
                raw=raw*np.float64(s.scale)
            else:raise ValueError('unsupported bridge node')
            elapsed=time.perf_counter()-tick
            # Detailed MAC/oracle timers are subsets, so keep a disjoint non-MAC timer.
            qstart=time.perf_counter();c,u,diagnostics=quantize(raw,self.scales[key],self.name)
            state=State(c,u,self.scales[key]);values[key]=state
            trace[key]={'codes':array_hash(c),'state':array_hash(reconstructed(u,state.scale,self.name)),
                        'raw':array_hash(raw),'mac':array_hash(mac) if mac is not None else None,
                        'diagnostics':diagnostics}
            timing['quantization_trace']+=time.perf_counter()-qstart
            timing['nonmac']+=elapsed
        timing['nonmac']-=timing['mac']+timing['oracle']
        timing['execution']=time.perf_counter()-start
        if not np.isfinite(output).all():raise ValueError('nonfinite output')
        import torch
        top5=torch.from_numpy(output).topk(5,dim=1).indices.tolist()
        return {'layers':trace,'output':array_hash(output),'top5':top5[0],
                'diagnostic_signature':digest({k:v['diagnostics'] for k,v in trace.items()}),
                'timing':timing,'oracle_dot_checks':oracle_count}

def numerical(record):
    return {k:record[k] for k in ('layers','output','top5','diagnostic_signature')}
