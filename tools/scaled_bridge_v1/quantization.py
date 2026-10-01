from __future__ import annotations
from functools import lru_cache
from fractions import Fraction
import json
import numpy as np
from .common import ROOT, FORMATS

@lru_cache(None)
def codebook(name):
    if name not in FORMATS:raise ValueError('outside frozen shortlist')
    from public.formats.oracle.number_format import NumberFormat
    fmt=NumberFormat(json.loads((ROOT/f'public/formats/manifests/accepted/{name}.json').read_text()))
    by_value={}
    for c in range(1<<fmt.bits):
        v=fmt.decode(c)
        if v.is_finite():by_value[Fraction(v)]=min(c,by_value.get(Fraction(v),c))
    values=sorted(by_value);shift=max(v.denominator.bit_length()-1 for v in values)
    levels=np.array([float(v) for v in values],dtype=np.float64)
    codes=np.array([by_value[v] for v in values],dtype=np.uint8)
    units=np.array([int(v*(1<<shift)) for v in values],dtype=np.int16)
    boundaries=(levels[:-1]+levels[1:])/2
    ties=np.array([(int(b)&1,int(b))<(int(a)&1,int(a)) for a,b in zip(codes[:-1],codes[1:])])
    return levels,codes,units,boundaries,ties,shift

def quantize(raw,scale,name,*,b_weight=False):
    raw=np.asarray(raw,dtype=np.float32 if b_weight else np.float64)
    scale=np.asarray(scale,dtype=np.float32 if b_weight else np.float64)
    if not np.isfinite(raw).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('nonfinite bridge input/scale')
    levels,codes,units,bounds,ties,shift=codebook(name)
    with np.errstate(over='ignore'):
        y=raw/scale
    pos=np.searchsorted(bounds,y,side='left');adj=np.minimum(pos,len(bounds)-1)
    tie=y==bounds[adj]
    pos+=tie&ties[adj]
    c=codes[pos];u=units[pos]
    diagnostics={'elements':int(raw.size),'clipped_low':int(np.sum(y<levels[0])),
                 'clipped_high':int(np.sum(y>levels[-1])),'ties':int(tie.sum()),
                 'zeros':int(np.sum(u==0)),'normalization_overflow':int(np.sum(~np.isfinite(y)))}
    return c,u,diagnostics

def reconstructed(units,scale,name):
    return np.ldexp(np.asarray(units,dtype=np.float64),-codebook(name)[5])*np.asarray(scale,dtype=np.float64)
