from __future__ import annotations
import time
import numpy as np
from .common import *

def panel(name,mode,backend,target):
    started=time.perf_counter()
    from tools.experiment_b.classifier import configure,load_model,image_batch
    from tools.experiment_b.common import dataset
    from .export import load_export
    from .engine import Engine,numerical
    configure('cuda' if backend=='cuda' else 'cpu')
    ex,arrays,ref=load_export(name)
    graph,transform,original=load_model('resnet18','cpu');del original
    _,rows,payload=dataset('imagenet_screen_1k')
    engine=Engine(ex,arrays,backend,mode)
    setup=time.perf_counter()-started;folder=run_root()/name/f'{mode}-{backend}'
    for i in range(target):
        path=folder/f'{i:04d}.json'
        if path.exists():
            old=unseal(path)
            if old['export']!=ref or old['sample']!=rows[i] or old['mode']!=mode or old['backend']!=backend:
                raise ValueError('panel checkpoint drift')
            continue
        tick=time.perf_counter();inputs=image_batch(rows[i:i+1],payload,transform,'cpu').numpy()
        preprocess=time.perf_counter()-tick
        value=engine.run(inputs,oracle=(backend=='cpp' and i==0))
        if backend=='cpp' and mode=='wide' and i==0:
            from .reference import FXWideReference
            tick=time.perf_counter();witness=FXWideReference(graph,ex,arrays).evaluate(inputs)
            if witness!=numerical(value):
                failed=[k for k in witness['layers'] if witness['layers'][k]!=value['layers'][k]]
                raise ValueError('independent full FX graph mismatch '+str(failed[:4]))
            immutable(run_root()/name/'wide-reference.json',{'export':ref,'sample':rows[0],'numerical':witness,
                      'seconds':time.perf_counter()-tick,'basis':'independent FX and exact integer-grid FP64 convolution under per-channel <2^53 absolute-sum proof'})
        record={'export':ref,'sample':rows[i],'index':i,'mode':mode,'backend':backend,**value,
                'preprocess_seconds':preprocess,'setup_seconds':setup if i==0 else 0.}
        # Fail closed if a code/binary/configuration drift occurred during compute.
        if sources()!=ex['sources']:raise ValueError('bridge sources changed during worker')
        immutable(path,record)
        print(f'{name} {mode}-{backend} {i+1}/{target}: {value["timing"]["execution"]:.3f}s',flush=True)

def gate(name):
    from .engine import numerical
    from .export import load_export
    ex,_,ref=load_export(name)
    conf=run_root()/'conformance.json';unseal(conf)
    proof=run_root()/name/'wide-reference.json';unseal(proof)
    refs=[]
    for mode in ('wide','control'):
        for i in range(8):
            paths=[run_root()/name/f'{mode}-{b}'/f'{i:04d}.json' for b in ('cpp','cuda')]
            a,b=map(unseal,paths)
            if a['export']!=ref or b['export']!=ref or a['sample']!=b['sample'] or numerical(a)!=numerical(b):
                raise ValueError(f'CPU/CUDA eight-image gate mismatch {name} {mode} {i}')
            if i==0 and a['oracle_dot_checks']!=3*len(ex['bounds']):raise ValueError('missing actual-node rational dot checks')
            refs.extend(map(reference,paths))
    if any(unseal(run_root()/name/'B'/f'{i:04d}.json')['top5']!=ex['retained_B_prefix'][i]['top5'] for i in range(8)):
        raise ValueError('missing B replay')
    record={'status':'pass','export':ref,'images_per_arm':8,'arms':['wide','control'],
            'records':refs,'primitive_conformance':reference(conf),'independent_wide_graph':reference(proof),
            'scope':'new scaled contract only; old unscaled admission is unchanged'}
    immutable(run_root()/name/'gate.json',record)
    print(name+' eight-image CPU/CUDA gate PASS',flush=True)
    return record
