"""Original B batch-eight runtime replay with separate trace output namespace."""
from __future__ import annotations
import time
import numpy as np
import torch
from tools.experiment_b.classifier import QDQInterpreter,quantized_node
from .common import *
from .quantization import quantize
from .export import load_export,calibration

class TracedB(QDQInterpreter):
    def __init__(self,engine,name,nodes):
        super().__init__(engine.module,engine.quantizer,{k:float(v.cpu()) for k,v in engine.scales.items()})
        self.name=name;self.traces=[];self.codes={};self.state_scales={};self.nodes={x['name']:x for x in nodes}
    def run_node(self,node):
        raw=torch.fx.Interpreter.run_node(self,node)
        if node.op=='output':return raw
        key=node.name;quantized=quantized_node(self.module,node)
        if not self.traces:self.traces=[{} for _ in range(raw.shape[0])]
        if quantized:
            result=self.quantizer(raw,self.scales[key])
            rawcpu=raw.detach().cpu().numpy();scale=float(self.scales[key].cpu())
            codes,units,_=quantize(rawcpu,scale,self.name,b_weight=True)
            self.codes[key]=codes;self.state_scales[key]=scale
        else:
            result=raw;parent=self.nodes[key]['inputs'][0]
            codes=self.codes[parent].reshape(tuple(raw.shape));self.codes[key]=codes
            self.state_scales[key]=self.state_scales[parent]
        values=result.detach().cpu().numpy()
        for i,trace in enumerate(self.traces):
            diag=quantize(rawcpu[i:i+1],scale,self.name,b_weight=True)[2] if quantized else {'skipped':True}
            trace[key]={'codes':array_hash(codes[i:i+1]),'state':array_hash(values[i:i+1].astype(np.float64)),
                        'diagnostics':diag,'negative_zero_values':int(np.sum((values[i]==0)&np.signbit(values[i])))}
        return result

def replay(name,target):
    from tools.experiment_b.classifier import configure,load_model,prepare_qdq,image_batch
    from tools.experiment_b.runner import runtime
    from tools.experiment_b.common import dataset
    ex,_,ref=load_export(name);configure('cuda')
    if runtime('cuda')!=ex['B_runtime']:raise ValueError('cannot reproduce B in changed runtime')
    graph,transform,original=load_model('resnet18','cuda');del original
    config=unseal(checked(ex['B_configuration']));maxima,_=calibration(config)
    with torch.inference_mode():engine,scales=prepare_qdq(graph,name,'maxabs',{k:np.empty(0) for k in maxima},maxima,'cuda')
    if scales!=ex['scales']:raise ValueError('replay scales changed')
    _,rows,payload=dataset('imagenet_screen_1k');folder=run_root()/name/'B'
    for start in range(0,target,8):
        paths=[folder/f'{i:04d}.json' for i in range(start,start+8)]
        if all(p.exists() for p in paths):
            for i,p in enumerate(paths,start):
                r=unseal(p)
                if r['export']!=ref or r['sample']!=rows[i] or r['top5']!=ex['retained_B_prefix'][i]['top5']:
                    raise ValueError('B trace checkpoint drift')
            continue
        tick=time.perf_counter();inputs=image_batch(rows[start:start+8],payload,transform,'cuda')
        prep=time.perf_counter()-tick
        tracer=TracedB(engine,name,ex['nodes']);tick=time.perf_counter()
        with torch.inference_mode():
            # Uninstrumented original B interpreter and traced replay must agree.
            expected=engine.run(inputs);observed=tracer.run(inputs);baseline=graph(inputs)
        if not torch.equal(expected,observed):raise ValueError('B trace instrumentation changes output')
        top=observed.topk(5,dim=1).indices.cpu().tolist();fp=baseline.topk(5,dim=1).indices.cpu().tolist()
        elapsed=time.perf_counter()-tick
        for i,path in enumerate(paths,start):
            j=i-start
            if top[j]!=ex['retained_B_prefix'][i]['top5'] or fp[j]!=ex['retained_FP32_prefix'][i]['top5']:
                raise ValueError('original batch-8 retained Top-5 reproduction failed')
            record={'export':ref,'sample':rows[i],'index':i,'batch_start':start,'batch_images':8,
                    'top5':top[j],'FP32_top5':fp[j],'layers':tracer.traces[j],
                    'retained_B_prediction':reference(ROOT/'artifacts/experiment_b/predictions'/digest(config)/(rows[i]['sha256']+'.json')),
                    'retained_FP32_prediction':reference(ROOT/'artifacts/experiment_b/predictions'/config['baseline_sha256']/(rows[i]['sha256']+'.json')),
                    'preparation_batch_seconds':prep,'execution_batch_seconds':elapsed,
                    'scope':'original B batch8 Top5 reproduced; separate layer trace'}
            if path.exists():
                old=unseal(path)
                if any(old[k]!=record[k] for k in ('export','sample','index','top5','FP32_top5','layers')):
                    raise ValueError('replayed B trace drift')
            else:immutable(path,record)
        print(f'{name} B reproduced {start+8}/{target}',flush=True)
