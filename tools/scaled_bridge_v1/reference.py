"""Independent FX traversal and exact-grid binary64 convolution graph witness."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from .common import array_hash,digest
from .quantization import codebook

class FXWideReference(torch.fx.Interpreter):
    def __init__(self,graph,export,arrays):
        super().__init__(graph)
        self.ex,self.arrays=export,arrays
        self.nodes={x['name']:x for x in export['nodes']}
        self.scales={};self.codes={};self.trace={}
        lev,code,units,mid,tie,self.shift=codebook(export['format'])
        self.levels=torch.from_numpy(lev);self.codebook=torch.from_numpy(code)
        self.unitbook=torch.from_numpy(units).double();self.mid=torch.from_numpy(mid);self.tie=torch.from_numpy(tie)
    def real(self,u,scale):return (u*float(2.**-self.shift))*float(scale)
    def run_node(self,node):
        spec=self.nodes[node.name];key=node.name;op=spec['op'];args,kwargs=self.fetch_args_kwargs_from_env(node)
        inputs=spec['inputs'];mac=None
        if op=='input':raw=next(self.args_iter).double()
        elif op=='output':return self.real(args[0],self.scales[inputs[0]])
        elif op in ('identity','flatten'):
            result=args[0].flatten(1) if op=='flatten' else args[0]
            self.scales[key]=self.scales[inputs[0]];self.codes[key]=self.codes[inputs[0]].reshape(result.shape)
            self.trace[key]={'codes':array_hash(self.codes[key]),'state':array_hash(self.real(result,self.scales[key]).numpy()),
                             'raw':None,'mac':None,'diagnostics':{'skipped':True}}
            return result
        elif op in ('conv','linear'):
            weight=torch.from_numpy(self.arrays[key+'_units'].astype(np.float64))
            # All integer products and all prefixes/subsets are <2^53 units by
            # the export certificate, making this independent reduction exact.
            if op=='conv':dot=F.conv2d(args[0],weight,None,**spec['attrs'])
            else:dot=F.linear(args[0],weight,None)
            mac=dot*float(2.**(-2*self.shift))
            raw=mac*float(self.scales[inputs[0]])
            shape=(1,-1,1,1) if op=='conv' else (1,-1)
            raw=raw*torch.tensor(self.ex['scales']['weight_scales'][key],dtype=torch.float64).reshape(shape)
            raw=raw+torch.from_numpy(self.arrays[key+'_bias'].astype(np.float64)).reshape(shape)
        elif op=='add':raw=self.real(args[0],self.scales[inputs[0]])+self.real(args[1],self.scales[inputs[1]])
        elif op=='relu':raw=self.real(torch.relu(args[0]),self.scales[inputs[0]])
        elif op=='maxpool':raw=self.real(F.max_pool2d(args[0],**spec['attrs']),self.scales[inputs[0]])
        elif op=='avgpool':
            raw=args[0].sum(dim=(-2,-1),keepdim=True)*float(2.**-self.shift)
            raw=raw/float(args[0].shape[-2]*args[0].shape[-1]);raw=raw*float(self.scales[inputs[0]])
        else:raise ValueError('reference encountered unsupported FX node')
        scale=self.ex['scales']['activation_scales'][key];y=raw/float(scale)
        pos=torch.bucketize(y.contiguous(),self.mid,right=False);adj=pos.clamp_max(len(self.mid)-1)
        tied=y==self.mid[adj];pos=pos+(tied&self.tie[adj]).long()
        result=self.unitbook[pos];codes=self.codebook[pos].numpy()
        diag={'elements':raw.numel(),'clipped_low':int((y<self.levels[0]).sum()),
              'clipped_high':int((y>self.levels[-1]).sum()),'ties':int(tied.sum()),
              'zeros':int((result==0).sum()),'normalization_overflow':int((~torch.isfinite(y)).sum())}
        self.scales[key]=scale;self.codes[key]=codes
        if op=='linear':mac=mac[:,:,None,None]
        self.trace[key]={'codes':array_hash(codes),'state':array_hash(self.real(result,scale).numpy()),
                         'raw':array_hash(raw.numpy()),'mac':array_hash(mac.numpy()) if mac is not None else None,
                         'diagnostics':diag}
        return result
    def evaluate(self,inputs):
        with torch.inference_mode():output=self.run(torch.from_numpy(inputs).double())
        return {'layers':self.trace,'output':array_hash(output.numpy()),'top5':output.topk(5,dim=1).indices.tolist()[0],
                'diagnostic_signature':digest({k:v['diagnostics'] for k,v in self.trace.items()})}
