"""Read-only B evidence verification and frozen FX export; no B runner writes."""
from __future__ import annotations
import copy
import operator
import numpy as np
from .common import *
from .quantization import codebook, quantize

def anchors():
    from tools.experiment_b import common as b
    ledger_path=ROOT/'results/summaries/b-stage-paired-1k-v2/analysis.json'
    ledger=unseal(ledger_path)
    rows=b.dataset('imagenet_screen_1k')[1]
    items=[]
    for name in FORMATS:
        entry=next(x for x in ledger['configurations'] if
                   (x['model'],x['format'],x['recipe'])==('resnet18',name,'maxabs'))
        config=unseal(checked(entry['configuration']));summary=unseal(checked(entry['summary']))
        ident=b.digest(config)
        if ident!=entry['configuration_sha256'] or config['source_sha256']!=b.source_identity():
            raise ValueError('original B configuration/source drift')
        if config['protocol']!=b.PROTOCOL or config['model_context']!=b.frozen_inputs('resnet18'):
            raise ValueError('B frozen context/protocol drift')
        predictions={}
        for key,identity,expected in (
            ('B',ident,summary['prediction_digest']),
            ('FP32',config['baseline_sha256'],summary['baseline_prediction_digest'])):
            records=[unseal(b.BASE/'predictions'/identity/(r['sha256']+'.json')) for r in rows]
            if any(r['sample']!=s or r['configuration_sha256']!=identity or
                   r['batch_start']!=8*(i//8) or r['batch_images']!=8
                   for i,(r,s) in enumerate(zip(records,rows))):
                raise ValueError('retained B batch membership/sample identity mismatch')
            if b.digest(records)!=expected:raise ValueError('B prediction digest drift')
            predictions[key]=records
        top1=100*sum(r['top5'][0]==int(r['sample']['label']) for r in predictions['B'])/1000
        top5=100*sum(int(r['sample']['label']) in r['top5'] for r in predictions['B'])/1000
        baseline=100*sum(r['top5'][0]==int(r['sample']['label']) for r in predictions['FP32'])/1000
        if abs(top1-summary['metrics']['top1_percent'])>1e-9 or abs(top5-summary['metrics']['top5_percent'])>1e-9:
            raise ValueError('historical B metrics mismatch')
        items.append({'format':name,'configuration':entry['configuration'],'summary':entry['summary'],
                      'config':config,'retained':predictions,'historical_1000':{'B_top1':top1,'B_top5':top5,'FP32_top1':baseline},
                      'analysis_ledger':reference(ledger_path)})
    return rows,items

def calibration(config):
    from tools.experiment_b import common as b
    from tools.experiment_b.quantizer import scale_for
    folder=b.BASE/'calibration/resnet18'/config['calibration']['identity']
    summary=unseal(folder/'summary.json');provenance=unseal(folder/'provenance.json')
    if b.digest(summary)!=config['calibration']['summary_sha256'] or b.digest(provenance)!=config['calibration']['identity']:
        raise ValueError('calibration summary/provenance drift')
    rows=b.dataset('imagenet_calibration_2k')[1];maxima={};refs=[]
    for batch in summary['batch_checkpoints']:
        start=batch['start'];path=folder/f'{start:05d}.json';meta=unseal(path);data=path.with_suffix('.npz')
        if file_hash(path)!=batch['sha256'] or file_hash(data)!=meta['arrays_sha256'] or meta['samples']!=rows[start:start+8]:
            raise ValueError('calibration checkpoint drift')
        if meta['calibration_sha256']!=config['calibration']['identity']:raise ValueError('calibration identity changed')
        for k,v in meta['maxima'].items():maxima[k]=max(maxima.get(k,0),v)
        refs.append({'metadata':reference(path),'arrays':reference(data)})
    scales={k:scale_for(v,config['format']) for k,v in maxima.items()}
    if scales!=config['scales']['activation_scales']:raise ValueError('activation scales not reproduced')
    return maxima,{'summary':reference(folder/'summary.json'),'provenance':reference(folder/'provenance.json'),
                   'batch_evidence':refs}

def topology(graph):
    import torch
    nn=torch.nn
    result=[]
    for node in graph.graph.nodes:
        row={'name':node.name,'fx_op':node.op,'target':str(node.target),
             'args_repr':repr(node.args),'kwargs_repr':repr(node.kwargs),
             'inputs':[x.name for x in node.all_input_nodes],'attrs':{}}
        if node.op=='placeholder':row['op']='input'
        elif node.op=='output':row['op']='output'
        elif node.op=='call_function' and node.target==operator.add:row['op']='add'
        elif node.op=='call_function' and node.target==torch.flatten:row['op']='flatten'
        elif node.op=='call_module':
            mod=graph.get_submodule(node.target)
            if isinstance(mod,nn.Conv2d):
                row['op']='conv';row['attrs']={k:list(getattr(mod,k)) for k in ('stride','padding','dilation')}
                row['attrs']['groups']=mod.groups
            elif isinstance(mod,nn.Linear):row['op']='linear'
            elif isinstance(mod,nn.ReLU):row['op']='relu';row['attrs']={'inplace':mod.inplace}
            elif isinstance(mod,nn.MaxPool2d):
                row['op']='maxpool'
                row['attrs']={k:([getattr(mod,k)]*2 if isinstance(getattr(mod,k),int) else list(getattr(mod,k))) for k in ('kernel_size','stride','padding','dilation')}
                if mod.ceil_mode:raise ValueError('ceil mode unsupported')
            elif isinstance(mod,nn.AdaptiveAvgPool2d) and mod.output_size==(1,1):row['op']='avgpool'
            elif isinstance(mod,(nn.Identity,nn.Dropout)):row['op']='identity'
            else:raise ValueError(f'unsupported exported module {type(mod)}')
        else:raise ValueError(f'unsupported FX node {node}')
        result.append(row)
    return result

def export_all():
    import torch
    from tools.experiment_b.classifier import configure,load_model,prepare_qdq
    from tools.experiment_b.runner import runtime
    from .native import build
    configure('cuda');rows,items=anchors();graph,transform,original=load_model('resnet18','cuda');del original
    native={b:reference(build(b)) for b in ('cpp','cuda')}
    immutable(BASE/'protocol.json',{'contract':CONTRACT,'protocol':PROTOCOL,'samples':rows[:128]})
    exports=[]
    for item in items:
        config=item['config'];name=item['format']
        if runtime('cuda')!=config['runtime']:raise ValueError('B runtime differs from retained batch-8 anchor')
        maxima,cal=calibration(config)
        with torch.inference_mode():
            engine,scales=prepare_qdq(graph,name,'maxabs',{k:np.empty(0) for k in maxima},maxima,'cuda')
        if scales!=config['scales']:raise ValueError('B weight/activation scales drift')
        nodes=topology(graph);arrays={};bounds=[];levels,codes,units,mid,ties,shift=codebook(name)
        for node in nodes:
            if node['op'] not in ('conv','linear'):continue
            mod=graph.get_submodule(node['target']);key=node['name'];w=mod.weight.detach().cpu().numpy()
            ws=np.array(scales['weight_scales'][key],dtype=np.float32).reshape((-1,)+(1,)*(w.ndim-1))
            c,u,_=quantize(w,ws,name,b_weight=True)
            arrays[key+'_original_weight']=w;arrays[key+'_codes']=c;arrays[key+'_units']=u
            # Preserve the original B signed-zero reconstruction separately;
            # candidate codes deliberately canonicalize zero under the contract.
            arrays[key+'_B_reconstructed']=engine.module.get_submodule(node['target']).weight.detach().cpu().numpy()
            original_codes=c.copy()
            original_codes[(u==0)&np.signbit(w)]=1<<(6 if name=='fp7_e3m3' else 5)
            arrays[key+'_B_codes']=original_codes
            reconstructed=np.ldexp(u.astype(np.float32),-shift)*ws
            reconstructed=np.where((u==0)&np.signbit(w),np.float32(-0.),reconstructed)
            if not np.array_equal(reconstructed.view('u4'),arrays[key+'_B_reconstructed'].view('u4')):
                raise ValueError('exported B weight codes do not reproduce original FP32 reconstruction')
            arrays[key+'_bias']=mod.bias.detach().cpu().numpy() if mod.bias is not None else np.zeros(w.shape[0],dtype=np.float32)
            maxa=int(np.max(np.abs(units.astype(np.int64))))
            bound=maxa*np.abs(u.reshape(u.shape[0],-1).astype(np.int64)).sum(1)
            if np.max(bound)>=2**53:raise ValueError('scaled graph exceeds exact binary64 grid bound')
            bounds.append({'node':key,'K':int(np.prod(w.shape[1:])),'shape':list(w.shape),
                           'maximum_activation_units':maxa,'per_channel_prefix_units':bound.tolist(),
                           'max_prefix_units':int(np.max(bound)),'integer_signed_bits':int(np.max(bound)).bit_length()+1,
                           'fp32_all_prefix_exact_sufficient':bool(np.max(bound)<2**24),
                           'weight_units_sha256':array_hash(u)})
        logical={'case':'resnet18/'+name+'/maxabs','format':name,'contract':CONTRACT,'protocol':PROTOCOL,
                 'B_configuration':item['configuration'],'B_summary':item['summary'],
                 'B_source_sha256':config['source_sha256'],'B_runtime':config['runtime'],
                 'calibration':cal,'folded_FX_topology':str(graph.graph),'nodes':nodes,
                 'scales':scales,'format_manifest':reference(ROOT/f'public/formats/manifests/accepted/{name}.json'),
                 'input_transform':repr(transform),'model_context':config['model_context'],
                 'codebook':{'levels':levels.tolist(),'codes':codes.tolist(),'units':units.tolist(),
                             'shift':shift,'boundaries':mid.tolist(),'choose_upper_tie':ties.tolist()},
                 'bounds':bounds,'array_identities':{k:array_hash(v) for k,v in arrays.items()},
                 'sources':sources(),'native':native,'historical_1000':item['historical_1000'],
                 'ordered_samples_sha256':digest(rows),'retained_B_prefix':item['retained']['B'][:128],
                 'retained_FP32_prefix':item['retained']['FP32'][:128]}
        folder=BASE/'exports'/digest(logical);arr=save_npz(folder/'constants.npz',arrays)
        path=folder/'export.json';immutable(path,{**logical,'constants':arr})
        exports.append(reference(path))
    enrollment={'version':'scaled-bridge-enrollment-1','protocol':reference(BASE/'protocol.json'),
                'exports':exports,'sources':sources(),'native':native,'B_analysis':items[0]['analysis_ledger']}
    immutable(enrollment_path(),enrollment)
    return enrollment

def load_export(name):
    enrollment=unseal(enrollment_path())
    if enrollment['sources']!=sources():raise ValueError('bridge execution source drift after enrollment')
    for ref in enrollment['exports']:
        data=unseal(checked(ref))
        if data['format']==name:
            from tools.experiment_b.common import source_identity
            if source_identity()!=data['B_source_sha256']:raise ValueError('frozen B source drift')
            with np.load(checked(data['constants']),allow_pickle=False) as archive:
                arrays={k:archive[k] for k in archive.files}
            if {k:array_hash(v) for k,v in arrays.items()}!=data['array_identities']:raise ValueError('export array drift')
            return data,arrays,ref
    raise ValueError('unknown bridge case')
