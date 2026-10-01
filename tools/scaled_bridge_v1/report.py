from __future__ import annotations
from collections import Counter
import csv
import io
import statistics
import numpy as np
from .common import *
from .controller import ledger,records

def paired(left,right,labels):
    result={}
    for metric in ('top1','top5'):
        a=np.array([p[0]==y if metric=='top1' else y in p for p,y in zip(left,labels)],dtype=int)
        b=np.array([p[0]==y if metric=='top1' else y in p for p,y in zip(right,labels)],dtype=int)
        differences=a-b;counts=np.array([(differences==v).sum() for v in (-1,0,1)])
        draws=np.random.default_rng(20260928).multinomial(len(labels),counts/len(labels),size=10000)
        bootstrap=100*(draws[:,2]-draws[:,0])/len(labels)
        result[metric]={'left_percent':100*float(a.mean()),'right_percent':100*float(b.mean()),
                        'delta_pp':100*float(differences.mean()),'paired_pointwise95_pp':np.quantile(bootstrap,[.025,.975]).tolist(),
                        'left_only_correct':int(((a==1)&(b==0)).sum()),'right_only_correct':int(((a==0)&(b==1)).sum())}
    result['changed_top1_indices']=[i for i,(a,b) in enumerate(zip(left,right)) if a[0]!=b[0]]
    result['changed_top5_indices']=[i for i,(a,b) in enumerate(zip(left,right)) if a!=b]
    return result

def compare_layers(left,right):
    first=Counter();disagree=Counter();counts=Counter()
    for a,b in zip(left,right):
        found=False
        for node,x in a['layers'].items():
            y=b['layers'][node]
            if x['codes']!=y['codes']:
                disagree[node]+=1
                if not found:first[node]+=1;found=True
            for side,value in (('left',x),('right',y)):
                for key in ('clipped_low','clipped_high','ties','zeros','normalization_overflow'):
                    counts[side+'_'+key]+=value['diagnostics'].get(key,0)
    return {'first_code_divergence_images':dict(first),'per_node_code_disagreement_images':dict(disagree),
            'diagnostic_element_counts':dict(counts)}

def summary():
    budget=ledger();result={'contract':CONTRACT,'protocol':PROTOCOL,'source_digest':digest(sources()),
        'budget':{k:v for k,v in budget.items() if k!='attempts'},'attempts':len(budget['attempts']),
        'failed_or_interrupted':sum(not a['completion'] or a['completion']['status']!='complete' for a in budget['attempts']),
        'cases':[]}
    for name in FORMATS:
        counts={f'{m}-{b}':len(records(name,m,b)) for m in ('wide','control') for b in ('cpp','cuda')}
        row={'format':name,'counts':counts,'B_replayed':len(list((run_root()/name/'B').glob('*.json'))),
             'gate':(run_root()/name/'gate.json').exists()}
        if enrollment_path().exists():
            from .export import load_export
            ex,_,ref=load_export(name);row['historical_1000']=ex['historical_1000'];row['export']=ref
            row['maximum_integer_bits']=max(x['integer_signed_bits'] for x in ex['bounds'])
            wide=records(name,'wide','cuda');control=records(name,'control','cuda');n=min(len(wide),len(control))
            if n:
                w=wide[:n];c=control[:n];labels=[int(r['sample']['label']) for r in w]
                predictions={'wide':[r['top5'] for r in w],'control':[r['top5'] for r in c],
                    'B':[r['top5'] for r in ex['retained_B_prefix'][:n]],'FP32':[r['top5'] for r in ex['retained_FP32_prefix'][:n]]}
                row['paired_images']=n;row['comparisons']={a+'_minus_'+b:paired(predictions[a],predictions[b],labels)
                    for a,b in (('wide','B'),('control','B'),('wide','FP32'),('control','FP32'),('control','wide'))}
                row['arithmetic_layers']=compare_layers(c,w)
                bt=[unseal(run_root()/name/'B'/f'{i:04d}.json') for i in range(min(n,row['B_replayed']))]
                if bt:row['transfer_layers']=compare_layers(w[:len(bt)],bt)
            row['timings']={}
            for m in ('wide','control'):
                for b in ('cpp','cuda'):
                    rs=records(name,m,b)
                    if rs:
                        # Exclude rational witnesses from steady execution;
                        # their whole worker cost remains in the budget.
                        row['timings'][m+'-'+b]={
                            'median_execution_seconds':statistics.median(r['timing']['execution']-r['timing']['oracle'] for r in rs),
                            'median_MAC_seconds':statistics.median(r['timing']['mac'] for r in rs),
                            'median_other_seconds':statistics.median(r['timing']['nonmac']+r['timing']['quantization_trace'] for r in rs),
                            'median_preprocess_seconds':statistics.median(r['preprocess_seconds'] for r in rs),
                            'first_setup_seconds':rs[0]['setup_seconds']}
        result['cases'].append(row)
    return result

def write_report():
    data=summary();folder=ROOT/'results/summaries/scaled-bridge-v1'/data['source_digest']
    # Every report revision gets its own content address; no earlier sealed
    # scientific report is replaced when a longer prefix completes.
    folder=folder/digest(data);immutable(folder/'summary.json',data)
    text=io.StringIO();writer=csv.writer(text)
    writer.writerow(['format','paired_images','B_top1','wide_top1','control_top1','FP32_top1','wide_minus_B_pp','control_minus_wide_pp'])
    for row in data['cases']:
        if 'comparisons' not in row:continue
        c=row['comparisons'];writer.writerow([row['format'],row['paired_images'],c['wide_minus_B']['top1']['right_percent'],
            c['wide_minus_B']['top1']['left_percent'],c['control_minus_B']['top1']['left_percent'],c['wide_minus_FP32']['top1']['right_percent'],
            c['wide_minus_B']['top1']['delta_pp'],c['control_minus_wide']['top1']['delta_pp']])
    (folder/'summary.csv').write_text(text.getvalue())
    lines=['# Scaled FP6/FP7 bridge pilot — 2026-09-28','',
        'This is a development-informed paired pilot, not independent confirmation. Historical B 1,000-image values and current panels are separate.',
        '',f'Worker budget: {data["budget"]["charged_seconds"]:.1f} / 14,400 seconds charged; {data["budget"]["remaining_seconds"]:.1f} seconds remain. Failed/interrupted attempts retain their reservation.',
        '', '| Format | Paired images | B top-1 | Wide top-1 | Control top-1 | FP32 top-1 | Wide − B (pp) | Control − wide (pp) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in data['cases']:
        if 'comparisons' not in row:
            lines.append(f'| {row["format"]} | 0 | — | — | — | — | — | — |');continue
        c=row['comparisons'];values=[c['wide_minus_B']['top1']['right_percent'],c['wide_minus_B']['top1']['left_percent'],
            c['control_minus_B']['top1']['left_percent'],c['wide_minus_FP32']['top1']['right_percent'],c['wide_minus_B']['top1']['delta_pp'],c['control_minus_wide']['top1']['delta_pp']]
        lines.append(f'| {row["format"]} | {row["paired_images"]} | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines+=['','## Four study questions','']
    for row in data['cases']:
        lines.append(f'### {row["format"]}')
        lines.append(f'B batch-eight replay: {row["B_replayed"]} images verified against retained candidate and FP32 Top-5; eight-image CPU/CUDA gate: {row["gate"]}.')
        if 'comparisons' in row:
            c=row['comparisons'];t=c['wide_minus_B'];a=c['control_minus_wide']
            lines.append(f'Transfer (wide minus B): top-1 {t["top1"]["delta_pp"]:+.3f} pp, pointwise paired95% {t["top1"]["paired_pointwise95_pp"]}; {len(t["changed_top1_indices"])} changed Top-1 and {len(t["changed_top5_indices"])} changed ordered Top-5 lists.')
            lines.append(f'Arithmetic only (control minus wide): top-1 {a["top1"]["delta_pp"]:+.3f} pp, pointwise paired95% {a["top1"]["paired_pointwise95_pp"]}; {len(a["changed_top1_indices"])} changed Top-1 and {len(a["changed_top5_indices"])} changed ordered Top-5 lists.')
            lines.append('First code divergences versus B: '+json.dumps(row.get('transfer_layers',{}).get('first_code_divergence_images',{}))+'.')
            lines.append('First code divergences between accumulator arms: '+json.dumps(row['arithmetic_layers']['first_code_divergence_images'])+'.')
            for label,timing in row['timings'].items():lines.append(f'{label}: median execution {timing["median_execution_seconds"]:.3f}s/image, MAC {timing["median_MAC_seconds"]:.3f}s, other {timing["median_other_seconds"]:.3f}s; preprocessing {timing["median_preprocess_seconds"]:.3f}s; first setup {timing["first_setup_seconds"]:.3f}s.')
            useful=row['gate'] and t['top1']['left_percent']>=40
            lines.append(f'Next accumulator sweep ready under this new contract: {useful}. This does not grant old unscaled native acceptance.')
        lines.append('')
    lines+=['## Next sweep specification (not launched)','',
       'For cases passing the gate and quality rule, retain every scale, bias, code store, reduction order and paired image identity. Replace only the code-level dot accumulator:',
       '',
       '- IEEE binary16: exact product plus prior state, one round to nearest even after each MAC, gradual subnormals, IEEE infinity/NaN on overflow; nonfinite graph state is a reported failure, never clamped silently.',
       '- Custom float: 8 exponent bits, bias127, 12 fraction bits, sign bit (21 total), RNE after each exact MAC, gradual subnormals and IEEE specials. This keeps FP32 range while testing a precision between FP16 and FP32; exact width must be frozen before a sweep.',
       '- Code-domain integer/fixed: zero point0, domain scale2^(-2*codebook_shift), exact integer products and integer adds. Use each export’s proved signed width for the lossless arm; test explicitly narrower widths with signed endpoint saturation after each addition (not wrap), then exact power-of-two conversion. Scales remain the bridge’s external FP32 bits and are applied with the same FP64 postops.',
       '',
       'Independent rational accumulator witnesses and paired native gates are required for each future arm. No broad accumulator sweep has been launched.',
       '',
       'Resume: `.venv-b/bin/python -m tools.run.scaled_bridge run`',
       '',f'Sealed machine results: `{(folder/"summary.json").relative_to(ROOT)}`.',
       'Each output record binds its sample, export, code/state hashes, diagnostics and timing. Source revisions use separate namespaces; the worker ledger is shared across revisions. Original B, old unscaled certificates and the old24-hour ledger remain unchanged.']
    report=ROOT/'docs/analysis/scaled-bridge-v1-results-2026-09-28.md';report.write_text('\n'.join(lines)+'\n')
    print(report);print(json.dumps({k:data[k] for k in ('budget','attempts','failed_or_interrupted')}))
    return data
