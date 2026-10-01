"""Read-only scientific audit; preserves the enrolled executor source identity.

Sealed JSON sorts object keys, so first-divergence reports must explicitly use
the exported FX order rather than the dictionary's serialized key order.
"""
from collections import Counter
import csv
import io
import json
from pathlib import Path
import shutil
from tools.scaled_bridge_v1.common import *
from tools.scaled_bridge_v1.controller import records,ledger
from tools.scaled_bridge_v1.export import load_export,anchors
from tools.scaled_bridge_v1.engine import numerical
from tools.scaled_bridge_v1.report import summary

def first_divergences(left,right,order):
    counts=Counter()
    for a,b in zip(left,right):
        node=next((key for key in order if key in a['layers'] and a['layers'][key]['codes']!=b['layers'][key]['codes']),None)
        if node:counts[node]+=1
    return dict(counts)

def audit():
    data=summary();_,historical=anchors()
    for row in data['cases']:
        name=row['format'];ex,_,ref=load_export(name)
        if row['historical_1000']!=next(r['historical_1000'] for r in historical if r['format']==name):
            raise ValueError('historical metrics mismatch')
        if not row['gate']:raise ValueError('incomplete admission')
        gate=unseal(run_root()/name/'gate.json')
        for item in gate['records']:unseal(checked(item))
        witness=unseal(checked(gate['independent_wide_graph']))
        if witness['numerical']!=numerical(records(name,'wide','cpp')[0]):raise ValueError('independent wide witness mismatch')
        for mode in ('wide','control'):
            cpu=records(name,mode,'cpp');gpu=records(name,mode,'cuda')
            if len(cpu)!=8 or len(gpu) not in (32,128):raise ValueError('unexpected completed panel')
            if any(numerical(a)!=numerical(b) for a,b in zip(cpu,gpu)):raise ValueError('native gate no longer verifies')
        wide=records(name,'wide','cuda');control=records(name,'control','cuda');n=len(wide)
        if len(control)!=n:raise ValueError('unpaired final panel')
        b=[unseal(run_root()/name/'B'/f'{i:04d}.json') for i in range(n)]
        order=[node['name'] for node in ex['nodes']]
        for i,(w,c,original) in enumerate(zip(wide,control,b)):
            for value in (w,c,original):
                if value['export']!=ref or value['sample']!=ex['retained_B_prefix'][i]['sample']:raise ValueError('sample/export drift')
            if original['top5']!=ex['retained_B_prefix'][i]['top5'] or original['FP32_top5']!=ex['retained_FP32_prefix'][i]['top5']:
                raise ValueError('B batch replay drift')
            if original['batch_start']!=i//8*8 or original['batch_images']!=8:raise ValueError('B batch membership drift')
            for item in (original['retained_B_prediction'],original['retained_FP32_prediction']):unseal(checked(item))
        row['transfer_layers']['first_code_divergence_images']=first_divergences(wide,b,order)
        row['arithmetic_layers']['first_code_divergence_images']=first_divergences(control,wide,order)
        row['exact_control_full_numerical_equal_images']=sum(numerical(a)==numerical(b) for a,b in zip(wide,control))
        row['arithmetic_raw_MAC_disagreement_images']=sum(any(a['layers'][k]['mac']!=b['layers'][k]['mac'] for k in a['layers']) for a,b in zip(wide,control))
        row['gate_reference']=reference(run_root()/name/'gate.json')
        row['all_layers_FP32_prefix_exact_sufficient']=all(b['fp32_all_prefix_exact_sufficient'] for b in ex['bounds'])
        row['sealed_panel_records']={arm:[reference(run_root()/name/arm/f'{i:04d}.json') for i in range(n)]
                                    for arm in ('B','wide-cuda','control-cuda')}
        decision=unseal(run_root()/name/'promotion.json')
        if decision['panel']!=32 or not decision['quality_pass']:raise ValueError('missing frozen promotion decision')
        row['promotion_reference']=reference(run_root()/name/'promotion.json')
        row['gate_primitive_reference']=gate['primitive_conformance']
        conformance=unseal(checked(gate['primitive_conformance']))
        if conformance['status']!='pass' or conformance['sources']!=sources():raise ValueError('primitive witness drift')
        for native in ex['native'].values():checked(native)
    data['audit']={'source':reference(Path(__file__)),'result':'pass','first_divergence_order':'exported FX topology',
                   'audit_date':'2026-09-29','pilot_date':'2026-09-28',
                   'checks':'sealed records, retained B identities/runtime/batch8 Top5, sample pairing, eight-image gates, independent wide graph witness, original1000-image anchors'}
    data['budget_attempt_records']=[{'reservation':reference(p),'completion':reference(p.with_suffix('.complete'))}
        for p in sorted((BASE/'budget/attempts').glob('*.json'))]
    folder=ROOT/'results/summaries/scaled-bridge-v1-audited'/digest(data)
    immutable(folder/'summary.json',data)
    table=io.StringIO();writer=csv.writer(table)
    writer.writerow(['format','images_per_arm','comparison','metric','left_percent','right_percent','delta_pp','paired_pointwise95_low_pp','paired_pointwise95_high_pp'])
    for row in data['cases']:
        for name,comparison in row['comparisons'].items():
            for metric in ('top1','top5'):
                values=comparison[metric]
                writer.writerow([row['format'],row['paired_images'],name,metric,values['left_percent'],values['right_percent'],values['delta_pp'],*values['paired_pointwise95_pp']])
    (folder/'comparisons.csv').write_text(table.getvalue())
    # Preserve the complete enrolled implementation and libraries, without
    # changing the source files to which completed evidence is bound.
    archive=BASE/'implementations'/data['source_digest']
    for rel,expected in sources().items():
        src=ROOT/rel;dst=archive/'files'/rel
        if file_hash(src)!=expected:raise ValueError('source drift while archiving')
        if dst.exists():
            if file_hash(dst)!=expected:raise ValueError('archive drift')
        else:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    immutable(archive/'manifest.json',{'sources':sources(),'enrollment':reference(enrollment_path())})
    enrollment=unseal(enrollment_path())
    archived=[]
    for backend,ref in enrollment['native'].items():
        original=checked(ref);dest=archive/'native'/backend/'bridge.so'
        if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,dest)
        if file_hash(dest)!=ref['sha256']:raise ValueError('archived binary drift')
        build=original.with_suffix('.json');unseal(build)
        saved=dest.with_suffix('.json')
        if not saved.exists():shutil.copyfile(build,saved)
        if file_hash(build)!=file_hash(saved):raise ValueError('archived compiler manifest drift')
        archived.append({'binary':reference(dest),'compiler_manifest':reference(saved)})
    immutable(archive/'native-manifest.json',{'native':archived})
    lines=['# Scaled FP6/FP7 arithmetic bridge: audited paired pilot','',
       'Pilot completed 2026-09-28; audited 2026-09-29. ResNet18/maxabs; selected using earlier development evidence. These panels are not independent final confirmation.',
       '',f'All three cases passed the independent wide-graph witness, actual-node rational dot checks and eight-image CPU/CUDA gates in both arms. Primitive conformance passed 98,768 checks. Ledger: {data["budget"]["charged_seconds"]:.1f} /14,400 worker-seconds charged, including setup, probes, both backends and both arms; {data["failed_or_interrupted"]} failed/interrupted attempts. Coding and compilation excluded.',
       '', '| Format | Images/arm | B top-1 | Wide top-1 | FP32-control top-1 | FP32 baseline | Wide − B | Control − wide |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in data['cases']:
        c=row['comparisons'];t=c['wide_minus_B']['top1'];a=c['control_minus_wide']['top1']
        values=[t['right_percent'],t['left_percent'],a['left_percent'],c['wide_minus_FP32']['top1']['right_percent']]
        lines.append(f'| {row["format"]} | {row["paired_images"]} | '+' | '.join(f'{v:.3f}%' for v in values)+f' | {t["delta_pp"]:+.3f} pp | {a["delta_pp"]:+.3f} pp |')
    lines+=['','## 1. Was the B anchor reproduced?','',
        'Yes, each selected case reproduces all saved candidate and paired FP32 ordered Top-5 lists for its panel, using the original runtime and original batch membership of eight. New layer traces are separate artifacts. The historical 1,000-image B top-1 anchors remain 66.2%,65.3%,69.7%, versus70.1% FP32; those are not the current panel scores.',
        '', '## 2. What changed when transferring B to the scaled code-domain contract?','',
        'B performs FP32 normalization, operand reconstruction and framework reduction. The new contract retains weight codes and scale bits, factors scales after the exact code-domain dot, and uses explicitly rounded FP64 postops and stores. It canonicalizes stored zero. Candidate-minus-B therefore measures the combined transfer; it is not an accumulator-only effect.']
    for row in data['cases']:
        t=row['comparisons']['wide_minus_B'];layer=row['transfer_layers']
        lines+=['',f'**{row["format"]}:** top-1 delta {t["top1"]["delta_pp"]:+.3f} pp (paired pointwise95% {t["top1"]["paired_pointwise95_pp"]}), top-5 accuracy delta {t["top5"]["delta_pp"]:+.3f} pp. Changed Top-1 predictions: {len(t["changed_top1_indices"])}; changed ordered Top-5 lists: {len(t["changed_top5_indices"])}.',
                'First differing layer codes in FX execution order: '+json.dumps(layer['first_code_divergence_images'])+'.',
                'Clipping counts (wide/B): '+str(layer['diagnostic_element_counts']['left_clipped_low']+layer['diagnostic_element_counts']['left_clipped_high'])+'/'+str(layer['diagnostic_element_counts']['right_clipped_low']+layer['diagnostic_element_counts']['right_clipped_high'])+'.']
    lines+=['','## 3. What changed from accumulator arithmetic alone?','',
        'Both candidate arms use the same code graph, external scales, bias, postops and store rules. The matched control changes only the dot to sequential FP32 fused multiply-add. Reduction order and each rounding boundary are frozen.']
    for row in data['cases']:
        t=row['comparisons']['control_minus_wide']
        lines+=['',f'**{row["format"]}:** control-minus-wide top-1 {t["top1"]["delta_pp"]:+.3f} pp (paired pointwise95% {t["top1"]["paired_pointwise95_pp"]}); changed Top-1/ordered Top-5: {len(t["changed_top1_indices"])}/{len(t["changed_top5_indices"])}. Full numerical records agree on {row["exact_control_full_numerical_equal_images"]}/{row["paired_images"]} images; {row["arithmetic_raw_MAC_disagreement_images"]} images have a differing raw dot result.',
                'First differing stored codes: '+json.dumps(row['arithmetic_layers']['first_code_divergence_images'])+'.']
    lines+=['','A zero-width empirical bootstrap interval when no outcomes differ is descriptive of this panel; it is not proof of population equivalence.',
        'The E2M3 certificate is stronger: all21 MAC layers have absolute grid-prefix bounds below2^24 (largest3,527,832), so FP32 is exact for the entire admitted code domain. It is a negative control for FP32 accumulator loss under this scaled contract. The corresponding maximum signed integer widths are23bits(E2M3),29bits(E3M2),31bits(FP7); per-channel proofs are in each export.',
        '', '## 4. Which cases are ready for the next accumulator sweep?','']
    ready=[]
    for row in data['cases']:
        useful=row['comparisons']['wide_minus_B']['top1']['left_percent']>=40
        if useful:ready.append(row['format'])
    lines+=['New scaled-contract gates and the frozen quality rule support: '+', '.join(ready)+'. No old unscaled native admission is extended; in particular, the old unscaled FP7 acceptance is still absent.',
        '', 'Next sweep specification only; no new sweep launched:',
        '', '- FP16: code-level exact product plus prior accumulator, one RNE after every fused MAC, gradual subnormals and IEEE infinity/NaN. Record nonfinite failures; never silently clamp them.',
        '- Custom21 float: sign1/exponent8/bias127/fraction12, RNE after each exact MAC, gradual subnormals and IEEE specials. It retains FP32 range while testing an intermediate precision. Freeze the manifest before running.',
        '- Integer/fixed grid: signed zero-point0 accumulator, exact product grid scale2^(-2×codebook shift). Use each exported per-channel absolute-sum width as the lossless reference, then explicitly narrower widths with endpoint saturation after every add. No wrap. Apply the same retained external scales and FP64 postops after exact power-of-two conversion.',
        '', 'Every future arm needs independent rational witnesses and paired native gates before quality evaluation.',
        '', '## Runtime and continuation','',
        '| Format | Wide CUDA median | Control CUDA median | Wide CPU median | Control CPU median |',
        '|---|---:|---:|---:|---:|']
    for row in data['cases']:
        lines.append('| '+row['format']+' | '+' | '.join(f'{row["timings"][k]["median_execution_seconds"]:.3f}s' for k in ('wide-cuda','control-cuda','wide-cpp','control-cpp'))+' |')
    lines+=['','These are measured per-image execution times including node quantization and trace hashes, excluding preprocessing/setup and independent rational witness overhead. Detailed MAC/non-MAC/preprocess/setup timings and all paired confidence intervals are in the machine result.',
      '', 'One-line resume and audited report:', '', '```bash',
      'bash tools/run/scaled_bridge.sh', '```',
      '', f'Sealed audited results: `{(folder/"summary.json").relative_to(ROOT)}`.',
      'The executor-generated preliminary report uses JSON key order for first-divergence summaries; this audit supersedes that field with the exported FX execution order. All numerical inference records and their identities are unchanged.',
      'The original B source identity is verified again by the audit; original B evidence and the historical E1 budget were not modified.',
      f'Complete paired Top-1/Top-5 comparisons and pointwise confidence intervals: `{(folder/"comparisons.csv").relative_to(ROOT)}`.',
      f'Archived enrolled source, libraries and compiler manifests: `{archive.relative_to(ROOT)}`.']
    doc=ROOT/'docs/analysis/scaled-bridge-v1-results-2026-09-28.md';doc.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(doc),'summary':reference(folder/'summary.json'),'budget':data['budget'],
                      'cases':[{k:r[k] for k in ('format','paired_images','gate','exact_control_full_numerical_equal_images')} for r in data['cases']]},indent=2))
    return data

if __name__=='__main__':audit()
