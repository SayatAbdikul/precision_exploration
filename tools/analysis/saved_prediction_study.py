"""Recompute saved-prediction analyses and freeze a prospective comparison design.

No model inference. Existing 128-image observations are development data, not
independent confirmation. Detector AP remains a dataset-level metric.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from pathlib import Path

import numpy as np

from public.analysis.phase3.statistics import paired_classification, coco_metrics, resampled_coco
from tools.analysis.experiment_b_coverage import audit
from tools.experiment_b.common import ROOT, dataset, digest, formats, seal, unseal
from tools.phase3.common import read, reference, checked
from tools.phase3.worker_locks import exclusive

OUT = ROOT / 'results/summaries/saved-prediction-study-v1'
MATRIX = ROOT / 'public/experiments/configs/breadth-study/comparison-matrix-v1.json'
SEED = 20260925


def immutable(path, value):
    if path.exists():
        if unseal(path) != value:
            raise ValueError(f'frozen artifact differs; use a new version: {path}')
    else:
        seal(path, value)


def classification(left, right):
    if not left or len(left) != len(right) or any(a['sample'] != b['sample'] for a,b in zip(left,right)):
        raise ValueError('classification predictions are not paired')
    rows = [{'sample_id': a['sample']['sha256'], 'ground_truth': int(a['sample']['label']),
             'fp32_prediction': a['top5'], 'candidate_prediction': b['top5']} for a,b in zip(left,right)]
    return paired_classification(rows, [r['sample_id'] for r in rows], resamples=5000, seed=SEED)


def analyze():
    coverage = audit()
    manifest = {f['name']: f for f in formats()}
    populations = {m: dataset('coco_screen_1k' if m=='yolov8n' else 'imagenet_screen_1k')[1][:128]
                   for m in ('resnet18','mobilenet_v2','mobilenet_v3_large','yolov8n')}
    baselines, predictions, ledger = {}, {}, []
    detector_truth = None
    baseline_ap = {}
    for component in ('experiment_b','experiment_b_ext'):
        base = ROOT / 'artifacts' / component
        state = read(base / 'status.json')
        for task in state['tasks']:
            if task['status'] != 'completed':
                continue
            model, name, recipe = (task[k] for k in ('model','format','recipe'))
            identity = task['configuration_sha256']
            conf_path = base / 'configurations' / f'{identity}.json'
            summary_path = base / 'summaries' / f'{identity}-128.json'
            config, summary = unseal(conf_path), unseal(summary_path)
            baseline_id = config['baseline_sha256']
            if (component,baseline_id) not in baselines:
                baselines[(component,baseline_id)] = [unseal(base/'predictions'/baseline_id/(r['sha256']+'.json'))
                                                     for r in populations[model]]
            baseline = baselines[(component,baseline_id)]
            candidate = [unseal(base/'predictions'/identity/(r['sha256']+'.json')) for r in populations[model]]
            if digest(baseline) != summary['baseline_prediction_digest'] or digest(candidate) != summary['prediction_digest']:
                raise ValueError('prediction digest mismatch')
            for records, expected_id in ((candidate,identity),(baseline,baseline_id)):
                if any(r['sample'] != sample or r['configuration_sha256'] != expected_id for r,sample in zip(records,populations[model])):
                    raise ValueError('prediction/baseline context mismatch')
            row = {'model': model, 'format': name, 'recipe': recipe, 'family': manifest[name]['family'],
                   'bits': manifest[name]['bits'], 'images':128, 'component':component,
                   'configuration':reference(conf_path), 'summary':reference(summary_path),
                   'configuration_sha256':identity, 'baseline_sha256':baseline_id,
                   'source_sha256':config['source_sha256'], 'protocol':config['protocol'],
                   'evidence_level':'development_FP32_QDQ_not_exact_acceptance',
                   'prediction_digest':summary['prediction_digest'], 'baseline_prediction_digest':summary['baseline_prediction_digest']}
            if model == 'yolov8n':
                if detector_truth is None:
                    path=ROOT/'data/raw/coco2017/annotations/instances_val2017.json'
                    declared=read(ROOT/'public/workloads/datasets/coco2017.json')['annotation_sha256']['instances_val2017']
                    checked({'path':str(path.relative_to(ROOT)), 'sha256':declared})
                    # Use the saved evaluator's SHA-ordered cloned image IDs.
                    # COCO sorts IDs; changing their order changes AP tie-breaking
                    # when quantization produces equal detection scores.
                    annotations=read(path)
                    ids=[int(r['image_id']) for r in populations[model]]
                    detector_truth,_=resampled_coco(annotations,[],ids)
                    remap={old:new for new,old in enumerate(ids,1)}
                def detections(records):
                    return [{**d,'image_id':remap[d['image_id']]} for r in records for d in r['detections']]
                if baseline_id not in baseline_ap:
                    baseline_ap[baseline_id]=coco_metrics(detector_truth,detections(baseline))
                a=baseline_ap[baseline_id]
                b=coco_metrics(detector_truth,detections(candidate))
                if not np.allclose(b,[summary['metrics']['map50_95'],summary['metrics']['map50']],rtol=0,atol=1e-12):
                    raise ValueError('recomputed COCO AP differs from saved summary')
                row.update(metric='map50_95', baseline_percent=float(100*a[0]), candidate_percent=float(100*b[0]),
                           delta_pp=float(100*(b[0]-a[0])), delta_interval_pp=None,
                           uncertainty='COCO image-bootstrap not run here; descriptive point estimates only',
                           empty_prediction_images=sum(not r['detections'] for r in candidate))
            else:
                stats=classification(baseline,candidate)
                top=stats['metrics']['top1']
                if abs(100*top['candidate']-summary['metrics']['top1_percent'])>1e-12:
                    raise ValueError('recomputed classification differs from saved summary')
                row.update(metric='top1',baseline_percent=100*top['fp32'], candidate_percent=100*top['candidate'],
                           delta_pp=100*top['delta'],delta_interval_pp=[100*x for x in top['delta_interval']],
                           statistics=stats, uncertainty='pointwise paired bootstrap; exploratory, no multiplicity correction',
                           unique_top1_predictions=len({r['top5'][0] for r in candidate}))
            ledger.append(row)
            predictions[(model,name,recipe)] = candidate
        print(f'{component}: recomputed {len(ledger)} configurations',flush=True)
    indexed={(r['model'],r['format'],r['recipe']):r for r in ledger}
    comparisons=[]
    for model,name in sorted({(r['model'],r['format']) for r in ledger}):
        a,b=[indexed[(model,name,recipe)] for recipe in ('maxabs','percentile_99_9')]
        left,right=[predictions[(model,name,recipe)] for recipe in ('maxabs','percentile_99_9')]
        if a['baseline_prediction_digest']!=b['baseline_prediction_digest']:
            raise ValueError('recipe comparison has unequal baselines')
        pair={'model':model,'format':name,'family':a['family'],'metric':a['metric'],
              'left_recipe':'maxabs','right_recipe':'percentile_99_9',
              'difference_pp':b['candidate_percent']-a['candidate_percent'],
              'left_configuration':a['configuration_sha256'],'right_configuration':b['configuration_sha256']}
        if model!='yolov8n':
            stats=classification(left,right)
            pair.update(delta_interval_pp=[100*x for x in stats['metrics']['top1']['delta_interval']],
                        paired_counts=stats['metrics']['top1']['paired_counts'],
                        changed_top1_images=sum(x['top5'][0]!=y['top5'][0] for x,y in zip(left,right)))
        else:
            pair.update(delta_interval_pp=None, uncertainty='dataset-level AP difference; no CI computed')
        comparisons.append(pair)
    exact=[]
    for path in sorted((ROOT/'artifacts/phase3/runs').glob('*/summary.json')):
        summary=read(path)
        if summary.get('scope')!='screen' or summary.get('status')!='completed':
            continue
        job=read(path.parent/'job.json')
        prepared=read(checked(job['prepared']))
        config=read(checked(prepared['configuration']))
        model,name=config['model'],config['formats']['activation']['name']
        pairs=read(checked(summary['paired']))
        if len(pairs)!=1000 or len(summary['image_records'])!=1000:
            raise ValueError('incomplete exact screen')
        for pair,item in zip(pairs,summary['image_records']):
            record=read(checked(item))
            candidate=record['backends']['cuda']['prediction']
            if (pair['sample_sha256']!=record['sample']['sha256'] or pair['candidate_prediction']!=candidate
                    or pair['fp32_prediction']!=record['paired']['fp32_prediction']
                    or pair['ground_truth']!=record['paired']['ground_truth']):
                raise ValueError('exact paired artifact differs from saved predictions')
        ids=[r['sample_id'] for r in pairs]
        stats=paired_classification(pairs,ids,resamples=5000,seed=SEED)
        small=pairs[:128]
        if [r['sample_sha256'] for r in small]!=[r['sha256'] for r in populations[model]]:
            raise ValueError('exact and QDQ panels do not match')
        small_stats=paired_classification(small,ids[:128],resamples=5000,seed=SEED)
        cross=[]
        for recipe in ('maxabs','percentile_99_9'):
            candidate=predictions[(model,name,recipe)]
            aa=[{'sample':r['sample'],'top5':p['candidate_prediction']} for p,r in zip(small,candidate)]
            gap=classification(aa,candidate)
            cross.append({'recipe':recipe,'b_minus_a_top1_pp':100*gap['metrics']['top1']['delta'],
                          'delta_interval_pp':[100*x for x in gap['metrics']['top1']['delta_interval']],
                          'interpretation':'confounded recipe_and_execution_difference; not a causal backend or PTQ effect'})
        exact.append({'model':model,'format':name,'summary':reference(path),'paired':summary['paired'],
                      'images':1000,'statistics_1000':stats,'statistics_shared_128':small_stats,
                      'descriptive_b_comparisons':cross,'evidence_level':'retained_exact_A_screen',
                      'verification':'paired artifact and all referenced prediction-file hashes verified; not a new native acceptance'})
        print(f'{model}/{name}: analyzed retained exact 1000-image screen',flush=True)
    result={'version':'saved-prediction-study-v1','coverage':coverage,'b_configurations':ledger,
            'paired_recipe_comparisons':comparisons,'exact_a_screens':exact,
            'analysis_source':reference(Path(__file__)),
            'limits':['All 128-image data were observed before this design freeze.',
                      'Intervals are pointwise exploratory image-bootstrap intervals, not final or simultaneous guarantees.',
                      'COCO AP was recomputed at dataset level; detector uncertainty remains open.',
                      'A/B raw gaps change both recipe and execution; they cannot isolate either factor.',
                      'No inference or confirmation data collection was performed.']}
    immutable(OUT/'analysis.json',result)
    return result


def freeze_matrix(analysis):
    panels={}
    for prefix in ('imagenet','coco'):
        ev,rows,_=dataset(prefix+'_screen_1k')
        cal,calrows,_=dataset(prefix+'_calibration_2k')
        if {r['sha256'] for r in rows}&{r['sha256'] for r in calrows}:
            raise ValueError('calibration/evaluation leakage')
        panels[prefix]={'evaluation_manifest':ev,'calibration_manifest':cal,'calibration_images':len(calrows),
                        'ordered_development_image_sha256':[r['sha256'] for r in rows],
                        'development_sizes':[8,32,128,256,1000],
                        'confirmation_status':'not certified unused; audit earlier benchmark/calibration exposure first'}
    selected=[('resnet18','int4'),('resnet18','posit4_es0'),('resnet18','nf4'),
              ('mobilenet_v2','int4'),('mobilenet_v2','mxfp4_e2m1'),('mobilenet_v2','log4'),
              ('mobilenet_v3_large','fp6_e3m2'),('mobilenet_v3_large','q1_6'),
              ('mobilenet_v3_large','binary_pm1'),('mobilenet_v3_large','ternary'),
              ('yolov8n','int8'),('yolov8n','bfp6')]
    fmt={f['name']:f for f in formats()}
    accepted={(r['model'],r['format']) for r in analysis['exact_a_screens']}
    bridge=[]
    for model,name in selected:
        bridge.append({'id':f'E1/{model}/{name}','model':model,'format':name,'format_sha256':fmt[name]['sha256'],
                       'family':fmt[name]['family'],'initial_images':32,'maximum_development_images':128,
                       'selection_reason':'architecture/family/arithmetic coverage fixed before new outcomes; not ranked by current accuracy',
                       'arms':['strict_A_exact','same_A_graph_and_quantizers_FP32_reduction_control'],
                       'matched_factors':['checkpoint','graph','input_images','stored_weight_codes','scales','quantizer_rounding',
                                         'observer_policy','nonfinite_policy','operator_exceptions','preprocessing'],
                       'changed_factor':'accumulation_execution_semantics_only',
                       'status':'requires_matched_QDQ_control' if (model,name) in accepted else 'requires_native_acceptance_and_matched_QDQ_control',
                       'reuse_current_B_as_matched_control':False})
    promotions=[]
    for pair in analysis['paired_recipe_comparisons']:
        chosen='percentile_99_9' if pair['difference_pp']>0 else 'maxabs'
        interval=pair['delta_interval_pp']
        ambiguous=(interval is not None and interval[0]<=0<=interval[1] and pair['changed_top1_images']>0)
        recipes=['maxabs','percentile_99_9'] if ambiguous or pair['model']=='yolov8n' else [chosen]
        promotions.append({'model':pair['model'],'format':pair['format'],'recipes':recipes,
                           'target_images':1000,'additional_images_per_unchanged_configuration':872,
                           'semantics':'existing_versioned_FP32_QDQ_only','status':'planned_not_launched',
                           'selection_rule':'retain >=1 recipe per datatype/model; both for paired uncertainty or unquantified detector uncertainty'})
    matrix={'version':'breadth-comparison-matrix-v1','status':'frozen_prospective_design; executable arms require listed gates',
            'frozen_after':'existing 200x128 B and 8x1000 exact-A development observations',
            'analysis':reference(OUT/'analysis.json'),'panels':panels,
            'hypotheses':[
                'Recipe sensitivity depends on datatype family and model; maxabs/percentile exploration alone does not isolate datatype quality.',
                'QDQ/exact agreement must be measured using matched numerical configurations before transferring exploration conclusions.',
                'Complete hardware support cost can change conclusions drawn from multiplier cost alone.'],
            'E0_exact_compatibility':[{'model':m,'format':f,'pilot_images':8,'match':['all_layer_codes','outputs','execution_modes','quantizer_events'],
                                       'historical_acceptance_unchanged':True} for m,f in sorted(accepted)],
            'E1_matched_arithmetic':bridge,
            'E2_failure_diagnosis':[{'model':m,'format':f,'images':128,'arms':['strict_control','MSE_clipping_only','adaptive_rounding_only'],
                                     'arithmetic':'same exact backend in every arm','status':'requires_versioned_recipe_implementations',
                                     'extend_to_256_if':'paired interval crosses the declared 1pp development effect threshold',
                                     'interpretation':'development ablations; published methods adapted only with explicit labelling'}
                                    for m,f in [('resnet18','int4'),('mobilenet_v2','int4'),('mobilenet_v2','int5'),('mobilenet_v2','int6')]],
            'E3_B_breadth_extensions':promotions,
            'E3_exact_promotion_rule':{'status':'deferred until E0/E1 and throughput evidence',
                'requirements':['retain datatype-family and all-model coverage','include matched conventional baselines',
                                'retain uncertainty cases and plausible hardware specialists','do not select only current quality winners'],
                'budget_gate':'record measured seconds/image by family and total planned GPU/CPU hours before enrolling large exact runs'},
            'E4_robustness':{'initial_calibration_seeds':[20260925,20260926,20260927],
                'scope':'central contenders and matched baselines only','status':'not launched; freeze calibration manifests before fitting',
                'confirmation':'freeze methods then audit untouched benchmark complement; full validation for selected benchmark claims'},
            'E5_hardware':{'costs':['decode','multiply','accumulate','scale','bias','requantize','memory','data_movement'],
                           'native_INT4_speed_from_INT8_simulation':False},
            'invariants':['Do not alter historical A or B identities/acceptance status.',
                          'Changed recipe or arithmetic gets a new configuration; reuse predictions only under unchanged numerical semantics.',
                          'Keep all 25 datatypes and four models in the evidence ledger; blocked/deferred rows stay visible.',
                          'Any change to this matrix requires a new version and a recorded reason.',
                          'This supersedes no historical D4 completion requirement and declares no final ranking.']}
    immutable(MATRIX,matrix)
    return matrix


def report(analysis,matrix):
    fields=['model','format','family','bits','recipe','images','metric','baseline_percent','candidate_percent','delta_pp','delta_interval_pp','configuration_sha256']
    with (OUT/'configurations.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r.get(k) for k in fields} for r in analysis['b_configurations'])
    rows=['# Saved-prediction analysis and frozen comparison matrix','',
          'Recomputed 200 saved B configurations (25 datatypes × 4 models × 2 recipes),',
          'including dataset-level COCO AP, and verified paired predictions for eight exact-A screens.',
          'No new quality inference was performed. All reported panels are development data.','',
          'Machine-readable evidence: `results/summaries/saved-prediction-study-v1/analysis.json`.',
          'Frozen design: `public/experiments/configs/breadth-study/comparison-matrix-v1.json`.','',
          '## Paired recipe effects','',
          'Differences below are percentile minus maxabs, in percentage points. The classifier',
          'intervals are pointwise paired image-bootstrap intervals (5,000 draws), not simultaneous',
          'guarantees. Detector entries have point estimates only; AP is never averaged per image.','',
          '| Model | Percentile higher / tied / lower | Largest absolute recipe difference |',
          '| --- | --- | --- |']
    for model in ('resnet18','mobilenet_v2','mobilenet_v3_large','yolov8n'):
        pairs=[p for p in analysis['paired_recipe_comparisons'] if p['model']==model]
        counts=[sum(p['difference_pp']>0 for p in pairs),sum(p['difference_pp']==0 for p in pairs),sum(p['difference_pp']<0 for p in pairs)]
        largest=max(pairs,key=lambda p:abs(p['difference_pp']))
        rows.append(f"| {model} | {' / '.join(map(str,counts))} | {largest['format']}: {largest['difference_pp']:+.3f} pp |")
    rows+=['','## Retained exact A','', '| Model / format | Exact top-1 (1k) | Loss vs paired FP32 |','| --- | --- | --- |']
    for r in analysis['exact_a_screens']:
        s=r['statistics_1000']['metrics']['top1']
        rows.append(f"| {r['model']} / {r['format']} | {100*s['candidate']:.2f}% | {100*s['delta']:+.2f} pp |")
    rows+=['','The JSON also contains A/B comparisons on the identical 128 image IDs.',
           'These differences mix recipe and execution changes and cannot identify their causes.','',
           '## Frozen next comparisons','',
           '- E0: eight accepted integer graphs, eight-image exact compatibility each.',
           '- E1: twelve family/model coverage cases, 32 images initially, up to 128; matched QDQ controls need implementation.',
           '- E2: four severe-loss cases, 128-image controlled clipping/rounding ablations.',
           f"- E3 B: {sum(len(r['recipes']) for r in matrix['E3_B_breadth_extensions'])} existing QDQ configurations enrolled to 1k, preserving every datatype/model pair; not launched.",
           '- Larger exact runs remain gated by compatibility, matching, measured throughput and an explicit compute budget.',
           '- Final confirmation requires an exposure audit; previously evaluated images are not newly untouched data.','',
           'The matrix is frozen after observing the old development panel. It is prospective for new',
           'runs, not a retrospective preregistration. All ten accepted datatype families and four models',
           'are represented in E1; unavailable exact arms remain explicitly blocked.','']
    (ROOT/'docs/analysis/saved-prediction-study-v1.md').write_text('\n'.join(rows))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        for folder in ('experiment_b','experiment_b_ext'):
            stack.enter_context(exclusive(ROOT/'artifacts'/folder/'controller.lock'))
        analysis=analyze()
        matrix=freeze_matrix(analysis)
        report(analysis,matrix)
    print(MATRIX)


if __name__=='__main__':
    main()
