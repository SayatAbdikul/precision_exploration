"""Read-only inference evidence audit and paired development analysis for B.

Writes only new analysis outputs. Never imports the model runners, acquires native
leases, changes inference sources, or launches inference. Recipe comparisons use
identical image IDs; unequal enrollments are compared on their common prefix.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from tools.experiment_b.common import ROOT, dataset, digest, file_hash, formats, seal, unseal
from tools.phase3.common import checked, reference

OUT = ROOT / 'results/summaries/b-stage-paired-1k-v1'
SEED = 20260927
RECIPES = ('maxabs', 'percentile_99_9')
MODELS = ('resnet18', 'mobilenet_v2', 'mobilenet_v3_large', 'yolov8n')


def paired_outcomes(left, right, *, resamples=10000):
    """Multinomial sampling of -1/0/+1 equals paired image resampling of accuracy."""
    a, b = np.asarray(left, dtype=np.int8), np.asarray(right, dtype=np.int8)
    if a.shape != b.shape or a.ndim != 1 or not len(a) or not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError('paired correctness vectors must be equally sized nonempty binary arrays')
    counts = np.bincount(b-a+1, minlength=3)
    rng = np.random.default_rng(SEED)
    draws = rng.multinomial(len(a), counts/len(a), size=resamples)
    sampled = 100*(draws[:, 2]-draws[:, 0])/len(a)
    discordant = int(counts[0]+counts[2])
    return {'images': len(a), 'left_percent': float(100*a.mean()), 'right_percent': float(100*b.mean()),
            'difference_pp': float(100*(b-a).mean()),
            'pointwise_95_interval_pp': np.quantile(sampled, [.025, .975]).tolist(),
            'left_only_correct': int(counts[0]), 'right_only_correct': int(counts[2]),
            'both_correct': int((a*b).sum()), 'both_wrong': int(((1-a)*(1-b)).sum()),
            'mcnemar_exact_p': float(binomtest(int(counts[2]), discordant).pvalue) if discordant else 1.0,
            'zero_discordance_upper_95_percent': float(100*(1-.05**(1/len(a)))) if not discordant else None,
            'bootstrap': {'method': 'paired_binary_outcome_multinomial', 'resamples': resamples, 'seed': SEED}}


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result, previous = [None]*len(values), 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(values)-rank)*values[index]))
        result[index] = previous
    return result


def correctness(records):
    result = []
    for record in records:
        label, top = int(record['sample']['label']), record['top5']
        if len(top) != 5 or len(set(top)) != 5 or any(type(x) is not int or not 0 <= x < 1000 for x in top):
            raise ValueError('invalid saved top-five prediction')
        result.append((top[0] == label, label in top))
    return np.asarray(result, dtype=np.int8)


def compare_records(left, right):
    if len(left) != len(right) or any(a['sample'] != b['sample'] for a, b in zip(left, right)):
        raise ValueError('predictions are not paired on identical ordered samples')
    a, b = correctness(left), correctness(right)
    return {'top1': paired_outcomes(a[:, 0], b[:, 0]), 'top5': paired_outcomes(a[:, 1], b[:, 1]),
            'changed_top1_images': sum(a['top5'][0] != b['top5'][0] for a, b in zip(left, right)),
            'changed_top5_lists': sum(a['top5'] != b['top5'] for a, b in zip(left, right))}


def prediction_records(root, identity, population):
    records = [unseal(root/'predictions'/identity/(sample['sha256']+'.json')) for sample in population]
    if any(r['sample'] != s or r['configuration_sha256'] != identity for r, s in zip(records, population)):
        raise ValueError('prediction identity/population mismatch')
    return records


def analyze(out):
    selected_path = ROOT/'artifacts/breadth_study/selected_b_v1/plan.json'
    selected = unseal(selected_path)
    enrollment = {t['configuration_sha256']: t for g in selected['groups'].values() for t in g['tasks']}
    manifest = {f['name']: f for f in formats()}
    populations, inputs = {}, {}
    for model in MODELS:
        prefix = 'coco' if model == 'yolov8n' else 'imagenet'
        ev, rows, _ = dataset(prefix+'_screen_1k')
        cal, calrows, _ = dataset(prefix+'_calibration_2k')
        if {r['sha256'] for r in rows} & {r['sha256'] for r in calrows}:
            raise ValueError('calibration/evaluation overlap')
        populations[model] = rows
        inputs[prefix] = {'evaluation': ev, 'calibration': cal, 'ordered_image_digest': digest(rows)}
    baselines, records_by_key, ledger = {}, {}, []
    for component in ('experiment_b', 'experiment_b_ext'):
        root = ROOT/'artifacts'/component
        state_path = root/'status.json'
        for task in json.loads(state_path.read_text())['tasks']:
            if task['status'] != 'completed':
                continue
            identity = task['configuration_sha256']
            model, name, recipe = (task[k] for k in ('model', 'format', 'recipe'))
            n = 1000 if identity in enrollment else 128
            conf_path = root/'configurations'/f'{identity}.json'
            config = unseal(conf_path)
            if (digest(config) != identity or (config['model_context']['model'], config['format'], config['recipe']) != (model, name, recipe)
                    or (identity in enrollment and file_hash(conf_path) != enrollment[identity]['configuration_file_sha256'])):
                raise ValueError('configuration drift')
            summary_path = root/'summaries'/f'{identity}-{n}.json'
            summary = unseal(summary_path)
            if summary['configuration_sha256'] != identity or summary['panel_images'] != n:
                raise ValueError('summary identity/panel mismatch')
            population = populations[model][:n]
            bid = config['baseline_sha256']
            cache_key = component, bid, n
            if cache_key not in baselines:
                baselines[cache_key] = prediction_records(root, bid, population)
            baseline = baselines[cache_key]
            candidate = prediction_records(root, identity, population)
            if digest(candidate) != summary['prediction_digest'] or digest(baseline) != summary['baseline_prediction_digest']:
                raise ValueError('saved summary does not match prediction digests')
            prefix = unseal(root/'summaries'/f'{identity}-128.json')
            if digest(candidate[:128]) != prefix['prediction_digest'] or digest(baseline[:128]) != prefix['baseline_prediction_digest']:
                raise ValueError('original 128-image prefix changed')
            row = {'model': model, 'format': name, 'family': manifest[name]['family'], 'bits': manifest[name]['bits'],
                   'recipe': recipe, 'images': n, 'component': component, 'configuration_sha256': identity,
                   'configuration': reference(conf_path), 'summary': reference(summary_path),
                   'source_sha256': config['source_sha256'], 'protocol': config['protocol'],
                   'prediction_digest': summary['prediction_digest'], 'baseline_sha256': bid,
                   'baseline_prediction_digest': summary['baseline_prediction_digest'],
                   'evidence_level': 'development_FP32_QDQ_not_exact_acceptance'}
            if model == 'yolov8n':
                row.update(metric='map50_95', candidate_percent=100*summary['metrics']['map50_95'],
                           baseline_percent=100*summary['metrics']['fp32_map50_95'],
                           delta_pp=100*summary['metrics']['delta_map50_95'],
                           empty_prediction_images=sum(not r['detections'] for r in candidate),
                           detection_count=sum(len(r['detections']) for r in candidate),
                           output_digest=digest([r['detections'] for r in candidate]),
                           uncertainty='dataset-level AP from sealed evaluator summary; paired-bootstrap shortlisted separately')
            else:
                stats = compare_records(baseline, candidate)
                if not np.isclose(stats['top1']['right_percent'], summary['metrics']['top1_percent'], rtol=0, atol=1e-10):
                    raise ValueError('recomputed top1 differs from summary')
                if not np.isclose(stats['top5']['right_percent'], summary['metrics']['top5_percent'], rtol=0, atol=1e-10):
                    raise ValueError('recomputed top5 differs from summary')
                row.update(metric='top1', candidate_percent=stats['top1']['right_percent'], baseline_percent=stats['top1']['left_percent'],
                           delta_pp=stats['top1']['difference_pp'], statistics=stats,
                           unique_top1_predictions=len({r['top5'][0] for r in candidate}),
                           output_digest=digest([r['top5'] for r in candidate]))
                if n == 1000:
                    row['extension_872_vs_fp32'] = compare_records(baseline[128:], candidate[128:])
                records_by_key[model, name, recipe] = candidate
            ledger.append(row)
            if len(ledger) % 20 == 0:
                print(f'Verified {len(ledger)}/200 configurations', flush=True)
    indexed = {(r['model'], r['format'], r['recipe']): r for r in ledger}
    if len(indexed) != 200 or len(ledger) != 200 or sum(r['images'] == 1000 for r in ledger) != 171:
        raise ValueError('unexpected completed B coverage')
    comparisons = []
    for model in MODELS:
        for name in manifest:
            a, b = [indexed[model, name, recipe] for recipe in RECIPES]
            n = min(a['images'], b['images'])
            if a['baseline_sha256'] != b['baseline_sha256']:
                raise ValueError('recipe pair has different baselines')
            pair = {'model': model, 'format': name, 'family': a['family'], 'images': n,
                    'available_images': [a['images'], b['images']], 'metric': a['metric'],
                    'left_recipe': RECIPES[0], 'right_recipe': RECIPES[1],
                    'left_configuration': a['configuration_sha256'], 'right_configuration': b['configuration_sha256']}
            if model == 'yolov8n':
                if n != 1000:
                    raise ValueError('detector enrollment unexpectedly incomplete')
                pair.update(left_percent=a['candidate_percent'], right_percent=b['candidate_percent'],
                            difference_pp=b['candidate_percent']-a['candidate_percent'],
                            pointwise_95_interval_pp=None, uncertainty='AP difference is descriptive pending paired image bootstrap')
            else:
                aa, bb = [records_by_key[model, name, recipe][:n] for recipe in RECIPES]
                stats = compare_records(aa, bb)
                pair.update(statistics=stats, **{k: stats['top1'][k] for k in ('left_percent', 'right_percent', 'difference_pp', 'pointwise_95_interval_pp')})
                pair['initial_128'] = compare_records(aa[:128], bb[:128])
                if n == 1000:
                    pair['extension_872'] = compare_records(aa[128:], bb[128:])
            comparisons.append(pair)
    classifier = [r for r in comparisons if r['model'] != 'yolov8n']
    for r, value in zip(classifier, holm([r['statistics']['top1']['mcnemar_exact_p'] for r in classifier])):
        r['mcnemar_holm_75_p'] = value
    aliases = []
    for model in MODELS:
        for recipe in RECIPES:
            groups = {}
            for row in ledger:
                if row['model'] == model and row['recipe'] == recipe and row['images'] == 1000:
                    groups.setdefault(row['output_digest'], []).append(row['format'])
            aliases += [{'model': model, 'recipe': recipe, 'images': 1000, 'formats': names,
                         'scope': 'identical saved top-five lists or detections; not proof of all-logit or arithmetic equivalence'}
                        for names in groups.values() if len(names) > 1]
    result = {'version': 'b-stage-paired-1k-v1', 'analysis_source': reference(Path(__file__)),
              'selected_plan': reference(selected_path), 'inputs': inputs, 'configurations': ledger,
              'paired_recipe_comparisons': comparisons, 'observed_output_aliases': aliases,
              'coverage': {'configurations': 200, 'at_1000': 171, 'at_128': 29, 'datatype_model_pairs': 100,
                           'paired_at_1000': sum(r['images'] == 1000 for r in comparisons),
                           'classifier_pairs_at_1000': sum(r['images'] == 1000 for r in classifier)},
              'limitations': ['Development analysis; shortlist selected after observing these data.',
                              '128-image selection panel and 872-image extension reported separately; neither is final confirmation.',
                              'Recipe comparisons with unequal enrollment use only the common 128 images.',
                              'Pointwise bootstrap intervals and exploratory Holm-adjusted tests do not undo adaptive enrollment or selection.',
                              'Zero observed correctness discordance does not establish equivalence; a binomial upper bound is supplied.',
                              'QDQ accuracy and runtime do not establish exact arithmetic quality or native hardware speed.',
                              'Detector AP is dataset-level, never an average of per-image AP.']}
    seal(out/'analysis.json', result)
    fields = ['model', 'format', 'family', 'bits', 'recipe', 'images', 'metric', 'baseline_percent', 'candidate_percent', 'delta_pp', 'configuration_sha256']
    with (out/'configurations.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(ledger)
    fields = ['model', 'format', 'family', 'images', 'left_percent', 'right_percent', 'difference_pp', 'pointwise_95_interval_pp', 'available_images']
    with (out/'recipe-pairs.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(comparisons)
    print(json.dumps(result['coverage']), flush=True)


def cross_compare(left_key, right_key, out):
    analysis = unseal(out/'analysis.json')
    index = {(r['model'], r['format'], r['recipe']): r for r in analysis['configurations']}
    left, right = index[tuple(left_key)], index[tuple(right_key)]
    if left['model'] != right['model'] or left['baseline_prediction_digest'] != right['baseline_prediction_digest']:
        raise ValueError('cross-format comparison requires same model, panel and baseline')
    n = min(left['images'], right['images'])
    model = left['model']
    _, rows, _ = dataset(('coco' if model == 'yolov8n' else 'imagenet')+'_screen_1k')
    records = [prediction_records(ROOT/'artifacts'/r['component'], r['configuration_sha256'], rows[:n]) for r in (left, right)]
    return left, right, rows[:n], records


def detector_bootstrap(out, left_key, right_key, resamples):
    from public.analysis.phase3.coco_cache import paired_coco_cached
    left, right, rows, (a, b) = cross_compare(left_key, right_key, out)
    manifest = json.loads((ROOT/'public/workloads/datasets/coco2017.json').read_text())
    path = checked({'path': 'data/raw/coco2017/annotations/instances_val2017.json',
                    'sha256': manifest['annotation_sha256']['instances_val2017']})
    annotations = json.loads(path.read_text())
    stats = paired_coco_cached(annotations, [d for r in a for d in r['detections']],
                              [d for r in b for d in r['detections']], [int(r['image_id']) for r in rows],
                              resamples=resamples, seed=SEED)
    for field, row in [('fp32', left), ('candidate', right)]:
        if not np.isclose(100*stats['metrics']['map50_95'][field], row['candidate_percent'], rtol=0, atol=1e-10):
            raise ValueError('recomputed AP differs from audited summary')
    result = {'left': left_key, 'right': right_key, 'left_configuration': left['configuration_sha256'],
              'right_configuration': right['configuration_sha256'], 'statistics': stats,
              'labels': 'fp32 and candidate metric keys denote left and right arms, both are B QDQ configurations',
              'analysis_source': reference(Path(__file__)),
              'scope': 'selected exploratory pointwise paired image bootstrap; no final ranking'}
    slug = '__'.join('-'.join(x) for x in (left_key, right_key))
    seal(out/'detector-bootstrap'/f'{slug}.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['analyze', 'detector-bootstrap'])
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--left', nargs=3)
    parser.add_argument('--right', nargs=3)
    parser.add_argument('--resamples', type=int, default=500)
    args = parser.parse_args()
    if args.action == 'analyze':
        analyze(args.output)
    else:
        detector_bootstrap(args.output, args.left, args.right, args.resamples)


if __name__ == '__main__':
    main()
