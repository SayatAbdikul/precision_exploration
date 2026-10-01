"""Prospective useful-quality E1: frozen canonical graphs and matched arithmetic.

All evidence is task-owned. Historical A/B matrices, runners and results remain
immutable. A B score motivates a case but is never assigned to its A graph.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

from tools.experiment_b.common import atomic_json, unseal
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference
from tools.phase3.execution_guard import verify_guard
from tools.phase3.worker_locks import exclusive, pool_locks
from tools.run.exact_execution import immutable, implementation, archive_implementation, event_signature

BASE = ROOT/'artifacts/breadth_study/useful_quality_e1_v1'
CASES = [('resnet18', 'int8'), ('mobilenet_v2', 'int8'), ('resnet18', 'fp6_e2m3'),
         ('resnet18', 'fp6_e3m2'), ('resnet18', 'fp7_e3m3')]
PROTOCOL = {
    'version': 'useful-quality-matched-e1-v1', 'created_before_new_inference': True,
    'configurations': 'canonical frozen A graphs; B maxabs is separate motivational evidence, not a translated recipe',
    'arms': ['strict_declared_Model_C', 'matched_sequential_FP32_reductions', 'paired_full_precision_baseline',
             'ordinary_B_maxabs_QDQ_separate_configuration'],
    'invariants': ['checkpoint', 'preprocessing', 'graph', 'stored_W_A_codes', 'scales', 'bias_policy',
                   'operator_policy', 'recipe', 'sample_order', 'sequential_reduction_order'],
    'arithmetic_changes': {
        'MAC': 'sequential RNE FP32 FMA of the same decoded operands (unscaled integer codes for integer MAC); cast final state into original accumulator before unchanged bias/store',
        'residual': 'one RNE FP32 rounding of aligned two-input sum in original sum domain; integer pairs use the equivalent exact LUT',
        'average_pool': 'sequential RNE FP32 sum in original domain; original division and output quantizer retained',
        'other': 'ReLU/ReLU6, maxpool, flatten, bias encoding/addition, scale application and output RNE/saturation unchanged',
        'excluded_ops': 'no detector, shared-scale, softmax, nonlinear-LUT, binary, logarithmic or NF4 graphs'},
    'panel': {'order': 'first ascending frozen imagenet_screen_1k image SHA256', 'initial': 32,
              'pilot': 8, 'maximum': 128, 'timing_probe': 1,
              'exposure': 'all panels are development; existing data exposure is retained, not independent confirmation'},
    'reuse': 'only verified original graph/configuration/source and sample identities; retain original exact execution provenance and timings; no B-to-A reuse',
    'statistics': {'bootstrap_draws': 10000, 'seed': 20260927, 'confidence': .95,
                   'paired': True, 'zero_discordance': 'one-sided exact binomial upper bound, not evidence of equality'},
    'promotion': {'useful_top1_percent': 50, 'useful_max_loss_to_FP32_pp': 20,
                  'informative_min_top1_percent': 40, 'collapsed_max_top1_percent': 10,
                  'rule': 'promote if either arithmetic arm has top1>=50% and loss<=20pp, or either has top1>=40% with observed layer/output arithmetic divergence; both<=10% never promote; fixed candidate order, cost gate applies',
                  'uncertainty_reason': '128 narrows the 32-image accuracy and discordance uncertainty; no equivalence or final superiority claim',
                  'anchors': 'use the same 128-image population for useful integer anchors and promoted floating cases'},
    'compute': {'threads_per_worker': 4, 'initial_workers': 2, 'maximum_workers': 3,
                'maximum_new_inference_worker_hours': 24, 'maximum_extension_worker_hours': 12,
                'estimate': '1.5 times measured median per-mode CUDA seconds, including input/diagnostic preparation; exact imported images have zero new inference cost',
                'concurrency': 'at most 3 only if timing pilot maximum RSS < 2GiB and MemAvailable >=12GiB; otherwise 2, or 1 below6GiB',
                'budget_policy': 'complete admitted 32-image panels; allocate bounded 128-image extensions in frozen case order from remaining measured budget; record any budget stop'},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sources():
    paths = ['tools/breadth_study/useful_quality.py', 'tools/run/useful_quality.py',
             'tools/breadth_study/matched_control_v2.py', 'tools/breadth_study/e1.py',
             'development/acceptance_proofs/family_static.py',
             'development/acceptance_proofs/dyadic_fp_mac_store.py',
             'tools/phase3/acceptance.py', 'tools/phase3/evidence.py', 'tools/phase3/controller_worker.py',
             'public/analysis/phase3/finite_fp64_nonmac.py']
    return {**implementation(), **{p: reference(ROOT/p)['sha256'] for p in paths}}


def assert_plan(plan):
    if plan['sources'] != sources() or plan['guard_sha256'] != verify_guard():
        raise ValueError('useful-quality execution source/runtime changed')
    for key in ('graph', 'configuration', 'baseline', 'protocol', 'mapping', 'static_proof'):
        checked(plan[key])


def static_proof(prepared):
    if prepared['format'] == 'int8':
        from tools.phase3.acceptance import verify_acceptance
        path = ROOT/'artifacts/phase3/acceptance'/prepared['configuration_sha256']/'prepared-accepted.json'
        acceptance = verify_acceptance(read(path))
        proof = read(checked(acceptance['proof']))
        return {'kind': 'accepted_integer', 'prepared': reference(path), 'acceptance': acceptance,
                'proof': proof, 'graph': prepared['graph']}
    from development.acceptance_proofs.family_static import verify_static
    result = verify_static(prepared)
    return {'kind': 'reproduced_dyadic_graph', 'proof': result, 'graph': prepared['graph'],
            'mac_proof_summary': reference(ROOT/'artifacts/phase3/proof-development/dyadic-fp-mac-store-v1/summary.json'),
            'nonmac_proof_summary': reference(ROOT/'results/summaries/phase3-finite-fp64-nonmac.json')}


def freeze_study():
    from tools.phase3.preparation_inventory import preparation_records
    from tools.phase3.baselines import screen_rows
    from public.quantization.graph.executable import graph_sha256
    prepared = preparation_records()
    population = screen_rows('resnet18', campaign(), verify_images=False)[0][:128]
    if population != screen_rows('mobilenet_v2', campaign(), verify_images=False)[0][:128]:
        raise ValueError('models do not share the declared image order')
    protocol = {**PROTOCOL, 'sample_rows': population, 'sample_digest': digest({'rows': population}),
                'contracts': [reference(ROOT/'docs/contracts'/name) for name in
                              ('arithmetic.md', 'operator-semantics.md', 'experiment-a.md')]}
    protocol_path = BASE/'protocol.json'
    immutable(protocol_path, protocol)
    current = sources()
    archive = archive_implementation(current)
    b_analysis = unseal(ROOT/'results/summaries/b-stage-paired-1k-v2/analysis.json')
    b_index = {(r['model'], r['format'], r['recipe']): r for r in b_analysis['configurations']}
    entries = []
    for model, fmt in CASES:
        case = model+'/'+fmt
        p = prepared[case]
        graph, config = read(checked(p['graph'])), read(checked(p['configuration']))
        manifest = read(checked(campaign()['inputs'][model]))
        brow = b_index[model, fmt, 'maxabs']
        bconfig = unseal(checked(brow['configuration']))
        calibration = read(checked(config['calibration']))
        for key, expected in [('checkpoint_sha256', manifest['checkpoint_sha256']),
                              ('preprocessing_sha256', manifest['preprocessing_sha256']),
                              ('source_graph_sha256', manifest['deployment_graph_sha256'])]:
            if bconfig['model_context'][key] != expected or calibration['context'][key] != expected:
                raise ValueError('candidate/B checkpoint, preprocessing or folded graph mismatch')
        node_map = [{**node, 'constant_inputs': {name: {'shape': graph['constants'][name]['shape'],
                    'encoding': graph['constants'][name]['encoding'], 'codes_sha256': digest({'codes': graph['constants'][name]['codes']})}
                    for name in node['inputs'] if name in graph['constants']}} for node in graph['nodes']]
        mapping = {'case': case, 'graph': p['graph'], 'graph_sha256': graph_sha256(graph),
                   'configuration': p['configuration'], 'model_manifest': campaign()['inputs'][model],
                   'checkpoint': {'path': manifest['checkpoint_path'], 'sha256': manifest['checkpoint_sha256']},
                   'preprocessing': {'path': manifest['preprocessing_path'], 'sha256': manifest['preprocessing_sha256']},
                   'folded_graph': {'path': manifest['deployment_graph_path'], 'sha256': manifest['deployment_graph_sha256']},
                   'calibration': config['calibration'], 'calibration_context': calibration['context'],
                   'calibration_mapping_policy': calibration['mapping_policy'], 'recipe': graph['provenance'],
                   'formats': config['formats'], 'inputs': graph['inputs'], 'nodes': node_map,
                   'operator_counts': dict(Counter(n['op'] for n in graph['nodes'])),
                   'rounding_and_saturation': 'defined by exact graph manifest_hashes and per-operator attrs; RNE output stores',
                   'manifest_hashes': graph['manifest_hashes'], 'sample_digest': digest({'rows': population}),
                   'ordinary_B': {'configuration': brow['configuration'], 'summary_1000': brow['summary'],
                       'configuration_sha256': brow['configuration_sha256'], 'component': brow['component'],
                       'model_context': bconfig['model_context'], 'protocol': bconfig['protocol'],
                       'scales': bconfig['scales'], 'baseline_sha256': bconfig['baseline_sha256'],
                       'relation': 'different configuration; no translation or pure arithmetic B-to-A attribution',
                       'differences': ['A MSE necessary integer scales or unscaled floating formats versus B maxabs external scales',
                                       'A explicit aligned/quantized residual inputs versus framework residual addition',
                                       'A accumulator-domain bias store versus framework bias addition',
                                       'canonical executable graph versus folded FX QDQ graph and framework reduction order']}}
        mapping_path = BASE/'mappings'/f'{model}-{fmt}.json'
        immutable(mapping_path, mapping)
        try:
            proof = static_proof(p)
        except (ValueError, FileNotFoundError) as error:
            entries.append({'case': case, 'status': 'blocked_static_proof', 'reason': str(error),
                            'mapping': reference(mapping_path)})
            continue
        proof_path = BASE/'proofs'/f'{model}-{fmt}.json'
        immutable(proof_path, proof)
        screen = None
        if fmt == 'int8':
            found = [path for path in (ROOT/'artifacts/phase3/runs').glob('*/summary.json')
                     if (lambda s: s.get('configuration_sha256') == p['configuration_sha256']
                         and s.get('scope') == 'screen' and s.get('images') == 1000)(read(path))]
            if len(found) != 1:
                raise ValueError('integer anchor needs its unique accepted exact screen')
            screen = reference(found[0])
        plan = {'version': 'useful-quality-e1-case-v1', 'case': case, 'model': model, 'format': fmt,
                'graph': p['graph'], 'configuration': p['configuration'], 'prepared': p,
                'graph_sha256': graph_sha256(graph), 'deployed_nodes': [n['name'] for n in graph['nodes']],
                'mapping': reference(mapping_path), 'static_proof': reference(proof_path),
                'protocol': reference(protocol_path), 'sources': current, 'source_archive': archive,
                'guard_sha256': verify_guard(), 'retained_exact_screen': screen,
                'baseline': read(ROOT/'results/summaries/phase3-baselines.json')['models'][model]}
        path = BASE/'jobs'/digest(plan)/'plan.json'
        immutable(path, plan)
        entries.append({'case': case, 'status': 'static_passed_native_gate_required', 'plan': reference(path)})
    enrollment = {'version': 'useful-quality-e1-enrollment-v1', 'protocol': reference(protocol_path), 'cases': entries}
    immutable(BASE/'enrollment.json', enrollment)
    return enrollment


def validate_record(record, plan, sample, mode, backend, index):
    if (record['job_sha256'] != digest(plan) or record['sample'] != sample or record['mode'] != mode
            or record['backend'] != backend or record['index'] != index
            or record['graph_sha256'] != plan['graph_sha256']
            or set(record['layers']) != set(plan['deployed_nodes'])):
        raise ValueError('useful-quality checkpoint identity, sample or layer coverage mismatch')
    if len(record['prediction']) != 5 or len(set(record['prediction'])) != 5:
        raise ValueError('invalid saved top-five prediction')
    if record.get('original_record'):
        original = read(checked(record['original_record']))
        expected = original['backends'][backend]
        if (original['sample'] != sample or original['graph_sha256'] != record['graph_sha256']
                or any(record[k] != expected[k] for k in ('layers', 'output_sha256', 'prediction', 'diagnostics'))):
            raise ValueError('imported exact record no longer matches original evidence')


def paired_stats(left, right):
    import numpy as np
    a, b = np.asarray(left, dtype=int), np.asarray(right, dtype=int)
    if a.ndim != 1 or len(a) != len(b) or not len(a) or not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError('paired correctness needs matching nonempty binary vectors')
    counts = np.bincount(b-a+1, minlength=3)
    rng = np.random.default_rng(PROTOCOL['statistics']['seed'])
    draws = rng.multinomial(len(a), counts/len(a), PROTOCOL['statistics']['bootstrap_draws'])
    delta = 100*(draws[:, 2]-draws[:, 0])/len(a)
    return {'left_percent': float(a.mean()*100), 'right_percent': float(b.mean()*100),
            'difference_pp': float((b-a).mean()*100), 'pointwise_95_interval_pp': np.quantile(delta, [.025, .975]).tolist(),
            'left_only_correct': int(counts[0]), 'right_only_correct': int(counts[2]),
            'discordant_images': int(counts[0]+counts[2]),
            'zero_discordance_upper_95_percent': 100*(1-.05**(1/len(a))) if not counts[0]+counts[2] else None}


def report_panel(path, images):
    plan = unseal(path)
    population = unseal(checked(plan['protocol']))['sample_rows'][:images]
    rows = {}
    for mode in ('exact', 'control'):
        rows[mode] = []
        for index, sample in enumerate(population):
            record = unseal(path.parent/f'{index:04d}-{mode}-cuda.json')
            validate_record(record, plan, sample, mode, 'cuda', index)
            rows[mode].append(record)
    for a, b in zip(rows['exact'], rows['control']):
        if a['ground_truth'] != b['ground_truth'] or a['fp32_prediction'] != b['fp32_prediction']:
            raise ValueError('arithmetic arms have different paired baselines or labels')
    stats = {}
    for rank in (1, 5):
        vectors = {mode: [r['ground_truth'] in r['prediction'][:rank] for r in records] for mode, records in rows.items()}
        fp32 = [r['ground_truth'] in r['fp32_prediction'][:rank] for r in rows['exact']]
        stats[f'top{rank}'] = {'control_minus_exact': paired_stats(vectors['exact'], vectors['control']),
                               'exact_minus_FP32': paired_stats(fp32, vectors['exact']),
                               'control_minus_FP32': paired_stats(fp32, vectors['control'])}
    graph = read(checked(plan['graph']))
    divergences = []
    for a, b in zip(rows['exact'], rows['control']):
        changed = [node['name'] for node in graph['nodes'] if a['layers'][node['name']] != b['layers'][node['name']]]
        divergences.append({'index': a['index'], 'sample_sha256': a['sample']['sha256'], 'first_layer': changed[0] if changed else None,
                            'changed_layers': changed, 'output_matches': a['output_sha256'] == b['output_sha256']})
    result = {'plan': reference(path), 'images': images, 'statistics': stats,
              'all_layer_agreement_images': sum(not r['changed_layers'] for r in divergences),
              'output_agreement_images': sum(r['output_matches'] for r in divergences), 'divergences': divergences,
              'records': {mode: [reference(path.parent/f'{i:04d}-{mode}-cuda.json') for i in range(images)]
                          for mode in ('exact', 'control')},
              'scope': 'paired canonical-A arithmetic development; not a B translation or independent confirmation'}
    immutable(path.parent/f'summary-{images}.json', result)
    return result


def promotion(summary):
    top = summary['statistics']['top1']
    exact, control = top['control_minus_exact']['left_percent'], top['control_minus_exact']['right_percent']
    if max(exact, control) <= PROTOCOL['promotion']['collapsed_max_top1_percent']:
        return False, 'both arithmetic arms collapsed; additional images uninformative'
    useful = any(top[key]['right_percent'] >= 50 and top[key]['difference_pp'] >= -20
                 for key in ('exact_minus_FP32', 'control_minus_FP32'))
    informative = max(exact, control) >= 40 and summary['all_layer_agreement_images'] < summary['images']
    return (True, 'useful absolute accuracy; narrow accuracy/arithmetic uncertainty') if useful else (
        (True, 'useful-adjacent quality with observed arithmetic divergence') if informative else
        (False, 'does not meet frozen useful/informative quality rule'))


def worker(path, images, fds):
    from tools.breadth_study.e1 import validate_lease
    validate_lease(fds)
    import torch
    import numpy as np
    from public.analysis.phase3.diagnostics import LayerDiagnostics, ReferenceSamples
    from public.inference.tensor import QUANTIZATION_OBSERVER, parse_encoding
    from public.quantization.ptq.encoding import float_tensor
    from public.quantization.graph.executable import execute
    from tools.breadth_study.matched_control_v2 import MatchedControl, native_conformance
    from tools.exact_execution_v2.engine import PreparedGraph
    from tools.exact_execution_v2.activation_trace import ActivationTrace
    from tools.phase3.runtime import load_runtime
    from tools.phase3.baselines import screen_rows
    from tools.phase3.controller_worker import rational_admission_guard
    from tools.run.phase3_thread_benchmark import set_threads
    started = time.perf_counter()
    plan = unseal(path)
    assert_plan(plan)
    if images not in (1, 8, 32, 128):
        raise ValueError('unsupported useful-quality panel')
    if static_proof(plan['prepared']) != unseal(checked(plan['static_proof'])):
        raise ValueError('static proof no longer reproduces')
    graph = read(checked(plan['graph']))
    population, payload, _ = screen_rows(plan['model'], campaign(), verify_images=False)
    if population[:128] != unseal(checked(plan['protocol']))['sample_rows']:
        raise ValueError('prospective image order changed')
    baseline = read(checked(plan['baseline']))
    retained = None
    if plan['retained_exact_screen']:
        from tools.phase3.evidence import verify_complete
        verify_complete(checked(plan['retained_exact_screen']), current_execution=True)
        retained = read(checked(plan['retained_exact_screen']))
    torch.set_num_interop_threads(1)
    set_threads(4)
    sample, observe, manifest = load_runtime(plan['model'], campaign())
    name = next(iter(graph['inputs']))
    encoding = parse_encoding(graph['inputs'][name])
    for backend in ('cpp', 'cuda'):
        immutable(path.parent/f'control-conformance-{backend}.json', native_conformance(backend))
    engines = {}
    for mode in ('exact', 'control'):
        for backend in ('cpp', 'cuda'):
            count = min(8, images) if backend == 'cpp' else images
            if images > 8 and not (path.parent/'native-admission.json').exists():
                raise ValueError('32/128 panel requires completed native admission')
            if not (mode == 'exact' and retained):
                tick = time.perf_counter()
                engines[mode, backend] = (PreparedGraph if mode == 'exact' else MatchedControl)(graph, backend=backend)
                print(f"{plan['case']} {mode}/{backend} engine setup {time.perf_counter()-tick:.2f}s", flush=True)
            for index, row in enumerate(population[:count]):
                paired = baseline['records'][index]
                if row['sha256'] != paired['sample_sha256'] or reference(payload/row['relative_path'])['sha256'] != row['sha256']:
                    raise ValueError('evaluation pairing or payload changed')
                destination = path.parent/f'{index:04d}-{mode}-{backend}.json'
                if destination.exists():
                    validate_record(unseal(destination), plan, row, mode, backend, index)
                    continue
                if mode == 'exact' and retained:
                    original_ref = retained['image_records'][index]
                    old = read(checked(original_ref))
                    if backend not in old['backends']:
                        pilot = unseal(checked(plan['static_proof']))['acceptance']['pilot']
                        original_ref = read(checked(pilot))['image_records'][index]
                        old = read(checked(original_ref))
                    actual = old['backends'][backend]
                    record = {**actual, 'job_sha256': digest(plan), 'sample': row, 'mode': mode, 'backend': backend,
                              'index': index, 'graph_sha256': old['graph_sha256'], 'original_record': original_ref,
                              'source_kind': 'retained_verified_exact', 'new_compute_seconds': 0,
                              'ground_truth': paired['ground_truth'], 'fp32_prediction': paired['fp32_prediction'],
                              'diagnostic_signature': event_signature(actual['diagnostics'])}
                    validate_record(record, plan, row, mode, backend, index)
                else:
                    tick = time.perf_counter()
                    fp32, _ = sample(payload/row['relative_path'])
                    if list(fp32.shape) != manifest['input_shape']:
                        raise ValueError('preprocessing changed')
                    refs = ReferenceSamples(campaign()['diagnostics']['sample_elements_per_layer_image'])
                    observe(fp32, refs)
                    tensor = float_tensor(fp32.numpy(), encoding)
                    prep_seconds = time.perf_counter()-tick
                    diagnostics = LayerDiagnostics(refs)
                    with ActivationTrace(), rational_admission_guard():
                        token = QUANTIZATION_OBSERVER.set(diagnostics.quantization)
                        tick = time.perf_counter()
                        try:
                            actual = engines[mode, backend].execute({name: tensor}, observer=diagnostics)
                        finally:
                            QUANTIZATION_OBSERVER.reset(token)
                        seconds = time.perf_counter()-tick
                    output = next(iter(actual['outputs'].values()))
                    values = output.values()
                    if not all(np.isfinite(float(v)) for v in values):
                        raise ValueError('candidate output is nonfinite')
                    record = {'job_sha256': digest(plan), 'sample': row, 'mode': mode, 'backend': backend, 'index': index,
                              'graph_sha256': actual['graph_sha256'], 'layers': actual['layers'],
                              'execution_modes': actual['execution_modes'], 'output_sha256': digest(output.document()),
                              'prediction': sorted(range(len(values)), key=lambda i: (-values[i], i))[:5],
                              'ground_truth': paired['ground_truth'], 'fp32_prediction': paired['fp32_prediction'],
                              'diagnostics': diagnostics.records, 'diagnostic_signature': event_signature(diagnostics.records),
                              'source_kind': 'new_verified_execution', 'input_preparation_seconds': prep_seconds,
                              'seconds': seconds, 'new_compute_seconds': seconds+prep_seconds,
                              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                    if mode == 'exact' and backend == 'cpp' and index == 0:
                        legacy_diag = LayerDiagnostics(refs)
                        with ActivationTrace(), rational_admission_guard():
                            token = QUANTIZATION_OBSERVER.set(legacy_diag.quantization)
                            tick = time.perf_counter()
                            try:
                                legacy = execute(graph, {name: tensor}, backend='cpp', observer=legacy_diag)
                            finally:
                                QUANTIZATION_OBSERVER.reset(token)
                        compatibility = {'graph': plan['graph'], 'sample': row, 'seconds': time.perf_counter()-tick,
                                         'matches': legacy['layers'] == record['layers'] and
                                         digest(next(iter(legacy['outputs'].values())).document()) == record['output_sha256'] and
                                         event_signature(legacy_diag.records) == record['diagnostic_signature']}
                        immutable(path.parent/'prepared-reference-compatibility.json', compatibility)
                        if not compatibility['matches']:
                            raise ValueError('prepared/reference exact execution mismatch')
                    assert_plan(plan)
                if backend == 'cuda' and index < 8:
                    cpu = unseal(path.parent/f'{index:04d}-{mode}-cpp.json')
                    fields = ('layers', 'output_sha256', 'prediction')
                    if mode == 'control' or not retained:
                        fields += ('diagnostic_signature',)
                    if any(record[key] != cpu[key] for key in fields):
                        raise ValueError(f'{mode} native CPP/CUDA discrepancy')
                immutable(destination, record)
                atomic_json(path.parent/'progress.json', {'case': plan['case'], 'mode': mode, 'backend': backend,
                            'images': index+1, 'target': count, 'updated_at': now(), 'seconds': record['new_compute_seconds']})
                print(f"{plan['case']} {mode}/{backend} {index+1}/{count}: {record['new_compute_seconds']:.2f}s", flush=True)
    if images >= 8:
        for mode in ('exact', 'control'):
            for index in range(8):
                a, b = [unseal(path.parent/f'{index:04d}-{mode}-{backend}.json') for backend in ('cpp', 'cuda')]
                if any(a[k] != b[k] for k in ('layers', 'output_sha256', 'prediction')):
                    raise ValueError('resumed native gate mismatch')
        immutable(path.parent/'native-admission.json', {'plan': reference(path), 'static_proof': plan['static_proof'],
                  'exact_and_control_cpp_cuda_images': 8, 'status': 'study_only_admitted',
                  'historical_A_acceptance_changed': False,
                  'records': [reference(path.parent/f'{i:04d}-{mode}-{backend}.json')
                              for mode in ('exact', 'control') for backend in ('cpp', 'cuda') for i in range(8)]})
    if images >= 32:
        report_panel(path, images)
    invocation = {'plan': reference(path), 'images': images, 'wall_seconds': time.perf_counter()-started,
                  'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'finished_at': now()}
    immutable(path.parent/'invocations'/f'{time.time_ns()}.json', invocation)


def memory_available():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].split()[0])*1024


def measured_cost(path):
    records = [unseal(p) for p in path.parent.glob('[0-9][0-9][0-9][0-9]-*-cuda.json')]
    seconds = sum(unseal(p).get('new_compute_seconds', 0)
                  for p in path.parent.glob('[0-9][0-9][0-9][0-9]-*-*.json'))
    compatibility = path.parent/'prepared-reference-compatibility.json'
    if compatibility.exists():
        seconds += unseal(compatibility)['seconds']
    rate = {}
    for mode in ('exact', 'control'):
        mode_records = [r for r in records if r['mode'] == mode]
        rate[mode] = statistics.median(r['new_compute_seconds'] for r in mode_records) if mode_records else None
    return {'new_compute_seconds': seconds, 'median_seconds': rate,
            'max_rss_kib': max((r.get('peak_rss_kib', 0) for r in records), default=0)}


def run(stage):
    enrollment = freeze_study()
    state = {'status': 'starting', 'stage': stage, 'started_at': now(), 'pid': os.getpid(),
             'cases': [dict(r) for r in enrollment['cases']]}
    with exclusive(BASE/'controller.lock'), ExitStack() as leases:
        controller = leases.enter_context(exclusive(ROOT/'artifacts/phase3/controller/controller.lock'))
        native = leases.enter_context(pool_locks(ROOT))
        fds = (controller.fileno(), native['cuda'], native['cpu'])
        atomic_json(BASE/'status.json', state)
        paths = [checked(row['plan']) for row in enrollment['cases'] if 'plan' in row]
        def dispatch(path, images):
            log_path = path.parent/f'run-{images}.log'
            env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE'}
            with log_path.open('ab') as log:
                result = subprocess.run([sys.executable, '-m', 'tools.run.useful_quality', 'worker', '--plan', str(path),
                                         '--images', str(images), '--lease-fds', *map(str, fds)], cwd=ROOT,
                                        env=env, pass_fds=fds, stdout=log, stderr=subprocess.STDOUT)
            return {'case': unseal(path)['case'], 'images': images, 'returncode': result.returncode,
                    'log': str(log_path.relative_to(ROOT))}
        def batch(jobs, workers):
            results = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(dispatch, path, count) for path, count in jobs]
                for future in as_completed(futures):
                    result = future.result(); results.append(result)
                    state.setdefault('invocations', []).append(result)
                    atomic_json(BASE/'status.json', state)
            return results
        try:
            # Native pilots also provide per-arm throughput before any quality promotion.
            count = 1 if stage == 'probe' else 8
            state['status'] = 'timing_probe' if stage == 'probe' else 'native_pilots'
            atomic_json(BASE/'status.json', state)
            batch([(path, count) for path in paths if not (path.parent/'native-admission.json').exists()], 2)
            if stage == 'probe':
                state['status'] = 'probe_completed'
                return
            admitted = [path for path in paths if (path.parent/'native-admission.json').exists()]
            costs = {unseal(path)['case']: measured_cost(path) for path in admitted}
            available = memory_available()
            maximum_rss = max((r['max_rss_kib'] for r in costs.values()), default=0)*1024
            workers = 3 if available >= 12*2**30 and maximum_rss < 2*2**30 else (2 if available >= 6*2**30 else 1)
            measurement = {'pilot_costs': costs, 'available_memory_bytes': available, 'workers': workers,
                           'threads_per_worker': 4, 'measured_at': now()}
            immutable(BASE/'measurements'/f'{time.time_ns()}.json', measurement)
            state.update(status='panels_32', concurrency=workers, measurement=measurement)
            atomic_json(BASE/'status.json', state)
            batch([(path, 32) for path in admitted if not (path.parent/'summary-32.json').exists()], workers)
            decisions = []
            remaining = min(12*3600, 24*3600-sum(measured_cost(p)['new_compute_seconds'] for p in paths))
            extensions = []
            for path in paths:
                case = unseal(path)['case']
                if not (path.parent/'summary-32.json').exists():
                    decisions.append({'case': case, 'promote': False, 'reason': 'native admission or initial panel incomplete'})
                    continue
                result = unseal(path.parent/'summary-32.json')
                eligible, reason = promotion(result)
                rates = measured_cost(path)['median_seconds']
                estimate = 1.5*96*sum(rates.values())
                allowed = eligible and estimate <= remaining
                decisions.append({'case': case, 'quality_eligible': eligible, 'promote': allowed,
                                  'reason': reason if allowed or not eligible else 'measured extension exceeds remaining budget',
                                  'estimated_extension_seconds': estimate})
                if allowed:
                    extensions.append((path, 128)); remaining -= estimate
            decision_path = BASE/'promotion.json'
            # A resume preserves the original decisions, including allocated cost.
            if decision_path.exists():
                saved = unseal(decision_path)
                extensions = [(p, 128) for p in paths if any(d['case'] == unseal(p)['case'] and d['promote'] for d in saved['decisions'])]
            else:
                immutable(decision_path, {'protocol': enrollment['protocol'], 'decisions': decisions,
                                         'unallocated_extension_seconds': remaining})
            state.update(status='panels_128', promotion=reference(decision_path))
            atomic_json(BASE/'status.json', state)
            batch([(p, n) for p, n in extensions if not (p.parent/'summary-128.json').exists()], workers)
            state['status'] = 'completed_admitted_panels' if all((p.parent/'summary-32.json').exists() for p in paths) and all(
                (p.parent/'summary-128.json').exists() for p, _ in extensions) else 'partial_with_failures'
        except BaseException as error:
            state.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            state['updated_at'] = now()
            atomic_json(BASE/'status.json', state)


def status():
    result = read(BASE/'status.json') if (BASE/'status.json').exists() else {'status': 'not_started'}
    result['checkpoints'] = []
    if (BASE/'enrollment.json').exists():
        for row in unseal(BASE/'enrollment.json')['cases']:
            item = {'case': row['case'], 'static': row['status']}
            if 'plan' in row:
                path = checked(row['plan'])
                item['counts'] = {f'{m}/{b}': len(list(path.parent.glob(f'*-{m}-{b}.json')))
                                  for m in ('exact', 'control') for b in ('cpp', 'cuda')}
                item['native_admission'] = (path.parent/'native-admission.json').exists()
                item['panels'] = [n for n in (32, 128) if (path.parent/f'summary-{n}.json').exists()]
                if (path.parent/'progress.json').exists():
                    item['progress'] = read(path.parent/'progress.json')
            result['checkpoints'].append(item)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'probe', 'run', 'start', 'worker', 'status'))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--images', type=int, choices=(1, 8, 32, 128))
    parser.add_argument('--lease-fds', type=int, nargs=3)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(freeze_study(), indent=2))
    elif args.action == 'status':
        status()
    elif args.action == 'worker':
        worker(args.plan, args.images, args.lease_fds)
    elif args.action == 'start':
        freeze_study()
        with exclusive(BASE/'controller.lock'):
            pass
        with (BASE/'run.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, '-m', 'tools.run.useful_quality', 'run'], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': child.pid, 'log': str(BASE/'run.log')}))
    else:
        run(args.action)


if __name__ == '__main__':
    main()
