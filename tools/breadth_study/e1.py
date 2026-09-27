"""Resumable E1 matched accumulation experiments; historical A stays immutable."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools.experiment_b.common import atomic_json, seal, unseal
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference
from tools.phase3.execution_guard import verify_guard
from tools.phase3.worker_locks import exclusive, pool_locks
from tools.run.exact_execution import immutable, implementation, archive_implementation, event_signature

BASE = ROOT / 'artifacts/breadth_study/e1_v1'
MATRIX = ROOT / 'public/experiments/configs/breadth-study/comparison-matrix-v1.json'
PROTOCOL = {
    'version': 'matched-sequential-fp32-reductions-v1',
    'changed_factor': 'FP32 reduction state; original sequential order and exact decoded operands',
    'integer_domain': 'original unscaled codes; original exact product scale restored after reduction',
    'MAC_boundary': 'FP32 final reduction state cast into original accumulator before unchanged bias/store',
    'non_MAC': 'residual sums, pooling sums, softmax denominator and DFL projection use FP32; other arithmetic unchanged',
    'invariants': ['graph', 'stored_codes', 'scales', 'quantizers', 'bias_policy', 'preprocessing', 'image_order'],
    'interpretation': 'isolates accumulation precision; not ordinary cuDNN QDQ and not a native low-bit throughput claim',
    'initial_images': 32, 'maximum_images': 128, 'control_native_pilot_images': 8,
    'bootstrap': {'draws': 5000, 'seed': 20260926, 'interval': .95},
}


def sources():
    paths = ['tools/breadth_study/e1.py', 'tools/breadth_study/matched_control.py', 'tools/run/e1.py']
    return {**implementation(), **{p: reference(ROOT / p)['sha256'] for p in paths}}


def prepare():
    from tools.phase3.preparation_inventory import preparation_records
    prepared = preparation_records()
    matrix = unseal(MATRIX)
    current = sources()
    archive = archive_implementation(current)
    inventory = []
    for row in matrix['E1_matched_arithmetic']:
        key = f"{row['model']}/{row['format']}"
        p = prepared[key]
        accepted = ROOT / 'artifacts/phase3/acceptance' / p['configuration_sha256'] / 'prepared-accepted.json'
        if not accepted.exists():
            inventory.append({'case': key, 'status': 'native_acceptance_required', 'prepared':
                              reference(checked(p['configuration']).parent / 'prepared.json')})
            continue
        strict = []
        for path in (ROOT / 'artifacts/phase3/runs').glob('*/summary.json'):
            summary = read(path)
            if (summary.get('scope') == 'screen' and summary.get('images') == 1000
                    and summary['configuration_sha256'] == p['configuration_sha256']):
                strict.append(path)
        if len(strict) != 1:
            raise ValueError('accepted initial E1 case needs its unique preserved exact-A screen')
        from fractions import Fraction
        from tools.phase3.acceptance import prove_integer_graph
        graph = read(checked(p['graph']))
        screen = read(strict[0])
        first = read(checked(screen['image_records'][0]))
        manifest = read(checked(campaign()['inputs'][row['model']]))
        shapes = {name: value['shape'] for name, value in first['backends']['cuda']['layers'].items()}
        integer_proof = prove_integer_graph(graph, {name: manifest['input_shape'] for name in graph['inputs']}, shapes)
        bounds = [Fraction(r['dot_bound']) for r in integer_proof['mac_bounds']]
        bounds += [Fraction(r['maximum_abs_sum_codes']) for r in integer_proof['other_sum_bounds']]
        equivalence = {'status': 'proven_exact_integer_FP32_reductions' if integer_proof['status'] == 'accepted'
                       and all(v <= 2**24 for v in bounds) else 'requires_empirical_comparison',
                       'largest_absolute_reduction_bound': str(max(bounds)), 'FP32_exact_integer_limit': 2**24,
                       'scope': 'unscaled integer FMA/sums only; original bias, division and output store retained'}
        plan = {'version': 'e1-matched-case-v1', 'case': key, 'model': row['model'], 'format': row['format'],
                'matrix': reference(MATRIX), 'protocol': PROTOCOL, 'graph': p['graph'],
                'configuration': p['configuration'], 'accepted_prepared': reference(accepted),
                'strict_screen': reference(strict[0]), 'guard_sha256': verify_guard(),
                'sources': current, 'source_archive': archive, 'integer_equivalence': equivalence}
        path = BASE / 'jobs' / digest(plan) / 'plan.json'
        immutable(path, plan)
        inventory.append({'case': key, 'status': 'ready', 'plan': reference(path)})
    result = {'matrix': reference(MATRIX), 'protocol': PROTOCOL, 'cases': inventory, 'sources': current}
    immutable(BASE / 'inventories' / f'{digest(result)}.json', result)
    return result


def validate_lease(fds):
    names = ['artifacts/phase3/controller/controller.lock', 'artifacts/phase3/locks/native-worker.lock',
             'artifacts/phase3/locks/cpu-native-worker.lock']
    if fds is None or len(fds) != 3:
        raise ValueError('E1 worker requires controller and native leases')
    for fd, name in zip(fds, names):
        actual, expected = os.fstat(fd), (ROOT / name).stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError('E1 worker has an unrelated resource lease')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def paired_metrics(rows):
    import numpy as np
    a = np.array([r['strict_top5'][0] == r['ground_truth'] for r in rows], dtype=float)
    b = np.array([r['control_top5'][0] == r['ground_truth'] for r in rows], dtype=float)
    rng = np.random.default_rng(PROTOCOL['bootstrap']['seed'])
    delta = b-a
    draws = delta[rng.integers(0, len(rows), (PROTOCOL['bootstrap']['draws'], len(rows)))].mean(axis=1)*100
    return {'images': len(rows), 'strict_top1_percent': float(a.mean()*100),
            'control_top1_percent': float(b.mean()*100), 'control_minus_strict_pp': float(delta.mean()*100),
            'paired_95_percent_interval_pp': np.quantile(draws, [.025, .975]).tolist(),
            'output_code_agreement_images': sum(r['output_codes_match'] for r in rows),
            'all_layer_code_agreement_images': sum(r['all_layers_match'] for r in rows),
            'interpretation': 'development paired pointwise interval; no final ranking'}


def worker(path, images, fds):
    validate_lease(fds)
    import torch
    from public.analysis.phase3.diagnostics import LayerDiagnostics, ReferenceSamples
    from public.inference.tensor import QUANTIZATION_OBSERVER, parse_encoding
    from public.quantization.ptq.encoding import float_tensor
    from tools.breadth_study.matched_control import MatchedControl, native_conformance
    from tools.exact_execution_v2.activation_trace import ActivationTrace
    from tools.phase3.acceptance import verify_acceptance
    from tools.phase3.baselines import screen_rows
    from tools.phase3.evidence import verify_complete
    from tools.phase3.runtime import load_runtime
    from tools.run.phase3_thread_benchmark import set_threads
    plan = unseal(path)
    if plan['sources'] != sources() or plan['guard_sha256'] != verify_guard():
        raise ValueError('E1 implementation or frozen dependencies changed')
    verify_acceptance(read(checked(plan['accepted_prepared'])))
    summary = read(checked(plan['strict_screen']))
    verify_complete(checked(plan['strict_screen']), current_execution=True)
    graph = read(checked(plan['graph']))
    population, payload, _ = screen_rows(plan['model'], campaign(), verify_images=False)
    sample, observe, manifest = load_runtime(plan['model'], campaign())
    torch.set_num_interop_threads(1)
    set_threads(4)
    input_name = next(iter(graph['inputs']))
    encoding = parse_encoding(graph['inputs'][input_name])
    identity = digest(plan)
    for backend, count in [('cpp', 8), ('cuda', images)]:
        immutable(path.parent / f'control-conformance-{backend}.json', native_conformance(backend))
        engine = MatchedControl(graph, backend=backend)
        with ActivationTrace() as trace:
            for index, row in enumerate(population[:count]):
                original = read(checked(summary['image_records'][index]))
                if original['sample'] != row or reference(payload / row['relative_path'])['sha256'] != row['sha256']:
                    raise ValueError('E1 paired image identity mismatch')
                destination = path.parent / f'{index:04d}-control-{backend}.json'
                if destination.exists():
                    record = unseal(destination)
                    if (record['job_sha256'] != identity or record['sample'] != row or record['backend'] != backend
                            or record['strict_record'] != summary['image_records'][index]):
                        raise ValueError('E1 checkpoint belongs to different numerical inputs')
                    trace.replay(record['activation_calls'])
                    continue
                fp32, _ = sample(payload / row['relative_path'])
                if list(fp32.shape) != manifest['input_shape']:
                    raise ValueError('E1 preprocessing changed')
                references = ReferenceSamples(campaign()['diagnostics']['sample_elements_per_layer_image'])
                observe(fp32, references)
                tensor = float_tensor(fp32.numpy(), encoding)
                diagnostics = LayerDiagnostics(references)
                trace.calls = []
                token = QUANTIZATION_OBSERVER.set(diagnostics.quantization)
                tick = time.perf_counter()
                try:
                    output = engine.execute({input_name: tensor}, observer=diagnostics)
                finally:
                    QUANTIZATION_OBSERVER.reset(token)
                seconds = time.perf_counter()-tick
                values = next(iter(output['outputs'].values()))
                scores = values.values()
                if not all(__import__('math').isfinite(float(v)) for v in scores):
                    raise ValueError('control produced nonfinite outputs')
                strict = original['backends']['cuda']
                record = {'job_sha256': identity, 'sample': row, 'backend': backend,
                          'strict_record': summary['image_records'][index], 'graph_sha256': output['graph_sha256'],
                          'output_sha256': digest(values.document()), 'layers': output['layers'],
                          'execution_modes': output['execution_modes'], 'diagnostics': diagnostics.records,
                          'diagnostic_signature': event_signature(diagnostics.records), 'seconds': seconds,
                          'activation_calls': trace.calls, 'ground_truth': original['paired']['ground_truth'],
                          'strict_top5': strict['prediction'],
                          'control_top5': sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:5],
                          'output_codes_match': digest(values.document()) == strict['output_sha256'],
                          'all_layers_match': output['layers'] == strict['layers']}
                if plan['integer_equivalence']['status'] == 'proven_exact_integer_FP32_reductions' and not record['all_layers_match']:
                    raise ValueError('control violates the proven integer/FP32 equivalence; investigate implementation')
                if backend == 'cuda' and index < 8:
                    cpu = unseal(path.parent / f'{index:04d}-control-cpp.json')
                    if any(record[k] != cpu[k] for k in ('layers', 'output_sha256', 'diagnostic_signature')):
                        raise ValueError('matched control C++/CUDA disagreement')
                if sources() != plan['sources'] or verify_guard() != plan['guard_sha256']:
                    raise ValueError('E1 execution dependencies changed')
                immutable(destination, record)
                print(f"{plan['case']} {backend} {index+1}/{count}: {seconds:.3f}s, exact output agreement={record['output_codes_match']}", flush=True)
    records = [unseal(path.parent / f'{i:04d}-control-cuda.json') for i in range(images)]
    # Recheck cross-backend checkpoints even if this invocation computed nothing.
    for i in range(8):
        cpu = unseal(path.parent / f'{i:04d}-control-cpp.json')
        if any(cpu[k] != records[i][k] for k in ('layers', 'output_sha256', 'diagnostic_signature')):
            raise ValueError('resumed E1 native control disagreement')
    report = {'plan': reference(path), 'images': images, 'native_control_images': 8, 'native_control_matches': True,
              'metrics': paired_metrics(records), 'scope': 'E1_development_matched_accumulation_only',
              'records': [reference(path.parent / f'{i:04d}-control-cuda.json') for i in range(images)]}
    immutable(path.parent / f'summary-{images}.json', report)


def run(images, wait):
    inventory = prepare()
    paths = [checked(r['plan']) for r in inventory['cases'] if r['status'] == 'ready']
    status = {'status': 'waiting_for_native_resources', 'pid': os.getpid(), 'images': images,
              'ready_cases': [r['case'] for r in inventory['cases'] if r['status'] == 'ready'],
              'blocked_cases': [r for r in inventory['cases'] if r['status'] != 'ready']}
    with exclusive(BASE / 'controller.lock'):
        atomic_json(BASE / 'status.json', status)
        while True:
            b_state = ROOT / 'artifacts/breadth_study/selected_b_v1/status.json'
            if wait and b_state.exists() and read(b_state)['status'] != 'completed':
                status['waiting_for'] = 'selected_B_extensions_to_complete'
                atomic_json(BASE / 'status.json', status)
                time.sleep(15)
                continue
            leases = ExitStack()
            try:
                controller = leases.enter_context(exclusive(ROOT / 'artifacts/phase3/controller/controller.lock'))
                native = leases.enter_context(pool_locks(ROOT))
                break
            except RuntimeError:
                leases.close()
                if not wait:
                    raise
                time.sleep(15)
        with leases:
            fds = (controller.fileno(), native['cuda'], native['cpu'])
            status['status'] = 'running'
            atomic_json(BASE / 'status.json', status)
            def dispatch(path):
                env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE'}
                subprocess.run([sys.executable, '-m', 'tools.run.e1', 'worker', '--plan', str(path), '--images', str(images),
                                '--lease-fds', *map(str, fds)], cwd=ROOT, env=env, pass_fds=fds, check=True)
                return reference(path.parent / f'summary-{images}.json')
            try:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    status['results'] = [f.result() for f in as_completed([pool.submit(dispatch, p) for p in paths])]
                status['status'] = 'ready_cases_completed; other_families_require_native_acceptance'
            except BaseException as error:
                status.update(status='failed', error=f'{type(error).__name__}: {error}')
                raise
            finally:
                atomic_json(BASE / 'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'start', 'worker', 'status'))
    parser.add_argument('--images', type=int, choices=(32, 128), default=32)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--lease-fds', type=int, nargs=3)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(), indent=2))
    elif args.action == 'worker':
        worker(args.plan, args.images, args.lease_fds)
    elif args.action == 'run':
        run(args.images, args.wait)
    elif args.action == 'status':
        print((BASE / 'status.json').read_text())
    else:
        prepare()
        with exclusive(BASE / 'controller.lock'):
            pass
        with (BASE / 'run.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, '-m', 'tools.run.e1', 'run', '--images', str(args.images), '--wait'],
                                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(json.dumps({'pid': child.pid, 'log': str(BASE / 'run.log')}))


if __name__ == '__main__':
    main()
