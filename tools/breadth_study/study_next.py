"""Queue admitted E1 family studies and E2 behind the initial matched controls."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.experiment_b.common import atomic_json, unseal
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference
from tools.phase3.execution_guard import verify_guard
from tools.phase3.worker_locks import exclusive, pool_locks
from tools.run.exact_execution import immutable
from tools.breadth_study.prospective_execution import create_plan

BASE = ROOT / 'artifacts/breadth_study/next_study_v1'
MATRIX = ROOT / 'public/experiments/configs/breadth-study/comparison-matrix-v1.json'


def pilot_task(model, fmt, prepared):
    from tools.phase3.screening import pipeline_identity
    config = read(checked(prepared['configuration']))
    return {'kind': 'pilot', 'model': model, 'format': fmt,
            'configuration_sha256': prepared['configuration_sha256'],
            'prepared': reference(checked(prepared['configuration']).parent / 'prepared.json'),
            'baseline': read(ROOT / 'results/summaries/phase3-baselines.json')['models'][model],
            'campaign_sha256': digest(campaign()), 'source_sha256': config['runtime']['source_sha256'],
            'pipeline_sha256': pipeline_identity(), 'dependencies': [], 'resource': 'cuda'}


def family_inventory():
    from tools.phase3.preparation_inventory import preparation_records
    from development.acceptance_proofs.family_static import verify_static
    from tools.breadth_study.family_admission import verify
    matrix, prepared = unseal(MATRIX), preparation_records()
    gate = {r['configuration']: r for r in read(ROOT / 'results/summaries/phase3-gate-inventory.json')['records']}
    rows = []
    for case in matrix['E1_matched_arithmetic']:
        key = f"{case['model']}/{case['format']}"
        if key in {'resnet18/int4', 'mobilenet_v2/int4'}:
            continue
        p = prepared[key]
        try:
            proof = verify_static(p)
        except ValueError as error:
            rows.append({'case': key, 'status': 'blocked_static_proof', 'reason': str(error),
                         'pending_mac_nodes': gate[key]['pending_mac_nodes'],
                         'pending_nonmac_nodes': gate[key]['pending_nonmac_nodes']})
            continue
        paired = []
        for path in (ROOT / 'artifacts/phase3/runs').glob('*/summary.json'):
            summary = read(path)
            if (summary['configuration_sha256'] == p['configuration_sha256'] and summary['scope'] == 'pilot'
                    and summary['images'] == 8 and summary['backend_equality']):
                paired.append(path)
        if len(paired) > 1:
            raise ValueError('ambiguous current paired native pilot')
        if not paired:
            rows.append({'case': key, 'status': 'static_passed_native_pilot_required',
                         'task': pilot_task(case['model'], case['format'], p)})
            continue
        pilot_ref = reference(paired[0])
        admission = verify(p, pilot_ref)
        admission_path = BASE / 'admissions' / f'{digest(admission)}.json'
        immutable(admission_path, admission)
        path = create_plan(model=case['model'], fmt=case['format'], graph=p['graph'], admission=reference(admission_path),
                           purpose='E1', arm='matched_arithmetic', strict_pilot=pilot_ref,
                           baseline=read(ROOT / 'results/summaries/phase3-baselines.json')['models'][case['model']])
        rows.append({'case': key, 'status': 'ready', 'plan': reference(path)})
    inventory = {'matrix': reference(MATRIX), 'cases': rows, 'guard_sha256': verify_guard(),
                 'orchestrator': reference(__file__)}
    immutable(BASE / 'inventories' / f'{digest(inventory)}.json', inventory)
    return inventory


def run_job(path, images, fds):
    environment = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                   'OPENBLAS_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE'}
    subprocess.run([sys.executable, '-m', 'tools.run.study_next', 'worker', '--plan', str(path),
                    '--images', str(images), '--lease-fds', *map(str, fds)],
                   cwd=ROOT, env=environment, pass_fds=fds, check=True)
    return reference(path.parent / f'summary-{images}.json')


def strict_screen(model, fmt):
    from tools.phase3.preparation_inventory import preparation_records
    identity = preparation_records()[f'{model}/{fmt}']['configuration_sha256']
    candidates = [p for p in (ROOT / 'artifacts/phase3/runs').glob('*/summary.json')
                  if (lambda s: s['configuration_sha256'] == identity and s['scope'] == 'screen' and s['images'] == 1000)(read(p))]
    if len(candidates) != 1:
        raise ValueError('E2 strict comparison needs a unique retained screen')
    return candidates[0]


def e2_comparison(model, fmt, jobs, images):
    from tools.breadth_study.e1 import paired_metrics
    from tools.phase3.evidence import verify_complete
    old_path = strict_screen(model, fmt)
    verify_complete(old_path, current_execution=True)
    old = read(old_path)
    strict = [read(checked(r)) for r in old['image_records'][:images]]
    comparisons = {}
    for arm, path in jobs.items():
        summary = unseal(path.parent / f'summary-{images}.json')
        candidate = [unseal(checked(r)) for r in summary['records']]
        if len(strict) != images or len(candidate) != images or summary['images'] != images:
            raise ValueError('E2 comparison population is incomplete')
        if any(a['sample'] != b['sample'] or a['paired']['ground_truth'] != b['ground_truth'] for a, b in zip(strict, candidate)):
            raise ValueError('E2 candidate/strict pairing mismatch')
        metrics = paired_metrics([{'ground_truth': b['ground_truth'], 'strict_top5': a['backends']['cuda']['prediction'],
             'control_top5': b['prediction'], 'output_codes_match': False, 'all_layers_match': False}
             for a, b in zip(strict, candidate)])
        for key in ('output_code_agreement_images', 'all_layer_code_agreement_images'):
            metrics.pop(key)
        metrics['recipe_minus_strict_pp'] = metrics.pop('control_minus_strict_pp')
        metrics['recipe_top1_percent'] = metrics.pop('control_top1_percent')
        comparisons[arm] = {'summary': reference(path.parent / f'summary-{images}.json'), 'metrics': metrics}
    result = {'model': model, 'format': fmt, 'images': images, 'strict_screen': reference(old_path),
              'comparisons': comparisons, 'interpretation': 'paired development recipe ablations, not confirmation'}
    output = BASE / 'E2' / f'{model}-{fmt}-{images}.json'
    immutable(output, result)
    return result


def fit_in_cuda_environment(fds, *, preflight=False):
    from tools.breadth_study.e2 import CASES
    from tools.breadth_study.cuda_fitting import prepare, finish
    results = []
    for model, fmt in CASES:
        path = prepare(model, fmt)
        subprocess.run([str(ROOT / '.venv-b/bin/python'), '-m', 'tools.run.study_next',
                        'fit-preflight' if preflight else 'fit', '--plan', str(path),
                        '--lease-fds', *map(str, fds)], cwd=ROOT, pass_fds=fds, check=True)
        if not preflight:
            results.append(finish(path))
    return results


def run_e2(fds, status, workers):
    from tools.breadth_study.e2 import CASES, clipping
    status['status'] = 'E2_recipe_fitting'
    atomic_json(BASE / 'status.json', status)
    prepared_recipes = fit_in_cuda_environment(fds)
    for model, fmt in CASES:
        prepared_recipes.append(clipping(model, fmt))
    jobs = {}
    for ref in prepared_recipes:
        p = unseal(checked(ref))
        path = create_plan(model=p['model'], fmt=p['format'], graph=p['graph'], admission=p['proof'],
                           purpose='E2', arm=p['arm'],
                           baseline=read(ROOT / 'results/summaries/phase3-baselines.json')['models'][p['model']])
        jobs.setdefault((p['model'], p['format']), {})[p['arm']] = path
    status['status'] = 'E2_exact_inference'
    status['E2_jobs'] = [reference(p) for arms in jobs.values() for p in arms.values()]
    atomic_json(BASE / 'status.json', status)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, path, 128, fds) for arms in jobs.values() for path in arms.values()]
        status['E2_results'] = [future.result() for future in as_completed(futures)]
    extend = []
    for (model, fmt), arms in jobs.items():
        comparison = e2_comparison(model, fmt, arms, 128)
        intervals = [r['metrics']['paired_95_percent_interval_pp'] for r in comparison['comparisons'].values()]
        if any(low <= 1 <= high for low, high in intervals):
            extend.append((model, fmt, arms))
    status['E2_extensions_to_256'] = [f'{m}/{f}' for m, f, _ in extend]
    atomic_json(BASE / 'status.json', status)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, path, 256, fds) for _, _, arms in extend for path in arms.values()]
        for future in as_completed(futures):
            future.result()
    for model, fmt, arms in extend:
        e2_comparison(model, fmt, arms, 256)


def run_stages(status):
    from tools.breadth_study.e1 import BASE as INITIAL
    atomic_json(BASE / 'status.json', status)
    while True:
        prior = read(INITIAL / 'status.json')
        if prior['status'].startswith('ready_cases_completed'):
            break
        if prior['status'] == 'failed':
            raise RuntimeError('initial E1 failed; fix it before downstream inference')
        time.sleep(15)
    with ExitStack() as stack:
        controller = stack.enter_context(exclusive(ROOT / 'artifacts/phase3/controller/controller.lock'))
        native = stack.enter_context(pool_locks(ROOT))
        fds = (controller.fileno(), native['cuda'], native['cpu'])
        status['status'] = 'E2_GPU_preflight'
        atomic_json(BASE / 'status.json', status)
        fit_in_cuda_environment(fds, preflight=True)
        status['status'] = 'E1_family_expansion'
        atomic_json(BASE / 'status.json', status)
        try:
            inventory = family_inventory()
            status['blocked_families'] = [r for r in inventory['cases'] if r['status'] == 'blocked_static_proof']
            status['family_jobs'] = inventory['cases']
            atomic_json(BASE / 'status.json', status)

            # Production pilot instrumentation is process-global: finish any
            # missing pilot before concurrent child jobs or recipe preparation.
            for row in inventory['cases']:
                if row['status'] == 'static_passed_native_pilot_required':
                    from tools.phase3.controller_worker import execute_task
                    execute_task(row['task'], native['cuda'])
            inventory = family_inventory()
            status['family_jobs'] = inventory['cases']
            ready = [checked(r['plan']) for r in inventory['cases'] if r['status'] == 'ready']
            complete = [p for p in ready if (p.parent / 'summary-32.json').exists()]
            pending = [p for p in ready if p not in complete]
            # Revalidate completed jobs without repeating their inferences.
            with ThreadPoolExecutor(max_workers=2) as pool:
                status['family_results'] = [f.result() for f in as_completed(
                    [pool.submit(run_job, p, 32, fds) for p in complete])]
            status['concurrency'] = {'E1_workers': min(1, len(pending)),
                                     'E2_workers': 3 if pending else 4, 'threads_per_worker': 4}
            atomic_json(BASE / 'status.json', status)
            with ThreadPoolExecutor(max_workers=1) as family_pool:
                families = [family_pool.submit(run_job, p, 32, fds) for p in pending]
                run_e2(fds, status, workers=3 if pending else 4)
                status['status'] = 'E2_completed_waiting_for_E1'
                atomic_json(BASE / 'status.json', status)
                status['family_results'].extend(f.result() for f in as_completed(families))
            status['status'] = 'admitted_E1_and_E2_completed; unresolved_family_proofs_remain'
        except BaseException as error:
            status.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            atomic_json(BASE / 'status.json', status)


def run():
    with exclusive(BASE / 'controller.lock'):
        status = {'pid': os.getpid(), 'status': 'waiting_for_initial_E1', 'matrix': reference(MATRIX)}
        try:
            run_stages(status)
        except BaseException as error:
            status.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            atomic_json(BASE / 'status.json', status)


def start():
    with exclusive(BASE / 'controller.lock'):
        pass
    with (BASE / 'run.log').open('ab') as log:
        child = subprocess.Popen([sys.executable, '-m', 'tools.run.study_next', 'run'], cwd=ROOT,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    print(json.dumps({'pid': child.pid, 'log': str(BASE / 'run.log')}), flush=True)
    from tools.breadth_study.ternary_extension import start as start_ternary
    start_ternary()


def controller_processes(pid):
    pending, seen, rows = [pid], set(), []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            directory = Path(f'/proc/{current}')
            fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            command = (directory / 'cmdline').read_bytes().decode().split('\0')
            for child_list in (directory / 'task').glob('*/children'):
                try:
                    pending.extend(map(int, child_list.read_text().split()))
                except FileNotFoundError:
                    pass
            rows.append({'pid': current, 'state': fields[0], 'threads': int(fields[17]),
                         'cpu_seconds': (int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),
                         'resident_mib': int(fields[21])*os.sysconf('SC_PAGE_SIZE')/2**20,
                         'entrypoint': command[2:4] if len(command)>3 and command[1]=='-m' else command[:1]})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return rows


def progress():
    """Compact read-only progress from the controller and sealed checkpoints."""
    state = read(BASE / 'status.json')
    result = {k: state[k] for k in ('pid', 'status', 'error', 'concurrency') if k in state}
    result['controller_processes'] = controller_processes(state['pid'])
    result['blocked_families'] = [r['case'] for r in state.get('blocked_families', [])]
    refs = [r['plan'] for r in state.get('family_jobs', []) if 'plan' in r]
    refs += state.get('E2_jobs', [])
    result['jobs'] = []
    for ref in refs:
        path = checked(ref)
        plan = unseal(path)
        counts = {f'{mode}_{backend}': len(list(path.parent.glob(f'[0-9]*-{mode}-{backend}.json')))
                  for mode in ('exact', 'control') for backend in ('cpp', 'cuda')}
        summaries = sorted(path.parent.glob('summary-*.json'))
        result['jobs'].append({'case': f"{plan['model']}/{plan['format']}", 'experiment': plan['purpose'],
                               'arm': plan['arm'], 'saved_images': counts,
                               'completed_panels': sorted(unseal(p)['images'] for p in summaries)})
    from tools.breadth_study.e2 import BASE as RECIPES
    result['rounding_fit'] = []
    for path in sorted((RECIPES / 'cuda_recipes').glob('*/plan.json')):
        plan = unseal(path)
        result['rounding_fit'].append({'case': plan['case'], 'layers': len(plan['layers']),
                                      'fitted_layers': len(list(path.parent.glob('layers/*/result.json')))})
    followup = ROOT / 'artifacts/breadth_study/ternary_e1_v1/status.json'
    if followup.exists():
        result['ternary_extension'] = read(followup)
        if 'proof' in result['ternary_extension']:
            checked(result['ternary_extension']['proof'])
            result['blocked_families'] = [name for name in result['blocked_families']
                                          if name != result['ternary_extension']['case']]
    return result


def restart_at_checkpoint(path):
    """Restart only this controller's process group after a saved image."""
    import signal
    status = read(BASE / 'status.json')
    pid = status['pid']
    def owned():
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if command[1:4] != [b'-m', b'tools.run.study_next', b'run'] or os.getpgid(pid) != pid:
            raise ValueError('recorded PID is not the detached study controller')
    owned()
    path = path.resolve()
    from tools.breadth_study.prospective_execution import BASE as JOBS
    if path.parent.parent != JOBS or path.name != 'plan.json':
        raise ValueError('checkpoint plan is outside the study namespace')
    plan = unseal(path)
    if plan['purpose'] != 'E1':
        raise ValueError('scheduler upgrade expects an active E1 checkpoint')
    before = set(path.parent.glob('[0-9]*-*.json'))
    if not before:
        raise ValueError('no completed checkpoint to protect')
    print(f'Waiting for the next saved image of {plan["model"]}/{plan["format"]}', flush=True)
    deadline = time.monotonic()+1800
    while set(path.parent.glob('[0-9]*-*.json')) == before:
        owned()
        if time.monotonic() >= deadline:
            raise TimeoutError('no new image checkpoint; controller was left running')
        time.sleep(2)
    owned()
    saved = [reference(p) for p in sorted(path.parent.glob('[0-9]*-*.json'))]
    journal = {'old_pid': pid, 'plan': reference(path), 'saved_records': saved,
               'reason': 'reserve one E1 worker and let E2 use three concurrent workers',
               'orchestrator': reference(__file__)}
    immutable(BASE / 'restarts' / f'{digest(journal)}.json', journal)
    os.killpg(pid, signal.SIGTERM)
    for _ in range(30):
        try:
            with ExitStack() as stack:
                stack.enter_context(exclusive(BASE / 'controller.lock'))
                stack.enter_context(exclusive(ROOT / 'artifacts/phase3/controller/controller.lock'))
                stack.enter_context(pool_locks(ROOT))
            break
        except RuntimeError:
            time.sleep(1)
    else:
        raise RuntimeError('prior workers still hold resource locks; no duplicate was launched')
    for ref in saved:
        checked(ref)
    start()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'start', 'worker', 'status', 'fit', 'fit-preflight', 'restart-at-checkpoint', 'ternary-start'))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--images', type=int, choices=(32, 128, 256), default=128)
    parser.add_argument('--lease-fds', type=int, nargs=3)
    args = parser.parse_args()
    if args.action in {'fit', 'fit-preflight'}:
        from tools.breadth_study.e1 import validate_lease
        validate_lease(args.lease_fds)
        from tools.breadth_study.cuda_fitting import fit
        fit(args.plan, first_only=args.action == 'fit-preflight')
    elif args.action == 'worker':
        from tools.breadth_study.prospective_execution import worker
        worker(args.plan, args.images, args.lease_fds)
    elif args.action == 'prepare':
        print(json.dumps(family_inventory(), indent=2))
    elif args.action == 'run':
        run()
    elif args.action == 'status':
        print(json.dumps(progress(), indent=2))
    elif args.action == 'restart-at-checkpoint':
        restart_at_checkpoint(args.plan)
    elif args.action == 'ternary-start':
        from tools.breadth_study.ternary_extension import start as start_ternary
        start_ternary()
    else:
        start()


if __name__ == '__main__':
    main()
