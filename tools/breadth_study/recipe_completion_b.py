"""Resume only the seven sealed percentile B counterparts, without retuning."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

from tools.experiment_b.common import ROOT, atomic_json, dataset, digest, file_hash, seal, unseal
from tools.phase3.worker_locks import exclusive
from tools.breadth_study.balanced_b import verify_one

BASE = ROOT / 'artifacts/breadth_study/recipe_completion_b_v1'
MATRIX = ROOT / 'public/experiments/configs/breadth-study/b-recipe-completion-v1.json'
SOURCES = (Path(__file__), ROOT/'tools/run/recipe_completion_b.py',
           ROOT/'tools/breadth_study/balanced_b.py')
COMPONENTS = {'original': 'experiment_b', 'extension': 'experiment_b_ext'}


def now():
    return datetime.now(timezone.utc).isoformat()


def modules():
    from tools.experiment_b import runner as original
    from tools.experiment_b_ext import runner as extension
    return {'original': original, 'extension': extension}


def check_prefix(group, task):
    root = ROOT/group['root']
    identity = task['configuration_sha256']
    config_path = root/'configurations'/f'{identity}.json'
    if file_hash(config_path) != task['configuration_file_sha256']:
        raise ValueError('frozen configuration file changed')
    prefix_path = root/'summaries'/f'{identity}-128.json'
    prefix = unseal(prefix_path)
    population = dataset('imagenet_screen_1k')[1][:128]
    records = [unseal(root/'predictions'/identity/(row['sha256']+'.json')) for row in population]
    if (file_hash(prefix_path) != task['summary_file_sha256_128'] or
            prefix['prediction_digest'] != task['prediction_digest_128'] or
            digest(records) != task['prediction_digest_128'] or
            any(record['sample'] != row or record['configuration_sha256'] != identity
                for record, row in zip(records, population))):
        raise ValueError('frozen 128-image prefix changed')


def prepare():
    matrix = unseal(MATRIX)
    selection = matrix['balance_extensions']
    if (matrix['version'] != 'b-recipe-completion-v1' or len(selection) != 7 or
            len({(r['model'], r['format'], r['recipe']) for r in selection}) != 7 or
            any(r['retained_images'] != 128 or r['target_images'] != 1000 or
                r['additional_images'] != 872 or r['recipe'] != 'percentile_99_9'
                for r in selection)):
        raise ValueError('unexpected seven-recipe enrollment')
    groups = {}
    for component, module in modules().items():
        root = ROOT/'artifacts'/COMPONENTS[component]
        source = module.source_identity()
        tasks = []
        for row in selection:
            if row['component'] != COMPONENTS[component]:
                continue
            identity = row['configuration_sha256']
            path = ROOT/row['configuration']['path']
            config = unseal(path)
            key = row['model'], row['format'], row['recipe']
            if (path != root/'configurations'/f'{identity}.json' or
                    file_hash(path) != row['configuration']['sha256'] or digest(config) != identity or
                    config['source_sha256'] != source or
                    (config['model_context']['model'], config['format'], config['recipe']) != key):
                raise ValueError(f'frozen B identity drift: {key}')
            prefix_path = root/'summaries'/f'{identity}-128.json'
            if (ROOT/row['retained_summary']['path'] != prefix_path or
                    file_hash(prefix_path) != row['retained_summary']['sha256'] or
                    file_hash(ROOT/row['paired_maxabs_summary']['path']) != row['paired_maxabs_summary']['sha256']):
                raise ValueError(f'frozen enrollment summary changed: {key}')
            prefix = unseal(prefix_path)
            if prefix['configuration_sha256'] != identity or prefix['panel_images'] != 128:
                raise ValueError(f'invalid retained B prefix: {key}')
            task = {'model': key[0], 'format': key[1], 'recipe': key[2],
                    'format_sha256': config['format_sha256'], 'configuration_sha256': identity,
                    'configuration_file_sha256': file_hash(path),
                    'prediction_digest_128': prefix['prediction_digest'],
                    'summary_file_sha256_128': file_hash(prefix_path),
                    'baseline_sha256': config['baseline_sha256'], 'blocked_reason': None}
            check_prefix({'root': str(root.relative_to(ROOT))}, task)
            tasks.append(task)
        groups[component] = {'root': str(root.relative_to(ROOT)), 'source_sha256': source, 'tasks': tasks}
    if sum(len(group['tasks']) for group in groups.values()) != 7:
        raise ValueError('seven unchanged configurations were not found')
    plan = {'version': 'b-recipe-completion-v1', 'matrix_file_sha256': file_hash(MATRIX),
            'sources': {str(p.relative_to(ROOT)): file_hash(p) for p in SOURCES},
            'groups': groups, 'target_images': 1000, 'additional_candidate_images': 6104}
    path = BASE/'plan.json'
    if path.exists():
        if unseal(path) != plan:
            raise ValueError('frozen recipe completion plan changed; use a new version')
    else:
        for source_path, identity in plan['sources'].items():
            archive = BASE/'source'/f'{identity}-{Path(source_path).name}'
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes((ROOT/source_path).read_bytes())
        seal(path, plan)
    check_plan(plan)
    return plan


def check_plan(plan):
    if file_hash(MATRIX) != plan['matrix_file_sha256']:
        raise ValueError('frozen matrix changed during run')
    for path, identity in plan['sources'].items():
        archive = BASE/'source'/f'{identity}-{Path(path).name}'
        if file_hash(ROOT/path) != identity or file_hash(archive) != identity:
            raise ValueError('controller/helper source or source archive changed')
    for component, module in modules().items():
        if module.source_identity() != plan['groups'][component]['source_sha256']:
            raise ValueError('original numerical runner source changed')


@contextmanager
def guard(module, original, group, component, plan):
    """Guard both extension writes and the original evaluator it reuses."""
    root = (ROOT/group['root']).resolve()
    candidates = {task['configuration_sha256'] for task in group['tasks']}
    baselines = {task['baseline_sha256'] for task in group['tasks']}
    samples = {row['sha256'] for row in dataset('imagenet_screen_1k')[1]}
    accepted = set()
    patched = list({id(value): value for value in (module, original)}.values())
    saved = [(value, value.seal, value.atomic_json, value.BASE) for value in patched]
    writer = module.seal
    status_writer = module.atomic_json

    def guarded_seal(path, payload):
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise ValueError('write outside enrolled component root')
        relative = path.relative_to(root)
        if relative.parts[0] == 'configurations':
            if (len(relative.parts) != 2 or path.stem not in candidates or
                    digest(payload) != path.stem or not path.exists() or unseal(path) != payload):
                raise ValueError('refusing a new or changed B configuration before candidate evaluation')
            accepted.add(path.stem)
            return
        if relative.parts[0] == 'predictions':
            if (len(relative.parts) != 3 or relative.parts[1] not in candidates | baselines or
                    path.stem not in samples or payload['configuration_sha256'] != relative.parts[1] or
                    payload['sample']['sha256'] != path.stem):
                raise ValueError('prediction outside frozen enrollment')
            if relative.parts[1] in candidates and relative.parts[1] not in accepted:
                raise ValueError('candidate configuration must pass identity guard before any prediction write')
        elif relative.parts[0] == 'summaries':
            if path.stem not in {f'{identity}-1000' for identity in candidates}:
                raise ValueError('summary outside pending enrollment')
        elif relative.parts[0] not in ('calibration', 'validation'):
            raise ValueError('unexpected runner evidence path')
        if path.exists():
            previous = unseal(path)
            if previous != payload:
                raise ValueError(f'refusing to rewrite existing evidence: {path}')
            return
        if relative.parts[0] == 'calibration':
            raise ValueError('run-only campaign requires existing calibration')
        writer(path, payload)

    def redirected_atomic(path, payload):
        check_plan(plan)
        path = Path(path).resolve()
        if path != root/'status.json':
            raise ValueError('unexpected direct runner status write')
        status_writer(BASE/f'{component}-status.json', payload)

    try:
        for value in patched:
            value.seal, value.atomic_json, value.BASE = guarded_seal, redirected_atomic, root
        yield
    finally:
        for value, previous_seal, previous_atomic, previous_base in saved:
            value.seal, value.atomic_json, value.BASE = previous_seal, previous_atomic, previous_base


def completed(plan):
    return [(component, task) for component, group in plan['groups'].items() for task in group['tasks']
            if (ROOT/group['root']/'summaries'/f"{task['configuration_sha256']}-1000.json").exists()]


def verify(plan):
    check_plan(plan)
    for group in plan['groups'].values():
        for task in group['tasks']:
            check_prefix(group, task)
    for component, task in completed(plan):
        verify_one(plan['groups'][component], task)
    return len(completed(plan))


def run(limit=None, max_run_seconds=3600):
    with exclusive(BASE/'controller.lock'):
        plan = prepare()
        state = {'pid': os.getpid(), 'status': 'running', 'started_at': now(),
                 'plan_sha256': digest(plan), 'selected_configurations': 7,
                 'completed_before': verify(plan), 'max_run_seconds': max_run_seconds}
        atomic_json(BASE/'status.json', state)
        started = time.monotonic()
        stop_reason = None

        def stop(signum, _frame):
            nonlocal stop_reason
            stop_reason = 'wall_time_budget' if signum == signal.SIGALRM else 'user_stop'
            raise InterruptedError(f'{stop_reason}; atomic image checkpoints retained')

        previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
        signal.setitimer(signal.ITIMER_REAL, max_run_seconds)
        try:
            remaining = limit
            runners = modules()
            for component, module in runners.items():
                check_plan(plan)
                group = plan['groups'][component]
                pending = [task for task in group['tasks'] if not
                           (ROOT/group['root']/'summaries'/f"{task['configuration_sha256']}-1000.json").exists()]
                if remaining is not None:
                    pending = pending[:remaining]
                if not pending:
                    continue
                active_group = {**group, 'tasks': pending}
                state.update(component=component, pending_this_component=len(pending), updated_at=now())
                atomic_json(BASE/'status.json', state)
                inventory = module.inventory()
                if inventory['source_sha256'] != group['source_sha256']:
                    raise ValueError('inventory source identity mismatch')
                inventory['configurations' if component == 'original' else 'tasks'] = pending
                args = SimpleNamespace(device='cuda', images=1000, models=None, formats=None, limit=None)
                with guard(module, runners['original'], active_group, component, plan):
                    result = module.execute(args, inventory)
                if result:
                    raise InterruptedError(f'{component} stopped with exit code {result}')
                if remaining is not None:
                    remaining -= len(pending)
            count = verify(plan)
            state.update(status='completed' if count == 7 else 'partial', completed_configurations=count)
        except (KeyboardInterrupt, InterruptedError) as error:
            state.update(status='budget_limited' if stop_reason == 'wall_time_budget' else 'interrupted',
                         stop_reason=stop_reason or 'runner_interrupted', error=f'{type(error).__name__}: {error}')
        except BaseException as error:
            state.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            state.update(updated_at=now(), elapsed_seconds=time.monotonic()-started,
                         completed_configurations=len(completed(plan)))
            atomic_json(BASE/'status.json', state)
        print(json.dumps(state, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'status', 'verify'))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-run-seconds', type=float, default=3600)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if not 0 < args.max_run_seconds <= 3600:
        parser.error('--max-run-seconds must be positive and at most the authorized 3600-second budget')
    if args.action == 'run':
        run(args.limit, args.max_run_seconds)
        return
    plan = prepare()
    if args.action == 'status':
        state_path = BASE/'status.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'status': 'prepared'}
        state['completed_configurations'] = len(completed(plan))
        for component in plan['groups']:
            path = BASE/f'{component}-status.json'
            if path.exists():
                current = json.loads(path.read_text())
                state[f'{component}_progress'] = current.get('progress')
                state[f'{component}_counts'] = current.get('counts')
        print(json.dumps(state, indent=2))
    else:
        count = verify(plan)
        print(json.dumps({'plan_sha256': digest(plan), 'selected_configurations': 7,
                          'completed_configurations': count, 'additional_candidate_images': 6104,
                          'verified_retained_prefixes': 7, 'status': 'verified'}))


if __name__ == '__main__':
    main()
