"""Resume the twelve frozen B comparison counterparts to 1,000 images.

The numerical runners and their original configuration identities are unchanged.
This controller only selects, verifies, and resumes the frozen enrollment.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

from tools.experiment_b.common import ROOT, atomic_json, dataset, digest, file_hash, seal, unseal
from tools.phase3.worker_locks import exclusive

BASE = ROOT / 'artifacts/breadth_study/balanced_b_v1'
MATRIX = ROOT / 'public/experiments/configs/breadth-study/b-deeper-comparisons-v1.json'
COMPONENTS = {'experiment_b': 'original', 'experiment_b_ext': 'extension'}


def now():
    return datetime.now(timezone.utc).isoformat()


def modules():
    from tools.experiment_b import runner as original
    from tools.experiment_b_ext import runner as extension
    return {'original': original, 'extension': extension}


def prepare():
    selection = unseal(MATRIX)['balance_extensions']
    if (len(selection) != 12 or len({(r['model'], r['format'], r['recipe']) for r in selection}) != 12
            or any(r['retained_images'] != 128 or r['target_images'] != 1000 or r['additional_images'] != 872
                   for r in selection)):
        raise ValueError('unexpected balance enrollment')
    groups = {}
    for component, module in modules().items():
        root_name = 'experiment_b' if component == 'original' else 'experiment_b_ext'
        root = ROOT / 'artifacts' / root_name
        source = module.source_identity()
        historical = {(r['model'], r['format'], r['recipe']): r
                      for r in json.loads((root/'status.json').read_text())['tasks'] if r['status'] == 'completed'}
        tasks = []
        for row in selection:
            if row['component'] != root_name:
                continue
            key = row['model'], row['format'], row['recipe']
            task = historical[key]
            identity = row['configuration_sha256']
            path = ROOT / row['configuration']['path']
            config = unseal(path)
            if (task['configuration_sha256'] != identity or path != root/'configurations'/f'{identity}.json'
                    or file_hash(path) != row['configuration']['sha256'] or digest(config) != identity
                    or config['source_sha256'] != source
                    or (config['model_context']['model'], config['format'], config['recipe']) != key):
                raise ValueError(f'historical B configuration drift: {key}')
            prefix = unseal(root/'summaries'/f'{identity}-128.json')
            population = dataset('imagenet_screen_1k')[1][:128]
            records = [unseal(root/'predictions'/identity/(sample['sha256']+'.json')) for sample in population]
            if (prefix['configuration_sha256'] != identity or prefix['panel_images'] != 128
                    or digest(records) != prefix['prediction_digest']
                    or any(record['sample'] != sample or record['configuration_sha256'] != identity
                           for record, sample in zip(records, population))):
                raise ValueError(f'historical B prediction drift: {key}')
            tasks.append({'model': key[0], 'format': key[1], 'recipe': key[2],
                          'format_sha256': config['format_sha256'], 'configuration_sha256': identity,
                          'configuration_file_sha256': file_hash(path),
                          'prediction_digest_128': prefix['prediction_digest'],
                          'blocked_reason': None})
        groups[component] = {'root': str(root.relative_to(ROOT)), 'source_sha256': source, 'tasks': tasks}
    if sum(len(g['tasks']) for g in groups.values()) != 12:
        raise ValueError('twelve unchanged configurations were not found')
    plan = {'version': 'balanced-b-v1', 'matrix_file_sha256': file_hash(MATRIX),
            'controller_source_sha256': file_hash(__file__), 'groups': groups,
            'target_images': 1000, 'additional_candidate_images': 10464}
    path = BASE/'plan.json'
    if path.exists():
        if unseal(path) != plan:
            raise ValueError('frozen balanced B plan changed; use a new version')
    else:
        seal(path, plan)
        archive = BASE/'source'/f"{plan['controller_source_sha256']}.py"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(Path(__file__).read_bytes())
    return plan


def check_plan(plan):
    if file_hash(MATRIX) != plan['matrix_file_sha256'] or file_hash(__file__) != plan['controller_source_sha256']:
        raise ValueError('frozen selection or controller source changed during run')


@contextmanager
def guard(module, group, component, plan):
    old_seal, old_atomic = module.seal, module.atomic_json
    allowed = {task['configuration_sha256'] for task in group['tasks']}

    def guarded_seal(path, payload):
        path = Path(path)
        if path.parent.name == 'configurations':
            if path.stem not in allowed or digest(payload) != path.stem or unseal(path) != payload:
                raise ValueError('refusing a new or changed B configuration')
            return
        if path.parent.name == 'summaries' and path.exists():
            previous = unseal(path)
            if {k: v for k, v in previous.items() if k != 'timing'} != {k: v for k, v in payload.items() if k != 'timing'}:
                raise ValueError('completed B summary changed on resume')
            return
        old_seal(path, payload)

    def redirected_atomic(path, payload):
        check_plan(plan)
        path = Path(path)
        if path.name == 'status.json':
            path = BASE/f'{component}-status.json'
        old_atomic(path, payload)

    module.seal, module.atomic_json = guarded_seal, redirected_atomic
    try:
        yield
    finally:
        module.seal, module.atomic_json = old_seal, old_atomic


def verify_one(group, task):
    root = ROOT/group['root']
    identity = task['configuration_sha256']
    summary = unseal(root/'summaries'/f'{identity}-1000.json')
    config = unseal(root/'configurations'/f'{identity}.json')
    population = dataset('imagenet_screen_1k')[1]
    if (summary['panel_images'] != 1000 or summary['configuration_sha256'] != identity
            or file_hash(root/'configurations'/f'{identity}.json') != task['configuration_file_sha256']):
        raise ValueError('balanced B summary/configuration mismatch')
    for prediction_id, field in ((identity, 'prediction_digest'),
                                 (config['baseline_sha256'], 'baseline_prediction_digest')):
        records = [unseal(root/'predictions'/prediction_id/(sample['sha256']+'.json')) for sample in population]
        if (digest(records) != summary[field] or any(record['sample'] != sample or
                record['configuration_sha256'] != prediction_id for record, sample in zip(records, population))):
            raise ValueError('balanced B prediction digest or population mismatch')
    prefix = unseal(root/'summaries'/f'{identity}-128.json')
    candidate_prefix = [unseal(root/'predictions'/identity/(sample['sha256']+'.json')) for sample in population[:128]]
    if digest(candidate_prefix) != prefix['prediction_digest']:
        raise ValueError('historical B prefix changed')


def completed(plan):
    return [(component, task) for component, group in plan['groups'].items() for task in group['tasks']
            if (ROOT/group['root']/'summaries'/f"{task['configuration_sha256']}-1000.json").exists()]


def run(limit=None):
    with exclusive(BASE/'controller.lock'):
        plan = prepare()
        state = {'pid': os.getpid(), 'status': 'running', 'started_at': now(),
                 'plan_sha256': digest(plan), 'selected_configurations': 12,
                 'completed_before': len(completed(plan))}
        atomic_json(BASE/'status.json', state)
        remaining_budget = limit
        try:
            for component, module in modules().items():
                check_plan(plan)
                group = plan['groups'][component]
                pending = [task for task in group['tasks'] if not
                           (ROOT/group['root']/'summaries'/f"{task['configuration_sha256']}-1000.json").exists()]
                if remaining_budget is not None:
                    pending = pending[:remaining_budget]
                if not pending:
                    continue
                state.update(component=component, pending_this_component=len(pending), updated_at=now())
                atomic_json(BASE/'status.json', state)
                module.BASE = ROOT/group['root']
                if component == 'extension':
                    from tools.experiment_b import runner as original
                    original.BASE = module.BASE
                inventory = module.inventory()
                inventory['configurations' if component == 'original' else 'tasks'] = pending
                args = SimpleNamespace(device='cuda', images=1000, models=None, formats=None, limit=None)
                with guard(module, group, component, plan):
                    result = module.execute(args, inventory)
                if result:
                    raise InterruptedError(f'{component} stopped with exit code {result}')
                if remaining_budget is not None:
                    remaining_budget -= len(pending)
            for component, task in completed(plan):
                verify_one(plan['groups'][component], task)
            state.update(status='completed' if len(completed(plan)) == 12 else 'partial',
                         completed_configurations=len(completed(plan)))
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'failed',
                         error=f'{type(error).__name__}: {error}')
            raise
        finally:
            state['updated_at'] = now()
            atomic_json(BASE/'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'start', 'status'))
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.action == 'prepare':
        plan = prepare()
        print(json.dumps({'plan_sha256': digest(plan), 'groups':
              {k: len(v['tasks']) for k, v in plan['groups'].items()}, 'additional_candidate_images': 10464}))
    elif args.action == 'status':
        plan = prepare()
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
    elif args.action == 'start':
        prepare()
        with exclusive(BASE/'controller.lock'):
            pass
        command = [sys.executable, '-m', 'tools.run.balanced_b', 'run']
        if args.limit is not None:
            command += ['--limit', str(args.limit)]
        with (BASE/'run.log').open('ab') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': child.pid, 'log': str(BASE/'run.log')}))
    else:
        def stop(_signal, _frame):
            raise InterruptedError('stop requested; image checkpoints retained')
        signal.signal(signal.SIGTERM, stop)
        run(args.limit)


if __name__ == '__main__':
    main()
