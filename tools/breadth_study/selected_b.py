"""Select the frozen B extensions without changing their numerical runners."""
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

BASE = ROOT / 'artifacts/breadth_study/selected_b_v1'
MATRIX = ROOT / 'public/experiments/configs/breadth-study/comparison-matrix-v1.json'


def selected_keys(matrix):
    keys = [(r['model'], r['format'], recipe)
            for r in matrix['E3_B_breadth_extensions'] for recipe in r['recipes']]
    if len(keys) != len(set(keys)) or len(keys) != 171 or len({k[:2] for k in keys}) != 100:
        raise ValueError('frozen enrollment must contain 171 recipes and all 100 model/datatype pairs')
    if any(r['target_images'] != 1000 or r['additional_images_per_unchanged_configuration'] != 872
           for r in matrix['E3_B_breadth_extensions']):
        raise ValueError('unsupported frozen image enrollment')
    return set(keys)


def prepare():
    from tools.experiment_b import runner as original
    from tools.experiment_b_ext import runner as extension
    selected = selected_keys(unseal(MATRIX))
    groups = {}
    panels = {model: dataset('coco_screen_1k' if model == 'yolov8n' else 'imagenet_screen_1k')[1][:128]
              for model in ('resnet18', 'mobilenet_v2', 'mobilenet_v3_large', 'yolov8n')}
    found = set()
    for component, module in (('original', original), ('extension', extension)):
        source = module.source_identity()
        root = ROOT / ('artifacts/experiment_b' if component == 'original' else 'artifacts/experiment_b_ext')
        state = json.loads((root / 'status.json').read_text())
        tasks = []
        for task in state['tasks']:
            key = task['model'], task['format'], task['recipe']
            if key not in selected:
                continue
            if component == 'original' and task.get('blocked_reason'):
                continue  # These rows are owned by the separately validated extension.
            if key in found or task['status'] != 'completed':
                raise ValueError('missing or duplicate completed B configuration')
            identity = task['configuration_sha256']
            path = root / 'configurations' / f'{identity}.json'
            config = unseal(path)
            if (digest(config) != identity or config['source_sha256'] != source
                    or (config['model_context']['model'], config['format'], config['recipe']) != key):
                raise ValueError('selected B configuration identity drift')
            summary = unseal(root / 'summaries' / f'{identity}-128.json')
            rows = panels[key[0]]
            records = [unseal(root / 'predictions' / identity / (r['sha256'] + '.json')) for r in rows]
            if (summary['panel_images'] != 128 or summary['configuration_sha256'] != identity
                    or digest(records) != summary['prediction_digest']
                    or any(p['sample'] != r or p['configuration_sha256'] != identity for p, r in zip(records, rows))):
                raise ValueError('saved 128-image evidence does not match frozen selection')
            tasks.append({'model': key[0], 'format': key[1], 'recipe': key[2],
                          'configuration_sha256': identity, 'configuration_file_sha256': file_hash(path),
                          'prediction_digest_128': summary['prediction_digest'],
                          'format_sha256': config['format_sha256'], 'blocked_reason': None})
            found.add(key)
        groups[component] = {'source_sha256': source, 'tasks': tasks, 'root': str(root.relative_to(ROOT))}
    if found != selected:
        raise ValueError('selected configuration missing from historical B ledger')
    plan = {'version': 'selected-b-extension-v1', 'matrix_file_sha256': file_hash(MATRIX),
            'selector_source_sha256': file_hash(__file__), 'groups': groups, 'target_images': 1000}
    path = BASE / 'plan.json'
    if path.exists() and unseal(path) != plan:
        raise ValueError('selected B plan changed; use a new version instead of overwriting')
    if not path.exists():
        seal(path, plan)
    return plan


def verify_configuration(path, payload, allowed):
    identity = digest(payload)
    if identity not in allowed or path.stem != identity or not path.exists() or unseal(path) != payload:
        raise ValueError('refusing a new or changed configuration in a historical B extension')


@contextmanager
def selected_orchestration(module, group, component, plan):
    """Intercept only evidence writes, keeping original numerical functions intact."""
    old_seal, old_atomic = module.seal, module.atomic_json
    allowed = {t['configuration_sha256'] for t in group['tasks']}
    def guarded_seal(path, payload):
        path = Path(path)
        if path.parent.name == 'configurations':
            verify_configuration(path, payload, allowed)
            return  # Historical configuration bytes remain unchanged.
        if path.parent.name == 'summaries' and path.exists():
            previous = unseal(path)
            if {k: v for k, v in previous.items() if k != 'timing'} != {k: v for k, v in payload.items() if k != 'timing'}:
                raise ValueError('completed B summary changed on resume')
            return  # Retain the first complete timing measurement.
        old_seal(path, payload)
    def redirect_state(path, payload):
        if file_hash(__file__) != plan['selector_source_sha256'] or file_hash(MATRIX) != plan['matrix_file_sha256']:
            raise ValueError('selected B orchestration changed during execution')
        path = Path(path)
        if path.name == 'status.json':
            path = BASE / f'{component}-status.json'
        old_atomic(path, payload)
    module.seal, module.atomic_json = guarded_seal, redirect_state
    try:
        yield
    finally:
        module.seal, module.atomic_json = old_seal, old_atomic


def run():
    from tools.experiment_b import runner as original
    from tools.experiment_b_ext import runner as extension
    with exclusive(BASE / 'controller.lock'):
        plan = prepare()
        state = {'pid': os.getpid(), 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
                 'plan_sha256': digest(plan), 'selected_configurations': 171}
        atomic_json(BASE / 'status.json', state)
        args = SimpleNamespace(device='cuda', images=1000, models=None, formats=None, limit=None)
        try:
            for component, module in (('original', original), ('extension', extension)):
                group = plan['groups'][component]
                state['component'] = component
                atomic_json(BASE / 'status.json', state)
                if component == 'original':
                    original.BASE = ROOT / group['root']
                inventory = module.inventory()
                inventory['configurations' if component == 'original' else 'tasks'] = group['tasks']
                with selected_orchestration(module, group, component, plan):
                    result = module.execute(args, inventory)
                if result:
                    raise InterruptedError(f'{component} stopped with exit code {result}')
            # Validate all result seals before declaring the selected campaign done.
            for group in plan['groups'].values():
                for task in group['tasks']:
                    identity = task['configuration_sha256']
                    summary = unseal(ROOT / group['root'] / 'summaries' / f'{identity}-1000.json')
                    if summary['panel_images'] != 1000 or summary['configuration_sha256'] != identity:
                        raise ValueError('selected extension summary mismatch')
                    config = unseal(ROOT / group['root'] / 'configurations' / f'{identity}.json')
                    population = dataset('coco_screen_1k' if task['model'] == 'yolov8n' else 'imagenet_screen_1k')[1]
                    for prediction_id, digest_key in ((identity, 'prediction_digest'),
                                                     (config['baseline_sha256'], 'baseline_prediction_digest')):
                        records = [unseal(ROOT / group['root'] / 'predictions' / prediction_id / (r['sha256'] + '.json'))
                                   for r in population]
                        if (digest(records) != summary[digest_key] or any(p['sample'] != r or
                                p['configuration_sha256'] != prediction_id for p, r in zip(records, population))):
                            raise ValueError('complete B summary/prediction population mismatch')
            state['status'] = 'completed'
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'failed',
                         error=f'{type(error).__name__}: {error}')
            raise
        finally:
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            atomic_json(BASE / 'status.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'start', 'status'))
    args = parser.parse_args()
    if args.action == 'status':
        for path in sorted(BASE.glob('*status.json')):
            state = json.loads(path.read_text())
            print(json.dumps({'file': path.name, **{k: v for k, v in state.items()
                  if k not in ('tasks', 'quantizer_validation', 'shared_validation', 'runtime')}}, indent=2))
    elif args.action == 'prepare':
        plan = prepare()
        print(json.dumps({'plan_sha256': digest(plan), 'configurations':
              {k: len(v['tasks']) for k, v in plan['groups'].items()}, 'additional_candidate_images': 149112}))
    elif args.action == 'start':
        prepare()
        # The child acquires the same lock before doing work; duplicate starts fail closed.
        with exclusive(BASE / 'controller.lock'):
            pass
        with (BASE / 'run.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, '-m', 'tools.run.selected_b', 'run'], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(json.dumps({'pid': child.pid, 'log': str(BASE / 'run.log')}))
    else:
        def stop(_signal, _frame):
            raise InterruptedError('stop requested; image checkpoints retained')
        signal.signal(signal.SIGTERM, stop)
        run()


if __name__ == '__main__':
    main()
