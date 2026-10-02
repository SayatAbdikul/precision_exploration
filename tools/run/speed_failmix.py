"""Lane S1, protocol speed-protocol-v1-addendum-1 (review 1, finding N5): real-image batches in which only some
images fail under a float accumulator.

  .venv-b/bin/python -m tools.run.speed_failmix scan   fast path, chooses images only: per case, fp16.x<e> for
                                                      e = rule+1, rule+2, ... on images 0-64 (batch 8) until all 64
                                                      fail; then images 0-1000 at every exponent with 0 < failed < 64
                                                      and at the two exponents around the 0 -> 64 jump. Picks per case
                                                      the exponent with the most mixed 8-aligned batches and the
                                                      8-aligned 32-image window with the most mixed 8-image batches.
                                                      -> artifacts/speed_v1/validation/failmix-picks.json (written once)
  .venv-b/bin/python -m tools.run.speed_failmix run    for every pick: the archive's own predict on the window
                                                      (tools.run.speed_archive_predict, redirected into
                                                      artifacts/speed_v1/archive-runs/), then the fast compare at
                                                      batches 1, 8 and 16 (tools.run.speed_engine_validate compare)
Identity is decided by the archive comparison only. GPU via artifacts/agent_orchestration/gpu_run.sh (GPU_LANE=S1).
Development evidence (ImageNet screen-1k).
"""
from __future__ import annotations
import json
import subprocess
import sys
import time

from tools.experiment_b.common import ROOT

SPEED = ROOT / 'artifacts/speed_v1'
PICKS = SPEED / 'validation' / 'failmix-picks.json'
PLAN = SPEED / 'validation' / 'plan.json'
R1 = '1f75c9232c8a0202482fe9bce9a46360c7cc0999c5b0bcf7df5ebcc0446fc863'
RN = '7c6344af732d1bf324c07b285921f63393150cdb3875356c86237d934fdc2f92'
PY = str(ROOT / '.venv-b/bin/python')
CASES = ('resnet18-int8-default-b2', 'mobilenet_v2-int8-default-b2', 'mobilenet_v3_large-int8-default-b2',
         'mobilenet_v3_large-fp7_e3m3-default-b2')
WINDOW, BATCH = 32, 8


def policy_of(e):
    return 'fp16' if e == 0 else f'fp16.x{e}'


def rule_exponent(name):
    return 0 if name == 'fp16' else int(name[len('fp16.x'):])


def failed_indices(case, e, start, stop):
    from tools.scaled_bridge_fast import worker
    ctx = worker.context(case, policy_of(e))
    items, _, _, timing = worker.execute(ctx, start, stop, BATCH)
    return [it['index'] for it in items if it['failure'] is not None], timing['wall_seconds']


def mixed_batches(failed, start, stop, size=BATCH):
    bad = set(failed)
    out = []
    for f in range(start, stop, size):
        n = sum(1 for i in range(f, min(stop, f + size)) if i in bad)
        if 0 < n < min(stop, f + size) - f:
            out.append(f)
    return out


def scan():
    if PICKS.exists():
        raise SystemExit(f'{PICKS} exists')
    plan = json.loads(PLAN.read_text())
    report = {}
    for case in CASES:
        rule = rule_exponent(plan[case]['policies']['fp16_rule'])
        small = []
        for e in range(rule + 1, rule + 17):
            failed, wall = failed_indices(case, e, 0, 64)
            small.append({'exponent': e, 'policy': policy_of(e), 'failed_of_64': len(failed), 'wall_seconds': round(wall, 2)})
            print(json.dumps({'case': case, **small[-1]}), flush=True)
            if len(failed) == 64:
                break
        partial = [r['exponent'] for r in small if 0 < r['failed_of_64'] < 64]
        none = [r['exponent'] for r in small if r['failed_of_64'] == 0]
        full = [r['exponent'] for r in small if r['failed_of_64'] == 64]
        wide = set(partial)
        if none:
            wide.add(max(none))
        if full:
            wide.add(min(full))
        large = []
        for e in sorted(wide):
            failed, wall = failed_indices(case, e, 0, 1000)
            mixed = mixed_batches(failed, 0, 1000)
            large.append({'exponent': e, 'policy': policy_of(e), 'failed_of_1000': len(failed), 'failed': failed,
                          'mixed_batches_of_8': len(mixed), 'wall_seconds': round(wall, 2)})
            print(json.dumps({'case': case, **{k: v for k, v in large[-1].items() if k != 'failed'}}), flush=True)
        pick = None
        best = max(large, key=lambda r: (r['mixed_batches_of_8'], -r['exponent']), default=None)
        if best and best['mixed_batches_of_8']:
            windows = []
            for s in range(0, 1000 - WINDOW + 1, BATCH):
                m = mixed_batches(best['failed'], s, s + WINDOW)
                nf = sum(1 for i in best['failed'] if s <= i < s + WINDOW)
                windows.append((len(m), s, nf))
            count, s, nf = max(windows, key=lambda w: (w[0], -w[1]))
            pick = {'policy': best['policy'], 'exponent': best['exponent'], 'start': s, 'stop': s + WINDOW,
                    'mixed_batches_of_8_in_window': count, 'failed_in_window': nf}
        report[case] = {'rule_policy': plan[case]['policies']['fp16_rule'], 'scan_0_64': small, 'scan_0_1000': large,
                        'pick': pick}
    PICKS.write_text(json.dumps({'protocol': 'speed-protocol-v1-addendum-1 (n5_mixed_failure_batches)',
                                 'images': 'imagenet_screen_1k (development)', 'batch': BATCH, 'window': WINDOW,
                                 'written': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'cases': report}, indent=1))
    print('SCAN-DONE', sum(1 for r in report.values() if r['pick']), 'picks', flush=True)


def run():
    picks = json.loads(PICKS.read_text())['cases']
    rc = 0
    for case, entry in sorted(picks.items()):
        pick = entry['pick']
        if pick is None:
            continue
        digest_ = RN if case.startswith('resnet18-') else R1
        start, stop, policy = str(pick['start']), str(pick['stop']), pick['policy']
        for cmd in ([PY, '-m', 'tools.run.speed_archive_predict', digest_, case, start, stop, '8', policy],
                    [PY, '-m', 'tools.run.speed_engine_validate', 'compare', case, policy, '--start', start, '--stop', stop,
                     '--batches', '1,8,16']):
            code = subprocess.run(cmd, cwd=ROOT).returncode
            print(json.dumps({'case': case, 'policy': policy, 'window': [pick['start'], pick['stop']], 'step': cmd[2],
                              'exit': code}), flush=True)
            rc |= code != 0
            if code:
                break
    print('RUN-DONE rc', int(rc), flush=True)
    return int(rc)


if __name__ == '__main__':
    if sys.argv[1] == 'scan':
        scan()
    else:
        sys.exit(run())
