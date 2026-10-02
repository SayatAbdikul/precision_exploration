"""Lane S1 part 1c(ii), MobileNet sat.w at a width WITH events (added r3).

The planned MobileNet widths (network structural width - 4) gave no saturation event on the first 64 images, and
structural - 10 collapses (Top-1 0). This tool picks, per MobileNet case, a global width in between that has events
and then compares the fast path with a fresh archive run at that width.

  .venv-b/bin/python -m tools.run.speed_satw scan            fast path, images 0-64, batch 8, widths structural-5..-9 of
                                                             every MobileNet case; picks the widest width with events on
                                                             >= 1 image (Top-1 and event counts recorded) ->
                                                             artifacts/speed_v1/validation/satw-picks.json (written once)
  .venv-b/bin/python -m tools.run.speed_satw run MODEL [B,..] for the picked widths of MODEL's cases: the archive's own
                                                             predict (tools.run.speed_archive_predict, redirected into
                                                             artifacts/speed_v1/archive-runs/) then the fast compare at
                                                             batches 1, 8 and 32 (tools.run.speed_engine_validate)
The scan only chooses which width is tested; identity is decided by the archive comparison. GPU via gpu_run.sh.
"""
from __future__ import annotations
import json
import subprocess
import sys

from tools.experiment_b.common import ROOT

SPEED = ROOT / 'artifacts/speed_v1'
PICKS = SPEED / 'validation' / 'satw-picks.json'
R1 = '1f75c9232c8a0202482fe9bce9a46360c7cc0999c5b0bcf7df5ebcc0446fc863'
PY = str(ROOT / '.venv-b/bin/python')


def scan():
    from tools.scaled_bridge_fast import worker
    plan = json.loads((SPEED / 'validation' / 'plan.json').read_text())
    if PICKS.exists():
        raise SystemExit(f'{PICKS} exists')
    picks = {}
    for case, entry in plan.items():
        if not case.startswith('mobilenet'):
            continue
        structural = int(entry['policies']['sat_w_events'][len('sat.w'):]) + 4
        rows = []
        for width in range(structural - 5, structural - 10, -1):
            ctx = worker.context(case, f'sat.w{width}')
            items = worker.execute(ctx, 0, 64, 8)[0]
            rows.append({'width': width, 'images_with_events': sum(1 for it in items if it['events']),
                         'top1': sum(int(it['top5'][0] == it['label']) for it in items) / len(items)})
            print(json.dumps({'case': case, **rows[-1]}), flush=True)
        chosen = next((r['width'] for r in rows if r['images_with_events']), None)
        picks[case] = {'network_structural_width': structural, 'scan': rows,
                       'picked': None if chosen is None else f'sat.w{chosen}'}
    PICKS.write_text(json.dumps({'images': '0-64 of imagenet_screen_1k (development)', 'batch': 8,
                                 'rule': 'widest of structural-5..-9 with events on >= 1 image', 'picks': picks}, indent=1))


def run(model, batches='1,8,32'):
    picks = json.loads(PICKS.read_text())['picks']
    rc = 0
    for case, entry in sorted(picks.items()):
        if not case.startswith(model + '-') or entry['picked'] is None:
            continue
        policy = entry['picked']
        for cmd in ([PY, '-m', 'tools.run.speed_archive_predict', R1, case, '0', '64', '8', policy],
                    [PY, '-m', 'tools.run.speed_engine_validate', 'compare', case, policy, '--start', '0', '--stop', '64',
                     '--batches', batches]):
            code = subprocess.run(cmd, cwd=ROOT).returncode
            print(json.dumps({'case': case, 'policy': policy, 'step': cmd[2], 'exit': code}), flush=True)
            rc |= code != 0
            if code:
                break
    return rc


if __name__ == '__main__':
    if sys.argv[1] == 'scan':
        scan()
    else:
        sys.exit(run(*sys.argv[2:4]))
