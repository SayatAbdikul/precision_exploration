"""Lane S1, protocol speed-protocol-v1-addendum-1 (review 1, finding N3): what a fast-engine process costs before
its first images.

  .venv-b/bin/python -m tools.run.speed_setup            (GPU through gpu_run.sh; shared GPU, indicative)
(a) wall time of the fast archive's `run.sh root` (its digest check; CPU), 3 repetitions;
(b) per case (policy wide) a child process that records: interpreter start to first line (from the parent's spawn
    time), `from tools.scaled_bridge_fast import worker` (torch, CUDA library), worker.context (export, codes, native
    library, CUDA context), one 8-image batch (images 0-8). The parent records spawn-to-exit wall time.
Result: artifacts/speed_v1/setup-v1/setup.json (written once). gpu_run.sh admission waits are not included.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time

from tools.experiment_b.common import ROOT

OUT = ROOT / 'artifacts/speed_v1/setup-v1/setup.json'
ARCHIVE = ROOT / 'artifacts/speed_v1/implementations/edfecd5cdb0dc5adf06eff2807afa023b12d781cf0d08961a0b023da510a293b'
PY = str(ROOT / '.venv-b/bin/python')
CASES = ('resnet18-int8-default-b2', 'mobilenet_v3_large-int8-default-b2')

CHILD = r'''
import json, sys, time
t_first = time.time()
from tools.scaled_bridge_fast import worker
t_import = time.time()
ctx = worker.context(sys.argv[1], 'wide')
t_context = time.time()
items, _, _, timing = worker.execute(ctx, 0, 8, 8)
t_batch = time.time()
print('CHILD ' + json.dumps({'first_line_epoch': t_first, 'import_seconds': t_import - t_first,
      'context_seconds': t_context - t_import, 'first_batch_seconds': t_batch - t_context, 'images': len(items)}))
'''


def main():
    if OUT.exists():
        raise SystemExit(f'{OUT} exists')
    result = {'protocol': 'speed-protocol-v1-addendum-1 (n3_process_setup)', 'shared_gpu': True, 'root': [], 'cases': {}}
    for _ in range(3):
        tick = time.time()
        proc = subprocess.run([str(ARCHIVE / 'run.sh'), 'root'], cwd=ROOT, capture_output=True, text=True)
        result['root'].append({'wall_seconds': time.time() - tick, 'exit': proc.returncode,
                               'resolves_to_own_digest': proc.stdout.strip().endswith(ARCHIVE.name)})
    for case in CASES:
        rows = []
        for _ in range(2):
            spawn = time.time()
            proc = subprocess.run([PY, '-c', CHILD, case], cwd=ROOT, capture_output=True, text=True)
            wall = time.time() - spawn
            line = next((l for l in proc.stdout.splitlines() if l.startswith('CHILD ')), None)
            child = json.loads(line[len('CHILD '):]) if line else {'stderr_tail': proc.stderr[-400:]}
            if line:
                child['interpreter_start_seconds'] = child.pop('first_line_epoch') - spawn
            rows.append({'exit': proc.returncode, 'spawn_to_exit_seconds': wall, **child})
            print(json.dumps({'case': case, **rows[-1]}), flush=True)
        result['cases'][case] = rows
    result['written'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print('SETUP-DONE', flush=True)


if __name__ == '__main__':
    main()
