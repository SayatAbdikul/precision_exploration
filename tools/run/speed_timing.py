"""Lane S1 part 1e: seconds per image of the predict loop, archive against fast path (protocol speed-protocol-v1).

  .venv-b/bin/python -m tools.run.speed_timing archive DIGEST CASE POLICY START STOP BATCHES OUT.json
  .venv-b/bin/python -m tools.run.speed_timing fast CASE POLICY START STOP BATCHES OUT.json
BATCHES: comma list (e.g. 8,32). Per batch size: wall-clock of the whole loop over [START, STOP) (preprocessing,
engine, item assembly; the archive's loop is its predict loop: image_batch then Engine.run(trace='none'); the fast
loop is worker.execute, which preprocesses the next batch on a CPU thread), after a 1-batch warm-up of the same
batch size outside the timed loop. Process setup (imports, export, engine construction) is timed separately.
The archive is imported from its folder with its run root redirected to artifacts/speed_v1/archive-runs/<digest>
(nothing is written under artifacts/scaled_bridge_v2). The items of every batch size must equal each other and,
across the two paths, the items digests must be equal (checked when the summary is written).
Run both paths back to back inside one `gpu_run.sh --exclusive` call for a reportable seconds-per-image number.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

EVENTS = ('saturated_elements', 'high_clamps', 'low_clamps', 'nonfinite_elements', 'nonfinite_steps')


def archive_runner(digest_, case, policy):
    root = Path.cwd()
    sys.path.insert(0, str(root / 'artifacts/scaled_bridge_v2/implementations' / digest_ / 'py'))
    import scaled_bridge_v2.common as common
    import scaled_bridge_v2.worker as worker
    if common.digest(common.engine_sources()) != digest_:
        raise SystemExit('archive does not resolve to its digest')
    target = root / 'artifacts/speed_v1/archive-runs' / digest_
    common.run_root = worker.run_root = (lambda: target)
    from tools.experiment_b.classifier import image_batch
    ex, arrays, ref, graph, transform, rows, payload, engine = worker.context(case, 'cuda', policy)

    def loop(start, stop, batch):
        images = []
        for first in range(start, stop, batch):
            last = min(stop, first + batch)
            inputs = image_batch(rows[first:last], payload, transform, 'cpu').numpy()
            records = engine.run(inputs, trace='none')
            for index, record in zip(range(first, last), records):
                item = {'index': index, 'sha256': rows[index]['sha256'], 'label': int(rows[index]['label']),
                        'top5': record['top5'], 'output': record['output'], 'top1_tied_classes': record['top1_tied_classes']}
                if engine.policy.parameterised:
                    item['failure'] = record['failure']; item['events'] = {}
                    for node, stats in record['accumulator'].items():
                        hit = {k: stats[k] for k in EVENTS if stats.get(k)}
                        if hit:
                            item['events'][node] = hit
                images.append(item)
        return images
    return loop, common.digest


def fast_runner(case, policy):
    from tools.scaled_bridge_fast import worker
    from tools.scaled_bridge_fast.common import digest
    ctx = worker.context(case, policy)
    return (lambda start, stop, batch: worker.execute(ctx, start, stop, batch)[0]), digest


def main():
    mode = sys.argv[1]
    tick = time.perf_counter()
    if mode == 'archive':
        digest_, case, policy, start, stop, batches, out = sys.argv[2:9]
        loop, digest = archive_runner(digest_, case, policy)
        path = f'archive {digest_[:8]}'
    else:
        case, policy, start, stop, batches, out = sys.argv[2:8]
        loop, digest = fast_runner(case, policy)
        path = 'fast'
    import torch
    torch.cuda.synchronize()
    setup = time.perf_counter() - tick
    start, stop = int(start), int(stop)
    runs = []; reference = None
    for batch in [int(b) for b in batches.split(',')]:
        loop(start, start + batch, batch)                       # warm-up (not timed)
        torch.cuda.synchronize(); t0 = time.perf_counter(); epoch0 = time.time()
        items = loop(start, stop, batch)
        torch.cuda.synchronize(); wall = time.perf_counter() - t0
        sha = digest(items)
        reference = reference or sha
        if sha != reference:
            raise ValueError('items differ between batch sizes')
        runs.append({'batch': batch, 'images': stop - start, 'wall_seconds': wall, 'seconds_per_image': wall / (stop - start),
                     'loop_start_epoch': epoch0, 'loop_end_epoch': epoch0 + wall})
        print(json.dumps({'path': path, 'case': case, 'policy': policy, **runs[-1]}), flush=True)
    result = {'path': path, 'case': case, 'policy': policy, 'start': start, 'stop': stop, 'setup_seconds': setup,
              'runs': runs, 'items_sha': reference, 'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(),
              'written': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'argv': sys.argv[1:]}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if Path(out).exists():
        raise SystemExit(f'{out} exists (written once)')
    Path(out).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
