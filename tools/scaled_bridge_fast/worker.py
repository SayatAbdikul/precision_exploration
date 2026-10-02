"""predict, identity comparisons and gates of the fast path (same record format as the archives' predict)."""
from __future__ import annotations
import json
import threading
import time
from pathlib import Path
import numpy as np
from .common import base, run_root, reference_root, immutable, digest, unseal, identity, V2, RESNET_DIGEST, BASE_DIGEST

_b = base()
from scaled_bridge_v2.common import reference                     # noqa: E402
from scaled_bridge_v2.worker import locate                        # noqa: E402
from scaled_bridge_v2.export import load_export                   # noqa: E402
from scaled_bridge_v2.accumulators import resolve                 # noqa: E402
from scaled_bridge_v2.engine import numerical                     # noqa: E402

EVENTS = ('saturated_elements', 'high_clamps', 'low_clamps', 'nonfinite_elements', 'nonfinite_steps')
GATE_IMAGES = 64


def context(case, policy):
    import torch
    from tools.experiment_b.classifier import configure, load_model
    from tools.experiment_b.common import dataset
    from .engine import FastEngine
    configure('cuda')
    ex, arrays, ref = load_export(case, locate(case))
    graph, transform, original = load_model(ex['model'], 'cpu')
    del original, graph
    _, rows, payload = dataset('imagenet_screen_1k')
    if digest(rows) != ex['retained']['ordered_samples_sha256']:
        raise ValueError('ordered development samples changed')
    engine = FastEngine(ex, arrays, policy)
    torch.cuda.synchronize()
    return ex, ref, transform, rows, payload, engine


class Prefetch:
    """Preprocesses the next batch on the CPU (the archive's image_batch, unchanged) while the GPU runs."""
    def __init__(self, rows, payload, transform, ranges):
        from tools.experiment_b.classifier import image_batch
        self.ranges, self.results = list(ranges), {}
        self.cv = threading.Condition()

        def work():
            for first, last in self.ranges:
                tick = time.perf_counter()
                value = image_batch(rows[first:last], payload, transform, 'cpu').numpy()
                with self.cv:
                    self.results[first] = (value, time.perf_counter() - tick)
                    self.cv.notify_all()
                    while len(self.results) >= 2:
                        self.cv.wait()
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def get(self, first):
        with self.cv:
            while first not in self.results:
                self.cv.wait(timeout=600)
                if first not in self.results and not self.thread.is_alive():
                    raise RuntimeError('preprocessing thread ended')
            value = self.results.pop(first)
            self.cv.notify_all()
            return value


def items_of(engine, records, rows, first, nodes):
    out = []
    for index, record in zip(range(first, first + len(records)), records):
        item = {'index': index, 'sha256': rows[index]['sha256'], 'label': int(rows[index]['label']),
                'top5': record['top5'], 'output': record['output'], 'top1_tied_classes': record['top1_tied_classes']}
        if engine.policy.parameterised:
            item['failure'] = record['failure']; item['events'] = {}
            for node, stats in record['accumulator'].items():
                nodes[node] = {k: v for k, v in stats.items() if k not in EVENTS}
                hit = {k: stats[k] for k in EVENTS if stats.get(k)}
                if hit:
                    item['events'][node] = hit
        out.append(item)
    return out


def execute(ctx, start, stop, batch, trace='none', keep=False):
    """Run [start, stop) at a batch size; returns (items, nodes, records or None, timing)."""
    ex, ref, transform, rows, payload, engine = ctx
    ranges = [(f, min(stop, f + batch)) for f in range(start, stop, batch)]
    pre = Prefetch(rows, payload, transform, ranges)
    images, nodes, kept = [], {}, []
    gpu = prep_wait = 0.; tick = time.perf_counter()
    for first, last in ranges:
        t0 = time.perf_counter()
        inputs, _ = pre.get(first)
        prep_wait += time.perf_counter() - t0
        records = engine.run(inputs, trace=trace)
        gpu += records[0]['timing']['execution']
        images.extend(items_of(engine, records, rows, first, nodes))
        if keep:
            kept.extend(records)
    wall = time.perf_counter() - tick
    return images, nodes, (kept if keep else None), {'wall_seconds': wall, 'engine_seconds': gpu,
                                                      'preprocess_wait_seconds': prep_wait, 'images': stop - start}


def require_gates(case, policy):
    """The fast path runs a case only with its identity gate under this digest, and a parameterised policy only
    with a passing identity gate of the same family on a case of the same model (as the 1f75c923 archive)."""
    path = run_root() / case / 'gate.json'
    if not path.exists():
        raise ValueError(f'{case}: no fast-path case gate under this digest ({path})')
    record = unseal(path)
    if record['status'] != 'pass' or record['identity'] != identity():
        raise ValueError(f'{case}: the fast-path case gate does not pass for this digest')
    kind = resolve(policy)
    if kind.parameterised:
        model = case.split('-')[0]
        found = [p for p in sorted(run_root().glob(f'{model}-*/gate-*.json'))
                 if unseal(p)['status'] == 'pass' and resolve(unseal(p)['policy']).kind == kind.kind]
        if not found:
            raise ValueError(f'{case}: no passing {kind.kind} fast-path policy gate on a {model} case under this digest')


def predict(case, policy, backend, start, stop, batch):
    """Compact predictions, the archives' format; one sealed file per call under this digest's run root."""
    if backend != 'cuda':
        raise ValueError('the fast path runs on CUDA only')
    require_gates(case, policy)
    tick = time.time()
    ctx = context(case, policy)
    ex, ref, transform, rows, payload, engine = ctx
    if not 0 <= start < stop <= len(rows) or batch < 1:
        raise ValueError('prediction range is outside the development screen list')
    path = run_root() / case / 'predictions' / f'{policy}-{backend}-{start:05d}-{stop:05d}.json'
    if path.exists():
        return unseal(path)
    setup = time.time() - tick
    images, nodes, _, timing = execute(ctx, start, stop, batch)
    sealed = reference_root(case) / case / f'{policy}-cuda'
    compared = 0
    for item in images:
        old = sealed / f'{item["index"]:04d}.json'
        if old.exists():
            old = unseal(old)
            if old['output'] != item['output'] or old['top5'] != item['top5'] or old['sample'] != rows[item['index']]:
                raise ValueError('fast prediction differs from the archive\'s sealed batch-1 record')
            compared += 1
    result = {'case': case, 'policy': policy, 'definition': engine.policy.definition, 'backend': backend,
              'start': start, 'stop': stop, 'batch': batch, 'export': ref, 'engine_sources': digest(identity()),
              'execution_path': 'scaled_bridge_fast (lane S1): archived kernels on device pointers, GPU post-operations',
              'base_engine_sources': BASE_DIGEST,
              'list': 'imagenet_screen_1k (development)', 'ordered_samples_sha256': digest(rows),
              'nodes': nodes, 'images': images, 'compared_with_sealed_batch1': compared,
              'execution_seconds': timing['engine_seconds'], 'wall_seconds': timing['wall_seconds'],
              'setup_seconds': setup, 'gpu_lock_declared': False,
              'event_fields': 'events lists, per MAC node with any event, the non-zero counts; node constants '
                              '(width or format, outputs per image) are under nodes'}
    immutable(path, result)
    ledger(['predict', case, policy, start, stop, batch], tick, images=len(images))
    return result


def ledger(arguments, start, **extra):
    record = {'arguments': [str(a) for a in arguments], 'started_epoch': start, 'wall_seconds': time.time() - start, **extra}
    immutable(run_root() / 'ledger' / f'{int(start * 1000)}-{threading.get_native_id()}.json', record)


# -- identity evidence ----------------------------------------------------------------------------------------
def archive_items(case, policy, start, stop, extra_roots=()):
    """Archive predict items for [start, stop) from sealed predict files (read-only); None where missing."""
    found, nodes, files = {}, None, []
    for root in (reference_root(case), *extra_roots):
        for path in sorted((root / case / 'predictions').glob(f'{policy}-cuda-*.json')):
            a, b = (int(x) for x in path.stem.split('-')[-2:])
            if b <= start or a >= stop:
                continue
            doc = unseal(path); files.append(str(path.relative_to(V2.parent.parent)))
            if nodes is not None and doc['nodes'] and nodes != doc['nodes']:
                raise ValueError('archive files disagree on node constants')
            nodes = doc['nodes'] or nodes
            for item in doc['images']:
                if start <= item['index'] < stop:
                    if item['index'] in found and found[item['index']] != item:
                        raise ValueError('archive files disagree')
                    found[item['index']] = item
    return found, nodes, files


def compare(case, policy, start, stop, batch, ctx=None, extra_roots=()):
    """Fast predict items against the archive's sealed predict items, field by field; returns an identity record."""
    ctx = ctx or context(case, policy)
    images, nodes, _, timing = execute(ctx, start, stop, batch)
    old, old_nodes, files = archive_items(case, policy, start, stop, extra_roots)
    same = differ = 0; first_difference = None
    for item in images:
        o = old.get(item['index'])
        if o is None:
            continue
        if o == item:
            same += 1
        else:
            differ += 1
            if first_difference is None:
                first_difference = {'index': item['index'], 'fields': sorted(k for k in item if o.get(k) != item.get(k))}
    nodes_equal = None if not old else (old_nodes == nodes if resolve(policy).parameterised else True)
    return {'case': case, 'policy': policy, 'start': start, 'stop': stop, 'batch': batch, 'compared': same + differ,
            'identical': same, 'different': differ, 'first_difference': first_difference, 'node_constants_equal': nodes_equal,
            'archive_files': files, 'fast_timing': timing}


def regress(case, policy, old_backend, images, old_root, batch=8):
    """Full-trace fast records against the sealed full-trace records of an earlier run root (numerical())."""
    ctx = context(case, policy)
    _, rows = ctx[1], ctx[3]
    _, _, records, timing = execute(ctx, 0, images, batch, trace='full', keep=True)
    same = 0; bad = []
    for i, record in enumerate(records):
        old = unseal(Path(old_root) / case / f'{policy}-{old_backend}' / f'{i:04d}.json')
        if old['export'] != ctx[1] or old['sample'] != rows[i]:
            raise ValueError('regression record belongs to another export or sample')
        if numerical(record) == numerical(old):
            same += 1
        else:
            a, b = numerical(record), numerical(old)
            bad.append({'image': i, 'keys': sorted(k for k in a if a[k] != b.get(k)),
                        'layers': sorted(k for k in a['layers'] if a['layers'][k] != b['layers'].get(k))[:5]})
    return {'case': case, 'policy': policy, 'old_backend': old_backend, 'old_root': Path(old_root).name, 'images': images,
            'batch': batch, 'identical': same, 'differences': bad, 'fast_timing': timing}
