"""Stage profiles of `predict` (trace none): the archives' NumPy path and the fast path (lane S1, part 1a).

Usage (repository root, GPU through gpu_run.sh):
  python -m tools.scaled_bridge_fast.profile archive DIGEST OUT.json IMAGES BATCH CASE:POLICY [CASE:POLICY ...]
  python -m tools.scaled_bridge_fast.profile fast OUT.json IMAGES BATCH CASE:POLICY [CASE:POLICY ...]

archive: the archived package is imported from its archive folder and its functions are wrapped with timers inside
this process only (nothing is written under artifacts/scaled_bridge_v2). The loop is the archive's predict loop
without its file output: image_batch -> Engine.run(trace='none') -> the per-image item assembly. The native call
of the archive (cudaMalloc, host-to-device copies, kernel, device-to-host copy, cudaFree, all inside the archived
library) is split by replaying every call's operands through the same kernel on device pointers (the fast
library: the archived kernel.h with the archive's nvcc flags), timed with synchronisation and excluded from the
loop time: kernel_only = replay time; transfers_and_allocation = native library call - kernel_only.
fast: the fast engine with a synchronisation after each stage (stage shares), then the same images again without
instrumentation (the loop time that counts).
Times are wall-clock seconds under a shared GPU slot unless the job ran with --exclusive: indicative shares only.
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

EVENTS = ('saturated_elements', 'high_clamps', 'low_clamps', 'nonfinite_elements', 'nonfinite_steps')


def items(records, first, rows, parameterised):
    out = []
    for index, record in zip(range(first, first + len(records)), records):
        item = {'index': index, 'sha256': rows[index]['sha256'], 'label': int(rows[index]['label']),
                'top5': record['top5'], 'output': record['output'], 'top1_tied_classes': record['top1_tied_classes']}
        if parameterised:      # the archives' predict item: only nodes with a non-zero event count
            item['failure'] = record['failure']; item['events'] = {}
            for n, stats in record['accumulator'].items():
                hit = {k: stats[k] for k in EVENTS if stats.get(k)}
                if hit:
                    item['events'][n] = hit
        out.append(item)
    return out


def archive_mode(digest_, out, images, batch, configs):
    root = Path.cwd()
    sys.path.insert(0, str(root / 'artifacts/scaled_bridge_v2/implementations' / digest_ / 'py'))
    import torch
    import scaled_bridge_v2.engine as E
    import scaled_bridge_v2.native as N
    from scaled_bridge_v2.common import digest, engine_sources
    from scaled_bridge_v2.worker import locate
    from scaled_bridge_v2.export import load_export
    from tools.experiment_b.classifier import configure, load_model, image_batch
    from tools.experiment_b.common import dataset
    from tools.scaled_bridge_fast.native import FastNative
    if digest(engine_sources()) != digest_:
        raise SystemExit('archive does not resolve to its digest')
    configure('cpu')
    fast = FastNative()
    T = defaultdict(float); C = defaultdict(int)
    state = {'replay': 0.}

    def timed(name, fn):
        def wrapper(*a, **k):
            tick = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                T[name] += time.perf_counter() - tick; C[name] += 1
        return wrapper

    class LibProxy:
        """Times the archived library calls; replays each call's operands through the same kernel on the GPU."""
        def __init__(self, lib):
            self.lib = lib

        def __getattr__(self, name):
            fn = getattr(self.lib, name)
            if not name.startswith('bridge2_conv'):
                return fn

            def call(*a):
                tick = time.perf_counter()
                result = fn(*a)
                T['native_library_call'] += time.perf_counter() - tick; C['native_library_call'] += 1
                return result
            return call

    original_run = N.Native.run

    def run(self, x, w, attrs, shift, policy, operand, params=None):
        if not isinstance(self.lib, LibProxy):
            self.lib = LibProxy(self.lib)
        tick = time.perf_counter()
        y, events = original_run(self, x, w, attrs, shift, policy, operand, params)
        T['native_run'] += time.perf_counter() - tick; C['native_run'] += 1
        # Kernel-only replay (excluded from the loop time).
        r0 = time.perf_counter()
        gx = torch.from_numpy(x).cuda(); gw = torch.from_numpy(w).cuda(); torch.cuda.synchronize()
        k0 = time.perf_counter()
        gy, gev, _ = fast.run(gx, gw, attrs, shift, policy, operand, params)
        torch.cuda.synchronize()
        T['kernel_only (replay on device pointers)'] += time.perf_counter() - k0
        if gy.cpu().numpy().tobytes() != y.tobytes():
            raise ValueError('replayed kernel differs from the archived library')
        state['replay'] += time.perf_counter() - r0
        return y, events
    N.Native.run = run
    E.quantize = timed('store_quantize (codebook search, codes, diagnostics)', E.quantize)
    E.array_hash = timed('hash (array_hash of outputs)', E.array_hash)
    E.stable_top5 = timed('top5 (contract tie order)', E.stable_top5)
    original_value = E.Engine._value

    def value(self, node, args):
        tick = time.perf_counter()
        try:
            return original_value(self, node, args)
        finally:
            T['op_' + node['op']] += time.perf_counter() - tick; C['op_' + node['op']] += 1
    E.Engine._value = value
    original_mac = E.Engine._mac

    def mac(self, node, state_, oracle, timing):
        tick = time.perf_counter(); before = T['native_run']; replay_before = state['replay']
        try:
            return original_mac(self, node, state_, oracle, timing)
        finally:
            T['mac_post (range checks, event stats, scale, bias)'] += (time.perf_counter() - tick - (T['native_run'] - before)
                                                                        - (state['replay'] - replay_before))
    E.Engine._mac = mac
    _, rows, payload = dataset('imagenet_screen_1k')
    results = []
    for config in configs:
        case, policy = config.split(':')
        T.clear(); C.clear(); state['replay'] = 0.
        ex, arrays, ref = load_export(case, locate(case))
        graph, transform, original = load_model(ex['model'], 'cpu'); del original, graph
        tick = time.perf_counter()
        engine = E.Engine(ex, arrays, 'cuda', policy)
        setup = time.perf_counter() - tick
        out_items = []
        wall = time.perf_counter()
        for first in range(0, images, batch):
            last = min(images, first + batch)
            t0 = time.perf_counter()
            inputs = image_batch(rows[first:last], payload, transform, 'cpu').numpy()
            T['preprocess (image_batch, CPU)'] += time.perf_counter() - t0
            t0 = time.perf_counter(); r0 = state['replay']
            records = engine.run(inputs, trace='none')
            T['engine_run'] += time.perf_counter() - t0 - (state['replay'] - r0)
            t0 = time.perf_counter()
            out_items.extend(items(records, first, rows, engine.policy.parameterised))
            T['item_assembly'] += time.perf_counter() - t0
        total = time.perf_counter() - wall - state['replay']
        stages = dict(T)
        stages['native_python (geometry, output arrays, result checks)'] = stages.pop('native_run') - stages['native_library_call']
        stages['transfers_and_allocation (library call - kernel_only)'] = stages['native_library_call'] - stages['kernel_only (replay on device pointers)']
        inside = sum(v for k, v in stages.items() if k not in ('engine_run', 'native_library_call') and not k.startswith('preprocess')
                     and k != 'item_assembly')
        stages['engine_other (graph walk, record dicts, input cast, nonfinite checks)'] = stages['engine_run'] - inside
        results.append({'path': f'archive {digest_[:8]}', 'case': case, 'policy': policy, 'images': images, 'batch': batch,
                        'engine_setup_seconds': setup, 'loop_wall_seconds': total, 'seconds_per_image': total / images,
                        'stages_seconds': stages, 'calls': dict(C), 'items_sha': digest(out_items)})
        print(json.dumps({'case': case, 'policy': policy, 'seconds_per_image': round(total / images, 4)}), flush=True)
    return results


def fast_mode(out, images, batch, configs):
    import torch
    from tools.scaled_bridge_fast import worker, engine as FE
    from tools.scaled_bridge_fast.common import digest
    T = defaultdict(float); C = defaultdict(int)
    sync = torch.cuda.synchronize

    def timed(name, fn):
        def wrapper(*a, **k):
            sync(); tick = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                sync(); T[name] += time.perf_counter() - tick; C[name] += 1
        return wrapper
    results = []
    for config in configs:
        case, policy = config.split(':')
        tick = time.perf_counter()
        ctx = worker.context(case, policy)
        setup = time.perf_counter() - tick
        engine = ctx[-1]
        # Uninstrumented pass first (the loop time that counts), then the instrumented pass on the same images.
        plain_items, _, _, timing = worker.execute(ctx, 0, images, batch)
        T.clear(); C.clear()
        native_run = engine.native.run
        engine.native.run = timed('kernel (archived kernel on device pointers, incl. launch)', native_run)
        for name in ('quantize', '_value'):
            original = getattr(engine, name)
            if name == '_value':
                def value(node, args, _o=original):
                    sync(); t = time.perf_counter()
                    try:
                        return _o(node, args)
                    finally:
                        sync(); T['op_' + node['op']] += time.perf_counter() - t; C['op_' + node['op']] += 1
                engine._value = value
            else:
                engine.quantize = timed('store_quantize (GPU)', original)
        original_mac = engine._mac

        def mac(node, s, checks, _o=original_mac):
            sync(); t = time.perf_counter(); before = T['kernel (archived kernel on device pointers, incl. launch)']
            try:
                return _o(node, s, checks)
            finally:
                sync(); T['mac_post (GPU)'] += time.perf_counter() - t - (T['kernel (archived kernel on device pointers, incl. launch)'] - before)
        engine._mac = mac
        wall = time.perf_counter()
        inst_items, _, _, inst_timing = worker.execute(ctx, 0, images, batch)
        instrumented = time.perf_counter() - wall
        engine.native.run = native_run; engine._mac = original_mac
        del engine.quantize, engine._value
        if inst_items != plain_items:
            raise ValueError('instrumented and plain fast runs differ')
        stages = dict(T)
        stages['engine_other (graph walk, checks, host transfer of output and stats, records)'] = \
            inst_timing['engine_seconds'] - sum(T.values())
        stages['preprocess_wait (CPU preprocessing not hidden by the prefetch thread)'] = inst_timing['preprocess_wait_seconds']
        results.append({'path': 'fast', 'case': case, 'policy': policy, 'images': images, 'batch': batch,
                        'engine_setup_seconds': setup, 'loop_wall_seconds': timing['wall_seconds'],
                        'seconds_per_image': timing['wall_seconds'] / images,
                        'engine_seconds_uninstrumented': timing['engine_seconds'],
                        'instrumented_wall_seconds': instrumented, 'stages_seconds_instrumented': stages, 'calls': dict(C),
                        'items_sha': digest(plain_items)})
        print(json.dumps({'case': case, 'policy': policy, 'seconds_per_image': round(timing['wall_seconds'] / images, 4)}), flush=True)
    return results


def main():
    mode = sys.argv[1]
    if mode == 'archive':
        digest_, out, images, batch, *configs = sys.argv[2:]
        results = archive_mode(digest_, out, int(images), int(batch), configs)
    elif mode == 'fast':
        out, images, batch, *configs = sys.argv[2:]
        results = fast_mode(out, int(images), int(batch), configs)
    else:
        raise SystemExit(__doc__)
    payload = {'mode': mode, 'argv': sys.argv[1:], 'results': results, 'written': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
               'note': 'shared GPU slot unless stated: stage shares are indicative, not throughput claims'}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if Path(out).exists():
        raise SystemExit(f'{out} exists (written once)')
    Path(out).write_text(json.dumps(payload, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
