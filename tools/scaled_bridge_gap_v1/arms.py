"""1k-panel workers: sealed v1 Engine arms and the read-only B batch-8 replay.

Both workers call the sealed v1 code unchanged and write only into
artifacts/scaled_bridge_gap_v1.  Images 0..127 are compared on the fly with
v1's sealed records and any difference stops the job.
"""
from __future__ import annotations
import json
import time
import numpy as np
from .common import *


def retained(name, ex):
    """Full-1k retained B and FP32 prediction records, digest-verified."""
    from tools.experiment_b import common as b
    config = unseal(checked(ex['B_configuration']))
    summary = unseal(checked(ex['B_summary']))
    ident = b.digest(config)
    rows = b.dataset('imagenet_screen_1k')[1]
    out = {}
    for key, identity, expected in (('B', ident, summary['prediction_digest']),
                                    ('FP32', config['baseline_sha256'], summary['baseline_prediction_digest'])):
        records = [unseal(b.BASE / 'predictions' / identity / (r['sha256'] + '.json')) for r in rows]
        if b.digest(records) != expected or any(r['sample'] != s for r, s in zip(records, rows)):
            raise ValueError('retained prediction digest drift')
        if records[:128] != ex[f'retained_{key}_prefix']:
            raise ValueError('retained prefix differs from v1 export')
        out[key] = records
    return config, out


def _check_sources(ex):
    from tools.scaled_bridge_v1.common import sources as v1_sources
    if v1_sources() != ex['sources']:
        raise ValueError('sealed v1 sources changed')


def run_arm(name, mode, start, stop):
    require_protocol()
    if mode not in ('wide', 'control'):
        raise ValueError('mode')
    started = time.perf_counter()
    from tools.experiment_b.classifier import configure, load_model, image_batch
    from tools.experiment_b.common import dataset
    from tools.scaled_bridge_v1.export import load_export
    from tools.scaled_bridge_v1.engine import Engine, numerical
    configure('cuda')
    ex, arrays, ref = load_export(name)
    _check_sources(ex)
    graph, transform, original = load_model('resnet18', 'cpu'); del original
    _, rows, payload = dataset('imagenet_screen_1k')
    if len(rows) != PANEL:
        raise ValueError('panel size')
    engine = Engine(ex, arrays, 'cuda', mode)
    setup = time.perf_counter() - started
    folder = arm_folder(name, mode); folder.mkdir(parents=True, exist_ok=True)
    v1folder = v1_run_root() / name / f'{mode}-cuda'
    done = 0; exec_total = 0.
    for i in range(start, stop):
        path = folder / f'{i:04d}.json'
        if path.exists():
            old = unseal(path)
            if old['export'] != ref or old['sample'] != rows[i] or old['mode'] != mode:
                raise ValueError('gap checkpoint drift')
            continue
        tick = time.perf_counter()
        inputs = image_batch(rows[i:i + 1], payload, transform, 'cpu').numpy()
        preprocess = time.perf_counter() - tick
        value = engine.run(inputs, oracle=False)
        record = {'export': ref, 'sample': rows[i], 'index': i, 'mode': mode, 'backend': 'cuda', **value,
                  'preprocess_seconds': preprocess, 'setup_seconds': setup if done == 0 else 0.}
        if i < 128:
            sealed = unseal(v1folder / f'{i:04d}.json')
            if sealed['export'] != ref or sealed['sample'] != rows[i] or numerical(sealed) != numerical(record):
                raise ValueError(f'v1 128-image reproduction FAILED {name} {mode} image {i}')
            record['v1_reproduced'] = True
        immutable(path, record)
        done += 1; exec_total += value['timing']['execution']
        if done % 50 == 0:
            print(f'{name} {mode} {i + 1}/{stop}: mean {exec_total / done:.3f}s/image', flush=True)
    _check_sources(ex)
    print(json.dumps({'job': 'arm', 'format': name, 'mode': mode, 'start': start, 'stop': stop, 'new_images': done,
                      'setup_seconds': setup, 'execution_seconds': exec_total,
                      'wall_seconds': time.perf_counter() - started}), flush=True)


def run_replay(name, start, stop):
    require_protocol()
    if start % 8 or stop % 8:
        raise ValueError('replay bounds must respect original batch-8 membership')
    started = time.perf_counter()
    import torch
    from tools.experiment_b.classifier import configure, load_model, prepare_qdq, image_batch
    from tools.experiment_b.runner import runtime
    from tools.experiment_b.common import dataset
    from tools.scaled_bridge_v1.export import load_export, calibration
    from tools.scaled_bridge_v1.replay import TracedB
    ex, _, ref = load_export(name); configure('cuda')
    _check_sources(ex)
    if runtime('cuda') != ex['B_runtime']:
        raise ValueError('cannot reproduce B in changed runtime')
    config, kept = retained(name, ex)
    graph, transform, original = load_model('resnet18', 'cuda'); del original
    maxima, _ = calibration(config)
    with torch.inference_mode():
        engine, scales = prepare_qdq(graph, name, 'maxabs', {k: np.empty(0) for k in maxima}, maxima, 'cuda')
    if scales != ex['scales']:
        raise ValueError('replay scales changed')
    _, rows, payload = dataset('imagenet_screen_1k')
    folder = arm_folder(name, 'B'); folder.mkdir(parents=True, exist_ok=True)
    v1folder = v1_run_root() / name / 'B'
    setup = time.perf_counter() - started; batches = 0; exec_total = 0.
    for s in range(start, stop, 8):
        paths = [folder / f'{i:04d}.json' for i in range(s, s + 8)]
        if all(p.exists() for p in paths):
            continue
        tick = time.perf_counter(); inputs = image_batch(rows[s:s + 8], payload, transform, 'cuda')
        prep = time.perf_counter() - tick
        tracer = TracedB(engine, name, ex['nodes']); tick = time.perf_counter()
        with torch.inference_mode():
            expected = engine.run(inputs); observed = tracer.run(inputs); baseline = graph(inputs)
        if not torch.equal(expected, observed):
            raise ValueError('B trace instrumentation changes output')
        top = observed.topk(5, dim=1).indices.cpu().tolist(); fp = baseline.topk(5, dim=1).indices.cpu().tolist()
        elapsed = time.perf_counter() - tick
        batches += 1; exec_total += elapsed
        for i, path in enumerate(paths, s):
            j = i - s
            if top[j] != kept['B'][i]['top5'] or fp[j] != kept['FP32'][i]['top5']:
                raise ValueError(f'original batch-8 retained Top-5 reproduction failed at image {i}')
            record = {'export': ref, 'sample': rows[i], 'index': i, 'batch_start': s, 'batch_images': 8,
                      'top5': top[j], 'FP32_top5': fp[j], 'layers': tracer.traces[j],
                      'preparation_batch_seconds': prep, 'execution_batch_seconds': elapsed,
                      'scope': 'original B batch8 Top5 reproduced against retained 1k records; separate v1 TracedB layer trace'}
            if i < 128:
                sealed = unseal(v1folder / f'{i:04d}.json')
                if any(sealed[k] != record[k] for k in ('export', 'sample', 'index', 'top5', 'FP32_top5', 'layers')):
                    raise ValueError(f'v1 128-image B trace reproduction FAILED {name} image {i}')
                record['v1_reproduced'] = True
            immutable(path, record)
        if batches % 10 == 0:
            print(f'{name} B {s + 8}/{stop}', flush=True)
    _check_sources(ex)
    print(json.dumps({'job': 'replay', 'format': name, 'start': start, 'stop': stop, 'new_batches': batches,
                      'setup_seconds': setup, 'execution_seconds': exec_total,
                      'wall_seconds': time.perf_counter() - started}), flush=True)
