"""Panel execution, throughput measurement and the per-case gate."""
from __future__ import annotations
import time
import numpy as np
from .common import (ROOT, BASE, digest, unseal, reference, immutable, engine_sources, run_root,
                     under_gpu_lock, Stopwatch)

POLICY_ARMS = ('wide', 'control')
GATE_IMAGES = 8
REPLAY_IMAGES = 32


def locate(case):
    """The one export of a case, whichever exporter (b1 adapter, B2 adapter) wrote it."""
    found = sorted((BASE / 'exports').glob(f'*/{case}/export.json'))
    if len(found) != 1:
        raise ValueError(f'expected exactly one export of {case}, found {len(found)}')
    return found[0]


def context(case, backend, policy):
    from tools.experiment_b.classifier import configure, load_model
    from tools.experiment_b.common import dataset
    from .export import load_export
    from .engine import Engine
    configure('cpu')
    ex, arrays, ref = load_export(case, locate(case))
    graph, transform, original = load_model(ex['model'], 'cpu')
    del original
    _, rows, payload = dataset('imagenet_screen_1k')
    if digest(rows) != ex['retained']['ordered_samples_sha256']:
        raise ValueError('ordered development samples changed')
    engine = Engine(ex, arrays, backend, policy)
    immutable(run_root() / case / 'certificate.json', {'export': ref, 'certificates': engine.certificates})
    return ex, arrays, ref, graph, transform, rows, payload, engine


def panel(case, policy, backend, target):
    """Batch-1 records with full traces (the gating and replay evidence)."""
    from tools.experiment_b.classifier import image_batch
    from .engine import numerical
    watch = Stopwatch('panel', [case, policy, backend, target]); started = time.perf_counter()
    ex, arrays, ref, graph, transform, rows, payload, engine = context(case, backend, policy)
    setup = time.perf_counter() - started
    folder = run_root() / case / f'{policy}-{backend}'; computed = 0; seconds = 0.
    for i in range(target):
        path = folder / f'{i:04d}.json'
        if path.exists():
            old = unseal(path)
            if old['export'] != ref or old['sample'] != rows[i] or old['policy'] != policy or old['backend'] != backend:
                raise ValueError('panel checkpoint drift')
            continue
        tick = time.perf_counter()
        inputs = image_batch(rows[i:i + 1], payload, transform, 'cpu').numpy()
        preprocess = time.perf_counter() - tick
        value = engine.run(inputs, oracle=(backend == 'cpp' and i == 0))[0]
        if backend == 'cpp' and policy == 'wide' and i == 0:
            from .reference import FXReference
            tick = time.perf_counter()
            witness = FXReference(graph, ex, arrays).evaluate(inputs)
            if witness != numerical(value):
                failed = [k for k in witness['layers'] if witness['layers'][k] != value['layers'][k]]
                raise ValueError('independent full FX graph mismatch ' + str(failed[:4]))
            immutable(run_root() / case / 'wide-reference.json', {
                'export': ref, 'sample': rows[0], 'numerical': witness, 'seconds': time.perf_counter() - tick,
                'basis': 'independent FX traversal; binary64 framework convolution on 16-bit limbs; '
                         'Python integer recombination and correctly rounded conversion'})
        record = {'export': ref, 'sample': rows[i], 'index': i, 'policy': policy, 'backend': backend, **value,
                  'preprocess_seconds': preprocess, 'setup_seconds': setup if i == 0 else 0.,
                  'gpu_lock_declared': under_gpu_lock(), 'engine_sources': digest(engine_sources())}
        immutable(path, record)
        computed += 1; seconds += value['timing']['execution']
        print(f'{case} {policy}-{backend} {i + 1}/{target}: {value["timing"]["execution"]:.3f}s', flush=True)
    watch.close(computed_images=computed, execution_seconds=seconds)


def throughput(case, policy, backend, images, batch, trace):
    """Timed run at a chosen batch size and trace level; never the evidence of record.

    Outputs and predictions are compared with the sealed batch-1 records, so a
    faster configuration cannot silently change results.
    """
    from tools.experiment_b.classifier import image_batch
    watch = Stopwatch('throughput', [case, policy, backend, images, batch, trace])
    ex, arrays, ref, graph, transform, rows, payload, engine = context(case, backend, policy)
    folder = run_root() / case / f'{policy}-{backend}'
    execution = []; preprocess = []; compared = 0
    for start in range(0, images, batch):
        tick = time.perf_counter()
        inputs = image_batch(rows[start:start + batch], payload, transform, 'cpu').numpy()
        preprocess.append(time.perf_counter() - tick)
        records = engine.run(inputs, trace=trace)
        execution.append(records[0]['timing']['execution'])
        for offset, record in enumerate(records):
            path = folder / f'{start + offset:04d}.json'
            if path.exists():
                old = unseal(path)
                if old['output'] != record['output'] or old['top5'] != record['top5']:
                    raise ValueError('batched execution differs from the sealed batch-1 record')
                if trace != 'none' and any(old['layers'][k]['codes'] != v['codes'] for k, v in record['layers'].items()):
                    raise ValueError('batched layer codes differ from the sealed batch-1 record')
                compared += 1
    total = len(execution) * batch
    result = {'case': case, 'policy': policy, 'backend': backend, 'batch': batch, 'trace': trace,
              'images': total, 'compared_with_sealed_batch1': compared,
              'execution_seconds_per_image': float(np.sum(execution)) / total,
              'steady_execution_seconds_per_image': float(np.sum(execution[1:])) / max(1, total - batch),
              'preprocess_seconds_per_image': float(np.sum(preprocess)) / total,
              'gpu_lock_declared': under_gpu_lock(), 'export': ref}
    folder = run_root() / case / 'throughput'
    folder.mkdir(parents=True, exist_ok=True)
    immutable(folder / f'{policy}-{backend}-b{batch}-{trace}-{images}-{int(time.time())}.json', result)
    watch.close(images=total)
    print(result, flush=True)
    return result


def require_gates(case, policy):
    """`predict` runs only gated evidence: the case gate of this engine digest and, for a parameterised
    accumulator, a passing policy gate of the same family (saturating integer or float) on a case of the same
    model under this digest."""
    from .export import load_export
    from .accumulators import resolve
    ex, _, ref = load_export(case, locate(case))
    path = run_root() / case / 'gate.json'
    if not path.exists():
        raise ValueError(f'{case}: no case gate under this engine digest ({path}); run the gate first')
    gate_record = unseal(path)
    if gate_record['status'] != 'pass' or gate_record['export'] != ref or gate_record['engine_sources'] != digest(engine_sources()):
        raise ValueError(f'{case}: the case gate does not pass for this export and engine digest')
    kind = resolve(policy)
    if kind.parameterised:
        found = []
        for other in sorted(run_root().glob(f'{ex["model"]}-*/gate-*.json')):
            record = unseal(other)
            if record['status'] == 'pass' and resolve(record['policy']).kind == kind.kind \
                    and record['engine_sources'] == digest(engine_sources()):
                found.append(other)
        if not found:
            raise ValueError(f'{case}: no passing {kind.kind} policy gate on a {ex["model"]} case under this digest')


def predict(case, policy, backend, start, stop, batch):
    """Compact predictions for panels beyond the gating prefix: one sealed file per call, no layer traces.

    Per image: Top-5 (contract tie order), output hash, classes tied at the maximum and, for a parameterised
    accumulator, the failure record and the per-node event counts that are not zero. Images that also have a
    sealed batch-1 record must agree with it. Images come from the development screen-1k list only.
    """
    from tools.experiment_b.classifier import image_batch
    require_gates(case, policy)
    watch = Stopwatch('predict', [case, policy, backend, start, stop, batch])
    ex, arrays, ref, graph, transform, rows, payload, engine = context(case, backend, policy)
    if not 0 <= start < stop <= len(rows) or batch < 1:
        raise ValueError('prediction range is outside the development screen list')
    path = run_root() / case / 'predictions' / f'{policy}-{backend}-{start:05d}-{stop:05d}.json'
    if path.exists():
        return unseal(path)
    sealed = run_root() / case / f'{policy}-{backend}'
    events = ('saturated_elements', 'high_clamps', 'low_clamps', 'nonfinite_elements', 'nonfinite_steps')
    images = []; nodes = {}; compared = 0; seconds = 0.
    for first in range(start, stop, batch):
        last = min(stop, first + batch)
        inputs = image_batch(rows[first:last], payload, transform, 'cpu').numpy()
        records = engine.run(inputs, trace='none')
        seconds += records[0]['timing']['execution']
        for index, record in zip(range(first, last), records):
            item = {'index': index, 'sha256': rows[index]['sha256'], 'label': int(rows[index]['label']),
                    'top5': record['top5'], 'output': record['output'],
                    'top1_tied_classes': record['top1_tied_classes']}
            if engine.policy.parameterised:
                item['failure'] = record['failure']; item['events'] = {}
                for node, stats in record['accumulator'].items():
                    nodes[node] = {k: v for k, v in stats.items() if k not in events}
                    hit = {k: stats[k] for k in events if stats.get(k)}
                    if hit:
                        item['events'][node] = hit
            old = sealed / f'{index:04d}.json'
            if old.exists():
                old = unseal(old)
                if old['output'] != record['output'] or old['top5'] != record['top5'] or old['sample'] != rows[index]:
                    raise ValueError('compact prediction differs from the sealed batch-1 record')
                compared += 1
            images.append(item)
        print(f'{case} {policy}-{backend} predicted {last}/{stop}', flush=True)
    result = {'case': case, 'policy': policy, 'definition': engine.policy.definition, 'backend': backend,
              'start': start, 'stop': stop, 'batch': batch, 'export': ref, 'engine_sources': digest(engine_sources()),
              'list': 'imagenet_screen_1k (development)', 'ordered_samples_sha256': digest(rows),
              'nodes': nodes, 'images': images, 'compared_with_sealed_batch1': compared,
              'execution_seconds': seconds, 'gpu_lock_declared': under_gpu_lock(),
              'event_fields': 'events lists, per MAC node with any event, the non-zero counts; node constants '
                              '(width or format, outputs per image) are under nodes'}
    immutable(path, result)
    watch.close(images=len(images))
    return result


def gate(case):
    from .engine import numerical
    from .export import load_export
    ex, _, ref = load_export(case, locate(case))
    root = run_root() / case
    certificate = unseal(root / 'certificate.json')
    proof = root / 'wide-reference.json'; unseal(proof)
    books = sorted(ex['codebooks'])
    conformance = [run_root() / 'conformance' / f'{name}.json' for name in books]
    for path in conformance:
        if unseal(path)['status'] != 'pass':
            raise ValueError('primitive conformance missing')
    macs = len(certificate['certificates']); refs = []
    for policy in POLICY_ARMS:
        for i in range(GATE_IMAGES):
            paths = [root / f'{policy}-{b}' / f'{i:04d}.json' for b in ('cpp', 'cuda')]
            a, b = map(unseal, paths)
            if a['export'] != ref or b['export'] != ref or a['sample'] != b['sample'] or numerical(a) != numerical(b):
                raise ValueError(f'CPU/CUDA eight-image gate mismatch {case} {policy} {i}')
            if i == 0 and a['oracle_dot_checks'] < 3 * macs:
                raise ValueError('missing actual-node rational dot checks')
            refs.extend(map(reference, paths))
    replayed = [unseal(root / 'B' / f'{i:04d}.json') for i in range(REPLAY_IMAGES)]
    if any(r['top5'] != ex['retained']['B_prefix'][i]['top5'] for i, r in enumerate(replayed)):
        raise ValueError('B replay does not reproduce the retained Top-5')
    record = {'status': 'pass', 'case': case, 'export': ref, 'images_per_arm': GATE_IMAGES, 'arms': list(POLICY_ARMS),
              'records': refs, 'primitive_conformance': [reference(p) for p in conformance],
              'independent_wide_graph': reference(proof), 'certificate': reference(root / 'certificate.json'),
              'B_replay_images': REPLAY_IMAGES, 'engine_sources': digest(engine_sources()),
              'scope': 'scaled contract v2; no legacy unscaled certificate is extended'}
    immutable(root / 'gate.json', record)
    print(case + ' eight-image CPU/CUDA gate PASS', flush=True)
    return record


LOSSLESS = ('sat.abs-0', 'sat.struct-0')


def policy_gate(case, policy):
    """Gate of one parameterised accumulator policy on one case.

    Requires the primitive policy witnesses, actual-node rational dot checks on
    the first image, identical CPU and CUDA records on eight images and, for a
    width equal to a certified lossless width, identity with the exact arm and
    no saturation on every image of the panel.
    """
    from .engine import numerical
    from .export import load_export
    from .accumulators import resolve
    ex, _, ref = load_export(case, locate(case))
    root = run_root() / case
    certificate = unseal(root / 'certificate.json'); macs = len(certificate['certificates'])
    witness = run_root() / 'conformance' / 'policies.json'
    if unseal(witness)['status'] != 'pass' or not resolve(policy).parameterised:
        raise ValueError('policy conformance missing or not a parameterised policy')
    refs = []
    for i in range(GATE_IMAGES):
        paths = [root / f'{policy}-{b}' / f'{i:04d}.json' for b in ('cpp', 'cuda')]
        a, b = map(unseal, paths)
        if a['export'] != ref or b['export'] != ref or a['sample'] != b['sample'] or numerical(a) != numerical(b):
            raise ValueError(f'CPU/CUDA eight-image gate mismatch {case} {policy} {i}')
        if i == 0 and a['oracle_dot_checks'] < 3 * macs and a['failure'] is None:
            raise ValueError('missing actual-node rational dot checks')
        refs.extend(map(reference, paths))
    panel_records = [unseal(p) for p in sorted((root / f'{policy}-cuda').glob('[0-9]*.json'))]
    graph_witness = None
    if 'B2_export' in ex['provenance']:
        # Contract 2.2 / PROTOCOL_MN: the whole-graph policy witness on the first image of the CPU panel.
        path = root / f'graph-witness-{policy}-cpp-0000.json'
        if not path.exists() or unseal(path)['status'] != 'pass' or unseal(path)['record'] != reference(root / f'{policy}-cpp' / '0000.json'):
            raise ValueError(f'{case} {policy}: whole-graph policy witness missing or failed')
        graph_witness = reference(path)
    lossless = None
    if policy in LOSSLESS:
        exact = [unseal(root / 'wide-cuda' / f'{i:04d}.json') for i in range(len(panel_records))]
        for r, w in zip(panel_records, exact):
            if r['layers'] != w['layers'] or r['output'] != w['output'] or r['top5'] != w['top5'] or r['failure'] is not None \
                    or any(v['saturated_elements'] for v in r['accumulator'].values()):
                raise ValueError(f'{case} {policy}: a certified lossless width differs from the exact arm')
        lossless = {'images_identical_to_wide': len(panel_records), 'saturated_elements': 0}
    record = {'status': 'pass', 'case': case, 'policy': policy, 'definition': resolve(policy).definition, 'export': ref,
              'images_per_backend_gate': GATE_IMAGES, 'panel_images_cuda': len(panel_records), 'records': refs,
              'primitive_policy_conformance': reference(witness), 'certificate': reference(root / 'certificate.json'),
              'oracle_dot_checks_first_image': unseal(root / f'{policy}-cpp' / '0000.json')['oracle_dot_checks'],
              'lossless_identity': lossless, 'images_failed_nonfinite': sum(r['failure'] is not None for r in panel_records),
              'graph_witness_first_image': graph_witness,
              'engine_sources': digest(engine_sources())}
    immutable(root / f'gate-{policy}.json', record)
    print(f'{case} {policy} policy gate PASS', flush=True)
    return record


def regress(case, policy, backend, images, old_run):
    """Re-run sealed records of an earlier engine digest and require identical numerical content."""
    from tools.experiment_b.classifier import image_batch
    from .engine import numerical
    matches = [p for p in (BASE / 'runs').glob(old_run + '*') if p.is_dir()]
    if len(matches) != 1 or matches[0] == run_root():
        raise ValueError('old run digest prefix must select exactly one earlier run root')
    old_root = matches[0]
    watch = Stopwatch('regress', [case, policy, backend, images, old_root.name])
    ex, arrays, ref, graph, transform, rows, payload, engine = context(case, backend, policy)
    same = 0
    for i in range(images):
        old = unseal(old_root / case / f'{policy}-{backend}' / f'{i:04d}.json')
        if old['export'] != ref or old['sample'] != rows[i]:
            raise ValueError('regression record belongs to another export or sample')
        value = engine.run(image_batch(rows[i:i + 1], payload, transform, 'cpu').numpy())[0]
        if numerical(value) != numerical(old):
            raise ValueError(f'{case} {policy}-{backend} image {i}: differs from the sealed record of {old_root.name[:16]}')
        same += 1
    record = {'status': 'pass', 'case': case, 'policy': policy, 'backend': backend, 'images_identical': same,
              'old_engine_sources': old_root.name, 'engine_sources': digest(engine_sources()), 'export': ref}
    immutable(run_root() / 'regress' / f'{case}-{policy}-{backend}-{images}-{old_root.name[:16]}.json', record)
    watch.close(images=same)
    print(f'{case} {policy}-{backend}: {same}/{images} identical to {old_root.name[:16]}', flush=True)
    return record
