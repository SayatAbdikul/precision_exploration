"""Replay of lane L1's B2 simulator next to the exact engine, and the MB2 case gate.

B2's own engine (tools/experiment_b2, original CUDA runtime, batches of eight) is rebuilt from the export's
configuration and run on the leading development images. Per image the replay records B2's logits under both
top-k conventions, the codes B2 stores at every quantising boundary, and how many of those codes differ from
the engine's exact (wide) arm, which is re-run here and must equal its sealed panel record.
This file is not an engine source.
"""
from __future__ import annotations
import time
from types import SimpleNamespace
import numpy as np
from .common import (digest, unseal, reference, immutable, array_hash, run_root, engine_sources, Stopwatch,
                     under_gpu_lock)
from .worker import locate, POLICY_ARMS, GATE_IMAGES, REPLAY_IMAGES

BATCH = 8


def b2_engine(ex):
    """B2's interpreter for the exported configuration; the configuration identity must be reproduced."""
    from tools.experiment_b2 import frozen  # noqa: F401  (registers the frozen recipes)
    from tools.experiment_b2.runner import build
    prov = ex['provenance']
    args = SimpleNamespace(model=ex['model'], format=ex['format'], recipe=prov['B2_recipe_name'], device='cuda')
    setup, _, engine, _, _, identity = build(args, seal_configuration=False)
    if identity != prov['B2_configuration_sha256']:
        raise ValueError('B2 configuration identity is not reproduced')
    return setup, engine


def traced_run(engine, tensor):
    """B2 output plus the level index B2 stores at every quantising node (instrumentation must not change it)."""
    import torch
    indices = {}

    def run_node(node):
        raw = torch.fx.Interpreter.run_node(engine, node)
        row = engine.plan.get(node.name)
        if row is not None and row['quantizes']:
            quantizer, scale = engine.quantizers[row['signedness']], engine.scales[node.name]
            indices[node.name] = quantizer.indices(raw, scale).cpu().numpy()
            return quantizer(raw, scale)
        return raw

    with torch.inference_mode():
        expected = engine.run(tensor)
        engine.run_node = run_node
        try:
            observed = engine.run(tensor)
        finally:
            del engine.run_node
    if not torch.equal(expected, observed):
        raise ValueError('B2 trace instrumentation changes the output')
    return observed, indices


def replay(case, target):
    import torch
    from tools.experiment_b.classifier import image_batch
    from .engine import stable_top5, numerical
    from .worker import context
    if target > REPLAY_IMAGES:
        raise ValueError('the MB2 protocol replays 32 images')
    watch = Stopwatch('b2-replay', [case, target])
    ex, arrays, ref, graph, transform, rows, payload, engine = context(case, 'cuda', 'wide')
    setup, b2 = b2_engine(ex)
    if digest(setup['rows']) != ex['retained']['ordered_samples_sha256'] or setup['rows'] != rows:
        raise ValueError('B2 and engine sample lists differ')
    retained = {r['index']: r['top5'] for r in ex['retained']['B2_prefix']}
    stored = [n['name'] for n in ex['nodes'] if n.get('store') is not None]
    books = {n['name']: engine.books[n['store']['codebook']] for n in ex['nodes'] if n.get('store') is not None}
    folder = run_root() / case / 'B2'; computed = 0
    for start in range(0, target, BATCH):
        paths = [folder / f'{i:04d}.json' for i in range(start, start + BATCH)]
        if all(p.exists() for p in paths):
            continue
        cached = np.array(setup['inputs'][start:start + BATCH], dtype=np.float32)
        inputs = image_batch(rows[start:start + BATCH], payload, transform, 'cpu').numpy()
        same_inputs = bool(np.array_equal(cached.view('u4'), inputs.view('u4')))
        tick = time.perf_counter()
        observed, indices = traced_run(b2, torch.from_numpy(cached).to('cuda'))
        elapsed = time.perf_counter() - tick
        if set(indices) != set(stored):
            raise ValueError('B2 quantising nodes differ from the stored boundaries of the export')
        top = observed.topk(5, dim=1).indices.cpu().tolist()
        logits = observed.double().cpu().numpy()
        capture = {}
        exact = engine.run(inputs, trace='full', capture=capture)
        for i, path in enumerate(paths, start):
            j = i - start
            sealed = unseal(run_root() / case / 'wide-cuda' / f'{i:04d}.json')
            if numerical(exact[j]) != numerical(sealed) or sealed['sample'] != rows[i]:
                raise ValueError('the exact arm re-run in the replay differs from its sealed panel record')
            if i in retained and retained[i] != top[j]:
                raise ValueError('B2 retained Top-5 is not reproduced by the replay')
            layers, changed = {}, {}
            for key in stored:
                codes = books[key].codes[indices[key][j:j + 1]]
                layers[key] = {'codes': array_hash(codes)}
                changed[key] = [int((codes != capture[key][j:j + 1]).sum()), int(codes.size)]
            record = {'export': ref, 'sample': rows[i], 'index': i, 'batch_start': start, 'batch_images': BATCH,
                      'top5': top[j], 'retained_top5': retained.get(i), 'stable_top5': stable_top5(logits[j]),
                      'top1_tie_count': int(np.sum(logits[j] == logits[j].max())),
                      'top1_tied_classes': [int(c) for c in np.flatnonzero(logits[j] == logits[j].max())],
                      'output': array_hash(logits[j:j + 1]), 'layers': layers,
                      'changed_codes_vs_wide': changed,
                      'max_abs_logit_difference_vs_wide': float(np.abs(logits[j] - capture['output'][j]).max()),
                      'inputs_bit_identical_to_engine_inputs': same_inputs,
                      'execution_batch_seconds': elapsed, 'gpu_lock_declared': under_gpu_lock(),
                      'engine_sources': digest(engine_sources()),
                      'scope': 'B2 simulator batch-8 replay; top5 is torch.topk order, stable_top5 is the contract '
                               'order (lowest class index first) on the same logits; codes compared with the '
                               'exact arm at every storing boundary'}
            immutable(path, record)
            computed += 1
        print(f'{case} B2 replayed {start + BATCH}/{target}', flush=True)
    watch.close(computed_images=computed)


def gate(case):
    """MB2 gate of one case (PROTOCOL_MB2 witnesses). Paired differences are reported, never gated."""
    from .engine import numerical
    from .export import load_export
    ex, _, ref = load_export(case, locate(case))
    root = run_root() / case
    certificate = unseal(root / 'certificate.json')
    if certificate['export'] != ref:
        raise ValueError('certificate belongs to another export')
    proof = root / 'wide-reference.json'
    if unseal(proof)['export'] != ref:
        raise ValueError('whole-graph reference belongs to another export')
    conformance = [run_root() / 'conformance' / f'{name}.json' for name in sorted(ex['codebooks'])]
    for name, path in zip(sorted(ex['codebooks']), conformance):
        record = unseal(path)
        if record['status'] != 'pass' or record['codebook'] != ex['codebooks'][name]:
            raise ValueError('primitive conformance missing for a codebook of the export')
    if 'weight_check' not in ex['provenance'] or 'codebook_check' not in ex['provenance']:
        raise ValueError('adapter checks missing')
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
    replayed = [unseal(root / 'B2' / f'{i:04d}.json') for i in range(REPLAY_IMAGES)]
    retained = sum(r['retained_top5'] is not None for r in replayed)
    if any(r['export'] != ref or (r['retained_top5'] is not None and r['retained_top5'] != r['top5']) for r in replayed):
        raise ValueError('B2 replay does not reproduce the retained Top-5')
    # Review 3, non-blocking 4: every replayed image must have a retained B2 prediction, and the replay inputs must
    # be bit-identical to the engine inputs.
    if retained != REPLAY_IMAGES:
        raise ValueError(f'B2 retained predictions exist for {retained} of {REPLAY_IMAGES} replayed images; the gate needs all')
    if not all(r['inputs_bit_identical_to_engine_inputs'] for r in replayed):
        raise ValueError('B2 replay inputs are not bit-identical to the engine inputs')
    # Contract 2.2: the whole-graph witness of the control arm on the first image.
    witness = root / 'graph-witness-control-cpp-0000.json'
    if not witness.exists() or unseal(witness)['status'] != 'pass' \
            or unseal(witness)['record'] != reference(root / 'control-cpp' / '0000.json'):
        raise ValueError('whole-graph control-arm witness missing or failed')
    record = {'status': 'pass', 'case': case, 'export': ref, 'images_per_arm': GATE_IMAGES, 'arms': list(POLICY_ARMS),
              'records': refs, 'primitive_conformance': [reference(p) for p in conformance],
              'independent_wide_graph': reference(proof), 'certificate': reference(root / 'certificate.json'),
              'B2_replay_images': REPLAY_IMAGES, 'B2_retained_top5_reproduced_images': retained,
              'replay_inputs_bit_identical_images': REPLAY_IMAGES, 'control_graph_witness': reference(witness),
              'graph_check': ex['provenance'].get('graph_check'),
              'adapter_weight_check': ex['provenance']['weight_check'],
              'engine_sources': digest(engine_sources()),
              'scope': 'contract 2.1, repaired-recipe graph; bit identity with the B2 simulator is not a gate'}
    immutable(root / 'gate.json', record)
    print(case + ' MB2 gate PASS', flush=True)
    return record
