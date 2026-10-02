"""Original B batch-eight runtime replay (read-only anchor) with separate traces."""
from __future__ import annotations
import time
import numpy as np
import torch
from tools.experiment_b.classifier import QDQInterpreter, quantized_node
from .common import ROOT, digest, unseal, reference, checked, immutable, array_hash, run_root, Stopwatch, under_gpu_lock
from .codebooks import Codebook, quantize
from .engine import stable_top5
from .export import load_export, b_calibration


class TracedB(QDQInterpreter):
    def __init__(self, engine, book, nodes):
        super().__init__(engine.module, engine.quantizer, {k: float(v.cpu()) for k, v in engine.scales.items()})
        self.book = book; self.traces = []; self.codes = {}
        self.nodes = {x['name']: x for x in nodes}

    def run_node(self, node):
        raw = torch.fx.Interpreter.run_node(self, node)
        if node.op == 'output':
            return raw
        key = node.name; quantized = quantized_node(self.module, node)
        if not self.traces:
            self.traces = [{} for _ in range(raw.shape[0])]
        if quantized:
            result = self.quantizer(raw, self.scales[key])
            codes, _, diagnostics = quantize(raw.detach().cpu().numpy(), float(self.scales[key].cpu()), self.book, b_weight=True)
        else:
            result = raw
            codes = self.codes[self.nodes[key]['inputs'][0]].reshape(tuple(raw.shape))
            diagnostics = [{'skipped': True}] * raw.shape[0]
        self.codes[key] = codes
        for i, trace in enumerate(self.traces):
            trace[key] = {'codes': array_hash(codes[i:i + 1]), 'diagnostics': diagnostics[i]}
        return result


def replay(case, target):
    from tools.experiment_b.classifier import configure, load_model, prepare_qdq, image_batch
    from tools.experiment_b.runner import runtime
    from tools.experiment_b.common import dataset
    watch = Stopwatch('replay', [case, target])
    ex, _, ref = load_export(case); configure('cuda')
    prov = ex['provenance']
    if runtime('cuda') != prov['B_runtime']:
        raise ValueError('cannot reproduce B in a changed runtime')
    model, fmt, recipe = ex['model'], ex['format'], ex['recipe']
    graph, transform, original = load_model(model, 'cuda'); del original
    config = unseal(checked(prov['B_configuration']))
    arrays, maxima, _ = b_calibration(model, config, need_samples=recipe != 'maxabs')
    with torch.inference_mode():
        engine, scales = prepare_qdq(graph, fmt, recipe, arrays, maxima, 'cuda')
    if scales != config['scales']:
        raise ValueError('replay scales changed')
    book = Codebook(ex['codebooks'][fmt])
    _, rows, payload = dataset('imagenet_screen_1k'); folder = run_root() / case / 'B'
    if target > len(ex['retained']['B_prefix']):
        raise ValueError('replay target exceeds the retained B prefix in the export')
    computed = 0
    for start in range(0, target, 8):
        paths = [folder / f'{i:04d}.json' for i in range(start, start + 8)]
        if all(p.exists() for p in paths):
            for i, p in enumerate(paths, start):
                r = unseal(p)
                if r['export'] != ref or r['sample'] != rows[i] or r['top5'] != ex['retained']['B_prefix'][i]['top5']:
                    raise ValueError('B trace checkpoint drift')
            continue
        inputs = image_batch(rows[start:start + 8], payload, transform, 'cuda')
        tracer = TracedB(engine, book, ex['nodes']); tick = time.perf_counter()
        with torch.inference_mode():
            expected = engine.run(inputs); observed = tracer.run(inputs); baseline = graph(inputs)
        if not torch.equal(expected, observed):
            raise ValueError('B trace instrumentation changes output')
        top = observed.topk(5, dim=1).indices.cpu().tolist(); fp = baseline.topk(5, dim=1).indices.cpu().tolist()
        logits = observed.double().cpu().numpy(); fp_logits = baseline.double().cpu().numpy()
        elapsed = time.perf_counter() - tick
        for i, path in enumerate(paths, start):
            j = i - start
            if top[j] != ex['retained']['B_prefix'][i]['top5'] or fp[j] != ex['retained']['FP32_prefix'][i]['top5']:
                raise ValueError('original batch-8 retained Top-5 reproduction failed')
            record = {'export': ref, 'sample': rows[i], 'index': i, 'batch_start': start, 'batch_images': 8,
                      'top5': top[j], 'FP32_top5': fp[j], 'layers': tracer.traces[j],
                      'stable_top5': stable_top5(logits[j]),
                      'top1_tie_count': int(np.sum(logits[j] == logits[j].max())),
                      'top1_tied_classes': [int(c) for c in np.flatnonzero(logits[j] == logits[j].max())],
                      'FP32_top1_tie_count': int(np.sum(fp_logits[j] == fp_logits[j].max())),
                      'output': array_hash(logits[j:j + 1]),
                      'execution_batch_seconds': elapsed, 'gpu_lock_declared': under_gpu_lock(),
                      'scope': 'original B batch-8 Top-5 reproduced; separate layer-code trace; tie diagnostics added'}
            immutable(path, record)
            computed += 1
        print(f'{case} B reproduced {start + 8}/{target}', flush=True)
    watch.close(computed_images=computed)
