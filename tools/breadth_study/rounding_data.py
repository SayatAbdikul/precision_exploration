"""Frozen calibration-only linear patches for the E2 rounding adaptation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from tools.breadth_study.recipes import ROUNDING
from tools.experiment_b.common import dataset, digest, frozen_inputs, seal, unseal
from tools.phase3.common import ROOT, checked, reference
from tools.run.exact_execution import immutable

BASE = ROOT / 'artifacts/breadth_study/rounding_calibration_v1'


def collect(model_name):
    import torch
    from tools.experiment_b.classifier import load_model, image_batch, configure
    configure('cpu')
    context = frozen_inputs(model_name)
    manifest, population, payload = dataset('imagenet_calibration_2k')
    indices = np.linspace(0, len(population)-1, 512, dtype=int)
    rows = [population[int(i)] for i in indices]
    plan = {'model': model_name, 'context': context, 'policy': ROUNDING,
            'calibration_manifest': manifest, 'selected_images': rows,
            'implementation': reference(__file__)}
    work = BASE / digest(plan)
    immutable(work / 'plan.json', plan)
    if (work / 'summary.json').exists():
        saved = unseal(work / 'summary.json')
        for item in saved['layers'].values():
            checked(item['data'])
        return reference(work / 'summary.json')
    graph, transform, original = load_model(model_name, 'cpu')
    del original
    modules = {node.name: graph.get_submodule(node.target) for node in graph.graph.nodes
               if node.op == 'call_module' and isinstance(graph.get_submodule(node.target), (torch.nn.Conv2d, torch.nn.Linear))}
    class Observe(torch.fx.Interpreter):
        ordinal = 0
        samples = {}
        def run_node(self, node):
            if node.name in modules:
                args, _ = self.fetch_args_kwargs_from_env(node)
                x, module = args[0], modules[node.name]
                if isinstance(module, torch.nn.Linear):
                    self.samples[node.name] = x.detach().cpu().numpy().reshape(-1, 1, module.in_features).copy()
                else:
                    kh, kw = module.kernel_size
                    sh, sw = module.stride
                    ph, pw = module.padding
                    dh, dw = module.dilation
                    h, w = x.shape[-2:]
                    oh, ow = (h+2*ph-dh*(kh-1)-1)//sh+1, (w+2*pw-dw*(kw-1)-1)//sw+1
                    padded = torch.nn.functional.pad(x, (pw, pw, ph, ph))
                    seed = int(hashlib.sha256(node.name.encode()).hexdigest()[:8], 16)
                    samples = []
                    for j in range(ROUNDING['patches_per_image']):
                        position = (seed+self.ordinal*2654435761+j*104729) % (oh*ow)
                        y, col = divmod(position, ow)
                        patch = padded[0, :, y*sh:y*sh+dh*(kh-1)+1:dh, col*sw:col*sw+dw*(kw-1)+1:dw]
                        samples.append(patch.reshape(module.groups, -1).numpy().copy())
                    self.samples[node.name] = np.stack(samples)
            return super().run_node(node)
    observer = Observe(graph)
    collected = {name: [] for name in modules}
    for ordinal, row in enumerate(rows):
        if reference(payload / row['relative_path'])['sha256'] != row['sha256']:
            raise ValueError('rounding calibration image payload drift')
        destination = work / f'image-{ordinal:04d}.npz'
        meta = destination.with_suffix('.json')
        if meta.exists():
            record = unseal(meta)
            if record['sample'] != row or record['plan_sha256'] != digest(plan):
                raise ValueError('rounding calibration checkpoint changed')
            checked(record['data'])
        else:
            observer.ordinal, observer.samples = ordinal, {}
            with torch.inference_mode():
                observer.run(image_batch([row], payload, transform, 'cpu'))
            if set(observer.samples) != set(modules):
                raise ValueError('rounding calibration omits a layer')
            temporary = destination.with_suffix('.partial')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **observer.samples)
            temporary.replace(destination)
            immutable(meta, {'sample': row, 'plan_sha256': digest(plan), 'data': reference(destination)})
        with np.load(destination, allow_pickle=False) as saved:
            for name in modules:
                collected[name].append(saved[name])
        if ordinal % 32 == 0:
            print(f'{model_name}: rounding calibration {ordinal+1}/512', flush=True)
    layers = {}
    for name, module in modules.items():
        groups = module.groups if isinstance(module, torch.nn.Conv2d) else 1
        weights = module.weight.detach().numpy().copy()
        destination = work / f'layer-{name}.npz'
        with destination.open('wb') as stream:
            np.savez_compressed(stream, patches=np.concatenate(collected[name]), weights=weights)
        layers[name] = {'data': reference(destination), 'groups': groups, 'weight_shape': list(weights.shape),
                        'patches': sum(len(chunk) for chunk in collected[name])}
    summary = {'plan': reference(work / 'plan.json'), 'model': model_name, 'images': len(rows), 'layers': layers,
               'scope': 'training-calibration-only; no evaluation images or labels used'}
    immutable(work / 'summary.json', summary)
    return reference(work / 'summary.json')
