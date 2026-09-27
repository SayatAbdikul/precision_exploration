"""Explicit E2 recipe adaptations; calibration only, no evaluation fitting."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from public.inference.tensor import Encoding, Tensor, parse_encoding
from public.quantization.graph.executable import validate_graph
from public.quantization.ptq.encoding import scalar_float_codes
from tools.experiment_b.common import seal, unseal
from tools.phase3.common import ROOT, checked, digest, read, reference
from tools.run.exact_execution import immutable

CLIPPING = {
    'name': 'activation_MSE_Lloyd_refinement_v1',
    'changed_factor': 'activation scale search only; original calibration samples and weight codes retained',
    'initial_scales': ['original', 'maxabs', 'abs_percentile90', 'abs_percentile99', 'abs_percentile99_9'],
    'updates': 'RNE integer assignments followed by least-squares scale; 25 iterations per start',
    'selection': 'minimum FP64 calibration reconstruction MSE, then smallest scale; original included',
    'important': 'strict A already uses 100-coarse/50-fine MSE search; this is a refinement, not first use of MSE',
}
ROUNDING = {
    'name': 'symmetric_patch_adaptive_rounding_v1',
    'source_method': 'https://proceedings.mlr.press/v119/nagel20a.html',
    'scope': 'AdaRound-inspired symmetric linear patch reconstruction adaptation, not a paper reproduction',
    'changed_factor': 'stored weight floor/ceil choices only; original weight/activation scales and all other graph attrs retained',
    'calibration_selection': '512 evenly spaced indices from the SHA-ordered frozen training calibration manifest',
    'patches_per_image': 2, 'iterations': 1000, 'batch_size': 128, 'seed': 20260926,
    'optimizer': 'Adam', 'learning_rate': .001, 'warmup_fraction': .2,
    'stretch': [-.1, 1.1], 'beta_cosine': [20, 2], 'regularization': .01,
    'loss': 'symmetric linear output MSE normalized by mean squared full-precision target, plus mean rounding penalty',
    'deployment': 'hard decisions only; retain original layer codes if full calibration patch MSE increases',
    'tiny_scale_channels': 'keep original codes when the weight scale is below FP32 normal range',
    'differences_from_paper': ['frozen per-channel scales', 'sampled linear patches', 'symmetric inputs',
                               'no activation function in local loss', '1000 steps', 'normalized loss'],
}


def signed_codes(codes, bits):
    codes = np.asarray(codes, dtype=np.int64)
    return np.where(codes >= 1 << (bits-1), codes-(1 << bits), codes)


def refined_scale(values, format_name, original):
    values = np.asarray(values, dtype=np.float64).ravel()
    if not values.size or not np.isfinite(values).all():
        raise ValueError('clipping requires finite nonempty calibration samples')
    bits = int(format_name.removeprefix('int'))
    lo, hi = -(1 << (bits-1)), (1 << (bits-1))-1
    def score(scale):
        if not np.isfinite(scale) or scale <= 0:
            return float('inf')
        codes = scalar_float_codes(values, Encoding(format_name, (str(scale),)))
        return float(np.mean((values-signed_codes(codes, bits)*scale)**2))
    initial = float(original)
    candidates = {initial}
    if np.any(values):
        seeds = [initial, np.max(np.abs(values))/hi]
        seeds += [float(np.quantile(np.abs(values), q))/hi for q in (.9, .99, .999)]
        for scale in seeds:
            if scale <= 0:
                continue
            for _ in range(25):
                assignments = np.clip(np.rint(values/scale), lo, hi)
                denominator = float(np.dot(assignments, assignments))
                if not denominator:
                    break
                proposed = float(np.dot(values, assignments)/denominator)
                if not np.isfinite(proposed) or proposed <= 0:
                    break
                candidates.add(proposed)
                if proposed == scale:
                    break
                scale = proposed
    # Use exact-boundary encoding for every evaluated candidate. FP64 is used
    # for the declared fitting objective, not the deployed integer arithmetic.
    best = min((score(s), s) for s in candidates)
    return {'scale': str(best[1]), 'mse': best[0], 'original_mse': score(initial), 'candidates': len(candidates)}


def clipping_graph(prepared):
    config = read(checked(prepared['configuration']))
    graph = read(checked(prepared['graph']))
    calibration = read(checked(config['calibration']))
    inventory = read(ROOT / f"artifacts/phase3/calibration/{config['model']}.json")
    row = next(r for r in inventory['records'] if r['format'] == prepared['format'])
    observation_path = checked(row['observations'])
    observation = read(observation_path)
    if observation['observations'] != calibration['observations']:
        raise ValueError('clipping observations differ from strict A')
    fits, encodings = {}, {}
    with np.load(observation_path.with_suffix('.npz'), allow_pickle=False) as saved:
        for name, encoding in calibration['encodings'].items():
            values = saved[name]
            expected = calibration['observations']['nodes'][name]['sample_sha256']
            if hashlib.sha256(values.astype('<f4').tobytes()).hexdigest() != expected:
                raise ValueError('strict calibration sample hash mismatch')
            fits[name] = refined_scale(values, prepared['format'], parse_encoding(encoding).scales[0])
            encodings[name] = Encoding(prepared['format'], (fits[name]['scale'],)).document()
    changed = deepcopy(graph)
    changed['inputs'] = {name: encodings[name] for name in graph['inputs']}
    for node in changed['nodes']:
        if 'output' in node['attrs']:
            node['attrs']['output'] = encodings[node['name']]
        if 'alignment' in node['attrs']:
            node['attrs']['alignment'] = encodings[node['name']]
    changed['provenance'] = {**changed['provenance'], 'prospective_E2': CLIPPING, 'strict_graph': prepared['graph']}
    validate_graph(changed)
    if changed['constants'] != graph['constants']:
        raise ValueError('activation clipping changed weight codes')
    return changed, {'policy': CLIPPING, 'strict_calibration': config['calibration'],
                     'observations': row['observations'], 'observation_arrays': reference(observation_path.with_suffix('.npz')),
                     'fits': fits}


def adaptive_round(weights, patches, original_codes, encoding, *, device, directory,
                   iterations=ROUNDING['iterations'], seed=ROUNDING['seed']):
    """Fit only floor/ceil bits; resume Adam and the minibatch RNG together."""
    import torch
    from math import cos, pi
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # weights [G,O,K], inputs [N,G,K], per-output-channel original scales.
    shape = tuple(weights.shape)
    if len(shape) != 3 or patches.ndim != 3 or patches.shape[1:] != (shape[0], shape[2]):
        raise ValueError('adaptive rounding group/patch geometry mismatch')
    bits = int(encoding.format.removeprefix('int'))
    lo, hi = -(1 << (bits-1)), (1 << (bits-1))-1
    raw_scales = np.array([float(s) for s in encoding.scales]).reshape(shape[0], shape[1], 1)
    frozen_channels = raw_scales < np.finfo(np.float32).tiny
    scales = torch.tensor(np.where(frozen_channels, 1, raw_scales), dtype=torch.float32, device=device)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    x = torch.tensor(patches, dtype=torch.float32, device=device)
    if not torch.isfinite(w).all() or not torch.isfinite(x).all() or not (scales > 0).all():
        raise ValueError('nonfinite adaptive-rounding inputs')
    normalized = w/scales
    lower = torch.floor(normalized)
    fraction = (normalized-lower).clamp(0, 1)
    alpha = torch.log((fraction+.1)/(1.1-fraction)).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([alpha], lr=ROUNDING['learning_rate'])
    generator = torch.Generator(device='cpu').manual_seed(seed)
    start = 0
    context = {'policy': ROUNDING, 'iterations': iterations, 'seed': seed, 'shape': list(shape),
               'device': device, 'torch': str(torch.__version__), 'implementation': reference(__file__),
               'weights_sha256': hashlib.sha256(np.asarray(weights, dtype='<f4').tobytes()).hexdigest(),
               'patches_sha256': hashlib.sha256(np.asarray(patches, dtype='<f4').tobytes()).hexdigest(),
               'encoding': encoding.document(), 'original_codes_sha256': hashlib.sha256(np.asarray(original_codes, dtype=np.uint8).tobytes()).hexdigest()}
    immutable(directory / 'context.json', context)
    for meta_path in sorted(directory.glob('checkpoint-*.json'), reverse=True):
        meta = unseal(meta_path)
        if meta['context_sha256'] != digest(context):
            raise ValueError('rounding optimizer checkpoint context changed')
        payload = torch.load(checked(meta['payload']), map_location=device, weights_only=True)
        with torch.no_grad():
            alpha.copy_(payload['alpha'])
        optimizer.load_state_dict(payload['optimizer'])
        generator.set_state(payload['rng'].cpu())
        start = payload['step']
        break
    def output(inputs, weight):
        return torch.einsum('ngk,gok->ngo', inputs, weight)
    with torch.no_grad():
        target = output(x, w)
        normalizer = target.square().mean().clamp_min(1e-12)
    for step in range(start, iterations):
        choice = torch.randint(len(x), (min(ROUNDING['batch_size'], len(x)),), generator=generator).to(device)
        h = (torch.sigmoid(alpha)*1.2-.1).clamp(0, 1)
        soft = (lower+h).clamp(lo, hi)*scales
        loss = (output(x[choice], soft)-target[choice]).square().mean()/normalizer
        if step >= iterations*ROUNDING['warmup_fraction']:
            progress = (step-iterations*.2)/(iterations*.8)
            beta = 2+18*(1+cos(pi*progress))/2
            loss = loss + ROUNDING['regularization']*(1-(2*h-1).abs().pow(beta)).mean()
        if not torch.isfinite(loss):
            raise ValueError('adaptive rounding fitting diverged')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step+1) % 100 == 0 or step+1 == iterations:
            destination = directory / f'checkpoint-{step+1:04d}.pt'
            temporary = destination.with_suffix('.partial')
            torch.save({'alpha': alpha.detach(), 'optimizer': optimizer.state_dict(),
                        'rng': generator.get_state(), 'step': step+1}, temporary)
            temporary.replace(destination)
            seal(destination.with_suffix('.json'), {'context_sha256': digest(context), 'payload': reference(destination)})
    with torch.no_grad():
        hard = (lower+(alpha >= 0).float()).clamp(lo, hi)
        original = torch.tensor(signed_codes(original_codes, bits).reshape(shape), device=device, dtype=torch.float32)
        frozen = torch.tensor(frozen_channels, device=device)
        hard = torch.where(frozen, original, hard)
        original_error = float((output(x, original*scales)-target).square().mean())
        error = float((output(x, hard*scales)-target).square().mean())
        retained = error > original_error
        selected = original if retained else hard
        codes = (selected.cpu().numpy().astype(np.int64) & ((1 << bits)-1)).astype(np.uint8)
    return codes, {'policy': ROUNDING, 'original_patch_mse': original_error, 'fitted_patch_mse': error,
                   'frozen_tiny_scale_channels': int(frozen_channels.sum()),
                   'retained_original_codes': retained, 'changed_weight_codes': int(np.count_nonzero(codes.ravel() != np.asarray(original_codes).ravel())),
                   'optimization_context': reference(directory / 'context.json')}
