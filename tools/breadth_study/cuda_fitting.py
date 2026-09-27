"""Isolate calibration-only CUDA fitting from the frozen exact runtime.

The parent validates strict inputs and accepts new graphs in the original
environment. A leased CUDA PyTorch child fits discrete weight choices only.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from public.inference.tensor import Tensor
from public.quantization.ptq.encoding import float_tensor
from tools.breadth_study.e2 import BASE, case_plan, commit_graph, recipe_sources
from tools.breadth_study.recipes import ROUNDING, adaptive_round, signed_codes
from tools.breadth_study.rounding_data import collect
from tools.experiment_b.common import unseal
from tools.phase3.common import ROOT, checked, digest, read, reference
from tools.phase3.execution_guard import verify_guard
from tools.run.exact_execution import immutable, archive_implementation


def bridge_sources():
    return {**recipe_sources(), **{p: reference(ROOT / p)['sha256'] for p in
            ('tools/breadth_study/cuda_fitting.py', 'tools/run/study_next.py')}}


def prepare(model, fmt):
    base, _, prepared = case_plan(model, fmt, 'adaptive_rounding_only')
    calibration_ref = collect(model)
    calibration = unseal(checked(calibration_ref))
    graph = read(checked(prepared['graph']))
    layers = []
    for node in graph['nodes']:
        if node['op'] not in {'conv2d', 'depthwise_conv2d', 'linear'}:
            continue
        weights = Tensor.from_document(graph['constants'][node['inputs'][1]])
        data = calibration['layers'][node['name']]
        with np.load(checked(data['data']), allow_pickle=False) as saved:
            source = saved['weights']
        if tuple(source.shape) != weights.shape or float_tensor(source, weights.encoding).codes != weights.codes:
            raise ValueError('calibration source weights do not reproduce strict stored codes')
        layers.append({'node': node['name'], 'constant': node['inputs'][1], **data})
    sources = bridge_sources()
    plan = {**base, 'version': 'cuda-calibration-bridge-v1', 'calibration': calibration_ref,
            'layers': layers, 'bridge_sources': sources, 'bridge_archive': archive_implementation(sources)}
    path = BASE / 'cuda_recipes' / digest(plan) / 'plan.json'
    immutable(path, plan)
    return path


def fit(path, *, first_only=False):
    # Deliberately no exact-runtime guard here: this process never executes
    # candidate inference. Its optimizer runtime is part of each checkpoint.
    from tools.experiment_b.classifier import configure
    configure('cuda')
    plan = unseal(path)
    if plan['bridge_sources'] != bridge_sources():
        raise ValueError('GPU fitting sources changed')
    graph = read(checked(plan['strict_graph']))
    for layer in plan['layers'][:1] if first_only else plan['layers']:
        work = path.parent / 'layers' / layer['node']
        weights = Tensor.from_document(graph['constants'][layer['constant']])
        with np.load(checked(layer['data']), allow_pickle=False) as saved:
            source, patches = saved['weights'], saved['patches']
        groups = layer['groups']
        codes, report = adaptive_round(source.reshape(groups, source.shape[0]//groups, -1), patches,
                                      np.asarray(weights.codes), weights.encoding, device='cuda', directory=work)
        output = work / 'codes.npy'
        if output.exists():
            if not np.array_equal(np.load(output, allow_pickle=False), codes):
                raise ValueError('resumed GPU fit changed discrete weight choices')
        else:
            temporary = output.with_suffix('.partial')
            with temporary.open('wb') as stream:
                np.save(stream, codes, allow_pickle=False)
            temporary.replace(output)
        immutable(work / 'result.json', {'plan': reference(path), 'codes': reference(output), 'fit': report})
        print(f"E2 GPU fit {plan['case']} {layer['node']}: {report['changed_weight_codes']} codes changed", flush=True)


def finish(path):
    plan = unseal(path)
    if plan['bridge_sources'] != bridge_sources() or plan['guard_sha256'] != verify_guard():
        raise ValueError('GPU fit parent sources or exact runtime changed')
    prepared = read(checked(plan['strict_prepared']))
    graph = read(checked(plan['strict_graph']))
    changed, fits = deepcopy(graph), {}
    for layer in plan['layers']:
        result = unseal(path.parent / 'layers' / layer['node'] / 'result.json')
        if result['plan'] != reference(path):
            raise ValueError('GPU fitting result belongs to another recipe')
        codes = np.load(checked(result['codes']), allow_pickle=False)
        weights = Tensor.from_document(graph['constants'][layer['constant']])
        with np.load(checked(layer['data']), allow_pickle=False) as data:
            source = data['weights']
        groups = layer['groups']
        shape = (groups, source.shape[0]//groups, int(np.prod(source.shape[1:])))
        bits = int(weights.encoding.format.removeprefix('int'))
        scales = np.array([float(s) for s in weights.encoding.scales]).reshape(groups, shape[1], 1)
        tiny = scales < np.finfo(np.float32).tiny
        normalized = source.reshape(shape) / np.where(tiny, 1, scales).astype(np.float32)
        lower = np.clip(np.floor(normalized), -(1 << (bits-1)), (1 << (bits-1))-1)
        upper = np.clip(np.ceil(normalized), -(1 << (bits-1)), (1 << (bits-1))-1)
        if codes.shape != shape or codes.dtype != np.uint8 or np.any(codes >= 1 << bits):
            raise ValueError('GPU fitting codes have invalid geometry or dtype')
        signed = signed_codes(codes, bits)
        original = signed_codes(weights.codes, bits).reshape(shape)
        if not np.all(np.where(tiny, signed == original, (signed == lower) | (signed == upper))):
            raise ValueError('GPU fitting changed more than permitted floor/ceil choices')
        changed['constants'][layer['constant']] = Tensor(weights.shape, tuple(codes.ravel().tolist()), weights.encoding).document()
        fits[layer['node']] = result
    changed['provenance'] = {**changed['provenance'], 'prospective_E2': ROUNDING, 'strict_graph': prepared['graph']}
    # commit_graph verifies all graph arithmetic again in the exact environment.
    return commit_graph(plan, path.parent, prepared, changed,
                        {'policy': ROUNDING, 'calibration': plan['calibration'], 'layers': fits,
                         'fitting_plan': reference(path), 'bridge_sources': plan['bridge_sources']})
