"""Prepare the two versioned E2 interventions without changing strict A."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from public.inference.tensor import Tensor
from public.quantization.graph.executable import validate_graph
from tools.breadth_study.recipes import CLIPPING, ROUNDING, clipping_graph, adaptive_round
from tools.breadth_study.rounding_data import collect
from tools.experiment_b.common import unseal
from tools.phase3.common import ROOT, checked, digest, read, reference, write, campaign
from tools.phase3.execution_guard import verify_guard
from tools.phase3.preparation_inventory import preparation_records
from tools.run.exact_execution import immutable, archive_implementation

BASE = ROOT / 'artifacts/breadth_study/e2_v1'
CASES = [('resnet18', 'int4'), ('mobilenet_v2', 'int4'), ('mobilenet_v2', 'int5'), ('mobilenet_v2', 'int6')]


def recipe_sources():
    return {p: reference(ROOT / p)['sha256'] for p in (
        'tools/breadth_study/recipes.py', 'tools/breadth_study/rounding_data.py',
        'tools/breadth_study/e2.py', 'tools/run/e2.py')}


def case_plan(model, fmt, arm):
    prepared = preparation_records()[f'{model}/{fmt}']
    src = recipe_sources()
    plan = {'case': f'{model}/{fmt}', 'model': model, 'format': fmt, 'arm': arm,
            'strict_prepared': reference(checked(prepared['configuration']).parent / 'prepared.json'),
            'strict_graph': prepared['graph'], 'matrix': reference(ROOT / 'public/experiments/configs/breadth-study/comparison-matrix-v1.json'),
            'policy': CLIPPING if arm == 'MSE_clipping_only' else ROUNDING,
            'sources': src, 'source_archive': archive_implementation(src), 'guard_sha256': verify_guard()}
    work = BASE / 'recipes' / digest(plan)
    immutable(work / 'plan.json', plan)
    return plan, work, prepared


def commit_graph(plan, work, prepared, graph, fit):
    from tools.phase3.acceptance import prove_integer_graph
    if plan['sources'] != recipe_sources() or plan['guard_sha256'] != verify_guard():
        raise ValueError('E2 recipe source changed during fitting')
    validate_graph(graph)
    manifest = read(checked(campaign()['inputs'][plan['model']]))
    shape_ref = read(ROOT / 'results/summaries/phase3-shapes.json')['models'][plan['model']]
    shapes = read(checked(shape_ref))['shapes']
    proof = prove_integer_graph(graph, {k: manifest['input_shape'] for k in graph['inputs']}, shapes)
    if proof['status'] != 'accepted':
        immutable(work / 'blocked-proof.json', proof)
        raise ValueError('changed recipe graph fails the original integer arithmetic proof')
    path = work / 'graph.json'
    if path.exists():
        if read(path) != graph:
            raise ValueError('changed recipe graph is immutable')
    else:
        write(path, graph)
    immutable(work / 'fit.json', fit)
    immutable(work / 'proof.json', {'proof': proof, 'graph': reference(path), 'shape_evidence': shape_ref,
                                   'implementation': reference(ROOT / 'tools/phase3/acceptance.py')})
    report = {'plan': reference(work / 'plan.json'), 'graph': reference(path), 'fit': reference(work / 'fit.json'),
              'proof': reference(work / 'proof.json'), 'status': 'static_integer_proof_passed; native_pilot_required',
              'model': plan['model'], 'format': plan['format'], 'arm': plan['arm']}
    immutable(work / 'prepared.json', report)
    print(f"{plan['case']} {plan['arm']}: graph prepared, native validation required", flush=True)
    return reference(work / 'prepared.json')


def clipping(model, fmt):
    plan, work, prepared = case_plan(model, fmt, 'MSE_clipping_only')
    if (work / 'prepared.json').exists():
        saved = unseal(work / 'prepared.json')
        for key in ('graph', 'fit', 'proof'):
            checked(saved[key])
        return reference(work / 'prepared.json')
    graph, fit = clipping_graph(prepared)
    return commit_graph(plan, work, prepared, graph, fit)


def rounding(model, fmt, *, device='cuda'):
    from tools.experiment_b.classifier import configure
    configure(device)
    plan, work, prepared = case_plan(model, fmt, 'adaptive_rounding_only')
    if (work / 'prepared.json').exists():
        return reference(work / 'prepared.json')
    calibration_ref = collect(model)
    calibration = unseal(checked(calibration_ref))
    graph = read(checked(prepared['graph']))
    changed = deepcopy(graph)
    fits = {}
    for node in changed['nodes']:
        if node['op'] not in {'conv2d', 'depthwise_conv2d', 'linear'}:
            continue
        key = node['inputs'][1]
        weights = Tensor.from_document(graph['constants'][key])
        layer = calibration['layers'][node['name']]
        with np.load(checked(layer['data']), allow_pickle=False) as saved:
            source, patches = saved['weights'], saved['patches']
        if tuple(source.shape) != weights.shape:
            raise ValueError('rounding source weight layout differs from strict graph')
        # Re-encode source weights with original scales before fitting. This
        # proves the patch loader and strict A use the same folded constants.
        from public.quantization.ptq.encoding import float_tensor
        if float_tensor(source, weights.encoding).codes != weights.codes:
            raise ValueError('rounding source weights do not reproduce strict stored codes')
        groups = layer['groups']
        source = source.reshape(groups, source.shape[0]//groups, -1)
        codes, fit = adaptive_round(source, patches, np.asarray(weights.codes), weights.encoding,
                                    device=device, directory=work / 'layers' / node['name'])
        changed['constants'][key] = Tensor(weights.shape, tuple(codes.ravel().tolist()), weights.encoding).document()
        fits[node['name']] = fit
        print(f"{plan['case']} rounding {node['name']}: {fit['changed_weight_codes']} codes changed", flush=True)
    changed['provenance'] = {**changed['provenance'], 'prospective_E2': ROUNDING, 'strict_graph': prepared['graph']}
    if changed['inputs'] != graph['inputs'] or changed['nodes'] != graph['nodes']:
        raise ValueError('weight rounding changed graph operators or activation scales')
    return commit_graph(plan, work, prepared, changed, {'policy': ROUNDING, 'calibration': calibration_ref, 'layers': fits})
