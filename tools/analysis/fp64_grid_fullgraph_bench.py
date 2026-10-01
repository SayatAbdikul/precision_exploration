"""Bounded first-image compatibility and stage timing for the versioned FP64 grid.

This is a benchmark/proof probe, not a study checkpoint.  It reads a frozen v1
image and compares every layer and diagnostic signature with the retained slow
rational record.  The result is written as an immutable standalone artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

from tools.breadth_study import useful_quality as v1
from tools.experiment_b.common import unseal
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference
from tools.run.exact_execution import event_signature, immutable

BASE = ROOT / 'artifacts/breadth_study/useful_quality_e1_v2/benchmarks'


def bench(case: str, backend: str, index: int = 0, mode: str = 'exact',
          no_oracle: bool = False):
    import numpy as np
    import torch

    from public.analysis.phase3.diagnostics import LayerDiagnostics, ReferenceSamples
    from public.inference.tensor import QUANTIZATION_OBSERVER, parse_encoding
    from public.quantization.ptq.encoding import float_tensor
    from tools.exact_execution_v2.activation_trace import ActivationTrace
    from tools.exact_execution_v2.optimized_engine import OptimizedPreparedGraph, runtime_identity
    from tools.exact_execution_v3.optimized_matched_control import OptimizedMatchedControl
    from tools.phase3.baselines import screen_rows
    from tools.phase3.controller_worker import rational_admission_guard
    from tools.phase3.runtime import load_runtime
    from tools.run.phase3_thread_benchmark import set_threads

    if mode not in {'exact', 'control'} or backend not in {'cpp', 'cuda'} or case not in {
            'resnet18/fp6_e2m3', 'resnet18/fp6_e3m2', 'resnet18/fp7_e3m3'}:
        raise ValueError('benchmark is limited to the three selected floating cases')
    if index != 0:
        raise ValueError('only the predeclared first image is benchmarked')
    enrollment = unseal(v1.BASE / 'enrollment.json')
    entry = next(row for row in enrollment['cases'] if row['case'] == case)
    plan_path = checked(entry['plan'])
    plan = unseal(plan_path)
    graph = read(checked(plan['graph']))
    population, payload, _ = screen_rows(plan['model'], campaign(), verify_images=False)
    row = population[index]
    old_path = plan_path.parent / f'{index:04d}-{mode}-{backend}.json'
    if no_oracle and (mode != 'exact' or case != 'resnet18/fp7_e3m3'
                      or old_path.is_file()):
        raise ValueError('no-oracle benchmark is limited to FP7 exact without a retained image')
    if not no_oracle and not old_path.is_file():
        raise ValueError('retained rational oracle image is required for this benchmark')
    oracle = None if no_oracle else unseal(old_path)
    if oracle is not None:
        v1.validate_record(oracle, plan, row, mode, backend, index)

    torch.set_num_interop_threads(1)
    set_threads(4)
    sample, observe, manifest = load_runtime(plan['model'], campaign())
    name = next(iter(graph['inputs']))
    t_start = time.perf_counter()
    engine = (OptimizedPreparedGraph if mode == 'exact' else OptimizedMatchedControl)(graph, backend=backend)
    setup_seconds = time.perf_counter() - t_start

    phases = {'mac': 0.0, 'store': 0.0, 'residual': 0.0, 'observer': 0.0,
              'quantization_observer': 0.0}
    counts = {'mac': 0, 'store': 0, 'residual': 0, 'observer': 0,
              'quantization_observer': 0}

    def instrument(target, method, key):
        original = getattr(target, method)

        def measured(*args, **kwargs):
            start = time.perf_counter()
            result = original(*args, **kwargs)
            phases[key] += time.perf_counter() - start
            counts[key] += 1
            return result

        setattr(target, method, measured)

    instrument(engine._operators.native, 'flex_conv2d', 'mac')
    instrument(engine._operators.native, 'flex_gemm', 'mac')
    instrument(engine._operators, '_store', 'store')
    instrument(engine._residuals, 'add', 'residual')

    start = time.perf_counter()
    fp32, _ = sample(payload / row['relative_path'])
    if list(fp32.shape) != manifest['input_shape']:
        raise ValueError('preprocessing shape changed')
    refs = ReferenceSamples(campaign()['diagnostics']['sample_elements_per_layer_image'])
    observe(fp32, refs)
    tensor = float_tensor(fp32.numpy(), parse_encoding(graph['inputs'][name]))
    prep_seconds = time.perf_counter() - start

    diagnostics = LayerDiagnostics(refs)

    class TimedObserver:
        def begin_node(self, node):
            return diagnostics.begin_node(node)

        def __call__(self, node, value):
            tick = time.perf_counter()
            result = diagnostics(node, value)
            phases['observer'] += time.perf_counter() - tick
            counts['observer'] += 1
            return result

        def quantization(self, values, tensor):
            tick = time.perf_counter()
            result = diagnostics.quantization(values, tensor)
            phases['quantization_observer'] += time.perf_counter() - tick
            counts['quantization_observer'] += 1
            return result

    timer = TimedObserver()
    with ActivationTrace(), rational_admission_guard():
        token = QUANTIZATION_OBSERVER.set(timer.quantization)
        start = time.perf_counter()
        try:
            result = engine.execute({name: tensor}, observer=timer)
        finally:
            QUANTIZATION_OBSERVER.reset(token)
        execution_seconds = time.perf_counter() - start
    output = next(iter(result['outputs'].values()))
    values = output.values()
    prediction = sorted(range(len(values)), key=lambda i: (-values[i], i))[:5]
    numerical = {'layers': result['layers'], 'output_sha256': digest(output.document()),
                 'prediction': prediction, 'diagnostic_signature': event_signature(diagnostics.records)}
    matches = {key: numerical[key] == oracle[key] for key in numerical} if oracle else None
    if matches is not None and not all(matches.values()):
        raise ValueError(f'optimized full graph differs from retained rational oracle: {matches}')
    if mode == 'exact' and (len(result['execution_modes']) != 21 or set(result['execution_modes'].values()) != {
            'certified_fp64_grid_v2'}):
        raise ValueError('selected graph did not use certified grid for all 21 MAC nodes')
    total = setup_seconds + prep_seconds + execution_seconds
    document = {'version': 'fp64-grid-v2-first-image-benchmark', 'case': case,
                'mode': mode,
                'backend': backend, 'sample': row, 'plan': reference(plan_path),
                'oracle': reference(old_path) if oracle is not None else None,
                'runtime_identity': runtime_identity(),
                'optimized_control_source_sha256': (
                    reference(ROOT / 'tools/exact_execution_v3/optimized_matched_control.py')['sha256']
                    if mode == 'control' else None),
                'numerical': numerical, 'matches': matches,
                'execution_modes': result['execution_modes'],
                'seconds': {'setup': setup_seconds, 'input_preparation': prep_seconds,
                            'execution': execution_seconds, 'total_new_compute': total,
                            'phases': phases, 'phase_counts': counts},
                'old_execution_seconds': oracle.get('seconds') if oracle is not None else None,
                'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'workload_context': 'campaign-concurrent if other host processes are active'}
    variant = ('optimized-control-v3-sourcebound' if mode == 'control' else
               'exact-no-oracle' if no_oracle else mode)
    path = BASE / f'{case.replace("/", "-")}-{variant}-{backend}-image{index}.json'
    immutable(path, document)
    return {'path': str(path.relative_to(ROOT)), 'case': case, 'mode': mode, 'backend': backend,
            'old_execution_seconds': document['old_execution_seconds'], 'seconds': document['seconds'],
            'peak_rss_kib': document['peak_rss_kib'], 'matches': matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case')
    parser.add_argument('backend', choices=('cpp', 'cuda'))
    parser.add_argument('--mode', choices=('exact', 'control'), default='exact')
    parser.add_argument('--no-oracle', action='store_true')
    args = parser.parse_args()
    print(json.dumps(bench(args.case, args.backend, mode=args.mode,
                           no_oracle=args.no_oracle), indent=2), flush=True)


if __name__ == '__main__':
    main()
