"""One arm/backend per v2 invocation, with checked v1 evidence migration."""
from __future__ import annotations

from datetime import datetime, timezone
import resource
import time

from tools.experiment_b.common import atomic_json, unseal
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference
from tools.run.exact_execution import immutable, event_signature
from tools.breadth_study import useful_quality as v1


def validate_record(record, plan, sample, mode, backend, index):
    if (record['job_sha256'] != digest(plan) or record['sample'] != sample
            or record['mode'] != mode or record['backend'] != backend
            or record['index'] != index or record['graph_sha256'] != plan['graph_sha256']
            or set(record['layers']) != set(plan['deployed_nodes'])):
        raise ValueError('v2 checkpoint identity, population or layer coverage mismatch')
    if len(record['prediction']) != 5 or len(set(record['prediction'])) != 5:
        raise ValueError('invalid v2 top-five prediction')
    if record['diagnostic_signature'] != event_signature(record['diagnostics']):
        raise ValueError('v2 quantizer diagnostic signature changed')
    if plan['format'] != 'int8' and mode == 'exact':
        strategies = record.get('execution_modes', {})
        if (set(strategies) != set(plan['mac_nodes'])
                or any(strategy != 'certified_fp64_grid_v2' for strategy in strategies.values())):
            raise ValueError('floating exact checkpoint did not use every certified native MAC')
    if plan['format'] != 'int8' and mode == 'control':
        strategies = record.get('execution_modes', {})
        if (set(strategies) != set(plan['mac_nodes'])
                or any(not strategy.startswith('matched_fp32_sequential_')
                       for strategy in strategies.values())):
            raise ValueError('floating control checkpoint did not use every matched FP32 MAC')
    if 'v1_record' in record:
        from tools.breadth_study.useful_quality_v2 import verify_v1_archive
        parent = unseal(checked(plan['parent_plan']))
        verify_v1_archive(parent)
        old = unseal(checked(record['v1_record']))
        v1.validate_record(old, parent, sample, mode, backend, index)
        keys = ('layers', 'output_sha256', 'prediction', 'diagnostics',
                'ground_truth', 'fp32_prediction', 'graph_sha256')
        if any(old[k] != record[k] for k in keys):
            raise ValueError('migrated checkpoint differs from original v1 evidence')
        if record['new_compute_seconds'] != 0 or record['source_kind'] != 'migrated_v1_original_provenance':
            raise ValueError('migrated compute/provenance mislabeled')
    elif 'original_record' in record:
        original = read(checked(record['original_record']))
        expected = original['backends'][backend]
        keys = ('layers', 'output_sha256', 'prediction', 'diagnostics')
        if (original['sample'] != sample or original['graph_sha256'] != record['graph_sha256']
                or any(record[k] != expected[k] for k in keys)
                or record['source_kind'] != 'retained_verified_exact'
                or record['new_compute_seconds'] != 0):
            raise ValueError('historical exact checkpoint identity changed')
    elif record['source_kind'] != 'new_verified_execution':
        raise ValueError('unknown v2 checkpoint provenance')


def _migrate_old(parent_path, parent, plan, sample, mode, backend, index):
    if mode == 'control':
        changed = [name for name, sha in parent['sources'].items()
                   if name in plan['sources'] and plan['sources'][name] != sha]
        if changed:
            raise ValueError('matched-control implementation changed; v1 control cannot be migrated: ' + ', '.join(changed))
    old_path = parent_path.parent / f'{index:04d}-{mode}-{backend}.json'
    if not old_path.exists():
        return None
    old = unseal(old_path)
    v1.validate_record(old, parent, sample, mode, backend, index)
    if old['graph_sha256'] != plan['graph_sha256']:
        raise ValueError('v1 graph differs from v2 canonical graph')
    record = {**old, 'job_sha256': digest(plan), 'v1_record': reference(old_path),
              'source_kind': 'migrated_v1_original_provenance',
              'new_compute_seconds': 0,
              'migration': 'same frozen graph/sample/numerical arm; original v1 source and timing retained'}
    validate_record(record, plan, sample, mode, backend, index)
    return record


def _import_retained_exact(plan, sample, paired, retained, backend, index):
    from tools.breadth_study.useful_quality_v2 import verify_v1_archive
    parent = unseal(checked(plan['parent_plan']))
    verify_v1_archive(parent)
    original_ref = retained['image_records'][index]
    old = read(checked(original_ref))
    if backend not in old['backends']:
        pilot = unseal(checked(plan['static_proof']))['acceptance']['pilot']
        original_ref = read(checked(pilot))['image_records'][index]
        old = read(checked(original_ref))
    actual = old['backends'][backend]
    record = {**actual, 'job_sha256': digest(plan), 'sample': sample, 'mode': 'exact',
              'backend': backend, 'index': index, 'graph_sha256': old['graph_sha256'],
              'original_record': original_ref, 'source_kind': 'retained_verified_exact',
              'new_compute_seconds': 0, 'ground_truth': paired['ground_truth'],
              'fp32_prediction': paired['fp32_prediction'],
              'diagnostic_signature': event_signature(actual['diagnostics'])}
    validate_record(record, plan, sample, 'exact', backend, index)
    return record


def requires_legacy_whole_graph(mode, backend, index):
    """The frozen protocol requires first-image uncompiled parity for exact only."""
    return mode == 'exact' and backend == 'cpp' and index == 0


def worker(path, mode, backend, images, fds):
    from tools.breadth_study.e1 import validate_lease
    from tools.breadth_study.useful_quality_v2 import source_identity, verify_v1_archive, verify_v2_guard, check_static_replay
    validate_lease(fds)
    if mode not in ('exact', 'control') or backend not in ('cpp', 'cuda') or images not in (1, 8, 32, 128):
        raise ValueError('invalid v2 work stage')
    if backend == 'cpp' and images > 8:
        raise ValueError('CPU work stops at the eight-image native gate')
    plan = unseal(path)
    if (plan['version'] != 'useful-quality-e1-case-v2'
            or plan['sources'] != source_identity()
            or plan['guard_sha256'] != verify_v2_guard()):
        raise ValueError('v2 runtime changed since enrollment')
    checked(plan['optimized_certificate'])
    parent_path = checked(plan['parent_plan'])
    parent = unseal(parent_path)
    verify_v1_archive(parent)
    for key in ('graph', 'configuration', 'baseline', 'protocol', 'mapping', 'static_proof'):
        checked(plan[key])
    check_static_replay(plan)
    if images > 8 and not (path.parent / 'native-admission.json').exists():
        raise ValueError('v2 panels require completed eight-image CPU/CUDA gate')

    import numpy as np
    import torch
    from public.analysis.phase3.diagnostics import LayerDiagnostics, ReferenceSamples
    from public.inference.tensor import QUANTIZATION_OBSERVER, parse_encoding
    from public.quantization.graph.executable import execute
    from public.quantization.graph.executable import graph_sha256
    from public.quantization.ptq.encoding import float_tensor
    from tools.breadth_study.matched_control_v2 import MatchedControl, native_conformance
    from tools.exact_execution_v2.activation_trace import ActivationTrace
    from tools.exact_execution_v2.optimized_engine import OptimizedPreparedGraph
    from tools.exact_execution_v3.optimized_matched_control import OptimizedMatchedControl, optimized_control_conformance
    from tools.phase3.baselines import screen_rows
    from tools.phase3.controller_worker import rational_admission_guard
    from tools.phase3.runtime import load_runtime
    from tools.run.phase3_thread_benchmark import set_threads

    started = time.perf_counter()
    torch.set_num_interop_threads(1)
    set_threads(4)
    graph = read(checked(plan['graph']))
    if graph_sha256(graph) != plan['graph_sha256']:
        raise ValueError('v2 graph content differs from frozen canonical identity')
    population, payload, _ = screen_rows(plan['model'], campaign(), verify_images=False)
    if population[:128] != unseal(checked(plan['protocol']))['sample_rows']:
        raise ValueError('v2 population differs from the frozen scientific protocol')
    baseline = read(checked(plan['baseline']))
    retained = read(checked(plan['retained_exact_screen'])) if plan['retained_exact_screen'] else None
    if retained is not None:
        from tools.phase3.evidence import verify_complete
        verify_complete(checked(plan['retained_exact_screen']), current_execution=False)
    sample_fn, observe, manifest = load_runtime(plan['model'], campaign())
    input_name = next(iter(graph['inputs']))
    input_encoding = parse_encoding(graph['inputs'][input_name])
    immutable(path.parent / f'native-conformance-{mode}-{backend}.json', native_conformance(backend))
    if mode == 'control' and plan['format'] != 'int8':
        control_conformance = path.parent / f'optimized-control-conformance-{backend}.json'
        if control_conformance.exists():
            saved_conformance = unseal(control_conformance)
            if saved_conformance.get('status') != 'passed' or saved_conformance.get('backend') != backend:
                raise ValueError('optimized-control native conformance is invalid')
        else:
            immutable(control_conformance, optimized_control_conformance(backend))
    engine = None
    for index, row in enumerate(population[:images]):
        paired = baseline['records'][index]
        if row['sha256'] != paired['sample_sha256'] or reference(payload / row['relative_path'])['sha256'] != row['sha256']:
            raise ValueError('v2 image payload, label or baseline changed')
        destination = path.parent / f'{index:04d}-{mode}-{backend}.json'
        if destination.exists():
            validate_record(unseal(destination), plan, row, mode, backend, index)
            continue
        migrated = None
        if plan['format'] == 'int8':
            migrated = _migrate_old(parent_path, parent, plan, row, mode, backend, index)
        if migrated is not None:
            record = migrated
        elif mode == 'exact' and retained is not None:
            record = _import_retained_exact(plan, row, paired, retained, backend, index)
        else:
            if engine is None:
                tick = time.perf_counter()
                engine_type = (OptimizedPreparedGraph if mode == 'exact' else
                               MatchedControl if plan['format'] == 'int8' else OptimizedMatchedControl)
                engine = engine_type(graph, backend=backend)
                print(f'{plan["case"]} {mode}/{backend} engine setup {time.perf_counter()-tick:.2f}s', flush=True)
            tick = time.perf_counter()
            fp32, _ = sample_fn(payload / row['relative_path'])
            if list(fp32.shape) != manifest['input_shape']:
                raise ValueError('v2 preprocessing changed')
            refs = ReferenceSamples(campaign()['diagnostics']['sample_elements_per_layer_image'])
            observe(fp32, refs)
            tensor = float_tensor(fp32.numpy(), input_encoding)
            prep_seconds = time.perf_counter() - tick
            diagnostics = LayerDiagnostics(refs)
            with ActivationTrace(), rational_admission_guard():
                token = QUANTIZATION_OBSERVER.set(diagnostics.quantization)
                tick = time.perf_counter()
                try:
                    actual = engine.execute({input_name: tensor}, observer=diagnostics)
                finally:
                    QUANTIZATION_OBSERVER.reset(token)
                seconds = time.perf_counter() - tick
            output = next(iter(actual['outputs'].values()))
            values = output.values()
            if not all(np.isfinite(float(v)) for v in values):
                raise ValueError('v2 candidate output is nonfinite')
            record = {'job_sha256': digest(plan), 'sample': row, 'mode': mode, 'backend': backend,
                      'index': index, 'graph_sha256': actual['graph_sha256'], 'layers': actual['layers'],
                      'execution_modes': actual['execution_modes'], 'output_sha256': digest(output.document()),
                      'prediction': sorted(range(len(values)), key=lambda i: (-values[i], i))[:5],
                      'ground_truth': paired['ground_truth'], 'fp32_prediction': paired['fp32_prediction'],
                      'diagnostics': diagnostics.records,
                      'diagnostic_signature': event_signature(diagnostics.records),
                      'source_kind': 'new_verified_execution',
                      'input_preparation_seconds': prep_seconds, 'seconds': seconds,
                      'new_compute_seconds': seconds + prep_seconds,
                      'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            if mode == 'exact' or plan['format'] != 'int8':
                oracle_path = parent_path.parent / f'{index:04d}-{mode}-{backend}.json'
                if oracle_path.exists():
                    oracle = unseal(oracle_path)
                    v1.validate_record(oracle, parent, row, mode, backend, index)
                    keys = ('layers', 'output_sha256', 'prediction', 'diagnostic_signature')
                    matched = all(record[k] == oracle[k] for k in keys)
                    numerical = {key: record[key] for key in keys}
                    proof = {'v1_oracle': reference(oracle_path), 'v2_numerical_sha256': digest(numerical),
                             'fields': list(keys), 'matches': matched}
                    immutable(path.parent / f'{index:04d}-{mode}-{backend}-oracle.json', proof)
                    if not matched:
                        raise ValueError('optimized execution differs from retained v1 arm oracle')
                elif requires_legacy_whole_graph(mode, backend, index):
                    legacy_diag = LayerDiagnostics(refs)
                    with ActivationTrace(), rational_admission_guard():
                        token = QUANTIZATION_OBSERVER.set(legacy_diag.quantization)
                        tick = time.perf_counter()
                        try:
                            legacy = execute(graph, {input_name: tensor}, backend='cpp', observer=legacy_diag)
                        finally:
                            QUANTIZATION_OBSERVER.reset(token)
                    legacy_time = time.perf_counter() - tick
                    matched = (legacy['layers'] == record['layers']
                               and digest(next(iter(legacy['outputs'].values())).document()) == record['output_sha256']
                               and event_signature(legacy_diag.records) == record['diagnostic_signature'])
                    compatibility_path = path.parent / f'{mode}-reference-compatibility.json'
                    comparison = {'graph': plan['graph'], 'sample': row, 'seconds': legacy_time,
                                  'matches': matched,
                                  'v2_numerical_sha256': digest({key: record[key] for key in
                                                                 ('layers', 'output_sha256', 'diagnostic_signature')})}
                    if compatibility_path.exists():
                        prior = unseal(compatibility_path)
                        if {k: v for k, v in comparison.items() if k != 'seconds'} != {
                                k: v for k, v in prior.items() if k != 'seconds'}:
                            raise ValueError('resumed legacy compatibility differs')
                    else:
                        immutable(compatibility_path, comparison)
                    if not matched:
                        raise ValueError('optimized exact execution differs from legacy rational oracle')
            validate_record(record, plan, row, mode, backend, index)
        if backend == 'cuda' and index < 8:
            cpu = unseal(path.parent / f'{index:04d}-{mode}-cpp.json')
            validate_record(cpu, plan, row, mode, 'cpp', index)
            fields = ('layers', 'output_sha256', 'prediction')
            if mode != 'exact' or plan['format'] != 'int8':
                fields += ('diagnostic_signature',)
            if any(record[k] != cpu[k] for k in fields):
                raise ValueError('v2 CPP/CUDA discrepancy')
        if plan['sources'] != source_identity() or plan['guard_sha256'] != verify_v2_guard():
            raise ValueError('v2 numerical runtime changed before checkpoint commit')
        immutable(destination, record)
        atomic_json(path.parent / f'progress-{mode}-{backend}.json',
                    {'case': plan['case'], 'mode': mode, 'backend': backend,
                     'images': index + 1, 'target': images,
                     'updated_at': datetime.now(timezone.utc).isoformat(),
                     'seconds': record['new_compute_seconds']})
        print(f'{plan["case"]} {mode}/{backend} {index+1}/{images}: {record["new_compute_seconds"]:.2f}s', flush=True)
    immutable(path.parent / 'invocations' / f'{time.time_ns()}.json',
              {'plan': reference(path), 'mode': mode, 'backend': backend, 'images': images,
               'wall_seconds': time.perf_counter() - started,
               'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'finished_at': datetime.now(timezone.utc).isoformat()})
