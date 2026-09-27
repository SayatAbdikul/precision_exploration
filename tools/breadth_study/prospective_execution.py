"""Native-gated exact E2 and expanded E1 image jobs with immutable checkpoints."""
from __future__ import annotations

import time
from pathlib import Path

from tools.experiment_b.common import unseal
from tools.phase3.common import ROOT, checked, read, reference, digest, campaign
from tools.phase3.execution_guard import verify_guard
from tools.run.exact_execution import immutable, implementation, archive_implementation, event_signature
from tools.breadth_study.e1 import validate_lease, paired_metrics

BASE = ROOT / 'artifacts/breadth_study/prospective_execution_v1'


def sources():
    paths = ['tools/breadth_study/prospective_execution.py', 'tools/breadth_study/matched_control_v2.py',
             'tools/breadth_study/family_admission.py', 'tools/breadth_study/e1.py', 'tools/run/study_next.py']
    return {**implementation(), **{p: reference(ROOT / p)['sha256'] for p in paths}}


def create_plan(*, model, fmt, graph, admission, purpose, arm, baseline, strict_pilot=None):
    current = sources()
    plan = {'version': 'prospective-exact-job-v1', 'model': model, 'format': fmt, 'graph': graph,
            'admission': admission, 'purpose': purpose, 'arm': arm, 'baseline': baseline,
            'strict_pilot': strict_pilot, 'sources': current, 'source_archive': archive_implementation(current),
            'guard_sha256': verify_guard(), 'maximum_images': 256 if purpose == 'E2' else 128,
            'diagnostics': 'original observers and sampling; activation cache starts cold per image in every newly run mode',
            'matrix': reference(ROOT / 'public/experiments/configs/breadth-study/comparison-matrix-v1.json')}
    path = BASE / digest(plan) / 'plan.json'
    immutable(path, plan)
    return path


def verify_admission(plan, graph):
    from public.quantization.graph.executable import graph_sha256
    admission = unseal(checked(plan['admission']))
    if plan['purpose'] == 'E2':
        from tools.phase3.acceptance import prove_integer_graph
        if admission['graph'] != plan['graph']:
            raise ValueError('recipe proof belongs to a different graph')
        shapes = read(checked(admission['shape_evidence']))['shapes']
        manifest = read(checked(campaign()['inputs'][plan['model']]))
        actual = prove_integer_graph(graph, {k: manifest['input_shape'] for k in graph['inputs']}, shapes)
        checked(admission['implementation'])
        if actual != admission['proof'] or actual['status'] != 'accepted':
            raise ValueError('E2 graph arithmetic proof does not reproduce')
    elif plan['purpose'] == 'E1':
        from tools.breadth_study.family_admission import verify
        from tools.phase3.preparation_inventory import preparation_records
        prepared = preparation_records()[f"{plan['model']}/{plan['format']}"]
        if prepared['graph'] != plan['graph'] or verify(prepared, plan['strict_pilot']) != admission:
            raise ValueError('expanded E1 admission does not reproduce')
    else:
        raise ValueError('unknown prospective experiment')


def worker(path, images, fds):
    validate_lease(fds)
    import torch
    import numpy as np
    from public.analysis.phase3.diagnostics import LayerDiagnostics, ReferenceSamples
    from public.inference.tensor import QUANTIZATION_OBSERVER, parse_encoding
    from public.quantization.ptq.encoding import float_tensor
    from public.quantization.graph.executable import execute
    from tools.breadth_study.matched_control_v2 import MatchedControl, native_conformance
    from tools.exact_execution_v2.engine import PreparedGraph
    from tools.exact_execution_v2.activation_trace import ActivationTrace
    from tools.phase3.baselines import screen_rows
    from tools.phase3.runtime import load_runtime
    from tools.phase3.controller_worker import rational_admission_guard
    from tools.phase3.integer_store import accelerated_stores
    from tools.run.phase3_thread_benchmark import set_threads
    plan = unseal(path)
    if (images not in (32, 128, 256) or images > plan['maximum_images']
            or plan['sources'] != sources() or plan['guard_sha256'] != verify_guard()):
        raise ValueError('prospective job sources, population or runtime changed')
    graph = read(checked(plan['graph']))
    verify_admission(plan, graph)
    population, payload, _ = screen_rows(plan['model'], campaign(), verify_images=False)
    sample, observe, manifest = load_runtime(plan['model'], campaign())
    torch.set_num_interop_threads(1)
    set_threads(4)
    name = next(iter(graph['inputs']))
    encoding = parse_encoding(graph['inputs'][name])
    baseline = read(checked(plan['baseline']))
    identity = digest(plan)
    historical = read(checked(plan['strict_pilot'])) if plan['strict_pilot'] else None
    modes = [('exact', 'cpp', 8), ('exact', 'cuda', images)]
    if plan['purpose'] == 'E1':
        # Fail before expensive full images if a native control ABI has drifted.
        for backend in ('cpp', 'cuda'):
            immutable(path.parent / f'conformance-{backend}.json', native_conformance(backend))
        modes += [('control', 'cpp', 8), ('control', 'cuda', images)]
    engines = {}
    for mode, backend, count in modes:
        engines[mode, backend] = (PreparedGraph if mode == 'exact' else MatchedControl)(graph, backend=backend)
        for index, row in enumerate(population[:count]):
            if row['sha256'] != baseline['records'][index]['sample_sha256']:
                raise ValueError('prospective baseline/image pairing changed')
            if reference(payload / row['relative_path'])['sha256'] != row['sha256']:
                raise ValueError('prospective evaluation payload drift')
            destination = path.parent / f'{index:04d}-{mode}-{backend}.json'
            if destination.exists():
                record = unseal(destination)
                if (record['job_sha256'] != identity or record['sample'] != row or record['mode'] != mode
                        or record['backend'] != backend):
                    raise ValueError('resumed prospective image provenance mismatch')
                continue
            fp32, _ = sample(payload / row['relative_path'])
            if list(fp32.shape) != manifest['input_shape']:
                raise ValueError('prospective preprocessing mismatch')
            references = ReferenceSamples(campaign()['diagnostics']['sample_elements_per_layer_image'])
            observe(fp32, references)
            tensor = float_tensor(fp32.numpy(), encoding)
            diagnostics = LayerDiagnostics(references)
            with ActivationTrace(), rational_admission_guard():
                token = QUANTIZATION_OBSERVER.set(diagnostics.quantization)
                tick = time.perf_counter()
                try:
                    result = engines[mode, backend].execute({name: tensor}, observer=diagnostics)
                finally:
                    QUANTIZATION_OBSERVER.reset(token)
                seconds = time.perf_counter()-tick
            output = next(iter(result['outputs'].values()))
            values = output.values()
            if not all(np.isfinite(float(v)) for v in values):
                raise ValueError('prospective output is nonfinite')
            record = {'job_sha256': identity, 'sample': row, 'mode': mode, 'backend': backend,
                      'graph_sha256': result['graph_sha256'], 'layers': result['layers'],
                      'execution_modes': result['execution_modes'], 'output_sha256': digest(output.document()),
                      'prediction': sorted(range(len(values)), key=lambda i: (-values[i], i))[:5],
                      'ground_truth': baseline['records'][index]['ground_truth'],
                      'fp32_prediction': baseline['records'][index]['fp32_prediction'],
                      'diagnostics': diagnostics.records, 'diagnostic_signature': event_signature(diagnostics.records),
                      'seconds': seconds}
            if mode == 'exact' and historical and index < 8:
                old = read(checked(historical['image_records'][index]))
                expected = old['backends'][backend]
                if old['sample'] != row or record['layers'] != expected['layers'] or record['output_sha256'] != expected['output_sha256']:
                    raise ValueError('prepared family execution differs from retained native pilot')
            if mode == 'exact' and backend == 'cpp' and index == 0 and plan['purpose'] == 'E2':
                # Independently check the uncompiled execution path on every new
                # recipe graph before admitting its prepared implementation.
                observer = LayerDiagnostics(references)
                with ActivationTrace(), accelerated_stores():
                    token = QUANTIZATION_OBSERVER.set(observer.quantization)
                    try:
                        legacy = execute(graph, {name: tensor}, backend='cpp', observer=observer)
                    finally:
                        QUANTIZATION_OBSERVER.reset(token)
                if (legacy['layers'] != record['layers'] or digest(next(iter(legacy['outputs'].values())).document()) != record['output_sha256']
                        or event_signature(observer.records) != record['diagnostic_signature']):
                    raise ValueError('new recipe prepared/reference execution mismatch')
                immutable(path.parent / 'prepared-legacy-compatibility.json', {'graph': plan['graph'], 'sample': row, 'matches': True})
            if backend == 'cuda' and index < 8:
                cpu = unseal(path.parent / f'{index:04d}-{mode}-cpp.json')
                if any(record[k] != cpu[k] for k in ('layers', 'output_sha256', 'diagnostic_signature')):
                    raise ValueError('prospective C++/CUDA conformance mismatch')
            if sources() != plan['sources'] or verify_guard() != plan['guard_sha256']:
                raise ValueError('prospective numerical sources changed')
            immutable(destination, record)
            print(f"{plan['model']}/{plan['format']} {plan['arm']} {mode}/{backend} {index+1}/{count}: {seconds:.2f}s", flush=True)
    exact = [unseal(path.parent / f'{i:04d}-exact-cuda.json') for i in range(images)]
    for mode in ('exact', 'control') if plan['purpose'] == 'E1' else ('exact',):
        for i in range(8):
            cpu = unseal(path.parent / f'{i:04d}-{mode}-cpp.json')
            gpu = unseal(path.parent / f'{i:04d}-{mode}-cuda.json')
            if any(cpu[k] != gpu[k] for k in ('layers', 'output_sha256', 'diagnostic_signature')):
                raise ValueError('resumed prospective native pilot mismatch')
    report = {'plan': reference(path), 'images': images, 'status': 'completed_development_experiment',
              'native_pilot_images': 8, 'native_pilot_matches': True,
              'top1_percent': 100*sum(r['prediction'][0] == r['ground_truth'] for r in exact)/images,
              'fp32_top1_percent': 100*sum(r['fp32_prediction'][0] == r['ground_truth'] for r in exact)/images,
              'records': [reference(path.parent / f'{i:04d}-exact-cuda.json') for i in range(images)]}
    if plan['purpose'] == 'E1':
        controls = [unseal(path.parent / f'{i:04d}-control-cuda.json') for i in range(images)]
        report['matched_metrics'] = paired_metrics([{'ground_truth': a['ground_truth'], 'strict_top5': a['prediction'],
            'control_top5': b['prediction'], 'output_codes_match': a['output_sha256'] == b['output_sha256'],
            'all_layers_match': a['layers'] == b['layers']} for a, b in zip(exact, controls)])
        report['control_records'] = [reference(path.parent / f'{i:04d}-control-cuda.json') for i in range(images)]
    immutable(path.parent / f'summary-{images}.json', report)
