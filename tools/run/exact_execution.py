"""Resumable compatibility and performance validation for prepared exact graphs.

This command writes diagnostic evidence in its own namespace. It does not
append predictions to a historical quality campaign or grant new acceptance.
"""
from __future__ import annotations

import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
from time import perf_counter

from tools.experiment_b.common import digest, seal, unseal
from tools.phase3.common import ROOT, campaign, checked, read, reference
from tools.phase3.execution_guard import verify_guard, machine_identity
from tools.phase3.worker_locks import exclusive, pool_locks


def implementation():
    paths = sorted((ROOT / 'tools/exact_execution_v2').glob('*.py'))
    paths += [Path(__file__), ROOT / 'tools/phase3/integer_store.py']
    return {str(p.relative_to(ROOT)): reference(p)['sha256'] for p in paths}


def immutable(path, payload):
    if path.exists():
        if unseal(path) != payload:
            raise ValueError(f'immutable evidence differs: {path}')
    else:
        seal(path, payload)


def archive_implementation(sources):
    directory = ROOT / 'artifacts/exact_execution_v2/implementations' / digest(sources)
    files = {}
    for name, sha in sources.items():
        source = checked({'path':name,'sha256':sha})
        target = directory / 'files' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise ValueError('immutable implementation archive changed')
        if not target.exists():
            target.write_bytes(data)
        files[name] = reference(target)
    immutable(directory / 'index.json', {'implementation':sources,'files':files})
    return reference(directory / 'index.json')


def event_signature(records):
    return digest({name: {key: row.get(key) for key in ('quantizer_event_counts', 'event_status', 'event_sampling')}
                   for name, row in records.items()})


def validate_row(row, expected, sample, mode, identity):
    from tools.phase3.common import digest as exact_digest
    if (row['job_sha256'] != identity or row['mode'] != mode or row['sample'] != sample
            or row['output_sha256'] != expected['output_sha256']
            or row['layers_sha256'] != exact_digest(expected['layers'])
            or row['diagnostic_signature'] != event_signature(expected['diagnostics'])
            or not row['matches_retained_pilot']):
        raise ValueError('compatibility checkpoint does not match retained pilot')


def verify_saved_report(plan, report):
    """Verify complete checkpoints against their reference pilot on no-op resume."""
    from tools.experiment_b.common import dataset
    pilot=read(checked(plan['pilot']))
    _,population,payload=dataset('coco_screen_1k' if plan['model']=='yolov8n' else 'imagenet_screen_1k')
    expected={(i,'prepared_v2') for i in range(plan['images'])}
    expected|={(i,'legacy_integer_store_v1') for i in range(plan['baseline_images'])}
    seen=set()
    for item in report['records']:
        row=unseal(checked(item))
        pair=(row['index'],row['mode'])
        if pair not in expected or pair in seen:
            raise ValueError('duplicate or unexpected validation checkpoint')
        original=read(checked(pilot['image_records'][row['index']]))
        validate_row(row,original['backends']['cuda'],original['sample'],row['mode'],digest(plan))
        if row['sample']!=population[row['index']] or row['graph_sha256']!=original['graph_sha256']:
            raise ValueError('resumed graph/population identity mismatch')
        if reference(payload/row['sample']['relative_path'])['sha256']!=row['sample']['sha256']:
            raise ValueError('resumed image payload drift')
        seen.add(pair)
    if (seen!=expected or report['job_sha256']!=digest(plan) or report['prepared_images']!=plan['images']
            or not report['all_layer_codes_outputs_execution_modes_and_quantizer_events_match']):
        raise ValueError('incomplete resumed validation')


def worker(path, lease_fds):
    # Keep the parent's kernel locks alive if it exits while workers finish.
    # A bare --worker invocation cannot bypass the controller/native locks.
    lock_paths = [ROOT/'artifacts/phase3/controller/controller.lock',
                  ROOT/'artifacts/phase3/locks/native-worker.lock', ROOT/'artifacts/phase3/locks/cpu-native-worker.lock']
    if not lease_fds or len(lease_fds)!=3:
        raise ValueError('worker requires inherited controller and native leases')
    for fd, lock_path in zip(lease_fds, lock_paths):
        actual, expected = os.fstat(fd), lock_path.stat()
        if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):
            raise ValueError('worker lease does not own the expected lock')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from public.analysis.phase3.diagnostics import LayerDiagnostics, ReferenceSamples
    from public.inference.native import NativeBackend
    from public.inference.tensor import QUANTIZATION_OBSERVER, parse_encoding
    from public.quantization.graph.executable import execute
    from public.quantization.ptq.encoding import float_tensor
    from tools.exact_execution_v2.engine import PreparedGraph
    from tools.exact_execution_v2.activation_trace import ActivationTrace
    from tools.phase3.common import digest as exact_digest
    from tools.phase3.evidence import verify_complete
    from tools.phase3.integer_store import accelerated_stores
    from tools.phase3.runtime import load_runtime
    from tools.phase3.baselines import screen_rows
    from tools.run.phase3_thread_benchmark import set_threads

    begin = perf_counter()
    plan = unseal(path)
    identity = digest(plan)
    if plan['implementation'] != implementation() or plan['guard_sha256'] != verify_guard():
        raise ValueError('execution implementation/dependencies changed')
    approval = read(checked(plan['acceptance']))
    if approval['status'] != 'accepted' or approval['pilot'] != plan['pilot']:
        raise ValueError('missing accepted reference pilot')
    print(f"{plan['model']}/{plan['format']}: verifying retained pilot", flush=True)
    _, _, (_, config, graph, _, accepted, _) = verify_complete(checked(plan['pilot']), current_execution=True)
    if config['model'] != plan['model'] or config['formats']['activation']['name'] != plan['format']:
        raise ValueError('reference configuration mismatch')
    frozen = campaign()
    population, payload, _ = screen_rows(plan['model'], frozen, verify_images=False)
    torch.set_num_interop_threads(1)
    sample, observe, manifest = load_runtime(plan['model'], frozen)
    effective = set_threads(plan['threads'])
    native = NativeBackend('cuda')
    if tuple(native.flex_gemm([1]*64, [1]*64, batch=1, channels=1, k=64, accumulator='int64_accumulator')) != (64,):
        raise ValueError('native warmup failed')
    start = perf_counter()
    engine = PreparedGraph(graph, backend='cuda')
    graph_preparation_seconds = perf_counter() - start
    input_name = next(iter(graph['inputs']))
    encoding = parse_encoding(graph['inputs'][input_name])
    setup_seconds = perf_counter() - begin
    rows, computed = [], 0
    modes = [('prepared_v2',plan['images'])]
    if plan['baseline_images']:
        modes.append(('legacy_integer_store_v1',plan['baseline_images']))
    for mode, count in modes:
      with ActivationTrace() as activation_trace:
        for index, selected in enumerate(population[:count]):
            if selected != accepted[index]['sample']:
                raise ValueError('pilot/evaluation image mismatch')
            expected = accepted[index]['backends']['cuda']
            prepared_input = None
            destination = path.parent / f'{index:02d}-{mode}.json'
            if destination.exists():
                row = unseal(destination)
                validate_row(row, expected, selected, mode, identity)
                activation_trace.replay(row['activation_calls'])
                rows.append(row)
                continue
            if prepared_input is None:
                if reference(payload / selected['relative_path'])['sha256'] != selected['sha256']:
                    raise ValueError('input image payload hash mismatch')
                start = perf_counter()
                fp32, _ = sample(payload / selected['relative_path'])
                if list(fp32.shape) != manifest['input_shape']:
                    raise ValueError('preprocessing shape mismatch')
                refs = ReferenceSamples(frozen['diagnostics']['sample_elements_per_layer_image'])
                observe(fp32, refs)
                tensor = float_tensor(fp32.numpy(), encoding)
                prepared_input = tensor, refs, perf_counter() - start
            tensor, refs, prep_seconds = prepared_input
            diagnostics = LayerDiagnostics(refs)
            activation_trace.calls=[]
            token = QUANTIZATION_OBSERVER.set(diagnostics.quantization)
            start = perf_counter()
            try:
                if mode == 'prepared_v2':
                    result = engine.execute({input_name: tensor}, observer=diagnostics)
                else:
                    with accelerated_stores():
                        result = execute(graph, {input_name: tensor}, backend='cuda', observer=diagnostics)
            finally:
                elapsed = perf_counter() - start
                QUANTIZATION_OBSERVER.reset(token)
            output = next(iter(result['outputs'].values()))
            failures = [label for label, match in {
                'layers':result['layers']==expected['layers'],
                'output':exact_digest(output.document())==expected['output_sha256'],
                'events':event_signature(diagnostics.records)==event_signature(expected['diagnostics']),
                'execution_modes':result['execution_modes']==expected['execution_modes'],
                'graph':result['graph_sha256']==accepted[index]['graph_sha256']}.items() if not match]
            if failures:
                raise ValueError(f'exact compatibility failure: {plan["model"]}/{plan["format"]} image {index} {mode}: {failures}')
            row = {'job_sha256': identity, 'mode': mode, 'sample': selected, 'index': index,
                   'input_sha256': exact_digest(tensor.document()), 'output_sha256': exact_digest(output.document()),
                   'layers_sha256': exact_digest(result['layers']), 'diagnostic_signature': event_signature(diagnostics.records),
                   'graph_sha256': result['graph_sha256'], 'layer_comparisons': len(result['layers']),
                   'matches_retained_pilot': True, 'seconds': elapsed, 'input_preparation_seconds': prep_seconds,
                   'activation_calls':activation_trace.calls}
            immutable(destination, row)
            rows.append(row)
            computed += 1
            print(f"{plan['model']}/{plan['format']} {index+1}/{plan['images']} {mode}: {elapsed:.3f}s MATCH", flush=True)
    if plan['implementation'] != implementation() or plan['guard_sha256'] != verify_guard():
        raise ValueError('execution source changed during validation')
    prepared = [r for r in rows if r['mode'] == 'prepared_v2']
    baseline = [r for r in rows if r['mode'] == 'legacy_integer_store_v1']
    result = {'job_sha256': identity, 'scope': 'diagnostic_compatibility_not_quality_campaign',
              'status': 'validated_eight_image_compatibility' if len(prepared) == 8 else 'partial_compatibility',
              'prepared_images': len(prepared), 'model': plan['model'], 'format': plan['format'],
              'all_layer_codes_outputs_execution_modes_and_quantizer_events_match': True,
              'thread_settings': effective, 'graph_preparation_seconds': graph_preparation_seconds,
              'setup_seconds_this_invocation': setup_seconds, 'wall_seconds_this_invocation': perf_counter()-begin,
              'computed_images_this_invocation': computed,
              'prepared_seconds': [r['seconds'] for r in prepared],
              'prepared_mean_seconds': statistics.mean(r['seconds'] for r in prepared),
              'prepared_warm_mean_seconds': statistics.mean(r['seconds'] for r in prepared[1:]) if len(prepared)>1 else None,
              'legacy_seconds': [r['seconds'] for r in baseline],
              'paired_speedup_legacy_over_prepared': (sum(r['seconds'] for r in baseline) /
                  sum(r['seconds'] for r in prepared if r['index'] < plan['baseline_images'])) if baseline else None,
              'timing_limit': 'wall time with full diagnostics; concurrent suites include resource contention; not GPU-only latency',
              'baseline_order': 'prepared pass then legacy pass; both start with cold activation cache',
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              'records': [reference(path.parent / f"{r['index']:02d}-{r['mode']}.json") for r in rows]}
    # Invocation timing may change on resume; immutable per-image evidence remains.
    seal(path.parent / 'validation.json', result)


def run(args):
    if not 1 <= args.images <= 8 or not 0 <= args.baseline_images <= args.images or not 1 <= args.workers <= 4 or not 1 <= args.threads <= 8:
        raise ValueError('use 1–8 images, <=4 workers, 1–8 threads, and baseline images <= images')
    with ExitStack() as stack:
        controller_lease=stack.enter_context(exclusive(ROOT / 'artifacts/phase3/controller/controller.lock'))
        native_leases=stack.enter_context(pool_locks(ROOT))
        lease_fds=(controller_lease.fileno(),native_leases['cuda'],native_leases['cpu'])
        guard, sources = verify_guard(), implementation()
        source_archive = archive_implementation(sources)
        plans = []
        for path in sorted((ROOT / 'artifacts/phase3/acceptance').glob('*/acceptance.json')):
            approval = read(path)
            job = read(checked(approval['pilot']).parent / 'job.json')
            prepared = read(checked(job['prepared']))
            config = read(checked(prepared['configuration']))
            model, fmt = config['model'], config['formats']['activation']['name']
            if model not in args.models or fmt not in args.formats:
                continue
            plan = {'version': 'exact-prepared-validation-v2.0.0', 'model': model, 'format': fmt,
                    'configuration_sha256': approval['configuration_sha256'], 'acceptance': reference(path),
                    'pilot': approval['pilot'], 'images': args.images, 'baseline_images': args.baseline_images,
                    'threads': args.threads, 'suite_workers': args.workers, 'guard_sha256': guard,
                    'implementation': sources, 'source_archive':source_archive, 'machine': machine_identity()}
            target = ROOT / 'artifacts/exact_execution_v2' / digest(plan) / 'plan.json'
            immutable(target, plan)
            plans.append(target)
        if {(unseal(p)['model'], unseal(p)['format']) for p in plans} != {(m,f) for m in args.models for f in args.formats}:
            raise ValueError('requested matrix lacks accepted exact reference pilots')
        start = perf_counter()
        def dispatch(path):
            existing = path.parent / 'validation.json'
            if existing.exists():
                report = unseal(existing)
                plan = unseal(path)
                verify_saved_report(plan,report)
                return {'validation': reference(existing), 'resumed': True}
            environment = {**os.environ, 'OMP_NUM_THREADS': str(args.threads), 'MKL_NUM_THREADS': str(args.threads),
                           'OPENBLAS_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE'}
            subprocess.run([sys.executable, '-m', 'tools.run.exact_execution', '--worker', str(path),
                            '--lease-fds',*[str(fd) for fd in lease_fds]],
                           cwd=ROOT, env=environment, pass_fds=lease_fds, check=True)
            return {'validation': reference(path.parent / 'validation.json'), 'resumed': False}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = [future.result() for future in as_completed([pool.submit(dispatch, p) for p in plans])]
        elapsed = perf_counter()-start
        if implementation() != sources or verify_guard() != guard:
            raise ValueError('suite execution dependencies changed')
        result = {'version': 'exact-prepared-suite-v2.0.0', 'guard_sha256': guard, 'implementation': sources,
                  'workers': args.workers, 'threads_per_worker': args.threads, 'plans': [reference(p) for p in plans],
                  'results': sorted(results, key=lambda r:r['validation']['path']), 'wall_seconds_this_invocation': elapsed,
                  'new_inferences': sum(unseal(checked(r['validation']))['computed_images_this_invocation'] for r in results if not r['resumed']),
                  'status': 'completed', 'scope': 'compatibility_and_throughput_only'}
        suite_id = digest([reference(p) for p in plans])
        target = ROOT / 'artifacts/exact_execution_v2/suites' / suite_id / f'{digest(result)}.json'
        immutable(target, result)
        print(target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['resnet18', 'mobilenet_v2'])
    parser.add_argument('--formats', nargs='+', default=['int4', 'int5', 'int6', 'int8'])
    parser.add_argument('--images', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--baseline-images', type=int, default=0)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--lease-fds', type=int, nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    worker(args.worker,args.lease_fds) if args.worker else run(args)


if __name__ == '__main__':
    main()
