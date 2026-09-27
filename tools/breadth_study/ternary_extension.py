"""Gated ternary E1 continuation after the main E1/E2 resource queue."""
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.experiment_b.common import atomic_json, unseal
from tools.phase3.common import ROOT, checked, read, reference, digest
from tools.phase3.execution_guard import verify_guard
from tools.phase3.worker_locks import exclusive, pool_locks
from tools.run.exact_execution import immutable, archive_implementation

BASE = ROOT / 'artifacts/breadth_study/ternary_e1_v1'
CASE = 'mobilenet_v3_large/ternary'


def extra_sources():
    return {p: reference(ROOT / p)['sha256'] for p in (
        'tools/breadth_study/ternary_extension.py',
        'development/acceptance_proofs/mapped_mean.py',
        'development/acceptance_proofs/mapped_mac_store.py',
        'public/analysis/phase3/finite_fp64_nonmac.py')}


def static_proof(prepared):
    from development.acceptance_proofs.mapped_mean import prove_graph_means
    from development.acceptance_proofs.mapped_mac_store import prove_graph
    from public.analysis.phase3.finite_fp64_nonmac import prove_nonmac
    if prepared['model'] != 'mobilenet_v3_large' or prepared['format'] != 'ternary':
        raise ValueError('extension admits only the frozen selected ternary case')
    graph = read(checked(prepared['graph']))
    shapes_ref = read(ROOT / 'results/summaries/phase3-shapes.json')['models'][prepared['model']]
    shapes = read(checked(shapes_ref))['shapes']
    mac = prove_graph(graph, 'ternary')
    means = prove_graph_means(graph, shapes)
    other = [r for r in prove_nonmac(graph, shapes)['records'] if r['op'] != 'adaptive_average_pool2d']
    names = [r['node'] for r in mac['records']] + [r['node'] for r in means['records']] + [r['node'] for r in other]
    if (mac['status'] != 'conditional_output_codes_proven' or means['status'] != 'mean_stores_proven'
            or any(r['status'] == 'pending' for r in other) or len(names) != len(set(names))
            or set(names) != {n['name'] for n in graph['nodes']}):
        raise ValueError('ternary static proof does not cover every deployed operator')
    return {'case': CASE, 'graph': prepared['graph'], 'shape_evidence': shapes_ref,
            'MAC': mac, 'means': means, 'other_nonMAC': other, 'guard_sha256': verify_guard(),
            'sources': extra_sources(), 'status': 'static_complete_native_pilot_required'}


def admission(prepared, pilot):
    from tools.phase3.acceptance import verify_pilot
    from tools.breadth_study.family_admission import verify_shapes
    proof = static_proof(prepared)
    graph, _, _, records = verify_pilot(prepared, pilot)
    verify_shapes(graph, read(checked(proof['shape_evidence']))['shapes'], records)
    return {'static': proof, 'pilot': pilot, 'images': len(records),
            'status': 'study_ternary_graph_and_native_pilot_verified',
            'scope': 'prospective E1 only; historical A acceptance registry unchanged'}


@contextmanager
def execution_adapter():
    """Versioned process-local admission hook; frozen files remain untouched."""
    from tools.breadth_study import prospective_execution as execution
    original_sources, original_admission = execution.sources, execution.verify_admission
    def sources():
        return {**original_sources(), **extra_sources()}
    def verify(plan, graph):
        from tools.phase3.preparation_inventory import preparation_records
        if plan['purpose'] != 'E1' or f"{plan['model']}/{plan['format']}" != CASE:
            raise ValueError('ternary adapter rejects unrelated experiment plans')
        prepared = preparation_records()[CASE]
        if graph != read(checked(prepared['graph'])) or plan['graph'] != prepared['graph']:
            raise ValueError('ternary adapter graph differs from frozen preparation')
        if unseal(checked(plan['admission'])) != admission(prepared, plan['strict_pilot']):
            raise ValueError('ternary graph/native admission does not reproduce')
    execution.sources, execution.verify_admission = sources, verify
    try:
        yield execution
    finally:
        execution.sources, execution.verify_admission = original_sources, original_admission


def prepare():
    from tools.phase3.preparation_inventory import preparation_records
    prepared = preparation_records()[CASE]
    proof = static_proof(prepared)
    path = BASE / 'proofs' / f'{digest(proof)}.json'
    immutable(path, {**proof, 'source_archive': archive_implementation(extra_sources())})
    return reference(path)


def run():
    from tools.breadth_study.study_next import BASE as MAIN, pilot_task
    from tools.phase3.preparation_inventory import preparation_records
    with exclusive(BASE / 'controller.lock'):
        status = {'status': 'verifying_static_proof', 'pid': os.getpid(), 'case': CASE}
        try:
            status['proof'] = prepare()
            status['status'] = 'waiting_for_main_E1_E2_queue'
            atomic_json(BASE / 'status.json', status)
            while True:
                prior = read(MAIN / 'status.json')['status']
                if prior.startswith('admitted_E1_and_E2_completed'):
                    break
                if prior == 'failed':
                    raise RuntimeError('main queue failed; repair it before the ternary continuation')
                time.sleep(30)
            while True:
                stack = ExitStack()
                try:
                    controller = stack.enter_context(exclusive(ROOT / 'artifacts/phase3/controller/controller.lock'))
                    native = stack.enter_context(pool_locks(ROOT))
                    break
                except RuntimeError:
                    stack.close()
                    time.sleep(10)
            with stack:
                fds = (controller.fileno(), native['cuda'], native['cpu'])
                prepared = preparation_records()[CASE]
                def pilots():
                    return [p for p in (ROOT / 'artifacts/phase3/runs').glob('*/summary.json')
                            if (lambda d: d['configuration_sha256'] == prepared['configuration_sha256']
                                and d['scope'] == 'pilot' and d['images'] == 8 and d['backend_equality'])(read(p))]
                saved = pilots()
                if not saved:
                    status['status'] = 'paired_native_pilot'
                    atomic_json(BASE / 'status.json', status)
                    from tools.phase3.controller_worker import execute_task
                    execute_task(pilot_task('mobilenet_v3_large', 'ternary', prepared), native['cuda'])
                    saved = pilots()
                if len(saved) != 1:
                    raise ValueError('ternary needs one complete paired native pilot')
                pilot = reference(saved[0])
                accepted = admission(prepared, pilot)
                path = BASE / 'admissions' / f'{digest(accepted)}.json'
                immutable(path, accepted)
                with execution_adapter() as execution:
                    plan = execution.create_plan(model='mobilenet_v3_large', fmt='ternary', graph=prepared['graph'],
                         admission=reference(path), purpose='E1', arm='matched_arithmetic', strict_pilot=pilot,
                         baseline=read(ROOT / 'results/summaries/phase3-baselines.json')['models']['mobilenet_v3_large'])
                status.update(status='E1_matched_32', plan=reference(plan))
                atomic_json(BASE / 'status.json', status)
                env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                       'OPENBLAS_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE'}
                subprocess.run([sys.executable, '-m', 'tools.breadth_study.ternary_extension', 'worker',
                                str(plan), *map(str, fds)], cwd=ROOT, env=env, pass_fds=fds, check=True)
                status.update(status='completed_ternary_E1', summary=reference(plan.parent / 'summary-32.json'))
        except BaseException as error:
            status.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            atomic_json(BASE / 'status.json', status)


def start():
    try:
        with exclusive(BASE / 'controller.lock'):
            pass
    except RuntimeError:
        print(json.dumps({'ternary': 'already_running'}), flush=True)
        return
    with (BASE / 'run.log').open('ab') as log:
        child = subprocess.Popen([sys.executable, '-m', 'tools.breadth_study.ternary_extension', 'run'],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    print(json.dumps({'ternary_pid': child.pid, 'log': str(BASE / 'run.log')}), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run()
    elif sys.argv[1] == 'worker':
        with execution_adapter() as execution:
            execution.worker(Path(sys.argv[2]), 32, tuple(map(int, sys.argv[3:])))
    else:
        raise ValueError('use the study controller entrypoint')
