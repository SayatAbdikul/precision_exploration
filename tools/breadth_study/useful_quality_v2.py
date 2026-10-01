"""Budgeted, resumable controller for the frozen useful-quality E1 population.

This controller does not mutate v1 plans or records. Its v2 plans bind the v1
scientific protocol and a separately archived optimized execution runtime.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from uuid import uuid4

from tools.experiment_b.common import atomic_json, unseal
from tools.phase3.common import ROOT, checked, digest, read, reference
from tools.phase3.execution_guard import GUARD_PATH, observed_guard
from public.experiments.registry.identity import canonical_json_bytes
from tools.phase3.worker_locks import exclusive, pool_locks
from tools.run.exact_execution import immutable, implementation, archive_implementation
from tools.breadth_study import useful_quality as v1

BASE = ROOT / 'artifacts/breadth_study/useful_quality_e1_v2'
OLD_BASE = v1.BASE
ORDER = tuple(f'{model}/{fmt}' for model, fmt in v1.CASES)
MODES = ('exact', 'control')
BACKENDS = ('cpp', 'cuda')
LIMIT_SECONDS = 24 * 3600
EXTENSION_SECONDS = 12 * 3600
SAFETY = 1.5
SCHEMA = 'useful-quality-e1-optimized-controller-v2'
INTERRUPTION_CHARGE = BASE / 'v1-interruption-charge.json'
CERTIFICATE_PATH = BASE / 'optimized-runtime-certificate.json'
ADDITIVE_NATIVE = frozenset((
    'public/inference/native_fp64_grid_v2.py',
    'public/inference/cpp/fp64_grid_v2.cpp',
    'public/cuda/kernels/fp64_grid_v2.cu',
))


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_v2_guard():
    """Verify the old numerical source inventory plus separate v2 additions."""
    expected = read(ROOT / GUARD_PATH)
    actual = observed_guard()
    if {k: v for k, v in actual.items() if k != 'source_sha256'} != {
            k: v for k, v in expected.items() if k != 'source_sha256'}:
        raise ValueError('legacy execution environment changed outside v2 native source inventory')
    paths = []
    for folder in ('public/inference', 'public/quantization', 'public/cuda', 'public/formats/oracle'):
        paths.extend(p for p in (ROOT / folder).rglob('*') if p.suffix in {'.py', '.h', '.cpp', '.cu'})
    if not all((ROOT / name).is_file() for name in ADDITIVE_NATIVE):
        raise ValueError('versioned native source inventory incomplete')
    old = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(paths) if str(p.relative_to(ROOT)) not in ADDITIVE_NATIVE}
    if hashlib.sha256(canonical_json_bytes(old)).hexdigest() != expected['source_sha256']:
        raise ValueError('frozen legacy numerical source identity changed')
    return digest(expected)


def source_identity():
    """Archive Python sources and native binary/proof identities used by v2."""
    from tools.exact_execution_v2.optimized_engine import runtime_identity
    paths = ('tools/breadth_study/useful_quality_v2.py',
             'tools/breadth_study/useful_quality_v2_worker.py',
             'tools/run/useful_quality_v2.py',
             'tools/breadth_study/useful_quality.py',
             'tools/breadth_study/e1.py',
             'tools/breadth_study/matched_control_v2.py',
             'tools/experiment_b/common.py',
             'tools/phase3/common.py',
             'tools/phase3/runtime.py', 'tools/phase3/baselines.py',
             'tools/phase3/worker_locks.py', 'tools/phase3/execution_guard.py',
             'tools/phase3/acceptance.py', 'tools/phase3/evidence.py',
             'tools/phase3/controller_worker.py',
             'tools/run/phase3_thread_benchmark.py',
             'public/analysis/phase3/diagnostics.py',
             'public/analysis/phase3/finite_fp64_nonmac.py',
             'public/quantization/ptq/encoding.py',
             'public/experiments/registry/identity.py',
             'development/acceptance_proofs/family_static.py',
             'development/acceptance_proofs/dyadic_fp_mac_store.py')
    versioned = sorted((ROOT / 'tools/exact_execution_v3').glob('*.py'))
    own = {**implementation(),
           **{p: reference(ROOT / p)['sha256'] for p in paths},
           **{p: reference(ROOT / p)['sha256'] for p in ADDITIVE_NATIVE},
           **{str(p.relative_to(ROOT)): reference(p)['sha256'] for p in versioned}}
    runtime = runtime_identity()
    if any(name in own and own[name] != sha for name, sha in runtime.items()):
        raise ValueError('optimized runtime identity disagrees with actual source bytes')
    if not any(name.endswith('.so') for name in runtime):
        raise ValueError('optimized runtime identity omits its versioned native binary')
    return {**own, **runtime}


def validate_certificate(certificate_ref, parents, current):
    """Reproduce graph certificates and checked full-image proof references."""
    from public.inference.native_fp64_grid_v2 import GraphGridCertificate
    from tools.exact_execution_v2.optimized_engine import runtime_identity
    if checked(certificate_ref) != CERTIFICATE_PATH.resolve():
        raise ValueError('optimized certificate must use the budget-audited v2 path')
    cert = unseal(checked(certificate_ref))
    if cert.get('version') != 'useful-quality-fp64-grid-v2-certificate' or cert.get('status') != 'passed':
        raise ValueError('optimized execution has no passed integration certificate')
    if cert.get('runtime_identity') != runtime_identity():
        raise ValueError('optimized certificate differs from current native/source identity')
    if not any(k.endswith('cpp.so') for k in current) or not any(k.endswith('cuda.so') for k in current):
        raise ValueError('optimized CPU/CUDA native binaries are not both archived')
    floating = {plan['case']: plan for _, plan in parents if plan['format'] != 'int8'}
    entries = cert.get('selected_graphs', {})
    if not set(floating) <= set(entries):
        raise ValueError('optimized certificate omits a selected floating graph')
    for case, plan in floating.items():
        grid = GraphGridCertificate(read(checked(plan['graph']))).document()
        item = entries[case]
        if item.get('graph_sha256') != plan['graph_sha256'] or item.get('grid_certificate_sha256') != digest(grid):
            raise ValueError(f'optimized grid certificate changed for {case}')
    calibration = cert.get('timing_reservations', {})
    positive = 0
    for case, arms in calibration.items():
        if case not in ORDER or not isinstance(arms, dict):
            raise ValueError('unexpected optimized timing case')
        for mode, backends in arms.items():
            if mode not in MODES or not isinstance(backends, dict):
                raise ValueError('unexpected optimized timing arm')
            for backend, value in backends.items():
                if (backend not in BACKENDS or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value <= 0 or value >= LIMIT_SECONDS):
                    raise ValueError('invalid optimized timing calibration')
                positive += 1
    if positive < 2:
        raise ValueError('optimized certificate lacks measured CPU/CUDA timing calibration')
    if (not isinstance(cert.get('pre_enrollment_compute_seconds'), (int, float))
            or not math.isfinite(cert['pre_enrollment_compute_seconds'])
            or cert['pre_enrollment_compute_seconds'] <= 0):
        raise ValueError('optimized benchmark compute is not charged')
    if cert['pre_enrollment_compute_seconds'] + 1e-6 < benchmark_compute_floor():
        raise ValueError('optimized benchmark certificate undercharges sealed full-image attempts')
    validation = cert.get('validation', {})
    comparisons = validation.get('full_image_comparisons', {})
    required = {'resnet18/fp6_e2m3': {'cpp', 'cuda'}}
    for case, backends in required.items():
        if not set(backends) <= set(comparisons.get(case, {})):
            raise ValueError('E2M3 CPU/CUDA retained-oracle comparisons are missing')
    for case, backends in comparisons.items():
        if case not in floating:
            raise ValueError('unexpected full-image comparison case')
        for backend, item in backends.items():
            if backend not in BACKENDS:
                raise ValueError('unexpected full-image comparison backend')
            proof = unseal(checked(item))
            matched = proof.get('matches')
            required_fields = ('layers', 'output_sha256', 'prediction', 'diagnostic_signature')
            if (not isinstance(matched, dict) or any(matched.get(key) is not True for key in required_fields)
                    or proof.get('runtime_identity') != cert['runtime_identity']
                    or proof.get('case') != case or proof.get('backend') != backend
                    or proof.get('plan') != reference(next(path for path, p in parents if p['case'] == case))):
                raise ValueError('full-image optimized/oracle comparison did not pass')
            oracle = unseal(checked(proof['oracle']))
            if (proof.get('sample') != oracle['sample']
                    or any(proof.get('numerical', {}).get(key) != oracle[key] for key in required_fields)):
                raise ValueError('full-image comparison numerical fields differ from retained oracle')
            modes = proof.get('execution_modes', {})
            graph = read(checked(floating[case]['graph']))
            mac_nodes = {node['name'] for node in graph['nodes'] if node['op'] in ('conv2d', 'linear')}
            if set(modes) != mac_nodes or any(value != 'certified_fp64_grid_v2' for value in modes.values()):
                raise ValueError('full-image comparison did not exercise all certified MACs')
    native = validation.get('native_conformance', {})
    if any(native.get(backend, {}).get('status') != 'passed' for backend in BACKENDS):
        raise ValueError('optimized CPU/CUDA small-kernel conformance is incomplete')
    controls = validation.get('matched_control_comparisons', {})
    if 'cuda' not in controls.get('resnet18/fp6_e2m3', {}):
        raise ValueError('optimized floating control lacks a retained full-image CUDA comparison')
    control_source = current['tools/exact_execution_v3/optimized_matched_control.py']
    for case, backends in controls.items():
        if case not in floating:
            raise ValueError('unexpected matched-control comparison case')
        for backend, item in backends.items():
            if backend not in BACKENDS:
                raise ValueError('unexpected matched-control comparison backend')
            proof = unseal(checked(item))
            matched = proof.get('matches')
            required_fields = ('layers', 'output_sha256', 'prediction', 'diagnostic_signature')
            if (proof.get('mode') != 'control' or proof.get('case') != case or proof.get('backend') != backend
                    or proof.get('plan') != reference(next(path for path, p in parents if p['case'] == case))
                    or proof.get('runtime_identity') != cert['runtime_identity']
                    or proof.get('optimized_control_source_sha256') != control_source
                    or not isinstance(matched, dict)
                    or any(matched.get(key) is not True for key in required_fields)):
                raise ValueError('optimized matched-control full-image proof did not pass under the archived source')
            oracle = unseal(checked(proof['oracle']))
            if (proof.get('sample') != oracle['sample']
                    or any(proof.get('numerical', {}).get(key) != oracle[key] for key in required_fields)):
                raise ValueError('matched-control comparison differs from retained v1 oracle')
    return cert


def verify_v1_archive(plan):
    """Check the frozen parent source bytes without comparing against v2 code."""
    index = unseal(checked(plan['source_archive']))
    if index['implementation'] != plan['sources']:
        raise ValueError('v1 parent plan/source archive mismatch')
    if set(index['files']) != set(plan['sources']):
        raise ValueError('v1 source archive incomplete')
    for name, item in index['files'].items():
        # A new runtime may legitimately change a current source. The sealed
        # archive must still contain exactly the old bytes named by the plan.
        checked(item)
        if item['sha256'] != plan['sources'][name]:
            raise ValueError('v1 source archive differs from parent plan')
    for name in ('tools/breadth_study/useful_quality.py',
                 'tools/breadth_study/matched_control_v2.py',
                 'tools/breadth_study/e1.py'):
        if reference(ROOT / name)['sha256'] != plan['sources'][name]:
            raise ValueError('frozen v1 study criterion or matched-control source changed')
    for key in ('graph', 'configuration', 'baseline', 'protocol', 'mapping', 'static_proof'):
        checked(plan[key])


def old_plans():
    enrollment = unseal(OLD_BASE / 'enrollment.json')
    if enrollment['version'] != 'useful-quality-e1-enrollment-v1':
        raise ValueError('unexpected parent enrollment')
    if tuple(row['case'] for row in enrollment['cases']) != ORDER:
        raise ValueError('frozen candidate order changed')
    if enrollment['protocol']['sha256'] != '0315ab8af441761d7ad2e58c8bbdecc47dc40819f9766a52173c0b8300cb91de':
        raise ValueError('frozen scientific protocol changed')
    checked(enrollment['protocol'])
    result = []
    for row in enrollment['cases']:
        if row['status'] != 'static_passed_native_gate_required':
            raise ValueError(f"{row['case']} lacks frozen static proof")
        parent_path = checked(row['plan'])
        plan = unseal(parent_path)
        verify_v1_archive(plan)
        if plan['case'] != row['case'] or plan['protocol'] != enrollment['protocol']:
            raise ValueError('parent enrollment/plan mismatch')
        result.append((parent_path, plan))
    return enrollment, result


def validate_static_v2(plan):
    """Replay the frozen static proofs under the archived legacy source digest.

    The v1 proof helpers reject additive source files by design. This verifier
    retains all their numerical comparisons while taking the old identity from
    the separately reproduced v2 legacy guard, never from a monkeypatch.
    """
    from development.acceptance_proofs import dyadic_fp_mac_store
    from public.analysis.phase3.finite_fp64_nonmac import prove_nonmac
    from public.quantization.graph.executable import graph_sha256
    from tools.phase3.acceptance import prove_integer_graph
    from tools.phase3.baselines import screen_rows
    from tools.phase3.common import campaign, file_hash
    from tools.phase3.evidence import verify_complete
    verify_v2_guard()
    parent = unseal(checked(plan['parent_plan']))
    verify_v1_archive(parent)
    expected_source = read(ROOT / GUARD_PATH)['source_sha256']
    config = read(checked(plan['configuration']))
    graph = read(checked(plan['graph']))
    prepared = plan['prepared']
    saved = unseal(checked(plan['static_proof']))
    campaign_sha = digest(campaign())
    case = config['model'] + '/' + config['formats']['activation']['name']
    if (case != plan['case'] or prepared['configuration_sha256'] != digest(config)
            or prepared['graph'] != plan['graph'] or prepared['configuration'] != plan['configuration']
            or config['runtime']['source_sha256'] != expected_source
            or config['campaign_sha256'] != campaign_sha
            or graph_sha256(graph) != plan['graph_sha256']
            or saved['graph'] != plan['graph']):
        raise ValueError('frozen static classifier/configuration identity changed')
    if saved['kind'] == 'accepted_integer':
        accepted = read(checked(saved['prepared']))
        acceptance = saved['acceptance']
        proof = saved['proof']
        if (accepted.get('screen_acceptance') is None
                or read(checked(accepted['screen_acceptance'])) != acceptance
                or read(checked(acceptance['proof'])) != proof
                or acceptance['source_sha256'] != expected_source
                or acceptance['configuration_sha256'] != digest(config)
                or acceptance['status'] != 'accepted'
                or proof['source_sha256'] != expected_source
                or proof['pilot'] != acceptance['pilot']
                or proof['proof_implementation'] != reference(ROOT / 'tools/phase3/acceptance.py')):
            raise ValueError('retained integer acceptance/proof identity changed')
        verify_complete(checked(acceptance['pilot']), current_execution=False)
        pilot = read(checked(acceptance['pilot']))
        if (pilot['scope'] != 'pilot' or pilot['images'] != 8
                or pilot['configuration_sha256'] != digest(config)
                or pilot['source_sha256'] != expected_source
                or not pilot['backend_equality'] or not pilot['diagnostics_complete']):
            raise ValueError('retained integer pilot is incomplete')
        population = screen_rows(config['model'], campaign(), verify_images=False)[0]
        expected_nodes = {n['name'] for n in graph['nodes']}
        layers = []
        for index, item in enumerate(pilot['image_records']):
            record = read(checked(item))
            a, b = record['backends']['cpp'], record['backends']['cuda']
            if (record['sample'] != population[index]
                    or record['graph_sha256'] != plan['graph_sha256']
                    or a['layers'] != b['layers'] or a['output_sha256'] != b['output_sha256']
                    or set(a['layers']) != expected_nodes or set(a['diagnostics']) != expected_nodes
                    or set(b['diagnostics']) != expected_nodes):
                raise ValueError('retained integer pilot node/sample evidence changed')
            layers.append({name: row['shape'] for name, row in a['layers'].items()})
        if any(shapes != layers[0] for shapes in layers[1:]):
            raise ValueError('retained integer pilot shapes vary')
        manifest = read(checked(campaign()['inputs'][config['model']]))
        actual = prove_integer_graph(graph, {name: manifest['input_shape'] for name in graph['inputs']}, layers[0])
        if actual['status'] != 'accepted' or any(proof.get(key) != value for key, value in actual.items()):
            raise ValueError('retained integer accumulator proof does not reproduce')
        return {'kind': saved['kind'], 'configuration': case, 'mac_nodes': len(actual['mac_bounds']),
                'nonmac_bounds': len(actual['other_sum_bounds']), 'source_sha256': expected_source}
    if saved['kind'] != 'reproduced_dyadic_graph':
        raise ValueError('unknown frozen static proof type')
    fmt = plan['format']
    if fmt not in dyadic_fp_mac_store.FORMATS:
        raise ValueError('floating proof format outside dyadic family')
    directory = ROOT / 'artifacts/phase3/proof-development/dyadic-fp-mac-store-v1'
    summary = read(checked(saved['mac_proof_summary']))
    manifest = read(checked(summary['manifest'], directory))
    implementation_hash = file_hash(dyadic_fp_mac_store.__file__)
    if (manifest['source_sha256'] != expected_source or manifest['campaign_sha256'] != campaign_sha
            or manifest['implementation_sha256'] != implementation_hash):
        raise ValueError('retained dyadic proof implementation or source changed')
    matching = [read(checked(item, directory)) for item in summary['records']]
    matching = [item for item in matching if item['configuration'] == case]
    if len(matching) != 1:
        raise ValueError('selected graph lacks a unique retained MAC proof')
    record = matching[0]
    actual = dyadic_fp_mac_store.prove_graph(graph, fmt)
    if (record['graph'] != plan['graph'] or record['configuration_artifact'] != plan['configuration']
            or record['source_sha256'] != expected_source or record['campaign_sha256'] != campaign_sha
            or record['proof_implementation_sha256'] != implementation_hash
            or record['graph_acceptance'] is not False or record['pending_nonmac_nodes']
            or actual != record['proof'] or actual['status'] != 'conditional_output_codes_proven'
            or actual['pending_channels']):
        raise ValueError('retained dyadic MAC/store proof does not reproduce')
    nonmac = read(checked(saved['nonmac_proof_summary']))
    rows = [row for row in nonmac['records'] if row['configuration'] == case]
    if (nonmac['source_sha256'] != expected_source or nonmac['campaign_sha256'] != campaign_sha
            or len(rows) != 1 or rows[0]['graph'] != plan['graph'] or rows[0]['pending_nodes']):
        raise ValueError('retained non-MAC proof identity or coverage changed')
    shapes_summary = read(ROOT / 'results/summaries/phase3-shapes.json')
    shape_ref = shapes_summary['models'][config['model']]
    shapes = read(checked(shape_ref))['shapes']
    nonmac_actual = prove_nonmac(graph, shapes)
    if (shapes_summary['campaign_sha256'] != campaign_sha
            or nonmac_actual['records'] != rows[0]['records'] or nonmac_actual['pending_nodes']
            or len(nonmac_actual['records']) != len(graph['nodes']) - actual['mac_nodes']):
        raise ValueError('retained non-MAC proof does not reproduce')
    expected = {'configuration': case, 'status': 'static_graph_arithmetic_reproduced_native_pilot_required',
                'configuration_sha256': digest(config), 'graph': plan['graph'],
                'mac_nodes': actual['mac_nodes'], 'nonmac_nodes': len(nonmac_actual['records']),
                'shape_evidence': shape_ref, 'source_sha256': expected_source,
                'proof_implementation_sha256': implementation_hash}
    if saved['proof'] != expected:
        raise ValueError('frozen static proof result differs from independent v2 replay')
    return {'kind': saved['kind'], 'configuration': case,
            'mac_nodes': actual['mac_nodes'], 'nonmac_nodes': len(nonmac_actual['records']),
            'source_sha256': expected_source}


def check_static_replay(plan):
    record = unseal(checked(plan['static_replay']))
    if (record['parent_plan'] != plan['parent_plan']
            or record['static_proof'] != plan['static_proof']
            or record['sources'] != plan['sources']
            or record['guard_sha256'] != plan['guard_sha256']
            or record['result']['configuration'] != plan['case']
            or record['result']['kind'] != unseal(checked(plan['static_proof']))['kind']
            or record['result']['source_sha256'] != read(ROOT / GUARD_PATH)['source_sha256']):
        raise ValueError('versioned static proof replay identity changed')
    return record['result']


def prepare(*, certificate):
    """Create content-addressed v2 plans once the optimized proof is sealed."""
    if certificate is None:
        raise ValueError('optimized runtime certificate is required before enrollment')
    certificate = reference(Path(certificate))
    checked(certificate)
    enrollment, parents = old_plans()
    current = source_identity()
    validate_certificate(certificate, parents, current)
    archive = archive_implementation(current)
    guard = verify_v2_guard()
    entries = []
    for parent_path, parent in parents:
        graph = read(checked(parent['graph']))
        plan = {**parent, 'version': 'useful-quality-e1-case-v2',
                'parent_plan': reference(parent_path), 'parent_sources': parent['sources'],
                'sources': current, 'source_archive': archive,
                'guard_sha256': guard, 'optimized_certificate': certificate,
                'mac_nodes': [node['name'] for node in graph['nodes']
                              if node['op'] in ('conv2d', 'linear')],
                'migration': 'v1 records retain original source/timing; strict floating v2 executes anew and compares retained v1 oracle where present',
                'budget': {'scope': 'lifetime v1+v2 new compute', 'limit_seconds': LIMIT_SECONDS,
                           'extension_limit_seconds': EXTENSION_SECONDS}}
        replay = validate_static_v2(plan)
        replay_path = BASE / 'proofs-v2' / f'{digest({"case": parent["case"], "sources": current, "result": replay})}.json'
        immutable(replay_path, {'parent_plan': reference(parent_path), 'static_proof': plan['static_proof'],
                                'sources': current, 'guard_sha256': guard, 'result': replay})
        plan['static_replay'] = reference(replay_path)
        path = BASE / 'jobs' / digest(plan) / 'plan.json'
        immutable(path, plan)
        entries.append({'case': parent['case'], 'plan': reference(path)})
    result = {'version': SCHEMA, 'scientific_protocol': enrollment['protocol'],
              'optimized_certificate': certificate, 'cases': entries,
              'source_archive': archive, 'budget_seconds': LIMIT_SECONDS}
    immutable(BASE / 'enrollment.json', result)
    return result


def plans():
    enrollment = unseal(BASE / 'enrollment.json')
    if enrollment['version'] != SCHEMA or tuple(row['case'] for row in enrollment['cases']) != ORDER:
        raise ValueError('v2 enrollment mismatch')
    result = []
    for row in enrollment['cases']:
        path = checked(row['plan'])
        plan = unseal(path)
        if plan['case'] != row['case'] or plan['version'] != 'useful-quality-e1-case-v2':
            raise ValueError('v2 plan mismatch')
        checked(plan['optimized_certificate'])
        checked(plan['parent_plan'])
        check_static_replay(plan)
        if plan['sources'] != source_identity() or plan['guard_sha256'] != verify_v2_guard():
            raise ValueError('optimized runtime or guarded legacy execution changed')
        result.append((path, plan))
    return result


def old_sunk_seconds():
    """Count v1 compute and setup once, including partial invocations."""
    total = 0.0
    cases = {}
    for parent_path, parent in old_plans()[1]:
        folder = parent_path.parent
        image_seconds = sum(float(unseal(p).get('new_compute_seconds', 0.0))
                            for p in folder.glob('[0-9][0-9][0-9][0-9]-*-*.json'))
        legacy = folder / 'prepared-reference-compatibility.json'
        if legacy.exists():
            image_seconds += float(unseal(legacy)['seconds'])
        invocations = sum(float(unseal(p)['wall_seconds']) for p in (folder / 'invocations').glob('*.json'))
        # A completed invocation includes its records and setup. Partial runs
        # have records but no invocation, so retain whichever audit is larger.
        seconds = max(image_seconds, invocations)
        cases[parent['case']] = {'record_and_compatibility_seconds': image_seconds,
                                 'completed_invocation_wall_seconds': invocations,
                                 'charged_seconds': seconds}
        total += seconds
    return total, cases


def benchmark_compute_floor():
    """Include superseded probes as sunk cost, not just selected proofs."""
    total = 0.0
    for path in (BASE / 'benchmarks').glob('*.json'):
        seconds = unseal(path).get('seconds', {}).get('total_new_compute')
        if (not isinstance(seconds, (int, float)) or not math.isfinite(seconds)
                or seconds <= 0 or seconds > LIMIT_SECONDS):
            raise ValueError(f'invalid sealed benchmark compute time: {path}')
        total += float(seconds)
    return total


def pre_enrollment_cost():
    """Charge benchmark images before any v2 stage allocation."""
    if not CERTIFICATE_PATH.exists():
        return 0.0
    cert = unseal(CERTIFICATE_PATH)
    seconds = cert.get('pre_enrollment_compute_seconds')
    if (not isinstance(seconds, (int, float)) or not math.isfinite(seconds)
            or seconds <= 0 or seconds > LIMIT_SECONDS):
        raise ValueError('optimized certificate lacks a valid pre-enrollment compute charge')
    if seconds + 1e-6 < benchmark_compute_floor():
        raise ValueError('optimized certificate no longer covers sealed benchmark compute')
    return float(seconds)


def interrupted_charge():
    """Fail closed on v1 workers stopped before their invocation was sealed."""
    unfinished = []
    for parent_path, parent in old_plans()[1]:
        folder = parent_path.parent
        if (any(folder.glob('[0-9][0-9][0-9][0-9]-*-*.json'))
                and not any((folder / 'invocations').glob('*.json'))):
            unfinished.append(parent['case'])
    if not INTERRUPTION_CHARGE.exists():
        return 0.0, bool(unfinished), unfinished
    record = unseal(INTERRUPTION_CHARGE)
    if (record.get('version') != 'useful-quality-v1-interruption-charge-v1'
            or not set(unfinished) <= set(record.get('cases', {}))
            or not set(record.get('cases', {})) <= set(ORDER)):
        raise ValueError('v1 interruption charge does not cover every unsealed invocation')
    total = 0.0
    for case, item in record['cases'].items():
        seconds = item.get('uncheckpointed_worker_seconds')
        if (not isinstance(seconds, (int, float)) or seconds < 0 or seconds > 86400
                or not item.get('stopped_at') or not item.get('basis')):
            raise ValueError(f'invalid v1 interruption charge for {case}')
        if case in unfinished:
            parent = next(path for path, plan in old_plans()[1] if plan['case'] == case)
            latest = read(parent.parent / 'progress.json')['updated_at']
            if datetime.fromisoformat(latest) > datetime.fromisoformat(item['stopped_at']):
                raise ValueError('v1 work resumed after its interruption charge')
        total += float(seconds)
    return total, False, unfinished


def audit_interruption():
    """Seal a conservative charge after the v1 controller has quiesced."""
    with exclusive(ROOT / 'artifacts/phase3/controller/controller.lock'), pool_locks(ROOT):
        _, pending, unfinished = interrupted_charge()
        if not pending:
            return unseal(INTERRUPTION_CHARGE) if INTERRUPTION_CHARGE.exists() else {
                'version': 'useful-quality-v1-interruption-charge-v1', 'cases': {}}
        stopped = datetime.now(timezone.utc)
        cases = {}
        parent = {plan['case']: path for path, plan in old_plans()[1]}
        for case in unfinished:
            folder = parent[case].parent
            progress = read(folder / 'progress.json')
            checkpoint = datetime.fromisoformat(progress['updated_at'])
            if checkpoint.tzinfo is None or checkpoint > stopped:
                raise ValueError('invalid last v1 checkpoint timestamp')
            seconds = (stopped - checkpoint).total_seconds()
            cases[case] = {'last_checkpoint_at': progress['updated_at'],
                           'stopped_at': stopped.isoformat(),
                           'uncheckpointed_worker_seconds': seconds,
                           'basis': 'conservative elapsed time from last saved image to observed quiescence'}
        document = {'version': 'useful-quality-v1-interruption-charge-v1',
                    'cases': cases, 'quiesced_at': stopped.isoformat()}
        immutable(INTERRUPTION_CHARGE, document)
        return document


def committed_allocations():
    spent, reserved, extension_spent = 0.0, 0.0, 0.0
    records = []
    for path in sorted((BASE / 'allocations').glob('*.json')):
        row = unseal(path)
        completion = BASE / 'completions' / path.name
        if completion.exists():
            done = unseal(completion)
            if done['allocation'] != reference(path):
                raise ValueError('completion/allocation mismatch')
            charge = max(0.0, float(done['wall_seconds']))
            spent += charge
            if row['stage'] == 'extension':
                extension_spent += charge
            state = 'completed'
        else:
            # A crash leaves this reservation in force until reconciled.
            reserved += row['reserved_seconds']
            state = 'reserved_unreconciled'
        records.append({**row, 'state': state})
    return spent, reserved, extension_spent, records


def budget():
    old, cases = old_sunk_seconds()
    interrupted, pending, unfinished = interrupted_charge()
    pre_enrollment = pre_enrollment_cost()
    spent, reserved, extension_spent, allocations = committed_allocations()
    used = old + interrupted + pre_enrollment + spent + reserved
    return {'limit_seconds': LIMIT_SECONDS, 'v1_sunk_seconds': old,
            'v1_uncheckpointed_seconds': interrupted,
            'v2_pre_enrollment_compute_seconds': pre_enrollment,
            'v1_interruption_audit_pending': pending,
            'v1_unsealed_invocations': unfinished,
            'v1_cases': cases, 'v2_spent_seconds': spent,
            'v2_reserved_seconds': reserved,
            'remaining_seconds': 0.0 if pending else max(0.0, LIMIT_SECONDS - used),
            'over_limit_seconds': max(0.0, used - LIMIT_SECONDS),
            'extension_remaining_seconds': max(0.0, EXTENSION_SECONDS - extension_spent -
                                                sum(r['reserved_seconds'] for r in allocations
                                                    if r['stage'] == 'extension' and r['state'] != 'completed')),
            'allocations': allocations}


def counts(path):
    return {(mode, backend): len(list(path.parent.glob(f'[0-9][0-9][0-9][0-9]-{mode}-{backend}.json')))
            for mode in MODES for backend in BACKENDS}


def verify_resume_state(items):
    """Audit all saved checkpoints once before treating counts as progress."""
    from tools.breadth_study.useful_quality_v2_worker import validate_record
    for path, plan in items:
        rows = unseal(checked(plan['protocol']))['sample_rows']
        for mode in MODES:
            for backend in BACKENDS:
                files = sorted(path.parent.glob(f'[0-9][0-9][0-9][0-9]-{mode}-{backend}.json'))
                if len(files) > (8 if backend == 'cpp' else 128):
                    raise ValueError('v2 checkpoint count exceeds its declared panel')
                for index, record_path in enumerate(files):
                    if record_path.name != f'{index:04d}-{mode}-{backend}.json':
                        raise ValueError('v2 checkpoint sequence contains a gap')
                    validate_record(unseal(record_path), plan, rows[index], mode, backend, index)
        if (path.parent / 'native-admission.json').exists():
            verify_gate(path)
        for images in (32, 128):
            if (path.parent / f'summary-{images}.json').exists():
                if not (path.parent / 'native-admission.json').exists():
                    raise ValueError('v2 panel exists without native admission')
                report_panel(path, images)


def measured_rates(path, mode, backend):
    plan = unseal(path)
    records = [unseal(p) for p in path.parent.glob(f'[0-9][0-9][0-9][0-9]-{mode}-{backend}.json')]
    rates = [r['new_compute_seconds'] for r in records if r.get('source_kind') == 'new_verified_execution']
    if rates:
        return statistics.median(rates)
    certificate = unseal(checked(plan['optimized_certificate']))
    calibrated = certificate.get('timing_reservations', {}).get(plan['case'], {}).get(mode, {}).get(backend)
    if calibrated is not None:
        if not isinstance(calibrated, (int, float)) or not 0 < calibrated < 86400:
            raise ValueError('invalid optimized timing reservation')
        return float(calibrated)
    parent = checked(plan['parent_plan'])
    old = [unseal(p) for p in parent.parent.glob(f'[0-9][0-9][0-9][0-9]-{mode}-{backend}.json')]
    rates = [r['new_compute_seconds'] for r in old if r.get('source_kind') == 'new_verified_execution']
    if rates:
        return statistics.median(rates)
    siblings = []
    for old_path, sibling in old_plans()[1]:
        if sibling['model'] != plan['model']:
            continue
        values = [unseal(p)['new_compute_seconds'] for p in old_path.parent.glob(
            f'[0-9][0-9][0-9][0-9]-{mode}-{backend}.json')
            if unseal(p).get('source_kind') == 'new_verified_execution']
        if values:
            siblings.append(statistics.median(values))
    if siblings:
        return max(siblings)
    # First timing probe only: upper-envelope reservation. Its actual timing
    # becomes mandatory before an eight-image or larger stage can be admitted.
    return 3600.0 if backend == 'cpp' else 1200.0


@dataclass(frozen=True)
class Task:
    path: Path
    mode: str
    backend: str
    target: int
    stage: str

    @property
    def case(self):
        return unseal(self.path)['case']


def estimate(task):
    n = counts(task.path)[task.mode, task.backend]
    if n >= task.target:
        return 0.0
    # Integer anchors retain their unchanged frozen execution; floating
    # controls use the newly validated optimized subclass and execute afresh.
    parent = checked(unseal(task.path)['parent_plan'])
    old_count = len(list(parent.parent.glob(f'[0-9][0-9][0-9][0-9]-{task.mode}-{task.backend}.json')))
    importable = unseal(task.path)['format'] == 'int8'
    missing_compute = max(0, task.target - max(n, old_count if importable else n))
    if task.mode == 'exact' and unseal(task.path)['format'] == 'int8':
        missing_compute = 0  # verified retained 1000-image strict screen
    rate = measured_rates(task.path, task.mode, task.backend)
    setup = 90.0 if missing_compute or n < task.target else 0.0
    if task.target == 1 and task.mode == 'exact' and task.backend == 'cpp' and old_count == 0:
        setup += 4200.0  # mandatory first-image legacy comparison if no oracle
    return SAFETY * (missing_compute * rate + setup)


def verify_gate(path):
    from tools.breadth_study.useful_quality_v2_worker import validate_record
    plan = unseal(path)
    rows = unseal(checked(plan['protocol']))['sample_rows']
    conformance_refs = []
    for mode in MODES:
        for backend in BACKENDS:
            base = path.parent / f'native-conformance-{mode}-{backend}.json'
            result = unseal(base)
            if result.get('status') != 'passed' or result.get('backend') != backend:
                raise ValueError('v2 native reduction conformance is incomplete')
            conformance_refs.append(reference(base))
    if plan['format'] != 'int8':
        for backend in BACKENDS:
            control = path.parent / f'optimized-control-conformance-{backend}.json'
            result = unseal(control)
            if (result.get('status') != 'passed' or result.get('backend') != backend
                    or result.get('matches_frozen_v2_layers_outputs_and_all_quantizer_events') is not True):
                raise ValueError('optimized control post-operation conformance is incomplete')
            conformance_refs.append(reference(control))
    for mode in MODES:
        for index in range(8):
            a, b = [unseal(path.parent / f'{index:04d}-{mode}-{backend}.json') for backend in BACKENDS]
            validate_record(a, plan, rows[index], mode, 'cpp', index)
            validate_record(b, plan, rows[index], mode, 'cuda', index)
            fields = ('layers', 'output_sha256', 'prediction')
            if mode != 'exact' or plan['format'] != 'int8':
                fields += ('diagnostic_signature',)
            if any(a[key] != b[key] for key in fields):
                raise ValueError('v2 CPP/CUDA eight-image mismatch')
            if plan['format'] != 'int8':
                if a['source_kind'] != 'new_verified_execution' or b['source_kind'] != 'new_verified_execution':
                    raise ValueError('floating gate requires eight new v2 executions per backend and arm')
    immutable(path.parent / 'native-admission.json',
              {'plan': reference(path), 'images_per_mode_backend': 8,
               'scope': 'v2 study only; historical acceptance unchanged',
               'conformance': conformance_refs,
               'records': [reference(path.parent / f'{i:04d}-{m}-{b}.json')
                           for m in MODES for b in BACKENDS for i in range(8)]})


def report_panel(path, images):
    from tools.breadth_study.useful_quality_v2_worker import validate_record
    plan = unseal(path)
    rows = unseal(checked(plan['protocol']))['sample_rows'][:images]
    arms = {}
    for mode in MODES:
        arm = [unseal(path.parent / f'{i:04d}-{mode}-cuda.json') for i in range(images)]
        for i, record in enumerate(arm):
            validate_record(record, plan, rows[i], mode, 'cuda', i)
        arms[mode] = arm
    for a, b in zip(arms['exact'], arms['control']):
        if (a['ground_truth'], a['fp32_prediction']) != (b['ground_truth'], b['fp32_prediction']):
            raise ValueError('paired label or baseline mismatch')
    stats = {}
    for rank in (1, 5):
        fp = [r['ground_truth'] in r['fp32_prediction'][:rank] for r in arms['exact']]
        ex = [r['ground_truth'] in r['prediction'][:rank] for r in arms['exact']]
        co = [r['ground_truth'] in r['prediction'][:rank] for r in arms['control']]
        stats[f'top{rank}'] = {'control_minus_exact': v1.paired_stats(ex, co),
                               'exact_minus_FP32': v1.paired_stats(fp, ex),
                               'control_minus_FP32': v1.paired_stats(fp, co)}
    nodes = [n['name'] for n in read(checked(plan['graph']))['nodes']]
    differences = []
    for a, b in zip(arms['exact'], arms['control']):
        changed = [name for name in nodes if a['layers'][name] != b['layers'][name]]
        differences.append({'index': a['index'], 'sample_sha256': a['sample']['sha256'],
                            'first_layer': changed[0] if changed else None,
                            'changed_layers': changed, 'output_matches': a['output_sha256'] == b['output_sha256']})
    result = {'plan': reference(path), 'images': images, 'statistics': stats,
              'all_layer_agreement_images': sum(not x['changed_layers'] for x in differences),
              'output_agreement_images': sum(x['output_matches'] for x in differences),
              'divergences': differences, 'records': {m: [reference(path.parent / f'{i:04d}-{m}-cuda.json')
                                                        for i in range(images)] for m in MODES},
              'scope': 'paired canonical-A arithmetic development; v1 provenance retained for migrated records'}
    immutable(path.parent / f'summary-{images}.json', result)
    return result


def ready_tasks(paths):
    """Return work that can advance now, with admission and pairing dependencies."""
    ready = []
    for path, plan in paths:
        have = counts(path)
        admitted = (path.parent / 'native-admission.json').exists()
        fully_probed = all(have[mode, backend] >= 1 for mode in MODES for backend in BACKENDS)
        for mode in MODES:
            if have[mode, 'cpp'] < 1:
                ready.append(Task(path, mode, 'cpp', 1, 'probe'))
            elif fully_probed and have[mode, 'cpp'] < 8:
                ready.append(Task(path, mode, 'cpp', 8, 'native_gate'))
            if have[mode, 'cpp'] >= 1 and have[mode, 'cuda'] < 1:
                ready.append(Task(path, mode, 'cuda', 1, 'probe'))
            elif fully_probed and have[mode, 'cpp'] >= 8 and have[mode, 'cuda'] < 8:
                ready.append(Task(path, mode, 'cuda', 8, 'native_gate'))
        if admitted:
            for mode in MODES:
                if have[mode, 'cuda'] < 32:
                    ready.append(Task(path, mode, 'cuda', 32, 'panel'))
    # Already admitted panels get GPU time while slow unrelated CPU gates run.
    priority = {'panel': 0, 'probe': 1, 'native_gate': 2}
    ready.sort(key=lambda t: (priority[t.stage], ORDER.index(t.case),
                              MODES.index(t.mode), BACKENDS.index(t.backend)))
    return ready


def memory_workers(paths):
    available = v1.memory_available()
    if available < 6 * 2**30:
        return 1
    rss_values = [unseal(p).get('peak_rss_kib', 0) for path, _ in paths
                  for p in path.parent.glob('[0-9][0-9][0-9][0-9]-*-*.json')]
    for old_path, _ in old_plans()[1]:
        rss_values.extend(unseal(p).get('peak_rss_kib', 0) for p in old_path.parent.glob(
            '[0-9][0-9][0-9][0-9]-*-*.json'))
    rss = max(rss_values, default=0) * 1024
    all_probed = all(all(counts(path)[mode, backend] >= 1 for mode in MODES for backend in BACKENDS)
                     for path, _ in paths)
    return 3 if all_probed and available >= 12 * 2**30 and rss < 2 * 2**30 else 2


def allocate(task, *, current_budget):
    cost = estimate(task)
    if cost > current_budget['remaining_seconds']:
        return None
    if task.stage == 'extension' and cost > current_budget['extension_remaining_seconds']:
        return None
    ident = uuid4().hex
    record = {'id': ident, 'version': SCHEMA, 'case': task.case, 'mode': task.mode,
              'backend': task.backend, 'target': task.target, 'stage': task.stage,
              'plan': reference(task.path), 'reserved_seconds': cost, 'allocated_at': now()}
    path = BASE / 'allocations' / f'{ident}.json'
    immutable(path, record)
    return path


def dispatch(task, allocation, fds):
    log = task.path.parent / f'v2-{task.stage}-{task.target}-{task.mode}-{task.backend}.log'
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'OPENBLAS_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE'}
    args = [sys.executable, '-m', 'tools.run.useful_quality_v2', 'worker',
            '--plan', str(task.path), '--mode', task.mode, '--backend', task.backend,
            '--images', str(task.target), '--lease-fds', *map(str, fds)]
    start = time.perf_counter()
    reserved = unseal(allocation)['reserved_seconds']
    timed_out = False
    with log.open('ab') as stream:
        try:
            result = subprocess.run(args, cwd=ROOT, env=env, pass_fds=fds,
                                    stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=max(1.0, reserved))
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            # subprocess.run terminates only this task-owned child. Earlier
            # image seals remain valid; the interrupted image is recomputed
            # only after a separate budget decision.
            timed_out = True
            returncode = 124
    elapsed = time.perf_counter() - start
    outcome = {'allocation': reference(allocation), 'returncode': returncode,
               'timed_out_at_reserved_budget': timed_out,
               'wall_seconds': elapsed, 'log': str(log.relative_to(ROOT)),
               'finished_at': now()}
    immutable(BASE / 'completions' / allocation.name, outcome)
    return outcome


def run():
    items = plans()
    state = {'version': SCHEMA, 'status': 'starting', 'pid': os.getpid(),
             'started_at': now(), 'invocations': []}
    with exclusive(BASE / 'controller.lock'), ExitStack() as locks:
        controller = locks.enter_context(exclusive(ROOT / 'artifacts/phase3/controller/controller.lock'))
        native = locks.enter_context(pool_locks(ROOT))
        fds = (controller.fileno(), native['cuda'], native['cpu'])
        verify_resume_state(items)
        running = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            try:
                while True:
                    for path, _ in items:
                        if all(n >= 8 for n in counts(path).values()) and not (path.parent / 'native-admission.json').exists():
                            verify_gate(path)
                        if (path.parent / 'native-admission.json').exists() and all(counts(path)[m, 'cuda'] >= 32 for m in MODES):
                            if not (path.parent / 'summary-32.json').exists():
                                report_panel(path, 32)
                    ready = ready_tasks(items)
                    occupied = {(task.path, task.mode, task.backend) for task in running.values()}
                    cpu_busy = sum(task.backend == 'cpp' for task in running.values())
                    gpu_busy = sum(task.backend == 'cuda' for task in running.values())
                    capacity = memory_workers(items)
                    blocked = []
                    for task in ready:
                        if len(running) >= capacity:
                            break
                        if ((task.path, task.mode, task.backend) in occupied
                                or (task.backend == 'cuda' and gpu_busy >= 1)
                                or (task.backend == 'cpp' and cpu_busy >= max(1, capacity - gpu_busy))):
                            continue
                        allocation = allocate(task, current_budget=budget())
                        if allocation is None:
                            blocked.append({'case': task.case, 'mode': task.mode, 'backend': task.backend,
                                            'target': task.target, 'stage': task.stage,
                                            'estimated_seconds': estimate(task)})
                            continue
                        future = pool.submit(dispatch, task, allocation, fds)
                        running[future] = task
                        occupied.add((task.path, task.mode, task.backend))
                        cpu_busy += task.backend == 'cpp'
                        gpu_busy += task.backend == 'cuda'
                    state.update(status='running' if running else ('budget_limited' if blocked else 'stages_complete'),
                                 budget=budget(), blocked=blocked, updated_at=now())
                    atomic_json(BASE / 'status.json', state)
                    if not running:
                        break
                    done, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in done:
                        task = running.pop(future)
                        outcome = future.result()
                        state['invocations'].append({'case': task.case, 'mode': task.mode,
                                                     'backend': task.backend, 'target': task.target,
                                                     'returncode': outcome['returncode']})
                        if outcome['returncode']:
                            state.update(status='budget_limited' if outcome['timed_out_at_reserved_budget'] else 'failed',
                                         failed_task={'case': task.case, 'mode': task.mode,
                                                      'backend': task.backend, 'target': task.target},
                                         budget=budget(), updated_at=now())
                            atomic_json(BASE / 'status.json', state)
                            return
                # Extension allocation is frozen in candidate order after all
                # available 32-image panels. Every arm is costed again at dispatch.
                selected = make_promotion(items)
                for path in selected:
                    for mode in MODES:
                        task = Task(path, mode, 'cuda', 128, 'extension')
                        if counts(path)[mode, 'cuda'] >= 128:
                            continue
                        allocation = allocate(task, current_budget=budget())
                        if allocation is None:
                            state.setdefault('extension_budget_limited', []).append(
                                {'case': task.case, 'mode': mode, 'estimated_seconds': estimate(task)})
                            state['status'] = 'budget_limited'
                            atomic_json(BASE / 'status.json', state)
                            break
                        outcome = dispatch(task, allocation, fds)
                        state['invocations'].append({'case': task.case, 'mode': mode,
                                                     'backend': 'cuda', 'target': 128,
                                                     'returncode': outcome['returncode']})
                        if outcome['returncode']:
                            state.update(status='budget_limited' if outcome['timed_out_at_reserved_budget'] else 'failed',
                                         failed_task={'case': task.case, 'mode': mode,
                                                      'backend': 'cuda', 'target': 128})
                            atomic_json(BASE / 'status.json', state)
                            return
                    if all(counts(path)[mode, 'cuda'] >= 128 for mode in MODES):
                        report_panel(path, 128)
            finally:
                state['budget'] = budget()
                state['updated_at'] = now()
                atomic_json(BASE / 'status.json', state)


def make_promotion(items):
    if (BASE / 'promotion.json').exists():
        saved = unseal(BASE / 'promotion.json')
        if saved['scientific_protocol'] != reference(OLD_BASE / 'protocol.json'):
            raise ValueError('frozen promotion protocol changed')
        return [path for path, plan in items
                if any(d['case'] == plan['case'] and d['promote'] for d in saved['decisions'])]
    decisions = []
    remaining = budget()['remaining_seconds']
    extension_remaining = budget()['extension_remaining_seconds']
    for path, plan in items:
        summary = path.parent / 'summary-32.json'
        if not summary.exists():
            decisions.append({'case': plan['case'], 'promote': False,
                              'reason': 'native admission, panel or budget incomplete'})
            continue
        eligible, reason = v1.promotion(unseal(summary))
        cost = sum(estimate(Task(path, mode, 'cuda', 128, 'extension')) for mode in MODES)
        allowed = eligible and cost <= remaining and cost <= extension_remaining
        decisions.append({'case': plan['case'], 'quality_eligible': eligible,
                          'promote': allowed, 'reason': reason if allowed or not eligible else 'measured lifetime or extension budget insufficient',
                          'estimated_extension_seconds': cost})
        if allowed:
            remaining -= cost
            extension_remaining -= cost
    immutable(BASE / 'promotion.json', {'scientific_protocol': reference(OLD_BASE / 'protocol.json'),
                                        'decisions': decisions, 'remaining_seconds': remaining,
                                        'extension_remaining_seconds': extension_remaining})
    return [path for path, plan in items
            if any(d['case'] == plan['case'] and d['promote'] for d in decisions)]


def status():
    state = read(BASE / 'status.json') if (BASE / 'status.json').exists() else {'status': 'not_started'}
    state['budget'] = budget()
    if (BASE / 'enrollment.json').exists():
        state['cases'] = [{'case': p['case'], 'counts': {f'{m}/{b}': n for (m, b), n in counts(path).items()},
                           'native_admission': (path.parent / 'native-admission.json').exists(),
                           'panels': [n for n in (32, 128) if (path.parent / f'summary-{n}.json').exists()]}
                          for path, p in plans()]
    print(json.dumps(state, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'status', 'worker', 'audit-interruption'))
    parser.add_argument('--certificate', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--mode', choices=MODES)
    parser.add_argument('--backend', choices=BACKENDS)
    parser.add_argument('--images', type=int, choices=(1, 8, 32, 128))
    parser.add_argument('--lease-fds', type=int, nargs=3)
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(certificate=args.certificate), indent=2))
    elif args.action == 'run':
        run()
    elif args.action == 'status':
        status()
    elif args.action == 'audit-interruption':
        print(json.dumps(audit_interruption(), indent=2))
    else:
        from tools.breadth_study.useful_quality_v2_worker import worker
        worker(args.plan, args.mode, args.backend, args.images, args.lease_fds)


if __name__ == '__main__':
    main()
