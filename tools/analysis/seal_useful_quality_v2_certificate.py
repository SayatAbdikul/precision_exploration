"""Seal the pre-enrollment FP64-grid v2 evidence and honest compute charge.

This script records the already completed bounded benchmarks. It does not run
any study image or change the frozen scientific protocol.
"""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from public.inference.native_fp64_grid_v2 import GraphGridCertificate
from tools.breadth_study import useful_quality_v2 as study
from tools.exact_execution_v2.optimized_engine import runtime_identity
from tools.experiment_b.common import unseal
from tools.phase3.common import ROOT, checked, digest, read, reference
from tools.run.exact_execution import immutable


BENCH = study.BASE / 'benchmarks'
NATIVE = ROOT / 'results/summaries/fp64-grid-v2-certificates-2026-09-28.json'
JUNIT = ROOT / 'results/summaries/fp64-grid-v2-conformance-junit.xml'
FAILED_SANDBOX_SETUP_SECONDS = 16.0  # observed 15.3548 s, rounded up


def benchmark(name: str) -> tuple[Path, dict]:
    path = BENCH / name
    return path, unseal(path)


def image_rate(item: dict) -> float:
    return float(item['seconds']['input_preparation'] + item['seconds']['execution'])


def main() -> None:
    _, parents = study.old_plans()
    by_case = {plan['case']: plan for _, plan in parents}
    native = json.loads(NATIVE.read_text())
    if set(native['cases']) != {'fp6_e2m3', 'fp6_e3m2', 'fp7_e3m3'}:
        raise ValueError('native grid certificate missing a selected format')
    junit = ET.parse(JUNIT).getroot()
    suites = [junit] if junit.tag == 'testsuite' else list(junit.iter('testsuite'))
    if not suites or any(int(s.get(key, '0')) for s in suites for key in ('failures', 'errors')):
        raise ValueError('native/post-operation conformance suite did not pass')
    test_count = sum(int(s.get('tests', '0')) for s in suites)
    if test_count < 30:
        raise ValueError('conformance test report is unexpectedly incomplete')

    e2_cpp_path, e2_cpp = benchmark('resnet18-fp6_e2m3-cpp-image0.json')
    e2_cuda_path, e2_cuda = benchmark('resnet18-fp6_e2m3-cuda-image0.json')
    e3_cpp_path, e3_cpp = benchmark('resnet18-fp6_e3m2-cpp-image0.json')
    control_cpp_path, control_cpp = benchmark(
        'resnet18-fp6_e2m3-optimized-control-v3-sourcebound-cpp-image0.json')
    control_cuda_path, control_cuda = benchmark(
        'resnet18-fp6_e2m3-optimized-control-v3-sourcebound-cuda-image0.json')
    fp7_cpp_path, fp7_cpp = benchmark('resnet18-fp7_e3m3-exact-no-oracle-cpp-image0.json')
    fp7_cuda_path, fp7_cuda = benchmark('resnet18-fp7_e3m3-exact-no-oracle-cuda-image0.json')
    numerical_fields = ('layers', 'output_sha256', 'prediction', 'diagnostic_signature')
    parity = {key: fp7_cpp['numerical'][key] == fp7_cuda['numerical'][key]
              for key in numerical_fields}
    if not all(parity.values()):
        raise ValueError('FP7 native CPU/CUDA first-image numerical divergence')
    if any(item['matches'] is not None or item['oracle'] is not None
           for item in (fp7_cpp, fp7_cuda)):
        raise ValueError('FP7 probe must not masquerade as whole-graph rational parity')
    cross = study.BASE / 'fp7-first-image-crossbackend-proof.json'
    immutable(cross, {'scope': 'first-image optimized CPU/CUDA parity only; no FP7 whole-graph rational oracle',
                      'cpp': reference(fp7_cpp_path), 'cuda': reference(fp7_cuda_path),
                      'numerical_fields': list(numerical_fields), 'matches': parity,
                      'source_identity': runtime_identity()})

    selected = {}
    for case in ('resnet18/fp6_e2m3', 'resnet18/fp6_e3m2', 'resnet18/fp7_e3m3'):
        plan = by_case[case]
        graph = read(checked(plan['graph']))
        selected[case] = {'graph_sha256': plan['graph_sha256'],
                          'grid_certificate_sha256': digest(GraphGridCertificate(graph).document())}

    timings = {
        'resnet18/fp6_e2m3': {
            'exact': {'cpp': image_rate(e2_cpp), 'cuda': image_rate(e2_cuda)},
            'control': {'cpp': image_rate(control_cpp), 'cuda': image_rate(control_cuda)}},
        'resnet18/fp6_e3m2': {
            'exact': {'cpp': image_rate(e3_cpp), 'cuda': image_rate(e2_cuda)},
            'control': {'cpp': image_rate(control_cpp), 'cuda': image_rate(control_cuda)}},
        'resnet18/fp7_e3m3': {
            'exact': {'cpp': image_rate(fp7_cpp), 'cuda': image_rate(fp7_cuda)},
            'control': {'cpp': image_rate(control_cpp), 'cuda': image_rate(control_cuda)}}}
    provenance = {
        'resnet18/fp6_e2m3': {'exact': {'cpp': reference(e2_cpp_path), 'cuda': reference(e2_cuda_path)},
                              'control': {'cpp': reference(control_cpp_path), 'cuda': reference(control_cuda_path)}},
        'resnet18/fp6_e3m2': {'exact': {'cpp': reference(e3_cpp_path), 'cuda': 'E2M3 CUDA same-graph proxy'},
                              'control': {'cpp': 'E2M3 CPU same-graph proxy', 'cuda': 'E2M3 CUDA same-graph proxy'}},
        'resnet18/fp7_e3m3': {'exact': {'cpp': reference(fp7_cpp_path), 'cuda': reference(fp7_cuda_path)},
                              'control': {'cpp': 'E2M3 CPU same-graph proxy', 'cuda': 'E2M3 CUDA same-graph proxy'}}}
    floor = study.benchmark_compute_floor()
    certificate = {
        'version': 'useful-quality-fp64-grid-v2-certificate', 'status': 'passed',
        'runtime_identity': runtime_identity(), 'selected_graphs': selected,
        'timing_reservations': timings, 'timing_provenance': provenance,
        'pre_enrollment_compute_seconds': floor + FAILED_SANDBOX_SETUP_SECONDS,
        'pre_enrollment_charge_basis': {'sealed_full_image_benchmark_seconds': floor,
                                         'failed_sandbox_setup_upper_seconds': FAILED_SANDBOX_SETUP_SECONDS},
        'validation': {
            'native_conformance': {
                backend: {'status': 'passed', 'grid_certificate': reference(NATIVE),
                          'junit': reference(JUNIT), 'test_count': test_count}
                for backend in ('cpp', 'cuda')},
            'full_image_comparisons': {
                'resnet18/fp6_e2m3': {'cpp': reference(e2_cpp_path), 'cuda': reference(e2_cuda_path)},
                'resnet18/fp6_e3m2': {'cpp': reference(e3_cpp_path)}},
            'matched_control_comparisons': {
                'resnet18/fp6_e2m3': {'cpp': reference(control_cpp_path),
                                       'cuda': reference(control_cuda_path)}},
            'fp7_first_image_crossbackend': reference(cross),
            'fp7_whole_graph_rational_oracle': 'absent; strict native admission remains blocked under the current lifetime cap'}}
    immutable(study.CERTIFICATE_PATH, certificate)
    study.validate_certificate(reference(study.CERTIFICATE_PATH), parents, study.source_identity())
    print(json.dumps({'certificate': reference(study.CERTIFICATE_PATH),
                      'sealed_benchmark_seconds': floor,
                      'charged_pre_enrollment_seconds': certificate['pre_enrollment_compute_seconds'],
                      'fp7_crossbackend': reference(cross), 'conformance_tests': test_count}, indent=2))


if __name__ == '__main__':
    main()
