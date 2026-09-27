"""Read-only Phase 3 proof handoff: verify development evidence and live gates."""
import argparse
import json
import os
import subprocess
import sys

from public.inference.conformance_job import source_identity
from tools.phase3.common import ROOT, campaign, checked, digest, file_hash, read


def preflight(root=ROOT):
    plan = campaign(root)
    campaign_id, source = digest(plan), source_identity()
    inventory_path = root / 'results/summaries/phase3-gate-inventory.json'
    inventory = read(inventory_path)
    if inventory['campaign_sha256'] != campaign_id or inventory['source_sha256'] != source:
        raise ValueError('frozen gate inventory differs from current campaign/engine')
    all_keys = {row['configuration'] for row in inventory['records']}
    expected = {f'{model}/{fmt}' for model in plan['models'] for fmt in plan['formats']}
    if len(inventory['records']) != 100 or all_keys != expected:
        raise ValueError('gate inventory is missing or duplicates configurations')
    base = root / 'artifacts/phase3/proof-development'
    reports = {}
    for label, directory, count in (('shared', base / 'shared-v1', 16),
                                    ('remaining', base / 'remaining-v2', 76)):
        summary = read(directory / 'summary.json')
        manifest = read(checked(summary['manifest'], directory))
        if (summary['configuration_count'] != count or summary['graph_acceptance'] is not False or
                manifest['campaign_sha256'] != campaign_id or manifest['source_sha256'] != source or
                len(summary['records']) != count or summary['failures']):
            raise ValueError(f'{label} development proof is incomplete or claims acceptance')
        for ref in manifest['sources'].values():
            checked(ref, directory)
        for ref in manifest.get('reports', {}).values():
            checked(ref, root)
        reports[label] = (directory, summary)
    development = {}
    for label, (directory, summary) in reports.items():
        for ref in summary['records']:
            record = read(checked(ref, directory))
            key = record['configuration']
            local_only = (record.get('graph_acceptance', False) is False and
                          (label != 'shared' or record['status'] == 'development_local_bounds_only_not_graph_acceptance'))
            if (key in development or key not in expected or record['campaign_sha256'] != campaign_id or
                    record['source_sha256'] != source or not local_only):
                raise ValueError('duplicated, stale or promoted development proof')
            gate = next(row for row in inventory['records'] if row['configuration'] == key)
            if record['graph'] != gate['graph'] or record['configuration_artifact'] != gate['configuration_artifact']:
                raise ValueError('development proof belongs to a different graph')
            development[key] = {'family': label, 'proof': ref}
    ready_integer = {f'{model}/{fmt}' for model in ('resnet18', 'mobilenet_v2')
                     for fmt in ('int4', 'int5', 'int6', 'int8')}
    if set(development) != expected - ready_integer:
        raise ValueError('development evidence does not cover all 92 previously blocked graphs')
    local_store = {}
    for label, directory, count, module in (('q1_6', base / 'q1_6-mac-store-v1', 4, 'q1_6_mac_store'),
                                             ('dyadic_fp', base / 'dyadic-fp-mac-store-v1', 24, 'dyadic_fp_mac_store'),
                                             ('posit', base / 'posit-mac-store-v1', 12, 'posit_mac_store'),
                                             ('mapped', base / 'mapped-mac-store-v1', 8, 'mapped_mac_store')):
        summary = read(directory / 'summary.json')
        manifest = read(checked(summary['manifest'], directory))
        if (summary['configuration_count'] != count or len(summary['records']) != count or
                summary['graph_acceptance'] is not False or manifest['campaign_sha256'] != campaign_id or
                manifest['source_sha256'] != source or
                file_hash(root / 'development/acceptance_proofs' / (module + '.py')) != manifest['implementation_sha256']):
            raise ValueError(f'{label} local output-code proof is incomplete or stale')
        for ref in manifest['sources'].values():
            checked(ref, directory)
        for ref in (manifest['gate_inventory'], manifest['nonmac_report']):
            checked(ref, root)
        for ref in summary['records']:
            record = read(checked(ref, directory))
            key = record['configuration']
            gate = next(row for row in inventory['records'] if row['configuration'] == key)
            if (key in local_store or key not in development or record['campaign_sha256'] != campaign_id or
                    record['source_sha256'] != source or record['graph_acceptance'] is not False or
                    record['graph'] != gate['graph'] or record['configuration_artifact'] != gate['configuration_artifact'] or
                    record['proof_implementation_sha256'] != manifest['implementation_sha256']):
                raise ValueError(f'{label} output-code record identity mismatch')
            local_store[key] = record
    static_dir = base / 'family-static-v2'
    static_summary = read(static_dir / 'summary.json')
    static_manifest = read(checked(static_summary['manifest'], static_dir))
    if (static_summary['configuration_count'] != 21 or len(static_summary['records']) != 21 or
            static_summary['graph_acceptance'] is not False or static_manifest['campaign_sha256'] != campaign_id or
            static_manifest['source_sha256'] != source or
            file_hash(root / 'development/acceptance_proofs/family_static.py') != static_manifest['implementation_sha256']):
        raise ValueError('family static verifier evidence is incomplete or stale')
    for ref in static_manifest['sources'].values():
        checked(ref, static_dir)
    static_keys = set()
    for ref in static_summary['records']:
        record = read(checked(ref, static_dir))
        key = record['configuration']
        if (key in static_keys or key not in local_store or record['graph_acceptance'] is not False or
                record['source_sha256'] != source or record['campaign_sha256'] != campaign_id or
                record['verification']['status'] != 'static_graph_arithmetic_reproduced_native_pilot_required' or
                record['verification']['graph'] != next(row['graph'] for row in inventory['records'] if row['configuration'] == key)):
            raise ValueError('family static record is duplicated or belongs to another graph')
        static_keys.add(key)
    locally_complete = {key for key, row in local_store.items() if key.split('/')[0] != 'yolov8n' and
                        row['proof']['status'] == 'conditional_output_codes_proven' and not row['pending_nonmac_nodes']}
    if static_keys != locally_complete:
        raise ValueError('recomputed classifier set differs from local MAC/non-MAC coverage')
    accepted = []
    for row in inventory['records']:
        ref = row['configuration_artifact']
        config = read(checked(ref, root))
        path = root / 'artifacts/phase3/acceptance' / digest(config) / 'prepared-accepted.json'
        if not path.exists():
            continue
        approval = read(path)
        decision = read(checked(approval['screen_acceptance'], root))
        if (decision['configuration_sha256'] != digest(config) or decision['source_sha256'] != source or
                decision['status'] != 'accepted'):
            raise ValueError('retained graph approval identity mismatch')
        accepted.append(row['configuration'])
    missing_approval = sorted(expected - set(accepted))
    result = {'schema_version': 'phase3-development-preflight-1.0.0',
              'campaign_sha256': campaign_id, 'source_sha256': source,
              'development_configurations': len(development), 'accepted_configurations': len(accepted),
              'missing_graph_acceptance_count': len(missing_approval),
              'missing_graph_acceptance': missing_approval,
              'local_mac_nodes_bounded': reports['shared'][1]['mac_nodes'] + reports['remaining'][1]['mac_nodes_checked'],
              'exact_ordinary_fp64_prefix_mac_nodes': reports['remaining'][1]['ordinary_prefix_exact_mac_nodes'],
              'pending_nonmac_nodes': reports['remaining'][1]['remaining_nonmac_nodes'] +
                                      sum(len(row['pending_nonmac_nodes']) for row in inventory['records']
                                          if row['configuration'] in development and development[row['configuration']]['family'] == 'shared'),
              'pending_posit_bias_layers': reports['remaining'][1]['remaining_bias_layers'],
              'conditional_mac_output_code_proven_configurations': sum(
                  row['proof']['status'] == 'conditional_output_codes_proven' for row in local_store.values()),
              'classifier_graphs_with_local_mac_and_nonmac_proofs': len(locally_complete),
              'recomputed_static_classifier_candidates': len(static_keys),
              'pending_sensitive_dyadic_fp_channels': sum(
                  row['proof']['pending_channels'] for key, row in local_store.items() if key.split('/')[1].startswith('fp')),
              'pending_sensitive_posit_channels': sum(
                  row['proof']['pending_channels'] for key, row in local_store.items() if key.split('/')[1].startswith('posit')),
              'pending_sensitive_mapped_channels': sum(
                  row['proof']['pending_channels'] for key, row in local_store.items()
                  if key.split('/')[1] in {'binary_pm1', 'ternary'}),
              'resume_accepted_work_command': '.venv/bin/python -m tools.run.phase3_resume',
              'noninteger_screening_verifier_available': False,
              'status': 'acceptance_and_screening_dispatch_still_required'}
    return result


def test_archived_proofs(root=ROOT):
    """Run the retained proof tests without depending on the development checkout."""
    base = root / 'artifacts/phase3/proof-development'
    source_dirs, tests = [], []
    for label, version, module in (('shared', 'shared-v1', 'shared_mac_proof'),
                                    ('remaining', 'remaining-v2', 'remaining_proof')):
        directory = base / version
        manifest = read(directory / 'manifest.json')
        prefix = f'development/{label}_proofs/'
        sources = manifest['sources']
        source_dirs.append(str(checked(sources[prefix + module + '.py'], directory).parent))
        tests.append(str(checked(sources[prefix + 'test_' + module + '.py'], directory)))
    for version, module in (('q1_6-mac-store-v1', 'q1_6_mac_store'),
                            ('dyadic-fp-mac-store-v1', 'dyadic_fp_mac_store'),
                            ('posit-mac-store-v1', 'posit_mac_store'),
                            ('mapped-mac-store-v1', 'mapped_mac_store')):
        directory = base / version
        manifest = read(directory / 'manifest.json')
        tests.append(str(checked(manifest['sources']['test_' + module + '.py'], directory)))
    static_dir = base / 'family-static-v2'
    static_manifest = read(static_dir / 'manifest.json')
    tests.append(str(checked(static_manifest['sources']['test_family_static.py'], static_dir)))
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONPATH': os.pathsep.join((str(root), *source_dirs, os.environ.get('PYTHONPATH', '')))}
    return subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *tests],
                          cwd=root, env=env, check=False).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', action='store_true', help='print counts and next command only')
    parser.add_argument('--test', action='store_true', help='also run archived focused proof tests')
    args = parser.parse_args()
    result = preflight()
    if args.summary:
        result.pop('missing_graph_acceptance')
    print(json.dumps(result, indent=2), flush=True)
    if args.test:
        raise SystemExit(test_archived_proofs())


if __name__ == '__main__':
    main()
