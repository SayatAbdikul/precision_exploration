"""Prepare resumable native pilots for the 92 graphs awaiting acceptance.

The batch uses the production controller's locks and worker implementation.
It cannot run beside an active controller. Pilots never issue graph acceptance.
"""
import argparse
import json
import sys

from public.inference.conformance_job import source_identity
from tools.analysis.phase3_development_preflight import preflight
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.screening import pipeline_identity


def pilot_inventory(root=ROOT):
    report = preflight(root)
    plan, source, pipeline = campaign(root), source_identity(), pipeline_identity(root)
    prepared = preparation_records(root)
    baselines = read(root / 'results/summaries/phase3-baselines.json')['models']
    blocked = set(report['missing_graph_acceptance'])
    static_ready = set()
    base = root / 'artifacts/phase3/proof-development'
    for version in ('q1_6-mac-store-v1', 'dyadic-fp-mac-store-v1', 'posit-mac-store-v1', 'mapped-mac-store-v1'):
        directory = base / version
        summary = read(directory / 'summary.json')
        for ref in summary['records']:
            row = read(checked(ref, directory))
            if (row['configuration'].split('/')[0] != 'yolov8n' and
                    row['proof']['status'] == 'conditional_output_codes_proven' and
                    not row['pending_nonmac_nodes']):
                static_ready.add(row['configuration'])
    tasks = []
    for key in sorted(blocked):
        row = prepared[key]
        model, fmt = key.split('/')
        config = read(checked(row['configuration'], root))
        if config['runtime']['source_sha256'] != source or config['campaign_sha256'] != digest(plan):
            raise ValueError('prepared pilot identity differs from frozen execution')
        prepared_path = checked(row['configuration'], root).parent / 'prepared.json'
        priority = '0' if key in static_ready else '1'
        tasks.append({'id': priority + '-' + row['configuration_sha256'] + '-pilot',
                      'kind': 'pilot', 'model': model, 'format': fmt,
                      'configuration_sha256': row['configuration_sha256'],
                      'prepared': reference(prepared_path, root), 'baseline': baselines[model],
                      'campaign_sha256': digest(plan), 'source_sha256': source,
                      'pipeline_sha256': pipeline, 'dependencies': [], 'resource': 'cuda'})
    if len(tasks) != 92 or len(static_ready) != 21:
        raise ValueError('pilot population or local static-proof priority changed')
    inventory = {'schema_version': 'phase3-development-pilot-inventory-1.0.0',
                 'campaign_sha256': digest(plan), 'source_sha256': source,
                 'pipeline_sha256': pipeline, 'planned_configurations': len(tasks),
                 'tasks': tasks, 'blocked': [], 'phase3_complete': False}
    return inventory, static_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='start pilots; requires the current controller to have stopped')
    parser.add_argument('--workers', type=int, choices=range(1, 7), default=2)
    args = parser.parse_args()
    inventory, static_ready = pilot_inventory()
    print(json.dumps({'pilot_configurations': len(inventory['tasks']),
                      'locally_static_ready_first': len(static_ready),
                      'native_backends': ['cpp', 'cuda'], 'images_per_graph': 8,
                      'workers': args.workers, 'will_run': args.run}, indent=2), flush=True)
    if not args.run:
        return 0
    from tools.phase3.controller import schedule
    from tools.phase3.controller_resources import GIB
    try:
        code = schedule(inventory, gpu_workers=args.workers, ram_per_worker=3*GIB)
    except RuntimeError as error:
        if 'live local writer' in str(error):
            print('The existing Phase 3 controller still owns the native locks; rerun after it finishes.', file=sys.stderr)
            return 2
        raise
    if code == 1:
        return 1
    status = read(ROOT / 'artifacts/phase3/controller/status.json')
    return 0 if status['counts']['completed'] == len(inventory['tasks']) else 2


if __name__ == '__main__':
    raise SystemExit(main())
