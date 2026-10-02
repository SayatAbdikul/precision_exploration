"""Lane S1 identity evidence for the fast exact-engine path (protocol speed-protocol-v1, part 1c).

  regress CASE...                    the MN-R record sets of these cases (runs/1f75c923.../regress/*.json): fast path,
                                     full trace, 8 images, numerical() against the sealed records of the old root
  trace CASE... [--batch 8]          every sealed batch-1 full-trace record folder <policy>-cuda of the case in the
                                     archive's run root (7c6344af for ResNet18, 1f75c923 for the MobileNets):
                                     fast path with full trace, numerical() equality
  compare CASE POLICY... [--start 0 --stop 64 --batches 8]
                                     fast predict items against the archives' sealed predict items (and against
                                     fresh archive runs under artifacts/speed_v1/archive-runs/), per batch size
Results: artifacts/speed_v1/validation/{regress,compare}/*.json (one file per set; an existing file is kept).
Run on the GPU through artifacts/agent_orchestration/gpu_run.sh (GPU_LANE=S1).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from tools.scaled_bridge_fast.common import SPEED, V2, BASE_DIGEST, run_root, digest, identity, unseal, immutable

OUT = SPEED / 'validation'
ARCHIVE_RUNS = SPEED / 'archive-runs'
RESNET = '7c6344af732d1bf324c07b285921f63393150cdb3875356c86237d934fdc2f92'


def regress_sets(cases):
    folder = V2 / 'runs' / BASE_DIGEST / 'regress'
    for path in sorted(folder.glob('*.json')):
        r = unseal(path)
        if r['case'] in cases and r['status'] == 'pass':
            yield path.stem, r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    a = sub.add_parser('regress'); a.add_argument('cases', nargs='+')
    a = sub.add_parser('trace'); a.add_argument('cases', nargs='+'); a.add_argument('--batch', type=int, default=8)
    a = sub.add_parser('compare'); a.add_argument('case'); a.add_argument('policies', nargs='+')
    a.add_argument('--start', type=int, default=0); a.add_argument('--stop', type=int, default=64)
    a.add_argument('--batches', default='8')
    args = p.parse_args()
    from tools.scaled_bridge_fast import worker
    tag = digest(identity())[:16]
    failures = 0
    if args.action == 'regress':
        for name, r in regress_sets(set(args.cases)):
            target = OUT / 'regress' / tag / f'{name}.json'
            if target.exists():
                continue
            old_root = V2 / 'runs' / r['old_engine_sources']
            result = worker.regress(r['case'], r['policy'], r['backend'], r['images_identical'], old_root, batch=8)
            result.update(regress_record=str(Path(V2.name) / 'runs' / BASE_DIGEST / 'regress' / f'{name}.json'),
                          fast_identity=identity())
            immutable(target, result)
            ok = result['identical'] == result['images'] and not result['differences']
            failures += not ok
            print(json.dumps({'set': name, 'identical': result['identical'], 'images': result['images'], 'ok': ok}), flush=True)
    elif args.action == 'trace':
        # Every sealed batch-1 full-trace record set of the case in the archive's own run root (gate panels:
        # <policy>-cuda/NNNN.json), fast path with full trace at --batch, numerical() equality.
        for case in args.cases:
            root = V2 / 'runs' / (RESNET if case.startswith('resnet18-') else BASE_DIGEST)
            for folder in sorted((root / case).glob('*-cuda')):
                policy = folder.name[:-len('-cuda')]
                if policy == 'B':
                    continue
                count = len(list(folder.glob('[0-9][0-9][0-9][0-9].json')))
                target = OUT / 'trace' / tag / f'{case}--{policy}--{count}--b{args.batch}.json'
                if target.exists() or not count:
                    continue
                result = worker.regress(case, policy, 'cuda', count, root, batch=args.batch)
                result.update(sealed_folder=str(folder.relative_to(V2.parent.parent)), fast_identity=identity())
                immutable(target, result)
                ok = result['identical'] == result['images'] and not result['differences']
                failures += not ok
                print(json.dumps({'case': case, 'policy': policy, 'identical': result['identical'], 'images': count,
                                  'batch': args.batch, 'ok': ok}), flush=True)
    else:
        for policy in args.policies:
            ctx = None                     # one engine per policy, reused across the batch sizes (r3)
            for batch in [int(b) for b in args.batches.split(',')]:
                target = OUT / 'compare' / tag / f'{args.case}--{policy}--{args.start}-{args.stop}--b{batch}.json'
                if target.exists():
                    continue
                ctx = ctx or worker.context(args.case, policy)
                result = worker.compare(args.case, policy, args.start, args.stop, batch, ctx=ctx,
                                        extra_roots=(ARCHIVE_RUNS / BASE_DIGEST, ARCHIVE_RUNS / '7c6344af732d1bf324c07b285921f63393150cdb3875356c86237d934fdc2f92'))
                result['fast_identity'] = identity()
                immutable(target, result)
                ok = result['compared'] >= args.stop - args.start and result['different'] == 0 and result['node_constants_equal'] is not False
                failures += not ok
                print(json.dumps({k: result[k] for k in ('case', 'policy', 'batch', 'compared', 'identical', 'different', 'node_constants_equal')}), flush=True)
    print('VALIDATE-DONE failures', failures, flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
