"""Fast exact-engine path (lane S1): build, root, predict (the archives' CLI), compare, regress."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('build'); sub.add_parser('root'); sub.add_parser('gates'); sub.add_parser('archive')
    p = sub.add_parser('predict'); p.add_argument('case'); p.add_argument('policy'); p.add_argument('backend')
    p.add_argument('start', type=int); p.add_argument('stop', type=int); p.add_argument('--batch', type=int, default=8)
    p = sub.add_parser('compare'); p.add_argument('case'); p.add_argument('policy'); p.add_argument('start', type=int)
    p.add_argument('stop', type=int); p.add_argument('--batch', type=int, default=8)
    p = sub.add_parser('regress'); p.add_argument('case'); p.add_argument('policy'); p.add_argument('backend')
    p.add_argument('images', type=int); p.add_argument('old_root'); p.add_argument('--batch', type=int, default=8)
    args = parser.parse_args(argv)
    if args.action == 'build':
        from .native import build
        print(build())
    elif args.action == 'gates':
        from .archive import gates
        print(json.dumps(gates(), indent=0))
    elif args.action == 'archive':
        from .archive import archive
        print(archive())
    elif args.action == 'root':
        from .common import run_root
        print(run_root())
    elif args.action == 'predict':
        from .worker import predict
        r = predict(args.case, args.policy, args.backend, args.start, args.stop, args.batch)
        print(json.dumps({k: r[k] for k in ('case', 'policy', 'backend', 'start', 'stop', 'compared_with_sealed_batch1')}))
    elif args.action == 'compare':
        from .worker import compare
        r = compare(args.case, args.policy, args.start, args.stop, args.batch)
        print(json.dumps({k: v for k, v in r.items() if k != 'archive_files'}))
    elif args.action == 'regress':
        from .worker import regress
        print(json.dumps(regress(args.case, args.policy, args.backend, args.images, args.old_root, args.batch)))
