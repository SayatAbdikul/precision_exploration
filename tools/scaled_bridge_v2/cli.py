"""Scaled code-domain bridge v2: export, witness, gate and report commands."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('build')
    p = sub.add_parser('export'); p.add_argument('model'); p.add_argument('format'); p.add_argument('recipe')
    p = sub.add_parser('conformance'); p.add_argument('codebooks', nargs='+'); p.add_argument('--case')
    sub.add_parser('policy-conformance')
    p = sub.add_parser('panel'); p.add_argument('case'); p.add_argument('policy'); p.add_argument('backend'); p.add_argument('images', type=int)
    p = sub.add_parser('throughput'); p.add_argument('case'); p.add_argument('policy'); p.add_argument('backend')
    p.add_argument('images', type=int); p.add_argument('--batch', type=int, default=1); p.add_argument('--trace', default='full')
    p = sub.add_parser('predict'); p.add_argument('case'); p.add_argument('policy'); p.add_argument('backend')
    p.add_argument('start', type=int); p.add_argument('stop', type=int); p.add_argument('--batch', type=int, default=8)
    p = sub.add_parser('replay'); p.add_argument('case'); p.add_argument('images', type=int)
    p = sub.add_parser('gate'); p.add_argument('case')
    p = sub.add_parser('policy-gate'); p.add_argument('case'); p.add_argument('policy')
    p = sub.add_parser('regress'); p.add_argument('case'); p.add_argument('policy'); p.add_argument('backend')
    p.add_argument('images', type=int); p.add_argument('old_run')
    p = sub.add_parser('seal-protocol'); p.add_argument('name')
    p = sub.add_parser('report'); p.add_argument('--milestone', default='M1'); p.add_argument('--run'); p.add_argument('--cutoff', type=float)
    sub.add_parser('archive')
    p = sub.add_parser('b2-export'); p.add_argument('folder')
    p = sub.add_parser('b2-replay'); p.add_argument('case'); p.add_argument('images', type=int)
    p = sub.add_parser('b2-gate'); p.add_argument('case')
    p = sub.add_parser('graph-witness'); p.add_argument('case'); p.add_argument('policies', nargs='+')
    p.add_argument('--backend', default='cpp'); p.add_argument('--image', type=int, default=0)
    p = sub.add_parser('fp16-policy'); p.add_argument('case')
    sub.add_parser('root')
    args = parser.parse_args(argv)
    if args.action == 'build':
        from .native import build
        for backend in ('cpp', 'cuda'):
            print(build(backend))
    elif args.action == 'export':
        from .export import export_b1
        print(json.dumps(export_b1(args.model, args.format, args.recipe)))
    elif args.action == 'conformance':
        from .conformance import codebook_checks
        for name in args.codebooks:
            codebook_checks(name, args.case)
    elif args.action == 'policy-conformance':
        from .conformance import policy_checks
        policy_checks()
    elif args.action == 'panel':
        from .worker import panel
        panel(args.case, args.policy, args.backend, args.images)
    elif args.action == 'throughput':
        from .worker import throughput
        throughput(args.case, args.policy, args.backend, args.images, args.batch, args.trace)
    elif args.action == 'predict':
        from .worker import predict
        result = predict(args.case, args.policy, args.backend, args.start, args.stop, args.batch)
        print(json.dumps({k: result[k] for k in ('case', 'policy', 'backend', 'start', 'stop', 'compared_with_sealed_batch1')}))
    elif args.action == 'replay':
        from .replay import replay
        replay(args.case, args.images)
    elif args.action == 'gate':
        from .worker import gate
        gate(args.case)
    elif args.action == 'policy-gate':
        from .worker import policy_gate
        policy_gate(args.case, args.policy)
    elif args.action == 'regress':
        from .worker import regress
        regress(args.case, args.policy, args.backend, args.images, args.old_run)
    elif args.action == 'seal-protocol':
        from . import contract
        print(contract.seal_protocol(args.name.lower(), getattr(contract, 'PROTOCOL_' + args.name.upper())))
    elif args.action == 'report':
        from . import report
        if args.milestone == 'MS':
            report.write_policy_report(args.run or report.MS_RUN, args.cutoff or report.MS_LEDGER_CUTOFF)
        elif args.milestone == 'MB2':
            report.write_b2_report(args.run, args.cutoff)
        elif args.milestone == 'MN':
            report.write_mn_report(args.run, args.cutoff)
        else:
            report.write_report(args.milestone, args.run, args.cutoff)
    elif args.action == 'archive':
        from .archive import archive
        print(archive())
    elif args.action == 'b2-export':
        from .b2_adapter import export_b2
        print(json.dumps(export_b2(args.folder)))
    elif args.action == 'b2-replay':
        from .b2_replay import replay
        replay(args.case, args.images)
    elif args.action == 'b2-gate':
        from .b2_replay import gate
        gate(args.case)
    elif args.action == 'graph-witness':
        from .graph_witness import witness
        for policy in args.policies:
            witness(args.case, policy, args.backend, args.image)
    elif args.action == 'fp16-policy':
        from .common import unseal, run_root
        certs = unseal(run_root() / args.case / 'certificate.json')['certificates'].values()
        ub = max(c['max_abs_prefix_units'].bit_length() - c['product_shift'] for c in certs)
        print('fp16' if 15 - ub >= 0 else f'fp16.x{15 - ub}')
    elif args.action == 'root':
        from .common import run_root
        print(run_root())
