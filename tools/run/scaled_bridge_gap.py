"""1k simulator-versus-exact gap study on the sealed scaled bridge v1 engine."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('protocol', 'arm', 'replay', 'analyze', 'diagnose', 'ties', 'stagefix'))
    parser.add_argument('arguments', nargs='*'); args = parser.parse_args()
    a = args.arguments
    if args.action == 'protocol':
        from tools.scaled_bridge_gap_v1.common import write_protocol
        print(write_protocol())
    elif args.action == 'arm':
        from tools.scaled_bridge_gap_v1.arms import run_arm
        run_arm(a[0], a[1], int(a[2]), int(a[3]))
    elif args.action == 'replay':
        from tools.scaled_bridge_gap_v1.arms import run_replay
        run_replay(a[0], int(a[1]), int(a[2]))
    elif args.action == 'analyze':
        from tools.scaled_bridge_gap_v1.analysis import write_results
        write_results()
    elif args.action == 'diagnose':
        from tools.scaled_bridge_gap_v1.diagnostics import main as diag
        diag(a)
    elif args.action == 'ties':
        from tools.scaled_bridge_gap_v1.ties import main as ties
        ties(a)
    elif args.action == 'stagefix':
        from tools.scaled_bridge_gap_v1.stagefix import main as stagefix
        stagefix(a)


if __name__ == '__main__':
    main()
