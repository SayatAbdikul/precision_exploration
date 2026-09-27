"""Prepare controlled clipping and adaptive-rounding recipe graphs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def main():
    from tools.breadth_study.e2 import CASES, clipping, rounding
    from tools.breadth_study.rounding_data import collect
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'collect', 'clip', 'round'))
    parser.add_argument('--model', choices=('resnet18', 'mobilenet_v2'))
    parser.add_argument('--format', choices=('int4', 'int5', 'int6'))
    args = parser.parse_args()
    if args.action == 'prepare':
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(collect, model) for model in ('resnet18', 'mobilenet_v2')]
            futures += [pool.submit(clipping, model, fmt) for model, fmt in CASES]
            for future in futures:
                print(future.result(), flush=True)
    elif args.action == 'collect':
        print(collect(args.model))
    elif args.action == 'clip':
        print(clipping(args.model, args.format))
    else:
        from tools.phase3.worker_locks import pool_locks
        from tools.phase3.common import ROOT
        with pool_locks(ROOT):
            print(rounding(args.model, args.format))


if __name__ == '__main__':
    main()
