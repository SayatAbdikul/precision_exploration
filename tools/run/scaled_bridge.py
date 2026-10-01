"""Resume the bounded scaled ResNet18 FP6/FP7 arithmetic bridge."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import argparse
import json
from tools.scaled_bridge_v1 import controller

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','check','status','report','worker','build'))
    parser.add_argument('arguments',nargs='*');args=parser.parse_args()
    if args.action=='worker':controller.worker(args.arguments)
    elif args.action=='status':
        from tools.scaled_bridge_v1.report import summary
        print(json.dumps(summary(),indent=2))
    elif args.action=='report':
        from tools.scaled_bridge_v1.report import write_report
        write_report()
    elif args.action=='build':
        from tools.scaled_bridge_v1.native import build
        for b in ('cpp','cuda'):print(build(b))
    else:
        try:controller.run(check_only=args.action=='check')
        except controller.BudgetStop as error:
            print('Budget stop:',error)
            from tools.scaled_bridge_v1.report import write_report
            write_report()

if __name__=='__main__':main()
