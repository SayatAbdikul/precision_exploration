"""Resume the gated E1 family expansion and E2 recipe experiments."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from tools.breadth_study.study_next import main

if __name__ == '__main__':
    main()
