"""Run the frozen twelve B counterparts with per-image resume."""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from tools.breadth_study.balanced_b import main

if __name__ == '__main__':
    main()
