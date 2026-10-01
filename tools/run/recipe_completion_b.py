"""Resume seven sealed percentile B configurations using the unchanged CUDA runner."""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from tools.breadth_study.recipe_completion_b import main

if __name__ == '__main__':
    main()
