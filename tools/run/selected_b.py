"""Run the frozen selected B extensions with per-image resume."""
import os

# Match the original launcher before importing torch or initializing CUDA.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from tools.breadth_study.selected_b import main

if __name__ == '__main__':
    main()
