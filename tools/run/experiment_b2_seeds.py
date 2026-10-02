"""Run lane Q5 calibration-subset cells (see tools/experiment_b2_seeds/cells.py)."""
import os

# Must be fixed before the first CUDA context for deterministic cuBLAS operations (as the matrix runner does).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __name__ == "__main__":
    from tools.experiment_b2_seeds.cells import main
    raise SystemExit(main())
