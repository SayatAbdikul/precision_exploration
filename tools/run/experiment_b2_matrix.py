"""Run one Experiment B2 matrix job: cells, baseline, regress-shared or campaign."""
import os

# Must be fixed before the first CUDA context for deterministic cuBLAS operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __name__ == "__main__":
    from tools.experiment_b2.matrix import main
    raise SystemExit(main())
