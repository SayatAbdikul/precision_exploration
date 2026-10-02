"""Run one Experiment B2 detector job: run, regress, stats or recipes."""
import os

# Must be fixed before the first CUDA context for deterministic cuBLAS operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __name__ == "__main__":
    from tools.experiment_b2_det.runner import main
    raise SystemExit(main())
