"""Run one read-only Experiment B2 verification job: bias-residual, vendor-check or vendor-probe."""
import os

# Must be fixed before the first CUDA context for deterministic cuBLAS operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __name__ == "__main__":
    from tools.experiment_b2.verify import main
    raise SystemExit(main())
