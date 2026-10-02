"""Run Experiment B2 matrix cells with non-finite audit SQNR recorded as text (see tools/experiment_b2/matrix_finite.py)."""
import os

# Must be fixed before the first CUDA context for deterministic cuBLAS operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __name__ == "__main__":
    from tools.experiment_b2.matrix_finite import main
    raise SystemExit(main())
