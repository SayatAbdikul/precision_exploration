"""Lane Q4: B2 activation attribution and repair arms (see tools/experiment_b2_attrib/runner.py)."""
import os

# Must be fixed before the first CUDA context for deterministic cuBLAS operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __name__ == "__main__":
    from tools.experiment_b2_attrib.runner import main
    raise SystemExit(main())
