"""Lane Q4 (agent r3): low-memory gate and arms jobs (see tools/experiment_b2_attrib/lmrunner.py)."""
import os

# Must be fixed before the first CUDA context for deterministic cuBLAS operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __name__ == "__main__":
    from tools.experiment_b2_attrib.lmrunner import main
    raise SystemExit(main())
