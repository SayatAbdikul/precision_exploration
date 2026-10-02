"""Lane Q4 (agent r6): pcu per-channel arms, protocol addendum 8 (see tools/experiment_b2_attrib/pcurunner.py)."""
import os

# Must be fixed before the first CUDA context for deterministic cuBLAS operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __name__ == "__main__":
    from tools.experiment_b2_attrib.pcurunner import main
    raise SystemExit(main())
