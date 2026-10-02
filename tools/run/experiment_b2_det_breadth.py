"""Run one GPU job of the detector breadth study (lane Q6): gate, formats, seeds, groups or fine."""
import os

# Must be fixed before the first CUDA context for deterministic cuBLAS operations (as tools/run/experiment_b2_det.py).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __name__ == "__main__":
    from tools.experiment_b2_det_breadth.run import main
    raise SystemExit(main())
