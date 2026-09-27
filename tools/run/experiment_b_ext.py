"""Run/resume the versioned Experiment B extension."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from tools.experiment_b_ext.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
