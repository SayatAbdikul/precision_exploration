"""Lane Q8: the v1 runner (tools.run.experiment_b2_axe, unchanged) with the memory-lean bias correction.

    GPU_LANE=Q8 artifacts/agent_orchestration/gpu_run.sh --min-free-mib 3000 \
        .venv-b/bin/python -m tools.run.experiment_b2_axe_lean run --case resnet18-int6 --arms ... --out ...

Patches, inside this process only, ``tools.experiment_b2_axe.build.bias_correct`` with
``tools.experiment_b2_axe.lean.bias_correct_lean`` (bit-identical computation, activations parked in host memory),
and adds the two lean files to the source hashes the runner records.  Records written by this entry point carry
``"bias_correction_path": "lean"`` only through those source hashes; the folder they go to says so too.
"""
import sys

import tools.run.experiment_b2_axe as runner
from tools.experiment_b2_axe import build, lean

LEAN_SOURCES = ("tools/experiment_b2_axe/lean.py", "tools/run/experiment_b2_axe_lean.py")

build.bias_correct = lean.bias_correct_lean
runner.OWN_SOURCES = tuple(runner.OWN_SOURCES) + LEAN_SOURCES

if __name__ == "__main__":
    sys.exit(runner.main())
