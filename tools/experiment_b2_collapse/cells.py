"""B2 matrix cells redirected into this lane's folder (rule: never write into artifacts/experiment_b2/matrix).

``tools.experiment_b2.matrix.run_cell`` (read-only) is called unchanged inside ``redirected()``,
which patches, in this process only, ``matrix.MATRIX`` to ``artifacts/experiment_b2_collapse/matrix``
and adds this file to the record's ``own_sources``.  The configuration identity does not depend
on either.  When the original matrix holds a cell with the same identity, its readout arrays
are compared with the redirected cell's and the result is written to ``matrix/checks/``.

    GPU_LANE=Q3 artifacts/agent_orchestration/gpu_run.sh [--heavy 8000] .venv-b/bin/python -m \
        tools.run.experiment_b2_collapse cell --model M --format F --recipes R1,R2
"""
from __future__ import annotations

from contextlib import contextmanager
import json

import numpy as np

from tools.experiment_b.common import ROOT, file_hash, seal, unseal

from .common import BASE

THIS = "tools/experiment_b2_collapse/cells.py"


@contextmanager
def redirected():
    from tools.experiment_b2 import matrix
    original_matrix, original_own = matrix.MATRIX, matrix.own_sources

    def own_sources():
        return {**original_own(), THIS: file_hash(ROOT / THIS)}

    matrix.MATRIX, matrix.own_sources = BASE / "matrix", own_sources
    try:
        yield matrix, original_matrix
    finally:
        matrix.MATRIX, matrix.own_sources = original_matrix, original_own


def compare(record, original_matrix):
    """Compare a redirected cell with the original matrix cell of the same identity, if one exists."""
    from tools.experiment_b2 import readout
    found = sorted((original_matrix / "cells").glob(
        f"{record['model']}--{record['format']}--{record['recipe_name']}--1000--{record['configuration_sha256'][:12]}.json"))
    if not found:
        return None
    original = unseal(found[0])
    mine = readout.load(ROOT / record["readout_file"])
    theirs = readout.load(ROOT / original["readout_file"])
    return {"original_cell": str(found[0].relative_to(ROOT)),
            "identity_equal": original["configuration_sha256"] == record["configuration_sha256"],
            "logits_sha256_equal": original["logits_sha256"] == record["logits_sha256"],
            "readout_arrays_equal": all(np.array_equal(mine[k], theirs[k]) for k in readout.DTYPES)}


def run_cells(model, format_name, recipes, device):
    import torch  # noqa: F401
    from tools.experiment_b2 import frozen  # noqa: F401
    with redirected() as (matrix, original_matrix):
        for recipe in recipes:
            record = matrix.run_cell(model, format_name, recipe, device)
            check = compare(record, original_matrix)
            if check is not None:
                path = BASE / "matrix" / "checks" / f"{model}--{format_name}--{recipe}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    seal(path, check)
            print(json.dumps({"model": model, "format": format_name, "recipe": recipe, "status": record["status"],
                              "top1_expected": round(record["readout"]["top1_expected_percent"], 2),
                              "check": check}), flush=True)
    return 0
