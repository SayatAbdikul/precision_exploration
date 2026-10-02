"""Lane Q2 reproduction gate (rule 12): a cell run through this lane's redirected machinery equals L1's sealed cell.

The GPU runs (``tools.run.experiment_b2_rank reproduce``) wrote redirected cells into
artifacts/experiment_b2_rank/reproduction/cells/.  This test re-checks them from the sealed records themselves:
same configuration identity, same logits sha256, every readout array equal, readout files unchanged since sealing.
"""
from pathlib import Path

import numpy as np
import pytest

from tools.experiment_b.common import ROOT, file_hash, unseal
from tools.experiment_b2 import readout

REPRO = ROOT / "artifacts/experiment_b2_rank/reproduction/cells"
L1 = ROOT / "artifacts/experiment_b2/matrix/cells"
CASES = [("resnet18", "fp6_e2m3", "default"), ("resnet18", "mxfp8_e4m3", "cum5_act_maxabs")]


def _one(folder: Path, model, name, recipe):
    found = sorted(folder.glob(f"{model}--{name}--{recipe}--1000--*.json"))
    assert len(found) == 1, (folder, model, name, recipe, found)
    return unseal(found[0])


@pytest.mark.parametrize("model,name,recipe", CASES)
def test_redirected_cell_reproduces_l1_bit_for_bit(model, name, recipe):
    if not REPRO.exists():
        pytest.skip("reproduction cells not run on this checkout")
    mine, theirs = _one(REPRO, model, name, recipe), _one(L1, model, name, recipe)
    assert mine["configuration_sha256"] == theirs["configuration_sha256"]
    assert mine["logits_sha256"] == theirs["logits_sha256"]
    assert mine["rows_sha256"] == theirs["rows_sha256"]
    for record in (mine, theirs):
        assert file_hash(ROOT / record["readout_file"]) == record["readout_file_sha256"]
    assert mine["readout_file"].startswith("artifacts/experiment_b2_rank/")
    a, b = readout.load(ROOT / mine["readout_file"]), readout.load(ROOT / theirs["readout_file"])
    for key in readout.DTYPES:
        assert np.array_equal(a[key], b[key]), key
