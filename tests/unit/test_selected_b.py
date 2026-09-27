from copy import deepcopy
from pathlib import Path

import pytest

from tools.breadth_study.selected_b import MATRIX, selected_keys, verify_configuration
from tools.experiment_b.common import digest, seal, unseal


def test_selection_preserves_all_pairs_and_only_frozen_recipes():
    matrix = unseal(MATRIX)
    keys = selected_keys(matrix)
    assert len(keys) == 171
    assert len({(m, f) for m, f, _ in keys}) == 100
    bad = deepcopy(matrix)
    bad['E3_B_breadth_extensions'].append(bad['E3_B_breadth_extensions'][0])
    with pytest.raises(ValueError):
        selected_keys(bad)


def test_reject_numerically_changed_or_unselected_configuration(tmp_path):
    payload = {'scale': '1/8', 'recipe': 'maxabs'}
    identity = digest(payload)
    path = tmp_path / f'{identity}.json'
    seal(path, payload)
    verify_configuration(path, payload, {identity})
    with pytest.raises(ValueError):
        verify_configuration(path, {**payload, 'scale': '1/4'}, {identity})
    with pytest.raises(ValueError):
        verify_configuration(path, payload, set())
