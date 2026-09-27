from pathlib import Path

import pytest

from tools.run.exact_execution import worker


def test_internal_worker_cannot_bypass_controller_or_native_leases():
    with pytest.raises(ValueError,match='inherited controller and native leases'):
        worker(Path('/nonexistent/plan.json'),None)


def test_unrelated_open_descriptors_are_not_worker_leases(tmp_path):
    with (tmp_path/'unrelated').open('w') as stream:
        with pytest.raises(ValueError,match='expected lock'):
            worker(Path('/nonexistent/plan.json'),[stream.fileno()]*3)
