import numpy as np
import pytest
import torch

from tools.run.phase3_residual_rounding_propagation import ReplaceResidual, array_hash


def test_replacement_requires_exact_fresh_target_and_single_execution():
    original = torch.tensor([[1.0, -2.0]])
    replacement = np.array([[0.5, -1.0]], dtype=np.float32)
    observer = ReplaceResidual("add", array_hash(original.numpy()), replacement)
    observer.record("unrelated", original)
    assert observer.calls == 0
    observer.record("add", original)
    assert observer.calls == 1
    assert np.array_equal(original.numpy(), replacement)
    with pytest.raises(ValueError, match="differs"):
        observer.record("add", original)
    observer = ReplaceResidual("add", array_hash(np.array([[1.0, -2.0]], dtype=np.float32)), replacement)
    with pytest.raises(ValueError, match="differs"):
        observer.record("add", torch.zeros((1, 2)))
