from fractions import Fraction

import numpy as np

from public.inference.tensor import Encoding
from tools.breadth_study.recipes import refined_scale, adaptive_round, signed_codes


def test_mse_refinement_keeps_original_candidate_and_handles_zero():
    values = np.r_[np.linspace(-1, 1, 101), 20.].astype(np.float32)
    result = refined_scale(values, 'int4', Fraction(1, 4))
    assert 0 < float(result['scale'])
    assert result['mse'] <= result['original_mse']
    zero = refined_scale(np.zeros(10), 'int4', Fraction(1))
    assert zero['scale'] == '1.0' and zero['mse'] == 0


def test_adaptive_rounding_is_discrete_scale_preserving_and_resumable(tmp_path, monkeypatch):
    from pathlib import Path
    from tools.breadth_study import recipes
    from tools.experiment_b.common import file_hash
    original_checked = recipes.checked
    monkeypatch.setattr(recipes, 'reference', lambda p: {'path': str(Path(p).resolve()), 'sha256': file_hash(p)})
    monkeypatch.setattr(recipes, 'checked', lambda r: original_checked(r, root=tmp_path))
    weights = np.array([[[.49, .49], [-.49, -.49]]], dtype=np.float32)
    patches = np.array([[[1., 1.]], [[2., 2.]], [[-1., -1.]], [[.5, .5]]], dtype=np.float32)
    original = np.zeros((1, 2, 2), dtype=np.uint8)
    encoding = Encoding('int4', (1, 1), axis=0)
    codes, report = adaptive_round(weights, patches, original, encoding, device='cpu', directory=tmp_path, iterations=20)
    resumed, resumed_report = adaptive_round(weights, patches, original, encoding, device='cpu', directory=tmp_path, iterations=20)
    np.testing.assert_array_equal(codes, resumed)
    assert report == resumed_report
    signed = signed_codes(codes, 4)
    assert np.all((signed == np.floor(weights)) | (signed == np.ceil(weights)))
    assert encoding.scales == (1, 1)
    if not report['retained_original_codes']:
        assert report['fitted_patch_mse'] <= report['original_patch_mse']
