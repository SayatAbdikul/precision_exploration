from fractions import Fraction
import numpy as np

from public.analysis.phase3.head_store import calibration_positions, store_head
from public.inference.tensor import Encoding
from public.quantization.ptq.encoding import float_tensor


def test_frozen_flat_observation_misses_three_coordinate_channels():
    result = calibration_positions((1, 84, 8400))
    assert result["coordinate_channels_sampled"] == [0]
    assert result["coordinate_samples"] == 1 and result["score_samples"] == 15
    assert result["positions"][0] == {"flat_index": 0, "channel": 0, "box_index": 0}


def test_final_store_preserves_reference_quantization_and_exposes_clipping():
    values = np.linspace(0, 640, 168, dtype=np.float32).reshape(1, 84, 2)
    values[:, 4:] /= 640
    encoding = Encoding("int8", (Fraction(21102811755426018, 10**17),))
    original = values.copy()
    actual, codes, diagnostics = store_head(values, encoding)
    expected = float_tensor(values, encoding)
    assert np.array_equal(actual, np.asarray([float(v) for v in expected.values()], dtype=np.float32).reshape(values.shape))
    assert codes.ravel().tolist() == list(expected.codes)
    assert np.array_equal(values, original)
    assert diagnostics["positive_scores_rounded_to_zero"] > 0
    assert diagnostics["maximum_representable"] == str(127*encoding.scales[0])
