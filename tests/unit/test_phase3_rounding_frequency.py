import numpy as np
import pytest

from tools.run.phase3_residual_rounding_frequency import count_pairs, rounding_table
from public.inference.tensor import Encoding


def test_frequency_counts_repeated_ordered_pairs_without_dropping_multiplicity():
    expected = np.zeros((2, 2), dtype=np.uint8)
    actual = expected.copy()
    actual[0, 1] = 1
    result = count_pairs([0, 0, 1, 0], [1, 1, 0, 0], expected, actual)
    assert result["elements"] == 4
    assert result["stored_output_discrepancies"] == 2
    assert result["observed_pairs"] == 3
    assert result["discrepant_pairs"][0]["count"] == 2
    with pytest.raises(ValueError, match="outside"):
        count_pairs([2], [0], expected, actual)


def test_table_is_symmetric_and_reproduces_log6_discrepancies():
    expected, actual = rounding_table(Encoding("log6"))
    assert np.array_equal(expected, expected.T)
    assert np.array_equal(actual, actual.T)
    assert np.count_nonzero(expected != actual) == 52
