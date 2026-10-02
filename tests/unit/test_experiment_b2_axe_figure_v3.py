"""Lane Q8 figure v3: derived sat.wP points (above L8's event-free width) are separated from measured ones."""
from tools.experiment_b2_axe.figure_v3 import split_sat


def test_split_sat_separates_derived_points():
    rows = [{"case": "c", "method": "sat", "P": 22, "top1_expected": 69.0, "source": "derived: P >= W_noevent=21, equals wide"},
            {"case": "c", "method": "sat", "P": 21, "top1_expected": 69.0, "source": "measured"},
            {"case": "c", "method": "sat", "P": 20, "top1_expected": 68.0, "source": "measured"},
            {"case": "c", "method": "axe", "P": 22, "top1_expected": 1.0, "source": "measured"},
            {"case": "d", "method": "sat", "P": 23, "top1_expected": 5.0, "source": "derived: x"}]
    measured, derived = split_sat(rows, "c")
    assert measured == [(20, 68.0), (21, 69.0)]
    assert derived == [(22, 69.0)]
