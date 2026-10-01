import pytest

from tools.hardware.ics55_ecc_pilot_summary import _ecc_qor, _instance_area, _metric


REPORT = """| DIE Area ( um^2 )      | 9605.568064 = 98.008000 * 98.008000 |
| CORE Area ( um^2 )     | 8685.600000 = 94.000000 * 92.400000 |
| All Instances | 2330   | 1            | 4066.44 | 1          |
"""


def test_ecc_area_parser_reads_area_column_not_instance_count():
    assert _metric(REPORT, "DIE Area ( um^2 )") == 9605.568064
    assert _metric(REPORT, "CORE Area ( um^2 )") == 8685.6
    assert _instance_area(REPORT) == 4066.44
    with pytest.raises(ValueError):
        _instance_area(REPORT + REPORT)


def test_ecc_qor_is_explicitly_estimated():
    row = "clk                       6.818        0.0       0     314MHz      0.173        0.0       0\n"
    assert _ecc_qor(row) == (6.818, 314)
    with pytest.raises(ValueError):
        _ecc_qor(row + row)
