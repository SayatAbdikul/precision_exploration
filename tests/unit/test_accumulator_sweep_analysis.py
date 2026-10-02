import pytest

from tools.analysis import accumulator_sweep as a


def test_published_width_formula():
    # contract 2.1 quotes 26 and 34 bits for fp6_e2m3 and fp7_e3m3 at n = 4,608
    assert a.published_width('fp6_e2m3') == 26 and a.published_width('fp7_e3m3') == 34
    assert a.published_width('int8') == 8 + 8 + 13 + 1
    assert a.published_width('int6') == 26
    assert a.published_width('fp8_e4m3fn') == 2 * (16 + 3) + 12
    assert a.published_width('posit8_es1') is None


def test_policy_family():
    assert a.policy_family('sat.w17') == ('uniform', 17)
    assert a.policy_family('sat.struct-4') == ('per_node', 4)
    assert a.policy_family('fp16.x-11') == ('fp16', -11)
    assert a.policy_family('f21') == ('f21', 0)
    assert a.policy_family('wide') == ('wide', None)
    with pytest.raises(ValueError):
        a.policy_family('bogus')


def test_hardware_join_table():
    areas = a.hardware_areas(8)
    assert set(w for w, _ in areas) == {4, 5, 6, 8}
    assert {16, 20, 24, 28, 32} <= set(acc for _, acc in areas)
    assert areas[(8, 32)] > areas[(8, 16)] > 0


def test_case_format():
    assert a.case_format('resnet18-int8-default_signed-b2') == 'int8'
    assert a.case_format('resnet18-fp8_e4m3fn-default-b2') == 'fp8_e4m3fn'


def test_join_rows_measured_derived_not_run_and_cert_cap():
    rows = {'sat.w16': {'top1_expected': 10.0, 'top1_lowest_index': 11.0}}
    areas = {(8, 16): 1.0, (8, 20): 2.0, (8, 24): 3.0, (8, 28): 4.0, (8, 32): 5.0}
    out = a.join_rows_for('resnet18-int8-default-b2', 27, rows, 70.0, 71.0, 21, areas)
    by = {r['accumulator_bits']: r for r in out}
    assert sorted(by) == [16, 20, 24]  # 28 and 32 lie above the certified 27 bits
    assert by[16]['source'] == 'measured' and by[16]['top1_expected'] == 10.0 and by[16]['core_area_8ns'] == 1.0
    assert by[20]['source'] == 'not run' and by[20]['top1_expected'] is None  # 20 <= W_noevent 21, no run
    assert by[24]['source'].startswith('derived') and by[24]['top1_lowest_index'] == 71.0
    assert all(r['multiplier_bits'] == 8 for r in out)
    six = a.join_rows_for('resnet18-int6-default-b2', 23, {}, 60.0, 60.0, None, {(6, 16): 9.0})
    assert [r['accumulator_bits'] for r in six] == [16, 20] and all(r['source'] == 'not run' for r in six)
    assert six[0]['multiplier_bits'] == 6 and six[0]['core_area_8ns'] == 9.0 and six[1]['core_area_8ns'] is None
