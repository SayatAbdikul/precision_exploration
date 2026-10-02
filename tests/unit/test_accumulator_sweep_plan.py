from tools.accumulator_sweep_v1 import plan
from tools.accumulator_sweep_v1.load import covering


def test_ladder():
    assert plan.ladder(27) == [25, 22, 19, 16, 13, 10, 7, 4]
    assert len(plan.ladder(63)) == plan.LADDER_MAX_RUNGS


def r(events, pct):
    return {'events': events, 'expected_percent': pct}


def test_bracket_rule():
    res = {25: r(0, 70), 22: r(0, 70), 19: r(5, 69), 16: r(100, 30), 13: r(128, 1)}
    assert plan.ladder_done(res, 70)
    assert plan.bracket(res, 70, 27) == (14, 22)
    # widest rung already has events: top is W_cert - 1
    res = {25: r(1, 70), 22: r(9, 60), 19: r(100, 20)}
    assert plan.bracket(res, 70, 27) == (17, 26)
    # no rung reached half: bottom from the narrowest rung
    res = {25: r(0, 70), 22: r(3, 68)}
    assert not plan.ladder_done(res, 70)
    assert plan.bracket(res, 70, 27) == (20, 25)


def test_screen_grid_and_priorities():
    g = plan.screen_grid('c', 27, 14, 22, True, 'fp16.x-11', 'f21')
    pol = [(x['priority'], x['policy']) for x in g]
    assert pol[0] == (1, 'wide') and (1, 'sat.w22') in pol and (1, 'sat.w14') in pol and (1, 'sat.w13') not in pol
    assert (2, 'control') in pol and (2, 'fp16.x-11') in pol and (2, 'f21') in pol
    assert [p for pr, p in pol if pr == 3] == []  # 16 and 20 are inside the bracket, 24 above, 28 32 > cert
    g = plan.screen_grid('c', 27, 18, 22, True, 'fp16.x-11', 'f21')
    assert [x['policy'] for x in g if x['priority'] == 3] == ['sat.w16']
    g = plan.screen_grid('c', 47, 18, 22, False, 'fp16.x-13', 'f21')
    assert [x['policy'] for x in g if x['priority'] == 3] == []
    assert [x['policy'] for x in g if x['priority'] == 4] == ['sat.struct-4', 'sat.struct-8']


def test_extensions():
    assert plan.extend_top(22, True, 27) == 23
    assert plan.extend_top(26, True, 27) is None
    assert plan.extend_top(22, False, 27) is None
    res = {22: r(0, 70), 21: r(1, 70), 20: r(9, 34), 19: r(9, 10)}
    assert plan.extend_bottom(res, 70) == 18      # only one width below W_half = 20
    res[18] = r(9, 1)
    assert plan.extend_bottom(res, 70) is None
    assert plan.extend_bottom({22: r(0, 70), 21: r(3, 60)}, 70) == 20  # half not reached


def test_hardware_derived():
    assert plan.hardware_derived(27, 23) == [24, 28, 32]
    assert plan.hardware_derived(23, 21) == [24, 28, 32]


def test_covering_tiles():
    files = [(0, 32, 'a'), (0, 64, 'b'), (64, 128, 'c'), (128, 1000, 'd'), (64, 1000, 'e')]
    assert [f[2] for f in covering(files, 0, 1000)] == ['b', 'e']
    assert covering(files, 0, 500) is None
    assert [f[2] for f in covering(files, 0, 128)] == ['b', 'c']


def test_refine():
    res = {25: r(0, 70), 22: r(0, 70), 19: r(5, 69), 16: r(100, 30), 13: r(128, 1)}
    assert plan.refine(res, 70, 27) == [21, 20, 18, 17]
    res.update({21: r(0, 70), 20: r(2, 70), 18: r(9, 40), 17: r(50, 33)})
    assert plan.refine(res, 70, 27) == []
    assert plan.bracket(res, 70, 27) == (15, 21)
    # widest rung has events: refine up to W_cert - 1
    assert plan.refine({25: r(3, 70), 22: r(9, 20)}, 70, 27) == [26, 24, 23]


def test_width_cap_63():
    assert plan.ladder(75)[:2] == [61, 58] and len(plan.ladder(75)) == plan.LADDER_MAX_RUNGS
    res = {61: r(5, 70), 58: r(9, 60), 55: r(9, 20)}
    assert plan.bracket(res, 70, 75)[1] == 63
    assert plan.refine(res, 70, 75) == [63, 62, 57, 56]
    assert plan.extend_top(63, True, 75) is None
