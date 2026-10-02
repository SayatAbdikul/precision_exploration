from tools.accumulator_sweep_mn import plan


def r(events, acc):
    return {'events': events, 'expected_percent': acc}


def test_ladder_starts_below_certificate_step_two():
    assert plan.ladder(25) == [24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2]
    assert plan.ladder(45)[:3] == [44, 42, 40] and len(plan.ladder(45)) == 12


def test_ladder_done_at_half():
    assert not plan.ladder_done({24: r(0, 70), 22: r(5, 60)}, 70)
    assert plan.ladder_done({24: r(0, 70), 18: r(9, 35)}, 70)


def test_refine_fills_single_gaps():
    res = {24: r(0, 70), 22: r(0, 70), 20: r(3, 69), 18: r(9, 50), 16: r(9, 1)}
    assert plan.refine(res, 70, 25) == [21, 17]
    res.update({21: r(0, 70), 17: r(9, 20)})
    assert plan.refine(res, 70, 25) == []
    assert plan.bracket(res, 70, 25) == (15, 21)


def test_refine_all_events_asks_up_to_cert():
    res = {20: r(5, 60), 18: r(9, 1)}
    assert plan.refine(res, 70, 23) == [22, 21, 19]


def test_bracket_no_half_and_clip():
    assert plan.bracket({24: r(0, 70), 22: r(1, 69)}, 70, 25) == (20, 24)
    assert plan.bracket({4: r(1, 1), 3: r(1, 0)}, 70, 25) == (2, 24)


def test_struct_ladder_and_set():
    assert plan.struct_next({}, 70) == 1
    assert plan.struct_next({1: r(9, 69)}, 70) == 2
    assert plan.struct_next({1: r(9, 69), 2: r(9, 30)}, 70) is None
    assert plan.struct_set({1: r(9, 69), 2: r(9, 30)}, 70) == [1, 2, 3]
    assert plan.struct_set({1: r(9, 30)}, 70) == [1, 2]
    res = {d: r(9, 60) for d in range(1, 5)}
    res[5] = r(9, 2)
    assert plan.struct_set(res, 70) == [3, 4, 5, 6]
    assert plan.struct_set({d: r(9, 60) for d in plan.STRUCT_LADDER}, 70) == [6, 7, 8]
    assert plan.struct_next({d: r(9, 60) for d in plan.STRUCT_LADDER}, 70) is None


def test_stress_widths():
    assert plan.stress_widths(13, 19) == [19, 17, 15, 13]
    assert plan.stress_widths(12, 19) == [19, 17, 15, 13, 12]
    assert plan.stress_widths(5, 5) == [5]


def test_screen_grid_priorities():
    g = plan.screen_grid('c', 15, 17, 'fp16.x-9', 'f21', [1, 2])
    assert [(x['priority'], x['policy']) for x in g] == [
        (1, 'wide'), (1, 'control'), (2, 'sat.w17'), (2, 'sat.w16'), (2, 'sat.w15'), (3, 'fp16.x-9'), (3, 'f21'),
        (4, 'sat.struct-1'), (4, 'sat.struct-2')]
    g = plan.screen_grid('c', 13, 18, 'fp16.x-4', 'f21', [1, 2], stress=True)
    assert [x['policy'] for x in g if x['priority'] == 2] == ['sat.w18', 'sat.w16', 'sat.w14', 'sat.w13']
    assert not [x for x in g if x['priority'] == 4]


def test_extensions():
    assert plan.extend_top(21, True, 25) == 22 and plan.extend_top(24, True, 25) is None
    assert plan.extend_top(21, False, 25) is None
    assert plan.extend_bottom({17: r(0, 70), 16: r(0, 30), 15: r(0, 1)}, 70) == 14
    assert plan.extend_bottom({17: r(0, 70), 16: r(0, 30), 15: r(0, 1), 14: r(0, 0)}, 70) is None
    assert plan.extend_bottom({17: r(0, 70), 16: r(0, 60)}, 70) == 15


def test_sensitivity_names():
    assert plan.sensitivity_policies(-9, 24) == ['fp16.x-12', 'fp16.x-11', 'fp16.x-10', 'fp16.x-8', 'fp16.x-7',
                                                 'fp16.x-6', 'f21.x-9']
    assert plan.sensitivity_policies(-1, 16)[3] == 'fp16'
