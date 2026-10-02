import numpy as np
import pytest

from tools.accumulator_sweep_v1.score import image_scores, derived, node_event_rates
from tools.accumulator_sweep_v1 import certs


def rec(i, label, tied, rest=(), events=None, failure=None):
    top5 = sorted(tied) + list(rest)
    return {'index': i, 'label': label, 'top1_tied_classes': list(tied), 'top5': top5[:5], 'events': events or {},
            'failure': failure}


def test_expected_and_lowest_index_credit():
    images = [rec(0, 3, [3]), rec(1, 7, [2, 7]), rec(2, 2, [2, 7]), rec(3, 9, [1, 4, 9]), rec(4, 5, [6])]
    s = image_scores(images)
    assert s['expected'].tolist() == pytest.approx([1, .5, .5, 1 / 3, 0])
    assert s['lowest'].tolist() == [1, 0, 1, 0, 0]
    assert s['top1'].tolist() == [3, 2, 2, 1, 6]
    assert s['tied'].tolist() == [1, 2, 2, 3, 1]


def test_failed_image_scores_zero_and_counts_event():
    images = [rec(0, 3, [3]), {'index': 1, 'label': 3, 'top1_tied_classes': None, 'top5': None,
                               'events': {'conv1': {'nonfinite_elements': 5}}, 'failure': {'kind': 'x'}}]
    s = image_scores(images)
    assert s['expected'].tolist() == [1, 0] and s['lowest'].tolist() == [1, 0]
    assert s['failed'].tolist() == [False, True] and s['event'].tolist() == [False, True] and s['top1'][1] == -1


def test_tie_order_and_contiguity_are_checked():
    bad = rec(0, 3, [3, 5]); bad['top5'] = [5, 3]
    with pytest.raises(ValueError):
        image_scores([bad])
    with pytest.raises(ValueError):
        image_scores([rec(0, 1, [1]), rec(2, 1, [1])])


def test_node_event_rates():
    images = [rec(0, 1, [1], events={'a': {'saturated_elements': 2, 'high_clamps': 3}}), rec(1, 1, [1])]
    r = node_event_rates(images, {'a': {'elements': 4}, 'b': {'elements': 4}})
    assert r['a'] == {'images_with_event': 1, 'image_rate': .5, 'output_rate': .25}
    assert r['b']['images_with_event'] == 0


def grid(rows):
    return {w: {'events': e, 'changed': c, 'expected_percent': p} for w, e, c, p in rows}


def test_derived_widths_uniform_family():
    g = grid([(24, 0, 0, 70.0), (23, 3, 0, 70.0), (22, 9, 2, 69.6), (21, 20, 9, 69.0), (20, 90, 40, 60.0),
              (19, 100, 300, 34.0), (18, 100, 600, 10.0)])
    d = derived(g, 70.0)
    assert d == {'noevent': 24, 'same': 23, 'acc_0.5': 22, 'acc_1.0': 21, 'half': 19}


def test_derived_requires_every_safer_setting():
    # accuracy recovers at 20 by chance below a failing 21: 'from upward' stops at 22
    g = grid([(23, 0, 0, 70.0), (22, 1, 1, 69.8), (21, 5, 9, 68.0), (20, 9, 12, 69.9)])
    d = derived(g, 70.0)
    assert d['acc_0.5'] == 22 and d['acc_1.0'] == 22 and d['half'] is None


def test_derived_none_when_safest_fails():
    g = grid([(20, 4, 3, 60.0), (19, 9, 9, 30.0)])
    d = derived(g, 70.0)
    assert d['noevent'] is None and d['same'] is None and d['acc_1.0'] is None and d['half'] == 19


def test_derived_per_node_family_direction():
    g = grid([(0, 0, 0, 70.0), (1, 0, 0, 70.0), (2, 5, 1, 69.9), (4, 50, 30, 66.0), (6, 99, 500, 20.0)])
    d = derived(g, 70.0, safer_is_larger=False)
    assert d == {'noevent': 1, 'same': 1, 'acc_0.5': 2, 'acc_1.0': 2, 'half': 6}


def test_float_exponent_rule():
    rows = {'a': {'abs': 27, 'struct': 26, 'range': 26, 'bound_units': (1 << 25) + 5, 'product_shift': 0, 'K': 9},
            'b': {'abs': 10, 'struct': 10, 'range': 10, 'bound_units': 300, 'product_shift': 0, 'K': 9}}
    assert certs.magnitude_exponent(rows) == 26
    assert certs.float_policy('fp16', rows) == 'fp16.x-11'   # |sum| < 2^26, scaled < 2^15
    assert certs.float_policy('f21', rows) == 'f21'          # plain f21 is proven overflow-free and normal
    small = {'a': dict(rows['b'], bound_units=1000, product_shift=4)}  # |sum| < 2^6
    assert certs.float_policy('fp16', small) == 'fp16'
    tiny = {'a': dict(rows['b'], bound_units=1000, product_shift=30)}  # smallest step 2^-30 < 2^-14
    assert certs.float_exponent('fp16', tiny) == 15 - (10 - 30)
