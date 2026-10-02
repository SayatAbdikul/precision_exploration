"""Scoring of compact `predict` records and the derived widths of the accumulator sweep (protocol v1, item d).

All functions are pure; reading files is in `load.py`.
"""
import numpy as np


def image_scores(images):
    """Per-image arrays from a list of predict image records (ordered by index).

    expected: tie-aware expected credit, 1/t when the label is among the t tied maxima (0 for a failed image);
    lowest: 1 when the lowest-index maximum (contract tie order: Top-5[0]) is the label;
    top1: the lowest-index Top-1 class (-1 for a failed image); event: any non-zero accumulator event count;
    failed: the image has no prediction (non-finite accumulator).
    """
    n = len(images)
    expected = np.zeros(n); lowest = np.zeros(n, dtype=np.int8); top1 = np.full(n, -1, dtype=np.int64)
    event = np.zeros(n, dtype=bool); failed = np.zeros(n, dtype=bool); tied = np.ones(n, dtype=np.int64)
    for i, r in enumerate(images):
        if r['index'] != images[0]['index'] + i:
            raise ValueError('image records are not contiguous')
        event[i] = bool(r.get('events'))
        if r.get('failure') is not None:
            failed[i] = True
            continue
        ties = r['top1_tied_classes']
        if r['top5'][0] != min(ties):
            raise ValueError('Top-5 is not in contract tie order')
        tied[i] = len(ties)
        top1[i] = r['top5'][0]
        expected[i] = (1.0 / len(ties)) if r['label'] in ties else 0.0
        lowest[i] = int(r['top5'][0] == r['label'])
    return {'expected': expected, 'lowest': lowest, 'top1': top1, 'event': event, 'failed': failed, 'tied': tied}


def node_event_rates(images, nodes):
    """Per MAC node: share of images with an event at that node and share of its outputs with an event."""
    out = {}
    n = len(images)
    for name, const in nodes.items():
        hit = 0; elems = 0
        for r in images:
            ev = (r.get('events') or {}).get(name)
            if ev:
                hit += 1
                elems += ev.get('saturated_elements', 0) + ev.get('nonfinite_elements', 0)
        out[name] = {'images_with_event': hit, 'image_rate': hit / n if n else 0.0,
                     'output_rate': elems / (n * const['elements']) if n and const.get('elements') else None}
    return out


def derived(grid, wide_expected_percent, safer_is_larger=True):
    """Derived widths of one family from per-setting summaries.

    grid: {setting: {'events': int, 'changed': int, 'expected_percent': float}} where setting is the register
    width W (uniform family, safer_is_larger=True) or the deficit d (per-node family, safer_is_larger=False).
    A threshold "from S upward" requires the condition at S and at every measured setting that is safer than S.
    Settings safer than the safest measured one are not measured; they are covered by monotonicity only for the
    event criterion (no saturation at W implies none at any W' > W, so the result equals the exact arm), which is
    why every 'from upward' value is reported only when the safest measured setting satisfies the criterion.
    Returns None for a criterion that no measured setting satisfies.
    """
    order = sorted(grid, reverse=safer_is_larger)  # safest first
    def narrowest(ok):
        best = None
        for s in order:
            if not ok(grid[s]):
                break
            best = s
        return best
    result = {
        'noevent': narrowest(lambda g: g['events'] == 0),
        'same': narrowest(lambda g: g['changed'] == 0),
        'acc_0.5': narrowest(lambda g: g['expected_percent'] >= wide_expected_percent - 0.5),
        'acc_1.0': narrowest(lambda g: g['expected_percent'] >= wide_expected_percent - 1.0),
    }
    half = [s for s in order if grid[s]['expected_percent'] <= wide_expected_percent / 2]
    result['half'] = half[0] if half else None  # safest (widest W / smallest d) setting at or below half
    return result
