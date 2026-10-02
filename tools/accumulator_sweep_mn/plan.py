"""Location rules and 1k grid of protocol accumulator-sweep-mn-protocol-v1 (pure functions, unit tested).

Settings are policy names of contract 2.1/2.2.  Uniform family: register width W of `sat.w<W>`.  Per-node family:
deficit d of `sat.struct-<d>` (each MAC node gets its own certified structural width minus d).
Adapted from lane L8's tools/accumulator_sweep_v1/plan.py: the ladder starts at the certified width and steps by 2
(not 3), the refinement (L8 addendum 3) is part of the frozen rule, and a per-node ladder fixes the 1k d set.
"""
LOCATION = (0, 128)
SCREEN = (0, 1000)
LADDER_STEP = 2
LADDER_MAX_RUNGS = 12
BOTTOM_MARGIN = 2
MIN_WIDTH = 2
MAX_WIDTH = 63  # contract 2.1: saturating registers of 2 to 63 bits
STRUCT_LADDER = (1, 2, 3, 4, 5, 6, 7, 8)
STRUCT_BELOW = 2   # 1k d set: d_half(128) - STRUCT_BELOW ... d_half(128) + STRUCT_ABOVE, clipped to >= 1
STRUCT_ABOVE = 1
FP16_SENSITIVITY = (-3, -2, -1, 1, 2, 3)
SENSITIVITY_CASES = ('mobilenet_v2-int8-default-b2', 'mobilenet_v3_large-int8-default-b2',
                     'mobilenet_v3_large-fp7_e3m3-default-b2')


def ladder(w_cert_abs):
    """Descending location rungs: W_cert_abs - 1, then every LADDER_STEP bits, at most LADDER_MAX_RUNGS rungs.
    (sat.w<W_cert_abs> is lossless by the certificate and is not run.)"""
    rungs, w = [], w_cert_abs - 1
    while w >= MIN_WIDTH and len(rungs) < LADDER_MAX_RUNGS:
        if w <= MAX_WIDTH:
            rungs.append(w)
        w -= LADDER_STEP
    return rungs


def at_or_below_half(result, wide_expected):
    return result['expected_percent'] <= wide_expected / 2


def ladder_done(results, wide_expected):
    """True once a measured rung has expected Top-1 at or below half of the exact arm (128 images)."""
    return any(at_or_below_half(r, wide_expected) for r in results.values())


def refine(results, wide_expected, w_cert_abs, step=LADDER_STEP):
    """Widths still to measure next to the two transitions (evaluated after every finished batch of sat.w runs).

    (a) W0 = the narrowest measured width such that no measured width >= W0 has an event: if a narrower width was
        measured, ask for W0-1 .. W0-step+1; if every measured width has an event, ask for every width from the
        widest measured + 1 to W_cert_abs - 1.
    (b) Wh = the widest measured width at or below half of the exact arm: if a wider width was measured, ask for
        Wh+1 .. Wh+step-1.
    Widths already measured, below MIN_WIDTH or at/above W_cert_abs (lossless) are dropped.  Widest first.
    """
    order = sorted(results, reverse=True)
    add = set()
    clean = None
    for w in order:
        if results[w]['events']:
            break
        clean = w
    if clean is not None and clean != order[-1]:
        add.update(range(clean - step + 1, clean))
    elif clean is None and order and order[0] < min(w_cert_abs - 1, MAX_WIDTH):
        add.update(range(order[0] + 1, min(w_cert_abs, MAX_WIDTH + 1)))
    half = [w for w in order if at_or_below_half(results[w], wide_expected)]
    if half and half[0] != order[0]:
        add.update(range(half[0] + 1, half[0] + step))
    return sorted((w for w in add if w not in results and MIN_WIDTH <= w < w_cert_abs and w <= MAX_WIDTH), reverse=True)


def bracket(results, wide_expected, w_cert_abs):
    """1k bracket [bottom, top] from the 128-image sat.w results ({W: {'events', 'expected_percent'}}).

    top: the narrowest measured W such that no measured width >= W has an event (W_cert_abs - 1 if the widest one
    has an event).  bottom: (widest measured W at or below half of the exact arm) - BOTTOM_MARGIN, or the narrowest
    measured W - BOTTOM_MARGIN if none reached half.  Clipped to [MIN_WIDTH, W_cert_abs - 1].
    """
    order = sorted(results, reverse=True)
    top = None
    for w in order:
        if results[w]['events']:
            break
        top = w
    if top is None:
        top = w_cert_abs - 1
    half = [w for w in order if at_or_below_half(results[w], wide_expected)]
    anchor = half[0] if half else order[-1]
    bottom = max(MIN_WIDTH, anchor - BOTTOM_MARGIN)
    top = min(top, w_cert_abs - 1, MAX_WIDTH)
    return min(bottom, top), top


def struct_next(results, wide_expected):
    """Per-node location ladder (128 images): the next deficit d to measure, ascending through STRUCT_LADDER, or
    None once a measured d is at or below half of the exact arm or the ladder is exhausted."""
    if any(at_or_below_half(r, wide_expected) for r in results.values()):
        return None
    for d in STRUCT_LADDER:
        if d not in results:
            return d
    return None


def struct_set(results, wide_expected):
    """1k d set from the per-node ladder: d_half - STRUCT_BELOW .. d_half + STRUCT_ABOVE (clipped to >= 1), where
    d_half is the smallest measured d at or below half; if no d reached half, the STRUCT_BELOW + 1 largest measured
    values."""
    half = sorted(d for d, r in results.items() if at_or_below_half(r, wide_expected))
    if half:
        dh = half[0]
        return list(range(max(1, dh - STRUCT_BELOW), dh + STRUCT_ABOVE + 1))
    measured = sorted(results)
    return measured[-(STRUCT_BELOW + 1):]


def stress_widths(bottom, top):
    """Reduced uniform grid of a stress case: the bracket top and every second width down to the bottom (the bottom
    is always included)."""
    widths = list(range(top, bottom - 1, -2))
    if widths[-1] != bottom:
        widths.append(bottom)
    return widths


def screen_grid(case, bottom, top, fp16_policy, f21_policy, struct_ds, stress=False):
    """The 1k runs of one case, each tagged with its priority (lower runs first).

    1: wide, control (all cases, before the location phase); 2: sat.w bracket (stress: reduced);
    3: the rule fp16 and f21 policies; 4: sat.struct-<d> for the location d set (not for a stress case).
    """
    runs = [(1, 'wide'), (1, 'control')]
    widths = stress_widths(bottom, top) if stress else list(range(top, bottom - 1, -1))
    runs += [(2, f'sat.w{w}') for w in widths]
    runs += [(3, fp16_policy), (3, f21_policy)]
    if not stress:
        runs += [(4, f'sat.struct-{d}') for d in struct_ds]
    return [{'case': case, 'policy': p, 'priority': pr} for pr, p in runs]


def extend_top(top, events_at_top, w_cert_abs):
    """Upward extension: while the widest bracket width has an event on the 1k images, add the next wider width
    (below W_cert_abs)."""
    return top + 1 if events_at_top and top + 1 < w_cert_abs and top + 1 <= MAX_WIDTH else None


def extend_bottom(results_1k, wide_expected):
    """Downward extension: while fewer than BOTTOM_MARGIN bracket widths lie below W_half (1k), or no width reached
    half, add the next narrower width (down to MIN_WIDTH)."""
    order = sorted(results_1k, reverse=True)
    bottom = order[-1]
    if bottom <= MIN_WIDTH:
        return None
    half = [w for w in order if at_or_below_half(results_1k[w], wide_expected)]
    if not half:
        return bottom - 1
    return bottom - 1 if half[0] - bottom < BOTTOM_MARGIN else None


def sensitivity_policies(fp16_exponent, ub):
    """Exponent-sensitivity runs of the location phase (SENSITIVITY_CASES): fp16.x<e+k> for k in FP16_SENSITIVITY and
    f21 with the fp16 rule's binary point, f21.x<15-ub> (expected identical to plain f21: no overflow, no subnormal)."""
    out = []
    for k in FP16_SENSITIVITY:
        e = fp16_exponent + k
        out.append('fp16' if e == 0 else f'fp16.x{e}')
    if 15 - ub != 0:
        out.append(f'f21.x{15 - ub}')
    return out
