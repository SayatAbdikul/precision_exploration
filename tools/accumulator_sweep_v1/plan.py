"""Location rule and 1k grid of protocol accumulator-sweep-protocol-v1 (pure functions, unit tested).

Settings are policy names of contract 2.1.  Widths are register widths W of `sat.w<W>`.
"""
LOCATION = (0, 128)
SCREEN = (0, 1000)
LADDER_STEP = 3
LADDER_MAX_RUNGS = 12
HARDWARE_WIDTHS = (16, 20, 24, 28, 32)
STRUCT_D_PRIMARY = (4, 8)
STRUCT_D_SECONDARY = (2, 12)
FP16_SENSITIVITY = (-3, 3)
BOTTOM_MARGIN = 2
MIN_WIDTH = 2
MAX_WIDTH = 63  # contract 2.1: saturating registers of 2 to 63 bits (addendum 1)


def ladder(w_cert_abs):
    """Descending location rungs: W_cert_abs - 2, then every LADDER_STEP bits, at most LADDER_MAX_RUNGS."""
    rungs, w = [], w_cert_abs - 2
    while w >= MIN_WIDTH and len(rungs) < LADDER_MAX_RUNGS:
        if w <= MAX_WIDTH:  # inadmissible rungs above 63 bits are skipped and not counted (addendum 1)
            rungs.append(w)
        w -= LADDER_STEP
    return rungs


def ladder_done(results, wide_expected):
    """True once a measured rung has expected Top-1 at or below half of the exact arm (128 images)."""
    return any(r['expected_percent'] <= wide_expected / 2 for r in results.values())


def refine(results, wide_expected, w_cert_abs):
    """Location refinement after the ladder: the widths between rungs next to the two transitions.

    Below the narrowest event-free rung W0 (when the next rung has an event): W0-1 .. W0-STEP+1; above the widest
    rung Wh at or below half (when the rung above is not): Wh+1 .. Wh+STEP-1.  Returns widths not yet measured.
    """
    order = sorted(results, reverse=True)
    add = set()
    clean = None
    for w in order:
        if results[w]['events']:
            break
        clean = w
    if clean is not None and clean != order[-1]:
        add.update(range(clean - LADDER_STEP + 1, clean))
    elif clean is None and order and order[0] < min(w_cert_abs - 1, MAX_WIDTH):
        add.update(range(order[0] + 1, min(w_cert_abs, MAX_WIDTH + 1)))
    half = [w for w in order if results[w]['expected_percent'] <= wide_expected / 2]
    if half and half[0] != order[0]:
        add.update(range(half[0] + 1, half[0] + LADDER_STEP))
    return sorted((w for w in add if w not in results and MIN_WIDTH <= w < w_cert_abs), reverse=True)


def clean_top_missing(top):
    return top is None


def bracket(results, wide_expected, w_cert_abs):
    """1k bracket [bottom, top] from the location rungs (results: {W: {'events', 'expected_percent'}}).

    top: the narrowest rung W such that no rung >= W has an event on the 128 images; if the widest rung already
    has an event, W_cert_abs - 1 (the certified width itself needs no run).
    bottom: (widest rung with expected Top-1 <= half of the exact arm) - BOTTOM_MARGIN; if no rung reached half,
    (narrowest rung) - BOTTOM_MARGIN.  Both clipped to [MIN_WIDTH, W_cert_abs - 1].
    """
    order = sorted(results, reverse=True)
    top = None
    for w in order:
        if results[w]['events']:
            break
        top = w
    if top is None:
        top = w_cert_abs - 1
    half = [w for w in order if results[w]['expected_percent'] <= wide_expected / 2]
    anchor = half[0] if half else order[-1]
    bottom = max(MIN_WIDTH, anchor - BOTTOM_MARGIN)
    top = min(top, w_cert_abs - 1, MAX_WIDTH)
    if bottom > top:
        bottom = top
    return bottom, top


def screen_grid(case, w_cert_abs, bottom, top, integer, fp16_policy, f21_policy,
                struct_primary=STRUCT_D_PRIMARY, struct_secondary=STRUCT_D_SECONDARY):
    """The 1k runs of one case, each tagged with its priority (lower runs first)."""
    runs = [(1, 'wide')] + [(1, f'sat.w{w}') for w in range(top, bottom - 1, -1)]
    runs += [(2, 'control'), (2, fp16_policy), (2, f21_policy)]
    if integer:
        runs += [(3, f'sat.w{w}') for w in HARDWARE_WIDTHS if w < bottom and w <= w_cert_abs]
    runs += [(4, f'sat.struct-{d}') for d in struct_primary]
    runs += [(6, f'sat.struct-{d}') for d in struct_secondary]
    return [{'case': case, 'policy': p, 'priority': pr} for pr, p in runs]


def hardware_derived(w_cert_abs, top_noevent_1k):
    """Hardware widths not run because they lie above a width with no event on the 1000 images (equal to the
    exact arm by construction) or at or above the certified width."""
    return [w for w in HARDWARE_WIDTHS if top_noevent_1k is not None and w > top_noevent_1k or w >= w_cert_abs]


def extend_top(top, events_at_top, w_cert_abs):
    """Upward extension: while the top width has an event on the 1k images, add top + 1 (below W_cert_abs)."""
    return top + 1 if events_at_top and top + 1 < w_cert_abs and top + 1 <= MAX_WIDTH else None


def extend_bottom(results_1k, wide_expected):
    """Downward extension: the bracket must contain at least BOTTOM_MARGIN widths below W_half (1k).

    results_1k: {W: {'expected_percent'}} for the contiguous 1k bracket.  Returns the next width to add or None.
    """
    order = sorted(results_1k, reverse=True)
    bottom = order[-1]
    if bottom <= MIN_WIDTH:
        return None
    half = [w for w in order if results_1k[w]['expected_percent'] <= wide_expected / 2]
    if not half:
        return bottom - 1
    return bottom - 1 if half[0] - bottom < BOTTOM_MARGIN else None
