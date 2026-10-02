import numpy as np

from tools.experiment_b2_recon.collapse_ties import SHARES, bounds_from_top5, exact_shares, status, verdict

CLASSES = 30


def _random_mask(rng, images, max_tie):
    mask = np.zeros((images, CLASSES), dtype=bool)
    for i in range(images):
        k = int(rng.integers(1, max_tie + 1))
        mask[i, rng.choice(CLASSES, size=k, replace=False)] = True
    return mask


def _top5_from_mask(rng, mask):
    """A top-5 list as topk may return it: maxima first, in an arbitrary order, then other classes."""
    top5 = np.zeros((mask.shape[0], 5), dtype=np.int64)
    for i, row in enumerate(mask):
        maxima = rng.permutation(np.flatnonzero(row))
        others = rng.permutation(np.flatnonzero(~row))
        top5[i] = np.concatenate([maxima, others])[:5]
    return top5


def test_bounds_are_exact_when_every_tie_is_listed():
    rng = np.random.default_rng(0)
    mask = _random_mask(rng, 200, 5)
    got = bounds_from_top5(_top5_from_mask(rng, mask), mask.sum(axis=1), classes=CLASSES)
    want = exact_shares(mask)
    for share in SHARES:
        assert np.allclose(got[share][0], got[share][1])
        assert np.allclose(got[share][0], want[share][0])


def test_bounds_contain_the_exact_shares_for_large_ties():
    rng = np.random.default_rng(1)
    for _ in range(5):
        mask = _random_mask(rng, 150, 12)
        got = bounds_from_top5(_top5_from_mask(rng, mask), mask.sum(axis=1), classes=CLASSES)
        want = exact_shares(mask)
        for share in SHARES:
            lower, upper = got[share]
            assert (lower <= want[share][0] + 1e-12).all() and (want[share][0] <= upper + 1e-12).all()


def test_tie_aware_collapse_where_topk_first_index_misses_it():
    # class 7 is a maximum on every image but topk lists it first on only a third of them
    mask = np.zeros((300, CLASSES), dtype=bool)
    mask[:, 7] = True
    mask[np.arange(300), 10 + np.arange(300) % 3] = True
    top5 = np.array([[7, 10 + i % 3, 0, 1, 2] if i % 3 == 0 else [10 + i % 3, 7, 0, 1, 2] for i in range(300)])
    shares = bounds_from_top5(top5, mask.sum(axis=1), classes=CLASSES)
    assert status(*shares["among_maxima"]) == "collapsed"
    assert np.isclose(shares["expected"][0][7], 0.5)
    assert status(*shares["expected"]) == "not_collapsed"
    assert status(*shares["lowest_index"]) == "collapsed"  # 7 < 10, 11, 12
    assert (top5[:, 0] == 7).mean() < 0.9


def test_status_undetermined_and_verdicts():
    lower, upper = np.array([0.85, 0.0]), np.array([0.95, 0.1])
    assert status(lower, upper) == "undetermined"
    assert verdict("collapsed", "not_collapsed") == "supported"
    assert verdict("not_collapsed", "collapsed") == "not_supported"
    assert verdict("undetermined", "not_collapsed") == "undecided"
    assert verdict("collapsed", "collapsed") == "undecided"
    assert verdict(None, "collapsed") is None
