"""Lane S1 part 3: fast_cocoeval equals pycocotools' accumulate (through stats.Evaluated.metrics) on synthetic rows
with many score ties, ignored detections and crowd ground truth, for the full set and for bootstrap draws."""
import numpy as np
from pycocotools.cocoeval import Params
from tools.analysis.fast_cocoeval import FastAccumulate
from tools.experiment_b2_det.stats import Evaluated


def synthetic(seed, n=30, K=4):
    rng = np.random.default_rng(seed)
    p = Params(iouType='bbox'); p.catIds = list(range(1, K + 1)); p.imgIds = list(range(n))
    A = len(p.areaRng); T = len(p.iouThrs)
    rows = [None] * (K * A * n)
    for k in range(K):
        for i in range(n):
            if rng.random() < 0.15:
                continue
            G = int(rng.integers(0, 4)); D = int(rng.integers(0, 6))
            scores = np.round(rng.random(D), 1)                     # coarse scores: many ties
            match = (rng.random((T, D)) < np.linspace(0.8, 0.2, T)[:, None]).astype(float)
            ignore = rng.random((T, D)) < 0.1
            rows[k * A * n + i] = {'dtScores': list(scores), 'dtMatches': match, 'dtIgnore': ignore,
                                   'gtIgnore': (rng.random(G) < 0.2).astype(int)}
    ev = object.__new__(Evaluated)
    ev.cache = type('Cache', (), {})(); ev.cache.parameters = p; ev.cache.rows = rows
    ev.images, ev.categories, ev.areas = n, K, A
    return ev


def test_point_and_draws_equal_pycocotools():
    for seed in range(4):
        ev = synthetic(seed)
        fast = FastAccumulate(ev)
        assert np.array_equal(fast.metrics(), ev.metrics())
        rng = np.random.default_rng(seed)
        for _ in range(10):
            draw = (rng.integers(0, ev.images, size=ev.images) + 1).tolist()
            assert np.array_equal(fast.order(draw)[0], fast.order_by_key(draw))
            assert np.array_equal(fast.metrics(draw), ev.metrics(draw))
