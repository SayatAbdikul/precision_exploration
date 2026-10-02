"""Vectorised COCOeval.accumulate for the detector bootstrap (lane S1, protocol speed-protocol-v1 part 3).

Input: a `tools.experiment_b2_det.stats.Evaluated` (its per-image evaluation rows, area 'all', maxDets 100, computed
once by pycocotools' COCOeval.evaluate). Output: [mAP50-95, mAP50] exactly as `Evaluated.metrics(draws)`.

Why this is the same computation. accumulate concatenates, per category, the rows of the drawn images in draw
order, sorts the detections by score with a stable sort (ties: draw position, then the order within the image),
and per IoU threshold reads the interpolated precision at 101 recall thresholds. Ignored detections change
neither the true- nor the false-positive count, so they only repeat the previous precision value and can be
dropped. At the m-th true positive (F false positives before it) the precision is m / ((F + m) + eps) with the
same binary64 operations; the interpolated precision at a recall threshold r is the maximum of these values over
the true positives from the m_r-th on, m_r = min{m : fl(m / npig) >= r} (0 at r = 0, where the maximum runs over
every true positive), and 0 when fewer than m_r true positives exist. A category with npig = 0 keeps -1. The
(T, R, K) array is then reduced with the same masked numpy mean as stats.Evaluated.metrics.
"""
from __future__ import annotations
import numpy as np

EPS = np.spacing(1)


class FastAccumulate:
    def __init__(self, evaluated):
        cache = evaluated.cache; p = cache.parameters
        n, K, A = evaluated.images, evaluated.categories, evaluated.areas
        self.n, self.K, self.T = n, K, len(p.iouThrs)
        self.thresholds = np.asarray(p.recThrs, dtype=np.float64)
        max_det = p.maxDets[-1]
        rows = cache.rows
        self.G = np.zeros((K, n), dtype=np.int64)
        cat, img, within, score, tp, fp = [], [], [], [], [], []
        for k in range(K):
            for i in range(n):
                e = rows[k * A * n + i]
                if e is None:
                    continue
                self.G[k, i] = np.count_nonzero(np.asarray(e['gtIgnore']) == 0)
                s = np.asarray(e['dtScores'][0:max_det], dtype=np.float64)
                if not len(s):
                    continue
                dtm = np.asarray(e['dtMatches'])[:, 0:max_det]; dtig = np.asarray(e['dtIgnore'])[:, 0:max_det]
                t_ = np.logical_and(dtm, np.logical_not(dtig)); f_ = np.logical_and(np.logical_not(dtm), np.logical_not(dtig))
                keep = (t_ | f_).any(0)          # ignored at every threshold: no effect on any count
                if not keep.any():
                    continue
                d = np.flatnonzero(keep)
                cat.append(np.full(len(d), k)); img.append(np.full(len(d), i)); within.append(d)
                score.append(s[d]); tp.append(t_[:, d]); fp.append(f_[:, d])
        self.cat = np.concatenate(cat) if cat else np.zeros(0, np.int64)
        self.img = np.concatenate(img) if img else np.zeros(0, np.int64)
        self.within = np.concatenate(within) if within else np.zeros(0, np.int64)
        scores = np.concatenate(score) if score else np.zeros(0)
        self.tp = np.concatenate(tp, axis=1) if tp else np.zeros((self.T, 0), bool)
        self.fp = np.concatenate(fp, axis=1) if fp else np.zeros((self.T, 0), bool)
        # Rank of -score (equal binary64 scores share a rank, as in the stable sort of -dtScores).
        _, rank = np.unique(-scores, return_inverse=True)
        self.base_key = self.cat.astype(np.int64) * (int(rank.max()) + 1 if len(rank) else 1) + rank
        self.width = int(max_det) + 1
        if len(self.base_key) and (int(self.base_key.max()) + 1) * n * self.width >= 2**62:
            raise ValueError('sort key would overflow int64')
        # Dense tie-group id (category, score rank), ascending = category ascending, score descending.
        _, group = np.unique(self.base_key, return_inverse=True)
        self.groups = int(group.max()) + 1 if len(group) else 0
        self.group = group.astype(np.uint16 if self.groups <= 1 << 16 else np.int64)
        # Detections of each image, by (category, within-image index): concatenated in draw order they form the
        # sequence in (draw position, within) order; a stable sort by tie group then gives (group, j, within).
        by_image = np.lexsort((self.within, self.cat, self.img))
        self.image_dets = by_image
        counts = np.bincount(self.img, minlength=n) if len(self.img) else np.zeros(n, np.int64)
        self.image_count = counts
        self.image_start = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        if self.T > 16:
            raise ValueError('more than 16 IoU thresholds')
        weights = (1 << np.arange(self.T, dtype=np.uint16))[:, None]
        self.tp_bits = (self.tp.astype(np.uint16) * weights).sum(0).astype(np.uint16)
        self.fp_bits = (self.fp.astype(np.uint16) * weights).sum(0).astype(np.uint16)

    def order(self, draws=None):
        """Detection instances of a draw in accumulate's order: (category, -score, draw position, within)."""
        n = self.n
        d = np.arange(n) if draws is None else np.asarray(draws, dtype=np.int64) - 1
        length = self.image_count[d]; total = int(length.sum())
        offset = np.repeat(self.image_start[d] - (np.cumsum(length) - length), length)
        sequence = self.image_dets[offset + np.arange(total)]
        return sequence[np.argsort(self.group[sequence], kind='stable')], np.bincount(d, minlength=n)

    def order_by_key(self, draws=None):
        """The same order from one composite integer key (first implementation; kept for the unit test)."""
        n = self.n
        d = np.arange(n) if draws is None else np.asarray(draws, dtype=np.int64) - 1
        c = np.bincount(d, minlength=n)
        by_image = np.argsort(d, kind='stable')
        starts = np.concatenate([[0], np.cumsum(c)[:-1]])
        rep = c[self.img]; total = int(rep.sum())
        p_inst = np.repeat(np.arange(len(rep)), rep)
        offs = np.arange(total) - np.repeat(np.cumsum(rep) - rep, rep)
        j_inst = by_image[np.repeat(starts[self.img], rep) + offs]
        key = (self.base_key[p_inst] * n + j_inst) * self.width + self.within[p_inst]
        return p_inst[np.argsort(key)]

    def metrics(self, draws=None):
        """[mAP50-95, mAP50] for 1-based image positions `draws` (default: every image once, in order)."""
        K, T = self.K, self.T
        order, c = self.order(draws)
        npig = self.G @ c
        cats = self.cat[order]
        tpb = self.tp_bits[order]; fpb = self.fp_bits[order]
        R = len(self.thresholds)
        precision = np.full((T, R, K), -1.0)
        valid = npig > 0
        # m_r per category: smallest m with fl(m / npig) >= threshold.
        safe = np.where(valid, npig, 1).astype(np.float64)
        m = np.ceil(self.thresholds[None, :] * safe[:, None]).astype(np.int64)
        for _ in range(3):
            m = np.where((m > 0) & ((m - 1) / safe[:, None] >= self.thresholds[None, :]), m - 1, m)
            m = np.where(m / safe[:, None] < self.thresholds[None, :], m + 1, m)
        need = np.maximum(m, 1)
        first = np.searchsorted(cats, np.arange(K), side='left')           # segment start per category
        for t in range(T):
            bit = np.uint16(1 << t)
            hit = np.flatnonzero(tpb & bit)
            q = np.zeros((K, R))
            if len(hit):
                false = np.flatnonzero(fpb & bit)
                hc = cats[hit]
                count = np.bincount(hc, minlength=K)
                tp_first = np.concatenate([[0], np.cumsum(count)[:-1]])
                # m-th TP of its category (1-based) and the FPs of its category before it.
                mm = (np.arange(len(hit)) - tp_first[hc] + 1).astype(np.float64)
                ff = (np.searchsorted(false, hit) - np.searchsorted(false, first)[hc]).astype(np.float64)
                prec = mm / (ff + mm + EPS)
                z = (K - hc).astype(np.float64) + 1j * prec
                best = np.maximum.accumulate(z[::-1])[::-1].imag              # suffix maximum within the category
                ok = (need <= count[:, None]) & (count[:, None] > 0)
                index = np.where(ok, tp_first[:, None] + need - 1, 0)
                q = np.where(ok, best[index], 0.0)
            precision[t] = np.where(valid[None, :], q.T, -1.0)
        if not (precision > -1).any():
            raise ValueError('COCO population has no evaluable ground truth')
        return np.array([precision[precision > -1].mean(), precision[0][precision[0] > -1].mean()])

    def bootstrap(self, resamples, seed, confidence=0.95):
        """Per-draw metrics with the draw sequence of stats.Evaluated.bootstrap (same generator, same calls)."""
        from public.analysis.phase3.statistics import settings
        rng = settings(confidence, resamples, seed)
        out = np.empty((resamples, 2))
        for index in range(resamples):
            out[index] = self.metrics((rng.integers(0, self.n, size=self.n) + 1).tolist())
        return out
