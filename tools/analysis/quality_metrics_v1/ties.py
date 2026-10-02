"""Part C (classifier): expected-credit minus lowest-index top-1 and the spread of one random tie break."""
from __future__ import annotations

import numpy as np

FLOOR = 0.10


def unit_ties(unit):
    """Per unit: expected minus lowest (pp), SD of top-1 under one uniformly random tie break (pp)."""
    e, low = unit["credit_expected"], unit["credit_lowest"]
    n = len(e)
    tied_label = (e > 0) & (e < 1)
    var = (e[tied_label] * (1 - e[tied_label])).sum()
    return {"expected_minus_lowest_pp": 100 * float(e.mean() - low.mean()),
            "random_tie_sd_pp": 100 * float(np.sqrt(var)) / n,
            "label_tied_images": int(tied_label.sum()), "tied_top1_images": int((unit["tie_size"] > 1).sum()),
            "topk_minus_expected_pp": 100 * float(unit["credit_topk"].mean() - e.mean())}


def family_of(unit):
    if unit["source"] == "matrix":
        fam = unit["family"]
        return {"shared_exponent": "shared_exponent"}.get(fam, fam)
    return unit["source"]


def summarise(rows):
    """Median, quartiles and max of |expected - lowest| and the random-tie SD per family and bit width."""
    groups = {}
    for r in rows:
        if r["top1_expected"] < 100 * FLOOR:
            continue
        groups.setdefault((r["source"], r["family"], r["bits"]), []).append(r)
    out = []
    for (source, family, bits), rs in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        d = np.array([r["expected_minus_lowest_pp"] for r in rs])
        s = np.array([r["random_tie_sd_pp"] for r in rs])
        out.append({"source": source, "family": family, "bits": bits, "units": len(rs),
                    "diff_median": float(np.median(d)), "diff_q25": float(np.percentile(d, 25)),
                    "diff_q75": float(np.percentile(d, 75)), "diff_min": float(d.min()), "diff_max": float(d.max()),
                    "abs_diff_max": float(np.abs(d).max()), "random_sd_median": float(np.median(s)),
                    "random_sd_max": float(s.max())})
    return out
