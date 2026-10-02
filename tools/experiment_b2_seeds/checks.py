"""Lane Q5, review 1 follow-up (CPU, read-only on the v1 summaries): two checks of ``analysis.py`` v1.

1. Signs in exact arithmetic.  ``analysis.signs_consistent`` takes ``numpy.sign`` of a float mean, so a per-seed
   difference that is exactly zero can come out as +/-4e-17.  Here every per-seed paired difference is recomputed as a
   rational number (expected credit = 1/e for an image whose label is tied with e classes at the top, lowest-index
   credit 0 or 1), the protocol's zero rule is applied (an exact zero is not a sign, so it breaks consistency), and the
   sign columns of ``contrasts.csv`` are compared with the exact result.
2. Image-sampling SD as the protocol defines it (square root of the mean over the five seeds of the bootstrap
   variance) against the v1 value (mean over seeds of the bootstrap SD), with the change of the seed share.

``analysis.py`` itself is left unchanged: its file hash is recorded in ``results/summaries/b2-seeds-v1/summary.json``.
Writes one JSON file once (default ``artifacts/experiment_b2_seeds/checks/review1-checks-v1.json``).
Development evidence on the 1k screen.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from tools.experiment_b.common import ROOT, file_hash
from tools.experiment_b2 import readout
from . import analysis
from . import cells as lane

SUMMARY = analysis.OUT
DEFAULT_OUT = lane.BASE / "checks" / "review1-checks-v1.json"
SEEDS = tuple(analysis.SEEDS)


def exact_top1(arrays, rule):
    """Number of correct images (a Fraction) under a readout rule, from the integer readout arrays."""
    g = arrays["greater"].astype(np.int64)
    e = arrays["equal"].astype(np.int64)
    if rule == "expected":
        counts = Counter(e[g == 0].tolist())
        return sum((Fraction(n, k) for k, n in counts.items()), Fraction(0))
    if rule == "lowest_index":
        low = arrays["equal_lower"].astype(np.int64)
        return Fraction(int(((g == 0) & (low == 0)).sum()))
    raise ValueError(rule)


def exact_signs_consistent(values):
    """Protocol zero rule on exact values: consistent only if every value is non-zero and all share one sign."""
    signs = {(v > 0) - (v < 0) for v in values}
    return len(signs) == 1 and 0 not in signs


def parse(side):
    fmt, rec = side.split(":")
    return fmt, rec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"written once: {out}")
    lane_cells, _ = analysis.load()
    exact = {}
    for key, cell in lane_cells.items():
        if key[3] not in SEEDS:
            continue
        arrays = readout.load(ROOT / cell["record"]["readout_file"])
        exact[key] = {rule: exact_top1(arrays, rule) for rule in analysis.READOUTS}
        # the float readout agrees with the exact count
        for rule, name in analysis.READOUTS.items():
            if abs(float(exact[key][rule]) - float(cell["credit"][name].sum())) > 1e-9:
                raise ValueError(f"exact and float credit differ for {key} {rule}")
    boot = analysis.Bootstrap()

    sign_rows, zeros, mismatches = [], [], []
    image = {"rows": 0, "max_abs_image_sd_change_pp": 0.0, "max_abs_seed_share_change": 0.0, "worst": None}
    contrasts_csv = SUMMARY / "contrasts.csv"
    with contrasts_csv.open() as stream:
        contrast_rows = list(csv.DictReader(stream))
    for row in contrast_rows:
        model = row["model"]
        (lf, lr), (rf, rr) = parse(row["left"]), parse(row["right"])
        for label, name in analysis.READOUTS.items():
            diffs = [(exact[(model, lf, lr, s)][label] - exact[(model, rf, rr, s)][label]) * 100 / 1000 for s in SEEDS]
            for s, d in zip(SEEDS, diffs):
                if d == 0:
                    zeros.append({"model": model, "left": row["left"], "right": row["right"], "readout": label, "seed": s})
            c3, c5 = exact_signs_consistent(diffs[:3]), exact_signs_consistent(diffs)
            sd3 = float(row[f"{label}_sd_first3"])
            sd5 = float(row[f"{label}_sd"])
            n3 = (not c3) or sd3 > analysis.DELTA / 2
            n5 = (not c5) or sd5 > analysis.DELTA / 2
            got = {"signs_consistent_3": c3, "signs_consistent_5": c5, "needs_five_after_3": n3, "needs_five_after_5": n5}
            for column, value in got.items():
                if (row[f"{label}_{column}"] == "True") != value:
                    mismatches.append({"model": model, "left": row["left"], "right": row["right"], "readout": label,
                                       "column": column, "csv": row[f"{label}_{column}"], "exact": value})
            sign_rows.append(1)
            # image SD as the protocol defines it
            vectors = [lane_cells[(model, lf, lr, s)]["credit"][name] - lane_cells[(model, rf, rr, s)]["credit"][name]
                       for s in SEEDS]
            _image_compare(image, boot, vectors, row, label, model, row["left"] + " - " + row["right"])
    with (SUMMARY / "seed-cells.csv").open() as stream:
        cell_rows = list(csv.DictReader(stream))
    for row in cell_rows:
        model = row["model"]
        for label, name in analysis.READOUTS.items():
            vectors = [lane_cells[(model, row["format"], row["recipe"], s)]["credit"][name] for s in SEEDS]
            _image_compare(image, boot, vectors, row, label, model, f"{row['format']}:{row['recipe']}")

    counts = {}
    for label in analysis.READOUTS:
        counts[label] = {
            "signs_change_5_exact": sum(not exact_signs_consistent(
                [(exact[(r["model"], *parse(r["left"]), s)][label] - exact[(r["model"], *parse(r["right"]), s)][label])
                 for s in SEEDS]) for r in contrast_rows),
            "needs_five_after_3_csv": sum(r[f"{label}_needs_five_after_3"] == "True" for r in contrast_rows),
            "needs_five_after_5_csv": sum(r[f"{label}_needs_five_after_5"] == "True" for r in contrast_rows),
        }
    result = {
        "version": "b2-seeds-review1-checks-v1",
        "evidence_class": "development evidence (1k screen); check of the v1 summaries, nothing re-measured",
        "inputs": {"contrasts.csv": file_hash(contrasts_csv), "seed-cells.csv": file_hash(SUMMARY / "seed-cells.csv"),
                   "analysis.py": file_hash(ROOT / "tools/experiment_b2_seeds/analysis.py"),
                   "summary.json analysis.py": json.loads((SUMMARY / "summary.json").read_text())["sources"][
                       "tools/experiment_b2_seeds/analysis.py"]},
        "sign_check": {"contrast_readout_pairs": len(sign_rows), "exact_zero_differences": zeros,
                       "column_mismatches": mismatches, "counts": counts},
        "image_sd_check": image,
        "sources": {"tools/experiment_b2_seeds/checks.py": file_hash(Path(__file__))},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    print(json.dumps({"out": str(out), "zeros": len(zeros), "mismatches": len(mismatches), "counts": counts,
                      "image": {k: v for k, v in image.items() if k != "worst"}}))
    return 0


def _image_compare(image, boot, vectors, row, label, model, what):
    sds = np.array([boot.sd(v) for v in vectors])
    v1 = float(sds.mean())
    protocol = float(np.sqrt((sds ** 2).mean()))
    if abs(v1 - float(row[f"{label}_image_sd_n1000"])) > 5e-4:
        raise ValueError(f"v1 image SD not reproduced for {model} {what} {label}")
    seed_sd = float(row[f"{label}_sd"])
    share_v1 = seed_sd ** 2 / (seed_sd ** 2 + v1 ** 2)
    share_protocol = seed_sd ** 2 / (seed_sd ** 2 + protocol ** 2)
    image["rows"] += 1
    change = abs(protocol - v1)
    if change > image["max_abs_image_sd_change_pp"]:
        image["max_abs_image_sd_change_pp"] = change
        image["worst"] = {"model": model, "what": what, "readout": label, "v1": v1, "protocol": protocol}
    image["max_abs_seed_share_change"] = max(image["max_abs_seed_share_change"], abs(share_protocol - share_v1))


if __name__ == "__main__":
    raise SystemExit(main())
