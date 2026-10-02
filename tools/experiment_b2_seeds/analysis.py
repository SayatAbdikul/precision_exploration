"""Lane Q5 analysis (CPU): seed spread, contrasts, variance decomposition, calibration size, matrix outcomes.

Reads the lane's cell records (artifacts/experiment_b2_seeds/cells) and, read-only, the full-set matrix cells
(artifacts/experiment_b2/matrix/cells, through ``tools.analysis.b2_matrix.load_cells``).  Writes the summary set
once (results/summaries/b2-seeds-v1/); an existing file is never replaced.  Rules: protocol
public/experiments/configs/breadth-study/b2-seeds-protocol-v1.json, ``analysis_rules_stated_in_advance``.
Every number is development evidence on the 1k screen.
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
from pathlib import Path

import numpy as np

from tools.analysis import b2_matrix
from tools.analysis.b2_ties import RESAMPLES, SEED
from tools.experiment_b.common import ROOT, file_hash, formats, unseal
from tools.experiment_b2 import readout
from . import cells as lane
from . import subsets

OUT = ROOT / "results/summaries/b2-seeds-v1"
MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large")
DELTA = 1.0
SEEDS = lane.SEEDS
READOUTS = {"expected": "top1_expected", "lowest_index": "top1_lowest_index"}
EIGHT = ("int8", "posit8_es1", "fp8_e4m3fn", "log8", "mxfp8_e4m3")
SIX = ("int6", "fp6_e2m3", "fp6_e3m2", "bfp6", "log6", "posit6_es1")
SIX_SCALAR = ("int6", "fp6_e2m3", "fp6_e3m2", "log6", "posit6_es1")
PROPOSAL = (("int8", "default"), ("posit8_es1", "default"), ("fp8_e4m3fn", "default"), ("log8", "default"),
            ("mxfp8_e4m3", "cum5_act_maxabs"), ("fp7_e3m3", "default"), ("fp6_e2m3", "default"),
            ("bfp6", "cum5_act_maxabs"), ("int6", "default"), ("log6", "default"), ("posit6_es1", "default"),
            ("nf4", "minimal"))
SIZES = {"L32": 32, "L128": 128, "S": 400, "H": 1000}


def recipe(name):
    return lane.recipe_for(name, "default")


def contrast_list():
    """``(group, (left format, left recipe), (right format, right recipe))`` per model, fixed in the protocol."""
    out = [("8_bit", (a, recipe(a)), (b, recipe(b))) for a, b in itertools.combinations(EIGHT, 2)]
    out += [("7_vs_8", ("fp7_e3m3", "default"), ("int8", "default")),
            ("7_vs_8", ("fp7_e3m3", "default"), ("fp8_e4m3fn", "default"))]
    out += [("6_bit", (a, recipe(a)), (b, recipe(b))) for a, b in itertools.combinations(SIX, 2)]
    out += [("recipe_rule_c", (f, "default"), (f, "minimal")) for f in SIX_SCALAR]
    return out


class Bootstrap:
    """The project's paired image bootstrap (10000 resamples, seed 20260927) as one resampling matrix."""

    def __init__(self, n=1000):
        index = np.random.default_rng(SEED).integers(0, n, size=(RESAMPLES, n))
        self.weights = np.zeros((RESAMPLES, n), dtype=np.float64)
        np.add.at(self.weights, (np.arange(RESAMPLES)[:, None], index), 1.0)
        self.n = n

    def draws(self, vector):
        return 100 * (self.weights @ np.asarray(vector, dtype=np.float64)) / self.n

    def sd(self, vector):
        return float(self.draws(vector).std(ddof=1))


def load():
    """Lane cells keyed by (model, format, recipe, subset) and full-set matrix cells keyed as in b2_matrix."""
    lane_cells = {}
    for path in sorted((lane.BASE / "cells").glob("*.json")):
        record = unseal(path)
        key = (record["model"], record["format"], record["recipe_name"], record["subset"]["name"])
        if key in lane_cells:
            raise ValueError(f"duplicate lane cell {key}")
        file = ROOT / record["readout_file"]
        if file_hash(file) != record["readout_file_sha256"]:
            raise ValueError(f"readout drift {file}")
        arrays = readout.load(file)
        if len(arrays["label"]) != 1000 or record["inference_batch_size"] != 8:
            raise ValueError(f"not a batch-8 1k cell: {path}")
        lane_cells[key] = {"record": record, "credit": readout.credits(arrays), "file": str(path.relative_to(ROOT)),
                           "record_sha256": file_hash(path), "labels": arrays["label"]}
    full = b2_matrix.load_cells(MODELS)
    for (model, *_), cell in lane_cells.items():
        reference = full[(model, "fp32", "baseline")]
        if not np.array_equal(cell["labels"], readout.load(ROOT / reference["record"]["readout_file"])["label"]):
            raise ValueError("lane cell is not on the matrix screen order")
    return lane_cells, full


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"n": int(values.size), "mean": float(values.mean()), "sd": float(values.std(ddof=1)) if values.size > 1 else None,
            "min": float(values.min()), "max": float(values.max())}


def signs_consistent(values):
    signs = {int(np.sign(v)) for v in values}
    return len(signs) == 1 and 0 not in signs


def needs_five(values):
    s = stats(values)
    return (not signs_consistent(values)) or (s["sd"] is not None and s["sd"] > DELTA / 2)


def r(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    return value


def csv_text(rows):
    if not rows:
        return ""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: r(v) for k, v in row.items()})
    return stream.getvalue()


def analyse(lane_cells, full):
    boot = Bootstrap()
    tables, notes = {}, {}

    def credit(model, fmt, rec, subset, rule="top1_expected"):
        if subset == "FULL":
            return full[(model, fmt, rec)]["credit"][rule]
        return lane_cells[(model, fmt, rec, subset)]["credit"][rule]

    def have(model, fmt, rec, subsets_):
        return all((model, fmt, rec, s) in lane_cells for s in subsets_)

    # 1. every lane cell
    tables["cells"] = [{"model": m, "format": f, "recipe": rc, "subset": s, "kind": c["record"]["subset"]["kind"],
                        "range_images": c["record"]["subset"]["range_images"],
                        "bias_images": c["record"]["subset"]["bias_images"],
                        "top1_expected": 100 * c["credit"]["top1_expected"].mean(),
                        "top1_lowest_index": 100 * c["credit"]["top1_lowest_index"].mean(),
                        "tied_top1_images": c["record"]["readout"]["images_with_tied_top1"],
                        "configuration_sha256": c["record"]["configuration_sha256"], "record": c["file"]}
                       for (m, f, rc, s), c in sorted(lane_cells.items())]

    # 2. per-cell seed spread and variance decomposition
    seed_rows = []
    for model in MODELS:
        for fmt, rec in sorted({(f, rc) for (m, f, rc, s) in lane_cells if m == model and s in SEEDS}):
            if not have(model, fmt, rec, SEEDS):
                continue
            row = {"model": model, "format": fmt, "recipe": rec}
            for label, rule in READOUTS.items():
                values = [100 * credit(model, fmt, rec, s, rule).mean() for s in SEEDS]
                st = stats(values)
                full_value = 100 * credit(model, fmt, rec, "FULL", rule).mean()
                image_sd = float(np.mean([boot.sd(credit(model, fmt, rec, s, rule)) for s in SEEDS]))
                row.update({f"{label}_mean": st["mean"], f"{label}_sd": st["sd"], f"{label}_min": st["min"],
                            f"{label}_max": st["max"], f"{label}_range": st["max"] - st["min"],
                            f"{label}_full_2000": full_value, f"{label}_mean_minus_full": st["mean"] - full_value,
                            f"{label}_sd_exceeds_half_delta": st["sd"] > DELTA / 2,
                            f"{label}_image_sd_n1000": image_sd,
                            f"{label}_seed_share": st["sd"] ** 2 / (st["sd"] ** 2 + image_sd ** 2)})
                for s, v in zip(SEEDS, values):
                    row[f"{label}_{s}"] = v
            seed_rows.append(row)
    tables["seed-cells"] = seed_rows

    # 3. contrasts
    contrast_rows = []
    for model in MODELS:
        for group, (lf, lr), (rf, rr) in contrast_list():
            if not (have(model, lf, lr, SEEDS) and have(model, rf, rr, SEEDS)):
                continue
            row = {"model": model, "group": group, "left": f"{lf}:{lr}", "right": f"{rf}:{rr}"}
            for label, rule in READOUTS.items():
                diffs = [100 * (credit(model, lf, lr, s, rule) - credit(model, rf, rr, s, rule)).mean() for s in SEEDS]
                st = stats(diffs)
                fd, flow, fhigh = b2_matrix.interval(credit(model, lf, lr, "FULL", rule), credit(model, rf, rr, "FULL", rule))
                image_sd = float(np.mean([boot.sd(credit(model, lf, lr, s, rule) - credit(model, rf, rr, s, rule))
                                          for s in SEEDS]))
                row.update({f"{label}_mean": st["mean"], f"{label}_sd": st["sd"], f"{label}_min": st["min"],
                            f"{label}_max": st["max"],
                            f"{label}_signs_consistent_3": signs_consistent(diffs[:3]),
                            f"{label}_signs_consistent_5": signs_consistent(diffs),
                            f"{label}_needs_five_after_3": needs_five(diffs[:3]),
                            f"{label}_needs_five_after_5": needs_five(diffs),
                            f"{label}_sd_first3": stats(diffs[:3])["sd"],
                            f"{label}_full_2000": fd, f"{label}_full_low": flow, f"{label}_full_high": fhigh,
                            f"{label}_full_sign_matches_seed_mean": bool(np.sign(fd) == np.sign(st["mean"])),
                            f"{label}_image_sd_n1000": image_sd,
                            f"{label}_seed_share": st["sd"] ** 2 / (st["sd"] ** 2 + image_sd ** 2)})
                if label == "expected":
                    for s, d in zip(SEEDS, diffs):
                        row[f"expected_{s}"] = d
                    per_seed = [b2_matrix.interval(credit(model, lf, lr, s, rule), credit(model, rf, rr, s, rule))
                                for s in SEEDS]
                    row["expected_separated_seeds"] = sum(lo > 0 or hi < 0 for _, lo, hi in per_seed)
                    row["expected_separated_full"] = bool(flow > 0 or fhigh < 0)
            contrast_rows.append(row)
    tables["contrasts"] = contrast_rows

    # 4. calibration size
    size_rows = []
    for model in MODELS:
        for fmt in lane.LADDER_FORMATS:
            ladder = {"L32": [f"L32_{k}" for k in range(subsets.LADDER_REPLICATES)],
                      "L128": [f"L128_{k}" for k in range(subsets.LADDER_REPLICATES)],
                      "S": list(SEEDS), "H": ["H0", "H1"]}
            for name, members in ladder.items():
                present = [s for s in members if (model, fmt, "default", s) in lane_cells]
                if not present:
                    continue
                for label, rule in READOUTS.items():
                    values = [100 * credit(model, fmt, "default", s, rule).mean() for s in present]
                    st = stats(values)
                    fv = 100 * credit(model, fmt, "default", "FULL", rule).mean()
                    size_rows.append({"model": model, "format": fmt, "readout": label, "range_images": SIZES[name],
                                      "bias_images": min(SIZES[name], 256), "replicates": st["n"],
                                      "mean": st["mean"], "sd": st["sd"], "min": st["min"], "max": st["max"],
                                      "mean_minus_full": st["mean"] - fv, "full_2000": fv,
                                      "values": " ".join(f"{v:.2f}" for v in values)})
    tables["size"] = size_rows

    # 5. bias-only arm against the seed arm
    bias_rows = []
    for model in MODELS:
        for fmt in lane.LADDER_FORMATS:
            bo = [f"B{k}" for k in range(subsets.SEEDS)]
            if not (have(model, fmt, "default", bo) and have(model, fmt, "default", SEEDS)):
                continue
            for label, rule in READOUTS.items():
                b = stats([100 * credit(model, fmt, "default", s, rule).mean() for s in bo])
                s_ = stats([100 * credit(model, fmt, "default", s, rule).mean() for s in SEEDS])
                fv = 100 * credit(model, fmt, "default", "FULL", rule).mean()
                bias_rows.append({"model": model, "format": fmt, "readout": label, "bias_only_mean": b["mean"],
                                  "bias_only_sd": b["sd"], "bias_only_min": b["min"], "bias_only_max": b["max"],
                                  "seed_mean": s_["mean"], "seed_sd": s_["sd"], "full_2000": fv,
                                  "bias_only_variance_fraction_of_seed": (b["sd"] ** 2 / s_["sd"] ** 2) if s_["sd"] else None})
    tables["bias-only"] = bias_rows

    # 5b. protocol addendum 1 (mechanism check, added after seeing the seed arm): range-only versus bias-only
    mech_rows = []
    for model in lane.ADDENDUM_MODELS:
        for fmt in lane.ADDENDUM_FORMATS:
            arms = {"seed": [f"S{k}" for k in range(3)], "range_only": [f"R{k}" for k in range(3)],
                    "bias_only": [f"B{k}" for k in range(3)]}
            if not all(have(model, fmt, "default", members) for members in arms.values()):
                continue
            fv = 100 * credit(model, fmt, "default", "FULL").mean()
            row = {"model": model, "format": fmt, "recipe": "default", "full_2000": fv}
            for arm, members in arms.items():
                values = [100 * credit(model, fmt, "default", s).mean() for s in members]
                row[f"{arm}_mean_k012"] = float(np.mean(values))
                row[f"{arm}_minus_full"] = float(np.mean(values)) - fv
                row[f"{arm}_values"] = " ".join(f"{v:.2f}" for v in values)
            mech_rows.append(row)
    tables["addendum-mechanism"] = mech_rows

    # 6. matrix outcomes per seed (report only)
    entries = {row["name"]: row for row in formats()}
    outcome_rows, class_rows, rule_rows = [], [], []
    for subset in ("FULL",) + SEEDS:
        for fmt, rec in PROPOSAL:
            if subset != "FULL" and not all(have(m, fmt, rec, [subset]) for m in MODELS):
                continue
            drops = []
            values = {}
            for m in MODELS:
                v = 100 * credit(m, fmt, rec, subset).mean()
                values[m] = v
                drops.append(100 * full[(m, "fp32", "baseline")]["credit"]["top1_expected"].mean() - v)
            mean_drop, worst = float(np.mean(drops)), float(np.max(drops))
            outcome_rows.append({"subset": subset, "format": fmt, "recipe": rec, **{f"top1_{m}": values[m] for m in MODELS},
                                 "mean_drop_vs_fp32": mean_drop, "worst_drop_vs_fp32": worst,
                                 "passes_floor": mean_drop <= b2_matrix.FLOOR_MEAN_DROP and worst <= b2_matrix.FLOOR_WORST_DROP})
        for m in MODELS:
            for fmt, rec in PROPOSAL:
                bits = int(entries[fmt]["bits"])
                anchor = b2_matrix.INTEGER_OF_BITS.get(bits)
                if anchor is None or anchor == fmt:
                    continue
                if subset != "FULL" and not (have(m, fmt, rec, [subset]) and have(m, anchor, "default", [subset])):
                    continue
                d, lo, hi = b2_matrix.interval(credit(m, fmt, rec, subset), credit(m, anchor, "default", subset))
                class_rows.append({"subset": subset, "model": m, "format": f"{fmt}:{rec}", "anchor": f"{anchor}:default",
                                   "difference": d, "low": lo, "high": hi, "class": b2_matrix.classify(d, lo, hi)})
            for fmt in SIX_SCALAR:
                if subset != "FULL" and not (have(m, fmt, "default", [subset]) and have(m, fmt, "minimal", [subset])):
                    continue
                a, b = credit(m, fmt, "default", subset), credit(m, fmt, "minimal", subset)
                d, lo, hi = b2_matrix.interval(a, b)
                rule_rows.append({"subset": subset, "model": m, "format": fmt, "default": 100 * a.mean(),
                                  "minimal": 100 * b.mean(), "default_minus_minimal": d, "low": lo, "high": hi,
                                  "better": "default" if lo > 0 else "minimal" if hi < 0 else "not_separated",
                                  "eligible": bool(max(a.mean(), b.mean()) >= b2_matrix.ELIGIBLE_FRACTION)})
    tables["proposal-floor"] = outcome_rows
    tables["class-vs-int"] = class_rows
    tables["rule-c-six-bit"] = rule_rows

    # summaries of the outcome tables
    def changes(rows, key_fields, value_field):
        by = {}
        for row in rows:
            by.setdefault(tuple(row[k] for k in key_fields), {})[row["subset"]] = row[value_field]
        out = []
        for key, values in by.items():
            seeds = [values[s] for s in SEEDS if s in values]
            if "FULL" in values and seeds:
                out.append({**dict(zip(key_fields, key)), "full": values["FULL"],
                            "seeds_equal_full": sum(v == values["FULL"] for v in seeds), "seeds": len(seeds),
                            "per_seed": " ".join(str(values.get(s, "-")) for s in SEEDS)})
        return out

    notes["proposal_floor_changes"] = [row for row in changes(outcome_rows, ("format", "recipe"), "passes_floor")
                                       if row["seeds_equal_full"] != row["seeds"]]
    notes["class_changes"] = [row for row in changes(class_rows, ("model", "format"), "class")
                              if row["seeds_equal_full"] != row["seeds"]]
    rule_summary = []
    for subset in ("FULL",) + SEEDS:
        rows = [row for row in rule_rows if row["subset"] == subset and row["eligible"]]
        if not rows:
            continue
        lost_default = [f"{row['model']}/{row['format']}" for row in rows if row["high"] < 0]
        lost_minimal = [f"{row['model']}/{row['format']}" for row in rows if row["low"] > 0]
        rule_summary.append({"subset": subset, "eligible_six_bit_scalar_cells": len(rows),
                             "R_bits6_pick_default_separated_worse": len(lost_default),
                             "R_bits7_pick_minimal_separated_worse": len(lost_minimal),
                             "R_bits6_better_at_six_bits": len(lost_default) < len(lost_minimal),
                             "cells_default_worse": " ".join(lost_default), "cells_minimal_worse": " ".join(lost_minimal)})
    tables["rule-c-summary"] = rule_summary
    notes["rule_c_six_bit_pick_changes"] = [row for row in changes(rule_rows, ("model", "format"), "better")
                                            if row["seeds_equal_full"] != row["seeds"]]
    return tables, notes


def recommendation(tables):
    rows = tables["contrasts"]
    out = {"contrasts": len(rows)}
    for label in READOUTS:
        out[label] = {
            "needs_five_after_3": sum(bool(row[f"{label}_needs_five_after_3"]) for row in rows),
            "needs_five_after_5": sum(bool(row[f"{label}_needs_five_after_5"]) for row in rows),
            "sign_change_5": sum(not row[f"{label}_signs_consistent_5"] for row in rows),
            "sd_above_half_delta_5": sum((row[f"{label}_sd"] or 0) > DELTA / 2 for row in rows),
            "needs_five_with_abs_mean_above_delta": [f"{row['model']}:{row['left']}-{row['right']}" for row in rows
                                                     if row[f"{label}_needs_five_after_5"] and abs(row[f"{label}_mean"]) > DELTA],
        }
    cells = tables["seed-cells"]
    out["cells"] = len(cells)
    out["cells_sd_above_half_delta"] = {label: [f"{c['model']}:{c['format']}:{c['recipe']}" for c in cells
                                                if c[f"{label}_sd"] > DELTA / 2] for label in READOUTS}
    out["median_cell_seed_sd"] = {label: float(np.median([c[f"{label}_sd"] for c in cells])) for label in READOUTS}
    out["median_cell_image_sd"] = {label: float(np.median([c[f"{label}_image_sd_n1000"] for c in cells]))
                                   for label in READOUTS}
    out["median_contrast_seed_sd"] = {label: float(np.median([c[f"{label}_sd"] for c in rows])) for label in READOUTS}
    out["median_contrast_image_sd"] = {label: float(np.median([c[f"{label}_image_sd_n1000"] for c in rows]))
                                       for label in READOUTS}
    return out


def write_once(folder, name, text, partial):
    path = Path(folder) / name
    if path.exists() and not partial:
        raise FileExistsError(f"summary exists and is written once: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--partial", action="store_true", help="scratch run: allow overwriting files in --out")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if not args.partial and out.resolve() != OUT.resolve():
        raise SystemExit("--out other than the summary folder needs --partial")
    lane_cells, full = load()
    tables, notes = analyse(lane_cells, full)
    for name, rows in tables.items():
        write_once(out, f"{name}.csv", csv_text(rows), args.partial)
    expected = {arm: len(lane.arm_jobs(arm)) * len(MODELS) for arm in lane.ARMS}
    measured = {arm: sum((m, f, rc, s) in lane_cells for m in MODELS for f, rc, s in lane.arm_jobs(arm))
                for arm in lane.ARMS}
    summary = {
        "version": "b2-seeds-v1", "protocol": {"path": str(subsets.PROTOCOL_FILE.relative_to(ROOT)),
                                               "sha256": file_hash(subsets.PROTOCOL_FILE)},
        "evidence_class": "development evidence (1k screen); seeds are disjoint subsamples of the one 2k calibration list",
        "cells_planned": expected, "cells_measured": measured,
        "reproduction": [unseal(p) | {"file": str(p.relative_to(ROOT))} for p in sorted((lane.BASE / "reproduction").glob("*.json"))],
        "recommendation_inputs": recommendation(tables), "notes": notes,
        "sources": {f"tools/experiment_b2_seeds/{n}": file_hash(ROOT / "tools/experiment_b2_seeds" / n)
                    for n in ("subsets.py", "cells.py", "analysis.py")},
        "lane_cell_records": {c["file"]: c["record_sha256"] for c in lane_cells.values()},
        "bootstrap": {"method": "paired image bootstrap, tools.analysis.b2_matrix.interval / b2_ties.paired",
                      "resamples": RESAMPLES, "seed": SEED},
    }
    for row in summary["reproduction"]:
        row.pop("own_sources", None)
    write_once(out, "summary.json", json.dumps(summary, indent=1, sort_keys=True, default=float), args.partial)
    print(json.dumps({"out": str(out), "cells_measured": measured, "tables": {k: len(v) for k, v in tables.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
