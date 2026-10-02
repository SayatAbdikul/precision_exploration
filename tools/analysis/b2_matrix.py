"""Tables of the B2 format matrix (protocol b2-matrix-protocol-v1): read-only over sealed cell records.

Usage: .venv/bin/python -m tools.analysis.b2_matrix [--tag NAME] [--digits N]
``--digits 4 --tag d4`` writes the same analysis with 4-decimal values (the document renders 1 decimal from it,
so that a displayed value is the rounding of the exact value, not of a 2-decimal value).
Writes results/summaries/b2-matrix-v1/ (a file that exists with other content is never replaced:
pass --tag to write a new versioned set).  Everything is development evidence (1k screen).

``--partial DIR`` is a development aid while the campaign runs: it analyses the formats whose cells
exist for every requested model, skips what is missing, and writes to ``DIR`` (never to results/).

The analysis rules (a)-(d) are those fixed in the protocol, section
``analysis_rules_stated_in_advance``; the functions below implement them on plain dictionaries
``(model, format, recipe) -> per-image credit`` so that they can be unit-tested without evidence files.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from tools.analysis.b2_ties import RESAMPLES, SEED, paired
from tools.analysis.b_stage_balanced_comparisons import correctness, paired_outcomes
from tools.experiment_b.common import ROOT, dataset, digest, file_hash, formats, unseal
from tools.experiment_b2 import readout

MATRIX = ROOT / "artifacts/experiment_b2/matrix"
OUT = ROOT / "results/summaries/b2-matrix-v1"
MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large")
RECIPES = ("default", "minimal")
V1_RECIPES = ("v1_maxabs", "v1_percentile_99_9")
INTRINSIC = {"default": "cum5_act_maxabs", "minimal": "cum1_fused"}
SHARED = ("bfp6", "mxfp4_e2m1", "mxfp6_e3m2", "mxfp8_e4m3")
FAMILY = {"integer": "integer", "fixed_point": "fixed_point", "float": "float", "posit": "posit",
          "logarithmic": "logarithmic", "bfp": "shared_exponent", "mx_float": "shared_exponent",
          "codebook": "codebook", "binary": "binary_ternary", "ternary": "binary_ternary"}
INTEGER_OF_BITS = {8: "int8", 6: "int6", 5: "int5", 4: "int4"}
RULE_THRESHOLDS = (5, 6, 7, 8)
ELIGIBLE_FRACTION = 0.10
FLOOR_MEAN_DROP, FLOOR_WORST_DROP, PROPOSAL_SIZE, PROPOSAL_MINIMUM = 5.0, 10.0, 10, 6
_W = {}


def interval(left, right):
    """``b2_ties.paired`` (left minus right, points) computed with one shared resampling matrix per length.

    Returns ``(difference, low, high)`` in points; the interval is the pointwise 95 percent paired
    image bootstrap of the mean credit difference (10000 resamples, the project seed).
    """
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("credit vectors are not paired")
    n = len(left)
    if n not in _W:
        index = np.random.default_rng(SEED).integers(0, n, size=(RESAMPLES, n))
        weights = np.zeros((RESAMPLES, n), dtype=np.float64)
        np.add.at(weights, (np.arange(RESAMPLES)[:, None], index), 1.0)
        check = paired(left, right)["pointwise_95_interval_pp"]
        mine = np.percentile(weights @ (left - right) / n, [2.5, 97.5]) * 100
        if not np.allclose(check, mine, atol=1e-9):
            raise ValueError("fast bootstrap differs from tools.analysis.b2_ties.paired")
        _W[n] = weights
    low, high = np.percentile(_W[n] @ (left - right) / n, [2.5, 97.5]) * 100
    return 100 * float((left - right).mean()), float(low), float(high)


def load_cells(models=MODELS):
    cells = {}
    for path in sorted((MATRIX / "cells").glob("*.json")):
        record = unseal(path)
        if record["model"] not in models:
            continue
        key = (record["model"], record["format"], record["recipe_name"])
        if key in cells:
            raise ValueError(f"duplicate cell record: {key}")
        file = ROOT / record["readout_file"]
        if file_hash(file) != record["readout_file_sha256"]:
            raise ValueError(f"readout record drift: {file}")
        arrays = readout.load(file)
        if record["images"] != 1000 or len(arrays["label"]) != 1000 or record["inference_batch_size"] != 8:
            raise ValueError(f"cell is not a batch-8 run of the 1k screen: {path}")
        cells[key] = {"record": record, "arrays": arrays, "credit": readout.credits(arrays),
                      "record_sha256": file_hash(path)}
    labels = {m: {cell["arrays"]["label"].tobytes() for (model, _, _), cell in cells.items() if model == m} for m in models}
    rows = {m: {cell["record"]["rows_sha256"] for (model, _, _), cell in cells.items() if model == m} for m in models}
    if any(len(v) > 1 for v in labels.values()) or any(len(v) > 1 for v in rows.values()):
        raise ValueError("cells of one model are not on the same ordered screen rows")
    return cells


def v1_cells(rows, models=MODELS):
    """Sealed v1 / extension 1k results: binary top-1 correctness per image under the sealed topk readout."""
    result = {}
    with (ROOT / "results/summaries/b-stage-paired-1k-v2/configurations.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["model"] not in models or row["metric"] != "top1" or row["images"] != "1000":
                continue
            root = ROOT / ("artifacts/experiment_b_ext" if row["format"] in SHARED else "artifacts/experiment_b")
            folder = root / "predictions" / row["configuration_sha256"]
            records = [unseal(folder / (sample["sha256"] + ".json")) for sample in rows]
            if any(r["sample"] != s for r, s in zip(records, rows)):
                raise ValueError("v1 prediction/sample mismatch")
            credit = correctness(records)[:, 0].astype(np.float64)
            if abs(100 * credit.mean() - float(row["candidate_percent"])) > 1e-6:
                raise ValueError("v1 predictions do not reproduce the sealed summary")
            result[(row["model"], row["format"], "v1_" + row["recipe"])] = credit
    return result


def ordering(credits):
    """``credits``: format -> per-image credit.  Returns the order (best first) and, for every pair
    ``(higher, lower)`` in that order, ``(difference, low, high, separated)``."""
    names = sorted(credits, key=lambda f: (-credits[f].mean(), f))
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d, low, high = interval(credits[a], credits[b])
            pairs[(a, b)] = (d, low, high, bool(low > 0 or high < 0))
    return names, pairs


def reversed_pairs(first, second):
    """Format pairs whose separated order in ``first`` is the opposite separated order in ``second``."""
    (_, pairs_a), (rank_b, pairs_b) = first, second
    return [(x, y) for (x, y), value in pairs_a.items()
            if value[3] and x in rank_b and y in rank_b and pairs_b.get((y, x), (0, 0, 0, False))[3]]


def classify(d, low, high):
    """Rule (b): the class of a format against an anchor from the paired difference and its interval."""
    if abs(d) <= 1.0:
        return "within_one_point"
    if d < -1.0 and high < 0:
        return "separated_below"
    if d > 1.0 and low > 0:
        return "separated_above"
    return "not_resolved"


def candidate_rules():
    rules = {"R_default": lambda bits: "default", "R_minimal": lambda bits: "minimal"}
    for threshold in RULE_THRESHOLDS:
        rules[f"R_bits({threshold})"] = lambda bits, t=threshold: "default" if bits >= t else "minimal"
    return rules


def recipe_pairs(credit, entries, models, names):
    rows = []
    for model in models:
        for name in names:
            default, minimal = credit[(model, name, "default")], credit[(model, name, "minimal")]
            d, low, high = interval(default, minimal)
            rows.append({"model": model, "format": name, "family": FAMILY[entries[name]["family"]],
                         "bits": int(entries[name]["bits"]), "default": 100 * float(default.mean()),
                         "minimal": 100 * float(minimal.mean()), "default_minus_minimal": d, "low": low, "high": high,
                         "better": "default" if low > 0 else "minimal" if high < 0 else "not_separated",
                         "eligible": bool(max(default.mean(), minimal.mean()) >= ELIGIBLE_FRACTION)})
    return rows


def score_rules(pair_rows):
    """Rule (c): score every candidate rule on the eligible cells; returns the scores and the winner's name."""
    scores = []
    for label, rule in candidate_rules().items():
        losses, regret, count, lost = 0, 0.0, 0, []
        for row in pair_rows:
            if not row["eligible"]:
                continue
            sign = 1 if rule(row["bits"]) == "default" else -1
            gain, upper = sign * row["default_minus_minimal"], (row["high"] if sign == 1 else -row["low"])
            if upper < 0:
                losses += 1
                lost.append(f"{row['model']}/{row['format']}")
            regret += max(0.0, -gain)
            count += 1
        scores.append({"rule": label, "eligible_cells": count, "cells_significantly_worse": losses,
                       "mean_regret_points": round(regret / max(count, 1), 3),
                       "parameters": 1 if label.startswith("R_bits") else 0, "cells_lost": " ".join(lost)})
    best = min(scores, key=lambda s: (s["cells_significantly_worse"], s["mean_regret_points"], s["parameters"]))
    return scores, best["rule"]


def family_exceptions(pair_rows, best, models):
    """Per-family exceptions to the selected rule.

    The protocol adopts an exception only when it holds on all three models with intervals excluding
    zero.  Operational form (fixed before the MobileNetV3 cells existed): for one family, another
    candidate rule replaces the selected one when, for every format of the family on which the two
    rules pick different recipes, the other rule's pick is better with an interval excluding zero on
    every model.  Among several such rules the one that changes the most formats is taken.  Everything
    else is listed as an observation and not adopted.
    """
    rules = candidate_rules()
    base = rules[best]
    families, observations = {}, []
    for row in pair_rows:
        families.setdefault(row["family"], {}).setdefault(row["format"], []).append(row)

    def better(row, pick):
        return row["low"] > 0 if pick == "default" else row["high"] < 0

    adopted = []
    for family, group in sorted(families.items()):
        options = []
        for label, other in rules.items():
            differing = [name for name, rows in group.items() if other(rows[0]["bits"]) != base(rows[0]["bits"])]
            if differing and all(len(group[name]) == len(models) and all(better(row, other(row["bits"])) for row in group[name])
                                 for name in differing):
                options.append((len(differing), label, differing))
        if options:
            size = max(option[0] for option in options)
            _, label, differing = next(option for option in options if option[0] == size)
            adopted.append({"family": family, "formats": sorted(differing), "equivalent_rules_for_this_family":
                            [option[1] for option in options if sorted(option[2]) == sorted(differing)],
                            "recipe": {name: rules[label](group[name][0]["bits"]) for name in sorted(differing)}})
        for name, rows in group.items():
            pick = base(rows[0]["bits"])
            other = "minimal" if pick == "default" else "default"
            worse = [row["model"] for row in rows if better(row, other)]
            if worse:
                observations.append({"format": name, "family": family, "bits": rows[0]["bits"], "rule_pick": pick,
                                     "models_where_the_other_recipe_is_separated_better": worse,
                                     "default_minus_minimal": {row["model"]: round(row["default_minus_minimal"], DIGITS) for row in rows}})
    picks = {name: base(rows[0]["bits"]) for group in families.values() for name, rows in group.items()}
    for exception in adopted:
        picks.update(exception["recipe"])
    return {"adopted": adopted, "observations": observations}, picks


def propose(credit, entries, models, names, picks, excluded=()):
    """Rule (d): candidates per (family, bit width), quality floor, selection; anchors are added by the caller.

    ``picks``: format -> recipe (the rule of (c) with its adopted exceptions).
    """
    fp32 = {m: credit[(m, "fp32", "baseline")] for m in models}
    candidates, considered = {}, []
    for name in names:
        bits, family = int(entries[name]["bits"]), FAMILY[entries[name]["family"]]
        if name == "int8" or name in excluded:
            continue
        recipe = picks[name]
        used = INTRINSIC[recipe] if name in SHARED else recipe
        if any((m, name, used) not in credit for m in models):
            continue
        top1 = [100 * float(credit[(m, name, used)].mean()) for m in models]
        drops = [100 * float(fp32[m].mean()) - value for m, value in zip(models, top1)]
        entry = {"format": name, "family": family, "bits": bits, "recipe": used, "top1": [round(v, DIGITS) for v in top1],
                 "mean_top1": float(np.mean(top1)), "mean_drop": float(np.mean(drops)), "worst_drop": float(max(drops)),
                 "passes_floor": bool(np.mean(drops) <= FLOOR_MEAN_DROP and max(drops) <= FLOOR_WORST_DROP)}
        considered.append(entry)
        key = (family, bits)
        if key not in candidates or entry["mean_top1"] > candidates[key]["mean_top1"]:
            candidates[key] = entry
    passing = sorted((c for c in candidates.values() if c["passes_floor"]), key=lambda c: (c["mean_drop"], c["format"]))
    selected, note = list(passing), "all passing candidates"
    if len(passing) > PROPOSAL_SIZE:
        keep = []
        for family in sorted({c["family"] for c in passing}):
            own = sorted((c for c in passing if c["family"] == family), key=lambda c: c["bits"])
            keep += [own[0]] + ([own[-1]] if len(own) > 1 else [])
        fill = [c for c in passing if c not in keep][:max(0, PROPOSAL_SIZE - len(keep))]
        selected = keep + fill
        note = (f"{len(passing)} candidates pass; kept the lowest and highest passing bit width of every family "
                f"({len(keep)}) and filled {len(fill)} by ascending mean drop")
    stress = []
    if len(passing) < PROPOSAL_MINIMUM:
        for family in sorted({c["family"] for c in candidates.values()}):
            floor_bits = min((c["bits"] for c in passing if c["family"] == family), default=33)
            lower = [c for c in candidates.values() if c["family"] == family and c["bits"] < floor_bits and not c["passes_floor"]]
            if lower:
                stress.append({**max(lower, key=lambda c: c["bits"]), "flag": "stress_configuration_below_the_floor"})
        note = f"only {len(passing)} candidates pass; one flagged stress configuration per family added"
    order = lambda c: (-c["bits"], c["mean_drop"], c["format"])  # noqa: E731
    return {"selected": sorted(selected, key=order), "stress": sorted(stress, key=order), "selection_note": note,
            "candidates_per_family_and_bits": sorted(candidates.values(), key=lambda c: (c["family"], -c["bits"])),
            "all_formats_under_the_rule": sorted(considered, key=order),
            "passing_candidates": len(passing)}


def write(path, text):
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"{path} exists with other content; use --tag for a new versioned set")
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return "written"


def table(rows):
    fields = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


DIGITS = 2  # decimals of percentage-point values in the tables; v1 used 2, ``--digits`` writes a wider set


def r(value, digits=None):
    return None if value is None else round(float(value), DIGITS if digits is None else max(digits, DIGITS))


def rounded(rows, digits=None):
    digits = DIGITS if digits is None else digits
    return [{k: (round(v, digits) if isinstance(v, float) else v) for k, v in row.items()} for row in rows]


def stage1_reference():
    """Expected and lowest-index top-1 of the stage-1 r3c tie records (27 default cells), for a cross-check."""
    result = {}
    with (ROOT / "results/summaries/b2-baseline-repair-v1/logit-ties-1k.csv").open() as stream:
        for row in csv.DictReader(stream):
            result[(row["model"], row["format"], row["recipe"])] = (float(row["top1_expected"]), float(row["top1_lowest_index"]))
    return result


def signed_int8_alias(cells, rows1k, models):
    """q1_6 against signed INT8: equal readout arrays under minimal, and the sealed stage-1 ``default_signed``
    INT8 top-5 lists against the q1_6 default cell."""
    from tools.experiment_b.validation import verify_prediction
    result = {}
    for model in models:
        if (model, "q1_6", "minimal") not in cells or (model, "q1_6", "default") not in cells:
            continue
        minimal = all(np.array_equal(cells[(model, "q1_6", "minimal")]["arrays"][k], cells[(model, "int8", "minimal")]["arrays"][k])
                      for k in readout.DTYPES)
        signed = sorted((ROOT / "artifacts/experiment_b2/runs").glob(f"*/{model}--int8--default_signed--1000--*.json"))
        equal = None
        if signed:
            identity = unseal(signed[0])["configuration_sha256"]
            folder = ROOT / "artifacts/experiment_b2/predictions" / identity
            top5 = cells[(model, "q1_6", "default")]["arrays"]["top5_topk"]
            equal = sum(verify_prediction(folder / (row["sha256"] + ".json"), identity, row)["top5"] == [int(x) for x in mine]
                        for row, mine in zip(rows1k, top5))
        result[model] = {"minimal_readout_identical_to_int8_minimal": bool(minimal),
                         "default_top5_lists_equal_to_sealed_int8_default_signed": equal}
    return result


def analyse(cells, entries, models, names, old, rows1k=None, partial=False):
    """All tables of the matrix; ``old`` holds the sealed v1 credits (may be empty)."""
    primary = {key: cell["credit"]["top1_expected"] for key, cell in cells.items()}
    lowest = {key: cell["credit"]["top1_lowest_index"] for key, cell in cells.items()}
    fp32 = {m: primary[(m, "fp32", "baseline")] for m in models}
    shared = [f for f in names if f in SHARED]

    # 1. every cell
    cell_rows = []
    for (model, name, recipe), cell in sorted(cells.items()):
        record, credit = cell["record"], cell["credit"]
        base = {v: k for k, v in INTRINSIC.items()}.get(recipe, recipe)
        arm = "baseline" if name == "fp32" else "intrinsic" if recipe in INTRINSIC.values() else "matrix"
        row = {"model": model, "format": name, "family": record["family"], "bits": record["bits"], "recipe": recipe,
               "arm": arm, "configuration": record["configuration_sha256"][:12], "stage1_cell": bool(record.get("stage1")),
               **{k: r(record["readout"][f"{k}_percent"]) for k in ("top1_expected", "top1_lowest_index", "top1_topk",
                                                                    "top5_expected", "top5_lowest_index")},
               "tied_top1_images": record["readout"]["images_with_tied_top1"],
               "tied_top1_images_involving_label": record["readout"]["images_with_tied_top1_involving_the_label"],
               "largest_top1_tie": record["readout"]["largest_top1_tie"]}
        if arm != "baseline":
            anchors = [("fp32", fp32[model])]
            if (model, "int8", base) in primary:
                anchors.append(("int8", primary[(model, "int8", base)]))
            for label, other in anchors:
                d, low, high = interval(credit["top1_expected"], other)
                row.update({f"d_{label}": r(d), f"d_{label}_low": r(low), f"d_{label}_high": r(high)})
            reference = cells[(model, "fp32", "baseline")]["credit"]
            d5, low5, high5 = interval(credit["top5_expected"], reference["top5_expected"])
            second = paired_outcomes(reference["top1_lowest_index"].astype(np.int8), credit["top1_lowest_index"].astype(np.int8))
            row.update({"d5_fp32": r(d5), "d5_fp32_low": r(low5), "d5_fp32_high": r(high5),
                        "lowest_d_fp32": r(second["difference_pp"]),
                        "lowest_d_fp32_low": r(second["pointwise_95_interval_pp"][0]),
                        "lowest_d_fp32_high": r(second["pointwise_95_interval_pp"][1]),
                        "lowest_mcnemar_p": float(f"{second['mcnemar_exact_p']:.3g}")})
            if (model, "int8", base) in lowest:
                second = paired_outcomes(lowest[(model, "int8", base)].astype(np.int8), credit["top1_lowest_index"].astype(np.int8))
                row.update({"lowest_d_int8": r(second["difference_pp"]),
                            "lowest_d_int8_low": r(second["pointwise_95_interval_pp"][0]),
                            "lowest_d_int8_high": r(second["pointwise_95_interval_pp"][1])})
            occupancy = (record.get("occupancy") or {}).get("summary") or {}
            row.update({k: r(occupancy.get(k), 3) for k in ("median_sqnr_db", "min_sqnr_db", "median_entropy_bits",
                                                           "median_levels_used_fraction", "median_fraction_at_extremes",
                                                           "max_fraction_at_extremes", "median_fraction_zero",
                                                           "median_fraction_blocks_clipped")})
        cell_rows.append(row)

    # 2. orderings: B2 recipes (expected credit; lowest index as a sensitivity check) and sealed v1 (topk)
    groups = {}
    for model in models:
        for recipe in RECIPES:
            groups[(model, recipe)] = {f: primary[(model, f, recipe)] for f in names}
        for recipe in V1_RECIPES:
            group = {f: old[(model, f, recipe)] for f in names if (model, f, recipe) in old}
            if group:
                groups[(model, recipe)] = group
    orders = {key: ordering(value) for key, value in groups.items()}
    order_rows, change_rows, sensitivity = [], [], []
    for (model, recipe), (ranked, pairs) in orders.items():
        for rank, name in enumerate(ranked, 1):
            row = {"model": model, "recipe": recipe, "rank": rank, "format": name, "bits": int(entries[name]["bits"]),
                   "top1": r(100 * groups[(model, recipe)][name].mean()),
                   "readout": "sealed_topk" if recipe.startswith("v1") else "expected_credit"}
            if rank < len(ranked):
                d, low, high, flag = pairs[(name, ranked[rank])]
                row.update({"minus_next": r(d), "minus_next_low": r(low), "minus_next_high": r(high), "separated_from_next": flag})
            # the best-ranked format this one is not separated from (the top of its tie group)
            row["not_separated_from_rank"] = next(i for i, other in enumerate(ranked, 1)
                                                  if other == name or not pairs[(other, name)][3])
            order_rows.append(row)
    for model in models:
        for before, after in [(a, b) for a in V1_RECIPES + ("minimal",) for b in RECIPES if a != b]:
            if (model, before) not in orders:
                continue
            (rank_a, _), (rank_b, _) = orders[(model, before)], orders[(model, after)]
            common = [f for f in rank_a if f in rank_b]
            a_only = [f for f in rank_a if f in common]
            b_only = [f for f in rank_b if f in common]
            rho = spearmanr([a_only.index(f) for f in common], [b_only.index(f) for f in common]).statistic
            flipped = reversed_pairs(orders[(model, before)], orders[(model, after)])
            floor = lambda key, f: 100 * groups[key][f].mean() < 1.0  # noqa: E731
            collapsed = [(x, y) for x, y in flipped if any(floor(key, x) and floor(key, y) for key in ((model, before), (model, after)))]
            change_rows.append({"model": model, "from": before, "to": after, "formats_in_common": len(common),
                                "spearman": r(rho, 3), "separated_pairs_reversed": len(flipped),
                                "of_which_both_below_1_percent_in_one_ordering": len(collapsed),
                                "pairs_first_above_second_in_from": " ".join(f"{x}>{y}" for x, y in flipped),
                                "missing_in_from": " ".join(f for f in names if f not in rank_a)})
        for recipe in RECIPES:
            first = [100 * primary[(model, f, recipe)].mean() for f in names]
            second = [100 * lowest[(model, f, recipe)].mean() for f in names]
            gaps = np.abs(np.array(first) - np.array(second))
            sensitivity.append({"model": model, "recipe": recipe,
                                "spearman_expected_vs_lowest_index": r(spearmanr(first, second).statistic, 4),
                                "largest_abs_gap_points": r(gaps.max()), "format_of_largest_gap": names[int(gaps.argmax())],
                                "formats_with_gap_above_1_point": int((gaps > 1.0).sum())})

    # 3. per bit width: against INT8 of the same recipe and against the integer of the same width
    width_rows = []
    for model in models:
        for recipe in RECIPES:
            for name in names:
                bits = int(entries[name]["bits"])
                row = {"model": model, "recipe": recipe, "format": name, "family": FAMILY[entries[name]["family"]], "bits": bits,
                       "top1": r(100 * primary[(model, name, recipe)].mean())}
                for label, anchor in (("int8", "int8"), ("same_width_integer", INTEGER_OF_BITS.get(bits))):
                    if anchor is None or anchor == name or (model, anchor, recipe) not in primary:
                        row.update({f"vs_{label}": None, f"vs_{label}_low": None, f"vs_{label}_high": None,
                                    f"class_{label}": "anchor" if anchor == name else "no_anchor"})
                        continue
                    d, low, high = interval(primary[(model, name, recipe)], primary[(model, anchor, recipe)])
                    row.update({f"vs_{label}": r(d), f"vs_{label}_low": r(low), f"vs_{label}_high": r(high),
                                f"class_{label}": classify(d, low, high)})
                width_rows.append(row)

    # 4. recipe rule (primary readout decides; the lowest-index readout is a sensitivity check)
    pair_rows = recipe_pairs(primary, entries, models, names)
    scores, best = score_rules(pair_rows)
    exceptions, picks = family_exceptions(pair_rows, best, models)
    pairs_lowest = recipe_pairs(lowest, entries, models, names)
    scores_lowest, best_lowest = score_rules(pairs_lowest)
    exceptions_lowest, picks_lowest = family_exceptions(pairs_lowest, best_lowest, models)
    base_picks = {name: candidate_rules()[best](int(entries[name]["bits"])) for name in names}

    # 5. exact-engine proposal
    alias = signed_int8_alias(cells, rows1k, models) if rows1k is not None and "q1_6" in names else {}
    q16_alias = bool(alias) and len(alias) == len(models) and all(v["minimal_readout_identical_to_int8_minimal"] for v in alias.values())
    excluded = ("q1_6",) if q16_alias else ()
    proposal = propose(primary, entries, models, names, picks, excluded)
    proposal_lowest = propose(lowest, entries, models, names, picks_lowest, excluded)
    proposal_plain = propose(primary, entries, models, names, base_picks, excluded)
    label = lambda group: [f"{c['format']}/{c['recipe']}" for c in group]  # noqa: E731
    int8_recipe = picks["int8"]
    proposal = {"status": "PROPOSAL for the owner's sign-off; selected by the rule fixed in the protocol; development evidence",
                "recipe_rule": best, "model_order": list(models),
                "anchors": [{"format": "fp32", "recipe": "baseline", "top1": [r(100 * fp32[m].mean()) for m in models]},
                            {"format": "int8", "recipe": int8_recipe,
                             "top1": [r(100 * primary[(m, "int8", int8_recipe)].mean()) for m in models]}],
                **proposal,
                "q1_6_excluded_as_alias_of_signed_int8": q16_alias, "q1_6_alias_checks": alias,
                "recipe_per_format": picks,
                "sensitivity_lowest_index_readout": {"recipe_rule": best_lowest, "exceptions": exceptions_lowest["adopted"],
                                                     "selected": label(proposal_lowest["selected"]),
                                                     "stress": label(proposal_lowest["stress"])},
                "sensitivity_rule_without_exceptions": {"selected": label(proposal_plain["selected"]),
                                                        "stress": label(proposal_plain["stress"])}}
    proposal_rows = []
    for role, group in (("anchor", proposal["anchors"]), ("selected", proposal["selected"]), ("stress", proposal["stress"])):
        for c in group:
            proposal_rows.append({"role": role, "format": c["format"], "family": c.get("family", "anchor"),
                                  "bits": c.get("bits", 32 if c["format"] == "fp32" else 8), "recipe": c["recipe"],
                                  **{f"top1_{m}": v for m, v in zip(models, c["top1"])},
                                  "mean_drop_vs_fp32": r(c.get("mean_drop", np.mean([a - b for a, b in zip(proposal["anchors"][0]["top1"], c["top1"])]))),
                                  "worst_drop_vs_fp32": r(c.get("worst_drop", max(a - b for a, b in zip(proposal["anchors"][0]["top1"], c["top1"])))),
                                  "passes_floor": c.get("passes_floor", True)})

    # 6. shared-exponent arms and scale metadata
    shared_rows, meta_rows = [], []
    for model in models:
        for name in shared:
            for recipe in RECIPES:
                if (model, name, INTRINSIC[recipe]) not in primary:
                    if partial:
                        continue
                    raise SystemExit(f"intrinsic cell missing: {model}/{name}/{INTRINSIC[recipe]}")
                d, low, high = interval(primary[(model, name, recipe)], primary[(model, name, INTRINSIC[recipe])])
                occupancy = cells[(model, name, recipe)]["record"]["occupancy"]
                shared_rows.append({"model": model, "format": name, "recipe": recipe, "intrinsic_recipe": INTRINSIC[recipe],
                                    "searched_activation_blocks": r(100 * primary[(model, name, recipe)].mean()),
                                    "intrinsic_activation_blocks": r(100 * primary[(model, name, INTRINSIC[recipe])].mean()),
                                    "searched_minus_intrinsic": r(d), "low": r(low), "high": r(high),
                                    "median_fraction_blocks_clipped_stored": r(occupancy["summary"].get("median_fraction_blocks_clipped"), 4),
                                    "median_fraction_blocks_clipped_patches": r(occupancy["patch_summary"].get("median_fraction_blocks_clipped"), 4)})
            meta = cells[(model, name, "default")]["record"]["block_metadata"]
            bits = int(entries[name]["bits"])
            meta_rows.append({"model": model, "format": name, "element_bits": bits,
                              **{f"{k}_scale_bits_per_element": r(meta[k]["scale_bits_per_element"], 4)
                                 for k in ("weights", "stored_activations_per_image", "convolution_patches_per_image")},
                              "effective_weight_bits": r(bits + meta["weights"]["scale_bits_per_element"], 3),
                              "effective_stored_activation_bits": r(bits + meta["stored_activations_per_image"]["scale_bits_per_element"], 3)})

    # 7. stage-1 cells under the fixed readout, cross-checked against the r3c tie records
    reference, stage1_rows = stage1_reference(), []
    for (m, f, rc), c in sorted(cells.items()):
        if not c["record"].get("stage1") or f == "fp32":
            continue
        row = {"model": m, "format": f, "recipe": rc, "sealed_topk_top1": r(c["record"]["stage1"]["sealed_top1_percent"]),
               "expected_top1": r(c["record"]["readout"]["top1_expected_percent"]),
               "lowest_index_top1": r(c["record"]["readout"]["top1_lowest_index_percent"]),
               "tied_top1_images": c["record"]["readout"]["images_with_tied_top1"],
               "sealed_top5_lists_reproduced": c["record"]["stage1"]["sealed_top5_lists_reproduced"],
               "agrees_with_r3c_tie_record": None}
        if (m, f, rc) in reference:
            expected, low_index = reference[(m, f, rc)]
            row["agrees_with_r3c_tie_record"] = bool(abs(expected - c["record"]["readout"]["top1_expected_percent"]) < 1e-9
                                                     and abs(low_index - c["record"]["readout"]["top1_lowest_index_percent"]) < 1e-9)
        stage1_rows.append(row)
    digests = {}
    for (model, name, recipe), cell in cells.items():
        digests.setdefault((model, cell["record"]["logits_sha256"]), []).append(f"{name}/{recipe}")
    aliases = [{"model": model, "cells": sorted(group)} for (model, _), group in sorted(digests.items()) if len(group) > 1]

    counts = {}
    for row in width_rows:
        for label in ("int8", "same_width_integer"):
            key = f"{row['recipe']}|vs_{label}|{row[f'class_{label}']}"
            counts[key] = counts.get(key, 0) + 1
    regression = [unseal(p) for p in sorted((MATRIX / "regression").glob("*--r2.json"))]
    summary = {"version": "b2-matrix-summary-1", "evidence": "development evidence: frozen 1k ImageNet screen, on which the recipes were selected",
               "partial": bool(partial), "models": list(models), "formats": list(names),
               "protocol": cells[(models[0], "fp32", "baseline")]["record"]["protocol"],
               "cells": len(cells), "stage1_cells_reused": len(stage1_rows),
               "stage1_cells_agreeing_with_r3c_tie_records": sum(row["agrees_with_r3c_tie_record"] is True for row in stage1_rows),
               "stage1_cells_disagreeing_with_r3c_tie_records": sum(row["agrees_with_r3c_tie_record"] is False for row in stage1_rows),
               "own_sources": sorted({json.dumps(c["record"]["own_sources"], sort_keys=True) for c in cells.values()}),
               "source_sha256": sorted({c["record"]["source_sha256"] for c in cells.values()}),
               "cell_records_sha256": digest(sorted(c["record_sha256"] for c in cells.values())),
               "readout": {"primary": "expected_credit", "secondary": "lowest_class_index",
                           "fp32_tied_images": {m: cells[(m, "fp32", "baseline")]["record"]["readout"]["images_with_tied_top1"] for m in models},
                           "ordering_sensitivity": sensitivity},
               "bootstrap": {"method": "paired image bootstrap of the mean credit difference (tools.analysis.b2_ties.paired)",
                             "resamples": RESAMPLES, "seed": SEED, "interval": "pointwise 95 percent, no multiplicity correction"},
               "bit_width_class_counts": dict(sorted(counts.items())),
               "recipe_rule": {"scores": scores, "selected": best, "family_exceptions_adopted": exceptions["adopted"],
                               "observations_not_adopted": exceptions["observations"],
                               "sensitivity_lowest_index_readout": {"scores": scores_lowest, "selected": best_lowest,
                                                                    "family_exceptions_adopted": exceptions_lowest["adopted"]}},
               "proposal": proposal, "identical_logit_aliases": aliases,
               "shared_regression": [{k: row[k] for k in ("model", "format", "recipe_name", "passed",
                                                          "logit_bit_identical_images", "weights_bit_identical",
                                                          "sealed_extension_top5_reproduced_images")} for row in regression]}
    tables = {"cells": cell_rows, "ordering": order_rows, "ordering-changes": change_rows, "bit-width": width_rows,
              "recipe-pairs": rounded(pair_rows), "proposal": proposal_rows, "shared-arms": shared_rows,
              "shared-scale-metadata": meta_rows, "stage1-posthoc-readout": stage1_rows}
    return tables, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="")
    parser.add_argument("--partial", default="", help="development only: analyse what exists and write to this directory")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--skip-v1", action="store_true", help="development only (with --partial)")
    parser.add_argument("--digits", type=int, default=2,
                        help="decimals of the percentage-point values (v1: 2); a value other than 2 needs --tag or --partial")
    args = parser.parse_args()
    global DIGITS
    DIGITS = args.digits
    if args.digits != 2 and not (args.tag or args.partial):
        raise SystemExit("--digits other than 2 writes a new versioned set: pass --tag (or --partial)")
    suffix = f"-{args.tag}" if args.tag else ""
    models = tuple(args.models.split(","))
    entries = {row["name"]: row for row in formats()}
    names = list(entries)
    cells = load_cells(models)
    rows1k = dataset("imagenet_screen_1k")[1]
    missing = [(m, f, rc) for m in models for f in names for rc in RECIPES if (m, f, rc) not in cells]
    missing += [(m, f, INTRINSIC[rc]) for m in models for f in SHARED for rc in RECIPES if (m, f, INTRINSIC[rc]) not in cells]
    missing += [(m, "fp32", "baseline") for m in models if (m, "fp32", "baseline") not in cells]
    if args.partial:
        out = Path(args.partial).resolve()
        if ROOT / "results" in out.parents or ROOT / "docs" in out.parents:
            raise SystemExit("--partial must not write under results/ or docs/")
        names = [f for f in names if all((m, f, rc) in cells for m in models for rc in RECIPES)]
        print(f"PARTIAL: {len(missing)} cells missing; {len(names)} formats complete on {models}")
        if not names:
            raise SystemExit("PARTIAL: no format has both recipes on every requested model; narrow --models")
    else:
        out = OUT
        if missing or models != MODELS or args.skip_v1:
            raise SystemExit(f"{len(missing)} cells missing (first: {missing[:5]}); the full analysis needs all models and v1")
    old = {} if args.skip_v1 else v1_cells(rows1k, models)
    tables, summary = analyse(cells, entries, models, names, old, rows1k, partial=bool(args.partial))
    for label, rows in tables.items():
        if rows:
            print(label, write(out / f"{label}{suffix}.csv", table(rows)))
    print("summary", write(out / f"summary{suffix}.json", json.dumps(summary, indent=1, sort_keys=True) + "\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
