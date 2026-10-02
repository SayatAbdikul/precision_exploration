"""Markdown tables of the B2 format matrix, generated from the summary tables (numbers are never typed by hand).

Usage: .venv/bin/python -m tools.analysis.b2_matrix_report [--source DIR] [--tag NAME] > tables.md
       .venv/bin/python -m tools.analysis.b2_matrix_report --assemble --tag d4 [--out FILE]
The document is assembled from the 4-decimal set (``b2_matrix --digits 4 --tag d4``) since revision 3, so that each
displayed 1-decimal value is the rounding of the exact value.
The first form prints sections ``<!-- name -->``.  ``--assemble`` writes docs/analysis/b2-matrix-2026-10-01.md
from tools/analysis/b2_matrix_doc_template.md by replacing every ``{{T:name}}`` with the table of that name.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tools.experiment_b.common import ROOT

SOURCE = ROOT / "results/summaries/b2-matrix-v1"
SHORT = {"resnet18": "R18", "mobilenet_v2": "MBv2", "mobilenet_v3_large": "MBv3-L"}
RECIPES = ("default", "minimal")
V1 = ("v1_maxabs", "v1_percentile_99_9")
INTRINSIC = {"default": "cum5_act_maxabs", "minimal": "cum1_fused"}
CLASS = {"within_one_point": "within", "separated_below": "BELOW", "separated_above": "ABOVE", "not_resolved": "unresolved",
         "anchor": "anchor", "no_anchor": ""}
FAMILIES = ["integer", "fixed_point", "float", "posit", "logarithmic", "bfp", "mx_float", "codebook", "ternary", "binary"]


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def md(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    lines += ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def f1(value, digits=1):
    return "" if value in ("", None) else f"{float(value):.{digits}f}"


def separated(low, high):
    """Whether a pointwise interval excludes zero, read from the table value (render from a --digits 4 set:
    a 2-decimal bound of 0.0 can hide an interval that excludes zero)."""
    low, high = float(low), float(high)
    return "search better" if low > 0 else ("search worse" if high < 0 else "no")


def span(row, key, digits=1):
    if row.get(key) in ("", None):
        return ""
    return f"{float(row[key]):+.{digits}f} [{float(row[key + '_low']):+.{digits}f}, {float(row[key + '_high']):+.{digits}f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--tag", default="")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "docs/analysis/b2-matrix-2026-10-01.md"))
    args = parser.parse_args()
    source, suffix = Path(args.source), (f"-{args.tag}" if args.tag else "")
    cells = read(source / f"cells{suffix}.csv")
    summary = json.loads((source / f"summary{suffix}.json").read_text())
    models = summary["models"]
    cell = {(row["model"], row["format"], row["recipe"]): row for row in cells}
    bits = {row["format"]: int(row["bits"]) for row in cells}
    family = {row["format"]: row["family"] for row in cells}
    names = sorted((f for f in summary["formats"]), key=lambda f: (-bits[f], FAMILIES.index(family[f]), f))
    shared = [f for f in names if (models[0], f, INTRINSIC["default"]) in cell]
    out, tables = [], {}

    def section(name, text):
        out.append(f"<!-- {name} -->\n{text}\n")
        tables[name] = text

    # matrix of expected-credit top-1
    header = ["format", "bits"] + [f"{SHORT[m]} {rc}" for m in models for rc in RECIPES]
    rows = [["fp32", 32] + [f1(cell[(m, "fp32", "baseline")]["top1_expected"]) if rc == "default" else "" for m in models for rc in RECIPES]]
    for name in names:
        rows.append([name, bits[name]] + [f1(cell[(m, name, rc)]["top1_expected"]) for m in models for rc in RECIPES])
        if name in shared:
            rows.append([f"{name}, max-abs activation blocks", bits[name]]
                        + [f1(cell[(m, name, INTRINSIC[rc])]["top1_expected"]) for m in models for rc in RECIPES])
    section("matrix", md(header, rows))

    # paired difference against FP32 with intervals
    for recipe in RECIPES:
        rows = [[name] + [span(cell[(m, name, recipe)], "d_fp32") for m in models] for name in names]
        section(f"vs-fp32-{recipe}", md(["format"] + [SHORT[m] for m in models], rows))

    # ties and readout sensitivity
    header = ["format"] + [f"{SHORT[m]} {rc}" for m in models for rc in RECIPES]
    rows = [[name] + [f"{cell[(m, name, rc)]['tied_top1_images']} ({float(cell[(m, name, rc)]['top1_lowest_index']) - float(cell[(m, name, rc)]['top1_expected']):+.1f})"
                      for m in models for rc in RECIPES] for name in names]
    section("ties", md(header, rows))

    # orderings
    order = read(source / f"ordering{suffix}.csv")
    lines = []
    for model in models:
        for recipe in RECIPES + V1:
            ranked = sorted((row for row in order if row["model"] == model and row["recipe"] == recipe), key=lambda r: int(r["rank"]))
            if not ranked:
                continue
            live = [row for row in ranked if float(row["top1"]) >= 1.0]
            dead = [row["format"] for row in ranked if float(row["top1"]) < 1.0]
            text = ""
            for row in live:
                text += f"{row['format']} {float(row['top1']):.1f}"
                if row is not live[-1]:
                    text += " **>** " if row["separated_from_next"] == "True" else " ~ "
            lines.append(f"- **{SHORT[model]}, {recipe}** ({len(ranked)} formats): {text}"
                         + (f"; below 1 percent: {', '.join(dead)}" if dead else ""))
    section("orderings", "\n".join(lines))
    changes = read(source / f"ordering-changes{suffix}.csv")
    section("ordering-changes", md(["model", "from", "to", "formats in common", "Spearman", "separated pairs reversed",
                                    "of which both formats below 1 percent in one ordering"],
                                   [[SHORT[r["model"]], r["from"], r["to"], r["formats_in_common"], f1(r["spearman"], 3),
                                     r["separated_pairs_reversed"], r["of_which_both_below_1_percent_in_one_ordering"]]
                                    for r in changes]))
    section("ordering-reversals", "\n".join(
        f"- {SHORT[r['model']]}, {r['from']} -> {r['to']}: {r['pairs_first_above_second_in_from'] or 'none'}"
        + (f" (no v1 1k result: {r['missing_in_from']})" if r["missing_in_from"] else "") for r in changes))
    section("readout-sensitivity", md(["model", "recipe", "Spearman expected vs lowest index", "largest gap (points)", "format", "formats with gap > 1"],
                                      [[SHORT[r["model"]], r["recipe"], f1(r["spearman_expected_vs_lowest_index"], 4), f1(r["largest_abs_gap_points"], 2),
                                        r["format_of_largest_gap"], r["formats_with_gap_above_1_point"]]
                                       for r in summary["readout"]["ordering_sensitivity"]]))

    # per bit width
    width = {(row["model"], row["format"], row["recipe"]): row for row in read(source / f"bit-width{suffix}.csv")}
    for recipe in RECIPES:
        rows = []
        for name in names:
            if name == "int8":
                continue
            rows.append([name, bits[name]] + [f"{span(width[(m, name, recipe)], 'vs_int8')} {CLASS[width[(m, name, recipe)]['class_int8']]}"
                                              for m in models])
        section(f"vs-int8-{recipe}", md(["format", "bits"] + [SHORT[m] for m in models], rows))
        rows = []
        for name in names:
            if width[(models[0], name, recipe)]["class_same_width_integer"] in ("anchor", "no_anchor"):
                continue
            rows.append([name, bits[name]] + [f"{span(width[(m, name, recipe)], 'vs_same_width_integer')} "
                                              f"{CLASS[width[(m, name, recipe)]['class_same_width_integer']]}" for m in models])
        section(f"vs-same-width-{recipe}", md(["format", "bits"] + [SHORT[m] for m in models], rows))
    # intrinsic arm of the shared-exponent formats against INT8 of the base recipe (review 1, finding 2): the
    # same rule (b) classes, computed from the cells table (d_int8 of an intrinsic cell is against INT8 of the
    # recipe it modifies: cum5_act_maxabs -> default, cum1_fused -> minimal)
    from tools.analysis.b2_matrix import classify
    rows = []
    for name in shared:
        for recipe in RECIPES:
            entries = []
            for m in models:
                row = cell[(m, name, INTRINSIC[recipe])]
                label = classify(float(row["d_int8"]), float(row["d_int8_low"]), float(row["d_int8_high"]))
                entries.append(f"{span(row, 'd_int8')} {CLASS[label]}")
            rows.append([f"{name}, max-abs activation blocks", bits[name], f"{INTRINSIC[recipe]} (vs INT8 {recipe})"] + entries)
    section("vs-int8-intrinsic", md(["format", "bits", "recipe"] + [SHORT[m] for m in models], rows))
    grouped = {}
    for (model, name, recipe), row in width.items():
        if name != "int8":
            grouped.setdefault((recipe, bits[name]), {}).setdefault(row["class_int8"], []).append(name)
    rows, letter = [], {"within_one_point": "w", "separated_below": "B", "separated_above": "A", "not_resolved": "u"}
    for (recipe, width_bits), classes in sorted(grouped.items(), key=lambda item: (item[0][0], -item[0][1])):
        own = [f for f in names if bits[f] == width_bits and f != "int8"]

        def all_models(label):
            return [f for f in own if classes.get(label, []).count(f) == len(models)]
        within, below, above = all_models("within_one_point"), all_models("separated_below"), all_models("separated_above")
        mixed = [f"{f} ({'/'.join(letter[width[(m, f, recipe)]['class_int8']] for m in models)})"
                 for f in own if f not in within + below + above]
        rows.append([recipe, width_bits, ", ".join(within) or "-", ", ".join(below) or "-",
                     ", ".join([f"{f} (A/A/A)" for f in above] + mixed) or "-"])
    section("bit-width-classes", md(["recipe", "bits", "within one point of INT8 on all models", "separated below on all models",
                                     "other (class per model: w within, B below, A above, u unresolved)"], rows))

    # recipe rule
    pairs = {(row["model"], row["format"]): row for row in read(source / f"recipe-pairs{suffix}.csv")}
    rows = []
    for name in names:
        rows.append([name, bits[name]] + [f"{float(pairs[(m, name)]['default_minus_minimal']):+.1f} "
                                          f"[{float(pairs[(m, name)]['low']):+.1f}, {float(pairs[(m, name)]['high']):+.1f}]"
                                          + ("" if pairs[(m, name)]["eligible"] == "True" else " n/e") for m in models])
    section("recipe-pairs", md(["format", "bits"] + [f"{SHORT[m]} default - minimal" for m in models], rows))
    section("recipe-rules", md(["rule", "eligible cells", "cells where the pick is separated-worse", "mean regret (points)"],
                               [[s["rule"], s["eligible_cells"], s["cells_significantly_worse"], s["mean_regret_points"]]
                                for s in summary["recipe_rule"]["scores"]]))
    low = summary["recipe_rule"]["sensitivity_lowest_index_readout"]
    section("recipe-rules-lowest-index", md(["rule", "eligible cells", "cells where the pick is separated-worse", "mean regret (points)"],
                                            [[s["rule"], s["eligible_cells"], s["cells_significantly_worse"], s["mean_regret_points"]]
                                             for s in low["scores"]]))
    section("recipe-rule-lost", "\n".join(f"- {s['rule']}: {s['cells_lost'] or 'none'}" for s in summary["recipe_rule"]["scores"]))

    # proposal
    proposal = read(source / f"proposal{suffix}.csv")
    section("proposal", md(["role", "format", "family", "bits", "recipe"] + [SHORT[m] for m in models] + ["mean drop", "worst drop", "floor"],
                           [[r["role"], r["format"], r["family"], r["bits"], r["recipe"]] + [f1(r[f"top1_{m}"]) for m in models]
                            + [f1(r["mean_drop_vs_fp32"]), f1(r["worst_drop_vs_fp32"]), "pass" if r["passes_floor"] == "True" else "FAIL"]
                            for r in proposal]))
    section("candidates", md(["family", "bits", "best format", "recipe"] + [SHORT[m] for m in models] + ["mean drop", "worst drop", "floor"],
                             [[c["family"], c["bits"], c["format"], c["recipe"]] + [f1(v) for v in c["top1"]]
                              + [f1(c["mean_drop"]), f1(c["worst_drop"]), "pass" if c["passes_floor"] else "FAIL"]
                              for c in summary["proposal"]["candidates_per_family_and_bits"]]))

    # shared exponent
    arms = read(source / f"shared-arms{suffix}.csv") if (source / f"shared-arms{suffix}.csv").exists() else []
    section("shared-arms", md(["model", "format", "recipe", "searched", "max-abs", "searched - max-abs", "interval excludes 0",
                               "blocks clipped (stored)"],
                              [[SHORT[r["model"]], r["format"], r["recipe"], f1(r["searched_activation_blocks"]),
                                f1(r["intrinsic_activation_blocks"]),
                                f"{float(r['searched_minus_intrinsic']):+.1f} [{float(r['low']):+.1f}, {float(r['high']):+.1f}]",
                                separated(r["low"], r["high"]),
                                f1(r["median_fraction_blocks_clipped_stored"], 3)] for r in arms]))
    meta = read(source / f"shared-scale-metadata{suffix}.csv")
    section("shared-metadata", md(["model", "format", "element bits", "weights", "stored activations", "convolution patches",
                                   "effective weight bits", "effective stored-activation bits"],
                                  [[SHORT[r["model"]], r["format"], r["element_bits"], f1(r["weights_scale_bits_per_element"], 3),
                                    f1(r["stored_activations_per_image_scale_bits_per_element"], 3),
                                    f1(r["convolution_patches_per_image_scale_bits_per_element"], 3),
                                    f1(r["effective_weight_bits"], 2), f1(r["effective_stored_activation_bits"], 2)] for r in meta]))

    # stage 1 and occupancy
    stage1 = read(source / f"stage1-posthoc-readout{suffix}.csv")
    section("stage1", md(["model", "format", "recipe", "sealed topk", "expected credit", "lowest index", "tied images"],
                         [[SHORT[r["model"]], r["format"], r["recipe"], f1(r["sealed_topk_top1"]), f1(r["expected_top1"], 2),
                           f1(r["lowest_index_top1"]), r["tied_top1_images"]] for r in stage1]))
    for recipe in RECIPES:
        rows = []
        for name in names:
            rows.append([name, bits[name]] + [f1(cell[(m, name, recipe)]["median_sqnr_db"]) for m in models]
                        + [f1(cell[(m, name, recipe)]["median_entropy_bits"], 2) for m in models]
                        + [f1(cell[(m, name, recipe)]["median_levels_used_fraction"], 2) for m in models]
                        + [f1(cell[(m, name, recipe)]["max_fraction_at_extremes"], 2) for m in models])
        section(f"occupancy-{recipe}", md(["format", "bits"] + [f"SQNR dB {SHORT[m]}" for m in models]
                                          + [f"entropy {SHORT[m]}" for m in models] + [f"levels used {SHORT[m]}" for m in models]
                                          + [f"max at extremes {SHORT[m]}" for m in models], rows))
    if not args.assemble:
        print("\n".join(out))
        return 0
    import re
    template = (ROOT / "tools/analysis/b2_matrix_doc_template.md").read_text()
    unknown = sorted(set(re.findall(r"\{\{T:([a-z0-9-]+)\}\}", template)) - set(tables))
    if unknown or re.search(r"\{\{[A-Z]+\}\}", template):
        raise SystemExit(f"template has unknown tables {unknown} or unfilled placeholders")
    text = re.sub(r"\{\{T:([a-z0-9-]+)\}\}", lambda match: tables[match.group(1)], template)
    Path(args.out).write_text(text)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
