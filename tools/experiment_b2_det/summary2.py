"""Versioned summaries after review 1 (protocol addendum 3); CPU, no model.

``.venv-b/bin/python -m tools.experiment_b2_det.summary2 [--tag r2]`` writes, under
``results/summaries/b2-detector-v1/``, new files beside the first versions (which stay as written):

- ``freeze-tests-r2``: every freeze decision with what the written rule prescribes, what ``default`` does, what
  ``conformant`` does, and whether ``default`` deviates from the rule;
- ``conformant-1000-r2``: the protocol-conformant recipe beside the frozen one for the nine sentinel formats
  (screen1k), the exit statement for both at INT8, and the dev128 INT8 values;
- ``effects-128-r2``, ``effects-1000-r2``: every run with its paired differences; ``conformant`` is referenced to
  ``default``; rows that are one configuration under two names say so (``same_configuration_as``);
- ``frozen-recipe-r2``: the frozen recipe with the protocol deviation stated, corrected reasons, and the
  conformant recipe.

Everything is development evidence (COCO screen list).
"""
from __future__ import annotations

import argparse

from . import summary
from .arms4 import conformant_arm
from .recipe import hardware_semantics, named
from .report import reference as first_reference
from .report import rows as effect_rows

NOTE = summary.NOTE
STATISTICS = summary.STATISTICS

# Whether a recipe carries the choice that a freeze decision is about (classifier-default switch kept, or
# contract-changing option / head exemption adopted).
CARRIES = {
    "bias correction": lambda r: r.bias_correction == "empirical",
    "unsigned codes": lambda r: r.unsigned,
    "activation MSE ranges": lambda r: r.activation_range == "mse",
    "weight MSE ranges": lambda r: r.weight_range == "mse",
    "code pass-through (max-pool, slice, upsample)": lambda r: r.passthrough != "none",
    "fused SiLU boundary": lambda r: r.boundaries == "fused_silu",
    "concatenation pass-through": lambda r: r.passthrough == "nonarith_concat",
    "head: projection constant not quantized": lambda r: not r.q_projection,
    "head: box logits not quantized": lambda r: not r.q_box_logits,
    "head: box expectation not quantized": lambda r: not r.q_dfl,
    "head: decoded boxes not quantized": lambda r: not r.q_boxes,
    "head: class logits not quantized": lambda r: not r.q_class_logits,
    "head: sigmoid scores not quantized": lambda r: not r.q_scores,
    "head: 84-channel joins not quantized": lambda r: not r.q_joins,
}


def rule_outcome(row):
    """What the written rule prescribes for one freeze decision (protocol v1 freeze_rule with addendum 1).

    Classifier-default switch: kept unless its leave-one-out arm is better on dev128 with an interval excluding
    zero.  Contract option or head exemption: adopted only if the test was met by an arm measured before the
    freeze; a tensor whose quantization was not tested stays quantized."""
    met = bool(row["test_met_on_dev128"])
    if row["kind"] == "default":
        return not met
    return met and row["decision_pair_measured"] == "before"


def freeze_tests():
    document = summary.freeze_tests()
    default, conformant = named("default"), named("conformant")
    for row in document["rows"]:
        carries = CARRIES[row["decision"]]
        rule = rule_outcome(row)
        row.update(rule_prescribes=rule, default_carries=bool(carries(default)),
                   conformant_carries=bool(carries(conformant)),
                   default_deviates_from_rule=bool(carries(default)) != rule)
        if bool(carries(conformant)) != rule:
            raise ValueError(f"conformant recipe does not follow the rule: {row['decision']}")
    document["rule"]["rule_prescribes"] = (
        "true = the written rule keeps the classifier-default switch / adopts the option or exemption. A contract "
        "option counts only if an arm measured before the freeze (20:04:01) met the test; an untested head tensor "
        "stays quantized (protocol v1 freeze_rule item 4, addendum 1). default_deviates_from_rule marks the protocol "
        "deviation of the frozen recipe (addendum 3).")
    return document


def _pair(left, right):
    return summary.difference(left, right)


def conformant_table(images=1000):
    found = summary.records(images)
    vec = lambda key: summary.vector(summary.stem(found[key], images))  # noqa: E731
    fp32 = vec(("fp32", "fp32"))
    int8_conformant = vec(("int8", "conformant")) if ("int8", "conformant") in found else None
    rows = []
    for name in summary.SENTINELS:
        arm = conformant_arm(name)
        if (name, arm) not in found or (name, "default") not in found:
            rows.append({"format": name, "conformant_arm": arm, "missing": True})
            continue
        frozen, mine = vec((name, "default")), vec((name, arm))
        row = {"format": name, "conformant_arm": arm, "configuration": found[(name, arm)]["configuration_sha256"][:12],
               "frozen": summary.cell(frozen), "conformant": summary.cell(mine),
               "conformant_minus_frozen": _pair(frozen, mine), "conformant_minus_fp32": _pair(fp32, mine),
               "frozen_minus_fp32": _pair(fp32, frozen),
               "conformant_minus_int8_conformant": (_pair(int8_conformant, mine)
                                                    if name != "int8" and int8_conformant is not None else None),
               "detections": mine["ties"]["detections"], "distinct_scores": mine["ties"]["distinct_scores"]}
        for recipe in ("maxabs", "percentile_99_9"):
            old = summary.vector(summary.sealed_stem(name, recipe, images))
            row[f"conformant_minus_v1_{recipe}"] = _pair(old, mine)
        rows.append(row)
    exits = {}
    for recipe in ("default", "conformant"):
        if ("int8", recipe) in found:
            mine = vec(("int8", recipe))
            against = _pair(fp32, mine)
            exits[recipe] = {**summary.cell(mine), "minus_fp32": against,
                             "point_estimate_within_one_point": bool(abs(against["delta"]) <= 1.0),
                             "interval_within_one_point": bool(against["interval"][0] >= -1.0
                                                               and against["interval"][1] <= 1.0)}
    dev = {}
    small = summary.records(128)
    if all(("int8", r) in small for r in ("default", "conformant")) and ("fp32", "fp32") in small:
        v = {key: summary.vector(summary.stem(small[key], 128)) for key in
             (("fp32", "fp32"), ("int8", "default"), ("int8", "conformant"))}
        dev = {"fp32": summary.cell(v[("fp32", "fp32")]), "default": summary.cell(v[("int8", "default")]),
               "conformant": summary.cell(v[("int8", "conformant")]),
               "conformant_minus_default": _pair(v[("int8", "default")], v[("int8", "conformant")]),
               "conformant_minus_fp32": _pair(v[("fp32", "fp32")], v[("int8", "conformant")])}
    return {"panel": f"screen{images}", "interpretation": NOTE, "statistics": STATISTICS, "fp32": summary.cell(fp32),
            "recipes": {"frozen": "`default` (frozen 2026-10-01 20:04:01)",
                        "conformant": "the recipe the written freeze rule prescribes (addendum 3): `default` with the "
                                      "box logits not quantized and the joins quantized (score group, signed codes); "
                                      "for bfp6 and mxfp8_e4m3 the registered arm default_fp32_box_logits, because "
                                      "the engine has no score-only join store for block formats"},
            "exit_statement_int8": exits, "dev128_int8": dev, "rows": rows,
            "decision": "open: the owner chooses which recipe the paper uses (addendum 3)"}


def reference(name):
    return "default" if name == "conformant" else first_reference(name)


def aliases(images):
    groups = {}
    for (name, recipe), record in summary.records(images).items():
        groups.setdefault((name, record["configuration_sha256"]), []).append(recipe)
    return [{"format": name, "configuration": identity[:12], "names": sorted(names)}
            for (name, identity), names in sorted(groups.items()) if len(names) > 1]


def effects(images):
    table = effect_rows(images, reference_of=reference)
    same = {}
    for group in aliases(images):
        for recipe in group["names"]:
            same[(group["format"], recipe)] = [x for x in group["names"] if x != recipe]
    for row in table:
        row["same_configuration_as"] = ";".join(same.get((row["format"], row["recipe"]), []))
    return table


def _fmt(row):
    d = row["dev128"]
    return f"{d['delta']:+.2f} [{d['interval'][0]:+.2f}, {d['interval'][1]:+.2f}] ({d['base']} -> {d['arm']})"


def recipe_statement():
    tests = {row["decision"]: row for row in freeze_tests()["rows"]}
    base = summary.recipe_statement()
    differences = []
    for entry in summary.DIFFERENCES:
        entry = dict(entry)
        if entry["switch"] == "head: decoded boxes and 84-channel joins":
            boxes, joins = tests["head: decoded boxes not quantized"], tests["head: 84-channel joins not quantized"]
            entry["why"] = (f"decoded boxes: test met on dev128, {_fmt(boxes)}. Joins: PROTOCOL DEVIATION. No arm tested "
                            "the join exemption before the freeze, so the written rule keeps the joins quantized; the "
                            f"exemption was adopted with the box exemption. Measured afterwards: {_fmt(joins)}, test "
                            "not met.")
        elif entry["switch"].startswith("head: projection constant"):
            box = tests["head: box logits not quantized"]
            cls = tests["head: class logits not quantized"]
            entry["why"] = ("projection constant, box expectation and sigmoid scores: putting each back did not meet "
                            f"the test on dev128. Class logits: exemption {_fmt(cls)}, test not met. Box logits: "
                            f"PROTOCOL DEVIATION. Exemption {_fmt(box)} met the test before the freeze (bootstrap "
                            "19:57:51, frozen.py 20:04:01); the first agent had classed the logit exemptions as "
                            "diagnostics and kept the box logits quantized.")
        differences.append(entry)
    conformant = named("conformant")
    return {**base, "differences_from_classifier_default": differences,
            "protocol_deviation": {
                "where": ["box logits quantized although their exemption met the freeze test on dev128",
                          "84-channel joins not quantized although no test admitted the exemption"],
                "source": "public/experiments/configs/breadth-study/b2-detector-protocol-v1-addendum-3.json",
                "status": "the frozen recipe is unchanged and every number reported for it stands; the owner decides "
                          "between `default` and `conformant`"},
            "conformant": {
                "name": "conformant", "switches": conformant.as_dict(),
                "difference_from_default": {"q_box_logits": False, "q_joins": True},
                "hardware_semantics": hardware_semantics(conformant),
                "notes": ["box logits leave the accelerator as wide values (no k-bit store of the 64-channel "
                          "box-distribution logits); the softmax and expectation run on them and are requantized",
                          "the joins re-store only the score group (boxes are not a k-bit store) with the signed "
                          "codebook, as engine.analyze defines a join (never non-negative); the scores are thus "
                          "unsigned k-bit codes after the sigmoid and signed k-bit codes after each join",
                          "bfp6 and mxfp8_e4m3: default_fp32_box_logits (no join store), because block formats "
                          "store a join only as a whole 84-channel tensor"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="r2")
    parser.add_argument("--only", nargs="*", default=None,
                        choices=("freeze-tests", "conformant-1000", "effects-128", "effects-1000", "frozen-recipe"))
    args = parser.parse_args()
    wanted = lambda key: args.only is None or key in args.only  # noqa: E731
    if wanted("freeze-tests"):
        document = freeze_tests()
        summary.write(f"freeze-tests-{args.tag}", document, document["rows"])
    if wanted("conformant-1000"):
        document = conformant_table()
        summary.write(f"conformant-1000-{args.tag}", document, document["rows"])
    for images in (128, 1000):
        if wanted(f"effects-{images}"):
            table = effects(images)
            summary.write(f"effects-{images}-{args.tag}",
                          {"panel": "dev128" if images == 128 else "screen1k", "interpretation": NOTE,
                           "statistics": STATISTICS, "same_configuration": aliases(images), "rows": table}, table)
    if wanted("frozen-recipe"):
        summary.write(f"frozen-recipe-{args.tag}", recipe_statement())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
