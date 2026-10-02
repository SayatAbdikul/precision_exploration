"""Unit tests of the B2 matrix analysis rules (tools/analysis/b2_matrix.py); no evidence files are read."""
import numpy as np
import pytest

from tools.analysis import b2_matrix as analysis
from tools.analysis.b2_ties import paired

MODELS = ("m1", "m2", "m3")
N = 400


def vector(percent, shift=0):
    """A binary credit vector with the given accuracy; ``shift`` rotates which images are correct."""
    values = np.zeros(N)
    values[:round(percent * N / 100)] = 1.0
    return np.roll(values, shift)


def test_fast_interval_equals_the_project_bootstrap():
    rng = np.random.default_rng(0)
    for _ in range(3):
        left, right = rng.random(1000).round(1), rng.integers(0, 2, 1000).astype(float)
        d, low, high = analysis.interval(left, right)
        reference = paired(left, right)
        assert d == pytest.approx(reference["difference_pp"], abs=1e-9)
        assert [low, high] == pytest.approx(reference["pointwise_95_interval_pp"], abs=1e-9)
    with pytest.raises(ValueError):
        analysis.interval(np.zeros(5), np.zeros(6))


def test_classification_against_an_anchor():
    assert analysis.classify(-1.0, -3.0, 0.5) == "within_one_point"
    assert analysis.classify(0.4, 0.1, 0.9) == "within_one_point"
    assert analysis.classify(-1.2, -2.0, -0.1) == "separated_below"
    assert analysis.classify(-1.2, -2.0, 0.1) == "not_resolved"
    assert analysis.classify(2.0, 0.5, 3.0) == "separated_above"
    assert analysis.classify(2.0, -0.5, 3.0) == "not_resolved"


def test_ordering_and_reversed_pairs():
    first = analysis.ordering({"a": vector(80), "b": vector(60), "c": vector(60, shift=5)})
    second = analysis.ordering({"a": vector(50), "b": vector(70), "c": vector(70, shift=5)})
    assert first[0] == ["a", "b", "c"] and second[0] == ["b", "c", "a"]
    assert first[1][("a", "b")][3] and not first[1][("b", "c")][3]
    assert sorted(analysis.reversed_pairs(first, second)) == [("a", "b"), ("a", "c")]
    assert analysis.reversed_pairs(first, first) == []


def pair_row(model, name, family, bits, default, minimal, eligible=True):
    d, low, high = analysis.interval(vector(default), vector(minimal))
    return {"model": model, "format": name, "family": family, "bits": bits, "default": default, "minimal": minimal,
            "default_minus_minimal": d, "low": low, "high": high, "eligible": eligible}


def synthetic_pairs(posit_models=MODELS):
    rows = []
    for model in MODELS:
        rows += [pair_row(model, "int8", "integer", 8, 70, 69.75), pair_row(model, "int6", "integer", 6, 60, 66),
                 pair_row(model, "int4", "integer", 4, 1, 30), pair_row(model, "tern", "binary_ternary", 2, 0, 0, eligible=False),
                 pair_row(model, "posit6", "posit", 6, 60 if model in posit_models else 40, 40),
                 pair_row(model, "posit4", "posit", 4, 0, 20)]
    return rows


def test_rule_scoring_picks_the_threshold_with_the_fewest_separated_losses():
    scores, best = analysis.score_rules(synthetic_pairs())
    by_rule = {row["rule"]: row for row in scores}
    assert all(row["eligible_cells"] == 15 for row in scores)          # the collapsed format is not scored
    assert by_rule["R_default"]["cells_significantly_worse"] == 9      # int6, int4, posit4 on three models
    assert by_rule["R_minimal"]["cells_significantly_worse"] == 3      # posit6 on three models
    assert by_rule["R_bits(6)"]["cells_significantly_worse"] == 3      # int6
    assert by_rule["R_bits(7)"]["cells_significantly_worse"] == 3      # posit6
    assert by_rule["R_bits(5)"] == {**by_rule["R_bits(6)"], "rule": "R_bits(5)"}
    assert by_rule["R_bits(6)"]["mean_regret_points"] < by_rule["R_bits(7)"]["mean_regret_points"]
    assert best == "R_bits(5)"                                         # equal score: the first candidate is kept
    result, picks = analysis.family_exceptions(synthetic_pairs(), best, MODELS)
    assert [(e["family"], e["formats"]) for e in result["adopted"]] == [("integer", ["int6"])] and picks["int6"] == "minimal"


def test_family_exception_needs_every_model():
    rows = synthetic_pairs()
    result, picks = analysis.family_exceptions(rows, "R_bits(7)", MODELS)
    assert [(e["family"], e["formats"], e["recipe"]) for e in result["adopted"]] == [("posit", ["posit6"], {"posit6": "default"})]
    assert picks == {"int8": "default", "int6": "minimal", "int4": "minimal", "tern": "minimal", "posit6": "default",
                     "posit4": "minimal"}
    rows = synthetic_pairs(posit_models=("m1", "m2"))
    result, picks = analysis.family_exceptions(rows, "R_bits(7)", MODELS)
    assert result["adopted"] == [] and picks["posit6"] == "minimal"
    observed = {row["format"]: row["models_where_the_other_recipe_is_separated_better"] for row in result["observations"]}
    assert observed == {"posit6": ["m1", "m2"]}


def entries(spec):
    return {name: {"family": family, "bits": bits} for name, (family, bits) in spec.items()}


def credits(table, recipe="default"):
    credit = {(m, "fp32", "baseline"): vector(75) for m in MODELS}
    for name, values in table.items():
        used = analysis.INTRINSIC[recipe] if name in analysis.SHARED else recipe
        for model, value in zip(MODELS, values):
            credit[(model, name, used)] = vector(value)
    return credit


def test_proposal_floor_best_per_family_and_stress_configurations():
    spec = {"int8": ("integer", 8), "int6": ("integer", 6), "int4": ("integer", 4), "fa8": ("float", 8), "fb8": ("float", 8),
            "posit8": ("posit", 8), "bfp6": ("bfp", 6), "mxfp6_e3m2": ("mx_float", 6), "q1_6": ("fixed_point", 8)}
    table = {"int8": (74, 74, 74), "int6": (72, 71, 64), "int4": (30, 30, 30), "fa8": (73, 73, 73), "fb8": (74, 74, 73),
             "posit8": (74.5, 74.5, 74.5), "bfp6": (71, 71, 71), "mxfp6_e3m2": (72, 72, 72), "q1_6": (74, 74, 74)}
    picks = {name: "default" for name in spec}
    result = analysis.propose(credits(table), entries(spec), MODELS, list(spec), picks, excluded=("q1_6",))
    selected = {c["format"]: c for c in result["selected"]}
    # int8 is an anchor, q1_6 excluded, fb8 beats fa8, mxfp6 beats bfp6, int6 fails the worst-model floor (11 points)
    assert sorted(selected) == ["fb8", "mxfp6_e3m2", "posit8"]
    assert selected["mxfp6_e3m2"]["recipe"] == "cum5_act_maxabs"
    assert not any(c["passes_floor"] for c in result["candidates_per_family_and_bits"] if c["format"] in ("int6", "int4"))
    # fewer than six pass: one stress configuration per family, the highest failing width below the passing ones
    assert [(c["format"], c["flag"]) for c in result["stress"]] == [("int6", "stress_configuration_below_the_floor")]
    assert result["passing_candidates"] == 3


def test_proposal_is_capped_with_family_extremes_first():
    spec, table = {"int8": ("integer", 8)}, {"int8": (74, 74, 74)}
    for index, bits in enumerate(range(12, 0, -1)):          # twelve passing float widths, better at more bits
        spec[f"f{bits}"], table[f"f{bits}"] = ("float", bits), (74 - 0.2 * index,) * 3
    spec["p8"], table["p8"] = ("posit", 8), (73.9,) * 3
    picks = {name: "default" for name in spec}
    result = analysis.propose(credits(table), entries(spec), MODELS, list(spec), picks)
    names = [c["format"] for c in result["selected"]]
    assert len(names) == analysis.PROPOSAL_SIZE and result["passing_candidates"] == 13 and result["stress"] == []
    assert {"f12", "f1", "p8"} <= set(names)                 # lowest and highest passing width of every family
    assert "f2" not in names and "f3" not in names           # the fill is by ascending mean drop


def test_summary_tables_do_not_overwrite(tmp_path):
    target = tmp_path / "a.csv"
    assert analysis.write(target, "x\n") == "written"
    assert analysis.write(target, "x\n") == "unchanged"
    with pytest.raises(SystemExit):
        analysis.write(target, "y\n")
    assert analysis.table([{"a": 1}, {"a": 2, "b": 3}]) == "a,b\n1,\n2,3\n"
