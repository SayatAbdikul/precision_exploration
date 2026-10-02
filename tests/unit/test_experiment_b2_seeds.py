"""Lane Q5: calibration subsets reproduce the v1 range statistics, and the subset machinery is sound."""
import numpy as np
import pytest

from tools.experiment_b.common import dataset, digest, unseal
from tools.experiment_b2 import data
from tools.experiment_b2_seeds import cells, subsets

MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large")


def test_seed_subsets_partition_the_list():
    spec = subsets.definitions()
    seeds = [set(spec[f"S{k}"]["range_batches"]) for k in range(subsets.SEEDS)]
    assert all(len(s) == subsets.SEED_BATCHES for s in seeds)
    assert set().union(*seeds) == set(range(subsets.BATCHES))
    assert sum(len(s) for s in seeds) == subsets.BATCHES  # pairwise disjoint
    for k in range(subsets.SEEDS):
        assert spec[f"S{k}"]["bias_batches"] == spec[f"S{k}"]["range_batches"]
        assert spec[f"B{k}"]["range_batches"] == list(range(subsets.BATCHES))
        assert spec[f"B{k}"]["bias_batches"] == spec[f"S{k}"]["range_batches"]
    for k in range(subsets.LADDER_REPLICATES):
        small, mid = set(spec[f"L32_{k}"]["range_batches"]), set(spec[f"L128_{k}"]["range_batches"])
        assert len(small) == 4 and len(mid) == 16 and small < mid < seeds[k]
    halves = [set(spec[f"H{j}"]["range_batches"]) for j in (0, 1)]
    assert len(halves[0]) == len(halves[1]) == 125 and not halves[0] & halves[1]
    assert halves[0] | halves[1] == set(range(subsets.BATCHES))
    assert subsets.definitions() == spec  # frozen seeds: deterministic


def test_bias_rows_rule():
    _, rows, _ = dataset(subsets.LIST)
    assert subsets.bias_rows(rows, range(subsets.BATCHES)) == rows[:256]  # the original rule
    spec = subsets.definitions()
    chosen = subsets.bias_rows(rows, spec["S3"]["bias_batches"])
    members = subsets.subset_rows(rows, spec["S3"]["range_batches"])
    assert len(members) == 400 and len(chosen) == 256 and chosen == members[:256]
    assert [r["sha256"] for r in chosen] == sorted(r["sha256"] for r in chosen)
    assert len(subsets.bias_rows(rows, spec["L32_1"]["bias_batches"])) == 32
    full = subsets.describe("FULL", rows)
    assert full["bias_rows_sha256"] == digest(rows[:256]) and full["range_images"] == 2000


@pytest.mark.parametrize("model", MODELS)
def test_full_set_reproduces_v1_ranges_bit_for_bit(model):
    arrays, maxima, identity = data.v1_calibration(model)
    observations = subsets.Observations(model)
    mine, mine_maxima, mine_identity = observations.calibration("FULL")
    assert mine_identity == identity
    assert set(mine) == set(arrays) and mine_maxima == maxima
    for key in arrays:
        assert mine[key].dtype == arrays[key].dtype and np.array_equal(mine[key], arrays[key])
        assert mine[key].tobytes() == arrays[key].tobytes()
    # The five seed subsets split every node's samples exactly and their maxima recombine to the full maxima.
    parts = [observations.calibration(f"S{k}") for k in range(subsets.SEEDS)]
    for key in arrays:
        assert sum(p[0][key].size for p in parts) == arrays[key].size
        assert max(p[1][key] for p in parts) == maxima[key]
        assert all(p[1][key] <= maxima[key] for p in parts)
    assert all(p[2]["b2_seeds_subset"]["name"] == f"S{k}" for k, p in enumerate(parts))
    assert parts[0][2]["identity"] == identity["identity"]


def test_compact_configuration_keeps_activation_scales_by_value():
    configuration = {"format": "int8", "calibration": {"identity": "x"},
                     "scales": {"activation_scales": {"a": 0.5}, "weight_scales": {"c": [1.0, 2.0]},
                                "bias_correction": {"c": {"max_abs_correction": 0.1}}, "activation_search": {"a": {}}}}
    compact = cells.compact_configuration(configuration)
    assert compact["scales"]["activation_scales"] == {"a": 0.5}
    assert "weight_scales" not in compact["scales"] and compact["scales"]["weight_scales_sha256"] == digest({"c": [1.0, 2.0]})
    assert compact["scales"]["sha256"] == digest(configuration["scales"])
    assert compact["calibration"] == {"identity": "x"}


def test_arm_jobs_match_protocol_counts():
    counts = {arm: len(cells.arm_jobs(arm)) * len(MODELS) for arm in cells.ARMS}
    assert counts == {"seed_default": 180, "seed_minimal": 90, "ladder": 96, "bias_only": 60}
    assert ("mxfp8_e4m3", "cum5_act_maxabs", "S0") in cells.arm_jobs("seed_default")
    addendum = {arm: len(cells.arm_jobs(arm)) * len(cells.ADDENDUM_MODELS) for arm in cells.ADDENDUM_ARMS}
    assert addendum == {"addendum_range_only": 18, "addendum_bias_only": 18}
    spec = subsets.definitions()
    assert spec["R1"]["range_batches"] == spec["S1"]["range_batches"]
    assert spec["R1"]["bias_batches"] == list(range(subsets.BATCHES))  # original bias images


def test_redirected_cells_reproduce_matrix_records():
    """GPU evidence written by ``tools.run.experiment_b2_seeds reproduce`` (skipped until it exists)."""
    found = sorted((cells.BASE / "reproduction").glob("*.json")) if (cells.BASE / "reproduction").exists() else []
    if len(found) < len(cells.REPRODUCTION):
        pytest.skip("reproduction evidence not yet written")
    for path in found:
        record = unseal(path)
        assert record["identity_reproduced"] and record["all_readout_arrays_equal"] and record["passed"], path
        assert record["configuration_sha256"] == record["matrix_configuration_sha256"]


def _transform(model):
    from torchvision import models
    from public.workloads.models.torchvision_eval import MODEL_SPECS
    _, weight_class, weight_name = MODEL_SPECS[model]
    return getattr(getattr(models, weight_class), weight_name).transforms()


@pytest.mark.parametrize("model,subset", [("resnet18", "S0"), ("mobilenet_v3_large", "L32_1")])
def test_subset_bias_inputs_match_cache_and_jpeg(model, subset):
    """Bias-correction inputs of a subset: the right rows, cache rows bit-equal to the cache, decoded rows bit-equal
    to a one-image decode of the JPEG (whose file hash ``image_batch`` checks).  Needs the existing 256-image cache
    entry, which is keyed by the torch build: run with ``CUDA_VISIBLE_DEVICES= .venv-b/bin/python -m pytest``."""
    from tools.experiment_b.classifier import image_batch
    transform = _transform(model)
    try:
        cached, cached_rows = data.cached_inputs(subsets.LIST, subsets.BIAS_IMAGES, transform, build=False)
    except FileNotFoundError:
        pytest.skip("256-image calibration cache entry not available for this torch build (use .venv-b)")
    context = cells.Context(model, "cpu")
    out, wanted = context.bias_inputs(subset, transform, data.cached_inputs)
    _, rows, payload = dataset(subsets.LIST)
    spec = subsets.definitions()[subset]
    assert wanted == subsets.bias_rows(rows, spec["bias_batches"])
    assert out.dtype == np.float32 and out.shape == (len(wanted),) + tuple(cached.shape[1:])
    where = {row["sha256"]: i for i, row in enumerate(cached_rows)}
    overlap = [i for i, row in enumerate(wanted) if row["sha256"] in where]
    decoded = [i for i, row in enumerate(wanted) if row["sha256"] not in where]
    assert context.bias_info == {"images": len(wanted), "from_cache": len(overlap), "decoded": len(decoded)}
    assert decoded  # both paths are exercised
    for i in overlap:
        assert np.array_equal(out[i], np.asarray(cached[where[wanted[i]["sha256"]]]))
    for i in sorted(set(decoded[:6] + decoded[-6:])):
        assert np.array_equal(out[i], image_batch([wanted[i]], payload, transform, "cpu").numpy()[0])


def test_exact_signs_and_credit():
    from fractions import Fraction
    from tools.experiment_b2_seeds import checks
    assert checks.exact_signs_consistent([Fraction(1, 3), Fraction(2)]) is True
    assert checks.exact_signs_consistent([Fraction(1, 3), Fraction(0)]) is False  # protocol zero rule
    assert checks.exact_signs_consistent([Fraction(-1), Fraction(1, 7)]) is False
    # 1/3 + 1/3 + 1/3 - 1 is exactly zero in rational arithmetic (it is not in float)
    arrays = {"greater": np.array([0, 0, 0, 1]), "equal": np.array([3, 3, 3, 1]), "equal_lower": np.array([0, 1, 2, 0])}
    assert checks.exact_top1(arrays, "expected") - 1 == 0
    assert checks.exact_top1(arrays, "lowest_index") == 1
