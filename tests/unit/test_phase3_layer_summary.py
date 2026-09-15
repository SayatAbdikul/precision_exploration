import pytest

from public.analysis.phase3.layer_summary import summarize


def image(*, mse, count, events=None):
    return {"backends": {"cuda": {"diagnostics": {"layer": {
        "mse": mse, "sampled_elements": count, "population_elements": count*10,
        "fp32": {"mean": 1, "variance": 0}, "candidate": {"zero_fraction": .5, "outlier_fraction": 0},
        "quantizer_event_counts": events, "event_status": "observed" if events else "no_instrumented_quantizer_calls"}}}}}


def test_pools_by_sample_count_and_retains_missing_event_coverage():
    records = [image(mse=1, count=2, events={"overflow": 1, "sampled_stores": 2}), image(mse=3, count=6)]
    layer = summarize(records, "cuda")["layers"][0]
    assert layer["sample_mse"] == 2.5
    assert layer["images_with_observed_quantizer_calls"] == 1
    assert layer["quantizer_event_counts"] == {"overflow": 1, "sampled_stores": 2}
    assert summarize([image(mse=0, count=3)], "cuda")["layers"][0]["quantizer_event_counts"] is None


def test_different_layer_coverage_cannot_be_silently_pooled():
    a, b = image(mse=1, count=1), image(mse=2, count=1)
    b["backends"]["cuda"]["diagnostics"] = {}
    with pytest.raises(ValueError, match="coverage"):
        summarize([a, b], "cuda")
