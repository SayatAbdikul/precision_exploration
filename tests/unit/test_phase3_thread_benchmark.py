import pytest

from tools.run.phase3_thread_benchmark import summarize


def sample(backend, threads, image, seconds, output="matching"):
    return {"backend": backend, "threads": threads, "sample_sha256": image, "output_sha256": output,
            "layers_sha256": "layers-" + image, "inference_with_diagnostics_seconds": seconds,
            "native_dispatch_seconds": 1, "store_seconds": 2, "diagnostic_seconds": 3, "cpu_seconds": 4}


def test_thread_summary_compares_identical_images_and_ranks_measured_times():
    rows = [sample("cpp", 4, "a", 20), sample("cpp", 4, "b", 22),
            sample("cuda", 8, "a", 18), sample("cuda", 8, "b", 19)]
    report = summarize(rows)
    assert report[0]["backend"] == "cuda"
    assert report[0]["mean_seconds"] == 18.5
    assert report[1]["median_seconds"] == 21


def test_thread_summary_rejects_output_changes_even_when_faster():
    with pytest.raises(ValueError, match="disagree"):
        summarize([sample("cpp", 4, "a", 20), sample("cuda", 8, "a", 1, "wrong")])


def test_thread_summary_rejects_unmatched_image_subsets():
    with pytest.raises(ValueError, match="identical images"):
        summarize([sample("cpp", 4, "a", 20), sample("cuda", 8, "b", 1)])


def test_thread_summary_rejects_duplicate_images():
    with pytest.raises(ValueError, match="duplicate"):
        summarize([sample("cpp", 4, "a", 20), sample("cpp", 4, "a", 21)])
