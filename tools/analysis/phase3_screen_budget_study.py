"""Retrospective subset stability on completed screens; no new inference or D4 decision."""
import numpy as np

from tools.phase3.common import ROOT, checked, digest, read, reference, write


def study(root=ROOT):
    expected = {(m, f) for m in ("resnet18", "mobilenet_v2")
                for f in ("int4", "int5", "int6", "int8")}
    sources, data, timing_rows, sample_ids = [], {}, [], None
    for path in sorted((root / "results/summaries").glob("phase3-analysis-*.json")):
        analysis = read(path)
        key = (analysis["model"], analysis["format"])
        if analysis["scope"] != "screen" or key not in expected:
            continue
        if key in data or analysis["images"] != 1000:
            raise ValueError("duplicate or incomplete screen")
        summary = read(checked(analysis["evidence"][0], root))
        paired = read(checked(summary["paired"], root))
        paired.sort(key=lambda row: row["sample_id"])
        ids = [row["sample_id"] for row in paired]
        if len(ids) != 1000 or len(set(ids)) != 1000 or (sample_ids is not None and sample_ids != ids):
            raise ValueError("screens must have the same 1,000 unique image IDs")
        sample_ids = ids
        candidate = np.array([row["candidate_correct"] for row in paired], dtype=np.int8)
        baseline = np.array([row["fp32_correct"] for row in paired], dtype=np.int8)
        delta = candidate - baseline
        metric = analysis["statistics"]["metrics"]["top1"]
        if not np.isclose(delta.mean(), metric["delta"], atol=1e-12, rtol=0):
            raise ValueError("paired outcomes do not reproduce published delta")
        data[key] = (candidate, delta)
        timings = summary["timings_seconds"]["cuda"]
        if len(timings) != 1000:
            raise ValueError("screen timing count does not match its images")
        timing_rows.append({"model": key[0], "format": key[1],
                            "mean_recorded_seconds_per_image": float(np.mean(timings)),
                            "sum_recorded_inference_hours": float(np.sum(timings) / 3600)})
        sources.append({"analysis": reference(path, root), "screen": analysis["evidence"][0],
                        "paired": summary["paired"]})
    if set(data) != expected:
        raise ValueError("study needs all eight completed integer screens")

    rng = np.random.default_rng(20260925)
    panels = np.array([rng.permutation(1000) for _ in range(1000)])
    rows, rankings = [], []
    for size in (128, 256, 512):
        indices = panels[:, :size]
        for (model, fmt), (candidate, delta) in sorted(data.items()):
            panel_delta = delta[indices].mean(axis=1)
            rows.append({"model": model, "format": fmt, "images": size,
                         "full_1k_delta_pp": float(100 * delta.mean()),
                         "panel_delta_pp_2_5_to_97_5_percentiles":
                             (100 * np.quantile(panel_delta, [0.025, 0.975])).tolist(),
                         "fraction_panels_with_negative_delta": float((panel_delta < 0).mean())})
        for model in ("resnet18", "mobilenet_v2"):
            top1 = np.stack([data[(model, fmt)][0][indices].mean(axis=1)
                             for fmt in ("int4", "int5", "int6", "int8")], axis=1)
            rankings.append({"model": model, "images": size,
                             "fraction_strict_int4_int5_int6_int8_order":
                                 float(np.all(np.diff(top1, axis=1) > 0, axis=1).mean())})
    document = {"schema_version": "phase3-retrospective-screen-budget-1.0.0",
                "seed": 20260925, "panels": 1000, "sampling": "nested uniform subsets without replacement",
                "source_images": 1000, "sources": sources,
                "implementation": reference(__file__, root), "numpy_version": np.__version__,
                "results": rows, "ranking_stability": rankings, "historical_screen_timings": timing_rows,
                "limits": ["retrospective analysis of eight integer classifier screens only",
                           "panel percentiles are not confidence intervals for population accuracy",
                           "does not validate family ranking, detector AP, or a stopping policy",
                           "sub-1k panels cannot cover all 1,000 ImageNet classes",
                           "historical inference timings include diagnostics and possible contention; not isolated throughput",
                           "no new inference; no change to campaign, acceptance, or D4"]}
    path = root / "artifacts/phase3/reviews/screen-budget" / f"{digest(document)}.json"
    write(path, document)
    return path, document


if __name__ == "__main__":
    path, result = study()
    print(path)
    for row in result["results"]:
        if row["images"] == 256:
            print(row)
    print(result["ranking_stability"])
