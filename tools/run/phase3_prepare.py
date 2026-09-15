"""Freeze Phase 3 inputs and expand calibration from verified observations.

This does not grant accumulator acceptance or submit full screening jobs.
Individual completed calibration artifacts are reused by content and context.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

from public.inference.conformance_job import source_identity
from public.quantization.calibration.artifact import create, validate, frozen_context
from tools.phase3.common import ROOT, MODELS, read, write, reference, checked, digest, campaign

PLAN = "public/experiments/configs/experiment_a/phase3-screen-v1.json"


def freeze(root=ROOT):
    if (root / PLAN).exists():
        return campaign(root)
    final = root / "results/summaries/phase2-final-verification.json"
    evidence = read(final)
    if evidence["status"] != "complete" or evidence["remaining_gates"] or evidence["source_sha256"] != source_identity():
        raise ValueError("Phase 3 requires a verified, completed Phase 2 starting point")
    snapshot = root / "artifacts/phase3/phase2-freeze" / reference(final, root)["sha256"]
    snapshot.mkdir(parents=True, exist_ok=True)
    # Preserve the source and curated records before Phase 3 instrumentation.
    for folder in ("public/inference", "public/quantization", "public/cuda", "public/formats/oracle"):
        for path in (root / folder).rglob("*"):
            if path.suffix in {".py", ".h", ".cpp", ".cu"}:
                destination = snapshot / path.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
    for path in (root / "results/summaries").glob("phase2-*.json"):
        shutil.copyfile(path, snapshot / path.name)
    snapshot_index = {str(path.relative_to(snapshot)): reference(path, root)["sha256"]
                      for path in sorted(snapshot.rglob("*")) if path.is_file()}
    write(snapshot / "index.json", snapshot_index)
    inputs = {"phase2": reference(snapshot / final.name, root),
              "phase2_snapshot": reference(snapshot / "index.json", root),
              "formats": reference(root / "public/formats/manifests/accepted/index.json", root),
              "datasets": reference(root / "data/manifests/index.json", root),
              "baseline": reference(root / "results/summaries/phase1-fp32-baselines.json", root),
              "accumulator_candidates": reference(snapshot / "phase2-wide-policy-candidates.json", root)}
    for model in MODELS:
        inputs[model] = reference(root / f"public/workloads/models/manifests/{model}.json", root)
    formats = read(checked(inputs["formats"], root))["manifests"]
    plan = {"schema_version": "phase3-campaign-1.0.0", "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": inputs, "models": list(MODELS), "formats": [row["name"] for row in formats],
            "configuration_count": len(MODELS) * len(formats), "images_per_configuration": 1000,
            "arithmetic": {"mac_model": "C", "reduction_order": "sequential", "uniform_weight_activation_output": True,
                           "accumulator_policy": "per_graph_evidence_required", "operators": "strict"},
            "ptq": {"retraining": False, "optional_external_scaling": False, "bias_correction": False,
                    "reconstruction": False, "mapping_policy": "mse_numpy_f64_100coarse_50fine_v1"},
            "statistics": {"confidence": 0.95, "classification_resamples": 5000, "detector_resamples": 2000,
                           "seed": 310911, "interval": "paired_percentile", "units": "fraction",
                           "promising_max_loss": 0.01, "catastrophic_min_loss": 0.20,
                           "catastrophic_requires_diagnosis": True, "hard_top_n": None},
            "diagnostics": {"sample_elements_per_layer_image": 4096, "outlier": "abs(x-mean)>6*std",
                            "sampling": "evenly_spaced_flat_v1", "sensitivity_images": "first_eight_by_sha256_v1"},
            "screen_gate": ["calibration", "encoded_graph", "accumulator_acceptance", "native_pilot", "diagnostics"]}
    write(root / PLAN, plan)
    return plan


def calibrate(model, plan, root=ROOT):
    import numpy as np
    context = frozen_context(root, model)
    observations = [path for path in sorted((root / "artifacts/calibration").glob(f"{model}-observations-*.json"))
                    if read(path)["context"] == context]
    if not observations:
        raise ValueError(f"verified observations required for {model}")
    observation_path = observations[-1]
    saved = read(observation_path)
    with np.load(observation_path.with_suffix(".npz"), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    records = []
    for format_name in plan["formats"]:
        existing = list((root / "artifacts/calibration").glob(f"{model}-{format_name}-*.json"))
        if existing:
            path = sorted(existing)[-1]
            document = validate(read(path), root, verify_payloads=False)
            if document["context"] != context or document["format"] != format_name:
                raise ValueError("cached calibration context mismatch")
            # Verify observations as well, even when this format was calibrated earlier.
            if document["observations"] != saved["observations"]:
                raise ValueError("cached calibration observations differ")
        else:
            document = create(context, arrays, saved["observations"], format_name)
            validate(document, root, verify_payloads=False)
            path = root / f"artifacts/calibration/{model}-{format_name}-{digest(document)}.json"
            write(path, document)
        records.append({"model": model, "format": format_name, "calibration": reference(path, root),
                        "observations": reference(observation_path, root), "images": 2000})
        write(root / f"artifacts/phase3/calibration/{model}.json", {"campaign_sha256": digest(plan), "records": records})
        print(f"{model} {format_name}: calibration verified ({len(records)}/{len(plan['formats'])})", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"Frozen {plan['configuration_count']} definitions: {PLAN}")
        return
    for model in args.models:
        calibrate(model, plan)


if __name__ == "__main__":
    main()
