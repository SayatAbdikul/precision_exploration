from __future__ import annotations

import json
from pathlib import Path

from public.formats.oracle import NumberFormat, load_manifest
from public.formats.oracle.manifest import manifest_sha256
from public.experiments.registry.identity import IdentityError, validate_experiment
from public.workloads.datasets import verify_phase1_subsets
from public.workloads.models.identity import verify_model_manifest


ROOT = Path(__file__).resolve().parents[2]


def test_all_accepted_manifests_validate_and_match_index() -> None:
    index = json.loads((ROOT / "public/formats/manifests/accepted/index.json").read_text())
    assert index["accepted_count"] == 25
    for row in index["manifests"]:
        manifest = load_manifest(ROOT / row["path"], role="weight")
        assert manifest_sha256(manifest) == row["sha256"]
        fmt = NumberFormat(manifest)
        assert fmt.code_count == 1 << row["bits"]


def test_frozen_dataset_manifests() -> None:
    report = verify_phase1_subsets(ROOT)
    assert report["counts"] == {
        "imagenet_calibration_2k": 2000, "imagenet_screen_1k": 1000,
        "imagenet_evaluation_10k": 10000, "coco_calibration_2k": 2000,
        "coco_screen_1k": 1000, "coco_evaluation_5k": 5000,
    }


def test_frozen_model_payloads_and_graphs() -> None:
    index = json.loads((ROOT / "public/workloads/models/manifests/index.json").read_text())
    assert index["efficientnet_b0"]["status"] == "intentionally_skipped"
    for item in index["manifests"]:
        manifest = verify_model_manifest(ROOT / item["path"], repository_root=ROOT)
        assert len(manifest["checkpoint_sha256"]) == 64


def test_resolved_phase1_configuration_verifies_real_references() -> None:
    model = json.loads((ROOT / "public/workloads/models/manifests/resnet18.json").read_text())
    datasets = json.loads((ROOT / "data/manifests/index.json").read_text())["records"]
    fp8 = load_manifest(ROOT / "public/formats/manifests/accepted/fp8_e4m3fn.json")
    accumulator = {
        "schema_version": "1.0.0", "name": "fp16_e5m10_accumulator", "family": "float", "bits": 16,
        "signed": True, "encoding": "sign_e5m10", "rounding": "rne", "overflow": "infinity",
        "underflow": "subnormal", "zero": "signed_zero", "scaling": {"mode": "none", "granularity": "none"},
        "float": {"exp_bits": 5, "mantissa_bits": 10, "bias": 15, "subnormals": True, "nan": True, "infinity": True},
    }
    manifests = {fp8["name"]: fp8, accumulator["name"]: accumulator}
    config = {
        "schema_version": "1.0.0",
        "model": {"name": "resnet18", "checkpoint_sha256": model["checkpoint_sha256"],
                  "graph_version": model["deployment_graph_version"], "graph_sha256": model["deployment_graph_sha256"],
                  "preprocessing_version": "imagenet_resnet18_v1"},
        "dataset": {"name": "imagenet_1k", "version": "frozen_phase1",
                    "calibration": {"name": "imagenet_calibration_2k", "path": datasets["imagenet_calibration_2k"]["path"],
                                    "sha256": datasets["imagenet_calibration_2k"]["sha256"], "split": "train"},
                    "evaluation": {"name": "imagenet_screen_1k", "path": datasets["imagenet_screen_1k"]["path"],
                                   "sha256": datasets["imagenet_screen_1k"]["sha256"], "split": "validation"},
                    "sets_disjoint": True},
        "formats": {
            **{role: {"name": fp8["name"], "sha256": manifest_sha256(fp8)}
               for role in ("weight", "activation", "output")},
            "accumulator": {"name": accumulator["name"], "sha256": manifest_sha256(accumulator)},
        },
        "ptq": {"experiment": "A", "method": "mse", "weight_granularity": "per_output_channel",
                "activation_granularity": "per_tensor", "rounding": "rne",
                "scaling_policy": "intrinsic_or_required_only", "retraining": False,
                "bias_correction": False, "reconstruction": False},
        "arithmetic": {"mac_model": "full_product", "product_precision": "exact_to_accumulator",
                       "accumulator_policy": "family_appropriate_wide", "reduction_order": "sequential",
                       "bias_domain": "accumulator", "requantization_point": "operator_output_store"},
        "operators": {"policy": "strict", "semantics_version": "1.0.0", "batchnorm_folded": True},
        "runtime": {"backend": "reference", "semantic_version": "1.0.0", "kernel_version": "oracle-1.0.0"},
        "seeds": {"calibration": 20250904, "evaluation": 20250904},
    }
    assert validate_experiment(config, manifests=manifests, repository_root=ROOT) == config
    changed = json.loads(json.dumps(config))
    changed["dataset"]["evaluation"]["sha256"] = "0" * 64
    try:
        validate_experiment(changed, manifests=manifests, repository_root=ROOT)
    except IdentityError as exc:
        assert "list hash mismatch" in str(exc)
    else:
        raise AssertionError("changed dataset hash was accepted")
