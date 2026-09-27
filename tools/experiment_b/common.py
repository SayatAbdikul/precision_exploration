"""Frozen inputs, atomic evidence and the complete B coverage inventory."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/experiment_b"
MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large", "yolov8n")
RECIPES = ("maxabs", "percentile_99_9")
SEMANTICS = "b_scalar_fp32_qdq_wideacc_v1"
SAMPLING = "rotating_channel_stratified_256_per_image_v1"
PROTOCOL = {
    "version": "experiment-b-exploration-1.0.0",
    "semantics": SEMANTICS,
    "image_stages": [128, 1000],
    "panel_order": "ascending_frozen_image_sha256",
    "calibration_images": 2000,
    "observer": SAMPLING,
    "recipes": list(RECIPES),
    "weights": "per_output_channel_external_fp32_scale",
    "activations": "static_per_node_external_fp32_scale",
    "scale_rule": "positive_abs_threshold_divided_by_positive_max_finite_level",
    "maxabs": "maximum_over_all_observed_elements",
    "percentile_99_9": "numpy_linear_0.999_quantile_of_abs_values; activation_stratified_sample; weights_all_elements",
    "rounding": "fp32_normalization_and_codebook_midpoints; manifest_code_tie_preference",
    "overflow": "explicit_B_finite_clipping_before_encoding",
    "bias_and_reductions": "ordinary_fp32_framework_ops; TF32_disabled; no_A_Model_C_equivalence_claim",
    "operators": "quantize_input_and_arithmetic_outputs; skip_identity_dropout_flatten; no_first_last_exemption",
    "shared_formats": "blocked_until_reduction_axis_block_quantization_is_validated",
    "detector": "blocked_until_head_observer_and_operator_policy_are_validated",
    "inference_batch_size": 8,
    "calibration_batch_size": 8,
    "interpretation": "exploratory_development_metrics; not_final_ranking_or_native_acceptance",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("w") as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def seal(path, payload):
    atomic_json(path, {"payload": payload, "sha256": digest(payload)})


def unseal(path):
    doc = json.loads(Path(path).read_text())
    if set(doc) != {"payload", "sha256"} or digest(doc["payload"]) != doc["sha256"]:
        raise ValueError(f"evidence integrity failure: {path}")
    return doc["payload"]


def source_identity():
    paths = list((ROOT / "tools/experiment_b").glob("*.py"))
    for folder in ("public/formats/oracle", "public/workloads/models", "public/workloads/datasets"):
        paths.extend((ROOT / folder).glob("*.py"))
    paths.append(ROOT / "tools/run/experiment_b.py")
    paths.append(ROOT / "tools/phase3/worker_locks.py")
    paths.extend((ROOT / "public/formats/manifests/accepted").glob("*.json"))
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)})


def formats():
    from public.formats.oracle.manifest import manifest_sha256
    index = json.loads((ROOT / "public/formats/manifests/accepted/index.json").read_text())
    for entry in index["manifests"]:
        if manifest_sha256(json.loads((ROOT / entry["path"]).read_text())) != entry["sha256"]:
            raise ValueError("accepted format manifest drift")
    return index["manifests"]


def dataset(name):
    from public.workloads.datasets.identity import load_tsv
    index = json.loads((ROOT / "data/manifests/index.json").read_text())["records"]
    record = index[name]
    if file_hash(ROOT / record["path"]) != record["sha256"]:
        raise ValueError("dataset list identity drift")
    rows = sorted(load_tsv(ROOT / record["path"]), key=lambda row: row["sha256"])
    if len(rows) != record["count"] or len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError("dataset count/uniqueness mismatch")
    return record, rows, ROOT / "data/raw" / record["logical_payload_root"]


def frozen_inputs(model):
    path = ROOT / f"public/workloads/models/manifests/{model}.json"
    manifest = json.loads(path.read_text())
    for key in ("checkpoint", "preprocessing", "deployment_graph"):
        if file_hash(ROOT / manifest[f"{key}_path"]) != manifest[f"{key}_sha256"]:
            raise ValueError(f"{key} drift for {model}")
    prefix = "coco" if model == "yolov8n" else "imagenet"
    cal, cal_rows, _ = dataset(prefix + "_calibration_2k")
    ev, ev_rows, _ = dataset(prefix + "_screen_1k")
    if {r['sha256'] for r in cal_rows} & {r['sha256'] for r in ev_rows}:
        raise ValueError("calibration/evaluation overlap")
    return {"model": model, "manifest_sha256": file_hash(path), "checkpoint_sha256": manifest["checkpoint_sha256"],
            "preprocessing_sha256": manifest["preprocessing_sha256"], "source_graph_sha256": manifest["deployment_graph_sha256"],
            "calibration_sha256": cal["sha256"], "evaluation_sha256": ev["sha256"]}


def inventory():
    rows = []
    for model in MODELS:
        for fmt in formats():
            reason = None
            if model == "yolov8n":
                reason = "detector_head_observer_and_QDQ_operator_validation_pending"
            elif fmt["family"] in {"bfp", "mx_float"}:
                reason = "intrinsic_shared_reduction_axis_block_QDQ_validation_pending"
            for recipe in RECIPES:
                rows.append({"model": model, "format": fmt["name"], "format_sha256": fmt["sha256"],
                             "recipe": recipe, "implementation_status": "blocked" if reason else "implemented_pending_validation",
                             "blocked_reason": reason})
    return {"protocol": PROTOCOL, "source_sha256": source_identity(), "configurations": rows,
            "datatype_model_pairs": 100, "configuration_count": len(rows),
            "planned_exploration_images": 128, "experiment_a_complete": False}
