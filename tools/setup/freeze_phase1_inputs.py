#!/usr/bin/env python3
"""Freeze the owner-approved Phase 1 datatype, dataset, and model identities."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from public.formats.oracle.manifest import manifest_sha256, validate_manifest
from public.workloads.models.identity import checkpoint_sha256, graph_identity
from public.workloads.models.deployment import prepare_deployment, GRAPH_VERSION


ROOT = Path(__file__).resolve().parents[2]
NONE = {"mode": "none", "granularity": "none"}
MAPPING = {"mode": "required_mapping", "granularity": "per_tensor"}


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def base(name: str, family: str, bits: int, encoding: str, *, scaling: dict[str, Any] = NONE) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0", "name": name, "family": family, "bits": bits,
        "signed": True, "encoding": encoding, "rounding": "rne", "overflow": "saturate",
        "underflow": "not_applicable", "zero": "code_0", "scaling": scaling,
    }


def integer(bits: int) -> dict[str, Any]:
    value = base(f"int{bits}", "integer", bits, "twos_complement", scaling=MAPPING)
    value["numeric"] = {"integer_bits": bits - 1, "fractional_bits": 0, "zero_point": 0}
    return value


def fixed() -> dict[str, Any]:
    value = base("q1_6", "fixed_point", 8, "twos_complement_fixed")
    value["numeric"] = {"integer_bits": 1, "fractional_bits": 6, "zero_point": 0}
    return value


def minifloat(name: str, bits: int, exp: int, mantissa: int, bias: int, *, infinity: bool) -> dict[str, Any]:
    value = base(name, "float", bits, f"sign_e{exp}m{mantissa}")
    value.update({
        "underflow": "subnormal", "overflow": "infinity" if infinity else "saturate",
        "zero": "signed_zero",
        "float": {"exp_bits": exp, "mantissa_bits": mantissa, "bias": bias,
                  "subnormals": True, "nan": True, "infinity": infinity},
    })
    return value


def block(name: str, family: str, bits: int, element: dict[str, Any]) -> dict[str, Any]:
    value = {**element, "float": dict(element["float"])}
    if name in {"mxfp6_e3m2", "mxfp4_e2m1"}:
        value["float"]["nan"] = False
    value.update({
        "name": name, "family": family, "bits": bits,
        "scaling": {"mode": "intrinsic_shared", "granularity": "block", "scale_format": "e8m0", "scale_bits": 8},
        "block": {"shared_scale_format": "e8m0", "block_size": 32, "block_axis": "reduction_k",
                  "incomplete_block_policy": "scale_valid_values_zero_pad"},
    })
    return value


def posit(bits: int, es: int) -> dict[str, Any]:
    value = base(f"posit{bits}_es{es}", "posit", bits, "posit_standard")
    value["zero"] = "zero_and_nar"
    value["posit"] = {"es": es}
    return value


def logarithmic(bits: int, integer_bits: int, fractional_bits: int) -> dict[str, Any]:
    value = base(f"log{bits}", "logarithmic", bits, "zero_then_signed_log2")
    value["zero"] = "code_zero"
    value["logarithmic"] = {"integer_bits": integer_bits, "fractional_bits": fractional_bits,
                            "linear_near_zero": False}
    return value


def candidates() -> list[dict[str, Any]]:
    fp8e4m3 = minifloat("fp8_e4m3fn", 8, 4, 3, 7, infinity=False)
    fp6e3m2 = minifloat("fp6_e3m2", 6, 3, 2, 3, infinity=False)
    fp6e2m3 = minifloat("fp6_e2m3", 6, 2, 3, 1, infinity=False)
    fp4e2m1 = minifloat("fp4_e2m1", 4, 2, 1, 1, infinity=False)
    bfp = base("bfp6", "bfp", 6, "signed_fixed_mantissa", scaling={"mode": "intrinsic_shared", "granularity": "block", "scale_format": "e8m0", "scale_bits": 8})
    bfp["zero"] = "code_000000"
    bfp["numeric"] = {"integer_bits": 0, "fractional_bits": 5, "zero_point": 0}
    bfp["block"] = {"shared_scale_format": "e8m0", "block_size": 32, "block_axis": "reduction_k", "incomplete_block_policy": "scale_valid_values_zero_pad"}
    nf4 = base("nf4", "codebook", 4, "bitsandbytes_normalfloat4", scaling=MAPPING)
    nf4["zero"] = "code_0111"
    nf4["codebook"] = {"values": [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]}
    binary = base("binary_pm1", "binary", 1, "binary_pm1", scaling=MAPPING)
    binary["zero"] = "not_representable"
    ternary = base("ternary", "ternary", 2, "ternary_zero_pos_neg_reserved", scaling=MAPPING)
    ternary["zero"] = "code_00; code_11_reserved"
    return [
        integer(8), integer(6), integer(5), integer(4), fixed(), fp8e4m3,
        minifloat("fp8_e5m2", 8, 5, 2, 15, infinity=True),
        minifloat("fp7_e3m3", 7, 3, 3, 3, infinity=False), fp6e3m2, fp6e2m3,
        minifloat("fp5_e2m2", 5, 2, 2, 1, infinity=False), fp4e2m1, bfp,
        block("mxfp8_e4m3", "mx_float", 8, fp8e4m3),
        block("mxfp6_e3m2", "mx_float", 6, fp6e3m2),
        block("mxfp4_e2m1", "mx_float", 4, fp4e2m1),
        posit(8, 1), posit(6, 1), posit(4, 0),
        logarithmic(8, 3, 3), logarithmic(6, 2, 2), logarithmic(4, 1, 1),
        nf4, binary, ternary,
    ]


def freeze_candidates() -> dict[str, Any]:
    output = ROOT / "public/formats/manifests/accepted"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for manifest in candidates():
        validate_manifest(manifest, role="weight")
        digest = manifest_sha256(manifest)
        path = output / f"{manifest['name']}.json"
        dump(path, manifest)
        rows.append({"name": manifest["name"], "family": manifest["family"], "bits": manifest["bits"],
                     "path": path.relative_to(ROOT).as_posix(), "sha256": digest})
    aggregate = hashlib.sha256("".join(f"{row['name']}:{row['sha256']}\n" for row in rows).encode()).hexdigest()
    index = {"schema_version": "1.0.0", "decision": "D1", "status": "accepted", "accepted_count": len(rows),
             "aggregate_sha256": aggregate, "manifests": rows}
    dump(output / "index.json", index)
    return index


def copy_dataset_manifest(source: str, destination: str, logical_root: str) -> dict[str, Any]:
    src, dst = ROOT / source, ROOT / destination
    dst.parent.mkdir(parents=True, exist_ok=True)
    # csv.writer emits CRLF by default; tracked canonical lists use LF so Git
    # and non-Windows consumers hash the same byte stream.
    dst.write_bytes(src.read_bytes().replace(b"\r\n", b"\n"))
    payload = dst.read_bytes()
    with dst.open(encoding="utf-8", newline="") as stream:
        count = sum(1 for _ in csv.DictReader(stream, delimiter="\t"))
    return {"path": destination, "sha256": hashlib.sha256(payload).hexdigest(), "count": count,
            "logical_payload_root": logical_root}


def copy_selection_record(source: str, destination: str) -> dict[str, Any]:
    src, dst = ROOT / source, ROOT / destination
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    payload = dst.read_bytes()
    return {"path": destination, "sha256": hashlib.sha256(payload).hexdigest()}


def freeze_datasets() -> dict[str, Any]:
    selections = {
        "imagenet_calibration_2k": copy_selection_record("data/raw/imagenet1k/train_calibration_2k/selection.json", "data/manifests/selection/imagenet1k_train_2k.json"),
        "imagenet_screen_1k": copy_selection_record("data/raw/imagenet1k/val_screen_1k/selection.json", "data/manifests/selection/imagenet1k_val_1k.json"),
        "imagenet_evaluation_10k": copy_selection_record("data/raw/imagenet1k/val_evaluation_10k/selection.json", "data/manifests/selection/imagenet1k_val_10k.json"),
        "coco_calibration_2k": copy_selection_record("data/raw/coco2017/train2017_calibration_2k/selection.json", "data/manifests/selection/coco2017_train_2k.json"),
        "coco_evaluation_5k": copy_selection_record("data/raw/coco2017/val2017/selection.json", "data/manifests/selection/coco2017_val_5k.json"),
    }
    selections["coco_screen_1k"] = selections["coco_evaluation_5k"]
    records = {
        "imagenet_calibration_2k": copy_dataset_manifest("data/raw/imagenet1k/train_calibration_2k/manifest.tsv", "data/manifests/calibration/imagenet1k_train_2k.tsv", "imagenet1k/train_calibration_2k"),
        "imagenet_screen_1k": copy_dataset_manifest("data/raw/imagenet1k/val_screen_1k/manifest.tsv", "data/manifests/evaluation/imagenet1k_val_1k.tsv", "imagenet1k/val_screen_1k"),
        "imagenet_evaluation_10k": copy_dataset_manifest("data/raw/imagenet1k/val_evaluation_10k/manifest.tsv", "data/manifests/evaluation/imagenet1k_val_10k.tsv", "imagenet1k/val_evaluation_10k"),
        "coco_calibration_2k": copy_dataset_manifest("data/raw/coco2017/train2017_calibration_2k/manifest.tsv", "data/manifests/calibration/coco2017_train_2k.tsv", "coco2017/train2017_calibration_2k"),
        "coco_screen_1k": copy_dataset_manifest("data/raw/coco2017/val2017/screen_manifest.tsv", "data/manifests/evaluation/coco2017_val_1k.tsv", "coco2017/val2017"),
        "coco_evaluation_5k": copy_dataset_manifest("data/raw/coco2017/val2017/manifest.tsv", "data/manifests/evaluation/coco2017_val_5k.tsv", "coco2017/val2017"),
    }
    for name, record in records.items():
        record["selection_record"] = selections[name]
    index = {"schema_version": "1.0.0", "selection_seed": 20250904,
             "imagenet_revision": "49e2ee26f3810fb5a7536bbf732a7b07389a47b5",
             "imagenet_full_50k": {"status": "deferred_to_phase_5_by_owner", "payload_downloaded": False},
             "coco_release": "2017", "records": records}
    dump(ROOT / "data/manifests/index.json", index)
    return index


def freeze_models() -> dict[str, Any]:
    import torch
    import torchvision
    from torchvision import models
    from ultralytics import YOLO

    torch.set_num_threads(4)
    model_specs = [
        ("resnet18", models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "resnet18-f37072fd.pth", 224),
        ("mobilenet_v2", models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V2, "mobilenet_v2-7ebf99e0.pth", 224),
        ("mobilenet_v3_large", models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2, "mobilenet_v3_large-5c1a4163.pth", 224),
    ]
    manifests = []
    graph_root = ROOT / "artifacts/folded_graphs"
    graph_root.mkdir(parents=True, exist_ok=True)
    for name, constructor, weights, filename, shape in model_specs:
        checkpoint = ROOT / "artifacts/checkpoints" / filename
        model = constructor(weights=None).eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        _, nodes, fold_evidence = prepare_deployment(model, [1, 3, shape, shape])
        graph_hash = graph_identity(architecture=name, graph_version=GRAPH_VERSION, nodes=nodes)
        graph_path = graph_root / f"{name}-{graph_hash}.json"
        graph_path.write_text(json.dumps({"architecture": name, "graph_version": GRAPH_VERSION, "nodes": nodes}, sort_keys=True, separators=(",", ":")) + "\n")
        upstream = weights.meta["_metrics"]["ImageNet-1K"]
        manifest = {"schema_version": "1.0.0", "name": name, "framework": "torchvision", "framework_version": torchvision.__version__,
            "checkpoint_identifier": f"{weights.__class__.__name__}.{weights.name}", "checkpoint_source": weights.url,
            "checkpoint_path": f"artifacts/checkpoints/{filename}", "checkpoint_size_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_sha256(checkpoint), "license": "BSD-3-Clause (torchvision code); ImageNet weights terms follow source dataset",
            "load_procedure": "constructor(weights=None); load_state_dict(weights_only=True); eval()", "evaluation_mode": True,
            "deployment_graph_version": GRAPH_VERSION, "deployment_graph_sha256": graph_hash,
            "input_shape": [1, 3, shape, shape], "output": "1000 ImageNet-1K class logits", "dataset": "ILSVRC2012",
            "evaluator": "public.workloads.models.torchvision_eval:1.0.0", "metrics": ["top1_percent", "top5_percent"],
            "upstream_reference": {"population": "ImageNet-1K full validation set", "top1_percent": upstream["acc@1"],
                                   "top5_percent": upstream["acc@5"], "source": "torchvision weight metadata"}}
        prep = "imagenet_resnet18" if name == "resnet18" else "imagenet_mobilenet"
        prep_path = f"public/workloads/datasets/{prep}_preprocessing.json"
        manifest.update({"preprocessing_version": prep + "_v1", "preprocessing_path": prep_path,
                         "preprocessing_sha256": checkpoint_sha256(ROOT / prep_path),
                         "deployment_graph_path": graph_path.relative_to(ROOT).as_posix(),
                         "fold_verification": fold_evidence})
        dump(ROOT / f"public/workloads/models/manifests/{name}.json", manifest)
        manifests.append(manifest)

    checkpoint = ROOT / "artifacts/checkpoints/yolov8n.pt"
    yolo = YOLO(checkpoint)
    _, nodes, fold_evidence = prepare_deployment(yolo.model.eval(), [1, 3, 640, 640])
    graph_hash = graph_identity(architecture="yolov8n", graph_version=GRAPH_VERSION, nodes=nodes)
    (graph_root / f"yolov8n-{graph_hash}.json").write_text(json.dumps({"architecture": "yolov8n", "graph_version": GRAPH_VERSION, "nodes": nodes}, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {"schema_version": "1.0.0", "name": "yolov8n", "framework": "ultralytics", "framework_version": "8.3.0",
        "checkpoint_identifier": "ultralytics-assets/v8.3.0/yolov8n.pt", "checkpoint_source": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "checkpoint_path": "artifacts/checkpoints/yolov8n.pt", "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256(checkpoint), "license": "AGPL-3.0; checkpoint provenance Ultralytics assets",
        "load_procedure": "ultralytics.YOLO(checkpoint); model.val()", "evaluation_mode": True,
        "deployment_graph_version": GRAPH_VERSION, "deployment_graph_sha256": graph_hash,
        "input_shape": [1, 3, 640, 640], "output": "COCO 80-category detections", "dataset": "COCO 2017",
        "evaluator": "ultralytics.YOLO.val:8.3.0", "metrics": ["map50_95", "map50"],
        "upstream_reference": {"population": "COCO val2017 5k", "map50_95": 0.373,
                               "source": "Ultralytics YOLOv8 model performance table"}}
    prep_path = "public/workloads/datasets/coco2017.json"
    manifest.update({"preprocessing_version": "coco2017_ultralytics_v1", "preprocessing_path": prep_path,
                     "preprocessing_sha256": checkpoint_sha256(ROOT / prep_path),
                     "deployment_graph_path": f"artifacts/folded_graphs/yolov8n-{graph_hash}.json",
                     "fold_verification": fold_evidence})
    dump(ROOT / "public/workloads/models/manifests/yolov8n.json", manifest)
    manifests.append(manifest)
    index = {"schema_version": "1.0.0", "mandatory": [item["name"] for item in manifests],
             "efficientnet_b0": {"status": "intentionally_skipped", "decision_D3": "open"},
             "manifests": [{"name": item["name"], "path": f"public/workloads/models/manifests/{item['name']}.json"} for item in manifests]}
    dump(ROOT / "public/workloads/models/manifests/index.json", index)
    return index


def main() -> None:
    summary = {"datatypes": freeze_candidates(), "datasets": freeze_datasets(), "models": freeze_models()}
    dump(ROOT / "results/summaries/phase1-input-freeze.json", summary)
    print(json.dumps({"accepted_datatypes": summary["datatypes"]["accepted_count"],
                      "dataset_manifests": len(summary["datasets"]["records"]),
                      "models": len(summary["models"]["mandatory"])}, sort_keys=True))


if __name__ == "__main__":
    main()
