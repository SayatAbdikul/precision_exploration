#!/usr/bin/env python3
"""Reproduce classifier baselines and validate folded graphs on fixed real images."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(*, resume=False):
    import torch
    import torchvision.models as models
    from PIL import Image
    from public.workloads.models.deployment import fold_batchnorm, capture_graph, GRAPH_VERSION
    from public.workloads.models.identity import graph_identity, verify_model_manifest
    from public.workloads.models.torchvision_eval import evaluate, MODEL_SPECS
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    index = json.loads((ROOT / "data/manifests/index.json").read_text())["records"]
    record = index["imagenet_screen_1k"]
    payload_root = ROOT / "data/raw" / record["logical_payload_root"]
    with (ROOT / record["path"]).open() as stream:
        samples = list(csv.DictReader(stream, delimiter="\t"))[:8]
    report = {"schema_version": "1.0.0", "graph_version": GRAPH_VERSION, "classifiers": [], "graphs": []}
    for name in ("resnet18", "mobilenet_v2", "mobilenet_v3_large", "yolov8n"):
        manifest = verify_model_manifest(ROOT / f"public/workloads/models/manifests/{name}.json", repository_root=ROOT)
        checkpoint = ROOT / manifest["checkpoint_path"]
        if name != "yolov8n":
            output = ROOT / f"artifacts/per_image_predictions/verification/{name}.jsonl"
            if resume and output.is_file():
                from public.workloads.models.identity import classification_metrics
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                result = {"predictions": {"sha256": hashlib.sha256(output.read_bytes()).hexdigest()},
                          "metrics": classification_metrics(rows)}
            else:
                result = evaluate(model_name=name, dataset_root=payload_root, sample_list=ROOT / record["path"],
                                  output=output, checkpoint=checkpoint)
            baseline = ROOT / f"artifacts/per_image_predictions/{name}_imagenet1k_screen.jsonl"
            if result["predictions"]["sha256"] != hashlib.sha256(baseline.read_bytes()).hexdigest():
                raise ValueError(f"{name} baseline predictions did not reproduce")
            report["classifiers"].append({"model": name, "sample_count": 1000, "result": "byte_identical",
                                           "prediction_sha256": result["predictions"]["sha256"], "metrics": result["metrics"]})
            constructor, weight_class, weight_name = MODEL_SPECS[name]
            model = getattr(models, constructor)(weights=None).eval()
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            transform = getattr(getattr(models, weight_class), weight_name).transforms()
            inputs = [transform(Image.open(payload_root / row.get("relative_path", row.get("file_name", ""))).convert("RGB")).unsqueeze(0) for row in samples]
            sample_ids = [row["relative_path"] for row in samples]
        else:
            import cv2
            from ultralytics import YOLO
            from ultralytics.data.augment import LetterBox
            model = YOLO(checkpoint).model.eval()
            coco = index["coco_screen_1k"]
            coco_root = ROOT / "data/raw" / coco["logical_payload_root"]
            with (ROOT / coco["path"]).open() as stream:
                coco_samples = list(csv.DictReader(stream, delimiter="\t"))[:8]
            inputs = []
            for row in coco_samples:
                bgr = cv2.imread(str(coco_root / row.get("relative_path", row.get("file_name", ""))))
                rgb = LetterBox(new_shape=(640, 640), auto=False)(image=bgr)[:, :, ::-1].copy()
                inputs.append(torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255)
            sample_ids = [row["image_id"] for row in coco_samples]
        folded = fold_batchnorm(model)
        maximum_error = 0.
        repeated = hashlib.sha256()

        def compare(a, b):
            nonlocal maximum_error
            if isinstance(a, torch.Tensor):
                if name == "yolov8n" and a.ndim == 3 and a.shape[1] == 84:
                    # Decoded xywh coordinates are in pixels; class scores are
                    # probabilities. Do not use the score tolerance for pixels.
                    torch.testing.assert_close(a[:, :4], b[:, :4], atol=0.01, rtol=1e-4)
                    torch.testing.assert_close(a[:, 4:], b[:, 4:], atol=1e-5, rtol=1e-4)
                else:
                    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
                maximum_error = max(maximum_error, float((a - b).abs().max()))
            else:
                assert len(a) == len(b)
                for left, right in zip(a, b):
                    compare(left, right)

        with torch.inference_mode():
            for value in inputs:
                original, result = model(value), folded(value)
                compare(original, result)
                second = folded(value)
                def exact(a, b):
                    if isinstance(a, torch.Tensor):
                        assert torch.equal(a, b)
                        repeated.update(a.numpy().tobytes())
                    else:
                        for left, right in zip(a, b): exact(left, right)
                exact(result, second)
                if name != "yolov8n":
                    assert torch.equal(original.topk(5).indices, result.topk(5).indices)
        nodes = capture_graph(folded, torch.zeros(manifest["input_shape"]))
        digest = graph_identity(architecture=name, graph_version=GRAPH_VERSION, nodes=nodes)
        if digest != manifest["deployment_graph_sha256"]:
            raise ValueError(f"{name} graph did not reproduce: {digest}")
        report["graphs"].append({"model": name, "graph_sha256": digest, "sample_ids": sample_ids,
                                  "fold_parity": "passed", "atol": 1e-4, "rtol": 1e-4,
                                  "max_absolute_error": maximum_error, "repeat_inference": "byte_identical",
                                  "coordinate_atol_pixels": 0.01 if name == "yolov8n" else None,
                                  "class_score_atol": 1e-5 if name == "yolov8n" else None,
                                  "folded_output_sha256": repeated.hexdigest()})
        print(f"verified {name}", flush=True)
    output = ROOT / "results/summaries/phase1-workload-verification.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Reuse already generated classifier verification predictions after rechecking their hashes")
    main(**vars(parser.parse_args()))
