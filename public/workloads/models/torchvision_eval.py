"""Deterministic torchvision FP32 classifier evaluation entry point."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

from .identity import checkpoint_sha256, classification_metrics, write_prediction_jsonl


MODEL_SPECS = {
    "resnet18": ("resnet18", "ResNet18_Weights", "IMAGENET1K_V1"),
    "mobilenet_v2": ("mobilenet_v2", "MobileNet_V2_Weights", "IMAGENET1K_V2"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights", "IMAGENET1K_V2"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights", "IMAGENET1K_V1"),
}


def evaluate(
    *,
    model_name: str,
    dataset_root: str | Path,
    sample_list: str | Path,
    output: str | Path,
    checkpoint: str | Path | None = None,
    batch_size: int = 32,
    workers: int = 0,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        import torchvision.models as models
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("torch, torchvision, NumPy, and Pillow are required for workload evaluation") from exc

    if model_name not in MODEL_SPECS:
        raise ValueError(f"unsupported frozen torchvision model: {model_name}")
    constructor_name, weights_class_name, weights_name = MODEL_SPECS[model_name]
    weights = getattr(getattr(models, weights_class_name), weights_name)
    if checkpoint is None:
        model = getattr(models, constructor_name)(weights=weights)
        checkpoint_identity = {"source": weights.url, "sha256": checkpoint_sha256(Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name)}
    else:
        checkpoint_path = Path(checkpoint)
        model = getattr(models, constructor_name)(weights=None)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        checkpoint_identity = {"path": checkpoint_path.as_posix(), "sha256": checkpoint_sha256(checkpoint_path)}
    model = model.eval().to(device)
    transform = weights.transforms()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    root = Path(dataset_root)
    samples: list[tuple[str, int]] = []
    with Path(sample_list).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"relative_path", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"sample list must contain tab-separated columns {sorted(required)}")
        for row in reader:
            samples.append((row["relative_path"], int(row["label"])))

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            tensors = [transform(Image.open(root / relative).convert("RGB")) for relative, _ in batch]
            logits = model(torch.stack(tensors).to(device))
            top5 = logits.topk(5, dim=1).indices.cpu().tolist()
            for (relative, label), predictions in zip(batch, top5):
                rows.append(
                    {
                        "sample_id": relative,
                        "ground_truth": label,
                        "top5_predictions": predictions,
                        "fp32_prediction": predictions[0],
                        "fp32_correct": predictions[0] == label,
                    }
                )

    evidence = write_prediction_jsonl(output, rows)
    return {
        "model": model_name,
        "weights": f"{weights_class_name}.{weights_name}",
        "checkpoint": checkpoint_identity,
        "preprocessing": repr(transform),
        "seed": seed,
        "workers": workers,
        "device": device,
        "elapsed_seconds": time.monotonic() - started,
        "metrics": classification_metrics(rows),
        "predictions": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--summary")
    args = parser.parse_args()
    result = evaluate(
        model_name=args.model,
        dataset_root=args.dataset_root,
        sample_list=args.sample_list,
        output=args.output,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        device=args.device,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.summary:
        summary = Path(args.summary)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
