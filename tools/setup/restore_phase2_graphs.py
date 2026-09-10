"""Recreate frozen classifier graph payloads only when their hashes reproduce."""
from __future__ import annotations

import hashlib
import json
import argparse
import os
from pathlib import Path

from public.workloads.models.deployment import prepare_deployment, GRAPH_VERSION

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("resnet18", "mobilenet_v2", "mobilenet_v3_large", "yolov8n"),
                        default=["resnet18", "mobilenet_v2", "mobilenet_v3_large", "yolov8n"])
    args = parser.parse_args()
    import torch
    from torchvision import models
    torch.set_num_threads(4)
    records = []
    output = ROOT / "results/summaries/phase2-graph-restoration.json"
    previous = json.loads(output.read_text())["records"] if output.exists() else []
    for name in args.models:
        manifest = json.loads((ROOT / f"public/workloads/models/manifests/{name}.json").read_text())
        checkpoint = ROOT / manifest["checkpoint_path"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["checkpoint_sha256"]:
            raise ValueError(f"checkpoint identity mismatch: {name}")
        if name == "yolov8n":
            os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "cache/ultralytics"))
            os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
            from ultralytics import YOLO
            model = YOLO(checkpoint).model.eval()
        else:
            model = getattr(models, name)(weights=None)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        _, nodes, evidence = prepare_deployment(model, manifest["input_shape"])
        payload = (json.dumps({"architecture": name, "graph_version": GRAPH_VERSION, "nodes": nodes},
                              sort_keys=True, separators=(",", ":")) + "\n").encode()
        digest = hashlib.sha256(payload).hexdigest()
        matches = digest == manifest["deployment_graph_sha256"]
        path = ROOT / manifest["deployment_graph_path"] if matches else ROOT / "artifacts/folded_graphs" / f"{name}-unverified-{digest}.json"
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("refusing to overwrite different graph bytes")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        record = {"model": name, "status": "frozen_hash_reproduced" if matches else "host_graph_hash_mismatch",
                  "expected_sha256": manifest["deployment_graph_sha256"], "actual_sha256": digest,
                  "artifact": str(path.relative_to(ROOT)), "fold_verification": evidence}
        records.append(record)
        print(f"{name}: {record['status']}", flush=True)
    records = [row for row in previous if row["model"] not in args.models] + records
    output.write_text(json.dumps({"schema_version": "2.0.0", "torch": torch.__version__, "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
