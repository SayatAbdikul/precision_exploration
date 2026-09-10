"""Read-only Phase 2 input/tool readiness report; missing data stays explicit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def inspect():
    index = json.loads((ROOT / "public/workloads/models/manifests/index.json").read_text())
    models = {}
    for row in index["manifests"]:
        manifest = json.loads((ROOT / row["path"]).read_text())
        record = {}
        for label, path_key, hash_key in [("checkpoint", "checkpoint_path", "checkpoint_sha256"),
                                          ("graph", "deployment_graph_path", "deployment_graph_sha256")]:
            path = ROOT / manifest[path_key]
            record[label] = "missing" if not path.is_file() else (
                "verified" if hashlib.sha256(path.read_bytes()).hexdigest() == manifest[hash_key] else "hash_mismatch")
        models[row["name"]] = record
    tools = {name: shutil.which(name) for name in ("g++", "nvcc", "nvidia-smi", "yosys", "sta", "iverilog", "vvp")}
    gpu = subprocess.run([tools["nvidia-smi"], "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True) if tools["nvidia-smi"] else None
    datasets = json.loads((ROOT / "data/manifests/index.json").read_text())["records"]
    return {"schema_version": "2.0.0", "python": sys.version,
            "dependencies": {name: importlib.util.find_spec(name) is not None for name in ("pytest", "jsonschema", "yaml", "numpy", "torch", "torchvision", "ultralytics")},
            "tools": tools, "gpu": {"accessible": gpu is not None and gpu.returncode == 0,
                                     "output": (gpu.stdout + gpu.stderr).strip() if gpu else "nvidia-smi missing"},
            "models": models,
            "dataset_roots": {name: {"logical_root": row["logical_payload_root"],
                                      "directory_present": (ROOT / "data/raw" / row["logical_payload_root"]).is_dir()}
                              for name, row in datasets.items()},
            "note": "Dataset directory presence is not payload/hash verification. GPU visibility is process/sandbox dependent."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(inspect(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result)


if __name__ == "__main__":
    main()
