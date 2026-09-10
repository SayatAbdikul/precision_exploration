"""Native-resolution classifier conformance on a deterministic frozen image set.

All input identities are checked before lowering or execution. The small set
is a network conformance gate, not the Phase 1 quality baseline population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from public.experiments.registry.identity import canonical_json_bytes
from public.inference.conformance_job import source_identity
from public.inference.tensor import Encoding
from public.quantization.graph.executable import execute, graph_sha256
from public.quantization.graph.torchvision import lower
from public.quantization.ptq.encoding import direct_float_tensor
from public.workloads.datasets.identity import load_tsv, sha256
from public.workloads.models.torchvision_eval import MODEL_SPECS

ROOT = Path(__file__).resolve().parents[2]


def select_samples(root=ROOT):
    """Freeze eight samples by SHA order from the already frozen screen list."""
    index = json.loads((root / "data/manifests/index.json").read_text())
    record = index["records"]["imagenet_screen_1k"]
    path = root / record["path"]
    if sha256(path) != record["sha256"]:
        raise ValueError("frozen screening list hash mismatch")
    rows = load_tsv(path)
    if len(rows) != record["count"]:
        raise ValueError("frozen screening list count mismatch")
    selected = sorted(rows, key=lambda row: row["sha256"])[:8]
    identity = {"selection": "first eight rows sorted by SHA256, version 1", "source_list_sha256": record["sha256"],
                "samples": selected}
    return identity, root / "data/raw" / record["logical_payload_root"]


def verify_images(samples, payload_root):
    verified, missing = [], []
    for row in samples:
        path = payload_root / row["relative_path"]
        if not path.resolve().is_relative_to(payload_root.resolve()):
            raise ValueError("image path escapes payload root")
        if not path.is_file():
            missing.append(str(path))
        elif sha256(path) != row["sha256"]:
            raise ValueError(f"frozen image hash mismatch: {path}")
        else:
            verified.append(path)
    if missing:
        raise FileNotFoundError("frozen conformance images are absent:\n" + "\n".join(missing))
    return verified


def save_checkpoint(path, record):
    payload = {**record, "record_sha256": hashlib.sha256(canonical_json_bytes(record)).hexdigest()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)


def load_checkpoint(path, context):
    if not path.exists():
        return {**context, "backends": {}}
    record = json.loads(path.read_text())
    digest = record.pop("record_sha256")
    if hashlib.sha256(canonical_json_bytes(record)).hexdigest() != digest:
        raise ValueError("fixed-image checkpoint hash mismatch")
    if {key: value for key, value in record.items() if key != "backends"} != context:
        raise ValueError("fixed-image checkpoint identity mismatch")
    if not set(record["backends"]).issubset({"cpp", "cuda"}):
        raise ValueError("fixed-image checkpoint contains an unsupported backend")
    return record


def compare_backends(graph, inputs, record, path, *, runner=execute):
    """Keep each finished backend through interruption, and recheck on resume."""
    expected = None
    for backend in ("cpp", "cuda"):
        if backend not in record["backends"]:
            print(f"{record['sample']['relative_path']} {backend}: executing", flush=True)
            started = time.perf_counter()
            result = runner(graph, inputs, backend=backend)
            record["backends"][backend] = {"seconds": time.perf_counter()-started, "layers": result["layers"],
                "outputs": {name: value.document() for name, value in result["outputs"].items()}}
            save_checkpoint(path, record)
        actual = record["backends"][backend]
        if expected is not None and (actual["layers"] != expected["layers"] or actual["outputs"] != expected["outputs"]):
            raise ValueError(f"network backend mismatch on image {record['sample']['relative_path']}")
        expected = actual
        print(f"{record['sample']['relative_path']} {backend}: all {len(actual['layers'])} layer hashes verified", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("resnet18", "mobilenet_v2", "mobilenet_v3_large"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    selection, default_root = select_samples()
    payload_root = args.dataset_root or default_root
    paths = verify_images(selection["samples"], payload_root)
    manifest_path = ROOT / f"public/workloads/models/manifests/{args.model}.json"
    manifest = json.loads(manifest_path.read_text())
    for path_key, hash_key in (("checkpoint_path", "checkpoint_sha256"),
                               ("deployment_graph_path", "deployment_graph_sha256"),
                               ("preprocessing_path", "preprocessing_sha256")):
        if sha256(ROOT / manifest[path_key]) != manifest[hash_key]:
            raise ValueError(f"frozen model identity mismatch: {path_key}")
    if args.preflight:
        print(f"verified model identities and all {len(paths)} frozen conformance images")
        return
    import numpy as np
    import torch
    import torchvision
    from torchvision import models
    from PIL import Image
    if torch.__version__.split("+")[0] != "2.3.0" or torchvision.__version__.split("+")[0] != manifest["framework_version"]:
        raise ValueError("use the frozen torch 2.3.0 / torchvision 0.18.0 workload environment")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    _, weights_class, weights_name = MODEL_SPECS[args.model]
    transform = getattr(getattr(models, weights_class), weights_name).transforms()
    model = getattr(models, args.model)(weights=None)
    model.load_state_dict(torch.load(ROOT / manifest["checkpoint_path"], map_location="cpu", weights_only=True))
    model.eval()
    encoding = Encoding("fp6_e3m2")
    execution_source = source_identity()
    graph = lower(model, input_encoding=encoding, weight_format=encoding.format, accumulator="fp32_e8m23_accumulator",
                  provenance={"kind": "frozen_image_conformance", "model_manifest_sha256": sha256(manifest_path),
                              "source_graph_sha256": manifest["deployment_graph_sha256"]})
    identity = graph_sha256(graph)
    artifact = ROOT / "artifacts/folded_graphs" / f"{args.model}-phase2-{identity}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(canonical_json_bytes(graph))
    job = {"model": args.model, "source_sha256": execution_source, "graph_sha256": identity,
           "model_manifest_sha256": sha256(manifest_path), "selection": selection}
    job_sha256 = hashlib.sha256(canonical_json_bytes(job)).hexdigest()
    work = ROOT / "artifacts/fixed_image_runs" / job_sha256
    records = []
    for row, path in zip(selection["samples"], paths):
        with Image.open(path) as image:
            inputs_fp32 = transform(image.convert("RGB")).unsqueeze(0)
        if list(inputs_fp32.shape) != manifest["input_shape"]:
            raise ValueError("preprocessed input does not match frozen native resolution")
        inputs = {next(iter(graph["inputs"])): direct_float_tensor(inputs_fp32.numpy(), encoding)}
        with torch.inference_mode():
            fp32_logits = model(inputs_fp32).cpu().numpy()
        if not np.isfinite(fp32_logits).all():
            raise ValueError("FP32 reference produced nonfinite logits")
        context = {"job_sha256": job_sha256, "sample": row,
                   "encoded_input_sha256": hashlib.sha256(canonical_json_bytes(next(iter(inputs.values())).document())).hexdigest(),
                   "fp32_logits": fp32_logits.flatten().tolist()}
        checkpoint = work / f"{row['sha256']}.json"
        record = load_checkpoint(checkpoint, context)
        records.append(compare_backends(graph, inputs, record, checkpoint))
    if source_identity() != execution_source:
        raise RuntimeError("engine sources changed during execution; rerun before publishing evidence")
    report = {"schema_version": "2.0.0", "source_sha256": execution_source, "model": args.model,
              "job_sha256": job_sha256, "checkpoints": str(work.relative_to(ROOT)),
              "model_manifest_sha256": sha256(manifest_path), "selection": selection,
              "graph_sha256": identity, "graph_artifact": str(artifact.relative_to(ROOT)), "records": records,
              "scope": "eight frozen images, native resolution, layerwise CPP/CUDA equality; FP32 logits recorded, not full quality reproduction"}
    output = ROOT / "results/summaries" / f"phase2-{args.model}-fixed-images.json"
    temporary = output.with_suffix(".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
