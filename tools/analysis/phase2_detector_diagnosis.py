"""Reproduce the strict FP6 detector's range failure without changing its policy.

The FP32 observation pass checks lowering independently of native arithmetic.
The final-store probe isolates the representational limit: even perfect FP32
predictions lose their pixel range when stored in the required FP6 encoding.
It is diagnostic evidence, not an alternative exact-inference implementation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from public.experiments.registry.identity import canonical_json_bytes, experiment_sha256
from public.inference.conformance_job import source_identity
from public.inference.reference.arithmetic import format_named
from public.inference.tensor import Encoding
from public.inference.workload_job import detector_predictions, validate_job
from public.quantization.graph.executable import graph_sha256
from public.workloads.datasets.identity import load_tsv, sha256

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOB = "02870391c93c230373e2094674c081ce4fa94ad86bca1adb70ba2cb1cd8757d6"


def diagnose(job=DEFAULT_JOB):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "cache/ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics.data.augment import LetterBox
    from public.quantization.calibration.observer import Observer, detector_observe
    from public.quantization.graph.yolo import lower
    from public.quantization.ptq.encoding import float_tensor

    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    work = ROOT / "artifacts/workload_runs" / job
    configuration = validate_job(json.loads((work / "configuration.json").read_text()))
    if experiment_sha256(configuration) != job:
        raise ValueError("workload configuration hash mismatch")
    if (configuration["model"] != "yolov8n"
            or configuration["formats"]["activation"]["name"] != "fp6_e3m2"
            or configuration["evaluation"]["selection"] != "first_eight_by_sha256_v1"):
        raise ValueError("this diagnostic isolates the eight-image strict FP6 detector probe")
    graph = json.loads((work / "graph.json").read_text())
    identity = graph_sha256(graph)
    summary = json.loads((work / "summary.json").read_text())
    if summary["graph_sha256"] != identity or summary["job_sha256"] != job:
        raise ValueError("workload summary identity mismatch")
    encoding = Encoding("fp6_e3m2")
    box_nodes = [node for node in graph["nodes"] if node["op"] == "decode_boxes"]
    if len(box_nodes) != 3 or any(node["attrs"]["output"] != encoding.document() for node in box_nodes):
        raise ValueError("unexpected detector coordinate storage policy")
    if graph["nodes"][-1]["attrs"]["output"] != encoding.document():
        raise ValueError("unexpected final detector output storage policy")

    manifest_path = ROOT / "public/workloads/models/manifests/yolov8n.json"
    manifest = json.loads(manifest_path.read_text())
    if (ultralytics.__version__ != manifest["framework_version"]
            or torch.__version__.split("+")[0] != "2.3.0"
            or sha256(ROOT / manifest["checkpoint_path"]) != manifest["checkpoint_sha256"]):
        raise ValueError("frozen detector runtime/checkpoint mismatch")
    model = ultralytics.YOLO(ROOT / manifest["checkpoint_path"]).model.eval()
    constants = {}
    observation_graph = lower(model, input_encoding=encoding, weight_format="fp6_e3m2",
        accumulator=configuration["formats"]["accumulator"]["name"],
        provenance={"kind": "fp32_calibration_observation_plan"},
        constant_observer=lambda name, tensor: constants.__setitem__(name, tensor))
    if observation_graph["nodes"] != graph["nodes"]:
        raise ValueError("FP32 observation plan differs from workload lowering")
    record = json.loads((ROOT / "data/manifests/index.json").read_text())["records"][configuration["evaluation"]["name"]]
    rows = sorted(load_tsv(ROOT / record["path"]), key=lambda row: row["sha256"])[:8]
    letterbox = LetterBox(new_shape=(640, 640), auto=False, stride=32)
    fmt = format_named(encoding.format)
    positive = sorted({float(fmt.decode(code)) for code in range(1 << fmt.bits)
                       if fmt.decode(code).is_finite() and fmt.decode(code) > 0})
    maximum, minimum = positive[-1], positive[0]
    evidence, all_predictions, references = [], [], []
    for row in rows:
        stored_path = work / f"{row['sha256']}.json"
        stored = json.loads(stored_path.read_text())
        digest = stored.pop("record_sha256")
        if (hashlib.sha256(canonical_json_bytes(stored)).hexdigest() != digest
                or stored["sample"] != row or stored["graph_sha256"] != identity
                or stored["job_sha256"] != job):
            raise ValueError("per-image workload record mismatch")
        if (set(stored["backends"]) != {"cpp", "cuda"}
                or stored["backends"]["cpp"]["layers"] != stored["backends"]["cuda"]["layers"]):
            raise ValueError("saved native layer comparison failed")
        image = cv2.imread(str(ROOT / "data/raw" / record["logical_payload_root"] / row["file_name"]))
        if image is None:
            raise ValueError("cannot decode frozen image")
        original_shape = image.shape[:2]
        transformed = letterbox(image=image)[:, :, ::-1].transpose(2, 0, 1)
        inputs = torch.from_numpy(np.ascontiguousarray(transformed)).float().unsqueeze(0) / 255
        observer = Observer()
        with torch.inference_mode():
            observed = detector_observe(observation_graph, constants, inputs, observer)
            expected = model(inputs)[0]
        # Accepted Phase 1 fold tolerances, separately for pixel coordinates
        # and class probabilities, do not relax exact native conformance.
        torch.testing.assert_close(observed[:, :4], expected[:, :4], rtol=1e-4, atol=.01)
        torch.testing.assert_close(observed[:, 4:], expected[:, 4:], rtol=1e-4, atol=1e-5)
        baseline = expected.cpu().numpy()
        encoded = float_tensor(baseline, encoding)
        head_predictions = detector_predictions(encoded, original_shape, row["image_id"])
        candidates = stored["prediction"]
        all_predictions.extend(candidates)
        evidence.append({"image_id": int(row["image_id"]), "sample_sha256": row["sha256"],
            "matching_native_layers": len(stored["backends"]["cpp"]["layers"]),
            "fp32_observation_parity": True,
            "fp32_max_coordinate_error": float((observed[:, :4] - expected[:, :4]).abs().max()),
            "fp32_max_probability_error": float((observed[:, 4:] - expected[:, 4:]).abs().max()),
            "fp32_coordinate_elements": int(baseline[:, :4].size),
            "fp32_coordinates_above_candidate_max": int((baseline[:, :4] > maximum).sum()),
            "fp32_probability_elements": int(baseline[:, 4:].size),
            "fp32_probabilities_below_half_candidate_min": int((baseline[:, 4:] < minimum / 2).sum()),
            "strict_prediction_count": len(candidates),
            "strict_degenerate_prediction_count": sum(p["bbox"][2] == 0 or p["bbox"][3] == 0 for p in candidates),
            "ideal_fp32_final_store_probe_prediction_count": len(head_predictions),
            "ideal_fp32_final_store_probe_degenerate_count": sum(p["bbox"][2] == 0 or p["bbox"][3] == 0 for p in head_predictions)})
        references.append({"path": str(stored_path.relative_to(ROOT)), "sha256": sha256(stored_path)})
        print(f"diagnosed image {len(evidence)}/{len(rows)}; FP32 lowering parity passed", flush=True)
    if json.loads((work / "predictions.json").read_text()) != all_predictions:
        raise ValueError("combined predictions differ from verified image records")
    references.extend({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in
        (work / "configuration.json", work / "graph.json", work / "summary.json", work / "predictions.json", manifest_path))
    return {"schema_version": "phase2-detector-diagnosis-1.0.0",
        "status": "diagnosed_candidate_range_failure", "source_sha256": source_identity(),
        "job_sha256": job, "graph_sha256": identity, "evidence": references,
        "format": {"name": encoding.format, "max_finite": maximum, "min_positive": minimum,
                   "scaling": "none", "overflow": "saturate"},
        "native_resolution": [640, 640], "coordinate_store_nodes": [node["name"] for node in box_nodes],
        "metrics": summary["metrics"], "images": evidence,
        "conclusion": "The accepted unscaled FP6 format cannot store native pixel coordinates above 24. "
            "Candidate coordinate stores saturate before NMS; letterbox unpadding can then clip boxes to zero area. "
            "Class probabilities below 0.03125 also round to zero. The two native backends agree on all layers, "
            "and an independent FP32 observation of the same lowering matches the frozen model. This is an "
            "expected catastrophic outcome of this format under the strict policy, not evidence of an engine mismatch.",
        "scope": "Eight frozen screening images; neither a family-wide quality claim nor an accepted accumulator policy. "
            "The ideal FP32 final-store probe isolates a storage limit and is not a production inference regime.",
        "policy": "Optional scaling or higher precision detector output exceptions belong to a separately identified "
            "Experiment B/practical ablation; the strict Experiment A runtime remains unchanged."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", default=DEFAULT_JOB)
    parser.add_argument("--output", type=Path, default=ROOT / "results/summaries/phase2-detector-diagnosis.json")
    args = parser.parse_args()
    report = diagnose(args.job)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
