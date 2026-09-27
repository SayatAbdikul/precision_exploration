"""Resumable Experiment B extension for the 74 previously blocked entries."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from tools.experiment_b import runner as prior_runner
from tools.experiment_b.common import ROOT, atomic_json, dataset, digest, file_hash, formats, frozen_inputs, seal, unseal
from tools.experiment_b.classifier import configure, image_batch as classifier_batch, load_model
from tools.experiment_b.validation import validate_fold
from tools.phase3.worker_locks import exclusive, pool_locks
from .detector import DetectorEngine, DetectorObserver, image_batch as detector_batch, load_detector, prepare_detector
from .shared import BlockQuantizer, codebook, numpy_block_qdq, prepare_shared

BASE = ROOT / "artifacts/experiment_b_ext"
MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large", "yolov8n")
SHARED = frozenset(("bfp6", "mxfp8_e4m3", "mxfp6_e3m2", "mxfp4_e2m1"))
RECIPES = ("maxabs", "percentile_99_9")
PROTOCOL = {
    "version": "experiment-b-extension-1.0.0",
    "purpose": "complete_previously_blocked_74_exploration_entries",
    "images": [128, 1000], "panel_order": "ascending_frozen_image_sha256",
    "shared": "intrinsic_e8m0_k32_fp32_qdq; stored_activations_channel_axis; convolution_patches_and_weights_reduction_k",
    "shared_scale_selection": "block_maxabs_or_block_99_9_percentile_then_smallest_covering_e8m0_scale",
    "shared_accumulator": "ordinary_deterministic_fp32_framework_reduction",
    "detector": "folded_ultralytics_yolov8n_explicit_dfl_and_decode_boxes_qdq",
    "detector_calibration": "2000_frozen_COCO_train_images; all_node_maxima; rotating_channel_256_sample_per_image",
    "detector_head": "independent_boxes_4_and_scores_80_scale_at_84_channel_joins",
    "detector_metrics": "COCOeval_bbox_map50_95_and_map50_on_matching_128_or_1000_image_subset",
    "scope": "development_FP32_QDQ_surrogate; no_exact_A_or_hardware_equivalence_claim",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def source_identity():
    own = sorted((ROOT / "tools/experiment_b_ext").glob("*.py"))
    own += [ROOT / "tools/run/experiment_b_ext.py"]
    return digest({"prior_source": prior_runner.source_identity(),
                   "extension": {str(p.relative_to(ROOT)): file_hash(p) for p in own}})


def inventory():
    rows = []
    for entry in formats():
        for model in MODELS:
            if model == "yolov8n" or entry["name"] in SHARED:
                for recipe in RECIPES:
                    rows.append({"model": model, "format": entry["name"],
                                 "format_sha256": entry["sha256"], "recipe": recipe})
    if len(rows) != 74 or len({(r["model"], r["format"], r["recipe"]) for r in rows}) != 74:
        raise ValueError("extension coverage is not exactly the previously blocked 74")
    return {"protocol": PROTOCOL, "source_sha256": source_identity(), "tasks": rows,
            "prior_completed_expected": 126, "total_matrix": 200}


def validate_shared(device):
    import torch
    rng = np.random.default_rng(20260925)
    checked = []
    for name in SHARED:
        levels, boundaries, _ = codebook(name)
        values = rng.normal(0, 3, (5, 67)).astype(np.float32)
        values[0, :] = 0
        values[1, 31] = 300
        values[2, 34] = -100
        for recipe in RECIPES:
            actual = BlockQuantizer(name, recipe, device)(torch.tensor(values, device=device)).cpu().numpy()
            np.testing.assert_array_equal(actual, numpy_block_qdq(values, name, recipe))
        checked.append({"format": name, "elements": int(values.size), "recipes": list(RECIPES)})
    return {"status": "passed", "device": device, "checked": checked,
            "claim": "torch_matches_declared_numpy_block_QDQ; not_exact_Model_C_equivalence"}


def detector_calibration(graph, constants, context, device, progress):
    import torch
    record, rows, payload = dataset("coco_calibration_2k")
    provenance = {"model_context": context, "protocol": PROTOCOL, "source_sha256": source_identity(),
                  "graph_sha256": digest(graph["nodes"]), "list_sha256": record["sha256"]}
    identity = digest(provenance)
    directory = BASE / "calibration" / identity
    directory.mkdir(parents=True, exist_ok=True)
    seal(directory / "provenance.json", provenance)
    engine = DetectorEngine(graph, constants, device)
    observer = DetectorObserver()
    engine.observer = observer
    arrays, maxima, channels, checkpoints = {}, {}, None, []
    for start in range(0, len(rows), 8):
        batch = rows[start:start+8]
        npz = directory / f"{start:05d}.npz"
        meta_file = directory / f"{start:05d}.json"
        if meta_file.exists():
            for row in batch:
                path = (payload / row["file_name"]).resolve()
                if not path.is_relative_to(payload.resolve()) or file_hash(path) != row["sha256"]:
                    raise ValueError("COCO calibration payload drift")
            meta = unseal(meta_file)
            if meta["identity"] != identity or meta["samples"] != batch or file_hash(npz) != meta["npz_sha256"]:
                raise ValueError("detector calibration checkpoint mismatch")
            with np.load(npz, allow_pickle=False) as saved:
                batch_arrays = {key: saved[key] for key in saved.files}
        else:
            observer.samples, observer.maxima, observer.channels = {}, {}, {}
            engine.ordinal = start
            inputs, _ = detector_batch(batch, payload, device)
            with torch.inference_mode():
                engine.run(inputs)
            batch_arrays = {key: np.concatenate(chunks).astype(np.float32)
                            for key, chunks in observer.samples.items()}
            tmp = npz.with_suffix(".partial")
            with tmp.open("wb") as stream:
                np.savez_compressed(stream, **batch_arrays)
                stream.flush(); os.fsync(stream.fileno())
            tmp.replace(npz)
            meta = {"identity": identity, "samples": batch, "npz_sha256": file_hash(npz),
                    "maxima": observer.maxima, "channels": observer.channels}
            seal(meta_file, meta)
        if channels is not None and meta["channels"] != channels:
            raise ValueError("detector calibration channel coverage changed")
        channels = meta["channels"]
        if set(batch_arrays) != set(channels) or set(batch_arrays) != set(meta["maxima"]):
            raise ValueError("detector calibration node coverage mismatch")
        for key, values in batch_arrays.items():
            if not values.size or not np.isfinite(values).all():
                raise ValueError("invalid detector calibration array")
            arrays.setdefault(key, []).append(values)
            maxima[key] = max(maxima.get(key, 0), meta["maxima"][key])
        checkpoints.append({"start": start, "sha256": file_hash(meta_file)})
        if start % 64 == 0 or start+len(batch) == len(rows):
            progress({"stage": "detector_calibration", "images": start+len(batch), "total": len(rows)})
    merged = {key: np.concatenate(chunks) for key, chunks in arrays.items()}
    summary = {"identity": identity, "images": len(rows), "checkpoints": checkpoints,
               "nodes": {key: {"sample_count": int(values.size), "channels": channels[key],
                               "maxabs": maxima[key]} for key, values in merged.items()}}
    seal(directory / "summary.json", summary)
    return merged, maxima, {"identity": identity, "summary_sha256": digest(summary)}


def predictions(output, shapes, rows):
    from ultralytics.utils import ops
    from ultralytics.data.converter import coco80_to_coco91_class
    categories = coco80_to_coco91_class()
    detections = ops.non_max_suppression(output, conf_thres=.001, iou_thres=.7,
                                          multi_label=True, max_det=300)
    result = []
    for detection, shape, sample in zip(detections, shapes, rows):
        ops.scale_boxes((640, 640), detection[:, :4], shape)
        boxes = ops.xyxy2xywh(detection[:, :4]); boxes[:, :2] -= boxes[:, 2:]/2
        result.append([{"image_id": int(sample["image_id"]), "category_id": categories[int(row[5])],
                        "bbox": [round(float(v), 3) for v in box], "score": round(float(row[4]), 5)}
                       for row, box in zip(detection, boxes)])
    return result


def evaluate_detector(engine, identity, device, limit, progress):
    import torch
    _, population, payload = dataset("coco_screen_1k")
    rows = population[:limit]
    directory = BASE / "predictions" / identity
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    elapsed, computed = 0.0, 0
    for start in range(0, limit, 4):
        batch = rows[start:start+4]
        for row in batch:
            path = (payload / row["file_name"]).resolve()
            if not path.is_relative_to(payload.resolve()) or file_hash(path) != row["sha256"]:
                raise ValueError("COCO evaluation payload drift")
        paths = [directory / (row["sha256"] + ".json") for row in batch]
        saved = [unseal(path) if path.exists() else None for path in paths]
        for row, record in zip(batch, saved):
            if record is not None and (record["configuration_sha256"] != identity or record["sample"] != row):
                raise ValueError("detector prediction provenance mismatch")
        if any(record is None for record in saved):
            tick = time.monotonic()
            inputs, shapes = detector_batch(batch, payload, device)
            with torch.inference_mode():
                output = engine.run(inputs)
                candidate = predictions(output, shapes, batch)
            elapsed += time.monotonic()-tick
            computed += len(batch)
            for index, (row, result) in enumerate(zip(batch, candidate)):
                record = {"configuration_sha256": identity, "sample": row, "detections": result,
                          "batch_start": start, "batch_images": len(batch)}
                if saved[index] is None:
                    seal(paths[index], record)
                    saved[index] = record
                elif saved[index]["detections"] != result:
                    raise ValueError("resumed detector batch changed saved predictions")
        records.extend(saved)
        if start % 32 == 0 or start+len(batch) == limit:
            progress({"stage": "detector_evaluation", "configuration_sha256": identity,
                      "images": len(records), "total": limit})
    return records, {"computed_images_this_invocation": computed,
                     "seconds_this_invocation": elapsed,
                     "images_per_hour_this_invocation": computed/elapsed*3600 if elapsed else None}


def detector_metrics(candidate, baseline):
    from public.analysis.phase3.statistics import coco_metrics, resampled_coco
    manifest = json.loads((ROOT / "public/workloads/datasets/coco2017.json").read_text())
    annotations_file = ROOT / "data/raw/coco2017/annotations/instances_val2017.json"
    if file_hash(annotations_file) != manifest["annotation_sha256"]["instances_val2017"]:
        raise ValueError("COCO annotation drift")
    annotations = json.loads(annotations_file.read_text())
    ids = [int(row["sample"]["image_id"]) for row in candidate]
    if ids != [int(row["sample"]["image_id"]) for row in baseline]:
        raise ValueError("detector paired image order mismatch")
    truth, (a, b) = resampled_coco(annotations, [[d for record in baseline for d in record["detections"]],
                                                [d for record in candidate for d in record["detections"]]], ids)
    left, right = coco_metrics(truth, a), coco_metrics(truth, b)
    return {"images": len(ids), "fp32_map50_95": float(left[0]), "map50_95": float(right[0]),
            "delta_map50_95": float(right[0]-left[0]), "fp32_map50": float(left[1]),
            "map50": float(right[1]), "delta_map50": float(right[1]-left[1]),
            "interpretation": "development_panel_descriptive_only"}


def execute(args, plan):
    import torch
    configure(args.device)
    env = prior_runner.runtime(args.device)
    env.update({"ultralytics": version("ultralytics"), "pycocotools": version("pycocotools")})
    source = plan["source_sha256"]
    state = {"version": PROTOCOL["version"], "source_sha256": source, "runtime": env,
             "started_at": now(), "pid": os.getpid(), "status": "starting",
             "tasks": [{**row, "status": "pending"} for row in plan["tasks"]]}
    def persist(info=None):
        if info is not None:
            state["progress"] = info
        state["updated_at"] = now()
        state["counts"] = dict(Counter(row["status"] for row in state["tasks"]))
        atomic_json(BASE / "status.json", state)
    def progress(info):
        persist(info); print(json.dumps(info, sort_keys=True), flush=True)
    prior_runner.BASE = BASE
    cache = {}
    completed_this_invocation = 0
    with exclusive(BASE / "controller.lock"), pool_locks(ROOT):
        persist()
        try:
            validation = validate_shared(args.device)
            seal(BASE / "validation" / f"{digest({'source': source, 'runtime': env})}.json", validation)
            state["shared_validation"] = validation
            state["status"] = "running"; persist()
            for task in state["tasks"]:
                model, name, recipe = task["model"], task["format"], task["recipe"]
                if args.models and model not in args.models or args.formats and name not in args.formats:
                    continue
                if args.limit is not None and completed_this_invocation >= args.limit:
                    break
                if source_identity() != source:
                    raise ValueError("extension source changed during run")
                task["status"] = "running"; persist()
                if model not in cache:
                    context = frozen_inputs(model)
                    if model == "yolov8n":
                        graph, constants, original = load_detector(args.device)
                        _, rows, payload = dataset("coco_screen_1k")
                        inputs, _ = detector_batch(rows[:1], payload, args.device)
                        with torch.inference_mode():
                            actual = DetectorEngine(graph, constants, args.device).run(inputs)
                            expected = original(inputs)[0]
                        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
                        fold_check = {"status": "passed", "images": 1,
                                      "max_absolute_error": float((actual-expected).abs().max())}
                        arrays = maxima = calibration = None
                        if any(t["model"] == model and t["format"] not in SHARED and
                               (not args.formats or t["format"] in args.formats) for t in state["tasks"]):
                            arrays, maxima, calibration = detector_calibration(graph, constants, context, args.device, progress)
                        baseline_id = digest({"context": context, "source_sha256": source, "runtime": env,
                                              "mode": "folded_fp32_detector_baseline", "protocol": PROTOCOL})
                        baseline, _ = evaluate_detector(DetectorEngine(graph, constants, args.device), baseline_id,
                                                        args.device, args.images, progress)
                        cache[model] = (context, graph, constants, arrays, maxima, calibration, baseline, baseline_id, fold_check)
                    else:
                        graph, transform, original = load_model(model, args.device)
                        _, rows, payload = dataset("imagenet_screen_1k")
                        fold_check = validate_fold(graph, original, classifier_batch(rows[:8], payload, transform, args.device))
                        baseline_id = digest({"context": context, "source_sha256": source, "runtime": env,
                                              "mode": "folded_fp32_classifier_baseline", "protocol": PROTOCOL})
                        baseline, _ = prior_runner.evaluate(graph, transform, model, baseline_id,
                                                            args.device, args.images, progress)
                        cache[model] = (context, graph, transform, None, None, None, baseline, baseline_id, fold_check)
                context, graph, extra, arrays, maxima, calibration, baseline, baseline_id, fold_check = cache[model]
                tick = time.monotonic()
                if model == "yolov8n":
                    engine, scales = prepare_detector(graph, extra, name, recipe, arrays, maxima, args.device)
                else:
                    engine, scales = prepare_shared(graph, name, recipe, args.device)
                configuration = {"model_context": context, "format": name, "format_sha256": task["format_sha256"],
                                 "recipe": recipe, "protocol": PROTOCOL, "runtime": env, "source_sha256": source,
                                 "calibration": calibration if calibration else "not_used_for_intrinsic_shared_scales",
                                 "scales": scales, "baseline_sha256": baseline_id}
                identity = digest(configuration)
                task["configuration_sha256"] = identity
                seal(BASE / "configurations" / f"{identity}.json", configuration)
                summary_file = BASE / "summaries" / f"{identity}-{args.images}.json"
                if summary_file.exists():
                    summary = unseal(summary_file)
                    if summary["configuration_sha256"] != identity or summary["panel_images"] != args.images:
                        raise ValueError("extension summary provenance mismatch")
                    task.update(status="completed", metrics=summary["metrics"], timing=summary["timing"])
                    persist(); continue
                if model == "yolov8n":
                    candidate, timing = evaluate_detector(engine, identity, args.device, args.images, progress)
                    measured = detector_metrics(candidate, baseline)
                else:
                    candidate, timing = prior_runner.evaluate(engine, extra, model, identity, args.device, args.images, progress)
                    measured = prior_runner.metrics(candidate, baseline)
                if source_identity() != source or frozen_inputs(model) != context:
                    raise ValueError("extension inputs/source changed before commit")
                summary = {"configuration_sha256": identity, "panel_images": args.images,
                           "metrics": measured, "timing": timing, "fold_validation": fold_check,
                           "native_acceptance": False, "prediction_digest": digest(candidate),
                           "baseline_prediction_digest": digest(baseline)}
                seal(summary_file, summary)
                task.update(status="completed", metrics=measured, timing=timing,
                            preparation_seconds=time.monotonic()-tick)
                completed_this_invocation += 1
                progress({"stage": "configuration_completed", "model": model, "format": name,
                          "recipe": recipe, **measured, **timing})
            state["status"] = "extension_finished" if all(t["status"] == "completed" for t in state["tasks"]) else "invocation_finished"
            persist()
        except (KeyboardInterrupt, InterruptedError):
            for row in state["tasks"]:
                if row["status"] == "running": row["status"] = "interrupted"
            state["status"] = "interrupted"; persist()
            return 130
        except Exception as error:
            for row in state["tasks"]:
                if row["status"] == "running": row["status"] = "failed"
            state.update(status="failed", error=f"{type(error).__name__}: {error}"); persist()
            raise
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--images", type=int, choices=(128, 1000), default=128)
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--formats", nargs="+")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.status:
        state = json.loads((BASE / "status.json").read_text())
        print(json.dumps({k: v for k, v in state.items() if k not in ("tasks", "shared_validation")}, indent=2))
        return 0
    plan = inventory()
    if args.formats and set(args.formats) - {row["format"] for row in plan["tasks"]}:
        parser.error("unknown extension format")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    seal(BASE / "inventory.json", plan)
    if args.prepare:
        print(json.dumps({"configurations": len(plan["tasks"]), "inventory": str(BASE / "inventory.json")}, indent=2))
        return 0
    def stop(_signal, _frame):
        raise InterruptedError("stop requested")
    signal.signal(signal.SIGTERM, stop)
    return execute(args, plan)
