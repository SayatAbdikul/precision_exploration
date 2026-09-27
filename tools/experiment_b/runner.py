"""Checkpointed B exploration with explicit blocked coverage and provenance."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from .common import (BASE, ROOT, MODELS, RECIPES, PROTOCOL, atomic_json, dataset,
                     digest, file_hash, formats, frozen_inputs, inventory, seal,
                     source_identity, unseal)


def now():
    return datetime.now(timezone.utc).isoformat()


def runtime(device):
    import torch
    import torchvision
    import PIL
    return {"device": device, "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "pillow": PIL.__version__, "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "capability": list(torch.cuda.get_device_capability(0)) if device == "cuda" else None,
            "deterministic": True, "tf32": False, "threads": 4,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def collect(model, graph, transform, provenance, device, progress):
    import torch
    from .classifier import ObservingInterpreter, image_batch
    record, rows, payload = dataset("imagenet_calibration_2k")
    identity = digest(provenance)
    directory = BASE / "calibration" / model / identity
    directory.mkdir(parents=True, exist_ok=True)
    seal(directory / "provenance.json", provenance)
    observer = ObservingInterpreter(graph)
    combined, maxima, channel_counts, files = {}, {}, None, []
    batch_size = PROTOCOL["calibration_batch_size"]
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        meta_path = directory / f"{start:05d}.json"
        npz_path = directory / f"{start:05d}.npz"
        # Input bytes are checked even when a saved calibration batch is reused.
        for row in batch:
            path = (payload / row["relative_path"]).resolve()
            if not path.is_relative_to(payload.resolve()) or file_hash(path) != row["sha256"]:
                raise ValueError("calibration payload drift")
        if meta_path.exists():
            meta = unseal(meta_path)
            if meta["calibration_sha256"] != identity or meta["samples"] != batch or file_hash(npz_path) != meta["arrays_sha256"]:
                raise ValueError("calibration checkpoint mismatch")
            with np.load(npz_path, allow_pickle=False) as saved:
                arrays = {key: saved[key] for key in saved.files}
        else:
            observer.ordinal = start
            observer.samples = {}
            observer.maxima = {}
            observer.channel_counts = {}
            inputs = image_batch(batch, payload, transform, device)
            with torch.inference_mode():
                observer.run(inputs)
            arrays = observer.samples
            temporary = npz_path.with_suffix(".partial")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(npz_path)
            meta = {"calibration_sha256": identity, "samples": batch, "arrays_sha256": file_hash(npz_path),
                    "maxima": observer.maxima, "channel_counts": observer.channel_counts}
            seal(meta_path, meta)
        if channel_counts is not None and meta["channel_counts"] != channel_counts:
            raise ValueError("calibration graph/channel coverage changed")
        channel_counts = meta["channel_counts"]
        if set(arrays) != set(channel_counts) or set(arrays) != set(meta["maxima"]):
            raise ValueError("calibration node coverage mismatch")
        for name, values in arrays.items():
            if values.ndim != 2 or values.shape[0] != len(batch) or not np.isfinite(values).all():
                raise ValueError("invalid saved calibration observations")
            combined.setdefault(name, []).append(values.ravel())
            maxima[name] = max(maxima.get(name, 0), meta["maxima"][name])
        files.append({"start": start, "sha256": file_hash(meta_path)})
        if start % 128 == 0 or start + len(batch) == len(rows):
            progress({"stage": "calibration", "model": model, "images": start+len(batch), "total": len(rows)})
    # The rotating schedule visits every channel even when a tensor has >256.
    if any(channels > len(rows) * 256 for channels in channel_counts.values()):
        raise ValueError("observer cannot cover every channel")
    arrays = {name: np.concatenate(chunks) for name, chunks in combined.items()}
    summary = {"identity": identity, "images": len(rows), "list_sha256": record["sha256"],
               "batch_checkpoints": files, "nodes": {k: {"sample_count": int(v.size), "channels": channel_counts[k],
                                                           "maxabs": maxima[k]} for k, v in arrays.items()}}
    seal(directory / "summary.json", summary)
    return arrays, maxima, {"identity": identity, "summary_sha256": digest(summary)}


def metrics(predictions, baseline):
    correct = sum(p["top5"][0] == int(p["sample"]["label"]) for p in predictions)
    top5 = sum(int(p["sample"]["label"]) in p["top5"] for p in predictions)
    baseline_correct = sum(p["top5"][0] == int(p["sample"]["label"]) for p in baseline)
    n = len(predictions)
    return {"images": n, "top1_percent": 100*correct/n, "top5_percent": 100*top5/n,
            "fp32_top1_percent": 100*baseline_correct/n, "delta_top1_pp": 100*(correct-baseline_correct)/n,
            "interpretation": "development_panel_descriptive_only"}


def evaluate(engine, transform, model, identity, device, limit, progress):
    import torch
    from .classifier import image_batch
    from .validation import verify_prediction
    _, rows, payload = dataset("imagenet_screen_1k")
    rows = rows[:limit]
    directory = BASE / "predictions" / identity
    directory.mkdir(parents=True, exist_ok=True)
    # Verify every panel payload even on a no-op resume.
    for row in rows:
        path = (payload / row["relative_path"]).resolve()
        if not path.is_relative_to(payload.resolve()) or file_hash(path) != row["sha256"]:
            raise ValueError("evaluation payload drift")
    predictions, elapsed, computed = [], 0.0, 0
    batch_size = PROTOCOL["inference_batch_size"]
    for start in range(0, limit, batch_size):
        batch = rows[start:start+batch_size]
        existing = {row["sha256"]: directory / (row["sha256"] + ".json") for row in batch}
        valid = {row["sha256"]: verify_prediction(existing[row["sha256"]], identity, row)
                 for row in batch if existing[row["sha256"]].exists()}
        if len(valid) != len(batch):
            # Recompute the original full batch after an interruption; unchanged
            # batch membership prevents resume-dependent convolution algorithms.
            tick = time.monotonic()
            inputs = image_batch(batch, payload, transform, device)
            with torch.inference_mode():
                output = engine.run(inputs) if hasattr(engine, "run") else engine(inputs)
            if not torch.isfinite(output).all() or tuple(output.shape) != (len(batch), 1000):
                raise ValueError("invalid classifier output")
            top5 = output.topk(5, dim=1).indices.cpu().tolist()
            duration = time.monotonic() - tick
            elapsed += duration
            computed += len(batch)
            for row, classes in zip(batch, top5):
                saved = {"configuration_sha256": identity, "sample": row, "top5": classes,
                         "batch_start": start, "batch_images": len(batch), "batch_seconds": duration}
                if row["sha256"] in valid:
                    if valid[row["sha256"]]["top5"] != classes:
                        raise ValueError("resumed batch changed saved predictions")
                else:
                    seal(existing[row["sha256"]], saved)
                    valid[row["sha256"]] = saved
        predictions.extend(valid[row["sha256"]] for row in batch)
        progress({"stage": "evaluation", "model": model, "configuration_sha256": identity,
                  "images": len(predictions), "total": limit})
    return predictions, {"computed_images_this_invocation": computed, "seconds_this_invocation": elapsed,
                         "images_per_hour_this_invocation": computed / elapsed * 3600 if elapsed else None}


def execute(args, plan):
    import torch
    from .classifier import configure, load_model, image_batch, prepare_qdq
    from .validation import validate_quantizers, validate_fold
    from tools.phase3.worker_locks import exclusive, pool_locks
    configure(args.device)
    env = runtime(args.device)
    source = plan["source_sha256"]
    state = {"version": PROTOCOL["version"], "started_at": now(), "pid": os.getpid(), "runtime": env,
             "source_sha256": source, "protocol_sha256": digest(PROTOCOL), "status": "starting",
             "tasks": [{**row, "status": "blocked" if row["blocked_reason"] else "pending"}
                       for row in plan["configurations"]]}
    def persist(extra=None):
        if extra:
            state["progress"] = extra
        state["updated_at"] = now()
        state["counts"] = dict(Counter(t["status"] for t in state["tasks"]))
        atomic_json(BASE / "status.json", state)
    def progress(info):
        persist(info)
        print(json.dumps(info, sort_keys=True), flush=True)
    cache = {}
    done = 0
    with exclusive(BASE / "controller.lock"), pool_locks(ROOT):
        persist()
        try:
            checks = validate_quantizers(args.device)
            seal(BASE / "validation" / f"{digest({'source':source,'runtime':env})}.json", checks)
            state["quantizer_validation"] = checks
            state["status"] = "running"
            # Round-robin model coverage before deeper format exploration.
            priority = {row["name"]: i for i, row in enumerate(formats())}
            tasks = sorted(state["tasks"], key=lambda t: (priority[t['format']], MODELS.index(t['model']), RECIPES.index(t['recipe'])))
            for task in tasks:
                if task["status"] == "blocked" or args.models and task["model"] not in args.models or args.formats and task["format"] not in args.formats:
                    continue
                if args.limit is not None and done >= args.limit:
                    break
                if source_identity() != source:
                    raise ValueError("execution source changed during campaign")
                task["status"] = "running"
                persist()
                model = task["model"]
                if model not in cache:
                    context = frozen_inputs(model)
                    graph, transform, original = load_model(model, args.device)
                    _, rows, payload = dataset("imagenet_screen_1k")
                    check = validate_fold(graph, original, image_batch(rows[:8], payload, transform, args.device))
                    del original
                    provenance = {"context": context, "source_sha256": source, "runtime": env,
                                  "protocol": PROTOCOL, "graph": str(graph.graph)}
                    arrays, maxima, calibration = collect(model, graph, transform, provenance, args.device, progress)
                    baseline_id = digest({"provenance": provenance, "mode": "folded_fp32_baseline"})
                    baseline, _ = evaluate(graph, transform, model, baseline_id, args.device, args.images, progress)
                    cache[model] = (context, graph, transform, arrays, maxima, calibration, baseline, baseline_id, check)
                context, graph, transform, arrays, maxima, calibration, baseline, baseline_id, check = cache[model]
                tick = time.monotonic()
                with torch.inference_mode():
                    engine, scales = prepare_qdq(graph, task["format"], task["recipe"], arrays, maxima, args.device)
                configuration = {"model_context": context, "format": task["format"], "format_sha256": task["format_sha256"],
                                 "recipe": task["recipe"], "protocol": PROTOCOL, "runtime": env, "source_sha256": source,
                                 "calibration": calibration, "scales": scales, "baseline_sha256": baseline_id}
                identity = digest(configuration)
                seal(BASE / "configurations" / f"{identity}.json", configuration)
                task["configuration_sha256"] = identity
                task["preparation_seconds"] = time.monotonic() - tick
                predictions, timing = evaluate(engine, transform, model, identity, args.device, args.images, progress)
                summary = {"configuration_sha256": identity, "panel_images": args.images,
                           "metrics": metrics(predictions, baseline), "timing": timing,
                           "fold_validation": check, "native_acceptance": False,
                           "prediction_digest": digest(predictions), "baseline_prediction_digest": digest(baseline)}
                if source_identity() != source or frozen_inputs(model) != context:
                    raise ValueError("execution inputs/source changed before result commit")
                seal(BASE / "summaries" / f"{identity}-{args.images}.json", summary)
                task.update(status="completed", metrics=summary["metrics"], timing=timing)
                progress({"stage": "configuration_completed", "model": model, "format": task["format"],
                          "recipe": task["recipe"], **summary["metrics"], **timing})
                del engine
                done += 1
            state["status"] = "supported_subset_finished" if not any(t['status'] == 'pending' for t in state['tasks']) else "invocation_finished"
            state["all_coverage_complete"] = all(t['status'] == 'completed' for t in state['tasks'])
            persist()
        except (KeyboardInterrupt, InterruptedError):
            for task in state["tasks"]:
                if task["status"] == "running":
                    task["status"] = "interrupted"
            state["status"] = "interrupted"
            persist()
            print("Stopped; completed calibration batches and image predictions are retained.", flush=True)
            return 130
        except Exception as error:
            for task in state["tasks"]:
                if task["status"] == "running":
                    task["status"] = "failed"
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            persist()
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
        print(json.dumps({k:v for k,v in state.items() if k not in ('tasks','quantizer_validation')}, indent=2))
        return 0
    plan = inventory()
    if args.formats and set(args.formats) - {row['name'] for row in formats()}:
        parser.error("unknown format selection")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.run and not any(not t['blocked_reason'] and (not args.models or t['model'] in args.models)
                            and (not args.formats or t['format'] in args.formats) for t in plan['configurations']):
        parser.error("selection contains no implemented configurations; inspect inventory blocked reasons")
    seal(BASE / "inventory.json", plan)
    if args.prepare:
        print(json.dumps({"configurations": len(plan['configurations']),
                          "implementation_status": dict(Counter(r['implementation_status'] for r in plan['configurations'])),
                          "inventory": str(BASE / 'inventory.json')}, indent=2))
        return 0
    def stop(_signal, _frame):
        raise InterruptedError("stop requested")
    signal.signal(signal.SIGTERM, stop)
    return execute(args, plan)
