"""Propagate retained FP64/unrounded residual alternatives through an FP32 tail."""
import fcntl
import hashlib
from time import perf_counter

import numpy as np

from public.analysis.phase3.statistics import paired_classification
from public.inference.conformance_job import source_identity
from public.inference.tensor import Tensor, parse_encoding
from public.quantization.ptq.encoding import float_tensor
from tools.phase3.baselines import screen_rows
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.provenance import source_reference
from tools.phase3.runtime import load_runtime
from tools.phase3.sensitivity import top5


def array_hash(value):
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes()).hexdigest()


class ReplaceResidual:
    def __init__(self, target, original_hash, replacement):
        self.target, self.original_hash = target, original_hash
        self.replacement, self.calls = np.asarray(replacement, dtype=np.float32), 0

    def record(self, name, value):
        if name == self.target:
            import torch
            if self.calls or tuple(value.shape) != self.replacement.shape or array_hash(value.detach().cpu().numpy()) != self.original_hash:
                raise ValueError("FP32 residual differs before intervention or target executes twice")
            value.copy_(torch.from_numpy(self.replacement).to(device=value.device))
            self.calls += 1


def main():
    plan, source = campaign(), source_identity()
    frequency_ref = reference(ROOT / "results/summaries/phase3-residual-rounding-frequency.json")
    frequency = read(checked(frequency_ref))
    previous = read(checked(frequency["job"]))
    if (frequency["status"] != "completed" or frequency["images"] != 8 or len(frequency["image_records"]) != 8
            or previous["source_sha256"] != source or previous["campaign_sha256"] != digest(plan)):
        raise ValueError("propagation requires complete current eight-image branch evidence")
    candidates, model, target = previous["candidates"], previous["model"], previous["target"]
    for candidate in candidates:
        config = read(checked(candidate["configuration"]))
        checked(candidate["graph"])
        if config != configuration(model, candidate["format"], plan):
            raise ValueError("retained branch encoding belongs to a stale configuration")
    names = set(candidates[0]["node"]["inputs"])
    baseline_ref = read(ROOT / "results/summaries/phase3-baselines.json")["models"][model]
    baseline = read(checked(baseline_ref))
    job = {"schema_version": "phase3-residual-rounding-propagation-1.0.0", "scope": "diagnostic_one_residual_precision",
           "source_sha256": source, "campaign_sha256": digest(plan), "frequency_evidence": frequency_ref,
           "baseline": baseline_ref, "images": 8, "model": model, "target": target,
           "implementations": [source_reference(ROOT / p) for p in (
               "tools/run/phase3_residual_rounding_propagation.py", "tools/phase3/runtime.py", "tools/phase3/sensitivity.py",
               "public/analysis/phase3/statistics.py")]}
    identity = digest(job)
    work = ROOT / "artifacts/phase3/residual-rounding-propagation" / identity
    lock_path = ROOT / "artifacts/phase3/locks/cpu-rounding-propagation.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(work / "job.json", job)
        population, payload, _ = screen_rows(model, plan, ROOT)
        sample, observe, _ = load_runtime(model, plan, ROOT)
        class Capture:
            def __init__(self):
                self.values = {}
            def record(self, name, value):
                if name in names or name == target:
                    self.values[name] = value.detach().cpu().contiguous().numpy().copy()
        records, image_refs = [], []
        for ordinal, (row, previous_ref) in enumerate(zip(population[:8], frequency["image_records"])):
            old = read(checked(previous_ref))
            if old["sample"] != row or old["job_sha256"] != digest(previous) or digest({k:v for k,v in old.items() if k != "record_sha256"}) != old["record_sha256"]:
                raise ValueError("retained branch image identity mismatch")
            tensors_path = checked(old["tensors"])
            path = work / f"{row['sha256']}.json"
            if path.exists():
                record = read(path)
                if record["job_sha256"] != identity or record["sample"] != row or digest({k:v for k,v in record.items() if k != "record_sha256"}) != record["record_sha256"]:
                    raise ValueError("propagation checkpoint identity mismatch")
            else:
                started = perf_counter()
                inputs, _ = sample(payload / row["relative_path"])
                captured = Capture()
                fp32 = top5(observe(inputs, captured))
                if set(captured.values) != names | {target}:
                    raise ValueError("fresh branch capture is incomplete")
                original_hash = array_hash(captured.values[target])
                outcomes = {}
                with np.load(tensors_path, allow_pickle=False) as tensors:
                    for candidate in candidates:
                        fmt = candidate["format"]
                        alignment = parse_encoding(candidate["node"]["attrs"]["alignment"])
                        # Verify saved branches against this fresh FP32 pass before
                        # propagating either precision alternative.
                        for side, name in zip(("left", "right"), candidate["node"]["inputs"]):
                            stored = float_tensor(captured.values[name], parse_encoding(candidate["inputs"][name]))
                            aligned = Tensor.quantize(stored.values(), stored.shape, alignment)
                            if not np.array_equal(np.asarray(aligned.codes, dtype=np.uint8).reshape(aligned.shape), tensors[fmt+"_"+side]):
                                raise ValueError("fresh encoded branch differs from retained frequency evidence")
                        size = tensors[fmt+"_exact_table"].shape[0]
                        decoded = np.asarray([float(v) for v in Tensor((size,), tuple(range(size)), alignment).values()], dtype=np.float32)
                        for policy in ("exact", "fp64"):
                            codes = tensors[fmt+"_"+policy+"_table"][tensors[fmt+"_left"], tensors[fmt+"_right"]]
                            replacement = decoded[codes]
                            intervention = ReplaceResidual(target, original_hash, replacement)
                            prediction = top5(observe(inputs, intervention))
                            if intervention.calls != 1:
                                raise ValueError("residual intervention was not executed")
                            outcomes[fmt+"_"+policy] = {"prediction": prediction, "replacement_sha256": array_hash(replacement)}
                record = {"job_sha256": identity, "sample": row, "branch_evidence": previous_ref,
                          "fp32_prediction": fp32, "frozen_reference": baseline["records"][ordinal],
                          "fresh_branch_codes_verified": True, "outcomes": outcomes, "seconds": perf_counter()-started}
                record["record_sha256"] = digest(record)
                write(path, record)
                print(ordinal+1, {k:v["prediction"] for k,v in outcomes.items()}, flush=True)
            if source_identity() != source:
                raise ValueError("engine changed during residual propagation")
            records.append(record)
            image_refs.append(reference(path))
        options = {k:plan["statistics"][k] for k in ("confidence", "seed")}
        analyses = {}
        for alternative in records[0]["outcomes"]:
            pairs = [{**r["frozen_reference"], "fp32_prediction": r["fp32_prediction"],
                      "candidate_prediction": r["outcomes"][alternative]["prediction"]} for r in records]
            analyses[alternative] = paired_classification(pairs, [r["sample_id"] for r in pairs],
                resamples=plan["statistics"]["classification_resamples"], **options)
        report = {"schema_version": "phase3-residual-rounding-propagation-summary-1.0.0", "status": "completed",
                  "scope": job["scope"], "job": reference(work / "job.json"), "images": len(records),
                  "classification": "DIAGNOSTIC_ONLY", "image_records": image_refs, "statistics": analyses,
                  "top5_list_precision_matches": {c["format"]: sum(r["outcomes"][c["format"]+"_exact"]["prediction"] ==
                      r["outcomes"][c["format"]+"_fp64"]["prediction"] for r in records) for c in candidates},
                  "limits": ["one residual in otherwise FP32 execution; not strict end-to-end quality or graph acceptance",
                             "both alternatives retain identical encoded/aligned inputs and prescribed output stores",
                             "the exact alternative removes only residual accumulation rounding, using oracle-defined finite values",
                             "eight selected diagnostic images; all quality deltas use the same fresh FP32 reference"]}
        write(work / "summary.json", report)
        write(ROOT / "results/summaries/phase3-residual-rounding-propagation.json", report)
        print(report["top5_list_precision_matches"], flush=True)


if __name__ == "__main__":
    main()
