"""Count log residual rounding discrepancies on retained FP32 branch samples."""
from fractions import Fraction
from itertools import product
from time import perf_counter
import fcntl
import os

import numpy as np

from public.inference.conformance_job import source_identity
from public.inference.reference.arithmetic import binary, format_named
from public.inference.tensor import Tensor, parse_encoding
from public.quantization.ptq.encoding import float_tensor
from tools.phase3.baselines import screen_rows
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference
from tools.phase3.runtime import load_runtime


def rounding_table(encoding):
    fmt = format_named(encoding.format)
    if encoding.axis is not None or encoding.block_size is not None:
        raise ValueError("rounding frequency requires ordinary scalar encoding")
    codes = tuple(range(1 << fmt.bits))
    values = Tensor((len(codes),), codes, encoding).values()
    if any(not fmt.decode(c).is_finite() for c in codes):
        raise ValueError("frequency table requires all codes finite")
    exact = [binary(a, b, "add") for a, b in product(values, repeat=2)]
    acc = format_named("fp64_e11m52_accumulator")
    rounded = [acc.rounded(v) for v in exact]
    expected = np.asarray(Tensor.quantize(exact, (len(exact),), encoding).codes, dtype=np.uint8)
    actual = np.asarray(Tensor.quantize(rounded, (len(rounded),), encoding).codes, dtype=np.uint8)
    return expected.reshape(len(codes), len(codes)), actual.reshape(len(codes), len(codes))


def count_pairs(left_codes, right_codes, expected, actual):
    left, right = np.asarray(left_codes, dtype=np.int64), np.asarray(right_codes, dtype=np.int64)
    if left.shape != right.shape or expected.shape != actual.shape or expected.shape[0] != expected.shape[1]:
        raise ValueError("residual pair/table shape mismatch")
    size = expected.shape[0]
    if np.any(left < 0) or np.any(right < 0) or np.any(left >= size) or np.any(right >= size):
        raise ValueError("residual code outside table")
    counts = np.bincount((left*size+right).ravel(), minlength=size*size).reshape(size, size)
    mismatch = expected != actual
    selected = np.argwhere((counts > 0) & mismatch)
    return {"elements": int(left.size), "stored_output_discrepancies": int(counts[mismatch].sum()),
            "observed_pairs": int(np.count_nonzero(counts)),
            "discrepant_pairs": [{"left_code": int(a), "right_code": int(b), "count": int(counts[a, b]),
                                  "exact_output_code": int(expected[a, b]), "fp64_output_code": int(actual[a, b])}
                                 for a, b in selected]}


def main():
    plan, source = campaign(), source_identity()
    model, target = "resnet18", "add"
    inventory = preparation_records()
    candidates = []
    for name in ("log6", "log8"):
        prepared = inventory[f"{model}/{name}"]
        config = read(checked(prepared["configuration"]))
        if config != configuration(model, name, plan) or config["runtime"]["source_sha256"] != source:
            raise ValueError("rounding frequency needs current preparation")
        graph = read(checked(prepared["graph"]))
        domains = dict(graph["inputs"])
        node = None
        for item in graph["nodes"]:
            if item["name"] == target:
                node = item
                break
            domains[item["name"]] = item["attrs"].get("output", domains[item["inputs"][0]])
        if (node is None or node["op"] != "elementwise" or node["attrs"].get("operation") != "add"
                or node["attrs"]["accumulator"] != "fp64_e11m52_accumulator"
                or node["attrs"]["alignment"] != node["attrs"]["output"]):
            raise ValueError("unexpected residual arithmetic domain")
        candidates.append({"format": name, "configuration": prepared["configuration"], "graph": prepared["graph"],
                           "node": node, "inputs": {n: domains[n] for n in node["inputs"]}})
    names = set(candidates[0]["node"]["inputs"])
    if any(set(c["node"]["inputs"]) != names for c in candidates):
        raise ValueError("candidate branches do not align")
    implementations = [source_reference(ROOT / p) for p in ("tools/run/phase3_residual_rounding_frequency.py", "tools/phase3/runtime.py")]
    job = {"schema_version": "phase3-residual-rounding-frequency-1.0.0", "scope": "diagnostic_fp32_branch_code_frequency",
           "model": model, "target": target, "images": 8, "source_sha256": source, "campaign_sha256": digest(plan),
           "candidates": candidates, "implementations": implementations}
    identity = digest(job)
    work = ROOT / "artifacts/phase3/residual-rounding-frequency" / identity
    lock_path = ROOT / "artifacts/phase3/locks/cpu-rounding-frequency.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(work / "job.json", job)
        tables = {c["format"]: rounding_table(parse_encoding(c["node"]["attrs"]["alignment"])) for c in candidates}
        population, payload, _ = screen_rows(model, plan, ROOT)
        sample, observe, _ = load_runtime(model, plan, ROOT)
        class Capture:
            def __init__(self):
                self.values = {}
            def record(self, name, value):
                if name in names:
                    self.values[name] = value.detach().cpu().contiguous().numpy().copy()
        image_refs = []
        totals = {c["format"]: {"elements": 0, "stored_output_discrepancies": 0} for c in candidates}
        for ordinal, row in enumerate(population[:8]):
            path = work / f"{row['sha256']}.json"
            if path.exists():
                record = read(path)
                if record["job_sha256"] != identity or record["sample"] != row or digest({k:v for k,v in record.items() if k != "record_sha256"}) != record["record_sha256"]:
                    raise ValueError("rounding frequency checkpoint identity mismatch")
                checked(record["tensors"])
            else:
                started = perf_counter()
                inputs, _ = sample(payload / row["relative_path"])
                capture = Capture()
                observe(inputs, capture)
                if set(capture.values) != names:
                    raise ValueError("FP32 branch capture is incomplete")
                outcomes, arrays = [], {}
                for candidate in candidates:
                    alignment = parse_encoding(candidate["node"]["attrs"]["alignment"])
                    branches = []
                    for name in candidate["node"]["inputs"]:
                        stored = float_tensor(capture.values[name], parse_encoding(candidate["inputs"][name]))
                        branches.append(Tensor.quantize(stored.values(), stored.shape, alignment))
                    left, right = branches
                    expected, actual = tables[candidate["format"]]
                    outcomes.append({"format": candidate["format"], **count_pairs(left.codes, right.codes, expected, actual)})
                    arrays[candidate["format"]+"_left"] = np.asarray(left.codes, dtype=np.uint8).reshape(left.shape)
                    arrays[candidate["format"]+"_right"] = np.asarray(right.codes, dtype=np.uint8).reshape(right.shape)
                    arrays[candidate["format"]+"_exact_table"] = expected
                    arrays[candidate["format"]+"_fp64_table"] = actual
                tensors = work / f"{row['sha256']}.npz"
                partial = tensors.with_suffix(".partial")
                with partial.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                os.replace(partial, tensors)
                record = {"job_sha256": identity, "sample": row, "outcomes": outcomes, "tensors": reference(tensors),
                          "seconds": perf_counter()-started}
                record["record_sha256"] = digest(record)
                write(path, record)
                print(ordinal+1, outcomes, flush=True)
            for outcome in record["outcomes"]:
                for key in ("elements", "stored_output_discrepancies"):
                    totals[outcome["format"]][key] += outcome[key]
            if source_identity() != source:
                raise ValueError("engine changed during rounding frequency study")
            image_refs.append(reference(path))
        report = {"schema_version": "phase3-residual-rounding-frequency-summary-1.0.0", "status": "completed",
                  "scope": job["scope"], "job": reference(work / "job.json"), "images": len(image_refs),
                  "classification": "DIAGNOSTIC_ONLY", "image_records": image_refs, "totals": totals,
                  "limits": ["FP32 branches stored/aligned in the candidate format; not strict-graph activation frequencies",
                             "compares declared FP64 addition with no accumulation rounding on the same stored inputs",
                             "only the first ResNet residual and eight frozen images; no downstream prediction or quality claim",
                             "oracle-defined finite values and unchanged output quantization; not exact irrational logarithms"]}
        write(work / "summary.json", report)
        write(ROOT / "results/summaries/phase3-residual-rounding-frequency.json", report)
        print(totals, flush=True)


if __name__ == "__main__":
    main()
