"""Calibration subsets as unions of whole v1 observation batches, and the range statistics of a subset.

The v1 observation cache stores, per batch of 8 consecutive rows of the sha256-sorted calibration list,
256 sampled values per image per quantizing node and the exact max-abs of the batch.  A subset is a set
of whole batches; its range statistics are computed exactly as ``tools.experiment_b2.data.v1_calibration``
computes them for the full list, restricted to the selected batches (samples concatenated in ascending
batch order, maxima the maximum over the selected batches starting from 0).  With all 250 batches the
result is bit-identical to ``v1_calibration`` (tests/unit/test_experiment_b2_seeds.py).
"""
from __future__ import annotations

import numpy as np

from tools.experiment_b.common import ROOT, dataset, digest, file_hash, unseal
from tools.experiment_b import common as v1

PROTOCOL_FILE = ROOT / "public/experiments/configs/breadth-study/b2-seeds-protocol-v1.json"
LIST = "imagenet_calibration_2k"
BATCH = 8
BATCHES = 250
SEED_RNG = 20261002
HALF_RNG = 20261003
SEEDS = 5
SEED_BATCHES = 50
BIAS_IMAGES = 256
LADDER_PREFIX = {"L32": 4, "L128": 16}
LADDER_REPLICATES = 3


def permutation(seed=SEED_RNG):
    return np.random.default_rng(seed).permutation(BATCHES)


def definitions():
    """Every subset of the protocol: name -> {"range_batches", "bias_batches", "kind"} (batch indices, sorted)."""
    perm = permutation()
    half = permutation(HALF_RNG)
    full = list(range(BATCHES))
    out = {"FULL": {"kind": "full", "range_batches": full, "bias_batches": full}}
    for k in range(SEEDS):
        segment = perm[SEED_BATCHES * k:SEED_BATCHES * (k + 1)]
        chosen = sorted(int(x) for x in segment)
        out[f"S{k}"] = {"kind": "seed", "range_batches": chosen, "bias_batches": chosen}
        out[f"B{k}"] = {"kind": "bias_only", "range_batches": full, "bias_batches": chosen}
        # Addendum 1 (range-only): ranges from S_k, bias correction from the original first 256 images.
        out[f"R{k}"] = {"kind": "range_only", "range_batches": chosen, "bias_batches": full}
        if k < LADDER_REPLICATES:
            for name, count in LADDER_PREFIX.items():
                nested = sorted(int(x) for x in segment[:count])
                out[f"{name}_{k}"] = {"kind": "ladder", "range_batches": nested, "bias_batches": nested}
    for j, part in enumerate((half[:BATCHES // 2], half[BATCHES // 2:])):
        chosen = sorted(int(x) for x in part)
        out[f"H{j}"] = {"kind": "ladder", "range_batches": chosen, "bias_batches": chosen}
    return out


def subset_rows(rows, batches):
    """Rows of the given batches, in list (ascending sha256) order."""
    return [row for b in sorted(batches) for row in rows[b * BATCH:(b + 1) * BATCH]]


def bias_rows(rows, batches, limit=BIAS_IMAGES):
    """First ``limit`` images of the subset in ascending sha256 order (the original rule applied to the subset)."""
    return subset_rows(rows, batches)[:limit]


def describe(name, rows):
    spec = definitions()[name]
    ranges, bias = subset_rows(rows, spec["range_batches"]), bias_rows(rows, spec["bias_batches"])
    return {"name": name, "kind": spec["kind"], "protocol_sha256": file_hash(PROTOCOL_FILE),
            "range_images": len(ranges), "range_batches": spec["range_batches"],
            "range_rows_sha256": digest(ranges), "bias_images": len(bias), "bias_rows_sha256": digest(bias)}


class Observations:
    """Verified per-batch view of one model's v1 observation cache (same checks as ``v1_calibration``)."""

    def __init__(self, model):
        record, rows, _ = dataset(LIST)
        folder = ROOT / "artifacts/experiment_b/calibration" / model
        matching = []
        for candidate in (p for p in folder.iterdir() if p.is_dir()):
            provenance = unseal(candidate / "provenance.json")
            if (digest(provenance) == candidate.name and provenance["source_sha256"] == v1.source_identity()
                    and provenance["protocol"] == v1.PROTOCOL and provenance["context"] == v1.frozen_inputs(model)):
                matching.append((candidate, provenance))
        if len(matching) != 1:
            raise ValueError(f"expected exactly one current v1 calibration cache for {model}, found {len(matching)}")
        folder, provenance = matching[0]
        summary = unseal(folder / "summary.json")
        if summary["identity"] != folder.name or summary["images"] != len(rows) or summary["list_sha256"] != record["sha256"]:
            raise ValueError("v1 calibration summary does not match the frozen calibration list")
        if v1.PROTOCOL["calibration_batch_size"] != BATCH or len(rows) != BATCH * BATCHES:
            raise ValueError("unexpected calibration batching")
        self.model, self.rows, self.batches, self.maxima = model, rows, [], []
        for b in range(BATCHES):
            start = b * BATCH
            meta = unseal(folder / f"{start:05d}.json")
            npz = folder / f"{start:05d}.npz"
            if (meta["calibration_sha256"] != folder.name or meta["samples"] != rows[start:start + BATCH]
                    or file_hash(npz) != meta["arrays_sha256"]):
                raise ValueError("v1 calibration checkpoint mismatch")
            with np.load(npz, allow_pickle=False) as saved:
                self.batches.append({key: saved[key].ravel() for key in saved.files})
            self.maxima.append(dict(meta["maxima"]))
        self.identity = {"identity": folder.name, "summary_sha256": digest(summary), "graph": provenance["graph"],
                         "runtime": provenance["runtime"], "reused_from": "artifacts/experiment_b/calibration (read-only)"}
        self.summary = summary

    def ranges(self, batches):
        """``(arrays, maxima)`` of the selected batches, computed as ``v1_calibration`` does for all of them."""
        batches = sorted(int(b) for b in batches)
        if len(set(batches)) != len(batches) or not batches or batches[0] < 0 or batches[-1] >= BATCHES:
            raise ValueError("invalid batch selection")
        combined, maxima = {}, {}
        for b in batches:
            for key, values in self.batches[b].items():
                combined.setdefault(key, []).append(values)
            for key, value in self.maxima[b].items():
                maxima[key] = max(maxima.get(key, 0), value)
        arrays = {key: np.concatenate(chunks) for key, chunks in combined.items()}
        if len(batches) == BATCHES:
            for key, values in arrays.items():
                node = self.summary["nodes"][key]
                if node["sample_count"] != values.size or node["maxabs"] != maxima[key]:
                    raise ValueError("v1 calibration arrays do not reproduce the sealed summary")
        return arrays, maxima

    def calibration(self, name):
        """``(arrays, maxima, identity)`` for a named subset; the full set keeps the v1 identity unchanged."""
        spec = definitions()[name]
        arrays, maxima = self.ranges(spec["range_batches"])
        if len(spec["range_batches"]) == BATCHES:
            identity = dict(self.identity)
        else:
            identity = {**self.identity, "b2_seeds_subset": describe(name, self.rows)}
        return arrays, maxima, identity
