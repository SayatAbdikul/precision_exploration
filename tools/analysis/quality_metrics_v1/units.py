"""Classifier units (matrix cells, AdaRound arms, accumulator-sweep policies) as per-image arrays against FP32.

Every unit becomes a dict of per-image arrays on the 1k screen rows of its network:

- ``label``; ``credit_expected`` / ``credit_lowest`` / ``credit_topk`` (top-1 credit under each readout);
- ``argmax_lowest`` (-1 where not determinable), ``top5`` (n x 5), ``tie_size``;
- ``in_tie``: 1 / 0 / -1 (FP32 top-1 class c0 is in the quantized top tie set / is not / undetermined).

Tie-set rules (protocol part A, ``tie_set``) are implemented in ``matrix_membership``, ``topk_membership``.
"""
from __future__ import annotations

import json

import numpy as np

from tools.experiment_b.common import ROOT

MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large")
RECON = ROOT / "artifacts/experiment_b2_recon/evals"
INTRINSIC_OF = {"cum5_act_maxabs": "default", "cum1_fused": "minimal"}


def matrix_membership(c0, top5_lowest, tie_size):
    """c0 in T for a lowest-index top-5 list: exact when k <= 5; when k > 5 the list holds the five lowest members."""
    top5 = np.asarray(top5_lowest, dtype=np.int64)
    k = np.asarray(tie_size, dtype=np.int64)
    c0 = np.asarray(c0, dtype=np.int64)
    columns = np.arange(5)[None, :]
    listed_in_tie = ((top5 == c0[:, None]) & (columns < k[:, None])).any(axis=1)
    result = listed_in_tie.astype(np.int8)
    big = k > 5
    unknown = big & ~listed_in_tie & (c0 > top5[:, 4])
    result[unknown] = -1
    return result


def topk_membership(c0, top5_topk, tie_size):
    """c0 in T for a stored CUDA topk list (unspecified order among equal values): exact when k <= 5."""
    top5 = np.asarray(top5_topk, dtype=np.int64)
    k = np.asarray(tie_size, dtype=np.int64)
    c0 = np.asarray(c0, dtype=np.int64)
    columns = np.arange(5)[None, :]
    listed_in_tie = ((top5 == c0[:, None]) & (columns < k[:, None])).any(axis=1)
    result = listed_in_tie.astype(np.int8)
    result[(k > 5) & ~listed_in_tie] = -1
    return result


def topk_lowest_argmax(top5_topk, tie_size):
    """Lowest-index argmax from a topk list: min of the first k entries when k <= 5, else -1."""
    top5 = np.asarray(top5_topk, dtype=np.int64)
    k = np.asarray(tie_size, dtype=np.int64)
    masked = np.where(np.arange(5)[None, :] < np.minimum(k, 5)[:, None], top5, 10**9)
    result = masked.min(axis=1)
    result[k > 5] = -1
    return result


def matrix_units():
    """All 177 matrix cell records (including FP32 baselines) as units; checks via b2_matrix.load_cells."""
    from tools.analysis.b2_matrix import load_cells
    from tools.experiment_b2 import readout
    units = {}
    for (model, fmt, recipe), cell in load_cells().items():
        a = cell["arrays"]
        credit = readout.credits(a)
        group = INTRINSIC_OF.get(recipe, recipe)
        name = fmt if recipe in ("default", "minimal", "baseline") else f"{fmt}@{recipe}"
        units[("matrix", model, group, name)] = {
            "source": "matrix", "model": model, "group": group, "name": name, "format": fmt, "recipe": recipe,
            "family": cell["record"]["family"], "bits": cell["record"]["bits"],
            "label": a["label"].astype(np.int64), "tie_size": a["tie_size"].astype(np.int64),
            "argmax_lowest": a["argmax_lowest"].astype(np.int64), "top5": a["top5_lowest"].astype(np.int64),
            "top5_topk": a["top5_topk"].astype(np.int64),
            "credit_expected": credit["top1_expected"], "credit_lowest": credit["top1_lowest_index"],
            "credit_topk": credit["top1_topk"], "rows_sha256": cell["record"]["rows_sha256"],
            "record": str(cell["record"]["readout_file"]),
        }
    return units


def fp32_reference(units, model):
    ref = units[("matrix", model, "baseline", "fp32")]
    if (ref["tie_size"] != 1).any():
        raise ValueError(f"FP32 {model} has a tied top-1; the agreement definitions assume a unique c0")
    return ref


def recon_units(labels_by_model, rows_by_model):
    """AdaRound evaluation arms (and their FP32 evals) as units; labels come from the matrix FP32 cell."""
    units = {}
    for model in MODELS:
        for path in sorted((RECON / model).glob("*.npz")):
            record = json.loads(path.with_suffix(".json").read_text())
            payload = record.get("payload", record)
            if payload["images"] != 1000 or payload["rows_sha256"] != rows_by_model[model]:
                raise ValueError(f"AdaRound eval not on the matrix rows: {path}")
            with np.load(path) as saved:
                a = {k: saved[k] for k in saved.files}
            label = labels_by_model[model]
            k = a["tie_size"].astype(np.int64)
            among = a["label_among_maxima"].astype(np.float64)
            top5 = a["top5_topk"].astype(np.int64)
            if not ((top5[:, 0] == label) <= (among > 0)).all():
                raise ValueError(f"topk credits a label outside the maxima: {path}")
            arm = path.stem
            units[("recon", model, "adaround", arm)] = {
                "source": "recon", "model": model, "group": "adaround", "name": arm, "format": payload.get("weight_format"),
                "recipe": payload.get("recipe_name"), "family": None, "bits": None,
                "label": label, "tie_size": k, "argmax_lowest": topk_lowest_argmax(top5, k), "top5": top5,
                "top5_topk": top5, "credit_expected": among / k,
                "credit_lowest": a["label_is_lowest_index_maximum"].astype(np.float64),
                "credit_topk": (top5[:, 0] == label).astype(np.float64), "rows_sha256": payload["rows_sha256"],
                "record": str(path.relative_to(ROOT)), "topk_list": True,
            }
    return units


def sweep_units(label_resnet18):
    """Accumulator-sweep policy rows at 1,000 images (ResNet18), exact tie lists."""
    import csv
    from tools.accumulator_sweep_v1.load import assemble
    rows = list(csv.DictReader((ROOT / "results/summaries/accumulator-sweep-v1/policies.csv").open()))
    units = {}
    for row in rows:
        if row["images"] != "1000":
            continue
        case, policy = row["case"], row["policy"]
        data = assemble(case, policy, 0, 1000)
        images = data["images"]
        label = np.array([r["label"] for r in images], dtype=np.int64)
        if not np.array_equal(label, label_resnet18):
            raise ValueError(f"sweep rows are not in matrix order: {case} {policy}")
        n = len(images)
        failed = np.array([r.get("failure") is not None for r in images])
        top5 = np.full((n, 5), -1, dtype=np.int64)
        k = np.ones(n, dtype=np.int64)
        expected = np.zeros(n)
        ties = []
        for i, r in enumerate(images):
            if failed[i]:
                ties.append(())
                continue
            t = r["top1_tied_classes"]
            if r["top5"][0] != min(t):
                raise ValueError("sweep top-5 not in contract tie order")
            top5[i] = r["top5"]
            k[i] = len(t)
            expected[i] = (1.0 / len(t)) if r["label"] in t else 0.0
            ties.append(tuple(t))
        unit = {"source": "sweep", "model": "resnet18", "group": case, "name": policy, "format": case.split("-")[1],
                "recipe": case, "family": None, "bits": None, "label": label, "tie_size": k,
                "argmax_lowest": top5[:, 0].copy(), "top5": top5, "top5_topk": top5,
                "credit_expected": expected, "credit_lowest": (top5[:, 0] == label).astype(np.float64),
                "credit_topk": (top5[:, 0] == label).astype(np.float64), "ties": ties, "failed": failed,
                "record": f"{case}/{policy}", "summary_top1_expected": float(row["top1_expected"])}
        if abs(100 * expected.mean() - float(row["top1_expected"])) > 1e-9:
            raise ValueError(f"sweep scores do not reproduce policies.csv: {case} {policy}")
        units[("sweep", "resnet18", case, policy)] = unit
    return units


def membership(unit, c0):
    if unit["source"] == "matrix":
        return matrix_membership(c0, unit["top5"], unit["tie_size"])
    if unit["source"] == "recon":
        return topk_membership(c0, unit["top5"], unit["tie_size"])
    result = np.array([1 if c in t else 0 for c, t in zip(c0, unit["ties"])], dtype=np.int8)
    return result


def load_all(with_sweep=True):
    units = matrix_units()
    labels = {m: units[("matrix", m, "baseline", "fp32")]["label"] for m in MODELS}
    rows = {m: units[("matrix", m, "baseline", "fp32")]["rows_sha256"] for m in MODELS}
    units.update(recon_units(labels, rows))
    if with_sweep:
        units.update(sweep_units(labels["resnet18"]))
    for model in MODELS:
        fp32_reference(units, model)
    for key, unit in units.items():
        c0 = units[("matrix", unit["model"], "baseline", "fp32")]["argmax_lowest"]
        unit["c0"] = c0
        unit["in_tie"] = membership(unit, c0)
    return units
