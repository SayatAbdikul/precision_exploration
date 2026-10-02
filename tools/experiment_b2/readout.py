"""Tie-aware readout of quantized logits (B2 matrix stage, 2026-10-01).

Quantized logits tie, and ``topk`` picks among equal values in an unspecified
way.  From this stage on every B2 figure is read out with two fixed rules:

* primary, *expected credit*: top-1 credit ``1/k`` when the label is among the
  ``k`` classes that share the maximum (the mean over uniformly random
  tie-breaking); top-5 credit is the probability that the label is among five
  classes drawn in descending logit order with uniformly random order inside
  every group of equal logits;
* secondary, *lowest index*: classes are ordered by (logit descending, class
  index ascending), which an integer argmax can implement.

Three counts per image decide both rules for top-1 and top-5: the number of
classes with a logit strictly greater than the label's (``greater``), the
number equal to it including the label (``equal``) and the number equal to it
with a smaller class index (``equal_lower``).
"""
from __future__ import annotations

import os

import numpy as np

VERSION = "b2-tie-readout-1"
RULES = {"primary": "expected_credit", "secondary": "lowest_class_index"}
DTYPES = {"label": np.int16, "greater": np.uint16, "equal": np.uint16, "equal_lower": np.uint16,
          "tie_size": np.uint16, "argmax_lowest": np.int16, "top5_lowest": np.int16, "top5_topk": np.int16}


def batch_readout(output, labels):
    """Per-image readout arrays (NumPy) for one ``[images, classes]`` logit tensor and its labels."""
    import torch
    if output.ndim != 2 or labels.shape != (output.shape[0],) or not torch.isfinite(output).all():
        raise ValueError("invalid logits for the readout")
    classes = output.shape[1]
    own = output.gather(1, labels[:, None])
    same = output == own
    index = torch.arange(classes, device=output.device)[None, :]
    top = output.amax(dim=1, keepdim=True)
    maxima = output == top
    # A stable descending sort keeps the lower class index first among equal logits.
    order = torch.sort(output, dim=1, descending=True, stable=True).indices[:, :5]
    arrays = {"label": labels, "greater": (output > own).sum(dim=1), "equal": same.sum(dim=1),
              "equal_lower": (same & (index < labels[:, None])).sum(dim=1), "tie_size": maxima.sum(dim=1),
              "argmax_lowest": maxima.int().argmax(dim=1), "top5_lowest": order,
              "top5_topk": output.topk(5, dim=1).indices}
    return {key: value.cpu().numpy().astype(DTYPES[key]) for key, value in arrays.items()}


def concatenate(parts):
    return {key: np.concatenate([part[key] for part in parts]) for key in DTYPES}


def check(arrays):
    """Internal consistency of a readout record; raises on any contradiction."""
    n = len(arrays["label"])
    for key, dtype in DTYPES.items():
        value = arrays[key]
        if value.dtype != dtype or value.shape[0] != n:
            raise ValueError(f"readout array {key} has the wrong type or length")
    g, e, low = (arrays[k].astype(np.int64) for k in ("greater", "equal", "equal_lower"))
    label = arrays["label"].astype(np.int64)
    if (e < 1).any() or (low >= e).any() or (arrays["tie_size"] < 1).any():
        raise ValueError("impossible tie counts")
    lowest = arrays["top5_lowest"].astype(np.int64)
    if not (lowest[:, 0] == arrays["argmax_lowest"]).all():
        raise ValueError("lowest-index top-5 does not start with the lowest-index argmax")
    if not (((g == 0) & (low == 0)) == (lowest[:, 0] == label)).all():
        raise ValueError("lowest-index top-1 disagrees with the tie counts")
    if not ((g + low < 5) == (lowest == label[:, None]).any(axis=1)).all():
        raise ValueError("lowest-index top-5 disagrees with the tie counts")
    if not (((g == 0) & (e == arrays["tie_size"])) | (g > 0)).all():
        raise ValueError("label among the maxima but its tie group differs from the maximum group")
    topk = arrays["top5_topk"].astype(np.int64)
    if ((topk == label[:, None]).any(axis=1) & (g >= 5)).any() or ((topk[:, 0] == label) & (g > 0)).any():
        raise ValueError("topk credits a label that no tie order can place there")
    return n


def credits(arrays):
    """Per-image credit vectors (float64 in [0, 1]) under every readout rule."""
    g, e, low = (arrays[k].astype(np.float64) for k in ("greater", "equal", "equal_lower"))
    label = arrays["label"].astype(np.int64)
    slots = 5 - g
    topk = arrays["top5_topk"].astype(np.int64)
    return {
        "top1_expected": np.where(g == 0, 1.0 / e, 0.0),
        "top1_lowest_index": ((g == 0) & (low == 0)).astype(np.float64),
        "top1_strict": ((g == 0) & (e == 1)).astype(np.float64),
        "top1_optimistic": (g == 0).astype(np.float64),
        "top1_topk": (topk[:, 0] == label).astype(np.float64),
        "top5_expected": np.where(slots <= 0, 0.0, np.minimum(1.0, slots / e)),
        "top5_lowest_index": (g + low < 5).astype(np.float64),
        "top5_topk": (topk == label[:, None]).any(axis=1).astype(np.float64),
    }


def summary(arrays):
    n = check(arrays)
    result = {f"{key}_percent": 100 * float(value.mean()) for key, value in credits(arrays).items()}
    g, e = arrays["greater"].astype(np.int64), arrays["equal"].astype(np.int64)
    result.update({"images": n, "images_with_tied_top1": int((arrays["tie_size"] > 1).sum()),
                   "images_with_tied_top1_involving_the_label": int(((g == 0) & (e > 1)).sum()),
                   "images_with_label_tied_across_the_top5_boundary": int(((g < 5) & (g + e > 5)).sum()),
                   "largest_top1_tie": int(arrays["tie_size"].max()), "version": VERSION, "rules": RULES})
    return result


def save(path, arrays):
    """Write a compressed record once; an existing file is never replaced."""
    check(arrays)
    if path.exists():
        raise FileExistsError(f"readout record exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **{key: arrays[key] for key in DTYPES})
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)  # fails if the target appeared meanwhile
    temporary.unlink()


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in DTYPES}
    check(arrays)
    return arrays
