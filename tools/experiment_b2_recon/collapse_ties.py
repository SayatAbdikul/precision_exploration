"""Tie-aware prediction-collapse check (r4; review 2, finding 1).

Protocol: public/experiments/configs/breadth-study/b2-recon-protocol-v1-addendum-2.json.

The r3 check (``collapse_check.py``) counted the first index ``torch.topk``
returns, whose order among tied maxima is unspecified; in the INT4 cells up to
100 percent of the images are tied.  This module replaces that readout by three
tie-aware shares per class, each bounded from the stored top-5 lists (exact
when an image's tie has at most 5 classes) or exact when a full maxima file
from the GPU step exists:

* ``among_maxima`` (primary): the class attains the maximum logit;
* ``expected``: the class is predicted under uniform random tie-breaking;
* ``lowest_index``: the class is the lowest-index maximum.

Subcommands: ``readout`` (CPU; writes the sealed summary, refuses to
overwrite) and ``exact MODEL LABEL...`` (GPU; re-evaluates arms through the
unchanged evaluation path and stores the packed maxima mask, only after
checking that the stored top-5 lists and tie sizes are reproduced).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from tools.experiment_b.common import ROOT, seal, unseal

from .collapse_check import COLLAPSE_SHARE, PAIRS, file_sha256, readout as r3_readout

ADDENDUM = "public/experiments/configs/breadth-study/b2-recon-protocol-v1-addendum-2.json"
DIAG = ROOT / "artifacts/experiment_b2_recon/diagnostics"
EVALS = ROOT / "artifacts/experiment_b2_recon/evals"
EXACT = DIAG / "maxima-r4"
CLASSES = 1000
SHARES = ("among_maxima", "expected", "lowest_index")


def bounds_from_top5(top5, tie_size, classes=CLASSES):
    """Lower and upper bound of each tie-aware share for every class, from the stored top-5 lists.

    Returns ``{share: (lower[classes], upper[classes])}`` as fractions of the images.
    """
    top5 = np.asarray(top5, dtype=np.int64)
    tie = np.asarray(tie_size, dtype=np.int64)
    n = top5.shape[0]
    out = {share: (np.zeros(classes), np.zeros(classes)) for share in SHARES}
    for i in range(n):
        listed = top5[i, :min(int(tie[i]), 5)]
        complete = tie[i] <= 5
        weight = 1.0 / tie[i]
        lowest = int(listed.min())
        for share in SHARES:
            lower, upper = out[share]
            if share == "among_maxima":
                lower[listed] += 1
                upper[listed] += 1
            elif share == "expected":
                lower[listed] += weight
                upper[listed] += weight
            elif complete:
                lower[lowest] += 1
                upper[lowest] += 1
            else:
                upper[lowest] += 1
            if not complete:
                unlisted = np.ones(classes, dtype=bool)
                unlisted[listed] = False
                if share == "among_maxima":
                    upper[unlisted] += 1
                elif share == "expected":
                    upper[unlisted] += weight
                else:  # an unlisted class can be the lowest maximum only if below every listed one
                    unlisted[lowest:] = False
                    upper[unlisted] += 1
    return {share: (lower / n, upper / n) for share, (lower, upper) in out.items()}


def exact_shares(mask):
    """Exact shares from the full ``[images, classes]`` boolean maxima mask (lower = upper)."""
    mask = np.asarray(mask, dtype=bool)
    n = mask.shape[0]
    tie = mask.sum(axis=1)
    if (tie < 1).any():
        raise ValueError("an image without a maximum")
    among = mask.sum(axis=0) / n
    expected = (mask / tie[:, None]).sum(axis=0) / n
    lowest = np.bincount(mask.argmax(axis=1), minlength=mask.shape[1]) / n
    return {"among_maxima": (among, among), "expected": (expected, expected), "lowest_index": (lowest, lowest)}


def status(lower, upper, threshold=COLLAPSE_SHARE):
    """``collapsed`` / ``not_collapsed`` / ``undetermined`` for one share (addendum 2)."""
    if float(np.max(lower)) >= threshold:
        return "collapsed"
    if float(np.max(upper)) < threshold:
        return "not_collapsed"
    return "undetermined"


def verdict(with_status, without_status):
    """Addendum-1 rule on the tie-aware statuses (``None`` when an arm is missing)."""
    if with_status is None or without_status is None:
        return None
    if with_status == "not_collapsed":
        return "not_supported"
    if with_status == "collapsed" and without_status == "not_collapsed":
        return "supported"
    return "undecided"


def summarise(shares):
    row = {}
    for share, (lower, upper) in shares.items():
        top = int(np.argmax(lower))
        row[share] = {"class_of_max_lower": top, "max_lower": round(float(lower[top]), 4),
                      "max_upper": round(float(np.max(upper)), 4), "class_of_max_upper": int(np.argmax(upper)),
                      "status": status(lower, upper)}
    return row


def exact_path(model, label):
    return EXACT / model / f"{label}.npz"


def record_row(record_path):
    record = unseal(record_path)
    npz_path = record_path.with_suffix(".npz")
    with np.load(npz_path) as arrays:
        top5, tie, among = arrays["top5_topk"], arrays["tie_size"], arrays["label_among_maxima"]
    if top5.shape[0] != 1000:
        raise ValueError(f"{record_path}: expected 1000 rows")
    exact_file = exact_path(record["model"], record["label"])
    source = "top5_bounds"
    shares = bounds_from_top5(top5, tie)
    exact_ok = False
    if exact_file.exists():
        exact_ok = unseal(exact_file.with_suffix(".json"))["reproduced"]["all"]
        if not exact_ok:  # addendum 2: a rerun that does not reproduce the record is not used, only reported
            source = "top5_bounds (exact rerun did not reproduce the record; not used)"
    if exact_ok:
        with np.load(exact_file) as stored:
            mask = np.unpackbits(stored["maxima_packbits"], axis=1, count=CLASSES).astype(bool)
        if not np.array_equal(mask.sum(axis=1), tie):
            raise ValueError(f"{exact_file}: tie sizes differ from the record")
        shares, source = exact_shares(mask), "exact_maxima_mask"
    r3 = r3_readout(top5[:, 0], tie, among)
    return {"model": record["model"], "label": record["label"], "arm": record["arm"],
            "weight_format": record.get("weight_format"), "activation_format": record.get("activation_format"),
            "weight_scale_rule": record.get("weight_scale_rule"), "record_sha256": file_sha256(record_path),
            "npz_sha256": file_sha256(npz_path), "source": source,
            "images_tied": int((tie > 1).sum()), "images_tie_over_5": int((tie > 5).sum()),
            "mean_tie_size": round(float(tie.mean()), 3), "max_tie_size": int(tie.max()),
            "top1_expected_pct": r3["top1_expected_pct"],
            "r3_topk_first_index": {k: r3[k] for k in ("most_frequent_class", "most_frequent_share", "collapsed")},
            "shares": summarise(shares)}


def table():
    return [record_row(path) for model_dir in sorted(p for p in EVALS.iterdir() if p.is_dir())
            for path in sorted(model_dir.glob("*.json"))]


def find(rows, model, arm, aformat):
    """The default-grid INT4 record of one arm (same selection as collapse_check.int4_verdicts)."""
    hits = [r for r in rows if r["model"] == model and r["arm"] == arm and r["weight_format"] == "int4"
            and r["activation_format"] == aformat and r["weight_scale_rule"] in (None, "mse_per_channel")
            and "mse_per_layer" not in r["label"] and "b2in" not in r["label"]]
    if len(hits) > 1:
        raise ValueError(f"ambiguous {model} {arm} {aformat}")
    return hits[0] if hits else None


def verdicts(rows):
    out = []
    for model in sorted({r["model"] for r in rows}):
        for aformat in ("int4", "int8"):
            for with_arm, without_arm in PAIRS:
                a, b = find(rows, model, with_arm, aformat), find(rows, model, without_arm, aformat)
                entry = {"model": model, "cell": f"W4/A{aformat[3:]}", "with_bias_correction": with_arm,
                         "without_bias_correction": without_arm,
                         "with": a and {"label": a["label"], "source": a["source"], "shares": a["shares"],
                                        "top1_expected_pct": a["top1_expected_pct"]},
                         "without": b and {"label": b["label"], "source": b["source"], "shares": b["shares"],
                                           "top1_expected_pct": b["top1_expected_pct"]}}
                for share in SHARES:
                    entry[f"verdict_{share}"] = verdict(a and a["shares"][share]["status"],
                                                        b and b["shares"][share]["status"])
                entry["verdict_primary"] = entry["verdict_among_maxima"]
                entry["verdict_r3_superseded"] = (None if not (a and b) else
                                                  ("not_supported" if not a["r3_topk_first_index"]["collapsed"] else
                                                   "supported" if not b["r3_topk_first_index"]["collapsed"]
                                                   else "undecided"))
                out.append(entry)
    return out


def undetermined_labels(entries):
    """Records of the verdict cells whose status is undetermined under any share (the GPU step's worklist)."""
    wanted = set()
    for entry in entries:
        for side in ("with", "without"):
            arm = entry[side]
            if arm and any(arm["shares"][s]["status"] == "undetermined" for s in SHARES):
                wanted.add((entry["model"], arm["label"]))
    return sorted(wanted)


def cmd_readout(args):
    out_path = DIAG / f"prediction-collapse-ties-{args.tag}.json"
    rows = table()
    entries = verdicts(rows)
    pending = undetermined_labels(entries)
    if args.dry_run:
        for e in entries:
            print(e["model"], e["cell"], e["with_bias_correction"], "->", {s: e[f"verdict_{s}"] for s in SHARES},
                  "r3:", e["verdict_r3_superseded"])
        print(json.dumps({"undetermined_records": pending}))
        return
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite {out_path.relative_to(ROOT)}; use a new --tag")
    seal(out_path, {"protocol": ADDENDUM, "supersedes_readout_of": "artifacts/experiment_b2_recon/diagnostics/"
                    "prediction-collapse-r3.json (topk first index)",
                    "evidence": "development_evidence_screen1k (re-read of sealed records)",
                    "collapse_share": COLLAPSE_SHARE, "records": len(rows), "int4_verdicts": entries,
                    "undetermined_verdict_records": pending, "rows": rows})
    print(json.dumps({"wrote": str(out_path.relative_to(ROOT)), "records": len(rows),
                      "undetermined_verdict_records": pending}))
    for e in entries:
        print(e["model"], e["cell"], e["with_bias_correction"], "->", e["verdict_primary"],
              {s: e[f"verdict_{s}"] for s in SHARES[1:]}, "r3:", e["verdict_r3_superseded"])


def cmd_exact(args):
    """GPU: re-evaluate arms and store their full maxima masks (addendum 2, conditional step)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import time
    import torch
    from tools.experiment_b2 import frozen  # noqa: F401
    from tools.experiment_b2.common import BIAS_CORRECTION_IMAGES, PROTOCOL as B2_PROTOCOL, source_identity
    from tools.experiment_b2.data import cached_inputs, v1_calibration
    from tools.experiment_b2.recipe import named
    from tools.experiment_b2.runner import model_setup
    from .engine import build_engine, state_digest
    from .evaluate import GROUPS, label, readout, run_screen, write_arrays
    from .fit import find_fit, load_fit, recon_sources

    specs = {label(s): s for group in GROUPS.values() for s in group()}
    todo = [name for name in args.labels if not exact_path(args.model, name).with_suffix(".json").exists()]
    if not todo:
        print(json.dumps({"model": args.model, "status": "all exact files exist"}))
        return
    device = "cuda"
    setup = model_setup(args.model, device)
    graph, rows, inputs = setup["graph"], setup["rows"], setup["inputs"]
    arrays, maxima, _ = v1_calibration(args.model)
    bias_inputs, _ = cached_inputs("imagenet_calibration_2k", BIAS_CORRECTION_IMAGES, setup["transform"], build=False)
    folder = EXACT / args.model
    folder.mkdir(parents=True, exist_ok=True)
    for name in todo:
        tick = time.monotonic()
        spec = specs[name]
        record_path = EVALS / args.model / f"{name}.json"
        record = unseal(record_path)
        recipe = named(spec["recipe"]) if spec["recipe"] else None
        learned = None
        if spec["fit"]:
            rule, mode, seed = spec["fit"]
            learned, _ = load_fit(find_fit(args.model, spec["weight_format"], rule, seed, mode), graph, device)
        correction = None if spec["bias_correction"] == "recipe" else spec["bias_correction"]
        run, _, deployed = build_engine(graph, spec["weight_format"], spec["activation_format"], recipe, arrays,
                                        maxima, device, weight_rule=spec["weight_rule"], learned=learned,
                                        bias_inputs=bias_inputs, bias_correction=correction)
        logits, top5 = run_screen(run, inputs, rows, device, B2_PROTOCOL["inference_batch_size"])
        _, per_image = readout(logits, top5, rows)
        with np.load(record_path.with_suffix(".npz")) as saved:
            same = {k: bool(np.array_equal(saved[k], per_image[k])) for k in per_image}
        mask = (logits == logits.amax(dim=1, keepdim=True)).numpy()
        same["tie_size_from_mask"] = bool(np.array_equal(mask.sum(axis=1), per_image["tie_size"]))
        same["deployed_state"] = state_digest(deployed) == record["deployed_state_sha256"]
        same["all"] = all(same.values())
        npz = exact_path(args.model, name)
        write_arrays(npz, {"maxima_packbits": np.packbits(mask, axis=1)})
        seal(npz.with_suffix(".json"), {
            "protocol": ADDENDUM, "model": args.model, "label": name, "record": str(record_path.relative_to(ROOT)),
            "record_sha256": file_sha256(record_path), "npz": str(npz.relative_to(ROOT)),
            "npz_sha256": file_sha256(npz), "reproduced": same, "b2_source_sha256": source_identity(),
            "recon_sources": recon_sources(), "evidence": "development_evidence_screen1k",
            "seconds": round(time.monotonic() - tick, 1)})
        print(json.dumps({"model": args.model, "label": name, "reproduced": same,
                          "seconds": round(time.monotonic() - tick, 1)}), flush=True)
        del run, deployed
        torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("readout")
    p.add_argument("--tag", default="r4")
    p.add_argument("--dry-run", action="store_true", help="print the verdicts and the worklist; write nothing")
    p = sub.add_parser("exact")
    p.add_argument("model")
    p.add_argument("labels", nargs="+")
    args = parser.parse_args(argv)
    {"readout": cmd_readout, "exact": cmd_exact}[args.command](args)


if __name__ == "__main__":
    main()
