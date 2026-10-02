"""Prediction-collapse check over the sealed evaluation records (r3, CPU only).

Protocol: public/experiments/configs/breadth-study/b2-recon-protocol-v1-addendum-1.json.

Step 1 seals the r2 bias-correction diagnostic (written unsealed by r2) as a new
attestation file. Step 2 reads every sealed evaluation record of this lane and
counts how many distinct top-1 classes each arm predicts on the 1k screen; a
record "collapses" when one class takes at least 90 percent of the images. The
INT4 cells then decide whether the B2 default recipe's chance-level accuracy at
4 bits comes from bias correction (N-default against N-nobc, identical except
``bias_correction='none'``). Nothing is run on a model; no existing file is
changed (outputs are new files and are never overwritten).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from tools.experiment_b.common import ROOT, seal, unseal

ADDENDUM = "public/experiments/configs/breadth-study/b2-recon-protocol-v1-addendum-1.json"
DIAG = ROOT / "artifacts/experiment_b2_recon/diagnostics"
R2_DIAGNOSTIC = DIAG / "bias-correction-resnet18-int4-per-layer-r2.json"
EVALS = ROOT / "artifacts/experiment_b2_recon/evals"
COLLAPSE_SHARE = 0.90
PAIRS = (("N-default", "N-nobc"), ("L-bc", "L-nobc"))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _iso(path):
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(Path(path).stat().st_mtime))


def readout(top1, tie_size, label_among_maxima):
    """Statistics of one arm's 1k-screen predictions (pure function, unit tested)."""
    top1 = np.asarray(top1)
    classes, counts = np.unique(top1, return_counts=True)
    first = int(np.argmax(counts))
    share = float(counts[first]) / top1.size
    tie = np.asarray(tie_size, dtype=np.float64)
    return {"images": int(top1.size), "distinct_top1_classes": int(classes.size),
            "most_frequent_class": int(classes[first]), "most_frequent_share": round(share, 4),
            "collapsed": share >= COLLAPSE_SHARE,
            "mean_tie_size": round(float(tie.mean()), 3), "max_tie_size": int(tie.max()),
            "top1_expected_pct": round(100.0 * float((np.asarray(label_among_maxima) / tie).mean()), 3)}


def verdict(with_bc, without_bc):
    """Addendum-1 criterion for one cell (None when an arm is missing)."""
    if with_bc is None or without_bc is None:
        return None
    if not with_bc["collapsed"]:
        return "not_supported"
    return "supported" if not without_bc["collapsed"] else "undecided"


def seal_r2_diagnostic(out):
    from tools.experiment_b2.common import source_identity
    script = ROOT / "tools/experiment_b2_recon/diagnose_bias.py"
    payload = {"protocol": ADDENDUM, "what": "after-the-fact seal of the r2 diagnostic (unsealed when written)",
               "file": str(R2_DIAGNOSTIC.relative_to(ROOT)), "file_sha256": file_sha256(R2_DIAGNOSTIC),
               "file_mtime": _iso(R2_DIAGNOSTIC), "content": json.loads(R2_DIAGNOSTIC.read_text()),
               "script": str(script.relative_to(ROOT)), "script_sha256": file_sha256(script),
               "script_mtime": _iso(script), "b2_source_sha256_at_seal": source_identity(),
               "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "limit": "fixes the content from the seal onwards; the script mtime preceding the JSON mtime is the "
                        "only evidence that this script version produced it"}
    seal(out, payload)
    return payload


def collapse_table():
    rows = []
    for model_dir in sorted(p for p in EVALS.iterdir() if p.is_dir()):
        for record_path in sorted(model_dir.glob("*.json")):
            record = unseal(record_path)
            arrays = np.load(record_path.with_suffix(".npz"))
            if arrays["top5_topk"].shape[0] != 1000:
                raise ValueError(f"{record_path}: expected 1000 rows")
            row = {"model": record["model"], "label": record["label"], "arm": record["arm"],
                   "weight_format": record.get("weight_format"), "activation_format": record.get("activation_format"),
                   "weight_scale_rule": record.get("weight_scale_rule"), "record_sha256": file_sha256(record_path),
                   "npz_sha256": file_sha256(record_path.with_suffix(".npz"))}
            row.update(readout(arrays["top5_topk"][:, 0], arrays["tie_size"], arrays["label_among_maxima"]))
            rows.append(row)
    return rows


def int4_verdicts(rows):
    def find(model, arm, aformat):
        hits = [r for r in rows if r["model"] == model and r["arm"] == arm and r["weight_format"] == "int4"
                and r["activation_format"] == aformat and r["weight_scale_rule"] in (None, "mse_per_channel")
                and "mse_per_layer" not in r["label"] and "b2in" not in r["label"]]
        if len(hits) > 1:
            raise ValueError(f"ambiguous {model} {arm} {aformat}: {[h['label'] for h in hits]}")
        return hits[0] if hits else None

    out = []
    for model in sorted({r["model"] for r in rows}):
        for aformat in ("int4", "int8"):
            for with_arm, without_arm in PAIRS:
                a, b = find(model, with_arm, aformat), find(model, without_arm, aformat)
                out.append({"model": model, "cell": f"W4/A{aformat[3:]}", "with_bias_correction": with_arm,
                            "without_bias_correction": without_arm,
                            "with": None if a is None else {k: a[k] for k in ("label", "distinct_top1_classes",
                                                                             "most_frequent_class", "most_frequent_share",
                                                                             "top1_expected_pct")},
                            "without": None if b is None else {k: b[k] for k in ("label", "distinct_top1_classes",
                                                                                "most_frequent_class", "most_frequent_share",
                                                                                "top1_expected_pct")},
                            "verdict": verdict(a, b)})
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="r3")
    args = parser.parse_args(argv)
    seal_path = DIAG / f"bias-correction-resnet18-int4-per-layer-r2.seal-{args.tag}.json"
    out_path = DIAG / f"prediction-collapse-{args.tag}.json"
    for path in (seal_path, out_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path.relative_to(ROOT)}; use a new --tag")
    seal_r2_diagnostic(seal_path)
    rows = collapse_table()
    verdicts = int4_verdicts(rows)
    seal(out_path, {"protocol": ADDENDUM, "evidence": "development_evidence_screen1k (re-read of sealed records)",
                    "collapse_share": COLLAPSE_SHARE, "records": len(rows), "int4_verdicts": verdicts,
                    "collapsed_records": [f"{r['model']}/{r['label']}" for r in rows if r["collapsed"]],
                    "rows": rows})
    print(json.dumps({"wrote": [str(seal_path.relative_to(ROOT)), str(out_path.relative_to(ROOT))],
                      "records": len(rows), "collapsed": sum(r["collapsed"] for r in rows)}))
    for v in verdicts:
        w, wo = v["with"] or {}, v["without"] or {}
        print(v["model"], v["cell"], v["with_bias_correction"], w.get("distinct_top1_classes"),
              w.get("most_frequent_share"), "|", v["without_bias_correction"], wo.get("distinct_top1_classes"),
              wo.get("most_frequent_share"), "->", v["verdict"])


if __name__ == "__main__":
    main()
