"""Effect tables of the detector study from the sealed run records and the stored bootstrap vectors.

``.venv-b/bin/python -m tools.experiment_b2_det.report --images 128`` writes
``results/summaries/b2-detector-v1/effects-<images>[-TAG].{json,csv}`` (never overwrites a different file).
Every row: mAP, the paired difference to FP32 and to its reference arm, with pointwise 95 percent intervals.
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from tools.experiment_b.common import ROOT, atomic_json, digest, unseal
from .runner import BASE
from .stats import CONFIDENCE, RESAMPLES, SEED, paired

OUT = ROOT / "results/summaries/b2-detector-v1"
LADDER = ["v1_maxabs", "cum1_head_logits", "cum2_passthrough", "cum3_unsigned", "cum4_act_mse", "cum5_weight_mse",
          "cum6_bias_correction", "cum7_concat_passthrough", "cum8_fused_silu"]


def reference(name):
    if name in LADDER[1:]:
        return LADDER[LADDER.index(name) - 1]
    for prefix, base in (("cum5_", "cum5_weight_mse"), ("cum6_", "cum6_bias_correction"), ("full_", "cum8_fused_silu"),
                         ("default_", "default")):
        if name.startswith(prefix):
            return base
    return None if name in ("fp32", "v1_maxabs") else "v1_maxabs"


def vector(stem):
    with np.load(BASE / "bootstrap" / f"{stem}.npz") as saved:
        return {"point": saved["point"], "draws": saved["draws"], "ties": json.loads(str(saved["ties"]))}


def rows(images, reference_of=reference):
    records = {}
    for path in sorted((BASE / "runs" / str(images)).glob("*.json")):
        record = unseal(path)
        records[(record["format"], record["recipe_name"])] = record
    fp32 = vector(f"{records[('fp32', 'fp32')]['configuration_sha256']}-{images}-index")
    table = []
    for (name, recipe), record in sorted(records.items()):
        mine = vector(f"{record['configuration_sha256']}-{images}-index")
        row = {"format": name, "recipe": recipe, "configuration": record["configuration_sha256"][:12], "images": images,
               "map50_95": 100 * float(mine["point"][0]), "map50": 100 * float(mine["point"][1]),
               "detections": mine["ties"]["detections"], "distinct_scores": mine["ties"]["distinct_scores"]}
        against = paired(fp32, mine)
        row.update(delta_fp32=against["map50_95"]["delta"], delta_fp32_interval=against["map50_95"]["interval"],
                   delta_fp32_map50=against["map50"]["delta"], delta_fp32_map50_interval=against["map50"]["interval"])
        base = reference_of(recipe)
        if base is not None and (name, base) in records:
            other = paired(vector(f"{records[(name, base)]['configuration_sha256']}-{images}-index"), mine)
            row.update(reference=base, delta_reference=other["map50_95"]["delta"],
                       delta_reference_interval=other["map50_95"]["interval"])
        table.append(row)
    return table


def write(name, table, extra=None):
    document = {"panel": "dev128" if table and table[0]["images"] == 128 else "screen1k",
                "interpretation": "development evidence; pointwise paired image-bootstrap intervals, no multiplicity adjustment",
                "statistics": {"resamples": RESAMPLES, "seed": SEED, "confidence": CONFIDENCE, "units": "mAP points"},
                "rows": table, **(extra or {})}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{name}.json"
    if target.exists():
        if json.loads(target.read_text())["sha256"] == digest(document):
            print(f"unchanged: {target.name}")
            return
        raise SystemExit(f"{target} exists with different content; use --tag for a new versioned file")
    atomic_json(target, {"payload": document, "sha256": digest(document)})
    keys = ["format", "recipe", "configuration", "images", "map50_95", "map50", "delta_fp32", "delta_fp32_interval",
            "reference", "delta_reference", "delta_reference_interval", "detections", "distinct_scores"]
    with (OUT / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in table:
            writer.writerow({k: (json.dumps([round(x, 3) for x in v]) if isinstance(v, list) else
                                 round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    print(f"wrote {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, choices=(128, 1000), default=128)
    parser.add_argument("--tag", default="")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    table = rows(args.images)
    if args.show:
        for row in table:
            interval = row.get("delta_reference_interval")
            print(f"{row['format']:11s} {row['recipe']:26s} {row['map50_95']:6.2f}  vsFP32 {row['delta_fp32']:+6.2f} "
                  f"[{row['delta_fp32_interval'][0]:+.2f},{row['delta_fp32_interval'][1]:+.2f}]"
                  + (f"  vs {row['reference']:22s} {row['delta_reference']:+6.2f} [{interval[0]:+.2f},{interval[1]:+.2f}]"
                     if interval else ""))
        return 0
    write(f"effects-{args.images}" + (f"-{args.tag}" if args.tag else ""), table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
