"""Summaries of the detector breadth study (lane Q6), written once to results/summaries/b2-detector-breadth-v1/.

    .venv/bin/python -m tools.experiment_b2_det_breadth.summary            # writes the files (refuses to overwrite)
    .venv/bin/python -m tools.experiment_b2_det_breadth.summary --show     # prints the tables, writes nothing
    .venv/bin/python -m tools.experiment_b2_det_breadth.summary --tag r2   # a regenerated version under new names

Every number is development evidence on the COCO 1k screen.  L7's records and bootstrap vectors are read
(read-only) for FP32, INT8 and the nine sentinel formats; the 16 other formats, the calibration subsets and the
attribution arms are this study's.  Units: mAP points.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tools.experiment_b.common import ROOT, formats, unseal
from tools.experiment_b2_det.arms4 import conformant_arm
from tools.experiment_b2_det.engine import BLOCK_FORMATS
from tools.experiment_b2_det.stats import interval, paired

from . import core
from . import stats as qstats

OUT = ROOT / "results/summaries/b2-detector-breadth-v1"
CLASSIFIER_CELLS = ROOT / "results/summaries/b2-matrix-v1/cells-d4.csv"
ORDERS = ("index", "reverse", "random0", "random1", "random2", "random3")
MARGIN = 0.5


# ------------------------------------------------------------------------------------------------ evidence
def l7_vector(identity, tag="index"):
    path = core.L7 / "bootstrap" / f"{identity}-{core.IMAGES}-{tag}.npz"
    if not path.exists():
        return None
    with np.load(path) as saved:
        return {"point": saved["point"], "draws": saved["draws"], "ties": json.loads(str(saved["ties"]))}


def l7_imageid(identity):
    path = core.L7 / "imageorder" / f"{identity}-{core.IMAGES}-index-imageid.npz"
    if not path.exists():
        return None
    with np.load(path) as saved:
        return {"point": saved["point"], "ties": json.loads(str(saved["ties"]))}


class Evidence:
    def __init__(self):
        self.l7 = core.l7_runs()
        self.mine = core.runs()
        self.by_label = {(r["format"], r["label"]): r for r in self.mine}

    def record(self, fmt, label):
        """This study's record first, then L7's (by recipe name)."""
        if (fmt, label) in self.by_label:
            return self.by_label[(fmt, label)], "Q6"
        if (fmt, label) in self.l7:
            return self.l7[(fmt, label)], "L7"
        return None, None

    def vector(self, record, source, tag="index"):
        identity = record["configuration_sha256"]
        return qstats.load(identity, tag) if source == "Q6" else l7_vector(identity, tag)

    def imageid(self, record, source):
        identity = record["configuration_sha256"]
        return qstats.load_imageid(identity) if source == "Q6" else l7_imageid(identity)

    def fp32(self):
        record = self.l7[("fp32", "fp32")]
        return record, self.vector(record, "L7")


def pair(left, right):
    """right minus left (points) with the paired interval, or the point difference only."""
    if left is None or right is None:
        return None
    if len(left["draws"]) and len(right["draws"]):
        return paired(left, right)["map50_95"]
    return {"delta": float(100 * (right["point"][0] - left["point"][0])), "interval": None}


def fmt_delta(d):
    if d is None:
        return ""
    if d["interval"] is None:
        return f"{d['delta']:+.2f}"
    return f"{d['delta']:+.2f} [{d['interval'][0]:+.2f}, {d['interval'][1]:+.2f}]"


def v1_values():
    """Sealed v1 mAP50-95 (points) of every format under max-abs and percentile 99.9 (Experiment B-extension)."""
    from tools.experiment_b_ext import runner as ext
    folder, out = ROOT / "artifacts/experiment_b_ext", {}
    for path in sorted((folder / "summaries").glob("*-1000.json")):
        configuration = unseal(folder / "configurations" / f"{path.name.split('-')[0]}.json")
        if configuration["model_context"].get("model") != "yolov8n" or configuration["source_sha256"] != ext.source_identity():
            continue
        out[(configuration["format"], configuration["recipe"])] = 100 * unseal(path)["metrics"]["map50_95"]
    return out


# ------------------------------------------------------------------------------------------------- tables
def format_table(ev):
    fp32_record, fp32 = ev.fp32()
    v1 = v1_values()
    rows = []
    for entry in formats():
        fmt = entry["name"]
        default_record, default_source = ev.record(fmt, "default")
        for recipe in ("default", "conformant"):
            arm = "default" if recipe == "default" else conformant_arm(fmt)
            record, source = ev.record(fmt, arm)
            row = {"format": fmt, "family": entry["family"], "bits": entry["bits"], "recipe": recipe, "arm": arm,
                   "rule_conformance": ("frozen_default" if recipe == "default" else
                                       "nearest_available_joins_not_stored" if fmt in BLOCK_FORMATS else "conformant"),
                   "source": source, "v1_maxabs": v1.get((fmt, "maxabs")), "v1_percentile_99_9": v1.get((fmt, "percentile_99_9"))}
            if record is None:
                at_chance = default_record is not None and 100 * default_record["map50_95"] < core.CHANCE_MAP
                row.update(status="not_run_stop_rule" if at_chance else "missing")
                rows.append(row)
                continue
            vector = ev.vector(record, source)
            int8_record, int8_source = ev.record("int8", "default" if recipe == "default" else "conformant")
            int8 = ev.vector(int8_record, int8_source)
            row.update(status="measured", configuration=record["configuration_sha256"][:12],
                       map50_95=100 * record["map50_95"], map50=100 * record["map50"],
                       at_chance=100 * record["map50_95"] < core.CHANCE_MAP,
                       minus_fp32=pair(fp32, vector), minus_int8_same_recipe=None if fmt == "int8" else pair(int8, vector),
                       minus_default=None if recipe == "default" else pair(ev.vector(default_record, default_source), vector),
                       detections=(vector or {}).get("ties", {}).get("detections"),
                       distinct_scores=(vector or {}).get("ties", {}).get("distinct_scores"))
            rows.append(row)
    return rows, {"fp32_map50_95": 100 * fp32_record["map50_95"], "fp32_configuration": fp32_record["configuration_sha256"][:12]}


def tie_table(ev):
    rows = []
    for entry in formats():
        fmt = entry["name"]
        for recipe in ("default", "conformant"):
            arm = "default" if recipe == "default" else conformant_arm(fmt)
            record, source = ev.record(fmt, arm)
            if record is None:
                continue
            index = ev.vector(record, source)
            if index is None:
                continue
            points = {}
            for tag in ORDERS:
                v = ev.vector(record, source, tag)
                if v is not None:
                    points[tag] = 100 * float(v["point"][0])
            image = ev.imageid(record, source)
            if image is not None:
                points["imageid"] = 100 * float(image["point"][0])
            ties = index["ties"]
            fixed = points["index"]
            moves = {k: v - fixed for k, v in points.items() if k != "index"}
            largest = max(moves.items(), key=lambda kv: abs(kv[1])) if moves else (None, 0.0)
            rows.append({"format": fmt, "recipe": recipe, "arm": arm, "source": source, "orders": len(points),
                         "fixed_rule": fixed, "imageid": points.get("imageid"),
                         **{f"order_{k}": v for k, v in points.items() if k not in ("index", "imageid")},
                         "range_over_orders": max(points.values()) - min(points.values()),
                         "largest_move_from_fixed": largest[1], "largest_move_order": largest[0],
                         "envelope_low": 100 * ties["false_positives_first"][0], "envelope_high": 100 * ties["true_positives_first"][0],
                         "tied_share": ties["detections_with_a_tied_score"] / max(1, ties["detections_scored"]),
                         "distinct_scores": ties["distinct_scores"], "detections": ties["detections"]})
    return rows


def rank(values):
    from scipy.stats import rankdata
    return rankdata([-v for v in values], method="average")


def classifier_table(format_rows):
    from scipy.stats import kendalltau, spearmanr
    cells = {}
    with CLASSIFIER_CELLS.open() as stream:
        for row in csv.DictReader(stream):
            if row["recipe"] == "default":
                cells[(row["model"], row["format"])] = float(row["top1_expected"])
    models = sorted({m for m, _ in cells})
    detector = {(r["format"], r["recipe"]): r.get("map50_95") for r in format_rows}
    names = [e["name"] for e in formats() if detector.get((e["name"], "default")) is not None]
    det_default = [detector[(f, "default")] for f in names]
    det_conf = [detector[(f, "conformant")] if detector.get((f, "conformant")) is not None else detector[(f, "default")]
                for f in names]
    table = []
    det_rank, conf_rank = rank(det_default), rank(det_conf)
    model_ranks = {m: rank([cells[(m, f)] for f in names]) for m in models}
    for i, f in enumerate(names):
        table.append({"format": f, "detector_default": det_default[i], "detector_rank_default": float(det_rank[i]),
                      "detector_conformant_or_default_at_chance": det_conf[i], "detector_rank_conformant": float(conf_rank[i]),
                      **{f"{m}_top1_default": cells[(m, f)] for m in models},
                      **{f"{m}_rank": float(model_ranks[m][i]) for m in models}})
    correlations = []
    for m in models:
        top1 = [cells[(m, f)] for f in names]
        for label, det in (("default", det_default), ("conformant", det_conf)):
            correlations.append({"classifier": m, "detector_recipe": label, "formats": len(names),
                                 "spearman": float(spearmanr(det, top1).statistic), "kendall_tau_b": float(kendalltau(det, top1).statistic)})
    return table, correlations


def seed_table(ev):
    rows, gaps = [], []
    seed_runs = [r for r in ev.mine if r["study_part"] == "B"]
    keys = sorted({(r["format"], r["recipe_name"]) for r in seed_runs})
    by = {(r["format"], r["recipe_name"], r["subset"]): r for r in seed_runs}
    subsets = sorted({r["subset"] for r in seed_runs})
    for fmt, arm in keys:
        values = [100 * by[(fmt, arm, s)]["map50_95"] for s in subsets if (fmt, arm, s) in by]
        full, _ = ev.record(fmt, arm)
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
        rows.append({"format": fmt, "arm": arm, "recipe": "default" if arm == "default" else "conformant",
                     "subsets": len(values), **{f"subset_{s}": 100 * by[(fmt, arm, s)]["map50_95"] for s in subsets if (fmt, arm, s) in by},
                     "mean": float(np.mean(values)), "sd": sd, "range": max(values) - min(values),
                     "full_2000": 100 * full["map50_95"] if full else None,
                     "full_minus_mean": (100 * full["map50_95"] - float(np.mean(values))) if full else None,
                     "sd_above_half_margin": None if sd is None else sd > MARGIN / 2, "sd_above_margin": None if sd is None else sd > MARGIN})
    for recipe in ("default", "conformant"):
        for left_fmt, right_fmt in (("int8", "posit8_es1"), ("bfp6", "log8"), ("int8", "fp8_e4m3fn")):
            left_arm = "default" if recipe == "default" else conformant_arm(left_fmt)
            right_arm = "default" if recipe == "default" else conformant_arm(right_fmt)
            full_l, src_l = ev.record(left_fmt, left_arm)
            full_r, src_r = ev.record(right_fmt, right_arm)
            if full_l is None or full_r is None:
                continue
            full_gap = pair(ev.vector(full_l, src_l), ev.vector(full_r, src_r))
            per = []
            for s in subsets:
                left = by.get((left_fmt, left_arm, s)) if left_fmt not in BLOCK_FORMATS else full_l
                right = by.get((right_fmt, right_arm, s))
                if left is None or right is None:
                    continue
                lv = qstats.load(left["configuration_sha256"]) if left_fmt not in BLOCK_FORMATS else ev.vector(full_l, src_l)
                rv = qstats.load(right["configuration_sha256"])
                d = pair(lv, rv)
                if d is not None:
                    per.append({"subset": s, **d})
            if not per:
                continue
            sign = np.sign(full_gap["delta"])
            gaps.append({"recipe": recipe, "pair": f"{right_fmt} minus {left_fmt}",
                         "calibration_invariant_side": left_fmt if left_fmt in BLOCK_FORMATS else None,
                         "full_2000": full_gap, "per_subset": per,
                         "subsets_same_sign_as_full": int(sum(np.sign(p["delta"]) == sign for p in per)),
                         "subsets_interval_excludes_zero": int(sum(p["interval"] is not None and (p["interval"][0] > 0 or p["interval"][1] < 0) for p in per)),
                         "gap_sd_over_subsets": float(np.std([p["delta"] for p in per], ddof=1)) if len(per) > 1 else None})
    return rows, gaps


def attribution_table(ev):
    fp32_record, fp32 = ev.fp32()
    rows = []
    for fmt in core.ATTRIBUTION_FORMATS:
        base_record, base_source = ev.record(fmt, "default")
        base = ev.vector(base_record, base_source)
        loss = 100 * (fp32_record["map50_95"] - base_record["map50_95"])
        for r in [r for r in ev.mine if r["study_part"] in ("C", "C-fine") and r["format"] == fmt]:
            vector = qstats.load(r["configuration_sha256"])
            gain = pair(base, vector)
            rows.append({"format": fmt, "stage": "group" if r["study_part"] == "C" else "fine",
                         "group": r["attribution_group"], "unit": r.get("unit"), "label": r["label"],
                         "map50_95": 100 * r["map50_95"], "default": 100 * base_record["map50_95"], "loss_to_fp32": loss,
                         "gain_over_default": gain, "share_of_loss": None if gain is None else gain["delta"] / loss,
                         "minus_fp32": pair(fp32, vector), "detections": (vector or {}).get("ties", {}).get("detections")})
    return rows


# ------------------------------------------------------------------------------------------------- output
def flat(row):
    out = {}
    for k, v in row.items():
        if isinstance(v, dict) and "delta" in v:
            out[k] = round(v["delta"], 4)
            out[f"{k}_low"] = None if v["interval"] is None else round(v["interval"][0], 4)
            out[f"{k}_high"] = None if v["interval"] is None else round(v["interval"][1], 4)
        elif isinstance(v, float):
            out[k] = round(v, 4)
        else:
            out[k] = v
    return out


def write(name, payload, rows, tag):
    suffix = f"-{tag}" if tag else ""
    OUT.mkdir(parents=True, exist_ok=True)
    targets = [OUT / f"{name}{suffix}.json"] + ([OUT / f"{name}{suffix}.csv"] if rows else [])
    for target in targets:
        if target.exists():
            raise SystemExit(f"refusing to overwrite {target} (use --tag)")
    targets[0].write_text(json.dumps(payload, indent=1, default=float) + "\n")
    if rows:
        flats = [flat(r) for r in rows]
        fields = list(dict.fromkeys(k for r in flats for k in r))
        with targets[1].open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flats)
    return [str(t.relative_to(ROOT)) for t in targets]


def build():
    ev = Evidence()
    format_rows, header = format_table(ev)
    classifier_rows, correlations = classifier_table(format_rows)
    seed_rows, gaps = seed_table(ev)
    common = {"protocol": core.PROTOCOL, "panel": "coco_screen_1k (1000 images)", "interpretation": "development_evidence",
              "units": "mAP50-95 points", "statistics": "paired image bootstrap, 2000 draws, seed 310911, pointwise 95%"}
    return {
        "formats-1000": ({**common, **header, "rows": format_rows}, format_rows),
        "ties-1000": ({**common, "rows": tie_table(ev)}, None),
        "classifier-order": ({**common, "classifier_source": str(CLASSIFIER_CELLS.relative_to(ROOT)),
                              "readout": "classifier top1_expected under the classifier `default` recipe",
                              "correlations": correlations, "rows": classifier_rows}, classifier_rows),
        "seeds-1000": ({**common, "subsets": "5 disjoint 400-image subsets of coco_calibration_2k (protocol part B)",
                        "margin": MARGIN, "rows": seed_rows, "close_gaps": gaps}, seed_rows),
        "attribution-1000": ({**common, "status": "diagnostic only", "rows": attribution_table(ev)}, None),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    tables = build()
    if args.show:
        for name, (payload, _) in tables.items():
            print(f"== {name}")
            for row in payload.get("rows", []):
                print(json.dumps(flat(row), default=float))
            for extra in ("correlations", "close_gaps"):
                for row in payload.get(extra, []):
                    print(json.dumps(row, default=float))
        return 0
    written = []
    for name, (payload, rows) in tables.items():
        if name == "ties-1000" or name == "attribution-1000":
            rows = payload["rows"]
        written += write(name, payload, rows, args.tag)
    print("\n".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
