"""Assemble parts A-E into one summary folder (written once; an existing file is never replaced)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from tools.experiment_b.common import ROOT
from . import PROTOCOL, SEED, agreement, fixed_width, resolution, ties, units as U
from . import detector as D

FLOOR = 0.10


def _clean(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


class Writer:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.files = []

    def _path(self, name):
        path = self.out / name
        if path.exists():
            raise FileExistsError(f"summary exists, not replaced: {path}")
        self.files.append(path)
        return path

    def csv(self, name, rows):
        rows = [{k: _clean(v) for k, v in r.items() if not isinstance(v, np.ndarray)} for r in rows]
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with self._path(name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 6) if isinstance(v, float) else (json.dumps(v) if isinstance(v, (list, tuple, dict)) else v))
                            for k, v in r.items()})

    def json(self, name, data):
        self._path(name).write_text(json.dumps(data, indent=1, default=_clean) + "\n")


def own(u):
    return u["source"] == "matrix" and u["group"] in ("default", "minimal") and "@" not in u["name"]


# ------------------------------------------------------------------------------------------------ part A
def part_a(w, units):
    unit_rows, pair_rows, agree = agreement.analyse(units)
    w.csv("partA-units.csv", unit_rows)
    w.csv("partA-pairs.csv", pair_rows)
    summary = agreement.resolution_summary(pair_rows)
    w.json("partA-resolution.json", summary)
    return unit_rows, pair_rows, summary


# ------------------------------------------------------------------------------------------------ part B (classifier)
def contrasts(units):
    """(kind, name, diff_expected, left_binary, right_binary): right minus left."""
    out = []
    for key, u in units.items():
        if not own(u) or u["credit_expected"].mean() < FLOOR or u["name"] == "fp32":
            continue
        f = units[("matrix", u["model"], "baseline", "fp32")]
        out.append(("cell_vs_fp32", f"{u['model']}/{u['group']}/{u['name']}", u["credit_expected"] - f["credit_expected"],
                    f["credit_lowest"], u["credit_lowest"]))
        i8 = units.get(("matrix", u["model"], u["group"], "int8"))
        if i8 is not None and u["name"] != "int8":
            out.append(("cell_vs_int8", f"{u['model']}/{u['group']}/{u['name']}", u["credit_expected"] - i8["credit_expected"],
                        i8["credit_lowest"], u["credit_lowest"]))
    keys = sorted(k for k, u in units.items() if own(u) and u["credit_expected"].mean() >= FLOOR and u["name"] != "fp32")
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ua, ub = units[a], units[b]
            if ua["model"] != ub["model"]:
                continue
            if ua["group"] == ub["group"]:
                kind = "format_pair"
            elif ua["name"] == ub["name"]:
                kind = "recipe_pair"
            else:
                continue
            out.append((kind, f"{ua['model']}/{ua['group']}/{ua['name']}~{ub['group']}/{ub['name']}",
                        ub["credit_expected"] - ua["credit_expected"], ua["credit_lowest"], ub["credit_lowest"]))
    for key, u in units.items():
        if u["source"] == "sweep" and u["name"] != "wide" and u["credit_expected"].mean() >= FLOOR:
            wide = units[("sweep", "resnet18", u["group"], "wide")]
            out.append(("accumulator_vs_wide", f"{u['group']}/{u['name']}", u["credit_expected"] - wide["credit_expected"],
                        wide["credit_lowest"], u["credit_lowest"]))
    return out


def part_b_classifier(w, units):
    cs = contrasts(units)
    rows, discord = [], {}
    for kind, name, d, left, right in cs:
        lo, ro = int(((left == 1) & (right == 0)).sum()), int(((left == 0) & (right == 1)).sum())
        discord.setdefault(kind, []).append(((lo + ro) / 1000, ro / (lo + ro) if lo + ro else 0.5))
        res = resolution.subsample_contrast(d, left.astype(np.int8), right.astype(np.int8))
        widths_e = {n: v["width_expected_median"] for n, v in res.items()}
        widths_b = {n: v["width_binary_median"] for n, v in res.items()}
        row = {"kind": kind, "contrast": name, "diff_expected_pp": 100 * d.mean(), "discordance": (lo + ro) / 1000,
               "slope_expected": resolution.slope(widths_e), "slope_binary": resolution.slope(widths_b)}
        for n, v in res.items():
            row.update({f"{k}_n{n}": v[k] for k in v if k != "blocks"})
        for N in (4000, 9000):
            row[f"projected_halfwidth_expected_n{N}"] = widths_e[1000] / 2 * math.sqrt(1000 / N)
        rows.append(row)
    w.csv("partB-subsample.csv", rows)
    # summary per kind and n
    summary = []
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        entry = {"kind": kind, "contrasts": len(rs)}
        for n in resolution.NS:
            for k in ("width_expected_median", "width_binary_median", "se_expected_mean"):
                entry[f"{k}_n{n}_median"] = float(np.median([r[f"{k}_n{n}"] for r in rs]))
            if n <= 256:
                ratio = [r[f"between_block_sd_expected_n{n}"] / r[f"se_expected_mean_n{n}"] for r in rs if r[f"se_expected_mean_n{n}"] > 0]
                entry[f"between_block_sd_over_se_n{n}_median"] = float(np.median(ratio)) if ratio else None
        sl = [r["slope_expected"] for r in rs if r["slope_expected"] is not None]
        entry.update(slope_expected_median=float(np.median(sl)), slope_expected_q10=float(np.percentile(sl, 10)),
                     slope_expected_q90=float(np.percentile(sl, 90)))
        dd = np.array([x[0] for x in discord[kind]])
        entry.update({f"discordance_q{q}": float(np.percentile(dd, q)) for q in (5, 25, 50, 75, 95)})
        for N in (1000, 4000, 9000):
            entry[f"halfwidth_expected_n{N}_median"] = float(np.median([r["width_expected_median_n1000"] / 2 * math.sqrt(1000 / N) for r in rs]))
        summary.append(entry)
    w.csv("partB-subsample-summary.csv", summary)
    return rows, summary, discord


def part_b_coverage(w, discord):
    obs = discord["cell_vs_fp32"] + discord["cell_vs_int8"]
    pd = np.percentile([o[0] for o in obs], [5, 25, 50, 75, 95])
    asym = np.percentile([o[1] for o in obs if o[0] > 0], [5, 25, 50, 75, 95])
    scenarios = [(f"pd_q{qa}_asym_q{qb}", float(a), float(b)) for qa, a in zip((5, 25, 50, 75, 95), pd)
                 for qb, b in zip((5, 25, 50, 75, 95), asym)]
    scenarios += [(f"null_pd{int(100 * p)}", p, 0.5) for p in (0.01, 0.02, 0.05, 0.10, 0.20)]
    rows = []
    for s, (name, p, a) in enumerate(scenarios):
        for n in resolution.NS:
            c = resolution.coverage(p * (1 - a), p * a, n, scenario=s)
            rows.append({"scenario": name, "discordance": p, "right_only_share": a, "true_diff_pp": 100 * p * (2 * a - 1), "n": n, **c})
    w.csv("partB-coverage.csv", rows)
    return rows


def part_b_power(w, discord):
    rows = []
    for kind, obs in sorted(discord.items()):
        dd = np.array([o[0] for o in obs])
        for q in (25, 50, 75):
            p = float(np.percentile(dd, q))
            row = {"kind": kind, "discordance_quantile": q, "discordance": p}
            for n in (1000, 4000, 9000):
                row[f"mdd_pp_n{n}"] = 100 * resolution.mdd(n, p)
            for m in (0.25, 0.5, 1.0):
                for k in ("difference", "equivalence", "noninferiority"):
                    row[f"images_{k}_margin{m}"] = resolution.images_needed(p, m / 100, k)
            rows.append(row)
    checks = []
    med = float(np.median([o[0] for o in discord["format_pair"]]))
    for i, n in enumerate((1000, 4000, 9000)):
        delta = resolution.mdd(n, med)
        checks.append({"discordance": med, "n": n, "mdd_pp": 100 * delta,
                       "simulated_power_exact_mcnemar": resolution.simulated_power(n, med, delta, seed_tag=i)})
    w.csv("partB-power.csv", rows)
    w.csv("partB-power-check.csv", checks)
    return rows, checks


# ------------------------------------------------------------------------------------------------ detector (B, C, D)
def det_results():
    out = {}
    for spec in D.configurations():
        path = D.WORK / f"{spec[0]}--{spec[1]}.npz"
        if not path.exists():
            raise FileNotFoundError(f"detector worker output missing: {path}")
        with np.load(path) as z:
            out[spec] = {"info": json.loads(str(z["info"])), **{k: z[k] for k in z.files if k != "info"}}
    return out


def stored_bootstrap(info):
    path = ROOT / "artifacts/experiment_b2_det/bootstrap" / f"{info['configuration_sha256']}-1000-index.npz"
    with np.load(path) as z:
        return z["point"], z["draws"]


def part_det(w, det):
    fp = det[("fp32", "fp32")]
    # B: AP resolution against n
    rows = []
    _, fdraws = stored_bootstrap(fp["info"])
    for spec in sorted(D.SUBSAMPLE - {("fp32", "fp32")}) + [("int8", "conformant-minus-default")]:
        if spec[1] == "conformant-minus-default":
            left, right = det[("int8", "default")], det[("int8", "conformant")]
        else:
            left, right = fp, det[spec]
        row = {"contrast": f"{spec[0]} {spec[1]}" + ("" if spec[1].startswith("conformant-") else " minus FP32")}
        for n in D.SUB_N:
            widths, diffs = [], []
            for r in range(D.SUB_PERMS):
                a, b = left[f"r{r}_n{n}"], right[f"r{r}_n{n}"]  # blocks x (1+draws) x 2
                for k in range(a.shape[0]):
                    dd = 100 * (b[k, 1:, 0] - a[k, 1:, 0])
                    lo, hi = np.quantile(dd, [0.025, 0.975])
                    widths.append(hi - lo)
                    diffs.append(100 * (b[k, 0, 0] - a[k, 0, 0]))
            row[f"width_n{n}_median"] = float(np.median(widths))
            row[f"block_sd_n{n}"] = float(np.std(diffs, ddof=1)) if len(diffs) > 2 else None
        _, ldraws = stored_bootstrap(left["info"])
        _, rdraws = stored_bootstrap(right["info"])
        dd = 100 * (rdraws[:, 0] - ldraws[:, 0])
        lo, hi = np.quantile(dd, [0.025, 0.975])
        row.update(width_n1000=float(hi - lo), se_n1000=float(dd.std(ddof=1)),
                   diff_n1000=100 * (right["info"]["point"][0] - left["info"]["point"][0]))
        row["slope"] = resolution.slope({128: row["width_n128_median"], 256: row["width_n256_median"],
                                         512: row["width_n512_median"], 1000: row["width_n1000"]})
        row["projected_halfwidth_n4000"] = row["width_n1000"] / 2 * math.sqrt(1000 / 4000)
        row["mdd_n1000"] = 2.80 * row["se_n1000"]
        row["mdd_n4000"] = 2.80 * row["se_n1000"] / 2
        row["images_for_halfwidth_0_25"] = math.ceil(1000 * (row["width_n1000"] / 2 / 0.25) ** 2)
        rows.append(row)
    w.csv("partB-detector-resolution.csv", rows)
    # all configurations against FP32 at 1k from the stored bootstrap
    se_rows = []
    for spec, r in det.items():
        if spec == ("fp32", "fp32"):
            continue
        _, d = stored_bootstrap(r["info"])
        dd = 100 * (d[:, 0] - fdraws[:, 0])
        se_rows.append({"format": spec[0], "recipe": spec[1], "diff": 100 * (r["info"]["point"][0] - fp["info"]["point"][0]),
                        "se_n1000": float(dd.std(ddof=1)), "width_n1000": float(np.subtract(*np.quantile(dd, [0.975, 0.025]))),
                        "projected_halfwidth_n4000": float(np.subtract(*np.quantile(dd, [0.975, 0.025]))) / 4})
    w.csv("partB-detector-se.csv", se_rows)
    # C: tie orders
    crow = []
    for spec, r in det.items():
        fixed = 100 * r["info"]["point"][0]
        entry = {"format": spec[0], "recipe": spec[1], "fixed_rule": fixed, "distinct_scores": r["info"]["distinct_scores"],
                 "detections": r["info"]["detections"]}
        for scheme in ("image_order", "uniform"):
            v = 100 * r[scheme][:, 0]
            entry.update({f"{scheme}_mean": float(v.mean()), f"{scheme}_sd": float(v.std(ddof=1)),
                          f"{scheme}_p2_5": float(np.percentile(v, 2.5)), f"{scheme}_p97_5": float(np.percentile(v, 97.5)),
                          f"{scheme}_min": float(v.min()), f"{scheme}_max": float(v.max()),
                          f"{scheme}_fixed_rank": float((v < fixed).mean()),
                          f"{scheme}_mean_minus_fixed": float(v.mean() - fixed),
                          f"{scheme}_map50_mean": float(100 * r[scheme][:, 1].mean())})
        crow.append(entry)
    w.csv("partC-detector-tie-orders.csv", crow)
    # D: agreement, AP by area, score step
    drow = []
    names = ["map50_95", "map50", "map75", "ap_small", "ap_medium", "ap_large"]
    for spec, r in det.items():
        entry = {"format": spec[0], "recipe": spec[1]}
        entry.update({n: 100 * v for n, v in zip(names, r["info"]["stats12"][:6])})
        if spec != ("fp32", "fp32"):
            entry.update({f"{n}_minus_fp32": 100 * (a - b) for n, a, b in zip(names, r["info"]["stats12"][:6], fp["info"]["stats12"][:6])})
            entry.update(r["info"]["agreement"])
        if "area_draws" in r and spec != ("fp32", "fp32"):
            dd = 100 * (r["area_draws"][:, :6] - fp["area_draws"][:, :6])
            for i, n in enumerate(names):
                lo, hi = np.quantile(dd[:, i], [0.025, 0.975])
                entry[f"{n}_minus_fp32_ci95"] = [float(lo), float(hi)]
        drow.append(entry)
    w.csv("partD-detector-agreement.csv", drow)
    steps = []
    for fmt in ("int8", "int6"):
        r = det[(fmt, "default_head_logits")]
        dflt = det[(fmt, "default")]["info"]
        for unit, s in r["info"]["score_step"].items():
            steps.append({"format": fmt, "fp32": 100 * fp["info"]["point"][0], "head_logits_wide": 100 * r["info"]["point"][0],
                          "default": 100 * dflt["point"][0], "score_unit": int(unit) / 1e5,
                          "head_logits_scores_requantised": 100 * s["point"][0],
                          "head_logits_threshold_only": 100 * s["threshold_only_point"][0],
                          "detections_wide": r["info"]["detections"], "detections_requantised": s["detections"],
                          "detections_default": dflt["detections"], "units_inferred": r["info"]["score_units_e5"],
                          "units_unexplained": r["info"]["score_units_unexplained"]})
    w.csv("partD-score-step.csv", steps)
    return rows, se_rows, crow, drow, steps


LEDGER_RATES = {
    "matrix_cell_1k_seconds": {"low": 6, "high": 55, "source": "surveys/2026-10-02-code-capabilities-data-costs.md s7 (cell_wall_seconds, scalar formats, 3 classifiers, shared GPU)"},
    "matrix_process_overhead_seconds": {"low": 30, "high": 40, "source": "same survey s7"},
    "exact_engine_resnet18_seconds_per_image": {"low": 0.15, "high": 0.35, "source": "same survey s7 (130-290 s per 1k)"},
    "exact_engine_mobilenet_seconds_per_image": {"low": 0.36, "high": 0.36, "source": "same survey s7 (about 6 min per 1k, estimated from 32-image runs)"},
    "detector_config_1k_gpu_minutes": {"low": 0.9, "high": 4.0, "source": "surveys/2026-10-02-detector-data-protocol.md s1 cost (median 0.9, up to 4 under contention)"},
    "detector_bootstrap_1k_cpu_minutes": {"low": 8, "high": 9, "source": "same survey (2,000-draw COCOeval bootstrap per configuration)"},
}


def ledger():
    """Confirmation compute ledger (projection from measured per-run rates; nothing here was run)."""
    r = LEDGER_RATES
    out = []
    def add(name, units, hours_low, hours_high, resource, note):
        out.append({"scenario": name, "units": units, "hours_low": round(hours_low, 2), "hours_high": round(hours_high, 2),
                    "resource": resource, "note": note})
    for configs in (12, 36):
        sec = configs * 9 * r["matrix_cell_1k_seconds"]["low"] + configs * r["matrix_process_overhead_seconds"]["low"]
        sec_hi = configs * 9 * r["matrix_cell_1k_seconds"]["high"] + configs * r["matrix_process_overhead_seconds"]["high"]
        add(f"simulator classifier cells on the 9k remainder", f"{configs} configurations", sec / 3600, sec_hi / 3600,
            "GPU (shared slot)", "cell time x 9 plus one process overhead each")
    for runs in (24, 56, 73):
        add("exact-engine ResNet18 runs on 9k", f"{runs} runs", runs * 9000 * r["exact_engine_resnet18_seconds_per_image"]["low"] / 3600,
            runs * 9000 * r["exact_engine_resnet18_seconds_per_image"]["high"] / 3600, "GPU", "56/73 = STATE.md grid sizes cited by the detector survey")
    add("exact-engine MobileNet runs on 9k", "10 runs", 10 * 9000 * 0.36 / 3600, 10 * 9000 * 0.36 / 3600, "GPU", "rate estimated, not measured at 1k")
    for configs in (6, 18):
        add("detector configurations on the 4k complement", f"{configs} configurations", configs * 4 * r["detector_config_1k_gpu_minutes"]["low"] / 60,
            configs * 4 * r["detector_config_1k_gpu_minutes"]["high"] / 60, "GPU", "x4 of the 1k time")
        add("detector COCOeval bootstrap on 4k (2,000 draws)", f"{configs} configurations", configs * 4 * 8 / 60, configs * 4 * 9 / 60,
            "CPU-hours", "accumulate time scales about linearly with images")
    return {"rates": LEDGER_RATES, "rows": out, "status": "projection from measured development rates; shared-GPU rates, not exclusive timings"}


# ------------------------------------------------------------------------------------------------ main
def main(out, parts=None):
    parts = parts or ["A", "B", "C", "D", "E"]
    w = Writer(out)
    units = U.load_all()
    result = {}
    if "A" in parts:
        result["A"] = part_a(w, units)
    if "B" in parts:
        rows, summary, discord = part_b_classifier(w, units)
        part_b_coverage(w, discord)
        part_b_power(w, discord)
        w.json("partB-ledger.json", ledger())
    if "C" in parts:
        trows = []
        for key, u in units.items():
            if u["name"] == "fp32":
                continue
            trows.append({"source": u["source"], "model": u["model"], "group": u["group"], "unit": u["name"],
                          "family": ties.family_of(u), "bits": u["bits"], "top1_expected": 100 * u["credit_expected"].mean(),
                          **ties.unit_ties(u)})
        w.csv("partC-classifier-ties.csv", trows)
        w.csv("partC-classifier-ties-by-family.csv", ties.summarise(trows))
    if "D" in parts:
        part_det(w, det_results())
    if "E" in parts:
        cells = fixed_width.grid(units)
        w.csv("partE-grid.csv", [{k: v for k, v in c.items() if k != "credit"} for c in cells.values()])
        w.csv("partE-orders.csv", fixed_width.orders(cells))
        w.csv("partE-float-accumulators.csv", fixed_width.float_types(units))
    protocol = ROOT / PROTOCOL
    w.json(f"manifest-{'-'.join(parts)}.json", {
        "written": datetime.now().astimezone().isoformat(timespec="seconds"), "protocol": PROTOCOL,
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(), "seed": SEED,
        "evidence": "development evidence: ImageNet and COCO screen-1k lists only",
        "analysis_code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT / "tools/analysis/quality_metrics_v1").glob("*.py"))},
        "files": [p.name for p in w.files]})
    return 0
