"""Lane Q4 v2 summaries (agent r6, protocol addendum 8): results/summaries/b2-attrib-v1/v2/ (written once).

    PYTHONPATH=. .venv/bin/python -m tools.experiment_b2_attrib.summarize2 [--write]

Changes from v1 (``summarize.py``, whose files stay):
* review B3: leave-one-out takes only records named ``wide:node.<node>`` without a ``+`` suffix;
* repair tables carry the ``pcu`` arms (unbiased per-channel sample) and label every ``pcf`` arm as superseded,
  with ``pcu`` minus ``pcf`` paired differences and the per-channel scale diagnostics (search-floor hits);
* accumulator: certified network maximum also for the pre-subtracted operand form (unchanged layers from the
  sealed scaled-bridge certificate);
* predictor statistics with bootstrap intervals, a within-cell (stratified) permutation test and Holm adjustment;
* stem rank by group SQNR and the largest-damage groups with their paired intervals in every attributed cell;
* sampler sensitivity: ``pcu`` next to the reviewer's coprime-stride reruns (review-scratch/out-*.json).

Development evidence: every top-1 number is the frozen ImageNet 1k screen (expected credit for tied logits).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from tools.experiment_b.common import ROOT, file_hash, unseal
from . import analysis, summarize

OUT = ROOT / "results/summaries/b2-attrib-v1/v2"
ADDENDUM_8 = "public/experiments/configs/breadth-study/b2-attrib-protocol-v1-addendum-8.json"
PROTOCOLS = summarize.PROTOCOLS + [ADDENDUM_8]
FOLD_V2 = ROOT / "artifacts/experiment_b2_attrib/folddiag/fold-diagnostic-v2-pcu.json"
REVIEW = ROOT / "artifacts/experiment_b2_attrib/review-scratch"
CERTS = {fmt: ROOT / "artifacts/scaled_bridge_v2/runs/1f75c9232c8a0202482fe9bce9a46360c7cc0999c5b0bcf7df5ebcc0446fc863"
         / f"mobilenet_v3_large-{fmt}-default-b2/certificate.json" for fmt in ("int8", "int6")}
MBV3 = summarize.MBV3
SEED = 20261002
RESAMPLES = 10000
FLOOR = 2.0 ** -6


# ---------------------------------------------------------------- statistics

def _rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_ci(x, y, strata=None):
    """Spearman rho, its scipy p, and a percentile bootstrap 95 % interval (resampling units within strata)."""
    base = analysis.spearman(list(x), list(y))
    if base is None:
        return None
    x, y = np.asarray(x, float), np.asarray(y, float)
    strata = np.zeros(len(x), int) if strata is None else np.asarray(strata)
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    draws = []
    for _ in range(RESAMPLES):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        r = _rho(x[idx], y[idx])
        if math.isfinite(r):
            draws.append(r)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {**base, "ci95": [round(float(low), 2), round(float(high), 2)], "bootstrap_valid": len(draws),
            "bootstrap": f"percentile, {RESAMPLES} resamples, strata={'cell' if len(groups) > 1 else 'none'}, seed {SEED}"}


def within_cell(cells, xkey, ykey):
    """Mean within-cell Spearman rho and its stratified permutation p (y permuted within each cell)."""
    usable = [(np.asarray([r[xkey] for r in rows], float), np.asarray([r[ykey] for r in rows], float))
              for rows in cells if len(rows) >= 4]
    observed = float(np.mean([_rho(x, y) for x, y in usable]))
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(RESAMPLES):
        value = float(np.mean([_rho(x, rng.permutation(y)) for x, y in usable]))
        hits += abs(value) >= abs(observed) - 1e-12
    return {"mean_within_cell_rho": round(observed, 3), "cells": len(usable),
            "per_cell_rho": [round(_rho(x, y), 3) for x, y in usable],
            "stratified_permutation_p": (hits + 1) / (RESAMPLES + 1), "permutations": RESAMPLES, "seed": SEED}


def holm(pvalues):
    """Holm-adjusted p-values for ``{label: p}``."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (label, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[label] = running
    return out


# ---------------------------------------------------------------- attribution cells

def attributed_cells():
    """``[(model, fmt, attribution)]`` of every cell with both reference arms (MobileNetV3 A1 and Part C)."""
    out = []
    for fmt in summarize.A_FORMATS:
        out.append((MBV3, fmt, analysis.attribution(MBV3, fmt, summarize.groups_of(MBV3)["role"], "role")))
    for model in ("resnet18", "mobilenet_v2"):
        for fmt in summarize.C_FORMATS:
            records = analysis.arm_records(model, fmt)
            if "ref_default" in records and "ref_weights_only" in records:
                out.append((model, fmt, analysis.attribution(model, fmt, summarize.groups_of(model)["role"], "role")))
    return out


def cell_view(model, fmt, result):
    groups = result["groups"]
    by_sqnr = sorted(groups, key=lambda g: -groups[g]["nsr_db"])
    by_damage = sorted(groups, key=lambda g: -groups[g]["damage_pp"])
    view = {"model": model, "format": fmt, "groups": len(groups), "activation_loss_pp": result["activation_loss_pp"],
            "group_sqnr_ranking_db": [[g, round(groups[g]["nsr_db"], 1)] for g in by_sqnr],
            "stem_rank_by_group_sqnr": by_sqnr.index("stem") + 1,
            "largest_damage": [{"group": g, "damage_pp": groups[g]["damage_pp"],
                                "ci95_pp": [-v for v in reversed(groups[g]["only_minus_weights_only"]["ci95_pp"])],
                                "recovery_pp": groups[g]["recovery_pp"], "recovery_ci95_pp": groups[g]["wide_minus_default"]["ci95_pp"],
                                "group_sqnr_db": round(groups[g]["nsr_db"], 1)} for g in by_damage[:3]]}
    records = analysis.arm_records(model, fmt)
    first, second = by_damage[0], by_damage[1]
    view["first_minus_second_damage"] = {"groups": [first, second], **analysis.paired(
        analysis.arm_credits(records[f"only:role.{second}"]), analysis.arm_credits(records[f"only:role.{first}"]))}
    view["predictors"] = {f"group_sqnr_vs_{t}": spearman_ci([groups[g]["nsr_db"] for g in groups], [groups[g][t] for g in groups])
                          for t in ("damage_pp", "recovery_pp", "kl_only")}
    view["predictors"]["min_sqnr_vs_damage_pp"] = spearman_ci([groups[g]["min_sqnr_db"] for g in groups],
                                                              [groups[g]["damage_pp"] for g in groups])
    return view


def predictors(cells, loo):
    rows = [[{"nsr_db": r["nsr_db"], "min_sqnr_db": r["min_sqnr_db"], **r} for r in result["groups"].values()]
            for _, _, result in cells]
    flat = [(i, r) for i, cell in enumerate(rows) for r in cell]
    pooled = {}
    for xkey, ykey in (("nsr_db", "damage_pp"), ("min_sqnr_db", "damage_pp"), ("nsr_db", "recovery_pp"), ("nsr_db", "kl_only")):
        ok = [(i, r) for i, r in flat if r[xkey] is not None and r[ykey] is not None]
        pooled[f"{xkey}_vs_{ykey}"] = {
            "pooled": spearman_ci([r[xkey] for _, r in ok], [r[ykey] for _, r in ok], [i for i, _ in ok]),
            "within_cell": within_cell([[r for r in cell if r[xkey] is not None and r[ykey] is not None] for cell in rows],
                                       xkey, ykey)}
    by_model = {}
    for model in (MBV3, "mobilenet_v2", "resnet18"):
        picked = [(i, r) for i, r in flat if cells[i][0] == model]
        by_model[model] = spearman_ci([r["nsr_db"] for _, r in picked], [r["damage_pp"] for _, r in picked],
                                      [i for i, _ in picked])
    per_cell = {f"{m}/{f}": cell_view(m, f, res) for m, f, res in cells}
    pvalues = {}
    for key, view in per_cell.items():
        for name, stat in view["predictors"].items():
            if stat:
                pvalues[f"{key}:{name}"] = stat["p"]
    for name, stat in pooled.items():
        pvalues[f"pooled:{name}"] = stat["pooled"]["p"]
        pvalues[f"within_cell:{name}"] = stat["within_cell"]["stratified_permutation_p"]
    for model, stat in by_model.items():
        if stat:
            pvalues[f"by_model:{model}:nsr_db_vs_damage_pp"] = stat["p"]
    for fmt, result in loo.items():
        if result and result["spearman_sqnr_vs_recovery"]:
            pvalues[f"leave_one_out:{fmt}:node_sqnr_vs_recovery"] = result["spearman_sqnr_vs_recovery"]["p"]
    adjusted = holm(pvalues)
    return {"per_cell": per_cell, "pooled": pooled, "by_model_group_sqnr_vs_damage": by_model,
            "holm": {"family": "every SQNR-predictor p-value in this block (per cell, pooled, within-cell, by model, leave-one-out)",
                     "tests": len(pvalues), "raw_p": pvalues, "adjusted_p": adjusted,
                     "below_0.05_after_adjustment": sorted(k for k, v in adjusted.items() if v < 0.05)}}


def leave_one_out(model, fmt):
    """Review B3 fix: only ``wide:node.<node>`` records without a ``+`` suffix."""
    records = analysis.arm_records(model, fmt)
    if "ref_default" not in records:
        return None
    loo = {k: v for k, v in records.items() if k.startswith("wide:node.") and "+" not in k}
    if not loo:
        return None
    default = analysis.arm_credits(records["ref_default"])
    sqnr = analysis.cell_sqnr(analysis.cells("default")[(model, fmt)])
    roles = {n: g for g, members in summarize.groups_of(model)["role"].items() for n in members}
    rows = {}
    for name, record in loo.items():
        node = name.split(".", 1)[1]
        c = analysis.arm_credits(record)
        rows[node] = {"role": roles.get(node), "sqnr_db": sqnr.get(node),
                      "recovery_pp": round(100 * float(c.mean() - default.mean()), 2),
                      "wide_minus_default": analysis.paired(c, default),
                      "kl_reduction": round(records["ref_default"]["kl_nats_mean"] - record["kl_nats_mean"], 4)}
    usable = [n for n in rows if rows[n]["sqnr_db"] is not None and math.isfinite(rows[n]["sqnr_db"])]
    return {"nodes": rows, "node_count": len(rows), "excluded": sorted(k for k in records if k.startswith("wide:node.") and "+" in k),
            "sum_recovery_pp": round(sum(r["recovery_pp"] for r in rows.values()), 2),
            "spearman_sqnr_vs_recovery": spearman_ci([rows[n]["sqnr_db"] for n in usable], [rows[n]["recovery_pp"] for n in usable]),
            "spearman_sqnr_vs_kl_reduction": analysis.spearman([rows[n]["sqnr_db"] for n in usable], [rows[n]["kl_reduction"] for n in usable])}


# ---------------------------------------------------------------- repairs

def certificate_bits(fmt):
    path = CERTS.get(fmt)
    if path is None or not path.exists():
        return None
    return {name: row["signed_bits_absolute"] for name, row in unseal(path)["certificates"].items()}


def accumulator_v2(meta, fmt):
    acc = summarize.accumulator(meta)
    if acc is None:
        return None
    layers = meta["accumulator"]["layers"]
    cert = certificate_bits(fmt)
    if cert:
        unchanged = max((b for name, b in cert.items() if name not in layers), default=0)
        arm = max([unchanged] + [v["bits_arm"] for v in layers.values()])
        pre = max([unchanged] + [v.get("bits_arm_presubtracted_operand", v["bits_arm"]) for v in layers.values()])
        acc.update(network_max_bits_arm_check=arm, network_max_bits_presubtracted=pre,
                   certificate_matches_layer_defaults=all(cert.get(n) == v["bits_default"] for n, v in layers.items()),
                   presubtracted_increase_layers=sorted(n for n, v in layers.items()
                                                        if v.get("bits_arm_presubtracted_operand", 0) > v["bits_default"]),
                   zero_points=sorted({v["zero_point"] for v in layers.values() if "zero_point" in v}))
        widest = [n for n, v in layers.items() if v.get("bits_arm_presubtracted_operand", v["bits_arm"]) == pre]
        acc["presubtracted_widest_layers"] = sorted(widest)
    zero_points = [v["zero_point"] for v in layers.values() if "zero_point" in v]
    if zero_points:
        acc["zero_point_range"] = [min(zero_points), max(zero_points)]
    return acc


def per_channel_diag(meta):
    pc = meta.get("per_channel") or {}
    if not pc:
        return None
    ratios = {n: v.get("min_ratio_to_maxabs_scale") for n, v in pc.items()}
    floor_hits = sorted(n for n, r in ratios.items() if r is not None and r <= FLOOR + 1e-12)
    samples = [v.get("sample") for v in pc.values() if v.get("sample")]
    spreads = [v.get("scale_spread_log2") for v in pc.values() if v.get("scale_spread_log2") is not None]
    out = {"boundaries": len(pc), "search_floor_hits": floor_hits,
           "median_scale_spread_log2": round(float(np.median(spreads)), 2) if spreads else None,
           "max_scale_spread_log2": round(float(np.max(spreads)), 2) if spreads else None}
    if samples:
        out["min_distinct_columns_fraction"] = round(min(s["distinct_columns"] / s["columns"] for s in samples if "columns" in s), 3)
    return out


def sampler(name):
    if "pcu:" in name:
        return "pcu (unbiased random sample; addendum 8)"
    if "pcf:" in name:
        return "pcf (border-column sample; superseded by pcu, review B1)"
    if "pc:" in name:
        return "pc (v1 sparse samples)"
    return None


def repair_table(model, fmt):
    records = analysis.arm_records(model, fmt)
    if "ref_default" not in records:
        return None
    table = analysis.arm_table(model, fmt, records)
    out = {}
    for name, row in table.items():
        record = records[name]
        if record["arm_kind"] != "recipe" and name != "ref_default":
            continue
        meta = record["meta"]
        entry = {**row, "sampler": sampler(name), "affine_applied": len(meta.get("affine_applied") or []),
                 "accumulator": accumulator_v2(meta, fmt), "per_channel_cost": summarize.channel_cost(meta),
                 "per_channel_diagnostics": per_channel_diag(meta), "stage": record.get("stage"),
                 "audit_median_sqnr_db": summarize.audit_median(record)}
        twin = name.replace("pcu:", "pcf:")
        if "pcu:" in name and twin in records:
            entry["pcu_minus_pcf"] = analysis.paired(analysis.arm_credits(record), analysis.arm_credits(records[twin]))
            entry["pcf_top1"] = table[twin]["top1_expected"]
        if record.get("selections"):
            entry["selections"] = record["selections"]
        out[name] = entry
    return out


def sampler_sensitivity():
    """pcu (this lane) next to the reviewer's coprime-stride reruns of the same pcf arms (both unbiased samplers)."""
    files = {"mobilenet_v2/int8": "out-mbv2-int8.json", "resnet18/int8": "out-r18-int8.json",
             "mobilenet_v3_large/int6": "out-mbv3-int6.json", "mobilenet_v3_large/int8": "out-mbv3-int8-unb.json"}
    rows = {}
    for cell, filename in files.items():
        path = REVIEW / filename
        if not path.exists():
            continue
        model, fmt = cell.split("/")
        records = analysis.arm_records(model, fmt)
        for key, value in json.loads(path.read_text())["arms"].items():
            if not key.endswith("@unbiased"):
                continue
            pcf_name = key[: -len("@unbiased")]
            pcu_name = pcf_name.replace("pcf:", "pcu:")
            pcu = records.get(pcu_name)
            rows[f"{cell}:{pcf_name}"] = {"pcf_border_sample": round(records[pcf_name]["readout"]["top1_expected_percent"], 2)
                                          if pcf_name in records else None,
                                          "reviewer_coprime_stride": round(value["top1_expected"], 2),
                                          "pcu": round(pcu["readout"]["top1_expected_percent"], 2) if pcu else None}
    return {"source": str(REVIEW.relative_to(ROOT)), "rows": rows}


def fold_diagnostic_v2():
    if not FOLD_V2.exists():
        return None
    data = json.loads(FOLD_V2.read_text())
    return {"file": str(FOLD_V2.relative_to(ROOT)), "sha256": file_hash(FOLD_V2),
            "models": {m: {"summary": v["summary"], "worst5": v["worst5"]} for m, v in data["models"].items()}}


def build():
    summary = {"evidence": "development: frozen ImageNet 1k screen (top-1 expected credit); audit on its first 128 images",
               "version": "v2 (protocol addendum 8; review 1 fixes)",
               "protocols": {p: file_hash(ROOT / p) for p in PROTOCOLS if (ROOT / p).exists()}}
    loo = {fmt: leave_one_out(MBV3, fmt) for fmt in summarize.A_FORMATS}
    summary["a2"] = loo
    cells = attributed_cells()
    summary["predictors"] = predictors(cells, loo)
    summary["b"] = {fmt: repair_table(MBV3, fmt) for fmt in summarize.B_FORMATS}
    summary["b_regression"] = {m: repair_table(m, "int8") for m in ("mobilenet_v2", "resnet18")}
    summary["c_repairs"] = {"mobilenet_v2/fp6_e2m3": repair_table("mobilenet_v2", "fp6_e2m3")}
    summary["sampler_sensitivity"] = sampler_sensitivity()
    summary["b_fold_diagnostic_v2"] = fold_diagnostic_v2()
    summary["records_manifest"] = summarize.records_manifest()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--out", default=None, help="dry-run folder (default: print only)")
    args = parser.parse_args(argv)
    summary = build()
    target = OUT if args.write else (Path(args.out) if args.out else None)
    if target is not None:
        if target.exists() and args.write:
            raise SystemExit(f"{target} exists: summaries are written once (use a new versioned folder)")
        target.mkdir(parents=True, exist_ok=not args.write)
        for key, value in summary.items():
            (target / f"{key}.json").write_text(json.dumps(value, indent=1, sort_keys=True, default=float) + "\n")
    print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in summary.items() if k != "protocols"},
                     default=str)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
