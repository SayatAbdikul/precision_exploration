"""Lane Q4 summaries (CPU): results/summaries/b2-attrib-v1/ (written once) from the arm records.

    PYTHONPATH=. .venv/bin/python -m tools.experiment_b2_attrib.summarize [--write]

Development evidence: every top-1 number is the frozen ImageNet 1k screen (expected credit for tied logits);
intervals are the project's paired bootstrap (tools/analysis/b2_matrix.interval).
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from tools.experiment_b.common import ROOT, file_hash
from . import analysis

OUT = ROOT / "results/summaries/b2-attrib-v1"
PROTOCOLS = ["public/experiments/configs/breadth-study/b2-attrib-protocol-v1.json",
             "public/experiments/configs/breadth-study/b2-attrib-protocol-v1-addendum-1.json",
             "public/experiments/configs/breadth-study/b2-attrib-protocol-v1-addendum-2.json",
             "public/experiments/configs/breadth-study/b2-attrib-protocol-v1-addendum-3.json",
             "public/experiments/configs/breadth-study/b2-attrib-protocol-v1-addendum-4.json",
             "public/experiments/configs/breadth-study/b2-attrib-protocol-v1-addendum-5.json",
             "public/experiments/configs/breadth-study/b2-attrib-protocol-v1-addendum-6.json",
             "public/experiments/configs/breadth-study/b2-attrib-protocol-v1-addendum-7.json"]
FOLD_DIAGNOSTIC = ROOT / "artifacts/experiment_b2_attrib/folddiag/fold-diagnostic-v1.json"
ARMS = ROOT / "artifacts/experiment_b2_attrib/arms"
MBV3 = "mobilenet_v3_large"
A_FORMATS = ("int8", "int6", "fp6_e2m3")
B_FORMATS = ("int8", "int6", "fp6_e2m3", "fp8_e4m3fn")
C_FORMATS = ("int6", "fp6_e2m3", "fp6_e3m2", "posit6_es1", "log6")


def groups_of(model):
    protocol = json.loads((ROOT / PROTOCOLS[0]).read_text())
    return protocol["groups"][model]


def accumulator(meta):
    acc = meta.get("accumulator")
    if not acc:
        return None
    layers = acc["layers"]
    increases = [v["bits_arm"] - v["bits_default"] for v in layers.values()]
    pre = [v["bits_arm_presubtracted_operand"] - v["bits_default"] for v in layers.values()
           if "bits_arm_presubtracted_operand" in v]
    return {"network_max_bits_default": acc["network_max_bits_default"], "network_max_bits_arm": acc["network_max_bits_arm"],
            "layers_changed": len(layers), "max_layer_increase_bits": max(increases, default=0),
            "layers_with_increase": sum(i > 0 for i in increases),
            "max_layer_increase_bits_presubtracted": max(pre, default=None) if pre else None}


def channel_cost(meta):
    pc = meta.get("per_channel") or {}
    if not pc:
        return None
    folded, requant = 0, 0
    for row in pc.values():
        labels = [label for _, label in row.get("consumers", [])]
        folded += sum(label in ("conv", "linear") for label in labels)
        requant += sum(label not in ("conv", "linear") for label in labels)
    return {"boundaries": len(pc), "consumer_edges_folded_into_weights": folded,
            "consumer_edges_needing_per_channel_requantization": requant}


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
        out[name] = {**row, "affine_applied": len(meta.get("affine_applied") or []),
                     "accumulator": accumulator(meta), "per_channel_cost": channel_cost(meta),
                     "stage": record.get("stage"), "audit_median_sqnr_db": audit_median(record)}
    return out


def audit_median(record):
    occupancy = record.get("occupancy")
    if not occupancy:
        return None
    values = [v for v in occupancy["sqnr_db"].values() if v is not None and math.isfinite(v)]
    return round(float(np.median(values)), 2) if values else None


def leave_one_out(model, fmt):
    records = analysis.arm_records(model, fmt)
    if "ref_default" not in records:
        return None
    loo = {k: v for k, v in records.items() if k.startswith("wide:node.")}
    if not loo:
        return None
    default = analysis.arm_credits(records["ref_default"])
    sqnr = analysis.cell_sqnr(analysis.cells("default")[(model, fmt)])
    roles = {n: g for g, members in groups_of(model)["role"].items() for n in members}
    rows = {}
    for name, record in loo.items():
        node = name.split(".", 1)[1]
        c = analysis.arm_credits(record)
        rows[node] = {"role": roles.get(node), "sqnr_db": sqnr.get(node),
                      "recovery_pp": round(100 * float(c.mean() - default.mean()), 2),
                      "wide_minus_default": analysis.paired(c, default),
                      "kl_reduction": round(records["ref_default"]["kl_nats_mean"] - record["kl_nats_mean"], 4)}
    x = [rows[n]["sqnr_db"] for n in rows if rows[n]["sqnr_db"] is not None]
    y = [rows[n]["recovery_pp"] for n in rows if rows[n]["sqnr_db"] is not None]
    k = [rows[n]["kl_reduction"] for n in rows if rows[n]["sqnr_db"] is not None]
    return {"nodes": rows, "sum_recovery_pp": round(sum(r["recovery_pp"] for r in rows.values()), 2),
            "spearman_sqnr_vs_recovery": analysis.spearman(x, y), "spearman_sqnr_vs_kl_reduction": analysis.spearman(x, k)}


def fixbc_table(model, fmt):
    """Addendum 3 diagnostic: wide/only arms with the default's corrected biases vs. the same arms with a refit."""
    records = analysis.arm_records(model, fmt)
    if "ref_default+fixbc" not in records or "ref_default" not in records:
        return None
    control = records["ref_default+fixbc"]["logits_sha256"] == records["ref_default"]["logits_sha256"]
    base_refit = analysis.arm_credits(records["ref_default"])
    base_fixed = analysis.arm_credits(records["ref_default+fixbc"])
    rows = {}
    for name, record in records.items():
        if not name.endswith("+fixbc") or name == "ref_default+fixbc":
            continue
        plain = name[: -len("+fixbc")]
        fixed = analysis.arm_credits(record)
        row = {"top1_fixbc": round(100 * float(fixed.mean()), 2), "kl_fixbc": record["kl_nats_mean"],
               "fixbc_minus_default": analysis.paired(fixed, base_fixed)}
        if plain in records:
            refit = analysis.arm_credits(records[plain])
            row.update({"top1_refit": round(100 * float(refit.mean()), 2), "kl_refit": records[plain]["kl_nats_mean"],
                        "refit_minus_default": analysis.paired(refit, base_refit),
                        "fixbc_minus_refit": analysis.paired(fixed, refit)})
        rows[plain] = row
    return {"control_logits_sha256_equal": control, "arms": rows}


def predictor_pool(attributions):
    """Across (model, format) pairs: do group NSR / min SQNR rank the group damage and recovery?"""
    rows = []
    for result in attributions:
        for group, row in result["groups"].items():
            rows.append({"model": result["model"], "format": result["format"], "group": group, **row})
    out = {"groups": len(rows)}
    for key in ("nsr_db", "min_sqnr_db"):
        usable = [r for r in rows if r[key] is not None]
        for target in ("damage_pp", "recovery_pp", "kl_only", "kl_reduction"):
            out[f"{key}_vs_{target}"] = analysis.spearman([r[key] for r in usable], [r[target] for r in usable])
    return out


def fold_diagnostic():
    """Addendum 6 (CPU): per-model summary of the pcf:all fold diagnostic, if it was run."""
    if not FOLD_DIAGNOSTIC.exists():
        return None
    data = json.loads(FOLD_DIAGNOSTIC.read_text())
    return {"file": str(FOLD_DIAGNOSTIC.relative_to(ROOT)), "sha256": file_hash(FOLD_DIAGNOSTIC),
            "models": {m: {"summary": v["summary"], "worst5": v["worst5"]} for m, v in data["models"].items()}}


def records_manifest():
    """sha256 of every arm record (JSON) the summaries read, per model--format folder."""
    out = {}
    for folder in sorted(p for p in ARMS.iterdir() if p.is_dir()):
        out[folder.name] = {f.stem: file_hash(f) for f in sorted(folder.glob("*.json"))}
    return out


def build():
    summary = {"evidence": "development: frozen ImageNet 1k screen (top-1 expected credit); audit on its first 128 images",
               "protocols": {p: file_hash(ROOT / p) for p in PROTOCOLS if (ROOT / p).exists()}}
    g = groups_of(MBV3)
    summary["a0_cross_cell"] = analysis.cross_cell()
    summary["a0_group_sqnr"] = {fmt: analysis.group_sqnr(MBV3, fmt, g["role"]) for fmt in B_FORMATS}
    attributions = []
    summary["a1"] = {}
    for fmt in A_FORMATS:
        records = analysis.arm_records(MBV3, fmt)
        if "ref_default" in records and "ref_weights_only" in records:
            entry = {scheme: analysis.attribution(MBV3, fmt, g[scheme], scheme) for scheme in ("role", "position")}
            summary["a1"][fmt] = entry
            attributions.append(entry["role"])
    summary["a2"] = {fmt: leave_one_out(MBV3, fmt) for fmt in A_FORMATS}
    summary["b"] = {fmt: repair_table(MBV3, fmt) for fmt in B_FORMATS}
    summary["b_regression"] = {m: repair_table(m, "int8") for m in ("mobilenet_v2", "resnet18")}
    summary["c"] = {}
    for model in ("resnet18", "mobilenet_v2"):
        for fmt in C_FORMATS:
            records = analysis.arm_records(model, fmt)
            if "ref_default" in records and "ref_weights_only" in records:
                result = analysis.attribution(model, fmt, groups_of(model)["role"], "role")
                summary["c"][f"{model}/{fmt}"] = result
                attributions.append(result)
    summary["c_repairs"] = {}
    for model in ("resnet18", "mobilenet_v2"):
        for fmt in C_FORMATS:
            table = repair_table(model, fmt)
            if table and len(table) > 1:
                summary["c_repairs"][f"{model}/{fmt}"] = table  # addendum 7: cross-network stem repair
    summary["a1_fixbc"] = {fmt: fixbc_table(MBV3, fmt) for fmt in A_FORMATS}
    summary["b_fold_diagnostic"] = fold_diagnostic()
    summary["records_manifest"] = records_manifest()
    summary["group_sqnr_predicts_effect"] = predictor_pool(attributions)
    summary["group_sqnr_predicts_effect_by_model"] = {
        m: predictor_pool([a for a in attributions if a["model"] == m]) for m in ("mobilenet_v3_large", "mobilenet_v2", "resnet18")}
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    summary = build()
    if args.write:
        if OUT.exists():
            raise SystemExit(f"{OUT} exists: summaries are written once (use a new versioned folder)")
        OUT.mkdir(parents=True)
        for key, value in summary.items():
            (OUT / f"{key}.json").write_text(json.dumps(value, indent=1, sort_keys=True, default=float) + "\n")
    print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in summary.items() if k != "protocols"},
                     default=str)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
