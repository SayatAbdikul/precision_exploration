"""Accumulator-width sweep of the existing integer MAC at equal clock speed.

Subcommands
  conformance  run the existing RTL conformance generator/oracle for every
               (multiplier width, accumulator width) pair (a small driver: the
               existing tool only ships int32/int64 accumulator manifests, so
               the driver supplies the same manifest with another width through
               the tool's own ``format_named`` hook; nothing existing is edited)
  synth        timing-driven mapping + OpenSTA for each verified pair and each
               clock target, with the stage-2 method imported unchanged from
               tools/hardware/integer_isoclock_matrix.py
  summarize    derived tables, fits, and the preliminary quality-cost join
               (see the second half of this file)

Only parameter values of public/generic_rtl/mac/integer_mac.sv are changed.
The RTL accumulator SATURATES on overflow (function ``saturate``); it does
not wrap, and no accumulator logic is added here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.hardware import integer_isoclock_matrix as m  # noqa: E402  stage-2 method, unmodified
from tools.hardware import integer_mac_baseline as base  # noqa: E402  existing conformance tool, unmodified
from public.formats.oracle.manifest import load_manifest  # noqa: E402
from public.inference.reference.arithmetic import Accumulator  # noqa: E402

WIDTHS = (4, 5, 6, 8)
ACC_WIDTHS = (16, 20, 24, 28, 32, 40, 48, 56, 64)
TARGETS_NS = (12, 10, 8, 6, 5, 4)
PERIOD_TOL_NS = 1e-4          # periods are printed to 6 decimals; compare with a tolerance
OUT = ROOT / "artifacts/ppa/ics55-integer-accwidth-v1"
SUMMARY = ROOT / "results/summaries/ics55-integer-accwidth-v1.json"
TABLE = ROOT / "results/tables/ics55-integer-accwidth-v1.csv"
JOIN_TABLE = ROOT / "results/tables/ics55-integer-quality-cost-prelim-v1.csv"


# ---------------------------------------------------------------- conformance
def accumulator_manifest(bits: int) -> dict:
    """The existing int32_accumulator manifest with another width (same rules)."""
    template = json.loads((ROOT / "public/formats/manifests/accumulators/int32_accumulator.json").read_text())
    return {**template, "name": f"int{bits}_accumulator", "bits": bits,
            "numeric": {**template["numeric"], "integer_bits": bits - 1}}


def install_width_hook() -> None:
    """Make the existing conformance tool accept intN_accumulator for any N."""
    original = base.format_named

    def hooked(name: str):
        if name.startswith("int") and name.endswith("_accumulator"):
            bits = int(name[3:-len("_accumulator")])
            if bits not in (32, 64):
                return Accumulator(accumulator_manifest(bits))
        return original(name)

    base.format_named = hooked


def check_hook_matches_shipped() -> None:
    """The hook's manifest, at 32 and 64 bits, equals the shipped manifests."""
    for bits in (32, 64):
        shipped = json.loads((ROOT / f"public/formats/manifests/accumulators/int{bits}_accumulator.json").read_text())
        shipped.pop("notes", None)  # free-text field only
        if accumulator_manifest(bits) != shipped:
            raise RuntimeError(f"generated manifest differs from shipped int{bits}_accumulator")


def run_conformance(cases: list[tuple[int, int]], out: Path, workers: int) -> list[dict]:
    install_width_hook()
    check_hook_matches_shipped()
    out.mkdir(parents=True, exist_ok=True)

    def one(case):
        w, a = case
        try:
            return base.run_one(w, w, a, output_dir=out)
        except Exception as exc:  # recorded, never hidden
            return {"configuration": f"int{w}xint{w}-acc{a}", "weight_bits": w, "activation_bits": w,
                    "accumulator_bits": a, "status": "fail", "error": str(exc)[:500]}

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, cases))
    return results


def verified_cases(path: Path) -> set[tuple[int, int]]:
    data = json.loads(path.read_text())
    return {(r["weight_bits"], r["accumulator_bits"]) for r in data["results"] if r["status"] == "pass"}


# ---------------------------------------------------------------- pure logic
def same_period(a: float, b: float, tol: float = PERIOD_TOL_NS) -> bool:
    """Periods are equal when they differ by less than tol (never exact float equality)."""
    return abs(a - b) <= tol


def netlist_groups(points: list[dict]) -> dict:
    """Group the points of one configuration by netlist hash (a netlist is
    identified by its hash, never by its printed period)."""
    groups: dict = {}
    for p in points:
        groups.setdefault(p["netlist_sha256"], []).append(p)
    return groups


def meets(p: dict) -> bool:
    return bool(p["valid"]) and p["worst_slack_ns"] >= 0


def smallest_valid_period(points: list[dict]) -> dict | None:
    """Smallest achieved period over valid points, with the netlist that gives
    it. Points whose period is within tolerance of the minimum are ties; if
    they carry different hashes all hashes are listed (no silent choice)."""
    valid = [p for p in points if p["valid"]]
    if not valid:
        return None
    best = min(p["min_period_ns"] for p in valid)
    tied = [p for p in valid if same_period(p["min_period_ns"], best)]
    hashes = sorted({p["netlist_sha256"] for p in tied})
    targets = sorted((p["target_ns"] for p in tied), reverse=True)
    # a netlist "meets" the target at which it was produced if slack >= 0 there
    first = max(tied, key=lambda p: p["target_ns"])
    return {"period_ns": best, "netlist_sha256": hashes[0], "tied_hashes": hashes,
            "targets_ns": targets, "first_target_ns": first["target_ns"],
            "meets_first_target": meets(first),
            "any_tied_point_meets_its_target": any(meets(p) for p in tied)}


def tightest_met_target(points: list[dict]) -> int | None:
    ok = [p["target_ns"] for p in points if meets(p)]
    return min(ok) if ok else None


def common_met_targets(by_config: dict, configs: list, targets) -> list:
    """Targets at which every listed configuration meets timing validly."""
    return [t for t in targets
            if all(any(p["target_ns"] == t and meets(p) for p in by_config.get(c, [])) for c in configs)]


def ols(xs: list[float], ys: list[float]) -> dict:
    """Straight-line least squares with residuals and R^2 (pure Python)."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        raise ValueError("need at least two matching points")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("x values are all equal")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - sum(r * r for r in resid) / ss_tot if ss_tot else 1.0
    return {"slope": slope, "intercept": intercept, "r2": r2, "residuals": resid,
            "max_abs_residual": max(abs(r) for r in resid), "n": n}


def index_points(records: list[dict]) -> dict:
    by_config: dict = {}
    for r in records:
        by_config.setdefault((r["width_bits"], r["accumulator_bits"]), []).append(r)
    return by_config


def area_at(by_config: dict, config: tuple, target: int, key: str = "core_area"):
    for p in by_config.get(config, []):
        if p["target_ns"] == target and meets(p):
            return p[key]
    return None


# ------------------------------------------------------- quality-cost join
def required_signed_bits(max_value: int) -> int:
    """Smallest signed two's-complement width n whose largest value 2^(n-1)-1
    is >= max_value (max_value >= 0)."""
    if max_value < 0:
        raise ValueError("max_value must be non-negative")
    n = 1
    while (1 << (n - 1)) - 1 < max_value:
        n += 1
    return n


def covering_width(required: int | None, widths=ACC_WIDTHS) -> int | None:
    """Smallest synthesised width >= required, None if required is None or too large."""
    if required is None:
        return None
    ok = [w for w in widths if w >= required]
    return min(ok) if ok else None


def read_requirement(bounds: dict) -> dict:
    """Static accumulator requirement of one graph from its accumulator-bounds
    record: for every reduction, |dot sum| <= dot_bound and the stored bias is
    <= maximum_stored_bias_codes, so |result| <= dot_bound + bias. Checks the
    record's own remaining_headroom against that arithmetic."""
    worst, worst_dot, node, checked = 0, 0, None, 0
    accs = set()
    for r in bounds["reductions"]:
        dot, bias = int(r["dot_bound"]), int(r["maximum_stored_bias_codes"])
        accs.add(r["accumulator"])
        bits = int(r["accumulator"].split("_")[0][3:])
        if int(r["remaining_headroom"]) != (1 << (bits - 1)) - 1 - (dot + bias):
            raise ValueError(f"{r['node']}: remaining_headroom disagrees with dot_bound + bias")
        checked += 1
        if dot + bias > worst:
            worst, node = dot + bias, r["node"]
        worst_dot = max(worst_dot, dot)
    return {"max_dot_plus_bias": worst, "node": node, "reductions": checked,
            "required_bits": required_signed_bits(worst),
            "required_bits_dot_only": required_signed_bits(worst_dot),
            "recorded_accumulators": sorted(accs)}


EVIDENCE = (("resnet18", 4), ("resnet18", 5), ("resnet18", 6), ("resnet18", 8),
            ("mobilenet_v2", 4), ("mobilenet_v2", 5), ("mobilenet_v2", 6), ("mobilenet_v2", 8))
ANALYSES = {  # results/summaries/phase3-analysis-<id>.json, from docs/architecture/phase3-eight-integer-review.md
    ("mobilenet_v2", 4): "1a2daf8c43db", ("mobilenet_v2", 5): "a89403d906f5",
    ("mobilenet_v2", 6): "f7ab263b2174", ("mobilenet_v2", 8): "cf9353a8e28d",
    ("resnet18", 4): "3c550d8ff844", ("resnet18", 5): "22a98c5deeb8",
    ("resnet18", 6): "7f7b315039cf", ("resnet18", 8): "d240b899d3f2"}


def load_evidence() -> list[dict]:
    audit = json.loads((ROOT / "results/summaries/phase3-accumulator-audit.json").read_text())
    rows = []
    for model, width in EVIDENCE:
        rec = next(r for r in audit["records"] if r["model"] == model and r["format"] == f"int{width}")
        row = {"model": model, "format": f"int{width}", "width_bits": width}
        bpath = ROOT / rec["bounds"]["path"]
        analysis = json.loads((ROOT / f"results/summaries/phase3-analysis-{ANALYSES[(model, width)]}.json").read_text())
        row.update({"analysis": f"results/summaries/phase3-analysis-{ANALYSES[(model, width)]}.json",
                    "images": analysis["images"], "scope": analysis["scope"],
                    "label": analysis["classification"]["label"],
                    "top1": analysis["statistics"]["metrics"]["top1"]["candidate"],
                    "top5": analysis["statistics"]["metrics"]["top5"]["candidate"],
                    "fp32_top1": analysis["statistics"]["metrics"]["top1"]["fp32"],
                    "top1_delta_interval": analysis["statistics"]["metrics"]["top1"]["delta_interval"]})
        if (analysis["model"], analysis["format"], analysis["images"]) != (model, f"int{width}", 1000) \
                or analysis["configuration_sha256"] != rec["configuration"]["sha256"]:
            raise ValueError(f"analysis for {model} int{width} is not the 1,000-image run of the audited configuration")
        if not bpath.exists() or base.sha(bpath) != rec["bounds"]["sha256"]:
            row.update({"required_bits": None, "requirement_note": "bounds file missing or hash mismatch"})
        else:
            row.update(read_requirement(json.loads(bpath.read_text())))
            row["bounds_path"], row["bounds_sha256"] = rec["bounds"]["path"], rec["bounds"]["sha256"]
            row["bounds_status"] = json.loads(bpath.read_text())["status"]
        rows.append(row)
    return rows


# --------------------------------------------------------------- derivations
def derive(records: list[dict], targets=TARGETS_NS) -> dict:
    by = index_points(records)
    configs = sorted(by)
    common = common_met_targets(by, configs, targets)
    per_config = {}
    for c in configs:
        best = smallest_valid_period(by[c])
        per_config[f"int{c[0]}-acc{c[1]}"] = {
            "distinct_netlists": len(netlist_groups(by[c])),
            "smallest_valid": best, "tightest_met_target_ns": tightest_met_target(by[c])}
    fits, steps = {}, {}
    for t in common:
        for key in ("core_area", "total_area"):
            for w in WIDTHS:
                ys = [area_at(by, (w, a), t, key) for a in ACC_WIDTHS]
                if None in ys:
                    continue
                fit = ols([float(a) for a in ACC_WIDTHS], ys)
                fits.setdefault(str(t), {}).setdefault(key, {})[f"int{w}"] = {
                    **fit, "acc_bits": list(ACC_WIDTHS), "areas": ys}
            for a in ACC_WIDTHS:
                ys = [area_at(by, (w, a), t, key) for w in WIDTHS]
                if None in ys:
                    continue
                fit = ols([float(w) for w in WIDTHS], ys)
                steps.setdefault(str(t), {}).setdefault(key, {})[f"acc{a}"] = {
                    **fit, "mult_bits": list(WIDTHS), "areas": ys,
                    "pairwise_steps": [{"from": w0, "to": w1, "per_bit": (y1 - y0) / (w1 - w0)}
                                       for (w0, y0), (w1, y1) in zip(zip(WIDTHS, ys), zip(WIDTHS[1:], ys[1:]))]}
    ratios = {}
    for t in common:
        core = fits[str(t)]["core_area"]
        step32 = steps[str(t)]["core_area"]["acc32"]["slope"]
        step64 = steps[str(t)]["core_area"]["acc64"]["slope"]
        acc_slopes = {k: v["slope"] for k, v in core.items()}
        a4, a8 = area_at(by, (4, 32), t), area_at(by, (8, 32), t)
        ratios[str(t)] = {
            "acc_bit_area_by_mult_width": acc_slopes,
            "mult_step_area_at_acc32": step32, "mult_step_area_at_acc64": step64,
            "acc_bit_over_mult_step_at_acc32": {k: v / step32 for k, v in acc_slopes.items()},
            "int8_minus_int4_at_acc32": a8 - a4,
            "int8_minus_int4_in_acc_bits": {k: (a8 - a4) / v for k, v in acc_slopes.items()},
            "int4_acc32_over_int8_acc32_core": a4 / a8,
            "int8_acc64_over_acc32_core": area_at(by, (8, 64), t) / a8}
    return {"common_met_targets_ns": common, "per_config": per_config, "acc_fits": fits,
            "mult_steps": steps, "ratios": ratios}


def stage2_reproduction(records: list[dict]) -> dict:
    """Do the eight stage-2 configurations reproduce stage 2 (same hash, same area)?"""
    path = ROOT / "results/tables/ics55-integer-isoclock-v1.csv"
    old = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["variant"] == "abc":
                old[(int(row["width_bits"]), int(row["accumulator_bits"]), int(row["target_ns"]))] = row
    same, differ, compared = 0, [], 0
    for r in records:
        key = (r["width_bits"], r["accumulator_bits"], r["target_ns"])
        if key in old:
            compared += 1
            if old[key]["netlist_sha256"] == r["netlist_sha256"] and float(old[key]["core_area"]) == r["core_area"]:
                same += 1
            else:
                differ.append(key)
    return {"compared": compared, "identical_netlist_and_area": same, "differing": differ}


COLUMNS = m.COLUMNS + ("distinct_netlist_index",)


def write_table(records: list[dict], path: Path) -> None:
    by = index_points(records)
    index = {}
    for c, pts in by.items():
        first = {}
        for p in sorted(pts, key=lambda p: -p["target_ns"]):
            first.setdefault(p["netlist_sha256"], len(first) + 1)
        index[c] = first
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in sorted(records, key=lambda r: (r["width_bits"], r["accumulator_bits"], -r["target_ns"])):
            row = m.table_rows([r])[0]
            row["distinct_netlist_index"] = index[(r["width_bits"], r["accumulator_bits"])][r["netlist_sha256"]]
            writer.writerow({k: row.get(k) for k in COLUMNS})


JOIN_COLUMNS = ("model", "format", "images", "top1", "top1_delta_interval_95", "top5", "fp32_top1", "screen_label",
                "accumulator_in_accuracy_run", "required_acc_bits_static_bound", "required_acc_bits_dot_only",
                "binding_node", "covering_synthesised_acc_bits", "mac_core_area_at_8ns",
                "mac_core_area_at_12ns", "smallest_valid_period_ns", "tightest_target_met_ns",
                "covering_acc_bits_if_dot_only", "mac_core_area_at_8ns_if_dot_only", "status")


def join_rows(evidence: list[dict], records: list[dict]) -> list[dict]:
    by = index_points(records)
    out = []
    for e in evidence:
        req = e.get("required_bits")
        cover = covering_width(req)
        row = {"model": e["model"], "format": e["format"], "images": e["images"], "top1": e["top1"],
               "top1_delta_interval_95": "[%.3f, %.3f]" % tuple(e["top1_delta_interval"]),
               "top5": e["top5"], "fp32_top1": e["fp32_top1"], "screen_label": e["label"],
               "accumulator_in_accuracy_run": "/".join(e.get("recorded_accumulators", [])) or "not established",
               "required_acc_bits_static_bound": req if req is not None else "not established",
               "required_acc_bits_dot_only": e.get("required_bits_dot_only", "not established"),
               "binding_node": e.get("node", "not established"),
               "covering_synthesised_acc_bits": cover if cover is not None else "not established"}
        dot_cover = covering_width(e.get("required_bits_dot_only"))
        row["covering_acc_bits_if_dot_only"] = dot_cover if dot_cover is not None else "not established"
        row["mac_core_area_at_8ns_if_dot_only"] = (area_at(by, (e["width_bits"], dot_cover), 8)
                                                   if dot_cover is not None else "not established")
        if cover is None:
            row["tightest_target_met_ns"] = "not established"
            row.update(mac_core_area_at_8ns="not established", mac_core_area_at_12ns="not established",
                       smallest_valid_period_ns="not established", status="not established")
        else:
            c = (e["width_bits"], cover)
            best = smallest_valid_period(by[c])
            row.update(mac_core_area_at_8ns=area_at(by, c, 8), mac_core_area_at_12ns=area_at(by, c, 12),
                       smallest_valid_period_ns=round(best["period_ns"], 6),
                       tightest_target_met_ns=tightest_met_target(by[c]), status="preliminary")
        out.append(row)
    return out


def plot(records: list[dict], target: int, png: Path, pdf: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # figure is optional
        return f"matplotlib unavailable: {exc}"
    by = index_points(records)
    colours = {4: "#0072B2", 5: "#E69F00", 6: "#009E73", 8: "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for w in WIDTHS:
        xs = [a for a in ACC_WIDTHS if area_at(by, (w, a), target) is not None]
        axes[0].plot(xs, [area_at(by, (w, a), target) for a in xs], marker="o", color=colours[w], label=f"INT{w}")
        best = [smallest_valid_period(by[(w, a)])["period_ns"] for a in ACC_WIDTHS]
        axes[1].plot(ACC_WIDTHS, best, marker="o", color=colours[w], label=f"INT{w}")
    axes[0].set_title(f"MAC core area at the {target} ns clock target", fontsize=10)
    axes[0].set_ylabel("core area (library units, mapped, no wires)")
    axes[1].set_title("smallest valid achieved period", fontsize=10)
    axes[1].set_ylabel("period (ns)")
    for ax in axes:
        ax.set_xlabel("accumulator width (bits)")
        ax.set_xticks(ACC_WIDTHS)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
        ax.legend(title="multiplier", fontsize=8)
    fig.suptitle("ICsprout55 integer MAC, Yosys/ABC mapping, typical corner", fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=160)
    fig.savefig(pdf)
    return None


def run_determinism(points, out: Path) -> dict:
    os.environ["TMPDIR"] = str(out)
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    for w, a, t in points:
        original = OUT / "points" / m.case_name(w, a) / f"t{t}ns"
        rerun_root = out / "points"
        m.run_point(w, a, t, rerun_root, resizer=True)
        again = rerun_root / m.case_name(w, a) / f"t{t}ns"
        result[f"int{w}-acc{a}-t{t}"] = {
            name: (original / name).read_bytes() == (again / name).read_bytes()
            for name in ("netlist.v", "yosys-stat.json", "opensta.txt", "drv.txt", "abc.constr")}
    return result


# --------------------------------------------------------------------- synth
def run_synth(cases: list[tuple[int, int]], targets, out: Path, workers: int) -> list[dict]:
    tmp = out / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    jobs = [(w, a, t) for (w, a) in cases for t in targets]
    for w, a, t in jobs:
        d = out / m.case_name(w, a) / f"t{t}ns"
        if d.exists():
            raise SystemExit(f"refusing to overwrite {d}")
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(m.run_point, w, a, t, out, resizer=True) for w, a, t in jobs]
        records = []
        for job, fut in zip(jobs, futures):
            records.append(fut.result())
            print("done", job, "valid" if records[-1]["valid"] else "INVALID", flush=True)
    return records


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("conformance")
    c.add_argument("--out-dir", type=Path, default=OUT / "conformance")
    c.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("synth")
    s.add_argument("--out-dir", type=Path, default=OUT / "points")
    s.add_argument("--conformance", type=Path, default=OUT / "conformance-summary.json")
    s.add_argument("--records", type=Path, default=OUT / "records.json")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--cases", nargs="+", default=None, help="W:ACC pairs (default: all verified)")
    s.add_argument("--targets", type=int, nargs="+", default=list(TARGETS_NS))
    d = sub.add_parser("determinism")
    d.add_argument("--out-dir", type=Path, default=OUT / "determinism-rerun")
    d.add_argument("--result", type=Path, default=OUT / "determinism-result.json")
    d.add_argument("--points", nargs="+", default=["6:40:5", "8:64:8"], help="W:ACC:TARGET")
    u = sub.add_parser("summarize")
    u.add_argument("--records", type=Path, default=OUT / "records.json")
    u.add_argument("--target", type=int, default=8, help="common clock target for the figure")
    args = ap.parse_args(argv)
    if args.cmd == "determinism":
        if args.out_dir.exists() or args.result.exists():
            raise SystemExit("refusing to overwrite an existing determinism rerun")
        pts = [tuple(int(x) for x in q.split(":")) for q in args.points]
        res = run_determinism(pts, args.out_dir)
        args.result.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
        print(json.dumps(res, indent=2))
        return
    if args.cmd == "summarize":
        for path in (SUMMARY, TABLE, JOIN_TABLE):
            if path.exists():
                raise SystemExit(f"refusing to overwrite {path}")
        data = json.loads(args.records.read_text())
        records = data["records"]
        conf = json.loads((OUT / "conformance-summary.json").read_text())
        function = {}
        for name in ("gatesim-result", "cec-result", "determinism-result"):
            f = OUT / ("function" if name != "determinism-result" else ".") / f"{name}.json"
            function[name] = json.loads(f.read_text()) if f.exists() else None
        derived = derive(records)
        evidence = load_evidence()
        rows = join_rows(evidence, records)
        write_table(records, TABLE)
        with JOIN_TABLE.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=JOIN_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        figure_note = plot(records, args.target, ROOT / "results/figures/ics55-integer-accwidth-v1.png",
                           ROOT / "results/figures/ics55-integer-accwidth-v1.pdf")
        summary = {
            "schema_version": "1.0.0",
            "scope": "accumulator-width sweep of the existing integer MAC at equal clock; saturating accumulator "
                     "as in the RTL; no new RTL; preliminary quality-cost join",
            "tool_versions": {"yosys": m.version("yosys"),
                              "openroad_wrapper": subprocess.run([str(m.WRAPPER), "--version"], capture_output=True,
                                                                 text=True, check=False).stdout.strip(),
                              "iverilog": conf["simulator_versions"]["iverilog"]},
            "constraints": {"clock_uncertainty_ns": m.UNCERTAINTY_NS, "input_delay_ns": m.IO_DELAY_NS,
                            "output_delay_ns": m.IO_DELAY_NS, "output_load_pf": m.OUTPUT_LOAD_PF,
                            "abc_driving_cell": m.DRIVING_CELL, "abc_output_load_ff": m.ABC_LOAD_FF,
                            "abc_delay_ps_rule": f"target_ns*1000 - {m.ABC_OVERHEAD_PS}"},
            "validity_rule": "worst setup path has no pin violating max_transition, max_capacitance or max_fanout",
            "period_comparison_tolerance_ns": PERIOD_TOL_NS,
            "hashes": {"liberty": m.digest(m.LIBERTY), "tech_lef": m.digest(m.TECH_LEF),
                       "cells_lef": m.digest(m.CELLS_LEF), "wrapper": m.digest(m.WRAPPER),
                       "stage2_matrix_script": m.digest(Path(m.__file__)),
                       "sweep_script": m.digest(Path(__file__)),
                       "rtl": {pp: m.digest(ROOT / pp) for pp in m.RTL}},
            "conformance": {"verified": sorted(f"int{r['weight_bits']}-acc{r['accumulator_bits']}"
                                               for r in conf["results"] if r["status"] == "pass"),
                            "not_verified": [r["configuration"] for r in conf["results"] if r["status"] != "pass"],
                            "summary": "artifacts/ppa/ics55-integer-accwidth-v1/conformance-summary.json"},
            "targets_ns": list(TARGETS_NS), "records": records, "derived": derived,
            "stage2_reproduction": stage2_reproduction(records),
            "function": function, "evidence": evidence, "join": rows,
            "join_caveats": ["preliminary", "strict-A recipe (simulator recipes being repaired elsewhere)",
                             "development evidence on the 1k screen", "MAC only: no scaling, requantisation, bias or "
                             "memory cost", "RTL accumulator saturates on overflow", "mapped netlists, no placement or "
                             "wire parasitics, typical corner, no power"],
            "figure": figure_note or "results/figures/ics55-integer-accwidth-v1.png/.pdf",
        }
        SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print("wrote", SUMMARY)
        return
    if args.cmd == "conformance":
        cases = [(w, a) for w in WIDTHS for a in ACC_WIDTHS]
        results = run_conformance(cases, args.out_dir, args.workers)
        summary = {"schema_version": "1.0.0",
                   "scope": "existing integer-mac conformance generator and oracle, accumulator width supplied "
                            "by a driver hook (same manifest as int32_accumulator with another width)",
                   "driver": str(Path(__file__).relative_to(ROOT)),
                   "driver_sha256": base.sha(Path(__file__)),
                   "generator_sha256": base.sha(Path(base.__file__)),
                   "rtl_sha256": base.sha(base.RTL), "harness_sha256": base.sha(base.HARNESS),
                   "oracle": "public.inference.reference.arithmetic.model_c",
                   "simulator_versions": {"iverilog": base.tool_version(["iverilog", "-V"]),
                                          "vvp": base.tool_version(["vvp", "-V"])},
                   "results": results}
        dest = args.out_dir.parent / "conformance-summary.json"
        if dest.exists():
            raise SystemExit(f"refusing to overwrite {dest}")
        dest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        for r in results:
            print(r["configuration"], r["status"], r.get("error", ""))
        print(dest)
        return
    if args.cmd == "synth":
        ok = verified_cases(args.conformance)
        cases = ([tuple(int(x) for x in p.split(":")) for p in args.cases] if args.cases
                 else [(w, a) for w in WIDTHS for a in ACC_WIDTHS])
        skipped = [c_ for c_ in cases if c_ not in ok]
        for c_ in skipped:
            print("SKIPPED (not verified):", c_)
        cases = [c_ for c_ in cases if c_ in ok]
        if args.records.exists():
            raise SystemExit(f"refusing to overwrite {args.records}")
        records = run_synth(cases, args.targets, args.out_dir, args.workers)
        args.records.write_text(json.dumps({"skipped_unverified": skipped, "records": records},
                                           indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
