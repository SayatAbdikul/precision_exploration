"""Timing-driven iso-clock synthesis and OpenSTA matrix for the integer MAC.

For each (INT width, accumulator width) configuration and each clock target:
Yosys ``synth`` + ``dfflibmap`` + ``abc -liberty -constr -D`` (timing-driven
technology mapping), then OpenSTA (embedded in OpenROAD, run through
tools/hardware/openroad_docker.sh) with the same constraints as stage 1.
Optionally the OpenROAD resizer is tried on points whose critical path
violates the library max-transition / max-capacitance limits.

No new RTL is used; the existing flow's helpers are imported.
Nothing outside --out-dir/--summary/--table is written.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from public.pdk_flow.ics55.validate_integer import (  # noqa: E402  (existing flow, reused)
    RTL, TOP, digest, version,
)

CASES = ((4, 32), (5, 32), (6, 32), (8, 32), (8, 64))
# Supplementary: also listed as passing in integer-mac-conformance-v1.json; they
# give the accumulator effect at other operand widths. Not part of the headline.
SUPPLEMENTARY = ((4, 64), (5, 64), (6, 64))
TARGETS_NS = (12, 10, 8, 6, 5, 4, 3, 2)
# Constraints identical to public/pdk_flow/common/opensta_matrix.tcl (stage 1).
UNCERTAINTY_NS = 0.10
IO_DELAY_NS = 0.20
OUTPUT_LOAD_PF = 0.01
# ABC constraint file: a driving cell and an output load from the library.
DRIVING_CELL = "BUFX2H7R"
# ABC's set_load is in femtofarads (ABC logs "Cout = <x> ff"), NOT in the
# liberty unit of 1 pF. 10 fF = 0.01 pF, the same load STA puts on every output.
ABC_LOAD_FF = 10
# ABC times only the combinational logic between sequential cells; the clock
# period also pays clk-to-Q (about 0.13 ns), setup (about 0.04 ns), the 0.10 ns
# uncertainty and, on output paths, the 0.20 ns output delay. Worst of those
# path classes (reg->output: 0.13 + 0.20 + 0.10) rounded up is 450 ps.
ABC_OVERHEAD_PS = 450
CONFORMANCE = ROOT / "results/summaries/integer-mac-conformance-v1.json"
WRAPPER = ROOT / "tools/hardware/openroad_docker.sh"
PDK = Path("/home/maveric/nursultan/texer.ai/ecc/chipcompiler/thirdparty/icsprout55-pdk")
CELL_DIR = PDK / "IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR"
LIBERTY = CELL_DIR / "liberty/ics55_LLSC_H7CR_typ_tt_1p2_25_nldm.lib"
TECH_LEF = PDK / "prtech/techLEF/N551P6M.lef"
CELLS_LEF = CELL_DIR / "lef/ics55_LLSC_H7CR.lef"


# ----------------------------------------------------------------- pure parts
def abc_delay_ps(target_ns: float) -> int:
    """ABC ``-D`` value in picoseconds for a clock target in nanoseconds."""
    value = round(target_ns * 1000) - ABC_OVERHEAD_PS
    if value <= 0:
        raise ValueError(f"clock target {target_ns} ns leaves no combinational budget")
    return value


def min_period_ns(period_ns: float, worst_slack_ns: float) -> float:
    """Achieved period: the clock period minus the worst setup slack."""
    return period_ns - worst_slack_ns


_PATH_LINE = re.compile(
    r"^\s*(?:(\d+\.\d+)\s+)?(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+[\^v]\s+(\S+)\s+\((\S+)\)\s*$")
_VIOL_LINE = re.compile(r"^(\S+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\(VIOLATED\)\s*$")
_SECTIONS = {"max slew": "max_slew", "max capacitance": "max_capacitance", "max fanout": "max_fanout"}


def parse_sta_report(text: str) -> dict:
    """Parse the report written by the Tcl in :func:`sta_tcl`.

    Requires exactly one worst-slack and one tns line and a path with an
    arrival time; raises ValueError otherwise so a broken run cannot pass.
    """
    slack = re.findall(r"^worst slack max\s+(-?\d+(?:\.\d+)?)\s*$", text, re.M)
    tns = re.findall(r"^tns max\s+(-?\d+(?:\.\d+)?)\s*$", text, re.M)
    if len(slack) != 1 or len(tns) != 1:
        raise ValueError("report needs exactly one 'worst slack max' and one 'tns max' line")
    arrival = re.findall(r"^\s*(-?\d+\.\d+)\s+data arrival time\s*$", text, re.M)
    setup = re.findall(r"^\s*(-?\d+\.\d+)\s+(?:-?\d+\.\d+\s+)?library setup time\s*$", text, re.M)
    if not arrival:
        raise ValueError("report has no timing path")
    pins = []
    for line in text.splitlines():
        match = _PATH_LINE.match(line)
        if match:
            cap, slew, _delay, _time, name, cell = match.groups()
            pins.append({"pin": name, "cell": cell, "slew_ns": float(slew),
                         "cap_pf": float(cap) if cap is not None else None})
    return {"worst_slack_ns": float(slack[0]), "tns_ns": float(tns[0]),
            "arrival_ns": float(arrival[0]), "setup_ns": float(setup[0]) if setup else None,
            "path_pins": pins, "path_cells": sum(1 for p in pins if p["cap_pf"] is not None)}


def parse_violations(text: str) -> dict:
    """Parse ``report_check_types -violators`` output into lists per check."""
    result = {key: [] for key in _SECTIONS.values()}
    section = None
    for line in text.splitlines():
        stripped = line.strip().lower()
        if stripped in _SECTIONS:
            section = _SECTIONS[stripped]
            continue
        match = _VIOL_LINE.match(line.strip())
        if match and section:
            pin, limit, value, slack = match.groups()
            result[section].append({"pin": pin, "limit": float(limit), "value": float(value),
                                    "slack": float(slack)})
    return result


def critical_path_violations(path_pins: list[dict], violations: dict) -> dict:
    """Violating pins that lie on the reported critical path, per check."""
    on_path = {p["pin"] for p in path_pins}
    return {key: sorted({v["pin"] for v in items if v["pin"] in on_path})
            for key, items in violations.items()}


def point_is_valid(path_pins: list[dict], violations: dict) -> bool:
    """Validity rule: the critical path touches no pin violating max slew,
    max capacitance or max fanout, so its delay lies in the characterised range.
    A report with no path pins can never be valid."""
    if not path_pins:
        return False
    return not any(critical_path_violations(path_pins, violations).values())


def point_meets(point: dict) -> bool:
    return bool(point["valid"]) and point["worst_slack_ns"] >= 0


def common_targets(points: dict, configs: list, targets: list) -> list:
    """Clock targets at which every listed configuration meets timing validly."""
    return [t for t in targets if all((c, t) in points and point_meets(points[(c, t)]) for c in configs)]


def ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return numerator / denominator


def pair_ratio_rows(points: dict, targets: list, numerator: tuple, denominator: tuple) -> list:
    """Area ratios numerator/denominator at the targets where both meet timing validly."""
    rows = []
    for t in common_targets(points, [numerator, denominator], targets):
        n, d = points[(numerator, t)], points[(denominator, t)]
        rows.append({"target_ns": t,
                     "core_ratio": ratio(n["core_area"], d["core_area"]),
                     "total_ratio": ratio(n["total_area"], d["total_area"])})
    return rows


def iso_clock_ratios(points: dict, targets: list, main_configs: list = None,
                     extra_accumulator_pairs: tuple = ()) -> dict:
    """The two hardware-hypothesis ratios at iso-clock.

    multiplier_effect: INT4/ACC32 over INT8/ACC32.
    accumulator_effect: INT8/ACC64 over INT8/ACC32 (an increase is ratio - 1).
    Each is given at every target where its two configurations both meet
    timing validly; ``all_main_common_targets`` are the targets where all of
    ``main_configs`` do, and the headline is taken there.
    extra_accumulator_pairs: further (ACC64 config, ACC32 config) pairs.
    """
    main_configs = list(main_configs) if main_configs else [(4, 32), (5, 32), (6, 32), (8, 32), (8, 64)]
    common = common_targets(points, main_configs, targets)
    out = {"all_main_common_targets": common,
           "multiplier_effect_int4_over_int8_acc32": pair_ratio_rows(points, targets, (4, 32), (8, 32)),
           "accumulator_effect_int8_acc64_over_acc32": pair_ratio_rows(points, targets, (8, 64), (8, 32))}
    for wide, narrow in extra_accumulator_pairs:
        out[f"accumulator_effect_int{wide[0]}_acc{wide[1]}_over_acc{narrow[1]}"] = \
            pair_ratio_rows(points, targets, wide, narrow)
    return out


# --------------------------------------------------------------- flow pieces
def case_name(width: int, acc: int) -> str:
    return f"int{width}-acc{acc}"


def require_conformance(width: int, acc: int) -> None:
    """Refuse to synthesise a configuration the conformance tool did not pass."""
    data = json.loads(CONFORMANCE.read_text())
    name = f"int{width}xint{width}-acc{acc}"
    status = {r["configuration"]: r["status"] for r in data["results"]}.get(name)
    if status != "pass":
        raise RuntimeError(f"{name} has no passing entry in {CONFORMANCE.name}; refusing to synthesise")


def run_checked(command: list[str], *, cwd: Path, log: Path, timeout: int) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False, timeout=timeout)
    log.write_text(f"command: {' '.join(command)}\nreturncode: {result.returncode}\n\nstdout:\n"
                   f"{result.stdout}\n\nstderr:\n{result.stderr}")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")
    return result


def synthesize(width: int, acc: int, target_ns: float, out: Path, yosys: str = "yosys") -> dict:
    """Timing-driven mapping: abc -liberty -constr -D <ps>. RTL paths are the
    fixed repository paths, as in the stage-1 flow."""
    out.mkdir(parents=True, exist_ok=True)
    delay = abc_delay_ps(target_ns)
    constr = out / "abc.constr"
    constr.write_text(f"set_driving_cell {DRIVING_CELL}\nset_load {ABC_LOAD_FF}\n")
    lines = [f"read_verilog -sv {ROOT / path}" for path in RTL]
    lines += [f"chparam -set {name} {value} {TOP}" for name, value in
              sorted({"W_BITS": width, "A_BITS": width, "ACC_BITS": acc}.items())]
    lines += [f"hierarchy -check -top {TOP}", "proc", "opt", f"synth -top {TOP}",
              f"dfflibmap -liberty {LIBERTY}",
              f"abc -liberty {LIBERTY} -constr {constr} -D {delay}",
              "opt_clean",
              f"write_verilog -noattr {out / 'netlist.v'}",
              f"tee -o {out / 'yosys-stat.json'} stat -json -liberty {LIBERTY}"]
    (out / "synthesis.ys").write_text("\n".join(lines) + "\n")
    run_checked([yosys, "-s", str(out / "synthesis.ys")], cwd=out, log=out / "yosys.log", timeout=300)
    log = (out / "yosys.log").read_text()
    if f"-D {delay}" not in log:
        raise RuntimeError("ABC did not receive the delay target")
    if f"Setting output load to be {ABC_LOAD_FF:.6f}" not in log:
        raise RuntimeError("ABC did not receive the output load")
    return {"abc_delay_ps": delay, "netlist": out / "netlist.v", "stat": out / "yosys-stat.json"}


def read_stat(stat_path: Path) -> dict:
    modules = json.loads(stat_path.read_text())["modules"]
    cores = [n for n in modules if "integer_mac" in n and "harness" not in n]
    top = modules[f"\\{TOP}"] if f"\\{TOP}" in modules else modules[TOP]
    design = json.loads(stat_path.read_text())["design"]
    if len(cores) != 1:
        raise ValueError(f"expected one core module, got {cores}")
    core = modules[cores[0]]
    return {"core_area": float(core["area"]), "total_area": float(design["area"]),
            "core_cells": int(core["num_cells"]), "total_cells": int(design["num_cells"]),
            "wrapper_area": float(top["area"])}


def sta_tcl(netlist: Path, period_ns: float, report: Path, drv: Path) -> str:
    """Same constraints as public/pdk_flow/common/opensta_matrix.tcl, plus
    max-slew/capacitance/fanout violator listing. The remove_from_collection
    shim is the one stage 1's wrapper defined for OpenROAD's Tcl."""
    return f"""read_lef {TECH_LEF}
read_lef {CELLS_LEF}
if {{[info commands remove_from_collection] eq ""}} {{
  proc remove_from_collection {{a b}} {{
    set r {{}}
    foreach x $a {{ if {{[lsearch -exact $b $x] < 0}} {{ lappend r $x }} }}
    return $r
  }}
}}
read_liberty {LIBERTY}
read_verilog {netlist}
link_design {TOP}
create_clock -name clk -period {period_ns} [get_ports clk]
set_clock_uncertainty {UNCERTAINTY_NS} [get_clocks clk]
set data_inputs [remove_from_collection [all_inputs] [get_ports clk]]
set_input_delay {IO_DELAY_NS} -clock clk $data_inputs
set_output_delay {IO_DELAY_NS} -clock clk [all_outputs]
set_load {OUTPUT_LOAD_PF} [all_outputs]
check_setup -verbose >{report}
report_checks -path_delay max -group_path_count 1 -fields {{slew cap input_pins}} -digits 6 >>{report}
report_worst_slack -max -digits 6 >>{report}
report_tns -max -digits 6 >>{report}
report_check_types -max_slew -max_capacitance -max_fanout -violators >{drv}
"""


def resizer_tcl(netlist: Path, period_ns: float, resized: Path) -> str:
    """Resizer on the unplaced netlist. No technology RC exists for ICS55 in
    this tree, so wire RC is set to zero, matching the zero-parasitic STA."""
    head = sta_tcl(netlist, period_ns, Path("/dev/null"), Path("/dev/null"))
    head = head.split("check_setup")[0]
    return head + f"""set_wire_rc -resistance 0 -capacitance 0
puts "RESIZER repair_design"
repair_design
puts "RESIZER repair_timing"
if {{[catch {{repair_timing -setup}} message]}} {{
  puts "RESIZER repair_timing failed: $message"
}}
write_verilog {resized}
"""


def run_openroad(tcl: Path, log: Path, timeout: int = 900) -> None:
    run_checked([str(WRAPPER), str(tcl)], cwd=tcl.parent, log=log, timeout=timeout)


def run_sta(netlist: Path, period_ns: float, out: Path, tag: str = "") -> dict:
    report, drv = out / f"opensta{tag}.txt", out / f"drv{tag}.txt"
    for stale in (report, drv):
        stale.unlink(missing_ok=True)
    tcl = out / f"sta{tag}.tcl"
    tcl.write_text(sta_tcl(netlist, period_ns, report, drv))
    run_openroad(tcl, out / f"sta{tag}.log")
    if not report.exists() or not drv.exists():
        raise RuntimeError("STA produced no report")
    parsed = parse_sta_report(report.read_text())
    violations = parse_violations(drv.read_text())
    on_path = critical_path_violations(parsed["path_pins"], violations)
    slews = [p["slew_ns"] for p in parsed["path_pins"]]
    return {"worst_slack_ns": parsed["worst_slack_ns"], "tns_ns": parsed["tns_ns"],
            "min_period_ns": min_period_ns(period_ns, parsed["worst_slack_ns"]),
            "path_cells": parsed["path_cells"], "path_worst_slew_ns": max(slews),
            "violations": {k: len(v) for k, v in violations.items()},
            "violations_on_critical_path": {k: len(v) for k, v in on_path.items()},
            "valid": point_is_valid(parsed["path_pins"], violations),
            "report_sha256": digest(report), "drv_sha256": digest(drv)}


def run_point(width: int, acc: int, target: int, root: Path, *, resizer: bool = True) -> dict:
    out = root.resolve() / case_name(width, acc) / f"t{target}ns"
    syn = synthesize(width, acc, target, out)
    record = {"width_bits": width, "accumulator_bits": acc, "target_ns": target,
              "abc_delay_ps": syn["abc_delay_ps"], "netlist_sha256": digest(syn["netlist"]),
              "stat_sha256": digest(syn["stat"]), **read_stat(syn["stat"])}
    sta = run_sta(syn["netlist"], target, out)
    record.update(variant="abc", **sta)
    record["resizer"] = None
    if resizer and not sta["valid"]:
        record["resizer"] = run_resizer_variant(width, acc, target, syn["netlist"], out)
    return record


def run_resizer_variant(width: int, acc: int, target: int, netlist: Path, out: Path) -> dict:
    rdir = out / "resizer"
    rdir.mkdir(exist_ok=True)
    resized = rdir / "netlist.v"
    resized.unlink(missing_ok=True)
    tcl = rdir / "resize.tcl"
    tcl.write_text(resizer_tcl(netlist, target, resized))
    try:
        run_openroad(tcl, rdir / "resize.log")
    except Exception as exc:  # recorded, never hidden
        return {"status": "failed", "reason": str(exc)[:300]}
    if not resized.exists():
        return {"status": "failed", "reason": "no netlist written"}
    stat = rdir / "yosys-stat.json"
    ys = rdir / "stat.ys"
    ys.write_text(f"read_liberty -lib {LIBERTY}\nread_verilog {resized}\nhierarchy -top {TOP}\n"
                  f"tee -o {stat} stat -json -liberty {LIBERTY}\n")
    run_checked(["yosys", "-s", str(ys)], cwd=rdir, log=rdir / "stat.log", timeout=120)
    design = json.loads(stat.read_text())["design"]
    sta = run_sta(resized, target, rdir)
    log = (rdir / "resize.log").read_text()
    failed = re.search(r"RESIZER repair_timing failed: (.*)", log)
    return {"status": "completed",
            "variant": "abc+repair_design" if failed else "abc+resizer",
            "repair_timing": f"failed: {failed.group(1)[:200]}" if failed else "ran", "total_area": float(design["area"]),
            "total_cells": int(design["num_cells"]), "netlist_sha256": digest(resized), **sta}


# ----------------------------------------------------------------- reporting
COLUMNS = ("width_bits", "accumulator_bits", "target_ns", "variant", "abc_delay_ps", "core_area",
           "total_area", "core_cells", "total_cells", "worst_slack_ns", "min_period_ns", "valid",
           "meets_timing", "max_slew_violations", "max_cap_violations", "path_worst_slew_ns",
           "path_cells", "netlist_sha256")


def table_rows(records: list[dict]) -> list[dict]:
    rows = []
    for r in records:
        rows.append({**{k: r.get(k) for k in COLUMNS}, "variant": "abc", "meets_timing": point_meets(r),
                     "max_slew_violations": r["violations"]["max_slew"],
                     "max_cap_violations": r["violations"]["max_capacitance"]})
        z = r.get("resizer")
        if z and z.get("status") == "completed":
            rows.append({**{k: z.get(k) for k in COLUMNS}, "width_bits": r["width_bits"],
                         "accumulator_bits": r["accumulator_bits"], "target_ns": r["target_ns"],
                         "variant": z["variant"], "abc_delay_ps": r["abc_delay_ps"], "core_area": None,
                         "meets_timing": point_meets(z), "max_slew_violations": z["violations"]["max_slew"],
                         "max_cap_violations": z["violations"]["max_capacitance"]})
    return rows


def pareto_front(points: list[dict]) -> list[dict]:
    """Points not dominated on (achieved period, total area): no other point is
    at least as good on both and strictly better on one. ABC is heuristic, so
    area is not monotone in the target; dominated points are not on the curve."""
    keep = []
    for p in points:
        dominated = any(q is not p and q["min_period_ns"] <= p["min_period_ns"]
                        and q["total_area"] <= p["total_area"]
                        and (q["min_period_ns"], q["total_area"]) != (p["min_period_ns"], p["total_area"])
                        for q in points)
        if not dominated:
            keep.append(p)
    return keep


def curves(records: list[dict]) -> dict:
    out = {}
    for (w, a) in sorted({(r["width_bits"], r["accumulator_bits"]) for r in records}):
        pts = sorted((r for r in records if (r["width_bits"], r["accumulator_bits"]) == (w, a)),
                     key=lambda r: -r["target_ns"])
        valid = [r for r in pts if r["valid"]]
        best = min(valid, key=lambda r: r["min_period_ns"]) if valid else None
        out[case_name(w, a)] = {
            "curve": [{"target_ns": r["target_ns"], "achieved_period_ns": r["min_period_ns"],
                       "total_area": r["total_area"], "core_area": r["core_area"], "valid": r["valid"],
                       "meets": point_meets(r)} for r in pts],
            "pareto_front": [{"target_ns": r["target_ns"], "achieved_period_ns": r["min_period_ns"],
                              "total_area": r["total_area"]} for r in
                             sorted(pareto_front(valid), key=lambda r: r["min_period_ns"])],
            "smallest_valid_achieved_period_ns": best["min_period_ns"] if best else None,
            "smallest_valid_at_target_ns": best["target_ns"] if best else None,
            "smallest_valid_point_meets_its_target": point_meets(best) if best else None}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/ppa/ics55-integer-isoclock-v1")
    ap.add_argument("--summary", type=Path, default=ROOT / "results/summaries/ics55-integer-isoclock-v1.json")
    ap.add_argument("--table", type=Path, default=ROOT / "results/tables/ics55-integer-isoclock-v1.csv")
    ap.add_argument("--targets", type=int, nargs="+", default=list(TARGETS_NS))
    ap.add_argument("--cases", nargs="+", default=[f"{w}:{a}" for w, a in CASES + SUPPLEMENTARY],
                    help="W:ACC pairs")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-resizer", action="store_true")
    args = ap.parse_args(argv)
    cases = [tuple(int(x) for x in c.split(":")) for c in args.cases]
    for w, a in cases:
        require_conformance(w, a)
    for p in (args.summary, args.table):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    root = args.out_dir.resolve()
    existing = sorted(str(root / case_name(w, a) / f"t{t}ns") for w, a in cases for t in args.targets
                      if (root / case_name(w, a) / f"t{t}ns").exists())
    if existing:
        raise SystemExit(f"refusing to overwrite {len(existing)} existing point directories, e.g. {existing[0]}")
    tmp = root / "tmp"          # keep Yosys/ABC scratch files inside the repository
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    jobs = [(w, a, t) for (w, a) in cases for t in args.targets]
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_point, w, a, t, root, resizer=not args.no_resizer) for w, a, t in jobs]
        records = []
        for job, fut in zip(jobs, futures):
            records.append(fut.result())
            print("done", job, "valid" if records[-1]["valid"] else "INVALID", flush=True)
    points = {((r["width_bits"], r["accumulator_bits"]), r["target_ns"]): r for r in records}
    ratios = iso_clock_ratios(
        points, sorted(args.targets, reverse=True), list(CASES),
        tuple(((w, 64), (w, 32)) for w in (4, 5, 6) if (w, 64) in cases and (w, 32) in cases)
    ) if all(c in cases for c in CASES) else None
    summary = {
        "schema_version": "1.0.0",
        "scope": "timing-driven iso-clock synthesis + OpenSTA for the existing integer MAC; no new RTL",
        "tool_versions": {"yosys": version("yosys"),
                          "openroad_wrapper": subprocess.run([str(WRAPPER), "--version"], capture_output=True,
                                                             text=True, check=False).stdout.strip()},
        "constraints": {"clock_uncertainty_ns": UNCERTAINTY_NS, "input_delay_ns": IO_DELAY_NS,
                        "output_delay_ns": IO_DELAY_NS, "output_load_pf": OUTPUT_LOAD_PF,
                        "abc_driving_cell": DRIVING_CELL, "abc_output_load_ff": ABC_LOAD_FF,
                        "abc_delay_ps_rule": f"target_ns*1000 - {ABC_OVERHEAD_PS}", "time_unit": "ns"},
        "validity_rule": "worst setup path has no pin violating max_transition, max_capacitance or max_fanout",
        "hashes": {"liberty": digest(LIBERTY), "tech_lef": digest(TECH_LEF), "cells_lef": digest(CELLS_LEF),
                   "wrapper": digest(WRAPPER), "matrix_script": digest(Path(__file__)),
                   "rtl": {p: digest(ROOT / p) for p in RTL}},
        "targets_ns": args.targets, "records": records, "curves": curves(records), "iso_clock": ratios,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    with args.table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(table_rows(records))


if __name__ == "__main__":
    main()
