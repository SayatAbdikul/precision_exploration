"""Bounded ICS55 integer-MAC synthesis and optional cell-delay/placement pilot.

All generated products stay beneath --work-dir.  This is intentionally not a
route or power flow: absent tools/collateral are recorded as unavailable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from public.analysis.hardware import parse_opensta_report, parse_yosys_stat
from public.pdk_flow.common.run_flow import run


ROOT = Path(__file__).resolve().parents[3]
CASES = ((4, 32), (5, 32), (6, 32), (8, 32), (8, 64))
PERIODS_NS = (10, 5, 2)
RTL = (
    "public/generic_rtl/mac/integer_mac.sv",
    "public/generic_rtl/mac/integer_mac_harness.sv",
)
TOP = "integer_mac_harness"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def version(executable: str, flag: str = "-V") -> str:
    try:
        result = subprocess.run([executable, flag], capture_output=True, text=True, check=False, timeout=10)
        lines = (result.stdout or result.stderr).strip().splitlines()
        return lines[0] if lines else "version unavailable"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"version unavailable: {exc}"


def core_name(stat_path: Path) -> str:
    modules = json.loads(stat_path.read_text())["modules"]
    matches = [name for name in modules if "integer_mac" in name and "harness" not in name]
    if len(matches) != 1:
        raise ValueError(f"expected one integer_mac core module; got {matches}")
    return matches[0]


def register_split(stat_path: Path, core_module: str) -> tuple[int, int]:
    modules = json.loads(stat_path.read_text())["modules"]
    def count(cells: dict[str, int]) -> int:
        return sum(value for name, value in cells.items() if name.lstrip("\\$_").upper().startswith(("DFF", "SDFF")))
    core = count(modules[core_module]["num_cells_by_type"])
    wrapper = count(modules[f"\\{TOP}"]["num_cells_by_type"])
    return core, wrapper


def run_sta(*, executable: str, liberty: Path, netlist: Path, period: int, output: Path) -> tuple[Path, Path]:
    report = output / f"opensta-{period}ns.txt"
    log = output / f"opensta-{period}ns.log"
    environment = {
        **os.environ,
        "TOP": TOP,
        "CLOCK_PORT": "clk",
        "LIBERTY_PATH": str(liberty),
        "NETLIST_PATH": str(netlist),
        "TARGET_PERIOD_NS": str(period),
        "REPORT_PATH": str(report),
    }
    script = ROOT / "public/pdk_flow/common/opensta_matrix.tcl"
    result = subprocess.run([executable, str(script)], cwd=output, env=environment,
                            capture_output=True, text=True, check=False, timeout=90)
    log.write_text(result.stdout + "\nSTDERR:\n" + result.stderr)
    if result.returncode or not report.exists():
        raise RuntimeError(f"OpenSTA failed with status {result.returncode}; see {log}")
    return report, log


def run_placement(*, executable: str, liberty: Path, tech_lef: Path,
                  cells_lef: Path, netlist: Path, output: Path) -> tuple[Path, Path]:
    placed = output / "placed.def"
    log = output / "openroad.log"
    environment = {**os.environ, "TOP": TOP, "NETLIST": str(netlist),
                   "LIBERTY": str(liberty), "TECH_LEF": str(tech_lef),
                   "CELLS_LEF": str(cells_lef), "OUTPUT_DEF": str(placed)}
    script = ROOT / "public/pdk_flow/common/openroad_pilot.tcl"
    result = subprocess.run([executable, str(script)], cwd=output, env=environment,
                            capture_output=True, text=True, check=False, timeout=180)
    log.write_text(result.stdout + "\nSTDERR:\n" + result.stderr)
    if result.returncode or not placed.exists():
        raise RuntimeError(f"OpenROAD failed with status {result.returncode}; see {log}")
    return placed, log


def stage_summary(statuses: list[str]) -> str:
    if "failed" in statuses:
        return "failed"
    if "unavailable" in statuses:
        return "unavailable"
    if "not_requested" in statuses:
        return "not_requested" if all(x == "not_requested" for x in statuses) else "completed"
    return "completed"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liberty", type=Path, required=True)
    parser.add_argument("--tech-lef", type=Path)
    parser.add_argument("--cells-lef", type=Path)
    parser.add_argument("--yosys", default="yosys")
    parser.add_argument("--opensta", help="Optional OpenSTA binary for cell-delay-only timing")
    parser.add_argument("--openroad", help="Optional OpenROAD binary for preliminary placement of INT4")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "artifacts/ppa/ics55-integer-validation")
    parser.add_argument("--summary", type=Path, default=ROOT / "results/summaries/ics55-integer-validation.json")
    parser.add_argument("--table", type=Path, default=ROOT / "results/tables/ics55-integer-validation.csv")
    args = parser.parse_args(argv)
    liberty = args.liberty.resolve(strict=True)
    pdk_root = liberty.parents[5] if len(liberty.parents) > 5 else liberty.parent
    revision = subprocess.run(["git", "-C", str(pdk_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False)
    pdk_commit = revision.stdout.strip() if revision.returncode == 0 else None
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    tool_version = version(args.yosys)
    placement_lefs = None
    if args.openroad and args.tech_lef and args.cells_lef:
        if args.tech_lef.is_file() and args.cells_lef.is_file():
            placement_lefs = (args.tech_lef.resolve(), args.cells_lef.resolve())
    source_hashes = {path: digest(ROOT / path) for path in RTL}
    pdk_hashes = {"liberty": digest(liberty)}
    for label, path in (("tech_lef", args.tech_lef), ("cells_lef", args.cells_lef)):
        if path and path.is_file():
            pdk_hashes[label] = digest(path.resolve())
    records = []
    for width, acc_bits in CASES:
        case_dir = work_dir / f"int{width}-acc{acc_bits}"
        case_dir.mkdir(parents=True, exist_ok=True)
        config_path = case_dir / "config.json"
        config = {
            "schema_version": "1.0.0", "pdk": "ics55", "library": "ics55_LLSC_H7CR",
            "corner": "typ_tt_1p2_25_nldm", "vt": "rvt", "rtl": list(RTL),
            "top": TOP, "parameters": {"W_BITS": width, "A_BITS": width, "ACC_BITS": acc_bits},
            "harness_version": "integer-mac-1.0.0", "target_clock": "unconstrained",
        }
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        row = {"width_bits": width, "accumulator_bits": acc_bits, "config_sha256": digest(config_path),
               "synthesis_status": "failed", "placement_status": "not_requested",
               "dut_latency_cycles": 1, "external_wrapper_latency_edges": 2,
               "pair_initiation_interval_cycles": 1,
               "k_product_dot_cycles": "K", "dot_initiation_interval_cycles": "K",
               "dot_throughput_expression": "f/K for back-to-back K-product dots",
               "measured_dot_throughput_per_second": None,
               "core_area_library_units": None, "wrapper_area_library_units": None,
               "total_area_library_units": None, "core_cells": None, "total_cells": None,
               "registers_total": None, "core_registers": None, "wrapper_registers": None,
               "timing": []}
        try:
            manifest = run(config_path=config_path, work_dir=case_dir, liberty=liberty, yosys=args.yosys)
            identity = {
                "rtl_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
                "harness_version": config["harness_version"], "pdk": "ics55",
                "library": config["library"], "corner": config["corner"], "vt": "rvt",
                "tool": "Yosys", "tool_version": tool_version, "target_clock": "unconstrained",
                "pipeline_stages": 0, "core_module": core_name(case_dir / "yosys-stat.json"),
                "config_sha256": row["config_sha256"], "liberty_sha256": pdk_hashes["liberty"],
                "accounting_boundary": "DUT includes accumulator/control; wrapper includes neutral launch/capture registers",
            }
            normalized = parse_yosys_stat(case_dir / "yosys-stat.json", top=TOP, identity=identity)
            core_registers, wrapper_registers = register_split(case_dir / "yosys-stat.json", identity["core_module"])
            if core_registers + wrapper_registers != normalized["metrics"]["register_count"]["value"]:
                raise ValueError("core and wrapper register counts do not match design total")
            row.update({"synthesis_status": "completed", "synthesis": normalized,
                        "run_manifest_sha256": digest(case_dir / "run-manifest.json"),
                        "netlist_sha256": manifest["outputs"]["netlist.v"]["sha256"],
                        "core_area_library_units": normalized["metrics"]["core_cell_area"]["value"],
                        "wrapper_area_library_units": normalized["metrics"]["wrapper_cell_area"]["value"],
                        "total_area_library_units": normalized["metrics"]["cell_area"]["value"],
                        "core_cells": normalized["metrics"]["core_cell_count"]["value"],
                        "total_cells": normalized["metrics"]["leaf_cell_count"]["value"],
                        "registers_total": normalized["metrics"]["register_count"]["value"],
                        "core_registers": core_registers, "wrapper_registers": wrapper_registers})
        except Exception as exc:
            row["error"] = str(exc)

        if args.openroad and (width, acc_bits) == (4, 32):
            if row["synthesis_status"] != "completed":
                row.update(placement_status="unavailable", placement_reason="synthesis prerequisite failed")
            elif not placement_lefs:
                row.update(placement_status="unavailable", placement_reason="technology or cell LEF missing")
            elif not shutil.which(args.openroad):
                row.update(placement_status="unavailable", placement_reason="OpenROAD executable unavailable")
            else:
                try:
                    placed, log = run_placement(executable=args.openroad, liberty=liberty,
                                                tech_lef=placement_lefs[0], cells_lef=placement_lefs[1],
                                                netlist=case_dir / "netlist.v", output=case_dir)
                    row.update(placement_status="completed", placed_def_sha256=digest(placed),
                               placement_log_sha256=digest(log))
                except Exception as exc:
                    row.update(placement_status="failed", placement_reason=str(exc))
                    log = case_dir / "openroad.log"
                    if log.exists():
                        row["placement_log_sha256"] = digest(log)

        for period in PERIODS_NS:
            timing_row = {"target_period_ns": period, "status": "not_requested",
                          "worst_slack_ns": None, "fmax_mhz": None}
            if args.opensta:
                if row["synthesis_status"] != "completed":
                    timing_row.update(status="unavailable", reason="synthesis prerequisite failed")
                elif not shutil.which(args.opensta):
                    timing_row.update(status="unavailable", reason="OpenSTA executable unavailable")
                else:
                    timing_identity = {**identity, "tool": "OpenSTA", "tool_version": version(args.opensta, "-version"),
                                       "target_clock": f"{period}ns", "time_unit": "ns",
                                       "netlist_sha256": row["netlist_sha256"],
                                       "timing_tcl_sha256": digest(ROOT / "public/pdk_flow/common/opensta_matrix.tcl")}
                    try:
                        report, log = run_sta(executable=args.opensta, liberty=liberty,
                                              netlist=case_dir / "netlist.v", period=period, output=case_dir)
                        timing = parse_opensta_report(report, identity=timing_identity)
                        timing_row.update(status="completed", normalized=timing,
                                          report_sha256=digest(report), log_sha256=digest(log),
                                          worst_slack_ns=timing["metrics"]["worst_slack"]["value"],
                                          fmax_mhz=timing["metrics"]["fmax"]["value"])
                    except Exception as exc:
                        timing_row.update(status="failed", reason=str(exc))
                        log = case_dir / f"opensta-{period}ns.log"
                        if log.exists():
                            timing_row["log_sha256"] = digest(log)
            row["timing"].append(timing_row)
        records.append(row)
        print(f"INT{width}/ACC{acc_bits}: {row['synthesis_status']}", flush=True)
    summary = {
        "schema_version": "1.0.0", "scope": "integer MAC core and neutral wrapper; no scaling/conversion",
        "evidence_level": "ics55_rvt_logic_synthesis; optional cell-delay-only STA/placement",
        "tool_versions": {"yosys": tool_version, "opensta": version(args.opensta, "-version") if args.opensta else None,
                          "openroad": version(args.openroad, "-version") if args.openroad else None},
        "pdk_inputs": {"pdk_root": str(pdk_root), "pdk_git_commit": pdk_commit,
                       "liberty_path": str(liberty), "hashes": pdk_hashes,
                       "corner": "typ_tt_1p2_25_nldm", "vt": "rvt"},
        "prior_pinned_comparison": {"release": "v1.10.102", "commit": "68d89edb47847671e18f9e65d66c0cd883995e05",
                                    "liberty_sha256": "15af0dceceeeb02e174174e7123c7d551551369bf5ad1e8417d3ce8dcbc24fcd",
                                    "status": "prior release payload unavailable on current host; not a same-library repeat"},
        "host_tool_inventory": {name: shutil.which(name) for name in ("yosys", "sta", "openroad", "iEDA")},
        "source_hashes": source_hashes,
        "flow_source_hashes": {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), ROOT / "public/pdk_flow/common/run_flow.py",
            ROOT / "public/pdk_flow/common/opensta_matrix.tcl",
            ROOT / "public/pdk_flow/common/openroad_pilot.tcl")},
        "clock_targets_ns": list(PERIODS_NS),
        "constraints": {"clock_uncertainty_ns": 0.10, "input_delay_ns": 0.20,
                        "output_delay_ns": 0.20, "output_load_pf": 0.01,
                        "mapping_clock_constraint": None,
                        "timing_tcl_sha256": digest(ROOT / "public/pdk_flow/common/opensta_matrix.tcl"),
                        "note": "Yosys/ABC mapping was unconstrained; listed 10/5/2 ns points require OpenSTA"},
        "algorithm": {"accepted_operand_pairs_per_cycle": 1,
                      "external_dot_end_to_captured_result_edge_offset": 2,
                      "internal_arithmetic_pipeline_boundaries": 0,
                      "dot_product_K_product_cycles": "K accepted pair cycles for a K-product dot",
                      "dot_result_initiation_interval": "K cycles for back-to-back K-product dots",
                      "dot_throughput_at_frequency_f": "f/K for fixed K with back-to-back dots; no fmax measured here"},
        "flow_capabilities": {
            "mapped_synthesis": {"status": stage_summary([row["synthesis_status"] for row in records]), "tool": "Yosys"},
            "post_synthesis_sta": {"status": stage_summary([entry["status"] for row in records for entry in row["timing"]]),
                                   "requested": bool(args.opensta)},
            "placement": {"status": records[0]["placement_status"], "requested": bool(args.openroad),
                          "scope": "INT4/ACC32 preliminary placement pilot"},
            "cts": {"status": "unvalidated", "missing": "validated placement and compatible CTS/tool setup"},
            "routing": {"status": "unvalidated", "missing": "validated placement/CTS and compatible router"},
            "parasitic_extraction": {"status": "unavailable_in_pdk_preview", "missing": "validated ICS55 RC collateral and extractor"},
            "activity_power": {"status": "unvalidated", "missing": "workload switching traces, complete timing/physical flow"},
            "physical_sram": {"status": "unavailable_in_pdk_preview", "missing": "characterized SRAM macro or validated ICS55 compiler"},
        },
        "records": records,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    columns = ("width_bits", "accumulator_bits", "synthesis_status", "core_area_library_units",
               "wrapper_area_library_units", "total_area_library_units", "core_cells", "total_cells",
               "registers_total", "core_registers", "wrapper_registers",
               "dut_latency_cycles", "external_wrapper_latency_edges", "pair_initiation_interval_cycles",
               "k_product_dot_cycles", "dot_initiation_interval_cycles", "dot_throughput_expression",
               "measured_dot_throughput_per_second", "placement_status", "target_period_ns", "timing_status",
               "worst_slack_ns", "fmax_mhz")
    with args.table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            for timing in row["timing"]:
                writer.writerow({**{key: row.get(key) for key in columns},
                                 "target_period_ns": timing["target_period_ns"],
                                 "timing_status": timing["status"],
                                 "worst_slack_ns": timing.get("worst_slack_ns"),
                                 "fmax_mhz": timing.get("fmax_mhz")})
    failures = []
    if any(row["synthesis_status"] != "completed" for row in records):
        failures.append("synthesis")
    if args.opensta and any(entry["status"] != "completed" for row in records for entry in row["timing"]):
        failures.append("requested STA")
    if args.openroad and records[0]["placement_status"] != "completed":
        failures.append("requested placement")
    if failures:
        raise SystemExit(f"incomplete stages: {', '.join(failures)}; inspect summary")


if __name__ == "__main__":
    main()
