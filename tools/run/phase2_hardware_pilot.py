"""Reproducible synthesis and exhaustive RTL simulation for four pilot families."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from public.generic_rtl.arithmetic.pilot import rtl, testbench
from public.pdk_flow.common.run_flow import run
from public.analysis.hardware.parser import parse_yosys_stat

ROOT = Path(__file__).resolve().parents[2]


def synthesis_metrics(work, evidence, tool_version):
    identity = {**evidence["config"], "rtl_sha256": evidence["rtl_sha256"],
                "tool": "yosys", "tool_version": tool_version,
                "liberty_sha256": evidence["external_resource_hashes"]["liberty_sha256"]}
    path = work / "yosys-stat.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != evidence["outputs"]["yosys-stat.json"]["sha256"]:
        raise ValueError("synthesis report hash mismatch")
    normalized = parse_yosys_stat(path, top="precision_pilot", identity=identity)
    raw = json.loads(path.read_text())
    # Older hierarchical Yosys stat reports exclude child area from the top
    # module. All-in area is the explicit design total, never the wrapper.
    metrics = normalized["metrics"]
    metrics["wrapper_cell_area"] = metrics["cell_area"]
    metrics["cell_area"] = {"value": raw["design"]["area"], "unit": "library_area_units"}
    metrics["core_cell_area"] = {"value": raw["modules"]["\\precision_primitive"]["area"], "unit": "library_area_units"}
    if abs(metrics["cell_area"]["value"] - metrics["wrapper_cell_area"]["value"] - metrics["core_cell_area"]["value"]) > 1e-5:
        raise ValueError("hierarchical synthesis areas do not sum")
    normalized["area_accounting"] = "design total = primitive core + registered wrapper"
    return normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yosys", required=True)
    parser.add_argument("--iverilog", required=True)
    parser.add_argument("--vvp", required=True)
    parser.add_argument("--liberty", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/summaries/phase2-hardware-pilot.json")
    parser.add_argument("--summarize-only", action="store_true", help="verify and normalize existing pilot synthesis reports")
    args = parser.parse_args()
    tool_version = subprocess.check_output([args.yosys, "-V"], text=True).strip()
    if args.summarize_only:
        report = json.loads(args.output.read_text())
        if report["tool_version"] != tool_version or report["liberty_sha256"] != hashlib.sha256(args.liberty.read_bytes()).hexdigest():
            raise ValueError("pilot tool/library identities changed")
        for record in report["runs"]:
            path = ROOT / record["run_manifest"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["run_manifest_sha256"]:
                raise ValueError("pilot manifest identity mismatch")
            record["synthesis"] = synthesis_metrics(path.parent, json.loads(path.read_text()), tool_version)
        report["timing"] = "separate evidence: phase2-sta-power-pilot.json, when available"
        report["power"] = "separate evidence: phase2-sta-power-pilot.json, when available"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    records = []
    for name in ("int4", "fp4_e2m1", "posit4_es0", "log4"):
        for stages in range(4):
            work = ROOT / "artifacts/ppa/phase2" / f"{name}-mul-p{stages}"
            work.mkdir(parents=True, exist_ok=True)
            design, latency = rtl(name, "mul", stages)
            source = work / "primitive.sv"
            source.write_text(design)
            tb = work / "testbench.sv"
            tb.write_text(testbench(name, "mul", latency))
            simulation = work / "sim"
            subprocess.run([args.iverilog, "-g2012", "-s", "tb", "-o", str(simulation), str(source), str(tb)], check=True)
            result = subprocess.check_output([args.vvp, str(simulation)], text=True)
            (work / "simulation.log").write_text(result)
            config = {"schema_version": "1.0.0", "pdk": "ics55", "library": "rvt",
                      "corner": "typ_tt_1p2_25_nldm", "vt": "rvt", "rtl": str(source.relative_to(ROOT)),
                      "top": "precision_pilot", "harness_version": "2.0.0", "internal_pipeline_stages": stages,
                      "pipeline_stages": latency, "target_clock": "10ns"}
            config_path = work / "config.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            evidence = run(config_path=config_path, work_dir=work, liberty=args.liberty, yosys=args.yosys)
            records.append({"format": name, "operation": "mul", "internal_stages": stages,
                            "latency_cycles": latency, "initiation_interval_cycles": 1,
                            "exhaustive_pairs": 256, "simulation": result.strip(),
                            "run_manifest": str((work / "run-manifest.json").relative_to(ROOT)),
                            "run_manifest_sha256": hashlib.sha256((work / "run-manifest.json").read_bytes()).hexdigest(),
                            "rtl_sha256": evidence["rtl_sha256"], "synthesis": synthesis_metrics(work, evidence, tool_version)})
            print(f"{name}: stages={stages}, simulation and mapped synthesis passed", flush=True)
    report = {"schema_version": "2.0.0", "scope": "primitive ROM baseline flow pilots; not arithmetic architecture ranking",
              "liberty_sha256": hashlib.sha256(args.liberty.read_bytes()).hexdigest(),
              "tool_version": tool_version, "runs": records,
              "timing": "separate evidence: phase2-sta-power-pilot.json, when available",
              "power": "separate evidence: phase2-sta-power-pilot.json, when available"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
