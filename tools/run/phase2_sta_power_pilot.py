"""Measure generic cell-delay timing and vectorless power with local iSTA/iPA."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess

from public.analysis.hardware.ista import parse_reports

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ieda", required=True)
    parser.add_argument("--yosys", required=True)
    parser.add_argument("--liberty", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", choices=["int4", "fp4_e2m1", "posit4_es0", "log4"], default=["int4", "fp4_e2m1", "posit4_es0", "log4"])
    parser.add_argument("--stages", nargs="+", type=int, default=[0,1,2,3])
    parser.add_argument("--periods", nargs="+", type=float, default=[10,5,2])
    args = parser.parse_args()
    if not re.search(r'time_unit\s*:\s*"1ns"', args.liberty.read_text()):
        raise ValueError("pilot report parser requires a Liberty time unit of 1ns")
    if any(stage not in range(4) for stage in args.stages):
        parser.error("stages must be in 0..3")
    records = []
    script = ROOT / "public/pdk_flow/common/ista_pilot.tcl"
    for name in args.formats:
        for stages in args.stages:
            source = ROOT / "artifacts/ppa/phase2" / f"{name}-mul-p{stages}" / "netlist.v"
            if not source.is_file():
                raise ValueError(f"synthesized netlist missing: {source}")
            original_source = source
            flattened = source.with_name("netlist-ista.v")
            adapter = source.with_name("ista-netlist.ys")
            adapter.write_text(f"read_verilog {json.dumps(str(source))}\n"
                               "hierarchy -top precision_pilot\nflatten\nclean -purge\nsplitnets -ports\nclean -purge\n"
                               f"write_verilog -noattr -noexpr {json.dumps(str(flattened))}\n")
            flattened_run = subprocess.run([args.yosys,"-s",str(adapter)],capture_output=True,text=True)
            adapter.with_suffix(".log").write_text(flattened_run.stdout+flattened_run.stderr)
            if flattened_run.returncode:
                raise RuntimeError("iSTA netlist adapter failed")
            source = flattened
            for period in args.periods:
                if period <= 0:
                    raise ValueError("clock period must be positive")
                output = source.parent / f"ista-{period:g}ns"
                output.mkdir(parents=True,exist_ok=True)
                sdc = output / "constraints.sdc"
                sdc.write_text(f"create_clock -name clk -period {period:g} [get_ports clk]\n"
                    "set_clock_uncertainty -setup 0.10 [get_clocks clk]\n"
                    "set_clock_uncertainty -hold 0.10 [get_clocks clk]\n"
                    "set_input_delay 0.20 -clock clk [get_ports {reset input_valid a[0] a[1] a[2] a[3] b[0] b[1] b[2] b[3]}]\n"
                    "set_output_delay 0.20 -clock clk [get_ports {output_valid result[0] result[1] result[2] result[3]}]\n"
                    "set_load 0.01 [get_ports {output_valid result[0] result[1] result[2] result[3]}]\n")
                command = [args.ieda,"-script",str(script),str(source),str(args.liberty.resolve()),str(sdc),str(output),"precision_pilot"]
                # Remove only reports owned by this invocation, so a failed tool
                # cannot pass by leaving stale reports from an earlier run.
                for suffix in (".rpt", ".pwr", "_instance.pwr"):
                    (output / ("precision_pilot" + suffix)).unlink(missing_ok=True)
                result = subprocess.run(command,cwd=output,text=True,capture_output=True,timeout=240)
                (output/"tool.log").write_text(result.stdout+result.stderr)
                reports = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in output.iterdir() if p.is_file() and p.suffix in {".rpt",".pwr",".sdc",".log"}}
                timing, power = output/"precision_pilot.rpt", output/"precision_pilot.pwr"
                complete = result.returncode == 0 and timing.is_file() and power.is_file()
                records.append({"format":name,"internal_stages":stages,"period_ns":period,"returncode":result.returncode,
                                "status":"measured" if complete else "failed",
                                "mapped_netlist_sha256":hashlib.sha256(original_source.read_bytes()).hexdigest(),
                                "netlist_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                                "netlist_adapter_sha256":hashlib.sha256(adapter.read_bytes()).hexdigest(),"reports":reports,
                                "normalized":parse_reports(timing,power) if complete else None})
                print(f"{name} p{stages} {period:g}ns: {records[-1]['status']}",flush=True)
                if not complete:
                    raise RuntimeError(f"STA/power did not generate required reports; inspect {output/'tool.log'}")
    report = {"schema_version":"2.0.0","tool":"iEDA iSTA/iPA",
              "tool_binary_sha256":hashlib.sha256(Path(args.ieda).read_bytes()).hexdigest(),
              "liberty_sha256":hashlib.sha256(args.liberty.read_bytes()).hexdigest(),
              "script_sha256":hashlib.sha256(script.read_bytes()).hexdigest(),
              "scope":"post-synthesis cell-delay timing; vectorless power with toggle=0.1; no extracted parasitics or workload activity",
              "runs":records}
    (ROOT/"results/summaries/phase2-sta-power-pilot.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
