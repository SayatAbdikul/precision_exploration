"""Hash and normalize the task-owned ECC INT4 physical pilot evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT = ROOT / "artifacts/ppa/ics55-hardware-foundation/integer4-ecc-readonly-runtime"
DEFAULT_SUMMARY = ROOT / "results/summaries/ics55-ecc-integer4-pilot-v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(report: str, name: str) -> float:
    matches = re.findall(r"^\|\s*" + re.escape(name) + r"\s*\|\s*([0-9.]+)", report, re.M)
    if len(matches) != 1:
        raise ValueError(f"expected one {name} metric, got {len(matches)}")
    return float(matches[0])


def _instance_area(report: str) -> float:
    matches = re.findall(r"^\| All Instances\s*\|\s*\d+\s*\|\s*[0-9.]+\s*\|\s*([0-9.]+)", report, re.M)
    if len(matches) != 1:
        raise ValueError(f"expected one all-instance area, got {len(matches)}")
    return float(matches[0])


def _ecc_qor(report: str) -> tuple[float, int]:
    matches = re.findall(r"^clk\s+([-+0-9.]+)\s+[-+0-9.]+\s+\d+\s+(\d+)MHz\s+", report, re.M)
    if len(matches) != 1:
        raise ValueError(f"expected one ECC clk QOR row, got {len(matches)}")
    return float(matches[0][0]), int(matches[0][1])


def summarize(project: Path) -> dict:
    project = project.resolve(strict=True)
    run = project / "runs/default"
    flow_path = run / "home/flow.json"
    flow = json.loads(flow_path.read_text())
    stages = [{"name": item["name"], "status": item["state"], "runtime": item["runtime"]}
              for item in flow["steps"]]
    place_report = run / "place_dreamplace/report/place.db.rpt"
    route_report = run / "route_ecc/report/route.db.rpt"
    drc_path = run / "drc_ecc/feature/drc.step.json"
    drc = json.loads(drc_path.read_text())
    place = place_report.read_text()
    route = route_report.read_text()
    sdc = run / "origin/integer_mac_harness.sdc"
    qor = run / "Synthesis_yosys/report/post_synthesis/qor_summary.rpt"
    ecc_wns, ecc_estimated_frequency = _ecc_qor(qor.read_text())
    source_list = project / "sources.f"
    rtl = [(project / line.strip()).resolve(strict=True) for line in source_list.read_text().splitlines()
           if line.strip() and not line.lstrip().startswith("#")]
    if len(rtl) != 2 or any(not path.is_relative_to(ROOT) for path in rtl):
        raise ValueError("unexpected RTL filelist in ECC pilot")
    deliverables = {
        "place_def_gz": run / "place_dreamplace/output/integer_mac_harness_place.def.gz",
        "route_def_gz": run / "route_ecc/output/integer_mac_harness_route.def.gz",
        "route_gds": run / "route_ecc/output/integer_mac_harness_route.gds",
        "filler_gds": run / "filler_ecc/output/integer_mac_harness_filler.gds",
    }
    for path in deliverables.values():
        path.resolve(strict=True)
    prior = json.loads((ROOT / "results/summaries/ics55-integer-validation.json").read_text())
    project_config = tomllib.loads((project / "ecc.toml").read_text())
    pdk_root = Path(project_config["pdk"]["root"]).resolve(strict=True)
    pdk_files = {
        "liberty": Path(prior["pdk_inputs"]["liberty_path"]),
        "tech_lef": pdk_root / "prtech/techLEF/N551P6M.lef",
        "cells_lef": pdk_root / "IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/lef/ics55_LLSC_H7CR.lef",
    }
    current_pdk_hashes = {name: sha(path.resolve(strict=True)) for name, path in pdk_files.items()}
    if current_pdk_hashes != prior["pdk_inputs"]["hashes"]:
        raise ValueError("ECC PDK inputs differ from pinned ICS55 matrix inputs")
    ecc_root = pdk_root.parents[2]
    def commit(path: Path) -> str:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    return {
        "schema_version": "ics55-ecc-integer-pilot-1.0.0",
        "summary_generator_sha256": sha(Path(__file__)),
        "evidence_level": "task_owned_ECC_physical_pilot_not_foundry_signoff",
        "design": "integer_mac_harness", "parameters": {"W_BITS": 4, "A_BITS": 4, "ACC_BITS": 32},
        "frequency_target_mhz": 100,
        "runtime": {"tool": "ECC", "version": "0.1.0a5", "placement": "DreamPlace",
                    "report_engine": "iEDA V23.03-OS-01", "editable_rebuild_disabled": True,
                    "ecc_source_commit": commit(ecc_root),
                    "dreamplace_source_commit": commit(ecc_root / "chipcompiler/thirdparty/ecc-dreamplace"),
                    "wrapper_sha256": sha(ROOT / "tools/hardware/ecc_readonly_runtime.py")},
        "inputs": {"ecc_toml_sha256": sha(project / "ecc.toml"),
                   "project_generator_sha256": sha(ROOT / "tools/hardware/prepare_ecc_pilot.py"),
                   "filelist_sha256": sha(source_list),
                   "rtl_source_hashes": {str(path.relative_to(ROOT)): sha(path) for path in rtl},
                   "sdc_sha256": sha(sdc),
                   "pdk_commit": prior["pdk_inputs"]["pdk_git_commit"],
                   "pdk_hashes": current_pdk_hashes},
        "stages": stages, "all_stages_success": all(item["status"] == "Success" for item in stages),
        "flow_sha256": sha(flow_path),
        "physical_metrics": {
            "placed_die_area_um2": _metric(place, "DIE Area ( um^2 )"),
            "placed_core_area_um2": _metric(place, "CORE Area ( um^2 )"),
            "placed_instance_area_um2": _instance_area(place),
            "routed_die_area_um2": _metric(route, "DIE Area ( um^2 )"),
            "routed_instance_area_um2": _instance_area(route),
            "drc_count_tool_reported": drc["drc"]["number"],
            "drc_gate_status": json.loads((run / "drc_ecc/checklist.json").read_text())["status"],
        },
        "evidence_hashes": {"place_db_report": sha(place_report), "route_db_report": sha(route_report),
                            "drc_step": sha(drc_path), "post_synthesis_qor": sha(qor),
                            **{name: sha(path) for name, path in deliverables.items()}},
        "timing": {"ecc_post_synthesis_qor_present": qor.exists(),
                   "ecc_post_synthesis_setup_wns_ns": ecc_wns,
                   "ecc_estimated_frequency_mhz": ecc_estimated_frequency,
                   "ecc_target_period_ns": 10.0,
                   "opensta_matrix_completed": False, "post_route_sta_completed": False,
                   "validated_fmax_mhz": None},
        "limits": ["ECC is a separate flow from the five unconstrained Yosys/ABC area points",
                   "route and DRC success are tool pilot results, not foundry signoff",
                   "no validated extracted RC or post-route STA", "no workload power or characterized SRAM"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    summary = summarize(args.project)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"ECC pilot: {len(summary['stages'])} stages, all_success={summary['all_stages_success']} -> {args.summary}")


if __name__ == "__main__":
    main()
