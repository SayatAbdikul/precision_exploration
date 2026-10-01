"""Compare raw MAC with exact power-of-two support under one ICS55 mapping flow.

Run from repository root with --liberty pointing at the identified RVT TT
Liberty. All generated netlists stay under --work-dir. These are unconstrained
cell-mapped area points, not timing or physical area.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from public.analysis.hardware.parser import parse_yosys_stat
from public.pdk_flow.common.run_flow import run
from tools.hardware.integer_mac_baseline import ROOT, sha
from tools.hardware.integer_output_conversion import OutputConfig

RAW_SOURCES = ("public/generic_rtl/mac/integer_mac.sv",
               "public/generic_rtl/mac/integer_mac_harness.sv")
SCALED_SOURCES = ("public/generic_rtl/mac/integer_mac.sv",
                  "public/generic_rtl/mac/integer_output_convert.sv",
                  "public/generic_rtl/mac/integer_mac_scaled.sv",
                  "public/generic_rtl/mac/integer_mac_scaled_harness.sv")
CASES = ((4, 32, 4, -1, False), (8, 64, 8, -2, True))


def _submodule(stat: dict, suffix: str) -> tuple[str, dict]:
    matches = [(name, module) for name, module in stat["modules"].items()
               if name.endswith("\\" + suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one {suffix} module, got {[name for name, _ in matches]}")
    return matches[0]


def _register_count(module: dict) -> int:
    return sum(count for name, count in module["num_cells_by_type"].items()
               if name.lstrip("\\$_").upper().startswith(("DFF", "SDFF")))


def mapped_case(*, label: str, top: str, sources: tuple[str, ...], parameters: dict,
                numerical: dict, folder: Path, liberty: Path, yosys: str) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    config = {"schema_version": "integer-output-area-1.0.0", "pdk": "ics55",
              "library": "ics55_LLSC_H7CR", "corner": "typ_tt_1p2_25_nldm", "vt": "rvt",
              "rtl": list(sources), "top": top, "parameters": parameters,
              "target_clock": "unconstrained", "numerical": numerical}
    config_path = folder / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    manifest = run(config_path=config_path, work_dir=folder, liberty=liberty, yosys=yosys)
    stat_path = folder / "yosys-stat.json"
    stat = json.loads(stat_path.read_text())
    tool_version = subprocess.run([yosys, "-V"], capture_output=True, text=True, check=True).stdout.strip()
    identity = {"rtl_sha256": manifest["rtl_sha256"], "harness_version": "integer-mac-harness-1.0.0",
                "pdk": "ics55", "library": config["library"], "corner": config["corner"], "vt": "rvt",
                "tool": "Yosys", "tool_version": tool_version, "target_clock": "unconstrained",
                "pipeline_stages": 1 if label == "scaled" else 0,
                "config_sha256": sha(config_path), "liberty_sha256": sha(liberty)}
    normalized = parse_yosys_stat(stat_path, top=top, identity=identity)
    top_module = stat["modules"]["\\" + top]
    total_area = normalized["metrics"]["cell_area"]["value"]
    wrapper_area = top_module["area"]
    wrapper_registers = _register_count(top_module)
    total_registers = normalized["metrics"]["register_count"]["value"]
    mac_name, mac_module = _submodule(stat, "integer_mac")
    conversion = None
    if label == "scaled":
        conversion_name, conversion_module = _submodule(stat, "integer_output_convert")
        conversion = {"module": conversion_name, "area_library_units": conversion_module["area"],
                      "registers": _register_count(conversion_module)}
    return {"label": label, "config_sha256": sha(config_path),
            "run_manifest_sha256": sha(folder / "run-manifest.json"),
            "netlist_sha256": manifest["outputs"]["netlist.v"]["sha256"],
            "rtl_source_hashes": manifest["rtl_source_hashes"],
            "normalized": normalized,
            "total_area_library_units": total_area,
            "wrapper_area_library_units": wrapper_area,
            "core_area_library_units": round(total_area - wrapper_area, 4),
            "total_registers": total_registers,
            "wrapper_registers": wrapper_registers,
            "core_registers": total_registers - wrapper_registers,
            "mac_module": mac_name, "mac_module_area_library_units": mac_module["area"],
            "mac_module_registers": _register_count(mac_module),
            "conversion_module": conversion}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liberty", type=Path, required=True)
    parser.add_argument("--yosys", default="yosys")
    parser.add_argument("--work-dir", type=Path,
                        default=ROOT / "artifacts/ppa/integer-output-conversion-v1")
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "results/summaries/integer-output-conversion-area-v1.json")
    args = parser.parse_args(argv)
    liberty = args.liberty.resolve(strict=True)
    rows = []
    for width, acc, out, exponent, relu in CASES:
        config = OutputConfig(acc, out, exponent, relu)
        base = args.work_dir / f"int{width}-acc{acc}-out{out}-e{exponent}-relu{int(relu)}"
        raw = mapped_case(label="raw", top="integer_mac_harness", sources=RAW_SOURCES,
                          parameters={"W_BITS": width, "A_BITS": width, "ACC_BITS": acc},
                          numerical={"mac_model": "C", "scale": "1", "output": "accumulator_code"},
                          folder=base / "raw", liberty=liberty, yosys=args.yosys)
        scaled = mapped_case(label="scaled", top="integer_mac_scaled_harness", sources=SCALED_SOURCES,
                             parameters={"W_BITS": width, "A_BITS": width, "ACC_BITS": acc,
                                         "OUT_BITS": out, "SCALE_EXP": exponent, "RELU": int(relu)},
                             numerical={"mac_model": "C", "conversion": config.identity()},
                             folder=base / "scaled", liberty=liberty, yosys=args.yosys)
        rows.append({"operand_bits": width, "accumulator_bits": acc, "output_bits": out,
                     "scale_exponent": exponent, "activation": "relu" if relu else "identity",
                     "raw": raw, "scaled": scaled,
                     "support_added_core_area_library_units": round(scaled["core_area_library_units"] - raw["core_area_library_units"], 4),
                     "support_added_core_registers": scaled["core_registers"] - raw["core_registers"],
                     "wrapper_area_change_library_units": round(scaled["wrapper_area_library_units"] - raw["wrapper_area_library_units"], 4),
                     "area_evidence": "unconstrained_ics55_rvt_mapped_synthesis",
                     "timing_ns": None, "fmax_mhz": None, "energy_per_operation_pj": None,
                     "network_quality_attached": False})
        print(f"INT{width}/ACC{acc}/OUT{out}: raw={raw['core_area_library_units']} scaled={scaled['core_area_library_units']}", flush=True)
    summary = {"schema_version": "integer-output-area-1.0.0",
               "pdk": "ics55", "liberty_sha256": sha(liberty),
               "library": "ics55_LLSC_H7CR", "corner": "typ_tt_1p2_25_nldm", "vt": "rvt",
               "driver_sha256": sha(Path(__file__)),
               "flow_driver_sha256": sha(ROOT / "public/pdk_flow/common/run_flow.py"),
               "parser_sha256": sha(ROOT / "public/analysis/hardware/parser.py"),
               "constraints": "unconstrained Yosys/ABC mapping; same source flow and Liberty",
               "accounting": "core includes architectural registers; neutral launch/capture wrapper separate; wrapper output widths differ",
               "rows": rows}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(args.summary)


if __name__ == "__main__":
    main()
