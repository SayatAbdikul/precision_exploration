# ICS55 Phase 1 pilot flow

This directory contains public orchestration and logical configuration. The ICsprout55 preview PDK is public and Apache-2.0 licensed; downloaded release payloads remain in the ignored `cache/` boundary and are supplied as explicit runtime arguments.

The Phase 1 pilot uses `pilot-config.json` and the shared driver:

```text
python -m public.pdk_flow.common.run_flow \
  --config public/pdk_flow/ics55/pilot-config.json \
  --work-dir artifacts/ppa/ics55-pilot \
  --liberty cache/icsprout55-pdk/releases/v1.10.102/ics55_LLSC_H7CR/liberty/ics55_LLSC_H7CR_typ_tt_1p2_25_nldm.lib
```

The validated Phase 1 timing step uses the tracked `public/pdk_flow/common/opensta_pilot.tcl` script with explicit `LIBERTY_PATH`, `NETLIST_PATH`, `TARGET_PERIOD_NS`, and `REPORT_PATH` environment inputs. It was run at 10 ns, 5 ns, and 2 ns using OpenSTA 3.1.0 in a Linux container.

Placement, routing, parasitic extraction, power, and SRAM evidence are unavailable in this validated host setup and are not silently approximated. See `docs/analysis/icsprout55-availability.md` and `results/summaries/ics55-phase1-pilot.json` for identities, hashes, results, and limits. SKY130 remains an unexercised fallback because the ICsprout55 RVT synthesis/STA route completed successfully.

## Integer MAC validation on the current host

Run `python -m public.pdk_flow.ics55.validate_integer --liberty PATH_TO_ICS55_RVT_TT_LIB --tech-lef PATH_TO_ICS55_TECH_LEF --cells-lef PATH_TO_ICS55_RVT_LEF` from the repository root. This runs a bounded INT4/5/6/8 ACC32 plus INT8 ACC64 mapped-synthesis matrix. Supply `--opensta PATH` only with a validated OpenSTA binary for cell-delay timing; supply `--openroad PATH` for preliminary INT4 placement only when the matching LEFs and OpenROAD binary are available. All stages retain distinct evidence labels. The full current-host command, numerical results, and capability limits are in `docs/analysis/ics55-integer-physical-flow-2026-09-27.md`.

A separate task-owned ECC runtime wrapper at `tools/hardware/ecc_readonly_runtime.py`
disables the installed DreamPlace editable package's automatic rebuild, which
otherwise writes a lock in a read-only external checkout. It enabled an
INT4/ACC32 harness pilot through placement, CTS, routing, DRC and filler.
`tools/hardware/ics55_ecc_pilot_summary.py` records its source, PDK, SDC,
flow-state, report, DEF and GDS identities in
`results/summaries/ics55-ecc-integer4-pilot-v1.json`. This ECC result is
separate from the five-case Yosys/ABC mapped-synthesis matrix; post-route STA,
validated RC, power and physical SRAM are still open.
