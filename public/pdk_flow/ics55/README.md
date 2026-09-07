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
