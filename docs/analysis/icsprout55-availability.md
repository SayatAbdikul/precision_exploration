# ICsprout55 availability and Phase 1 evidence assessment

Assessment date: **2026-09-07**

Scope: **public, generic precision-exploration hardware-flow foundation only**

## Conclusion

ICsprout55 is available to this repository as a public preview PDK. The Phase 1 RVT pilot therefore uses ICsprout55 directly; the SKY130 fallback is intentionally not exercised.

The validated local route is:

```text
generic registered smoke RTL
  -> native Yosys 0.65 mapped synthesis
  -> ICsprout55 v1.10.102 RVT typical Liberty
  -> OpenSTA 3.1.0 in a Linux container
  -> normalized, hashed Phase 1 results
```

This route does not claim placement, routing, extraction, power, SRAM characterization, a candidate arithmetic architecture, or private MANT implementation evidence.

## Repository findings

| Source | Phase 1 finding | Consequence |
|---|---|---|
| [openecos-projects/ecc](https://github.com/openecos-projects/ecc) | ECC is the command-line RTL-to-GDS flow and accepts an `ics55` PDK name/root. Its packaged installer/toolchain targets Linux x86-64. | ECC defines the intended open flow, but its packaged distribution was not run natively on this macOS arm64 host. |
| [openecos-projects/ecos-studio](https://github.com/openecos-projects/ecos-studio) | ECOS Studio is the GUI/project environment and links the public ICsprout55 PDK; ECC remains the automation core. | No GUI dependency is needed for reproducible Phase 1 evidence. |
| [openecos-projects/icsprout55-pdk](https://github.com/openecos-projects/icsprout55-pdk) | Apache-2.0 public preview with RVT/HVT/LVT standard-cell Liberty, LEF, GDS and Verilog collateral plus technology LEF. The repository also lists RAM, DRC/LVS, SPICE, PDN and RC collateral as unfinished. | RVT mapping and Liberty STA are defensible; memory, extracted timing, physical verification and power claims are not. |
| [parallaxsw/OpenSTA](https://github.com/parallaxsw/OpenSTA) | Stand-alone STA consumes Liberty, a gate-level netlist and timing constraints. | Provides the post-synthesis timing stage used here without implying routed interconnect timing. |

## Frozen inputs

| Item | Frozen identity |
|---|---|
| ICsprout55 repository | commit `68d89edb47847671e18f9e65d66c0cd883995e05` |
| PDK release | `v1.10.102` |
| RVT Liberty archive | SHA-256 `ef33eec4cd5f617d3dd0073122e556df06560dd65eebf805648640038dedc2b7` |
| RVT typical Liberty | `ics55_LLSC_H7CR_typ_tt_1p2_25_nldm.lib`; SHA-256 `15af0dceceeeb02e174174e7123c7d551551369bf5ad1e8417d3ce8dcbc24fcd` |
| Yosys | `0.65`, git SHA `aec814bdf3071f7e0fd0fbe43f7f711e99d01e24` |
| OpenSTA | `3.1.0`, git SHA prefix `2996e37a3a` |
| RTL | SHA-256 `36b91456236cc5a785f7b8b1b1d2f605be382d19f3a0c14152a7da5908915e3d` |
| Harness | registered input to registered output, latency 2 cycles, initiation interval 1 |

## Pilot results and limits

The two repeated synthesis runs produced identical mapped netlists and Yosys statistics. The all-in registered wrapper plus DUT uses 981.12 Liberty area units and 440 leaf cells, including 35 registers. The combinational DUT accounts for 676.20 Liberty area units and 338 cells; the registered wrapper accounts for 304.92 area units and 102 leaf cells.

The typical-corner post-synthesis timing sweep uses 0.10 ns clock uncertainty, 0.20 ns input/output delay, ideal clocks and a 0.01 pF output load:

| Target | Worst slack | TNS | Status |
|---:|---:|---:|---|
| 10 ns | 7.963998 ns | 0 ns | met |
| 5 ns | 2.963997 ns | 0 ns | met |
| 2 ns | -0.036003 ns | -0.048578 ns | violated |

OpenSTA reports a 2.04 ns minimum clock period and 491.16 MHz fmax for this smoke harness and Liberty model. These are cell-delay-only post-synthesis values. There is no wire-delay, extracted-parasitic, routed-area, leakage, dynamic-power, throughput-per-area or energy-per-operation evidence in Phase 1.

The canonical machine-readable result is `results/summaries/ics55-phase1-pilot.json`; raw generated reports remain under the ignored `artifacts/ppa/` boundary.

## Memory evidence capability matrix

| Evidence level | Availability | What Phase 1 may claim | Limitation |
|---|---|---|---|
| 1 — characterized ICsprout55 SRAM/compiler | unavailable | none | Public PDK TODO explicitly lists RAMs as unfinished. |
| 2 — validated technology-specific generated memory | unavailable | none | No validated ICsprout55 OpenRAM/compiler flow or macros were found. |
| 3 — synthesized local memory/register array | available | RVT standard-cell register implementation can be synthesized and timed | Not an SRAM macro; area, timing and energy do not represent SRAM. |
| 4 — analytical fallback | available | May support later sensitivity analysis if labeled analytical | Cannot be promoted to technology-characterized evidence. |

The strongest defensible Phase 1 memory evidence is level 3. Decision D9 remains open.

## SKY130 fallback rule

SKY130 should be used only if the pinned ICsprout55 standard-cell collateral becomes inaccessible or unusable for the required generic pilot. That condition did not occur: both mapped synthesis and post-synthesis STA completed with reproducible hashes. Any later SKY130 run must remain a separately identified cross-check and must not be scaled into ICsprout55 numbers.
