# ICsprout55 integer MAC timing matrix (2026-10-01)

Fills the `timing_status not_requested` gap of `results/tables/ics55-integer-validation.csv` (see `ics55-integer-physical-flow-2026-09-27.md`).

## Tools
- Image: `openroad/orfs:latest`, repository digest `openroad/orfs@sha256:6da005d1c3447799f7401215174196c75ba9ec7e28417624d0c20225f85d594c`.
- The image has no standalone `sta`. OpenSTA 3.1.0 (`sta::version`) is the engine embedded in `/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad`; `openroad -version` prints only `unknown` (no git describe). The image's Yosys (0.68+post) was not used; synthesis used host Yosys 0.53+67.
- `tools/hardware/opensta_docker.sh` runs openroad in the pinned image as the host user, mounting the repo and PDK (read-only) at identical paths, same cwd, passing the flow's environment variables. Because OpenROAD's `link_design` needs a technology, the shim first reads the tech and cell LEFs, defines a list-based `remove_from_collection` (absent in OpenROAD's Tcl) and then sources the unmodified `opensta_matrix.tcl`. The existing driver was not edited. The wrapper must be passed as an absolute path (the driver changes cwd).

## Command
Run from the repository root `/home/maveric/precision_exploration` (PDK root `/home/maveric/nursultan/texer.ai/ecc/chipcompiler/thirdparty/icsprout55-pdk`):
```
.venv/bin/python -m public.pdk_flow.ics55.validate_integer \
  --liberty /home/maveric/nursultan/texer.ai/ecc/chipcompiler/thirdparty/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/liberty/ics55_LLSC_H7CR_typ_tt_1p2_25_nldm.lib \
  --tech-lef /home/maveric/nursultan/texer.ai/ecc/chipcompiler/thirdparty/icsprout55-pdk/prtech/techLEF/N551P6M.lef \
  --cells-lef /home/maveric/nursultan/texer.ai/ecc/chipcompiler/thirdparty/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/lef/ics55_LLSC_H7CR.lef \
  --opensta /home/maveric/precision_exploration/tools/hardware/opensta_docker.sh \
  --work-dir artifacts/ppa/ics55-integer-timing-v1 \
  --summary results/summaries/ics55-integer-timing-v1.json \
  --table results/tables/ics55-integer-timing-v1.csv
```
Outputs: the summary JSON, CSV, and raw reports `artifacts/ppa/ics55-integer-timing-v1/<case>/opensta-{10,5,2}ns.txt`.

## Results
Area in library units (identical to the earlier validation table for all five rows). Min period = OpenSTA `report_clock_min_period`; fmax = 1000 / min period. Slack in ns (setup, with 0.10 ns uncertainty, 0.20 ns I/O delays).

| Config | Area | Slack 10 ns | Slack 5 ns | Slack 2 ns | Min period (ns) | Fmax (MHz) |
|---|---|---|---|---|---|---|
| INT4/ACC32 | 2581.04 | +2.793602 | -2.206399 | -5.206399 | 7.21 | 138.77 |
| INT5/ACC32 | 2864.40 | +3.073619 | -1.926380 | -4.926380 | 6.93 | 144.38 |
| INT6/ACC32 | 2976.12 | +3.167249 | -1.832750 | -4.832750 | 6.83 | 146.35 |
| INT8/ACC32 | 3417.12 | +2.790950 | -2.209050 | -5.209050 | 7.21 | 138.71 |
| INT8/ACC64 | 6368.04 | -1.926084 | -6.926085 | -9.926085 | 11.93 | 83.85 |

Only 10 ns is met, and only by the four 32-bit-accumulator designs; 5 and 2 ns fail everywhere for these unconstrained netlists (see Limits). Slack is consistent with period minus the period-independent path delay (e.g. INT8/ACC64: 11.93 - 10 = 1.93).
The 64-bit accumulator costs about 4.7 ns of period over INT8/ACC32 in these netlists (inflated by max-transition violations, see Limits). INT5 and INT6 are slightly faster than INT4 and INT8 in this netlist-level estimate, which reflects synthesis mapping choices rather than a trend to rely on.

## Limits (read before using any number above)
Corrections added after independent review (`handoffs/L4-hardware-timing-review.md`); the numbers are unchanged.
- **Synthesis was unconstrained.** Yosys `synth` plus `abc -liberty` ran with no clock target, no sizing and no buffering (`mapping_clock_constraint: null` in the summary JSON). Each configuration is therefore one netlist checked against three clocks, and carries one independent number, its minimum period (period minus slack). The 10, 5 and 2 ns slacks are the same path delay re-expressed; "5 and 2 ns fail everywhere" describes these netlists only, not what timing-driven synthesis could reach.
- **Every reported critical path violates the library max_transition limit** (0.795659 ns), so its delay is extrapolated beyond the characterised NLDM range. Worst slew on the reported critical path: INT4/ACC32 1.04 ns, INT5/ACC32 1.29, INT6/ACC32 0.91, INT8/ACC32 1.29, INT8/ACC64 2.49. Violating pins per netlist (whole design): max slew 101, 129, 67, 90, 967 and max capacitance 4, 3, 2, 4, 12, in the order INT4, INT5, INT6, INT8/ACC32, INT8/ACC64 (reviewer's `review/drv-<case>.log`). In INT8/ACC64 one OAI2BB1X0P5 stage drives 0.116 pF and contributes 1.43 ns. The absolute periods, and part of the "4.7 ns" accumulator cost, therefore come from unbuffered X0P5 drivers and not from logic depth alone. Treat all five minimum periods as not valid timing points; see `ics55-integer-isoclock-2026-10-01.md` for timing-driven netlists.
- **Constraints.** Ideal clock (no clock tree; only 0.10 ns uncertainty), 0.20 ns input and output delays, 0.01 pF load on each of the 65 outputs, no wire parasitics (all drivers unannotated). Liberty corner typical (tt, 1.2 V, 25 C). Mapped netlist only, no placement or routing, no power analysis; fmax is a post-synthesis estimate, not signoff.
- **Wrapper provenance.** The summary hashes `opensta_matrix.tcl` (sha256 `33b1e5c23020f0131bccd5c008d6160ae832a25749ffde98acbba08d84c7309f`) but not the wrapper. The wrapper `tools/hardware/opensta_docker.sh` has sha256 `2e615abc56eb8c8dcc0574c8c194d9c3cedd6a537790822451fb045196efa59c`. Its prelude (two `read_lef` commands and a list-based `remove_from_collection` shim) is part of the Tcl that actually ran. Its `-version` output is a fixed string, correct for the pinned digest.
- **Netlist identity.** The timed netlists are re-synthesised ones, identified by the hashes in `results/summaries/ics55-integer-timing-v1.json` and not by those of the 2026-09-27 validation summary. Netlist bytes depend on the RTL source path; the reviewer synthesised from a copy of the RTL in another directory and got different netlist hashes with identical `yosys-stat.json` and identical worst slack in all five cases.
- Raw reports live under the git-ignored `artifacts/`; only their hashes are in the summary.

## Untouched
The three open findings of `hardware-followup-review-2026-09-27.md` are untouched. No new unit test was added.
