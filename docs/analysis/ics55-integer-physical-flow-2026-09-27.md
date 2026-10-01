# ICS55 integer MAC physical-flow validation, 2026-09-27

This note records the physical evidence completed in the isolated
hardware-foundation worktree. It is a generic public integer MAC pilot. The
RTL includes a signed full-product multiply, a sequential accumulator,
dot-control and bias addition. It does not include scale application or output
requantization. No network quality result is assigned to these mapped blocks.

## Inputs and reproducibility

The original Phase 1 pilot used ICsprout55 release `v1.10.102`, commit
`68d89edb...`, RVT typical Liberty SHA-256 `15af0dce...`. That release payload
is no longer present in the original project's cache on this host. The
current measurements use an **independently identified** public ICS55 PDK
checkout at commit `c0a4ee0123a4c04812571dd71c3922d9b17ed5c0`, with
standard-cell package `ics55_LLSC_H7C_V1p10C100`, RVT library
`ics55_LLSC_H7CR_typ_tt_1p2_25_nldm.lib`, and corner
`typ_tt_1p2_25_nldm`. Its Liberty SHA-256 is
`38a9f9056a7e4417451e0aaa137efaf0461ee1fc574a8b2ad53622def34bb2b7`;
technology LEF and cell LEF SHA-256 values are in the [machine-readable
summary](../../results/summaries/ics55-integer-validation.json). This is not
a byte-for-byte repeat of the earlier PDK/library result.

The host's Yosys version is `0.53+67` (`4f968c669`). The earlier Phase 1
timing used Yosys `0.65` and OpenSTA `3.1.0`. OpenSTA and OpenROAD are absent
from this host's executable paths. We also inspected the cached `hdlc/sim`,
`patch_image`, and `claude-agent` Docker images with networking disabled; none
exposed `openroad`, `sta`, or `iEDA`. The installed ECC CLI is `0.1.0a5` with
`ecc_tools 0.1.0a7`, but its editable DreamPlace dependency attempts to create
`editable_rebuild.lock` in the external read-only ECC runtime. The exact
task-owned [command](../../artifacts/ppa/ics55-hardware-foundation/ecc-placement-attempt-command.txt)
and [failure log](../../artifacts/ppa/ics55-hardware-foundation/ecc-placement-attempt.log)
show `OSError: [Errno 30] Read-only file system`. ECC's own
[flow state](../../artifacts/ppa/ics55-hardware-foundation/smoke-ecc/runs/default/home/flow.json)
records synthesis as `Success` and floorplan, placement, CTS, and routing as
`Unstart`. That initial attempt did not validate placement. A later task-owned
wrapper disabled the editable package's automatic rebuild and completed a
separate ECC physical pilot, described below, without modifying the external
ECC installation.
The compact [host-capability record](../../results/summaries/ics55-host-capability-2026-09-27.json)
retains the executable and Docker inventory, ECC failure hash, and deepest
completed ECC stage.

Reproduce the bounded matrix from the repository root with the current PDK
checkout supplied as read-only inputs:

```bash
python -m public.pdk_flow.ics55.validate_integer \
  --liberty /home/maveric/nursultan/texer.ai/ecc/chipcompiler/thirdparty/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/liberty/ics55_LLSC_H7CR_typ_tt_1p2_25_nldm.lib \
  --tech-lef /home/maveric/nursultan/texer.ai/ecc/chipcompiler/thirdparty/icsprout55-pdk/prtech/techLEF/N551P6M.lef \
  --cells-lef /home/maveric/nursultan/texer.ai/ecc/chipcompiler/thirdparty/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/lef/ics55_LLSC_H7CR.lef
```

The runner uses the existing Yosys flow and hardware report parser. It emits
per-case RTL/config/netlist/library hashes and raw files under
`artifacts/ppa/ics55-integer-validation/`, the compact [JSON
summary](../../results/summaries/ics55-integer-validation.json), and the
[CSV table](../../results/tables/ics55-integer-validation.csv). The source
hashes in that summary are the identities of the specific RTL measured. The
Yosys/ABC mappings are **unconstrained**. The 10, 5, and 2 ns clocks are
selected from the earlier smoke timing pilot for optional post-synthesis STA;
there are no new slack or fmax values at those targets.

The review follow-up made stage status execution-derived. This reproduction
command requests synthesis and supplies PDK LEFs, but requests neither OpenSTA
nor OpenROAD, so the current summary reports `completed` synthesis and
`not_requested` STA/placement. Supplying a timing or placement executable
makes that stage required: a failed or unavailable requested stage produces a
nonzero CLI exit after the summary and CSV are written. A failed placement
pilot preserves the already completed synthesis row. The summary hashes the
driver, common flow, timing Tcl, placement Tcl, and explicit timing constants.

For a small witness before the integer matrix, the existing
`registered_smoke_arithmetic` RTL was mapped through the current ICS55 RVT
Liberty using `public.pdk_flow.common.run_flow`. Its all-in hierarchical
design area is `951.16` Liberty area units, including `646.24` for the
arithmetic DUT and `304.92` for its register wrapper; the netlist/stat/hash
manifest are in [smoke-current](../../artifacts/ppa/ics55-hardware-foundation/smoke-current/run-manifest.json).
This reproduces the **mapped-synthesis stage** of the prior witness on the
present host, not the prior numeric area or timing result.

## Mapped integer results

All values below are **Liberty cell area units**, not verified physical
µm². The DUT area includes its accumulator and dot control registers. The
wrapper area includes neutral input launch and output capture registers.
There is no placement area, clock tree, interconnect, memory, scale, or
conversion area in these mapped-synthesis numbers. Yosys's hierarchical `design.area` is used
for the total; the top module's local area alone excludes the instantiated
DUT in this tool version. The shared parser was corrected and tested for
this distinction.

| W/A, accumulator | DUT area | Wrapper area | Total area | DUT / wrapper registers | Total leaf cells |
|---|---:|---:|---:|---:|---:|
| INT4/INT4, 32 bit | 2006.48 | 574.56 | 2581.04 | 65 / 76 | 1125 |
| INT5/INT5, 32 bit | 2274.72 | 589.68 | 2864.40 | 65 / 78 | 1293 |
| INT6/INT6, 32 bit | 2371.32 | 604.80 | 2976.12 | 65 / 80 | 1337 |
| INT8/INT8, 32 bit | 2782.08 | 635.04 | 3417.12 | 65 / 84 | 1532 |
| INT8/INT8, 64 bit | 5249.16 | 1118.88 | 6368.04 | 129 / 148 | 3016 |

The DUT registers are architectural accumulator/result/valid state. The
wrapper adds launch/capture state. There is no internal arithmetic pipeline
boundary: output delay registers do not shorten the multiply/accumulate
feedback path. The DUT result is registered on the final accepted product;
the external launch-to-captured-result offset is two edges. The input pair
II is one cycle. A K-product dot needs K accepted pair cycles; for continuous
back-to-back K-product dots its throughput expression is `f/K`, not `f` dots/s.
No validated fmax for this five-case matrix exists, so numerical throughput is
missing rather than zero.

## Task-owned ECC INT4 physical pilot

[`ecc_readonly_runtime.py`](../../tools/hardware/ecc_readonly_runtime.py)
turns off the installed scikit-build editable finder's automatic rebuild at
process startup. The already installed ECC/DreamPlace binaries are then read
without writing to the external runtime. A smoke arithmetic run completed all
nine ECC stages. The separate `integer_mac_harness` INT4/ACC32 project used the
same two RTL source files, ICS55 PDK checkout, 100 MHz clock target, and
task-owned `ecc.toml`/filelist. It completed synthesis, floorplan, fanout
repair, DreamPlace placement, CTS, legalization, routing, DRC, and filler;
the ECC CLI exited zero. The compact [pilot summary](../../results/summaries/ics55-ecc-integer4-pilot-v1.json)
records config, RTL, PDK, generated SDC, flow-state, report, DEF and GDS
hashes. Reproduce with the ECC virtual environment and task-owned project:

```bash
python3 -m tools.hardware.prepare_ecc_pilot
/home/maveric/nursultan/texer.ai/ecc/.venv/bin/python -m tools.hardware.ecc_readonly_runtime run \
  --project artifacts/ppa/ics55-hardware-foundation/integer4-ecc-readonly-runtime --plain --overwrite
python3 -m tools.hardware.ics55_ecc_pilot_summary
```

The iEDA placement report gives a **9,605.568 µm² die**, **8,685.6 µm² core**,
and **4,045.72 µm² placed-instance area**. The routed report retains the die
and core dimensions and reports **4,066.44 µm² instance area** after physical
and timing cell changes. ECC's DRC gate reports zero violations for its own
pilot rule deck. The generated SDC defines a 10 ns clock and does not declare
input/output delays; the ECC post-synthesis QOR report lists 6.818 ns setup
WNS and an estimated 314 MHz. These timing numbers belong to the separate ECC
flow and are not a validated 10/5/2 ns OpenSTA matrix, post-route STA result,
or transferable fmax for the Yosys/ABC mapped rows. The route and DRC results
are tool pilot evidence, not foundry signoff. No validated extracted RC,
activity power, or physical SRAM result is claimed.

## Capability boundary

| Stage | Current status | Evidence or exact missing prerequisite |
|---|---|---|
| RTL elaboration and ICS55 RVT mapped synthesis | Completed | Five hashed integer netlists and the smoke witness. |
| Post-synthesis STA | One ECC QOR report; five-case matrix unvalidated | ECC reports one 10 ns INT4/ACC32 setup estimate. No OpenSTA executable or 10/5/2 ns integer matrix on this host. |
| Placement | ECC INT4/ACC32 pilot completed | DreamPlace produced a task-owned placed DEF and iEDA area report. |
| CTS | ECC INT4/ACC32 pilot completed | ECC flow state and output hashes recorded. |
| Routing and DRC | ECC INT4/ACC32 pilot completed | Routed DEF/GDS and ECC's zero-count DRC gate recorded; no foundry signoff claim. |
| Parasitic extraction and post-route STA | Unvalidated | Public PDK README lists RC collateral unfinished; no validated extraction or post-route timing. |
| Activity power | Unvalidated | No switching trace or complete timing/physical setup. No energy/op is reported. |
| Physical memory | Unavailable | Public PDK README lists RAMs unfinished; no characterized SRAM/compiler or validated ICS55 OpenRAM result. A standard-cell register array would remain a register array. |

The public PDK README's RAM/RC/PDN/verification TODOs remain capability
limits. Decision D9 and final energy/PPA ranking remain open. The next timing
prerequisite is a validated OpenSTA or equivalent matrix under explicit
constraints; post-route timing additionally needs trustworthy RC models. Workload energy
needs switching traces for the exact arithmetic/scaling configuration.
