# Hardware foundation review follow-up

Date: 2026-09-27. Scope: the public hardware foundation in this isolated
worktree. This follows the independent
[`hardware-foundation-review-2026-09-27.md`](/home/maveric/precision_exploration/docs/analysis/hardware-foundation-review-2026-09-27.md).
The original numerical campaign, retained graphs, PDK inputs and experiments
were read only. There is no new inference result or candidate ranking.

## Review findings

| Finding | Correction and evidence |
|---|---|
| Resolution content identity | `requirements.py` now verifies the accepted-format index against the campaign input hash and checks each referenced accumulator-resolution file's raw SHA-256 and model/format identity before reading its accumulator. Tampered bytes and misidentified content fail a focused regression. |
| Physical stage state | The ICS55 driver records execution-derived state for each requested stage and fails its CLI when a requested stage fails; successful earlier-stage evidence remains in the record. Driver, Tcl and constraint identities are recorded. |
| Binary and ternary code maps | The 25-format requirements output now lists every binary/ternary code via accepted `NumberFormat.decode`, including ternary's reserved NaN code. Regression checks the projected codes. |
| Packed-memory edge cases | The layout API checks actual crossings within each packing line, includes bias and accumulator padding in the total padding fraction, and rejects explicit zero block size. The published 50-format and 100-graph tables were regenerated. |
| Verification artifacts | MAC conformance accepts caller-specified artifact and summary paths; unit tests use temporary directories. Regenerated raw-MAC vectors are byte-identical to the reviewed streams. |

The updated [requirements projection](../../results/summaries/hardware-requirements-v1.json)
covers all 25 accepted formats. The raw MAC conformance
[summary](../../results/summaries/integer-mac-conformance-v1.json) retains ten
configurations and 164,489 checked cycles with 27,627 completed dots. The
bounded conversion below changes the integer requirement status only for its
precise supported domain. It does not resolve calibrated scale mapping.

## Exact bounded integer output conversion

[`integer_output_convert.sv`](../../public/generic_rtl/mac/integer_output_convert.sv)
accepts stored, signed INT32 or INT64 accumulator codes and emits signed INT4,
INT5, INT6 or INT8 codes. A compile-time exponent `E` from `-ACC_BITS` through
`OUT_BITS` defines the exact factor `2^E`. The block applies round to nearest
even, saturation to the output format and optional ReLU. RTL elaboration
rejects invalid parameter combinations. The accepted arithmetic oracle's
`encode` supplies expected output codes; unsupported general rational scales
are rejected by the configuration API. The path has one registered edge and
valid/reset behavior.

[`integer_mac_scaled.sv`](../../public/generic_rtl/mac/integer_mac_scaled.sv)
connects the existing sequential, saturating Model C MAC to that conversion.
It converts the accumulator after the stored bias has been added. Its result
is valid one edge after the raw MAC output. The [conformance
summary](../../results/summaries/integer-output-conversion-v1.json) contains
80 stage configurations (24,240 checked cycles, 22,800 output checks) and
three MAC integration configurations (10,328 cycles, 8,590 completed output
checks). It stores
source, accepted-manifest, vector, testbench and configuration hashes. Cases
include signed extrema, RNE ties, left and right shifts, saturation, ReLU,
valid gaps and resets. Reproduce with:

```bash
python3 -m tools.hardware.integer_output_conversion conformance
```

This is a static power-of-two subset. Calibrated external scales, dynamic
scale selection, residual alignment, general activations and exact complete
operator/graph equivalence remain unresolved. No retained network quality
score is attached to these conformance or mapped-area results.

The [ICS55 mapped-area comparison](../../results/summaries/integer-output-conversion-area-v1.json)
uses the same RVT typical Liberty, unconstrained Yosys/ABC flow and neutral
launch/capture structure as the raw MAC pilot. Both raw netlists are
byte-identical to the corresponding earlier INT4/ACC32 and INT8/ACC64
physical-matrix netlists. The core includes architectural registers; the
wrapper is reported separately because the converted output bus is narrower.

| W/A, accumulator, output | Raw core | MAC + conversion core | Added mapped core | Added core registers |
|---|---:|---:|---:|---:|
| INT4, 32 bit, INT4; `E=-1`, identity | 2,006.48 | 2,268.28 | 261.80 | 5 |
| INT8, 64 bit, INT8; `E=-2`, ReLU | 5,249.16 | 5,823.44 | 574.28 | 9 |

Areas are **Liberty area units**. The added core is the difference of fully
mapped hierarchical cores and includes any optimization of the raw MAC in its
new context. It is not the standalone converter module area. The added
register counts are the converter output and valid state. The comparison
supplies no slack, fmax, placed/routed area, energy, or physical-memory cost.
Reproduce with:

```bash
python3 -m tools.hardware.integer_output_synthesis --liberty \
  /home/maveric/nursultan/texer.ai/ecc/chipcompiler/thirdparty/icsprout55-pdk/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/liberty/ics55_LLSC_H7CR_typ_tt_1p2_25_nldm.lib
```

## Comparable tiled memory schedule

The [schedule report](memory-schedule-2026-09-27.md) defines a common tile
loop, 64-bit four-bank layout and 8 GB/s external budget for five retained
layer shapes across all 25 accepted formats and five scenarios. The
[625-row table](../../results/tables/memory-scheduled-layers.csv) separates
activation, weight, output, metadata, bias and optional accumulator-spill
traffic, and counts bank read/write contention and capacity. Under its base
schedule, MobileNetV2 depthwise INT4 is 9.33 external bits/MAC and 6.86
analytical GMAC/s; INT8 is 17.33 bits/MAC and 3.69 GMAC/s. Seven enlarged
depthwise tiles exceed the assumed 8 KiB bank and have null throughput.
These are declared schedule estimates, not measured traffic or physical SRAM
results.

## Reproduction and limits

The focused regression suite includes the five review fixes, conversion,
memory schedule, hardware parser and resource checks:

```bash
/home/maveric/precision_exploration/.venv/bin/python -m pytest -q \
  tests/unit/test_integer_mac_requirements.py \
  tests/unit/test_integer_output_conversion.py \
  tests/unit/test_ics55_integer_stage_status.py \
  tests/unit/test_ics55_ecc_pilot_summary.py \
  tests/unit/test_memory_model.py tests/unit/test_memory_schedule.py \
  tests/unit/test_hardware_parser.py tests/unit/test_phase3_resources.py
```

It passed 68 tests, including 17 requested-stage success/failure combinations. Raw MAC and
conversion conformance were independently regenerated. Test runs use
temporary files; generator runs deliberately refresh task-owned compact
summaries and ignored RTL artifacts. The complete physical evidence and exact
tool/collateral boundary are recorded in the [ICS55 report](ics55-integer-physical-flow-2026-09-27.md).

The remaining public-frontier inputs are exact graph scale mappings and full
operator RTL, matrix-level timing and physical signoff with parasitics, characterized
physical memory, representative switching activity, and workload-level power.
The memory schedule assumes accumulator registers and analytical metadata
widths. There is no final area/energy/quality Pareto ranking.

The task-owned ECC wrapper later completed an INT4/ACC32 harness physical
pilot through placement, CTS, routing, DRC and filler. Its [separate pilot
summary](../../results/summaries/ics55-ecc-integer4-pilot-v1.json) gives a
9,605.568 µm² die and 4,066.44 µm² routed instance area. ECC reported zero
violations in its pilot DRC gate. This is not the five-case OpenSTA timing
matrix or foundry signoff; the [physical report](ics55-integer-physical-flow-2026-09-27.md)
records exact hashes, stages, constraints and the remaining RC/power limits.
