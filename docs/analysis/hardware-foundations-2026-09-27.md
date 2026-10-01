# Public hardware foundations: integer MAC, ICS55 flow, packed memory

The [review follow-up](hardware-foundations-followup-2026-09-27.md) documents
subsequent corrections, the bounded integer output-conversion block, updated
physical attempts, and the comparable layer memory schedule. The evidence
below describes the initial foundation before those additions.

Date: 2026-09-27. Scope: architecture-independent public study. This report
covers three engineering assignments; it is not a candidate selection or a
complete quality–area–energy–throughput frontier. Existing B quality-loss
experiments were read for scope only and were not changed or relaunched.

## Delivered interfaces and evidence

| Assignment | Main implementation | Compact evidence | Reproduce |
|---|---|---|---|
| Numerical requirements and integer MAC | `public/generic_rtl/mac/requirements.py`, `integer_mac.sv`, `integer_mac_harness.sv` | `results/summaries/hardware-requirements-v1.json`, `integer-mac-conformance-v1.json` | `public/generic_rtl/mac/README.md` |
| ICS55 physical validation | `public/pdk_flow/ics55/validate_integer.py`, extended common flow/parser | `results/summaries/ics55-integer-validation.json`, `ics55-host-capability-2026-09-27.json`, `results/tables/ics55-integer-validation.csv` | `docs/analysis/ics55-integer-physical-flow-2026-09-27.md` |
| Packed memory and bandwidth | `public/analysis/memory_model.py`, `tools/analysis/memory_model.py` | `results/summaries/memory-model-1.0.0.json`, `results/tables/memory-*.csv`, `results/figures/memory-effective-bits.svg` | See `docs/analysis/memory-model-2026-09-27.md` |

All newly generated heavyweight RTL vectors, simulation executables, and
mapped netlists live under this task worktree's ignored `artifacts/` directory.
The compact results retain source/configuration hashes for review. The original
project's datasets, graphs, PDK caches, experiments, status files and libraries
were read-only inputs.

Final integration validation: **40 targeted tests passed** across the new MAC
and memory APIs and the existing hardware parser/resource tests. All ten
generated integer variants elaborated; their oracle simulations passed. A
separate hash audit confirmed that the simulated and synthesized RTL hashes
match, all generated variant hashes match their files, the five physical rows
contain no fabricated fmax, and the memory tables contain 25 formats and 100
hash-verified graph configurations. `git diff --check` passed.

## Numerical-to-hardware boundary

The requirements projection covers all **25 accepted datatypes** and records
W/A code widths and mapping, exact Model C product, sequential reduction,
resolved accumulator manifest references where available, accumulator
rounding/overflow, stored bias and its addition point, scale kind/precision,
block policy, output conversion, operator contract, and campaign identity.
Seven mapped-scale formats have no implementation-width specification in the
accepted manifest. The earlier 64-bit mapped-scale storage estimate remains an
analytical placeholder. Four intrinsic shared-scale formats have manifest
8-bit metadata and a 32-value block. Eleven model/format accumulator
resolution references appear in the checked index; unresolved graph cases
remain unresolved in this inventory. The full network identity additionally
requires the exact graph, calibration, scales and operator choices.

The new DUT supports the **raw signed integer Model C dot product at scale 1**,
using a complete W×A product, saturating INT32/INT64 after each sequential add,
and one stored accumulator-width bias addition at the end. W and A widths are
independent. Output is the accumulator code. Requantization, arbitrary mapped
scales, activation, residual alignment, and noninteger family datapaths are
requirements only. No network quality number attaches to the raw MAC or the
mapped-synthesis rows.

Icarus simulation compared the RTL to
`public.inference.reference.arithmetic.model_c` for ten declared
configurations: equal INT4/5/6/8 W/A with both INT32 and INT64, plus INT4×INT8
and INT8×INT4 with INT32. It checked **164,489 cycles and 27,627 completed dot
results**. INT4/5/6 primitive pairs were exhaustive, with boundary and seeded
INT8 pairs. Directed cases cover multi-operation dots, valid gaps, reset,
back-to-back dots, stored-bias RNE ties and saturation, and real INT8/INT32
accumulator overflow. INT64 accumulator overflow is impractical to reach in a
bounded stream; INT64 bias saturation was checked. The neutral wrapper passed
an independent launch-to-capture ordering test. These are dot-level results,
not full-operator equivalence.

The DUT accepts one product per cycle (product II=1). A K-product dot uses K
accepted cycles; consecutive K-product dots can deliver one completed dot
per K cycles. The DUT has a feedback register and an output register but no
arithmetic pipeline stage dividing the product/add/saturation feedback path.
The neutral wrapper launches at edge N and captures the corresponding valid
result after edge N+2. Its registers are excluded from DUT core area.

## Physical evidence boundary

The ICS55 study uses the explicit RVT typical-corner Liberty and standard-cell
LEF from the available local ICS55 checkout, recording its commit and hashes.
The originally pinned Liberty from the prior pilot was absent. Mapped logic
synthesis completed for INT4/5/6/8 with INT32 and INT8 with INT64, each with
the launch/DUT/capture wrapper. Core, wrapper and total areas are:

| W/A | Accumulator | Core | Wrapper | Total |
|---:|---:|---:|---:|---:|
| 4 | 32 | 2006.48 | 574.56 | 2581.04 |
| 5 | 32 | 2274.72 | 589.68 | 2864.40 |
| 6 | 32 | 2371.32 | 604.80 | 2976.12 |
| 8 | 32 | 2782.08 | 635.04 | 3417.12 |
| 8 | 64 | 5249.16 | 1118.88 | 6368.04 |

These values are **Liberty area units**, not a verified square-micrometre
physical area. The clock targets of 10, 5 and 2 ns are intended STA points;
no slack, fmax or measured throughput is supplied without STA.

The available flow did not validate placement, CTS, routing, extraction,
activity power or physical SRAM. Their metric fields are missing, never zero.
The ICS55 capability inventory and exact failed boundary are in the physical
validation report. In particular, a synthesized register array would be a
register array, not an SRAM. Vectorless/cell-only pilot power is preliminary;
this task did not create workload power or energy claims.

## Memory evidence boundary

The packed-memory model separates ideal payload density, word/bank/port-packed
logical allocation and verified physical implementation. Its all-format tables
include 4/5/6/7/8-bit packing, mapped-scale assumptions, intrinsic block
metadata, capacity/padding, accesses, delivered bandwidth and a common
compute/memory roof. Alternative MX/BFP block sizes are labeled hypothetical.
Graph and layer examples reuse retained shape and liveness observations for
ResNet18, MobileNetV2, MobileNetV3-Large and YOLOv8n; these are estimates under
the stated buffer/reuse schedule, not host-memory observations or measured
accelerator traffic. Physical SRAM area, latency and read/write energy are
unavailable, so no physical memory savings are credited.

In the common 32×33, 64-bit-word, two-bank example, INT4 and INT5 both occupy
5.94 logical bits per useful value because each 33-value row needs the same
word allocation after mapping metadata and bank padding. FP6 and FP7 both
occupy 7.76 bits/value, while BFP6 reaches 8.24 bits/value with intrinsic
scale metadata. This is a packing result under one stated layout, not an
area or energy ranking.

## Remaining inputs for the public frontier

- Implement and prove complete scaling, activation/conversion, output storage,
  residual alignment and any family-specific datapaths for an *exact* numerical
  configuration. Sequential rounded reduction must not silently become a tree
  or a one-round exact/Kulisch sum. MX/BFP hardware scale selection must match
  the accepted exhaustive MSE policy or receive a distinct configuration.
- Provide validated ICS55 timing and physical tools/collateral, including
  placement, CTS, routing, extraction and a characterized memory option before
  making routed area or memory-energy claims.
- Provide representative exact-inference activity traces, a validated power
  flow, and complete operator/memory accounting before energy/inference or a
  quality-attached hardware ranking is possible.

D7 finalist selection, D9 physical memory policy, D10 power methodology and
D11 candidate export remain open. The B FP32-QDQ analysis supplies exploratory
quality context and does not establish RTL equivalence.
