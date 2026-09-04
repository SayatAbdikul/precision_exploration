# Phase 9 — MANT tile/chip co-design

Scope: **external handoff context only; no implementation in this repository**  
Starts after: **external D12**  
Status: **out of repository scope**

## Source-roadmap tile tasks

- **M9.1** — Implement 4–8 complete tile-level candidates, not only primitives.
- **M9.2** — Evaluate independent storage and accumulator widths if evidence supports them.
- **M9.3** — Evaluate W/A format asymmetry in implementation.
- **M9.4** — Implement required conversion, requantization, and scale paths.
- **M9.5** — Add operator support only where workload evidence requires it.

## Source-roadmap tile PPA tasks

- **M9.6** — Synthesize complete tile variants.
- **M9.7** — Include memory, A/B, control, bus, scale/conversion, and instruction effects.
- **M9.8** — Measure tile area, fmax, power, and energy.

## Source-roadmap whole-chip tasks

- **M9.9** — Iso-area/iso-transistor comparison.
- **M9.10** — Explore tile count, local-memory capacity, and communication bandwidth.
- **M9.11** — Measure full-network throughput and energy/inference.
- **M9.12** — Compare uniform precision first.

## Quality tasks

- **N9.1** — Attach full measured accuracy/mAP to every private hardware point.
- **N9.2** — Confirm private numerical behavior on full or representative datasets where necessary.

## Decision D13 — Engineering objective

Freeze the acceptable quality loss and primary hardware objective only after a complete private Pareto frontier. Energy-first and throughput/mm²-first choices may select different points.

## Decision D14 — Mixed/heterogeneous precision

Consider only after uniform results show strong layer/family specialization or a meaningful Pareto gap. If selected, the external project must account for tile classes, conversions, control, mapping, and compiler complexity.

No directory or code for these tasks is created here.
