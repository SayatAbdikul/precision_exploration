# Phase 0 — Freeze the experiment contracts

Scope: **in repository**  
Criticality: **critical path**  
Status: **planned; source-stated decisions not yet ratified in repository**

## Purpose

Remove semantic ambiguity before large-scale implementation. One experiment configuration must determine one expected numerical result, and public precision-exploration code must remain independent of MANT.

## Numerical/research tasks

- **N0.1** — Write the canonical arithmetic contract for Model C.
- **N0.2** — Define the accumulator-format interface and every conversion/rounding point.
- **N0.3** — Apply the revised Experiment A scaling rule: intrinsic/necessary scaling only; no optional external scale.
- **N0.4** — Freeze deployment-graph order: eval mode → BN folding → calibration → PTQ.
- **N0.5** — Define operator semantics and strict/practical low-precision modes.

Primary locations: `docs/contracts/arithmetic.md`, `experiment-a.md`, `operator-semantics.md`.

## Software/reproducibility tasks

- **I0.1** — Define the canonical experiment configuration schema.
- **I0.2** — Define the datatype-manifest schema.
- **I0.3** — Define content-hash identity rules for runs and artifacts.
- **I0.4** — Define the public candidate package exported to an external private MANT repository.

Primary locations: `docs/contracts/config-and-identity.md`, `public/formats/manifests/`, `public/package/schema/`.

## Generic hardware tasks

- **H0.1** — Freeze area, fmax, II, latency, throughput, leakage, dynamic power, and energy/op definitions.
- **H0.2** — Freeze public hierarchy: primitive → MAC/FPU → scaling/memory → generic PE/tile/system.
- **H0.3** — Define the standard register-to-register timing harness.

Primary locations: `docs/contracts/hardware-metrics.md`, `public/generic_rtl/harness/`.

## External-boundary tasks from the source roadmap

- **M0.1** — MANT ISA, tile, interconnect, mapping/compiler, and golden model remain private and outside this repository.
- **M0.2** — Define only the export/import boundary; do not redesign MANT during the public study.

This repository implements only the exporter/schema side.

## Source-stated baseline decisions in force for the plan

- F1: Experiment A uses Model C.
- F2: Experiment A uses a sufficiently wide family-appropriate accumulator.
- F3: W and A formats are independently configurable.
- F4: accumulator is independently configurable.
- F5: controlled static PTQ, folded BN, fixed subsets, no retraining, no bias correction/reconstruction.
- F6: no optional external scale in Experiment A.
- F7: public study is architecture independent from MANT.
- F8: export Pareto plus guard candidates rather than one winner.

Their repository approval status is tracked separately in `docs/decisions/decision-register.md`.

## Exit criterion

- One canonical config uniquely determines the numerical result.
- All datatype/operator/accumulator/scale semantics are versioned and testable.
- Generic hardware comparison metrics and harness are unambiguous.
- The candidate-package boundary contains no private MANT knowledge.

## Blocks

All workload, oracle, inference, experiment, and hardware implementation.
