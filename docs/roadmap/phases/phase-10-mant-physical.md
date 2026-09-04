# Phase 10 — Detailed MANT physical validation

Scope: **external handoff context only; no implementation in this repository**  
Starts after: **external D13/final MANT shortlist**  
Status: **out of repository scope**

This roadmap phase concerns detailed physical validation of 2–4 **MANT candidates**. It is distinct from the detailed P&R of generic public finalists used for D10 in Phases 6–7.

## Source-roadmap physical-design tasks

- **H10.1** — Placement, CTS, routing, and extraction where supported.
- **H10.2** — Multiple placement seeds where practical.
- **H10.3** — RVT baseline with HVT/LVT sensitivity where available.
- **H10.4** — Routed area, timing, registers, and clock-tree contribution.

## Source-roadmap switching-power tasks

- **H10.5** — Generate datatype-specific representative CNN traces.
- **H10.6** — Test convergence across 32→64→128→256 images or equivalent.
- **H10.7** — Use SAIF for average activity and VCD/detail where useful.
- **H10.8** — Use delay-aware/glitch power only if reliable; otherwise disclose limitations.

## Source-roadmap final metrics

- **M10.1** — Images/s, images/s/mm², images/J, and energy/inference.
- **M10.2** — Memory capacity/bandwidth and interconnect/control energy.
- **M10.3** — Leakage energy equals leakage power times inference time.
- **M10.4** — Attach quality to the exact physical configuration.

## Source-roadmap robustness/reproducibility

- **N10.1** — Re-run essential numerical tests against final arithmetic semantics.
- **I10.1** — Freeze manifests, configs, RTL hashes, tool versions, and physical-flow settings.

## Exit criterion

Every final private point has a checkpoint → PTQ → exact inference → RTL → physical implementation → end-to-end quality/energy/throughput evidence chain.

No directory or code for these tasks is created here.
