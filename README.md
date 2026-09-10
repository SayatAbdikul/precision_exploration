# Precision Exploration

Architecture-independent exploration of numerical representations at 8 bits and below for edge CNN inference. The project compares complete numerical architectures—weight format, activation format, accumulator, scaling, signedness, MAC semantics, and generic hardware implementation—under measured quality, area, energy, and throughput.

This repository covers the **public precision-exploration study only**. It does not contain the MANT ISA, compiler, mapping, simulator, tile, RTL, interconnect, or chip implementation. The only MANT-facing output is a versioned candidate package that can be imported by a separate private repository after the public Pareto study.

The two HTML files in `docs/source/` are preserved planning inputs. Their statements are documented as proposals, source-stated baselines, or evidence-gated decisions; they are not silently treated as new user instructions. Where the earlier plan and later roadmap differ, the later roadmap refinement is called out explicitly.

## Research objective

> Which sub-8-bit numerical representation gives the best accuracy–area–energy–throughput trade-off for real edge CNN inference under actual silicon constraints?

The central hypothesis is that the smallest multiplier will not necessarily produce the best complete accelerator. Accumulator width, scaling and conversion, memory density, metadata, communication, control, and realistic switching activity can reorder the arithmetic-only Pareto frontier.

## In-repository scope

- FP32/FP16 reference reproduction and fixed ImageNet/COCO workload manifests.
- Datatype manifests and a high-precision arithmetic oracle.
- Controlled Experiment A and optimized Experiment B PTQ.
- Bit-exact reference, C++, and CUDA inference.
- Paired statistical screening, conservative promotion, and full validation.
- Generic arithmetic, MAC/FPU, scaling/conversion, memory, and PE/tile RTL DSE.
- ICS55 synthesis/P&R as the primary hardware flow, with optional cross-check flows.
- Generic end-to-end, confidence-aware Pareto analysis.
- Versioned export of approximately 5–10 Pareto and guard candidates.
- Public research/reproduction packaging.

## Explicitly out of scope

- MANT-specific ISA and instruction encoding.
- MANT golden model, mapping, compiler, and cycle simulator.
- MANT tile/uFPU, memory organization, interconnect, RTL, floorplan, and whole-chip choice.
- Retraining or quantization-aware training; the supplied plan uses PTQ only.
- Approximate arithmetic in the primary study; it is only an optional later branch.

## Start here

- `docs/analysis/document-synthesis.md` — reconciled analysis of the two inputs.
- `docs/architecture/repository-layout.md` — directory ownership and dependency rules.
- `docs/contracts/README.md` — Phase 0 contract set.
- `docs/methodology/README.md` — complete numerical, statistical, and hardware methods.
- `docs/roadmap/README.md` — work packages, phases, dependencies, and scope boundary.
- `docs/decisions/decision-register.md` — F1–F8 and D1–D14 with evidence requirements.
- `docs/publications/README.md` — public paper and release plan.

## Repository layout

| Path | Purpose |
|---|---|
| `docs/` | Sources, analysis, contracts, decisions, methodology, roadmap, and publication plan |
| `public/` | Architecture-independent numerical software and generic hardware study |
| `data/` | Tracked dataset/checkpoint identities and fixed sample lists; raw datasets remain external |
| `artifacts/` | Reproducible heavyweight outputs addressed by experiment/configuration identity |
| `cache/` | Disposable acceleration data |
| `results/` | Run databases, summaries, plots, tables, and exported candidate packages |
| `tests/` | Primitive conformance through full-pipeline regression |
| `tools/` | Thin setup, run, and report entry points |

## Phase boundary

Phases 0–7 are the core public study and are implemented here. Public research synthesis and release tasks from Phase 11 also belong here. Roadmap Phases 8–10 and the MANT engineering tasks in Phase 11 are documented for handoff context only and must execute in a separate private MANT repository.

## Current status

Phase 0 completed on 2026-09-04. Phase 1 completed and was reverified on 2026-09-08 after correcting the audit findings, with frozen workloads and datasets, 25 accepted datatype manifests, a validated oracle and regenerated exhaustive conformance tables, a tested experiment registry, reproducible FP32 baselines, and a public ICsprout55 RVT synthesis/STA pilot. Phase 2 is in progress: the exact reference/native engine, detector and shared-scale lowering, frozen COCO calibration/workload artifacts, and generic hardware pilots are implemented. The source-frozen gate record is `results/summaries/phase2-final-verification.json`; ImageNet validation recovery is complete and D3 retains EfficientNet as optional; all four models pass native-image C++/CUDA conformance; training calibration recovery, the original YOLO prediction artifact, and D2 counter evidence remain open. Time estimates are in `docs/roadmap/phases/phase-02-remaining-work.md`. See `docs/architecture/exact-engine.md` for execution commands and remaining gates, and `docs/roadmap/phases/phase-01-foundation.md` for the Phase 1 completion record and evidence limits.
