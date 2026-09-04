# Phase 0 contracts

The supplied roadmap makes WP0 a prerequisite for all implementation. These files collect the complete contract content present in the plan and roadmap so every backend, experiment, and RTL block can consume one definition.

Phase 0 baseline decisions F1–F8 were accepted by the project owner on 2026-09-04. Contract version 1.0.0 was then cross-checked against the schemas and witnesses listed below.

| Contract | Scope | Source roadmap tasks |
|---|---|---|
| `arithmetic.md` | Datatype interface, MAC A/B/C, accumulator, rounding, conversion, order, requantization | N0.1, N0.2 |
| `experiment-a.md` | Controlled static PTQ, graph order, intrinsic scaling, calibration/evaluation separation | N0.3, N0.4, F5, F6 |
| `operator-semantics.md` | Conv/DWConv/Linear, bias, residual, activations, pooling, nonlinear LUTs, strict/practical modes | N0.5 |
| `config-and-identity.md` | Canonical config, deterministic hash, manifests, artifacts, lifecycle, database | I0.1–I0.3 |
| `hardware-metrics.md` | Generic hierarchy, timing harness, PDK/corners, pipeline/frequency, power/memory/Pareto metrics | H0.1–H0.3 |
| `public-private-interface.md` | Versioned public candidate export to a separate private repository | I0.4, M0.1–M0.2 |

## Acceptance checklist

Each contract must record:

- owner and reviewers;
- semantic version;
- status: draft, accepted, superseded;
- acceptance date and linked decision record;
- exact fields left open for later gates;
- conformance tests proving implementations consume the contract;
- migration notes for any backward-incompatible revision.

## Machine-readable artifacts

| Artifact | Location | Phase 0 validation |
|---|---|---|
| Datatype manifest schema | `public/formats/manifests/datatype-manifest.schema.json` | Valid FP6 witness accepted; missing/invalid semantics rejected |
| Experiment config schema | `public/experiments/configs/experiment.schema.json` | Experiment A witness accepted; external scale/native MAC rejected |
| Candidate package schema | `public/package/schema/quantized-model-package.schema.json` | Public witness accepted; private MANT field rejected |
| Canonical identity vector | `tests/conformance/fixtures/canonical/` | Canonical SHA-256 recorded and reproducible |
| Timing harness contract | `public/generic_rtl/harness/README.md` | Accounting, timing, II, provenance, and invalid-point rules defined |

## Cross-contract traceability

| Semantic concept | Contract source | Schema representation | Later consumer |
|---|---|---|---|
| W/A/accumulator/output formats | `arithmetic.md` | Experiment `formats`; datatype manifest refs | Oracle, PTQ, inference, RTL, export |
| Encode/decode/round/overflow/underflow | `arithmetic.md` | Datatype manifest | Oracle, C++, CUDA, RTL conformance |
| MAC/product/accumulator/reduction | `arithmetic.md` | Experiment `arithmetic` | Exact operators, traces, RTL, package |
| Intrinsic/required versus optional scaling | `experiment-a.md` | Manifest `scaling`; experiment `ptq.scaling_policy` | Calibration, inference, scale RTL, analysis |
| Graph/BN/bias/operators | `operator-semantics.md` | Model graph identity plus `operators` and arithmetic bias fields | Graph conversion, exact runtime, package |
| Canonical run/artifact identity | `config-and-identity.md` | Schema versions, hashes, derived ID fields | Scheduler, database, artifact cache |
| Area/timing/power/energy/memory hierarchy | `hardware-metrics.md` | Hardware-run metadata and package evidence | PDK flow, parsers, Pareto analysis |
| External boundary | `public-private-interface.md` | Candidate package schema | Exporter and external importer only |

No semantic default is allowed solely in implementation code; it must resolve through the accepted contract/schema.

## Acceptance record

- Baseline decision approval: project owner, 2026-09-04.
- Contract and schema preparation/cross-check: Codex, 2026-09-04.
- Numerical review: Model C, accumulator, scaling, deployment graph, bias/operator points cross-consistent.
- Reproducibility review: schemas, canonical identity, lifecycle, and invalidation rules cross-consistent.
- Generic hardware review: hierarchy, metric formulas, timing harness, and evidence labels cross-consistent.
- Scope review: no MANT implementation directory; package schema rejects private fields.

## Phase 0 exit criterion

One canonical experiment configuration uniquely determines the expected numerical result, all generic hardware metrics have unambiguous definitions, and no public code needs unpublished MANT knowledge.
