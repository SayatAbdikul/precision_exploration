# Phase 0 contracts

The supplied roadmap makes WP0 a prerequisite for all implementation. These files collect the complete contract content present in the plan and roadmap so every backend, experiment, and RTL block can consume one definition.

Their content is currently **plan-derived**. A source statement described as “frozen” is recorded as such, but project acceptance still requires an owner, review record, and decision-register update.

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

## Phase 0 exit criterion

One canonical experiment configuration uniquely determines the expected numerical result, all generic hardware metrics have unambiguous definitions, and no public code needs unpublished MANT knowledge.
