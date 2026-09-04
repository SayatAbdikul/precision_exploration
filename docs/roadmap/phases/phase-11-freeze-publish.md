# Phase 11 — Freeze and publish

Scope: **public research/reproduction tasks in this repository; MANT engineering freeze external**  
Starts after: **public results stabilize; external MANT freeze depends on external Phase 10**  
Status: **planned/public and out-of-scope/external split**

## External engineering-freeze tasks

The source roadmap lists these for the private MANT repository:

- **M11.1** — Select primary numerical architecture plus optional backup.
- **M11.2** — Freeze storage width, W/A formats, accumulator, MAC, scaling/requantization, and required ISA/tile changes.
- **M11.3** — Update the MANT manual.
- **M11.4** — Freeze compiler/mapping assumptions for v1.

They are documented here but not executed or stored here.

## In-repository research synthesis

- **N11.1** — Produce final public Pareto figures and supported design rules.
- **N11.2** — Explain controlled Experiment A separately from optimized Experiment B.
- **N11.3** — The source asks for generic-versus-MANT agreement/divergence; this repository may incorporate only externally approved aggregate findings, never private implementation details.
- **N11.4** — Derive any new format/accumulation insight supported by measured data.

## In-repository release/reproduction

- **I11.1** — Release public datatype manifests and conformance tests.
- **I11.2** — Release PTQ/calibration and architecture-independent exact inference code.
- **I11.3** — Release generic RTL/PPA scripts and result tables where permitted.
- **I11.4** — Maintain the external private boundary; no MANT source enters this repository.

## Publication choice

- **P11.1** — Default: a public numerical-format paper here and a separate external MANT paper.
- **P11.2** — A combined co-design paper is considered only if architecture–format interaction becomes a central, publishable result and disclosure is separately authorized.

## Final public deliverables

- Source and accepted contract/decision history.
- Reproducible workload/checkpoint/subset manifests.
- Datatype manifests, oracle, and conformance vectors.
- Controlled A and optimized B result sets with full provenance.
- Accumulator, MAC, W/A, scaling, and sensitivity analyses.
- Generic arithmetic/support/memory/PE hardware database and physical evidence.
- Confidence-aware Pareto tables/figures and design rules.
- Pareto-plus-guard external candidate packages.
- Reproduction instructions and limitations.

## Completion condition for this repository

All public claims resolve to tracked configurations and content-addressed evidence; releaseable code/data tables reproduce the published figures; and no MANT-specific implementation is present.
