# Phase 1 — Build the foundation in parallel

Scope: **in repository**, except the source roadmap's MANT regression tasks  
Starts after: **Phase 0 contracts**  
Parallelism: **high**  
Status: **planned**

## Numerical/workload tasks

- **N1.1** — Freeze checkpoints for ResNet-18, MobileNetV2, MobileNetV3, and one tiny detector.
- **N1.2** — Optionally stage EfficientNet-B0/Lite0 for later D3 inclusion.
- **N1.3** — Reproduce FP32 Top-1/Top-5 and COCO mAP.
- **N1.4** — Freeze preprocessing and dataset versions.
- **N1.5** — Create disjoint ImageNet lists: 2k calibration plus 1k/10k/full evaluation.
- **N1.6** — Create fixed COCO calibration and screening lists.

Required evidence includes checkpoint/file hashes, preprocessing identity, graph version, dataset version, list hashes, and baseline metrics.

## Exact arithmetic and infrastructure tasks

- **I1.1** — Implement the high-precision datatype oracle.
- **I1.2** — Implement manifest parsing and validation.
- **I1.3** — Generate exhaustive ADD/MUL/convert truth tables where feasible.
- **I1.4** — Build scheduler, SQLite schema, artifact cache, and job lifecycle.
- **I1.5** — Implement deterministic config hashing and metadata capture.
- **I1.6** — Store per-image results for paired statistics.

## Generic hardware-flow tasks

- **H1.1** — Bring up ICS55 synthesis/P&R scripts with representative arithmetic blocks.
- **H1.2** — Validate timing harness, corners, and the RVT baseline.
- **H1.3** — Prepare the optional SKY130 cross-check flow if useful.
- **H1.4** — Investigate actual ICS55 SRAM/memory support early and record the available evidence level.
- **H1.5** — Build the standardized synthesis-result parser/database interface.

## External context only

The roadmap assigns these to a separate MANT repository; they are not performed or stored here:

- **M1.1** — Ensure the current simulator/golden-model regression tests pass.
- **M1.2** — Document current numerical assumptions as a compatibility baseline.
- **M1.3** — Do not implement new ISA support yet.

## Decision D1 — Initial candidate manifest set

Evidence required:

- complete definitions and semantic distinctness;
- relevance to ≤8-bit PTQ inference;
- oracle/backend implementability;
- redundancy review;
- expected hardware/research value;
- preservation of every major family.

More candidates increase truth-table, kernel, and screening cost; fewer candidates risk missing a family-level result. Remove only redundant or ill-defined variants.

## Outputs

- Reproducible FP32 baselines.
- Frozen dataset/checkpoint manifests.
- Validated initial format manifests and truth tables.
- Resumable/deduplicating scheduler and database.
- Working generic ICS55 pilot flow and recorded memory capability.

## Blocks

- WP1A workloads and WP1B oracle/infrastructure jointly block Experiment A.
- WP1B plus the operator contract blocks the exact engine.
- WP1C blocks later generic hardware DSE but does not block numerical Experiment A.
