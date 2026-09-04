# Phase 5 — Full numerical validation and high-confidence pruning

Scope: **in repository**  
Criticality: **strong numerical conclusions begin here, not at 1k**  
Starts after: **selected Phase 4 sweeps and D5/D6 policies**  
Status: **planned**

## Full quality and robustness tasks

- **N5.1** — Select approximately 20–40 configurations for full ImageNet/COCO validation where practical.
- **N5.2** — Report absolute metrics and deltas from the matching FP32 reference.
- **N5.3** — Repeat calibration seeds for approximately 5–10 finalists.
- **N5.4** — Analyze mean and worst-case degradation across workloads.
- **N5.5** — Freeze exact candidate packages/configurations for generic hardware evaluation.

## Statistical tasks

- **I5.1** — Produce final confidence intervals and paired-disagreement analysis.
- **I5.2** — Produce layer/class sensitivity summaries.
- **I5.3** — Detect calibration instability and model-specific failures.

## Hardware-candidate preparation

- **H5.1** — Translate approximately 8–15 strongest hardware-relevant configurations into exact RTL requirements.
- **H5.2** — Include format decode/encode, scale logic, accumulation, bias, conversion, and requantization—not only multiplication.

## External context only

The roadmap assigns these to a separate MANT repository:

- **M5.1** — Prepare private importer tests using synthetic public packages.
- **M5.2** — Avoid format-specific architectural commitment until public hardware evidence exists.

This repository may validate its exporter/schema, but contains no private importer.

## Decision D7 — Full validation to generic RTL set

Select approximately 8–15 numerical configurations using:

- measured full-dataset quality and confidence;
- robustness across workloads/seeds;
- family representation;
- near-Pareto status and uncertainty buffer;
- complete hardware plausibility, including support/memory cost.

Use fewer only when dominance evidence is strong; use more when needed to preserve a non-obvious hardware winner.

## Outputs

- Publication-quality full numerical result set.
- Robustness and failure/sensitivity analysis.
- Frozen config/manifest/artifact hashes for hardware evaluation.
- D7 RTL-requirement packages for approximately 8–15 candidates.

## Blocks

Generic hardware DSE in Phase 6.
