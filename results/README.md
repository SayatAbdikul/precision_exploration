# Results

## Directory roles

- `raw/` — ignored run-level logs/output not already represented as registered artifacts.
- `databases/` — ignored SQLite run/hardware registries; export compact snapshots when needed for publication.
- `summaries/` — tracked compact numerical, statistical, and hardware summaries with provenance.
- `tables/` — review/publication-ready tables generated from registered evidence.
- `figures/` — review/publication-ready plots generated from registered evidence.
- `candidate_packages/` — versioned public Pareto/guard exports or small manifests pointing to large content-addressed payloads.

## Required numerical results

- FP32 absolute baseline on the identical subset.
- Quantized absolute Top-1/Top-5 or mAP and delta from FP32.
- Paired confidence intervals and disagreement counts.
- Calibration seed, subset, checkpoint, graph, datatype, PTQ, and backend identities.
- Layer statistics, sensitivity, overflow/underflow/sign clipping, and failure diagnosis.
- Screening state and explicit promotion/pruning rationale.

## Required hardware results

- Numerical tuple: weight, activation, accumulator, output, scaling, MAC, workload.
- RTL architecture/pipeline/frequency identity and conformance status.
- PDK/library/corner/VT, tools, constraints, timing harness, and placement seed.
- Area, fmax, latency, II, throughput, leakage/dynamic/clock/total power, energy/op.
- Raw and all-in scaling/conversion cost.
- Ideal/packed/physical memory evidence.
- Generic images/s, images/s/mm², images/J, energy/inference, and measured quality.

## Pareto outputs

Store per-workload and aggregate/worst-case confidence-aware fronts at primitive, MAC/accumulator, all-in datapath, and generic-system levels. Keep potentially Pareto-optimal uncertainty-overlapping points distinct from confidently dominated points.

Do not publish a hardware point without the quality of the identical configuration. Do not replace the multi-objective result with one arbitrary composite score.

## Candidate packages

The D11 export contains approximately 5–10 public Pareto representatives and guard candidates, validated against `public/package/schema/`. It is the final boundary of this repository; downstream MANT evaluation occurs elsewhere.
