# Statistical, scheduling, and scope-control methodology

## Fixed subsets

For a given model and stage, every candidate sees the same ordered image list. Calibration and evaluation lists are separate and hashed.

| Workload | Calibration | Screening/validation |
|---|---|---|
| ImageNet | 2,000 class-balanced training images, two/class, fixed seed | Fixed stratified 1k, 5k/10k, and full validation lists |
| COCO | Approximately 1,000–2,000 stratified training images | Fixed screening and validation lists with detector-appropriate evaluation |

Repeat calibration with several alternative fixed seeds for approximately 5–10 finalists to test ranking stability.

## Paired classification statistics

For each image `i`:

```text
d_i = I[quantized correct] - I[FP32 correct]
delta_accuracy = (1/N) * sum_i d_i
```

Store all paired outcome counts:

- both correct;
- FP32 correct / quantized wrong;
- FP32 wrong / quantized correct;
- both wrong.

Use paired bootstrap resampling over image IDs with several thousand resamples to estimate confidence intervals for delta Top-1/Top-5. For COCO mAP, use dataset-level bootstrap/resampling appropriate to the evaluation implementation; do not model mAP as Bernoulli accuracy.

## Meaning of the 1k stage

The 1k screen is not a fine-ranking stage. It asks whether a configuration is functional, catastrophic, clearly promising, or uncertain. A few tenths of a percentage point is insufficient evidence for elimination.

Classify each result:

- **PROMISING** — advance;
- **UNCERTAIN** — advance because more data is required;
- **CATASTROPHIC/BROKEN** — diagnose before removing the configuration.

## Three pruning levels

| Level | Example | Elimination standard |
|---|---|---|
| Configuration | FP6 E3M2 + W6A6 + native accumulator + one scale policy | May be removed after diagnosed strong evidence |
| Format | FP6 E3M2 | Requires multiple reasonable configurations |
| Family | FP/minifloat, Posit, log, MX | Very conservative; never from one configuration/model |

Before removing a major family, evaluate at least one reasonable scaling configuration, one reasonable widened accumulator, at least two representative CNNs, and a second PTQ/calibration variant if baseline behavior is unexpectedly poor.

## Promotion funnel

### Stage 0 — sanity

Reject only objectively invalid/broken runs: oracle mismatch, invalid semantics, unsupported invalid-value explosion, constant/all-zero outputs from failure, or obvious near-random collapse. Failure does not automatically eliminate the datatype.

### Stage 1 — fixed 1k

Run all valid candidates. Promote promising and uncertain candidates. Diagnose catastrophic candidates before pruning. There is no hard top-N cutoff.

### Stage 2 — fixed 5k/10k

Promote:

- all promising and uncertainty-overlapping candidates;
- at least one strong representative of every surviving family;
- near-Pareto candidates and candidates inside an uncertainty buffer;
- unusually hardware-efficient candidates;
- important cross-workload specialists;
- representatives that may benefit from W/A asymmetry.

### Stage 3 — full validation

Evaluate approximately 20–40 configurations on full ImageNet/COCO where practical. Report absolute quality, delta from FP32, confidence intervals, mean/worst-case cross-workload degradation, layer/class sensitivity, and calibration-seed robustness.

### Stage 4 — hardware narrowing

| Evidence stage | Approximate set | Purpose |
|---|---:|---|
| Full numerical | 20–40 configurations | High-confidence quality |
| Arithmetic RTL | 8–15 | Real arithmetic area/fmax/power |
| MAC/FPU/generic PE/tile | 4–8 | Include support and memory overhead |
| Detailed public P&R | 2–4 | Final generic physical comparison |
| Export | Approximately 5–10 | Public Pareto representatives plus guards |

Counts are targets, not hard elimination quotas.

## Dominance and uncertainty

Accuracy alone does not safely prune a candidate. B safely dominates A only with convincing no-worse quality, area, energy, and throughput. Preserve a buffer around the apparent Pareto frontier for subset error, calibration variation, later accumulation/scaling optimization, and early hardware-estimate uncertainty.

Build per-workload frontiers first. Then report mean and worst-case loss:

```text
L_max(F) = max_over_models delta_quality(F, model)
```

Classify points as confidently Pareto-optimal, potentially Pareto-optimal/uncertainty-overlapping, or confidently dominated.

## Layer and error diagnostics

Collect per layer:

- min/max/percentiles, mean, variance, kurtosis;
- zero and outlier fractions;
- dynamic range, SQNR, and quantization MSE;
- overflow, underflow, and sign-clipping counts;
- one-layer-at-a-time quality sensitivity.

```text
S_i(F) = quality_FP32 - quality_with_only_layer_i_quantized_to_F
```

Use these diagnostics before labeling a family failure.

## Experiment identity and database

One canonical YAML/JSON config is the run source of truth. Its SHA-256 is the experiment ID. SQLite stores config identity, provenance, lifecycle, metrics, errors, and content-addressed artifact references. Export CSV/Parquet for analysis when needed.

Run lifecycle:

```text
PENDING -> RUNNING -> COMPLETED
                 \-> FAILED / INVALID
```

Writes are atomic; completed runs deduplicate; interrupted incomplete jobs resume/reschedule; failure logs are retained.

## Scheduling

- CPU: multiple independent experiment jobs, bounded by memory bandwidth; benchmark few-many-thread jobs versus many-few-thread jobs.
- Single weak GPU: normally one well-batched experiment at a time.
- Multiple GPUs/machines: distribute complete independent configurations.
- GPU inner parallelism: batch, output channels, output pixels, and independent Conv/GEMM tiles.
- Maintain deterministic per-output reduction order.
- Reuse valid calibration/quantized-weight artifacts when only downstream accumulator/runtime semantics change.

## Cache and artifact policy

Cache quantized weights, calibration ranges, scales/clipping, encoded tensors, truth tables, datatype metadata, layer statistics, and completed outputs. Cache contents are performance aids; authoritative reusable outputs belong in content-addressed artifacts and the database references them by hash.

## Provisional experiment counts

The plan's pilot estimate is:

| Dimension | Count |
|---|---:|
| Numerical formats | Approximately 25; D1 finalizes |
| Initial CNNs | 3 |
| Accumulators | 4 |
| PTQ configurations | 3 |

This gives 900 basic configurations; five workloads would give 1,500. Applying six W/A combinations naively to 900 creates 5,400. Therefore no full Cartesian product is allowed. W/A, signedness, block size, bias/range, and calibration ablations apply only after promotion.

The whole study target is approximately 2,000–5,000 useful inference jobs, tens of full-dataset configurations, and a handful of detailed RTL/P&R points.

## Workload reduction estimate

The plan illustrates:

```text
900 * 1,000 + 150 * 10,000 + 25 * 50,000
= approximately 3.65 million model-image evaluations
```

Running all 900 on all 50,000 ImageNet validation images would be 45 million. The staged funnel reduces the dominant workload by roughly an order of magnitude.
