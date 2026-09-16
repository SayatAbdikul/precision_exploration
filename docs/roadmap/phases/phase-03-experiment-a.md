# Phase 3 — Broad controlled Experiment A

Scope: **in repository; public Stage A**  
Starts after: **WP1A workloads plus Phase 2 exact engine**  
Status: **in progress — verified 2026-09-16; 1/100 full screens complete; D4 remains open**

The frozen matrix contains 25 accepted formats across four models (100 initial
configurations). All 100 calibration artifacts, all 100 encoded graphs and four
paired FP32 screen baselines are verified. The runner, paired statistics and sampled layer
diagnostics are implemented. ResNet18 INT8 exposed an INT32 stored-bias overflow;
its versioned INT64 replacement, ResNet18 INT6/5/4 with INT32, and MobileNetV2
INT8/INT64 passed eight-image C++/CUDA acceptance: five accepted configurations.
ResNet18 INT8 completed its full 1k screen and paired analysis: Top-1 68.4%
versus FP32 70.1%, delta -1.7 percentage points (95% paired CI [-3.0025, -0.4]),
with an `UNCERTAIN` retention label. MobileNetV2 INT8 is now screening.
MobileNetV3 INT8/FP64 and revised posit8 completed eight CPU images, as did
MobileNetV2 INT6/5/4; these configurations still need CUDA and acceptance.
YOLO INT8's completed CPU pilot produced
no detections; an eight-image final-store study isolates a calibration-coverage
confound. Log residual rounding studies retain boundary cases and their
downstream effects separately. Eight classifier one-operation studies are complete.
A separate accelerated ResNet18 BFP6 native one-image diagnostic also completed;
it is not canonical screen acceptance. Local bounds cover 4,221 ordinary MACs
and local coverage reaches 5,871 non-MAC nodes; shared MAC bounds and other
precision gates remain open. Hardware/storage priors cover all 100 prepared graphs.
The remaining 99 fixed-1k results or explicit diagnosed statuses, remaining
accumulator acceptance, failure diagnosis and D4 promotion are still required.

See [remaining work](phase-03-remaining-work.md), the
[execution guide](../../architecture/phase3-experiment-a.md), and
[machine-readable progress](../../../results/summaries/phase3-progress.json).

## Numerical screening tasks

- **N3.1** — Begin mainly with uniform W=A to control search size while preserving independent interfaces.
- **N3.2** — Use Model C plus a sufficiently wide family-appropriate accumulator.
- **N3.3** — Use no optional external scale for self-scaling formats.
- **N3.4** — Use required canonical mapping scale for INT/fixed point and intrinsic shared scale for MX/BFP.
- **N3.5** — Run the fixed 1k paired screen for every valid D1 candidate.
- **N3.6** — Collect layer distributions, SQNR, MSE, zero/outlier fractions, overflow/underflow/sign clipping.
- **N3.7** — Run selected one-layer-at-a-time sensitivity studies.

## Statistical/infrastructure tasks

- **I3.1** — Store per-image paired outcomes.
- **I3.2** — Compute paired-bootstrap confidence intervals for delta Top-1/Top-5.
- **I3.3** — Use detector-appropriate resampling for mAP.
- **I3.4** — Classify PROMISING, UNCERTAIN, or CATASTROPHIC/BROKEN.
- **I3.5** — Diagnose catastrophic configurations before removal.

## Early generic hardware estimates

- **H3.1** — Build rough analytical/logic-synthesis priors for major families.
- **H3.2** — Estimate nominal storage bits and metadata overhead.
- **H3.3** — Use estimates only as a preservation signal, never as final hardware claims.

## External context only

The roadmap assigns these to a separate MANT repository:

- **M3.1** — Continue current MANT development independently.
- **M3.2** — Do not use MANT fit to prune public format families.

This repository uses no MANT information in pruning.

## Decision D4 — Promote 1k to 5k/10k

Promote:

- promising candidates;
- statistically uncertain candidates;
- surviving-family representatives;
- near-Pareto candidates;
- candidates expected to have unusual hardware efficiency;
- important cross-workload specialists.

Diagnosed genuinely broken/catastrophic configurations may be pruned. A bad configuration does not eliminate its format or family. There is no hard top-N 1k cutoff.

## Outputs

- Complete fixed-1k results and paired data for all valid candidates.
- Diagnostic and layer-sensitivity evidence.
- Provisional quality/hardware preservation categories.
- D4 promotion list with reasons and confidence.

## Blocks

D4 blocks the deeper numerical branches in Phase 4.
