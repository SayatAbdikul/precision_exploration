# Proposed focused study within three to five weeks of experiment runtime

Scope recommendation superseded by the
[breadth-first proposal](breadth-first-study-proposal-2026-09-25.md), following
the owner's requirement to retain datatype, model and PTQ-method coverage.
This narrower alternative is retained for its conditional runtime calculations.

Status: proposal. Date: 2026-09-25. Hardware: the existing i9-12900K,
RTX 4070 Ti and 32 GB RAM workstation. Assumes continuous operation.
As requested, implementation time is excluded; the experiment clock starts
when the selected protocols, execution paths and hardware designs are runnable.
Calibration, PTQ fitting, verification runs, inference, analysis and hardware
simulation/synthesis/physical runs all consume this runtime budget.

## Scientific scope

A focused, rigorous study is plausible on this workstation. The full original
25-format/four-model campaign, large PTQ matrix, many full-dataset repeats and
complete physical frontier are not the deliverable of this time-limited study.
The earlier months-to-years estimate described that larger program.

Recommended question:

> How do calibration and scale policies change both quantized CNN quality and
> the accumulator/conversion hardware required to deliver that quality?

Focus on ResNet18 and MobileNetV2, INT4/6/8, and three recipes. This uses the
already supported integer graphs and existing controls. Keep the original
eight screens as retained evidence, with six directly used in the new matrix.
INT5 remains useful extra historical evidence.

| Recipe | Purpose |
| --- | --- |
| R0: frozen original MSE recipe | Existing strict control; six 1k results are already complete |
| R1: representative observer with otherwise matched calibration policy | Isolate the sampling/coverage change |
| R2: R1 plus one established reconstruction/rounding method | Test improvement beyond an observer correction |

Choose one reconstruction baseline after a timed reproduction pilot; QDrop or
BRECQ is a reasonable candidate. QDrop explicitly models activation quantization
during reconstruction and provides an official ResNet18/MobileNetV2 implementation:
[paper](https://arxiv.org/abs/2203.05740),
[official implementation](https://github.com/wimh966/QDrop).
An adaptation to our strict graph must be labeled as such. Published results
with different checkpoints/operator exceptions are not direct controls.

All new encoded graphs still need appropriate arithmetic, native, scale and
bias headroom checks. Inference timing assumes that their operators and stores
remain supported by the measured integer path. New unsupported operators or
arithmetic paths invalidate that timing assumption.

The novel contribution, if supported, should be a joint calibration/scale and
provably sufficient accumulator selection method with complete conversion-cost
accounting. Merely repairing sampling or reproducing known PTQ is not a new
top-tier contribution. The hypothesis can fail; publication quality cannot be
guaranteed by a schedule.

## Bounded numerical matrix

Two models × three widths × three recipes = 18 model/width/recipe configurations.
Six R0 controls already have 1k results; twelve configurations are new.

| Work | New model-image evaluations |
| --- | ---: |
| Twelve new configurations at 256 images | 3,072 |
| Extend those twelve to 1k | 8,928 |
| Four selected configurations from 1k to 10k | 36,000 |
| Two additional calibration seeds for two selected configurations, at 1k each | 4,000 |
| **Five-week core total** | **52,000** |

The four confirmations should cover a strong baseline and a candidate on each
model, with policies frozen after the 1k selection stage. Define the practical
quality margin before selection; do not call a candidate competitive merely
because it improves on a weak strict baseline. If no new candidate satisfies
the margin, retain the negative result and investigate the mechanism.

Use the existing nested ImageNet lists. The newly evaluated 9k complement is
confirmatory only after auditing evaluation history and freezing the selection.
Report it separately from the complete 10k result, which includes development
images. The seed repeats at 1k are robustness diagnostics, not the equivalent
of repeated full validation. Do not infer sub-percentage-point superiority
when intervals are too wide.

The three-week version uses 5k instead of 10k confirmations:
12,000 + 4 × 4,000 + 4,000 = **32,000 new evaluations**. It supports a narrower
result, with less precision and a shallower hardware investigation.

## Runtime budget

Use 120–160 aggregate images/hour as a conservative planning range for the
supported integer workloads, subject to a sustained benchmark of the revised
recipes. The existing current-host benchmark measured 162/hour with four workers
and 240/hour with six, but those were small diagnostic samples. Existing full
integer screen receipts also show completed roughly 20–22-hour 1k jobs; this
does not guarantee the same throughput for every new recipe.

At 120–160 aggregate images/hour, 52,000 evaluations require 13.5–18.1 days.
The four long confirmations must be admitted together when resources permit;
small benchmark throughput cannot be extrapolated through an underfilled queue.
No image-level sharding is assumed. Check the longest job as well as aggregate
work in the sustained benchmark.

| Five-week budget item | Allocation |
| --- | ---: |
| Core inference | 13.5–18.1 days |
| Native verification and calibration/proof sanity runs | 1.5 days |
| Calibration collection and PTQ fitting | 2.5 days |
| Bounded W/A, block-rescue and scale/accumulator interventions | 3 days |
| Hardware simulation/synthesis and limited physical runs | 3 days |
| Statistical analysis, failures, reruns and reserve | 5 days |
| **Total** | **28.5–33.1 days** |

The non-inference entries are **allocated caps, not measured completion times**.
Timed pilots must demonstrate that the chosen fitting and hardware experiments
fit those caps. Do not silently truncate a reference method and describe it as
a faithful reproduction. If they do not fit, reduce secondary experiments,
report the reduced scope, or use the five-week maximum; do not drop correctness
checks or necessary comparator evidence.

The three-week variant uses 8.3–11.1 days of core inference plus an 8.5-day
allowance: one day native checks, two fitting, one interventions, 1.5 hardware
synthesis, three analysis/retry reserve. Total **16.8–19.6 days**, conditional
on those caps. It does not promise detailed routed physical evidence.

At only 80 aggregate images/hour, the five-week core alone requires 27.1 days;
the 15-day allowance would push it past five weeks. The throughput condition is
therefore a real admission gate, not an optional optimization target.

## Experimental priorities inside the caps

1. Compare observer coverage with the old recipe, holding other policies fixed.
2. Use W-only/A-only and quantize/rescue block studies to distinguish cumulative
   error from a single sensitive operation. Use fixed small panels and avoid
   attributing causes to raw MSE rankings alone.
3. For the main cases, compare conventional fixed accumulator policies with
   proven sufficient widths; preserve counterexamples and rounding semantics.
4. Use hardware-realizable scale encodings and simulate exactly those coefficients.
   Report complete MAC/bias/requantization cost, not a free high-precision host
   conversion or multiplier-only efficiency.
5. Limit hardware runs to a small set of representative supported datapaths,
   with common synthesis/physical settings and workload activity. Report the
   actual evidence level if routing or SRAM modeling is unavailable.

The required contrast is the proposed mechanism against strong PTQ and hardware
baselines. A large matrix of familiar methods is secondary. Defer broad YOLO,
all-family and full PE/system Pareto claims to the larger program.

## Full-50k validation as a conditional extension

Replacing the four 10k confirmations with full 50k increases the total from
52,000 to **212,000 new evaluations**. At current 120–160 aggregate rates this
is 55–74 days of inference before other experiments, so it does not fit.

Admit this extension only if a faster production implementation is already
available and demonstrated to preserve the claimed semantics. A sustained
800–1,000 aggregate images/hour across the four relevant jobs would put the
212k workload at 8.8–11.0 days of pure inference, subject to job tails and
verification overhead. This is a required performance target, not a measured
capability or a promised optimization result.

A conventional GPU fake-quantization path can support a separately labeled
standard-PTQ study, but must not be presented as the exact sequential finite
accumulator experiment without sufficient equivalence evidence. Matching a
small image subset does not prove graph-wide equality.

For a competitive broad ImageNet comparison, full-50k headline measurements are
preferable. The 10k package is useful for a focused mechanism study with explicit
uncertainty and scope; it is not interchangeable with a fully validated
cross-family frontier. An additional workload or detailed physical finalist is
admitted only after the core experiment budget is secure.

## How this changes the current program

Adoption would dedicate the workstation to this bounded campaign and defer the
remaining broad matrix at saved checkpoints. The current pilot queue was not
changed by this proposal. A new campaign identity and revised deliverables are
needed; the smaller study does not complete the original Phase 3/D4 obligations.

The first experimental day should time a representative new recipe, verify
output/identity handling and assess the most important mechanism on a small
panel. If throughput or scientific differentiation is absent, adjust the claim
and scope before spending the remaining runtime budget.

Recommendation: use the **five-week core** as the budgeted study, seek a
validated faster path before admitting full-50k finals, and prioritize one
defensible new design insight over exhaustive format coverage.
