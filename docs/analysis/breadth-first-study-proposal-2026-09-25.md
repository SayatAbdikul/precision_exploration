# Broad datatype, model and PTQ study within a workstation budget

**Subsequent scope revision:** the owner chose to start the B-led transition.
The [B execution protocol](../architecture/experiment-b-exploration.md) supersedes
this proposal's requirement for all 100 strict-A screens before B. Its runtime
tables describe the earlier A/B allocation and are not an ETA for the new runner.

Status: proposal, 2026-09-25. This replaces the scope recommendation in the
[focused five-week proposal](five-week-study-proposal-2026-09-25.md), following
the owner's clarification that datatype, model and quantization-method breadth
is a primary requirement. It does not change a running experiment, numerical
contract, acceptance record or decision gate.

The target is three to five weeks of experiment execution on the existing
i9-12900K / RTX 4070 Ti / 32 GB workstation, excluding implementation time.
The broad study is conditional on a substantially faster, validated execution
path. Its completion time is not established by current measurements.

## Preserve the scientific coverage

Retain every accepted D1 datatype and all four D3 models. That gives **100
mandatory datatype/model pairs**, before varying the PTQ recipe.

| Datatype group | Accepted candidates | Count |
| --- | --- | ---: |
| Integer and fixed point | INT4/5/6/8; Q1.6 | 5 |
| Scalar floating point | FP8 E4M3FN/E5M2, FP7 E3M3, FP6 E3M2/E2M3, FP5 E2M2, FP4 E2M1 | 7 |
| Block/shared scale | BFP6, MXFP8 E4M3, MXFP6 E3M2, MXFP4 E2M1 | 4 |
| Tapered | Posit(4,0), Posit(6,1), Posit(8,1) | 3 |
| Logarithmic | LOG4/6/8 | 3 |
| Codebook | NF4 | 1 |
| Very low cardinality | Binary and ternary | 2 |
| **Total** | | **25** |

Models: ResNet18, MobileNetV2, MobileNetV3 Large and YOLOv8n. These cover the
project's existing residual classifier, mobile classifiers and detector scope;
they do not establish generality to transformers or language models. Additional
models, widths and encodings are stretch work after the core budget is secure.

The objective is useful coverage, not inflating the configuration count with
duplicate encodings or nominal recipe changes that produce identical graphs.
Different widths and exponent layouts remain distinct candidates.

## Preserve Experiments A and B

Experiment A remains the controlled format-native comparison: all 100
datatype/model pairs, with required/intrinsic scaling, the declared wide
accumulator and no optional external scaling or reconstruction. Every valid A
configuration receives the frozen 1k evaluation. A poor result characterizes
that controlled recipe; it does not establish the best achievable PTQ quality
of the datatype.

Experiment B measures the gains from declared calibration, scaling, rounding
and reconstruction refinements. Budget 100–200 additional compatible recipes,
aiming for at least one meaningful B variant per datatype/model pair. Larger
method sweeps remain selective. Reconstruction requires its own explicit B
policy, as the accepted A contract specifies.

Compare an A/B pair on identical evaluation images, and use a common valid
observer policy when attributing the difference to PTQ. Observer corrections
need versioned controls; comparing old sampling with corrected sampling plus
reconstruction does not isolate the reconstruction benefit. Numerical execution
semantics must also match for that attribution.

This clarifies and revises the initial proposal's ambiguous "representative
recipe" allocation: the mandatory 100 1k runs belong to **A**, and selected B
recipes receive **additional** 1k evaluations. A tuned B recipe cannot replace
an A result or close its original Phase 3 obligation.

## PTQ coverage: 200–300 distinct configurations as a target

Use a coverage table indexed by datatype, model and fully specified recipe.
Reserve approximately two to three recipe slots per pair, with the third slot
allocated according to applicability and measured fitting cost. A 300-slot
budget is an illustrative ceiling, not a claim that three published algorithms
already work on every one of the 100 pairs.

Include the following comparisons:

1. The original format-native strict recipe as a historical/control condition.
2. A matched calibration and scale-budget condition, with representative
   observer coverage and an explicit clipping objective. MSE versus a fixed
   percentile or min/max criterion provides an inexpensive method comparison.
3. Adaptive weight rounding and activation-aware reconstruction on a balanced,
   supported subset. AdaRound and QDrop are candidate references, with timed
   reproductions before admitting large sweeps. Use several families and all
   four models where a validated adaptation is available.

[AdaRound](https://proceedings.mlr.press/v119/nagel20a.html) changes weight
rounding; its weight-only result is not evidence for our full W/A graph.
[QDrop](https://arxiv.org/abs/2203.05740) includes activation quantization during
reconstruction. Neither source establishes that our posit, log, shared-scale
or detector adaptations are already supported. Adaptations need distinct names,
correctness checks, and stated optimization budgets. Do not describe an altered
or prematurely stopped method as a faithful reproduction.

The matched condition is a **new experimental policy**. Adding an external
scale to a formerly unscaled float/posit/log candidate changes its graph and
hardware costs. Freeze scale encoding, granularity, block shape and conversion
rules; account for their metadata and runtime work. For MX/BFP, keep intrinsic
scale behavior explicit. Offline weight-scale search and online activation-scale
selection have different costs. Do not silently replace the current intrinsic
MSE scale selector with a cheaper maximum-based selector.

Use the same calibration images, valid observer coverage, graph exceptions,
optimization budget and evaluation panels for comparisons intended to isolate
one factor. Keep original-versus-corrected-observer comparisons separate from
claims about the effect of a rounding method. Repair the known detector-head
sampling confound before interpreting datatype quality. Calibration-set size
is not reduced automatically just because evaluation panels are smaller.

Unsupported or redundant combinations are reported explicitly. They do not
count as completed independent configurations. PTQ failures are scientific
outcomes once implementation correctness is established; they are not evidence
that a datatype is intrinsically unusable.

## Spend more images only where they answer the next question

The following example budgets 300 configurations: 100 A controls and 200 B
recipes, distributed as 75 configurations per model. A smaller compatible
matrix costs less. All extensions reuse
earlier predictions only when graph, backend semantics and dataset identities
are unchanged.

| Stage | Coverage and purpose | Incremental model-image evaluations |
| --- | --- | ---: |
| Broad discovery | Up to 300 configurations, 128 fixed images each | 38,400 |
| Experiment A coverage | Extend all 100 A configurations to 1k | 87,200 |
| Experiment B comparison | Extend 40 B configurations to 1k, ten per model | 34,880 |
| Full confirmation | Six classifiers from 1k to ImageNet 50k; two detectors from 1k to COCO 5k | 302,000 |
| Calibration robustness | Two additional calibration seeds for one selected configuration on each of four models, at 1k each | 8,000 |
| **Full-confirmation example** | **300 small screens, 140 1k results, eight full-dataset results** | **470,480** |

Predeclare the A controls and a coverage-aware B promotion rule before looking
at evaluation results. Every datatype/model pair reaches the A table, including
poor performers. Reserve B promotions for family/model coverage, informative
method contrasts and uncertain cases; 40 is an allocation, not a hard quality
cutoff. A 60-B-run alternative adds 17,440 evaluations, for **487,920** total.
The eight final configurations should provide one same-format A/B comparison
on each model. Both members must be among the 1k configurations; otherwise add
their missing evaluations to the budget. This eight-run allocation supports
four deep A/B comparisons, not full validation of every datatype or method.

The 128-image stage is a coarse diagnostic screen. It cannot cover all 1,000
ImageNet classes and provides especially weak detector AP evidence for rare
categories. It must not eliminate a family, settle close rankings, or support
headline accuracy claims. Close method comparisons beyond the allocated 40–60
B runs consume extra budget. Reserve such runs inside the intervention cap,
or increase the explicit workload; do not describe all 300 configurations as
fully benchmarked. Counts are cumulative: 128→1k adds 872 images, rather than
another 1,000. The eight completed A screens need no repeat if their identities
remain valid. Calibration/training images are separate from these evaluation
counts; retain the frozen calibration budget unless a separate study changes it.

Choose finalists using the development panel and a fixed rule. Freeze the
policies, then report fresh complementary images separately from the complete
benchmark containing development images. Use paired comparisons and declared
practical margins; handle repeated looks and multiple claims explicitly.
Seed screens measure calibration sensitivity, not full-dataset repeatability.

A fallback with six 10k classifier confirmations instead of 50k requires
**230,480** evaluations for the 40-B-run allocation, or **247,920** for 60.
It preserves datatype/model breadth but gives weaker
final precision and a less complete standard-benchmark comparison. Full 50k is
the preferred headline target. There is no assumption that either version
guarantees a top-tier paper.

## Required execution speed, not a promised speedup

For the full-confirmation example with 40 B extensions, the totals are 420,360
classifier executions and 50,120 detector executions. The 10k alternative has
180,360 classifier executions and the same detector count. These are gross
budgets: existing eight
1k controls can reduce them only where their identities match the selected
protocol. A new quantizer, observer or accumulation policy prevents such reuse.

Schedule classification and detector batches separately for this calculation:

```text
inference_days = (classifier_evaluations / aggregate_classifier_images_per_hour
                + detector_evaluations / aggregate_detector_images_per_hour) / 24
```

Aggregate throughput already includes concurrency. Check job tails, memory,
startup, scoring and scheduling overhead in sustained benchmarks; do not divide
the result by worker count again or assume one long job is already shardable.

| Package | Required illustrative aggregate rates: classifier / detector | Inference | With 14-day additional allocation |
| --- | --- | ---: | ---: |
| A + 40 B comparisons + 10k classifier finals | 800 / 200 images/hour | 19.8 days | 33.8 days |
| A + 40 B comparisons + full 50k classifier finals | 1,500 / 250 images/hour | 20.0 days | 34.0 days |

At the same rates, extending 60 B recipes instead would increase those totals
to 35.4 and 35.1 days respectively. This additional A/B evidence revises the
initial proposal's approximately 31–32-day scenarios; it is not free.

The additional 14 days allocate four days to calibration/PTQ fitting, three to
native correctness/acceptance experiments, two to causal/method interventions,
two to a limited hardware experiment set and three to retries/analysis/reserve.
These are **budget caps, not measured durations**. Hardware routing and extensive
PTQ reconstruction may exceed them. Timed pilots must validate all major terms;
unused reserve cannot be assumed before those measurements. The table does not
include coding. CPU/GPU overlap is not credited until demonstrated without
lowering the measured inference rates.

Today's small integer benchmark reached about 240 aggregate images/hour, while
recent mixed noninteger pilots are much slower. There is no current-host
production detector throughput measurement supporting 200–250 images/hour.
Even extrapolating 240 classifier/hour and assuming 20 detector/hour gives about
136 days for the 10k version or 177 days for full confirmation, **inference
alone**. Neither the assumed detector rate nor the integer rate is a measured
all-format average. More concurrent workers alone cannot justify the target.

Therefore **five weeks is a performance-gated target; three weeks is not a
credible commitment from current evidence**. Admission should use a representative
timed workload spanning all family paths and models, weighted by the planned
counts, plus PTQ fitting pilots. If the projected time misses the budget, reduce
secondary recipe depth first while preserving all 100 pairs. Do not quietly
claim that the full confirmation package still fits.

## Execution semantics and Phase 3 obligations

A fast implementation has two scientifically different uses:

- A demonstrated equivalent implementation can accelerate the original
  arithmetic experiment. Its proof/acceptance requirements remain in force.
- A GPU quantize/dequantize implementation with ordinary floating-point
  reductions can support a separately defined datatype/PTQ quality study.
  Exact finite-accumulator experiments then investigate selected cases as a
  separate axis. It is not interchangeable with the existing sequential Model C.

Small image-level agreement alone does not prove equivalence. The latter option
may make broad coverage computationally practical, but changes what the broad
results establish. Do not use those predictions to close original Phase 3/D4
acceptance or claim finite-accumulator hardware quality without corresponding
evidence. The remaining native acceptances and frozen original A screens stay
separate obligations unless their governing protocol is explicitly revised.

Keep the exact reference for quantizer boundaries, accumulator/scale analysis,
counterexamples and selected full-graph validation. Use the fast path only after
its claimed semantics and all format/operator coverage pass their checks.
Quantization fitting and inference share one GPU; heavy reconstruction cannot
be treated as free background work. CPU-native verification and hardware tools
also need explicit RAM/core budgets.

## Research outcome and next decision

The broad study should answer whether datatype rankings survive changes in
calibration, quantization method, model architecture and complete arithmetic
cost. The public output should include the entire coverage matrix, failed cases,
uncertainty, scale/conversion overhead and reproducible identities. A new
predictive rule or mechanism connecting those factors would provide a stronger
contribution than configuration count alone. The existing
[FP8 versus INT8 study](https://arxiv.org/abs/2303.17951) already joins format,
quantization and hardware comparisons, so that combination alone is not novel.

Recommended scope: **all 25 datatypes and four models; 200–300 compatible PTQ
configurations in discovery across A and B; all 100 A configurations and 40–60
B configurations at 1k; eight deeper A/B results; selected calibration and
mechanism repeats**.
The next implementation milestone is representative throughput and correctness
validation of the bulk execution path. Only then freeze a three-to-five-week
execution schedule. Running jobs were not changed by this proposal.
