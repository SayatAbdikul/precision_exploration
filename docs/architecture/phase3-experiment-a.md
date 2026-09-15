# Phase 3 Experiment A execution

The [frozen campaign](../../public/experiments/configs/experiment_a/phase3-screen-v1.json)
contains 100 initial configurations: 25 accepted formats across ResNet18,
MobileNetV2, MobileNetV3 Large and YOLOv8n. Weight, activation and output formats
are uniform. Accumulator candidates require per-graph acceptance; the Phase 2
candidate policy is not blanket approval for a workload graph.

## Evidence and numerical boundaries

Phase 2 sources and summaries are archived under
`artifacts/phase3/phase2-freeze/`. Original baseline predictions and frozen
training/evaluation populations remain hash checked. Phase 3 configurations bind
calibration, all four format roles, source identity, campaign and any versioned
accumulator resolution. Prepared graph and encoded-weight artifacts live under
`artifacts/phase3/configurations/<configuration-sha>/`.

INT uses canonical per-output-channel weight and per-tensor activation scales.
MX/BFP retains intrinsic shared scaling. Self-scaling formats receive no optional
external scale. A larger accumulator does not authorize retraining, bias
correction, reconstruction or a different PTQ scale policy.

## Prepare and inspect

Run from the repository with the frozen `.venv`:

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m tools.run.phase3_prepare
OMP_NUM_THREADS=4 .venv/bin/python -m tools.run.phase3_baselines
OMP_NUM_THREADS=4 .venv/bin/python -m tools.run.phase3_graphs
.venv/bin/python -m tools.analysis.phase3_accumulator_audit --resolve
.venv/bin/python -m tools.analysis.phase3_status
```

Preparation reuses verified artifacts and records per-configuration failures.
Independent preparation workers use distinct `--report` paths under
`artifacts/phase3/preparation/`. Progress, accumulator auditing and storage reports
combine these inventories without concurrent writes to a shared report. This
allows other models to prepare while an expensive shared-format search runs.
`phase3_accumulator_audit --resolve` can add explicit INT64 candidates for proven
INT32 headroom failures; it does not accept those candidates. Re-run graph
preparation for affected configurations after a resolution. Unproved floating,
quire or nonlinear integer cases remain engineering work, not rejected formats.

`tools.analysis.phase3_integer_readiness` applies the existing finite-integer
proof to current graphs and retained FP32 shapes, archiving the implementation
and evidence without issuing acceptance. All four ResNet18 and all four
MobileNetV2 integer configurations pass these static checks. YOLO INT4/5/6 remain
outside the proven operator set. Native shape confirmation, eight-image
C++/CUDA equality and diagnostics are still required; a static pass is not a
replacement for the full-screen gate. See `phase3-integer-readiness.json`.

## Pilot, accept and screen

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m tools.run.phase3_screen \
  --model resnet18 --format int8 --images 8 --backends cuda cpp
```

The CLI prints the job identity and stores a summary in
`artifacts/phase3/runs/<job-sha>/summary.json`. Use its prepared graph and completed
pilot summary with `tools.run.phase3_accept --prepared PATH --pilot PATH`.
Acceptance verifies all eight image records, layer/output agreement, diagnostics,
source identities and concrete integer bounds. It writes a separate
`prepared-accepted.json` without changing the artifact referenced by the pilot.
FP64/quire acceptance still needs a separate sensitivity proof.

Once accepted, the full screen uses the same CLI:

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m tools.run.phase3_screen \
  --model resnet18 --format int8 --scope screen --images 1000 --backends cuda
```

The full-screen gate re-verifies the retained pilot and reproduces its accumulator
proof. Completed native images persist after interruption. Re-run the same
command to retry a failed attempt. If the process died while its registry lease
was RUNNING, append `--recover-stale-seconds 300` only after the lease has stopped
renewing for that interval. Separate CPU and CUDA worker locks prevent duplicate
writers within each resource class. Recovery is scoped to the requested job.
Failures remain in `results/databases/phase3.sqlite`; completed work is
never relabeled as a new numerical configuration.

`tools.run.phase3_finalize_records --job PATH` can repair publication after an
interruption only when every image/backend checkpoint is complete and verified.
It does no inference and cannot supply missing results.

## Paired analysis and diagnostics

Analyze a completed run with
`tools.analysis.phase3_results --summary artifacts/phase3/runs/<job-sha>/summary.json`.
The verifier checks that paired predictions actually match the retained image
records. Pilots receive `PILOT_ONLY`, never a D4 label.

The statistical policy was frozen before ranking: 95% paired percentile intervals,
5,000 classification resamples, 2,000 detector resamples and seed 310911. Metrics
and deltas use fractions. A candidate is PROMISING when all lower delta bounds
are at least -0.01; a primary upper bound below -0.20 requires diagnosis before
CATASTROPHIC/BROKEN classification. Other configurations remain UNCERTAIN.

The detector's FP32 screen1k reference is **mAP50–95 = 0.3927050762** and
**mAP50 = 0.5452584959**. These are recomputed on the frozen 1k population;
the full-5k metric is not used as its paired reference. Bootstrap resampling
clones repeated images and annotation identities for both predictors jointly.
Production detector analysis now caches image-local COCO matching and reruns
official COCO accumulation for each joint draw. It preserves duplicate-image
multiplicity and stable score-tie order. Seven tests cover repeated draws,
ties, crowd/ignore handling and empty predictions. A frozen-1k engineering
comparison reproduced direct evaluation exactly (two resamples; about 31.3 s
direct versus 14.6 s cached on this run). This small benchmark is not quality
evidence; production keeps all 2,000 frozen resamples. Analysis implementation
bytes are archived so later tool edits cannot erase their provenance.

Layer diagnostics align candidate nodes with FP32 node names and native shapes,
sampling up to 4,096 evenly spaced elements per layer/image. Reports include
distributions, MSE/SQNR, zeros/outliers, and sampled actual quantizer range
violations, underflow-to-zero and sign clipping. Intermediate alignment stores
are included and sampling denominators are explicit. These are sampled events,
not exhaustive accumulator-overflow counters; graph bounds and sensitivity
evidence remain separate requirements.

## Selected one-layer sensitivity

`tools.run.phase3_sensitivity --prepared PATH --layers fc conv1 --images 8 --backend cpp`
runs explicit classifier diagnostics. Exactly one selected MAC layer quantizes
its FP32 inputs, uses the prepared candidate weights and declared accumulator,
stores its output in the candidate format, then resumes the folded FP32 graph.
Fresh FP32 predictions and original frozen predictions are both retained. These
studies have `diagnostic_one_layer` scope and cannot enter strict screening or D4
as end-to-end quality results. Detector propagation and non-MAC interventions
remain separate implementation work.

ResNet18 INT4 studies completed eight images each for `fc` and `conv1`. All fresh
FP32 Top-5 lists matched their frozen references. The final-layer intervention
lost three Top-5 hits; the first-convolution intervention lost none. Both lost
one net Top-1 hit. These small-sample observations prioritize further diagnosis;
they do not establish a family ranking or justify pruning.

CPU diagnostic workers use their own lock and may overlap the CUDA screen;
their timings therefore include possible contention. Every image is checkpointed
and failures are retained under `artifacts/phase3/sensitivity/failures/`.

## Completion

`results/summaries/phase3-progress.json` records verified coverage. Storage priors
are generated by `tools.analysis.phase3_hardware_priors`; they are preservation
signals with explicit assumptions, not final area, timing or power results.
Phase 3 closes only after complete screening or diagnosed configuration statuses,
selected sensitivity studies and a D4 promotion list with reasons and confidence.
There is no top-N cutoff and one bad configuration does not eliminate a family.

## Shape, resource and nonlinear accumulator audits

Run `python -m tools.run.phase3_shapes` to retain FP32 tensor shapes on the first
SHA-ordered frozen image of each model. The artifacts identify the image, model
manifest and archived observation implementation. Shape coverage is distinct
from numerical acceptance. `tools.analysis.phase3_shape_check --pilot PATH`
checks these observations against a completed pilot; the ResNet18 pilot agrees
on all 784 comparisons (49 layers × eight images × two backends).

`python -m tools.analysis.phase3_hardware_priors` joins those shapes to verified
prepared graphs. It reports packed payload and scale metadata, live activation
buffers, accumulator storage alternatives, full patch-buffer estimates, explicit
LUT bits and logical arithmetic counts. The schedule assumes fresh output buffers
(including views), sequential node execution and release after the last consumer.
Constants and bias storage remain separate. One serial accumulator and one
accumulator per output are alternative allocation assumptions, not measured
parallelism. Internal working buffers and host allocator overhead are excluded.
Family-specific supporting functions and illustrative decode-ROM sizes are
included; these estimates do not establish area, energy or a hardware ranking.

`python -m tools.analysis.phase3_nonlinear_accumulators` probes the current integer
MobileNetV3 hard activations and YOLO DFL softmax domains. It enumerates every
input code for scalar hard activations and uses an equal-logit counterexample
for softmax. Exact rational operations followed by the candidate output store
provide the comparison. All four MobileNetV3 integer configurations and YOLO
INT8 show stored-output differences with their declared integer accumulators;
widening to INT64 does not remove fractional rounding. FP64 matches these bounded
probes, which is evidence for a precision revision, not whole-graph acceptance.
For YOLO INT4/5/6, the frozen probability output domain itself rounds uniform
1/16 probabilities to zero. That observation does not prove other softmax inputs
safe, nor does it justify eliminating a format. The probe report retains these
distinct causes and leaves active experiment definitions unchanged.

`python -m tools.run.phase3_resolve_nonlinear` reproduces the failing probes and
versions FP64 accumulator candidates for the four MobileNetV3 integer formats
and YOLO INT8. Each revision archives its diagnostic evidence and preserves any
earlier INT64 resolution. `tools.run.phase3_graphs` prepares the active revisions
through the existing graph path. This changes the concrete accumulator throughout
the graph, including MACs; weights, activations, output encodings and calibration
remain identical, as checked by `tools.analysis.phase3_fp64_revisions`.

The FP64 revision report covers 319 MAC layers with conservative local Model C
rounding bounds, including stored bias and subnormal absolute error. Every bound
excludes overflow for the admitted finite integer operands; the largest error
bound is approximately 2.33e-9 of an output quantization step. This is a local
bound for identical stored inputs, not network accuracy or proof against every
output-threshold crossing. The hard-activation and uniform-softmax probes match
the exact-rational stored outputs after revision. Other inputs, operators and
native-resolution behavior still need validation.

`python -m tools.analysis.phase3_fp64_nonmac` checks local output-code stability
for the remaining integer-domain operations. Residual additions use their common
alignment/output grid; global average pooling compares a conservative ordered
FP64 sum-and-division error bound with the nearest possible output threshold.
Exact threshold ties remain pending. Hard activations reuse exhaustive input-code
probes; movement, LUT and exact arithmetic operations are identified separately.
Across the five revisions, 201 nodes have local output-code equality evidence
and 213 do not round an accumulator. All 76 non-MAC nodes in each MobileNetV3
revision are covered. Box decoding is also covered for all three YOLO levels:
the proof propagates FP64 errors through both endpoint branches, center/size
arithmetic and stride multiplication. It compares the bound with exact output
threshold margins for every possible integer distance pair and fixed anchor.
YOLO retains three pending DFL nodes.
`phase3-fp64-nonmac.json` retains graph, shape and implementation references.
These checks assume identical stored inputs and fixed observed shapes; they do
not establish MAC output-code stability, native conformance or graph acceptance.

`python -m tools.analysis.phase3_dfl_sensitivity` compares declared FP64 DFL
arithmetic with 400/800-digit exponential references, prescribed probability
stores and exact rational projection sums. Deterministic cases cover all input
code gaps, selected high-logit counts and orders, peak positions, uniform values
and seeded mixed vectors. All 1,703 vectors at each of three levels matched:
81,744 probability codes and 5,109 projected output codes, with no discrepancy
between reference precisions. Projection rounding is also checked separately on
identical stored probabilities. Inputs, outputs and source references are retained
in `phase3-dfl-sensitivity.json`. The cases are not exhaustive for 16 logits, and
agreement between high precisions is not a transcendental error proof. DFL and
whole-graph acceptance therefore remain open.

`python -m tools.run.phase3_fp64_mac_checks` checks the first and longest-K MAC
of each revision using real weight rows, calibration scales and biases with a
fixed synthetic activation-code pattern. All 30 selected C++ accumulator states
(ten layers) match the reference before output quantization. These checks retain
full reduction lengths but are separate from image pilots and CUDA agreement.

`tools.run.phase3_family_mac_pilot --model MODEL` extends selected-row checks
to non-shared floating/fixed accumulator policies. Across four models, 73
configurations, 146 first/longest-K MAC layers and 438 selected accumulator
states match the C++ and reference paths before output storage. Inputs include
finite extreme codes; actual prepared weights, full reduction lengths and bias
values are retained. This covers FP, fixed operand, posit, logarithmic, NF4,
binary/ternary and the five revised integer-operand FP64 configurations. Shared
block layouts and integer product-domain accumulators need their separate checks.
Reports are `phase3-family-mac-pilot-<model>.json`. Bias application uses the
common Python path; these are not native image, independent bias-kernel, or
precision-sufficiency claims.

`tools.analysis.phase3_fixed_mac_bounds` checks the actual weights of all 12
posit configurations against their declared 64-bit fixed accumulator: 24
fractional bits initially, with the MobileNetV3 posit8 revision below using 28.
All 603 MAC layers have exact product-grid admission and enough
headroom for every finite activation-code sequence, every sequential prefix and
the stored bias. The absolute product bound includes padding. Bias is rounded
once as prescribed; its storage error is reported separately instead of being
called exact. `phase3-fixed-mac-bounds.json` archives the configuration, graph,
per-channel checks and implementation. This establishes no additional MAC-sum
rounding after the prescribed bias store for these finite domains. Nonfinite
activations, non-MAC arithmetic, native image checks and whole-graph acceptance
remain outside this local proof.

`tools.analysis.phase3_fixed_nonmac` adds finite-posit residual, global-pool,
hard-activation and box-decoding checks. Residuals and boxes prove their
intermediate arithmetic lies on the fixed grid with sufficient headroom. Global
pooling bounds all possible sum-lattice means, preserving exact threshold ties
only when they also lie on the accumulator grid; non-ties must stay more than
half an accumulator step from output thresholds. Hard activations enumerate all
finite input codes. Across the current 12 posit configurations, 231 nodes have
local output-code equality evidence and 552 do not round an accumulator. The
nine YOLO DFL nodes remain pending. These checks assume finite stored operands
and the retained shapes; they do not provide native image acceptance.

This audit reproduced a concrete MobileNetV3 posit8 failure: code 1 through
hard-swish rounds to zero with 24 fractional bits, whereas exact arithmetic
stores code 1. The domain appears at 21 nodes. `tools.run.phase3_resolve_posit`
archives those failures and versions `posit8_es1_quire64_f28_accumulator`, keeping
64 storage bits with four additional fractional bits. All 255 finite codes pass
the hard-swish and hard-sigmoid checks after revision, and all 64 MAC/76 non-MAC
checks in that graph pass locally. `tools.analysis.phase3_posit_revision`
verifies that only accumulator attributes/manifests changed at 131 nodes:
weights, input/output encodings, calibration, topology and provenance survive.
Six selected C++ accumulator states match the reference under the new manifest.
The prepared configuration is
`0d08915cb5e6812d39b0267c2271a4eb9d5a7c24687e2e2e8d1d5f04c91b7b91`.
Evidence is in `phase3-posit-revision.json`; original graph and failure evidence
remain immutable. The revision is a candidate, pending native images and
whole-graph acceptance; it does not change the policy for other posit workloads.

The registered runner now admits one CPU-only pilot alongside the single CUDA
worker. Resource-specific locks prevent duplicates, transactional targeted claims
prevent a worker from taking another job, and stale recovery is scoped to the
requested experiment. The MobileNetV3 INT8/FP64 one-image C++ pilot measures the
new wide-accumulator path at native resolution. It does not satisfy the required
eight-image C++/CUDA acceptance gate. Scheduler changes preserve the frozen
numerical engine and screening pipeline identities.

## Durable preparation and pilot extension

Shared-scale preparation now supports checkpoints below the graph level:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m tools.run.phase3_shared_graphs \
  --model resnet18 --formats mxfp6_e3m2 mxfp4_e2m1 --fast-fp32 \
  --report artifacts/phase3/preparation/resnet-shared-remainder.json
```

The wrapper installs a cache only in that preparation process. By default cache
misses call the unchanged exhaustive 255-scale MSE oracle. Each complete response commits
atomically to a SQLite WAL database under `artifacts/phase3/shared-scale-cache/`.
Keys include engine identity, format manifest, exact input values in order and
search options. Reuse verifies request identity and response hash; corrupt or
conflicting entries fail closed. This preserves the smallest-scale tie rule,
partial-block behavior and encoded tensor results. It changes no numerical
engine file, graph definition or running screen identity. The cache is an
operational checkpoint, not accumulator acceptance or a completed graph.

The optional `--fast-fp32` search evaluates the same 255 scales using exact
integer distances on a 2^-149 grid. It admits zero and finite FP32-representable
values with magnitude between 2^-100 and 2^100; other inputs fall back to the
original oracle. This conservative domain also ensures the original oracle's
200-digit Decimal conversion is exact. Ascending scale order preserves ties.
The command requires matching implementation hashes and retained equivalence
evidence from `tools.analysis.phase3_shared_scale_benchmark`: 64 actual cached
preparation blocks for each of four shared formats, 256 identical responses.
The retained benchmark measured approximately 48–62× faster scale searches.
This is a search-only measurement, not an inference or whole-graph speedup.
Preparation archives the benchmark and implementation; existing cache entries
remain reusable because both paths return identical results.

Per-model locks serialize cached preparation workers. Different models may
share committed block results. Progress snapshots are written every 256 blocks
under `artifacts/phase3/preparation-progress/`. After interruption, the same CLI
reuses saved blocks even when the enclosing tensor or graph never finished.

The MobileNetV3 INT8/FP64 C++ pilot completed eight images in 900–1,137 seconds
each, with all 1,120 native layer shapes matching the FP32 observations. Its
Top-1 is 2/8 versus FP32 6/8; Top-5 is 6/8 versus 8/8. Retained paired bootstrap
statistics are marked `PILOT_ONLY`. These results motivate numerical diagnosis
and selected layer studies; they do not make a promotion decision.

`tools.run.phase3_extend_pilot --summary PATH --images 8` seeds a larger pilot
from verified completed evidence. Source and target must share configuration,
graph, baseline, engine and pipeline identities; only pilot size and backend
coverage may grow or change. Source records remain unchanged, target records
retain reuse provenance, and conflicting existing results are rejected. Resource
locks prevent edits to an executing target. The normal screening CLI then skips
saved image/backend work. The MobileNetV3 eight-image C++ pilot uses this path;
CUDA comparison and whole-graph acceptance remain pending.

## Residual sensitivity studies

`tools.run.phase3_residual_sensitivity --prepared PATH --layers add_5 add_6
--images 8 --backend cpp` executes one residual addition in an otherwise FP32
classifier. Both branches are captured before the addition and copied to avoid
later in-place changes. Each uses its declared input encoding; the intervention
then applies the graph's alignment, accumulator and output store. Only the fresh
addition output is replaced before FP32 propagation resumes. Movement/view
operations are excluded. The runner preserves image checkpoints, separate job
identities and archived implementation dependencies; prior one-MAC evidence is
unchanged. Both selected MobileNetV3 residuals preserve FP32 correctness on the
eight diagnostic images, so neither alone explains the strict pilot loss. See
the [diagnosis](phase3-mobilenet-v3-int8-diagnosis.md).

## Detector output-store diagnosis

The YOLO INT8/FP64 C++ one-image pilot completed in about 3 hours 21 minutes,
with all 176 layer shapes matching the FP32 observations, and no detections.
`tools.run.phase3_head_store_sensitivity` isolates its final store on eight fresh
FP32 heads, retains full tensors/codes and recomputes paired COCO statistics.
This final store alone reduces mAP50–95 from 0.432137 to zero on those images.
The frozen per-tensor scale mixes pixel coordinates with probabilities; the
16-position calibration sampler misses three coordinate channels. This is an
explicit calibration-coverage confound, and the original baseline is preserved.
The [YOLO diagnosis](phase3-yolo-int8-diagnosis.md) records the evidence and limits.

`tools.analysis.phase3_posit_dfl` adds selected finite-vector checks for all
three posit detector domains. It tests 2,823 unique vectors and 45,168 probability
codes against 400/800-digit references, with matching projections. Three nodes
share each tested domain, verified through hashes of all attributes and weights.
This is bounded, deduplicated sensitivity coverage and does not close DFL or
whole-graph acceptance.
