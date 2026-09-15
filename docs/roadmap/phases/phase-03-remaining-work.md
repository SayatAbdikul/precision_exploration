# Phase 3 remaining work

Status: **in progress; no D4 decision and no complete 100-configuration screen.**

| Step | Verified progress | Remaining completion condition |
| --- | --- | --- |
| 1. Freeze definitions | `phase3-screen-v1.json`: 25 formats × four models, uniform W=A=output, strict Model C, fixed datasets and statistics | Keep accumulator revisions explicit and preserve original evidence |
| 2. Calibration and encoded weights | All 100 calibrations and all 100 current encoded graphs verified; shared preparation completed | Coverage complete for the current definitions; preserve evidence across any revisions |
| 3. Wide accumulators | ResNet18 and MobileNetV2 INT8/INT64 accepted; local finite bounds cover all 4,221 ordinary MACs, and current proofs cover 4,999 non-MAC nodes without double counting | Resolve 804 shared MACs and remaining non-MAC/precision gates; accept every remaining runnable graph with native evidence |
| 4. Runner | Paired predictions, per-backend image checkpoints, registry retries, live-worker lock and explicit stale recovery implemented | Exercise remaining workload/family paths and retain complete results |
| 5. Statistics | Paired classification bootstrap; FP32 detector rescored on screen1k; cached image matching reproduces direct COCO resampling on the frozen 1k benchmark | Compute statistics for complete screens with all frozen resamples |
| 6. Diagnostics | Aligned FP32 samples, distributions, SQNR/MSE, zero/outlier fractions and sampled pre-store events implemented | Diagnose actual failures and verify coverage across all workload operators |
| 7. Pilot | ResNet18 and MobileNetV2 INT8/INT64 completed eight matching C++/CUDA images; MobileNetV3 INT8 and revised posit8 plus ResNet18 INT6/5/4 completed eight CPU images; YOLO INT8 completed one CPU image; selected native MAC evidence retained | Remaining native image/family coverage, CUDA comparisons, runtime costs and complete native acceptance |
| 8. Hardware/storage | All 100 prepared graphs have storage, activation liveness, accumulator alternatives and logical arithmetic estimates; family support functions and archived Phase 2 primitives retained | Review assumptions for candidate preservation; no final PPA claims |
| 9. Fixed 1k screen | The accepted ResNet18 INT8/INT64 CUDA screen has started and checkpoints each image | Every valid configuration needs all 1,000 images or an explicit diagnosed status |
| 10. Sensitivity/diagnosis | ResNet18 INT32 bias failure identified; six classifier one-operation studies completed; an eight-image YOLO final-store-only study reproduces zero mAP and exposes frozen calibration sampling omissions | Resolve detector calibration coverage and MobileNetV3's strict pilot loss; separate representation limits from implementation/accumulator failures |
| 11. D4 | Conservative promotion rules implemented without top-N or automatic family elimination | Publish complete evidence, confidence intervals and promotion reasons after remaining gates close |

## Runtime and planning limits

The corrected ResNet18 INT8/INT64 pilot took about **106 seconds per CUDA image**
and **196 seconds per C++ image**, including sampled diagnostics. At that observed
CUDA rate, its 1k screen is approximately **29–30 hours**, plus preparation and
analysis. This is an extrapolation from eight images with some CPU preparation
overlap, not a clean throughput benchmark or a timing estimate for other formats.

The eight MobileNetV3 INT8/FP64 C++ images completed in **900–1,137 seconds each**
including diagnostics, about **2.14 hours total**. The first image was reused
from the earlier one-image pilot. A direct extrapolation is roughly **11.2 days
for a 1k CPU run**; this is not a GPU estimate or authorization to bypass its
pending acceptance gates. Timing includes contention with other work. Its CUDA
pilot still needs measurements. The YOLO INT8/FP64 C++ pilot took **12,059.85
seconds** for one image, an approximately **140-day** extrapolation for 1k CPU
images; this is not a validated production runtime estimate. The revised
MobileNetV3 posit8 C++ eight-image pilot took **299–304 seconds/image**,
approximately **3.5 days** if that rate held over 1k images. MobileNetV2 INT8's
eight paired images take **162–165 seconds on CUDA** and **179–184 seconds on
C++**, approximately **45–46 hours** for a 1k CUDA screen. ResNet18 INT6/5/4 CPU
pilots take **187–195 seconds/image**. These estimates include diagnostics and
possible contention, and do not replace actual screen measurements.

The other families need pilots before assigning credible end-to-end estimates.
Shared-format graph preparation and wide rational execution can be much slower.
The complete matrix cannot be assigned a trustworthy finish date from one pilot.
The current host uses one registered CUDA screen/pilot worker at a time;
one registered CPU-only pilot, sensitivity studies and preparation may overlap
when memory permits. CPU and CUDA jobs have separate worker locks and claim
their specific registry entries; recovery only affects the requested experiment.
Do not prune a format merely because its implementation or
accumulator policy is still pending.

## Retained failures and evidence

The original ResNet18 INT8/INT32 pilot was stopped after its first convolution's
stored-bias requirement reached **7,455,621,761 raw product-domain codes**. This
exceeds signed INT32 even before adding the conservative dot-product bound.
The old registry attempt and saved images remain diagnostic evidence. Backend
agreement did not prove that the accumulator was sufficiently wide.

The INT64 replacement retains the same operand formats, frozen calibration and
canonical scales. Its fast native reduction is admitted only with an exact
bound proving every prefix fits INT32; states are sign-extended and bias is
stored and added in INT64. Unsafe or fractional reductions use the exact
rational implementation. Boundary, cancellation, saturation, fractional,
out-of-range and wide-bias cases are tested against the reference.

The same audit found INT32 mapped-bias headroom failures for MobileNetV2 INT8,
INT6, INT5 and INT4, and MobileNetV3 Large INT8 and INT6. Together with ResNet18
INT8, all seven explicit INT64 candidates were prepared. ResNet18 INT8 and
MobileNetV2 INT8 now have native and whole-graph acceptance. A static inventory using retained FP32 shapes
passes the existing integer proof for all four ResNet18 and all four MobileNetV2
integer configurations. YOLO INT4/5/6 retain unsupported operations. This
inventory (`phase3-integer-readiness.json`) does not replace native pilot gates.
All 84 non-shared configurations and all 16
shared configurations are now prepared. Four independent CPU workers completed
the shared searches using separate report files and
`tools.run.phase3_shared_graphs`, whose persistent cache saves each
finished exhaustive block-scale search. Restarting preparation no longer discards
all scale-search work from an unfinished graph. Operational cache counts are
included in `phase3-progress.json` and do not imply graph completion.
The workers used `--fast-fp32`. The exact integer-grid search retains
all 255 candidate scales and the oracle's tie rule, with fallback outside its
proven FP32 input domain. The retained benchmark compares 256 actual cached
blocks across all four shared formats and records identical responses with
approximately **48–62× search-only speedups**. This does not estimate full graph
preparation or inference time. Evidence is in `phase3-shared-scale-benchmark.json`.

A separate nonlinear audit found fractional-precision failures in the original
integer-accumulator versions of all four MobileNetV3 integer configurations and
YOLO INT8. All five FP64 replacements are now versioned and prepared. Checks
confirm that only the accumulator policy changed: encoded weights, output and
activation scales, calibration and topology are preserved. The 319 local MAC
bounds exclude overflow and remain below 2.33e-9 output steps; 30 selected C++
accumulator states match exact Model C. Further non-MAC checks prove local output
codes for 201 nodes and identify 213 nodes without accumulator rounding. Every
MobileNetV3 non-MAC node and all three YOLO box-decoding nodes are covered.
The box proof includes every finite distance-code pair at each fixed spatial
anchor, staged rounding, stride multiplication and output saturation.
Residual and global-mean proofs compare rounding bounds with output
threshold margins, conditional on identical stored inputs and fixed shapes.
These local checks do not establish MAC output-code stability, native image
conformance or whole-graph acceptance. Evidence is in `phase3-fp64-nonmac.json`.
The three DFL nodes remain pending for acceptance. Targeted checks cover 1,703
vectors per node: every input-code gap with selected high-logit counts and
orders, peak positions, uniform endpoints and seeded mixed vectors. All 81,744
probability codes and 5,109 projected outputs match references using 400/800-digit
exponentials and exact rational projection sums. This is non-exhaustive evidence,
not a proof for all possible 16-logit vectors. The immutable inputs, outputs and
implementation references are in `phase3-dfl-sensitivity.json`.
YOLO INT4/5/6 also lose uniform probabilities
at their frozen output representation boundary, so lack of a difference in that
single probe is not acceptance. Evidence is in
`results/summaries/phase3-nonlinear-accumulator-audit.json`.
Original failing probes are archived in each resolution's immutable evidence;
`phase3-fp64-revisions.json` and `phase3-fp64-mac-checks.json` record the replacements.

Broader first/longest-K C++ MAC checks now cover 73 configurations and 146 layers
across all four models. All 438 selected accumulator states match the reference
before output storage. Finite synthetic activation patterns use actual prepared
weight rows and reduction lengths. This is distinct from native image coverage,
and shared layouts/integer product-domain policies retain their separate checks.
Evidence is in `phase3-family-mac-pilot-<model>.json`.

All 12 current posit configurations also have local proofs for all 603 MAC layers:
products lie exactly on the declared quire grid and every finite-code sequential
partial sum plus stored bias has sufficient headroom. The prescribed bias-store
rounding error is retained separately. Non-MAC arithmetic, nonfinite inputs and
whole-graph native acceptance remain open. Evidence is in
`phase3-fixed-mac-bounds.json`.

The posit non-MAC audit covers 783 of 792 nodes: 231 with local output-code
equality and 552 without accumulator rounding. YOLO's nine DFL nodes remain
pending. Exact pooling ties are handled explicitly, and all finite hard-activation
codes are checked. This found a real failure at 21 MobileNetV3 posit8 hard-swish
nodes: input code 1 became zero under the original 24-fractional-bit policy.
A versioned 64-bit accumulator with 28 fractional bits fixes that counterexample
and passes all 64 MAC/76 non-MAC local checks with preserved operands and
calibration. Six selected C++ states also match the reference. Original failures
remain archived; native image and whole-graph acceptance are pending. See
`phase3-fixed-nonmac.json` and `phase3-posit-revision.json`.

Selected DFL checks now also cover all three posit domains: **2,823 unique
16-logit vectors**, **45,168 probability codes**, and matching projected outputs
against the 400/800-digit references. Each domain is shared by three YOLO DFL
nodes after full attribute/encoding/weight hash comparison. These are deduplicated
checks, not three times as many tested vectors. They exclude NaR and remain
bounded sensitivity evidence, not whole-domain acceptance. See
`phase3-posit-dfl-sensitivity.json`.

The revised MobileNetV3 posit8 eight-image CPU pilot has Top-1 **7/8 versus
FP32 6/8** and Top-5 **8/8 for both**; all 1,120 layer shapes match. It reuses
the verified first image. CUDA and graph acceptance remain pending, and the
result remains `PILOT_ONLY`. MobileNetV2 INT8's accepted paired pilot preserves
the FP32 correctness outcomes: **7/8 Top-1 and 7/8 Top-5**.

ResNet18's CPU-only INT6/5/4 pilots are complete: Top-1 **4/8, 4/8, 1/8** and
Top-5 **8/8, 5/8, 3/8**, respectively, versus FP32 **5/8 and 8/8**. Native CUDA
comparisons remain required. The weak INT4 result motivates last-block studies;
it is not yet a complete-screen catastrophic label or grounds for eliminating
the integer family.

Broader finite FP64 bounds cover **3,133 MACs across 61 configurations**, with
overflow excluded. Two ResNet18 FP8 E5M2 layers have loose conservative
minimum-spacing bounds. All 366 retained selected FP64 native states and the
targeted cases preserve exact-reference output stores. Non-MAC checks also
identify log6/log8 residual threshold sensitivity. Eight-image frequency and
FP32-tail studies find changed rankings but unchanged Top-1/Top-5 correctness
for the selected operation. See the [finite accumulator evidence](../../architecture/phase3-finite-accumulator-evidence.md).

The MobileNetV3 eight-image pilot has Top-1 **2/8 versus FP32 6/8**, and Top-5
**6/8 versus FP32 8/8**. These are pilot results, not a D4 label. Sampled errors
are largest at `add_6`, `add_5` and `features_16_0`; this does not establish which
layer caused the mistakes. Separate eight-image one-MAC studies of
`features_16_0` and `classifier_3` both preserve FP32's 6/8 Top-1 and 8/8 Top-5
correctness. Further one-residual studies of `add_5` and `add_6` also preserve
these outcomes, including explicit branch encoding and alignment. None of the
four selected operations alone reproduces the strict loss; cumulative
error and other layers remain possible causes. See the
[pilot diagnosis](../../architecture/phase3-mobilenet-v3-int8-diagnosis.md).

YOLO INT8's strict one-image pilot produced no detections. A separate final-store
intervention on eight fresh FP32 heads gives mAP50–95 **0 versus 0.432137** for
the same fresh FP32 predictions. The frozen output domain caps coordinates at
26.80057 pixels and severely coarsens score values. The calibration observer
samples only coordinate channel 0 at spatial position 0 and misses coordinate
channels 1–3 in the final head. This is a calibration-coverage confound; verified
cache hashes alone do not prove representative sampling. The experiment does
not alter the strict campaign or eliminate INT8. Details, uncertainty and
reproduction are in the [YOLO diagnosis](../../architecture/phase3-yolo-int8-diagnosis.md).

The resource estimates use retained FP32 shapes for all four models. All
784 ResNet18 and 1,120 MobileNetV3 native layer/image/backend shape comparisons
agree with those observations. The other models still need native checks. Activation liveness
assumes copied views and last-consumer release; accumulator bank sizes are
explicit alternatives, not actual memory or hardware measurements.

The latest full test run passed **401 tests**; its log and XML are
`artifacts/phase3/pytest-finite-fp64.*`; the subsequent propagation intervention
test also passes. The CUDA suite passed **102 tests**,
followed by **10 final INT64 boundary tests** after tightening fast-path admission.
These overlap and must not be added as unique tests. Logs/XML are under
`artifacts/phase3/pytest-*`. Historical Phase 2 results remain archived and are
not presented as verification of the changed Phase 3 Python engine.

The continuation suite adds **25 passing focused tests**, including one-layer
FP32 propagation, interruption recovery, independent preparation inventories,
cached COCO resampling and archived analysis provenance. This overlaps existing
tests and is recorded separately in `artifacts/phase3/pytest-continuation.xml`.
