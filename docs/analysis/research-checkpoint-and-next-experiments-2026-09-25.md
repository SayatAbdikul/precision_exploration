# Research checkpoint and next experiments

Saved 2026-09-25 after the owner's request to preserve completed work and identify
the additional experiments needed for the research. This records the direction
agreed in the discussion; the successor execution manifest and promotion rules
must be finalized before new quality runs. It does not rewrite historical
experiment identities or declare the original Phase 3/D4 complete.

## Evidence to retain and reuse

| Evidence | State | Valid use |
| --- | --- | --- |
| Exact Experiment A | Eight accepted 1,000-image classifier screens: ResNet18 and MobileNetV2 INT4/5/6/8 | Historical exact controls, paired quality comparisons and diagnostics under their recorded recipe |
| Remaining exact-A native pilots | Saved controller state: five completed, four interrupted, 83 pending | Correctness/acceptance development; not 92 completed quality screens |
| Experiment B original runner | 126 completed 128-image configurations | Declared FP32-QDQ exploration |
| Experiment B extension | 74 completed 128-image configurations | Declared detector/shared-format FP32-QDQ exploration |
| Combined B coverage | 25 datatypes × four models × two recipes = 200 configurations, 25,600 candidate-image evaluations | Broad descriptive development coverage; not exact-A acceptance or final confirmation |
| Proof, failure and review artifacts | Retained, including counterexamples and unresolved gates | Reproducibility, diagnosis and requirements for new implementations |
| Exact performance evidence | Verified concurrency benchmarks and the one-image cProfile diagnostic | Optimization guidance; cProfile elapsed times are instrumented and not ordinary throughput |

The B coverage audit was rerun successfully before saving. The saved archive
also retains existing per-image predictions, calibration records, source
archives, configuration hashes, native libraries, dataset manifests, downloaded
model files and the local data payload. It preserves consistent SQLite copies.
Byte preservation is distinct from re-proving every existing scientific result.

The local checkpoint is
`backups/research-checkpoint-20260925T182113Z/`.
Its `SNAPSHOT.json`, `MANIFEST.jsonl`, `SHA256SUMS` and `README.md` specify archive
coverage, exclusions, verification and restoration. Virtual environments and
reinstallable package caches are omitted; installed-package inventories and
requirements are retained. This is a verified local copy on the same disk,
not an off-device backup. Git history and uncommitted working-tree changes are
both retained, without changing the current checkout or creating a commit.

## Research direction

Preserve breadth across all 25 datatypes, four models and compatible PTQ methods.
Assign an explicit evidence level to every entry. The contribution should test
how datatype, quantization policy, accumulation and complete hardware support
cost interact; a larger table alone does not establish a strong contribution.

There is no scientific requirement to run every strict-A configuration on
exactly 1,000 images. Small panels support exploration; they cannot establish
close rankings, eliminate a whole datatype family, or replace final quality
validation. The existing B protocol still specifies a 1k extension for at least
one recipe per datatype/model pair. Any reduction of that requirement belongs
in the successor protocol, with claim limits and deferred entries recorded.

The completed B runs use FP32 quantize/dequantize computation. Their predictions
remain useful exploratory evidence, but cannot be relabelled as exact low-bit
inference. New central claims about exact arithmetic require the declared
exact backend or a transformation with a justified equivalence guarantee.
Original FP32 checkpoints, offline calibration and FP32 reference baselines
remain separate from the candidate inference arithmetic.

## Work that needs no new quality inference

1. Merge the two B result inventories with exact-A controls in an evidence
   ledger. Preserve the original metrics and source/recipe/backend identities.
2. Analyze paired recipe differences, model dependence, clear failures and
   uncertainty using retained predictions. Report small-panel conclusions as
   descriptive; do not choose a favorable evaluation subset retrospectively.
3. Freeze hypotheses, selection rules, data roles and the next-run matrix.
   Include conventional baselines, every datatype family, all models, uncertain
   cases and plausible hardware specialists. Do not select only current winners.
4. Record calibration/evaluation histories before designating any complementary
   images as unused confirmation data. Keep the required calibration population
   independent of a smaller evaluation panel.

## New experiments, in order

### E0 — Faster exact execution: correctness and performance

First remove repeated graph validation/hashing and constant/weight preparation.
Then accelerate exact residual arithmetic, output conversion and other expensive
host operations. Profile representative noninteger and detector paths before
choosing their kernel work. Keep the reference implementation available.

Validate each admitted implementation with targeted arithmetic boundary cases,
the existing eight-image per-graph native comparison and matching layer codes,
final outputs and required diagnostic events. Eight matching images alone do
not prove a changed reduction order equivalent. Integer tensor-core reductions
require a bound proving the relevant overflow/rounding behavior is preserved;
other cases retain the declared sequential Model C arithmetic.

Measure warm image latency and sustained completed images/hour for supported
worker counts, including setup/checkpoint costs in campaign estimates. Cover
integer, floating/fixed, posit, shared, logarithmic/codebook/very-low-cardinality
representatives and YOLO as the paths become admitted. A 5–15-second integer
classifier inference is an engineering target, not a measured result or an
estimate for other families. Keep profiling runs labelled as diagnostics.

**Output:** bit-exact compatibility evidence, family-specific throughput and a
resource-aware resumable runner. This is the first gate before large exact runs.

### E1 — Matched arithmetic and PTQ comparisons

Use a small family/model-stratified set of configurations, provisionally 8–12,
chosen for coverage and arithmetic risk rather than their observed quality.
Start with native pilots and 32–128 fixed development images; expand only when
needed. Keep graph, stored weights, quantizer rules, scales, observer policy,
image IDs and operator exceptions fixed when isolating backend differences.

Compare the old exact executor with its optimized successor; matching output
codes are required for an execution-only change. If measuring the difference
between exact execution and QDQ, construct the same numerical configuration
for both and report the approximation separately. The existing strict-A versus
current-B result gap mixes recipe and execution changes and does not isolate PTQ.

For claims about PTQ, compare strict/control and improved recipes under the
same declared arithmetic. If intrinsic scales or nonfinite policies differ,
state the additional factor or add a matched control. Missing family admission
remains an explicit block; a surrogate cannot silently fill an exact result.

**Output:** credible matched A/B controls and a measured limit on interpreting
the existing QDQ exploration as guidance for exact inference.

### E2 — Explain the severe losses and test useful quantization changes

Prioritize the unresolved exact-A losses for MobileNetV2 INT4/5/6 and ResNet18
INT4. On a frozen 128–256-image development panel, vary one factor at a time:
calibration/scale policy, quantized W/A precision asymmetry, residual alignment,
or selected layer/block precision. Diagnostic precision changes are separate
ablations, not silent first/last-layer exceptions in the original strict A.

For YOLO, use matched observer/head-domain controls to distinguish calibration
coverage effects from datatype effects. Retain the original failed evidence.

Use the existing maxabs/percentile results first. Add MSE clipping and one
appropriate stronger rounding/reconstruction method on a bounded subset when
they answer the central hypothesis. A modified implementation of a published
method must be labelled as an adaptation. Offline fitting may use FP32; the
candidate inference must still follow its declared quantized arithmetic.

**Output:** causal failure evidence and meaningful PTQ ablations, with the
compute spent on newly informative comparisons rather than duplicate sweeps.

### E3 — Focused quality extensions

Extend selected configurations to 1k and, where uncertainty requires it, larger
development sets. Selection must preserve datatype/family and model coverage,
matched baselines, uncertain cases and plausible hardware specialists.
Configurations that remain at 128 images stay explicitly exploratory.

Reuse an earlier prediction only when the numerical configuration, runtime
semantics and image identity are unchanged. A valid 128→1k extension adds 872
images; changed quantization or arithmetic requires new predictions. Do not
pool QDQ predictions into an exact-A quality estimate.

**Output:** a justified shortlist and a frozen set of central comparisons. The
number promoted follows claim coverage and uncertainty, not a fixed top-N rule.

### E4 — Robustness and final confirmation

Repeat calibration for the main contenders and their matched baselines using
at least three declared seeds initially; add seeds if instability affects the
conclusion. Do not repeat all exploratory configurations by default.

Freeze the final method choices before evaluating confirmation data. For the
selected headline configurations use full ImageNet validation (50k) and COCO
validation (5k) where those benchmark claims are made, reporting the audited
unused complement separately from development images. Use paired inference
results and uncertainty appropriate to the metric and sampling design; close
claims may require more evidence. Dataset sizes are not publication guarantees.

**Output:** final quality, robustness and scope-of-generality evidence.

### E5 — Hardware and transfer evidence

Run numerical-to-hardware checks and representative generic RTL pilots while
the numerical shortlist is being refined, subject to shared workstation
capacity. Account for decoding, accumulation, scale selection, bias,
requantization, memory, and data movement, not only multiplication. Deepen
physical evaluation for a small set of claim-relevant finalists.

Do not label an INT8-accelerated simulation of INT4 values as a measurement of
native INT4 hardware speed. Tie hardware measurements to the same arithmetic
and scale policy used for the quality result. If claiming a transferable
selection rule, test frozen choices on an additional held-out workload rather
than adding every datatype to that workload immediately.

**Output:** evidence for the claimed quality/area/energy/throughput tradeoff and
for any generalization claim beyond the models used to develop the method.

## Scheduling and completion

Immediate next actions are the retained-result analysis and E0/E1 representative
pilots. Large E3/E4 runs follow measured family throughput and a versioned
experiment manifest. Keep the one-command resume requirement, immutable
per-image checkpoints, and explicit campaign/recipe/backend identities.

No new quality sweep was launched while saving this checkpoint. No completed
screen was rerun. Existing source identities and historical acceptance/D4
records remain as saved. The three-to-five-week experimental budget is a target;
it cannot be certified from the fast INT4 benchmark while detector and other
family throughput are unmeasured. The earlier 64/139-day strict-A calculations
were scenarios with assumed detector rates, not measured completion forecasts.

The smallest useful next package is a faster verified exact executor, matched
controls showing whether the existing exploration transfers to exact arithmetic,
and explanations of the major quality losses. That package determines which
larger experiments are worth the remaining compute.
