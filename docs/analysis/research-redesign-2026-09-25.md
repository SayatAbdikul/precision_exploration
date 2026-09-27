# Proposed research redesign: numerical formats, PTQ and complete hardware cost

Date: 2026-09-25. Status: **proposal, not an adopted campaign or D4 decision**.
Working assumptions: the primary audience is architecture/EDA, the current
workstation is the initial compute budget, and the public edge-CNN/PTQ scope
continues. The analysis covers the public phases 0–7 and 11; phases 8–10 remain
the separate downstream MANT study. No running experiment was changed.

## Recommendation

Use smaller initial evaluation sets, bring a bounded Experiment B forward,
and allocate expensive exact evaluation to unresolved scientific comparisons.
Keep controlled A and optimized B separately identifiable. A larger format
matrix by itself is insufficient: the paper needs a new, tested explanation or
method for selecting numerical architectures under a quality and hardware budget.

The strongest direction is **joint scale/accumulator/format selection under
complete implementation cost**, supported by controlled-versus-optimized PTQ
comparisons. Novelty is a hypothesis to establish against prior work, not a
claim already demonstrated by this repository.

## 1. What the project has actually established

The accepted [roadmap](../roadmap/README.md), [decisions](../decisions/decision-register.md),
[A contract](../contracts/experiment-a.md), and numerical/statistical/hardware
methodology were reviewed alongside the retained results and current code.
Several overview documents still contain September 16/20 progress snapshots;
the evidence below takes precedence over those historical counts.

| Area | Evidence as of this review | Meaning |
| --- | --- | --- |
| Foundation | Phase 1 verification passed; archived Phase 2 final report says complete with no remaining gates | Strong reproducibility and implementation foundation; does not imply Phase 3 graph acceptance |
| Preparation | 25 formats × four models have calibration and encoded graphs | Prepared artifacts, not 100 measured quality results |
| Accepted full screens | Eight integer classifier configurations, each 1,000 images | Only ResNet18 and MobileNetV2 INT4/5/6/8 have complete current screens |
| Other analyses | 17 retained `pilot` analyses | Includes historical/diagnostic configurations; not 17 additional full screens |
| Remaining acceptance | Development preflight verifies evidence for all 92; 21 classifier graphs have local MAC/non-MAC proofs; noninteger screen acceptance integration is unavailable | Finishing native pilots alone will not release all 92 screens |
| Active pilot batch | Snapshot 10:50 Almaty: four running, 88 pending, zero completed/failed | Live state, not a forecast or acceptance result |
| Hardware | Primitive synthesis/STA/vectorless pilots and all-100 storage/operator priors | No completed all-in, routed, workload-energy Pareto frontier |
| Validation payload | ImageNet 50k payload is explicitly deferred in the dataset manifest | Stage the final dataset before committing to Phase 5 execution |

Sources: [Phase 1 verification](../../results/summaries/phase1-completion-verification.json),
[Phase 2 verification](../../results/summaries/phase2-final-verification.json),
[eight-screen review](../architecture/phase3-eight-integer-review.md),
[development preflight](../../tools/analysis/phase3_development_preflight.py),
[dataset inventory](../../data/manifests/index.json),
[hardware pilot](../../results/summaries/phase2-hardware-pilot.json), and
[power pilot](../../results/summaries/phase2-sta-power-pilot.json).

| Model | INT4 Top-1 | INT5 | INT6 | INT8 | Matching FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 12.3% | 48.3% | 62.5% | 68.4% | 70.1% |
| MobileNetV2 | 0.2% | 6.4% | 46.0% | 70.6% | 72.1% |

These measurements establish the behavior of the strict frozen recipe. They
do not establish the best achievable low-bit integer quality. MobileNetV2
INT4/5/6 and ResNet18 INT4 exceed the severe-loss threshold with their paired
intervals, but the specific causes remain unresolved. Native agreement verifies
implementation consistency; it does not prove that a quantization policy is
representative or competitive.

## 2. Main scientific and execution weaknesses

1. **A single recipe can determine the apparent format ranking.** A permits
   calibrated mapping for integers and dynamic shared scaling for MX/BFP, but
   forbids optional scaling for direct FP/posit/log. This is a legitimate,
   explicitly defined control. It cannot by itself establish intrinsic format
   superiority independent of scaling policy or practical PTQ quality.
2. **Calibration coverage is a demonstrated confound.** The
   [observer](../../public/quantization/calibration/observer.py) samples the same
   16 flat positions in a node on every image. The
   [YOLO diagnosis](../architecture/phase3-yolo-int8-diagnosis.md) shows omitted
   coordinate channels and a final tensor mixing coordinate and probability
   units. More calibration images cannot repair a channel that is never sampled.
   This is a reason to audit classifiers too, not proof that all their losses
   share that cause.
3. **The global D4 barrier delays the informative work.** All 100 strict screens
   currently precede Phase 4. A difficult backend/proof path for one family can
   prevent useful PTQ and hardware experiments for already accepted candidates.
4. **Detailed exact execution is too expensive for the proposed final volume.**
   Six newer low-bit integer screens average roughly 71–80 recorded seconds per
   image, including diagnostics and contention. A 75-second illustrative rate
   makes one 50k run 43.4 worker-days. Smaller initial screens help, but cannot
   make dozens of full runs practical without much faster verified execution,
   additional compute, or a smaller final scope. This is not a wall-clock ETA.
5. **Runtime scale selection must be part of the hardware configuration.**
   [SharedEncoding](../../public/inference/tensor.py) uses `intrinsic_mse_v1`;
   [mse_scale](../../public/quantization/calibration/mse.py) evaluates 255 E8M0
   scales per block semantically. This can be useful as a quality reference.
   It must not be assigned the cost of a cheap maximum/exponent detector unless
   an equivalent implementation is demonstrated. Offline weight-scale search
   and online activation-scale search have very different deployment costs.
6. **Research breadth currently exceeds demonstrated depth.** There are many
   formats and arithmetic structures, but no completed competitive B baseline,
   cross-family full-quality comparison, or complete hardware ranking reversal.
   The next work should close these gaps before expanding the candidate universe.

## 3. Evidence for a smaller first screen

I performed a retrospective study using saved paired outcomes, with no new
inference: 1,000 seeded random panels, nested sizes 128/256/512, sampled without
replacement from the same completed 1k population for every candidate.
The [result artifact](../../artifacts/phase3/reviews/screen-budget/94f257292a77275ed7e4c10a06fa8ea335d2d63f59152bea005ba08ac2e3b289.json)
and [reproduction code](../../tools/analysis/phase3_screen_budget_study.py)
retain input hashes and limitations.

On 256-image panels, the strict INT4 < INT5 < INT6 < INT8 accuracy ordering
held in 99.9% of ResNet18 panels and 100% of MobileNetV2 panels. Gross losses
remained obvious. However, the central 95% of observed panel deltas for
MobileNetV2 INT8 ranged from -3.91 to +0.78 percentage points; for ResNet18 INT8,
from -4.30 to +0.39 points. These are **empirical panel percentiles, not population
confidence intervals**. Eight integer screens cannot validate the same policy
for other families or detector AP.

Recommendation: **256 images for coarse classifier exploration**, 1k for a
common all-class screen, 5k/10k for close comparisons, and full validation for
publication claims. A 256-image panel cannot represent every ImageNet class.
Use a frozen order sampled without inspecting candidate outcomes; do not select
a favorable panel from this retrospective study.

| Stage | Classification images | Detection images | Purpose |
| --- | ---: | ---: | --- |
| Native correctness pilot | Existing eight + targeted arithmetic witnesses | Existing eight + head/operator witnesses | Conformance; not quality ranking or a replacement for proofs |
| Exploration | 256; optional 128 for diagnostic interventions | 256–512 with audited category/size coverage | Large losses, candidate/recipe prioritization; small-set AP is diagnostic |
| Common screen | 1,000, one/class in the existing list | 1,000 fixed images | Common quality comparison and preservation decisions |
| Focused refinement | Nested 5,000/10,000 | Retain 1k development; use a separate training-derived development set if more tuning is necessary | Resolve close choices and freeze policies |
| Final confirmation | Full 50k benchmark plus untouched complement reported separately | Full val5k benchmark plus untouched complement separately | Final paired comparisons, seeds, hardware candidates |

A smaller evaluation set is independent of calibration size. Keep the existing
2k calibration definition as the historical A control. For new B recipes,
compare 256/1,024/2,000 calibration images on a small pilot only after improving
coverage; preserve common data budgets across competing recipes. Reconstruction
iterations, calibration images, and evaluation images must be separate fields.

The current 1k ImageNet list is nested in the 10k list by content hash; the
COCO 1k is nested in val5k. Sample-level reuse is therefore possible, but only
when checkpoint, graph, scales, arithmetic, backend equivalence, preprocessing
and sample identity remain valid. A changed quantizer needs new predictions.

## 4. Move a bounded Experiment B into the discovery stage

Use three explicit comparison arms rather than an uncontrolled format × method
Cartesian product:

| Arm | Policy | Question |
| --- | --- | --- |
| A0 | Existing frozen strict recipe | What happens under the original control? |
| A1 | Matched-budget range/scaling calibration with a common operator regime, where valid | Does the ranking survive comparable range adaptation? |
| B | Best result found under a declared PTQ budget, with practical exceptions separately labeled | What can actually be deployed for the quality target? |

A1 is a proposed new experiment identity, not a redefinition of the old A.
Compare within each arm; do not merge their rankings without labeling the
policy. A1 should include scale-free, power-of-two and general-scale options
only where meaningful, and record scale precision/granularity and support cost.

Start method development on ResNet18 and MobileNetV2 because their existing
screens give useful controls. A compact first matrix is:

| Recipe | First targets | Scope/control |
| --- | --- | --- |
| Existing MSE + nearest rounding | All existing candidates as supported | Historical control |
| Better observer + clipping/range search | INT8/6/4; scaled FP8/6/4 and shared representatives | Separate coverage correction from the format/method change |
| Bias correction and legal cross-layer equalization | INT4/6 and sensitive depthwise blocks | Verify FP32 function preservation; do not apply ReLU scale identities blindly to ReLU6, hard-swish or SE |
| AdaRound | INT4/6 weights, initially with high-precision activations as a diagnostic | Isolate weight-rounding benefit; label W4A32 distinctly from W4A4 |
| BRECQ or QDrop | INT4/6 W/A | Implement one strong reconstruction baseline first |
| PD-Quant | Small competitive subset | Independent stronger PTQ comparator after reproduction |
| Hardware-oriented shared scaling | MXFP8/6/4 and BFP | Compare explicit max/exponent policy with exhaustive MSE quality reference; charge online scale generation |

AdaRound studies data-dependent weight rounding; BRECQ studies block
reconstruction; QDrop incorporates activation quantization during reconstruction;
PD-Quant uses prediction differences. These provide established CNN PTQ
comparators rather than evidence that our own strict low-bit results are wrong.
Published accuracies cannot be compared directly without matching checkpoint,
operator exceptions, W/A bits, preprocessing and accumulator rules.
Sources: [AdaRound](https://proceedings.mlr.press/v119/nagel20a.html),
[BRECQ](https://openreview.net/pdf?id=POWv6hDd9XH),
[QDrop](https://arxiv.org/abs/2203.05740),
[PD-Quant](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_PD-Quant_Post-Training_Quantization_Based_on_Prediction_Difference_Metric_CVPR_2023_paper.html).

Weight equalization/bias correction has particular relevance to MobileNet
experiments; reproduce its applicability instead of assuming every nonlinear
graph admits the transformation. See
[Nagel et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.html).
Local reconstruction is an explicit extension of the current B policy, which
currently requires a policy decision to admit reconstruction. It is distinct
from full supervised retraining/QAT. Record the optimized parameters and
teacher losses; preserve the PTQ scope.

For noninteger families, establish compatible, justified adaptations before
claiming that an integer PTQ algorithm transfers unchanged. Every family gets
at least one reasonable optimized recipe; the entire list of methods is not
mandatory for every family. Binary/ternary remain extreme PTQ controls, and NF4
W=A failure cannot establish failure of its weight-only use.

## 5. Priority diagnostic experiments

**Calibration coverage.** Build a versioned observer using streaming histograms
or deterministic reservoirs spanning channels, spatial positions and images.
Record coverage and effective sampling weights. Compare old/new sampling with
all other choices fixed, and inspect scale/clipping stability as sample density
increases. Training calibration data determines revised scales; diagnostic
evaluation heads do not.

**Detector semantic boundaries.** Test separate coordinate/score encodings and
a practical high-precision head/DFL arm. Distinguish the observer correction
from the graph-policy change. Retain strict uniform results, and report the
fraction of operations, MACs and memory bytes left at higher precision. A
classification-oriented reconstruction loss may not handle box regression;
[Reg-PTQ](https://openaccess.thecvf.com/content/CVPR2024/papers/Ding_Reg-PTQ_Regression-specialized_Post-training_Quantization_for_Fully_Quantized_Object_Detector_CVPR_2024_paper.pdf)
is relevant detector prior work, not a directly validated YOLOv8n solution.

**Severe classifier losses.** Run paired W-only and A-only experiments, then
block-level quantize and block-level rescue interventions. The existing
one-layer-in-FP32 studies can miss cumulative error. Measure normalized error,
SQNR, clipping, changed prediction margins, residual input-scale mismatch and
prefix/suffix degradation. Avoid equating the largest raw MSE with the cause.

**Accumulation.** Distinguish (i) correctness for the declared finite arithmetic,
(ii) proof of no accidental overflow/output-code change relative to a wider
reference, and (iii) task-quality sensitivity to accumulator width. A meaningful
finite-accumulator B variant need not be identical to an ideal accumulator, but
must implement its specified semantics. This does not relax v1 acceptance or
admit known backend discrepancies. Build reusable compositional family/operator
proofs and integrate the 21 locally proved classifier candidates first.

## 6. Revised phases and decision gates

| Phase | Proposed change | Exit evidence |
| --- | --- | --- |
| 0–2 | Preserve completed versions; add a versioned protocol amendment and audited execution path | Data roles, semantics, acceptance categories and compute budget recorded |
| 3A: readiness | Fix observer coverage; establish realistic INT8 and scaled FP/MX anchors; benchmark the production exact path | Correctness and calibration sanity; feasible time estimate for full validation |
| 3B: broad discovery | 256-image A0 exploration across valid candidates, plus bounded early A1/B on two models | Coverage matrix, gross failure diagnoses, uncertainty and family preservation ledger |
| 3C: common screen | Extend informative configurations to 1k, including anchors, uncertain cases, hardware specialists and family representatives | Explicit promotion/defer reasons; no fixed top-N elimination |
| 4: depth | Start per candidate once its readiness is established; overlap W/A, scale, reconstruction and accumulator work | Recipe/range/accumulator interactions, matched-budget comparisons, frozen B policies |
| 5: confirmation | Freeze selected recipes before opening held-out outcomes; stage full ImageNet payload; repeat calibration seeds | Full quality, independent confirmation, direct candidate comparisons, robustness |
| 6A: early hardware | Begin representative complete MAC+scale+conversion and memory feasibility during Phases 3/4 | Measured cost priors with uncertainty; identify unsupported/expensive support logic |
| 6B: detailed hardware | Apply architecture sweeps and physical implementation to numerically validated candidates | Correct RTL, common flow, workload activity, complete cost, placement-seed variability |
| 7 | Test ranking reversals and quality-constrained frontiers under several resource budgets | Explain which mechanisms change winners; validate the selection method on held-out workloads |
| 8–10 | Preserve external MANT boundary | Versioned public export consumed separately |
| 11 | Begin claim/evidence tables now; release once results support them | Reproduction package, negative findings, limitations and a concrete contribution |

Replace the single global prerequisite with proposed **D4a** (a candidate can
enter depth safely) and **D4b** (coverage/promotion accounting is complete).
Record deferred configurations as incomplete or budget-deferred, never as
scientifically dominated merely because they are expensive. Completing a v2
gate would not retroactively complete the frozen v1 100-screen requirement.

D5 should allocate experiments by unresolved quality/cost uncertainty and
family coverage; D6 should freeze the best recipe **under a stated budget**,
not claim unbounded best-achievable PTQ. Bring the D9 memory-feasibility pilot
forward so an unavailable SRAM model is discovered before the paper depends on
it. D7/D8/D10/D11 retain their evidence requirements.

## 7. Statistical safeguards and dataset roles

The 256/1k/10k outcomes used to choose formats or recipes are development data.
For classification, reserve the 40k complement of the existing nested 10k list
for final confirmation; report the conventional full-50k result as well. For
COCO, reserve the 4k complement of screen1k and also report full val5k. Verify
all evaluation history and content hashes before describing a complement as
untouched. A full-set score containing development images is not an entirely
independent final test.

Use matched image IDs for direct candidate-versus-candidate comparisons as well
as candidate-versus-FP32. Overlapping separate baseline confidence intervals
are not a test of equivalence between two candidates. Predeclare practical
quality-loss margins and the primary contrasts; use simultaneous/multiplicity
control for confirmatory claims rather than selecting a favorable result from
hundreds of unadjusted tests.

For exploratory stages, show uncertainty and preserve close cases. Ordinary
95% bootstrap intervals repeatedly inspected while stopping adaptively do not
automatically retain 95% coverage. Either use fixed decision points with an
appropriate error allocation, or a sequential method whose assumptions match
the paired finite-population/stratified sampling design. Do not apply a generic
i.i.d. confidence-sequence formula blindly to these lists. Relevant foundation:
[Howard et al., confidence sequences](https://arxiv.org/abs/1810.08240).

For COCO, use image-level resampling with repeated-image multiplicity and the
actual dataset AP evaluator; retain AP-small/medium/large and category coverage.
Do not substitute Bernoulli intervals or rank rare categories from tiny panels.
For ImageNet, report the sampling design and repeat/sensitivity-check class
composition; one image/class does not estimate within-class variability well.

Use three calibration seeds for leading recipe comparisons, extending to five
where ranking is unstable and budget allows. Separate evaluation-image
uncertainty from calibration/optimization-seed variation. Keep paired calibration
subsets across methods; record failed seeds rather than reporting only the best.

## 8. Compute plan and practical scope

For the 92 uncompleted screens, 256 images each require 23,552 model-image
evaluations rather than 92,000. If 30 subsequently extend to 1k using saved
predictions, the cumulative cost is:

```text
92 × 256 + 30 × (1,000 − 256) = 45,872 evaluations
```

That is about 50.1% fewer evaluations in this illustrative discovery/common-screen
portion. The 30 is a budget example, not a cutoff. If uncertainty requires all
92 to extend, there is no image-count saving. Different model/family runtimes,
PTQ optimization, preparation, proofs and later validation prevent interpreting
this directly as a wall-clock speedup.

Likewise, three recipes × 256 images cost 768 evaluations, versus 1,000 for one
recipe, before recipe fitting/acceptance costs. This is why breadth of reasonable
methods can be preferable early. It does not make all combinations cheap.

Execution priorities:

1. Benchmark an audited metrics-only inference path versus sampled diagnostics.
   Store predictions for every image and collect expensive full diagnostics on
   a fixed representative panel; prove that instrumentation changes no outputs.
2. Keep the exact oracle for conformance and difficult witnesses. Build faster
   code-identical paths using bounded integer/dyadic arithmetic, specialized
   stores, batched independent outputs and proven transformations. Tensor-core
   or FP32 surrogates are exploratory unless their equality/error guarantee for
   the claimed semantics is established. A handful of matching images is not
   proof that reassociated reduction is exact.
3. Profile CPU fallback, output conversion, scale search, transfers and kernel
   occupancy. Optimize completed useful comparisons per hour rather than GPU
   utilization alone. Separate CPU proof/calibration work from GPU inference;
   choose concurrency from measured aggregate throughput and memory use.
4. Reuse immutable graphs, weights, FP32 references, and per-image predictions
   only across valid identities. Preserve the one-command resume interface with
   explicit campaign/stage/sample-list identity and safe stage extension.
5. Admit a full-validation package only after measuring its per-image time and
   total budget, including calibration seeds and detector cost. If it remains
   tens of seconds/image, obtain more compute or narrow the full-validation
   package before promising completion dates.

A practical initial paper package is 4–6 frozen numerical recipes across the
four existing workloads (16–24 primary full evaluations), with additional seeds
for the central winners and strongest baselines. Preserve more when uncertainty
or family roles demand it. Allocate 8–15 RTL configurations, then 2–4 physically
detailed representatives according to evidence, not arbitrary popularity.
These are initial planning targets, not eligibility rules.

## 9. What would make this a stronger research contribution

**Primary proposed contribution:** a validated procedure that selects format,
scaling and accumulator together, with complete quality/area/energy/throughput
accounting and explicit proof/evidence levels. Demonstrate where simpler
multiplier-only or nominal-bit rankings choose the wrong design.

Three falsifiable hypotheses should organize the experiments:

| Hypothesis | Required experiment | Result that would weaken it |
| --- | --- | --- |
| PTQ policy can change cross-format ordering | Matched data/budget A0/A1/B comparisons on multiple models, with direct paired differences | Rankings remain stable across reasonable recipes and seeds |
| Scale/accumulator/support cost can reverse arithmetic-only ranking | Same-quality complete MAC/scale/memory implementations and iso-area/energy frontiers | Complete cost preserves the arithmetic-only ordering across budgets |
| Tensor/operator structure predicts useful numerical choices | Develop selection rules on selected models/layers; test on held-out model/block types and compare to uniform anchors | No improvement in quality-constrained efficiency or poor transfer |

Potential mechanisms include range/scale policy versus bias headroom, residual
alignment, depthwise sensitivity, and semantically different detector outputs.
Show interaction ablations, not only one-factor sweeps. A new format or new
accumulator should be derived only if measured evidence exposes an exploitable
gap; inventing an encoding is not itself a contribution.

The scope should remain edge CNNs unless there is budget for a properly audited
transformer path. Add one transfer workload for finalists—e.g. the already
planned EfficientNet or another CNN with a meaningfully different graph—before
adding it to every combination. If claiming generality beyond CNNs, include a
vision-transformer workload with its required operator conformance and suitable
PTQ baselines; do not extrapolate CNN findings to LLMs.

The literature already includes INT/FP inference comparisons, shared-format
design spaces and scaled-format architectures. The proposed paper must clearly
state how its exact semantics, matched PTQ, accumulator analysis and complete
edge-system measurements add beyond them. Relevant anchors include
[FP8 versus INT8](https://arxiv.org/abs/2303.17951),
[Shared Microexponents](https://arxiv.org/abs/2302.08007),
[Microscaling Data Formats](https://arxiv.org/abs/2310.10537),
[OCP MX specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), and
[Avant-Garde, ISCA 2025](https://doi.org/10.1145/3695053.3731100).
Version the exact standard used; do not equate every similarly named shared
format with the same encoding, block layout or scale-generation algorithm.

For an architecture/EDA paper, a concrete design mechanism or validated
selection method plus realistic physical evidence is the strongest direction.
For an ML-methods paper, the quantization algorithm and generalization evidence
would need to become the main new contribution. A benchmark paper instead needs
an unusually useful, independently reproducible dataset and supported insights.
No amount of additional screening alone guarantees a top-tier publication.

## 10. Proposed next execution order

1. Record the v2 amendment, workload/recipe budgets, data roles and claim matrix.
2. Correct and validate calibration coverage; reproduce conventional INT8 and
   at least one deployable scaled FP/MX anchor under matched semantics.
3. Diagnose the four severe classifier cases with W/A and block-rescue studies;
   isolate detector sampling and output-domain effects.
4. Integrate reusable family acceptance for the most ready graphs and establish
   a fast audited production path; retain v1 pilot evidence under its own identity.
5. Run the 256-image A0/A1/early-B matrix, extend informative cases to 1k, and
   start representative complete hardware/memory pilots concurrently.
6. Freeze leading methods, run 5k/10k refinement and calibration robustness,
   then full held-out confirmation and detailed physical implementation.

Adopting this proposal requires new campaign and policy versions because F5/F6,
the fixed-1k gate, diagnostics policy and acceptance identities were previously
frozen. The eight completed screens remain usable historical controls and
retrospective budget evidence. Record any superseded unfinished v1 obligations
explicitly; do not report them as completed by the revised study.
