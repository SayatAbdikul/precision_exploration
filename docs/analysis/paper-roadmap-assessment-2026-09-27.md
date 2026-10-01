# Public paper roadmap assessment — 2026-09-27

This is a status assessment and proposed schedule, not a replacement for the
sealed comparison matrix. New comparisons require a versioned protocol before
execution. No experiment was launched or changed while preparing this report.
The scope is the public numerical-format/generic-hardware paper; private MANT
architecture work is separate.

## Verified position

| Work | Current evidence | Remaining work |
| --- | --- | --- |
| Broad B exploration | All 200 configurations retained; 183 at 1,000 images and 17 at 128; 83 full recipe pairs, including 58 classifier pairs; balanced paired analysis completed | Freeze claim-relevant promotions and new diagnostic protocols; preserve development/confirmation distinction |
| Historical exact A | Eight accepted 1,000-image integer classifier screens | Retain as controls; historical Phase 3/D4 remains open |
| E0 exact acceleration | Validated optimized execution for admitted cases; large integer speedup | Measure sustained throughput and improve slow noninteger/detector paths before large allocations |
| E1 matched arithmetic | Six completed 32-image cases: R18 INT4/Posit4, MNV2 INT4, MNV3 Q1.6/FP6/ternary | Six selected cases remain proof-blocked; validate useful-quality configurations too |
| E2 recipe diagnosis | Eight new exact recipe graphs completed: six at 256 images and two at 128 | Explain remaining failures and compare with an appropriate published PTQ baseline |
| E4 confirmation | Not launched | Independent calibration repeats, audited confirmation data, full benchmark evaluation for selected headline claims |
| E5 hardware | 16 primitive synthesis pilots and 48 timing/vectorless-power pilot runs | Claim-relevant RTL, complete support costs, workload activity, physical evidence and quality-linked Pareto analysis |
| Manuscript | Publication outline and requirements exist | Related-work/novelty audit, methods draft, final figures, limitations and reproduction package |

The 183 extended configurations are a subset of the 200, not an additional 183.
They comprise the earlier 171 extensions plus twelve balanced counterparts,
which added 10,464 candidate-image predictions without changing their numerical
identities. The [balanced results report](b-stage-balanced-results-2026-09-27.md)
and `results/summaries/b-stage-paired-1k-v2/` supersede the earlier coverage
snapshot for current planning; historical selections and evidence remain intact.
B uses declared FP32 quantize/dequantize simulation. Its completion does not
certify exact low-bit arithmetic. E1's matched sequential-FP32 control is also
distinct from ordinary framework QDQ B.

The completed E1 cases agree in final output and layer codes on their 32-image
panels. Several have collapsed task quality, so this does not establish that
ordinary B transfers faithfully to competitive exact configurations. Integer
reduction bounds and empirical noninteger agreement have different strengths.
The completed ternary panel has 3.125% top-1 for both exact and matched control,
versus 68.75% for its paired FP32 baseline, despite agreement at every checked
layer and output. Assignment 2, useful-quality matched exact panels, remains a
proposed next assignment and has not been launched by this roadmap update.

The balanced B panels retain large architecture-dependent recipe effects:
MNV3 INT8 is 4.10% under maxabs versus 68.90% under percentile clipping, while
R18 FP6 E3M2 is 65.30% versus 53.80%, respectively, on 1,000 images. R18 INT4
and NF4 maxabs remain at 0.20% and 1.10%; increasing the panel did not repair
their collapsed quality. These observations support targeted recipe diagnosis
and useful-quality exact controls, not universal datatype rankings. Selection
used the original 128-image results, so the larger panels remain development
evidence.

An independent completion check revalidated all twelve configuration/prediction
digests and historical prefixes, reproduced their twelve paired recipe effects
and intervals, checked all three retained detector-evidence hashes, and confirmed
the 183/17 coverage counts. The five existing paired-analysis tests also passed.
This checks the saved numerical evidence; hardware evidence remains separate.

E2 adaptive rounding improves MNV2 INT5 from 5.86% to 11.33% top-1 (+5.47
percentage points) and INT6 from 47.66% to 53.52% (+5.86 points), on 256-image
development panels. Their pointwise paired intervals exclude zero, but these
are exploratory, adaptively extended comparisons, not final superiority claims.
R18 INT4 remains at 14.45% with rounding; MNV2 INT4 is at 0% on 128 images.
Those losses need explanation before allocating large confirmation runs.

## Recommended scientific structure

1. Keep broad coverage: retain every datatype/model pair in the evidence ledger,
   with recipe, arithmetic, image count, uncertainty and admission status.
2. Use A as the controlled exact reference and B for broad PTQ exploration.
   E1 and E2 provide matched comparisons between arithmetic and recipe effects.
   Completing every old 1,000-image strict-A screen is not the scheduling
   prerequisite for a focused paper. Unfinished original requirements remain
   explicitly open rather than being declared complete.
3. Use the completed balanced 183-result analysis, preserving uncertainty cases
   and plausible hardware specialists as well as accuracy winners. Define
   central hypotheses before choosing the final expensive comparisons.
4. Diagnose severe quality losses with bounded W/A, residual and layer/block
   precision ablations. Record changed precision as a separate configuration.
   Add an appropriate faithful published PTQ baseline. The current rounding
   intervention is an AdaRound-inspired adaptation, not an AdaRound reproduction.
   Candidate references: [AdaRound](https://proceedings.mlr.press/v119/nagel20a.html)
   and [BRECQ](https://openreview.net/forum?id=POWv6hDd9XH). Audit recent related
   work before making novelty or state-of-the-art claims.
5. Promote a bounded exact set provisionally around 8–12 configurations, with
   family/model coverage in the mechanism studies. This is a budgeting proposal,
   not a frozen top-N rule. Size larger panels from uncertainty and measured cost.
6. Reserve full validation for a smaller set of central comparisons and matched
   baselines. A costing scenario is four classifier configurations and one or
   two detector configurations; the scientific claim, not this number, decides
   the final matrix. Use 50k ImageNet / 5k COCO for corresponding full-benchmark
   claims, and report the audited unused complement separately.
7. Test at least three calibration seeds for central contenders and baselines
   on a common robustness panel. Freeze calibration manifests before fitting.
   Full evaluation of one selected calibration instance must not be presented
   as a full-benchmark three-seed average. Expand repeats if instability matters.
8. Evaluate complete hardware costs for claim-relevant representatives, then
   deepen physical evidence for approximately 2–4 finalists. Cover decoding,
   multiply/accumulate, scales, bias, requantization, metadata, memory and data
   movement. Primitive ROM pilots and vectorless power are insufficient for
   final architecture rankings or workload-energy claims.

The intended contribution is an evidence-backed rule for when datatype,
quantization recipe and accumulation change the quality/cost ranking, including
whether complete support costs reverse a multiplier-only ranking. A large
configuration count or a small improvement over a weak baseline is insufficient
on its own. Transfer claims need a held-out workload test with frozen choices.

## Proposed calendar from September 27

These are planning ranges including implementation, experiments, analysis and
writing. They are not estimates solely of the current queued experiments, and
they are not a commitment that unresolved proofs can be completed on a date.

| Window | Deliverable |
| --- | --- |
| Sep 27–Oct 4 | Current admitted queue and balanced B analysis completed; freeze next protocol using B/E2 results; benchmark selected families; validate hardware flow; start methods and related work |
| Oct 4–11 | Resolve priority admission gaps; test stronger PTQ and causal ablations; run family-stratified exact panels; freeze final comparisons |
| Oct 11–25 | Run costed larger/full validation and calibration robustness; develop and evaluate shortlisted RTL in parallel |
| Oct 18–25 | First complete manuscript draft, with pending results visibly marked |
| Oct 25–Nov 8 | Complete workload-driven hardware/physical evidence, confirmation and final statistical analysis; generate final figures |
| Nov 8–22 | Reproduction audit, claim review, writing revisions and submission-ready package |

Working target: **6–8 weeks to a submission-ready focused public paper
(November 8–22)**, conditional on the first-week throughput and hardware gates.
A focused numerical experimental package can target **3–5 weeks
(October 18–November 1)**. That is not equivalent to finishing the original
full hardware/publication roadmap. Writing starts immediately and overlaps runs.
If the first-week gates fail, revise the scope explicitly or move the date;
do not silently substitute simulation for exact evidence.

The exhaustive original roadmap has substantially more arithmetic architecture,
memory and physical design-space exploration. It is a multi-month scope at
present, without a defensible fixed completion date from the existing pilots.
Journal/conference review and acceptance occur after submission and cannot be
scheduled from workstation throughput.

## Conditions behind the estimate

- The workstation runs most of the time and the study stays focused. Reuse
  predictions only for identical numerical configurations and image identities.
- Four full classifier evaluations represent roughly 200,000 image evaluations
  before reuse. At a hypothetical sustained aggregate 600–900 images/hour this
  is 9.3–13.9 machine-days, excluding fitting, checks and interruptions. This is
  an integer-path planning scenario, not a measured rate for every datatype.
- A slow path at 400 seconds/image costs 4.6 days for 1,000 images and about
  231 days for 50,000 images in one sequential job. Such a configuration cannot
  enter the proposed full-validation schedule unchanged. It needs validated
  acceleration/sharding or an explicit limitation of the corresponding claim.
- By October 4, record actual family-specific sustained rates, checkpoint/setup
  overhead, total planned work and available concurrency. Large-run enrollment
  must fit that budget; do not extrapolate fast integer throughput to YOLO or
  every noninteger family.
- Also validate a representative hardware flow end to end, including available
  PDK/memory support and workload activity. The existing flow pilots lack
  extracted parasitics and workload-based switching evidence.
- Overlap CPU hardware work with GPU inference only after measuring the impact
  on completed useful work/hour. Maximum utilization alone is not the goal.
- Preserve immutable execution sources and resumable per-image checkpoints.
  The current `study_next` resume command covers its admitted queue, not future
  unimplemented robustness, hardware or paper workflows.

## Sources checked

- `docs/analysis/breadth-study-launch-2026-09-26.md`
- `docs/analysis/b-stage-balanced-results-2026-09-27.md`
- `results/summaries/b-stage-paired-1k-v2/analysis.json`
- `results/summaries/b-stage-paired-1k-v2/balanced-evidence-ledger.json`
- `artifacts/breadth_study/balanced_b_v1/plan.json` and `status.json`
- `artifacts/breadth_study/ternary_e1_v1/status.json` and its sealed 32-image summary
- `public/experiments/configs/breadth-study/comparison-matrix-v1.json`
- Live `.venv/bin/python -m tools.run.study_next status`
- Sealed comparisons under `artifacts/breadth_study/next_study_v1/E2/`
- `results/summaries/phase2-hardware-pilot.json`
- `results/summaries/phase2-sta-power-pilot.json`
- `docs/analysis/research-checkpoint-and-next-experiments-2026-09-25.md`
- `docs/roadmap/phases/phase-06-generic-hardware.md`
- `docs/publications/README.md`
