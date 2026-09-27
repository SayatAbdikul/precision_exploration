# Phase 3 review of the eight completed integer screens

Reviewed 2026-09-25. This is an interim evidence review, **not D4 approval**.
The [content-addressed review record](../../artifacts/phase3/reviews/interim-eight-integer/7a83fa2239eecebb5653c52cca62cb2dee8defe9a8b2f8ad07a35213af93621f.json)
hash-checks each analysis, diagnostic, screen and acceptance reference and
recomputes the retained integer acceptance proofs.
The 29-task integer controller run completed without a failed task. Every
configuration below has whole-graph integer acceptance, eight paired native
pilot images, a complete 1,000-image CUDA screen, paired analysis and layer
diagnostics. The [recomputed D4 readiness](../../results/summaries/phase3-d4-readiness.json)
finds eight completed screens and 92 missing configurations; its decision
remains `D4_OPEN`.

The frozen statistical policy uses the same 1,000 images for candidate and FP32,
5,000 paired bootstrap resamples, a 95% interval and seed 310911. Changes and
intervals below are in **percentage points of top-1 accuracy**. The FP32 top-1
baselines are 72.1% for MobileNetV2 and 70.1% for ResNet18.

| Screen and analysis | Top-1 | Top-5 | Top-1 change [95% paired interval] | Review disposition |
| --- | ---: | ---: | ---: | --- |
| [MobileNetV2 INT4](../../results/summaries/phase3-analysis-1a2daf8c43db.json) | 0.2% | 0.9% | -71.9 [-74.7, -69.1] | Severe loss; cause unresolved |
| [MobileNetV2 INT5](../../results/summaries/phase3-analysis-a89403d906f5.json) | 6.4% | 14.8% | -65.7 [-68.7, -62.7] | Severe loss; cause unresolved |
| [MobileNetV2 INT6](../../results/summaries/phase3-analysis-f7ab263b2174.json) | 46.0% | 68.7% | -26.1 [-29.1, -23.0] | Severe loss; cause unresolved |
| [MobileNetV2 INT8](../../results/summaries/phase3-analysis-cf9353a8e28d.json) | 70.6% | 89.7% | -1.5 [-3.0, -0.1] | Retain; further evidence |
| [ResNet18 INT4](../../results/summaries/phase3-analysis-3c550d8ff844.json) | 12.3% | 33.3% | -57.8 [-60.9, -54.6] | Severe loss; cause unresolved |
| [ResNet18 INT5](../../results/summaries/phase3-analysis-22a98c5deeb8.json) | 48.3% | 75.0% | -21.8 [-24.7, -18.8] | Retain; further evidence |
| [ResNet18 INT6](../../results/summaries/phase3-analysis-7f7b315039cf.json) | 62.5% | 85.4% | -7.6 [-10.0, -5.2] | Retain; further evidence |
| [ResNet18 INT8](../../results/summaries/phase3-analysis-d240b899d3f2.json) | 68.4% | 89.1% | -1.7 [-3.0, -0.4] | Retain; further evidence |

All eight retain the campaign's `UNCERTAIN` screening label. The four severe
rows have `diagnosis_required: true`: their top-1 interval lies entirely beyond
the frozen 20-point catastrophic-loss threshold. The other four are not
`PROMISING` under the conservative one-point preservation tolerance. These are
quality findings, not failures of native C++/CUDA agreement or integer graph
acceptance. No configuration or integer family is removed by this review.

## Configuration and sensitivity review

The [MobileNetV2 INT4](../../results/summaries/phase3-diagnostics-1a2daf8c43db.json),
[INT5](../../results/summaries/phase3-diagnostics-a89403d906f5.json) and
[INT6](../../results/summaries/phase3-diagnostics-f7ab263b2174.json) layer
summaries put `add`, `add_2` and `add_1` among the largest sampled MSE nodes
in every low-bit screen. The strict top-1 results improve monotonically from
INT4 through INT8. That pattern is consistent with accumulated representation
error, but neither sampled MSE nor the bit-width trend identifies a causal
operator, calibration choice or exact scale failure. There is no retained
MobileNetV2 one-operation intervention for these three full-screen losses.
Their specific catastrophic diagnoses remain open.

The [ResNet18 INT4 diagnostics](../../results/summaries/phase3-diagnostics-3c550d8ff844.json)
put `fc`, `add_7` and `layer4_1_conv2` highest by sampled MSE. Four eight-image
one-operation studies separately tested
[`conv1`](../../results/summaries/phase3-sensitivity-260aa67dd4eb.json),
[`fc`](../../results/summaries/phase3-sensitivity-4d80322df03d.json),
[`layer4_1_conv2`](../../results/summaries/phase3-sensitivity-a4e4f8669412.json)
and [`add_7`](../../results/summaries/phase3-sensitivity-2669f173ebf3.json).
None reproduces the strict full-graph top-1 collapse on its small fixed subset.
Cumulative effects or other operations remain possible; the cause is not
verified. A D4 diagnosis artifact concluding `catastrophic_configuration`
would overstate this evidence today.

The [MobileNetV2 INT8](../../results/summaries/phase3-diagnostics-cf9353a8e28d.json),
[ResNet18 INT5](../../results/summaries/phase3-diagnostics-22a98c5deeb8.json),
[INT6](../../results/summaries/phase3-diagnostics-7f7b315039cf.json) and
[INT8](../../results/summaries/phase3-diagnostics-d240b899d3f2.json) diagnostics
are complete. Their measured losses retain them for further evidence, including
near-boundary ResNet18 INT5, whose top-1 interval crosses the catastrophic
threshold rather than lying wholly beyond it. No sampled quantizer layer in
these eight reports recorded a nonfinite input event. Layer summaries explicitly
warn that samples are descriptive and do not establish a causal attribution.

## Four D4 review dimensions

| Dimension | Review of current evidence | Remaining gate |
| --- | --- | --- |
| Sensitivity | The four ResNet18 INT4 interventions do not isolate the strict loss; the other three severe configurations have no matching causal intervention. | Explain the four severe configurations with specific, verified diagnosis evidence. |
| Hardware preservation | [All 100 prepared graphs have storage/operator priors](../../results/summaries/phase3-hardware-priors.json). Those priors are not routed area, measured energy or full-system PPA; no hardware-based elimination follows. | Measure or validate hardware preservation for candidates across the full matrix. |
| Family preservation | Eight integer classifier screens are the only complete measurements. INT8 retains an integer representative, while the other format families remain unmeasured. | Review all families and workload specialists after their screens; preserve every unmeasured family. |
| Calibration | The eight screens use the [frozen graph/calibration inventory](../../results/summaries/phase3-gate-inventory.json). Their quality losses do not prove a calibration cause. The separate [YOLO INT8 diagnosis](phase3-yolo-int8-diagnosis.md) identifies a detector calibration-coverage confound. | Resolve detector calibration policy in a versioned experiment and review representativeness before D4. |

The [D4 readiness verifier](../../tools/analysis/phase3_d4.py) rechecked the
complete native screen evidence and reproduced all eight paired classification
statistics. It promotes all eight only as preservation candidates under the
uncertainty buffer, with no removals. Its `D4_OPEN` decision is correct while
92 screens and the four diagnoses remain missing. The running 92-configuration
pilot batch is diagnostic and cannot substitute for those screens.
The [explicit D4 review validator](../../tools/analysis/phase3_review.py) also
recomputes each reviewed label from the measured interval. A severe loss without
a matching verified diagnosis remains pending and cannot close D4, even after
all 100 screens have completed.
