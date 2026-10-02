# B2 reconstruction baseline: AdaRound on the repaired simulator (2026-10-01)

Status: **complete (lane L6, runs r1 and r2; review corrections r3 and r4).** 18 of 18 primary fits and all 132 primary evaluation records
exist; the secondary "activation-aware input" group (6 fits, 12 arms, an adaptation) is complete (section 7); the
secondary max-abs group was not run.

All numbers are **development evidence**: top-1 on the frozen 1k screen (`imagenet_screen_1k`), which contains
dev512, the panel on which the B2 default recipe was selected. The rounding is learned on the first 1024 images of
the frozen calibration list (`imagenet_calibration_2k`, ImageNet training images, labels unused). The ImageNet
10k evaluation list beyond the screen was not read. Intervals are pointwise 95 percent paired image-bootstrap
intervals of the difference, in points (10000 resamples, seed 20260927, `tools.analysis.b2_ties.paired`), without
multiplicity correction. Unless stated otherwise a figure is top-1 under the expected-credit tie rule (credit 1/k
when the label is among the k classes tied for the maximum); the lowest-class-index argmax figure is given in
brackets. With FP32 activations no image is tied and both rules agree.

- Protocol, written before any measurement: `public/experiments/configs/breadth-study/b2-recon-protocol-v1.json`.
- Summaries: `results/summaries/b2-recon-v1/{matrix-1k,six-bit-order,faithfulness,questions,fits}--r2.json` and
  `matrix-1k--r2.csv`; figure `results/figures/b2-recon-matrix--r2.{pdf,png}`.
- Records: `artifacts/experiment_b2_recon/fits/<fit>/` (one npz per layer, `spec.json`, `complete.json`) and
  `artifacts/experiment_b2_recon/evals/<model>/<arm>.{json,npz}` (sealed; per-image top-5, tie size, tie flags).
- Diagnostics: `artifacts/experiment_b2_recon/diagnostics/bias-correction-resnet18-int4-per-layer-r2.json`
  (written unsealed by r2; sealed after the fact by r3 in `...-r2.seal-r3.json` with its file hash, its content,
  the script hash and both modification times) and `prediction-collapse-r3.json` (sealed). Both r3 steps follow
  `public/experiments/configs/breadth-study/b2-recon-protocol-v1-addendum-1.json`, written before them. The r3
  collapse readout is superseded by the tie-aware `prediction-collapse-ties-r4.json` (sealed) and the full maxima
  sets of five re-evaluated arms in `diagnostics/maxima-r4/<model>/<arm>.{npz,json}` (sealed), following
  `b2-recon-protocol-v1-addendum-2.json`, written before them.
- `tools/experiment_b2` is not tracked by git (lane L1's files), so that this lane left it unmodified is shown by
  its source hash only: `tools.experiment_b2.common.source_identity()` = `3966af3b5a...`, equal to the
  `b2_source_sha256` stored in all 144 evaluation records and every fit spec (rechecked by the reviewer and by r3).

Revision r3 (2026-10-02, after independent review 1, approve with fixes): the paper's W4/A8 number is now
labelled as its Table 7 setting (2048 images, 20000 iterations) and added to the differences (section 3); the
claim that the default recipe's INT4 failure is the bias-correction collapse is now checked per model (section 3;
r3's readout of that check is superseded by r4, below); the short answer on the 6-bit order now names the ResNet18 arms where FP6 is
significantly ahead; two ranges were corrected (section 5: 2.05 to 3.82 points; section 7: 17.0 to 29.5 percent).
No measured number changed and no summary file was regenerated.

Revision r4 (2026-10-02, after independent review 2, approve with fixes): the r3 collapse check read the first
index `topk` returns, whose order among tied maxima is unspecified, in cells where most images are tied. It is
replaced by a tie-aware readout (addendum 2; five arms re-evaluated bit-identically to store their full maxima
sets); section 3 now reports the criterion's verdicts as computed before the reading: bias correction is the
switch to a one-class output in 5 of 12 INT4 cells (ResNet18 W4/A8 nearest, ResNet18 W4/A4 nearest and learned,
MobileNetV2 W4/A4 nearest and learned), not in the other 7. The r3 sentence "holds for ResNet18 only" was wrong.
Short answer 2 and section 4 now include MobileNetV2 W4/A4 `L-bc` (0.23 against 27.2 without bias correction).
No accuracy number changed and no summary file was regenerated.

## Short answers

1. **MobileNetV3-Large INT8 gap: not closed.** Learned rounding followed by B2 bias correction (`L-bc`) gives 72.95
   (73.1), up 1.35 [0.05, 2.70] on the default recipe (71.6) but still 2.05 [0.25, 3.85] below FP32 (75.0). The
   published method without bias correction (`L-nobc`) gives 71.2, no better than the default. The gap is not a
   weight problem: INT8 weights with FP32 activations are at 75.3 with nearest and with learned rounding. It comes
   from the INT8 activations, which AdaRound does not touch.
2. **4-bit: AdaRound rescues 4-bit *weights*, not W4/A4.** W4/A8 with learned rounding and no bias correction:
   ResNet18 70.4 (FP32 70.1), MobileNetV2 71.5 (72.1), MobileNetV3 69.35 (75.0, -5.65); the default recipe gives
   0.1, 7.5 and 48.75. Weights only (W4/A32): 70.5, 71.7, 73.1. W4/A4 stays far from FP32 under every arm (best
   learned 24.8, 27.2, 0.1), because 4-bit activations, not the weights, now fail. Learning the rounding against
   quantized activations (an adaptation, section 7) lifts MobileNetV2 W4/A4 to 50.3 but no model to within 5
   points. B2's bias correction applied after learned rounding hurts at 4 bits (ResNet18 W4/A4 3.0 against
   24.8 without it, MobileNetV2 W4/A4 0.23 against 27.2, MobileNetV2 W4/A8 54.2 against 71.5); in both W4/A4
   cells it makes one class a maximum on almost every image (section 3).
3. **6-bit order: unchanged.** By the protocol's criterion (sign change with an interval excluding zero) no arm
   changes the INT6-versus-FP6 order on any model. MobileNetV3 stays decided by activations (FP6 ahead by 31 to 40
   points under every arm; with FP32 activations both formats are within 0.6 point). On MobileNetV2 INT6's lead under
   the default recipe (+3.9) is a bias-correction effect: it shrinks to +1.6 [0.03, 3.2] with `L-bc` and turns to
   -1.25 [-2.6, 0.1] (not significant) with `L-nobc`. ResNet18: no significant difference under `N-default` or `L-bc`,
   but FP6 is significantly ahead under the published method `L-nobc` (-1.68 [-2.93, -0.37]) and under the nearest
   arms without bias correction (`N-nobc` -1.48 [-2.90, -0.07], `N-minimal` -2.17 [-3.77, -0.66]); since FP6 is
   also (not significantly) ahead under `N-default`, the sign does not change and the criterion reports no change.
   Under the lowest-index tie rule none of these three ResNet18 intervals excludes zero.

Faithfulness: the ResNet18 W4/A32 reproduction is **consistent** with the paper by the criterion written before
measuring (drops of 0.6, 1.3 and 0.8 points for three seeds against a published 1.08).

## 1. What is implemented, and where each setting comes from

Paper: Nagel, Amjad, van Baalen, Louizos, Blankevoort, "Up or Down? Adaptive Rounding for Post-Training
Quantization", ICML 2020, arXiv 2004.10568v2. The PDF was read by run r1 (sections 3 to 5, Tables 2 to 8); run r2
did not re-read it and relies on the protocol, which records every setting and number used here.

| setting | paper | implemented (`tools/experiment_b2_recon`) | source |
|---|---|---|---|
| soft weight | eq. 22: `s * clip(floor(W/s) + h(V), n, p)` | `s * (lo + h(V) (hi - lo))`, `lo`/`hi` the neighbouring codebook levels of `W/s`; identical to eq. 22 on integer grids (unit test) | paper |
| rectified sigmoid | eq. 23, zeta 1.1, gamma -0.1 | same | paper |
| regulariser | eq. 24, `sum 1 - abs(2h - 1)^beta`, beta annealed high to low | same | paper |
| lambda | not stated | 0.01 | memory of AIMET |
| beta schedule, warm start | not stated | no regulariser for the first 20 percent of iterations, then 20 down to 2 on a cosine | memory of AIMET |
| objective | eq. 25: asymmetric, `f_a(Wx)` against `f_a(W_soft x_hat)` | same; `x_hat` from the graph whose preceding layers carry their learned hard weights; FP32 activations | paper |
| loss normalisation | Frobenius norm, normalisation not stated | squared error summed over output channels, mean over batch and positions; regulariser summed over the layer | memory of AIMET |
| `f_a` | "the activation function" | the nonlinearity that is the only consumer of the layer (ReLU, ReLU6, Hardswish, Hardsigmoid), else identity | paper, extended to Hardswish/Hardsigmoid |
| initialisation | Figure 3: initial `h(V)` is the FP32 weight | `h(V) = (W/s - lo) / (hi - lo)` | paper |
| scale | per layer, minimising `norm(W - W_bar)`, fixed before learning | faithfulness fits: one B2 MSE-search scale per layer; matrix fits: the B2 per-output-channel MSE scale (the default recipe's) | paper / B2 |
| layers | whole network, first and last included | every Conv2d and Linear | paper |
| batch norm | folded | folded (B2 graph) | paper |
| optimiser | Adam, default hyper-parameters | Adam, lr 1e-3, betas (0.9, 0.999) | paper |
| iterations, batch | 10000, 32 (Table 7: 20000) | 10000, 32 | paper |
| calibration images | 1024 ImageNet training images (Table 7: 2048) | first 1024 of the frozen calibration list (sha256 order) | paper |
| activations while learning | not stated | FP32 | memory of AIMET |
| activations at inference | FP32, or 8 bits with min/max ranges | B2 handling (fused boundaries, unsigned codes where proven non-negative, MSE scales from the FP32 observations) or FP32 | B2 |
| bias | no bias correction | arm `L-nobc`: none; arm `L-bc`: B2's empirical correction after the rounding is fixed | paper / B2 |
| hard decision | not stated | up when `h(V) >= 0.5`; no fallback to nearest | memory of AIMET |
| seeds | 5 | 3 for the faithfulness setting, 1 for the matrix; the seed changes the mini-batch order only | - |

The five rows marked "memory of AIMET" were not re-read from a source and are the most likely cause of any
disagreement with the published numbers. Non-uniform codebooks (`fp6_e2m3`, `posit8_es1`): the same variable
interpolates between two neighbouring levels of unequal distance. **This is an adaptation and not part of the
published method**, which is defined on a uniform grid; those rows are marked with an asterisk.

Runs are resumable per layer: a job stops starting layers after 600 s and is rerun (layer seed = 100003 * seed +
layer index, so a resumed fit equals an uninterrupted one). A fit is stored as packed up/down bits plus float32
scales per layer: 0.3 to 1.5 MB per fit, 21 MB for all fits of the lane; evaluation records 2.2 MB.

## 2. Tests

`tests/unit/test_experiment_b2_recon.py` (17 tests) and `tests/unit/test_experiment_b2_recon_report.py` (3 tests):
`.venv/bin/python -m pytest tests/unit/test_experiment_b2_recon.py tests/unit/test_experiment_b2_recon_report.py -q`
-> 20 passed. Covered: the rectified sigmoid (ends reached at finite V, gradient only inside), the regulariser
(hand values), the beta schedule, eq. 22 on an integer grid including clipped weights, neighbours on INT6 / FP6 /
Posit8 with nearest rounding being one of the two choices, a layer with a known optimum (three weights 0.3, 0.45,
0.25 with equal inputs: exactly one must round up; nearest rounds none), a 2 x 6 layer against exhaustive search
over all 64 choices per row (AdaRound finds the best and the second best, nearest the second and the seventh best),
convergence of every soft variable to exactly 0 or 1 and determinism, bit-identity of the builder with
`prepare_b2` for nearest rounding (5 format/recipe pairs), mixed INT4/INT8 precision, the layer record round trip
and write-once rule, a sequential two-layer fit on a small graph, and the report's credit rules and arm pairing.
Collapse checks (r3, r4): `tests/unit/test_experiment_b2_recon_collapse.py` (3 tests) and
`tests/unit/test_experiment_b2_recon_collapse_ties.py` (4 tests: tie-aware bounds equal the exact shares when every
tie is listed and contain them for larger ties, a case where `topk`'s first index misses a collapse, the
status/verdict rule). All four files: 27 passed.

On the real models the builder reproduces the sealed B2 top-5 lists on 1000 of 1000 images for all 45 nearest
arms with a frozen B2 recipe (3 models x 5 formats x 3 recipes), and the FP32 lists for all three models. All
records were made with one B2 source identity and one set of reconstruction sources.

A property found while testing: the method does not guarantee binary soft variables. A variable is pushed to 0 or 1
only if the curvature of its reconstruction term, about `s^2 E[x^2]`, is below `4 lambda`. After 10000 iterations
the fraction of soft variables still between 0.01 and 0.99 is 0 to 0.24 percent at 6 and 8 bits and 0.6 to 3.7
percent at INT4 (MobileNetV2 INT4 3.7 percent); they are rounded by the sign of V like the others.

## 3. Faithfulness check (ResNet18, INT4 weights, one scale per layer)

Setting: torchvision ResNet18 checkpoint frozen in this repository, INT4 codebook (-8..7), one MSE scale per layer,
every layer including first and last, 1024 calibration images, 10000 iterations, batch 32, seeds 0, 1, 2.
`results/summaries/b2-recon-v1/faithfulness--r2.json`.

| arm | this project, 1k screen (FP32 70.1) | drop vs FP32 [interval] | paper, 50k val (FP32 69.68) | paper drop |
|---|---|---|---|---|
| nearest, W4/A32 | 19.0 | 51.1 [47.9, 54.3] | 23.99 | 45.69 |
| empirical bias correction on it, W4/A32 | **0.1** (one class for all images) | 70.0 | 38.87 | 30.81 |
| AdaRound W4/A32, seed 0 | 69.5 | 0.6 [-0.9, 2.1] | 68.60 +- 0.09 (5 seeds) | 1.08 |
| AdaRound W4/A32, seed 1 | 68.8 | 1.3 [-0.2, 2.8] | | |
| AdaRound W4/A32, seed 2 | 69.3 | 0.8 [-0.7, 2.3] | | |
| AdaRound W4/A8 (B2 activations), seeds 0, 1, 2 | 69.4 (68.9), 68.6 (68.3), 69.4 (68.6) | 0.7, 1.5 [0.0, 3.0], 0.7 | 68.55 +- 0.01, **Table 7 setting: 2048 images, 20k iterations** | 1.13 |
| (paper only) AdaRound W4/A32 in the Table 7 setting | - | - | 68.71 +- 0.06 | 0.97 |
| nearest W4/A8 (B2 activations) | 19.0 | 51.1 | - | - |

**Verdict by the criterion written before measuring: consistent.** The published drop of 1.08 points lies inside
the seed-0 interval [-0.9, 2.1], and the three seeds lie within 0.7 point (68.8 to 69.5; mean 69.2, mean drop 0.9).
The W4/A8 rows are not a like-for-like comparison (corrected in r3): the paper's only ResNet18 W4/A8 AdaRound
number, 68.55, is from its Table 7, which uses 2048 calibration images and 20000 iterations, while this lane used
1024 images and 10000 iterations (the paper's setting for its W4/A32 ablations, 68.60). The comparison that the
paper does support is the cost of 8-bit activations at fixed weights: in Table 7 W4/A8 minus W4/A32 is -0.16 point
(68.55 against 68.71); here it is -0.1, -0.2 and +0.1 point for seeds 0, 1, 2 (B2 activation handling instead of
min/max ranges). Same direction and size, not tested by any criterion. The intervals are wide (about +-1.5 points) because
the screen has 1000 images; the check can exclude a gross failure, not a difference of a few tenths. 16.1 percent of
the weights round differently from nearest; 0.8 percent of the soft variables were not binary at the end.

Two setup checks disagree with the paper and are reported, not tuned:

- Nearest rounding is 5 points worse than published (drop 51.1 [47.9, 54.3] against 45.69). Candidate causes are
  the scale (the B2 grid search minimises the weight MSE with 8 points per octave and local refinement; the
  paper's minimiser is not specified) and the evaluation set. Not investigated.
- **B2's empirical bias correction collapses the network to one class (0.1) where the paper reports 38.87.** This
  is `tools.experiment_b2.engine.bias_correct` applied with FP32 activations, so activation quantization is not the
  cause. Diagnostic (r2, `tools/experiment_b2_recon/diagnose_bias.py` ->
  `artifacts/experiment_b2_recon/diagnostics/bias-correction-resnet18-int4-per-layer-r2.json`, 64 screen images):
  the corrected network matches the FP32 per-channel means layer by layer by construction, but its
  image-to-image variation disappears in layer 4. The standard deviation across images of the per-image channel
  means is, FP32 / uncorrected W4 / corrected W4: `layer4_1_conv1` 0.133 / 0.087 / 0.017, `layer4_1_conv2`
  1.23 / 0.75 / 0.040, `fc` 2.64 / 1.42 / 0.11. In `layer4_0_conv2`, `layer4_1_conv1` and `layer4_1_conv2` the RMS correction
  (0.36, 0.47, 0.55) exceeds the RMS of the FP32 channel means (0.22, 0.29, 0.34). With almost constant logits, every image takes the class with the largest
  corrected mean logit (463). The uncorrected W4 network still predicts 44 distinct classes on the same 64 images.
  Whether this is a defect of `bias_correct` (it belongs to lane L1) or a property of sequential mean correction
  of pre-activations before ReLU at large weight errors is **not settled**: it is reported as an open issue.
- Does the same collapse explain the B2 default recipe (per-channel scales, quantized activations) at INT4? The
  diagnostic above is per-layer W4/A32 only, so the predictions of the existing sealed records were checked
  (`N-default` and `N-nobc` differ only in bias correction, as do `L-bc` and `L-nobc`; collapse = one class takes
  at least 90 percent of the 1000 screen images). r3 (addendum 1, `collapse_check.py` ->
  `prediction-collapse-r3.json`) counted the first index `topk` returns; **that readout is superseded**: in the W4/A4
  cells 50 to 100 percent of the images are tied (mean tie 6.5 classes on ResNet18 W4/A4 `N-default`, 10.9 `L-bc`,
  20 on MobileNetV2 W4/A4), and the order of tied maxima in `topk` is unspecified (review 2). r4 (addendum 2,
  written first; `tools/experiment_b2_recon/collapse_ties.py` ->
  `artifacts/experiment_b2_recon/diagnostics/prediction-collapse-ties-r4.json`, sealed) counts a class per image
  only through the tie: primary share = the class attains the maximum logit ("among the maxima"), beside it the
  expected share under uniform random tie-breaking and the share as lowest-index maximum. Shares are bounded from
  the stored top-5 lists; the five records where a bound left the status open were re-evaluated on the GPU with
  the unchanged path (top-5, tie sizes and deployed state reproduced bit for bit) and their full maxima sets stored
  (`diagnostics/maxima-r4/`), so every verdict below is decided.

  **Verdicts as computed (primary share, addendum-1 rule).** Supported (the arm with bias correction collapses,
  the arm without does not), 5 of 12 cells: ResNet18 W4/A8 `N-default` (class 463 a maximum on 100 percent,
  without bias correction largest class 0.9 percent; 0.1 against 53.5), ResNet18 W4/A4 `N-default` (463 on 100
  percent, without at most 19 percent; 0.13 against 27.8), ResNet18 W4/A4 `L-bc` (463 on 94.4 percent, without at
  most 42 percent; 3.0 against 24.8), MobileNetV2 W4/A4 `N-default` (457 on 100 percent, without 43 to 46 percent;
  0.1 against 1.1), MobileNetV2 W4/A4 `L-bc` (457 on 99.9 percent, without at most 30 percent; 0.23 against
  27.2). Not supported, 7 of 12: ResNet18 W4/A8 `L-bc` (largest class 0.6 percent), MobileNetV2 W4/A8 `N-default`
  (class 600 on 81.6 percent; 7.5 against 2.25) and `L-bc` (6.2 percent), MobileNetV3 W4/A4 `N-default` (463 on
  87.7 percent; 0.10 against 0.20) and `L-bc` (733 on 89.6 percent; 0.13 against 0.05), MobileNetV3 W4/A8
  `N-default` and `L-bc` (at most 1 percent). None undecided. The r3 readout had ResNet18 W4/A4 `N-default`, ResNet18 W4/A4
  `L-bc` and MobileNetV2 W4/A4 `L-bc` as not supported; all three change to supported.

  Under the two other shares the verdicts differ where ties are large: under the expected share only ResNet18
  W4/A8 `N-default` is supported (a class at the maximum of ties averaging 6 to 21 classes is predicted on 5 to 22
  percent of the images under random tie-breaking); under the lowest-index share ResNet18 W4/A8 `N-default` and MobileNetV2
  W4/A4 `N-default` and `L-bc` are supported (class 419 is the lowest maximum on 98.9 and 90.7 percent).

  **Reading.** Bias correction is the switch that makes the output independent of the image on ResNet18 (W4/A8;
  W4/A4 with nearest and with learned rounding) and on MobileNetV2 W4/A4, always on the same class per model (463 is
  also the class of the per-layer W4/A32 collapse above). This shows the switch, not the mechanism: the per-channel
  cells were not diagnosed layer by layer. On MobileNetV2 W4/A4 with nearest rounding the network is already near
  chance without bias correction (1.1), so there bias correction decides how it fails, not whether; after learned
  rounding it destroys a 27.2-point network. Bias correction is not the cause on MobileNetV2 W4/A8 and MobileNetV3
  W4/A8 (it helps: 7.5 against 2.25, 48.75 against 5.9) nor on MobileNetV3 W4/A4, which is near collapse with nearest
  rounding with or without it (one class a maximum on 87.7 and 82.0 percent of the images) and at chance under all four
  arms (0.05 to 0.20). Collapse is also not specific to bias
  correction: MobileNetV2 Posit8 under the minimal recipe (no bias correction) has class 610 at the maximum on 92.3
  percent of the images (0.1).

Differences in setup, all of which remain: evaluation on the 1000-image screen, not the 50000-image validation
set; FP32 baseline 70.1 on the screen (paper 69.68, torchvision 69.76); 3 seeds instead of 5 (the seed changes only
the mini-batch order); scale from the B2 grid search instead of an unspecified minimiser; lambda, beta schedule,
warm start, loss normalisation and the hard decision from memory of AIMET; this repository's frozen calibration
images; own batch-norm folding and FX graph; for W4/A8, B2 activation handling (fused boundaries, unsigned codes,
MSE scales) instead of min/max ranges; for W4/A8, 1024 images and 10000 iterations against the paper's Table 7
setting of 2048 images and 20000 iterations (added in r3; the 2048/20000 setting was not run).

## 4. Matrix (three classifiers)

Arms: `N-default`, `N-nobc`, `N-minimal` (nearest rounding; the sealed B2 configurations, re-evaluated);
`L-nobc` (learned rounding on the default grid without bias correction: the published method on the B2 grid);
`L-bc` (learned rounding, then B2 bias correction: one switch away from `N-default`); `N-A32` / `L-A32` (the same
weights with FP32 activations). One fit per model and weight format (per-channel MSE scales, seed 0, FP32
activations while learning) serves every activation format. Rows marked * use a non-uniform codebook: the learned
rows there are the adaptation, not the published method. Full table with all intervals and the topk figure:
`results/summaries/b2-recon-v1/matrix-1k--r2.csv`; figure `results/figures/b2-recon-matrix--r2.pdf`.

Top-1, expected credit (lowest index):

**ResNet18 (FP32 70.1)**

| weights / activations | N-default | N-nobc | N-minimal | L-nobc | L-bc | N-A32 | L-A32 |
|---|---|---|---|---|---|---|---|
| INT8 / INT8 | 69.2 (68.6) | 69.3 (69.0) | 69.7 (69.4) | 69.7 (69.6) | 69.5 (69.2) | 69.0 | 69.9 |
| INT6 / INT6 | 65.9 (65.2) | 66.5 (66.8) | 65.8 (65.6) | 66.0 (65.8) | 68.1 (67.2) | 69.4 | 69.5 |
| INT4 / INT8 | 0.1 (0.1) | 53.5 (53.6) | 46.5 (46.7) | **70.4 (69.9)** | 67.8 (67.7) | 53.9 | 70.5 |
| INT4 / INT4 | 0.1 (0.2) | 27.8 (28.5) | 17.1 (18.0) | 24.8 (23.7) | 3.0 (5.2) | | |
| FP6 e2m3* | 66.7 (66.8) | 68.0 (66.9) | 68.0 (67.0) | 67.7 (66.7) | 67.8 (67.5) | 69.8 | 70.6 |
| Posit8 es1* | 69.6 (69.1) | 69.2 (69.4) | 4.4 (4.4) | 69.0 (68.8) | 69.6 (69.5) | 69.8 | 70.0 |

**MobileNetV2 (FP32 72.1)**

| weights / activations | N-default | N-nobc | N-minimal | L-nobc | L-bc | N-A32 | L-A32 |
|---|---|---|---|---|---|---|---|
| INT8 / INT8 | 72.3 (72.0) | 71.9 (72.0) | 72.3 (72.1) | 72.1 (72.1) | 72.25 (72.0) | 71.8 | 72.3 |
| INT6 / INT6 | 70.1 (70.3) | 67.1 (66.6) | 66.3 (66.2) | 69.7 (69.7) | 70.4 (70.1) | 68.3 | 72.3 |
| INT4 / INT8 | 7.5 (8.1) | 2.25 (2.3) | 1.1 (1.0) | **71.5 (71.3)** | 54.2 (54.2) | 2.3 | 71.7 |
| INT4 / INT4 | 0.1 (0.1) | 1.1 (1.2) | 0.5 (0.4) | 27.2 (25.6) | 0.2 (0.4) | | |
| FP6 e2m3* | 66.2 (65.4) | 67.0 (67.1) | 66.6 (66.5) | 71.0 (70.4) | 68.8 (68.4) | 68.9 | 71.7 |
| Posit8 es1* | 72.4 (71.9) | 72.1 (72.0) | 0.1 (0.1) | 72.25 (72.1) | 72.0 (72.4) | 71.7 | 72.2 |

**MobileNetV3-Large (FP32 75.0)**

| weights / activations | N-default | N-nobc | N-minimal | L-nobc | L-bc | N-A32 | L-A32 |
|---|---|---|---|---|---|---|---|
| INT8 / INT8 | 71.6 (71.5) | 70.5 (70.6) | 70.15 (69.9) | 71.2 (71.3) | 72.95 (73.1) | 75.3 | 75.3 |
| INT6 / INT6 | 21.0 (20.9) | 3.45 (3.5) | 2.55 (2.5) | 5.3 (5.2) | 23.4 (24.2) | 71.8 | 75.0 |
| INT4 / INT8 | 48.75 (49.0) | 5.9 (5.9) | 1.35 (1.4) | **69.35 (69.4)** | 69.1 (69.0) | 6.9 | 73.1 |
| INT4 / INT4 | 0.1 (0.1) | 0.2 (0.1) | 0.1 (0.1) | 0.05 (0.0) | 0.13 (0.3) | | |
| FP6 e2m3* | 58.4 (58.5) | 34.8 (35.1) | 36.8 (37.9) | 45.4 (46.3) | 57.8 (57.8) | 72.4 | 74.8 |
| Posit8 es1* | 73.4 (72.8) | 74.8 (74.7) | 0.0 (0.0) | 73.9 (73.5) | 72.7 (72.0) | 75.6 | 74.8 |

Selected paired differences (expected credit):

| model, cell | learned arm minus FP32 | learned arm minus N-default | learned minus nearest, same recipe |
|---|---|---|---|
| ResNet18 W4/A8, L-nobc | +0.33 [-0.90, 1.57] | +70.3 [67.5, 73.1] | +17.0 [14.4, 19.6] |
| ResNet18 W4/A4, L-nobc | -45.3 [-48.0, -42.4] | +24.7 [22.9, 26.5] | **-2.96 [-5.23, -0.61]** |
| ResNet18 INT6, L-bc | -1.98 [-3.27, -0.71] | +2.23 [1.08, 3.42] | (same) |
| MobileNetV2 W4/A8, L-nobc | -0.62 [-1.97, 0.73] | +64.0 [61.0, 66.9] | +69.2 [66.4, 72.1] |
| MobileNetV2 W4/A4, L-nobc | -44.9 [-47.6, -42.2] | +27.1 [25.4, 28.8] | +26.1 [24.3, 27.9] |
| MobileNetV2 FP6*, L-nobc | -1.15 [-2.63, 0.37] | +4.72 [2.90, 6.55] | +3.93 [2.32, 5.58] |
| MobileNetV3 INT8, L-bc | -2.05 [-3.85, -0.25] | +1.35 [0.05, 2.70] | (same) |
| MobileNetV3 INT8, L-nobc | -3.82 [-5.72, -2.00] | -0.42 [-2.02, 1.17] | +0.68 [-0.70, 2.10] |
| MobileNetV3 W4/A8, L-nobc | -5.65 [-7.75, -3.65] | +20.6 [17.8, 23.4] | +63.4 [60.3, 66.5] |
| MobileNetV3 INT6, L-bc | -51.6 [-54.6, -48.5] | +2.40 [0.35, 4.52] | (same) |
| W4/A32, L-A32 (ResNet18, MNv2, MNv3) | +0.4 [-0.9, 1.7]; -0.4 [-1.8, 1.0]; -1.9 [-3.4, -0.5] | - | +16.6; +69.4; +66.2 |

What the matrix shows beyond the three questions:

- With FP32 activations learned rounding removes the weight damage of every format on every model: all 15 `L-A32`
  cells are within 0.6 point of FP32 except MobileNetV3 INT4 (-1.9). The remaining losses at INT6, FP6 and W4/A4
  are activation losses.
- At 8 bits learned rounding changes little on ResNet18 and MobileNetV2 (every `L` minus `N` of the same recipe
  within 0.7 point). At 6 bits it gains 1 to 4 points where the nearest arm lost them to the weights: ResNet18
  INT6 `L-bc` +2.2 [1.1, 3.4], FP6* `L-bc` +1.2; MobileNetV2 INT6 `L-nobc` +2.6 [0.7, 4.5], FP6* `L-nobc` +3.9,
  `L-bc` +2.6.
- B2's bias correction after learned rounding helps at 6 and 8 bits on MobileNetV3 (INT8 +1.77 [0.17, 3.35]
  over `L-nobc`; FP6 57.8 against 45.4) but hurts at 4 bits (ResNet18 W4/A4 3.0 against 24.8; MobileNetV2 W4/A4
  0.23 against 27.2; MobileNetV2 W4/A8 54.2 against 71.5). Tie-aware (r4, section 3): on ResNet18 W4/A4 `L-bc`
  class 463, the class of the section 3 collapse, is a maximum on 94.4 percent of the images (expected share under
  random tie-breaking 19.4 percent), against at most 42 percent (expected at most 4.8) for any class without bias
  correction; on MobileNetV2 W4/A4 `L-bc` class 457 is a maximum on 99.9 percent (at most 30 percent without).
  MobileNetV2 W4/A8 `L-bc` does not concentrate (largest class a maximum on 6.2 percent of images), so its loss is
  a different effect.
- On ResNet18 W4/A4, learned rounding is 3 points *worse* than nearest without bias correction (24.8 against 27.8):
  rounding learned against FP32 activations is not optimal once 4-bit activations are added.
- The non-uniform adaptation behaves like the uniform case: FP6 weights only 68.9 -> 71.7 (MobileNetV2) and
  72.4 -> 74.8 (MobileNetV3); Posit8 weights are already near FP32 with nearest rounding.
- The tie rule does not change any conclusion: expected credit and lowest index differ by at most 2.2 points
  (ResNet18 W4/A4 `L-bc`, 900 tied images) and 1.6 (MobileNetV2 W4/A4 `L-nobc`), elsewhere by at most 1.2.

## 5. The three questions (criteria from the protocol)

`results/summaries/b2-recon-v1/questions--r2.json` and `six-bit-order--r2.json`.

**MobileNetV3-Large INT8 gap.** Criterion: closed if `L-bc` or `L-nobc` is within 1.0 point of FP32 with the
difference to `N-default` excluding zero. `L-bc` 72.95: -2.05 [-3.85, -0.25] from FP32, +1.35 [0.05, 2.70] over
`N-default`. `L-nobc` 71.2: -3.82 from FP32, -0.42 [-2.02, 1.17] against `N-default`. **Not closed by either arm.**
`L-bc` minus `L-nobc`: +1.77 [0.17, 3.35]. INT8 weights alone cost nothing (`N-A32` and `L-A32` both 75.3), so the
remaining 2.05 (`L-bc`) to 3.82 (`L-nobc`) points come from INT8 activation quantization, which weight rounding cannot repair. (Posit8
weights and activations under `N-nobc` reach 74.8 on the same model, -0.2 [-1.3, 0.9] from FP32.)

**4-bit rescue.** Criterion: W4/A4 is rescued if a learned arm is within 5 points of FP32. Best learned W4/A4:
ResNet18 24.8 (`L-nobc`), MobileNetV2 27.2 (`L-nobc`), MobileNetV3 0.13 (`L-bc`). **Not rescued on any model.**
The side that fails is the activations: W4/A8 learned reaches 70.4, 71.5 and 69.35 and W4/A32 70.5, 71.7, 73.1. For
the literature comparison (W4/A8): ResNet18 70.4 against FP32 70.1 here (1024 images, 10000 iterations, one
seed), and 68.55 against 69.68 in the paper, whose W4/A8 number comes from its Table 7 setting (2048 images,
20000 iterations; W4/A32 68.71 in the same setting), so the published W4/A8 drop of 1.13 is for twice the data
and iterations used here. The default recipe is at 0.1 / 7.5 / 48.75 on the same cells (ResNet18's 0.1 is the
bias-correction collapse of section 3; nearest without bias correction gives 53.5). So AdaRound turns 4-bit weights from unusable into
nearly free on ResNet18 and MobileNetV2 and into a 5.7-point loss on MobileNetV3, but it does not make a 4-bit
activation format usable. With the activation-aware adaptation of section 7 the best W4/A4 cells are 27.8, 50.3
and 0.2: still not rescued.

**6-bit order.** Criterion: the order changes only if the sign of INT6 minus FP6 changes against `N-default`
and the new interval excludes zero. INT6 minus FP6 e2m3 (points):

| model | N-default | N-nobc | N-minimal | L-nobc | L-bc | weights only N / L |
|---|---|---|---|---|---|---|
| ResNet18 | -0.77 [-1.92, 0.42] | -1.48 [-2.90, -0.07] | -2.17 [-3.77, -0.66] | -1.68 [-2.93, -0.37] | +0.29 [-0.85, 1.39] | -0.4 / -1.1 |
| MobileNetV2 | +3.86 [2.35, 5.41] | +0.13 [-1.84, 2.08] | -0.32 [-2.20, 1.54] | -1.25 [-2.64, 0.14] | +1.61 [0.03, 3.18] | -0.6 / +0.6 |
| MobileNetV3 | -37.4 [-40.4, -34.5] | -31.4 [-34.1, -28.6] | -34.2 [-37.2, -31.2] | -40.1 [-43.2, -37.1] | -34.4 [-37.3, -31.4] | -0.6 / +0.2 |

**No order change on any model.** ResNet18 has no significant order under `N-default`, and `L-bc` flips the sign
without significance; MobileNetV2's INT6 lead survives `L-bc` (smaller) and loses significance under `L-nobc`;
MobileNetV3's FP6 lead is an activation effect (both formats within 0.6 point of each other with FP32 activations)
that learned weight rounding cannot touch. The best arm per format is within about 0.6 point on ResNet18 (INT6
68.1 `L-bc`, FP6 68.0 `N-nobc`/`N-minimal`) and MobileNetV2 (INT6 70.4 `L-bc`, FP6 71.0 `L-nobc`); these
cross-arm comparisons were not in the protocol and are not tested.

## 6. Fit statistics and cost

`results/summaries/b2-recon-v1/fits--r2.json` (per layer: calibration reconstruction error with nearest, learned
and FP32 weights, flipped weights, unsettled variables, data placement, seconds).

- In all 18 primary fits, no layer has a higher calibration reconstruction error with learned than with nearest
  rounding. Weights rounded differently from nearest: 8 percent (ResNet18 INT8) to 19 percent (MobileNetV2 INT4).
- Unsettled soft variables at the end: 0 to 0.24 percent at 6 and 8 bits, 0.57 to 3.73 percent at INT4.
- Cost: the summed layer time per fit was 6 to 36 minutes for ResNet18 (21 layers), 35 to 79 for MobileNetV2 (53)
  and 38 to 76 for MobileNetV3 (64), on a GPU shared with up to three other lanes; layers whose data did not fit
  beside the other jobs were held in host memory (0 to 25 layers per fit). The one ResNet18 fit that ran on a
  nearly free GPU (faithfulness seed 2) took 9.5 minutes; the paper reports about 10 minutes for ResNet18.
  A layer held in host memory can take up to 637 s, so a few jobs that started such a layer just before the
  600 s cut-off ran past the 15-minute guideline. Evaluation: 1 to 164 s per arm, 35 minutes for all 129 primary arms (plus 3 FP32 records).
- Hardware: learned rounding changes only which of two neighbouring codes each weight takes. Codebooks, scales,
  boundaries, accumulators and the datapath are those of B2.

## 7. Secondary: activation-aware reconstruction input (adaptation, partial)

Protocol v1 listed this as a secondary group: `x_hat` (the reconstruction input) is taken from the B2 engine with
the activation boundaries quantized (recipe `default_no_bias_correction`, activation format = weight format), so the
rounding is learned against the activation quantization it will meet. **This is an adaptation, not the published
method.** Six fits (three models x INT4, INT6), run 2026-10-02 in queues F, G and H; arms `LQ-nobc` (no bias
correction) and `LQ-bc` (B2 bias correction afterwards). Rows `LQ-*` in `matrix-1k--r2.csv`.

| model, W/A | LQ-nobc | LQ-bc | LQ-nobc minus L-nobc | LQ-bc minus L-bc | LQ best minus FP32 |
|---|---|---|---|---|---|
| ResNet18 W4/A4 | 27.8 (27.2) | 0.8 (1.6) | +2.93 [1.30, 4.55] | -2.17 [-2.88, -1.54] | -42.3 |
| ResNet18 INT6 | 66.0 (66.5) | 68.1 (67.4) | -0.03 [-0.94, 0.86] | 0.00 [-0.95, 0.97] | -1.98 [-3.22, -0.76] |
| MobileNetV2 W4/A4 | **50.3 (49.8)** | 7.6 (8.3) | +23.1 [20.7, 25.5] | +7.37 [5.98, 8.81] | -21.8 [-24.5, -19.2] |
| MobileNetV2 INT6 | 70.7 (70.8) | **71.8 (71.1)** | +1.01 [-0.28, 2.33] | +1.33 [0.19, 2.52] | -0.33 [-1.63, 0.98] |
| MobileNetV3 W4/A4 | 0.2 (0.1) | 0.13 (0.2) | +0.15 [-0.04, 0.36] | 0.00 [-0.08, 0.09] | -74.8 |
| MobileNetV3 INT6 | 29.5 (29.5) | 26.2 (27.0) | +24.2 [21.5, 26.9] | +2.79 [0.44, 5.15] | -45.5 [-48.6, -42.4] |

Learning against quantized activations helps where activations are the problem and the network is not already
at chance: MobileNetV2 W4/A4 rises from 27.2 to 50.3, MobileNetV3 INT6 from 5.3 to 29.5, and MobileNetV2 INT6 with
bias correction comes within 0.3 point of FP32. It does nothing on ResNet18 INT6 and on MobileNetV3 W4/A4 (still at
chance), and adds 3 points on ResNet18 W4/A4 (27.8, equal to nearest without bias correction). No W4/A4 cell comes
within 5 points of FP32, so the answer to the 4-bit question does not change; MobileNetV3 INT6 remains broken by its
6-bit activations. Fit statistics: no layer worse than nearest except two layers of the MobileNetV2 INT4 fit;
17.0 to 29.5 percent of the weights flipped; 0.2 to 5.5 percent of the soft variables not binary at the end.

## 8. Limits, and what block-wise reconstruction would add

- One seed per matrix cell; the faithfulness seeds differ by up to 0.7 point, so single-cell differences below
  about 1 point between learned arms are not resolved even when an interval excludes zero.
- The screen contains dev512, on which the B2 default recipe was chosen. No setting of this lane was chosen on the
  screen (all fixed in the protocol), but every number here is development evidence.
- The rounding is learned with FP32 activations (as in the AIMET procedure, from memory); it does not know the
  activation quantization, which is where the remaining losses are.
- Five settings are from memory of AIMET (section 1).
- Block-wise reconstruction (BRECQ, Li et al. 2021) is out of scope. It replaces the per-layer target by the output
  of a whole residual block or inverted bottleneck, weighted with a Fisher-information estimate, so that the
  rounding errors of layers inside one block can cancel; it can also learn the activation step sizes jointly. In
  the literature this is the main gain over layer-wise AdaRound for MobileNet-type networks at 4 bits and below.
  Here the weights-only results leave little for it to gain at W4/A32 (within 0.4 point on ResNet18 and
  MobileNetV2, 1.9 on MobileNetV3); its possible gain is at W4/A4 and on MobileNetV3's activations, which would
  require learning activation scales (a change to the B2 activation recipe, not only to weights).

## 9. Not run

- Secondary group `maxabs` (INT4 learned on the minimal recipe's max-abs weight scales): not run. The minimal
  recipe is the stronger nearest arm only on ResNet18 W4/A4, where nearest without bias correction is stronger still.
- Secondary group `b2in` for FP6 and Posit8 (not in the protocol), and every secondary arm with INT8 activations.
- 5 seeds (3 run for the faithfulness setting, 1 for the matrix); the 2048-image / 20000-iteration setting of
  Table 7; the paper's MobileNetV2 W4 setting with cross-layer equalisation.

## 10. How to reproduce

    artifacts/agent_orchestration/gpu_run.sh .venv-b/bin/python -m tools.run.experiment_b2_recon fit \
        --model resnet18 --wformat int4 --rule mse_per_layer --seed 0     # exit 3 = budget reached, run again
    artifacts/agent_orchestration/gpu_run.sh .venv-b/bin/python -m tools.run.experiment_b2_recon eval \
        --model resnet18 --group faithful                                  # groups: matrix faithful b2in maxabs
    .venv/bin/python -m tools.run.experiment_b2_recon report --tag r2 --figure   # CPU, about 90 s
    artifacts/agent_orchestration/gpu_run.sh .venv-b/bin/python -m tools.experiment_b2_recon.diagnose_bias
    .venv/bin/python -m tools.experiment_b2_recon.collapse_check --tag r3   # CPU, seconds; refuses to overwrite
    .venv/bin/python -m tools.experiment_b2_recon.collapse_ties readout --dry-run   # r4 verdicts + GPU worklist
    artifacts/agent_orchestration/gpu_run.sh .venv-b/bin/python -m tools.experiment_b2_recon.collapse_ties exact \
        MODEL LABEL ...                                                 # full maxima sets of undetermined records
    .venv/bin/python -m tools.experiment_b2_recon.collapse_ties readout --tag r4   # writes the sealed r4 file once

Queue job lists of all runs: `artifacts/experiment_b2_recon/logs/queue-*.jobs` (queue A: faithfulness, ResNet18
and MobileNetV2; B: MobileNetV3; C: MobileNetV3 INT8 rerun after an out-of-memory failure; D/E: first b2in queues,
stopped for a trim; F/G/H: b2in). Every failed job in queues A and B was a CUDA out-of-memory error while other lanes
used the GPU; each was refilled by a later job (an evaluation fills only missing arms, a fit resumes per layer).
