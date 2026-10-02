# B2 collapse diagnosis: bias correction, posit under max-abs, wide-exponent formats under default (2026-10-02)

Lane Q3. Protocol `public/experiments/configs/breadth-study/b2-collapse-protocol-v1.json`
(sha256 `3c474583fbbbb77cca702bba7fffcabe7be07343f5acff5adbbb7c5880fe5045`, written 2026-10-02T11:39:55+05:00,
before any new measurement; hypotheses H1.0-H1.3, H2.1-H2.4, H3.1-H3.2 and the decision rules are stated there).
Addenda (written before the cells they govern): `b2-collapse-protocol-v1-addendum-1.json` (sha256 `6872c72d...3606`,
2026-10-02T16:21:37+05:00) and `-addendum-2.json` (sha256 `b1b40419...6eb1`, 16:23), which move the MXFP6 max-abs-weight
cells to a low-memory, bit-identical execution of the block bias correction (section 3.1); nothing measured changes.
`-addendum-3.json` (sha256 `4897a3b7...7e9b`, 2026-10-02T17:43, written after independent review 1 and before the
measurement it governs) adds one diagnostic on the correction images: the input-mean drift at every conv/linear input
(section 1.4).

**Revision r6 (after independent review 2, same file, section "Review 2").** No accuracy, summary or figure changes;
text only, every number re-read from the sealed records. Corrected: regime (ii) no longer says the weights-only pass
"repairs" the loss in all its integer cells: it avoids the collapse in all four, but repairs the loss against no
correction only on ResNet18 W4/A4 learned and MobileNetV2 W5/A5 (and in the six scalar mode-3 cells); on ResNet18
W5/A5 and MobileNetV2 W4/A4 learned it stays below no correction (short answer 1, sections 1.2 and 1.5, verdict,
recommendations 1 and 2). The attribution of the rule-(c) losses to the quantized-activation pass is shown for six of
the eight wide-exponent losses and inferred for the two MXFP6 cells (short answer 3, mode-3 verdict, recommendations 2
and 4). Limits now state that the regime boundaries are a post hoc reading of pre-specified arms on few cells, cite the
original-path identity check of the MobileNetV2 MXFP6 max-abs cell, the source hashes of the weight-placement record and
the brief overlap of two lane GPU jobs. Section 3.2 separates the largest always-off fraction from the largest excess
over FP32. Recommendation 2(a) no longer calls the 5-bit losses collapses.

**Revision r5 (after independent review 1, `artifacts/agent_orchestration/handoffs/Q3-collapse-diagnosis-review.md`).**
No accuracy changes. Corrected: the literal weights-only Appendix D pass is not "also collapsing" in general: it
collapses only at 4-bit nearest weights, avoids the collapse with learned 4-bit weights and at 5 bits, and repairs the
wide-exponent losses, so the verdicts
now name three regimes (short answers 1 and 3, sections 1.2, 1.5, 3.2, verdicts, recommendations); the mechanism
direction (section 1.4: the late inputs are attenuated, not raised; the correction lowers biases and the drift
compounds) now rests on sealed per-layer records plus the new drift records; the log6 row of table 2.2; the ResNet18
flagged-layer list; H3.1's mechanism clause holds on MobileNetV2 only; the cost range of recommendation 2(a).
Summaries: `results/summaries/b2-collapse-v1/*--{r1,r2}.{json,csv}` (each written once). **r2 supersedes r1 only for
MXFP6 E3M2**: it adds the three MXFP6 decomposition cells (ResNet18 max-abs weights, MobileNetV2 no correction and
max-abs weights); every mode-1 and mode-2 row and every other mode-3 number is identical in r1 and r2 (the four
MobileNetV3 max-abs rows now cite the reference `--b2` arm instead of its bit-identical lean twin; same values).
Figures: `results/figures/b2-collapse-{mode1,layers,mode2,mode3}--r2.{pdf,png}` (r1 kept; the text cites r2). Records:
`artifacts/experiment_b2_collapse/evals/`, for the redirected matrix cells `artifacts/experiment_b2_collapse/matrix/`,
and for the input-drift diagnostic `artifacts/experiment_b2_collapse/drift/input-drift--{resnet18,mobilenet_v2}--r1.json`.

**Evidence class.** Every accuracy here is measured on the frozen ImageNet 1k screen (`imagenet_screen_1k`, which
contains the dev512 images B2 was tuned on): development evidence only. Corrections and weight scales are computed on
the first 256 images of `imagenet_calibration_2k` (the existing B2 bias-correction cache entry, `build=False`) and the
B2 v1 calibration observations; learned rounding reuses the L6 fits read-only (no new AdaRound fit). Readout:
expected-credit top-1 (credit 1/k when the label is among k tied maxima); "collapse" (marked **c**) = one class
among the maxima on at least 90 percent of the images. Differences: paired image bootstrap of the expected credit
(`tools.analysis.b2_ties.paired`, 10,000 resamples, seed 20260927, pointwise 95 percent intervals, no multiplicity
correction). One seed, one correction set.

**Verdict vocabulary (protocol).** Defect = the code does not do what the cited method prescribes. Property = the
faithful method fails in this regime. Fragile design = a defensible recipe choice that fails for a family of
formats and has a cheap alternative.

## Short answers

1. **Mode 1, bias correction: no code defect; three regimes.** B2's `bias_correct` is a faithful, bit-reproduced
   implementation of DFQ Appendix D with one deviation that the protocol stated in advance (H1.0): it runs the
   correction pass with quantized activations, where Appendix D quantizes the weights only. The literal weights-only
   pass (`dfq_weights_only`) separates three regimes:
   (i) **4-bit nearest weights: property of the method.** The weights-only pass collapses as B2 does (ResNet18 W4 with
   FP32, 8-bit and 4-bit activations 0.1, 0.1 and 0.05; MobileNetV2 W4/A4 0.6, one class on 99-100 percent of the
   images): the global (Appendix D) correction fails at this weight error. MobileNetV3-Large W4/A4 is at chance
   (0.1-0.2) with or without any correction.
   (ii) **Learned 4-bit weights, 5 bits and the wide-exponent formats of mode 3: fragile design.** In all of these
   cells with that arm (the two MXFP6 cells have none) the weights-only pass does not collapse and is above B2's pass
   (+2.3 to +42.7 points, every paired interval
   excluding zero), so B2's choice to run the pass with quantized activations is the cause of B2's loss there. The
   weights-only pass repairs the loss against no correction on ResNet18 W4/A4 learned (45.7 against B2's 3.0,
   +42.7 [+40.2, +45.2]; no correction 24.8), on MobileNetV2 W5/A5 (65.4 against 32.5; no correction 36.6) and in
   every scalar mode-3 loss cell (short answer 3). In the two other integer cells it only avoids the collapse:
   ResNet18 W5/A5 51.3 against B2's 13.7 but 4.9 points below no correction (56.2; -4.9 [-7.3, -2.4]), and
   MobileNetV2 W4/A4 learned 25.3 against 0.2 (+25.0 [+22.7, +27.4]) but 1.9 below no correction (27.2; -1.9
   [-4.5, +0.7]), with one class among the maxima on 52.8 percent of the images (2.8 percent uncorrected).
   (iii) **MobileNetV3-Large integers: the same choice is what helps.** INT6 21.0 against 4.4 with the weights-only
   pass (3.5 uncorrected), INT5 8.1 against 0.6, INT8 learned 73.0 against 71.0.
   Mechanism in the collapse cells (section 1.4): the late inputs of the quantized network are smaller than the FP32
   ones, and through the layer's weights that deficit raises the pre-activation means (the inherited term of the
   correction is negative). The sequential correction lowers each bias, which switches part of the layer off,
   attenuates the next layer's input further and enlarges the next correction. In the ResNet18 faithfulness cell the
   inherited term at layer4_1_conv1 grows from -0.09 uncorrected to -0.45, the image-to-image spread of its
   pre-activation falls to 0.12 of FP32, and 96 percent of its channels are off on every correction image; the logit
   spread is 0.045 of FP32 and top-1 0.1. Not a sample-size effect (32 to 256 images). The own-error (local) form that
   AdaRound's baseline uses gives 37.7 (paper 38.87) where B2 gives 0.1, never collapses, and lifts MobileNetV2 W4/A4
   nearest from 1.1 to 20.9 and W5/A5 from 36.6 to 58.9; but it gives up the MobileNetV3 gains (INT6 4.2 against
   21.0) and loses 10 points on MobileNetV2 W4/A4 learned.
2. **Mode 2, posit under max-abs: fragile design, no slip.** Tables and scale mapping are exact; max-abs maps each
   channel maximum to maxpos (4096 / 256), where the posit grid is regime-only (weights from 0.19 to 0.63 of the
   maximum all become 0.25), so weight SQNR is about 10 dB in every layer. Anchoring the maximum at 1.0 brings posit8
   within 1.5 points of the MSE scale on all three networks; the one-class output on MobileNetV2 is not driven by a
   small node group but by the accumulated error (any stage group alone suffices).
3. **Mode 3, wide-exponent formats under default: bias correction is the switch (H3.1), not the weight range; the
   cause is B2's quantized-activation correction pass (regime (ii) of mode 1), a fragile design (shown for the six
   scalar cells; inferred for the two MXFP6 cells, where no weights-only arm was run).**
   Removing bias correction alone recovers 88 to 167 percent of the minimal-minus-default loss for FP8 E5M2, FP6 E3M2
   and Log6 on ResNet18 and MobileNetV2; removing the MSE weight search recovers at most 40 percent. The literal
   weights-only Appendix D correction recovers 109 to 217 percent in all six cells (63.0 to 63.9; +2.3 [+0.8, +3.9]
   to +42.7 [+39.6, +45.7] against default), and the own-error corrections reach 61.7 to 65.3, above or at both
   recipes (MobileNetV2 FP8 E5M2: default 23.2, minimal 41.9, weights-only 63.9, local 65.0). MXFP6 E3M2 follows the
   same rule on both networks (the correction variants were not run on this block format): removing bias
   correction recovers 99 percent of the loss (ResNet18 64.2 against default 51.6, minimal 64.4; MobileNetV2 41.7
   against 7.2, minimal 41.9, where the default cell predicts one class for 78 percent of the images), while max-abs
   weights recover nothing (50.7, -7 percent; 8.4, 3 percent).

## 1. Mode 1: bias correction

### 1.1 What the method prescribes, and what B2 does (line by line)

Sources read: Nagel et al. 2019 (DFQ, arXiv 1906.04721), section 4.2 and Appendix D; Nagel et al. 2020 (AdaRound,
arXiv 2004.10568), Table 8 and its bias-correction equation. Code read (read-only):
`tools/experiment_b2/engine.py::bias_correct`, its caller `prepare_b2`, and the L6 builder
`tools/experiment_b2_recon/engine.py::build_engine`.

| item | DFQ (App. D, empirical form) | B2 `bias_correct` | finding |
|---|---|---|---|
| reference graph | FP32 network | FP32 graph (`fp_graph`) fed FP32 inputs | same |
| quantized graph during the pass | **quantized weights only**; activations quantized afterwards | quantized weights **and** quantized activations at every planned boundary | deviation (tested: variant `dfq_weights_only`) |
| inputs | N calibration images | first 256 of `imagenet_calibration_2k`, chunks of 32 | same kind; N tested at 32/64/128/256 |
| statistic | per-output-channel mean of the pre-activation | mean over images and positions of the conv/linear output, float64 sums, channel axis 1, dims (0,2,3) or (0) | same |
| sign | subtract E[y_q] - E[y_fp] from the bias | `bias += mean_fp - mean_q` | same |
| order | topological; each layer after all producers are corrected | node order of the FX graph; the stored quantized outputs are shifted by the same vector, so every consumer sees corrected inputs | same (shift = recompute: variant `executed` gives the same result) |
| where it lands | bias of the layer, before the nonlinearity | folded conv bias; ReLU/ReLU6 is a separate node after it (fused boundary), so the shift is pre-activation | same |
| applied twice? | no | each conv/linear module is called by exactly one node (checked; `staged_correct` refuses shared modules) | no |

AdaRound's Table 8 lists "bias correction (Nagel et al. 2019)" on ResNet18 with 4-bit weights, one scale per layer
and FP32 activations: nearest 23.99, with empirical bias correction 38.87. Its bias-correction equation uses the
same input x in both terms, E[Wx] - E[W_q x]: the layer's own weight error only (a "local" form). DFQ's Appendix D
instead corrects the full output mean difference, which also contains the error inherited from upstream layers
(a "global" form). With y = Wx and the quantized network's input x_q:

E[y_fp] - E[y_q] = E[(W - W_q) x_q] (own error, "local") + E[W (x_fp - x_q)] (inherited drift).

B2 implements the global form. The lane re-implements it operation by operation (`correct.staged_correct`, policy
`global`) and checks bit-identity of the corrected biases against `bias_correct` (unit tests on int4, int6,
posit8, fp8_e5m2, log6, posit6, fp6_e3m2) and of the deployed state and 1000/1000 top-5 lists against the L6 and
matrix records (field `reproduction_check` of each record).

### 1.2 Fidelity (H1.0): no defect

- The lane's independent re-implementation (`global`) gives bit-identical corrected biases to `bias_correct` in the
  unit tests and reproduces the existing records: deployed-state SHA-256 equal and 1000/1000 top-5 lists equal for
  every `global` and `none` arm with an L6 counterpart (ResNet18 faithfulness cell, W4/A8, W4/A4 nearest and learned;
  MobileNetV2 W4/A4 learned; MobileNetV3 INT8 learned), and 1000/1000 top-5 lists against the matrix cells where only
  those exist (W5/A5, W6/A6).
- Recomputing each corrected layer instead of shifting its stored output (`executed`) gives the same top-1 on the
  ResNet18 cells (0.1 / 0.13) and 72.4 against 71.6 on MobileNetV3 INT8 nearest (float32 rounding at the
  activation-code boundaries, then different downstream corrections; within the screen's noise, see the paired table).
- The one deviation from DFQ Appendix D (quantized activations during the pass) is a recipe choice stated in the
  protocol (H1.0), not a defect, but it decides the outcome outside 4-bit nearest weights. The literal weights-only
  pass (`dfq_weights_only`) also collapses ResNet18 W4/A8 (0.1), ResNet18 W4/A4 nearest (0.05) and MobileNetV2 W4/A4
  nearest (0.6), and at FP32 activations it is the same computation as B2 (0.1); it does not collapse with learned
  4-bit weights (ResNet18 45.7, MobileNetV2 25.3 against B2's 3.0 and 0.2) or at 5 bits (51.3 and 65.4 against 13.7
  and 32.5), and it repairs every scalar mode-3 loss (section 3). So the deviation is the cause of B2's loss there;
  but the weights-only pass is not better than no correction in every such cell: it stays below it on ResNet18 W5/A5
  (-4.9 [-7.3, -2.4]) and MobileNetV2 W4/A4 learned (-1.9 [-4.5, +0.7]; one class among the maxima on 52.8 percent of
  the images). On MobileNetV3-Large the deviation is what makes the correction help (sections 1.5 and 4).

### 1.3 Results (ResNet18, MobileNetV2, MobileNetV3-Large; 1k screen, development evidence)

Top-1 (expected credit, percent); **c** = collapse.

| network / cell | none | B2 global | executed | DFQ App. D (W only) | cap RMS | linear-output only | local (own error, x_q) | analytic (own error, x_fp) |
|---|---|---|---|---|---|---|---|---|
| R18 faith-w4-a32-perlayer-N | 19.0 | 0.1**c** | 0.1**c** | - | 1.3 | 44.9 | 37.7 | 42.6 |
| R18 w4-a4-default-L | 24.8 | 3.0**c** | - | 45.7 | 29.8 | 20.1 | 25.6 | 25.9 |
| R18 w4-a4-default-N | 27.8 | 0.1**c** | 0.1**c** | 0.1**c** | 1.3**c** | 24.8 | 30.1 | 31.8 |
| R18 w4-a8-default-N | 53.5 | 0.1**c** | - | 0.1**c** | 11.4 | 63.0 | 59.2 | 60.0 |
| R18 w5-a5-default-N | 56.2 | 13.7 | - | 51.3 | 39.7 | 52.1 | 55.2 | 54.9 |
| R18 w6-a6-default-N | 66.5 | 65.9 | - | 67.8 | 65.9 | 66.0 | 65.8 | 65.8 |
| MBv2 w4-a4-default-L | 27.2 | 0.2**c** | - | 25.3 | 0.4**c** | 10.3 | 17.0 | 20.4 |
| MBv2 w4-a4-default-N | 1.1 | 0.1**c** | 0.1**c** | 0.6**c** | 0.1**c** | 0.5**c** | 20.9 | 22.1 |
| MBv2 w5-a5-default-N | 36.6 | 32.5 | - | 65.4 | 34.1 | 52.5 | 58.9 | 58.7 |
| MBv2 w6-a6-default-N | 67.1 | 70.1 | - | 71.7 | 69.8 | 69.9 | 71.0 | 70.3 |
| MBv3-L w4-a4-default-N | 0.2 | 0.1 | 0.1 | 0.1**c** | 0.1**c** | 0.1**c** | 0.2 | 0.1 |
| MBv3-L w5-a5-default-N | 0.6 | 8.1 | - | 0.6 | 6.5 | 4.4 | 1.0 | 1.1 |
| MBv3-L w6-a6-default-N | 3.5 | 21.0 | - | 4.4 | 20.0 | 23.4 | 4.2 | 4.3 |
| MBv3-L w8-a8-default-L | 71.2 | 73.0 | - | 71.0 | 71.7 | 73.2 | 71.2 | 71.1 |
| MBv3-L w8-a8-default-N | 70.5 | 71.6 | 72.4 | 71.0 | 71.2 | 72.0 | 70.2 | 70.8 |

Correction images (B2 global on the first N of the 256 images):

| network / cell | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| R18 faith-w4-a32-perlayer-N | 0.1 (58%) | 0.1 (100%) | 0.1 (100%) | 0.1 (100%) |
| R18 w4-a4-default-N | 0.1 (100%) | 0.1 (100%) | 0.1 (100%) | 0.1 (100%) |
| MBv2 w4-a4-default-N | 0.1 (100%) | 0.1 (100%) | 0.1 (100%) | 0.1 (100%) |
| MBv3-L w8-a8-default-N | 71.2 (0%) | 71.6 (0%) | 71.5 (0%) | 71.6 (0%) |

Paired differences against no correction (points, 95% interval):

| network / cell | global | dfq_weights_only | linear_only | local_empirical | analytic_fp32 |
|---|---|---|---|---|---|
| R18 faith-w4-a32-perlayer-N | -18.9 [-21.3, -16.4] | - | +25.9 [+22.8, +29.0] | +18.7 [+16.0, +21.5] | +23.6 [+20.8, +26.5] |
| R18 w4-a4-default-L | -21.8 [-23.8, -19.9] | +20.9 [+18.8, +22.9] | -4.8 [-6.2, -3.3] | +0.7 [-0.2, +1.7] | +1.1 [-0.0, +2.1] |
| R18 w4-a4-default-N | -27.6 [-29.8, -25.5] | -27.7 [-29.9, -25.6] | -3.0 [-5.1, -0.8] | +2.3 [+0.3, +4.2] | +4.0 [+2.1, +5.9] |
| R18 w4-a8-default-N | -53.4 [-56.4, -50.2] | -53.4 [-56.4, -50.2] | +9.6 [+7.2, +11.9] | +5.8 [+3.5, +8.0] | +6.5 [+4.2, +8.8] |
| R18 w5-a5-default-N | -42.5 [-45.5, -39.5] | -4.9 [-7.3, -2.4] | -4.1 [-6.0, -2.2] | -1.0 [-2.8, +0.8] | -1.2 [-3.0, +0.6] |
| R18 w6-a6-default-N | -0.6 [-2.4, +1.1] | +1.3 [-0.1, +2.7] | -0.5 [-1.6, +0.6] | -0.7 [-1.8, +0.4] | -0.7 [-1.7, +0.3] |
| MBv2 w4-a4-default-L | -26.9 [-28.7, -25.2] | -1.9 [-4.5, +0.7] | -16.8 [-19.1, -14.6] | -10.2 [-11.7, -8.8] | -6.8 [-8.2, -5.3] |
| MBv2 w4-a4-default-N | -1.0 [-1.7, -0.5] | -0.6 [-1.3, +0.1] | -0.6 [-1.3, +0.0] | +19.8 [+18.2, +21.4] | +21.0 [+19.3, +22.8] |
| MBv2 w5-a5-default-N | -4.1 [-7.1, -1.2] | +28.8 [+25.9, +31.6] | +15.8 [+13.0, +18.7] | +22.3 [+19.6, +25.0] | +22.1 [+19.4, +24.9] |
| MBv2 w6-a6-default-N | +3.0 [+1.0, +4.9] | +4.5 [+2.8, +6.4] | +2.7 [+0.8, +4.6] | +3.8 [+2.1, +5.5] | +3.2 [+1.6, +4.9] |
| MBv3-L w4-a4-default-N | -0.1 [-0.4, +0.1] | -0.1 [-0.4, +0.1] | -0.1 [-0.4, +0.1] | -0.0 [-0.2, +0.2] | -0.1 [-0.3, +0.2] |
| MBv3-L w5-a5-default-N | +7.5 [+6.0, +9.1] | +0.0 [-0.5, +0.4] | +3.8 [+2.7, +5.0] | +0.4 [-0.3, +1.1] | +0.6 [-0.1, +1.3] |
| MBv3-L w6-a6-default-N | +17.6 [+15.1, +19.9] | +0.9 [-0.0, +1.9] | +20.0 [+17.4, +22.6] | +0.7 [-0.3, +1.7] | +0.8 [-0.2, +1.9] |
| MBv3-L w8-a8-default-L | +1.8 [+0.2, +3.4] | -0.1 [-1.4, +1.1] | +2.1 [+0.5, +3.6] | +0.1 [-1.3, +1.4] | -0.1 [-1.4, +1.2] |
| MBv3-L w8-a8-default-N | +1.1 [-0.6, +2.9] | +0.6 [-0.7, +1.8] | +1.6 [-0.1, +3.2] | -0.3 [-1.7, +0.9] | +0.3 [-1.0, +1.6] |

### 1.4 Mechanism (H1.1) and sample size (H1.2)

| network / cell | layers with correction RMS > FP32 mean RMS | median inherited/local there | worst always-off layer: FP32 / uncorrected / corrected | logit spread corrected/FP32 (uncorrected/FP32) |
|---|---|---|---|---|
| R18 faith-w4-a32-perlayer-N | 4/21 | 23.7 | layer4_1_conv1: 0% / 0% / 96% | 0.04 (0.56) |
| R18 w4-a4-default-L | 3/21 | 49.3 | layer4_1_conv1: 0% / 0% / 59% | 0.17 (0.94) |
| R18 w4-a4-default-N | 4/21 | 34.3 | layer4_1_conv1: 0% / 0% / 97% | 0.07 (0.75) |
| R18 w4-a8-default-N | 3/21 | 35.2 | layer4_1_conv1: 0% / 0% / 94% | 0.06 (0.80) |
| R18 w5-a5-default-N | 3/21 | 41.3 | layer4_1_conv1: 0% / 0% / 31% | 0.19 (0.87) |
| R18 w6-a6-default-N | 0/21 | - | conv1: 12% / 12% / 12% | 0.60 (0.99) |
| MBv2 w4-a4-default-L | 16/53 | 0.7 | features_18_0: 0% / 0% / 100% | 0.02 (0.97) |
| MBv2 w4-a4-default-N | 17/53 | 0.5 | features_18_0: 0% / 51% / 100% | 0.01 (0.43) |
| MBv2 w5-a5-default-N | 14/53 | 0.4 | features_18_0: 0% / 3% / 90% | 0.13 (0.90) |
| MBv2 w6-a6-default-N | 8/53 | 0.6 | features_16_conv_1_0: 13% / 16% / 25% | 0.52 (0.96) |
| MBv3-L w4-a4-default-N | 21/64 | 1.9 | features_15_block_2_fc1: 0% / 22% / 32% | 0.08 (0.46) |
| MBv3-L w5-a5-default-N | 17/64 | 1.9 | features_15_block_2_fc1: 0% / 14% / 20% | 0.28 (0.64) |
| MBv3-L w6-a6-default-N | 15/64 | 2.9 | features_15_block_2_fc1: 0% / 11% / 16% | 0.28 (1.12) |
| MBv3-L w8-a8-default-L | 8/64 | 14.6 | features_15_block_2_fc2: 0% / 0% / 0% | 0.93 (1.01) |
| MBv3-L w8-a8-default-N | 9/64 | 6.7 | features_15_block_2_fc2: 0% / 0% / 0% | 0.93 (1.02) |

- **ResNet18** (figure `b2-collapse-layers--r2`): in the faithfulness cell the global correction exceeds the RMS of the
  FP32 channel means in 4 of 21 layers (layer2_0_conv2 and the last-stage layer4_0_conv2, layer4_1_conv1,
  layer4_1_conv2; in the other ResNet18 W4 and W5 cells only last-stage layers, plus fc in W4/A4 nearest); there the
  inherited part is 24 times the layer's own weight-error part (median). Direction (sealed per-layer records, and the
  input-drift records of addendum 3): at the late layers the inherited term E[W (x_fp - x_q)] is negative, so the
  quantized pre-activations sit above the FP32 ones, while the inputs themselves are smaller than in FP32
  (uncorrected, in the three W4 cells measured (faithfulness, W4/A4 nearest and learned) and at W5, the mean
  drift E[x_q] - E[x_fp] is negative at the
  inputs of layer4_0_conv2, layer4_1_conv1 and fc: faithfulness cell layer4_1_conv1 -0.020 against an FP32 input mean
  of 0.069, 21 percent of channels positive, fc -0.33 against 0.92; layer4_1_conv2 is mixed, 22 to 62 percent of
  channels positive with a mean within 0.006 of zero). The sequential correction lowers each bias to remove that excess; every lowered layer
  switches part of its output off, which attenuates the next layer's input further and enlarges the next correction
  (under the global correction every listed late input is attenuated on all channels, 0 percent
  positive, in every measured ResNet18 cell (FP6 E3M2 included): layer4_1_conv1 -0.034 to -0.064 against 0.069, fc -0.45 to -0.86 against 0.92;
  addendum-3 hypothesis D1 holds at 16 of 16 late inputs, D2 (uncorrected inputs not raised on more than 60 percent
  of channels) at 19 of 20, the exception being layer4_1_conv2 in the learned cell, 62 percent positive with a mean
  of +0.002; all 10 rebuilt arms have the sealed records' deployed-state SHA-256). H1.1's three quantitative
  predictions hold; its stated cause, "upstream noise rectified by ReLU raises input means", is rejected. From the end of stage 3 into the last stage the applied mean shift grows: -0.13 (layer3_1_conv2),
  -0.14 (layer4_0_conv1), -0.33 (layer4_0_conv2), -0.45 (layer4_1_conv1 and layer4_1_conv2); at layer4_1_conv1 the inherited term is -0.45 against
  -0.09 in the uncorrected network, and the image-to-image spread of its pre-activation (unchanged by a constant
  shift, so it measures what upstream corrections removed) is 0.017 against 0.090 uncorrected and 0.146 in FP32. A
  shift of -0.45 against so small a spread puts nearly every channel below the ReLU threshold on every image:
  96 percent always-off channels (FP32 0 percent, uncorrected 0.2 percent), and the image-to-image spread of the
  logits falls to 0.045 of FP32 (uncorrected 0.56). The earlier version of this document said that rectified
  upstream noise raises the input means and that the correction cancels a positive drift; the measurements
  contradict that: the inputs are attenuated and the correction compounds the attenuation. The same pattern holds in
  all four ResNet18 W4 cells (W4/A4 nearest: layer4_1_conv1 inherited -0.46 against -0.12 uncorrected, spread 0.014,
  97 percent off; learned: -0.37 against -0.06, spread 0.051, 59 percent off) and, weaker, at W5 (inherited -0.34
  against -0.04, spread 0.061, 31 percent off, logit spread 0.19, one class on 77 percent of the images, 13.7 top-1). Causal check (independent reviewer, post hoc,
  `artifacts/experiment_b2_collapse/review-scratch/indep_dfq.json`, own hook-based code on B2's weights): the global
  correction with the corrections of layer2_0_conv2, layer4_0_conv2 and layer4_1_conv2 skipped (the layers whose
  correction RMS exceeded the FP32 mean RMS in that pass) gives 20.1 instead of 0.1 (no correction 19.0), and with
  layer4 and fc left uncorrected 21.8: the few layers whose correction exceeds the FP32 mean RMS carry the collapse.
- **MobileNetV2**: the damage sits at the linear bottleneck outputs. In W4/A4 learned, the corrections of
  features_16_conv_2 and features_17_conv_2 (no nonlinearity after them) have RMS 0.94 and 1.31 against FP32
  channel-mean RMS 0.30 and 0.32, dominated by inherited drift (0.84 and 1.43 against own error 0.49 and 0.31); the
  shifted residual stream switches off 99.9 percent of the channels of features_18 (the last 1x1 conv with ReLU6;
  0 percent uncorrected), so the pooled features no longer depend on the image (logit spread 0.02 of FP32). Over all
  layers whose correction exceeds the FP32 mean RMS the inherited/own ratio is 0.7 (bottleneck projections carry
  large own error too), so the H1.1 prediction "inherited exceeds local in those layers" is not met on MobileNetV2
  as stated, but the collapse is again driven by chasing inherited drift at a few late layers. Direction
  (addendum-3 records): under the global correction the classifier input (the pooled features_18 output) falls by
  0.104 against an FP32 mean of 0.105 in both W4/A4 cells (0 percent of channels positive), and the inputs of
  features_17_conv_1_0 and features_17_conv_2 fall by 0.12 to 0.13 (0-1 and 12 percent of channels positive);
  uncorrected these inputs are mixed (15 to 41 percent positive, means -0.002 to -0.063). D1 as stated (at most 10
  percent of channels positive) fails at features_17_conv_2 in all three MobileNetV2 cells (11 to 12 percent), although
  its mean drift is negative; D2 holds at 9 of 9 late inputs; the 6 rebuilt arms match the sealed deployed states.
- **MobileNetV3-Large**: the global correction helps where B2 needs it (INT5 +7.5, INT6 +17.6, INT8 learned
  +1.8 points); there the inherited drift comes from activation quantization of the hard-swish / SE network and the
  own weight error is small, so the local forms do nothing (INT6: local 4.2, analytic 4.3, against 3.5 uncorrected).
  Lane Q4 owns the INT6 activation side; this lane only reports the correction dimension.
- **Sample size (H1.2) holds**: on ResNet18 W4/A4 nearest the collapse is present with 32, 64, 128 and 256
  correction images; in the faithfulness cell top-1 is 0.1 at every size (one-class share 58 percent at 32 images,
  100 percent from 64); on MobileNetV2 W4/A4 nearest 100 percent at every size; on MobileNetV3 INT8 the benefit is
  stable from 32 images (71.2 to 71.6). It is not a noisy-estimate problem: the correction estimates the drift
  correctly; cancelling it is what fails.

### 1.5 The repaired variants (H1.3)

- `local_empirical` (the layer's own error on the quantized input; AdaRound's form) never collapses, reproduces the
  AdaRound/DFQ baseline number in the faithfulness cell (37.7 against the paper's 38.87; nearest 19.0 here against the
  paper's 23.99), and beats no correction in the ResNet18 W4 cells (W4/A8 +5.8, W4/A4 nearest +2.3) and on MobileNetV2
  W4/A4 nearest (+19.8) and W5/A5 (+22.3). It loses 10.2 points on MobileNetV2 W4/A4 learned and gives up the
  MobileNetV3 gains.
- `analytic_fp32` (own error on the FP32 input, DFQ section 4.2 with empirical input means) behaves like the local
  form (faithfulness 42.6; MobileNetV2 W4/A4 learned -6.8).
- `linear_only` (correct only conv/linear outputs that feed no rectifier; per-layer-group correction) is the best arm
  on ResNet18 (faithfulness 44.9, W4/A8 63.0 against 53.5 uncorrected) and on MobileNetV3 (INT6 23.4, INT8 learned
  73.2), but on MobileNetV2, where the damage sits at those very layers, it fails (W4/A4 nearest 0.5 collapse,
  learned 10.3).
- `cap_rms` (correction scaled to at most the RMS of the FP32 channel means) only softens the collapse
  (ResNet18 W4/A4 nearest 1.3, still one class on 92 percent; W4/A8 11.4).
- `dfq_weights_only` (literal Appendix D) collapses in the three 4-bit nearest cells with quantized activations
  (ResNet18 W4/A8 0.1, W4/A4 0.05, MobileNetV2 W4/A4 0.6; in the faithfulness cell it is B2's computation), but not
  with learned 4-bit weights (ResNet18 45.7, the best arm there; MobileNetV2 25.3, 1.9 points below no correction,
  -1.9 [-4.5, +0.7], and one class among the maxima on 52.8 percent of the images against 2.8 percent uncorrected)
  or at 5 bits (ResNet18 51.3, 4.9 points below no correction, -4.9 [-7.3, -2.4]; MobileNetV2 65.4, the best arm
  there), and it repairs every scalar mode-3 loss
  (63.0 to 63.9, section 3). It gives up the MobileNetV3 gains (INT5 0.6, INT6 4.4, INT8 learned 71.0 against
  8.1, 21.0 and 73.0 with B2's pass).
- Protocol H1.3 as stated (local, analytic and linear-only avoid collapse in the 5 cells and lose at most 2 points
  against no correction there) is **not supported**: local and analytic avoid collapse everywhere but lose 10.2 and
  6.8 points on MobileNetV2 W4/A4 learned; linear-only collapses on MobileNetV2 W4/A4 nearest. `executed` equals B2
  within 0.3 points on ResNet18 but not on MobileNetV3 INT8 nearest (+0.8).
- **Guarded correction (exploratory, post hoc).** No single variant wins everywhere, but the protocol's pre-stated
  H1.1 indicator separates the cases: keep the global correction unless the logit spread on the correction images
  (label-free; no screen image) falls below 0.2 of FP32, otherwise use the local form. This rule picks the better of
  global and local in 14 of the 15 mode-1 cells (MobileNetV2 W6/A6: global 70.1 against local 71.0), never
  collapses and is never worse than B2; it still loses 10.2 points against no correction on MobileNetV2 W4/A4
  learned, and it does not trigger on the moderate wide-exponent losses of ResNet18 (section 3). The threshold was stated before the
  measurements as a collapse predictor, not as a recipe rule, and the 15 cells are development evidence: this is a
  candidate for the next protocol, not a result.

**Verdict (mode 1): no defect; three regimes.** (i) At 4-bit nearest weights (ResNet18 W4 with A32, A8 and A4;
MobileNetV2 W4/A4) the faithful weights-only Appendix D pass collapses too: a **property** of the global form at
large weight error. (ii) With learned 4-bit weights, at 5 bits and for the wide-exponent formats of mode 3 the
faithful pass does not collapse and is above B2's pass in every cell with that arm (MXFP6 has none; +2.3 to +42.7 points, paired intervals excluding
zero), so B2's quantized-activation pass is the cause of B2's loss: a **fragile design** choice with
cheap alternatives (the weights-only pass, or the own-error local and analytic forms). The faithful pass repairs the
loss against no correction on ResNet18 W4/A4 learned, MobileNetV2 W5/A5 and the six scalar mode-3 cells; on ResNet18
W5/A5 (-4.9 [-7.3, -2.4]) and MobileNetV2 W4/A4 learned (-1.9 [-4.5, +0.7], one class among the maxima on 52.8
percent of the images) it only avoids the collapse and stays below no correction. (iii) On MobileNetV3-Large INT5,
INT6 and INT8 that same choice is what makes the correction help (+7.5, +17.6, +1.8 points; the weights-only pass
gives +0.0, +0.9, -0.1), so no single form wins everywhere (section 4).

## 2. Mode 2: posit under max-abs weight scales (minimal recipe)

### 2.1 Is there a slip in the scale mapping or the codebook? (H2.1)

- Codebook: an independent posit decoder (regime, exponent, fraction from the bit pattern; written for this check)
  gives exactly the sorted level tables of `TableQuantizer` for `posit8_es1` (255 finite levels, maxpos 4096) and
  `posit6_es1` (63 levels, maxpos 256).
- Scale mapping: the max-abs per-channel weight scale (`recon.engine.weight_scale`, rule `maxabs_per_channel`, the
  rule of the minimal recipe) maps every channel's max-abs weight exactly to maxpos (ratio 4096.0 and 256.0 on every
  channel of a test layer; the quantized channel maximum equals the FP32 maximum).
- Rounding is to the nearest level by value (2100 -> 1024 on posit8, because the midpoint of 1024 and 4096 is 2560).
  The posit standard rounds on the bit string, which in the regime-only top of the grid puts the boundary at the
  geometric midpoint (2048 -> 4096). This is B2's convention for every codebook (it treats all formats as level
  tables), not a slip; it only changes which of two very coarse top levels a weight lands on.

H2.1 holds: no defect in the mapping or the table. The three checks are saved as unit tests
(`tests/unit/test_experiment_b2_collapse_posit.py`: decoder equals the tables, value-nearest rounding, max-abs maps
every channel maximum to maxpos).

### 2.2 Mechanism (H2.2)

Posit precision is tapered: posit8_es1 has 3 fraction bits around 1 (spacing 6 percent) and none at the top
(positive levels ... 128, 192, 256, 512, 1024, 4096). Max-abs maps each channel's largest weight to 4096, so the
bulk of the weights lands in the regime-only part of the grid: every weight between 0.19 and 0.63 of the channel
maximum becomes 0.25 of the maximum, every weight above 0.63 becomes the maximum, and the relative spacing below is
25 to 50 percent (figure `b2-collapse-mode2--r2`, left). The CPU weight analysis
(`artifacts/experiment_b2_collapse/weights/weight-placement-r1.json`) gives median per-layer weight SQNR:

| format | max-abs at maxpos (minimal) | max-abs at 1.0 | MSE search (default) | MSE scale / max-abs scale (median) |
|---|---|---|---|---|
| posit8_es1 | about 10 dB | 34.5-37 dB | 38-40 dB | 1093-1188 (max lands near 3.4-3.7; no clipping) |
| posit6_es1 | about 10 dB | 22.5-25 dB | 26-28 dB | 65-68 (max lands near 3.8-4.0; no clipping) |
| fp8_e5m2 | 25.7-26.4 dB | about 26 dB | 26.2-27.7 dB | 1.35-1.38 (MSE clips 7.8-8.5 percent of channels) |
| fp6_e3m2 | about 26 dB | 20-24 dB | 26-28 dB | 1.42-1.46 (MSE clips 5.5-6.3 percent of channels) |
| log6 | 24-26 dB | 10-16 dB | 26-27.5 dB | 3.5-4.5 (MSE clips 0-1.3 percent of channels) |

(ranges over the three networks; ratio = median over layers of the per-channel MSE scale divided by the max-abs
scale, a channel clips when its ratio is below 1). A 10 dB weight SQNR is a relative weight error of about 30 percent in every layer;
for comparison INT6 with an MSE scale is at 26-31 dB. The MSE search of the default recipe finds the dense part of
the posit grid by itself (it places the channel maximum near 3.4-3.7), which is why posit is at the top of the
default-recipe ranking and at chance under the minimal recipe.

### 2.3 Anchors (H2.3) and layer groups (H2.4)

Minimal recipe throughout; only the per-channel weight scale rule changes:
`maxabs_per_channel` (= minimal, reproduces the matrix cells), `anchor_one` (channel max-abs at the level 1.0),
`mse_per_channel` (the default recipe's weight search).

| network | format | arm | top-1 | one-class share | minus MSE weights |
|---|---|---|---|---|---|
| R18 | posit8_es1 | anchor:maxabs_per_channel | 4.4 | 23.1% | -64.9 [-67.8, -62.0] |
| R18 | posit8_es1 | anchor:anchor_one | 69.4 | 0.4% | +0.2 [-0.6, +0.9] |
| R18 | posit8_es1 | anchor:mse_per_channel | 69.2 | 0.4% | - |
| R18 | posit6_es1 | anchor:maxabs_per_channel | 3.8 | 34.0% | -60.3 [-63.0, -57.6] |
| R18 | posit6_es1 | anchor:anchor_one | 62.2 | 0.7% | -1.9 [-3.4, -0.4] |
| R18 | posit6_es1 | anchor:mse_per_channel | 64.1 | 0.7% | - |
| MBv2 | posit8_es1 | anchor:maxabs_per_channel | 0.1**c** | 92.3% | -72.0 [-74.7, -69.2] |
| MBv2 | posit8_es1 | anchor:anchor_one | 70.6 | 0.3% | -1.5 [-2.6, -0.3] |
| MBv2 | posit8_es1 | anchor:mse_per_channel | 72.1 | 0.4% | - |
| MBv2 | posit6_es1 | anchor:maxabs_per_channel | 0.1 | 85.8% | -54.4 [-57.2, -51.6] |
| MBv2 | posit6_es1 | anchor:anchor_one | 20.2 | 15.9% | -34.3 [-37.0, -31.5] |
| MBv2 | posit6_es1 | anchor:mse_per_channel | 54.5 | 1.5% | - |
| MBv3-L | posit8_es1 | anchor:maxabs_per_channel | 0.0 | 25.8% | -74.8 [-77.4, -72.2] |
| MBv3-L | posit8_es1 | anchor:anchor_one | 74.3 | 0.5% | -0.5 [-1.6, +0.7] |
| MBv3-L | posit8_es1 | anchor:mse_per_channel | 74.8 | 0.4% | - |
| MBv3-L | posit6_es1 | anchor:maxabs_per_channel | 0.1 | 54.5% | -46.4 [-49.3, -43.6] |
| MBv3-L | posit6_es1 | anchor:anchor_one | 21.6 | 26.2% | -24.9 [-27.6, -22.3] |
| MBv3-L | posit6_es1 | anchor:mse_per_channel | 46.5 | 2.5% | - |
| MBv2 | posit8_es1 | maxabs-except-classifier | 0.1 | 59.6% | - |
| MBv2 | posit8_es1 | maxabs-except-depthwise | 0.1**c** | 100.0% | - |
| MBv2 | posit8_es1 | maxabs-except-early | 0.1 | 61.5% | - |
| MBv2 | posit8_es1 | maxabs-except-first | 0.0**c** | 93.0% | - |
| MBv2 | posit8_es1 | maxabs-except-last_conv | 0.1**c** | 99.1% | - |
| MBv2 | posit8_es1 | maxabs-except-late | 0.0 | 51.0% | - |
| MBv2 | posit8_es1 | maxabs-except-mid | 0.1**c** | 99.7% | - |
| MBv2 | posit8_es1 | maxabs-only-classifier | 68.2 | 0.5% | - |
| MBv2 | posit8_es1 | maxabs-only-depthwise | 1.1 | 20.2% | - |
| MBv2 | posit8_es1 | maxabs-only-early | 0.1**c** | 97.2% | - |
| MBv2 | posit8_es1 | maxabs-only-first | 57.0 | 1.3% | - |
| MBv2 | posit8_es1 | maxabs-only-last_conv | 70.2 | 0.5% | - |
| MBv2 | posit8_es1 | maxabs-only-late | 0.2 | 77.0% | - |
| MBv2 | posit8_es1 | maxabs-only-mid | 0.8 | 50.6% | - |

- **H2.3 holds.** Anchoring the channel maximum at 1.0 recovers posit8 to within 2 points of the MSE weight scale on
  all three networks (ResNet18 +0.2 [-0.6, +0.9], MobileNetV2 -1.5 [-2.6, -0.3], MobileNetV3-Large -0.5
  [-1.6, +0.7]) from 0.0 to 4.4 under max-abs. For posit6 (2 fraction bits at 1, none at 1/16) anchor-at-1 stays below
  MSE (ResNet18 -1.9, MobileNetV2 -34.3, MobileNetV3-Large -24.9): with only 63 levels the weights below 1/16 of the
  channel maximum fall into the sparse lower tail, and the MSE search (median: channel maximum near 4 for posit6) balances both tails.
- **H2.4 is rejected.** On MobileNetV2 posit8 no small group drives the one-class output. Max-abs on the classifier
  alone (68.2), the last 1x1 conv alone (70.2) or the first conv alone (57.0) keeps the network working; max-abs on
  any stage group alone breaks it (early 0.1 with one class on 97 percent of the images, mid 0.8, late 0.2,
  depthwise 1.1), and max-abs everywhere except any one group is at chance (0.0-0.1). The damage is the accumulation
  of about 10 dB weight SQNR in every layer, not a particular node.

**Verdict (mode 2): fragile design, not a defect.** The table and the scale mapping are exact; the max-abs-to-maxpos
rule, natural for uniform grids, puts the weights of a tapered format in its sparsest region. Cheap alternatives
exist: the MSE search (already in the default recipe) or a fixed anchor at 1.0 for posit8.

## 3. Mode 3: wide-exponent formats under the default recipe

### 3.1 Which switches differ (verified before measuring)

For every non-integer format the default and minimal recipes produce the same boundary plan (boundary fusion and
the activation range search are identical on all three networks: unit test
`test_default_and_minimal_differ_only_in_weight_range_and_correction_for_non_integers`); the unsigned-code switch
has no effect because these codebooks are signed in both recipes. Two switches remain: the weight range (MSE search
against max-abs) and bias correction. Each is removed alone: `default_no_bias_correction` and
`default_weight_maxabs` (for the scalar formats the latter is computed with the lane's bit-identical lean
correction, arm `<fmt>-default_weight_maxabs-N--global`, because the reference `bias_correct` holds all correction
activations on the GPU; MXFP6 uses redirected matrix cells, and its two max-abs-weight cells use the block
correction's low-memory twin `tools/experiment_b2_collapse/lowmem.py`, which runs the same GPU operations on the same
8-image chunks in the same order but parks live activations in host memory: peak 0.7-0.8 GB (ResNet18) and 2.1 GB (MobileNetV2) allocated over the
whole cell, against 5-8 GB for the correction alone in the original.
It reproduces the matrix record `resnet18 mxfp6_e3m2 default` exactly (configuration identity, which contains the
per-layer correction report, logits SHA-256 and readout arrays) and the configuration identity of `mobilenet_v2
mxfp6_e3m2 default` (53 corrected layers); its `resnet18 mxfp6_e3m2 default_weight_maxabs` cell has the same identity
and logits SHA-256 as lane Q2's record of that cell, computed without it; unit test
`tests/unit/test_experiment_b2_collapse_lowmem.py`). Decision rule (protocol): a switch is the cause if
removing it alone recovers at least 50 percent of the minimal-minus-default loss and the paired interval of
(arm - default) excludes zero.

| network / format | minimal - default | default | minimal | default_no_bias_correction | default_weight_maxabs | bc:local_empirical | bc:analytic_fp32 | bc:linear_only | bc:cap_rms | bc:dfq_weights_only | cause by rule |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R18 fp8_e5m2 | +1.8 [+0.2, +3.4] | 61.2 | 63.0 | 63.6 (138%) | 59.8 (-79%) | 64.4 (179%) | 64.1 (162%) | 65.0 (214%) | 61.2 (0%) | 63.5 (130%) | default_no_bias_correction |
| R18 fp6_e3m2 | +9.3 [+7.2, +11.4] | 53.6 | 62.9 | 64.3 (114%) | 51.7 (-21%) | 64.4 (116%) | 65.3 (126%) | 64.6 (118%) | 53.9 (3%) | 63.7 (109%) | default_no_bias_correction |
| R18 log6 | +7.0 [+4.8, +9.2] | 54.5 | 61.5 | 60.6 (88%) | 54.7 (3%) | 62.8 (119%) | 61.7 (103%) | 62.8 (119%) | 55.0 (7%) | 63.2 (125%) | default_no_bias_correction |
| R18 posit6_es1 | -55.9 [-58.6, -53.0] | 59.6 | 3.8 | 64.1 (-8%) | 0.1 (106%) | - | - | - | - | - | no significant default-minus-minimal loss |
| R18 mxfp6_e3m2 | +12.8 [+10.6, +15.0] | 51.6 | 64.4 | 64.2 (99%) | 50.7 (-7%) | - | - | - | - | - | default_no_bias_correction |
| MBv2 fp8_e5m2 | +18.7 [+15.7, +21.6] | 23.2 | 41.9 | 54.5 (167%) | 30.7 (40%) | 65.0 (223%) | 63.9 (217%) | 45.6 (120%) | 23.5 (2%) | 63.9 (217%) | default_no_bias_correction |
| MBv2 fp6_e3m2 | +21.8 [+18.8, +24.7] | 20.9 | 42.6 | 49.2 (130%) | 27.6 (31%) | 64.6 (201%) | 64.5 (201%) | 42.1 (98%) | 19.7 (-5%) | 63.5 (196%) | default_no_bias_correction |
| MBv2 log6 | +27.1 [+24.2, +30.1] | 20.3 | 47.5 | 50.6 (111%) | 25.5 (19%) | 62.8 (156%) | 63.4 (158%) | 40.0 (72%) | 19.7 (-2%) | 63.0 (157%) | default_no_bias_correction |
| MBv2 posit6_es1 | -18.7 [-21.0, -16.5] | 18.8 | 0.1 | 54.5 (-191%) | 0.1 (100%) | - | - | - | - | - | no significant default-minus-minimal loss |
| MBv2 mxfp6_e3m2 | +34.7 [+31.7, +37.6] | 7.2 | 41.9 | 41.7 (99%) | 8.4 (3%) | - | - | - | - | - | default_no_bias_correction |
| MBv3-L fp8_e5m2 | -1.4 [-4.1, +1.2] | 46.6 | 45.1 | 52.5 (-418%) | 47.2 (-45%) | - | - | - | - | - | no significant default-minus-minimal loss |
| MBv3-L fp6_e3m2 | -2.3 [-5.0, +0.4] | 44.6 | 42.3 | 52.9 (-366%) | 44.3 (12%) | - | - | - | - | - | no significant default-minus-minimal loss |
| MBv3-L log6 | -37.2 [-40.2, -34.3] | 53.2 | 15.9 | 20.6 (87%) | 49.3 (10%) | - | - | - | - | - | no significant default-minus-minimal loss |
| MBv3-L posit6_es1 | -42.8 [-45.6, -39.9] | 42.9 | 0.1 | 46.5 (-8%) | 0.1 (100%) | - | - | - | - | - | no significant default-minus-minimal loss |
| MBv3-L mxfp6_e3m2 | -14.9 [-17.8, -12.1] | 43.0 | 28.1 | - | - | - | - | - | - | - | no significant default-minus-minimal loss |

(Percentages in brackets: share of the minimal-minus-default loss recovered by the arm; `bc:*` = default recipe with
the mode-1 correction variant in place of B2's global correction.)

### 3.2 Findings

- **H3.1, switch clause: holds on ResNet18 and MobileNetV2; bias correction is the switch.** Removing it alone
  recovers 88 to 167 percent of the loss for FP8 E5M2, FP6 E3M2 and Log6 on both networks, with intervals excluding
  zero; removing the MSE weight search instead does not (ResNet18: -79 to +3 percent; MobileNetV2: 19 to 40 percent).
  H3.2 (weight range) is rejected, as the CPU weight analysis predicted (the MSE weight scale is only 0.5 to 2 dB
  better than max-abs for these formats, section 2.2).
- **H3.1, mechanism clause (late always-off channels): holds on MobileNetV2 only.** On MobileNetV2 the global
  correction switches off 97.9 to 99.0 percent of the channels of features_18_0 (FP32 0 percent) and the logit spread
  falls to 0.077-0.089 of FP32 for E5M2, E3M2 and Log6. On ResNet18 it does not: for FP8 E5M2 no layer's correction
  exceeds the RMS of the FP32 channel means and the always-off fractions equal FP32's (worst layer conv1, 12.5
  percent in both); for FP6 E3M2 and Log6 one layer does (layer4_1_conv2), the largest always-off fraction is again
  conv1's 12.5 percent (equal to FP32), the largest excess over FP32 is 0.8 points (layer4_1_conv1: 0.8 percent
  against 0 in FP32), and the logit spread stays at 0.37-0.38 of FP32 (E5M2 0.49).
- **The weights-only Appendix D pass repairs every scalar loss cell, so the cause is B2's quantized-activation pass
  (regime (ii) of mode 1).** `bc:dfq_weights_only` gives 63.5 / 63.7 / 63.2 on ResNet18 (FP8 E5M2 / FP6 E3M2 / Log6;
  +2.3 [+0.8, +3.9], +10.1 [+8.2, +12.1], +8.7 [+6.7, +10.8] against default) and 63.9 / 63.5 / 63.0 on MobileNetV2
  (+40.6 [+37.6, +43.6], +42.6 [+39.6, +45.5], +42.7 [+39.6, +45.7]): 109 to 217 percent of the
  minimal-minus-default loss, above minimal in every cell. The faithful method does not fail here; the recipe's
  choice to correct with quantized activations does.
- **MXFP6 E3M2 (block format, r2) behaves the same way.** Removing bias correction alone recovers 99 percent of the
  loss on both networks: ResNet18 64.2 against default 51.6 (minimal 64.4; arm - default +12.6 [+10.5, +14.8]), MobileNetV2 41.7
  against 7.2 (minimal 41.9; +34.5 [+31.5, +37.5]). Max-abs weights recover nothing: ResNet18 50.7 (-0.9
  [-2.1, +0.2], -7 percent), MobileNetV2 8.4 (+1.2 [+0.3, +2.1], 3 percent). On MobileNetV2 the default MXFP6 cell is
  close to a mode-1 collapse: the lowest-index argmax is one class on 77.7 percent of the images (max-abs weights
  76.8 percent; no correction 3.2 percent, minimal 3.1 percent; the matrix cells store no one-class-among-maxima
  share, so this is the lowest-index readout).
- **Mechanism.** On MobileNetV2 it is the mode-1 collapse pattern at 6 to 8 bits: under the global correction the classifier input of FP8 E5M2 falls by 0.100 against an FP32
  mean of 0.105 (0 percent of channels positive; uncorrected +0.016, 51 percent positive), as in the W4/A4 cells. The logit spread
  under the global correction is 0.09 of FP32 for FP8 E5M2 (the guard of section 1.5 switches to the local form
  there: 65.0), and the features_18 ReLU6 output has 99.6 percent zero codes under default against 93.5 percent under
  minimal (matrix cells `mobilenet_v2 fp8_e5m2 default` and `minimal`, occupancy block, first 128 screen images).
  On ResNet18 the pattern is milder: the late inputs are attenuated under the global correction on every channel (FP6 E3M2: fc -0.45
  against 0.92, layer4_1_conv1 -0.034 against 0.069) and mixed without it (39 to 59 percent positive, means within
  0.02 of zero), but almost no channels are switched off (always-off fractions at most 0.8 points above FP32's in any
  layer), the logit spread falls to 0.37 to 0.49 of FP32, and the loss is a moderate 2 to 9 points. Since the weights-only pass, which corrects the same
  weights with FP32 activations, repairs both networks, the harmful part of B2's correction is the part that
  responds to activation quantization; why these coarse-mantissa codebooks (2 mantissa bits for E5M2 and E3M2)
  produce more of it than INT6 does was not isolated further.
- **The own-error forms repair it and beat both recipes.** With the default recipe's MSE weights and the local or
  analytic correction, FP8 E5M2 reaches 64.4 / 64.1 on ResNet18 (default 61.2, minimal 63.0) and 65.0 / 63.9 on
  MobileNetV2 (default 23.2, minimal 41.9); FP6 E3M2 64.4 / 65.3 on ResNet18 (53.6 / 62.9) and 64.6 / 64.5 on
  MobileNetV2 (20.9 / 42.6); Log6 62.8 / 61.7 on ResNet18 (54.5 / 61.5) and 62.8 / 63.4 on MobileNetV2
  (20.3 / 47.5). The literal weights-only Appendix D pass is close (63.0 to 63.9, see above). `linear_only` helps
  on ResNet18 (65.0 E5M2) but only partly on MobileNetV2 (40.0 to 45.6), and `cap_rms` does nothing (61.2, 23.5).
- **MobileNetV3-Large has no default-minus-minimal loss** for these formats (default is better or equal), but
  removing bias correction still helps FP8 E5M2 (52.5 against 46.6), FP6 E3M2 (52.9 against 44.6) and posit6
  (46.5 against 42.9), while it costs Log6 32.6 points (20.6 against 53.2).
- **posit6** has no default-minus-minimal loss (minimal is at chance, mode 2), but under the default recipe bias
  correction costs it 4.5 points on ResNet18 (no correction 64.1 against 59.6), 35.7 on MobileNetV2 (54.5 against
  18.8) and 3.6 on MobileNetV3-Large (46.5 against 42.9).

**Verdict (mode 3): fragile design (regime (ii) of mode 1), not a property of the method and not a defect.** Bias
correction is the switch (H3.1, switch clause, on ResNet18 and MobileNetV2, MXFP6 included), not the MSE weight
range (H3.2 rejected). Within bias correction, the cause is B2's choice to run the pass with quantized activations:
the literal weights-only Appendix D pass recovers 109 to 217 percent of every scalar loss, and the own-error forms
recover 103 to 223 percent, with the same code that reproduces the matrix cells. For MXFP6 E3M2 only the switch is
established (the correction variants were not run on the block engine). Of the eight rule-(c) losses of the
wide-exponent formats (FP8 E5M2, FP6 E3M2, Log6 and MXFP6 E3M2, each on ResNet18 and MobileNetV2), bias correction is
the switch in all eight, not the MSE weight range; that the cause within it is B2's quantized-activation pass, not the
cited method, is shown for six (the scalar cells, by the weights-only arm) and inferred for the two MXFP6 cells.

## 4. Recommendations to the owner

1. **Report B2's bias correction as DFQ Appendix D run with quantized activations, not as the published method
   failing.** The code is correct. The literal weights-only Appendix D pass fails only at 4-bit nearest weights (a
   property of the global form); B2's other failures (learned 4-bit weights, 5 bits, the wide-exponent formats) come
   from its quantized-activation pass (for MXFP6 E3M2 inferred, not measured). The weights-only pass avoids those
   failures but does not always beat no correction (ResNet18 W5/A5 -4.9, MobileNetV2 W4/A4 learned -1.9). Any
   comparison with AdaRound's "bias correction" number should use the local form (37.7 here against the paper's
   38.87); the global form, literal or B2's, gives 0.1 in that cell.
2. **Decide the recipe's correction form (owner decision).** Options, all measured here on development evidence:
   (a) keep B2's pass (status quo; collapses in every 4-bit cell on ResNet18/MobileNetV2; at 5 bits it costs 42.5 and
   4.1 points against no correction (ResNet18 W5/A5 13.7, one class among the maxima on 76.5 percent of the images,
   below the 90 percent collapse mark; MobileNetV2 W5/A5 32.5); against no correction it costs
   the wide-exponent formats 2.5 to 35.7 points on ResNet18 and MobileNetV2: FP8 E5M2 2.5 / 31.2, FP6 E3M2
   10.6 / 28.3, Log6 6.1 / 30.2, MXFP6 E3M2 12.6 / 34.5, posit6 4.5 / 35.7; on MobileNetV3-Large it costs FP8 E5M2,
   FP6 E3M2 and posit6 5.9, 8.4 and 3.6 points and gains Log6 32.6); (b) the literal weights-only Appendix D pass
   (repairs every scalar wide-exponent loss: 63.0 to 63.9, at or above minimal; no collapse with learned 4-bit
   weights or at 5 bits: ResNet18 45.7 and 51.3, MobileNetV2 25.3 and 65.4, of which ResNet18 W5/A5 and MobileNetV2
   W4/A4 learned stay below no correction, -4.9 and -1.9; still collapses at 4-bit nearest weights;
   gives up the MobileNetV3 gains: INT5 / INT6 / INT8 learned 0.6 / 4.4 / 71.0 against 8.1 / 21.0 / 73.0; not
   measured on MXFP6, posit6 or MobileNetV3 wide-exponent formats); (c) the local own-error form (never collapses;
   repairs every scalar wide-exponent loss and beats minimal by 1 to 23 points; loses the MobileNetV3 INT5/6/8
   gains of +7.5/+17.6/+1.8 and 10 points on MobileNetV2 W4/A4 learned); (d) the guarded rule of section 1.5
   (global unless the label-free logit-spread indicator on the correction images falls below 0.2; keeps the
   MobileNetV3 gains, removes every collapse, but does not fix the moderate ResNet18 wide-exponent losses).
   Whichever is chosen must be frozen in a new protocol and confirmed on held-out data before it replaces B2 in any
   table; changing it re-opens the recipe rule (c) result, because 8 of its 10 losses are bias-correction losses
   (removing the correction alone recovers 88 to 167 percent of each); for six of them (FP8 E5M2, FP6 E3M2 and Log6
   on ResNet18 and MobileNetV2) the weights-only arm shows that B2's quantized-activation pass is the cause, for the
   two MXFP6 E3M2 cells this is inferred (no weights-only arm on the block engine).
3. **Posit under the minimal recipe** should be described as a scale-rule/format mismatch, not a posit property:
   with the maximum anchored at 1.0 posit8 is within 1.5 points of the MSE scale on all three networks. If the
   minimal recipe stays in the paper as an ablation, add the anchor-at-1 arm for posits or footnote the mismatch.
4. **Rule (c) losses (wide-exponent formats)**: attribute all eight to B2's bias correction, not to the MSE weight
   range (removing the correction alone recovers 88 to 167 percent of each, MXFP6 included). Attribute six of them
   (FP8 E5M2, FP6 E3M2 and Log6 on ResNet18 and MobileNetV2) to B2's quantized-activation pass and not to the cited
   method: the weights-only pass puts them at 63.0 to 63.9 and the local and analytic variants at 61.7 to 65.3, at or
   above both recipes. For the two MXFP6 E3M2 losses the same cause is inferred, not shown; a weights-only arm on
   the block engine would settle it.
5. Practical: lanes running scalar B2 bias correction on MobileNets can use the lane's lean sequential correction
   (`correct.staged_correct`, policy `global`, about 1.2-1.6 GB peak, activations kept in host memory), which is
   bit-identical to `bias_correct` (unit tests; deployed-state and top-5 equality on 24 L6 counterparts; and
   the reference arm `resnet18 fp8_e5m2 default_weight_maxabs --b2` against its lean twin: deployed state and
   per-image credit equal; the same holds for the four MobileNetV3 max-abs reference arms and their lean twins).
   For the block formats, `tools/experiment_b2_collapse/lowmem.py` is a bit-identical low-memory twin of
   `blocks.bias_correct_blocks` (0.7 GB peak on ResNet18 and 2.1 GB on MobileNetV2 against 5-8 GB; checks in
   section 3.1); lanes with many block-format bias-correction jobs can run them outside the heavy-job queue with it.

## 5. Limits

- Development evidence only: the 1k screen contains dev512, on which B2 was chosen; nothing here is a held-out
  result. One correction set (the first 256 calibration images), one seed, nearest rounding except where L6 fits
  are reused; ties handled by expected credit.
- The instrumentation (always-off channels, spreads, local/inherited split) is computed on the correction images,
  not on screen images 0..63 as the protocol's instrumentation line said; this is stricter (no screen image is used
  for diagnostics) and is the reason the guard is label-free.
- The guard (section 1.5) and the variant ranking are post hoc on 15 + 10 cells; the 0.2 threshold was pre-stated only
  as a collapse predictor.
- The three regimes of mode 1 are a post hoc reading of pre-specified arms on development cells, one seed, not a
  tested classification. Regime (i) rests on three ResNet18 cells (W4 nearest with FP32, 8-bit and 4-bit activations;
  in the FP32-activation cell the weights-only pass is B2's computation by construction) and one MobileNetV2 cell
  (W4/A4 nearest) that is at 1.1 percent even uncorrected; MobileNetV3-Large W4/A4 is at chance in every arm and
  separates nothing. The nearest-versus-learned contrast inside 4 bits rests on two cells (ResNet18 and MobileNetV2
  W4/A4) with one learned-rounding fit seed each (the L6 fits, seed 0). Regime (ii) contains cells where the
  weights-only pass only avoids the collapse (ResNet18 W5/A5, MobileNetV2 W4/A4 learned; section 1.5).
- The input-drift diagnostic (addendum 3, section 1.4) was specified after the independent reviewer had measured the
  same quantity with the lane's build code (`artifacts/experiment_b2_collapse/review-scratch/drift_check.json`): it is
  a reproduction on rebuilt, digest-checked arms, not a blind test. The mechanism is described for the late layers
  where the correction exceeds the FP32 mean RMS; why the excess starts there (which weights make the missing input
  net inhibitory) was not traced further. The reviewer's skip-layer intervention (section 1.4) is post hoc and was
  run with the reviewer's code, on the ResNet18 faithfulness cell only.
- MXFP6 E3M2 (r2): the decomposition is complete for ResNet18 and MobileNetV2 (the protocol's cells); MobileNetV3 has
  no default-minus-minimal loss for MXFP6 and was not decomposed. The two max-abs-weight cells were computed with the
  low-memory twin of the block correction (addenda 1-2); its MobileNetV2 validation is by configuration identity
  only (the full logits/readout reproduction was done on ResNet18), as addendum 2 states. A second identity check
  exists for the MobileNetV2 max-abs cell itself: an older lane queue later rebuilt that configuration through the
  original path (`tools.run.experiment_b2_collapse cell`, the unmodified `blocks.bias_correct_blocks`, `--heavy
  8000`) and `matrix.run_cell` found the existing low-memory cell under the same identity (status "exists", cell
  `058874e281ad`; `artifacts/experiment_b2_collapse/logs/queue2-cells.log`, 2026-10-02T18:00:56). That is identity
  equality only (the cell is keyed by the first 12 hex digits of the configuration SHA-256; logits were not
  recomputed). The `bc:*` variants of
  section 1.5 were not run on MXFP6 (block engine), so the own-error repair is shown for the scalar formats only.
- GPU cost: 218 arm evaluations (2.6 GPU-hours of arm compute) plus 6 redirected matrix cells (about 0.6 GPU-hour,
  of which r4's three jobs about 0.5), plus r5's two input-drift jobs (283 s and 347 s, peak 1.4 and 1.6 GB), about
  3.4 GPU-hours on the shared GPU in all, above the protocol's 1.5-2 hour estimate (per-arm times were inflated by
  contention). For a few minutes around 17:50 two lane GPU jobs ran at once (r2's leftover cell queue, which had no
  stop switch and found its cell already present, beside r5's MobileNetV2 drift job), against the one-job rule; the
  cell job wrote nothing, and the drift values equal the independent reviewer's re-measurement.
- Provenance of the weight-placement record `artifacts/experiment_b2_collapse/weights/weight-placement-r1.json`
  (table 2.2): its `own_sources` lists hashes of earlier versions of `correct.py`, `build.py` and `evaluate.py` (the
  670 s run started about 11:55; those files were last edited at 11:57 and 12:02). `weights.py`, which computes the
  record, imports none of them and its listed hash equals the current file; the independent reviewer recomputed the
  ResNet18 rows with the current code and got the same values.
- Figure `b2-collapse-mode1--r1`: the purple points (`dfq_weights_only`, literal Appendix D) have no legend entry
  (the label was attached to the first panel row, which has no such arm); `b2-collapse-mode1--r2` has the entry.
- Figure `b2-collapse-mode3--r2`: arms at chance (posit6 max-abs weights, 0.1) are drawn as bars of zero height; the
  table above gives their values.

## 6. Reproduction

```
.venv/bin/python -m pytest tests/unit/test_experiment_b2_collapse.py tests/unit/test_experiment_b2_collapse_drive.py \
    tests/unit/test_experiment_b2_collapse_lowmem.py tests/unit/test_experiment_b2_collapse_drift.py \
    tests/unit/test_experiment_b2_collapse_posit.py -q                               # 31 tests
GPU_LANE=Q3 artifacts/agent_orchestration/gpu_run.sh --min-free-mib 4000 .venv-b/bin/python -m \
    tools.run.experiment_b2_collapse_drive --model M --group mode1|mode2|mode3 [--reverse] [--skip-heavy --lean-maxabs]
GPU_LANE=Q3 artifacts/agent_orchestration/gpu_run.sh [--heavy 8000] .venv-b/bin/python -m \
    tools.run.experiment_b2_collapse cell --model M --format F --recipes R1,R2      # redirected matrix cells
GPU_LANE=Q3 artifacts/agent_orchestration/gpu_run.sh --min-free-mib 4000 .venv-b/bin/python -m \
    tools.run.experiment_b2_collapse_lowmem --model resnet18 --format mxfp6_e3m2 --recipes default,default_weight_maxabs
GPU_LANE=Q3 artifacts/agent_orchestration/gpu_run.sh --min-free-mib 4000 .venv-b/bin/python -m \
    tools.run.experiment_b2_collapse_lowmem --model mobilenet_v2 --format mxfp6_e3m2 --validate default \
    --recipes default_weight_maxabs                                                  # low-memory block correction
GPU_LANE=Q3 artifacts/agent_orchestration/gpu_run.sh --min-free-mib 3000 .venv-b/bin/python -m \
    tools.run.experiment_b2_collapse_drift --model resnet18|mobilenet_v2 --tag r1     # input drift (addendum 3)
.venv/bin/python -m tools.run.experiment_b2_collapse weights --tag r1                 # CPU weight placement
.venv/bin/python -m tools.run.experiment_b2_collapse report --tag r2                  # summaries + figures, once per tag
```

Rule-12 check: the redirected cell `resnet18 int6 minimal` equals the matrix record (configuration identity, logits
SHA-256 and readout arrays; `artifacts/experiment_b2_collapse/matrix/checks/resnet18--int6--minimal.json`).
