# B2 activation-error attribution and MobileNetV3 repairs (lane Q4)

Status: **2026-10-02, final v2 (lane Q4, agents r1-r7; review 1 resolved).** Summaries:
`results/summaries/b2-attrib-v1/` (v1, superseded where v2 differs), `results/summaries/b2-attrib-v1/v2/` (review
fixes, `pcu` arms) and `results/summaries/b2-attrib-v1/v2.1/` (one correction of v2, section 3.1); each written once.
Figures: `results/figures/b2-attrib-{mbv3-groups,sqnr-predictor}-v1.{png,pdf}` and
`results/figures/b2-attrib-{mbv3-repairs,mbv3-leave-one-out}-v2.{png,pdf}` (the v1 repairs and leave-one-out
figures are superseded).

In brief (MobileNetV3-Large, B2 default recipe, ImageNet 1k screen; per-channel numbers from the `pcu` arms of
addendum 8, which replace the `pcf` arms after review 1):
- **INT8 (71.60, FP32 75.00):** the three stem boundaries carry 3.10 [1.4, 4.8] of the 3.40 pp activation loss. The
  stem repair r3b (per-channel scales at the two stem outputs plus zero points at the Hardswish outputs) reaches
  74.10, +2.50 [0.70, 4.35] over the default and -0.90 [-2.10, 0.30] to FP32. The number is sampler-sensitive: two
  other per-channel samples gave 74.65 (biased) and 76.13 (unbiased).
- **INT6 (21.00):** Hardswish outputs (23.1 pp damage) and the stem (15.5) dominate. Zero points alone give 57 %,
  `pcu:all` 66.0 %; INT6 stays about 9 pp or more below FP32.
- **FP6 E2M3 (58.43):** the stem dominates (8.7 pp). Per-channel scales help: `pcu:all` 68.8 %, the stem alone 63.8 %.
- **FP8 E4M3 (71.66):** no repair helps. The per-channel arms other than the stem arms lose 0.7-2.1 pp (three of
  four intervals exclude zero); zero points lose 3.1 pp where the search chose them.
- **Other networks:** at INT8 the affine and `pcu` arms neither harm nor help MobileNetV2 or ResNet18 (point
  estimates -0.32 to 0.00 pp, every interval includes zero). The earlier `pcf:all` -25 pp on MobileNetV2 came from
  the border-column sample (review B1), not from the repair or its fold.
- **SQNR:** no detectable association between group SQNR and accuracy damage, though a moderate one is not excluded
  (pooled Spearman -0.11 [-0.36, 0.14]; none of 42 predictor tests survives Holm); the stem ranks 4th of 13 by SQNR at MobileNetV3 INT8. Network-level
  median SQNR ranks the top-1 drop across formats (-0.92).

Evidence: **development evidence only.**
- Every accuracy number is top-1 with expected credit for tied logits on the frozen ImageNet 1k screen.
- Paired differences use the project's paired bootstrap (`tools/analysis/b2_matrix.interval`), 95 % intervals.
- Calibration uses only the frozen calibration list: the v1 observations, plus the first 256 calibration images for
  bias correction and for the `pcf`/`pcu` per-channel scales.
- Repairs (`affine`, `pc`, `pcf`, `pcu`) were chosen and evaluated on the same screen. Gains are optimistic by construction.
- Keeping boundaries in FP32 (`wide`, `only`) locates error. It is never a recipe.
- Whether any repair enters the recipe is the owner's decision.

Protocol: `public/experiments/configs/breadth-study/b2-attrib-protocol-v1.json` (sha256 f2f3ae1e…). Addenda, each
written before the measurements it covers:
- addendum 1 (7ae5f26d…): the low-memory bias-correction path and its gate;
- addendum 2 (c6a4da75…): `pcf` per-channel arms, meant to be fitted on the full calibration activations but drawn
  from border columns (review B1); superseded by addendum 8;
- addendum 3 (02a6971f…): the `+fixbc` diagnostic (the default's corrected biases instead of a per-arm refit);
- addendum 4 (9f2b73bd…): three attribution-driven repair arms (r3) at the stem;
- addendum 5 (cd5b881f…): one more `+fixbc` diagnostic at a single node;
- addendum 6 (bf731668…): a CPU check of the fold mechanism suspected behind the `pcf:all` regression (no accuracy
  measured; the regression later proved a sampler artefact);
- addendum 7 (f39f92ef…): one cross-network stem-repair arm (MobileNetV2 FP6 E2M3 `pcf:role.stem`; rerun as `pcu`
  under addendum 8);
- addendum 8 (bc2852da…): after independent review 1, the `pcu` per-channel arms (unbiased sample; they replace
  every `pcf` result), the v2 summaries with the review fixes, and the fold diagnostic rerun with `pcu` scales.

## 1. Setup

- Base: the frozen B2 default recipe. MSE activation scales per node, per-channel MSE weights, sequential empirical
  bias correction on 256 calibration images.
- An arm changes only the boundary plan or the activation code at some boundaries. Bias correction is refitted for
  every arm, as the recipe would do.
- Arms:
  - `only:G` quantizes only group G; every other boundary stays FP32. Damage of G = weights-only minus `only:G`.
  - `wide:G` keeps G in FP32 and quantizes the rest. Recovery of G = `wide:G` minus default.
  - `affine:T` gives the T boundaries an affine (zero-point) code; the search keeps the symmetric code where the
    zero point is not better on the calibration samples.
  - `pc:T` per-channel activation scales from the v1 samples. `pcu:T` (addendum 8) the same MSE search on the FP32
    activations of the 256 bias-correction images, anchored at the exact per-channel maximum, on a seeded simple
    random sample of 4 096 positions (image, row, column) per boundary, the same positions for every channel.
    Scales are folded into conv/linear consumers before weight quantization; other consumers need per-channel
    requantization.
  - `pcf:T` (addendum 2) was meant to be the `pcu` fit but kept every k-th value with k = H*W/16: on 112x112 maps
    all 4 096 search values of a channel came from image column 0, on 56x56 maps from two columns, on 28x28 maps
    from four (review finding B1). Its 36 records stay as measured, labelled superseded; no conclusion rests on them.
- Groups (protocol): 13 structural role groups of MobileNetV3 (stem = input, stem conv output, stem Hardswish;
  expand-conv outputs; ReLU and Hardswish outputs; depthwise outputs; SE pool, reduce, expand, gate, product;
  projections; residual adds; classifier) and 3 position groups (early, middle, late inverted-residual stages).
- Low-memory path (addendum 1). `tools/experiment_b2_attrib/lowmem.py` computes the frozen bias correction with the
  same kernels on the same tensors at about 1.0-1.1 GB peak (from 5-8 GB). CPU test: bit-identical to the frozen
  `bias_correct`. GPU gates against the sealed matrix cells passed for MobileNetV3 INT8, FP6 E2M3 and FP8 E4M3 (six
  checks each, including the logits sha256); the v1 verify job passed for INT8 and INT6. Every `ref_default` arm
  reproduces its sealed cell's top-1 (INT8 71.60, INT6 21.00, FP6 E2M3 58.43, FP8 E4M3 71.66).

## 2. Where the MobileNetV3 loss lives (Part A)

### 2.1 Stored per-boundary SQNR (A0, CPU)

- INT8: Hardswish outputs carry 38 % of the network's summed noise-to-signal, SE products 17 %, depthwise outputs 12 %.
- INT6: Hardswish 29 %, classifier group 28 % (global average pool 8.7 dB), projections 12 %.
- Across all sealed scalar default cells, the median per-boundary SQNR ranks the top-1 drop to FP32 well:
  Spearman -0.96 / -0.98 / -0.96 (ResNet18 / MobileNetV2 / MobileNetV3), pooled -0.92. Restricted to cells
  with a drop below 20 pp: -0.97 / -0.93 / -0.75, pooled -0.83. This is driven by the bit width.

### 2.2 Group attribution (A1)

Damage (`only:G`) and recovery (`wide:G`) in top-1 pp; recovery with its paired 95 % CI.

| group (nodes) | INT8 dmg | INT8 rec | INT6 dmg | INT6 rec [CI] | FP6 E2M3 dmg | FP6 rec [CI] |
|---|---|---|---|---|---|---|
| stem (3) | **3.10** | 1.40 [-0.5, 3.3] | **15.5** | 22.5 [19.8, 25.2] | **8.7** | 7.1 [4.8, 9.5] |
| Hardswish outputs (19) | 0.20 | 0.20 | **23.1** | 28.6 [25.8, 31.4] | 2.9 | 2.5 [0.4, 4.6] |
| ReLU outputs (11) | 0.40 | 0.00 | 1.8 | 2.2 [0.2, 4.2] | 1.3 | 0.8 |
| projections (15) | 0.00 | 0.08 | 1.2 | 3.2 [1.1, 5.3] | 0.4 | -0.6 |
| classifier (4) | -0.1 | -0.1 | 0.4 | 4.4 [2.9, 5.9] | 1.1 | 0.7 |
| residual adds (10) | 0.10 | 0.60 | 0.2 | 5.0 [2.7, 7.2] | 0.2 | -1.4 |
| depthwise outputs (9) | -0.6 | 0.55 | 0.6 | 0.1 | -0.2 | 2.4 [0.5, 4.2] |
| SE pool/reduce/expand/gate/product | ≤ 0.6 | -0.65 to 0.73 | ≤ 0.1 | 0.3 to 3.0 | ≤ 0.5 | -2.5 to 0.4 |
| expand-conv outputs (10) | -0.7 | **-1.90 [-3.6, -0.3]** | 1.4 | **-8.3 [-10.5, -6.1]** | -0.8 | **-10.2 [-12.6, -7.8]** |

Reference points: INT8 default 71.60, weights-only 75.00 (= FP32 75.00), activation loss 3.40 pp; INT6 21.00 /
74.80, loss 53.8; FP6 E2M3 58.43 / 74.60, loss 16.2.

Reading:
- **INT8: the stem carries most of the loss.** Quantizing only the three stem boundaries costs 3.10 [1.4, 4.8] of
  the 3.40 pp; keeping them FP32 recovers +1.40 [-0.5, 3.3] (an interval that includes zero). Every other group's
  damage is at most 0.6 pp. Every other recovery interval includes zero except the expand-conv outputs, whose
  interval [-3.6, -0.3] excludes it on the loss side (keeping them FP32 hurts; section 2.4 traces this to the
  bias-correction refit).
- **INT6:** Hardswish outputs (23.1) and the stem (15.5) dominate. **FP6 E2M3:** the stem (8.7), Hardswish second (2.9).
- **High SQNR does not mean harmless.** At INT8 the stem's group SQNR (32.4 dB from the summed noise-to-signal;
  boundaries 35-46 dB) ranks 4th of 13, behind the SE gate (39.8), SE expand (33.8) and SE reduce (33.2) groups,
  and the stem has by far the largest damage while those three cost nothing (-0.1 each). At INT6 the stem again
  ranks 4th of 13 (22.8 dB). Only at FP6 E2M3 is the stem the top-SQNR group (24.9 dB). The opposite case also
  occurs: at INT6 the Hardswish outputs have the lowest group SQNR (6.0 dB) and the largest damage (23.1 pp), while
  the classifier group, next lowest (6.1 dB; its global average pool node 8.7 dB), costs 0.4 pp.
- No detectable association between group SQNR and group damage, but with 13 groups per cell a moderate one is
  not excluded: Spearman(group SQNR, damage) -0.16 [-0.73, 0.47] at INT8, -0.37 [-0.81, 0.33] at INT6, -0.04
  [-0.70, 0.59] at FP6 (bootstrap 95 % intervals over groups; minimum SQNR -0.24 [-0.77, 0.35] at INT8). At INT8
  group SQNR ranks the output divergence of `only:G` (Spearman with the mean KL -0.66 [-0.95, -0.05], raw p 0.014),
  but this does not survive a Holm adjustment over the 42 SQNR-predictor tests of the v2 summary (none does).
  Pooled over formats and networks: section 4.
- Why the stem: on the 256 bias-correction images (FP32), the 16 channels of the stem conv output span per-channel
  maxima from 1.0 to 33.4 (5.1 octaves; 5 of 16 channels below 1/8 of the tensor maximum), the stem Hardswish 5.6
  octaves, the input only 0.23 octaves. A per-tensor scale leaves the small stem channels a few levels, and the
  first block is a depthwise convolution that processes each of those channels on its own.

### 2.3 Leave-one-out (A2)

Protocol rule: the two role groups with the largest recovery (more than one node), at most 40 nodes, one node kept
FP32 at a time. INT6 and FP6 E2M3 selected the Hardswish outputs and the stem (22 nodes); INT8 selected the stem and
the SE reduce outputs (11 nodes).

| node | role | INT6 recovery [CI] | FP6 E2M3 recovery [CI] | INT8 |
|---|---|---|---|---|
| features_11_block_0_2 (Hardswish after the expand conv, first 112-channel block) | hswish | **17.5 [15.1, 19.9]** | **-9.8 [-12.1, -7.6]** | (not selected) |
| features_7_block_0_2 (same position, first 80-channel block) | hswish | 11.5 [9.2, 13.8] | -0.9 [-2.8, 1.1] | (not selected) |
| features_0_0 (stem conv output) | stem | 10.5 [8.2, 12.8] | 3.7 [1.4, 6.1] | 1.43 [-0.17, 3.1] |
| features_0_2 (stem Hardswish) | stem | 9.8 [7.5, 12.0] | 4.5 [2.3, 6.7] | **1.78 [0.22, 3.4]** |
| x (input) | stem | 3.2 [1.1, 5.2] | -0.3 [-2.4, 1.8] | -0.27 [-1.65, 1.13] |
| SE reduce outputs (8 nodes) | se_reduce | (not selected) | (not selected) | 0.0 to 0.98 (one interval excludes 0: features_11, [0.05, 1.95]) |

- Single-node SQNR shows no detectable ranking of the nodes: Spearman(node SQNR, recovery) 0.10 [-0.30, 0.45] at
  INT6, -0.02 [-0.41, 0.37] at FP6 (22 nodes), -0.52 [-0.89, 0.23] at INT8 (10 nodes with a finite stored SQNR, p 0.13).
- Leave-one-out recoveries do not add up: INT6 sum 72.3 pp against a 53.8 pp activation loss; FP6 sum -2.8 pp
  against 16.2 pp; INT8 6.5 pp against 3.4 pp (22, 22 and 11 nodes). The v1 summary `a2.json` also counted the
  addendum-5 arm `wide:node.features_11_block_0_2+fixbc` as a 23rd node (69.36 / -5.71); `v2/a2.json` excludes it
  (review finding B3).
- At INT8 the stem Hardswish output alone recovers about half of the loss; at FP6 the two stem outputs recover 4.5
  and 3.7 pp. The input `x` matters only at INT6.

### 2.4 The wide-arm anomaly (addenda 3 and 5)

Keeping some boundaries in FP32 loses accuracy: the expand-conv outputs (-1.9 INT8, -8.3 INT6, -10.2 FP6), the late
stages at INT6 (-5.9), and the single node features_11_block_0_2 at FP6 (-9.8, while it gains 17.5 at INT6). The
candidate cause is the per-arm bias-correction refit. The `+fixbc` arms keep the default's corrected biases; the
control `ref_default+fixbc` reproduces `ref_default` bit for bit (logits sha256 equal at INT8, INT6 and FP6 E2M3).

| arm (recovery vs default, pp) | INT8 refit | INT8 fixbc [CI] | INT6 refit | INT6 fixbc [CI] | FP6 refit | FP6 fixbc [CI] |
|---|---|---|---|---|---|---|
| `ref_weights_only` | +3.4 | -0.2 [-2.3, 1.9] | +53.8 | **-17.8** (3.2 % top-1) | +16.2 | +2.6 [-0.1, 5.3] |
| `wide:role.expand_conv` | **-1.9 [-3.6, -0.3]** | **+0.05 [-1.5, 1.7]** | -8.3 | -16.9 | **-10.2** | **-2.0 [-4.4, 0.3]** |
| `wide:position.late` | +1.1 | +0.55 | **-5.9** | **+0.2 [-1.6, 2.0]** | +2.3 | +3.1 [1.4, 4.9] |
| `wide:role.hswish` | +0.2 | +0.35 | +28.6 | -11.0 | +2.5 | 0.0 |
| `wide:role.stem` | +1.4 | -0.15 | +22.5 | +9.4 [6.9, 11.8] | +7.1 | +1.5 [-0.8, 3.6] |
| `wide:role.se_pool` | 0.0 | +0.13 | +0.9 | +1.1 | -2.5 | -1.1 |

- The refit causes the expand-conv loss at INT8 (-1.9 becomes +0.05) and FP6 (-10.2 becomes -2.0; both intervals
  include zero) and the INT6 late-stage loss (-5.9 becomes +0.2).
- At INT8, with the default's biases, even weights-only (all activations FP32) stays at the default (71.4, -0.2):
  those biases are tuned to the quantized activations, so `fixbc` is not a clean control at INT8 either, and every
  listed INT8 arm is within noise of the default under it.
- At INT6 the default's corrected biases are large and specific to its plan: the weights-only network with them
  falls to 3.2 %. So `fixbc` is not a clean no-refit control at INT6 for any group whose error the default corrected.
- Bias-correction failures belong to lane Q3; this is reported to it, and the attribution is read in both readings.
- Single node (addendum 5): keeping `features_11_block_0_2` FP32 with fixed biases gives -2.9 [-5.0, -0.8] at FP6
  (refit -9.8) and -2.9 [-5.2, -0.6] at INT6 (refit +17.5). So the refit makes the FP6 loss three times larger, and
  the INT6 gain of this node exists only with a refit.

## 3. Repairs at one uniform precision (Part B)

Top-1 (expected); in brackets the paired difference to the default with its 95 % interval. Per-channel rows are the
`pcu` arms (addendum 8); the superseded `pcf` numbers are in section 3.2.

| arm | INT8 | INT6 | FP6 E2M3 | FP8 E4M3 |
|---|---|---|---|---|
| default | 71.60 | 21.00 | 58.43 | 71.66 |
| r1a `affine:kind.hardswish` | 72.15 (+0.55 [-1.25, 2.40]) | 57.06 (+36.1 [33.1, 39.1]) | 57.22 (-1.21 [-3.03, 0.67]) | 71.66 (no zero point chosen) |
| r1b `affine:signed` | 72.85 (+1.25 [-0.60, 3.10]) | 57.16 (+36.2 [33.2, 39.1]) | 58.48 (+0.05 [-2.07, 2.16]) | **68.53 (-3.13 [-4.83, -1.44])** |
| r2 `pcu:role.dw_conv` | 71.70 (+0.10 [-1.15, 1.35]) | **18.69 (-2.31 [-4.24, -0.41])** | 59.38 (+0.95 [-0.98, 2.89]) | **69.87 (-1.79 [-2.94, -0.69])** |
| r2 `pcu:role.se_product` | 70.90 (-0.70 [-2.00, 0.60]) | **17.86 (-3.14 [-5.14, -1.14])** | 56.74 (-1.68 [-3.66, 0.26]) | 71.00 (-0.66 [-1.74, 0.43]) |
| `pcu:kind.hardswish` | 72.40 (+0.80 [-0.90, 2.55]) | 54.09 (+33.1 [30.3, 36.0]) | 61.90 (+3.48 [1.23, 5.78]) | **69.77 (-1.89 [-3.19, -0.63])** |
| `pcu:all` | 72.58 (+0.98 [-1.05, 2.98]) | **66.01 (+45.0 [42.0, 48.1])** | **68.80 (+10.4 [8.0, 12.8])** | **69.60 (-2.06 [-3.57, -0.55])** |
| `pc:all` (v1 samples) | 69.39 (-2.21 [-4.33, -0.10]) | 61.60 (+40.6 [37.7, 43.6]) | 66.81 (+8.38 [5.94, 10.88]) | 68.04 (-3.62 [-5.23, -2.03]) |
| combination `affine:signed` + best of the four r2/`pcu` arms above | 72.20 (+0.60 [-1.45, 2.60]; `pcu:all`) | 65.71 (+44.7 [41.7, 47.7]; `pcu:all`) | 68.33 (+9.91 [7.52, 12.38]; `pcu:all`) | **68.34 (-3.32 [-5.11, -1.53]; `pcu:role.se_product`)** |
| r3a `pcu:role.stem` | 73.70 (+2.10 [0.25, 3.95]) | 38.31 (+17.3 [14.7, 19.9]) | 63.77 (+5.34 [3.14, 7.52]) | 71.52 (-0.14 [-1.37, 1.05]) |
| r3b `affine:kind.hardswish+pcu:role.stem` | **74.10 (+2.50 [0.70, 4.35])** | 62.52 (+41.5 [38.5, 44.6]) | 63.68 (+5.26 [2.92, 7.59]) | 71.52 (= r3a: no zero point chosen) |
| r3c `pcu:node.features_0_0` | 73.05 (+1.45 [-0.40, 3.30]) | 31.01 (+10.0 [7.6, 12.4]) | 61.72 (+3.29 [0.90, 5.60]) | 71.64 (-0.02 [-1.25, 1.22]) |

Source: `results/summaries/b2-attrib-v1/v2/b.json`; figure `results/figures/b2-attrib-mbv3-repairs-v2.png`. The
combination arm was selected inside its job by the addendum-8 rule (best of `pcu:role.dw_conv`, `pcu:role.se_product`,
`pcu:kind.hardswish`, `pcu:all`); where a boundary is named by both terms, the per-channel term wins, so in
`affine:signed+pcu:all` zero points remain only at the input `x` and `classifier_3`.

- FP32 is 75.00 on the screen. **INT8:** the attribution-driven r3b gives 74.10, +2.50 [0.70, 4.35] over the default
  and -0.90 [-2.10, 0.30] to FP32: it closes about 74 % of the 3.40 pp and is the only arm whose point estimate is
  within 1 pp of FP32. The stem part alone (r3a) gives +2.10 [0.25, 3.95] (62 %) and halves the mean KL (0.209 ->
  0.101; r3b 0.075). The zero points add +0.40 on top of it (point estimate). Every other INT8 arm's interval includes zero (best
  `affine:signed` +1.25, r3c +1.45, `pcu:all` +0.98). r3b was chosen after looking at the screen, and its value
  moves by 2 pp with the per-channel sample (section 3.2): optimistic by construction.
- **INT6:** a zero point at the Hardswish outputs lifts the network from 21 % to 57 %; per-channel scales
  everywhere (`pcu:all`) reach 66.0 %. Adding zero points to `pcu:all` gives nothing more (65.7 %: only two zero
  points survive the overlap rule), and the stem-only r3b reaches 62.5 %. Per-channel scales at the depthwise
  outputs or at the SE products alone lose 2.3 and 3.1 pp (both intervals exclude zero). Even the best repair
  leaves INT6 9 pp below FP32 (`pcu:all` -8.99 [-11.17, -6.85]).
- **FP6 E2M3:** only per-channel scales help (`pcu:all` +10.4 [8.0, 12.8]). The two stem boundaries (r3a, +5.3) give
  about half the gain of all 118 per-channel boundaries. FP6 stays at least 6.2 pp below FP32.
- **FP8 E4M3:** every repair is neutral or harmful. `pcu:all`, `pcu:role.dw_conv` and `pcu:kind.hardswish` lose
  1.8-2.1 pp (intervals exclude zero); the stem arms are neutral (-0.14, -0.02). Zero points are an integer-only
  repair: the affine search keeps z = 0 at most float boundaries, and where it chose one (12 boundaries in
  `affine:signed`) FP8 lost 3.1 pp.
- The v1 `pc` arms clip: their scales come from 400-530 sparse samples per channel, anchored at the sample maximum
  (`features_16_2` loses 9.4 dB on the audit), and `pc:all` loses 2.2 pp at INT8. The `pcu` arms use the exact
  per-channel maximum as the anchor and are the per-channel result to read.

Regression (MobileNetV2 and ResNet18 INT8; paired difference to the default; `results/summaries/b2-attrib-v1/v2/b_regression.json`):

| arm | MobileNetV2 (default 72.30, FP32 72.10) | ResNet18 (default 69.22, FP32 70.10) |
|---|---|---|
| `affine:signed` | 72.10 (-0.20 [-0.80, 0.35]) | 69.03 (-0.18 [-0.78, 0.42]) |
| `pc:all` (v1 samples) | **68.90 (-3.40 [-5.15, -1.70])** | **68.23 (-0.99 [-1.99, -0.02])** |
| `pcu:all` | 71.98 (-0.32 [-1.47, 0.80]) | 69.03 (-0.18 [-1.13, 0.77]) |
| r3 `pcu:role.stem` | 72.25 (-0.05 [-0.70, 0.55]) | 69.21 (-0.01 [-0.60, 0.57]) |
| combination `affine:signed+pcu:all` (the MobileNetV3 INT8 selection) | 72.30 (0.00 [-1.10, 1.10]) | 68.92 (-0.30 [-1.28, 0.65]) |
| superseded `pcf:all` / `pcf:role.stem` | 47.05 / 72.68 | 67.95 / 67.97 |

- No affine or `pcu` repair helps or harms the two networks that are already near FP32 at INT8: point estimates
  -0.32 to 0.00 pp, every interval includes zero. Only the v1 `pc:all` (clipping samples) loses accuracy.
- The earlier `pcf:all` -25.25 pp on MobileNetV2 and the -1.24 pp of `pcf:role.stem` on ResNet18 were artefacts of
  the border-column sample: `pcu` minus `pcf` is +24.93 [22.16, 27.74] (MobileNetV2 `all`) and +1.23 [0.05, 2.46]
  (ResNet18 stem), with the fold and everything else unchanged.
- The fold diagnostic (addendum 6, rerun with `pcu` scales under addendum 8) compares, for every dense consumer that
  `pcu:all` folds into, the default weight codes `Qw(W)` with the folded codes `Qw(W r)` (range-weighted relative
  weight error `||(W_eff - W) r|| / ||W r||`; no accuracy measured):

  | network (INT8, `pcu` scales) | dense folded consumers | median error default -> folded | max folded | consumers > 2x default | Spearman(scale spread, error ratio) |
  |---|---|---|---|---|---|
  | MobileNetV2 | 35 | 0.0076 -> 0.0088 | 0.061 | 3 | +0.61 |
  | MobileNetV3 | 48 | 0.0078 -> 0.0097 | 0.028 | 2 | +0.61 |
  | ResNet18 | 19 | 0.0123 -> 0.0164 | 0.021 | 0 | +0.59 |

  (`results/summaries/b2-attrib-v1/v2/b_fold_diagnostic_v2.json`; the v1 run with `pcf` scales gave nearly the same
  table.) The fold does raise the weight error, most in the same three MobileNetV2 projections
  (`features_16/13/6_conv_2`, fed by depthwise ReLU6 outputs whose channel scales span 11-12.5 octaves), and no
  input channel is silenced by it (MobileNetV2 10 -> 10). Yet `pcu:all` costs MobileNetV2 only -0.32 [-1.47, 0.80].
  **The fold has no measurable accuracy cost at INT8 on the screen**; the fold-mechanism explanation that the r5
  draft gave for the `pcf:all` loss is withdrawn, and no fold guard is needed.

### 3.1 Hardware consequences

Accumulator width, from the B2 integer weight codes with the closed form of `tools/scaled_bridge_v2/certificates.py`
(`signed_bits_absolute = bit_length(max|a| * sum|w|) + 1`, per output channel, maximum over channels). The per-layer
default widths equal each network's sealed certificate (`artifacts/scaled_bridge_v2/runs/1f75c923…/<model>-<format>-default-b2/certificate.json`):
network maximum 24 bits for MobileNetV3 INT8, 20 for MobileNetV3 INT6, 25 for MobileNetV2 INT8 and 27 for ResNet18 INT8.
- Zero point: an affine input `x = s (q - z)` is accumulated as `sum w q - z sum w`; the per-output-channel term
  `z sum(w)` is a constant that can start the accumulator. Bound `max|q| sum|w| + |z| |sum w|`.
  - `affine:kind.hardswish` at INT8: zero points -102 to -125 at the 21 Hardswish outputs (-102 to -124 at those
    feeding MAC layers); +1 bit in 10 of the 15
    MAC layers fed by them; `affine:signed`: +1 bit in 12 of 28 layers. With the `z sum(w)` constant in the
    accumulator **the certified network maximum stays 24 bits** (INT6: 20 bits, +1 bit in 11 layers).
  - The other form, subtracting z from the operand first (a `|q - z|`-wide operand, bound `max|q - z| sum|w|`), costs
    more: +1 bit in 14 of the 15 layers, and the network maximum rises to **25 bits at INT8** (`classifier_3`,
    24 -> 25, fed by the last Hardswish) and **21 at INT6**. So "stays 24 / 20" holds only for the constant-term form.
    (Per-layer widths from the arm records; unchanged layers from the sealed certificate, whose per-layer defaults
    equal the records' in every changed layer; `v2/b.json` field `accumulator`.)
  - On MobileNetV2 and ResNet18 INT8, `affine:signed` chooses zero points at 4 boundaries each (z -71 to -6 and
    -50 to -12; -15 to -6 where they feed MAC layers) and adds no accumulator bit in either form (`v2.1/accumulator_model_matched.json`).
- Per-channel scales: folding into a conv/linear consumer is exact in real arithmetic (the consumer's weights become
  `W[o,c] r_c` before weight quantization). With `pcu` scales no MobileNetV3 width changes (`pcu:all`: 0 layers
  wider, network maximum 24 INT8 / 20 INT6). On MobileNetV2 `pcu:all` widens 1 layer by 1 bit but narrows the widest
  one, so the network maximum falls from 25 to 24 bits; on ResNet18 it falls from 27 to 26. Other consumers
  (elementwise ops, pool, add, multiply, Hardswish/Hardsigmoid) need per-channel requantization. Edge counts on
  MobileNetV3: `pcu:all` 63 folded and 73 requantized (118 boundaries); `pcu:kind.hardswish` 15 / 12;
  `pcu:role.dw_conv` 0 / 9 (depthwise outputs feed activations); `pcu:role.se_product` 8 / 0; MobileNetV2 `pcu:all`
  52 / 21; ResNet18 `pcu:all` 19 / 17.
- r3 (stem): `pcu:role.stem` puts per-channel scales on 2 boundaries (stem conv output, stem Hardswish). One edge
  folds into the first depthwise conv's weights (exact: one input channel per output channel, certified width
  unchanged at 24 bits INT8 / 20 bits INT6). Two edges need per-channel requantization: the stem Hardswish input and
  the first residual add. r3b adds zero points at the other 20 Hardswish outputs (-102 to -125 at INT8): +1 bit in
  10 layers at INT8 (11 at INT6) with the constant term, certified network maximum unchanged at 24 (20) bits; with
  the operand-subtract form 13 layers widen and the maximum becomes 25 (21). On the producer side, per-channel
  output scales are free: the stem conv already requantizes its accumulator per output channel (per-channel weight
  scales), so only its multipliers change. The cost sits at the two consumers that cannot fold the scales.
- The combination `affine:signed+pcu:all` has the cost of `pcu:all` (63 folded, 73 requantized edges) plus zero
  points at `x` (z = -15, feeding the stem conv; no width change) and `classifier_3` (the logits; no MAC consumer).
  No certified width changes. (Review 1 noted that the `pcf` version's "+1 bit in 1 layer" came from the fold; with
  `pcu` scales that layer no longer widens.)
- Correction (v2.1): `v2/b_regression.json` computed its certificate cross-check fields
  (`network_max_bits_arm_check`, `network_max_bits_presubtracted`, `certificate_matches_layer_defaults`,
  `presubtracted_*`) for MobileNetV2 and ResNet18 against the MobileNetV3 certificate. They are recomputed with each
  network's own certificate in `results/summaries/b2-attrib-v1/v2.1/accumulator_model_matched.json`
  (`tools/experiment_b2_attrib/regacc.py`); the MobileNetV3 rows of v2 are unchanged (0 differences), 14
  MobileNetV2/ResNet18 rows change, and with the matching certificate every record's per-layer defaults equal it.
  No accuracy number is affected.

### 3.2 Sampler sensitivity of the per-channel arms

The `pcf` arms of addendum 2 drew their per-channel search values from border columns (review B1); `pcu` draws a
seeded simple random sample of 4 096 positions per boundary. The reviewer reran seven arms with a third sampler (the
`pcf` stride raised until coprime with the map size; `artifacts/experiment_b2_attrib/review-scratch/`, not part of
the lane's sealed records):

| arm | `pcf` (border columns) | reviewer, coprime stride | `pcu` (addendum 8) |
|---|---|---|---|
| MobileNetV3 INT8 r3b `affine:kind.hardswish+…:role.stem` | 74.65 | **76.13** | **74.10** |
| MobileNetV3 INT8 r3a `…:role.stem` | 73.23 | 73.75 | 73.70 |
| MobileNetV3 INT6 `…:all` | 65.04 | 66.91 | 66.01 |
| MobileNetV2 INT8 `…:all` | **47.05** | 72.38 | 71.98 |
| MobileNetV2 INT8 `…:role.stem` | 72.68 | 72.55 | 72.25 |
| ResNet18 INT8 `…:all` | 67.95 | 68.73 | 69.03 |
| ResNet18 INT8 `…:role.stem` | 67.97 | 69.15 | 69.21 |

(`results/summaries/b2-attrib-v1/v2/sampler_sensitivity.json`.)
- The two unbiased samplers agree within 0.05-0.90 pp on six arms and differ by **2.03 pp on r3b**, the arm with
  the largest MobileNetV3 INT8 gain. The per-channel numbers therefore carry a sampler uncertainty of up to about
  2 pp, similar to the half-width of their paired intervals; `pcu` uses one seed.
- Across all 31 MobileNetV3 per-channel arm pairs, `pcu` minus `pcf` lies between -3.29 (INT6 `role.stem`, [-5.67, -0.94])
  and +1.90 pp; on MobileNetV2 `all` it is +24.93: the border sample hurt most where the maps are 112x112.
- Search floor: the per-channel MSE search stops at 2^-6 of the channel maximum. Under `pcu:all` at least one
  channel reaches it at 4 MobileNetV3 boundaries (SE ReLU outputs `features_{4,5,6,14}_block_2_activation`; the
  first is identically zero on the calibration images, as in the sealed default cell), 14 MobileNetV2 boundaries
  (ReLU6 outputs with channel spreads of 5.8-12.5 octaves; median channel ratio 0.85-1.0) and 1 ResNet18 boundary
  (`relu`). These are near-silent channels for which the MSE optimum is a very small scale; the arms that contain
  them are neutral or better (MobileNetV2 `pcu:all` -0.32), so this was not pursued.

## 4. Cross-network (Part C)

The GPU cap (10 800 s of job wall) left room for four of the ten planned cells: ResNet18 and MobileNetV2 at INT6 and
FP6 E2M3. FP6 E3M2, posit6 and LOG6 were not run.

| cell | default | weights-only | FP32 | activation loss | largest damage (group SQNR) | Spearman(group SQNR, damage) |
|---|---|---|---|---|---|---|
| ResNet18 INT6 | 65.89 | 68.60 | 70.10 | 2.71 | stem 1.00 (29.0 dB), classifier 0.83 | +0.41 (6 groups) |
| MobileNetV2 INT6 | 70.10 | 72.70 | 72.10 | 2.60 | stem 1.40 (33.0 dB, the highest) | +0.49 (6 groups) |
| ResNet18 FP6 E2M3 | 66.66 | 69.90 | 70.10 | 3.24 | stem 1.40 (27.0 dB; recovery 1.5 [0.3, 2.8]), classifier 1.07, block outputs 1.00 | +0.37 (6 groups) |
| MobileNetV2 FP6 E2M3 | 66.23 | 72.30 | 72.10 | 6.07 | **stem 2.90 (29 dB, the highest; recovery 4.7 [3.2, 6.2])**, classifier 0.54 | +0.38 (6 groups) |

- In all seven attributed cells the stem has the largest point-estimate damage (second at MobileNetV3 INT6, behind
  the Hardswish outputs). The paired difference to the second group separates them only in four cells: MobileNetV3
  INT8 (stem minus SE product +2.5 [0.8, 4.2]), FP6 E2M3 (stem minus Hardswish +5.8 [3.5, 8.1]), INT6 (Hardswish
  minus stem +7.6 [4.8, 10.4]) and MobileNetV2 FP6 E2M3 (stem minus classifier +2.4 [0.9, 3.9]). In ResNet18 INT6
  (stem 1.0 [-0.2, 2.2] vs classifier 0.83; difference +0.17 [-1.0, 1.4]), ResNet18 FP6 E2M3 (1.4 [0.0, 2.8] vs 1.07
  and 1.00; +0.33 [-0.9, 1.6]) and MobileNetV2 INT6 (1.4 [0.2, 2.6] vs 0.6; +0.8 [-0.4, 2.1]) the stem is not
  separated from the runner-up. The stem's group SQNR ranks 1st of 6 in all four Part C cells, 1st of 13 at
  MobileNetV3 FP6 E2M3 and 4th of 13 at MobileNetV3 INT8 and INT6 (section 2.2).
- At INT6 the two healthy networks spread their small activation loss thin (every recovery interval includes zero).
  At FP6 E2M3 the stem is the one group whose recovery interval excludes zero in both (ResNet18 +1.5, MobileNetV2
  +4.7 of a 6.1 pp loss); MobileNetV2's expand activations add +1.35 [0.3, 2.4].
- Within a cell no association between group SQNR and damage is detectable, and with 6 to 13 groups none of
  moderate size is excluded: Spearman(group SQNR, damage) is positive (the wrong sign) in all four Part C cells
  (+0.37 to +0.49, each with a bootstrap interval spanning about [-0.8, 1.0]).
- Pooled over all 63 groups of the seven attributed cells: Spearman(group SQNR, damage) -0.11 [-0.36, 0.14] (p 0.38;
  bootstrap resampling groups within cells), minimum SQNR -0.14 [-0.37, 0.11], group SQNR vs recovery -0.09
  [-0.35, 0.17]. The pooled rho mixes cells; the mean within-cell rho is +0.15 (stratified permutation p 0.30).
  Group SQNR vs the mean KL of `only:G`: pooled -0.27 [-0.52, -0.01] (raw p 0.03), but the mean within-cell rho is
  -0.05 (p 0.75). By network: MobileNetV3 -0.30 [-0.58, 0.00] (39 groups, p 0.065), MobileNetV2 +0.45 [-0.19, 0.86]
  and ResNet18 +0.32 [-0.35, 0.73] (12 each). After a Holm adjustment over all 42 SQNR-predictor tests
  (`v2/predictors.json`), no p-value stays below 0.05. Reading: no detectable association; a moderate one is not
  excluded.
- Across cells (network level, section 2.1) the median SQNR does rank the drop (pooled -0.92). There it tracks the
  bit width and the format, not where the error enters.
- Does the stem repair transfer (addendum 7, one arm, rerun with `pcu` under addendum 8)? MobileNetV2 FP6 E2M3
  `pcu:role.stem`: 67.27, +1.04 [-0.18, 2.25] over the default 66.23 (superseded `pcf`: 66.96), although keeping the
  stem FP32 recovers 4.7 pp there. Per-channel scales at MobileNetV2's one stem output boundary (`features_0_2`,
  folded into the first depthwise conv) recover +1.04 of those 4.7 pp (upper interval end 2.25); most of the stem
  error there is not a per-tensor-scale problem of the stem output. The untested candidate is the input `x`, which the per-channel arms
  exclude (MobileNetV2 has no leave-one-out).

## 5. Limits

- Development evidence on the 1k screen; the repair arms were chosen on it, so their gains are optimistic. With
  `pcu`, MobileNetV3 INT8 r3b reads 74.10; with the reviewer's sampler 76.13, above FP32 75.00: held-out
  confirmation is needed before any claim.
- Attribution is not additive (section 2.3) and some wide arms lose accuracy because of the bias-correction refit
  (section 2.4). `fixbc` is not a clean no-refit control: the default's biases are tuned to its own quantized plan.
- Per-channel calibration: `pcu` uses 256 calibration images, one seed and 4 096 positions per boundary. Two unbiased
  samplers differ by 0.05-0.90 pp on six arms and by 2.03 pp on r3b (section 3.2), so the per-channel numbers carry
  a sampler uncertainty of up to about 2 pp. The search floor (2^-6 of the channel maximum) is reached by near-silent
  channels at 4 / 14 / 1 boundaries of MobileNetV3 / MobileNetV2 / ResNet18 (section 3.2). The 36 `pcf` records
  (border-column sample, review B1) are kept as measured and superseded; no conclusion rests on them.
- The fold diagnostic measures weight error, not accuracy. With `pcu` scales the MobileNetV2 regression is gone, so
  the fold's weight error has no measurable accuracy effect at INT8 on the screen; this does not exclude an effect
  at lower precision.
- Accumulator widths are certified for INT8 and INT6 only (the float formats have no integer certificate here). The
  v2 regression summary compared MobileNetV2/ResNet18 against the MobileNetV3 certificate; v2.1 corrects it
  (section 3.1).
- 62 low-memory records (agent r3/r4 era) carry the hash of an earlier `lmrunner.py` (changed at 14:05 to add
  `+fixbc`; the file is not part of the arm identity). The reviewer reproduced one of them (INT8 `affine:signed`)
  bit for bit with the current file.
- GPU: the shared card was crowded; four jobs died of CUDA out-of-memory with the ~1 GB low-memory path and were
  retried as heavy jobs (rule 7). Up to four Q4 jobs waited for the heavy lock at once (two retries, two v1 verify
  jobs started by agent r1), above addendum 2's "at most 2 running or waiting"; at most two Q4 jobs ran at a time.
- GPU budget (deviation): 12 773 s of job wall in total, 3.55 GPU-hours against the brief's "about 3". Up to agent
  r5: 9 281 s recorded in the two job ledgers plus a 1 200 s allowance for failed and old-path jobs without a ledger
  line (an estimate, not checkable exactly), i.e. 10 481 of the protocol's 10 800 s. Addendum 8 (the review-1
  re-measurement) set its own cap of 3 600 s and used 2 292 s (7 jobs, all exit 0, longest 742 s;
  `artifacts/experiment_b2_attrib/pcu/logs/jobs.jsonl`).
- To stay under the 10 800 s cap (the protocol's stage-start rule), three addendum-2 combination lines were not run
  with `pcf` (`artifacts/experiment_b2_attrib/lowmem/queues/z/combo.dropped`); addendum 8 ran their `pcu`
  counterparts, selected by its rule (FP8 `affine:signed+pcu:role.se_product`, MobileNetV2/ResNet18 INT8
  `affine:signed+pcu:all`). Part C covers 4 of 10 planned cells
  (FP6 E3M2, posit6 and LOG6 were not run).

## 6. Reproduction

- Gate: `GPU_LANE=Q4 artifacts/agent_orchestration/gpu_run.sh --min-free-mib 3000 .venv-b/bin/python -m tools.run.experiment_b2_attrib_lm gate --model M --format F`
- Arms: `... -m tools.run.experiment_b2_attrib_lm arms --model M --format F --stage S --arms a,b [--audit]`
- Queues: `tools/run/experiment_b2_attrib_lm_queue2.sh LIST LOGDIR 3000`; lists in `artifacts/experiment_b2_attrib/lowmem/queues/`.
- Records: `artifacts/experiment_b2_attrib/arms/<model>--<format>/`.
- `pcu` arms (addendum 8): `tools/run/experiment_b2_attrib_pcu_queue.sh LIST LOGDIR 3000` with the lists
  `artifacts/experiment_b2_attrib/pcu/queues/{a,b}.list` (lines `EST args` of `-m tools.run.experiment_b2_attrib_pcu arms ...
  [--combo] [--regression-combo-from mobilenet_v3_large/int8]`); ledger `artifacts/experiment_b2_attrib/pcu/logs/jobs.jsonl`.
- Fold diagnostic (CPU): `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. nice -n 10 .venv-b/bin/python -m tools.experiment_b2_attrib.folddiag`
  (`pcf` scales, v1) and `-m tools.experiment_b2_attrib.folddiag_pcu` (`pcu` scales, v2).
- Summaries and figures (each written once; the writers refuse to overwrite):
  - v1: `PYTHONPATH=. .venv/bin/python -m tools.experiment_b2_attrib.summarize --write` and `-m tools.experiment_b2_attrib.figures`
    (`results/figures/b2-attrib-*-v1.{png,pdf}`);
  - v2: `-m tools.experiment_b2_attrib.summarize2 --write` (`results/summaries/b2-attrib-v1/v2/`) and
    `-m tools.experiment_b2_attrib.figures2` (`results/figures/b2-attrib-{mbv3-repairs,mbv3-leave-one-out}-v2.{png,pdf}`);
  - v2.1: `-m tools.experiment_b2_attrib.regacc --write` (`results/summaries/b2-attrib-v1/v2.1/accumulator_model_matched.json`).
- Tests (CPU, `.venv`): `tests/unit/test_experiment_b2_attrib.py`, `_lowmem.py`, `_folddiag.py`, `_pcu.py`.

## 7. For the owner

Decisions (none is taken here; every number is development evidence on the screen, and the repairs were picked on it):
- **Stem repair for MobileNetV3 INT8 (r3b).** It is the only arm whose point estimate brings MobileNetV3 INT8 within
  1 pp of FP32: 74.10, +2.50 [0.70, 4.35] over the default, -0.90 [-2.10, 0.30] to FP32. Cost: per-channel scales at
  2 stem boundaries (1 exact fold into the first depthwise conv, 2 per-channel requantizations) and zero points at
  20 Hardswish outputs (+1 accumulator bit in 10 layers with the `z sum(w)` constant; certified maximum unchanged at
  24 bits, or 25 if z is subtracted from the operand). Its stem part is neutral on MobileNetV2 and ResNet18 INT8
  (-0.05, -0.01) and at FP8 E4M3 (-0.14), and transfers only weakly to MobileNetV2 FP6 E2M3 (+1.04 [-0.18, 2.25]).
  Against it: chosen on the screen, and its value moves by 2 pp with the per-channel sampler (74.10 / 76.13). If
  adopted, it would be a MobileNetV3-specific recipe arm with a fixed sampler and seed, confirmed on held-out data
  before any claim.
- **Affine activation codes for integer formats.** They lift MobileNetV3 INT6 from 21 % to 57 %, are neutral on
  MobileNetV2/ResNet18 INT8 (-0.20, -0.18; no accumulator bit there), and cost +1 bit in about a dozen MobileNetV3
  layers. They do nothing or harm for float formats (FP8 -3.1 pp where the search chose a zero point).
- **Per-channel activation scales everywhere (`pcu:all`).** The best single repair for MobileNetV3 INT6 (66.0 %) and
  FP6 E2M3 (68.8 %); neutral at INT8 on all three networks (+0.98, -0.32, -0.18) and harmful at FP8 E4M3 (-2.06
  [-3.57, -0.55]). Cost: 73 per-channel requantization edges on MobileNetV3 (63 more fold exactly), no certified
  width increase on MobileNetV3, network maximum 25 -> 24 bits on MobileNetV2 and 27 -> 26 on ResNet18. The fold
  guard proposed in the r5 version is not needed: the MobileNetV2 -25 pp came from the `pcf` sample (review B1).
- **Bias-correction refit.** The per-arm refit creates the wide-arm losses (expand-conv outputs, late stages,
  `features_11_block_0_2`). This is reported to lane Q3, which owns bias-correction failures.
- **SQNR as a diagnostic.** For the paper: per-group (and per-node) SQNR shows no detectable association with where
  the accuracy is lost (pooled Spearman -0.11 [-0.36, 0.14]; mean within-cell +0.15; none of 42 tests survives
  Holm), though a moderate association is not excluded with 6-13 groups per cell. Only the network-level median
  SQNR tracks the drop across formats (-0.92).
