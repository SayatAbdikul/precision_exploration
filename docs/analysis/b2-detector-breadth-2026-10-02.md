# Experiment B2 detector breadth: 25 formats, calibration subsets, 6-bit attribution (2026-10-02)

Lane Q6. YOLOv8n under the repaired B2 detector recipes of lane L7 (`docs/analysis/b2-detector-2026-10-01.md`),
FP32 quantize-dequantize surrogate with a wide FP32 accumulator (no exact-engine or hardware claim). **Everything
below is development evidence on the COCO 1k screen** (`coco_screen_1k`, 1000 images, iterated from the manifest).
Ranges come only from `coco_calibration_2k` (the sealed v1 per-batch observations). No image outside the 1k screen
was opened. Metric: COCO bbox mAP50-95 in points (fixed tie rule of L7). Intervals: paired image bootstrap of the
dataset-level AP difference (2000 draws, seed 310911, L7's draw sequence, so pairs with L7's vectors are valid),
pointwise 95 percent, no multiplicity adjustment.

Protocol: `public/experiments/configs/breadth-study/b2-detector-breadth-protocol-v1.json` (sha256 d2f15d44...,
written 11:36 before any measurement; two edits of its own "written" label inside the same minute, no other
change). Summaries: `results/summaries/b2-detector-breadth-v1/` (`formats-1000`, `ties-1000`, `classifier-order`,
`seeds-1000`, `attribution-1000`, JSON and CSV). Figure: `results/figures/b2-detector-breadth-v2.{png,pdf}` (use this one; it sets
Q1.6 apart with ‡ in panel a; v1 is kept unchanged and still lists q1_6 inside the format ordering).
Evidence: `artifacts/experiment_b2_det_breadth/`. Code: `tools/experiment_b2_det_breadth/` (wraps L7's runner;
nothing of L7 was edited). The owner has not chosen between `default` (frozen) and `conformant` (the recipe the
written freeze rule prescribes); both are reported. For the block formats (bfp6, mxfp8_e4m3, mxfp6_e3m2,
mxfp4_e2m1) the conformant column is `default_fp32_box_logits`, the **nearest available recipe** (joins not
stored), marked †.

## 1. Result in brief

- **The 25-format table is complete** (section 3). Under `default`: INT8 38.02 ~ Posit8 38.11 > MXFP8 35.36 >
  FP8 E4M3 34.42 ~ FP7 34.15 ~ BFP6 33.98 ~ LOG8 33.67 > FP6 E2M3 32.13 > INT6 30.94 > FP6 E3M2 26.96
  ~ Posit6 26.66 > LOG6 25.53 ~ FP8 E5M2 25.51 > MXFP6 23.33 > FP5 14.76 ~ INT5 14.07 > MXFP4 1.14 (above the
  1-point stop threshold by 0.14, so run in full, but practically at chance); at chance (stop rule, below 1 point,
  seven formats): NF4 0.52, LOG4, Posit4, FP4, INT4, ternary and binary (all at or below 0.01). FP32 39.05. Q1.6
  (37.70) is left out of this ordering on purpose: its gap to INT8 is a recipe artefact, not a format property
  (next bullet).
- **`conformant` helps most where the box logits are coarsest**: +1.4 to +2.7 points for FP8 E5M2, FP6 E3M2,
  Posit6, LOG6 and MXFP6†, +0.0 to +0.4 at INT5, FP5, Q1.6 and MXFP4†. Under `conformant` the order changes only
  among close neighbours (FP6 E3M2 28.94 ~ Posit6 28.03 ~ LOG6 27.71 ~ FP8 E5M2 27.35 > MXFP6† 26.07).
- **Q1.6 is INT8 with signed codes everywhere, bit for bit**: Q1.6 `default` detections are bit-identical to L7's
  INT8 `default_signed` (detection sha256 6bd8fcf9..., mAP 37.6968 in both records; scales exactly 64x apart). The
  0.33 [0.05, 0.61] gap to INT8 under `default` (0.31 [0.02, 0.54] under `conformant`) is therefore the recipe's
  missing unsigned codebook for the fixed-point family (the input image, sigmoid scores and box expectation are
  stored with signed codes, one bit fewer of resolution there), not a property of the Q1.6 format. It must not be
  read as a format ranking: under a calibrated per-tensor scale Q1.6 is the INT8 grid up to a power-of-two factor,
  so with the same unsigned rule it would be expected to match INT8 (not measured). (The `conformant` gap has
  the same mechanism; L7 has no signed `conformant` INT8 record to check it bit for bit.)
- **Detector order follows the classifier matrix**: Spearman 0.92 to 0.96 and Kendall tau-b 0.76 to 0.86 between
  detector mAP and each classifier's `default` top-1 over 25 formats (section 4). The seven chance-level formats and
  MXFP4 occupy ranks 18 to 25 of every list and raise these coefficients.
- **Calibration draw** (section 5): over five disjoint 400-image calibration subsets the SD of mAP is 0.11 to 0.51
  point; it exceeds half the 0.5 margin of the protocol draft (0.25) for 7 of 12 format-recipe cells (INT8, Posit8 (0.254,
  barely), FP6 E2M3 and FP8 E4M3 under `conformant`, FP6 E2M3 and FP8 E4M3 under `default`, INT6 under `conformant`)
  and the full margin (0.5) for one (FP6 E2M3 `conformant`, 0.51). In four cells every
  subset value is below the full-set value: INT8 `conformant` (37.15 to 38.23 against 38.35), INT8 `default` (37.50
  to 37.97 against 38.02), LOG8 `conformant` (33.78 to 34.10 against 34.35) and LOG8 `default` (33.13 to 33.60
  against 33.67). **Posit8 minus INT8 is
  non-negative on every subset** (+0.01 to +0.65 `default`, +0.23 to +1.29 `conformant`), as on the full set.
  **LOG8 minus BFP6 under `conformant` depends on calibration size and draw**: +0.13 with the full set, -0.12 to
  -0.44 on all five 400-image subsets, and no interval (full set or subset) excludes zero. Every subset puts LOG8
  `conformant` 0.25 to 0.57 below its full-set value (BFP6 takes no calibration), so this is a calibration-size
  effect as much as a draw effect. Under `default` the gap is negative throughout.
- **Where the 6-bit loss lives** (section 6; diagnostic only): the loss is mostly activations (keeping all
  activations wide recovers 6.19 of INT6's 8.10 points and 5.11 of FP6's 6.92; keeping weights wide recovers 2.78
  and 1.58). No single group holds it: the largest group is the stem (input image, model_0, model_1), +1.86
  [+1.27, +2.46] at INT6 and +2.29 [+1.64, +2.94] at FP6 E2M3; every other group recovers 0.05 to 1.33. The neck
  recovers nothing measurable (+0.18 and +0.05, intervals contain zero). Inside the stem the two convolutions carry
  it in both formats (model_0 +1.10 INT6 / +1.09 FP6, model_1 +0.94 / +0.88, all intervals above zero); the 6-bit
  input image is not resolved (+0.43 [-0.08, +0.93] INT6, +0.51 [-0.09, +1.05] FP6).

## 2. Method

- **Wrapper** (`tools/experiment_b2_det_breadth/core.py`): L7's `runner.Session` (model, sealed observations,
  inputs, engine build, fixed-rule NMS) is subclassed; outputs go to `artifacts/experiment_b2_det_breadth/`.
  **Reproduction gate** (before any new measurement, `reproduction/int8-default-gate.json`): the plain path, the
  calibration-subset path with all 250 batches and the attribution path with an empty group all rebuild L7's INT8
  `default` configuration identity (1834351b1f74), and the detections on the 1k screen are bit-identical to L7's
  stored file (mAP 38.0238).
- **Part A**: 16 formats under `default` and the conformant arm; tie orders as L7 (fixed rule, reverse, random0..3 on
  the GPU from the same engine; `imageid` = plain pycocotools order on the CPU). Stop rule (protocol): a format below
  1.0 mAP under `default` skips the conformant arm and the extra orders. Tie-order detections are stored as
  references into the fixed-rule file (exact round trip, unit-tested; 55 to 65 percent smaller).
- **Part B**: the 250 calibration batches (8 images each) were split into five disjoint subsets of 50 batches by
  `numpy.random.default_rng(20261002).permutation(250)`; a subset's ranges are its batches' stored samples and
  maxima combined exactly as L7's loader does (unit test: the union of all batches reproduces the sealed summary and
  calibration identity). INT8, INT6, FP6 E2M3, Posit8, LOG8 and FP8 E4M3 under both recipes (60 configurations).
  BFP6 and MXFP8 take no calibration (block scale at run time; unit test: the engine builds identically from empty
  observations), so their full-set value holds for every subset.
- **FP8 E4M3 subsets** were the protocol's budget-conditional part ("only if the GPU budget allows after parts A
  and C", stated here as required). They ran at 14:56 (queued at 11:43 behind the other subsets), after part A and
  the part C group stage and the FP6 finer stage, but before the INT6 finer stage finished (17:07, delayed by the
  shared heavy-job lock). The budget condition held: the lane used about 2 GPU-hours of the 4.5 budgeted.
- **Part C**: one group kept wide at a time around `default` at INT6 and FP6 E2M3: its activation boundaries not
  quantized and its convolution weights (DFL group: projection constants) FP32; a pass-through node outside the group
  whose source left the grid quantizes with its own observed range. Groups: stem (input, model_0-1), the backbone
  stages P2 (model_2), P3 (3-4), P4 (5-6), P5 (7-8), SPPF (9), neck (10-21), heads P3/P4/P5 (per-level box and
  class convolutions including the network outputs), DFL (per-level expectation and projection constants), scores
  (per-level sigmoid); plus L7's whole-network weights-only and activations-only arms. Finer stage in the worst group
  (largest point gain; the stem for both formats): input | model_0 | model_1.

## 3. The 25 formats (screen1k)

`d` = `default`, `c` = `conformant` († nearest available). Differences are paired against L7's FP32 (39.05) and
INT8 vectors of the same recipe. Rows of the nine L7 sentinels repeat L7's values.

| format | bits | v1 max-abs | `d` | `d` minus FP32 | `c` | `c` minus FP32 | `c` minus `d` | distinct scores d / c |
|---|---|---|---|---|---|---|---|---|
| int8 | 8 | 31.59 | 38.02 | -1.02 [-1.52, -0.78] | 38.35 | -0.69 [-1.22, -0.53] | +0.33 [+0.12, +0.46] | 88 / 54 |
| posit8_es1 | 8 | 0.00 | 38.11 | -0.93 [-1.37, -0.72] | 38.42 | -0.63 [-1.10, -0.46] | +0.30 [+0.10, +0.44] | 164 / 75 |
| q1_6 ‡ | 8 | 31.59 | 37.70 | -1.35 [-1.87, -1.12] | 38.04 | -1.01 [-1.50, -0.82] | +0.35 [+0.18, +0.49] | 78 / 56 |
| mxfp8_e4m3 † | 8 | 19.11 | 35.36 | -3.69 [-4.51, -3.32] | 36.24 | -2.81 [-3.59, -2.46] | +0.88 [+0.69, +1.10] | 36 / 36 |
| fp8_e4m3fn | 8 | 15.98 | 34.42 | -4.63 [-5.54, -4.25] | 35.30 | -3.74 [-4.67, -3.42] | +0.89 [+0.61, +1.09] | 112 / 51 |
| fp7_e3m3 | 7 | 15.70 | 34.15 | -4.90 [-5.84, -4.47] | 35.13 | -3.92 [-4.87, -3.49] | +0.98 [+0.74, +1.21] | 110 / 47 |
| bfp6 † | 6 | 11.96 | 33.98 | -5.07 [-6.18, -4.68] | 34.22 | -4.83 [-5.96, -4.48] | +0.24 [+0.08, +0.34] | 38 / 38 |
| log8 | 8 | 16.77 | 33.67 | -5.38 [-6.41, -4.93] | 34.35 | -4.70 [-5.70, -4.25] | +0.68 [+0.46, +0.91] | 112 / 61 |
| fp6_e2m3 | 6 | 4.10 | 32.13 | -6.92 [-7.97, -6.45] | 32.92 | -6.13 [-7.21, -5.68] | +0.79 [+0.51, +1.00] | 26 / 18 |
| int6 | 6 | 0.22 | 30.94 | -8.10 [-9.24, -7.53] | 31.05 | -8.00 [-9.18, -7.44] | +0.11 [-0.19, +0.35] | 26 / 16 |
| fp6_e3m2 | 6 | 5.87 | 26.96 | -12.09 [-13.31, -11.50] | 28.94 | -10.11 [-11.35, -9.64] | +1.98 [+1.59, +2.25] | 54 / 25 |
| posit6_es1 | 6 | 0.00 | 26.66 | -12.39 [-13.63, -11.71] | 28.03 | -11.02 [-12.25, -10.39] | +1.37 [+1.02, +1.68] | 40 / 19 |
| log6 | 6 | 6.60 | 25.53 | -13.52 [-14.92, -12.75] | 27.71 | -11.34 [-12.72, -10.70] | +2.18 [+1.80, +2.53] | 41 / 26 |
| fp8_e5m2 | 8 | 7.28 | 25.51 | -13.54 [-14.79, -12.79] | 27.35 | -11.70 [-12.96, -11.00] | +1.84 [+1.45, +2.17] | 56 / 28 |
| mxfp6_e3m2 † | 6 | 5.51 | 23.33 | -15.72 [-17.41, -15.06] | 26.07 | -12.98 [-14.60, -12.41] | +2.74 [+2.38, +3.10] | 24 / 24 |
| fp5_e2m2 | 5 | 0.00 | 14.76 | -24.29 [-26.41, -23.68] | 14.88 | -24.17 [-26.36, -23.60] | +0.12 [-0.19, +0.34] | 9 / 5 |
| int5 | 5 | 0.00 | 14.07 | -24.98 [-27.17, -24.18] | 14.11 | -24.94 [-27.07, -24.11] | +0.04 [-0.14, +0.34] | 13 / 7 |
| mxfp4_e2m1 † | 4 | 0.02 | 1.14 | -37.91 | 1.15 | -37.90 | +0.01 [-0.10, +0.05] | 10 / 10 |
| nf4 | 4 | 0.00 | 0.52 | -38.53 | stop rule | | | 6 |
| log4, posit4_es0, fp4_e2m1, int4 | 4 | 0.00 | 0.00-0.01 | -39.04 to -39.05 | stop rule | | | 2-4 |
| ternary, binary_pm1 | 2, 1 | 0.00 | 0.00 | -39.05 | stop rule | | | 1 |

‡ Q1.6 `default` is bit-identical to L7's INT8 `default_signed` (section 1): its row measures the signed-code
recipe on the INT8 grid, not a separate format, and its gap to INT8 is not a format ranking.

Readings. (1) The 8-bit formats other than FP8 E5M2, and FP7, BFP6, FP6 E2M3 and INT6, stay within 8.1 points of
FP32; FP6 E3M2, Posit6, LOG6, FP8 E5M2 and MXFP6 lose 12 to 16 points and the 5-bit formats about 25. (2) With 7 or fewer distinct score values (INT5, FP5 under `conformant`) the signed join
re-store has nothing left to remove and the recipe difference vanishes. (3) Ternary and binary emit the 300-detection
maximum on every image with a single score value. (4) The v1 max-abs column carries the v1 head (L7 section 3); the
gap to it measures the repair, not the format.

**Ties** (`ties-1000`): for the nine new formats above the stop threshold (int5, fp5_e2m2, fp6_e3m2, fp8_e5m2,
log6, posit6_es1, mxfp6_e3m2†, mxfp4_e2m1†, q1_6; 18 format-recipe cells), the seven orders move mAP by at most 0.37 (FP6 E3M2
`default`, reverse order); Posit6 `conformant` 0.36 (reverse), LOG6 0.24 (0.34 range under `default`), INT5 0.26,
FP5 0.20, all others at most 0.24; Q1.6 at most 0.06. Plain pycocotools (`imageid`) is within -0.16 to +0.19 of
the fixed rule. 99.1 to 100 percent of scored detections share their score. Every recipe difference of section 3
that is above 1 point exceeds these moves; those of INT5, FP5 and MXFP4† (below 0.13) do not. For the seven
formats stopped by the rule, the GPU tie orders were not run as the protocol says, but the statistics step still
evaluated the CPU `imageid` order beside the fixed rule (`ties-1000` rows with orders = 2; no GPU cost); it moves
mAP by at most 0.06 (NF4) and no reported number uses it.

## 4. Beside the classifier matrix

Spearman / Kendall tau-b between detector mAP50-95 and the classifier `default` top-1 (`results/summaries/
b2-matrix-v1/cells-d4.csv`, `top1_expected`) over the 25 formats; `conformant` column with `default` for the
chance-level formats:

| classifier | `default` | `conformant` |
|---|---|---|
| ResNet18 | 0.961 / 0.853 | 0.964 / 0.860 |
| MobileNetV2 | 0.921 / 0.763 | 0.925 / 0.769 |
| MobileNetV3-Large | 0.959 / 0.847 | 0.961 / 0.853 |

Per-format ranks are in `classifier-order.csv`. The detector shares the classifiers' split: the top group (INT8,
Posit8, and Q1.6, which is INT8 under signed codes), then the 8-bit floats, block formats and LOG8, then the 6-bit formats with 3 mantissa bits, then the
2-mantissa-bit formats, then 5-bit, then 4-bit and below at chance. Visible difference: on the detector BFP6 sits with
the 8-bit floats (classifiers: ResNet18 level with FP6 E2M3, MobileNetV2 about 6 points below FP8 E4M3) and INT6 is below FP6 E2M3 (MobileNetV2
has INT6 ahead). The largest rank disagreements: INT6 and INT5 on MobileNetV2 (classifier ranks 4 and 11, detector
10 and 17: the classifier ranks them six places higher), and INT6 on MobileNetV3-Large, where the classifier collapses
(top-1 21.0, rank 15) while the detector keeps 30.94 (rank 10): the MobileNetV3 INT6 collapse (studied by lane Q3)
has no counterpart on YOLOv8n. On ResNet18 LOG8 ranks 4 against the detector's 8. These are rank statistics over a list with a large chance block; no test is attached.

## 5. Calibration subsets (5 disjoint 400-image subsets)

| format | recipe | subsets s0..s4 | mean | SD | full 2000 | SD > 0.25 |
|---|---|---|---|---|---|---|
| int8 | `d` | 37.97 37.90 37.50 37.82 37.93 | 37.82 | 0.19 | 38.02 | no |
| int8 | `c` | 37.68 38.23 37.15 37.87 38.21 | 37.83 | 0.44 | 38.35 | yes |
| posit8_es1 | `d` | 37.98 38.00 38.15 38.10 38.23 | 38.09 | 0.11 | 38.11 | no |
| posit8_es1 | `c` | 37.97 38.46 38.45 38.64 38.49 | 38.40 | 0.25 | 38.42 | yes (0.254) |
| fp8_e4m3fn | `d` | 34.29 34.76 33.87 34.24 34.47 | 34.33 | 0.33 | 34.42 | yes |
| fp8_e4m3fn | `c` | 34.47 35.45 35.04 35.16 35.59 | 35.14 | 0.44 | 35.30 | yes |
| log8 | `d` | 33.31 33.13 33.44 33.60 33.22 | 33.34 | 0.19 | 33.67 | no |
| log8 | `c` | 33.78 34.01 33.87 34.10 33.93 | 33.94 | 0.12 | 34.35 | no |
| fp6_e2m3 | `d` | 32.21 31.90 31.42 32.14 32.41 | 32.02 | 0.38 | 32.13 | yes |
| fp6_e2m3 | `c` | 32.67 32.55 31.72 32.21 33.08 | 32.45 | 0.51 | 32.92 | yes |
| int6 | `d` | 30.91 30.65 31.04 31.05 30.90 | 30.91 | 0.16 | 30.94 | no |
| int6 | `c` | 31.20 30.58 30.94 31.12 31.42 | 31.05 | 0.31 | 31.05 | yes |

Close gaps per subset (paired intervals on each subset; BFP6 is calibration-invariant):

| recipe | gap | full set | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|---|---|
| `d` | posit8 - int8 | +0.09 [-0.32, +0.53] | +0.01 | +0.10 | +0.65 [+0.16, +1.12] | +0.28 | +0.30 |
| `c` | posit8 - int8 | +0.06 [-0.32, +0.49] | +0.28 | +0.23 | +1.29 [+0.89, +1.87] | +0.77 [+0.40, +1.24] | +0.28 |
| `d` | log8 - bfp6 | -0.31 [-1.03, +0.55] | -0.67 | -0.85 | -0.54 | -0.37 | -0.75 |
| `c` | log8 - bfp6 † | +0.13 [-0.54, +1.02] | -0.44 | -0.21 | -0.35 | -0.12 | -0.29 |
| `d` | fp8 - int8 | -3.60 [-4.38, -3.10] | -3.68 | -3.14 | -3.63 | -3.58 | -3.46 |
| `c` | fp8 - int8 | -3.05 [-3.82, -2.55] | -3.21 | -2.77 | -2.12 | -2.72 | -2.62 |

Intervals not shown contain zero (FP8 subsets: point values; FP8 was not bootstrapped per subset, protocol part B).
Readings. (1) The seed SD is of the order of the margin's half at 400 images; at 2000 images it is expected to be
smaller, but that was not measured (no second 2000-image draw exists). (2) The full-set value is at or above the
subset mean for 11 of 12 cells (by up to 0.52, INT8 `conformant`; INT6 `conformant` is the exception, 31.049 against
a subset mean of 31.052, 0.003 below); a 400-image calibration is on average slightly worse. In four cells every
subset is below the full set: INT8 `conformant` (0.13 to 1.20 below), INT8 `default` (0.06 to 0.53 below), LOG8
`conformant` (0.25 to 0.57 below, mean 0.42, subset SD 0.12) and LOG8 `default` (0.07 to 0.54 below); in the other
eight cells at least one subset is above the full set.
(3) Posit8 is never below INT8; the full-set "level" reading (+0.09, +0.06) is at the low end of the subset spread,
and on two `conformant` subsets Posit8 is ahead with an interval excluding zero. (4) The sign of LOG8 minus BFP6
under `conformant` (L7: "LOG8 moves level with BFP6") depends on calibration size and draw: +0.13 on the full set,
negative on all five 400-image subsets, and no interval excludes zero on any of them. Because BFP6 is
calibration-invariant and every subset lowers LOG8 `conformant` by 0.25 to 0.57, the flip reflects the smaller
calibration set at least as much as the draw; the data support "level", not an order, in either direction. (5) FP8 E4M3 stays 2.1 to 3.7 points below INT8 on every subset.

## 6. Where the 6-bit loss lives (diagnostic only; no recipe)

Gain over `default` when one group is kept wide (screen1k; share = gain / loss to FP32; shares overlap and are not
additive). INT6 `default` 30.94 (loss 8.10), FP6 E2M3 32.13 (loss 6.92).

| group | INT6 gain | share | FP6 E2M3 gain | share |
|---|---|---|---|---|
| all activations wide (weights quantized) | +6.19 [+5.66, +7.20] | 0.76 | +5.11 [+4.64, +5.96] | 0.74 |
| all weights wide (activations quantized; the 3 DFL projection constants stay quantized, as in L7's arm) | +2.78 [+2.06, +3.46] | 0.34 | +1.58 [+0.94, +2.22] | 0.23 |
| stem (input, model_0, model_1) | **+1.86 [+1.27, +2.46]** | 0.23 | **+2.29 [+1.64, +2.94]** | 0.33 |
| stage P2 (model_2) | +0.82 [+0.25, +1.46] | 0.10 | +0.33 [-0.23, +0.78] | 0.05 |
| stage P3 (model_3-4) | +1.33 [+0.75, +1.90] | 0.16 | +0.52 [-0.07, +0.92] | 0.08 |
| stage P4 (model_5-6) | +0.82 [+0.35, +1.20] | 0.10 | +0.22 [-0.27, +0.69] | 0.03 |
| stage P5 (model_7-8) | +0.64 [+0.24, +0.98] | 0.08 | +0.29 [-0.11, +0.68] | 0.04 |
| SPPF (model_9) | +0.65 [+0.19, +0.99] | 0.08 | +0.31 [-0.16, +0.65] | 0.05 |
| neck (model_10-21) | +0.18 [-0.29, +0.54] | 0.02 | +0.05 [-0.37, +0.40] | 0.01 |
| head P3 | +0.96 [+0.70, +1.27] | 0.12 | +0.65 [+0.44, +0.86] | 0.09 |
| head P4 | +0.69 [+0.50, +0.99] | 0.08 | +0.66 [+0.41, +0.92] | 0.10 |
| head P5 | +0.74 [+0.43, +1.01] | 0.09 | +0.71 [+0.50, +0.99] | 0.10 |
| DFL (expectation, projection) | +0.61 [+0.48, +0.78] | 0.08 | +1.01 [+0.85, +1.19] | 0.15 |
| scores (sigmoid) | +0.21 [+0.13, +0.33] | 0.03 | +0.31 [+0.13, +0.47] | 0.05 |

Finer stage inside the stem (one unit kept wide at a time; share = gain / loss to FP32):

| stem unit | INT6 gain | share | FP6 E2M3 gain | share |
|---|---|---|---|---|
| input image | +0.43 [-0.08, +0.93] | 0.05 | +0.51 [-0.09, +1.05] | 0.07 |
| model_0 (first Conv block, 3x3 stride 2) | +1.10 [+0.61, +1.67] | 0.14 | +1.09 [+0.47, +1.49] | 0.16 |
| model_1 (second Conv block, 3x3 stride 2) | +0.94 [+0.39, +1.43] | 0.12 | +0.88 [+0.33, +1.30] | 0.13 |

Readings. (1) The 6-bit loss is spread: the twelve group gains sum to 9.5 (INT6) and 7.4 (FP6) points of gain, more
than the loss, so the errors compound rather than add, and no group alone recovers more than a third. (2) The stem
is the largest single place in both formats, and inside it the loss sits in the first two convolutions (each about
+1 point, intervals above zero, nearly the same in INT6 and FP6 E2M3); quantizing the input image to 6 bits costs
at most about one point and is not resolved from zero in either format (the INT6 input uses the unsigned 64-level
codebook, FP6 E2M3 its signed grid; the two gains are alike). (3) At FP6 E2M3 the backbone stages after the
stem matter little (intervals contain zero); at INT6 every backbone stage matters (+0.6 to +1.3); why the two
formats differ there was not diagnosed. (4) The neck recovers nothing measurable in either format. (5) Head
stores at 6 bits (DFL expectation and scores together +0.8 to +1.3) are a smaller share than the backbone, unlike
INT8, where L7 found most of the residual in the head.

## 7. Limits

- Development evidence on the 1k screen only; formats and recipes were not selected on it, but the finer stage of
  section 6 was chosen adaptively (worst group) as the protocol prescribes. No multiplicity adjustment.
- One model (YOLOv8n), one input size, the FP32 surrogate; block formats under the nearest available recipe (†).
- Calibration subsets are 400 images, a fifth of the size of the calibration used elsewhere; their SD is not a
  2000-image seed SD. Subsets are drawn from one 2000-image list (no new images).
- The stop rule removed the conformant arm and GPU tie orders of seven formats (int4, fp4_e2m1, posit4_es0, log4,
  nf4, ternary, binary_pm1); only the CPU `imageid` order was evaluated for them, unused; MXFP4 (1.14) passed it by
  0.14 point and was run in full.
- Tie orders: seven orders are a sample, not a bound (envelopes in `ties-1000`); every reported number uses the
  fixed rule.
- Attribution arms change two things at once for a group (activations and weights) and are not additive; they say
  where wide precision would recover quality, not what a mixed-precision recipe would achieve, and none is proposed.
- The classifier comparison is a rank correlation over a list whose bottom eight entries are the seven chance-level
  formats and MXFP4 (1.14); it is descriptive.
- "All weights wide" (L7's `default_activations_only`, `quantize_weights=False`) leaves the three DFL projection
  constants quantized (they follow `q_projection`); the per-group DFL arm of section 6 does keep them FP32.
- Inherited from L7's statistic (not a lane change): the resampled dataset-level mAP is biased upward (FP32 bootstrap
  draws average 40.28 against the point value 39.05), so percentile intervals of "minus FP32" differences sit
  off-centre around the point estimate, most visibly for badly degraded formats (MXFP4 -37.91 [-40.97, -37.39],
  not shown in the table). The same applies to every large paired difference: the "minus INT8" columns of badly
  degraded formats and the large attribution gains of section 6 (e.g. "all activations wide" INT6 +6.19
  [+5.66, +7.20], bootstrap draw mean +6.41). Paired differences between two similar configurations (section 5,
  tie orders, small attribution gains) are much less affected. Paper tables should not quote these intervals until
  the owner decides on the interval method (e.g. bias-corrected or basic bootstrap).
- No seconds-per-image are reported (no run was exclusive). GPU: about 2 hours of shared-queue time in total.

## 8. Reproduce

```
export GPU_LANE=Q6; G="artifacts/agent_orchestration/gpu_run.sh --min-free-mib 5000 .venv-b/bin/python -m tools.run.experiment_b2_det_breadth"
$G gate
$G formats --formats int5 fp8_e5m2 fp6_e3m2 posit6_es1      # likewise the other 12 formats
$G seeds --formats int8 int6 fp6_e2m3 posit8_es1 log8 --subsets 0   # subsets 0..4; fp8_e4m3fn likewise
$G groups --format int6 --groups stem stage_p2 stage_p3 stage_p4 stage_p5 sppf neck
$G groups --format int6 --groups head_p3 head_p4 head_p5 dfl scores weights_only activations_only
$G fine --format fp6_e2m3 --group stem --units input model_0 model_1
# INT6 finer stage (OOM retry): gpu_run.sh --heavy 8000 --wait 7200 ... fine --format int6 --group stem --units input model_0 model_1
nice -n 10 .venv-b/bin/python -m tools.experiment_b2_det_breadth.stats --workers 4
.venv/bin/python -m tools.experiment_b2_det_breadth.summary      # write-once; --show prints
.venv/bin/python -m tools.experiment_b2_det_breadth.figure                                      # v1 (17:11)
.venv/bin/python -m tools.experiment_b2_det_breadth.figure --out-tag v2 --set-apart q1_6        # v2 (17:52)
.venv/bin/python -m pytest tests/unit/test_experiment_b2_det_breadth.py -q
```

Logs and completion markers: `artifacts/experiment_b2_det_breadth/logs/` (`<tag>.log`, `<tag>.done`). One job
(`f-int6`) died of CUDA out-of-memory on the shared GPU and wrote nothing; its retry with `--heavy 8000`
(`f-int6-r2`) did not obtain the shared heavy-job lock within 3600 s and never ran; the next retry (`f-int6-r3`,
`--heavy 8000 --wait 7200`) waited 1 h 53 min for the lock and completed (exit 0).

## 9. Revision note

17:33, after independent review 1 (`artifacts/agent_orchestration/handoffs/Q6-detector-breadth-review.md`): text
only. Corrected counts (nine new formats above the stop threshold, seven stopped by the rule, MXFP4 counted
separately), Q1.6 = INT8 `default_signed` bit for bit (section 1, table ‡), LOG8 vs BFP6 worded as calibration
size/draw dependence, INT6 `conformant` exception in section 5 reading (2), FP8 E4M3 subsets stated as the
budget-conditional part, DFL projection constants in the "all weights wide" arm, the largest classifier rank
disagreements (section 4), and the inherited bootstrap bias of "minus FP32" intervals (section 7). No measurement,
summary file or figure changed; the facts added were checked against the stored records (detection sha256 of Q1.6
`default` and L7 INT8 `default_signed` both 6bd8fcf9...; FP32 draw mean 40.28 vs point 39.05 from L7's sealed FP32
bootstrap; subset differences from `seeds-1000.csv`; ranks from `classifier-order.csv`; `engine.py` line 260 for
the projection constants).

17:55, after independent review 2: (a) section 1 bullet and section 5 reading (2) now name all four cells in which
every 400-image subset is below the full set (INT8 and LOG8 under both recipes; was two), from `seeds-1000.csv`;
(b) new figure `results/figures/b2-detector-breadth-v2.{png,pdf}` with Q1.6 set apart below a rule and marked ‡
(panel a ordering covers 24 formats; panels b and c unchanged; v1 kept); (c) section 3 ties paragraph and section 7
state that the CPU `imageid` order was still evaluated for the seven stopped formats (unused; at most 0.06 mAP);
(d) section 3 merged row reads -39.04 to -39.05; (e) section 7 extends the interval-method caveat to the large
paired differences ("minus INT8" of degraded formats; "all activations wide" INT6 +6.19 [+5.66, +7.20] with draw
mean +6.41, recomputed from the stored vectors). No measurement or summary file changed.
