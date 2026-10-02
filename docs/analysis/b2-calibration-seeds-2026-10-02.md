# B2 calibration seeds and calibration size (lane Q5, 2026-10-02)

**Evidence class: development evidence.** Every number is measured on the frozen 1k ImageNet screen, on which the B2
recipes were selected. The five calibration "seeds" are disjoint 400-image subsamples of the one existing 2,000-image
calibration list (2 training images per class); they are not independent draws from the ImageNet training set (no
other training images exist on this machine). No image of the 10k list beyond the 1k screen and no COCO image was
opened. Protocol: `public/experiments/configs/breadth-study/b2-seeds-protocol-v1.json` (sha256 `79c6bbe1...`, written
before measurement) and addendum 1 (`...-v1-addendum-1.json`, sha256 `3bdfc7a2...`, a mechanism check added **after**
seeing the seed arm). Tables: `results/summaries/b2-seeds-v1/` (written once). Figures:
`results/figures/b2-seeds-{contrasts,size}-v1.{png,pdf}` and `results/figures/b2-seeds-cells-v2.{png,pdf}` (v2 replaces
`b2-seeds-cells-v1` for use, same data, labels no longer overlap; v1 is kept as written). Readout: expected-credit top-1 (primary) and
lowest-index top-1; numbers below are expected credit unless stated. 462 cells (426 protocol + 36 addendum), 2.74 h of
summed GPU cell time (plan about 2 h, hard stop 3 h).

**Revision 2 (2026-10-02 about 13:50Z), after review 1** (`artifacts/agent_orchestration/handoffs/Q5-calibration-seeds-review.md`).
No cell was re-measured and no summary or figure changed; the corrections are to this text. Corrected: the
MobileNetV3-Large 1,000-image column of the ladder table (section 4; revision 1 showed replicate H0 alone, the table
now shows the H0/H1 mean of `size.csv`); the statement on MobileNetV3-Large 6-bit contrasts (section 1: the magnitudes
are unstable, but 10 of the 15 keep their sign on every seed); the MXFP8 proposal-floor ranges (seed values only); the
named wide-exponent gains on ResNet18; recommendation 2 (it rested on MobileNetV2 contrasts only); the list of largest
minimal-recipe seed SDs; ResNet18 Posit6 2k value written as 59.595 in both places. Added: an erratum to the protocol
and addendum (`public/experiments/configs/breadth-study/b2-seeds-protocol-v1-erratum-1.json`), an exact-arithmetic
check of the sign columns and of the image-SD definition (`artifacts/experiment_b2_seeds/checks/review1-checks-v1.json`,
section 7), a unit test of the bias-correction inputs of a subset, and the run record of section 9 (three CUDA OOM
failures).

**Revision 3 (2026-10-02 about 14:10Z), after review 2** (approve; wording items only). No cell, summary or number
changed. New figure `results/figures/b2-seeds-cells-v2.{png,pdf}` (`tools/experiment_b2_seeds/figures_v2.py`: same
`seed-cells.csv`, short names, cells ordered by width and recipe; the script fails if any two x tick labels overlap).
Wording corrected: recommendation 2 (states the criterion under which three seeds sufficed, and the draft rule's
outcome for the same contrasts), the span of the five sign-changing MobileNetV3-Large 6-bit contrasts (section 1),
the chance-level cells of the image-SD check and the per-node sample count (section 7), and the exact seed means in
erratum item (i) (section 9).

## 1. Result in brief

- **Seed spread is small at 8 bits on ResNet18 and MobileNetV2, large on MobileNetV3-Large and at 6 bits.** Seed SD of
  top-1 over five 400-image seeds: ResNet18 8-bit 0.23 to 0.64 pp (INT8 0.25, Posit8 0.26, FP8 E4M3 0.51, Log8 0.64,
  MXFP8 0.23); MobileNetV2 8-bit 0.09 to 0.47; **MobileNetV3-Large INT8 1.32, FP8 E4M3 1.51, FP7 1.43, Log8 0.74,
  Posit8 0.50, MXFP8 0.22**. At 6 bits on MobileNetV3-Large: INT6 6.9, FP6 E2M3 1.4, FP6 E3M2 11.3, Log6 10.3,
  Posit6 11.1, BFP6 1.1 pp. Median over the 54 (model, format, recipe) cells: seed SD 0.69 pp against an image-sampling
  SD of 1.39 pp at n = 1,000.
- **33 of 54 cells have a seed SD above delta/2 = 0.5 pp**, among them MobileNetV3-Large INT8 and FP8 E4M3 and
  ResNet18 FP8 E4M3 and Log8.
- **Contrasts.** Of the 96 predeclared same-width and recipe contrasts (32 per network), the draft's five-seed rule
  (sign change or seed SD > 0.5 pp) fires for **81 after three seeds and 84 after five**; the sign changes across the
  five seeds in 30 of 96 (34 of 96 under lowest index). Median seed SD of a contrast 1.41 pp against an image SD of
  0.89 pp. The rule fires almost everywhere because the seed SD of a difference at 400 images is of the order of a
  point; it does not single out the unstable contrasts.
- **Stable conclusions.** Every 8-bit contrast on MobileNetV2 that the matrix separated, except Log8 - MXFP8 (separated on 3 of 5), stays separated on all five
  seeds (INT8 and Posit8 above FP8 E4M3, Log8 and MXFP8 by 3.5 to 5.4 points; seed SD 0.35 to 0.55). On ResNet18,
  INT8 - MXFP8 (+1.37, 4 of 5 seeds separated), Posit8 - MXFP8 (+1.65, 5 of 5) and FP7 - INT8 (-1.47, 5 of 5) hold;
  the other ResNet18 8-bit contrasts the matrix separated (INT8 - FP8 E4M3, Posit8 - FP8 E4M3, Posit8 - Log8; 2k
  difference 1.55 to 1.97) keep their sign on all five seeds but are separated on only 3 or 4 of them.
  INT8 - Posit8 is -0.27 (SD 0.24) on ResNet18 and +0.35 (SD 0.19) on MobileNetV2: equal within delta on every seed.
- **Unstable conclusions.** On MobileNetV3-Large the **sizes** of the 6-bit contrasts are unstable (seed SD 1.76 to
  18.31 pp, median 10.44), and 5 of the 15 change sign across the seeds: INT6 - Posit6, FP6 E3M2 - BFP6,
  FP6 E3M2 - Log6, FP6 E3M2 - Posit6 and Log6 - Posit6 (per-seed values from -27.7 to +23.2 across the five; Log6 -
Posit6 alone -17.6 to +23.2). The other 10 keep their sign on every
  seed, for example INT6 - FP6 E2M3 (-38.4 to -18.9) and FP6 E2M3 - BFP6 (SD 1.76, +0.21 to +5.00): FP6 E2M3 is above
  every other 6-bit format on every seed, BFP6 above Log6 and Posit6, and INT6 below all but Posit6; the sizes of these
  gaps are not stable. On MobileNetV3-Large at 8 bits: INT8 - Posit8
  -3.33 (range -5.06 to -0.37; full set -1.76), INT8 - FP8 E4M3 +0.23 (-1.00 to +2.37), INT8 - MXFP8 -0.43 (-1.39 to
  +1.71; full set +1.54), FP8 E4M3 - Log8 +0.42 (-2.03 to +2.74). On ResNet18 the wide-exponent 6-bit contrasts shift
  by about half their full-set size (section 4).
- **Matrix outcomes (report only).** The exact-engine proposal's quality floor gives the same pass/fail for all twelve
  proposal formats measured here on every seed (closest: MXFP8 E4M3 mean drop 3.69 to 4.03 and worst drop 4.51 to 5.50
  over the five seeds, 4.28 and 5.54 with the full set; Log8 worst drop 4.31 to 6.32 over the seeds, 5.17 with the full
  set; both pass the 5.0 / 10.0 floor on every seed). Rule (c) at 6 bits: R_bits(6) loses fewer
  eligible 6-bit scalar cells than R_bits(7) on every seed (3 or 4 against 7 or 8; full set 4 against 7); 4 of the 15
  per-cell recipe picks change on at least one seed. The rule (b) class against the integer of the same width changes on
  at least one seed for 9 of 27 (model, format) pairs (for example ResNet18 FP8 E4M3: within one point on two seeds,
  separated below on three).
- **Calibration size.** Smaller calibration sets are not uniformly worse. ResNet18 and MobileNetV2 8-bit means at 32,
  128, 400 and 1,000 images are within 0.5 pp of the 2,000-image value. **MobileNetV3-Large INT6 is 26.9 points better
  with 32 images (47.9 vs 21.0)**, 19.1 with 128, 6.9 with 400 and 10.4 with 1,000. Two wide-exponent 6-bit formats on
  ResNet18 gain with 400 images: FP6 E3M2 +5.89 and Log6 +4.76 (five-seed means); Posit6 does not (+0.33). On
  MobileNetV2 FP6 E3M2, Log6 and Posit6 gain +5.62, +3.83 and +9.33. The 2,000-image calibration is therefore not a
  converged reference for these cells.
- **Mechanism (addendum 1, after seeing the data).** On ResNet18 the 400-image gain of FP6 E3M2 and Log6 comes from the
  range statistics (range-only arm +5.9 and +5.5 against seed +5.4 and +5.4; bias-only +0.4 and +1.2). On MobileNetV2
  it comes from the bias-correction images (bias-only +7.4 and +5.6; range-only -0.1 and -0.3): the original first 256
  images are worse than all three alternative 256-image draws for these formats. Posit6 on MobileNetV2 gains from both
  (+5.4 range, +5.8 bias, +11.4 together). On MobileNetV3-Large INT8 the seed SD is a range effect (bias-only SD 0.17
  against seed SD 1.32).

## 2. Per-cell seed spread (`seed-cells.csv`, figure `b2-seeds-cells-v2`; v1 has overlapping labels)

Five seeds, default-type recipe unless marked; "2k" is the existing matrix cell (2,000 range images, first 256 list images
for bias correction). Full table with min, max, lowest-index columns and per-seed values in `seed-cells.csv`.

| model | format | seed mean | seed SD | range | 2k | image SD (n=1000) |
|---|---|---:|---:|---:|---:|---:|
| ResNet18 | INT8 | 69.12 | 0.25 | 0.58 | 69.22 | 1.44 |
| ResNet18 | Posit8 es1 | 69.39 | 0.26 | 0.71 | 69.63 | 1.40 |
| ResNet18 | FP8 E4M3 | 67.90 | 0.51 | 1.40 | 67.67 | 1.38 |
| ResNet18 | Log8 | 68.26 | 0.64 | 1.45 | 68.08 | 1.37 |
| ResNet18 | MXFP8 E4M3 (cum5) | 67.75 | 0.23 | 0.64 | 67.73 | 1.39 |
| ResNet18 | FP7 E3M3 | 67.65 | 0.37 | 1.00 | 67.48 | 1.39 |
| ResNet18 | INT6 | 65.67 | 0.26 | 0.55 | 65.89 | 1.42 |
| ResNet18 | FP6 E2M3 | 67.30 | 0.19 | 0.52 | 66.66 | 1.38 |
| ResNet18 | FP6 E3M2 | 59.51 | 2.11 | 5.55 | 53.62 | 1.38 |
| ResNet18 | BFP6 (cum5) | 68.31 | 0.31 | 0.78 | 68.20 | 1.42 |
| ResNet18 | Log6 | 59.30 | 1.51 | 4.08 | 54.54 | 1.39 |
| ResNet18 | Posit6 es1 | 59.93 | 0.91 | 1.98 | 59.595 | 1.38 |
| MobileNetV2 | INT8 | 72.40 | 0.21 | 0.52 | 72.30 | 1.41 |
| MobileNetV2 | Posit8 es1 | 72.05 | 0.09 | 0.23 | 72.39 | 1.39 |
| MobileNetV2 | FP8 E4M3 | 67.93 | 0.47 | 1.15 | 67.71 | 1.42 |
| MobileNetV2 | Log8 | 68.56 | 0.38 | 0.97 | 68.17 | 1.41 |
| MobileNetV2 | MXFP8 E4M3 (cum5) | 67.00 | 0.41 | 0.99 | 66.56 | 1.43 |
| MobileNetV2 | FP7 E3M3 | 67.86 | 0.74 | 1.67 | 67.62 | 1.42 |
| MobileNetV2 | INT6 | 69.86 | 1.01 | 2.73 | 70.10 | 1.39 |
| MobileNetV2 | FP6 E2M3 | 66.97 | 0.61 | 1.63 | 66.23 | 1.42 |
| MobileNetV2 | FP6 E3M2 | 26.48 | 1.66 | 3.48 | 20.86 | 1.30 |
| MobileNetV2 | BFP6 (cum5) | 66.42 | 0.52 | 1.33 | 65.60 | 1.46 |
| MobileNetV2 | Log6 | 24.16 | 4.41 | 9.28 | 20.34 | 1.26 |
| MobileNetV2 | Posit6 es1 | 28.12 | 4.14 | 10.47 | 18.79 | 1.31 |
| MobileNetV3-L | INT8 | 70.47 | 1.32 | 3.52 | 71.60 | 1.43 |
| MobileNetV3-L | Posit8 es1 | 73.80 | 0.50 | 1.25 | 73.36 | 1.36 |
| MobileNetV3-L | FP8 E4M3 | 70.23 | 1.51 | 3.81 | 71.66 | 1.39 |
| MobileNetV3-L | Log8 | 69.82 | 0.74 | 2.01 | 69.83 | 1.40 |
| MobileNetV3-L | MXFP8 E4M3 (cum5) | 70.90 | 0.22 | 0.58 | 70.06 | 1.39 |
| MobileNetV3-L | FP7 E3M3 | 69.92 | 1.43 | 3.23 | 71.12 | 1.39 |
| MobileNetV3-L | INT6 | 27.92 | 6.92 | 17.93 | 21.00 | 1.31 |
| MobileNetV3-L | FP6 E2M3 | 58.79 | 1.38 | 3.30 | 58.42 | 1.48 |
| MobileNetV3-L | FP6 E3M2 | 43.09 | 11.33 | 28.86 | 44.57 | 1.43 |
| MobileNetV3-L | BFP6 (cum5) | 56.14 | 1.11 | 2.38 | 56.60 | 1.53 |
| MobileNetV3-L | Log6 | 45.42 | 10.31 | 22.41 | 53.16 | 1.43 |
| MobileNetV3-L | Posit6 es1 | 41.54 | 11.05 | 22.73 | 42.91 | 1.42 |

Minimal-recipe cells (6-bit scalar formats and NF4) are in the same table; the largest
seed SDs there are MobileNetV3-Large FP6 E2M3 7.73, Log6 4.79 and FP6 E3M2 2.82, ResNet18 Log6 1.45 and NF4 1.27; all
others are below 1 pp. For the shared-exponent formats
only bias correction depends on the calibration subset; their seed SDs (0.22 to 1.11) are bias-correction effects.

## 3. Contrasts and the five-seed rule (`contrasts.csv`, figure `b2-seeds-contrasts-v1`)

Per network, 10 pairs among the 8-bit formats, 2 FP7 pairs, 15 pairs among the 6-bit formats, 5 default-minus-minimal
pairs (rule c). Columns: needs five after 3 / after 5 seeds; seed mean, SD, min and max of the per-seed paired
difference; the 2k difference.

| network | group | contrasts | five needed after 3 | after 5 | sign changes (5 seeds) | median seed SD | median image SD |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | 8-bit | 10 | 7 | 8 | 4 | 0.59 | 0.55 |
| ResNet18 | FP7 vs 8-bit | 2 | 1 | 1 | 1 | 0.50 | 0.52 |
| ResNet18 | 6-bit | 15 | 12 | 12 | 3 | 1.38 | 0.82 |
| ResNet18 | rule (c) | 5 | 5 | 5 | 2 | 1.01 | 0.90 |
| MobileNetV2 | 8-bit | 10 | 4 | 4 | 1 | 0.42 | 0.79 |
| MobileNetV2 | FP7 vs 8-bit | 2 | 1 | 2 | 1 | 0.76 | 0.71 |
| MobileNetV2 | 6-bit | 15 | 14 | 15 | 4 | 3.09 | 1.50 |
| MobileNetV2 | rule (c) | 5 | 5 | 5 | 1 | 1.45 | 1.32 |
| MobileNetV3-L | 8-bit | 10 | 10 | 10 | 5 | 1.47 | 0.82 |
| MobileNetV3-L | FP7 vs 8-bit | 2 | 2 | 2 | 2 | 0.84 | 0.77 |
| MobileNetV3-L | 6-bit | 15 | 15 | 15 | 5 | 10.44 | 1.38 |
| MobileNetV3-L | rule (c) | 5 | 5 | 5 | 1 | 8.35 | 1.42 |

Selected contrasts (seed mean, SD, [min, max] over five seeds; 2k difference):

| network | contrast | seeds | 2k |
|---|---|---|---:|
| ResNet18 | INT8 - Posit8 | -0.27, 0.24 [-0.60, 0.00] | -0.42 |
| ResNet18 | INT8 - FP8 E4M3 | +1.22, 0.73 [+0.10, +2.05] | +1.55 |
| ResNet18 | FP8 E4M3 - Log8 | -0.36, 0.92 [-1.07, +1.23] | -0.42 |
| ResNet18 | INT6 - FP6 E2M3 | -1.62, 0.35 [-2.16, -1.22] | -0.77 |
| ResNet18 | FP6 E3M2 - Posit6 | -0.42, 2.71 [-4.51, +2.36] | -5.97 |
| ResNet18 | Log6 - Posit6 | -0.63, 1.36 [-2.10, +1.52] | -5.05 |
| MobileNetV2 | INT8 - Posit8 | +0.35, 0.19 [+0.03, +0.53] | -0.09 |
| MobileNetV2 | INT8 - FP8 E4M3 | +4.47, 0.55 [+3.97, +5.34] | +4.59 |
| MobileNetV2 | FP6 E2M3 - BFP6 | +0.56, 0.76 [-0.61, +1.31] | +0.64 |
| MobileNetV2 | Log6 - Posit6 | -3.96, 3.09 [-7.51, +0.67] | +1.55 |
| MobileNetV3-L | INT8 - Posit8 | -3.33, 1.77 [-5.06, -0.37] | -1.76 |
| MobileNetV3-L | INT8 - MXFP8 | -0.43, 1.25 [-1.39, +1.71] | +1.54 |
| MobileNetV3-L | Posit8 - FP8 E4M3 | +3.56, 1.92 [+1.32, +6.38] | +1.70 |
| MobileNetV3-L | FP8 E4M3 - Log8 | +0.42, 1.97 [-2.03, +2.74] | +1.82 |
| MobileNetV3-L | INT6 - FP6 E2M3 | -30.87, 7.77 [-38.39, -18.90] | -37.43 |
| MobileNetV3-L | Log6 - Posit6 | +3.88, 18.31 [-17.65, +23.24] | +10.25 |

## 4. Calibration size and the source of the seed effect (`size.csv`, `bias-only.csv`, `addendum-mechanism.csv`)

Ladder (top-1 minus the 2k value, mean over replicates; 3 replicates at 32 and 128 images, 5 at 400, 2 at 1,000;
bias correction on min(n, 256) images of the same subset):

| network | format | 32 | 128 | 400 | 1000 |
|---|---|---:|---:|---:|---:|
| ResNet18 | INT8 | -0.33 | -0.12 | -0.10 | -0.05 |
| ResNet18 | Posit8 | -0.49 | -0.46 | -0.24 | -0.22 |
| ResNet18 | FP8 E4M3 | +0.34 | +0.40 | +0.24 | +0.16 |
| ResNet18 | INT6 | -1.02 | -0.23 | -0.22 | -0.37 |
| MobileNetV2 | INT8 | +0.19 | +0.03 | +0.10 | +0.02 |
| MobileNetV2 | Posit8 | -0.19 | -0.42 | -0.34 | -0.13 |
| MobileNetV2 | FP8 E4M3 | -0.06 | +0.25 | +0.22 | -0.04 |
| MobileNetV2 | INT6 | -1.06 | +0.27 | -0.24 | -0.33 |
| MobileNetV3-L | INT8 | -0.13 | +0.35 | -1.13 | -0.30 |
| MobileNetV3-L | Posit8 | +0.52 | +0.37 | +0.44 | +0.14 |
| MobileNetV3-L | FP8 E4M3 | -0.84 | -1.34 | -1.43 | -0.62 |
| MobileNetV3-L | INT6 | +26.90 | +19.12 | +6.92 | +10.42 |

(Revision 1 printed +0.34, -1.53 and +12.59 in the last three rows of the 1,000 column: those were replicate H0 alone,
from a partial analysis before the three H1 cells finished. The figure `b2-seeds-size-v1` was right.)

Bias-only arm (ranges from all 2,000 images, bias correction from the first 256 images of S_k): seed SD of top-1
0.16 to 0.37 pp for the 8-bit formats on ResNet18 and MobileNetV2, 0.17 (INT8), 0.30 (Posit8), 0.47 (FP8 E4M3) and
0.72 (INT6) on MobileNetV3-Large, against 1.32, 0.50, 1.51 and 6.92 in the seed arm. At production range size, the
bias-correction draw alone moves 8-bit top-1 by about 0.2 to 0.5 pp (SD).

Addendum 1 (mean over k = 0, 1, 2, minus the 2k value):

| network | format | 2k | seed (S) | range-only (R) | bias-only (B) |
|---|---|---:|---:|---:|---:|
| ResNet18 | FP6 E3M2 | 53.62 | +5.36 | +5.90 | +0.42 |
| ResNet18 | Log6 | 54.54 | +5.41 | +5.51 | +1.18 |
| ResNet18 | Posit6 | 59.595 | +0.35 | +0.18 | -0.43 |
| MobileNetV2 | FP6 E3M2 | 20.86 | +6.43 | -0.12 | +7.44 |
| MobileNetV2 | Log6 | 20.34 | +5.65 | -0.34 | +5.64 |
| MobileNetV2 | Posit6 | 18.79 | +11.39 | +5.40 | +5.78 |

A plausible reading, not tested: more calibration images raise the per-node maxima, and the activation scale search
is anchored to the maximum (its grid is a ratio to the max-abs scale), so rare large activations pull the chosen
scale for the wide-exponent formats and for INT6 on MobileNetV3-Large.

## 5. Recommendation for the confirmation protocol (proposal, owner decision)

1. **Do not use the draft rule's SD clause as written.** With seeds of this kind the seed SD of a contrast exceeds
   delta/2 = 0.5 pp in 82 of 96 contrasts; the rule would extend almost every contrast to five seeds after the first
   three (81 of 96) and still does not separate stable from unstable contrasts. Replace it by a rule on the total
   uncertainty, for example: report SE_total = sqrt(SE_image^2 + SD_seed^2 / k) and extend to five seeds when
   SD_seed / sqrt(3) exceeds SE_image or the sign changes.
2. **Five seeds from the start** for every contrast on MobileNetV3-Large and for every 6-bit contrast. Under the
   candidate rule stated at the end of this item, the ResNet18 and MobileNetV2 8-bit contrasts that could stop at three
   seeds are the six MobileNetV2 contrasts between {INT8, Posit8} and {FP8 E4M3, Log8, MXFP8} (2k difference 4.1 to
   5.8 points, five-seed SD 0.35 to 0.55, three-seed SD 0.31 to 0.66, separated on five of five seeds); no ResNet18
   8-bit contrast has a 2k difference above 3 points (largest Posit8 - FP8 E4M3, 1.97), although ResNet18 Posit8 -
   MXFP8 (1.90) and FP7 - INT8 (-1.73) were also separated on five of five seeds. Under the draft rule instead, 15 of
   the 96 contrasts needed no extension after three seeds, and two of the six MobileNetV2 contrasts (INT8 - FP8 E4M3
   and Posit8 - FP8 E4M3, three-seed SD 0.66 and 0.58) did trigger it. The ResNet18 8-bit contrasts the
   matrix separated (1.5 to 2.0 points) kept their sign on five of five seeds but were separated on only 3 to 5, so they
   need five seeds if separation, not only the sign, is the claim. Across all 96 contrasts, the 10 whose 2k difference
   exceeds 3 points and whose seed SD over the first three seeds is below 0.7 pp (the six MobileNetV2 contrasts above,
   MobileNetV2 FP7 - INT8 and INT6 - FP6 E3M2, MobileNetV3-Large Posit8 - MXFP8, ResNet18 Log6 default - minimal) all
   kept their sign on five of five seeds and were separated on 4 or 5. A candidate rule is therefore "three seeds
   when the 2k difference exceeds 3 points and the three-seed SD is below 0.7 pp, five otherwise"; the thresholds
   were read off these data and would need a check on new calibration draws.
3. **Report the calibration size as a factor** for MobileNetV3-Large INT6 and the wide-exponent 6-bit formats: the
   2k calibration is not a converged reference for them, and a single draw of any size can move them by 5 to 27 points.
4. **Seeds should vary both the range images and the bias-correction images**: the two sources dominate in different
   cells (section 4).
5. Seeds drawn as subsamples of one 2k list understate between-list variation and, at 400 images, overstate the range
   part of the variation of a 2,000-image draw; a confirmation run with new calibration lists (training images not on
   this machine) would remove both limits.

## 6. Limits

- Development evidence on 1,000 images on which the recipes were selected; pointwise intervals, no multiplicity
  correction.
- Seeds are disjoint subsamples of one 2k list (2 per class), not independent draws; seed subsets have 400 range images
  against 2,000 in production (bias correction 256 in both), so the range part of the seed SD is for a smaller
  calibration than the one reported elsewhere. The bias-only arm gives the production-size bias part.
- Five seeds give a rough SD (its own relative standard error is about 35 percent).
- Addendum 1 (motivation corrected in erratum 1, section 9) was added after seeing the seed arm and covers 3 formats, 2 networks and 3 seeds; it is a mechanism
  check, not a test.
- Shared-exponent formats were run only in the proposal arm (`cum5_act_maxabs`); the searched default arm and NF4/ternary
  under other recipes were not run; ternary was not run (chance everywhere in the matrix).

## 7. Method

- **Subsets.** The v1 observation cache stores range statistics per batch of 8 consecutive rows of the
  sha256-sorted calibration list (250 batches): min(256, elements per image) sampled values per image and quantizing
  node (nodes with fewer elements store all of them, e.g. 120 for MobileNetV3-Large `features_11_block_2_activation`),
  and the exact
  max-abs of the batch. A subset is a set of whole batches. `numpy.random.default_rng(20261002).permutation(250)`
  split into five blocks of 50 gives the seeds S0..S4 (400 images each, disjoint, together the whole list). The
  ladder takes the first 4 and 16 batches of S0..S2 in permutation order (32 and 128 images, nested in the seed),
  and the two halves of a second permutation (`default_rng(20261003)`, 1,000 images each). The bias-only arm B0..B4
  keeps the ranges of all 2,000 images and draws only the bias-correction images from S_k.
- **Ranges from a subset.** Samples are concatenated in ascending batch order and maxima are the maximum over the
  selected batches, which is the `tools/experiment_b2/data.py::v1_calibration` code path restricted to those batches.
  With all 250 batches the arrays, maxima and identity record are bit-identical to `v1_calibration` for all three
  networks (`tests/unit/test_experiment_b2_seeds.py`), and the five seeds' samples add up exactly to the full set.
- **Bias correction from a subset.** The first min(256, n) images of the subset in ascending sha256 order (the original
  rule applied to the subset). Their preprocessed tensors come from the existing 256-image input-cache entry where they
  overlap (48 of 256 for S0) and otherwise from the v1 `image_batch` on the JPEG in the running process; every
  process checks that the two paths agree bit for bit on two cached images. Nothing was persisted and the 2,000-image
  cache entry was not built.
- **Cells.** Built by the unchanged B2 code (`runner.build`, `matrix.build_shared`, `matrix.evaluate`, batch 8,
  tie-aware readout) with three module attributes replaced inside the lane's own process
  (`tools/experiment_b2_seeds/cells.py`). Output goes to `artifacts/experiment_b2_seeds/` only. Rule 12 proof: the
  full-set subset through the wrapper reproduces the matrix configuration identity, all readout arrays and the logits
  sha256 of three existing cells (ResNet18 INT8 default, MobileNetV3-Large Posit8 default, ResNet18 MXFP8 E4M3
  cum5_act_maxabs; `artifacts/experiment_b2_seeds/reproduction/`).
- **Shared-exponent formats** (MXFP8 E4M3, BFP6, recipe `cum5_act_maxabs` as in the proposal) scale activation blocks
  by their own max-abs, so only their bias correction depends on the calibration subset; their minimal-recipe arm
  `cum1_fused` uses no calibration data and was not run.
- **Statistics.** Seed SD with ddof 1 over the five seeds; intervals are the project's pointwise 95 percent paired
  image bootstrap (10,000 resamples, seed 20260927, `tools.analysis.b2_matrix.interval`); the image-sampling SD is the
  SD of the same bootstrap distribution, averaged over the seeds. Seed share = seed variance / (seed + image variance).
- **Deviation from the protocol's image variance (erratum 1).** The protocol defines the image variance as the mean over
  seeds of the bootstrap variance; the v1 analysis averages the SDs. Recomputed with the protocol's definition
  (`tools/experiment_b2_seeds/checks.py`, output `artifacts/experiment_b2_seeds/checks/review1-checks-v1.json`): over
  all 54 seed cells and 96 contrasts and both readouts the image SD changes by at most 0.018 pp and the seed share by
  at most 0.12, both in MobileNetV3-Large NF4 minimal at chance level (lowest-index top-1 0.0 to 0.4 percent on the
  five seeds); under expected credit at most
  0.008 pp and 0.033. No median or conclusion of this document changes; the v1 summaries are kept as written.
- **Signs in exact arithmetic.** The v1 analysis takes the sign of a float per-seed difference, so an exact zero can
  appear as +/-4e-17. `checks.py` recomputes every per-seed difference as a rational number (credit 1/e for a tie of e
  classes) and applies the protocol's zero rule: 9 differences are exactly zero (1 under expected credit, ResNet18
  INT8 - Posit8 on S3, which v1 counted as a sign change only because its float residue was positive; 8 under lowest
  index), and the sign and five-seed columns of `contrasts.csv` agree with the exact evaluation in all 192
  contrast-readout pairs. Counts unchanged (30 and 34 sign changes; 81/84 and 86/87 five-seed flags).
- **Tests.** `.venv/bin/python -m pytest -q tests/unit/test_experiment_b2_seeds.py` (9 passed, 2 skipped: the
  bias-input test needs the cache entry of the CUDA environment) and
  `CUDA_VISIBLE_DEVICES= .venv-b/bin/python -m pytest -q tests/unit/test_experiment_b2_seeds.py` (11 passed). The
  bias-input test checks, for ResNet18 S0 and MobileNetV3-Large L32_1, that the rows are the subset's first
  min(256, n) rows, that rows taken from the cache are bit-equal to the cache entry and that decoded rows are bit-equal
  to a one-image decode of the hash-checked JPEG.

## 8. Proposal only: a second calibration draw for the accumulator sweep (not run)

The accumulator sweep (`docs/analysis/accumulator-sweep-2026-10-01.md`, lane L8) measured eight ResNet18 B2 cases on the
exact engine with one calibration draw. A second draw changes the activation scales and the corrected biases, so every
case needs a new B2 export, a new adapted exact-engine export and new gates before any width can be swept:

| step per case | cost basis (measured in L2/L8) |
|---|---|
| B2 export (ResNet18) | about 134 MB on disk, 4.5 to 6.5 GB GPU memory (heavy job, one at a time across lanes); exports have died of CUDA OOM under contention |
| adapted bridge-v2 export | tens to a few hundred MB per case (`artifacts/scaled_bridge_v2/exports/`) |
| MB2 gate (32 images, B2 replay vs engine inputs) and 4 policy gates | minutes each; B2 replays needed up to 8 GB and one case needed five OOM retries |
| sweep | the full L8 sweep cost 6.11 GPU-h for 8 cases (location 122 calls 1.09 h, screen 89 calls 5.02 h) at 0.15 to 0.35 s per image |

Options, for the owner:

- **Full repeat, one extra draw (8 cases):** about 6 GPU-h of engine time plus about 2 to 3 GPU-h of exports and gates,
  and roughly 1.5 to 3 GB of new exports, which alone would use most or all of the project's 3 GB disk allowance. Not
  feasible under the current cap without deleting exports after use.
- **Bracket check (recommended if any):** S1 and S2 of this study for INT8, FP8 E4M3 and Posit8 (6 cases), each export
  deleted after its gates and runs, and only the widths W_acc(1.0) - 1, W_acc(1.0), W_acc(1.0) + 1 and W_noevent on the
  1k screen (4 calls per case at about 4 to 6 minutes): about 6 x (20 min export and gates + 20 min sweep) = 4 GPU-h,
  peak disk about 0.5 GB at a time. It answers whether the measured collapse widths move by a bit with the calibration
  draw. Expected but not checked here: the certificate widths do not move (they bound code ranges and K, not scales);
  W_acc(1.0) and W_noevent depend on the activation code distribution and can.
- The ResNet18 seed SDs measured here (section 2) are 0.25 to 0.5 pp at 8 bits, so a width that keeps top-1 within
  0.5 pp of the exact arm under one draw is unlikely to fail by more than a point under another; the bracket check
  would test that, not assume it.

## 9. Run record and protocol erratum

- **Failures (all completed on retry; ledger `artifacts/experiment_b2_seeds/logs/ledger.txt`).** Three jobs died of
  CUDA out-of-memory under GPU contention: the MobileNetV3-Large ladder job at 09:22Z (rerun as a heavy job, done
  11:05Z), the ResNet18 `addendum_bias_only` job at 10:38Z (rerun, done 10:41Z) and the MobileNetV3-Large block-format
  seed job at 11:03Z (`--heavy 8000 --min-free-mib 5000`; one retry with `--heavy 8000 --min-free-mib 7000`, which
  stopped itself at its time limit with cells pending at 12:39Z and was finished by one continuation job at 12:56Z).
  Earlier exit code 3 entries are jobs that stopped themselves before 13 minutes with cells pending and were rerun.
  At 09:14Z the lane's own block worker, which had held the heavy lock for 1 h 49 min while waiting for free memory,
  was stopped and replaced. Every cell that exists was measured once to completion; none was overwritten.
- **Erratum 1** (`public/experiments/configs/breadth-study/b2-seeds-protocol-v1-erratum-1.json`, no new measurement):
  (i) addendum 1's motivation said every 400-image seed was 4 to 9 points above the 2k value for FP6 E3M2, Log6 and
  Posit6 on ResNet18 and MobileNetV2; the seed means above the 2k value are FP6 E3M2 +5.89 / +5.62, Log6 +4.76 /
  +3.83 and Posit6 +0.33 / +9.33 (ResNet18 / MobileNetV2), so the "4 to 9 points" holds for FP6 E3M2 on both
  networks and Log6 on ResNet18, approximately for MobileNetV2 Log6 (+3.83) and Posit6 (+9.33), and not for ResNet18 Posit6 (seeds 58.91 to 60.90 against 59.595) and not for every single seed
  (MobileNetV2 Log6 S2 20.06 against 20.34); (ii) the addendum said 382 protocol cells were measured before it, 386
  records are older than the file; (iii) the 36 addendum cells carry only the v1 protocol hash and are identified by
  arm, subset (R0..R2, B0..B2) and time; (iv) the cell records are in one flat folder, not `cells/<arm>/`; (v) and (vi)
  the image-variance definition and the float sign, section 7.
