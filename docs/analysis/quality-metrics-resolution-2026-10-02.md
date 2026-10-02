# Quality metrics, statistical resolution and ties (lane Q7, 2026-10-02)

Development evidence only: every number re-reads stored records of the ImageNet screen-1k and COCO screen-1k lists.
No model was run, no GPU was used, no held-out image or held-out file was read. Protocol
`public/experiments/configs/breadth-study/quality-metrics-protocol-v1.json` (sha256 `bbbb9a6e5424e5e9…`, written
2026-10-02 11:36 +05 before any metric was computed). Summaries `results/summaries/quality-metrics-v1/` (20 files,
written once, manifest `manifest-A-B-C-D-E.json` with the analysis-code hashes). Revision 1 (after independent review 1;
addendum-1 summaries stamped 12:53:56 +05, document saved 12:57 +05) adds protocol addendum
`public/experiments/configs/breadth-study/quality-metrics-protocol-v1-addendum-1.json` (sha256 `8c8dca5b2494…`) and the
summaries `results/summaries/quality-metrics-v1/addendum-1/` (10 files, manifest `manifest-addendum-1.json`), which give
code to numbers the first version derived by hand and correct three statements. Revision 2 (after independent review 2;
protocol addendum `quality-metrics-protocol-v1-addendum-2.json`, sha256 `e1f67405d3e1…`, frozen 13:11:07 +05; summaries
`results/summaries/quality-metrics-v1/addendum-2/`, 7 files, written 13:14 +05) separates the contrasts that are
identical to their reference (binary discordance 0) from the planning numbers, which changes the H3 recommendation,
and corrects the knife-edge statement; the changes of both revisions are listed in section 9. Figures
`results/figures/quality-metrics-{width-vs-n,coverage,fixed-width}-v1.{png,pdf}`. Every interval is a pointwise
95 percent paired interval with no multiplicity adjustment. Recommendations are labelled as such and are not owner decisions.

## Numbers first

- **Agreement resolves what top-1 cannot, but it measures closeness to FP32, not accuracy.** Over the 742 pairs of
  formats of one network and one recipe (matrix own cells, both top-1 >= 10 %), exact McNemar on lowest-index
  agreement with FP32 separates 61 pairs that top-1 does not, against 2 the other way (604 both, 75 neither). AdaRound
  arms: 421 against 15 of 1,037. Accumulator-sweep policies: 47 against 0 of 268. Example: ResNet18 default, INT8
  against Posit8 es1: top-1 +0.42 pp (not separated), agreement 96.2 % against 93.8 %, -2.4 pp, McNemar p = 0.00054.
  Interval widths of agreement differences are not narrower (median ratio 1.2 to the top-1 width); agreement
  separates more pairs because formats differ more in which images they change than in how many they get right.
  Across the 96 own cells with top-1 >= 10 %, agreement and the top-1 drop correlate at r = 0.989 (Spearman 0.988;
  `addendum-1/partA-agreement-correlation.json`).
- **Resolution follows 1/sqrt(n) in the median.** Median slope of log interval width against log n is -0.49 to -0.50
  for every classifier contrast type. For format pairs, recipe pairs and cells against FP32 or INT8 the 10th-90th
  percentile is -0.51 to -0.48; for accumulator arms against their wide arm it is -0.50 to -0.37 (shallowest -0.19, 45
  contrasts; `addendum-1/partB-slope-quantiles.csv`). The six slopes above -0.38 are arms with 2 to 5 discordant images
  in 1,000 (discordance 0.2 to 0.5 %), where small blocks hold zero or one discordant image and the median width is 0
  or one lattice step; their 1k widths are 0.40 to 0.58 pp. The
  between-block SD over the analytic SE is 0.94 at n = 128, which is the finite-population factor
  sqrt(1 - 128/1000) = 0.934 of disjoint blocks from 1,000 images: a consistency check of the SE, not an independent
  confirmation. **Identical contrasts.** 21 of the 1,028 classifier contrasts have binary discordance 0 at n = 1,000
  (the two arms give the same lowest-index outcome on every image): 15 of the 60 accumulator arms (7 `control` arms
  that equal the wide arm by construction, and 8 widths with no accumulator event on the screen images) and 3 format
  pairs plus 3
  cells against INT8 (`minimal` INT8 against `minimal` `q1_6`, all three networks;
  `addendum-2/partB-identical-contrasts.csv`). They are left out of every planning number below (revision 2;
  `addendum-2/partB-subsets.csv`). Median 95 % width at n = 128 / 256 / 512 / 1,000: format pairs 13.2 / 9.4 / 6.6 /
  4.8 pp; cell against FP32 10.6 / 7.6 / 5.3 / 3.9; accumulator arms that differ from their wide arm (45) 4.7 / 3.3 /
  2.4 / 1.7 (3.8 / 2.9 / 2.0 / 1.4 if the 15 identical arms are counted as width 0). For near-equal contrasts
  (|difference| < 2 pp; a selection made after the data were seen, registered in addendum 1 as descriptive, with 1 pp
  and 3 pp as sensitivity) that differ from their reference, the half-width at 1k is 1.31 pp (135 format pairs, median
  discordance 7.1 %), 1.80 (22 recipe pairs, 10.2 %), 1.26 (23 cells against INT8, 6.1 %), 0.78 (36 accumulator arms
  against wide, 2.8 %; interquartile range of discordance 1.3 to 4.3 %). Counting the identical contrasts, as revision 1
  did, gives 1.29 (7.05 %), 1.24 (6.0 %) and, for accumulator arms, 0.59 (1.6 %), which understates the H3 planning
  discordance. At 1 pp / 3 pp the format-pair half-width is 1.26 / 1.40 (discordance 6.6 / 7.7 %).
- **Projection (not a measurement) to the sealed sets:** near-equal format pairs 0.44 pp half-width on 9k ImageNet,
  recipe pairs 0.60, accumulator arms 0.26 (`addendum-2/partB-subsets.csv`, differing contrasts). Detector (COCO 4k): INT8 / Posit8 / INT8 conformant against FP32 about
  0.16 to 0.19 mAP, INT6 0.43, INT8 conformant minus default 0.08.
- **Minimum detectable difference** (alpha 0.05, power 0.8): at the median near-equal format-pair discordance 7.1 %:
  2.36 pp at 1k, 0.79 pp at 9k (`addendum-2/partB-subsets.csv`); for differing accumulator arms (2.8 %) 1.48 pp at 1k,
  0.49 pp at 9k; at the median over all format pairs (22.6 %,
  `partB-power.csv`) 4.2 pp at 1k and 1.4 pp at 9k. The analytic value is confirmed by simulation with the exact McNemar test (power 0.79 / 0.78 / 0.80 at
  n = 1k / 4k / 9k for discordance 22.6 %).
- **Interval coverage.** At n = 1,000 the project's percentile bootstrap covers 0.94 to 0.96 in all 30 scenarios;
  BCa 0.93 to 0.96, Wald 0.94 to 0.96. Every coverage below 0.90 is at discordance of 2.55 % or less (the five
  2.55 % scenarios and the 2 % null) and n <= 256 (`addendum-1/partB-coverage-extremes.csv`). The percentile and BCa
  minima at n = 128 (0.79 and 0.75) are in scenario
  `pd_q5_asym_q95`, whose discordance is balanced (right-only share 0.499, true difference -0.005 pp): a knife-edge
  case. 34 % of its percentile intervals have an end point exactly at 0, but only those whose lower end is 0 miss
  the negative truth: 18.05 % of data sets (lower end only 14.65 %, both ends 3.4 %); the 15.85 % whose upper end is
  0 cover it. Of the 21 % that miss, 18.05 points are lower-end-at-0 and 2.95 other
  (`addendum-2/partB-knife-edge-endpoints.csv`). Against
  a truth of 0 the same simulated data sets give 0.97 (percentile), 0.90 (BCa), 0.97 (Wald), and a new symmetric null at
  the same discordance gives 0.97 / 0.91 / 0.97 (`addendum-1/partB-coverage-knife-edge.csv`). Each simulated value has
  the Monte Carlo error of 2,000 data sets (SE about 0.005 near 0.95, more at low coverage); enumerating the discordant
  counts exactly (`addendum-2/partB-coverage-exact-cells.csv`, all nine scenarios with discordance <= 5 %, three
  bootstrap seeds) gives, for example, BCa 0.916 instead of 0.898 for the new null at n = 256, and leaves every
  conclusion of this bullet unchanged: in the 32 v1 scenario-n cells with discordance <= 5 %, BCa is worse than the
  percentile interval by >= 0.02 in 12 (simulation 13) and better in 2 (simulation 1: the one-sided case at n = 256 is
  +0.024 exactly, +0.017 simulated); the new null adds 2 worse cells (n = 128, 256). In the one-sided case
  (`pd_q5_asym_q5`, right-only share 0.017, true difference -2.46 pp) BCa is better: n = 128 percentile 0.82, BCa 0.94,
  Wald 0.82; n = 256 0.93 / 0.95 / 0.94; also `pd_q25_asym_q5` n = 128 0.93 / 0.96. BCa is worse than the percentile
  interval (by 0.02 or more) in 14 scenario-n cells, all at discordance <= 2.55 % or n = 128, including the
  null scenarios at 1 and 2 % discordance (BCa 0.90 to 0.94 at n = 256 to 1,000; `addendum-1/partB-coverage-bca-vs-percentile.csv`).
- **Ties.** Classifier expected-credit minus lowest-index top-1 over matrix cells >= 10 %: median +0.11 pp, 5th/95th
  percentile -1.07 / +0.81, largest |difference| 1.81 pp (11 of 114 cells above 1 pp); the SD of top-1 under one random
  tie break is median 0.46 pp, at most 0.96. Detector: over 200 random cross-image orders the AP SD is 0.02 to 0.08 points;
  under random image orders the fixed rule sits at the 1st percentile for INT6 `default` (fixed 30.94, expected 31.05)
  and at the 0th for INT6 `conformant` (31.05 against 31.18); under uniform random orders of equal scores at the 6th and
  8th (expected 31.02 and 31.13); the two schemes' means differ by up to 0.056 points. No verdict changes: INT8 `default` against FP32 is -1.03 under the expected AP
  (fixed rule -1.02), still just outside one point.
- **Detector error decomposition (INT6 `default`, 8.10 points below FP32; approximation).** About 7.3 points are lost
  before the head's projection, expectation and score stores (`default_head_logits`, where those stores are wide,
  gives 31.77; its class logits still come out of INT6 layers, so its scores take only 34 distinct values).
  Re-quantising those scores onto the INT6 score grid costs 0.41 to 0.45 points, of which dropping the low-score tail
  (172,218 to about 62,700 detections) is 0.25 to 0.33; the remaining 0.4 of the 0.83-point head step is DFL/projection
  codes and NMS on coarse scores. These steps are of the size of the fixed-rule tie deficit (0.11 to 0.15 points for
  these two arms), so the split is good to about +-0.15. FP32 true positives kept by INT6:
  87.3 % at IoU 0.5 and 80.6 % at IoU 0.75 (all scores); at score >= 0.25, 74.8 % (small objects 40.2 %, large 90.1 %).
  AP-small/medium/large: -6.2 [-7.5, -4.8], -9.4 [-11.4, -8.2], -8.8 [-11.0, -6.9]; INT8 `default` -0.81
  [-1.70, -0.15], -1.06 [-2.00, -0.43], -0.50 [-1.95, +0.21].
- **Format order at a fixed register width (ResNet18, saturating `sat.w<W>`).** Within 1 pp of their own exact arm:
  W = 16 INT6 only; W = 17 INT6, FP6 E2M3; W = 18 adds INT8 with signed activations; W = 19 to 22 INT8, INT8 signed,
  INT6, FP6; W = 23 to 31 adds FP7 E3M3; W = 32 adds Posit8 es1 (68.8 %). FP8 E4M3 needs 39 bits and FP8 E5M2 more
  than 63, so both are collapsed throughout 16-32. The leader changes from INT6 (W = 16) to FP6 E2M3 (17) to the
  INT8 recipes (18 and up).

## 1. Part A: classifier agreement with FP32

Units: 177 matrix cells (incl. 24 intrinsic-arm cells and 3 FP32 baselines), 144 AdaRound evaluations, 89 sweep policy
rows at 1,000 images (`partA-units.csv`, one row each). FP32 has no tied top-1 on any image of any network, so the FP32
class c0 is unique. AdaRound FP32 records equal the matrix FP32 top-5 on 1,000 of 1,000 images (all three networks).

What is computable (protocol part A, `tie_set`):

| record | lowest-index agreement | expected agreement 1[c0 in T]/k |
|---|---|---|
| matrix readout | exact (`argmax_lowest`) | exact when k <= 5; when k > 5, exact unless c0 is above the fifth-lowest tied class; bounds otherwise |
| AdaRound eval | exact when k <= 5 (first k entries of `top5_topk`); undetermined otherwise | as matrix, except no index order (c0 not listed and k > 5 = undetermined) |
| sweep prediction | exact (full tie list stored) | exact |

68 units with top-1 >= 10 % have undetermined images (at most 300; largest gap between bounds 2.4 pp); 47 AdaRound
arms have undetermined lowest-index agreement. These units are reported with bounds and are left out of
bootstrap comparisons of expected agreement (protocol); McNemar on lowest-index agreement is exact for every
matrix and sweep unit.

Selected cells (default recipe; top-1 expected credit, agreement lowest-index, flips correct-to-wrong / wrong-to-correct,
mean top-5 overlap with FP32):

| network | format | top-1 | agreement | flips c->w / w->c | top-5 overlap | McNemar p vs FP32 |
|---|---|---|---|---|---|---|
| ResNet18 | INT8 | 69.2 | 96.2 | 18 / 3 | 95.5 | 0.0015 |
| ResNet18 | Posit8 es1 | 69.6 | 93.8 | 20 / 10 | 92.6 | 0.099 |
| ResNet18 | FP8 E4M3 | 67.7 | 86.9 | 49 / 27 | 85.3 | 0.015 |
| ResNet18 | INT6 | 65.9 | 83.3 | 71 / 22 | 78.7 | < 1e-4 |
| MobileNetV2 | INT8 | 72.3 | 97.4 | 8 / 7 | 95.4 | 1.0 |
| MobileNetV2 | Posit8 es1 | 72.4 | 91.5 | 22 / 20 | 88.3 | 0.88 |
| MobileNetV2 | INT6 | 70.1 | 84.7 | 54 / 36 | 78.3 | 0.073 |
| MobileNetV3-L | INT8 | 71.6 | 82.6 | 66 / 31 | 74.2 | 0.00049 |
| MobileNetV3-L | Posit8 es1 | 73.4 | 89.8 | 42 / 20 | 82.9 | 0.0071 |
| MobileNetV3-L | INT6 | 21.0 | 21.9 | 551 / 10 | 19.6 | < 1e-4 |

MobileNetV2 INT8 and Posit8 are within 0.3 pp of FP32 in top-1 (72.30 and 72.39 against 72.10), yet Posit8 changes
the class of 3.3 times as many images (8.5 % against 2.6 %). Pairs resolved by agreement only include, on ResNet18 default, INT8 against
Posit8 (top-1 +0.42, agreement -2.4, p = 0.00054) and INT8 against Log8 (-1.13, -8.6, p < 1e-4); on MobileNetV2
default, INT8 against Posit8 (+0.09, -5.9); on MobileNetV3-Large default, FP8 E4M3 against INT8 (-0.06, -3.3,
p = 0.0095). (Revision 1: the first version attributed the last pair to MobileNetV2, where FP8 E4M3 against INT8 is
+4.59 pp in top-1 and +17.6 pp in agreement, both separated.) In 7 AdaRound
pairs both metrics separate with opposite signs (an arm closer to FP32 but less accurate); in matrix and sweep pairs
never. Error overlap (Jaccard and phi of the error sets) is in `partA-pairs.csv` for every pair.

Resolution counts (`partA-resolution.json`):

| selection | pairs (McNemar) | agreement only | top-1 only | both | neither | pairs (bootstrap, expected) | agreement only | top-1 only |
|---|---|---|---|---|---|---|---|---|
| matrix own cells | 742 | 61 | 2 | 604 | 75 | 210 | 34 | 1 |
| matrix incl. intrinsic arms | 1,055 | 94 | 6 | 834 | 121 | 334 | 58 | 4 |
| AdaRound arms | 1,037 | 421 | 15 | 463 | 138 | 1,411 | 473 | 43 |
| sweep policies | 268 | 47 | 0 | 81 | 140 | 268 | 43 | 13 |

## 2. Part B: resolution, coverage, power

Subsample widths (`partB-subsample.csv`, `-summary.csv`; 50 random permutations, disjoint blocks; Figure
`quality-metrics-width-vs-n-v1`): see Numbers first. The slope is -0.49 to -0.50 in the median for every contrast type, so
1k half-widths can be projected by sqrt(1000/N) with no correction in the median; for accumulator arms with fewer
than about 5 discordant images per 1,000 the 1k width is lattice-limited and the projection is rough (it is already
below 0.6 pp there). The v1 protocol seeds the permutations `default_rng(20261002 + r)`; the code used
`default_rng([20261002, 10, r])` (disclosed in addendum 1; no effect on any summary statistic beyond Monte Carlo noise),
and the between-block SD is reported for n <= 256 only.

Detector (`partB-detector-resolution.csv`; 3 permutations, 500-draw bootstraps per block, stored 2,000-draw vectors at 1k):

| contrast | width n=128 | 256 | 512 | 1000 | SE 1000 | slope | projected half-width 4k | MDD 4k | images for half-width 0.25 |
|---|---|---|---|---|---|---|---|---|---|
| INT8 default - FP32 | 2.28 | 1.70 | 1.17 | 0.74 | 0.19 | -0.54 | 0.19 | 0.27 | 2,217 |
| INT8 conformant - FP32 | 2.33 | 1.68 | 1.05 | 0.68 | 0.18 | -0.60 | 0.17 | 0.25 | 1,868 |
| Posit8 default - FP32 | 2.24 | 1.77 | 1.06 | 0.65 | 0.17 | -0.62 | 0.16 | 0.23 | 1,676 |
| FP8 E4M3 default - FP32 | 3.83 | 2.78 | 1.73 | 1.30 | 0.34 | -0.54 | 0.32 | 0.48 | 6,715 |
| INT6 default - FP32 | 4.63 | 3.60 | 2.63 | 1.71 | 0.44 | -0.48 | 0.43 | 0.61 | 11,645 |
| INT8 conformant - default | 0.80 | 0.64 | 0.62 | 0.34 | 0.09 | -0.38 | 0.08 | 0.12 | 456 |

The detector slopes are noisier (3 permutations, one block at n = 512); the stored 1k bootstrap is the basis of the
projection. Interval width grows with the size of the effect, as for classifiers.

Coverage (`partB-coverage.csv`, Figure `quality-metrics-coverage-v1`, which shows four balanced-asymmetry scenarios only,
not the one-sided and knife-edge cases discussed below; 25 scenarios at the 5/25/50/75/95th percentiles of
observed discordance (2.6 % to 55 %) and of its asymmetry, plus 5 null scenarios; 2,000 data sets x 2,000 resamples):
worst coverage by method at n = 128 / 256 / 512 / 1,000: percentile 0.79 / 0.90 / 0.93 / 0.94; BCa 0.75 / 0.88 / 0.91 /
0.93; Wald 0.82 / 0.94 / 0.92 / 0.94 (Monte Carlo SE 0.005; scenario of each minimum in
`addendum-1/partB-coverage-extremes.csv`). Medians are 0.94 to 0.95 throughout. The failures are all low-discordance
scenarios at small n, where few discordant images make the bootstrap distribution lattice-like. The percentile and BCa
minima at n = 128 and 256 are the knife-edge scenario `pd_q5_asym_q95` (true difference -0.005 pp; against a truth of
0 the same data sets give percentile 0.97 / 0.96, BCa 0.90 / 0.92); the Wald minimum at n = 128 and the case where BCa
helps (0.94 against 0.82) is the one-sided `pd_q5_asym_q5`. BCa is worse than percentile at low-discordance nulls. For
the nine scenarios with discordance <= 5 % (and the addendum-1 null) the coverage was also computed exactly over the
discordant counts, with three bootstrap seeds per count (`addendum-2/partB-coverage-exact-cells.csv`): BCa at the
symmetric nulls of 1 to 2.55 % discordance and n = 256 to 1,000 covers 0.906 to 0.947 (percentile 0.953 to 0.977); the
simulated values differ from the exact ones by up to about 0.02 in single cells, no conclusion changes.

Power: images for power 0.8, alpha 0.05, normal approximation, n = z^2 p_d / delta^2. The protocol table
`partB-power.csv` uses the 25/50/75th percentiles of discordance over all contrasts of a kind (identical contrasts
included). The table below (revision 2, `addendum-2/partB-power-grid.csv` and `addendum-2/partB-sufficiency.csv`) uses
the near-equal subset (|difference| < 2 pp, chosen after the data were seen) without the identical contrasts, at its
median and 75th-percentile discordance, the share of its contrasts whose own screen discordance needs at most 9,000
images, and the median over all differing contrasts of the kind. Near-equal pairs are the ones for which equivalence
and non-inferiority claims are made, but the selection is post hoc: for format pairs the all-contrast median (22.6 %)
needs 3.2 times more images (17,739 for a 1 pp difference instead of 5,573); for accumulator arms 1.3 times, for
recipe pairs 1.2 times. `difference` and `beyond_margin` use the same formula (z_0.975 + z_0.8, effect = margin).

| hypothesis type | subset, median discordance | margin 1.0 pp | 0.5 pp | 0.25 pp | at 75th pct: discordance, 1.0 / 0.5 pp | share <= 9k at 1.0 / 0.5 pp | all differing: median discordance, 1.0 pp |
|---|---|---|---|---|---|---|---|
| difference of 1 margin (two-sided) | 135 format pairs, 7.1 % | 5,573 | 22,291 | 89,164 | 10.3 %, 8,085 / 32,338 | 84 % / 3 % | 22.6 %, 17,739 |
| equivalence within +-margin (TOST, true 0) | 135 format pairs, 7.1 % | 6,081 | 24,322 | 97,286 | 10.3 %, 8,821 / 35,284 | 76 % / 3 % | 22.6 %, 19,355 |
| equivalence within +-margin | 22 recipe pairs, 10.2 % | 8,736 | 34,941 | 139,762 | 12.45 %, 10,662 / 42,648 | 50 % / 18 % | 12.4 %, 10,620 |
| interval beyond the margin (true 2 x margin) | 22 recipe pairs, 10.2 % | 8,006 | 32,024 | 128,094 | 12.45 %, 9,772 / 39,088 | 55 % / 18 % | 12.4 %, 9,733 |
| non-inferiority (one-sided, true 0) | 36 accumulator arms vs wide, 2.8 % | 1,732 | 6,925 | 27,698 | 4.3 %, 2,674 / 10,696 | 100 % / 61 % | 3.7 %, 2,288 |
| equivalence within +-margin | 36 accumulator arms vs wide, 2.8 % | 2,398 | 9,592 | 38,367 | 4.3 %, 3,704 / 14,816 | 100 % / 50 % | 3.7 %, 3,169 |

Revision 1 used the near-equal subsets with the identical contrasts counted (`addendum-1/partB-power-near-equal.csv`):
format pairs 7.05 % (5,534 / 6,038 at 1 pp, almost unchanged) and accumulator arms 1.6 %, which gave non-inferiority
3,957 and equivalence 5,481 at 0.5 pp. Those two H3 numbers were too low by a factor of 1.75: 15 of the 51 arms in
that subset cannot differ from the wide arm, so they say nothing about the discordance of an arm under test. The
per-contrast shares use each contrast's own 1k discordance, which for the near-equal accumulator arms rests on 2 to
72 discordant images and is itself uncertain.

The exact-McNemar simulation confirms the normal approximation (`partB-power-check.csv`: power 0.79 / 0.78 / 0.80 at
the MDD for n = 1k / 4k / 9k, discordance 22.6 %).

Compute ledger (`partB-ledger.json`; projections from shared-GPU development rates, not exclusive timings):

| work | size | hours | resource | rate source |
|---|---|---|---|---|
| simulator classifier cells, 9k | 12 / 36 configurations | 0.3-1.8 / 0.8-5.4 | GPU slot | matrix cell 6-55 s per 1k + 30-40 s per process |
| exact engine ResNet18, 9k | 24 / 56 / 73 runs | 9-21 / 21-49 / 27-64 | GPU | 0.15-0.35 s per image (L8 sweep) |
| exact engine MobileNet, 9k | 10 runs | about 9 | GPU | 6 min per 1k, estimated from 32-image runs |
| detector arms, COCO 4k | 6 / 18 configurations | 0.4-1.6 / 1.1-4.8 | GPU | 0.9-4 min per 1k |
| detector COCOeval bootstrap, 4k, 2,000 draws | 6 / 18 configurations | 3.2-3.6 / 9.6-10.8 | CPU-hours | 8-9 CPU-min per 1k |

## 3. Part C: ties

Classifiers (`partC-classifier-ties.csv`, `-by-family.csv`): expected minus lowest-index, cells >= 10 %:

| source | units | median | 5th pct | 95th pct | max abs | random-tie SD median / max |
|---|---|---|---|---|---|---|
| matrix | 114 | +0.11 | -1.07 | +0.81 | 1.81 | 0.46 / 0.96 |
| AdaRound | 111 | 0.00 | -0.63 | +0.88 | 1.58 | 0.27 / 1.10 |
| sweep | 68 | +0.25 | -0.51 | +0.91 | 1.31 | 0.48 / 0.71 |

By family (matrix): 8-bit integers +0.10 to +0.62 (lowest index below expected credit in all six cells), Posit8
+0.49 to +0.56; 5-bit integers -1.6 to +0.5; the stored `topk` order differs from the expected credit by up to 1.74 pp.

Detector (`partC-detector-tie-orders.csv`): 200 random image orders (pycocotools-equivalent orders) and 200 uniform orders
of equal scores at fixed image-local matching. The two schemes give means that differ by up to 0.056 points (INT6
`conformant`; then 0.047 BFP6 `default_fp32_box_logits`, 0.045 INT6 `default_head_logits`;
`addendum-1/partC-detector-tie-scheme-gap.csv`); the image-order mean is the higher one for every 6-bit
configuration. The fixed rule's percentile depends on the scheme: INT6 `default` 1 % (image orders) against 6 %
(uniform), INT6 `conformant` 0 % against 8 %. The table below uses the image-order scheme.

| format | recipe | fixed rule | expected (image orders) | SD | range of 200 | fixed-rule percentile |
|---|---|---|---|---|---|---|
| FP32 | | 39.05 | 39.05 | 0.000 | - | - |
| INT8 | default | 38.02 | 38.02 | 0.024 | 37.97-38.08 | 53 |
| INT8 | conformant | 38.35 | 38.34 | 0.029 | 38.27-38.42 | 74 |
| INT6 | default | 30.94 | 31.05 | 0.047 | 30.92-31.18 | 1 |
| INT6 | conformant | 31.05 | 31.18 | 0.054 | 31.06-31.36 | 0 |
| FP6 E2M3 | default / conformant | 32.13 / 32.92 | 32.20 / 32.99 | 0.06 / 0.07 | | 10 / 15 |
| Posit8 | default / conformant | 38.11 / 38.42 | 38.13 / 38.43 | 0.02 / 0.03 | | 21 / 34 |
| BFP6 | default / default_fp32_box_logits | 33.98 / 34.22 | 34.03 / 34.25 | 0.08 / 0.08 | | 23 / 37 |
| MXFP8 | default / default_fp32_box_logits | 35.36 / 36.24 | 35.41 / 36.30 | 0.04 / 0.04 | | 8 / 5 |

(`default_fp32_box_logits` is the nearest available recipe to `conformant` for the block formats.)

Verdict re-check under the expected AP: INT8 `default` - FP32 = -1.03 (exit target missed by 0.03, as under the fixed
rule); INT8 `conformant` -0.71 (inside); INT6 `conformant` - `default` = +0.13 (fixed +0.11; still within the tie
sensitivity); the sentinel order is unchanged. The within-image NMS-order part (GPU) was not run.

## 4. Part D: detector agreement and error decomposition

`partD-detector-agreement.csv` (21 configurations against FP32; matching by COCOeval, area all, maxDets 100):

| format, recipe | mAP - FP32 | small / medium / large | FP32 TP kept IoU .5 / .75 (all scores) | kept at score >= .25, IoU .5 | new FP (>= .25) | box IoU to FP32 (median) | score Spearman |
|---|---|---|---|---|---|---|---|
| INT8 default | -1.02 | -0.81 / -1.06 / -0.50 | 98.0 / 93.7 | 94.5 | 75 | 0.975 | 0.95 |
| INT8 conformant | -0.69 | -0.37 / -0.76 / -0.56 | 96.5 / 92.5 | 94.5 | 75 | 0.976 | 0.95 |
| Posit8 default | -0.93 | -0.94 / -1.72 / -0.83 | 98.5 / 93.9 | 96.6 | 147 | 0.963 | 0.97 |
| MXFP8 default | -3.69 | -3.22 / -3.68 / -4.56 | 96.4 / 90.0 | 92.6 | 266 | 0.930 | 0.89 |
| FP8 E4M3 default | -4.63 | -3.43 / -5.14 / -5.97 | 96.3 / 88.5 | 91.6 | 478 | 0.927 | 0.86 |
| LOG8 default | -5.38 | -3.29 / -4.82 / -7.72 | 95.8 / 86.9 | 89.4 | 405 | 0.914 | 0.85 |
| FP6 E2M3 default | -6.92 | -5.19 / -7.48 / -8.76 | 88.1 / 80.8 | 86.8 | 498 | 0.915 | 0.79 |
| INT6 default | -8.10 | -6.25 / -9.37 / -8.82 | 87.3 / 80.6 | 74.8 | 269 | 0.928 | 0.75 |

FP32 has 1,550 false positives at score >= 0.25 (5,447 detections). The 8-bit floats and LOG8 lose AP mainly through
box quality (median IoU to the FP32 box 0.91 to 0.93, mean change of IoU to the ground truth -0.02 to -0.03) and new
false positives at score >= 0.25 (FP8 E4M3 and FP7 E3M3 474 to 553, LOG8 337 / 405 for `conformant` / `default`,
MXFP8 265 / 266; `addendum-1/partD-new-false-positives.csv`), not through lost objects (96 % kept at IoU 0.5). INT6 is the other way round: it
keeps only 74.8 % of the confident FP32 true positives (small objects 40 %) and adds few new false positives.

Score step (`partD-score-step.csv`; CPU approximation, NMS ran on the `default_head_logits` scores): INT6 FP32 39.05 ->
`default_head_logits` (projection, expectation and score stores wide; scores still 34 distinct values because the class
logits are INT6 outputs) 31.77 -> scores re-quantised to one of the three inferred per-level grids (units 0.0123 / 0.0129 / 0.0148) 31.32 to
31.37 -> `default` 30.94. Dropping only the detections whose score is below half a step: 31.45 to 31.52. INT8: 38.16
(`default_head_logits`) -> 37.83 to 38.12 (re-quantised) -> 38.02 (`default`); threshold only 38.14 to 38.15. So at INT6 the score
quantisation step accounts for about 0.4 of the 8.1-point loss (roughly 5 %), and most of it is the dropped low-score tail.
All values here use the fixed tie rule; under the image-order expected AP the two measured arms are 31.93
(`default_head_logits`) and 31.05 (`default`), a head step of 0.87 instead of 0.83. The re-quantised synthetic sets were
not evaluated under random orders, so the 0.4-point score step carries a tie uncertainty of about 0.1 to 0.15 points.

## 5. Part E: format ranking at a fixed accumulator width

`partE-grid.csv`, `partE-orders.csv`, Figure `quality-metrics-fixed-width-v1`. One accumulator type: the uniform signed
saturating register `sat.w<W>` of lane L8 (widths in units of each case's product grid). Cells are measured at 1k where
measured, equal to the exact arm at or above the case's event-free width, and "collapsed, not measured" below the
narrowest measured width (monotone extrapolation, labelled).

| W | order (top-1 %) | within 1 pp of own exact arm | separated pairs / measured pairs |
|---|---|---|---|
| 16 | INT6 66.4 > FP6 65.2 > INT8s 2.4 > INT8 0.2 > others collapsed | INT6 | 5 / 6 |
| 17 | FP6 66.5 > INT6 65.9 > INT8s 48.7 > INT8 24.0 | INT6, FP6 | 5 / 6 |
| 18 | INT8s 68.2 > INT8 67.6 > FP6 66.5 > INT6 65.9 | INT8s, INT6, FP6 | 2 / 6 |
| 19-22 | INT8 ~69.0 / INT8s ~68.9 > FP6 66.5 > INT6 65.9 > FP7 (0.1 to 61.8) | INT8, INT8s, INT6, FP6 | 8 / 10 |
| 23-31 | INT8s 69.1 > INT8 69.0 > FP7 67.0-67.4 > FP6 66.5 > INT6 65.9 > Posit8 (0.1 to 51.4) | + FP7 | 6-12 / 10-15 |
| 32 | INT8s 69.1 > INT8 69.0 > Posit8 68.8 > FP7 67.3 > FP6 66.5 > INT6 65.9 | + Posit8 | 10 / 15 |

INT8s = INT8 with signed activations. Kendall tau between adjacent widths drops to 0.64 at W = 18 and 0.78 at W = 32
(entry of a new format); elsewhere 0.84 to 1.0. FP8 E4M3 (needs 39) and FP8 E5M2 (> 63) are collapsed throughout.
Float accumulators are a separate type (`partE-float-accumulators.csv`): fp16-rule and f21 arms are within 1 pp of exact for
all eight cases; one interval excludes 0, upward (Posit8 fp16.x-10 +0.63 [+0.03, +1.25]).

## 6. Recommendations for the confirmation protocol (recommendations, not decisions)

1. **Primary readouts.** Classifiers: keep expected-credit top-1 as primary and lowest-index top-1 as secondary
   (the difference is at most 1.8 pp per cell, median 0.1, and the random-tie SD is up to 1 pp). Add lowest-index
   agreement with FP32 and exact McNemar as a declared secondary metric for "closer to FP32" claims; never as a
   substitute for an accuracy claim.
2. **Detector tie rule.** Report the expected AP over random evaluation-image orders (the image-order scheme: 200
   permutations of the image list, which is what pycocotools' own tie handling depends on; CPU minutes) as primary and
   the fixed rule as secondary, because under that scheme the fixed rule sits at the 0-1st percentile for INT6
   (0.11 to 0.13 points low) while the order SD is only 0.02 to 0.08. Report the uniform-tie scheme as a sensitivity:
   its mean differs from the image-order mean by up to 0.056 points and puts the INT6 fixed rule at the 6-8th
   percentile (0.08 points low), so the size of the fixed-rule deficit depends on the scheme, its sign does not. Alternative if the owner prefers a deterministic
   number: keep the fixed rule primary and report the expected AP and its SD beside every 6-bit detector number.
3. **Intervals.** Keep the project's percentile paired bootstrap; do not switch to BCa. Reason: at n >= 1,000 all three
   intervals cover 0.93 to 0.96 in all 30 scenarios, so the choice matters only at small n and low discordance, and
   there BCa is worse in more cases than it is better: it under-covers at symmetric low-discordance nulls (exactly
   computed 0.906 to 0.947 at 1 to 2.55 % discordance, n = 256 to 1,000, where the percentile interval covers 0.953 to
   0.977) and is better only
   in the one-sided low-discordance case at n <= 256 (0.94 against 0.82 at n = 128). The percentile minimum of 0.79 is a
   knife-edge artefact of a near-null truth of -0.005 pp (0.97 against a truth of 0; 18 of its 21 missing points are
   intervals whose lower end is exactly 0). Do not use n <= 256 panels for
   contrasts with discordance below about 5 %.
4. **Margins and image counts.** These rest on the near-equal discordances of contrasts that differ from their
   reference (post hoc selection, addenda 1 and 2; `addendum-2/partB-power-grid.csv`, `partB-sufficiency.csv`); the
   all-contrast medians give 3.2 times more images for format pairs, 1.3 for accumulator arms and 1.2 for recipe pairs.
   Classifier H1/H2, format pairs (near-equal median discordance 7.1 %): delta = 1.0 pp is resolvable on the 9k
   remainder (difference 5,573, equivalence 6,081 images at the median; 84 % / 76 % of the 135 near-equal format pairs
   need at most 9,000; projected half-width 0.44 pp); 0.5 pp is not (22,000 or more). Recipe pairs (near-equal median
   10.2 %): delta = 1.0 pp is marginal on 9k: equivalence needs 8,736 images at the median, 10,662 at the 75th
   percentile, and more than 9,000 for half of the 22 near-equal recipe pairs; a difference beyond the margin needs
   8,006 at the median (55 % within 9,000). State recipe equivalence claims only with a wider margin or as
   differences beyond the margin. For a format pair at the all-contrast median discordance (22.6 %), a 1 pp
   difference needs about 17,700 images and is not resolvable on 9k; such pairs are usually not near-equal, and the
   protocol should state the discordance it assumes per hypothesis and check it on the screen before freezing n.
   H3 (accumulator arm against its wide arm, the 36 near-equal arms that differ from wide, median discordance 2.8 %,
   75th percentile 4.3 %): delta = 1.0 pp non-inferiority and equivalence are resolvable on 9k for every one of these
   arms (at most 3,704 images at the 75th percentile). Delta = 0.5 pp is not resolvable for all arms: non-inferiority
   needs 6,925 images at the median and 10,696 at the 75th percentile (61 % of the arms within 9,000), equivalence
   9,592 at the median (50 % within 9,000). Recommendation: H3 with delta = 1.0 pp on 9k; a 0.5 pp non-inferiority
   claim only for arms whose screen discordance is at most about 3.6 % (the value at which
   (z_0.95 + z_0.8)^2 p_d / 0.005^2 = 9,000), declared per arm before the run. Arms identical to the wide arm on the
   screen (controls, event-free widths) are not planning cases: controls are identical by construction; for event-free
   widths, count accumulator events on the 9k set. The 1k screen resolves neither margin (half-width 0.78 pp, MDD
   1.48 pp). Revision 1 gave 3,957 / 5,481 images and "9k suffices" for delta = 0.5 pp; that rested on a subset in
   which 15 of 51 arms were identical to wide.
   Detector: delta = 0.5 mAP is resolvable on the 4k complement for 8-bit configurations near FP32 (projected
   half-width 0.16-0.19, MDD 0.23-0.27) and for recipe differences (0.08), not for 6-bit formats (0.43, MDD 0.61);
   a 6-bit detector claim needs delta = 1.0 mAP or about 12,000 images.
5. **Compute.** The ResNet18 exact-engine grid dominates (24-73 runs on 9k: 9-64 GPU-hours at shared-GPU rates); the
   simulator and the detector together stay under about 10 GPU-hours plus about 11 CPU-hours of COCOeval bootstrap.
6. **H3 as written** compares widths across accumulator types; Part E shows the order of formats depends on W inside
   one type (leader changes at W = 16, 17, 18). State H3 per accumulator type.

## 7. Limits

- One calibration draw, one input size, the 1k screens; dev128 lies inside the 1k COCO screen.
- Planning discordances come from the screen. Per-contrast sufficiency shares use each contrast's own 1k discordance,
  which for accumulator arms rests on 2 to 72 discordant images; arms that were identical to wide on the screen can
  differ on new images (event-free widths), and were left out of the planning subsets rather than modelled.
- Agreement metrics need the FP32 class to be unique (true here) and are exact only where the tie set is known.
- Coverage simulates i.i.d. paired binary outcomes; class stratification of the 9k list (9 per class) is not modelled,
  and the expected-credit (fractional) outcome was not simulated.
- The score-step decomposition is a CPU approximation on stored detections (NMS ran on wide scores; per-level score
  scales were inferred from the stored values; DFL and projection effects are the residual).
- Part E below the measured brackets is extrapolated (collapsed), not measured. The ledger uses shared-GPU rates.
- The optional GPU within-image tie-order run was not done.
- The near-equal subset (|difference| < 2 pp) behind the margin recommendations was chosen after the data were seen;
  addendum 1 registers it as descriptive and adds 1 pp and 3 pp thresholds as sensitivity; without identical contrasts
  (addendum 2) the format-pair half-width is 1.26 / 1.31 / 1.40 pp at 1 / 2 / 3 pp, and the accumulator-arm subset is
  the same 36 arms at 2 and 3 pp (34 at 1 pp, discordance 2.45 %).
- Protocol deviations of v1 (disclosed in addendum 1, no effect on conclusions): permutation seeds
  `default_rng([20261002, 10, r])` instead of `default_rng(20261002 + r)`; between-block SD reported for n <= 256 only;
  `manifest-A-B-C-D-E.json` lists the pre-edit hash of `figures.py` (aee9194b…; edited afterwards for tick labels only,
  now 563af530…, recorded in `addendum-1/manifest-addendum-1.json`).

## 8. Reproduce

```
.venv/bin/python -m tools.run.quality_metrics det-jobs --workers 4      # CPU, ~40 min; artifacts/quality_metrics_v1/work/det/
.venv/bin/python -m tools.run.quality_metrics analyse --out <new folder> # CPU, ~6 min; refuses to overwrite
.venv/bin/python -m tools.analysis.quality_metrics_v1.figures <folder>   # results/figures/quality-metrics-*-v1
.venv/bin/python -m tools.run.quality_metrics addendum1 --out <new folder> # CPU, ~1 s; reads the v1 summaries (prints nothing)
.venv/bin/python -m tools.run.quality_metrics addendum2 --out <new folder> # CPU, ~40 s; reads the v1 summaries
.venv/bin/python -m pytest tests/unit/test_quality_metrics.py tests/unit/test_quality_metrics_addendum1.py \
    tests/unit/test_quality_metrics_addendum2.py -q                          # 15 passed
```
Code: `tools/analysis/quality_metrics_v1/` (units, agreement, resolution, ties, detector, fixed_width, report, figures,
addendum1, addendum2), `tools/run/quality_metrics.py`. No existing tracked file was edited.

## 9. Revision 1 change log (2026-10-02, after independent review 1)

Review: `artifacts/agent_orchestration/handoffs/Q7-metrics-resolution-review.md`, review 1 (approve with fixes; every v1
summary number it recomputed reproduced). No v1 summary changed; new numbers are in `addendum-1/`.

- Coverage (blocking 1): the statement that BCa is never better than the percentile interval was false (one-sided
  low-discordance case, n = 128: 0.94 against 0.82); the 0.79 / 0.75 minima were attributed to one-sided discordance but
  are in a balanced near-null scenario and are a knife-edge effect. Rewritten in Numbers first, section 2 and
  recommendation 3, with the knife-edge re-count and a new symmetric null at 2.55 % (addendum 1). The BCa minimum at
  n = 512 is 0.91, not 0.92. Recommendation 3 is unchanged, its reason is rewritten.
- Pair label (blocking 2): FP8 E4M3 against INT8 (-0.06, -3.3, p = 0.0095) is MobileNetV3-Large, not MobileNetV2.
- Slopes (blocking 3): the 10th-90th percentile range -0.51 to -0.48 holds for four of the five contrast kinds; for
  accumulator arms it is -0.50 to -0.37 (shallowest -0.19).
- Near-equal numbers, power table and r = 0.99 now come from code (`addendum1.py`) and versioned summaries; the
  post hoc selection is disclosed; the power table shows the all-contrast discordance beside it (3.2 times more images
  for format pairs). The "beyond the margin" row is `beyond_margin` in `resolution.images_needed`, now written.
- Tie schemes: the two schemes differ by up to 0.056 points, not "within 0.04"; the uniform scheme puts the INT6 fixed
  rule at the 6th / 8th percentile; recommendation 2 names the image-order scheme.
- Minor: ResNet18 INT8 McNemar p 0.0015 (was printed 0.0014); MobileNetV2 Posit8 is 0.29 pp above FP32 (was "within
  0.2 pp"); new false positives range now includes LOG8 `conformant` (337) and MXFP8 (265 / 266); "wide head stores"
  renamed to `default_head_logits` with its 34 distinct score values and the tie uncertainty of the score-step split;
  block-format "nearest" recipe named `default_fp32_box_logits`; between-block SD / SE = 0.94 identified as the
  finite-population factor; protocol deviations and the figures.py hash disclosed (section 7).

## 10. Revision 2 change log (2026-10-02, after independent review 2)

Review: same file, review 2 (approve with fixes; every revision-1 number it recomputed reproduced). Protocol addendum 2
frozen 13:11:07 +05 (clock reading) before its numbers were computed; summaries `addendum-2/` written 13:14 +05. No v1 or
addendum-1 summary changed.

- H3 planning numbers (blocking 1): the near-equal accumulator subset of revision 1 (51 arms) held 15 arms with
  binary discordance 0 against wide (7 controls, 8 widths without accumulator events on the screen). Without them
  (36 arms) the median discordance is 2.8 % (not 1.6 %), the 1k half-width 0.78 pp (not 0.59), non-inferiority at
  0.5 pp needs 6,925 images (not 3,957; 10,696 at the 75th percentile; 61 % of the arms within 9,000) and equivalence
  9,592 (not 5,481). Recommendation 4 now proposes delta = 1.0 pp for H3 on 9k and 0.5 pp non-inferiority only for arms
  with screen discordance <= 3.6 %. The accumulator median widths and the all-contrast median (3.7 %, not 1.95 %) are
  given without the identical arms; the 3 identical format pairs and 3 identical cells against INT8 are also left out
  (format-pair numbers change by <= 0.02 pp).
- Knife edge (blocking 2): 34 % of the percentile intervals have an end point at 0, but only the 18.05 % with the
  lower end at 0 miss the negative truth; 15.85 % have the upper end at 0 and cover it (`partB-knife-edge-endpoints.csv`).
- Recipe-pair equivalence (must fix): revision 1 wrote "equivalence needs about 6,000 images" for format and recipe
  pairs; 6,038 was the format-pair value. Recipe pairs need 8,736 at the median, and half of them more than 9,000;
  recommendation 4 now treats recipe equivalence at 1 pp as marginal. The full grid is `partB-power-grid.csv`.
- Coverage Monte Carlo error: exact enumeration over the discordant counts for every scenario with discordance <= 5 %
  (`partB-coverage-exact-cells.csv`) changes single values by up to about 0.02 (new null at n = 256: BCa 0.916, not
  0.898) and no conclusion.
- Minor: McNemar p for MobileNetV3-L INT8 0.00049 (printed 0.0004) and ResNet18 INT8 against Posit8 0.00054 (printed
  0.0005); "all low coverage is at 2.55 %" replaced by the exact statement (below 0.90 only at discordance <= 2.55 %,
  including the 2 % null, and n <= 256); revision-1 times in the header are now the file stamps (the handoff's 12:58 /
  13:10 were not clock readings); `addendum1` prints nothing, so its log is empty by design; event-free widths are no
  longer called identical "by construction".
