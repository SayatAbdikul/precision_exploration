# B2 format ranking: robustness to the recipe and to signedness (lane Q2, 2026-10-02)

All numbers are **development evidence on the ImageNet 1k screen list** (expected-credit top-1, 1,000 images,
pointwise 95 percent paired bootstrap intervals, seed 20260927, 10,000 resamples, no multiplicity adjustment:
261 same-width format pairs, 7,996 recipe-pair tests). Calibration only on the frozen v1 calibration lists.
Summaries: results/summaries/b2-rank-v1/ (written once). Figures: results/figures/b2-rank-tau-heatmap.png,
b2-rank-decomposition-v2.png (v1 of that figure has its legend over the ResNet18 bars), b2-rank-unsigned.png,
b2-rank-tau-heatmap-v2.png (revision 2: above-chance formats, ties declared; use it instead of the v1 heat map,
which covers all 25 formats with chance ties broken by summation order).

Revision 2 (2026-10-02, after review 1, artifacts/agent_orchestration/handoffs/Q2-rank-robustness-review.md):
protocol addendum-2 (b2-rank-protocol-v1-addendum-2.json, sha256 d3170e01...) added two derived summaries written
once beside the v1 files, rank-correlation-tol-v2.csv (rank statistics with ties declared at 10 decimals of mean
credit) and reversal-switches-v2.csv / addendum2-summary.json (which recipe switches differ in each reversal), by
`tools/experiment_b2_rank/addendum2.py`. No cell was rerun and no v1 summary changed. Corrected in this revision:
the bias-correction attribution of the reversals (section 1, 4), the Kendall tau headline ranges (now quoted from the
above-chance subset), two rounded unsigned differences, the 8-bit leaders on MNV3-L, the minimal-side pair list and
the meaning of "unchanged" for the proposal.

## 1. Numbers first

- **Cells.** 257 new cells (21 recipe arms x formats x 3 networks) plus L1's 150 matrix and 24 intrinsic cells reused
  read-only; every factorial is complete. Two redirected cells reproduce L1 cells bit for bit (identity, logits
  sha256, all readout arrays). 39 of 39 earlier runner records (artifacts/experiment_b2/runs/*) with the same
  configuration identity as a new cell give the same top-1. GPU: 2.30 h of cell time.
- **Q1.6 = signed INT8**: the int8 `default_signed` readout arrays equal L1's q1_6 `default` arrays on all three
  networks (q1_6 was not run again).
- **Orderings among strong recipes are stable overall, not pair by pair.** Kendall tau-b over the formats at or
  above 1 percent under either recipe (table 3, range over the three networks): default vs default_signed
  0.89-1.00; vs default with unsigned siblings 0.78-0.93; vs no bias correction 0.76-0.86; vs max-abs weight ranges
  0.60-0.71; vs minimal 0.55-0.65; vs v1 percentile 0.36-0.60; vs v1 max-abs 0.23-0.44. Over all 25 formats the
  values depend on how formats at exactly chance are tied (MNV2 has six at 0.001): with ties declared at 10 decimals
  (rank-correlation-tol-v2.csv) default vs no bias correction is 0.75-0.84, vs max-abs weights 0.69-0.74, vs minimal
  0.62-0.67; the v1 table 2 values differ from these by at most 0.04 over all widths and by up to 0.24 in the
  <=5-bit class, the above-chance values by at most 0.007 (8- and 6-bit classes: no change).
- **15 format pairs reverse between two strong recipes** with both intervals beyond 1 pp and neither format below
  1 percent (28 such recipe-pair reversals). **All 15 are 6-bit or <=5-bit pairs; no 8-bit pair reverses among
  strong recipes on any network.** 22 of the 28 reversals have `default_no_bias_correction` on one side, and 14 of
  the 15 pairs have at least one such reversal (the exception is ResNet18 int5/fp5_e2m2). Split by the switches that
  differ between the two recipes (reversal-switches-v2.csv): **bias correction alone** (default vs
  default_no_bias_correction) reverses **7 pairs** (ResNet18 mxfp6_e3m2/log6 and MNV3-L fp6_e2m3/posit6_es1,
  fp6_e3m2/bfp6, fp6_e3m2/fp6_e2m3, fp6_e3m2/log6, mxfp6_e3m2/log6, posit6_es1/log6); the unsigned sibling alone
  (default vs default_unsigned) 2 pairs (MNV3-L mxfp6_e3m2/log6 and posit6_es1/log6, both also under bias correction
  alone); the weight range alone 1 pair (ResNet18 int5/fp5_e2m2, also with default_signed and default_unsigned vs
  max-abs weights). The other **7 pairs reverse only when two switches change at once**: 5 only between
  default_no_bias_correction and default_unsigned (bias correction and the unsigned sibling: MNV2 fp6_e3m2/posit6_es1
  and posit6_es1/log6; MNV3-L fp6_e3m2/mxfp6_e3m2, fp6_e3m2/posit6_es1, mxfp6_e3m2/posit6_es1) and 2 only between
  default_no_bias_correction and default_weight_maxabs (ResNet18 fp6_e3m2/log6, int4/nf4). By rows: B alone 7, B+W 7,
  B+sibling 8, sibling alone 2, W alone 1, W+U 1, W+sibling 2.
  9 of the 15 pairs are MobileNetV3-Large 6-bit pairs; the formats involved are the wide-exponent ones
  (fp6_e3m2, mxfp6_e3m2, log6, posit6_es1) against fp6_e2m3/bfp6 and each other.
- **13 pairs reverse only against the sealed v1 max-abs/percentile results** (recipe artifacts), among them all four
  MobileNetV3-Large 8-bit INT8-versus-non-integer pairs (int8 vs fp8_e4m3fn, fp8_e5m2, log8, mxfp8_e4m3) and
  ResNet18 int6 vs fp6_e3m2/log6/mxfp6_e3m2. 5 further pairs reverse only between a minimal-side corner and a
  strong recipe (four ResNet18 posit pairs, fp6_e3m2/posit6_es1, mxfp6_e3m2/posit6_es1, posit6_es1/log6 and
  fp8_e5m2/posit8_es1, where posit collapses under max-abs weights, and MNV2 int5/fp5_e2m2), 22 only with a format at
  chance, 14 only by the point estimate.
- **Unsigned codes.** With the unsigned sibling at the nodes B2 proves non-negative, 8-bit formats move by at most
  about +2.7 pp except fp8_e5m2 (+2.6 RN18, +25.6 MNV2, +9.4 MNV3-L); 6-bit wide-exponent formats gain most
  (MNV2 fp6_e3m2 20.9 -> 55.8, log6 20.3 -> 54.5, mxfp6_e3m2 8.4 -> 30.4, posit6_es1 18.8 -> 42.7; RN18 fp6_e3m2
  +9.6 [+7.8, +11.5], mxfp6_e3m2 +11.5 [+9.7, +13.3], log6 +8.0 [+6.1, +9.8]); one loss: MNV3-L log6 53.2 -> 45.2
  (-7.9 [-10.3, -5.6]; differences are of unrounded means, signedness.csv). Integers: the B2
  default (unsigned) minus default_signed is -0.4 to +6.2 at 8 and 6 bits and +19.4 [+17.2, +21.5] for MNV2 int5.
- **INT versus the other formats when both get unsigned codes** (table 6): at 8 bits on ResNet18 every
  non-integer except fp8_e5m2 comes within 1 pp of INT8 (fp8_e4m3fn -0.8 [-1.9, +0.3], mxfp8 -0.2, log8 -0.4,
  posit8 +0.2), against -1.1 to -1.6 with signed codes; on MobileNetV2 fp8_e4m3fn/log8/mxfp8 stay 1.4-3.6 below
  INT8 (separated). So part of the INT8 advantage reported by the matrix is the unsigned-integer convention.
- **The membership of the exact-run proposal (results/summaries/b2-matrix-v1/proposal-d4.csv) is unchanged**
  under default, default_signed, default with unsigned siblings and the searched block arm: the same set of
  (role, format) entries; under default with unsigned siblings the order inside the selected list differs (log8
  before fp8_e4m3fn), which does not change which formats are run. It changes under every corner that
  drops bias correction (MXFP8 moves from selected to stress, BFP6 leaves) or MSE weight ranges (Posit8 moves to
  stress, Posit6 leaves) - a report only; the proposal is not changed.

## 2. Design (protocol b2-rank-protocol-v1, sha256 af1405d7..., addendum-1 sha256 e1f1eeb4..., addendum-2 sha256 d3170e01...)

- Recipe switches from minimal to default: U (unsigned activation codes at proven non-negative nodes; integers
  only in B2), W (MSE instead of max-abs weight range), B (empirical bias correction); both recipes keep fused-ReLU
  boundaries and MSE activation ranges. Integers: full 2^3 cube (default, default_signed, default_no_bias_correction,
  default_weight_maxabs, cum3_act_mse, minimal and two new corners rank_signed_no_bias_correction,
  rank_signed_weight_maxabs). Other scalar formats: 2x2 in (W, B) with the existing named arms. Block formats
  (bfp6, mxfp4/6/8): the intrinsic arm (max-abs activation blocks, the arm L1's rule (d) uses) as primary,
  cum5_act_maxabs = default, cum1_fused = minimal, plus rank_intrinsic_no_bias_correction and
  rank_intrinsic_weight_maxabs; the literal (searched) arm as sensitivity, complete on ResNet18 only.
  No CLE or AdaRound fit was run. AdaRound arms L-bc / L-nobc (lane L6b) enter for their 5 formats.
- Unsigned sibling rule (stated before measurement): U_n(F) = non-negative finite values of F's (n+1)-bit sibling
  (one more mantissa / fraction / log-fraction bit, posit(n+1, es)); for integers it is exactly B2's unsigned INT.
  Excluded: nf4, ternary, binary_pm1, q1_6. Injected in-process only; the accepted manifests are untouched.
- Reversal: X - Y beyond +1 pp under one recipe and beyond -1 pp under the other, by point estimate and by interval
  (both bounds beyond 1 pp). A reversal is flagged chance when either format is below 1 percent under either recipe
  (addendum-1: the protocol's literal wording was vacuous for a 1 pp margin); flagged reversals never count as robust.
  Strong recipes: default, default_signed, default_no_bias_correction, default_weight_maxabs, default with unsigned
  siblings, AdaRound L-bc/L-nobc. Weak: sealed v1 max-abs and percentile 99.9.
- Shapley decomposition of default minus minimal over the factorial corners (sums exactly to the total).
- Addendum-2 (after review 1, before computing anything it defines): rank statistics with mean credits rounded to
  10 decimals before Kendall tau-b / Spearman (point and every resample), so equal-up-to-summation-order means are
  ties; and a label of the switches (B bias correction, W weight range, U signed/unsigned integer codes, S unsigned
  sibling) that differ between the two recipes of each strong reversal. It also records the cells.py versions seen
  in cell records (293b80f4 before addendum-1: 83 factorial cells; 7ac9e5b4 at addendum-1: 23 factorial and 8
  searched cells; a4221295 from 16:55, a per-item part in run_jobs with no numerical change: 92 factorial and all 51
  unsigned cells). The source hash is recorded in each record but is not part of the configuration identity, and
  the reviewer reproduced a cell bit for bit.

## 3. Tables

Table 1 (default minus minimal and its Shapley parts, formats above 1 percent under either corner),
table 2 (Kendall tau-b, all formats, v1 statistic: depends on tie order at chance, see rank-correlation-tol-v2.csv), table 3 (tau-b, formats at or above 1 percent; quoted in section 1), table 4 (pair verdicts),
table 5 (unsigned minus signed under default), table 6 is int-vs-format-gaps.csv, table 7 (proposal by recipe)
follow at the end of this document, generated by `tools.experiment_b2_rank.report` from the summaries.

## 4. Reading

- **Robust orderings.** At 8 bits no same-width pair reverses between two strong recipes with both formats above
  chance (Posit8 collapses to 0.1 under max-abs weight ranges; those reversals are flagged chance). Under default,
  Posit8 is first on all three networks (RN18 69.6, MNV2 72.4, MNV3-L 73.4). INT8/Q1.6 come next on ResNet18 (69.2)
  and MNV2 (72.3/72.2), ahead of log8, fp8_e4m3fn and mxfp8 by 1.1-5.7 pp. On MNV3-L there is no INT8 lead: Q1.6 72.0,
  fp8_e4m3fn 71.7 and INT8 71.6 are within 0.4 pp under default, and without bias correction fp8_e4m3fn is ahead
  (72.5 vs INT8 70.5, Q1.6 69.9). fp8_e5m2 is last on every network. The MNV3-L INT8-below-FP8 results seen under v1
  max-abs are max-abs artifacts (INT8 itself collapsed to 4.1 there). At 6 bits on ResNet18 and MNV2 the order of the
  narrow-exponent formats (int6, fp6_e2m3, bfp6) is stable; the wide-exponent 6-bit formats (fp6_e3m2, mxfp6_e3m2,
  log6, posit6_es1) are the ones that reverse.
- **Recipe-sensitive orderings involve bias correction, often together with a second switch.** 22 of the 28 strong
  reversals and 14 of the 15 pairs involve the recipe without bias correction, but bias correction alone (default vs
  default_no_bias_correction) reverses 7 pairs (6 on MNV3-L, ResNet18 mxfp6_e3m2/log6). 5 pairs reverse only when
  bias correction is dropped and the unsigned sibling is switched on at the same time (both MNV2 pairs and three
  MNV3-L pairs), 2 only when bias correction is dropped and the weight range changes (ResNet18 fp6_e3m2/log6,
  int4/nf4), and 1 under the weight range alone (ResNet18 int5/fp5_e2m2: MSE weight ranges cost int5 23.6 pp of
  Shapley credit). So "bias correction reorders them" holds as stated for 7 pairs; for the other 8 the data show a
  reversal between two strong recipes that differ in two switches, which this design does not attribute to one of
  them. In the decomposition, bias correction is the largest Shapley part of default minus minimal for most 6-bit
  and <=5-bit cells (negative for the wide-exponent formats on RN18/MNV2, positive on MNV3-L); the MSE weight range
  is the decisive part for posit (about +36 to +74 pp at 8 and 6 bits, figure); the unsigned switch is small at
  8 bits and +1.3 to +14.5 at 6 and 5 bits for integers. Lane Q3 diagnoses why bias correction fails on
  wide-exponent formats; this lane only measures that it reorders them.
- **For the paper (claim C1).** C1 survives among strong recipes only in a narrow form: at <=6 bits, the recipe
  reorders same-width formats (15 pairs, 9 on MNV3-L; 7 of them under the bias-correction switch alone); at 8 bits
  the recipe changes levels, not the order, once max-abs and chance-level collapses are excluded. Claims built on
  v1 max-abs versus default (13 pairs) are recipe artifacts.

## 5. Limits

- Development evidence on 1k images; pointwise intervals with no multiplicity control over 7,996 tests.
- AdaRound covers 5 formats only; its rank correlations rest on 4-5 formats and are uninformative.
- Rank correlations over all formats depend on how formats at exactly chance are tied (review finding B): the v1
  table 2 breaks such ties by floating-point summation order, and its bootstrap intervals can exclude the point
  estimate. Quote the above-chance table 3, or the tie-declared rank-correlation-tol-v2.csv (addendum-2).
- The switch labels of the reversals say which recipe settings differ, not which one causes a reversal when two
  differ; the factorial has no corner with the unsigned sibling and without bias correction, so B+S reversals are
  not split further.
- In unsigned scalar cells the 'codes' field of the occupancy/codebook metadata is arange over the sibling levels,
  not the sibling's bit pattern (review note G); quantized values and every readout are unaffected.
- The searched block arm is complete on ResNet18 only (protocol priority 6, MobileNets, not run for budget).
- Unsigned siblings are a quality measurement only; hardware cost is out of scope and the candidate set is unchanged.
- The new integer and intrinsic corners are recipe definitions registered in tools/experiment_b2_rank/cells.py,
  not frozen B2 recipes.
- Execution: heavy-lock contention on the shared GPU delayed the bias-correction cells by several hours; one block
  unsigned bug (block plan marked unsigned) was fixed before any unsigned cell existed (addendum-1).

## Appendix: tables (generated by `.venv/bin/python -m tools.experiment_b2_rank.report`)

### decomposition
| model | format | bits | default | minimal | default - minimal [95%] | U (unsigned) | W (MSE weights) | B (bias corr.) |
|---|---|---|---|---|---|---|---|---|
| ResNet18 | int8 | 8 | 69.2 | 69.7 | -0.5 [-1.3, +0.3] | +0.0 [-0.4, +0.4] | -0.3 [-0.8, +0.1] | -0.2 [-0.7, +0.2] |
| ResNet18 | int6 | 6 | 65.9 | 65.8 | +0.1 [-1.6, +1.8] | +1.3 [+0.6, +2.1] | -1.0 [-1.8, -0.2] | -0.3 [-1.6, +1.1] |
| ResNet18 | int5 | 5 | 13.7 | 52.2 | -38.5 [-41.4, -35.6] | +4.3 [+3.6, +5.1] | -23.6 [-25.1, -22.1] | -19.2 [-21.1, -17.3] |
| ResNet18 | int4 | 4 | 0.1 | 17.1 | -17.0 [-18.8, -15.3] | +5.0 [+4.3, +5.8] | -3.9 [-5.0, -2.7] | -18.1 [-19.6, -16.7] |
| ResNet18 | q1_6 | 8 | 69.2 | 69.7 | -0.6 [-1.4, +0.3] | n/a | -0.3 [-0.9, +0.3] | -0.2 [-0.8, +0.3] |
| ResNet18 | fp8_e4m3fn | 8 | 67.7 | 68.6 | -1.0 [-2.2, +0.3] | n/a | -0.4 [-1.3, +0.5] | -0.5 [-1.4, +0.3] |
| ResNet18 | fp8_e5m2 | 8 | 61.2 | 63.0 | -1.8 [-3.4, -0.2] | n/a | +1.0 [+0.1, +2.0] | -2.8 [-4.2, -1.5] |
| ResNet18 | fp7_e3m3 | 7 | 67.5 | 68.4 | -0.9 [-2.1, +0.2] | n/a | -0.1 [-0.9, +0.7] | -0.9 [-1.7, -0.1] |
| ResNet18 | fp6_e3m2 | 6 | 53.6 | 62.9 | -9.3 [-11.4, -7.2] | n/a | +1.6 [+0.6, +2.6] | -10.9 [-12.8, -9.1] |
| ResNet18 | fp6_e2m3 | 6 | 66.7 | 68.0 | -1.3 [-2.9, +0.2] | n/a | -0.1 [-1.0, +0.8] | -1.3 [-2.5, -0.1] |
| ResNet18 | fp5_e2m2 | 5 | 39.1 | 56.9 | -17.8 [-20.6, -15.0] | n/a | -7.3 [-8.6, -5.9] | -10.5 [-12.6, -8.4] |
| ResNet18 | fp4_e2m1 | 4 | 0.1 | 19.2 | -19.1 [-21.0, -17.3] | n/a | +0.7 [-0.5, +1.8] | -19.8 [-21.5, -18.1] |
| ResNet18 | bfp6 | 6 | 68.2 | 68.8 | -0.6 [-1.7, +0.5] | n/a | -0.5 [-1.2, +0.2] | -0.1 [-0.9, +0.6] |
| ResNet18 | mxfp8_e4m3 | 8 | 67.7 | 68.7 | -1.0 [-2.1, +0.1] | n/a | +0.0 [+0.0, +0.0] | -1.0 [-2.1, +0.1] |
| ResNet18 | mxfp6_e3m2 | 6 | 50.7 | 64.5 | -13.7 [-15.9, -11.6] | n/a | -0.2 [-1.0, +0.6] | -13.5 [-15.6, -11.5] |
| ResNet18 | mxfp4_e2m1 | 4 | 0.2 | 41.6 | -41.4 [-43.9, -38.9] | n/a | -0.6 [-1.6, +0.3] | -40.8 [-43.2, -38.4] |
| ResNet18 | posit8_es1 | 8 | 69.6 | 4.4 | +65.3 [+62.4, +68.1] | n/a | +67.2 [+64.4, +69.9] | -1.9 [-2.7, -1.2] |
| ResNet18 | posit6_es1 | 6 | 59.6 | 3.8 | +55.8 [+53.0, +58.6] | n/a | +59.9 [+57.3, +62.4] | -4.1 [-5.1, -3.0] |
| ResNet18 | posit4_es0 | 4 | 0.1 | 24.0 | -24.0 [-26.0, -22.0] | n/a | +1.4 [+0.4, +2.5] | -25.4 [-27.2, -23.6] |
| ResNet18 | log8 | 8 | 68.1 | 68.4 | -0.3 [-1.4, +0.8] | n/a | +0.4 [-0.3, +1.1] | -0.7 [-1.5, +0.1] |
| ResNet18 | log6 | 6 | 54.5 | 61.5 | -7.0 [-9.2, -4.8] | n/a | -0.5 [-1.7, +0.6] | -6.4 [-8.3, -4.5] |
| ResNet18 | log4 | 4 | 0.1 | 15.1 | -15.0 [-16.8, -13.3] | n/a | +3.9 [+2.9, +4.9] | -18.9 [-20.6, -17.3] |
| ResNet18 | nf4 | 4 | 0.3 | 34.4 | -34.2 [-36.5, -31.9] | n/a | -2.7 [-4.0, -1.4] | -31.5 [-33.6, -29.4] |
| MNV2 | int8 | 8 | 72.3 | 72.3 | +0.0 [-0.8, +0.9] | -0.0 [-0.4, +0.4] | -0.1 [-0.6, +0.3] | +0.1 [-0.4, +0.6] |
| MNV2 | int6 | 6 | 70.1 | 66.3 | +3.8 [+1.9, +5.7] | +1.7 [+1.0, +2.5] | -0.7 [-1.7, +0.4] | +2.7 [+1.4, +4.1] |
| MNV2 | int5 | 5 | 32.5 | 31.5 | +1.1 [-1.8, +4.0] | +14.5 [+13.2, +15.7] | -11.9 [-13.3, -10.4] | -1.5 [-3.6, +0.6] |
| MNV2 | q1_6 | 8 | 72.2 | 72.3 | -0.1 [-0.9, +0.8] | n/a | -0.1 [-0.7, +0.4] | +0.0 [-0.6, +0.7] |
| MNV2 | fp8_e4m3fn | 8 | 67.7 | 66.7 | +1.0 [-0.8, +2.9] | n/a | +1.8 [+0.8, +2.8] | -0.8 [-2.3, +0.7] |
| MNV2 | fp8_e5m2 | 8 | 23.2 | 41.9 | -18.7 [-21.6, -15.7] | n/a | +2.5 [+1.0, +4.1] | -21.2 [-23.7, -18.7] |
| MNV2 | fp7_e3m3 | 7 | 67.6 | 66.6 | +1.0 [-0.8, +3.0] | n/a | +1.6 [+0.7, +2.5] | -0.6 [-2.1, +1.0] |
| MNV2 | fp6_e3m2 | 6 | 20.9 | 42.6 | -21.7 [-24.7, -18.8] | n/a | -0.1 [-1.5, +1.3] | -21.7 [-24.2, -19.0] |
| MNV2 | fp6_e2m3 | 6 | 66.2 | 66.6 | -0.4 [-2.3, +1.5] | n/a | -0.4 [-1.4, +0.6] | +0.0 [-1.6, +1.6] |
| MNV2 | fp5_e2m2 | 5 | 8.3 | 41.0 | -32.7 [-35.4, -29.9] | n/a | -6.9 [-8.3, -5.5] | -25.8 [-28.1, -23.5] |
| MNV2 | bfp6 | 6 | 65.6 | 63.9 | +1.7 [-0.5, +3.9] | n/a | -1.7 [-2.8, -0.7] | +3.4 [+1.6, +5.3] |
| MNV2 | mxfp8_e4m3 | 8 | 66.6 | 65.2 | +1.3 [-0.7, +3.3] | n/a | +0.0 [+0.0, +0.0] | +1.3 [-0.7, +3.3] |
| MNV2 | mxfp6_e3m2 | 6 | 8.4 | 41.9 | -33.5 [-36.3, -30.4] | n/a | +0.2 [-0.6, +1.1] | -33.7 [-36.5, -30.8] |
| MNV2 | posit8_es1 | 8 | 72.4 | 0.1 | +72.3 [+69.5, +75.0] | n/a | +72.2 [+69.4, +74.8] | +0.1 [-0.5, +0.8] |
| MNV2 | posit6_es1 | 6 | 18.8 | 0.1 | +18.7 [+16.5, +21.0] | n/a | +36.5 [+34.5, +38.6] | -17.9 [-19.3, -16.4] |
| MNV2 | log8 | 8 | 68.2 | 67.4 | +0.8 [-1.1, +2.7] | n/a | +0.5 [-0.4, +1.5] | +0.2 [-1.3, +1.8] |
| MNV2 | log6 | 6 | 20.3 | 47.5 | -27.2 [-30.1, -24.2] | n/a | -1.0 [-2.5, +0.4] | -26.1 [-28.7, -23.5] |
| MNV2 | nf4 | 4 | 0.1 | 1.5 | -1.4 [-2.1, -0.8] | n/a | +0.1 [-0.3, +0.5] | -1.5 [-2.0, -1.0] |
| MNV3-L | int8 | 8 | 71.6 | 70.2 | +1.4 [-0.4, +3.4] | +0.3 [-0.4, +1.1] | -0.5 [-1.3, +0.3] | +1.6 [+0.3, +2.9] |
| MNV3-L | int6 | 6 | 21.0 | 2.5 | +18.4 [+16.0, +20.9] | +3.2 [+2.4, +4.1] | -1.2 [-2.1, -0.3] | +16.4 [+14.5, +18.4] |
| MNV3-L | int5 | 5 | 8.1 | 0.8 | +7.3 [+5.8, +8.8] | +2.1 [+1.5, +2.7] | -0.7 [-1.3, +0.0] | +5.9 [+4.9, +6.8] |
| MNV3-L | q1_6 | 8 | 72.0 | 70.2 | +1.8 [-0.1, +3.7] | n/a | -0.4 [-1.4, +0.6] | +2.2 [+0.7, +3.8] |
| MNV3-L | fp8_e4m3fn | 8 | 71.7 | 68.5 | +3.2 [+1.5, +5.0] | n/a | +2.1 [+1.0, +3.2] | +1.1 [-0.1, +2.4] |
| MNV3-L | fp8_e5m2 | 8 | 46.6 | 45.2 | +1.4 [-1.3, +4.1] | n/a | +3.4 [+1.6, +5.0] | -1.9 [-4.2, +0.3] |
| MNV3-L | fp7_e3m3 | 7 | 71.1 | 69.0 | +2.1 [+0.3, +3.8] | n/a | +0.8 [-0.2, +1.9] | +1.2 [-0.1, +2.5] |
| MNV3-L | fp6_e3m2 | 6 | 44.6 | 42.3 | +2.3 [-0.4, +5.0] | n/a | +5.5 [+4.0, +7.1] | -3.2 [-5.3, -1.0] |
| MNV3-L | fp6_e2m3 | 6 | 58.4 | 36.8 | +21.7 [+18.7, +24.7] | n/a | -1.2 [-2.8, +0.4] | +22.9 [+20.4, +25.4] |
| MNV3-L | fp5_e2m2 | 5 | 10.6 | 0.9 | +9.7 [+8.0, +11.4] | n/a | -0.6 [-1.6, +0.3] | +10.3 [+8.8, +11.8] |
| MNV3-L | bfp6 | 6 | 56.6 | 42.4 | +14.2 [+11.4, +17.1] | n/a | +0.8 [-0.9, +2.6] | +13.3 [+11.1, +15.7] |
| MNV3-L | mxfp8_e4m3 | 8 | 70.1 | 66.2 | +3.9 [+1.8, +6.0] | n/a | +0.0 [+0.0, +0.0] | +3.9 [+1.8, +6.0] |
| MNV3-L | mxfp6_e3m2 | 6 | 44.9 | 29.0 | +15.9 [+13.1, +18.8] | n/a | +0.5 [-0.7, +1.7] | +15.4 [+12.8, +18.1] |
| MNV3-L | posit8_es1 | 8 | 73.4 | 0.0 | +73.4 [+70.6, +76.1] | n/a | +74.0 [+71.4, +76.6] | -0.7 [-1.3, -0.0] |
| MNV3-L | posit6_es1 | 6 | 42.9 | 0.1 | +42.8 [+39.9, +45.6] | n/a | +44.6 [+42.1, +47.1] | -1.8 [-3.2, -0.5] |
| MNV3-L | log8 | 8 | 69.8 | 69.4 | +0.5 [-1.4, +2.4] | n/a | +0.5 [-0.6, +1.6] | -0.0 [-1.5, +1.5] |
| MNV3-L | log6 | 6 | 53.2 | 15.9 | +37.3 [+34.3, +40.2] | n/a | +4.3 [+2.9, +5.8] | +33.0 [+30.5, +35.4] |
| MNV3-L | nf4 | 4 | 2.2 | 0.2 | +2.0 [+1.2, +2.8] | n/a | -0.2 [-0.7, +0.4] | +2.1 [+1.5, +2.7] |

### tau_all
v1 statistic (rank-correlation.csv): ties at chance broken by floating-point summation order; the tie-declared values are in rank-correlation-tol-v2.csv (addendum-2) and section 1.

| recipe A | recipe B | ResNet18 tau-b [95%] (n) | MNV2 tau-b [95%] (n) | MNV3-L tau-b [95%] (n) |
|---|---|---|---|---|
| default | minimal | 0.66 [0.57, 0.68] (25) | 0.60 [0.53, 0.70] (25) | 0.67 [0.57, 0.70] (25) |
| default | v1_maxabs | 0.59 [0.46, 0.63] (20) | 0.54 [0.39, 0.58] (23) | 0.40 [0.34, 0.53] (24) |
| default | v1_percentile_99_9 | 0.52 [0.41, 0.56] (21) | 0.60 [0.50, 0.65] (23) | 0.63 [0.55, 0.69] (22) |
| minimal | v1_maxabs | 0.86 [0.73, 0.89] (20) | 0.66 [0.54, 0.74] (23) | 0.56 [0.48, 0.68] (24) |
| default | default_no_bias_correction | 0.83 [0.75, 0.89] (25) | 0.81 [0.74, 0.86] (25) | 0.75 [0.71, 0.82] (25) |
| default | default_weight_maxabs | 0.69 [0.58, 0.70] (25) | 0.67 [0.61, 0.79] (25) | 0.74 [0.67, 0.77] (25) |
| default | default_signed | 1.00 [0.96, 1.00] (25) | 0.95 [0.93, 0.96] (25) | 0.98 [0.96, 1.00] (25) |
| default | default_unsigned | 0.93 [0.82, 0.93] (25) | 0.87 [0.84, 0.92] (25) | 0.89 [0.84, 0.93] (25) |
| default_no_bias_correction | default_weight_maxabs | 0.62 [0.50, 0.67] (25) | 0.64 [0.53, 0.68] (25) | 0.56 [0.50, 0.63] (25) |
| default | default_searched | 0.97 [0.94, 0.99] (25) | 0.99 [0.97, 1.00] (25) | 0.99 [0.96, 1.00] (25) |
| default | adaround_bc | 0.80 [0.60, 1.00] (5) | 0.80 [0.80, 1.00] (5) | 0.80 [0.80, 1.00] (5) |
| adaround_bc | adaround_nobc | 0.60 [0.60, 1.00] (5) | 0.60 [0.40, 1.00] (5) | 0.80 [0.80, 1.00] (5) |

### tau_all_above
| recipe A | recipe B | ResNet18 tau-b [95%] (n) | MNV2 tau-b [95%] (n) | MNV3-L tau-b [95%] (n) |
|---|---|---|---|---|
| default | minimal | 0.65 [0.56, 0.67] (23) | 0.56 [0.47, 0.62] (18) | 0.55 [0.49, 0.62] (18) |
| default | v1_maxabs | 0.44 [0.30, 0.51] (17) | 0.23 [0.12, 0.32] (15) | 0.41 [0.37, 0.50] (17) |
| default | v1_percentile_99_9 | 0.49 [0.41, 0.52] (19) | 0.36 [0.32, 0.44] (15) | 0.60 [0.55, 0.66] (16) |
| minimal | v1_maxabs | 0.84 [0.69, 0.88] (18) | 0.59 [0.40, 0.67] (14) | 0.52 [0.41, 0.60] (12) |
| default | default_no_bias_correction | 0.86 [0.78, 0.91] (23) | 0.82 [0.77, 0.85] (20) | 0.76 [0.71, 0.82] (18) |
| default | default_weight_maxabs | 0.60 [0.50, 0.62] (20) | 0.62 [0.57, 0.70] (18) | 0.71 [0.63, 0.74] (18) |
| default | default_signed | 1.00 [0.97, 1.00] (17) | 0.89 [0.85, 0.92] (17) | 0.98 [0.97, 1.00] (18) |
| default | default_unsigned | 0.93 [0.74, 0.94] (17) | 0.78 [0.68, 0.84] (17) | 0.84 [0.76, 0.88] (18) |
| default_no_bias_correction | default_weight_maxabs | 0.60 [0.49, 0.63] (23) | 0.54 [0.46, 0.59] (20) | 0.50 [0.45, 0.54] (18) |
| default | default_searched | 0.94 [0.91, 0.99] (17) | 0.99 [0.96, 1.00] (17) | 0.99 [0.93, 1.00] (18) |
| default | adaround_bc | 0.80 [0.60, 1.00] (5) | 0.67 [0.67, 1.00] (4) | 0.67 [0.67, 1.00] (4) |
| adaround_bc | adaround_nobc | 0.60 [0.60, 1.00] (5) | 0.60 [0.40, 1.00] (5) | 0.67 [0.67, 1.00] (4) |

### verdicts
| model | class | interval_only_with_chance_format | maxabs_percentile_only | minimal_side | point_only | survives_among_strong |
|---|---|---|---|---|---|---|
| ResNet18 | 8 | 0 | 0 | 1 | 3 | 0 |
| ResNet18 | 6 | 0 | 3 | 3 | 1 | 2 |
| ResNet18 | le5 | 5 | 0 | 0 | 1 | 2 |
| MNV2 | 8 | 4 | 1 | 0 | 1 | 0 |
| MNV2 | 6 | 5 | 5 | 0 | 1 | 2 |
| MNV2 | le5 | 0 | 0 | 1 | 1 | 0 |
| MNV3-L | 8 | 5 | 4 | 0 | 3 | 0 |
| MNV3-L | 6 | 3 | 0 | 0 | 3 | 9 |
| MNV3-L | le5 | 0 | 0 | 0 | 0 | 0 |

- **survives_among_strong** (15): ResNet18 fp6_e3m2/log6; ResNet18 mxfp6_e3m2/log6; ResNet18 int4/nf4; ResNet18 int5/fp5_e2m2; MNV2 fp6_e3m2/posit6_es1; MNV2 posit6_es1/log6; MNV3-L fp6_e2m3/posit6_es1; MNV3-L fp6_e3m2/bfp6; MNV3-L fp6_e3m2/fp6_e2m3; MNV3-L fp6_e3m2/log6; MNV3-L fp6_e3m2/mxfp6_e3m2; MNV3-L fp6_e3m2/posit6_es1; MNV3-L mxfp6_e3m2/log6; MNV3-L mxfp6_e3m2/posit6_es1; MNV3-L posit6_es1/log6
- **minimal_side** (5): ResNet18 fp6_e3m2/posit6_es1; ResNet18 mxfp6_e3m2/posit6_es1; ResNet18 posit6_es1/log6; ResNet18 fp8_e5m2/posit8_es1; MNV2 int5/fp5_e2m2
- **maxabs_percentile_only** (13): ResNet18 int6/fp6_e3m2; ResNet18 int6/log6; ResNet18 int6/mxfp6_e3m2; MNV2 fp6_e2m3/bfp6; MNV2 fp6_e2m3/mxfp6_e3m2; MNV2 fp6_e3m2/fp6_e2m3; MNV2 fp6_e3m2/mxfp6_e3m2; MNV2 mxfp6_e3m2/log6; MNV2 mxfp8_e4m3/log8; MNV3-L int8/fp8_e4m3fn; MNV3-L int8/fp8_e5m2; MNV3-L int8/log8; MNV3-L int8/mxfp8_e4m3

### signedness
| model | format | bits | unsigned | signed | unsigned - signed [95%] |
|---|---|---|---|---|---|
| ResNet18 | int8 | 8 | 69.2 | 69.2 | +0.1 [-0.6, +0.7] |
| ResNet18 | int6 | 6 | 65.9 | 61.8 | +4.1 [+2.8, +5.4] |
| ResNet18 | int5 | 5 | 13.7 | 7.7 | +6.0 [+4.7, +7.3] |
| ResNet18 | int4 | 4 | 0.1 | 0.1 | -0.0 [-0.1, +0.1] |
| ResNet18 | fp8_e4m3fn | 8 | 68.4 | 67.7 | +0.7 [-0.2, +1.7] |
| ResNet18 | fp8_e5m2 | 8 | 63.8 | 61.2 | +2.6 [+1.3, +4.0] |
| ResNet18 | fp7_e3m3 | 7 | 68.1 | 67.5 | +0.6 [-0.3, +1.5] |
| ResNet18 | fp6_e3m2 | 6 | 63.2 | 53.6 | +9.6 [+7.8, +11.5] |
| ResNet18 | fp6_e2m3 | 6 | 68.1 | 66.7 | +1.5 [+0.4, +2.7] |
| ResNet18 | fp5_e2m2 | 5 | 58.3 | 39.1 | +19.2 [+16.9, +21.5] |
| ResNet18 | fp4_e2m1 | 4 | 0.1 | 0.1 | -0.0 [-0.1, +0.1] |
| ResNet18 | bfp6 | 6 | 69.2 | 68.2 | +1.0 [+0.2, +1.8] |
| ResNet18 | mxfp8_e4m3 | 8 | 69.0 | 67.7 | +1.3 [+0.4, +2.2] |
| ResNet18 | mxfp6_e3m2 | 6 | 62.3 | 50.7 | +11.5 [+9.7, +13.3] |
| ResNet18 | mxfp4_e2m1 | 4 | 0.6 | 0.2 | +0.4 [+0.1, +0.7] |
| ResNet18 | posit8_es1 | 8 | 69.5 | 69.6 | -0.2 [-0.9, +0.5] |
| ResNet18 | posit6_es1 | 6 | 64.9 | 59.6 | +5.3 [+3.7, +6.9] |
| ResNet18 | posit4_es0 | 4 | 0.2 | 0.1 | +0.1 [-0.0, +0.3] |
| ResNet18 | log8 | 8 | 68.8 | 68.1 | +0.7 [-0.2, +1.7] |
| ResNet18 | log6 | 6 | 62.5 | 54.5 | +8.0 [+6.1, +9.8] |
| ResNet18 | log4 | 4 | 0.1 | 0.1 | -0.0 [-0.1, +0.1] |
| MNV2 | int8 | 8 | 72.3 | 72.2 | +0.1 [-0.6, +0.8] |
| MNV2 | int6 | 6 | 70.1 | 67.0 | +3.1 [+1.8, +4.5] |
| MNV2 | int5 | 5 | 32.5 | 13.2 | +19.4 [+17.2, +21.5] |
| MNV2 | int4 | 4 | 0.1 | 0.1 | +0.0 [+0.0, +0.0] |
| MNV2 | fp8_e4m3fn | 8 | 70.2 | 67.7 | +2.5 [+1.2, +3.8] |
| MNV2 | fp8_e5m2 | 8 | 48.8 | 23.2 | +25.6 [+23.1, +28.1] |
| MNV2 | fp7_e3m3 | 7 | 69.8 | 67.6 | +2.2 [+0.9, +3.5] |
| MNV2 | fp6_e3m2 | 6 | 55.8 | 20.9 | +34.9 [+32.2, +37.7] |
| MNV2 | fp6_e2m3 | 6 | 70.4 | 66.2 | +4.2 [+2.8, +5.6] |
| MNV2 | fp5_e2m2 | 5 | 33.2 | 8.3 | +24.9 [+22.6, +27.2] |
| MNV2 | fp4_e2m1 | 4 | 0.1 | 0.1 | +0.0 [+0.0, +0.0] |
| MNV2 | bfp6 | 6 | 69.5 | 65.6 | +3.9 [+2.5, +5.4] |
| MNV2 | mxfp8_e4m3 | 8 | 68.7 | 66.6 | +2.1 [+0.8, +3.5] |
| MNV2 | mxfp6_e3m2 | 6 | 30.4 | 8.4 | +22.0 [+19.6, +24.4] |
| MNV2 | mxfp4_e2m1 | 4 | 0.1 | 0.1 | -0.0 [-0.0, -0.0] |
| MNV2 | posit8_es1 | 8 | 72.7 | 72.4 | +0.3 [-0.6, +1.2] |
| MNV2 | posit6_es1 | 6 | 42.7 | 18.8 | +24.0 [+21.5, +26.3] |
| MNV2 | posit4_es0 | 4 | 0.1 | 0.1 | +0.0 [-0.0, +0.0] |
| MNV2 | log8 | 8 | 70.9 | 68.2 | +2.7 [+1.6, +3.9] |
| MNV2 | log6 | 6 | 54.5 | 20.3 | +34.1 [+31.2, +36.9] |
| MNV2 | log4 | 4 | 0.1 | 0.1 | +0.0 [+0.0, +0.0] |
| MNV3-L | int8 | 8 | 71.6 | 72.0 | -0.4 [-1.7, +0.9] |
| MNV3-L | int6 | 6 | 21.0 | 14.8 | +6.2 [+4.2, +8.2] |
| MNV3-L | int5 | 5 | 8.1 | 3.5 | +4.5 [+3.1, +6.0] |
| MNV3-L | int4 | 4 | 0.1 | 0.1 | +0.0 [-0.1, +0.1] |
| MNV3-L | fp8_e4m3fn | 8 | 72.6 | 71.7 | +0.9 [-0.2, +2.0] |
| MNV3-L | fp8_e5m2 | 8 | 56.0 | 46.6 | +9.4 [+7.5, +11.5] |
| MNV3-L | fp7_e3m3 | 7 | 71.7 | 71.1 | +0.6 [-0.5, +1.7] |
| MNV3-L | fp6_e3m2 | 6 | 47.8 | 44.6 | +3.3 [+1.3, +5.3] |
| MNV3-L | fp6_e2m3 | 6 | 58.7 | 58.4 | +0.2 [-1.7, +2.2] |
| MNV3-L | fp5_e2m2 | 5 | 14.5 | 10.6 | +3.9 [+1.9, +5.8] |
| MNV3-L | fp4_e2m1 | 4 | 0.1 | 0.1 | -0.1 [-0.1, +0.0] |
| MNV3-L | bfp6 | 6 | 56.3 | 56.6 | -0.3 [-2.6, +2.1] |
| MNV3-L | mxfp8_e4m3 | 8 | 71.3 | 70.1 | +1.3 [+0.2, +2.4] |
| MNV3-L | mxfp6_e3m2 | 6 | 59.4 | 44.9 | +14.5 [+12.2, +16.9] |
| MNV3-L | mxfp4_e2m1 | 4 | 0.5 | 0.5 | +0.0 [-0.4, +0.5] |
| MNV3-L | posit8_es1 | 8 | 74.1 | 73.4 | +0.7 [-0.1, +1.5] |
| MNV3-L | posit6_es1 | 6 | 53.0 | 42.9 | +10.1 [+8.0, +12.3] |
| MNV3-L | posit4_es0 | 4 | 0.1 | 0.1 | -0.0 [-0.0, +0.0] |
| MNV3-L | log8 | 8 | 71.7 | 69.8 | +1.9 [+0.6, +3.2] |
| MNV3-L | log6 | 6 | 45.2 | 53.2 | -7.9 [-10.3, -5.6] |
| MNV3-L | log4 | 4 | 0.1 | 0.1 | +0.0 [-0.0, +0.0] |

### proposal
| view | entries | passing | selected | stress | added | dropped | identical |
|---|---|---|---|---|---|---|---|
| default | 14 | 5 | posit8_es1 fp8_e4m3fn log8 mxfp8_e4m3 fp7_e3m3 | fp6_e2m3 bfp6 int6 log6 posit6_es1 nf4 ternary |  |  | True |
| default_signed | 14 | 5 | posit8_es1 fp8_e4m3fn log8 mxfp8_e4m3 fp7_e3m3 | fp6_e2m3 bfp6 int6 log6 posit6_es1 nf4 ternary |  |  | True |
| default_no_bias_correction | 13 | 4 | posit8_es1 fp8_e4m3fn log8 fp7_e3m3 | mxfp8_e4m3 fp6_e2m3 posit6_es1 int6 log6 nf4 ternary | stress:mxfp8_e4m3 | selected:mxfp8_e4m3 stress:bfp6 | False |
| default_weight_maxabs | 13 | 4 | fp8_e4m3fn log8 mxfp8_e4m3 fp7_e3m3 | posit8_es1 fp6_e2m3 bfp6 int6 log6 nf4 ternary | stress:posit8_es1 | selected:posit8_es1 stress:posit6_es1 | False |
| cum3_act_mse | 12 | 3 | log8 fp8_e4m3fn fp7_e3m3 | mxfp8_e4m3 posit8_es1 fp6_e2m3 int6 log6 nf4 ternary | stress:mxfp8_e4m3 stress:posit8_es1 | selected:posit8_es1 selected:mxfp8_e4m3 stress:bfp6 stress:posit6_es1 | False |
| rank_signed_no_bias_correction | 13 | 4 | posit8_es1 fp8_e4m3fn log8 fp7_e3m3 | mxfp8_e4m3 fp6_e2m3 posit6_es1 int6 log6 nf4 ternary | stress:mxfp8_e4m3 | selected:mxfp8_e4m3 stress:bfp6 | False |
| rank_signed_weight_maxabs | 13 | 4 | fp8_e4m3fn log8 mxfp8_e4m3 fp7_e3m3 | posit8_es1 fp6_e2m3 bfp6 int6 log6 nf4 ternary | stress:posit8_es1 | selected:posit8_es1 stress:posit6_es1 | False |
| minimal | 12 | 3 | log8 fp8_e4m3fn fp7_e3m3 | mxfp8_e4m3 posit8_es1 fp6_e2m3 int6 log6 nf4 ternary | stress:mxfp8_e4m3 stress:posit8_es1 | selected:posit8_es1 selected:mxfp8_e4m3 stress:bfp6 stress:posit6_es1 | False |
| default_unsigned | 14 | 5 | posit8_es1 log8 fp8_e4m3fn mxfp8_e4m3 fp7_e3m3 | fp6_e2m3 bfp6 log6 posit6_es1 int6 nf4 ternary |  |  | True |
| default_searched | 14 | 5 | posit8_es1 fp8_e4m3fn log8 mxfp8_e4m3 fp7_e3m3 | fp6_e2m3 bfp6 int6 log6 posit6_es1 nf4 ternary |  |  | True |
| default_no_bias_correction_searched | 12 | 4 | posit8_es1 fp8_e4m3fn log8 fp7_e3m3 | fp6_e2m3 posit6_es1 int6 log6 nf4 ternary |  | selected:mxfp8_e4m3 stress:bfp6 | False |
| default_weight_maxabs_searched | 11 | 3 | fp8_e4m3fn log8 fp7_e3m3 | posit8_es1 fp6_e2m3 int6 log6 nf4 ternary | stress:posit8_es1 | selected:posit8_es1 selected:mxfp8_e4m3 stress:bfp6 stress:posit6_es1 | False |
| minimal_searched | 12 | 3 | log8 fp8_e4m3fn fp7_e3m3 | mxfp8_e4m3 posit8_es1 fp6_e2m3 int6 log6 nf4 ternary | stress:mxfp8_e4m3 stress:posit8_es1 | selected:posit8_es1 selected:mxfp8_e4m3 stress:bfp6 stress:posit6_es1 | False |
