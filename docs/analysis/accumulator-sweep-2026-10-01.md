# Accumulator-width sweep, ResNet18 (lane L8, 2026-10-01/02)

Development evidence: ImageNet screen-1k list, images 0 to 999. One network, one calibration draw. Nothing ran on
held-out images. Summaries `results/summaries/accumulator-sweep-v1/` (written once, 2026-10-02 02:16 +05).
Figures `results/figures/accumulator-sweep-top1-vs-width-v1.{png,pdf}` and `accumulator-sweep-widths-v1.{png,pdf}`.
Protocol `public/experiments/configs/breadth-study/accumulator-sweep-protocol-v1.json` (sha256 `fb9c5624…d765`,
frozen 2026-10-01 22:14:29 +05), addendum 1 (`f17547d3…5b746`, 23:35:07), addendum 2 (`d3703516…9310`,
2026-10-02 02:15:43, deviations and budget outcome) and addendum 3 (2026-10-02, the location refinement rule, two
unretried calls and the L1 configuration re-seals; written after review 1).

**Revision 3 (2026-10-02, after review 2).** No number, run or summary changed. Two sentences changed. "All 21 MAC
nodes have events at W_half" now holds in 7 of the 8 cases; FP8 E5M2 has 20 of 21. The FP6 W=18 channel count of 51
is kept and explained: the store's top level is 7.0, not the 7.5 that review 2 assumed, and a script now backs the
count. The summaries record no hash of the analysis code. `artifacts/accumulator_sweep_v1/review2_check.json` records
it instead. It holds the sha256 of every analysis source and summary file. It also holds a rerun of the analysis into
scratch with the same code: all 7 CSVs are byte-identical, and summary.json is identical apart from the list of
addenda. From now on, `tools/analysis/accumulator_sweep.py` writes `analysis_code_sha256` into every new summary
version. This field is the only change to the analysis code. A scratch rerun after the change gives the same 7 CSVs,
byte for byte. Revision 2 is kept at `artifacts/accumulator_sweep_v1/doc-revisions/accumulator-sweep-2026-10-01.r2.md`.

**Revision 2 (2026-10-02, after review 1, `artifacts/agent_orchestration/handoffs/L8-accsweep-review.md`).** No number
and no run changed; the summaries are those of revision 1. Corrected sentences: the float accumulators (one expected-credit
interval excludes 0: posit8 fp16.x-10); events at W_acc(1.0) (FP6 E2M3: 987 of 1,000 images, not all); the FP6 W=18
mechanism (the clamps are mostly high, not negative); the FP7/FP8 E4M3 "shifted by 16 bits" note (event sets equal,
changed counts not); which layers saturate at collapse (depends on the width); the retries (4 of 6 failed calls, not
all). Added: the location refinement rule that set the brackets, and the re-sealed L1 configuration files (addendum 3).
Revision 1 is kept at `artifacts/accumulator_sweep_v1/doc-revisions/accumulator-sweep-2026-10-01.r1.md`.

## Numbers first

Uniform register `sat.w<W>`: the same signed saturating W-bit register at every MAC node. Widths count the sign bit and
are in units of the product grid. Top-1 is tie-aware expected credit on 1,000 images.

| Case (B2 recipe) | exact Top-1 | published formula | W_cert abs/struct | W_noevent | W_same | W_acc(0.5) | W_acc(1.0) | W_half | cert − W_acc(1.0) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 | 69.05 | 30 | 27/26 | 21 | 21 | 19 | 19 | 17 | 8 |
| INT8, signed activations | 69.06 | 30 | 26/25 | 21 | 21 | 19 | 18 | 16 | 8 |
| INT6 | 65.91 | 26 | 23/22 | 17 | 17 | 16 | 15 | 13 | 8 |
| FP6 E2M3 | 66.53 | 26 | 23/23 | 19 | 18 | 17 | 17 | 14 | 6 |
| FP7 E3M3 | 67.29 | 34 | 31/30 | 26 | 26 | 23 | 23 | 21 | 8 |
| FP8 E4M3FN | 67.60 | 50 | 47/46 | 42 | 42 | 40 | 39 | 37 | 8 |
| posit8 es1 | 69.65 | – | 50/49 | 35 | 35 | 33 | 32 | 30 | 18 |
| FP8 E5M2 | 60.91 | 80 | 75/74 | > 63 | > 63 | > 63 | > 63 | ≥ 63 | – |

- **Accuracy survives 6 to 8 bits below the certified lossless width** for the six minifloat and integer cases (18 bits for
  posit8). Within 2 to 3 bits below W_acc(1.0) it falls to half of the exact arm or less. Example, INT8: 69.05 / 69.03 / 67.60 / 24.02 / 0.18 % at
  W = 21 / 19 / 18 / 17 / 16. The narrowest width with no saturation event on any image lies 4 to 6 bits below the
  certificate (15 for posit8). From that width up, every output equals the exact arm bit for bit.
- Events do not mean errors. At W_acc(1.0), 987 to 1,000 images have a saturation event (all 1,000 in six cases; 987 for
  FP6 E2M3 at W=17), yet Top-1 is within 1 point (INT8 W=19: events on 1,000 images, 33 changed classes,
  −0.02 pp [−0.8, +0.8]).
- **The first layer saturates first in every case.** Just below W_noevent, only `conv1` has events on most images (INT8
  W=20: conv1 on 826 images, two other nodes on 1 image each; FP7 W=25: conv1 on 278). Which layer leads at collapse
  depends on the width:
  - At W_half, all 21 MAC nodes have events in 7 of the 8 cases (FP8 E5M2: 20 of 21). By the share of outputs with a clamp, `conv1` still leads in 6 of the 8
    cases (INT8 0.31, INT8 signed 0.52, FP6 0.42, FP7 0.40, FP8 E4M3 0.40, posit8 0.47). The exceptions are INT6 at W=13
    (`layer4_0_conv1` 0.45, `conv1` 0.32) and FP8 E5M2 at W=63 (`layer3_0_conv1` 0.89; 20 of 21 nodes have events).
  - The second node at W_half is a `layer4` 3×3 convolution in INT8, INT8 signed, FP6 and posit8. In FP7 E3M3 and
    FP8 E4M3FN it is a `layer1` convolution.
  - Two widths further down, at the bracket bottom, the deep 3×3 convolutions clamp on almost every output and overtake
    `conv1`. Examples: INT8 W=15, `layer4_1_conv2` 0.98 and `layer4_0_conv1` 0.96. INT6 W=11, `layer1_1_conv2` 0.995.
    These two were counted from the sealed prediction files; they are not in `node_events.csv`, which starts at W_half.
- **FP8 E5M2 cannot be bracketed.** The widest register the contract admits, 63 bits, gives chance-level Top-1
  (0.14 %, events on 1,000 of 1,000 images). Its certificate is 75 bits.
- **Float accumulators and FP32 are harmless here.** The fp16 rule policies, f21 and the FP32 control are within
  1 point of exact in every case, with no failed image. One expected-credit interval excludes 0, upward (posit8
  fp16.x-10, +0.63 [+0.03, +1.25]); under lowest index it does not. The FP32 control is bit-identical to exact on all 1,000 images
  for INT8, INT8-signed, INT6, FP6 and FP7. It differs in 162 / 7 / 28 outputs for FP8 E4M3 / posit8 / FP8 E5M2.
- **The B2 simulator agrees with exact execution** within ±0.3 points under the lowest-index and expected-credit
  rules for every B2 case (all intervals contain 0). Under the simulator's retained `torch.topk` order, INT6 differs by
  −1.9 pp [−3.4, −0.4].
- Preliminary join (8 ns, core area): an INT8 MAC with a 20-bit accumulator (1,902.6 units) keeps exact accuracy
  (69.15 %). The 16-bit one (1,558.5) gives 0.18 %. The 24-bit one (2,230.5) is lossless on these images.

## What was measured, and on which images

- Network and recipe: ResNet18 with lane L1's repaired quantisation recipe B2 (`default` calibration, one draw).
  It is executed exactly in the code domain by the archived scaled-bridge-v2 engine
  `artifacts/scaled_bridge_v2/implementations/7c6344af732d1bf324c07b285921f63393150cdb3875356c86237d934fdc2f92`
  (`$A/run.sh predict <case> <policy> cuda <start> <stop> --batch 8`, always through `gpu_run.sh`). Every dot
  product is accumulated on the exact product grid 2^-(input shift + weight shift). Arms differ only in the
  accumulator policy (contract 2.1).
- Images: screen-1k list, index 0 to 999, development evidence. The location phase used images 0 to 127. The 1k runs
  reuse those 128-image files as their first tile.
- Location rule (protocol `location_phase`, plus the refinement that addendum 3 writes down):
  - Ladder: `sat.w` at W_cert_abs − 2, then every 3 bits down, stopped once a rung is at or below half of exact.
  - Refinement (`plan.refine`), evaluated again after each finished batch of runs until it asks for nothing. It asks
    for the two widths below the narrowest event-free width and the two widths above the widest at-or-below-half width.
  - Bracket: from the narrowest width with no event on the 128 images, down to the widest width at or below half,
    minus 2.
  - The refinement matters: the ladder rungs alone would give other brackets, e.g. INT8 14–22 instead of 15–21.
  - Replaying the rule reproduces exactly the widths that were measured in all 10 cases
    (`artifacts/accumulator_sweep_v1/addendum3_check.json`). No 1k extension was triggered.
- Policies (contract 2.1):
  - `wide`: exact sum; the reference arm.
  - `control`: sequential binary32 FMA.
  - `sat.w<W>`: signed saturating register, clamped after every add, the same W at every MAC node. The bias is
    added outside the register, in binary64.
  - `sat.struct-<d>`: each node gets its certified structural width minus d.
  - `fp16.x<e>`, `f21.x<e>`: binary16 or 1-8-12 float accumulators, one round-to-nearest-even per multiply-add, with
    the binary point moved by e.
- Float exponent rule (protocol; certificate quantities only):
  - ub = max over nodes of (bit_length(max |prefix| in product units) − product shift).
  - e = 0 when ub ≤ emax and the product step is not below the smallest normal step; otherwise e = emax − ub.
  - This gives fp16.x-11 (INT8), x-10 (INT8 signed, posit8), x-7 (INT6), x-1 (FP6), x-5 (FP7), x-13 (FP8 E4M3) and
    x-27 (FP8 E5M2). f21 needs no shift.
- Certified width of a node = 1 + bit_length(max_c max|a| · Σ_k |w[c,k]|) over the admitted code domain (contract,
  "Width convention").
- Scoring: every accuracy is computed twice, as tie-aware expected credit (1/t when the label is among t tied maxima)
  and as lowest-index Top-1 (the contract tie rule). A failed image (non-finite float accumulator) scores 0 and is
  counted. Exact-arm ties: 35 to 178 of 1,000 images (328 for FP8 E5M2).
- Statistics: each policy is paired against `wide` of the same case on the same images.
  - Expected credit: `tools.analysis.b2_ties.paired`.
  - Lowest index: `tools.analysis.b_stage_balanced_comparisons.paired_outcomes`, which also gives the exact McNemar p.
  - Both use a 95 % pointwise paired bootstrap with 10,000 resamples and seed 20260927.
  - There is no correction for multiple comparisons; 81 non-reference policy rows are reported.
- Derived widths: the definitions are protocol item d, implemented in `tools/accumulator_sweep_v1/score.derived`.
  "From W upward" means the condition holds at W and at every measured wider width. Widths above the bracket top are
  covered by W_noevent: a register with no event equals the exact arm.

## The uniform family in detail

Per width: the number of images with a changed Top-1 class, with a saturation event, and with a changed output
(`policies.csv`).

| Case | Bottom … top of the 1k bracket, as W:changed/event/output |
|---|---|
| INT8 | 15:999/1000/1000, 16:997/1000/1000, 17:758/1000/1000, 18:129/1000/1000, 19:33/1000/997, 20:6/826/354, 21:0/0/0 |
| INT8 signed | 14:999, 15:999, 16:976, 17:438/1000/1000, 18:107/1000/999, 19:34/999/993, 20:11/826/348, 21:0/0/0 |
| INT6 | 11:998, 12:999, 13:983, 14:249/1000/1000, 15:109/1000/998, 16:28/894/376, 17:0/0/0 |
| FP6 E2M3 | 12:999, 13:997, 14:900, 15:290/1000/1000, 16:137/1000/999, 17:66/987/881, 18:0/63/0, 19:0/0/0 |
| FP7 E3M3 | 19:998, 20:998, 21:820, 22:249/1000/1000, 23:89/1000/999, 24:68/997/966, 25:7/278/69, 26:0/0/0 |
| FP8 E4M3FN | 35:999, 36:996, 37:716, 38:250/1000/1000, 39:100/1000/999, 40:55/997/966, 41:6/278/70, 42:0/0/0 |
| posit8 es1 | 28:999, 29:999, 30:989, 31:411/1000/1000, 32:97/1000/999, 33:36/999/991, 34:8/671/329, 35:0/0/0 |
| FP8 E5M2 | 61:1000, 62:999, 63:997 (events and output changes on all 1,000) |

Paired differences to exact (expected credit, pp, 95 % interval) near the transition:

- INT8: W18 −1.45 [−3.0, +0.1]; W19 −0.02 [−0.8, +0.8]; W20 +0.10 [−0.1, +0.4].
- INT8 signed: W17 −20.39 [−23.3, −17.5]; W18 −0.89 [−2.2, +0.4]; W19 −0.29 [−1.0, +0.4]; W20 −0.12 [−0.4, +0.1].
- INT6: W14 −6.78 [−8.7, −5.0]; W15 −0.57 [−1.5, +0.4]; W16 +0.45 [+0.0, +0.9].
- FP6: W15 −8.09 [−10.1, −6.1]; W16 −1.32 [−2.5, −0.1]; W17 −0.02 [−0.8, +0.7].
- FP7: W22 −5.52 [−7.3, −3.7]; W23 −0.09 [−1.1, +0.9]; W25 −0.33 [−0.6, −0.1].
- FP8 E4M3: W38 −5.59 [−7.5, −3.7]; W39 −0.66 [−1.6, +0.3]; W40 +0.27 [−0.5, +1.0].
- posit8: W32 −0.83 [−1.9, +0.3]; W33 −0.31 [−0.9, +0.3]; W34 −0.12 [−0.4, +0.2].

The lowest-index differences follow the same pattern; they are in `policies.csv`.

Notes:

- At FP6 W=18, 63 images have an event and no output changes.
  - All events are in `conv1`: 23,783 high clamps and 6,900 low clamps, on 7,780 output elements.
  - So the clamped sums are mostly positive. They are not mainly negative sums that ReLU zeroes.
  - Why no output changes is not established element by element. A likely reason, from the export constants only: at
    W=18 the high bound maps above the top level of the next store (`relu`, scale 0.358) in 51 of the 64 `conv1`
    channels. That top level is 7.0, not 7.5: the accepted fp6_e2m3 manifest has `nan: true`, so the export's
    codebook has 61 levels, ±56 units of 2^-3. All 51 channels also round to the top level. Against a top level of
    7.5, as review 2 assumed, the count would be 50. Script `artifacts/accumulator_sweep_v1/review2_check.py`,
    output `review2_check.json`. The low bound maps to ≤ 0, which ReLU zeroes, in all 64. A clamped element that ends beyond either
    bound would then store the same code as the exact sum.
- FP7 E3M3 and FP8 E4M3FN at W and W + 16:
  - The event-image sets are identical at every pair from 19/35 to 26/42; e.g. 278 event images at W 25 / 41.
  - At 25/41 and 26/42, the same nodes saturate on every image. At narrower pairs the node sets differ.
  - The changed-class counts are not the same: 7 / 6 at 25/41, 68 / 55 at 24/40, 89 / 100 at 23/39, 820 / 716 at
    21/37.
  - The conv1 part has a likely explanation. In the exports, the E4M3FN input and weight scales are exactly 1/16 of the
    E3M3 ones, and both grids are 2^4 finer. Equal real operands therefore give `conv1` sums exactly 2^16 times larger
    in product units. This was checked on the scales and grid shifts only, not on the image codes.
  - Deeper layers see differently quantised activations, so the two cases diverge there.
- Two expected-credit intervals exclude 0 in the positive direction:
  - INT6 W16: +0.45 [+0.02, +0.89].
  - posit8 fp16.x-10 (next section): +0.63 [+0.03, +1.25].
  - Under lowest index neither does: +0.5 [−0.2, +1.2], McNemar p = 0.27; and +0.3 [−0.5, +1.2], p = 0.65.
  - With 81 uncorrected rows, treat them as noise, not as a gain from saturation or from fp16.

**Per-node family (`sat.struct-<d>`): no 1k result.** The budget rule deferred it (below). The only observations are
on 128 images (INT8 `sat.struct-4`: 66.0 % against 70.3 %, 15 changed classes, events on 128 of 128) and the
32-image gate panels. The protocol's d values are undefined at 1k.

**Recipe sensitivity (original recipe B1).** The 1k runs are deferred. On the 128 location images:

| Case | W_noevent | W_same | W_acc(1.0) | W_half |
|---|---:|---:|---:|---:|
| INT8 B1 | 21 | 21 | 19 | 16 |
| INT8 B2 | 21 | 20 | 20 | 17 |
| FP7 B1 | 27 | 26 | 24 | 22 |
| FP7 B2 | 26 | 25 | 23 | 21 |

The certificates are 26 (INT8 B1) against 27 (INT8 B2), and 31 for FP7 under both recipes. The recipe moves the
transition by at most 1 bit.

## Float accumulators and the FP32 control (1k, against wide)

| Case | control | fp16 rule | f21 |
|---|---|---|---|
| INT8 | 0.00, bit-identical | fp16.x-11 +0.29 [−0.3, +0.9] | +0.38 [−0.1, +0.9] |
| INT8 signed | 0.00, bit-identical | fp16.x-10 −0.15 [−0.7, +0.4] | −0.16 [−0.8, +0.5] |
| INT6 | 0.00, bit-identical | fp16.x-7 −0.07 [−0.8, +0.7] | +0.07 [−0.8, +1.0] |
| FP6 E2M3 | 0.00, bit-identical | fp16.x-1 +0.31 [−0.7, +1.3] | +0.33 [−0.6, +1.2] |
| FP7 E3M3 | 0.00, bit-identical | fp16.x-5 −0.01 [−0.9, +0.8] | −0.00 [−0.8, +0.8] |
| FP8 E4M3FN | −0.08 [−0.3, +0.1], 162 outputs differ | fp16.x-13 −0.17 [−1.0, +0.7] | 0.00 [−0.8, +0.8] |
| posit8 es1 | 0.00, 7 outputs differ | fp16.x-10 +0.63 [+0.0, +1.2] | −0.34 [−0.8, +0.2] |
| FP8 E5M2 | −0.02 [−0.1, +0.0], 28 outputs differ | fp16.x-27 −0.80 [−1.9, +0.3] | −0.69 [−1.8, +0.4] |

(Differences are expected credit in pp with 95 % intervals.)

- No float policy produced a non-finite value on any image, so there were 0 failures.
- fp16 and f21 change every output and 24 to 131 Top-1 classes.
- One of the 16 float rows has an expected-credit interval that excludes 0: posit8 fp16.x-10, +0.63 [+0.03, +1.25],
  upward. Under lowest index it is +0.3 [−0.5, +1.2] (McNemar p = 0.65).
- No other float interval excludes 0 under either rule. With no multiplicity correction, I do not read this one as an
  effect.
- Exponent sensitivity, on 128 images (location phase): fp16 at e ± 3 gave the same Top-1 as at e for INT8, FP7 and
  posit8, except posit8 at x-13 (70.7 % against 69.9 %). f21 with the shifted point gave the same Top-1 as plain f21.
- As contract 2.1 states, plain fp16 overflows on integer codes. The rule's scale exponent is part of every fp16
  result.

## By-product: exact execution against the B2 simulator (1k, wide arm)

| Case | B2 Top-1, retained / lowest-index / expected | exact, lowest / expected | exact − B2, retained order | exact − B2, lowest index | exact − B2, expected | same class (retained / lowest) |
|---|---|---|---|---|---|---|
| INT8 | 69.0 / 68.6 / 69.22 | 68.7 / 69.05 | −0.3 [−1.0, +0.3] | +0.1 [−0.4, +0.6] | −0.17 [−0.6, +0.2] | 982 / 984 |
| INT6 | 66.9 / 65.2 / 65.89 | 65.0 / 65.91 | **−1.9 [−3.4, −0.4]** | −0.2 [−0.8, +0.4] | +0.02 [−0.4, +0.4] | 872 / 970 |
| FP6 E2M3 | 67.3 / 66.8 / 66.66 | 67.1 / 66.53 | −0.2 [−1.6, +1.2] | +0.3 [−0.5, +1.1] | −0.13 [−0.6, +0.4] | 888 / 969 |
| FP7 E3M3 | 68.0 / 67.3 / 67.48 | 67.5 / 67.29 | −0.5 [−1.9, +0.9] | +0.2 [−0.6, +1.0] | −0.19 [−0.8, +0.4] | 892 / 964 |
| FP8 E4M3FN | 68.5 / 67.9 / 67.67 | 67.8 / 67.60 | −0.7 [−2.3, +0.8] | −0.1 [−0.8, +0.7] | −0.06 [−0.7, +0.5] | 878 / 970 |
| posit8 es1 | 69.8 / 69.1 / 69.63 | 69.4 / 69.65 | −0.4 [−1.5, +0.7] | +0.3 [−0.4, +1.0] | +0.02 [−0.5, +0.5] | 946 / 979 |
| FP8 E5M2 | 61.0 / 60.4 / 61.18 | 60.2 / 60.91 | −0.8 [−3.0, +1.4] | −0.2 [−1.1, +0.7] | −0.27 [−0.9, +0.4] | 735 / 944 |
| INT8 signed | 68.7 / – / – | 68.6 / 69.06 | −0.1 [−0.8, +0.7] | – | – | 971 / – |

- Sources:
  - L1 matrix readouts (`artifacts/experiment_b2/matrix/readout/<configuration>.npz`), matched by the configuration
    digest of the adapted export.
  - For INT8 signed, the per-image predictions `artifacts/experiment_b2/predictions/e0b942d0…/`. These have the
    retained order only and no tie readout, so only that column exists.
- Under the tie rule, exact execution reproduces the simulator's accuracy. The same lowest-index class is predicted on
  944 to 984 of 1,000 images. The retained `torch.topk` order picks a different tied class on up to 265 images
  (FP8 E5M2). For INT6, that order alone moves Top-1 by 1.9 points. This extends lane L5's original-recipe finding to
  every B2 case.

## Preliminary join with ICS55 MAC area (integer cases)

- Table: `results/tables/ics55-integer-accwidth-v1.csv`, clock target **8 ns** (every configuration meets it), core
  area in library units.
- "derived" = above W_noevent, equal to the exact arm on these images. 28 and 32 bits lie above every integer
  certificate, so they are lossless by proof and not listed.

| Case | MAC (mult. bits, acc. bits) | core area | Top-1 expected / lowest | source |
|---|---|---:|---|---|
| INT8 | 8, 16 | 1,558.5 | 0.18 / 0.2 | measured |
| INT8 | 8, 20 | 1,902.6 | 69.15 / 68.7 | measured |
| INT8 | 8, 24 | 2,230.5 | 69.05 / 68.7 | derived |
| INT8 signed | 8, 16 | 1,558.5 | 2.37 / 2.4 | measured |
| INT8 signed | 8, 20 | 1,902.6 | 68.94 / 68.6 | measured |
| INT8 signed | 8, 24 | 2,230.5 | 69.06 / 68.6 | derived |
| INT6 | 6, 16 | 1,290.2 | 66.36 / 65.5 | measured |
| INT6 | 6, 20 | 1,615.9 | 65.91 / 65.0 | derived |

On these images, the accuracy-preserving step for INT8 is 20 bits, between 16 (collapse) and 24 (lossless). For INT6
it is 16 bits.

**This join is preliminary.** The ICS55 accumulator-width sweep was never reviewed, and RTL work is on hold. The
semantic differences between `public/generic_rtl/mac/integer_mac.sv` and the engine register (contract 2.1):

| Aspect | RTL `integer_mac` | Engine `sat.w<W>` |
|---|---|---|
| Operands | signed × signed; the table has `W_BITS` = `A_BITS` | weights signed. Activations are **unsigned** codes 0..2^b−1 in INT8/INT6 `default`, so the RTL would need a 9-bit / 7-bit signed activation port, which is not in the table. They are signed in `default_signed`, which matches. |
| Register and clamp | (ACC+1)-bit sum, clamped to [−2^(ACC−1), 2^(ACC−1)−1] after every MAC | the same clamp after every add |
| Start of a dot | `dot_start` adds the first product to 0 | from 0 |
| Bias | added **inside** the register at `dot_end` and clamped again | added outside, in binary64 after scaling, so it never saturates |
| Tap order | whatever the driver streams; the RTL does not fix it | input channel, kernel row, kernel column; padded taps add a zero product |
| Scale and zero point | codes at scale 1, no zero point | the same inside the register; scales applied afterwards in binary64 |
| Weight code range | full signed range | the exported B2 weight levels |

- Saturation events depend on the order of the taps. An RTL schedule with another order can saturate on other
  images.
- The RTL bias add can saturate once more.
- The INT8 `default` row joins accuracy measured with unsigned activations to the area of a signed 8 × 8 MAC. The
  signed-activation case is the like-for-like row, and its widths match within one bit (W_acc(1.0) 18 against 19).

## Published closed-form width (the [Agg24] formulas)

- Integer: ra + rb + ⌈log2 n⌉ + 1 (30 bits for INT8, 26 for INT6).
- Minifloat: 2^ea + ma + 2^eb + mb + ⌈log2 n⌉ − 1 (26 for FP6 E2M3, 34 for FP7 E3M3, 50 for FP8 E4M3, 80 for FP8 E5M2).
- n = 4,608, the formulas as verified in the related-work audit, entry [Agg24]. There is no formula for posit8.

I assume the formula counts the sign bit, as the certificates do. Whether the paper's convention is the same, and
whether its integer r counts a sign bit for unsigned activations, is **still open**. The published widths are 3 to 5
bits above the certificate. The certificate is 4 to 6 bits above W_noevent (posit8: 15), and 6 to 8 bits above W_acc(1.0) (posit8: 18).

## Execution, deviations, refusals and null results

GPU time:

- Lane GPU job time: **6.11 h**, against a protocol plan of 5 h. This is the engine ledger's `wall_seconds` of every
  `predict` since the freeze: location 122 calls, 1.09 h; screen 89 calls, 5.02 h.
- The protocol runs priorities 1 and 2 whatever they cost and priorities 3 to 6 only below 5 h. The total reached
  5.00 h when the 7 frozen cases finished priority 2, so priorities 4 to 6 were not started.
- Deferred, not run: 32 per-node runs (priorities 4 and 6) and 18 original-recipe 1k runs (priority 5). Priority 3 had
  no runs: every hardware width at or below the certificate lies inside the bracket or above W_noevent.

Timing (task step 1):

- The exclusive-mode timing had not been made in round 1, so the protocol's budget used a shared-mode figure. I made
  it afterwards: `gpu_run.sh --exclusive $A/run.sh predict resnet18-int8-default-b2 sat.struct-4 cuda 0 128 --batch 8`
  took 19.70 s of execution for 128 images, **0.154 s per image** at batch 8.
- Shared-mode 1k runs averaged about 0.22 s per image.

Addendum 3 records the location refinement rule (above, "What was measured") that the frozen protocol did not state,
the two unretried calls and the configuration re-seals (below). It changes no run.

Deviations (addendum 2):

- The exclusive timing came after the freeze.
- Runner extension, about 23:41: concurrent-job detection, a check that the location phase is complete, and marker
  tags. The rules that select runs were not changed.
- For about 3 minutes at about 23:35, the lane ran 4 GPU jobs instead of 3.
- Corrected times, from `stat`:
  - `GATES-DONE` was written 2026-10-01 23:35:19; its text says 23:45.
  - Addendum 1 was written 23:35:07; the round-1 handoff says 23:47.
  - The round-1 handoff's "Update 23:50" and "HANDOFF at 23:58" sections were written by 23:36:51.

Failures and limits on the runs:

- 6 engine calls failed transiently with `bridge v2 kernel error 2` (CUDA allocation under contention).
  - 4 were retried and sealed: FP6 `sat.w12` (location), FP7 B1 `wide` (location), INT6 `sat.w17` (1k),
    INT8 signed `sat.w19` (1k).
  - 2 location calls were not relaunched, because the rule no longer asked for them. Neither changes a bracket.
  - FP6 `sat.w9`: a ladder rung below `sat.w12`, which already gave 0.0 %. The ladder had stopped.
  - FP7 B1 `sat.w28`: it lies between `sat.w27` and `sat.w29`, which both have no event on the 128 images, so it
    equals the exact arm by construction.
- Lane L1's export, run by this lane's gates, rewrote three L1 configuration files in place under
  `artifacts/experiment_b2/configurations/`: `b7c945…` (21:16), `e0b942…` (21:31) and `4068da…` (23:28). Their
  content is unchanged: each payload digest equals the file name and the seal (addendum 3).
- `plan.py` was last changed at 23:32, after the freeze, to add the 63-bit cap of addendum 1. All 78 location `sat.w`
  calls of the cases located before then ran on the code loaded at 22:15, and the current rule reproduces their widths.
- The session's permission system refused stopping the round-1 screen runner and starting a second runner beside it.
  The two new cases therefore ran after the first runner finished, with no change to what was run.
- No case or policy was refused by the archive.

Null results:

- FP8 E5M2 has no transition at or below 63 bits.
- No float accumulator shows a difference under both tie rules. The one expected-credit exception, posit8 fp16.x-10
  (+0.63 [+0.03, +1.25]), is not resolved under lowest index.
- The per-node family has no 1k data.

## Limits

- One network (ResNet18), one calibration draw of one recipe (B2 `default`), and 1,000 development images. The widths
  were located on images 0 to 127 of the same list, so the 1k numbers are not independent of the location rule's
  input. A paired difference of about ±0.8 pp is the resolution at 1k.
- Ties: 3.5 to 33 % of exact-arm images have tied maxima, so expected credit and lowest index can rank arms
  differently near the transition.
- One tap order, the engine's. Saturation results depend on it.
- One register width for the whole network. Mixed widths per layer were planned only as `sat.struct-<d>` and are
  deferred.
- The hardware join is preliminary and limited to integer formats at one clock.

## Proposed frozen grid for the held-out confirmation (9,000 images; not run)

Per case: `wide`, `control`, the rule fp16 policy, `f21`, and `sat.w<W>` for every W from W_half to W_noevent of this
1k screen. The INT8 cases also get `sat.w16` (hardware width).

| Case | sat.w widths | runs |
|---|---|---:|
| INT8 B2 | 16–21 | 10 |
| INT8 signed B2 | 16–21 | 10 |
| INT6 B2 | 13–17 | 9 |
| FP6 E2M3 B2 | 14–19 | 10 |
| FP7 E3M3 B2 | 21–26 | 10 |
| FP8 E4M3FN B2 | 37–42 | 10 |
| posit8 es1 B2 | 30–35 | 10 |
| FP8 E5M2 B2 | none (wide, control, fp16.x-27, f21 only) | 4 |
| **total** | | **73** |

GPU time:

- At the exclusive rate (0.154 s per image): 73 × 9,000 × 0.154 s ≈ **28 GPU job hours**.
- At the shared rate seen here (about 0.22 s per image): about 40 h.

A reduced core grid keeps `wide`, `control`, the fp16 rule and the widths from W_acc(1.0) − 1 to W_noevent:

- 56 runs, about 22 h exclusive.
- Its widths: INT8 18–21, INT8 signed 17–21, INT6 14–17, FP6 16–19, FP7 22–26, FP8 E4M3 38–42, posit8 31–35, and
  E5M2 wide/control/fp16.
- With `f21` dropped, because nothing was resolved on the screen.

Before any of it runs, the owner should decide:

- whether the deferred per-node family and the B1 1k check belong in the confirmation;
- which of the two grids to run.
