# Claim-to-evidence ledger (DAC 2027 draft)

Lane P2, written 2026-10-02 (evening, +05); revision 2 the same evening after independent review 1
(`artifacts/agent_orchestration/handoffs/P2-manuscript-skeleton-review.md`), with the files on disk at about 18:05
(+05). This ledger is where every claim the paper could make is traced to its evidence. It runs no experiment and
adds no number: every number below is copied from the cited file and line, read on 2026-10-02. Line references
point to the file versions of that time; documents of running lanes may shift. Paths are relative to the
repository root.

**Data role of every number in this ledger: development evidence.** All quality numbers are on the frozen
screen-1k lists (`imagenet_screen_1k`, `coco_screen_1k`), which were also used to select recipes and cases. No
held-out result and no full-validation result exists. The interim held-out sets (the 9,000 ImageNet images of the
10k list outside the screen, and the 4,000 COCO images outside the screen) are sealed until the hypotheses are
frozen (`docs/decisions/owner-decisions-2026-10-01.md`, decision 2); the YOLOv8n FP32 baseline has already been
scored on all 5,000 COCO images and this must be disclosed (`docs/analysis/evaluation-history-audit-2026-10-01.md`
L91-103).

## Status vocabulary

| Status | Meaning |
|---|---|
| REVIEWED | An independent review file exists in `artifacts/agent_orchestration/handoffs/*-review.md` and its findings were applied (verdict approve or approve_with_fixes, re-reviewed or fixes confirmed). |
| REVIEWED-OPEN | Reviewed, but the review's findings are not yet applied or not re-checked. |
| UNREVIEWED | No review file. |
| INTERIM | The producing lane was still running when this ledger was written; the number may change. The source file's modification time is given. |
| MISSING | No document or summary holds the number yet. |

Caveat on REVIEWED: for the matrix, baseline-repair, reconstruction, detector, accumulator-sweep, gap and Q7
documents, the last review asked for fixes that the lane applied afterwards and no reviewer re-read (for Q7 the
numbers 1.31 / 0.78 / 2.36 come from its addendum 2). P2 review 1 reproduced those numbers from the CSV files, but
"REVIEWED" here means "reviewed, fixes applied by the lane", not "fixes re-reviewed".

Review files consulted (`artifacts/agent_orchestration/handoffs/*-review.md`, state at about 18:05): L1-baseline,
L1-matrix, L2-exact-engine (four reviews, last "approve, no blocking finding"), L3-evaluation-history,
L4-hardware-isoclock, L4-hardware-timing, L5-bridge-gap, L6-recon, L7-detector, L8-accsweep, Q3-collapse-diagnosis
(review 1, 17:35), Q4-activation-attribution (review 1, approve_with_fixes with one blocking finding), Q6-detector-
breadth (reviews 1 and 2), Q7-metrics-resolution, Q8-accumulator-aware-ptq (review 1, 17:51), S1-engine-review, and
lane P3's review of the integer accumulator-width hardware study
(`docs/analysis/ics55-integer-accwidth-review-2026-10-02.md`, verdict "Approve with corrections", L25-40). No review
file exists for Q1 (MobileNet sweep), Q2 (ranking robustness), Q5 (calibration seeds) or P1 (closest prior work).

Novelty verdicts are those of `docs/analysis/related-work-audit-2026-10-01.md` section 2 (C1 L932, C2 L964, C3
L1004, C4 L1049, C5 L1084), section 4 (L1153-1216, including the three conditions of the independent check) and
section 7.3 (L1362-1381).

Abbreviations: "sweep" = `docs/analysis/accumulator-sweep-2026-10-01.md`; "contract" =
`docs/analysis/scaled-bridge-v2-contract-2026-10-01.md`; "engine" = `docs/analysis/scaled-bridge-v2-engine-2026-10-01.md`;
"gap" = `docs/analysis/scaled-bridge-gap-2026-10-01.md`; "matrix" = `docs/analysis/b2-matrix-2026-10-01.md`;
"Q7" = `docs/analysis/quality-metrics-resolution-2026-10-02.md`. W = accumulator register width in bits, sign bit
included, in units of the exact product grid (sweep L31-32). "pp" = percentage points of top-1. Intervals are
pointwise 95 % paired image-bootstrap intervals (10,000 resamples, seed 20260927, no multiplicity correction)
unless stated otherwise.

---

## A. Headline (owner decision 1): failure width per format against the certified width, joined to MAC cost

### A1. Accuracy survives well below the certified lossless width, and collapses within a few bits

- **Wording.** "On ResNet18 with a repaired PTQ recipe, a uniform saturating accumulator keeps top-1 within 1 pp
  of exact execution down to 6 to 8 bits below the per-network certified lossless width for integer and minifloat
  formats (18 bits for posit8 es1), and falls to half of exact or less within 2 to 3 further bits."
- **Evidence.** sweep L36-43 (table), L45-49 (bullet). Per case, W_cert abs / W_acc(1.0) / W_half: INT8 27 / 19 / 17;
  INT8 signed activations 26 / 18 / 16; INT6 23 / 15 / 13; FP6 E2M3 23 / 17 / 14; FP7 E3M3 31 / 23 / 21;
  FP8 E4M3FN 47 / 39 / 37; posit8 es1 50 / 32 / 30; FP8 E5M2 75 / not bracketed (63-bit register gives 0.14 %, L63-64).
  Example, INT8: 69.05 / 69.03 / 67.60 / 24.02 / 0.18 % at W = 21 / 19 / 18 / 17 / 16 (L47-48). At W_acc(1.0),
  987 to 1,000 images have a saturation event; INT8 W=19: −0.02 pp [−0.8, +0.8] against exact, 33 changed classes
  (L49-51). Cross-checked against `results/summaries/accumulator-sweep-v1/widths.csv` (all widths equal).
- **n and role.** 1,000 screen images (ImageNet), development; widths located on images 0-127 of the same list
  (sweep L352: the 1k numbers are not independent of the location rule). One network, one calibration draw, one
  tap order (sweep L348-358). Resolution at 1k about ±0.8 pp (L352).
- **Status.** REVIEWED (L8 review 2: approve_with_fixes, fixes applied in revision 3; sweep L10-17).
- **Still needed.** (1) Second and third network: MobileNetV2/V3 sweep (A6, INTERIM, unreviewed). (2) Held-out
  confirmation on the sealed 9,000 images (sweep L360 proposes the frozen grid; not run). (3) Calibration-draw
  stability of W_acc: Q5 measured seeds on the simulator matrix only, not on the accumulator sweep; its section 8
  (`docs/analysis/b2-calibration-seeds-2026-10-02.md` L247) is a proposal, not run. (4) Other tap orders, tiled /
  two-stage accumulation and wrap-around registers (sweep L355; P1 Part B rows 7-8,
  `docs/analysis/closest-prior-work-accumulators-2026-10-02.md` L198-199). (5) H3 is now defined per accumulator
  type (saturating register only) with a "slack form" option in `docs/analysis/confirmation-protocol-draft2-2026-10-02.md`
  L127-158 (not signed).
- **Novelty (audit, narrowed by P1).** Mechanism (a), "the least-occupied gap that the evidence supports" (audit
  L1163-1176). Condition 1: scope to sub-8-bit non-integer PTQ formats with a measured failure width; the 8-bit
  neighbours [Nat25], [Blu24], [Des23], [Ber22], [Lut24] must be cited and compared (audit L1201-1204). P1's
  full-text reading (`closest-prior-work-accumulators-2026-10-02.md` L23-27, FINAL, unreviewed) finds every
  priority-1 paper 8-bit, training, or both, and none measuring a failure width for sub-8-bit formats under PTQ;
  its bounded search (L324-326) found no closer work. P1's defensible sentence 1 (L269-271) is this row's wording.
  Integer accumulator reduction with guarantees is prior art ([Col23], [Col24a], [Col24b], [Umu25]); the INT8/INT6
  rows are anchors, not the contribution.

### A2. The failure width depends on the format (H3) in absolute bits; below the certificate the slack is 6-8 bits on ResNet18

- **Wording (what the data allow today).** "Measured in absolute register bits, the narrowest lossless-in-accuracy
  width differs by format by more than 2 bits (INT6 15, FP6 E2M3 17, INT8 19, FP7 E3M3 23, posit8 32, FP8 E4M3
  39); measured against each format's certificate, it lies 6 to 8 bits below it for six of the seven bracketed
  cases and 18 bits below for posit8."
- **Evidence.** sweep L36-43 (columns W_acc(1.0) and cert − W_acc(1.0) = 8, 8, 8, 6, 8, 8, 18). Q7 L99-104 (format
  order at a fixed register width: W = 16 INT6 only; W = 17 adds FP6 E2M3; W = 18 adds INT8 signed; W = 23-31 adds
  FP7 E3M3; W = 32 adds posit8 es1 (68.8 %); FP8 E4M3 needs 39, FP8 E5M2 > 63; the leader changes INT6 → FP6 E2M3 →
  INT8).
- **n and role.** As A1.
- **Status.** REVIEWED (sweep, L8); Q7 REVIEWED (Q7 review 2, approve_with_fixes, all findings resolved).
- **MobileNets (INTERIM, unreviewed; A6).** Q1's scratch table (`docs/analysis/accumulator-sweep-mobilenet-2026-10-02.md`
  L15-26, 8 of 10 cases) gives W_acc(1.0) INT8 19 / 19, FP6 E2M3 15 / 17, FP7 E3M3 23 / –, FP8 E4M3 39 / 39
  (MobileNetV2 / MobileNetV3-Large), the same absolute widths as ResNet18 for INT8, FP7 and FP8 E4M3, while the
  certificates are lower (slack 4-6 bits against 6-8 on ResNet18, L29-30).
- **Still needed.** As A1. In addition the interpretation: absolute widths are in product-grid units, so a 2-bit
  difference between formats follows from the code grid alone (protocol draft 2 L148-151 says so). If the
  MobileNet numbers hold, the slack is not constant across networks (4-6 against 6-8) while the absolute width per
  format is nearly network-independent; neither reading has been reviewed. Posit8 (slack 18) is the exception that
  needs an explanation (not yet written in any document). The audit's fallback (L1209-1211) and protocol draft 2's
  slack form of H3 ([OC-6c], L145-151) decide which quantity is claimed; see `outline.md` §"Fallback".
- **Novelty.** As A1. The fixed-width ordering (Q7 L99) is a new presentation of the same data. "The certificate
  against the measured W_acc(1.0) ... which no published bound predicts" (P1 L261-262).

### A3. The per-layer certificate: 2 bits below the exact data-type bound, from the weights (the rest is form and convention)

- **Wording (P1 Part D, sentence 2).** "We certify a lossless accumulator width per layer from the exported weight
  codes and the input code domain, the extension of A2Q's weight-norm bound to minifloat and posit grids; it lies 2
  bits below the exact data-type bound for every ResNet18 case." Do **not** write "3 to 5 bits tighter than the
  published formula" (P1 L288).
- **Evidence.** Convention: contract L185-198. P1 Part C (`docs/analysis/closest-prior-work-accumulators-2026-10-02.md`
  L206-262), recomputed from the stored certificates: the published formula of [Agg24]/[Dam24] counts one sign bit,
  no guard bits, and has the product grid as LSB, the same unit and sign convention as the certificate (L213-215).
  Split of the gap per ResNet18 B2 case (table L238-247): FP6 E2M3 26 → 25 → 25 → 23 (form 1, convention 0,
  substance 2); FP7 E3M3 34 → 33 → 33 → 31 (1 / 0 / 2); FP8 E4M3FN 50 → 49 → 49 → 47 (1 / 0 / 2); FP8 E5M2
  80 → 79 → 77 → 75 (1 / 2 / 2: IEEE Inf/NaN take the top binade); INT8 30 → 28 → 29 → 27 (convention 1, substance
  2); INT8 signed activations 30 → 28 → 28 → 26 (convention 2, substance 2); INT6 26 → 24 → 25 → 23 (1 / 2).
  "The worst-channel Σ|w| is 21 to 26 % of K·max|w| in every case; a factor of 4 is 2 bits" (L249). Worked example
  FP6 E2M3 `layer4_1_conv1`, Σ|w| = 63,683 of 258,048 units (24.7 %), 3,566,248 < 2^22, 23 bits (L225-234); the
  contract's 62,997 / 24.4 % is the B1 max-abs export, same widths (L233-234). Engine L395-402 gives the same B2
  numbers.
- **n and role.** Static (no images). Weights are the B2 export of one calibration draw.
- **Status.** Certificate REVIEWED (L2 reviews 1-4); the convention and the split come from P1 (FINAL, unreviewed).
  The [Agg24] sign-bit question that was open (sweep L286-288; engine L403-405) is settled by P1 from [Dam24] eq. (2)
  (L150-153); [Ugu17] itself was not opened (HAL bot check, L154-155).
- **Still needed.** Print three columns per format as P1 recommends (L257-260): the formula as printed, the exact
  data-type bound in the project's convention, the certificate; claim only the last difference and attribute the
  idea to A2Q for integers. The same comparison for both MobileNets (Q1's table L15-26 lists formula 28 / 24 / 32 /
  48 against certificates 25 / 21 / 29 / 45 for MobileNetV2, INTERIM; not split). The SIRA-style interval bound
  ([Umu25]) as the audit asks (L1044-1047).
- **Novelty (narrowed).** The per-channel weight-ℓ1 bound with post-training minimisation is stated for integers in
  A2Q [Col23] Sec. 3.2 (P1 L19-21, L156-161, L251-255); SIRA [Umu25] does the same with interval arithmetic. What
  remains is "the non-integer transcription of a known integer bound, used as the lossless reference of a PTQ sweep"
  (P1 L295-298). Audit condition 2 (L1204-1207: tighter "for a reason other than convention") is met by the 2
  substance bits only. Not a contribution on its own; it is the method that makes A1 measurable.

### A4. FP32 (binary32 FMA) accumulation of the code-grid products changes no prediction in the admitted cases

- **Wording.** "A sequential binary32 FMA accumulator over the exact code-grid integer products, with scales applied
  afterwards in binary64, reproduces exact execution bit for bit on 1,000 images for INT8, INT8 signed, INT6, FP6
  E2M3 and FP7 E3M3; for FP8 E4M3 / posit8 / FP8 E5M2 the output changes on 162 / 7 / 28 images, the top-1 class
  on 8 / 0 / 1 of them, with no resolved accuracy change."
- **Evidence.** sweep L65-68 and table L196-203 (FP8 E4M3 −0.08 [−0.3, +0.1]; FP8 E5M2 −0.02 [−0.1, +0.0]);
  image counts from `results/summaries/accumulator-sweep-v1/policies.csv`, policy `control`, columns
  `output_changed_images` (162 / 7 / 28) and `changed_top1_images` (8 / 0 / 1);
  gap L85-89 for the original recipe (control − wide 0.0 in all three cases; one-sided 95 % Clopper-Pearson upper
  bound 0.299 % per image); engine L554 (MobileNets, control − wide identical on 32 of 32 images in all 10 cases).
- **n and role.** 1,000 images per case (ResNet18), development; MobileNets 32 images.
- **Status.** REVIEWED.
- **Still needed.** State the definition exactly (gap L265-269: this does not cover the simulator's own FP32
  accumulation over reconstructed operands). Exactness is proven only where the certificate fits binary32 (INT6 and
  FP6 E2M3 on ResNet18, 23 bits; P1 L275-276); for INT8 (27), FP7, FP8 and posit8 the result is empirical (audit
  L1035-1041). MobileNets at 1k (Q1, INTERIM): control = wide except MobileNetV3-L FP8 (3 top-1 changes, −0.03 pp)
  and MobileNetV2 FP8 (+0.05 pp) (`accumulator-sweep-mobilenet-2026-10-02.md` L38-40).
- **Novelty.** Settles a point on which [Agg24] (assumes FP32 adequate) and [Cuy26] (FP32 lossy for MX) disagree
  "in passing" (audit L1026-1031). Small.

### A5. Narrow floating-point accumulators (binary16 with a chosen binary point; a 21-bit 1-8-12 float) are harmless

- **Wording.** "With the binary point placed by a certificate-derived rule, binary16 and a 21-bit float accumulator
  stay within 1 pp of exact in every ResNet18 case; one of 16 intervals excludes zero, upward."
- **Evidence.** sweep L65-68, table L196-203 (e.g. FP6 E2M3 fp16.x-1 +0.31 [−0.7, +1.3]; FP8 E5M2 fp16.x-27 −0.80
  [−1.9, +0.3]; posit8 fp16.x-10 +0.63 [+0.0, +1.2]); 0 non-finite failures (L207). MobileNets (Q1, INTERIM,
  unreviewed): fp16 rule policy and f21 within [−1.0, +0.8] pp of exact, all intervals include 0 except
  MobileNetV3-L INT6 f21 +1.36 [+0.21, +2.53] (stress case at 20.5 %), 0 failed images
  (`accumulator-sweep-mobilenet-2026-10-02.md` L38-40).
- **n and role.** 1,000 images, development.
- **Status.** REVIEWED (ResNet18); MobileNet rows INTERIM.
- **Still needed.** A float failure point does not exist yet: no narrower float (FP12, bf16, operand-format
  accumulation, chunked fp16) was run, so the claim cannot answer [Blu24], [Nat25], [Wan18], [Ber22], [Lut24]
  (survey `artifacts/agent_orchestration/surveys/2026-10-02-related-work-paper-gaps.md` §2 table, row "Narrow FP
  accumulators"). See `gaps.md` G3. P1 names the missing design points (Part B rows 2, 4, 5, 9, 11; L193-202): a
  [Blu24]-style 12-bit M7E4 float with floor rounding and chunk 16, a mantissa sweep `fe5m<k>`, products rounded
  before accumulation, and an LSB-truncating fixed-point register. [Blu24] Table 8 already reports that a 12-bit
  float accumulator without fine-tuning drops ResNet18 from 69.75 to 60.14 % (P1 L28-30, L137-138; cite as their
  number).
- **Novelty.** Narrow FP accumulators at 8 bits are published (audit L1023-1025). Below 8 bits with code-domain
  exactness: not found by the audit or by P1's bounded search (P1 L324-326). P1's defensible sentence 4 (L277-278)
  is the safe wording.

### A6. The same on MobileNetV2 and MobileNetV3-Large (INTERIM)

- **Wording (only after Q1's write-once summary and a review).** "On MobileNetV2 and MobileNetV3-Large the
  saturating register keeps top-1 within 1 pp of exact down to 4 to 6 bits below the certificate, at the same
  absolute width as ResNet18 for INT8 (19 bits) and FP8 E4M3 (39 bits), and fails within 2 to 3 further bits."
- **Evidence.** `docs/analysis/accumulator-sweep-mobilenet-2026-10-02.md` (DRAFT, file time 18:03) section 1, from
  "a scratch run of tools/accumulator_sweep_mn/analysis2.py at 17:48" (L6-7), 8 of 10 cases, table L15-26
  (W_cert / W_noevent / W_acc(1.0) / W_half / slack): MobileNetV2 INT8 25 / 20 / 19 / 17 / 6 (exact 72.55 %), FP6
  E2M3 21 / 17 / 15 / 14 / 6, FP7 E3M3 29 / 24 / 23 / 21 / 6, FP8 E4M3FN 45 / 40 / 39 / 37 / 6; MobileNetV3-L INT8
  24 / 20 / 19 / 17 / 5 (72.00 %), INT6 (stress, 20.50 %) 20 / 16 / 14 / 12 / 6, FP6 E2M3 21 / 17 / 17 / 14 / 4
  (57.86 %), FP8 E4M3FN 45 / 40 / 39 / 37 / 6. Transition 2-3 bits (L28-29). Per-node slack follows fan-in,
  Spearman ρ(slack, log2 K) 0.84-0.96 (L31-34). MobileNetV2 INT6 and MobileNetV3-L FP7 E3M3 were still running.
  Certificates REVIEWED (engine L500-511; depthwise 4 to 5 bits below pointwise, L513-515).
- **n and role.** 1,000 screen images, development; widths located on images 0-127 as for ResNet18.
- **Status.** INTERIM, UNREVIEWED; `results/summaries/accumulator-sweep-mn-v1/` does not exist yet. One
  inconsistency to raise with Q1: the text defines W_half as the "narrowest width above half of the exact arm"
  (L11-13), but the table values follow the ResNet18 definition (widest width at or below half: MobileNetV2 INT8 18
  bits = 50 %, 17 bits = 1 %, W_half 17, L28-29).
- **Still needed.** Q1's write-once summary, the two remaining cases, and an independent review. Protocol draft 2
  leaves open whether MobileNet cases join H3 ([OC-8], `confirmation-protocol-draft2-2026-10-02.md` L154-158); by its
  10 pp floor MobileNetV3-L INT6 and FP6 E2M3 would not.
- **Novelty.** [Agg24] covers MobileNetV2 with closed-form widths; a measured failure width there would be new.

### A7. Join of accuracy to MAC area (integer only, preliminary, conditional)

- **Wording (today, integer only, with its conditions).** "For INT8 on ResNet18 a MAC with a 20-bit saturating
  accumulator keeps exact accuracy (69.15 %, core area 1,902.6 library units at 8 ns) where the 16-bit one
  collapses (0.18 %, 1,558.5); the 24-bit point (2,230.5) is lossless by derivation, not measured." Any such
  sentence must carry P3's conditions (below), or wait for an RTL variant.
- **Evidence.** sweep L72-74 and table L250-257 (16- and 20-bit rows "measured", 24-bit rows "derived": above
  W_noevent, so equal to the exact arm); area from `results/tables/ics55-integer-accwidth-v1.csv`. Area per
  accumulator bit: `docs/analysis/ics55-integer-accwidth-2026-10-01.md` L7 (INT8 82.8 core units per bit at 12 ns;
  "2.8 to 3.1 percent" of the ACC-32 core of each multiplier width) and fit table L144-149. P3's join from the
  reviewed widths (`docs/analysis/ics55-integer-accwidth-review-2026-10-02.md` L16-23, table L155-167): core area
  at 8 ns INT8 2,482 (certified, 27 bits) / 1,973 (no event, 21) / 1,804 (within 1 pp, 19); INT6 1,819 / 1,372 /
  1,223 (15 bits, extrapolated below 16); narrowing from the certified to the 1-pp width saves 679 ± 68 (27.3 %)
  for INT8 and 596 ± 36 (32.8 %) for INT6 in the study's flow, 376 (19.7 %) and 365 (26.3 %) with `-noalumacc`
  (L169-177). "No period saving can be claimed" (L179).
- **n and role.** Accuracy: 1,000 images, development. Hardware: mapped netlists, ideal clock, typical corner, no
  wires, **no power** (accwidth L221-228; isoclock L117-127).
- **Status.** Accuracy REVIEWED. Hardware study reviewed by P3: "Approve with corrections" (L25-40); finding F1
  is **blocking for the paper's join** (L31): the RTL accumulator is not the engine's `sat.w<W>` (unsigned
  activations in 20 of 21 MAC nodes need a W+1-bit signed port that was not synthesised; the bias is added inside
  the register with a second clamp, which also sets the critical path; the tap order is not fixed; L10-14). F3
  (L33): the area per accumulator bit is a flow artefact of Yosys `alumacc` (44.2 / 45.0 / 45.3 units per bit with
  `-noalumacc`, L15). F4 (L34): "one multiplier step buys 2.3 to 3.0 accumulator bits" holds at ACC 32 only (1.4 to
  1.8 at ACC 16, 3.8 to 4.8 at ACC 64). The sweep's own table of RTL-versus-engine differences (sweep L265-273) has
  seven aspects, three of which agree (clamp after every add, start from 0, scale/zero point); four differ
  (operand signedness, bias placement, tap order, weight code range). The only like-for-like row today is INT8
  `default_signed` (P3 L181-189, condition 2).
- **Still needed.** (1) An RTL variant with the bias outside the register and an unsigned activation port, and a
  fixed synthesis recipe (`alumacc` or not) (P3 handoff, open for owner). (2) MAC RTL for the non-integer formats
  (FP6, FP7, FP8, posit8) at certified and failure widths: **the headline's hardware half does not exist for any
  non-integer format**; RTL work is on hold by owner instruction (owner decisions, "Later the same day"). (3)
  Energy from workload-derived activity (publication plan: vectorless energy is not a headline). (4) Placement and
  routing for finalists.
- **Novelty.** Audit condition 3 (L1207-1209): "the headline must be the measured failure point joined to ASIC area
  and energy". Sub-8-bit ASIC cost of non-integer MACs is the open part of C2 (audit L987-990). P1 adds that four of
  six priority-1 papers have ASIC numbers for FP8 datapaths (7, 5, 16, 12 nm), so the integer-only join is "the weak
  half against this set" (P1 L34-35); P1's fallback wording is "integer MAC cost, minifloat accumulator widths"
  (L304-306). Must not write "first ASIC cost of minifloat accumulation" (P1 L289).

---

## B. Methods claim (owner decision 1): simulator against exact code-domain execution

### B1. Under one tie rule, the FP32 fake-quantisation simulator and exact execution agree within the 1k resolution

- **Wording (gap L263-264, supportable form).** "On the 1k development screen, with one deterministic tie rule for
  both, exact execution of the scaled contract and the FP32 QDQ simulator differ by +0.2, +0.1 and +0.2 pp top-1
  (pointwise 95 % intervals [−0.6, +1.0], [−0.9, +1.1], [−0.6, +1.0])" (ResNet18, original max-abs recipe, FP6 E2M3,
  FP6 E3M2, FP7 E3M3). For the repaired recipe, all seven B2 cases agree within ±0.3 pp under the lowest-index and
  expected-credit rules, every interval containing 0 (sweep L69-71, table L222-229).
- **Evidence.** gap L36-37, table L56-60; sweep L222-229.
- **n and role.** 1,000 images per case, development; the gap study's tie audit is post hoc (gap L3).
- **Status.** REVIEWED (L5 review 2; L8 review 2).
- **Still needed.** MobileNets at 1k (only 32-image panels: stored codes differ from B2 in 2.15 % to 23.35 % of codes,
  engine L532-541 (table), L545-549). Q1's 1k comparison is INTERIM and unreviewed
  (`accumulator-sweep-mobilenet-2026-10-02.md` L41-47): exact wide arm minus the B2 simulator, expected credit, 10
  cases from −0.85 [−1.65, −0.08] (MobileNetV2 FP7) to +1.11 [+0.15, +2.10] (MobileNetV3-L FP7); two of ten
  intervals exclude 0, in opposite directions; same lowest-index class on 665 to 989 of 1,000 images ("scatter of a
  few hundred discordant images, not ... a bias"). Both sides are scored with expected credit (L41-42). A factorial decomposition across formats and
  models (audit C4 L1078-1080) exists only for three ResNet18 cases (gap L168-257). P1 finds no paper that measures
  network-level simulator fidelity (P1 L36, L299-301).
- **Must not be claimed** (gap L261-264): "the simulator overstates exact accuracy by 4-5 points"; "the simulator is
  1-2 points optimistic"; do not round to "within ±1 pp".
- **Novelty.** C4 "apparently open in the specific form stated; the integer analogue is published" ([LiM21]);
  confidence low-medium (audit L1051-1053, L1068-1071).

### B2. Ties between equal quantised logits decide a measurable share of reported accuracy

- **Wording (gap L270-271).** "Sub-8-bit logits are frequently tied (18-30 % of images at top-1, 57-76 % at the Top-5
  boundary). Reported accuracies must fix a tie rule, or report the random-tie expectation, because the rule alone
  moves a single configuration's top-1 by up to 2.4 pp at n = 1000."
- **Evidence.** gap L34, L270-271; sweep L69-71 (INT6, retained `torch.topk` order alone: −1.9 pp [−3.4, −0.4]);
  matrix L74-77 (13 to 807, median 181 of 1,000 images tied in cells above 10 %); Q7 L82-88 (expected credit minus
  lowest index: median +0.11 pp, largest 1.81 pp; 11 of 114 cells above 1 pp).
- **n and role.** 1,000 images, development.
- **Status.** REVIEWED.
- **Still needed.** Nothing beyond held-out replication; the tie rule must be stated in the methods (done in
  `methods-draft.md`).
- **Novelty.** Not assessed by the audit (methodological observation). Treat as a methods point, not a contribution.

### B3. The exact engine is conformance-tested per format and per case

- **Wording.** "Every admitted case passes primitive rational conformance of its codebook, an independent
  first-image whole-graph reference, actual-node rational dot checks and CPU/CUDA identity before any panel."
- **Evidence.** engine L23-44 (23 of 23 ResNet18 cases, 17 scalar formats; witness counts 2,474 to 268,666 per
  codebook, L35-39); engine L471-490 (10 of 10 MobileNet cases; 8 of 8 policy gates); contract L65-77 (witness list).
  The fast path used for later runs is bit-identical (S1 review: "no_difference_found" twice and
  "approve_use_with_conditions"; `artifacts/agent_orchestration/handoffs/S1-engine-review.md` L4, L350, L468).
- **Status.** REVIEWED.
- **Novelty.** Infrastructure; supports C3/C4.

---

## C. Supporting claims (audit: supporting results, not the headline)

### C1. The PTQ recipe changes the cross-format ordering

- **Wording (within the audit verdict).** "Repairing the PTQ recipe reorders the formats: the rank correlation with
  the earlier max-abs ordering falls to Spearman 0.49-0.73, chiefly because posit8 moves from collapse to the top
  group; a faithful reconstruction method (AdaRound) does not change the 6-bit INT-versus-FP order."
- **Evidence.** matrix L66-73 (minimal recipe vs v1: Spearman 0.85-0.97; default: 0.49-0.73; 3 to 19 separated
  reversals per comparison not involving a posit format); `docs/analysis/b2-baseline-repair-2026-10-01.md` L31-35
  (INT8: MobileNetV3 4.1 % under v1 max-abs, 71.5 % under B2) and L47-48 (posit8 to 69.8 / 72.1 / 73.4);
  `docs/analysis/b2-reconstruction-baseline-2026-10-01.md` L63-72 (6-bit order unchanged under every arm).
- **n and role.** 1,000 images, 3 classifiers, development; one calibration draw.
- **Status.** matrix REVIEWED (L1-matrix review 2); baseline REVIEWED; reconstruction REVIEWED (L6 review 2).
  Ranking-robustness analysis (Q2): MISSING (`docs/analysis/b2-ranking-robustness-2026-10-02.md` is a DRAFT with no numbers, mtime 2026-10-02 17:48; protocol `b2-rank-protocol-v1.json`).
  Calibration-draw stability of the order (Q5, UNREVIEWED, lane done): `docs/analysis/b2-calibration-seeds-2026-10-02.md`
  L29-36: on MobileNetV2 every separated 8-bit contrast except Log8 − MXFP8 stays separated on all five 400-image
  seeds; on ResNet18 INT8 − MXFP8, Posit8 − MXFP8 and FP7 − INT8 hold; every 6-bit contrast on MobileNetV3-Large and
  several 8-bit ones there (e.g. INT8 − Posit8 −3.33, range −5.06 to −0.37) are unstable. The seeds are disjoint
  subsamples of one calibration list, not independent draws (L3-6).
- **Still needed.** Kendall-τ per recipe with bootstrap intervals and reversals that survive among strong recipes
  (audit L960-962: "C1 is defensible only if reversals survive among the strong recipes, not only between maxabs and
  the rest"); BRECQ or QDrop not run.
- **Novelty.** Partially covered, medium ([Kuz22], [ZhaM26], [Shen24], [Che25], [Agg24], [LiM21]) (audit L936-947).
  Must not be framed as new that "conclusions depend on the recipe".

### C2. Complete cost against multiplier-only cost

- **Wording (today, integer MAC only).** "In the same ICS55 synthesis flow, doubling the INT8 accumulator from 32 to
  64 bits raises core area by a factor of 1.892 (12 ns) to 1.948 (8 ns), while halving the multiplier width from 8
  to 4 bits saves 27.3 % of core area at a 32-bit accumulator."
- **Evidence.** `docs/analysis/ics55-integer-isoclock-2026-10-01.md` L7-17 (bullets "Multiplier effect" and
  "Accumulator effect"); accwidth L5-13. Do **not** cite the minimum periods of `ics55-integer-timing-2026-10-01.md`
  (its own Limits L38-46 disown them).
- **n and role.** Synthesis-level, typical corner, no wires, no power.
- **Status.** isoclock REVIEWED (L4 isoclock review 2); accwidth reviewed by P3 with corrections (F1 blocking for
  the join, F3 flow sensitivity of the per-bit slope, F4 ratio at ACC 32 only; see A7). The isoclock numbers above
  are not affected by F1 (they compare integer MACs with each other), but the same RTL caveats (bias inside the
  register, signed operands) apply to any use beside engine accuracy.
- **Still needed.** Non-integer MACs (decode, alignment, scale/requantisation unit, MX metadata) on the same flow;
  energy; a rank comparison multiplier-only versus complete PE at iso-quality (audit L999-1002). None exists.
- **Novelty.** Direction published ([Agg24], [Dam24], [Des23], [vB23], [Col24b]): "A paper that presents 'complete
  cost changes the ranking' as a discovery would be contradicted" (audit L996-998). Only sub-8-bit ASIC breadth and
  quality-attached Pareto remain.

---

## D. Secondary results

| ID | Claim as it could be worded | Evidence (path, line) and number | n, role | Status | Still needed | Audit / nearest prior work |
|---|---|---|---|---|---|---|
| D1 | YOLOv8n INT8 with the repaired detector recipe is within about 1 mAP point of FP32 | `docs/analysis/b2-detector-2026-10-01.md` L82-89: FP32 39.05; `default` 38.02 (−1.02 [−1.52, −0.78]); `conformant` 38.35 (−0.69 [−1.22, −0.53]); v1 max-abs 31.59 | COCO screen 1k, development | REVIEWED (L7) | Owner choice `default` vs `conformant` (L107-108); detector has no exact-engine run (no SiLU, concat, DFL) | Detection breadth is expected by the plan; no novelty claim |
| D2 | The detector's format order follows the classifier order | `docs/analysis/b2-detector-breadth-2026-10-02.md` L42-44: Spearman 0.92-0.96, Kendall τ-b 0.76-0.86 over 25 formats; the seven chance-level formats and MXFP4 occupy ranks 18-25 and raise the coefficients | COCO screen 1k, development | REVIEWED-OPEN (Q6 review 2 at 17:50, approve_with_fixes, one blocking text item; the lane applied it at 17:55, L312-318, not re-reviewed) | Re-review; rank agreement excluding chance-level formats; use figure `b2-detector-breadth-v2` (v1 lists Q1.6 as a format) | — |
| D3 | Calibration draw moves detector mAP by a fraction of a point | same file L45-56: SD over five disjoint 400-image subsets 0.11-0.51; in four cells every subset is below the full-set value (INT8 and LOG8 under both recipes); posit8 − INT8 ≥ 0 on every subset (+0.01 to +0.65 `default`, +0.23 to +1.29 `conformant`) | COCO screen 1k; calibration subsets of `coco_calibration_2k` | REVIEWED-OPEN (as D2) | Re-review | Audit DAC baseline 6 asks for ≥ 3 seeds (L1129-1131) |
| D4 | In one setting (ResNet18; M = N = 8 or 6 bits at all 21 MAC nodes; B2 per-channel MSE weight scales and unsigned activations; no equalisation; bias outside the register), accumulator-aware PTQ with fixed scales buys one bit below the certificate, while a saturating register reaches 7-8 bits below it. Not a statement about AXE in general. | `docs/analysis/accumulator-aware-ptq-2026-10-02.md` table L24-27 (narrowest P within 1 pp: INT8 AXE 25, naive 25, `sat.wP` 19, exploratory rescale 20, axers 19‡; INT6 21, 21, 15, 17, 17); ‡ "at exact equality with the threshold ... one image's credit decides 19 against 20" (L31, L36-40); L89 (INT8 P = 24: AXE − B2 −4.51 (−6.44 to −2.63)); conditions L105-113; bias excluded from the register L287-291 | ImageNet screen 1k, ResNet18 INT8/INT6 only, development; AXE arms are simulator results, `sat.wP` exact; `sat.wP` rows above the event-free width are derived, not measured (L80) | REVIEWED-OPEN (Q8 review 1 at 17:51, approve_with_fixes, three must-fix wording items; the lane revised the document at 18:03, round 3, not re-reviewed) | Re-review; MobileNet and W4A8; (M, N) chosen per P as published AXE does; the exploratory arms (rescale, axers) were added post hoc (addendum 2) | Answers audit DAC baseline 4 (A2Q/A2Q+/AXE at equal width, audit L1122-1126); [Col24b] |
| D5 | The B2 bias-correction pass, run with quantised activations, is behind the 4-bit-learned, 5-bit and wide-exponent collapses; at 4-bit nearest weights the faithful (DFQ App. D) correction itself fails | `docs/analysis/b2-collapse-diagnosis-2026-10-02.md` short answers L41-84 (revision r5): mode 1 "no code defect; three regimes" (L43-67): (i) W4 nearest weights: property of the method; (ii) learned W4, W5 and the wide-exponent formats: fragile design (B2's quantised-activation pass is the cause; e.g. ResNet18 W4/A4 learned 45.7 against 3.0, +42.7 [+40.2, +45.2]); (iii) MobileNetV3-L integers: the same choice helps (INT6 21.0 against 4.4). Mode 3: removing bias correction recovers 88 to 167 % of the loss (L75) | ImageNet screen 1k, development | REVIEWED-OPEN (Q3 review 1 at 17:35, approve_with_fixes; r5 applies it, not re-reviewed; the review asks a follow-up reviewer to check the three regimes, review file L89-92) | Re-review | [Nag19] bias correction |
| D6 | MobileNetV3 INT8 activation loss sits in the stem; a stem repair closes it | `docs/analysis/b2-activation-attribution-2026-10-02.md` L6-21: stem 3.10 of 3.40 pp; repair r3b 74.65 (+3.05 [1.1, 5.0] over default; −0.35 [−1.75, 1.05] to FP32); cost +1 accumulator bit in 10 layers. The Q4 review re-measured r3b with sound sampling: 76.13, +4.53 [2.63, 6.42] over default, above FP32 75.00 ("selection on the screen") (`artifacts/agent_orchestration/handoffs/Q4-activation-attribution-review.md` L30, L72) | ImageNet screen 1k; repairs chosen and evaluated on the same screen (optimistic) | REVIEWED-OPEN with a **blocking** finding (Q4 review 1, B1: the `pcf` per-channel scales were fitted on border pixels only; every claim resting on `pcf` arms, r3b included, is "NOT usable as written" and needs re-measurement under a new addendum, review L9-11, B1 at L15); SQNR claims are must-fix (review L55-60) | Re-measurement and re-review; whether any repair enters the recipe is an owner decision | [Nag19]; [Kri18] |
| D7 | Agreement with FP32 resolves format differences that top-1 cannot; resolution follows 1/√n | Q7 L21-29 (61 vs 2 pairs separated; r = 0.989 with top-1 drop); L30-52 (near-equal format pairs half-width 1.31 pp at 1k, accumulator arms 0.78); L56-60 (minimum detectable difference 2.36 pp at 1k, 0.79 pp at 9k for format pairs) | Re-reads stored 1k records | REVIEWED | — | Methods only |
| D8 | A faithful AdaRound reproduces the published 4-bit-weight result and rescues W4/A8, not W4/A4 | `docs/analysis/b2-reconstruction-baseline-2026-10-01.md` L55-62 (W4/A8 learned, no bias correction: 70.4 / 71.5 / 69.35; W4/A4 best 24.8 / 27.2 / 0.1); L73-75 (faithfulness: drops 0.6 / 1.3 / 0.8 vs published 1.08) | ImageNet screen 1k; rounding learned on 1,024 calibration images | REVIEWED | BRECQ/QDrop not run | Audit DAC baseline 1 ([Nag20], [Li21]) |
| D9 | INT8 anchors are within about one point of FP32 except MobileNetV3 | `docs/analysis/b2-baseline-repair-2026-10-01.md` L31-35 (B2 − FP32: ResNet18 −1.1 [−2.0, −0.2]; MobileNetV2 −0.1 [−0.8, +0.7]; MobileNetV3 −3.5 [−5.4, −1.5]); reconstruction L50-54 (MobileNetV3 L-bc −2.05 below FP32) | ImageNet screen 1k | REVIEWED | MobileNetV3 anchor remains weak (see D6) | Audit L953-956 ([Nag19] reading of MobileNet collapse) |
| D10 | Plain PTQ does not hold below 6 bits on these models | matrix L53-57 (best 5-bit cell 56.9; every 4-bit cell ≤ 2.2 under the default recipe) | ImageNet screen 1k | REVIEWED | — | Limitation, not a contribution |
| D11 | Evaluation hygiene: the interim held-out sets are untouched by candidates | `docs/analysis/evaluation-history-audit-2026-10-01.md` L91-103 | Record-based audit | REVIEWED (L3-evaluation-history review) | Repeat before unsealing | — |
| D12 | Calibration draw moves classifier accuracy little at 8 bits on ResNet18 and MobileNetV2, a lot on MobileNetV3-Large and at 6 bits; a 2,000-image calibration is not converged for some cells | `docs/analysis/b2-calibration-seeds-2026-10-02.md` L16-21 (seed SD of top-1 over five 400-image seeds: ResNet18 8-bit 0.23-0.64 pp, MobileNetV2 8-bit 0.09-0.47, MobileNetV3-L INT8 1.32, FP8 E4M3 1.51; at 6 bits on MobileNetV3-L INT6 6.9, FP6 E3M2 11.3; median over 54 cells 0.69 pp against an image-sampling SD of 1.39 pp at n = 1,000); L44-47 (MobileNetV3-L INT6 47.9 % with 32 calibration images against 21.0 with 2,000) | ImageNet screen 1k; seeds are disjoint 400-image subsamples of the one calibration list, not independent draws (L3-6); simulator (B2) only, no accumulator sweep | UNREVIEWED (lane Q5 done) | Review; W_acc on a second calibration draw (proposal L247, not run) | Audit DAC baseline 6 (≥ 3 seeds, L1129-1131) |

---

## E. Claims the paper must not make (from the sources)

| Do not say | Why | Source |
|---|---|---|
| "The simulator overstates exact accuracy by 4-5 points" / "is 1-2 points optimistic" | Tie-breaking artefact on this panel | gap L261-263 |
| "Complete cost changes the format ranking" as a discovery | Published direction | audit L996-998 |
| "FP32 accumulation is exact for minifloats" in general | Proven only where the certificate is ≤ 24 bits (E2M3); empirical elsewhere | audit L1035-1041; sweep L65-68 |
| Any held-out, full-validation, or 50k number | None exists | owner decisions; confirmation draft §9 |
| A first-class hardware result for non-integer formats | No RTL exists | owner decisions "Later the same day"; A7 |
| Mixed-precision or per-layer format selection as a contribution | Out of scope (decision 3); crowded (C5) | owner decisions; audit L1084-1104 |
| Anything about the private downstream architecture | Public/private rule | `docs/publications/README.md` L84 |
| "The certificate is 3 (to 5) bits tighter than the published formula" | 1 bit is formula form, 0-2 bits format convention, 2 bits substance | P1 L257-262, L288 |
| "First lossless / overflow-free accumulator width" or "first weight-dependent width" | A2Q [Col23] eq. 13 with post-training minimisation; [Umu25]; Kulisch | P1 L287 |
| "First study of narrow accumulators for low-precision floating-point inference" | [Nat25], [Blu24], [Sak19], [Wan18] | P1 L286 |
| "First ASIC cost of minifloat accumulation" | [Des23] 16 nm, [Ber22] 12 nm, [Lut24] 5 nm, [Nat25] ASAP7, [Joh18] 28 nm | P1 L289 |
| "FP32 accumulation is required for FP8" or "12-bit accumulators suffice" as general statements | Contradicted by [Blu24] (fine-tuned 12-bit works; zero-shot does not) | P1 L290 |
| The integer MAC area at a given W as the cost of the engine's `sat.w<W>` register, without conditions | RTL differs (unsigned activations, bias in register, tap order); per-bit slope is flow-dependent | P3 review L25-34, L181-189 |
| "Accumulator-aware PTQ costs N bits" as a statement about AXE in general | Holds for fixed M = N, B2 scales, no equalisation, bias outside | Q8 L105-113 |
| Any MobileNet accumulator width as final | Q1 numbers are a scratch run, 8 of 10 cases, unreviewed | Q1 L6-7 |
