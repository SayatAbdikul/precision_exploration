PROPOSED - awaiting owner sign-off

# Confirmation protocol, pre-registration draft 2 (2026-10-02, revision 3)

Status: draft 2, not signed, not frozen. Written by lane P4. It replaces the first draft
(`docs/analysis/confirmation-protocol-draft-2026-10-01.md`, not edited) as the text to be signed. Machine-readable twin:
`public/experiments/configs/breadth-study/confirmation-protocol-v1-draft2.json` (status "DRAFT - not signed"; owner
choices are null fields). The owner's choices are collected in `docs/decisions/decision-package-2026-10-02.md` and
marked **[OC-n]** here with that package's item number. Once every [OC-n] is filled and the INTERIM inputs of section 15
are final, a third party can execute this text without a further choice. Numbers are cited as `file:line`. Where two
sources disagree, section 16 says so; nothing was chosen silently. No held-out image, held-out prediction file or
held-out outcome was opened; the only files read about the held-out sets are the manifests.

Revision 1 (2026-10-02, lane P4 agent r2) resolves review 1
(`artifacts/agent_orchestration/handoffs/P4-confirmation-protocol-review.md`): H3 multiplicity rebuilt (two sequences
per case at half the case level each; the 0.5 pp claims secondary or a separate share) [B1]; H1 power restated at the
adjusted level, pair rule tied to the screen power [B2]; H1 pair rule made executable on lane Q2's `reversals.csv` [B3];
Bonferroni instead of Holm, every test form and level defined, MobileNet grid tied to the ResNet18 choice [B4];
simulator cell counts corrected (36 / 42 cells for S-A) [B5]; per-node arm restored to the sweep's 32 runs, halving
marked [B6]; engine base archive corrected to `1f75c923` [B7]; percentile coverage at the adjusted levels computed
exactly (section 5.2) and the resample count raised; INTERIM inputs re-read at 18:15-18:30; M1 offered as a choice;
caps and floors marked as proposals.

Revision 2 (2026-10-02, lane P4 agent r3; file written 18:57): the /1.4 rule checked at the exact nominal levels (5.2; m = 1.5
at C = 15); lane Q1's sweep finished (summaries written once 18:36,
`results/summaries/accumulator-sweep-mn-v1/`); the MobileNet grid (9.3), the H3 floor list and the M1 power-entry list
are now derived from those files by `artifacts/confirmation_protocol_v1/design_v3.py` -> `design-v3.json` (still
INTERIM until Q1's document is reviewed). Corrected: the sign label of Q1's simulator comparison (it is exact minus
simulator); the confirmatory 0.5 pp arm list, now selected at the calibrated level like its bounds (INT6 `sat.w16`
leaves the equal-shares list); H1 criterion (B) bounds at nominal /1.4; Q1 and Q2 line citations. Added: Q1's own
60-run MobileNet proposal as an option and a disagreement (16.14-16.15); a third R1 option for H1 from lane Q2's
interim finding; the M1 power entry on Q1's intervals. Package (`decision-package-2026-10-02.md`) and JSON twin
rewritten to this revision.

Revision 3 (2026-10-02, lane P4 agent r4, started 19:12) resolves review 2: coverage recomputed over the whole null
region in both directions (5.2; the divisor m = 1.4 / 1.5 is withdrawn, nominal levels are set per direction and
target) [blocking 1]; H1 power stated for criteria (A) and (B), screen rule by projected joint power [2]; detector
margin defined for every format, FP7 included [3]; family D defined under every [OC-5] option [4]; [OC-8] split into
inclusion, grid rule and failing cases [5]; INTERIM inputs re-read 19:09-19:25; K4 seed and batch fixed; P4 values
labelled as proposals; claim (a) of [OC-6c] made error-controlled; ledger at the S1 (19:11) and Q1 r4 (18:58) rates.

## 0. What changed since draft 1

| Draft 1 | Draft 2 | Why |
|---|---|---|
| Confirmation = ImageNet 40k complement | ImageNet 9k remainder of the 10k list and COCO 4k remainder | owner decision 2 (`docs/decisions/owner-decisions-2026-10-01.md:18`) |
| Margins without measured resolution | Margins from lane Q7's measurements | `quality-metrics-resolution-2026-10-02.md:348-373` |
| No tie rule | Classifier expected credit primary; detector [OC-4] | Q7 `:328-337` |
| Contrasts "at most 12", unnamed | Named families, mechanical H1 pair rule | draft 1 fact-check item 1 (`:101`) |
| H3 one width axis over three accumulator types | H3 per type (uniform saturating register); two fixed sequences per case | fact-check items 2-3; Q7 `:376-377` |
| Point versus interval undefined | Every criterion on interval bounds at a stated level | fact-check item 4 |
| Holm and max-t | Bonferroni between and within families, fixed sequences inside an H3 case | fact-check item 5; review 1 B4 |
| Detector without hypothesis | Detector family D, secondary [OC-6e] | fact-check item 6 |
| H1, H2, H3 a third of alpha each | H1, H3 and the new family M1 (simulator against exact) [OC-6d, OC-6f]; H2 no alpha [OC-7] | fact-check item 7; RTL on hold (`owner-decisions-2026-10-01.md:50-56`); M1 is new in draft 2 |
| No ledger, no engine requirement | Sections 10 and 11 | task; `S1-engine-review.md:677-729` |

## 1. Scope and evidence status

Everything measured so far (128- to 1,000-image panels, the B2 matrix, the accumulator sweeps, the detector studies)
is development evidence on the screen lists. The confirmation is the first use of images that no candidate, recipe,
grid or margin choice has touched. Scope (owner decision 3, `owner-decisions-2026-10-01.md:19`): post-training
quantisation, CNN workloads (ResNet18, MobileNetV2, MobileNetV3-Large, YOLOv8n), no mixed precision. Governance items
that change no run but fix the wording of results: proposals P1, P3 and P5 [OC-11]; the class of the scaled bridge
[OC-15]; unsigned activation codes [OC-17]; one PDK [OC-18] (`proposed-decision-updates-2026-10-01.md:7-47,49-59`).

## 2. Data roles

| Role | ImageNet | COCO |
|---|---|---|
| Calibration | `data/manifests/calibration/imagenet1k_train_2k.tsv` (2,000) | `coco2017_train_2k.tsv` (2,000) |
| Development | screen `imagenet1k_val_1k.tsv` (1,000) | screen `coco2017_val_1k.tsv` (1,000) |
| Confirmation (held out) | rows of `imagenet1k_val_10k.tsv` whose sha256 is not in the 1k list: 9,000 | rows of `coco2017_val_5k.tsv` whose sha256 is not in the 1k list: 4,000 |
| Later, own protocol | 40k complement of the 10k list in the 50k set [OC-19] | none |

Manifest facts (`artifacts/confirmation_protocol_v1/complement_digest.py` -> `complement-digests-v1.json`, manifests
only; reproduced by review 1 and by `complement_digest_v2.py --compare`, 4 of 4 digests equal): all six manifests
equal their sha256 in `data/manifests/index.json`; the 1k lists nest in the 10k / 5k lists (1,000 of 1,000); the
ImageNet remainder has 9,000 unique sha256, 9 per class in all 1,000 classes; the COCO remainder 4,000 unique; the
calibration lists share 0 hashes with the evaluation lists.
- ImageNet 9k digest (sha256 of the sorted sha256 column, newline-joined, trailing newline):
  `d47669bbf06d04c7442baa4a7411a9687d8c5d392418f9a474ac17feca161574`; rows in manifest order
  `03bb87539016fda88ceab1936417cba937ce69e0a36599f9a5a2e7ce0b7e1e3e`.
- COCO 4k digest `4fd234e061c1a69f30edf9d5f1c992c136b08a6c4dc9baa7c2252d703e418638`; rows in order
  `7abc4519c263b2d53457669206e7cac8d6e234d7eddffdb6a4bea12acb9d65af`.

Every held-out run iterates the complement rows in full-manifest order; never an image folder, a dataset yaml's `val`
entry, or Ultralytics `val`.

## 3. Audit that must pass before a complement is called untouched

1. Rerun the evaluation-history audit into a NEW file (`results/summaries/evaluation-history-audit-v1.json` exists;
   the documented command, `evaluation-history-audit-2026-10-01.md:113`, would overwrite it):
   `.venv/bin/python tools/analysis/evaluation_history_audit.py --out results/summaries/evaluation-history-audit-v2.json`
   and `.venv/bin/python -m pytest -q tests/unit/test_evaluation_history_audit.py`.
2. Pass (`evaluation-history-audit-2026-10-01.md:104-110`): ImageNet remainder digest in the evaluation unions is
   `e3b0c442...` (empty); its `index_mentions` are 9,000 and nothing else; no candidate run record or other mention of
   the COCO remainder; the YOLOv8n FP32 baseline records on all 5,000 images are the only COCO remainder use.
3. `python3 artifacts/confirmation_protocol_v1/complement_digest_v2.py --out artifacts/confirmation_protocol_v1/complement-digests-audit.json --compare artifacts/confirmation_protocol_v1/complement-digests-v1.json`
   must print "EQUAL (4 of 4)" and exit 0.
4. Disclose, not resolve by assumption: YOLOv8n FP32 scored all 5,000 COCO images (`owner-decisions-2026-10-01.md:33`);
   "untouched" means "on this machine, as far as its records show"; the phase 3 integer screening ran on a second
   machine whose records are absent (`evaluation-history-audit-2026-10-01.md:107`); public-set tuning of pretrained
   weights is not checked.
5. COCO consequence: detector contrasts between quantised configurations are primary within family D; "minus FP32" is
   reported with the disclosure.

If step 2 or 3 fails, no held-out run starts; the finding goes to the owner.

## 4. Hypotheses (operational definitions)

Common definitions. Top-1 = tie-aware expected credit (1/t when the label is among t tied maxima) on stored outputs;
lowest-index top-1 is secondary. "Wide" = the exact engine's exact-sum arm of the same case. A "case" = (network,
format, frozen B2 recipe), exported once from the frozen calibration list. Widths count the sign bit, in units of the
case's product grid (`accumulator-sweep-2026-10-01.md:94-110`). Width definitions of the sweep protocol
(`public/experiments/configs/breadth-study/accumulator-sweep-protocol-v1.json`, `definitions_uniform_family`), always
on the 1k screen: W_cert = certified absolute width; W_noevent = narrowest measured W from which upward no saturation
event occurs on any image; W_acc(d) = narrowest W from which upward expected-credit top-1 >= wide - d (a point
estimate); W_half = widest W at which top-1 <= half of wide. "From W upward" = at W and every measured wider width.

### H1, the recipe changes the ordering (confirmatory, family H1)

- Recipes: R2 = the frozen B2 recipe under `R_bits(6)` (`b2-matrix-2026-10-01.md:58,450`). R1 **[OC-6a]**: (i) B2
  `minimal`; (ii) the sealed v1 max-abs recipe; (iii) B2 `default` without its bias correction (Q2 view
  `default_no_bias_correction`). Option (iii) is added because lane Q2's interim analysis found that all 7 non-chance
  interval reversals among its strong recipes so far are `default` against `default_no_bias_correction` at 6 bits
  (Q2 handoff, 18:38 entry: partial analysis of 115 cells at 17:10; INTERIM); it overlaps the bias-correction decision [OC-12].
- Pair list, filled before signing from lane Q2's final `results/summaries/b2-rank-v1/reversals.csv` (INTERIM; file
  sha256 recorded in the signed JSON). Columns as written by `tools/experiment_b2_rank/analysis.py:269-287` (model,
  class, x, y, view_a, view_b, d_a, d_a_low, d_a_high, d_b, d_b_low, d_b_high, point_reversal, interval_reversal,
  chance_pair, category; view_a precedes view_b alphabetically; d = top1(x) - top1(y)).
  1. Rows of one network whose view pair is {V(R2), V(R1)}: V(R2) = `default` for width classes "8" and "6" (Q2's
     `default` view uses the intrinsic arm for block formats and reproduces `proposal-d4.csv`, Q2 handoff :32-34,67) and
     `minimal` for class "le5"; V(R1) = `minimal`, `v1_maxabs` or `default_no_bias_correction` per [OC-6a]. Under
     R1 = `minimal` class "le5" has no pair (V(R1) = V(R2)). FP7 E3M3 is alone in class "7", which is never paired
     (`analysis.py:40,251`).
  2. `reversals.csv` holds only rows with `point_reversal` = true (both |d| > 1.0 pp, opposite signs; `:276-278`);
     keep `chance_pair` = false (no format below 1 % in either view; `:265`).
  3. Pool **[OC-6g]**: (i) both formats in the promoted set [OC-1]; (ii) any formats of the B2 matrix.
  4. Screen power [threshold 0.8 is a lane P4 proposal, OC-6g]: for each view v, se9_v = (high - low) / (2 x 1.96 x 3)
     (Q2's 95 % percentile interval at n = 1,000 scaled to 9,000) and the projected component power
     P_v = Phi((|d_v| - c_v) / se9_v), with c_v = max(delta, z_A x se9_v) under criterion (A) and c_v = delta +
     z_B x se9_v under criterion (B), z_A and z_B at k = 6 (the cap; conservative when fewer pairs are taken). Keep
     pairs whose projected joint power P_a x P_b is at least 0.8. Screen values are inflated by the selection, so the
     true power is lower than projected.
  5. Rank by projected joint power (descending); ties by network (ResNet18, MobileNetV2, MobileNetV3-Large), class
     ("8", "6", "le5"), then x, then y. Take pairs in order, skipping a pair whose network already has 2 [proposed cap,
     OC-6g], up to 6 [proposed cap, OC-6g]. k = the number taken. Orientation as in Q2: D_R = top1(x, R) - top1(y, R).
- Level: a_pair = a_H1 / k, two-sided. Criterion (A) uses the two-sided (1 - a_pair) percentile interval without
  calibration (k = 6, equal thirds: z = 2.99): its conjunct |D_hat| >= delta needs at least 90 net discordant images
  of 9,000, whose probability under theta = 0 is below 3e-8 at every discordance up to 3 % (`design-v4.json`), where
  the bare zero-margin bound can err (5.2); above 1.5 % that bound errs at most 0.98 alpha (`coverage-v1.json`).
  Criterion (B) bounds against +-delta are 'beyond' tests (5.2); their nominal level is the target (k = 6: z = 2.99).
- Criterion **[OC-6b]**: (A) both intervals exclude 0, in opposite directions, and both point estimates satisfy
  |D| >= delta; or (B) both intervals lie entirely beyond delta in opposite directions (D_R1 > +delta and D_R2 <
  -delta, or the reverse). Intersection-union: no further split.
- Outcome: H1 "supported" if at least one pair meets the criterion; otherwise "not shown" (not evidence of absence).
  Reported per pair, descriptive: "reversal excluded" if under at least one recipe the (1 - a_pair) interval lies
  inside (-delta, +delta), or both intervals exclude 0 on the same side; "open" otherwise. The list is never widened.
- Power (`design-v4.json` `H1_power_equal_thirds`; se9 = sqrt(p_d / 9,000)): under criterion (A) a component claims
  when D_hat >= max(delta, z x se9). At 7.1 % discordance the point condition binds (z x se9 = 0.84 < 1.0), so at a
  true |D| of 1.0 pp a component's power is at most 0.5 at any image count; revision 1's 10,431 images were the power
  to exclude 0 only. Joint 80 % power for both components needs a true |D| of at least 1.35 / 1.73 / 2.13 pp at 7.1 /
  15 / 22.6 % discordance (k = 6; k = 3: 1.35 / 1.64 / 2.02). Criterion (B) needs 2.19 / 2.73 / 3.13 pp (k = 6, z 2.99;
  `design-v4.json` lists 3.12 at the uncapped z 2.98) and 2.13 / 2.64 / 3.02 at k = 3. Pairs chosen for |d| > 1 pp are
  not Q7's near-equal pairs (all-contrast median discordance 22.6 %, Q7 `:357-359`), so H1 can confirm only reversals
  of about 1.4-2.1 pp (A) or 2.2-3.1 pp (B) or more.

### H2, complete cost against multiplier-only cost (no alpha while no cost exists) [OC-7]

New RTL work is on hold (`owner-decisions-2026-10-01.md:50-56`); no MAC cost exists for non-integer formats. Proposed:
the quality side runs now as secondary equivalence contrasts E: every pair of promoted formats of equal nominal width
on the same network, TOST at delta = 1.0 pp with the pointwise two-sided 90 % percentile interval (no multiplicity
adjustment). The cost criterion is frozen now: "H2 holds for a pair (X, Y) equivalent under E if X has the lower
multiplier-only area and Y the lower complete-MAC area (decode, scale, accumulate at the certified width) at every
pipeline setting evaluated, from one PDK, with placement-seed spread reported." Evaluated later in an addendum, without
alpha (cost has no sampling interval). Alternative: H2 removed from this paper.

### H3, the accumulator failure width depends on the format (confirmatory, family H3)

- Accumulator type: the uniform signed saturating register `sat.w<W>` (clamp after every add, same W at every MAC node,
  bias added outside in binary64; `accumulator-sweep-2026-10-01.md:96-99`). Float accumulators (fp16 rule, `f21`) and
  the FP32 `control` are secondary non-inferiority contrasts against wide at margin -1.0 pp (pointwise one-sided 95 %,
  calibrated nominal 0.0432, z = 1.71; 5.2), not on the axis; those on the 0.5 pp list are also bounded at -0.5 pp.
- Cases: the 7 bracketed ResNet18 B2 cases (INT8, INT8 signed activations, INT6, FP6 E2M3, FP7 E3M3, FP8 E4M3FN,
  posit8 es1); FP8 E5M2 cannot be bracketed (`accumulator-sweep-2026-10-01.md:43`) and is descriptive; MobileNet cases
  per [OC-8]. C = number of H3 cases; a_case = a_H3 / C.
- Contrast at width W: Delta(W) = top1(sat.wW) - top1(wide), 9k, delta_acc = 1.0 pp. The bounds below are the ends of
  one percentile interval per width, each end at its own calibrated nominal level (5.2): the lower end (sequence N, a
  'toward' test) at nu_toward(a_case / 2), the upper end (sequence I, a 'beyond' test) at nu_beyond(a_case / 2).
  C = 7: z = 3.22 (N) and 3.05 (I); C = 15: 3.48 and 3.26 (`design-v4.json` `levels`).
- Sequence N (non-inferiority, one-sided a_case / 2): from W_noevent(1k) down through the grid widths; W passes if the
  lower bound > -1.0 pp; stop at the first width that does not pass. k*(c) = narrowest width of the uninterrupted
  passing run; "above the grid" if W_noevent(1k) does not pass.
- Sequence I (inferiority, one-sided a_case / 2): from the narrowest grid width upward; W passes if the upper bound
  < -1.0 pp; stop at the first width that does not pass. f(c) = widest width of the uninterrupted passing run;
  "open below" if the narrowest grid width does not pass.
- A width whose 9k run has 0 saturation events equals wide bit for bit; its interval is [0, 0], it passes N without a
  statistical test and is reported "identical". Error control: N and I each hold their error at a_case / 2 (fixed
  sequences), so a case at a_case and the family at a_H3 (Bonferroni over cases).
- Slack: s*(c) = W_cert_abs(c) - k*(c), also against the structural certificate and the published closed-form width
  ([Agg24]; `accumulator-sweep-2026-10-01.md:34-43,281-289`).
- Claim **[OC-6c]**: (a) draft 1 form, restated so that it is error-controlled: supported if two cases A, B of one
  network satisfy f(B) - k*(A) >= 2 (B shown to fail at a width at least 2 bits above a width at which A is shown
  non-inferior). Draft 1 compared k* values, and k* can result from a non-rejection in N, so a difference of k* values
  alone is descriptive. Raw widths are in format-specific grid units, so (a) follows largely from the grid (screen
  W_acc(1.0) INT8 19, FP8 E4M3 39, `accumulator-sweep-2026-10-01.md:36,41`). (b) Slack form, supported if two cases
  A, B of one network satisfy (W_cert_abs(A) - k*(A)) - (W_cert_abs(B) - f(B)) >= 2. Otherwise "not shown"; per-case
  brackets [f, k*] are reported.
- 0.5 pp claims **[OC-3]**: (i) secondary: every arm of the chosen grid with 0 < screen discordance against wide
  <= 3.27 % (Q7's rule, Q7 `:365-366`, at the calibrated pointwise level, z = 1.78 + 0.84) gets a pointwise one-sided
  bound against -0.5 pp at nominal 0.0374, labelled secondary: 22 ResNet18 arms (`h3-arm-discordance-v1.csv`; under
  the 56 grid the `f21` arms drop) and 8 in-grid arms of the MobileNet H3 cases (`design-v4.json`; 4 eligible per-node
  `sat.struct` arms are in no grid). (ii) Confirmatory: a_case is split 0.4 / 0.4 / 0.2 between N, I and a third fixed
  sequence over the named `sat.w` arms of the case (wide to narrow, -0.5 pp margin, 'toward' bound at
  nu_toward,0.5(0.2 a_case)), the arms selected by Q7's rule at that level (`design-v4.json` `half_margin_arms`):
  equal shares, C = 7 (z = 3.63, p_d <= 1.12 %): FP7 `sat.w25`, FP8 E4M3 `sat.w41`, INT8 `sat.w20`, INT8 signed
  `sat.w20`, posit8 `sat.w34`; C = 15 (z = 4.01, p_d <= 0.96 %): the same five plus MobileNetV3-L FP7 `sat.w23` and
  FP8 E4M3 `sat.w39`; H3-half shares, C = 7 (z = 3.56, p_d <= 1.16 %): the same five (INT6 `sat.w16` no longer
  qualifies). The third sequence is tested whether or not N passed at 1.0 pp (it has its own share). Under (ii), N and
  I run at 0.4 a_case each (C = 7 equal shares: z = 3.32 for N, 3.11 for I).
- Arms with 0 saturation events on the 9k cannot pass I (Delta = 0); they end sequence I where reached.
- Event counts: for every run width the 9k saturation-event count is a declared outcome [OC-2].
- Resolution (equal thirds; `design-v4.json` `H3_resolution_pp`): at 2.8 / 7.2 % discordance the N bound sits 0.57 /
  0.91 pp and the I bound 0.54 / 0.86 pp from the estimate at C = 7; 0.61 / 0.98 and 0.58 / 0.92 pp at C = 15.

### M1, simulator against exact execution [OC-6d]

New in draft 2 (the methods contribution of owner decision 1, `owner-decisions-2026-10-01.md:17`); the owner chooses
confirmatory (a share of alpha) or secondary.
- Contrast per case: top1(B2 simulator cell) - top1(exact wide), 9k, same export, same scoring.
- Cases: the 8 ResNet18 sweep cases; MobileNet cases admitted to H3 by [OC-8a], only those whose projected TOST power
  from the screen is at least 0.8 [threshold is a lane P4 proposal, OC-8a]: power = Phi((1 - |D_s|)/se9 - z) +
  Phi((1 + |D_s|)/se9 - z) - 1, with D_s the 1k simulator-minus-exact difference (expected credit) and se9 = (high -
  low) / (2 x 1.96 x 3) from Q1's pointwise 95 % interval (`results/summaries/accumulator-sweep-mn-v1/simulator.csv`,
  column `exact_minus_b2_expected`, which is exact minus simulator), z at the calibrated level for C_M = 8 + the number
  of MobileNet H3 cases (conservative; the final level uses the final C_M). Result (`design-v4.json` `M1_power_entry`;
  z = 3.32 at C_M = 16): enter MobileNetV2 INT8 (power 1.00), INT6 (0.97), FP6 E2M3 (1.00), FP8 E4M3FN (0.95) and
  MobileNetV3-L FP8 E4M3FN (0.98); not MobileNetV2 FP7 E3M3 (0.01; D_s +0.86), MobileNetV3-L INT8 (0.43) or FP7 E3M3
  (0.00; D_s -1.11). C_M = 13.
- Criterion: equivalence within 1.0 pp, TOST; each one-sided test at a_M1 / C_M (Bonferroni over cases), both sides
  'toward' tests at nu_toward(a_M1 / C_M) (5.2): the interval with both ends at that nominal lies inside (-1.0, +1.0).
  Per case "equivalent" or "not shown". C_M = 8: target 0.00208, nominal 0.00115, z = 3.05, half-width 0.72 pp at 5 %
  discordance; C_M = 13: target 0.00128, nominal 0.000691, z = 3.20, half-width 0.75 pp.
- The MobileNet cases not admitted (FP7 on both networks, whose 1k intervals exclude 0) are reported as secondary
  pointwise contrasts; a paper sentence on simulator fidelity must name them.
- Also reported (no alpha): per-image prediction-change rate with its one-sided 95 % upper bound; the v1 bridge gap
  rerun (`scaled-bridge-gap-2026-10-01.md:273-285`) only if [OC-10] includes it.

### D, detector (secondary, no alpha) [OC-3, OC-5, OC-6e]

YOLOv8n on COCO 4k. Family D = the contrasts "promoted format minus INT8" under the primary detector recipe of
[OC-5]: (a) `default`; (b) `conformant`; (c) both recipes run, `conformant` primary. n_D = the promoted quantised
formats run on the detector (9.5) minus INT8: 10 under either promoted set (NF4 and ternary fall under the 1-mAP stop
rule). Secondary, outside the family: every format minus FP32 (disclosed; 11 contrasts per recipe run); under (c) also
the `default` contrasts and the recipe differences `conformant` minus `default` per format (11 contrasts, margin
0.5 mAP); block formats use `default_fp32_box_logits` as their `conformant` recipe. Per contrast: point, two-sided
95 % interval (method [OC-9]); "within margin" if the 90 % interval lies inside (-delta_det, +delta_det); "different"
if the 95 % interval excludes 0. delta_det per format **[OC-3]**, named now: (a) by nominal width, 0.5 mAP at 8 bits
and 1.0 at 7 bits or fewer (FP7 E3M3 1.0); (b) lane Q7's resolvability rule (`:371-373`) applied per format: 0.5 mAP
if the projected 4k MDD (Q6's 1k "minus FP32" 95 % interval width / 2 x sqrt(1000/4000) x 2.80/1.96,
`b2-detector-breadth-2026-10-02.md:102-114`) is at most 0.5, else 1.0; result (`design-v4.json`
`detector_margin_rule_b`, the same under both recipes): 0.5 for INT8, posit8 es1, MXFP8 E4M3, FP8 E4M3FN and FP7 E3M3
(MDD 0.23-0.49); 1.0 for LOG8, BFP6, FP6 E2M3, INT6, posit6 es1 and LOG6 (0.52-0.78). Also: format order against the
classifier order (Spearman, Kendall tau-b), AP-small/medium/large. If [OC-6e] makes D confirmatory, D takes a share
a_D under [OC-6f] (the confirmatory families split 0.05 equally, or H3 keeps half and the rest split equally), and each
of its n_D contrasts is a TOST at delta_det: both one-sided tests at a_D / n_D, i.e. the two-sided (1 - 2 a_D / n_D)
interval of the chosen form [OC-9] lies inside (-delta_det, +delta_det); per contrast "equivalent" or "not shown". At
that level the MDDs grow by about (3.02 + 0.84) / 2.80 = 1.4 (a_D = 0.0125, n_D = 10), so 0.5 mAP stays resolvable
only for INT8 and posit8. (The coverage calibration of 5.2 is for classifier outcomes; none exists for COCOeval.)

### Exploratory only

Operator structure as a predictor of format choice; per-layer selection; any repair or recipe arm chosen on the screen
(section 9.6). Unadjusted intervals, labelled exploratory.

## 5. Readouts, tie rules, intervals, margins

### 5.1 Readouts and tie rules
- Classifiers: primary expected-credit top-1; secondary lowest-index top-1 (exact engine and simulator with the common
  rule, larger fc code then smaller class index, `scaled-bridge-gap-2026-10-01.md:36`); for "closer to FP32" wording
  only: lowest-index agreement with FP32 and exact McNemar (Q7 `:328-331`).
- Detector **[OC-4]**: (a) primary expected AP over 200 random permutations of the image order, fixed rule (sha256
  order) secondary, uniform-tie scheme as sensitivity; (b) fixed rule primary, expected AP and its SD beside every 6-bit
  number (Q7 `:332-337`).

### 5.2 Intervals and their coverage at the adjusted levels
- Classifiers: the project's unstratified paired image bootstrap, percentile interval (`tools/analysis/b2_ties.py:28,47-57`;
  lowest index `tools/analysis/b_stage_balanced_comparisons.py` `paired_outcomes`). Not BCa (Q7 `:338-345`). Seed
  **[OC-9]**. A new wrapper module (not an edit) draws the same index resampling with 200,000 resamples at adjusted
  levels; its unit test reproduces `b2_ties.paired` at 2.5 / 97.5 with 10,000 resamples and the same seed exactly.
  Each end of an interval is computed at its own nominal one-sided level (below); the ends need not be symmetric.
- Coverage (exact law of the bootstrap, i.i.d. paired binary outcomes, n = 9,000; each protocol written before its
  computation: `artifacts/confirmation_protocol_v1/coverage-protocol-v1.json`, `-v2.json`, `-v2-addendum-1.json`,
  `-v3.json`, `-v3-addendum-1.json`; outputs `coverage-v1.json`, `-v2.json`, `-v2a.json`, `-v3.json`, `-v3a.json`).
  A null at a non-zero margin delta exists at every discordance p_d >= delta (at p_d = delta every discordant image is
  a loss); the v1/v2 grids stopped at 1.5 % (review 2). Revision 3 takes the worst case over p_d from delta to 22.6 %
  (15 values for delta 1.0, 18 for 0.5, dense near delta), separately for the two directions of a bound:
  - 'toward' (claim: lower bound > -delta; sequence N, both one-sided tests of M1 and E, the 0.5 pp sequence, the
    secondary non-inferiority bounds): anti-conservative near p_d = delta; at nominal 0.00085 the error is 1.90 x
    nominal at p_d = 1.0 % (1.46 at 1.25 %, 1.07 at 2.8 %, 0.98 at 7.1 %); for delta 0.5 at 0.000159 it is 3.10-3.19 x
    near p_d = 0.5 % (`coverage-v3.json` `profiles_err_over_alpha`).
  - 'beyond' (claim: upper bound < -delta; sequence I, H1 criterion (B)): conservative at low discordance (0.31-0.49 x
    nominal at p_d = 1.0 %), 0.96-1.00 x at 15-22.6 %. Never computed before revision 3 (section 16, item 17).
  - zero margin (claim: an interval excludes 0): at p_d >= 0.5 % at most 1.06 x for nominal >= 0.00013, but 1.5-2.1 x
    at 0.0015-0.001 for p_d = 0.1 % (`coverage-v3a.json`; addendum 1). The only confirmatory zero-margin claim, H1
    criterion (A), is protected by its conjunct |D_hat| >= delta (section 4); secondary "different" labels at 0.025
    err at most 0.92 x (nominal 0.024).
  Rule (stated in `coverage-protocol-v3.json` before computing): for a target one-sided level t, the nominal level is
  nu(t) = the largest alpha_j = 0.05 x 0.93^j whose worst-case error over the grid is at most t, capped at t (so
  m = t / nu >= 1); a target between tabulated targets uses the next smaller one. Levels used (equal thirds;
  `design-v4.json` `levels`, which also lists the M1-secondary and H3-half schemes; every C from 7 to 17, C_M from 8 to
  18 and k from 1 to 6 in `coverage-v3.json` `enumerated_targets`): H3 N at C = 7 nominal 0.000643 (m 1.85, z 3.22),
  C = 15 0.000250 (m 2.22, z 3.48); H3 I at C = 7 0.00115 (m 1.04, z 3.05), C = 15 0.000556 (m 1.0, z 3.26); M1 C_M = 8
  0.00115 (z 3.05), C_M = 13 0.000691 (z 3.20); 0.5 pp third sequence 0.000140 (C = 7, m 3.40, z 3.63) and 0.0000305
  (C = 15, m 7.29, z 4.01); pointwise secondary 'toward' bounds at 0.05: 0.0432 (delta 1.0, z 1.71) and 0.0374
  (delta 0.5, z 1.78). The single divisor m = 1.4 / 1.5 of revision 2 is withdrawn: it did not hold the 'toward' error
  near p_d = delta and was needlessly conservative for 'beyond' bounds. An exact unconditional test would have more
  power near p_d = delta; it is not adopted (new code, new review). Not modelled: expected-credit fractions (ties) and
  the 9-per-class structure (the unstratified bootstrap ignores it; Q7 `:386-387`).
- Detector: COCOeval recomputed on images resampled with replacement, paired, 2,000 draws (Q7 `:234`). Interval form
  **[OC-9]**: basic bootstrap (2 x estimate - percentile ends) for every detector contrast, or percentile for every
  contrast. The resampled dataset-level mAP is biased upward (FP32 draws average 40.28 against 39.05; large paired
  differences sit off-centre; `b2-detector-breadth-2026-10-02.md:266-273`, INTERIM).

### 5.3 Margins [OC-3]
| Family | delta | Basis |
|---|---|---|
| H1 | 1.0 pp | power as in section 4 (adjusted level); Q7's 5,573 / 6,081 images are at unadjusted 95 % and 7.1 % discordance (`:351-353`) |
| E (H2 quality side) | 1.0 pp, pointwise TOST | Q7 `:351-353` |
| Recipe-pair equivalence | not claimed at 1.0 pp | Q7 `:354-357` |
| H3 | 1.0 pp; 0.5 pp per section 4 option | Q7 `:360-368` |
| M1 | 1.0 pp | section 4 |
| D | per format by [OC-3] (a) or (b): 0.5 or 1.0 mAP (section 4, D); recipe differences 0.5 mAP | Q7 `:371-373`; `design-v4.json` |

## 6. Primary contrast list

| ID | Family | Contrast | n | Level | Criterion |
|---|---|---|---|---|---|
| H1-1..H1-k (k <= 6) | H1 | D_R1, D_R2 of each listed pair | 9,000 | two-sided a_H1/k per component; (B) ends at nu_beyond (the target) | [OC-6b] |
| H3-c (C = 7, or 7 + MobileNet cases) | H3 | Delta(W), sequences N and I | 9,000 | one-sided a_H3/C/2 each; N at nu_toward, I at nu_beyond (5.2) | k*, f, claim [OC-6c] |
| M1-c (C_M = 8 + eligible MobileNet cases) | M1 [OC-6d] | simulator - exact wide | 9,000 | one-sided a_M1/C_M each side, at nu_toward | TOST 1.0 pp |

Secondary (pointwise, labelled): E pairs; float and FP32 accumulators against wide; 0.5 pp arms under option (i);
event counts; every promoted format against INT8 and FP32 per network; family D; seed sensitivity (section 8);
optional arms of 9.6.

## 7. Multiplicity

Family-wise alpha 0.05, Bonferroni over the confirmatory families **[OC-6f]**: (a) equal shares (H1, H3, M1: 0.0167
each; with D confirmatory as well, 0.0125 each; if M1 is secondary, H1 and H3 0.025 each); (b) H3 half (0.025), the
rest split equally. Inside H1, M1 and D: Bonferroni over pairs, cases or contrasts (Holm is not used: it needs
p-values, and the protocol defines intervals only). Inside H3: Bonferroni over cases, two fixed sequences per case at
a_case / 2 each (0.4 / 0.4 / 0.2 under OC-3 (ii)). Single look: each family is evaluated once, after all its runs
exist. Levels (equal thirds; target, then the calibrated nominal of 5.2; `design-v4.json` `levels`): H3 C = 7
a_case = 0.00238, each sequence 0.00119 (N nominal 0.000643, z 3.22; I 0.00115, z 3.05); C = 15 0.00111 / 0.000556
(N 0.000250, z 3.48; I 0.000556, z 3.26); H1 k = 6 a_pair = 0.00278 two-sided (criteria (A) and (B): z 2.99); M1
C_M = 8 one-sided 0.00208 (nominal 0.00115, z 3.05), C_M = 13 0.00128 (0.000691, z 3.20). Other C, C_M, k and shares:
`coverage-v3.json` `enumerated_targets` (C 7-17, C_M 8-18, k 1-6; family shares 0.025, 0.0167, 0.0125, 0.0083).

## 8. Calibration seeds [OC-16]

The primary result of every contrast uses the frozen calibration (B2 ranges from the 2k list, bias correction from its
first 256 images): one draw. Lane Q5's study (`b2-calibration-seeds-2026-10-02.md`, revised after its reviews, file
19:07; INTERIM until the lane closes): seed SD over five disjoint 400-image subsets of the 2k list is 0.23-0.64 pp at
8 bits on ResNet18, 0.09-0.47 on MobileNetV2, 0.22-1.51 on MobileNetV3-Large 8-bit and 1.1-11.3 at 6 bits there
(`:36-41`); 33 of 54 cells exceed delta/2 (`:43`); draft 1's rule (three seeds, five if a sign changes or seed SD >
delta/2) would fire for 81 of 96 contrasts after three seeds (`:44-45`). Q5 recommends SE_total = sqrt(SE_image^2 +
SD_seed^2 / k), five seeds from the start on MobileNetV3-Large and at 6 bits, seeds varying both range and correction
images, and new lists from new training images (`:221-250`). Options: (a) one frozen draw, Q5 cited; (b) (a) plus
Q5's subsets on the 9k for the simulator cells of the H1 pairs (S0-S2; S0-S4 for MobileNetV3-Large and 6-bit pairs),
reported as calibration sensitivity with SE_total, a pair whose sign changes across subsets labelled seed-unstable;
(c) new disjoint 2k lists (not on this machine); (d) addable to any: Q5's accumulator bracket check on the screen
before signing (about 4 GPU-h at old rates; `:316-341`). A 400-image subset changes calibration size as well as the
draw. Exact-engine grids use the one frozen draw under every option.

## 9. Promoted set and frozen run grids

### 9.1 Promoted set [OC-1]
Rule (d) (`public/experiments/configs/breadth-study/b2-matrix-protocol-v1.json:57`) as applied by
`tools/analysis/b2_matrix.py propose()`: `results/summaries/b2-matrix-v1/proposal-d4.csv` (sha256 prefix
`3e288cca80d1a529`; 14 rows = FP32 + 13 quantised: INT8; passing posit8 es1, FP8 E4M3FN, LOG8, MXFP8 E4M3 (intrinsic arm
`cum5_act_maxabs`), FP7 E3M3; stress FP6 E2M3, BFP6, INT6, LOG6, posit6 es1, NF4, ternary; `b2-matrix-2026-10-01.md:480-493`).
Option 12 drops NF4 and ternary (at or near chance on all three networks; the other reading of the stress clause,
`:465-475,520-525`). A configuration failing a gate is reported "not promoted, reason".

### 9.2 Exact grid, ResNet18 [OC-2]
Rule "full" (73): per case `wide`, `control`, the fp16 rule policy, `f21`, `sat.w<W>` for W_half(1k) <= W <=
W_noevent(1k), and `sat.w16` for INT8 cases (`accumulator-sweep-2026-10-01.md:362-376`). Rule "core" (56): `wide`,
`control`, fp16 rule, W_acc(1.0) - 1 <= W <= W_noevent; no `f21` (`:382-388`).

| Case (B2) | 73-run widths | runs | 56-run widths | runs |
|---|---|---:|---|---:|
| INT8 | 16-21 | 10 | 18-21 | 7 |
| INT8 signed | 16-21 | 10 | 17-21 | 8 |
| INT6 | 13-17 | 9 | 14-17 | 7 |
| FP6 E2M3 | 14-19 | 10 | 16-19 | 7 |
| FP7 E3M3 | 21-26 | 10 | 22-26 | 8 |
| FP8 E4M3FN | 37-42 | 10 | 38-42 | 8 |
| posit8 es1 | 30-35 | 10 | 31-35 | 8 |
| FP8 E5M2 | none (wide, control, fp16.x-27, f21) | 4 | wide, control, fp16 | 3 |
| total | | 73 | | 56 |

Under the 56 grid sequence I can rarely conclude: INT8 at W = 18 is -1.45 pp on the screen (72 discordant of 1,000);
at 9k its upper bound would sit about 3.14 x sqrt(0.072/9000) = 0.89 pp above the estimate, near -0.56, not below -1.0,
so f(INT8) would be "open below". The 73 grid adds W = 17 and 16 (24.0 and 0.18 % top-1,
`accumulator-sweep-2026-10-01.md:46`), where I is expected to pass.

### 9.3 Exact grid, MobileNets [OC-8]
Grid rule [OC-8b], default: the same rule as the ResNet18 choice of [OC-2] (full if 73, core if 56), applied with the
definitions of section 4 to lane Q1's write-once 1k summaries (`results/summaries/accumulator-sweep-mn-v1/widths.csv`,
written 18:36; sha256 in `design-v3.json` `inputs_sha256`, re-verified 19:10; Q1 document r4, section 1). Q1's two
reviews (`Q1-mn-accsweep-review.md`, approve with fixes) found the summaries right byte for byte; their open findings
are text. Derived by `design_v3.py` (full: wide, control, fp16 rule, `f21`, `sat.w` W_half..W_noevent, plus `sat.w16`
for INT8; core: wide, control, fp16 rule, `sat.w` W_acc(1.0)-1..W_noevent):

| Case | wide top-1 1k | W_cert | W_noevent | W_acc(1.0) | W_half | full: widths, runs | core: widths, runs | H3 floor |
|---|---:|---:|---:|---:|---:|---|---|---|
| MobileNetV2 INT8 | 72.55 | 25 | 20 | 19 | 17 | 16-20, 9 | 18-20, 6 | joins (+0.45) |
| MobileNetV2 INT6 | 69.71 | 21 | 16 | 15 | 13 | 13-16, 8 | 14-16, 6 | joins (-2.39) |
| MobileNetV2 FP6 E2M3 | 66.33 | 21 | 17 | 15 | 14 | 14-17, 8 | 14-17, 7 | joins (-5.77) |
| MobileNetV2 FP7 E3M3 | 66.77 | 29 | 24 | 23 | 21 | 21-24, 8 | 22-24, 6 | joins (-5.33) |
| MobileNetV2 FP8 E4M3FN | 67.41 | 45 | 40 | 39 | 37 | 37-40, 8 | 38-40, 6 | joins (-4.69) |
| MobileNetV3-L INT8 | 72.00 | 24 | 20 | 19 | 17 | 16-20, 9 | 18-20, 6 | joins (-3.00) |
| MobileNetV3-L INT6 (Q1 stress case) | 20.50 | 20 | 16 | 14 | 12 | 12-16, 9 | 13-16, 7 | fails (-54.50) |
| MobileNetV3-L FP6 E2M3 | 57.86 | 21 | 17 | 17 | 14 | 14-17, 8 | 16-17, 5 | fails (-17.14) |
| MobileNetV3-L FP7 E3M3 | 72.23 | 29 | 24 | 23 | 21 | 21-24, 8 | 22-24, 6 | joins (-2.77) |
| MobileNetV3-L FP8 E4M3FN | 71.38 | 45 | 40 | 39 | 37 | 37-40, 8 | 38-40, 6 | joins (-3.62) |
| total, all 10 cases | | | | | | 83 | 61 | |
| total, the 8 floor cases | | | | | | 66 | 49 | |

H3 floor [OC-8a option (b); 10 pp is a lane P4 proposal]: a MobileNet case joins H3 if its wide arm is within 10 pp
of FP32 on the screen (FP32 72.1 / 75.0, `b2-matrix-2026-10-01.md:91`; the bracket gives wide minus FP32). Eight cases
join, so C = 7 + 8 = 15. The two failing cases (MobileNetV3-L INT6 and FP6 E2M3) [OC-8c]: (i) run under the grid rule
and reported descriptively (full 17 runs, core 12; 0.3-0.9 GPU-h at new shared rates), or (ii) dropped.

Alternative grid rule [OC-8b]: lane Q1's own held-out proposal (`accumulator-sweep-mobilenet-2026-10-02.md:196-199`):
per case wide, W_noevent, W_acc(1.0), W_half, `sat.struct-d_half` and the fp16 rule policy, "10 x 6 = 60 runs" (59
distinct, since W_acc(1.0) = W_noevent = 17 for MobileNetV3-L FP6 E2M3). It has no `control` arm, no width strictly
between W_half and W_acc(1.0), and skips W_noevent - 1 where W_acc(1.0) < W_noevent - 1, so sequences N and I step
over gaps and k* and f(c) are resolved only at those widths; it adds the per-node rule at d_half.

### 9.4 Simulator cells (B2 matrix machinery, redirected outputs)
- S-A: every promoted quantised format x 3 classifiers under R2, plus FP32 x 3: 11 x 3 + 3 = 36 cells (12-set) or
  13 x 3 + 3 = 42 (14-set).
- S-B: the H1 pairs under R1: at most 12 cells; if [OC-6g] admits formats outside the promoted set, their R2 cells too
  (at most 12 more).
- S-M: B2 cells of the 8 ResNet18 sweep cases not in S-A: INT8 signed, FP8 E5M2 (+2). MobileNet M1 cases are in S-A.
- Totals: 36 + 12 + 2 = 50 (12-set) or 56 (14-set); 62 / 68 with pool (ii).
- S-seed [OC-16 (b)]: up to 3 subsets x 24 pair cells = 72 (more with five subsets).
- S-arm: recipe arms of 9.6 if chosen.
Inputs are streamed from the frozen complement list without an input cache (a 9k cache would be about 9 x 602 MB per
preprocessing). Bias-correction cells use the bit-identical low-memory paths (`b2-collapse-diagnosis-2026-10-02.md:576-584`)
or run as heavy jobs.

### 9.5 Detector arms [OC-5]
FP32 plus the promoted formats under the chosen detector recipe; formats below 1 mAP on the screen are not run (stop
rule; NF4 0.52, ternary 0.00, `b2-detector-breadth-2026-10-02.md:28,120-122`, INTERIM): 12 configurations under one
recipe, 23 under both (FP32 once; block formats use `default_fp32_box_logits` as the nearest `conformant` recipe).
FP32 is rerun on the 4k; its detections may be compared with the existing val5k FP32 file only after unblinding.

### 9.6 Optional arms [OC-10, OC-12, OC-13]
- Per-node family `sat.struct-<d>`: the sweep deferred 32 runs (priorities 4 and 6: d in {4, 8} and {2, 12} for the 8
  cases) and 18 original-recipe (B1) 1k runs (`accumulator-sweep-2026-10-01.md:297-300`). Proposed: run all 50 on the
  screen first; enter the confirmation as secondary only if they exist before signing. The halving to d in {4, 8}
  (16 runs) is a lane P4 proposal, not the sweep's plan.
- Bias-correction form (lane Q3, INTERIM; `b2-collapse-diagnosis-2026-10-02.md:546-566`): the chosen alternative form
  (literal weights-only Appendix D pass or local own-error form) as a secondary arm on FP8 E5M2, FP6 E3M2 and LOG6 on
  ResNet18 and MobileNetV2 (6 cells per form), if [OC-12] chooses so.
- MobileNetV3 stem repair r3b (lane Q4, INTERIM; `b2-activation-attribution-2026-10-02.md:441-449`): one secondary
  cell (MobileNetV3-L INT8; screen 74.10, or 76.13 under the reviewer's per-channel sampler, `:311-340`), with the
  sampler and its seed fixed in the signed JSON, if [OC-13] chooses so.
- AXE accumulator-aware PTQ (lane Q8, INTERIM): ResNet18 INT8 and INT6 at the narrowest P within 1 pp of wide on the
  screen (25 and 21, `accumulator-aware-ptq-2026-10-02.md:24-28`) and P - 1: 4 secondary cells, if [OC-10] adds them
  (reviewer baseline, `related-work-audit-2026-10-01.md:1123`).

## 10. Requirements on the exact engine

1. Base: the fast path `artifacts/speed_v1/implementations/edfecd5cdb0dc5adf06eff2807afa023b12d781cf0d08961a0b023da510a293b`,
   whose records carry `base_engine_sources` = `1f75c923...` for every case, ResNet18 included (record
   `artifacts/speed_v1/runs/edfecd5c.../resnet18-int8-default-b2/predictions/wide-cuda-00000-00064.json`;
   `speed-optimizations-2026-10-02.md:87,113`). The sealed ResNet18 1k files were produced by archive `7c6344af...`;
   the fast path reproduces them and the 1f75c923 MobileNet files: 0 differences in about 97,000 image records
   (`S1-engine-review.md:678-679`; `speed-optimizations-2026-10-02.md:91`). Usable now only on the screen-1k list
   (`S1-engine-review.md:683-690,698`).
2. Source change: both engines hard-code the screen-1k list (`S1-engine-review.md:698`). The change is confined to list
   selection (a manifest path plus its expected digest from section 2, iteration in manifest order, the list digest in
   every record) and, under [OC-21], compressed records. No change to kernels, certificates, policies, codebooks, export
   loading, hashing or tie lists.
3. New digest, then the re-checks of `S1-engine-review.md:716-729` part (d) in order: (1) only if `fastdot.cu`,
   `kernel.h`, `native.py` or `engine.py` changed: kernel.h sha256 `2fb29b0d...`, nvcc command, PTX identity of the 8
   entries, A's t1; (2) S1's regress 82, trace 100, compare 369 sets, then `gates` and `archive`; (3) D's
   `check_archive.py` and `trace_gates.py`; (4) sealed 1k comparisons (B's 18 ResNet18 pairs, wide and control of all 10
   MobileNet cases; about 15 GPU minutes); (5) C's worklists (18 GPU minutes), B's references 700-731, D's fp16 probe.
   A change confined to `worker.py` needs (2)-(5).
4. Screen equivalence: with the list parameter at the screen manifest, the changed engine reproduces, record for record,
   one sealed 1k file per (case, policy family) of every grid case.
5. Careful independent review of the change before any held-out call (owner requirement of 2026-10-02) and the owner's
   acceptance of it **[OC-20]**.
6. The two cases without a fast gate in the combined verdict (MobileNetV2 INT6, MobileNetV3-L FP7 E3M3;
   `S1-engine-review.md:692-696`) got gates from lane S1 at 18:07 and passed Q1's K2 and K3 checks (Q1 handoff,
   18:07-18:23 entries); reviewer D's `trace_gates.py` over these two gate files has not been run. They run on the fast
   path only after that check and the review of item 5; otherwise on the archive at old rates with the same list
   change and review.
7. Per runner start: K1 (archive manifest, run-root digest, torch 2.3.0+cu121, NumPy 1.26.4, driver 595.91.07;
   `S1-engine-review.md:702-703`). K4 (`S1-engine-review.md:710-712`; seed and batch are lane P4 proposals): after a
   family's runs, its held-out calls (case, policy, image range), sorted by that key, are sampled with
   `numpy.random.default_rng(20261004).choice(n_calls, ceil(n_calls / 10), replace=False)`; each sampled call is
   recomputed at batch 5 in a second process into a scratch root and compared record for record (identity only,
   outcome-blind); a difference stops that family's analysis. Batch 8; heavy jobs declared as rule 7 of the lane
   rules says.
8. Records: one sealed file per (case, policy, image range); `engine_sources` = the new digest; the list digest; no
   outcome computed by the runner.

## 11. Compute and disk ledger

Arithmetic: `artifacts/confirmation_protocol_v1/design_v4.py` -> `design-v4.json` `ledger_gpu_h` (supersedes the
ledgers of `ledger-v1.json`, `design-v2.json` and `design-v3.json`, kept unchanged). Hours = runs x 9,000 x seconds per
image + 30-40 s setup per process at new rates (one process per case; the lane rules' figure, an upper bound: S1
measures 2.1-2.3 s from spawn to the first batch, `speed-optimizations-2026-10-02.md:37,190`).

| Grid | Old engine, exclusive | Old engine, shared | New engine, exclusive | New engine, shared |
|---|---|---|---|---|
| ResNet18 73 runs | 73 x 9,000 x 0.154-0.1687 = 28.11-30.79 h | x 0.22-0.35 = 40.15-63.87 h | x 0.0061 + setup = 1.18-1.20 h | x 0.0116-0.0158 + setup = 2.18-2.97 h |
| ResNet18 56 runs | 21.56-23.62 | 30.80-49.00 | 0.92-0.94 | 1.69-2.30 |
| MobileNets 83 runs, full | 83 x 9,000 x 0.294 = 61.01 | x 0.26-0.64 = 53.95-132.80 | x 0.0062 + setup = 1.37-1.40 | x 0.0085-0.021 + setup = 1.85-4.47 |
| MobileNets 66 runs, full, floor cases | 48.51 | 42.90-105.60 | 1.09-1.11 | 1.47-3.55 |
| MobileNets 61 runs, core | 44.84 | 39.65-97.60 | 1.03-1.06 | 1.38-3.31 |
| MobileNets 60 runs, Q1's proposal | 44.10 | 39.00-96.00 | 1.01-1.04 | 1.36-3.26 |

Rates (s per image): ResNet18 old 0.154 exclusive (`accumulator-sweep-2026-10-01.md:306`) and 0.1687
(`speed-optimizations-2026-10-02.md:18`), shared about 0.22 (`accumulator-sweep-2026-10-01.md:307`), 0.324
(`speed-optimizations:53`), up to 0.35 (Q7 `:231`); new 0.0061 exclusive (`:19`), 0.0116-0.0126 shared one job
(`:30-31`), 0.0149 in the profile and 0.0158 at `sat.w20` (`:53,55`). MobileNet old 0.294 exclusive (`:20`),
0.26-0.64 shared (`accumulator-sweep-mobilenet-2026-10-02.md:109`), 0.599 (`speed-optimizations:53`); new 0.0062
exclusive (`:21`), 0.0085-0.021 shared on Q1's 1k files (Q1 now prints 0.0082-0.021, `:198-199`; the ledger keeps
0.0085, a 4 % difference at the low end; S1's profile 0.0136, `:53`, lies inside). INTERIM: S1 document (19:13).

Other work (GPU-h unless stated):
- Simulator cells, 9 x 6-55 s + 30-40 s setup per cell (Q7 `:230`): 36 cells 0.84-5.35; 42: 0.98-6.24; 50: 1.17-7.43;
  56: 1.31-8.32; 62: 1.45-9.21; 72 seed cells: 1.68-10.70; 6 cells 0.14-0.89; 4 cells 0.09-0.59 (AXE evaluation only;
  its fits are not in Q8's sources); 1 cell 0.02-0.15.
- Per-node 32 + B1 18 runs (ResNet18, new shared): screen 1k 0.23-0.31, 9k 1.52-2.06; halved (34 runs) 0.18-0.24 /
  1.05-1.43.
- Detector on 4k, 0.9-4 min per 1k (Q7 `:233`): 12 configurations 0.72-3.2; 23: 1.38-6.13. COCOeval bootstrap, 2,000
  draws, 8-9 CPU-min per 1k (Q7 `:234`): 12: 6.4-7.2 CPU-h; 23: 12.3-13.8 CPU-h. Tie orders: CPU minutes.
- Engine change re-checks: (4) about 15 and (5) about 18 GPU minutes (`S1-engine-review.md:722-724`) plus (2) and the
  screen equivalence; 0.5-2 GPU-h (estimate, not measured).
- v1 bridge gap rerun (optional): 5.2 GPU-h (wide and B replay), 7.9 with control, old v1 rates
  (`scaled-bridge-gap-2026-10-01.md:278-279`), plus its own list change.
- Recommended package of the decision package (simulator 50 cells, ResNet18 73, MobileNets 83, detector 23, per-node
  and B1 on screen and 9k, 6 bias-correction cells, r3b, AXE, 72 seed cells, re-checks): 10.8-37.7 GPU-h plus
  12.3-13.8 CPU-h (`design-v4.json` `recommended_package_sum`).

Disk: 1k exact prediction files average 2.10 MB (archive root, 17 files) and 2.72 MB (fast root, 102 files), at most
9.88 MB. At 9k, uncompressed: 73 runs 1.38-1.79 GB, 56 runs 1.06-1.37, MobileNets 83 runs 1.57-2.03, 61 runs
1.15-1.49; gzip -9 shrank 1k files 3.6-9.3 times (73 runs 0.15-0.50 GB; 83 MobileNet runs 0.17-0.56). The project may
grow 3 GB in total, shared by all lanes: the held-out records need an owner allowance and compressed storage [OC-21].

Confirmation ledger: new, separate from the historical ledgers, ceiling 1.5 x the chosen options' upper estimate;
unfinished cases are "budget-limited" [OC-14].
unfinished cases are "budget-limited" [OC-14].

## 12. Order of operations

1. Re-check every INTERIM input (section 15); fill the H1 pair list, the MobileNet grid, the H3 and M1 MobileNet case
   lists and the 0.5 pp arms by their rules; record source sha256s.
2. Owner fills every [OC-n]; the JSON twin is written as a new file `confirmation-protocol-v1.json` with status
   "SIGNED", owner and date; its sha256 is recorded in the owner's decision record. No held-out step before.
3. Audit (section 3). Stop if it fails.
4. Engine change, gates, re-checks, screen equivalence, independent review, owner acceptance (section 10); simulator and
   detector held-out loaders as new modules, each shown on the screen list to reproduce existing sealed cells.
5. Analysis code (section 14, including the interval wrapper and its unit test) written and run on screen data as a dry
   run; its sha256 recorded in an addendum before the first held-out call.
6. Runs, outcome-blind: FP32 baselines, exact grids, simulator cells, detector arms. Monitoring prints only failures,
   identity checks and counts, never an accuracy.
7. Unblinding: one run of the frozen analysis per family after all its runs exist; summaries written once to new
   versioned paths.
8. Anything else is exploratory and labelled.

## 13. Deviation, failure and stop rules

- Before the first held-out call a deviation is a numbered addendum written before the affected step. After it, a
  change of hypotheses, contrasts, margins, levels, recipes, promoted set or analysis code makes the affected result
  exploratory and needs a new protocol version.
- A crash or out-of-memory is rerun (identical configuration; out-of-memory retried once with `--heavy 8000`) and
  logged. An integrity failure (fail-closed check, K4 identity, engine disagreement with its control) stops that
  configuration; it is reported as failed, not replaced.
- No interim look at outcomes. A family is evaluated once.
- Compute: stop at the ledger ceiling (unfinished = "budget-limited", not negative). Disk: stop at the owner's allowance.
- A contrast that cannot be run (gate, budget) stays in its family as "not run"; its alpha is not redistributed. In an H3
  sequence a width not run stops the sequence there.

## 14. Analysis fixed before unblinding

- Per family: the tables of section 6 with point estimate, interval at the stated level, criterion outcome. H3: per
  case the passing runs of N and I, k*, f(c), s*, certificate and published-formula widths, 9k events per width. M1:
  equivalence verdict and change rate. D: AP and AP-small/medium/large under the tie rules of [OC-4].
- Code: the project functions of 5.2 through the new wrapper (200,000 resamples, nominal /m (section 5.2) against non-zero
  margins); COCOeval as in the detector studies.
- Labels: "confirmation (9k held-out)" or "confirmation (COCO 4k held-out; FP32 baseline previously seen)"; screen
  numbers are never pooled with held-out numbers.
- Every summary records the protocol sha256, the analysis-code sha256 and the list digest.

## 15. INTERIM inputs to re-check before signing (state at 19:24, file times of 2026-10-02)

| Input | File (modification time) | Used for |
|---|---|---|
| Q1 MobileNet sweep | summaries `results/summaries/accumulator-sweep-mn-v1/` (written once 18:36; sha256 equal to `design-v3.json` `inputs_sha256` at 19:10); document (19:16, text revision after review 2); `Q1-mn-accsweep-review.md` (19:13; review 2: approve with fixes, text findings only, summaries right byte for byte) | 9.3 grid; H3 and M1 MobileNet cases; 1k exact minus simulator excludes 0 for FP7 on both networks (MobileNetV2 -0.86 [-1.65, -0.08], MobileNetV3-L +1.11 [+0.15, +2.10], `:84-90`; simulator minus exact is the negative) |
| Q2 ranking robustness | `b2-ranking-robustness-2026-10-02.md` (19:22, first full text; not yet reviewed); summaries `results/summaries/b2-rank-v1/` incl. `reversals.csv` (now written); handoff (19:23) | H1 pair list (rule of section 4 not yet run on it); R1 option (iii) |
| Q3 collapse diagnosis | `b2-collapse-diagnosis-2026-10-02.md` (18:20; four forms at `:546-566`) | [OC-12], 9.6 |
| Q4 activation attribution | `b2-activation-attribution-2026-10-02.md` (18:55; owner section `:438-463`, r3b `:441-449`, sampler `:311-340`) | [OC-13] |
| Q5 calibration seeds | `b2-calibration-seeds-2026-10-02.md` (19:07); summaries `results/summaries/b2-seeds-v1/` (17:57); handoff (19:08) | section 8, [OC-16] |
| Q6 detector breadth | `b2-detector-breadth-2026-10-02.md` (17:52; table `:102-114`) | 9.5, D margins, [OC-9] |
| Q8 accumulator-aware PTQ | `accumulator-aware-ptq-2026-10-02.md` (18:03) | 9.6 |
| S1 speed | `speed-optimizations-2026-10-02.md` (19:13) | section 11 rates |
| S1 review | `S1-engine-review.md` (16:48; combined verdict) | section 10; the two gates S1 wrote at 18:07 (MobileNetV2 INT6, MobileNetV3-L FP7 E3M3): Q1's document now records a run of reviewer D's `trace_gates.py` over all fast gates (`accumulator-sweep-mobilenet-2026-10-02.md:159`), to be checked before item 6 of section 10 is taken as met |

Also before signing: rerun `design_v4.py`'s derivations as a new versioned file if Q1's summaries change (compare the
sha256s in `design-v3.json` `inputs_sha256`); run the H1 pair rule of section 4 on Q2's `reversals.csv` once Q2's
document is reviewed.

## 16. Where sources disagree (not resolved here)

1. Bootstrap seed: draft 1 proposes 20261001 (`confirmation-protocol-draft-2026-10-01.md:24`); the project functions and
   the sweep use 20260927 (`b2_ties.py:28`; `accumulator-sweep-2026-10-01.md:115`). [OC-9]
2. Stratification: draft 1 asks for class-stratified resampling (`:24`); the project functions do not (`b2_ties.py:53`).
3. Detector draws: 2,000 (Q7 `:234`) against 10,000 (draft 1 `:24`).
4. Old ResNet18 rate: 0.154 (`accumulator-sweep-2026-10-01.md:306`) against 0.1687 (`speed-optimizations-2026-10-02.md:17`);
   Q7 uses 0.15-0.35 (`:231`).
5. Old MobileNet rate: about 0.36 (Q7 "6 min per 1k", `:232`), 0.294 exclusive (S1 `:19`), 0.515-0.635 shared (Q1 `:58`).
6. Promoted set: 14 entries by the code, 12 by the other reading of the stress clause (`b2-matrix-2026-10-01.md:465-475`).
7. Data roles: draft 1 makes the 40k complement the confirmation set; owner decision 2 seals the 9k remainder; draft 1's
   fact-check (`:97-99`) notes the roadmap's other ordering. [OC-19]
8. The audit's documented rerun command overwrites its v1 summary (`evaluation-history-audit-2026-10-01.md:113`).
9. Seeds: draft 1's five-seed rule (`:30`) would fire for 81 of 96 contrasts (Q5 `:24-28`). [OC-16]
10. H3 width axis: draft 1 one axis over three types (`:17`); Q7 per type (`:376-377`).
11. W_half: revision 2 recorded that Q1's document worded W_half as "narrowest width above half"; Q1's document (file 19:16,
    `accumulator-sweep-mobilenet-2026-10-02.md:27`) now uses the protocol's definition ("widest width at which Top-1 is
    at most half"), as L8's protocol and Q1's code do (`tools/accumulator_sweep_mn/protocol.py:92`). Resolved.
12. "Identical to wide on the screen" (15 ResNet18 arms in `h3-arm-discordance-v1.csv`) means 0 top-1-discordant images;
    the FP8 E5M2 and posit8 controls differ in outputs (`accumulator-sweep-2026-10-01.md:67-68`).
13. Draft 2 revision 0 quoted a MobileNet new shared rate of 0.0073-0.0082 s per image from an earlier version of the S1
    document; the 18:08 version gives 0.0136-0.0149 (`:51,53`); section 11 uses the latter.
14. MobileNet held-out grid: this protocol applies the ResNet18 rule (83 or 61 runs, section 9.3); lane Q1 proposes
    10 x 6 = 60 runs (wide, W_noevent, W_acc(1.0), W_half, `sat.struct-d_half`, fp16 rule;
    `accumulator-sweep-mobilenet-2026-10-02.md:117-122`; 59 distinct). [OC-8]
15. Old MobileNet rate: Q1's r4 gives 0.26-0.64 s per image for the archive on the shared GPU (`:109`) and 0.0082-0.021
    for the fast path (`:198-199`); revision 2 used 515-635 s per 1k and S1's 0.0136-0.0149; section 11 now uses Q1's
    measured ranges.
16. Revision 1 of this draft labelled Q1's 1k simulator comparison "simulator minus exact"; Q1's `simulator.csv`
    column is `exact_minus_b2_expected` (left = exact). Corrected in revision 2; the M1 TOST is symmetric, so no rule
    changes.
17. `coverage-v2.json` checked "H1 criterion B" with the non-inferiority ('toward') null; criterion (B) and sequence I
    are 'beyond' tests, first computed in `coverage-v3.json`. The v2 verdict for those rows does not apply.
18. Stated times of this lane against file times: revision 2's header said 18:40-19:10 (file 18:57); the `written`
    field of `coverage-protocol-v3-addendum-1.json` says 19:19, its file time is 19:16:48 (before the run, 19:16:49; the
    field was not edited afterwards, so the sha256 recorded in `coverage-v3a.json` stays valid).
19. Detector margins: Q7 states 0.5 mAP for 8-bit configurations near FP32 and 1.0 for 6-bit formats (`:371-373`);
    FP7 E3M3 and the 8-bit formats far from FP32 (LOG8 -5.38 mAP, `b2-detector-breadth-2026-10-02.md:109`) fall under
    neither. [OC-3]

## 17. Sources read

Draft 1 and its fact-check; `proposed-decision-updates-2026-10-01.md`; `owner-decisions-2026-10-01.md`;
`decision-register.md`; `evaluation-history-audit-2026-10-01.md`; `quality-metrics-resolution-2026-10-02.md`;
`accumulator-sweep-2026-10-01.md` and its protocol; `b2-matrix-2026-10-01.md` 4.2 with rule (d) and `proposal-d4.csv`;
`b2-detector-2026-10-01.md`; `scaled-bridge-gap-2026-10-01.md`; `speed-optimizations-2026-10-02.md`; the combined verdict
of `S1-engine-review.md`; `docs/roadmap/phases/phase-05-full-validation.md`; `related-work-audit-2026-10-01.md` section 3;
the documents and handoffs of section 15; `tools/experiment_b2_rank/analysis.py` (column definitions);
`data/manifests/*` and `index.json`.
