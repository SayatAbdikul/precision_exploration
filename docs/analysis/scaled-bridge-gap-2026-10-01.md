# Simulator versus exact scaled bridge at 1,000 images (gap study v1)

Date: 2026-10-01. Lane L5. Evidence level: **development**. The three cases (ResNet18, maxabs; fp6_e2m3, fp6_e3m2, fp7_e3m3) and the 1k ImageNet screen were both used in earlier selection, so nothing here is independent confirmation. The tie audit (below) is a **post-hoc, exploratory** addendum.

Revision r2 (same day), after an independent review (`artifacts/agent_orchestration/handoffs/L5-bridge-gap-review.md`):
- **Corrected:** the first-divergence counts. The r1 code walked the sealed records' layers in alphabetical key order, not execution order, and its stage rule put the eight residual adds in the stem. Both are fixed and `summary.json` was regenerated. Only the first-divergence fields changed; every accuracy, interval and count elsewhere is identical. The r1 files are kept in `artifacts/scaled_bridge_gap_v1/superseded/`.
- **Rerun:** the six stage hybrids, with each residual add placed in its own block (addendum protocol `scaled-bridge-gap-diag-1a`, sealed before the rerun).
- **Reworded:** the scope of the FP32-accumulator claim; null results are now stated as bounds; Top-5 is reported; the selection explanation is weakened; three small factual corrections.

Revision r3 (same day), after the second review (same file). Wording only; no number in the result files changed and nothing was rerun:
- **Corrected:** the propagation sentence. Up and down propagated counts agree within 0.63 % from layer1 to the head, but not in the stem under the corrected stages (stem counts now given).
- **Corrected:** the source-hash disclosure, the cause of the 3 changed fc images in the fp7_e3m3 head hybrid (fc accumulation), and the job list (smoke jobs added).
- **Scoped:** "not a property of the arithmetic" now reads "on this panel", with the remaining ~1 pp interval stated.

Protocols, each sealed before its runs:
- 1k study: `artifacts/scaled_bridge_gap_v1/protocol-1k-v1.json` (version `scaled-bridge-gap-1k-1`).
- Mechanism diagnostics (step 4): `artifacts/scaled_bridge_gap_v1/protocol-diag-v1.json` (`scaled-bridge-gap-diag-1`).
- Tie audit: `artifacts/scaled_bridge_gap_v1/protocol-ties-v1.json` (`scaled-bridge-gap-ties-1`). This was written after the 1k results and the first diagnostic case had been seen, so it is post-hoc.
- Corrected stage hybrids: `artifacts/scaled_bridge_gap_v1/protocol-diag-1a-stagefix.json` (`scaled-bridge-gap-diag-1a`). This was written after review, before the rerun.

Machine results in `results/summaries/scaled-bridge-gap-v1/`:
- `summary.json` (sealed, r2) and `per_image.csv`: per-image predictions, the wide-versus-B transition and the first-divergence layers in execution order.
- `diagnostics.json` (r2): per-configuration accuracy and code flips. Per-stage tallies use the corrected stages.
- `ties.json`: tie counts, and every difference under a common tie rule.
- `stage_hybrids_corrected.json`: the rerun stage hybrids.
- Source hashes (checked against the live files in r3):
  - `diagnostics.json` and `stage_hybrids_corrected.json`: every recorded hash matches the current code.
  - `summary.json`: every recorded hash matches except `tools/scaled_bridge_gap_v1/diag_summary.py`, which was edited a minute after `summary.json` was written. The 1k summary does not use that file.
  - `ties.json` was written in r1. Its recorded hashes for `tools/run/scaled_bridge_gap.py`, `analysis.py`, `diagnostics.py` and `diag_summary.py` predate the r2 edits. Those edits add code paths and fix the first-divergence walk; they leave the tie computation unchanged.
  - The independent re-review regenerated the case payloads of all four result files in memory with the current code: 12 of 12 were identical to the stored files.

## Bottom line

- **On this development panel, the measured 1–2 point simulator-versus-exact gap is not explained by arithmetic. It comes from how the two sides break ties between equal quantized logits.** The final fc layer stores codes on a 6- or 7-bit grid, so the largest logit is shared by several classes on 18–30 % of images.
- B (the FP32 QDQ simulator) ranks with a batched CUDA FP32 `topk`. The exact engine ranks with a per-image CPU binary64 `topk`. These two calls order tied classes differently.
- Apply one deterministic rule to both arms: the larger fc code wins, and on a tie the smaller class index wins. Wide − B is then **+0.2, +0.1 and +0.2 pp** (fp6_e2m3, fp6_e3m2, fp7_e3m3), with intervals [−0.6, +1.0], [−0.9, +1.1] and [−0.6, +1.0]. Under the same rule, the 128-image v1 prefix gap of −3.9 to −4.7 pp becomes 0.0, −0.8 and 0.0.
- Under that rule, no single contract difference has a top-1 point estimate beyond ±0.5 pp in any case, and every interval includes zero. This is a bound, not a proof of no effect: the single-switch intervals reach from −1.1 to +1.6 pp.
- **Evidence level.** All of this is development evidence on the 1k screen, which was used in selection, and the tie analysis is post-hoc. Settling an effect of 1–2 pp, or confirming that it is absent, needs the 9,000 held-out images. Those are reserved until the hypotheses are frozen. The run is costed below (about 5.2 GPU-hours) and was not done.

## What ran

- **Engine.** The sealed v1 engine (`tools/scaled_bridge_v1`, source digest `5e11c339…`), exactly as enrolled, was called read-only from a new controller (`tools/scaled_bridge_gap_v1`). The v1 ledger, runs and reports were not touched.
- **Panel.** All 1,000 images of the frozen `imagenet_screen_1k`, in v1's SHA-256 order. Images 0–127 are v1's sealed panel. No image outside the 1k screen was used.
- **Arms.** Wide (exact code-domain dot) and control (sequential FP32 FMA dot) ran on CUDA at batch size 1. The B anchor was replayed with v1's `TracedB` in the original runtime, keeping the original batch-of-8 membership. All 1,000 B and FP32 Top-5 lists matched the retained `artifacts/experiment_b/predictions` records.
- **Reproduction gate.** For images 0–127, every record in all three arms equals v1's sealed record: layer code, state and raw hashes, output, Top-5 and diagnostics. That is 3 × 3 × 128 checks with 0 mismatches.
- **Cost.**
  - The 1k arms ran as 10 exclusive-lock GPU jobs with 0 failures: 3,168.8 GPU-locked seconds, plus 636 s of waiting for the lock.
  - The diagnostics ran as 3 jobs (576, 689 and 843 s) and the tie audit as 3 jobs (656, 562 and 431 s). The r2 stage-hybrid rerun took 3 jobs (295, 289 and 200 s). Each of these three harnesses also had one 16-image smoke job (12, 11 and 6 s). Two diagnostic jobs, all tie-audit jobs and the rerun ran in shared GPU mode, so their times are not clean timings.
  - In total, `artifacts/scaled_bridge_gap_v1/jobs.jsonl` lists 22 jobs (the 10 arm jobs, including a 3-image smoke, plus 12 above), and all 22 exited 0.
  - None of these figures is reported as throughput.

## Results (top-1, n = 1000, paired; intervals are pointwise 95 % paired-bootstrap intervals from `tools/analysis/b_stage_balanced_comparisons.paired_outcomes`)

"Measured" means each arm's own `topk`, as recorded. "Common rule" means one deterministic tie rule applied to both arms (larger fc code first, then smaller class index). "Random-tie expectation" means the expected top-1 if every top-1 tie were broken uniformly at random.

| Case | FP32 | B | Wide | Control | Wide − B, measured | B→wrong / B→correct | McNemar p | Wide − B, common rule | B→wrong / B→correct | Wide − B, random-tie expectation |
|---|---:|---:|---:|---:|---|---:|---:|---|---:|---:|
| fp6_e2m3 | 70.1 | 66.2 | 66.5 | 66.5 | +0.3 [−1.3, +1.9] | 31 / 34 | 0.80 | +0.2 [−0.6, +1.0] | 7 / 9 | +0.09 |
| fp6_e3m2 | 70.1 | 65.3 | 63.4 | 63.4 | −1.9 [−3.8, 0.0] | 57 / 38 | 0.064 | +0.1 [−0.9, +1.1] | 13 / 14 | −0.04 |
| fp7_e3m3 | 70.1 | 69.7 | 68.3 | 68.3 | −1.4 [−3.0, +0.2] | 41 / 27 | 0.11 | +0.2 [−0.6, +1.0] | 7 / 9 | +0.20 |

- "B→wrong" counts images where B is correct and wide is wrong. "B→correct" counts the reverse.
- Measured top-1 predictions changed on 123, 204 and 113 images, including changes from one wrong class to another. Under the common rule, the discordant images fall from 65, 95 and 68 to 16, 27 and 16.
- Top-1 under the common rule is 65.5 / 65.7 (B / wide) for fp6_e2m3, 62.9 / 63.0 for fp6_e3m2 and 68.3 / 68.5 for fp7_e3m3.
- **Pure tie cases.** Top-1 changed with identical stored codes at *every* layer on 36 images for fp6_e2m3 (7 B→wrong, 7 B→correct, 22 wrong→other), 6 for fp6_e3m2 (3 / 1 / 2) and none for fp7_e3m3. This undercounts the tie effect, because the logits depend only on the fc codes. Top-1 changes with identical *fc* codes number 83, 61 and 6 (table below).

**Top-5, the protocol's secondary outcome** (wide − B; correct only in B / only in wide):

| Case | Measured | Common rule |
|---|---|---|
| fp6_e2m3 | −0.7 [−1.4, −0.1], 9 / 2 | −0.1 [−0.3, 0.0], 1 / 0 |
| fp6_e3m2 | +0.2 [−0.7, +1.1], 9 / 11 | 0.0 [−0.6, +0.6], 5 / 5 |
| fp7_e3m3 | −0.2 [−1.0, +0.6], 10 / 8 | +0.2 [−0.4, +0.8], 4 / 6 |

The one measured Top-5 interval that excludes zero (fp6_e2m3, −0.7 pp) disappears under the common rule. Ties across the Top-5 boundary are even more common than top-1 ties (57–76 % of images; see below).

Each arm minus FP32. The FP32 baseline's logits are not quantized, so ties there should be rare. The saved records hold no FP32 logits, though, so this could not be checked:

| Case | B − FP32, measured | B − FP32, common rule | Wide − FP32, measured | Wide − FP32, common rule |
|---|---|---|---|---|
| fp6_e2m3 | −3.9 [−6.1, −1.7] | −4.6 [−6.7, −2.5] | −3.6 [−5.8, −1.4] | −4.4 [−6.5, −2.2] |
| fp6_e3m2 | −4.8 [−7.1, −2.5] | −7.2 [−9.5, −4.8] | −6.7 [−9.0, −4.4] | −7.1 [−9.4, −4.8] |
| fp7_e3m3 | −0.4 [−2.0, +1.2] | −1.8 [−3.4, −0.2] | −1.8 [−3.5, −0.1] | −1.6 [−3.3, +0.1] |

Control minus wide is 0.0 [0.0, 0.0] in every case. Across 3 × 1,000 image-runs, the control arm changed no top-1 prediction, no ordered Top-5 list and no stored code at any layer. Control and wide use the same engine `topk`, so ties cannot separate them.
- For fp6_e2m3 and fp6_e3m2, every per-layer record is identical.
- For fp7_e3m3, 451 of 1,000 images have an FP32 dot result that differs from the exact one in at least one layer, but none of those differences moves a stored code.
- The one-sided 95 % Clopper–Pearson upper bound on the per-image rate of a changed top-1 prediction, changed correctness or changed Top-5 list is 0.299 % for each case. For fp6_e2m3 and fp6_e3m2 the same bound also covers any numerical difference at all.

### The 128-image prefix versus the rest

| Case | Wide − B, 0–127, measured | Wide − B, 0–127, common rule | Wide − B, 128–999, measured | Wide − B, 128–999, common rule | B top-1, 0–127 vs 128–999 (measured) |
|---|---|---|---|---|---|
| fp6_e2m3 | −4.7 [−9.4, −0.8], 7 / 1 | 0.0 [0.0, 0.0], 0 / 0 | +1.0 [−0.7, +2.8], 24 / 33 | +0.2 [−0.7, +1.1] | 72.7 vs 65.3 |
| fp6_e3m2 | −3.9 [−9.4, +1.6], 9 / 4 | −0.8 [−2.3, 0.0], 1 / 0 | −1.6 [−3.7, +0.5], 48 / 34 | +0.2 [−0.9, +1.4] | 67.2 vs 65.0 |
| fp7_e3m3 | −4.7 [−9.4, −0.8], 7 / 1 | 0.0 [0.0, 0.0], 0 / 0 | −0.9 [−2.6, +0.8], 34 / 26 | +0.2 [−0.7, +1.2] | 71.1 vs 69.5 |

On the prefix, FP32 scores 71.9, against 69.8 on the rest. Nearly all of v1's 128-image gap came from tie resolution: under the common rule, 1 of 384 prefix image-runs is discordant.

The prefix was not uniformly easy for B. B − FP32 on images 0–127 against 128–999 is +0.8 / −4.6 pp for fp6_e2m3, −4.7 / −4.8 for fp6_e3m2 and −0.8 / −0.3 for fp7_e3m3. In tie-luck terms (defined in the tie section below), the prefix gap has two parts in every case:
- B was lucky: +2.9, +0.8 and +2.0 pp (z = +2.05, +0.47, +1.29).
- Wide was unlucky: −1.2, −1.5 and −2.0 pp (z = −0.84, −0.88, −1.39).

On images 128–999, B's tie luck is −0.4, +1.7 and +1.7 pp (z = −0.69, +2.25, +3.01), and wide's is +0.5, −0.1 and +0.5 pp.

### Where the two sides first diverge

Corrected in r2. The r1 numbers here (stem first on 457, 880 and 997 images) were wrong. They came from walking the layers in alphabetical order, in which the residual `add` nodes come first, and from counting those adds as stem. The table below uses execution order from the sealed v1 export graph, and each residual add belongs to its block. It matches the reviewer's independent recount.

| Case | No code differs at any layer | First difference in the stem (all at `conv1`) | … in layer1 | … in layer2–4 | B→wrong: stem first | B→correct: stem first |
|---|---:|---:|---:|---:|---:|---:|
| fp6_e2m3 | 348 | 50 | 306 | 296 | 1 of 31 | 3 of 34 |
| fp6_e3m2 | 28 | 396 | 452 | 124 | 17 of 57 | 18 of 38 |
| fp7_e3m3 | 1 | 774 | 212 | 13 | 30 of 41 | 22 of 27 |

- The ReLU and max-pool are never the first difference. After `conv1`, the most common first differences are the layer1 convolutions, for example `layer1_0_conv2` (134) and `layer1_1_conv1` (106) for fp6_e2m3, and `layer1_0_conv1` (172 and 133) for fp6_e3m2 and fp7_e3m3.
- Where the first difference falls is similar for gains and losses:
  - fp6_e3m2: stem 17 / 18, layer1 31 / 14, later layers 6 / 5, none 3 / 1 (B→wrong / B→correct).
  - fp7_e3m3: stem 30 / 22, layer1 9 / 5, layer2 2 / 0.
  - fp6_e2m3: stem 1 / 3, layer1 11 / 12, later layers 12 / 12, none 7 / 7.
  - The only visible asymmetry is in fp6_e3m2's layer1 (31 against 14). It is not tested, and the tie analysis below accounts for the overall imbalance.
- The control arm's stored codes equal the wide arm's at every layer on all 3 × 1,000 images, so there is no first divergence between them.
- Earlier, only changes with identical codes at *every* layer were attributed to ties. That understated the effect. The relevant condition is identical *fc* codes, because the logits are reconstructions of those codes. That count is given below.

## Top-1 ties between equal quantized logits (tie audit; post-hoc, exploratory)

- **Method.** `tools/scaled_bridge_gap_v1/ties.py` re-ran every diagnostic configuration on all 1,000 images per case. Each run passed the same endpoint gates as the diagnostics (below). Every configuration's own Top-5 also matched the diagnostic record on all 3 × 1,000 images.
- **What was recorded.** For every image and configuration, the audit recorded the set of classes whose fc code equals the maximum, and the Top-5 under the common rule.

| | fp6_e2m3 | fp6_e3m2 | fp7_e3m3 |
|---|---:|---:|---:|
| Images with a top-1 tie, B / wide | 192 / 185 | 297 / 301 | 179 / 178 |
| … of which the label is in the tie set, B / wide | 124 / 117 | 207 / 206 | 117 / 117 |
| Largest top-1 tie set, B / wide | 7 / 6 | 11 / 11 | 9 / 6 |
| Images with a tie across the Top-5 boundary, B / wide | 601 / 606 | 763 / 753 | 574 / 569 |
| Images whose fc codes are identical in B and wide | 716 | 293 | 73 |
| Measured top-1 changes with identical fc codes (B→wrong / B→correct / wrong→other) | 83 (20 / 22 / 41) | 61 (16 / 12 / 33) | 6 (3 / 3 / 0) |
| Measured top-1 changes with different fc codes (B→wrong / B→correct / wrong→other) | 40 (11 / 12 / 17) | 143 (41 / 26 / 76) | 107 (38 / 24 / 45) |
| Measured top-1 equals common-rule top-1, B / wide | 903 / 903 | 755 / 833 | 895 / 911 |

- **Changes with identical fc codes are pure tie-breaking.** They split evenly in sign: sign-test p = 0.88, 0.57 and 1.0.
- **Changes where the fc codes differ also involve ties.** When the codes differ, the tie *sets* still often overlap, so both rules again pick among near-equal classes. This is why the B→wrong / B→correct counts shrink to 7/9, 13/14 and 7/9 under the common rule.
- **Neither `topk` follows a simple index rule.** Among top-1 ties, B picks the largest index in 80, 174 and 96 of 192, 297 and 179 images, and the smallest in 95, 52 and 74. The engine's choice is similarly mixed. The label's relative position inside the tie set averages 0.47–0.54, so no index rule favours the label.

**How lucky each arm's own tie-breaking was.** On top-1 tie images, count the images where the rule picked the label, and compare with a uniformly random pick:

| Case | B measured: picked / expected (z, excess pp of panel) | Wide measured | B, common rule | Wide, common rule |
|---|---|---|---|---|
| fp6_e2m3 | 57 / 56.8 (z = +0.04, +0.02) | 56 / 53.7 (+0.44, +0.24) | 50 / 56.8 (−1.24) | 48 / 53.7 (−1.06) |
| fp6_e3m2 | 96 / 80.5 (z = **+2.28**, **+1.55**) | 79 / 82.1 (−0.46, −0.31) | 72 / 80.5 (−1.26) | 75 / 82.1 (−1.05) |
| fp7_e3m3 | 72 / 54.5 (z = **+3.27**, **+1.75**) | 56 / 54.5 (+0.28, +0.15) | 58 / 54.5 (+0.65) | 58 / 54.5 (+0.66) |

- **B's reported top-1 in the two cases that triggered diagnostics contains about 1.5–1.8 pp of favourable tie resolution.** That is the gap. A fresh tie-break does not reproduce it. Applying any one rule to both sides gives a wide − B within ±0.3 pp of zero:
  - B's own `topk` on both sides (X − nonmac32): 0.0, +0.3 and −0.1;
  - the common rule: +0.2, +0.1 and +0.2;
  - the random-tie expectation: +0.09, −0.04 and +0.20.
- **Why B was lucky cannot be settled here.** Selection is one candidate explanation, and it is weaker than r1 said:
  - All three cases were selected on earlier results, yet fp6_e2m3 shows no luck at all (z = +0.04).
  - For selection alone to produce z = +3.3, the candidates would have had to be nearly tied on true accuracy.
  - `topk` never sees the label. However, B's batched CUDA `topk` is deterministic in both values and indices, so a structural cause is not excluded.
  - B's `topk` applied to the exact codes (X − nonmac32) is also above the random-tie expectation: z = −0.12, +2.77 and +2.72 (excess −0.1, +1.9 and +1.5 pp).
  - The engine's CPU `topk` applied to B's codes (B + nonmac64) is close to random: z = +1.32, +0.51 and +0.65.
  - So the luck follows B's `topk` call rather than B's codes. It is not independent evidence, though, because the tie sets overlap heavily: 159 of 192, 226 of 297 and 138 of 179 of B's top-1 tie sets are identical under X − nonmac32.
  - **The held-out run should pre-register B's tie-luck z (measured rule against random) as a test.**
- **The tie rule alone moves B's own accuracy by up to 2.4 pp.** For fp6_e3m2, B is 65.3 measured against 62.9 under the common rule, with a random-tie expectation of 63.75.
- **What this means for the project.** Every quantized-logit accuracy in the project that comes from the B simulator carries this tie-breaking component. Its per-case standard deviation is about 0.5–0.7 pp, which is sd_random ≈ 5.3–6.8 images out of 1,000. Its sign varies from case to case. Paired comparisons with different tie rules on the two sides inherit it twice.

## Mechanism diagnostics (step 4; protocol `scaled-bridge-gap-diag-1`; triggered for fp6_e3m2 and fp7_e3m3, fp6_e2m3 run as the untriggered control)

**Harness.** `tools/scaled_bridge_gap_v1/diagnostics.py` is a diagnostic tool and **is not bit-certified** between its endpoints. It re-implements both semantics in torch on CUDA, keeping the original batch-of-8 membership. Each difference can be switched independently:
- `in`: input normalisation in FP32 or binary64.
- `mac`: three settings.
  - B: FP32 cuDNN convolution of FP32-reconstructed operands.
  - R64: the same operands in binary64, which removes FP32 accumulation only.
  - X: the exact code-domain dot, with the scales applied afterwards in binary64.
- `store`: store normalisation in FP32 or binary64, at the outputs of MAC nodes (conv and fc) only.
- `nonmac`: residual, ReLU, pooling and logit reconstruction in FP32 or binary64. It also covers the store normalisation of those non-MAC nodes.
- `mac` R64 also removes the FP32 bias add. `mac` X additionally factors the scales out of the dot.

Because the logits are FP32 when `nonmac` is B, `nonmac` also selects which `topk` is used: B's batched CUDA FP32 call or the engine's per-image CPU binary64 call. `nonmac` flips no code anywhere (below), so in practice only its `topk` part acts.

Only `mac` and `store` change codes. In propagated flips the design behaves as a complete 2 × 2: X − mac B equals B + store64, and X − store32 equals B + mac X.

**Endpoint gates.** On every image, the all-B configuration had to reproduce the B replay's stored-code hash at every node and B's Top-5, and the all-X configuration had to reproduce the sealed wide arm's hashes and Top-5. Both gates passed on all 3 × 1,000 images; a failure would have stopped the run. Zero canonicalisation is shown to be harmless by the same all-B endpoint.

### Accuracy per configuration (top-1 %, then difference from B in pp; measured / common rule)

Single switches applied to B:

| Configuration | fp6_e2m3 | fp6_e3m2 | fp7_e3m3 |
|---|---|---|---|
| B | 66.2 / 65.5 | 65.3 / 62.9 | 69.7 / 68.3 |
| B + in64 | +0.0 / +0.0 | +0.0 / +0.0 | +0.0 / +0.0 |
| B + mac R64 (FP32 accumulation removed) | −0.8 [−1.6, 0.0] / −0.3 [−1.0, +0.3] | +0.0 [−1.0, +1.0] / +0.3 [−0.7, +1.3] | −0.2 [−1.0, +0.6] / −0.2 [−1.0, +0.6] |
| B + mac X (exact MAC) | −0.3 [−1.1, +0.5] / +0.1 [−0.7, +0.9] | −0.3 [−1.3, +0.8] / +0.5 [−0.5, +1.6] | −0.2 [−1.0, +0.6] / −0.3 [−1.1, +0.5] |
| B + store64 | −0.2 [−0.7, +0.3] / +0.0 [−0.5, +0.5] | −0.8 [−1.5, −0.2] / +0.3 [−0.1, +0.8] | +0.0 [−0.3, +0.3] / +0.1 [0.0, +0.3] |
| B + nonmac64 (codes identical to B; only the `topk` changes) | **+0.7** [−0.8, +2.2] / +0.0 | **−1.2** [−2.9, +0.5] / +0.0 | **−1.4** [−3.0, +0.2] / +0.0 |

Single switches reversed from X (difference from B):

| Configuration | fp6_e2m3 | fp6_e3m2 | fp7_e3m3 |
|---|---|---|---|
| X (all exact) | +0.3 / +0.2 | −1.9 / +0.1 | −1.4 / +0.2 |
| X − in32 | +0.3 / +0.2 | −1.9 / +0.1 | −1.4 / +0.2 |
| X − mac B | +0.3 / +0.0 | −1.3 / +0.3 | −1.6 / +0.1 |
| X − store32 | +0.2 / +0.1 | −1.6 / +0.5 | −1.3 / −0.3 |
| X − nonmac32 (exact codes, B's `topk`) | **+0.0** / +0.2 | **+0.3** / +0.1 | **−0.1** / +0.2 |

Stage hybrids: X in one stage only, B everywhere else. Each cell gives the difference from B, measured / common rule, then the number of images whose fc codes differ from B's.

| Stage | fp6_e2m3 | fp6_e3m2 | fp7_e3m3 |
|---|---|---|---|
| stem (`x`, `conv1`, `bn1`, `relu`, `maxpool`) | +0.0 / +0.0; 5 | −0.1 / +0.1; 53 | +0.1 / −0.1; 132 |
| layer1 (with `add`, `add_1`) | −0.2 / +0.1 [−0.6, +0.8]; 192 | −0.2 / −0.3 [−1.2, +0.6]; 462 | +0.2 / +0.2 [−0.5, +0.9]; 732 |
| layer2 (with `add_2`, `add_3`) | +0.1 / +0.0; 78 | −0.1 / −0.2 [−0.8, +0.4]; 302 | +0.5 / +0.2 [−0.3, +0.8]; 495 |
| layer3 (with `add_4`, `add_5`) | +0.1 / +0.1; 30 | +0.2 / +0.0; 154 | −0.1 / +0.2; 314 |
| layer4 (with `add_6`, `add_7`) | +0.0 / +0.0; 14 | +0.1 / +0.0; 43 | +0.0 / +0.0; 105 |
| head (`avgpool`, `flatten`, `fc`, logits) | +0.7 / +0.0; 0 | −1.2 / +0.0; 0 | −1.4 / +0.0; 3 |

- **Stage assignment (corrected in r2).** diag-1 ran these hybrids with the residual adds counted as stem: its "stem" also switched all eight adds, and its "layerN" did not switch its own adds.
- **The rerun** (`scaled-bridge-gap-diag-1a`; 3 shared-mode jobs of 295, 289 and 200 s; all exited 0) places each add in its block. Its all-B endpoint was gated on every image, and its head hybrid was checked against the diag-1 record.
- **Result of the rerun.** On all 3 × 1,000 images, every corrected hybrid has the same fc codes and the same top-1 as the as-run hybrid, and the same propagated-flip totals. Moving the adds therefore changed nothing. This is expected, because non-MAC arithmetic flips no code. The table above is both the as-run and the corrected result; only its labels changed.
- **Effect sizes.** Under the common rule, every interval includes zero. The largest common-rule point estimate is 0.3 pp among the stage hybrids and 0.5 pp across all diagnostic configurations (both in absolute value). The intervals reach from −1.2 to +1.6 pp.
- **The head hybrid** changes no code in fp6_e2m3 and fp6_e3m2, so there it only swaps the `topk` and equals B + nonmac64. In fp7_e3m3 it changes the fc codes of 3 images, with propagated flips of 1 up and 2 down. They come from the fc accumulation: nonmac64 (which covers the binary64 pooling) flips no code locally, while mac R64 flips fc codes locally 1 up and 2 down, the same counts. Its common-rule accuracy is unchanged.

### Local code flips (teacher-forced: each switch applied to B's own inputs at every node; summed over 1,000 images)

| Switch | fp6_e2m3 up / down (sign p) | fp6_e3m2 | fp7_e3m3 |
|---|---|---|---|
| in64 | 0 / 0 | 0 / 0 | 0 / 0 |
| nonmac64 | 0 / 0 | 0 / 0 | 0 / 0 |
| mac R64 | 625 / 396 (8e−13) | 1,909 / 1,674 (9e−5) | 3,834 / 4,100 (0.003) |
| mac X | 640 / 468 (3e−7) | 1,927 / 1,667 (2e−5) | 3,885 / 4,204 (4e−4) |
| store64 | 136 / 74 (2e−5) | 356 / 299 (0.03) | 383 / 503 (6e−5) |

- **The flips are rare.** At most about 8,000 stored codes change per 1,000 images. Each image has about 5.9 million distinct stored codes (5,897,192, not counting identity and flatten nodes, which repeat their input), so the per-code rate is around 1e−6. The flips are spread over the stem and layer1–4. The fc layer flips at most 3 codes in 1,000 images.
- **The pooled sign tests are "significant", but the direction is not consistent.**
  - FP32 accumulation leans toward smaller codes than binary64 in fp6_e2m3 and fp6_e3m2, and toward larger codes in fp7_e3m3.
  - FP32 store normalisation leans the same way as FP32 accumulation within each case, so its direction also flips between the FP6 cases and FP7.
  - The tests treat codes as independent, which they are not (the codes come from shared images and channels), so the p-values overstate the evidence.
  - None of these leanings moves accuracy: every single-switch row is within 0.5 pp of B under the common rule.
- **Propagation is large and, from layer1 on, symmetric.** Through the network these few local flips grow into 21, 85 and 150 million propagated code differences (X versus B, over 1,000 images). In layer1, layer2, layer3, layer4 and the head, up and down counts agree to within 0.63 % in every case.
  - The stem (`conv1` to `maxpool`, corrected stages) is the exception: up / down 85 / 43 (fp6_e2m3), 731 / 700 (fp6_e3m2) and 2,014 / 2,429 (fp7_e3m3), that is 49 %, 4 % and 17 % apart (`diagnostics.json`, `configs.X.propagated_flips_per_stage.stem`).
  - The stem holds under 0.003 % of all propagated differences, and its leaning matches the local-flip leanings above (X above B in the two FP6 cases, below B in fp7_e3m3). No conclusion changes.
  - r1 and r2 said "within 0.7 % at every stage". That was true only under the as-run stage rule, which put the eight residual adds in the stem.

### Mechanism conclusion

- **No single contract difference accounts for the residual 1–2 points.**
  - Input normalisation and non-MAC arithmetic flip no codes.
  - FP32 accumulation, the exact MAC and store normalisation each flip about 1e−6 of the codes, with no consistent direction across cases.
  - Under a fixed tie rule, their point estimates lie within ±0.5 pp of B. Their intervals reach from −1.1 to +1.6 pp, so effects of about 1 pp are not excluded at n = 1000.
- **On this panel, the measured −1.9 and −1.4 pp come from tie-breaking between equal quantized logits.**
  - Changing only the `topk` (B + nonmac64) reproduces −1.2 and −1.4 pp with identical codes.
  - Giving the exact codes B's `topk` (X − nonmac32) gives +0.3 and −0.1 pp.
- **B's own tie resolution was favourable in the two triggered cases** (z = +2.3 and +3.3 against random tie-breaking). Selection on B's results is one possible reason. A structural property of B's batched `topk` is not excluded (see the tie section).
- **What remains looks like symmetric near-boundary rounding noise.** Once a common rule is applied, wide − B is +0.1 to +0.2 pp, with 16–27 discordant images per 1,000 and balanced directions (7/9, 13/14, 7/9).

## What the paper can and cannot claim

- **Not supported:** "the simulator overstates exact accuracy by 4–5 points". At n = 1000 the measured differences are +0.3, −1.9 and −1.4 pp, and with a common tie rule they are +0.2, +0.1 and +0.2 pp.
- **Also not supported:** "the simulator is 1–2 points optimistic". On this panel, that residual is a tie-breaking artefact plus favourable luck in B's own tie resolution, not a property of the arithmetic. The common-rule intervals still allow about 1 pp either way, so an arithmetic effect of that size is not excluded.
- **Supportable, worded as development evidence:** "On the 1k development screen, with one deterministic tie rule for both, exact execution of the scaled contract and the FP32 QDQ simulator differ by +0.2, +0.1 and +0.2 pp top-1 (pointwise 95 % intervals [−0.6, +1.0], [−0.9, +1.1], [−0.6, +1.0]). No individual arithmetic difference (input or store normalisation, FP32 accumulation, non-MAC arithmetic) has a point estimate beyond ±0.5 pp, and none is resolved from zero (intervals −1.1 to +1.6 pp)."
  - Do not round this to "within ±1 pp". The fp6_e3m2 interval reaches +1.1, and the intervals are pointwise.
- **Supported, with its definition stated:** "a sequential FP32 FMA accumulator over the code-grid integer products, with the scales applied afterwards in binary64 (the v1 contract's control arm), gives the same stored codes and predictions as exact accumulation for these three cases."
  - At n = 1000 per case there are zero changed predictions and zero changed stored codes. The one-sided 95 % upper bound on the per-image prediction-change rate is 0.30 % per case.
  - For fp6_e2m3 and fp6_e3m2 the FP32 dot results equal the exact ones on every image, so FP32 never rounded. That shows the significand is wide enough for these products. It is not a test of rounding behaviour.
  - For fp7_e3m3, the dot results differ on 451 images. The equivalence holds after the store quantisation.
  - **This does not cover the simulator's own FP32 accumulation** (cuDNN over FP32-reconstructed operands). That is a different computation, and it is not code-identical: B + mac R64 flips 1,021, 3,583 and 7,934 codes locally. Its fc codes equal B's on only 746, 309 and 83 of 1,000 images.
- **Methodological claim the paper should make:** "Sub-8-bit logits are frequently tied (18–30 % of images at top-1, 57–76 % at the Top-5 boundary). Reported accuracies must fix a tie rule, or report the random-tie expectation, because the rule alone moves a single configuration's top-1 by up to 2.4 pp at n = 1000."
- **Scope:** ResNet18, maxabs, three formats, one development panel. The cases were selected on earlier results, and the tie audit is post-hoc. No other model, recipe or accumulator width is covered.

## What would settle it, and what it costs (proposed, not run)

The remaining question is whether the exact contract and the simulator differ by more than a few tenths of a point, under one pre-registered tie rule. On the 1k panel, the common-rule interval is already about ±1 pp, because only 1.6–2.7 % of images are discordant. Settling a smaller effect, or confirming that the measured 1–2 pp gap is gone, needs data not used for selection.

The only such data on this machine are the 9,000 images of the 10k list outside the 1k screen. They are reserved as held-out data until the hypotheses are frozen (orchestrator notice 2, 2026-10-01), so this run is proposed only.
- **Cost from the measured per-1,000 job times** (wide about 340 s, B replay about 350 s, control about 360 s): wide plus B replay for 3 cases × 9,000 images is about 3 × 9 × 690 s ≈ 5.2 GPU-hours. Adding the control arm brings it to about 7.9 GPU-hours.
- **Precision.** With common-rule discordance of about 2–3 %, the 95 % half-width of wide − B would be about ±0.35 pp. Measured-rule comparisons would be about ±0.65 pp, which is one more reason not to use them.
- **Pre-registration** should fix the following before the held-out images are touched:
  - the tie rule, or the random-tie expectation as the primary outcome;
  - the three cases and the analysis;
  - B's tie-luck z (measured `topk` against random tie-breaking) as a test, which separates selection from a structural property of B's batched `topk`.
- **A cheaper alternative.** If the only question is the tie-rule-corrected difference, the wide and B arms are enough (about 5.2 GPU-hours). The ~1e−6 code-flip rates do not need the diagnostic harness again.
- **Logging.** The arms should record the fc codes, so that any tie rule can be applied offline without rerunning.

## Notes for other lanes

- **L2 (scaled bridge v2 contract):** the v2 contract should specify the ranking rule. Recommended: order by fc stored code, then by class index ascending, applied to the codes rather than to reconstructed floats. The B simulator side of any v2 comparison should be re-ranked with the same rule from its stored fc codes, and the per-image fc code hash (or the codes themselves) should be recorded.
- **L1 (quality-baseline repair):** B accuracies on low-bit logits include a tie-breaking component. Its per-case SD is about 0.5–0.7 pp at n = 1000, and between rules it reached 2.4 pp here. Comparisons that are B against B are paired, and the rule is shared, but they still inherit tie luck unless a fixed rule or the random-tie expectation is used.

## Reproduce

```
tools/run/scaled_bridge_gap_chain.sh          # nine GPU jobs; resumable; checks v1 on images 0..127
.venv-b/bin/python -m tools.run.scaled_bridge_gap analyze
tools/run/scaled_bridge_gap.sh diagnose run <format> 0 1000        # or tools/run/scaled_bridge_gap_diag_chain.sh
.venv-b/bin/python -m tools.run.scaled_bridge_gap diagnose summarize
tools/run/scaled_bridge_gap_ties_chain.sh                          # tie audit, one job per case
.venv-b/bin/python -m tools.run.scaled_bridge_gap ties summarize
tools/run/scaled_bridge_gap_stagefix_chain.sh                      # corrected stage hybrids (diag-1a), one job per case
.venv-b/bin/python -m tools.run.scaled_bridge_gap stagefix summarize
```

- `tools/run/scaled_bridge_gap.sh` now launches through `artifacts/agent_orchestration/gpu_run.sh` in shared mode. The 1k arms and the fp6_e3m2 diagnostic ran under the earlier exclusive-lock form. Results do not depend on the mode, because every record is gated against sealed hashes.
- The GPU jobs use `.venv-b`, the CUDA build of torch 2.3.0 in which v1 and B were run. `.venv` has only a CPU build of torch.
- Unit tests (`.venv`): `tests/unit/test_scaled_bridge_gap_v1.py` and `tests/unit/test_scaled_bridge_gap_ties.py`. These include r2 regression tests: first divergence in execution order against alphabetical order, and the corrected stage of every residual add.
