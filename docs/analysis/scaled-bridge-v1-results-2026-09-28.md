# Scaled FP6/FP7 arithmetic bridge: audited paired pilot

Pilot completed 2026-09-28; audited 2026-09-29. ResNet18/maxabs; selected using earlier development evidence. These panels are not independent final confirmation.

All three cases passed the independent wide-graph witness, actual-node rational dot checks and eight-image CPU/CUDA gates in both arms. Primitive conformance passed 98,768 checks. Ledger: 602.1 /14,400 worker-seconds charged, including setup, probes, both backends and both arms; 0 failed/interrupted attempts. Coding and compilation excluded.

| Format | Images/arm | B top-1 | Wide top-1 | FP32-control top-1 | FP32 baseline | Wide − B | Control − wide |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp6_e2m3 | 128 | 72.656% | 67.969% | 67.969% | 71.875% | -4.688 pp | +0.000 pp |
| fp6_e3m2 | 128 | 67.188% | 63.281% | 63.281% | 71.875% | -3.906 pp | +0.000 pp |
| fp7_e3m3 | 128 | 71.094% | 66.406% | 66.406% | 71.875% | -4.688 pp | +0.000 pp |

## 1. Was the B anchor reproduced?

Yes, each selected case reproduces all saved candidate and paired FP32 ordered Top-5 lists for its panel, using the original runtime and original batch membership of eight. New layer traces are separate artifacts. The historical 1,000-image B top-1 anchors remain 66.2%,65.3%,69.7%, versus70.1% FP32; those are not the current panel scores.

## 2. What changed when transferring B to the scaled code-domain contract?

B performs FP32 normalization, operand reconstruction and framework reduction. The new contract retains weight codes and scale bits, factors scales after the exact code-domain dot, and uses explicitly rounded FP64 postops and stores. It canonicalizes stored zero. Candidate-minus-B therefore measures the combined transfer; it is not an accumulator-only effect.

**fp6_e2m3:** top-1 delta -4.688 pp (paired pointwise95% [-9.375, -0.78125]), top-5 accuracy delta +0.781 pp. Changed Top-1 predictions: 15; changed ordered Top-5 lists: 79.
First differing layer codes in FX execution order: {"layer2_0_conv2": 12, "layer2_0_conv1": 5, "layer1_1_conv2": 7, "layer1_0_conv2": 18, "layer1_1_conv1": 20, "layer4_1_conv1": 4, "conv1": 12, "layer2_1_conv1": 1, "layer3_0_conv1": 2, "layer4_0_conv1": 2, "layer4_0_downsample_0": 1, "layer3_0_conv2": 1, "layer4_1_conv2": 1, "layer3_1_conv1": 2}.
Clipping counts (wide/B): 2/2.

**fp6_e3m2:** top-1 delta -3.906 pp (paired pointwise95% [-9.375, 1.5625]), top-5 accuracy delta +0.781 pp. Changed Top-1 predictions: 28; changed ordered Top-5 lists: 106.
First differing layer codes in FX execution order: {"layer1_0_conv2": 22, "layer1_0_conv1": 22, "layer1_1_conv1": 12, "conv1": 44, "layer2_0_conv1": 4, "layer1_1_conv2": 10, "layer2_0_conv2": 4, "layer3_1_conv2": 1, "layer2_1_conv1": 4, "layer2_0_downsample_0": 1, "layer4_0_conv2": 1}.
Clipping counts (wide/B): 0/0.

**fp7_e3m3:** top-1 delta -4.688 pp (paired pointwise95% [-9.375, -0.78125]), top-5 accuracy delta -1.562 pp. Changed Top-1 predictions: 13; changed ordered Top-5 lists: 87.
First differing layer codes in FX execution order: {"layer1_1_conv1": 3, "layer1_0_conv2": 6, "conv1": 95, "layer1_0_conv1": 22, "layer2_0_conv1": 1, "layer2_0_downsample_0": 1}.
Clipping counts (wide/B): 1/1.

## 3. What changed from accumulator arithmetic alone?

Both candidate arms use the same code graph, external scales, bias, postops and store rules. The matched control changes only the dot to sequential FP32 fused multiply-add. Reduction order and each rounding boundary are frozen.

**fp6_e2m3:** control-minus-wide top-1 +0.000 pp (paired pointwise95% [0.0, 0.0]); changed Top-1/ordered Top-5: 0/0. Full numerical records agree on 128/128 images; 0 images have a differing raw dot result.
First differing stored codes: {}.

**fp6_e3m2:** control-minus-wide top-1 +0.000 pp (paired pointwise95% [0.0, 0.0]); changed Top-1/ordered Top-5: 0/0. Full numerical records agree on 128/128 images; 0 images have a differing raw dot result.
First differing stored codes: {}.

**fp7_e3m3:** control-minus-wide top-1 +0.000 pp (paired pointwise95% [0.0, 0.0]); changed Top-1/ordered Top-5: 0/0. Full numerical records agree on 71/128 images; 57 images have a differing raw dot result.
First differing stored codes: {}.

A zero-width empirical bootstrap interval when no outcomes differ is descriptive of this panel; it is not proof of population equivalence.
The E2M3 certificate is stronger: all21 MAC layers have absolute grid-prefix bounds below2^24 (largest3,527,832), so FP32 is exact for the entire admitted code domain. It is a negative control for FP32 accumulator loss under this scaled contract. The corresponding maximum signed integer widths are23bits(E2M3),29bits(E3M2),31bits(FP7); per-channel proofs are in each export.

## 4. Which cases are ready for the next accumulator sweep?

New scaled-contract gates and the frozen quality rule support: fp6_e2m3, fp6_e3m2, fp7_e3m3. No old unscaled native admission is extended; in particular, the old unscaled FP7 acceptance is still absent.

Next sweep specification only; no new sweep launched:

- FP16: code-level exact product plus prior accumulator, one RNE after every fused MAC, gradual subnormals and IEEE infinity/NaN. Record nonfinite failures; never silently clamp them.
- Custom21 float: sign1/exponent8/bias127/fraction12, RNE after each exact MAC, gradual subnormals and IEEE specials. It retains FP32 range while testing an intermediate precision. Freeze the manifest before running.
- Integer/fixed grid: signed zero-point0 accumulator, exact product grid scale2^(-2×codebook shift). Use each exported per-channel absolute-sum width as the lossless reference, then explicitly narrower widths with endpoint saturation after every add. No wrap. Apply the same retained external scales and FP64 postops after exact power-of-two conversion.

Every future arm needs independent rational witnesses and paired native gates before quality evaluation.

## Runtime and continuation

| Format | Wide CUDA median | Control CUDA median | Wide CPU median | Control CPU median |
|---|---:|---:|---:|---:|
| fp6_e2m3 | 0.300s | 0.297s | 0.828s | 0.820s |
| fp6_e3m2 | 0.317s | 0.317s | 0.879s | 0.886s |
| fp7_e3m3 | 0.338s | 0.337s | 0.868s | 0.860s |

These are measured per-image execution times including node quantization and trace hashes, excluding preprocessing/setup and independent rational witness overhead. Detailed MAC/non-MAC/preprocess/setup timings and all paired confidence intervals are in the machine result.

One-line resume and audited report:

```bash
bash tools/run/scaled_bridge.sh
```

Sealed audited results: `results/summaries/scaled-bridge-v1-audited/18723721117929fa5d6d6302aa192d126ea881ee7059f5792b65152bd78c1af2/summary.json`.
The executor-generated preliminary report uses JSON key order for first-divergence summaries; this audit supersedes that field with the exported FX execution order. All numerical inference records and their identities are unchanged.
The original B source identity is verified again by the audit; original B evidence and the historical E1 budget were not modified.
Complete paired Top-1/Top-5 comparisons and pointwise confidence intervals: `results/summaries/scaled-bridge-v1-audited/18723721117929fa5d6d6302aa192d126ea881ee7059f5792b65152bd78c1af2/comparisons.csv`.
Archived enrolled source, libraries and compiler manifests: `artifacts/scaled_bridge_v1/implementations/5e11c33940783230a36ccda4ff624628be5137ca0bc345c8d2a5029f62154e51`.
