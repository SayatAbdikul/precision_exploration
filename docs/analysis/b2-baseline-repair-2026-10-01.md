# Experiment B2: repairing the PTQ baseline of the breadth simulator (2026-10-01)

Status: complete for stages S0-S6 of `public/experiments/configs/breadth-study/b2-baseline-repair-protocol-v1.json`.
All numbers are **development evidence**: the recipe was selected on dev512 (the first 512 images of the frozen
1k screen), so the 1k exit test overlaps the selection panel. The 488 images that were never used for selection
are reported separately as `heldout488`. The ImageNet 10k evaluation list and the 50k set were not touched.

Revision r2 (same day, after independent review 1, `artifacts/agent_orchestration/handoffs/L1-baseline-review.md`):
three passages that contradicted the saved evidence were corrected (section 2 items 2, 4 and 6; the residual
diagnosis in section 5), the vendor full-range collapse was explained (section 4), and the limits were extended
(section 11). No B2 measurement changed and no numeric source file was edited; the new evidence is under
`artifacts/experiment_b2/verification/`.

Revision r3 (same day, after independent review 2): the statement that the bias-correction residual is the same at
batch 32 and batch 8 was false for ResNet18 and is corrected (section 11); the QNNPACK ReLU6 failure is described
as what it is, a memory-layout defect (section 4); the MobileNetV3 QNNPACK arm is marked as a lower bound; and two
effects on the readout were measured for the first time: **the batch size changes ResNet18 predictions, and
quantized logits tie, so part of every top-1 figure is decided by the tie-breaking of `topk`** (new section 13).
No sealed B2 number changed and no numeric source file was edited. Protocol addendum:
`public/experiments/configs/breadth-study/b2-baseline-repair-protocol-v1-r3-addendum.json`. Final r3 records
carry the tag `r3c`.

All intervals in this document are **pointwise** 95 percent paired image-bootstrap intervals (10000 resamples,
`pointwise_95_interval_pp` of `tools.analysis.b_stage_balanced_comparisons.compare_records`; the summaries store
them under the shorter key `interval_pp`). No multiplicity correction is applied in any table.

## 1. Result in brief

INT8 top-1 on the frozen 1k screen (pointwise paired 95 percent bootstrap intervals, 10000 resamples):

| model | FP32 | B v1 max-abs | B v1 percentile | **B2 default** | B2 minus FP32 | vendor FX x86 default | B2 minus vendor |
|---|---|---|---|---|---|---|---|
| ResNet18 | 70.1 | 69.8 | 62.4 | **69.0** | -1.1 [-2.0, -0.2] | 69.4 | -0.4 [-1.5, +0.7] |
| MobileNetV2 | 72.1 | 66.5 | 66.4 | **72.0** | -0.1 [-0.8, +0.7] | 66.5 | +5.5 [+3.2, +7.7] |
| MobileNetV3-Large | 75.0 | 4.1 | 68.9 | **71.5** | -3.5 [-5.4, -1.5] | 65.7 | +5.8 [+3.4, +8.1] |

- Exit target ("within about 1 point of FP32, or at least on par with the vendor anchor"): MobileNetV2 meets the
  FP32 clause. ResNet18 is at -1.1 and statistically on par with the vendor anchor. **MobileNetV3-Large does not
  meet the FP32 clause (-3.5 points)**; it meets the target only through the vendor clause (+5.8 over the shipped
  x86 default, +2.0 [-0.1, +4.0] over the stronger QNNPACK full-range arm at 69.5, which itself has one wrong
  kernel and is a lower bound for a correct 8-bit vendor flow, section 4). That is not a met target.
- The +5.5 on MobileNetV2 is measured against a weak anchor. The shipped x86 default resolves activations with
  7 bits over pre-activation ranges; a working 8-bit vendor arm (QNNPACK full range with its defective ReLU6
  kernel replaced by an exact clamp, section 4) scores 71.5, and B2 is on par with it: +0.5 [-0.9, +2.0].
- On ResNet18 the repaired recipe is nominally 0.8 below v1 max-abs (69.0 versus 69.8, interval [-2.3, +0.6]):
  v1 max-abs was already sound there and nothing in B2 improves it.
- Under the repaired recipe the format ordering changes: Posit8 moves from about 0 percent in v1 to the top of
  the sentinel list on all three models (69.8 / 72.1 / 73.4). The cause is confirmed by the occupancy audit (section 7).
- Readout limit found in r3 (section 13): the logits are quantized like every other tensor, so the largest logit
  is often shared by several classes and `topk` picks one in an unspecified way. At INT8 this is small (13 to 42
  tied images per model; the sealed 69.0 / 72.0 / 71.5 become 69.2 / 72.3 / 71.6 under expected credit, and the
  exit statement is unchanged), but for the minifloat and log formats 10 to 19 percent of the images are tied
  and the sealed top-1 differs from the expected-credit top-1 by up to 0.8 point (1.0 for INT6; up to 1.7 under a
  lowest-index rule). Differences of about a point between formats in section 6 are therefore not resolved.
- The default recipe is not safe at 4 bits: INT4 and NF4 collapse to chance, and on ResNet18 INT4 the bias
  correction step itself costs 28 points (section 8). This is reported, not repaired (stop rule).

## 2. What was wrong in v1 (diagnosis, INT8, dev512)

Arms change one switch of `v1_maxabs` at a time; the cumulative ladder adds (a) to (e) in order. Entries are
top-1 and the paired difference to `v1_maxabs` in points with its 95 percent interval. Full tables (dev128 and
dev512, top-5, McNemar p, differences to FP32): `results/summaries/b2-baseline-repair-v1/diagnosis-512.{json,csv}`.

| arm | ResNet18 (FP32 71.09) | MobileNetV2 (FP32 73.83) | MobileNetV3-L (FP32 76.95) |
|---|---|---|---|
| v1_maxabs (reference) | 70.51 | 68.55 | 4.10 |
| v1_percentile_99_9 | 65.43, -5.08 [-8.20, -1.95] | 67.77, -0.78 [-4.49, +2.93] | 70.12, +66.02 [+61.72, +70.12] |
| (a) fused boundaries, ReLU/ReLU6 only | 70.12, -0.39 [-2.15, +1.37] | **73.83, +5.27 [+2.34, +8.20]** | 6.05, +1.95 [0.00, +4.10] |
| (a) fused, every nonlinearity | same as above | same as above | 23.63, +19.53 [+15.82, +23.24] |
| (b) unsigned codes | 68.95, -1.56 [-3.52, +0.39] | 68.36, -0.20 [-2.15, +1.76] | 3.52, -0.59 [-2.34, +1.17] |
| (c) activation range MSE | 70.31, -0.20 [-2.15, +1.76] | 72.46, +3.91 [+0.98, +7.03] | **72.07, +67.97 [+63.87, +72.07]** |
| (c) activation range percentile | 66.02, -4.49 [-7.81, -1.17] | 67.19, -1.37 [-5.08, +2.15] | 70.51, +66.41 [+62.11, +70.51] |
| (d) weight range MSE | 68.55, -1.95 [-3.71, -0.20] | 67.77, -0.78 [-3.12, +1.56] | 5.27, +1.17 [-0.78, +3.12] |
| (d) weight range percentile | 67.77, -2.73 [-4.69, -0.78] | 68.16, -0.39 [-2.73, +2.15] | 3.91, -0.20 [-1.95, +1.56] |
| (e) empirical bias correction | 67.97, -2.54 [-4.49, -0.59] | 70.12, +1.56 [-1.17, +4.49] | 47.66, +43.55 [+39.06, +48.05] |
| (g) FP32 logits | 70.31, -0.20 [-1.17, +0.78] | 68.16, -0.39 [-1.17, +0.39] | 3.91, -0.20 [-0.59, 0.00] |
| (g) FP32 input | 69.34, -1.17 [-2.54, +0.20] | 68.36, -0.20 [-2.34, +1.95] | 4.69, +0.59 [-1.17, +2.34] |
| (g) v1 percentile with FP32 logits | 68.55 (percentile alone 65.43) | 73.05 (percentile alone 67.77) | 72.27 (percentile alone 70.12) |
| ladder 1: (a) | 70.12 | 73.83 | 6.05 |
| ladder 2: + (b) | 70.31 | 73.05 | 8.01 |
| ladder 3: + (c) | 69.92 | 73.63 | 73.24 |
| ladder 4: + (d) | 69.73 | 73.24 | 72.27 |
| ladder 5: + (e) = default | 69.53, -0.98 [-2.93, +0.98] | 73.05, +4.49 [+1.56, +7.42] | 72.66, +68.55 [+64.45, +72.66] |
| ladder 5 with every nonlinearity fused | same | same | 73.05 |
| ladder 5 + CLE (f) | 69.34 | 73.05 (no eligible pair) | 72.46 |
| ladder 4 + CLE (f) | 70.31 | 73.24 (no eligible pair) | 71.09 |

What the hypotheses turned out to be:

1. **Double quantization at the pre-activation boundary is the MobileNetV2 defect.** v1 quantizes the conv output
   with its own (wide, signed) range and then the ReLU6 output again. Quantizing once after the nonlinearity
   closes the whole 5.3-point gap by itself. Confirmed.
2. **Outlier-driven max-abs ranges are the MobileNetV3 defect.** No arm that keeps max-abs activation ranges
   exceeds 51.6 on dev512: fusing every nonlinearity gives 23.63, bias correction 47.66, the full ladder with
   max-abs activation ranges (`cum5_act_maxabs`) 51.56, and every other max-abs arm stays between 3.5 and 8.0. The MSE scale search alone brings it to 72.07. Confirmed.
3. **The percentile recipe clips the logits.** Leaving the logits in FP32 under the v1 percentile recipe returns
   +3.1 (ResNet18), +5.3 (MobileNetV2) and +2.2 (MobileNetV3) points on dev512. On MobileNetV2 this is the whole
   loss of the percentile recipe. Confirmed. (Under max-abs or MSE ranges the logit boundary costs nothing.)
4. **Unsigned codes, per-channel weight MSE and bias correction are not supported at INT8.** Around the full
   ladder every leave-one-out interval on dev512 includes zero for every model. As single switches on top of
   v1 max-abs they are worse on ResNet18, with pointwise intervals that exclude zero: weight MSE -1.95
   [-3.71, -0.20], weight percentile -2.73 [-4.69, -0.78], bias correction -2.54 [-4.49, -0.59] (unsigned alone
   -1.56 [-3.52, +0.39]). They matter at 6 bits (section 8). Not confirmed at INT8.
5. **Cross-layer equalization gives nothing.** It was triggered by the protocol (ResNet18 and MobileNetV3 remained
   more than 1.0 point below FP32 on dev512). It is implemented only for exactly function-preserving
   conv-ReLU-conv pairs (ResNet18 8 pairs, MobileNetV3 16, MobileNetV2 none: ReLU6 and Hardswish are not
   positively homogeneous). Null result; it is not part of any frozen recipe.
6. **No B2 component is supported by ResNet18 INT8 evidence, and five arms are worse than v1 max-abs there**
   with a pointwise interval that excludes zero (dev512, no multiplicity correction): `v1_percentile_99_9` -5.08
   [-8.20, -1.95], activation percentile -4.49 [-7.81, -1.17], weight MSE -1.95 [-3.71, -0.20], weight percentile
   -2.73 [-4.69, -0.78] and bias correction -2.54 [-4.49, -0.59]. The other single switches and every ladder
   step have intervals that include zero. An arm changes 14 to 114 of the 512 top-1 predictions relative to v1
   max-abs (10 to 26 relative to the full ladder for its leave-one-out arms). The full ladder itself is -0.98
   [-2.93, +0.98] against v1 max-abs, and the freeze rule is unaffected because it is defined on the
   leave-one-out arms around the full ladder, whose intervals all include zero.

## 3. Frozen recipes (deliverable B)

Freeze rule, written before measuring (protocol S2): the default is the full ladder with deployable boundaries;
a step is dropped only if its leave-one-out arm is better on dev512 with an interval excluding zero for some
model. No arm qualified, so nothing was dropped. Recipes are in `tools/experiment_b2/frozen.py`.

| name | boundaries | unsigned | activation range | weight range | bias correction |
|---|---|---|---|---|---|
| `default` | fused_relu | yes (integers only) | MSE search | MSE search, per channel | empirical |
| `default_fused_all` | fused_all | yes | MSE | MSE | empirical |
| `default_signed` | fused_relu | no | MSE | MSE | empirical |
| `default_no_bias_correction` | fused_relu | yes | MSE | MSE | none |
| `default_weight_maxabs` | fused_relu | yes | MSE | max-abs, per channel | empirical |
| `minimal` | fused_relu | no | MSE | max-abs, per channel | none |

The same switches apply to every scalar format. The only per-format element is the scale search, whose grid is
derived from the codebook itself (ratio to the max-abs scale from 2^-6 up to the dynamic range of the codebook,
8 points per octave plus a local refinement; `tools/experiment_b2/scales.py`). The unsigned switch is defined for
integer formats only, because the accepted non-integer manifests have no unsigned variant.

Hardware-visible semantics each switch changes (also emitted per recipe by
`.venv-b/bin/python -m tools.run.experiment_b2 recipes` and stored in every export):

| switch | what the hardware cost model must follow |
|---|---|
| `fused_relu` | A conv/linear/add/mul whose only consumer is ReLU or ReLU6 keeps the wide accumulator through the clamp and is requantized **once**, after it (the clamp folds into the requantizer). Max-pool forwards codes unchanged (no requantizer). Hardswish/Hardsigmoid still take a requantized input and requantize their output. Fewer requantizers than v1, none added. |
| `fused_all` | As above, and Hardswish/Hardsigmoid are evaluated on the wide accumulator value before the single requantization: a wide-precision nonlinearity (multiplier or large LUT) is required. Ablation only. |
| `unsigned` | Non-negative boundaries (outputs of ReLU, ReLU6, Hardsigmoid, and pooling/add/mul/passthrough over them) store unsigned k-bit codes `0..2^k-1`. The activation operand of the next MAC is unsigned k-bit while weights stay signed: a signed-by-unsigned multiplier, one more magnitude bit in the product and accumulator. It is a distinct activation-format identity (`<format>:unsigned`), not an accepted manifest. |
| activation range MSE | Same metadata as v1: one FP32 scale per quantizing node. Only the offline value changes. |
| weight range MSE | Same metadata as v1: one FP32 scale per output channel. Weights beyond the top level are clipped offline. |
| bias correction | Bias constants are replaced offline by corrected FP32 values (first 256 calibration images, sequential). No new storage, no new operator. |
| equalization | Offline rescaling of weights and biases; no run-time cost. Not used in any frozen recipe. |

Input for decision D6: at 8 bits the two switches that carry the repair are `fused_relu` (free in hardware,
removes requantizers) and the activation MSE scale (free). `unsigned` has a real datapath cost and no measurable
INT8 benefit; it is worth 3 to 6 points at INT6 (section 8). Bias correction is free in hardware but harmful at
4 bits and on Posit8/MobileNetV3. A per-bit-width policy would need its own protocol; it was not done here.
One more hardware-visible choice is not fixed by any recipe (r3, section 13): which class wins when several logit
codes are equal. The recipe quantizes the logits, the argmax then runs on codes, and its tie rule (lowest index,
or none because the logits stay wide) has to be part of the contract.

## 4. Vendor anchor (deliverable C)

PyTorch FX graph-mode static INT8 (`torch.ao.quantization.quantize_fx`, torch 2.3.0, CPU) on the same frozen
checkpoints (unfolded torchvision modules; FX fuses BN), calibrated on **all 2000** frozen calibration images
(batch 32) and evaluated on the 1k screen with sealed per-image top-5. FX handled all three models.

| arm | ResNet18 | MobileNetV2 | MobileNetV3-L | notes |
|---|---|---|---|---|
| `x86_default` (anchor of record) | 69.4 | 66.5 | 65.7 | `get_default_qconfig_mapping('x86')`: HistogramObserver with `reduce_range=True` (7-bit quint8 affine activations), per-channel symmetric int8 weights |
| `x86_fullrange` | 67.9 | 0.1 | 64.7 | `reduce_range=False` on the x86 engine |
| `qnnpack_fullrange` | 69.4 | 0.1 | 69.5 | `reduce_range=False`, QNNPACK engine, per-channel weights; added after the x86 full-range arm failed |

The anchor of record is weak on the two MobileNets, for reasons that are properties of the vendor flow:

- `reduce_range=True` gives the activation scale of a 7-bit grid over the calibrated range (codes above 127 are
  still stored, so the eighth bit is head-room, not resolution).
- FX fuses conv+BN+ReLU but not ReLU6 or Hardswish. In the converted MobileNetV2 the 35 ReLU6 are separate
  modules that share the scale and zero point of the convolution output, so that output is quantized over the
  pre-activation range, negative half included; MobileNetV3 keeps 21 Hardswish and 8 Hardsigmoid modules. On the
  first MobileNetV2 layer the ReLU6 output occupies 16 codes (x86 default). This is the same pre-activation
  quantization that B2 removes with fused boundaries. No float conv or linear module is left in any converted
  graph (census with full module paths in `verification/vendor/`).

**Why the full-range arms collapse on MobileNetV2 (explained in r2, audit extended in r3).** Every quantized
operator module was audited on 8 screen images: the output is compared with the FP32 evaluation of its own
dequantized operands, requantized with the output parameters (a correct kernel differs by at most one code), and
a ReLU6 must equal a clamp of its input codes (`verify.kernel_audit`). In r2 the audit covered conv, linear and
ReLU6; since r3 it also covers Hardswish, Hardsigmoid, add, mul and pooling. Records:
`verification/vendor-probe/*--r3c.json`.

| arm | ResNet18: 31 audited (20 conv, 1 linear, 8 add, 2 pool) | MobileNetV2: 99 audited (52 conv, 1 linear, 35 ReLU6, 10 add, 1 pool) | MobileNetV3-L: 120 audited (62 conv, 2 linear, 21 Hardswish, 8 Hardsigmoid, 8 mul, 10 add, 9 pool) |
|---|---|---|---|
| `x86_default` | 0 wrong | 0 wrong | 0 wrong |
| `x86_fullrange` | 10 conv wrong (up to 74 codes) | 49 conv and the linear wrong (up to 255 codes) | 34 conv wrong (up to 125 codes) |
| `qnnpack_fullrange` | 0 wrong | all 35 ReLU6 wrong (up to 114 codes), nothing else | 1 conv wrong (204 codes) |

No add, mul, pooling, Hardswish or Hardsigmoid module is wrong in any arm.

- x86 engine with 8-bit activations: the convolution kernels themselves return wrong codes. The likely cause is
  16-bit saturation of the multiply-add in the x86 kernels, which is the documented reason the vendor default
  reduces the range; this is inferred from the error pattern, not measured (CPU: i9-12900K, AVX2 and AVX-VNNI,
  no AVX-512). It costs 1.5 points on ResNet18, 1.0 on MobileNetV3 and everything on MobileNetV2, plausibly
  because the shared pre-activation zero points there (91 to 223) keep every post-ReLU6 code high in the range.
- QNNPACK engine: all convolution kernels are right. The quantized ReLU6 fails because of a **memory-layout
  defect** (found by review 2, recorded in `verification/operator/relu6-layout--r3c.json`): on a contiguous
  quantized tensor `torch.ao.nn.quantized.ReLU6` equals the integer clamp exactly on both engines, in place or
  not; on a channels-last tensor the QNNPACK engine returns the correctly clamped codes in a permuted layout
  (same code histogram, output equal to the clamp read in the other layout; 73.8 percent of positions differ on
  a random tensor). Every quantized convolution returns a channels-last tensor, so all 35 ReLU6 of the converted
  MobileNetV2 fail (2.7 to 97.6 percent of the outputs of a module differ from the clamp; in all 35 the input is
  channels-last, the histogram is that of the clamp and the output equals the clamp in the other layout), and
  none fails on the x86 engine. Quantized Hardswish and Hardsigmoid are exact to one code on both engines and
  both layouts, and ResNet18 has no ReLU6, which is why the same arm works on the other two models.
- **The MobileNetV3 QNNPACK arm (69.5) is not a fully correct flow.** Its one wrong kernel,
  `features_4_block_2_fc2`, returns the zero-point code for every output (1 distinct code where the FP32
  evaluation of its own operands gives 49 on the 8 audit images; the x86 default returns 35 distinct codes at the
  same node), so the squeeze-excite gate of that block is the constant 0.5 instead of hardsigmoid(bias). The
  input of that kernel is the dead squeeze-excite ReLU (all zeros, observed scale 1.2e-7); the cause was not
  investigated. 69.5 is therefore a lower bound for a correct 8-bit vendor flow on MobileNetV3.
- MinMaxObserver instead of HistogramObserver fails the same way on both engines (0.2 and 0.1 percent, sealed
  probes `x86_minmax_fullrange` and `qnnpack_minmax_fullrange`); the observer is not the cause.

With the 35 ReLU6 modules replaced by an exact clamp of the codes (probe `qnnpack_fullrange_relu6fix`, same
calibration, nothing else changed) the 8-bit vendor flow gives **71.5** on MobileNetV2: -0.6 [-2.1, +0.9] against
FP32, +5.0 [+2.9, +7.0] over the x86 default. This is a diagnostic probe, not a shipped configuration, and it was
not sealed as per-image prediction files (the record holds the 1000 top-5 lists). The strongest vendor numbers
are therefore 69.4 (ResNet18, both engines), 71.5 (MobileNetV2, probe) and 69.5 (MobileNetV3, QNNPACK, a lower
bound as explained above). "Strongest arm" is chosen on the same 1k screen it is then compared on; the selection
works against B2 and everything here is development evidence.

Identity note: six of the nine sealed vendor configurations (`x86_default` and `x86_fullrange`) record a
`vendor_source_sha256` (`95218102...`) older than the current `vendor.py` (`e18ff04d...`), which was edited to
add the QNNPACK arm after they ran. All nine were rebuilt with the current file in r2: 1000 of 1000 top-5 lists
equal the sealed ones and every other identity field is equal (`verification/vendor/*.json`). The sealed runs
were left as they are; their identities cannot be regenerated from the current file.

Probe records: eleven of the twelve r2 probe records (`verification/vendor-probe/*--2000.json`) were made with
an earlier `verify.py` (`a6f64717...`) than the one that made the `relu6fix` record (`739ba1db...`), and lack the
top-5 lists and paired statistics. In r3 all twelve probes were run again with one final file
(`a17ce615...`, records `*--2000--r3c.json`): every prediction digest equals its r2 record. The r2 records are
kept. Verification jobs now refuse to write to an existing path (`--tag` gives a new file).

## 5. Exit test (deliverable D), INT8, top-1

| model | panel | FP32 | B2 default | B2 minus FP32 | vendor x86_default | B2 minus vendor |
|---|---|---|---|---|---|---|
| ResNet18 | screen1k | 70.10 | 69.00 | -1.10 [-2.00, -0.20] | 69.40 | -0.40 [-1.50, +0.70] |
| ResNet18 | dev512 | 71.09 | 69.53 | -1.56 [-2.93, -0.39] | 69.53 | 0.00 [-1.37, +1.37] |
| ResNet18 | heldout488 | 69.06 | 68.44 | -0.61 [-1.84, +0.61] | 69.26 | -0.82 [-2.66, +1.02] |
| MobileNetV2 | screen1k | 72.10 | 72.00 | -0.10 [-0.80, +0.70] | 66.50 | +5.50 [+3.20, +7.70] |
| MobileNetV2 | dev512 | 73.83 | 73.05 | -0.78 [-1.95, +0.20] | 67.19 | +5.86 [+2.73, +8.98] |
| MobileNetV2 | heldout488 | 70.29 | 70.90 | +0.61 [-0.41, +1.64] | 65.78 | +5.12 [+2.05, +8.20] |
| MobileNetV3-L | screen1k | 75.00 | 71.50 | -3.50 [-5.40, -1.50] | 65.70 | +5.80 [+3.40, +8.10] |
| MobileNetV3-L | dev512 | 76.95 | 72.66 | -4.30 [-6.84, -1.76] | 68.75 | +3.91 [+0.58, +7.23] |
| MobileNetV3-L | heldout488 | 72.95 | 70.29 | -2.66 [-5.53, +0.20] | 62.50 | +7.79 [+4.51, +11.07] |

Against the strongest vendor arm per model (screen1k, B2 default minus vendor): ResNet18 `qnnpack_fullrange`
69.4, -0.4 [-1.5, +0.7] as for the x86 default; MobileNetV2 `qnnpack_fullrange_relu6fix` probe 71.5, +0.5
[-0.9, +2.0]; MobileNetV3-L `qnnpack_fullrange` 69.5, +2.0 [-0.1, +4.0]. B2 INT8 is on par with the best 8-bit
vendor flow available here on all three models and is not shown to be better than it on any. Two caveats: the
MobileNetV2 arm exists only as a probe with a patched ReLU6, and the MobileNetV3 arm has one wrong kernel
(section 4), so a correct vendor flow could score higher than 69.5 and the +2.0 is an upper bound on B2's margin.
Under a fixed tie rule (section 13) these comparisons move by at most 0.4 point and no conclusion changes.

Ablation arms on screen1k (top-1): ResNet18 fused_all 69.0, signed 68.7, no bias correction 69.3, weight max-abs
69.2, minimal 69.4; MobileNetV2 72.0, 72.3, 72.1, 72.2, 72.2; MobileNetV3 71.9, 72.1, 70.6, 72.1, 69.9. No ablation
differs from the default with an interval excluding zero on screen1k. All pairings: `exit-test.json`.

**Residual diagnosis for MobileNetV3-Large (-3.5 points).** Not input or logits (FP32 input and logits: 72.27 on
dev512, no gain). Not the place of the Hardswish boundary (`fused_all` minus default on screen1k: +0.4 [-1.4, +2.2]).
Not weight range or equalization. The occupancy audit of the default INT8 configuration (dev128) puts the
lowest signal-to-quantization-noise boundaries at Hardswish outputs (mean 30.6 dB, minimum 22.6 dB at
`features_15_block_1_2`, then 22.7 at `features_14_block_0_2`) and squeeze-excite products (mean 32.7, minimum
22.6 at `mul_6`), against a median of 36.9 dB in ResNet18 and 45.6 dB in MobileNetV2. Among convolution
boundaries the weakest are the late depthwise ones (`features_14_block_1_0` 24.2 dB, `features_15_block_1_0`
25.9 dB); the projection convolutions of the same blocks (`block_3_0`) are at 34.1 and 32.8 dB and are not the
problem. Hardswish outputs range over
[-0.375, max], so they are not provably non-negative: they cannot use unsigned codes and a symmetric signed
code spends half of its levels on a sliver of the range. An affine (zero-point) activation code, a per-channel
activation scale or reconstruction-based rounding would be the next candidates; each changes the hardware
contract and none was tried (stop rule). The vendor's affine 8-bit arm reaches only 69.5, so asymmetry alone is
unlikely to close the gap.

## 6. Sentinel sweep with the frozen default (deliverable E), screen1k top-1

Difference to FP32 with interval; v1 columns are the sealed v1 1k results where they exist.

| format | ResNet18 (70.1) | v1 max / pct | MobileNetV2 (72.1) | v1 max / pct | MobileNetV3-L (75.0) | v1 max / pct |
|---|---|---|---|---|---|---|
| INT8 | 69.0, -1.1 [-2.0, -0.2] | 69.8 / 62.4 | 72.0, -0.1 [-0.8, +0.7] | 66.5 / 66.4 | 71.5, -3.5 [-5.4, -1.5] | 4.1 / 68.9 |
| INT6 | 66.9, -3.2 [-5.0, -1.4] | 51.5 / 60.6 | 70.5, -1.6 [-3.4, +0.2] | 0.3 / 51.9 | 21.1, -53.9 [-57.1, -50.8] | 0.1 / 14.4 |
| INT4 | 0.1 | 0.2 / 8.5 | 0.1 | 0.0 / 0.1 | 0.1 | 0.1 / 0.1 |
| FP8 E4M3 | 68.5, -1.6 [-3.3, +0.2] | 69.0 / - | 68.1, -4.0 [-6.1, -1.9] | 65.7 / 56.5 | 71.6, -3.4 [-5.2, -1.7] | 68.6 / - |
| FP7 E3M3 | 68.0, -2.1 [-3.7, -0.5] | 69.7 / - | 67.9, -4.2 [-6.3, -2.1] | 65.0 / 56.7 | 71.0, -4.0 [-5.9, -2.2] | 68.4 / 61.7 |
| FP6 E2M3 | 67.3, -2.8 [-4.7, -0.9] | 66.2 / 57.3 | 66.9, -5.2 [-7.3, -3.1] | 10.2 / 54.6 | 57.9, -17.1 [-19.8, -14.4] | 0.1 / 44.3 |
| LOG8 | 68.0, -2.1 [-3.7, -0.5] | 66.4 / - | 67.8, -4.3 [-6.2, -2.4] | 60.9 / 58.9 | 70.0, -5.0 [-7.0, -3.0] | 68.6 / - |
| Posit8 | 69.8, -0.3 [-1.4, +0.9] | 0.6 / 0.3 | 72.1, 0.0 [-1.3, +1.4] | 0.0 / 0.0 | 73.4, -1.6 [-3.1, -0.1] | 0.0 / 0.0 |
| NF4 | 0.4 | 1.1 / 35.0 | 0.1 | 0.1 / 0.2 | 2.2 | 0.0 / 0.0 |

Does the ordering change? Yes.

- v1 best-recipe order (ResNet18): INT8 > FP7 > FP8 > LOG8 > FP6 > INT6 > NF4 > INT4 > Posit8.
  B2 default order: Posit8 > INT8 > FP8 > FP7 > LOG8 > FP6 > INT6 > NF4 > INT4. MobileNetV2 and V3 likewise put
  Posit8 first (orders in `sentinel-1k.json`).
- Posit8 versus INT8 under the default (paired, Posit8 minus INT8): +0.8 [-0.2, +1.9], +0.1 [-1.1, +1.4],
  +1.9 [-0.2, +4.0]. Posit8 is at least on par with INT8; it is not shown to be better.
- FP8 E4M3 versus INT8: -0.5 [-2.2, +1.2], **-3.9 [-6.0, -1.8]** (MobileNetV2), +0.1 [-2.1, +2.3]. With a sound INT8
  anchor the 8-bit minifloat is behind INT8 on MobileNetV2: its 3 mantissa bits cap the per-boundary SQNR at
  about 31.5 dB whatever the scale, against 45.6 dB for INT8.
- INT6 becomes usable on ResNet18 and MobileNetV2 (66.9, 70.5) but not on MobileNetV3 (21.1).
- 4-bit formats are at chance under the default; for ResNet18 the v1 percentile recipe is better there
  (INT4 8.5, NF4 35.0). The default must not be used to rank 4-bit formats (section 8).

Every statement of the form "format X beats INT-k" drawn from the v1 matrix for the two MobileNets has to be
re-derived: the INT8 anchor moved by +5.5 points on MobileNetV2 and by +2.6 (versus v1 percentile) or +67.4 (versus v1
max-abs) on MobileNetV3.

Tie caveat (r3, section 13): in this sweep 10 to 19 percent of the images have a tied largest logit for the
minifloat and log formats, 5 to 10 percent for Posit8 and 10 to 47 percent for INT6. The sealed figures above are
what `topk` returned; under expected credit they move by -1.0 to +0.5 point. The coarse ordering (Posit8 and INT8
at the top, 4-bit formats at chance, FP6 and INT6 collapsing on MobileNetV3) does not depend on the tie rule;
orderings among FP8, FP7, LOG8 and INT6 within about a point do.

## 7. Posit8 code-occupancy audit (dev128, activations at every quantizing boundary)

| model | recipe | boundaries | elements at levels with abs value >= 1 (median over boundaries) | median SQNR | worst SQNR | median code entropy |
|---|---|---|---|---|---|---|
| ResNet18 | v1_maxabs | 49 | 99.2 % | 15.4 dB | 7.7 dB | 4.30 bits |
| ResNet18 | default | 31 | 5.9 % | 37.2 dB | 36.8 dB | 5.41 bits |
| MobileNetV2 | v1_maxabs | 100 | 99.2 % | 11.8 dB | 5.6 dB | 3.86 bits |
| MobileNetV2 | default | 65 | 14.0 % | 37.5 dB | 35.5 dB | 5.51 bits |
| MobileNetV3-L | v1_maxabs | 140 | 99.2 % | 13.7 dB | 0.0 dB | 3.90 bits |
| MobileNetV3-L | default | 121 | 7.6 % | 37.3 dB | 34.2 dB | 6.11 bits |

The suspicion is confirmed. Range-endpoint scaling maps the calibration maximum to the top posit level (4096),
so essentially every non-zero value falls in the upper regimes where Posit8 has 0 to 2 fraction bits. The scale
search moves the bulk below 1, where the codebook is dense. The weights need the same treatment: with the
activation search but max-abs weight scales Posit8 stays at chance (`default_weight_maxabs`: 0.1 / 0.1 / 0.1;
`minimal`: 4.2 / 0.1 / 0.0). For comparison under the default: INT8 median SQNR 36.9 / 45.6 / 36.1 dB, FP8 E4M3
31.5 / 31.6 / 31.6 dB. Per-boundary data: `artifacts/experiment_b2/occupancy/`, summary `occupancy.json`.

## 8. Ablations away from INT8 (screen1k; addition to the protocol, measurement only)

Default minus arm, points, with interval (`lowbit-ablation-1k.{json,csv}`):

| format | arm removed | ResNet18 | MobileNetV2 | MobileNetV3-L |
|---|---|---|---|---|
| INT6 | unsigned | +4.9 [+3.0, +6.9] | +3.2 [+1.5, +4.9] | +6.0 [+3.5, +8.4] |
| INT6 | bias correction | +0.2 [-1.9, +2.3] | +3.3 [+1.1, +5.4] | +17.7 [+15.1, +20.3] |
| INT6 | weight MSE (max-abs instead) | -1.6 [-3.4, +0.2] | -1.5 [-3.0, 0.0] | -2.3 [-4.8, +0.2] |
| INT4 | bias correction | **-28.2 [-31.0, -25.5]** | -1.0 [-1.7, -0.4] | -0.1 |
| INT4 | weight MSE | -20.4 [-22.9, -17.9] | -10.3 [-12.2, -8.4] | 0.0 |
| FP6 E2M3 | bias correction | -0.9 [-3.0, +1.1] | -0.5 [-2.7, +1.7] | +23.5 [+20.2, +26.8] |
| Posit8 | bias correction | +0.3 [-0.8, +1.4] | -0.1 [-1.6, +1.4] | -1.9 [-3.4, -0.4] |
| Posit8 | weight MSE | +69.7 | +72.0 | +73.3 |

Reading: unsigned codes and bias correction earn their place at 6 bits. Per-channel weight MSE is consistently,
if not significantly, slightly negative for integers at 6 bits and strongly negative at 4 bits, yet indispensable
for Posit8. At INT4 the default (0.1) is worse than its own ablations on ResNet18 (28.3 without bias correction,
20.5 with max-abs weights): with a signal this degraded the sequential correction replaces it with constants.
This agrees with the earlier exact-path finding that MSE clipping does not rescue 4 to 5 bits
(`artifacts/breadth_study/next_study_v1/E2/`). Posit8 on MobileNetV3 without bias correction scores 75.3
(FP32 75.0).

## 9. Tests and proofs (deliverable F)

- Unit tests: `tests/unit/test_experiment_b2.py` (25 tests) and `tests/unit/test_experiment_b2_guards.py`
  (6 tests, r2: the graph guard, the verification interpreter, the vendor census and the kernel audit) and
  `tests/unit/test_experiment_b2_verify_r3.py` (4 tests, r3: the tie statistics on a hand-checkable case, the
  refusal to overwrite a verification record, the layout helper, the extended kernel audit);
  `.venv/bin/python -m pytest tests/unit/test_experiment_b2.py tests/unit/test_experiment_b2_guards.py
  tests/unit/test_experiment_b2_verify_r3.py -q` (35 pass).
  Hand-checkable cases for the unsigned codebook (levels, ties to even, clipping), the boundary analysis on a
  graph with every operator kind, the MSE search (exact grid, clipping one outlier, independence of rows, tapered
  codebook, zero tensor), bias correction (per-channel means equal FP32 after correction on the small test graph
  executed in one batch; on the real models this holds only for the staged pass, see section 11; weights untouched),
  exemptions, equalization (function preserved, ranges balanced, ReLU6 pair skipped) and the export.
- v1 regression, synthetic graph (CPU, in the unit tests): with all switches off, B2 logits and quantized weights
  are bit-identical to v1 `prepare_qdq` for 7 formats x 2 v1 recipes.
- v1 regression, real models (GPU, sealed in `artifacts/experiment_b2/regression/`): for ResNet18, MobileNetV2 and
  MobileNetV3-Large, INT8 and Posit8, both v1 recipes, B2-off logits are bit-identical to v1 on 1000 of 1000
  images, the scales equal the sealed v1 configurations and the sealed v1 top-5 lists are reproduced on 1000 of
  1000 images (12 of 12 records pass). B2 therefore measures only the declared switches.

## 10. How to reproduce

GPU work runs in `.venv-b` (torch 2.3.0+cu121, the environment v1 used; `.venv` has CPU-only torch and runs the
unit tests). All stages are resumable and skip recorded runs.

```
.venv-b/bin/python -m tools.run.experiment_b2 inputs --model resnet18        # and mobilenet_v2: input cache
for st in regress diagnosis cle vendor exit audit sentinel lowbit export; do
  .venv-b/bin/python -m tools.run.experiment_b2_campaign $st                 # takes the shared GPU lock itself
done
.venv-b/bin/python -m tools.analysis.b2_report all --images 512              # tables under results/summaries/b2-baseline-repair-v1/
# single job, for example:
flock -w 3600 artifacts/agent_orchestration/gpu.lock .venv-b/bin/python -m tools.run.experiment_b2 run \
  --model mobilenet_v2 --format int8 --recipe default --images 1000 --stage exit
```

Read-only verification jobs added in r2 (they seal one record each under `artifacts/experiment_b2/verification/`
and never write configurations, predictions, summaries or run records):

```
flock -w 3600 artifacts/agent_orchestration/gpu.lock .venv-b/bin/python -m tools.run.experiment_b2_verify \
  bias-residual --model mobilenet_v3_large --format int8 --recipe default
.venv-b/bin/python -m tools.run.experiment_b2_verify vendor-check --model resnet18 --qconfig x86_default   # CPU
.venv-b/bin/python -m tools.run.experiment_b2_verify vendor-probe --model mobilenet_v2 --qconfig qnnpack_fullrange_relu6fix
```

Added in r3 (same rules; a job refuses to write to a record path that exists, `--tag NAME` writes a new file):

```
flock -w 3600 artifacts/agent_orchestration/gpu.lock .venv-b/bin/python -m tools.run.experiment_b2_verify \
  batch-sensitivity --model resnet18 --format int8 --recipe default --batches 8,32,1 --tag NAME   # also tie statistics
.venv-b/bin/python -m tools.run.experiment_b2_verify relu6-operator --tag NAME                      # CPU
.venv/bin/python -m tools.analysis.b2_ties --tag r3c    # logit-ties-1k.{csv,json}, batch-sensitivity-1k.json
```

The r3 campaign as it was run: `artifacts/experiment_b2/logs/verify-r3c-{cpu,gpu}.sh` (logs beside them).

The vendor anchor and these CPU jobs run in `.venv-b` as well (the sealed vendor runtime is torch 2.3.0+cu121 on
CPU; the input-cache key contains the torch build string, so `.venv` would rebuild the cache).

Evidence: `artifacts/experiment_b2/` (`configurations/`, `predictions/<configuration>/<image sha256>.json` in the v1
schema, `summaries/`, `runs/<stage>/`, `regression/`, `occupancy/`, `exports/`, `logs/`). Calibration observations
are the v1 cache, read and verified but never written. Recipe-as-data exports of the default INT8 configuration
(format `b2-recipe-export-1`, reader `tools.experiment_b2.export.load_export`):
`artifacts/experiment_b2/exports/03abf669.../`, `6dcf36e7.../`, `b2179648.../` (one per classifier; the model
is in each `export.json`).

Cost (sealed in every run record; `cost.json`): 224 runs, 2589 s of job wall-clock in total, of which 2201
GPU-seconds (diagnosis about 740, low-bit 583, sentinel 319, exit 255, audit 149, regression 143) and 382
CPU-seconds for the vendor anchor. Time spent waiting for the GPU lock is not included (it is in
`artifacts/experiment_b2/logs/campaign-<stage>.jsonl`). Revision r2 added, outside `cost.json` and sealed in each
verification record: 3 bias-residual jobs (37 GPU-seconds, lock waits under 3 s), 9 vendor rebuilds (266
CPU-seconds) and 12 vendor probes (444 CPU-seconds); about 8 earlier probe runs made while the audit was being
written (roughly 5 CPU-minutes) were overwritten by these final records. Revision r3 added, sealed in each
record: 33 batch-sensitivity and tie jobs (465 GPU-seconds, one lock acquisition per configuration, 174 s of lock
waiting in total), 12 vendor probes (475 CPU-seconds) and one operator test (2 CPU-seconds), all tagged `r3c`.
Intermediate r3 records made while `verify.py` was still being edited are kept and superseded: five
batch-sensitivity records (130 GPU-seconds; untagged, `r3`, `r3b`), six vendor probes (220 CPU-seconds; `r3`,
`r3b`) and two operator records, plus three interrupted jobs that wrote nothing.

## 11. Limits

- Development evidence only. Selection panel 512 images, exit test 1000 images containing it; intervals are
  about +-1 to +-2.5 points wide, so INT8 effects below roughly 2 points are not resolved. Intervals are
  pointwise and no multiplicity correction is applied to the diagnosis, exit, sentinel or low-bit tables; with
  about 30 arms per model, a few intervals that exclude zero are expected by chance.
- B2 is still the FP32 quantize-dequantize surrogate: wide FP32 accumulation, FP32 bias, FP32 scales. It is not
  the exact engine. The project's exact strict-A MobileNetV2 INT8 with MSE calibration scored 70.6 on this list (figure from the
  lane brief, not re-measured here);
  B2 default scores 72.0 and v1 66.5, so the surrogate no longer under-reports, but the 1.4-point difference to
  the exact path is unexplained here (lane L5 measures that gap).
- Activation ranges come from 256 stratified samples per image per node over 2000 images (the v1 observer);
  bias correction uses the first 256 calibration images.
- Bias correction is exact only for the staged pass that fits it. When the finished engine is executed again on
  the same 256 calibration images, the per-channel means of the conv and linear outputs no longer equal FP32
  (`verification/bias-residual/`): the largest residual is 0.0076 on ResNet18 (`fc`; 2 of 21 layers above 1e-3),
  0.012 on MobileNetV2 (`features_18_0`; 20 of 53) and 0.083 on MobileNetV3 (`features_16_0`; 34 of 64), where it
  is five times the correction applied to that layer (0.016); the median ratio of residual to applied correction
  is 0.002, 0.017 and 0.014. The residual has the same size at batch 32 and batch 8 but is not the same
  (corrected in r3): on the two MobileNets the two batch sizes agree to 1e-8 (3 layers differ, by at most 5e-9);
  on ResNet18 14 of 21 layers differ, 7 of them by more than 10 percent, the largest at `layer4_1_conv2` (0.0049
  at batch 8, 0.0029 at batch 32), and the maxima are 0.0076 (batch 8) and 0.0073 (batch 32). Cause, as
  dissected by the reviewer: the staged pass adds the correction to an output already computed, the engine
  recomputes the layer with the new bias, the two differ by about 1e-6, and rounding at each later boundary
  amplifies the difference. Consequences: (i) corrections in deep layers that are smaller than this residual are
  noise, (ii) the engine output depends on low-order floating-point bits, hence on the batch size (measured in
  r3, section 13: ResNet18 at batch 32 changes 14 of 1000 top-1 predictions) and presumably on the GPU and cuDNN
  version, which was not measured. B2 numbers reproduce bit for bit on this machine at the protocol batch size
  (the reviewers recomputed seven distinct default cells; r3 recomputed 33: the 27 sentinel cells, which
  include the three exit cells, and the six INT8 cells of the two v1 recipes; all gave 1000 of 1000 sealed
  top-5 lists).
- Tied logits (r3, section 13). Every top-1 and top-5 figure of B2, of v1 and of the vendor anchor is read from
  quantized logits with `topk`, whose choice among equal values is implementation-defined (the same logits give
  10 to 19 different top-1 classes on ResNet18 INT8 when `topk` runs on the CPU instead of the GPU). Section 13
  gives the size of the effect per configuration.
- The vendor anchor of record is weak on the MobileNets (section 4): 7-bit activation scale and no fusion of
  ReLU6 or Hardswish. "On par with the vendor" should be read against the strongest working vendor arm
  (69.4 / 71.5 / 69.5), and "+5.5 over the vendor default" on MobileNetV2 mostly measures the default's handicap.
- Unsigned codes rely on a structural non-negativity proof that looks only at FX-node operands of add and mul; a
  literal operand would be missed and the unsigned quantizer maps negative inputs to zero silently. Since r2
  `guards.check_graph` refuses such graphs in `runner.build` (the three classifiers pass), and the bias-residual
  job counted zero negative elements reaching any of the 18 / 36 / 33 unsigned boundaries on 256 images.
- The sealed configurations report an MSE-search ratio at the lower grid edge (2^-6) for
  `features_4_block_2_activation` of MobileNetV3. That node is a dead squeeze-excite ReLU (all zeros in
  calibration and in the occupancy audit), so the value is an artefact; no live node is near the grid edge
  (reviewer: next smallest ratio 0.166 at INT8, 0.071 at INT6).
- The vendor `graph_census` stored in the sealed configurations keeps only the last component of the module
  path; float and quantized modules are told apart by the full-path census in `verification/vendor/`.
- Deviation from the lane's GPU rule, reported to the orchestrator: each configuration is its own process, but
  one lock acquisition covers a batch of them capped at 240 s of started work, not one configuration.
- Records and summaries with a known blemish, kept under the no-overwrite rule: (i) the batch-sensitivity
  records name a field `paired_top1_sealed_batch8_minus_this`, but the value is this run minus the sealed run
  (the statistics helper returns right minus left); the summary `batch-sensitivity-1k.json` uses the correct
  name and `batch-sensitivity.json` (first export, same numbers, wrong field name) is superseded by it;
  (ii) `diagnosis-128.json` has no models and `diagnosis-128.csv` is empty, because no 128-image diagnosis runs
  exist: the dev128 figures are the prefix rows inside `diagnosis-512.*`; (iii) the untagged record
  `verification/batch-sensitivity/resnet18--int8--default--2483170b2bc6.json` reports that the sealed predictions
  were not reproduced: that job took top-5 on the CPU in float64, the runner takes it on the GPU in float32, and
  the tie-breaking differs (262 top-5 lists, 19 top-1 classes); the tagged records take it as the runner does
  and reproduce 1000 of 1000.
- The kernel audit and the code census use the first 8 screen images only; a kernel that fails only on other
  inputs would be missed.
- Protocol additions made after it was written, all measurement only: the `qnnpack_fullrange` vendor arm, the
  `g_percentile_no_logits` diagnostic arm, the low-bit ablation stage, the r2 verification jobs and vendor
  probes, and the r3 jobs (written down first in the r3 addendum, which was amended once, before the tie jobs
  ran). None was used to choose the default.

## 12. Open

1. MobileNetV3-Large INT8 is 3.5 points below FP32. Closing it needs a component outside this protocol (affine
   or per-channel activation codes, or reconstruction-based rounding) and an owner decision, because each
   changes the hardware contract.
2. The default is harmful at 4 bits. A bit-width-dependent policy (for example no bias correction and max-abs
   weights at 4 bits) needs its own protocol and selection data; D6 should decide whether "one recipe for all
   formats" is a requirement or whether a small per-family table is acceptable.
3. Whether `unsigned` belongs in the default is a hardware question: no INT8 benefit, 3 to 6 points at INT6, and
   a signed-by-unsigned MAC.
4. Not run: the 200-configuration matrix, shared-exponent formats and YOLO. MX/BFP need the block quantizer of
   `tools/experiment_b_ext/shared.py` given the same boundary plan (fused boundaries, one block scale search per
   block instead of block max-abs) and a definition of bias correction over block-scaled activations. YOLO needs
   the detector graph of `tools/experiment_b_ext/detector.py` taught the boundary analysis (SiLU is not fusable
   under `fused_relu`; concat and the DFL head need rules), the COCO calibration observations, and box-metric
   statistics instead of paired top-1.
5. Confirmation on held-out data (the 9000 evaluation images outside the 1k screen) is reserved and was not
   run. The frozen recipes and the export make that a single pass when it is released.
6. The vendor full-range collapse on MobileNetV2 is explained (section 4). Open: whether the paper's vendor
   anchor should be the shipped x86 default (weak, but what a user gets) or the strongest working arm per model;
   the MobileNetV2 one exists only as a probe with a patched ReLU6 and would have to be sealed with per-image
   predictions under its own protocol entry before it is cited. An independent runtime (ONNX Runtime or TFLite)
   is not installed and was not tried.
7. Bias correction could be made exact for the executed engine by recomputing each layer with its corrected
   bias inside the staged pass. That edits `engine.py` (a numeric source), changes every configuration identity
   and needs the affected cells re-measured; at INT8 the step is not supported by the evidence anyway.
8. Readout rule for tied logits (r3, section 13). Options: keep the logits wide (the diagnostic arm (g) shows no
   accuracy cost at INT8 and it removes the ties, but it is an exemption of the last boundary), or keep them
   quantized and fix the argmax rule (lowest index) in the protocol and in the hardware contract, or score with
   expected credit. Whatever is chosen has to be applied to the comparator as well; v1 matrix results carry the
   same unspecified rule. This belongs with D6.
9. Whether batch-size dependence of the engine output is acceptable for sealed evidence. It can be removed by
   evaluating at batch 1 semantics (per-image determinism costs throughput) or tolerated and stated; on this
   machine only ResNet18 shows it (14 of 1000 top-1 predictions between batch 8 and 32, +0.1 point).

## 13. Readout: tied logits and batch size (r3)

Protocol: the r3 addendum (written before the jobs; amended once before the tie jobs). All counts are exact on the
1k screen; they are development evidence about the readout, not a comparison of recipes. Records:
`verification/batch-sensitivity/*--r3c.json` and `verification/vendor-probe/*--r3c.json`; summaries
`results/summaries/b2-baseline-repair-v1/logit-ties-1k.{csv,json}` and `batch-sensitivity-1k.json`
(`.venv/bin/python -m tools.analysis.b2_ties`). Every job first recomputed the sealed configuration (identity
equal) and reproduced its sealed top-5 lists on 1000 of 1000 images.

**Tied logits.** B (v1 and B2) quantizes the logits like any other tensor, and the vendor flow returns
dequantized 8-bit logits. An image is tied when its largest logit is shared by two or more classes. Four rules:
strict (the label must be the unique maximum), expected (credit 1/k in a k-way tie, the mean over random
tie-breaking), lowest index (the smallest class index among the maxima wins; an integer argmax can implement
it), optimistic (the label is among the maxima). The sealed figure is whatever `topk` returned. FP32 has no tied
image on any model. Cells: sealed top-1; tied images of 1000; strict / expected / lowest index / optimistic.

| INT8 configuration | ResNet18 | MobileNetV2 | MobileNetV3-L |
|---|---|---|---|
| B v1 max-abs | 69.8; 72 tied; 67.6 / 69.9 / 69.8 / 72.3 | 66.5; 33 tied; 65.8 / 66.5 / 66.3 / 67.3 | 4.1; 66 tied; 4.0 / 4.0 / 4.1 / 4.1 |
| B v1 percentile | 62.4; 185 tied; 55.2 / 62.0 / 62.0 / 71.9 | 66.4; 157 tied; 60.3 / 66.5 / 67.0 / 73.8 | 68.9; 94 tied; 65.0 / 68.8 / 69.0 / 72.8 |
| **B2 default** | 69.0; 42 tied; 67.8 / 69.2 / 68.6 / 70.7 | 72.0; 14 tied; 71.9 / 72.3 / 72.0 / 72.7 | 71.5; 13 tied; 71.3 / 71.6 / 71.5 / 71.9 |
| vendor `x86_default` | 69.4; 52 tied; 67.9 / 69.3 / 69.3 / 70.9 | 66.5; 43 tied; 65.7 / 66.4 / 66.1 / 67.1 | 65.7; 28 tied; 64.7 / 65.5 / 65.4 / 66.4 |
| vendor `x86_fullrange` | 67.9; 31 tied; 66.9 / 67.6 / 67.5 / 68.3 | 0.1 (collapsed) | 64.7; 13 tied; 64.3 / 64.7 / 64.6 / 65.0 |
| vendor `qnnpack_fullrange` | 69.4; 31 tied; 68.2 / 69.2 / 69.2 / 70.2 | 0.1 (collapsed) | 69.5; 10 tied; 69.3 / 69.6 / 69.7 / 69.9 |
| vendor `qnnpack_fullrange_relu6fix` (probe) | n/a | 71.5; 21 tied; 70.9 / 71.5 / 71.7 / 72.1 | n/a |

| format, B2 default | ResNet18 | MobileNetV2 | MobileNetV3-L |
|---|---|---|---|
| int8 | 69.0; 42 tied; 67.8 / 69.2 / 68.6 / 70.7 | 72.0; 14 tied; 71.9 / 72.3 / 72.0 / 72.7 | 71.5; 13 tied; 71.3 / 71.6 / 71.5 / 71.9 |
| int6 | 66.9; 168 tied; 61.3 / 65.9 / 65.2 / 71.4 | 70.5; 104 tied; 67.4 / 70.1 / 70.3 / 73.4 | 21.1; 466 tied; 17.0 / 21.0 / 20.9 / 29.5 |
| int4 | 0.1; 980 tied; 0.0 / 0.1 / 0.2 / 0.9 | 0.1; 1000 tied; 0.0 / 0.1 / 0.1 / 2.0 | 0.1; 1000 tied; 0.0 / 0.1 / 0.1 / 0.7 |
| fp8_e4m3fn | 68.5; 186 tied; 61.9 / 67.7 / 67.9 / 74.7 | 68.1; 135 tied; 64.1 / 67.7 / 67.9 / 71.8 | 71.6; 111 tied; 68.0 / 71.7 / 71.5 / 75.7 |
| fp7_e3m3 | 68.0; 174 tied; 62.1 / 67.5 / 67.3 / 73.9 | 67.9; 115 tied; 64.6 / 67.6 / 67.3 / 70.9 | 71.0; 104 tied; 67.9 / 71.1 / 71.3 / 75.1 |
| fp6_e2m3 | 67.3; 183 tied; 61.4 / 66.7 / 66.8 / 73.3 | 66.9; 152 tied; 62.3 / 66.2 / 65.4 / 71.0 | 57.9; 150 tied; 54.6 / 58.4 / 58.5 / 62.6 |
| log8 | 68.0; 181 tied; 62.6 / 68.1 / 67.9 / 74.8 | 67.8; 111 tied; 65.2 / 68.2 / 67.9 / 71.6 | 70.0; 114 tied; 66.4 / 69.8 / 69.5 / 73.7 |
| posit8_es1 | 69.8; 102 tied; 66.4 / 69.6 / 69.1 / 73.1 | 72.1; 56 tied; 70.6 / 72.4 / 71.9 / 74.3 | 73.4; 47 tied; 72.0 / 73.4 / 72.8 / 74.9 |
| nf4 | 0.4; 606 tied; 0.1 / 0.3 / 0.1 / 1.5 | 0.1; 1000 tied; 0.0 / 0.1 / 0.2 / 5.8 | 2.2; 616 tied; 1.1 / 2.2 / 2.0 / 4.8 |

What this shows:

- At INT8 the effect is small for the frozen default: the sealed figure is within 0.3 point of expected credit
  and within 0.4 of the lowest-index rule on all three models, although the strict-to-optimistic band on
  ResNet18 is 2.9 points wide (67.8 to 70.7).
- The formats whose large magnitudes are coarse (minifloats, LOG8) tie on 10 to 19 percent of the images, because
  the winning logit sits in the coarse top of the codebook; the strict-to-optimistic band is 6 to 13 points
  wide. The sealed top-1 is not biased in one direction (expected credit minus sealed: -1.0 to +0.5 over the 27
  sentinel cells; lowest index minus sealed: -1.7 to +0.6) but it is one arbitrary draw from that band.
- The ordering of the sentinel formats is stable at the coarse level under the sealed, expected and
  lowest-index readouts (Posit8 and INT8 are the top two on ResNet18 and MobileNetV2; Posit8 leads on
  MobileNetV3 with FP8 and INT8 within 0.2 point of each other; 4-bit formats at chance; INT6 and FP6 collapsing
  on MobileNetV3) and unstable within about a point: on ResNet18 and MobileNetV2 LOG8 moves ahead of FP8 and FP7
  under expected credit, and on MobileNetV2 INT8 moves ahead of Posit8 under the lowest-index rule.
- v1 percentile has 94 to 185 tied images, consistent with its clipping of the logits (section 2 item 3).

Exit test under a fixed tie rule (post-hoc readout, same sealed runs; pointwise 95 percent paired image bootstrap
of the mean credit difference, 10000 resamples; B2 default minus the other):

| model | rule | minus FP32 | minus vendor x86 default | minus strongest vendor arm | minus B v1 max-abs |
|---|---|---|---|---|---|
| ResNet18 | sealed `topk` | -1.1 [-2.0, -0.2] | -0.4 [-1.5, +0.7] | -0.4 [-1.5, +0.7] | -0.8 [-2.3, +0.6] |
| ResNet18 | expected | -0.88 [-1.60, -0.20] | -0.12 [-1.07, +0.85] | +0.02 [-0.85, +0.88] | -0.65 [-1.90, +0.63] |
| ResNet18 | lowest index | -1.5 [-2.4, -0.6] | -0.7 [-1.8, +0.4] | -0.6 [-1.6, +0.4] | -1.2 [-2.7, +0.3] |
| MobileNetV2 | sealed `topk` | -0.1 [-0.8, +0.7] | +5.5 [+3.2, +7.7] | +0.5 [-0.9, +2.0] | +5.5 (no interval) |
| MobileNetV2 | expected | +0.20 [-0.60, +0.95] | +5.93 [+3.78, +8.20] | +0.80 [-0.60, +2.20] | +5.75 [+3.65, +7.90] |
| MobileNetV2 | lowest index | -0.1 [-0.9, +0.7] | +5.9 [+3.7, +8.2] | +0.3 [-1.1, +1.7] | +5.7 [+3.6, +7.9] |
| MobileNetV3-L | sealed `topk` | -3.5 [-5.4, -1.5] | +5.8 [+3.4, +8.1] | +2.0 [-0.1, +4.0] | +67.4 (no interval) |
| MobileNetV3-L | expected | -3.40 [-5.25, -1.50] | +6.05 [+3.75, +8.35] | +2.00 [0.00, +4.00] | +67.55 [+64.50, +70.45] |
| MobileNetV3-L | lowest index | -3.5 [-5.4, -1.6] | +6.1 [+3.7, +8.4] | +1.8 [-0.2, +3.8] | +67.4 [+64.4, +70.3] |

(The two sealed entries without an interval are differences of the sealed top-1 figures of section 1; the
strongest vendor arm is `qnnpack_fullrange` for ResNet18 and MobileNetV3 and the `relu6fix` probe for
MobileNetV2.) The exit statement of section 5 holds under every rule: MobileNetV2 meets the FP32 clause,
ResNet18 is about one point below FP32 (-0.9 to -1.5, every interval excluding zero) and on par with the vendor,
MobileNetV3-Large misses the FP32 clause by 3.4 to 3.5 points. Under the lowest-index rule ResNet18 is 1.5
points below FP32, further from the "about 1 point" target than the sealed readout suggests.

**Batch size** (INT8 default, 1000 screen images; sealed evaluation is batch 8):

| model | batch | top-1 | top-5 lists equal to sealed | top-1 class changed | images with any logit difference to batch 8 | largest logit difference |
|---|---|---|---|---|---|---|
| ResNet18 | 8 | 69.0 | 1000 | 0 | 0 | 0 |
| ResNet18 | 32 | 69.1 | 840 | 14 | 897 | 0.208 |
| ResNet18 | 1 | 69.0 | 1000 | 0 | 0 | 0 |
| MobileNetV2 | 32 | 72.0 | 1000 | 0 | 1 | 0.073 |
| MobileNetV2 | 1 | 72.0 | 1000 | 0 | 1 | 0.073 |
| MobileNetV3-L | 32 | 71.5 | 1000 | 0 | 1 | 0.080 |
| MobileNetV3-L | 1 | 71.5 | 1000 | 0 | 3 | 0.080 |

The FP32 graph gives identical top-5 lists at all three batch sizes on all three models (logit differences up to
4e-5). On ResNet18 at batch 32 the paired top-1 difference to the sealed run is +0.1 [-0.5, +0.8]: the accuracy
does not move, but 14 predictions do, so per-image predictions of B2 on ResNet18 are tied to the batch size as
well as to the machine. The mechanism is consistent with that of the bias-correction residual (section 11): a convolution
whose floating-point result depends on the batch size in its last bits, amplified by rounding at every later
quantizing boundary. The two MobileNets are almost insensitive on this machine.
