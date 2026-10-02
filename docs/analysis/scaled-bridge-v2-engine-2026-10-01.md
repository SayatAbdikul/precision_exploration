# Scaled bridge v2 engine: milestone M1 (ResNet18, scalar formats)

Run 2026-10-01. Contract and protocol: `docs/analysis/scaled-bridge-v2-contract-2026-10-01.md`
(contract 2.0, protocol M1, sealed before any panel at
`artifacts/scaled_bridge_v2/protocols/m1-ac8ff427bb80897f.json`). Machine result:
`results/summaries/scaled-bridge-v2/M1-2a64216184ac39ea/summary-r2.json` (per-layer widths, paired comparisons,
timings, cost, witness counts) and `table-r2.txt`. Engine sources digest `2a64216184ac39ea…`.

Revision note (2026-10-01, after review 1). Revision 1 of the summary (`summary.json`, `table.txt`) is kept
unchanged. Revision 2 has identical per-case content plus two explicit counts per case (images whose Top-1 class
and whose ordered Top-5 differ between the arms), the primitive witness count per codebook and a cost total
bounded by a recorded ledger cut-off. Four statements of the first version of this document were wrong or
overstated and are corrected below: the Top-1 claim for the control arm, the 32-bit claim, the witness counts
and the cost totals. The report tool no longer replaces a published file: it writes a new revision or reports
that an existing revision is identical. A third file, `summary-r3.json`, also exists: it was written by a
first version of the revised tool whose output depended on the live sources, and it differs from revision 2 only
in the `engine_sources` field (a placeholder text instead of the source list). The tool was corrected; revision 2
is the summary of record and the reproduce command below reports it as unchanged.

Everything below is development evidence on a 32-image prefix of the frozen screen-1k list (128 images for two
cases). It supports engine admission. It does not support quality claims or rankings.

## Result

**23 of 23 cases passed every gate.** ResNet18, original B recipes (family b1): `maxabs` for 17 scalar formats and
`percentile_99_9` for 6. Bit-exact execution under the scaled contract now covers every scalar format with a
dyadic grid, up from three.

Per case the gate is: primitive rational conformance for the codebook (2,474 to 268,666 checks per codebook, both
backends, both arms; the count grows with the number of code pairs, see below); an independent first-image whole-graph reference (FX traversal, framework binary64
convolution on 16-bit limbs, Python integer recombination); 63 actual-node rational dot checks per arm
(3 per MAC node); eight images with identical CPU and CUDA layer codes, raw states, dots, outputs, predictions
and diagnostics in both arms; the B anchor replayed in its original batches of eight with every retained Top-5
list reproduced.

Primitive witness counts per codebook (17 codebooks): binary_pm1 2,474; ternary 2,510; fp4_e2m1 3,310;
posit4_es0 3,566; int4 3,706; fp5_e2m2 6,254; int5 7,034; fp6_e2m3 and fp6_e3m2 18,286; posit6_es1 19,310;
int6 19,834; fp7_e3m3 66,926; fp8_e5m2 250,414; fp8_e4m3fn 262,510; posit8_es1 266,606; int8 and q1_6 268,666.
Only the five 8-bit codebooks reach about 250,000; most of each count is the exhaustive code-pair product table.

Regression against v1: for fp6_e2m3, fp6_e3m2 and fp7_e3m3 the v2 engine reproduces the sealed v1 records bit
for bit on 32 of 32 images in both arms (all layer codes, raw states, dots and outputs), and the v1 widths
(23, 29, 31 bits).

## Certified lossless accumulator widths

Signed two's-complement bits, maximum over the 21 MAC layers. "Absolute" is the v1 definition
(bit length of max|a|·Σ|w|, plus sign). "Structural" uses the fact that inputs behind a ReLU are non-negative.
Per-layer and per-channel values are in the summary and in each case's `certificate.json`.

| Format (maxabs) | Absolute | Structural | Layers exact in binary64 | Exact accumulator |
|---|---:|---:|---:|---|
| ternary | 10 | 10 | 21/21 | int64 |
| binary_pm1 | 14 | 14 | 21/21 | int64 |
| int4 | 17 | 16 | 21/21 | int64 |
| fp4_e2m1 | 18 | 17 | 21/21 | int64 |
| int5 | 20 | 19 | 21/21 | int64 |
| posit4_es0 | 20 | 19 | 21/21 | int64 |
| fp5_e2m2 | 21 | 20 | 21/21 | int64 |
| int6 | 22 | 21 | 21/21 | int64 |
| fp6_e2m3 | 23 | 22 | 21/21 | int64 |
| int8, q1_6 | 26 | 25 | 21/21 | int64 |
| fp6_e3m2 | 29 | 28 | 21/21 | int64 |
| fp7_e3m3 | 31 | 30 | 21/21 | int64 |
| posit6_es1 | 43 | 43 | 21/21 | int64 |
| fp8_e4m3fn | 47 | 46 | 21/21 | int64 |
| posit8_es1 | 59 | 59 | 1/21 | int64, one RNE conversion |
| fp8_e5m2 | 75 | 74 | 0/21 | two 64-bit limbs, one RNE conversion |

The percentile recipe gives the same widths as maxabs for int5, int6, int8, fp4_e2m1 and fp5_e2m2, and 18 bits
(absolute) for int4. The INT8 layers range from 20 bits (1×1 downsample, K=64) to 26 bits (K=4608).

The integer and minifloat formats up to 7 bits, ternary, binary and posit4_es0 fit a signed 32-bit accumulator on
ResNet18 (at most 31 bits). posit6_es1 does not: it needs 43 bits. The 8-bit minifloats and posit8_es1 need 47
to 75 bits. int8 and q1_6 (26 bits) fit. Wherever the width exceeds 25 bits FP32 accumulation is no longer
guaranteed lossless (next section).

## Accumulator arithmetic alone (control minus wide)

Top-1 accuracy is equal between the exact arm and the FP32-FMA control arm in all 23 cases. The predicted Top-1
class is the same on every image in 22 cases. For fp8_e4m3fn it differs on 3 of 128 images: image 55 (wide 811,
control 589, which is the label), image 105 (wide 911, which is the label; control 658) and image 127 (272
against 271, both wrong). One image gained and one lost cancel, so the accuracy difference is 0.00 points; the
classes are not identical. The arms are numerically identical (every dot, every code) for every format up to
fp6_e3m2. They differ where the width exceeds what binary32 can hold:

| Case | Images with a differing raw dot | Images with a differing stored code | Changed Top-1 class | Changed ordered Top-5 |
|---|---:|---:|---:|---:|
| fp7_e3m3 | 15/32 | 0 | 0 | 0 |
| posit6_es1 | 32/32 | 2 | 0 | 0 |
| fp8_e4m3fn | 128/128 | 89 | 3 | 7 |
| posit8_es1 | 32/32 | 23 | 0 | 0 |
| fp8_e5m2 | 32/32 | 16 | 0 | 0 |

So an FP32 accumulator perturbs stored codes for the 8-bit floating formats and, for fp8_e4m3fn, changes
individual predictions (3 of 128) with no net accuracy change on this panel. Whether it costs accuracy needs
the larger panels of the next stage.

## B anchor replay (paired, development)

"Stable" scores Top-1 with the contract's tie order against B's retained Top-1 (CUDA `torch.topk`).
"Tie-aware" gives credit 1/t when the label is among t classes tied at the maximum, on both sides.

| Case | n | B top-1 | Wide top-1 | Wide − B, stable [pointwise 95%] | Tie-aware wide / B |
|---|---:|---:|---:|---:|---:|
| int8 maxabs | 128 | 69.53 | 70.31 | +0.78 [0.00, +2.34] | 69.40 / 70.18 |
| fp8_e4m3fn maxabs | 128 | 71.88 | 70.31 | −1.56 [−4.69, +1.56] | 70.14 / 69.69 |
| fp7_e3m3 maxabs | 32 | 71.88 | 68.75 | −3.12 [−9.38, 0.00] | 65.62 / 67.19 |
| fp6_e2m3 maxabs | 32 | 71.88 | 65.62 | −6.25 [−15.62, 0.00] | 63.80 / 63.80 |
| fp6_e3m2 maxabs | 32 | 59.38 | 65.62 | +6.25 [−6.25, +18.75] | 61.98 / 61.98 |
| fp8_e5m2 maxabs | 32 | 62.50 | 59.38 | −3.12 [−9.38, 0.00] | 61.56 / 62.34 |
| int8 percentile | 32 | 59.38 | 56.25 | −3.12 [−9.38, 0.00] | 56.88 / 56.88 |
| int6 percentile | 32 | 62.50 | 62.50 | 0.00 [−9.38, +9.38] | 59.48 / 61.04 |
| int5 percentile | 32 | 56.25 | 46.88 | −9.38 [−21.88, 0.00] | 51.17 / 53.96 |
| fp5_e2m2 percentile | 32 | 56.25 | 59.38 | +3.12 [0.00, +9.38] | 59.06 / 59.06 |

The remaining cases (int4, int5/int6 maxabs, fp4, fp5 maxabs, posits, ternary, binary) are at or near chance
in B and in the bridge; they exercise the arithmetic, not the quality. All rows are in `table.txt`.

### Top-1 on quantised logits depends on tie order

B quantises the logits (no last-layer exemption). With a low-bit codebook several classes often share the
maximum logit: 8/32 images for fp6_e2m3 and 10/32 for fp6_e3m2, in the bridge and in B alike. Top-1 then
depends on which tied class `torch.topk` returns, which is unspecified and differs between B (CUDA, FP32) and
v1 (CPU, binary64). Evidence: on the three v1 cases v2 reproduces v1's logits exactly, yet with a defined tie
order the Top-1 class matches v1's on only 26, 27 and 30 of 32 images.

Scored tie-aware, wide and B are identical for fp6_e2m3 (63.80 vs 63.80) and fp6_e3m2 (61.98 vs 61.98), where
the stable order shows −6.25 and +6.25 points. On the two 128-image cases the tie-aware differences are −0.78
(int8) and +0.45 points (fp8_e4m3fn). This is a candidate mechanism for the 3.9 to 4.7 point "transfer" gap in
the v1 results; lane L5 is testing it at 1k images. v2 defines the order (value descending, class index
ascending) and records the tied classes on both sides.

## Speed

Per-image execution with full traces (every layer's codes, raw state and dot hashed), batch 1, median over the
panel. CUDA timings were taken under the shared GPU lock. CPU timings (4 threads) ran without the lock while
other lanes used the machine and are indicative only.

| | CUDA wide | CUDA control | CPU wide |
|---|---:|---:|---:|
| Range over 23 cases | 0.18–0.35 s | 0.19–0.33 s | 0.77–2.0 s |
| int8 maxabs | 0.285 s | 0.287 s | 0.84 s |
| fp8_e5m2 (two-limb) | 0.316 s | 0.321 s | 1.96 s |

Throughput options, int8 maxabs, wide arm, CUDA, 64 images under the lock; every output, prediction and layer
code was compared with the sealed batch-1 records and found identical:

| Batch | Trace | Seconds per image |
|---:|---|---:|
| 1 | full (evidence of record) | 0.285 |
| 1 | none | 0.240 |
| 8 | layer codes only | 0.210 |
| 8 | none | 0.210 |

The native dot products take about 0.024 s of this; the rest is host-side NumPy post-operations, stores and
hashing, which is where further speed would come from.

At about 0.3 s per image a 10k panel costs under one hour per arm, inside the 3-hour target.

Cost of this milestone's measurement runs, from the run ledger up to the recorded cut-off (the last ledger
entry, epoch 1790858356, 2026-10-01 17:39): 145 invocations, 1,364 s wall-clock in total, of which 887 s under
the GPU lock (conformance 18 s, B replay 198 s, panels 1,101 s, throughput runs 46 s). Revision 1 of the summary
was sealed before the three throughput runs and reported 142 invocations, 1,317 s and 841 s.

## Not admitted, and why

- `log4`, `log6`, `log8`, `nf4`: finite levels are not dyadic, so no exact integer grid exists. They stay on
  the rational fallback. An exact algebraic accumulator for the log formats (integer coefficients over powers
  of 2^(1/4)) would be a separate design; it is not attempted here.
- Shared-exponent formats (bfp6, mxfp*): milestone M4.
- MobileNetV2 and MobileNetV3-Large: milestones M2 and M3. The export already handles both graphs (155 and 189
  nodes); the engine fails closed on ReLU6, hard-swish, hard-sigmoid and multiply until their definitions are
  frozen and witnessed.

## Reproduce

```bash
.venv/bin/python -m pytest tests/unit/test_scaled_bridge_v2_engine.py tests/conformance/test_scaled_bridge_v2_native.py -q
tools/run/scaled_bridge_v2_cases.sh artifacts/scaled_bridge_v2/plans/m1-cases.txt   # resumable; GPU steps take the lock
.venv-b/bin/python -m tools.run.scaled_bridge_v2 report --milestone M1 --run 2a64216184ac39ea --cutoff 1790858356.2406776
```

The report command names the run root (engine source digest) and the ledger cut-off. It never replaces a
published file: with these arguments it reports that revision 2 is identical; with a later cut-off or new
evidence it writes the next revision.

---

# Milestone MS: accumulator policies (contract 2.1)

Run 2026-10-01. Contract 2.1 and protocol MS were frozen and sealed before any policy panel
(`docs/analysis/scaled-bridge-v2-contract-2026-10-01.md`, `artifacts/scaled_bridge_v2/protocols/ms-4373badeebae72a0.json`).
Engine sources digest `99de2758260f852e…`. Machine result (the record): revision 2,
`results/summaries/scaled-bridge-v2/MS-99de2758260f852e/summary-r2.json` and `table-r2.txt` (bounded to the
protocol prefixes and a ledger cut-off; revision 1, `summary.json`/`table.txt`, is kept and reports the same
numbers).

Everything below is development evidence on a 32-image prefix of the screen-1k list. It admits the policies. It
is not a quality result: 32 images cannot rank accumulators.

## Result

**14 of 14 policy gates passed**: seven policies on two ResNet18 cases (int8 maxabs and fp7_e3m3 maxabs, the M1
exports). The dot accumulator is now pluggable: `sat.w<W>`, `sat.abs-<d>`, `sat.struct-<d>` (saturating integer),
`fp16[.x<e>]` and `f21[.x<e>]` (float, one rounding per multiply-add), next to `wide` and `control`.

Witnesses, per the protocol:

- Primitive rational witnesses, both backends: 4,996 checks. Saturation at the high end (387 sequences) and the
  low end (363), exact landing on an end point, recovery after a clamp, widths 2 to 63, 64-bit operand products,
  cross-check with the public int32 accumulator manifest; overflow to +infinity (55) and -infinity (51), the tie
  at the binary16 overflow threshold, sticky infinity, subnormal ties and underflow to zero, scale exponents,
  operands up to 2^32; padded, strided, dilated and grouped geometry.
- Unit tests (20, CPU): a float reference written from scratch in exact rationals, IEEE float16 arithmetic from
  NumPy for binary16, an integer reference for saturation, and engine-level tests (failed image, batch
  invariance next to a failed image, code pass-through).
- Actual-node rational dot checks on the first image of every policy and case: four fixed outputs per MAC node
  (one of them interior) plus the first output that reports an event; value and event count must match.
- Eight images with identical CPU and CUDA records, including accumulator counts and failures.
- `sat.abs-0` and `sat.struct-0` are bit-identical to the exact arm on 32 of 32 images with no saturation.
- The engine sources changed, so `wide` and `control` were regressed against the sealed M1 records: 8 of 8
  images identical on CUDA for both arms and both cases.

## What the 32-image panels show

"Failed" counts images whose accumulator became infinite; such an image has no prediction and is scored wrong.
"Event" is a saturation (integer) or an infinite result (float).

| Case | Policy | Failed | Top-1 (exact arm) | Images with a changed stored code | Images with an event | MAC nodes with an event |
|---|---|---:|---:|---:|---:|---:|
| int8 | sat.abs-0, sat.struct-0 | 0 | 65.6 (65.6) | 0 | 0 | 0 of 21 |
| int8 | sat.struct-4 | 0 | 65.6 (65.6) | 32 | 32 | 1 |
| int8 | sat.w16 | 0 | 21.9 (65.6) | 32 | 32 | 18 |
| int8 | fp16 | 32 | 0.0 (65.6) | n/a | 32 | 1 |
| int8 | fp16.x-12 | 0 | 65.6 (65.6) | 32 | 0 | 0 |
| int8 | f21 | 0 | 68.8 (65.6) | 32 | 0 | 0 |
| fp7_e3m3 | sat.abs-0, sat.struct-0 | 0 | 68.8 (68.8) | 0 | 0 | 0 |
| fp7_e3m3 | sat.struct-4 | 0 | 59.4 (68.8) | 32 | 32 | 3 |
| fp7_e3m3 | sat.w16 | 0 | 0.0 (68.8) | 32 | 32 | 21 |
| fp7_e3m3 | fp16, fp16.x-12, f21 | 0 | 68.8 (68.8) | 32 | 0 | 0 |

- Four bits below the certified structural width, saturation happens almost only in the first convolution (`conv1`,
  the 7×7 convolution on the signed network input): 16% of its outputs for int8 (18-bit register) and 28%
  for fp7_e3m3 (23 bits), on every image. For fp7_e3m3 two more nodes saturate on a few outputs.
- `fp16` without a scale exponent fails on every int8 image at `conv1` (35% of the outputs infinite). This is
  the expected consequence of accumulating integer code products in binary16 and is recorded, not clamped.
  With the binary point moved (`fp16.x-12`) no image fails.
- The float accumulators change stored codes on every image without failing. Their Top-1 on 32 images is
  within two images of the exact arm; that is all 32 images can say.

Per-node widths and event frequencies for every policy are in the summary (`nodes_with_event`, `widths`).

## Speed

Median seconds per image, batch 1, full trace. The CUDA panels ran in shared GPU mode (up to four jobs at once),
so these are indicative and are not timing claims: CUDA 0.28 to 0.39 s for every policy (the exact arm takes
0.29 to 0.35 s); CPU (4 threads, two panels at a time) 1.4 to 2.8 s for the saturating integer and 2.6 to
4.4 s for the float accumulators. Cost of the MS runs: 37 invocations, 618 s of wall-clock.

## Archived implementation

`artifacts/scaled_bridge_v2/implementations/99de2758260f852ebaf5fafeb83e8bce48f84cbf8b7cefdb38b5f80f68bbb05b/`
holds the sources (`py/scaled_bridge_v2/`), the built CPU and CUDA libraries with their compiler manifests
(`lib/`), a sealed `manifest.json` and `run.sh`. Another lane runs it from the repository root without touching
the live sources:

```bash
A=artifacts/scaled_bridge_v2/implementations/99de2758260f852ebaf5fafeb83e8bce48f84cbf8b7cefdb38b5f80f68bbb05b
$A/run.sh root                                                        # prints the run root of this digest
artifacts/agent_orchestration/gpu_run.sh $A/run.sh panel resnet18-int8-maxabs-b1 sat.w20 cuda 32
```

Superseded (correction 2026-10-02): this archive has no compact `predict` command and no B2 adapter. For 1k
panels use the archive of digest `7c6344af…` (section MB2, "Running B2 graphs and the 1k sweep from the
archive") or the MobileNet archive of section MN; do not run `panel` or `throughput` on 1000 images here.

Evidence lands under `artifacts/scaled_bridge_v2/runs/99de2758…/<case>/<policy>-<backend>/`. The archive was
checked by running it: it reports the same run root, and runs from the archive at batch 8 (CUDA, `sat.w16`) and
batch 4 (CPU, `fp16`) reproduced the sealed batch-1 records (8 of 8 images). Later edits of `tools/scaled_bridge_v2` do
not affect it. Limits of this archive: it runs the b1 (original recipe) exports only. (When it was archived the
B2 adapter did not exist; it was written in milestone MB2 and is used through the later archives.)

## Reproduce

```bash
.venv/bin/python -m pytest tests/unit/test_scaled_bridge_v2_engine.py tests/unit/test_scaled_bridge_v2_policies.py tests/conformance/test_scaled_bridge_v2_native.py -q
artifacts/agent_orchestration/gpu_run.sh .venv-b/bin/python -m tools.run.scaled_bridge_v2 policy-conformance
tools/run/scaled_bridge_v2_policies.sh "resnet18-int8-maxabs-b1 resnet18-fp7_e3m3-maxabs-b1" "sat.abs-0 sat.struct-0 sat.struct-4 sat.w16 fp16 fp16.x-12 f21" 2a64216184ac39ea
.venv-b/bin/python -m tools.run.scaled_bridge_v2 report --milestone MS --run 99de2758260f852e --cutoff 1790861431.6161683
```

The MS summary of record is revision 2 (`results/summaries/scaled-bridge-v2/MS-99de2758260f852e/summary-r2.json`,
`table-r2.txt`). Review 2 found that the report read every record and the whole ledger of the MS run root, which
is also the root an archived implementation writes into, so a later sweep would have changed the published
numbers. The report now reads the protocol prefixes only (32 CUDA and 8 CPU records per policy), the four
protocol regressions and the ledger up to the start of the last MS invocation (epoch 1790861431.6161683).
Revision 2 equals revision 1 in every reported number and in the table; it drops the count of later ledger
entries and adds the bounds. Checked on a copy of the run root with longer panels, a later ledger entry and an
extra regression added: the bounded report still equals revision 2.


# Milestone MB2: repaired-recipe (B2) graphs

Engine digest `7c6344af732d1bf3…`. Contract 2.1 (keys `B2_recipe_graphs`, `maxpool_code_passthrough`,
`unsigned_codebooks`, `top_k`, `B2_boundary`, `B2_not_exact`, `width_convention`) and `PROTOCOL_MB2` were sealed
before MS (`artifacts/scaled_bridge_v2/protocols/mb2-e825eb2c19fb3c2e.json`); nothing in them was changed. All
of this is development evidence on the first 32 images of the screen-1k list.

## Result

5 of 5 ResNet18 B2-default cases pass the gate (int8, int6, fp6_e2m3, fp7_e3m3, fp8_e4m3fn; recipe: boundaries
fused across ReLU, unsigned codes for non-negative integer tensors, MSE-searched scales, empirical bias
correction). Summary: `results/summaries/scaled-bridge-v2/MB2-7c6344af732d1bf3/summary.json`, `table.txt`.

| Case | Bits abs/range/struct | B2 Top-1, retained order / lowest index | Exact Top-1 | Exact − B2, retained order [95%] | Exact − B2, lowest index [95%] | Tie-aware exact / B2 | Control − exact | Images with every code equal to B2 | Codes that differ from B2 |
|---|---|---|---:|---|---|---|---:|---:|---:|
| int8 | 27/26/26 | 68.75 / 68.75 | 68.75 | 0.00 [0, 0] | 0.00 [0, 0] | 68.75 / 68.75 | 0.00 | 0 of 32 | 0.98% |
| int6 | 23/22/22 | 71.88 / 68.75 | 68.75 | −3.12 [−9.38, 0] | 0.00 [0, 0] | 68.75 / 68.75 | 0.00 | 4 of 32 | 0.25% |
| fp6_e2m3 | 23/23/23 | 65.62 / 68.75 | 68.75 | +3.12 [0, +9.38] | 0.00 [0, 0] | 67.19 / 67.19 | 0.00 | 7 of 32 | 0.28% |
| fp7_e3m3 | 31/31/30 | 68.75 / 68.75 | 65.62 | −3.12 [−9.38, 0] | −3.12 [−9.38, 0] | 65.62 / 67.81 | 0.00 | 0 of 32 | 1.56% |
| fp8_e4m3fn | 47/47/46 | 68.75 / 68.75 | 68.75 | 0.00 [0, 0] | 0.00 [0, 0] | 69.38 / 68.75 | 0.00 | 0 of 32 | 1.98% |

Thirty-two images: one image is 3.12 points. These numbers gate the engine; they are not a quality result.

What each gate item covered:

- **Adapter** (`b2_adapter.py`, `b2-export <folder>`): for every convolution and the linear layer, the integer
  weight units rebuilt from the exported manifest codes reproduce B2's FP32 reconstructed weights bit for bit,
  signed zero included; every B2 codebook table (levels, codes, midpoint ties, FP32 midpoints) equals the engine
  codebook; activation scales are read from their hexadecimal FP32 form. Nothing is re-quantised. Adapted
  exports: `artifacts/scaled_bridge_v2/exports/9ecfd61292dc72b1…/<case>/` (16 MB each).
- **Graph shape** (per case): 31 storing boundaries, 17 arithmetic nodes that do not store (9 convolutions and
  8 residual adds fused into the following ReLU), 1 max-pool that forwards codes.
- **Unsigned codebooks**: `int8.unsigned` (0..255) and `int6.unsigned` (0..63) are separate codebooks, code =
  level, ties to the even code. Primitive rational conformance on both backends: 268,666 checks each for int8
  and int8.unsigned, 19,834 each for int6 and int6.unsigned, 18,286 (fp6_e2m3), 66,926 (fp7_e3m3), 262,510
  (fp8_e4m3fn). The oracle for an unsigned book is built from the bit count alone, not from the export.
- **Whole-graph witness**: the independent FX/limb reference (extended for unstored binary64 values and the
  code-forwarding max-pool) equals the exact arm on the first image of every case, every layer hash included.
- **Actual-node dots**: 83 rational dot checks per arm on the first image (21 MAC nodes).
- **CPU = CUDA** on 8 images, both arms, all five cases.
- **B2 replay**: B2's own simulator, rebuilt from the export's configuration (identity reproduced), reproduces
  its retained Top-5 on 32 of 32 images in every case. Its cached inputs are bit-identical to the engine's.
- **Regression**: with the engine sources changed, 18 of 18 sealed record sets are reproduced bit for bit on
  CUDA, 8 images each: the 7 MS policies and the exact arm against the MS root and the control arm against the
  M1 root, for int8 and fp7_e3m3 (original recipe).
- Tests: 41 pass on CPU, 46 with the CUDA variants.

## Exact engine against the B2 simulator

Bit identity with B2 is not expected and is not a gate (contract key `B2_boundary`): B2 reduces with FP32
framework convolutions on FP32 reconstructed operands and stores in FP32; the engine computes exact code-domain
dots and stores in binary64. Measured on 32 images:

- The input boundary is identical in all cases. At the first ReLU boundary at most 4 codes in a million differ
  (FP32 against exact arithmetic at near-tie stores). The difference then spreads with depth: 0.001 to 0.012%
  of codes at the end of the first block, 0.03 to 0.43% at the first block of layer 2, 0.7 to 6% at the end of
  layer 3, 3.8 to 26% at the end of layer 4 and 6 to 45% of the logit codes. Over all storing boundaries 0.25
  to 1.98% of codes differ.
- Images on which every stored code equals B2's: 0, 4, 7, 0, 0 of 32. The first differing boundary is the first
  ReLU on 113 of the 149 differing images.
- Top-1 class under the contract tie rule (applied to both sides) equals B2's on 32, 32, 32, 30 and 31 of 32
  images (int8, int6, fp6_e2m3, fp7_e3m3, fp8_e4m3fn). In accuracy that is one image lost for fp7_e3m3
  (−3.12 points) and no change for the others.
- A small rounding difference at the first boundary therefore reaches the logits on most images while leaving
  the predicted class unchanged on 157 of 160. A 1k comparison between the simulator and the exact engine should
  expect code-level differences everywhere and should be scored on predictions with the tie rule below.

## Top-k tie rule

Contract: descending output value, ties by ascending class index (lowest class index first). B2 retained its
predictions with `torch.topk` on CUDA FP32 logits, whose order among equal logits is unspecified. Because the
logits are quantised, equal maxima occur: B2's maximum logit is shared by several classes on 0, 3, 1, 3 and 0 of
32 images (int8, int6, fp6_e2m3, fp7_e3m3, fp8_e4m3fn), and the two conventions give a different B2 Top-1 class
on 0, 2, 1, 2 and 0 images and a different ordered Top-5 on 13, 25, 26, 28 and 25. For int6 and fp6_e2m3 the
whole exact-minus-B2 difference under the retained order (−3.12 and +3.12 points) is tie order: under the
contract rule it is 0.00. Every paired difference against B2 is therefore reported under both conventions, with
the tie-aware expected accuracy next to them.

## Accumulator arithmetic on B2 graphs (control minus exact)

Top-1 class identical on 32 of 32 images in all five cases. Stored codes identical on every image for int8,
int6, fp6_e2m3 and fp7_e3m3 (fp7_e3m3: the raw dot of `conv1` differs on 7 images without changing a code).
fp8_e4m3fn: the `conv1` dot differs on 32 images, stored codes on 16, ordered Top-5 on 1.

## Certified widths on B2 graphs

Per-node widths are in the summary (`cases[].layers`) and in each `certificate.json`. Network maxima
(absolute / structural) against the original recipe: int8 27/26 (26/25), int6 23/22 (22/21), fp6_e2m3 23/23
(23/22), fp7_e3m3 31/30 (31/30), fp8_e4m3fn 47/46 (47/46). The integer formats need one bit more than under the
original recipe because non-negative tensors now use the unsigned range (largest input code 255 instead of 128,
63 instead of 32). Smallest node width: 21 (int8), 17 (int6), 18 (fp6_e2m3), 26 (fp7_e3m3), 42 (fp8_e4m3fn).

Convention (contract key `width_convention`): 1 sign bit + bit_length(max over output channels of
max|a| · Σ|w|), a over every level of the input codebook, w the exported weight integers of the channel, unit =
product grid, per channel, then maximum over channels and nodes; the bias is not in this register. Worked
example on the B2 graph, fp6_e2m3, `layer4_1_conv1` (K = 4,608): the published minifloat formula gives 26 bits
(largest product counted as 2^12, tap count as 2^13); the same worst case without rounding both factors up is
4,608 · 56 · 56 = 14,450,688 < 2^24, 25 bits; the certificate uses the worst channel's trained weights,
Σ|w| = 63,683 units, 56 · 63,683 = 3,566,248 < 2^22, 23 bits. One bit is the separate rounding, two bits are
the actual weights. For int8, `layer4_1_conv1`: 255 · 144,542 = 36,858,210 < 2^26, 27 bits; behind a ReLU with
the unsigned range the sum is one-sided, so the structural width is 26. Caveat (review 2, item 8): the reading
of the published formula, including its sign-bit accounting, is taken from the related-work audit and has not
been checked against the paper itself; check it before the comparison is printed.

## What of B2 cannot be run exactly

- `quantize_input=false` (diagnostic recipes): the first convolution has no input code grid. The adapter and
  the engine refuse the export.
- `fused_all` and the MobileNet operators (hard-swish, hard-sigmoid, ReLU6, tensor multiply): not defined in
  contract 2.1; the engine rejects the operators. Next continuation.
- Equalised recipes (`equalization=cle`): not tried. The graph is the same operator set, so the adapter should
  apply; no case was gated.
- B2's FP32 convolution and FP32 stores are not reproduced by design: the engine is the exact reference, B2 is
  the simulator. The measured consequence is the section above.
- `quantize_logits=false` works already (an unstored output), but no such case was gated.

## Running B2 graphs and the 1k sweep from the archive

`artifacts/scaled_bridge_v2/implementations/7c6344af732d1bf324c07b285921f63393150cdb3875356c86237d934fdc2f92/`
is the archive of this engine (sources, both libraries with compiler manifests, sealed manifest, `run.sh`). It
supersedes the `99de2758…` archive, whose records it reproduces, and it runs both recipe families. `run.sh` now
refuses to run if its run root would not resolve to the archived digest.

```bash
A=artifacts/scaled_bridge_v2/implementations/7c6344af732d1bf324c07b285921f63393150cdb3875356c86237d934fdc2f92
artifacts/agent_orchestration/gpu_run.sh $A/run.sh predict resnet18-int8-default-b2 sat.w20 cuda 0 1000 --batch 8
```

`predict CASE POLICY BACKEND START STOP` is the compact runner for panels: one sealed file per call under
`runs/7c6344af…/<case>/predictions/` with, per image, Top-5 (contract tie order), output hash, tied classes,
the failure record and the non-zero accumulator event counts per node; no layer traces (about 1.6 KB per image).
Checked through the archive on 32 images: the exact arm at batch 8 equals the sealed batch-1 records (32 of
32); `sat.abs-0` equals the exact arm with no saturation (32 of 32); `sat.w16` on int8 saturates on every image
(Top-1 0 of 32; the certified width is 27); `f21` on fp7_e3m3 has no non-finite result and the exact arm's
Top-1. `panel` remains the full-trace command (38 KB per image) and should not be used for 1k panels.
Seconds per image in the summary (0.22 to 0.25 s on CUDA, batch 1, full trace) were measured in shared GPU
mode and are indicative only. Cost of the MB2 runs: 60 invocations, 328 s of wall-clock.

Incident, for the record: the first MB2 launch started five GPU jobs that each compiled the CUDA library of
the new source digest at the same time; the builds raced and left a library that did not match its manifest
(the loader refused it). One conformance record written during that window was deleted; its ledger entry
(`conformance int8`, 19:09:46, 3 s) remains and is counted in the 60 invocations, and what the deleted record
said can no longer be checked. The library was rebuilt once, and the runner scripts now build before starting
parallel jobs. The CPU half of that launch (certificates, whole-graph references, `wide-cpp` and `control-cpp`,
8 images per case) did not fail and was kept; review 3 reproduced it. The second launch lost three cases to
CUDA out-of-memory with four B2 replays at once; they were rerun one at a time. No record of a failed GPU job
was kept or reused.

## Reproduce

```bash
.venv/bin/python -m pytest tests/unit/test_scaled_bridge_v2_engine.py tests/unit/test_scaled_bridge_v2_policies.py tests/unit/test_scaled_bridge_v2_b2.py tests/conformance/test_scaled_bridge_v2_native.py -q
artifacts/agent_orchestration/gpu_run.sh .venv-b/bin/python -m tools.run.experiment_b2 export --model resnet18 --format int6 --recipe default   # lane L1's command; prints the export folder
.venv-b/bin/python -m tools.run.scaled_bridge_v2 b2-export artifacts/experiment_b2/exports/<folder>
tools/run/scaled_bridge_v2_b2.sh "resnet18-int6-default-b2"      # one case at a time: the B2 replay needs GPU memory
.venv-b/bin/python -m tools.run.scaled_bridge_v2 report --milestone MB2 --run 7c6344af732d1bf3 --cutoff 1790864059.7781281
```

# Milestone MN: MobileNetV2 and MobileNetV3-Large on repaired-recipe (B2) graphs (contract 2.2)

Run 2026-10-02, 00:00 to 03:57 (+05). Contract 2.2 and protocol MN were frozen and sealed before any MobileNet
panel (`artifacts/scaled_bridge_v2/protocols/mn-9f55f6139786bdfe.json`; contract document, section "Contract
2.2"). Engine sources digest `1f75c9232c8a0202…`. Machine result:
`results/summaries/scaled-bridge-v2/MN-1f75c9232c8a0202/summary.json` and `table.txt` (bounded to the protocol
prefixes and the ledger cut-off `MN_LEDGER_CUTOFF` in `report.py`). Everything below is **development evidence**
on prefixes of the screen-1k list (8 images CPU, 32 images CUDA and replay, 8 images per accumulator policy).
It admits the cases; it is not a quality comparison. No held-out image was used; no panel exceeds 32 images.

## Result

- **10 of 10 cases pass every gate**: `{mobilenet_v2, mobilenet_v3_large} × {int8, int6, fp6_e2m3, fp7_e3m3,
  fp8_e4m3fn}`, B2 recipe `default`. Per case: adapter checks (weights reproduce B2's FP32 reconstruction bit for
  bit; codebook tables; node rows equal the independently loaded FX graph), primitive rational conformance of
  every codebook including the 2.2 operators (int8: 446,607 checks, of which 171,072 stored code products and
  6,303 ReLU6/hard-sigmoid/hard-swish witnesses), certificates, first-image FX/limb whole-graph reference (wide),
  first-image whole-graph witness of the control arm written from the contract text and fed from lane L1's raw
  export (65 stored tensors and 53 MACs for V2, 121 and 64 for V3, 0 problems in all 10 cases), actual-node
  rational dot checks in both arms, identical CPU and CUDA records on 8 images in both arms, and a 32-image
  replay of B2's own simulator that reproduces B2's retained Top-5 on 32 of 32 images with inputs bit-identical
  to the engine inputs.
- **8 of 8 accumulator-policy gates pass** on the int8 case of each network (`sat.struct-0`, `sat.struct-4`,
  `fp16.x-9` for V2 / `fp16.x-8` for V3, `f21`): 8 images CPU = CUDA including counters, first-image rational
  dot checks, whole-graph policy witness on the first image (0 problems). The certified lossless width
  (`sat.struct-0`) is identical to the exact arm with no saturation on 8 of 8 images for both networks;
  `sat.struct-4` saturates on 8 of 8 images (38 nodes for V2, 44 for V3) and loses every Top-1 on these 8 images;
  `fp16.x<e>` and `f21` produce no non-finite value and change the Top-1 class on 0 (V2) and 1 (V3) of 8 images.
- **Regression (MN-R)**: 82 of 82 sealed record sets of the previous digest reproduce bit for bit on 8 images
  under the new sources: 8 ResNet18 B2 cases × (wide, control and the four policies L8 gated) on CUDA, wide and
  control on CPU, and the 2 original-recipe cases × 9 record sets (control against `2a642161…`, wide and the MS
  policies against `99de2758…`). Records: `runs/1f75c923…/regress/`.

## Certified widths by node kind

Signed bits, sign included, convention of section "Width convention" of the contract (absolute / structural /
closure; the closure width is new in 2.2 and propagates input intervals through hard-swish, ReLU6 and the
squeeze-excite product). Maximum over the nodes of each kind:

| Case | stem | depthwise (K = 9; V3 also 25) | pointwise | SE reduce | SE expand | classifier | network | SE product |
|---|---|---|---|---|---|---|---|---|
| V2 int8 | 19/19/19 | 20/20/20 | 25/24/24 | - | - | 24/24/24 | 25/24/24 | - |
| V2 int6 | 15/15/15 | 16/16/16 | 21/20/20 | - | - | 21/20/20 | 21/20/20 | - |
| V2 fp6_e2m3 | 16/16/16 | 16/16/16 | 21/20/20 | - | - | 21/21/20 | 21/21/20 | - |
| V2 fp7_e3m3 | 24/24/24 | 24/24/24 | 29/28/28 | - | - | 29/29/28 | 29/29/28 | - |
| V2 fp8_e4m3fn | 40/40/40 | 40/40/40 | 45/44/44 | - | - | 45/45/44 | 45/45/44 | - |
| V3 int8 | 19/19/19 | 20/20/20 | 24/24/23 | 23/23/23 | 23/22/22 | 24/24/23 | 24/24/23 | 17 |
| V3 int6 | 15/15/15 | 16/16/16 | 20/20/19 | 19/19/19 | 19/18/18 | 20/20/19 | 20/20/19 | 13 |
| V3 fp6_e2m3 | 16/16/16 | 17/17/17 | 21/21/21 | 21/21/20 | 20/19/19 | 21/21/21 | 21/21/21 | 13 |
| V3 fp7_e3m3 | 24/24/24 | 25/25/25 | 29/29/28 | 29/29/28 | 27/27/27 | 29/29/29 | 29/29/29 | 21 |
| V3 fp8_e4m3fn | 40/40/40 | 41/41/41 | 45/45/44 | 45/45/44 | 43/43/43 | 45/45/45 | 45/45/45 | 37 |

- Depthwise nodes (nine or twenty-five taps) need 4 to 5 bits fewer than the pointwise nodes of the same
  network; the network width is set by pointwise or classifier nodes, as in ResNet18 (whose B2 widths were
  27/26 for int8 with K up to 4,608). For int8 the MobileNets need 24 to 25 bits against ResNet18's 26 to 27.
- The squeeze-excite product (hard-sigmoid code × activation code) needs 17 bits for int8 (255 × 255 for the
  three unsigned ReLU blocks) and 13 for int6; it is a single product, not a sum, and is exact in binary64 in
  every case.
- The closure width is at most one bit below the structural width (hard-swish's minimum -0.375 bounds the
  negative side of the inputs that follow it; behind the average pool the closure uses a ≥ 0). `sat.struct-<d>`
  keeps the 2.1 structural width so that ResNet18 evidence stays comparable.
- Every MAC dot of every case is below 2^53 in product units (widest 45 bits, fp8_e4m3fn), so it is exact in
  binary64 before the scale factors are applied; no case needs the two-limb path (53 of 53 and 64 of 64 nodes).

## Exact engine against the B2 simulator (32 images, paired, development)

Top-1 percent and paired differences wide − B2 with the v1 paired bootstrap (pointwise 95%), B2 under its
retained `torch.topk` order:

| Case | wide | B2 | wide − B2 pp [95%] | wide − B2, contract tie rule | control − wide | stored codes ≠ B2 |
|---|---:|---:|---|---:|---:|---:|
| V2 int8 | 71.88 | 71.88 | +0.00 [+0.00, +0.00] | +0.00 | +0.00 | 7.24% |
| V2 int6 | 75.00 | 71.88 | +3.12 [+0.00, +9.38] | +0.00 | +0.00 | 2.15% |
| V2 fp6_e2m3 | 65.62 | 65.62 | +0.00 [-9.38, +9.38] | -3.12 | +0.00 | 3.53% |
| V2 fp7_e3m3 | 75.00 | 68.75 | +6.25 [+0.00, +15.62] | +6.25 | +0.00 | 6.05% |
| V2 fp8_e4m3fn | 71.88 | 75.00 | -3.12 [-9.38, +0.00] | -3.12 | +0.00 | 4.64% |
| V3 int8 | 68.75 | 65.62 | +3.12 [+0.00, +9.38] | +6.25 | +0.00 | 17.02% |
| V3 int6 | 15.62 | 15.62 | +0.00 [-9.38, +9.38] | +0.00 | +0.00 | 4.25% |
| V3 fp6_e2m3 | 62.50 | 59.38 | +3.12 [-6.25, +12.50] | +0.00 | +0.00 | 11.69% |
| V3 fp7_e3m3 | 68.75 | 68.75 | +0.00 [+0.00, +0.00] | +0.00 | +0.00 | 19.72% |
| V3 fp8_e4m3fn | 62.50 | 65.62 | -3.12 [-9.38, +0.00] | -3.12 | +0.00 | 23.35% |

- On 32 images no interval excludes zero in a way that matters: the largest difference is +6.25 pp (2 images).
  These are paired 32-image prefixes; they say the exact engine and B2 classify alike, not which is better.
- Stored codes differ far more than on ResNet18 (0.25 to 1.98% there). The divergence starts at
  rounding level (V2 int8: 1 per 100,000 codes at the first stored boundary, 0.8% by block 4) and grows with
  depth to 20-80% of the codes of the last blocks and the logits (V3 int8: `features_16_0` 82%, logits 77%), a
  cascade of ±1-code differences through 17 (V2) or 15 (V3) blocks with linear bottlenecks, squeeze-excite and
  hard-swish computed in binary64 here and in FP32 in B2. The predicted class is still equal on most images.
  This is the size of the "simulator vs exact" gap for deep MobileNets; it is not a gate (contract key
  `B2_boundary`).
- MobileNetV3 int6 collapses in B2 itself (15.6% Top-1 on these 32 images); the exact engine reproduces the
  collapse. It is a property of the B2 recipe at 6 bits on V3, not of the engine.
- Control (sequential FP32 FMA) minus exact: Top-1 identical on 32 of 32 images in all 10 cases.

## Speed (indicative only)

Median CUDA seconds per image, batch 1, full trace, shared GPU (other lanes' jobs running): 0.47 to 0.69 s; CPU
0.7 to 1.7 s. Not exclusive; do not report as throughput. Cost: 183 invocations, 1,811 s of wall-clock in the
ledger (`gpu_lock_declared` is false: jobs ran in `gpu_run.sh` shared slots, which do not set that flag).

## Archived implementation and how a sweep lane runs MobileNet cases

`artifacts/scaled_bridge_v2/implementations/1f75c9232c8a0202482fe9bce9a46360c7cc0999c5b0bcf7df5ebcc0446fc863/`
(1.6 MB; sources, CPU and CUDA libraries, manifest, `run.sh`). Checked by running it: `run.sh root` prints the
`1f75c923…` root; `predict mobilenet_v3_large-int8-default-b2 wide cuda 0 32 --batch 8` equals the sealed
batch-1 records on 32 of 32 images; `predict mobilenet_v2-fp6_e2m3-default-b2 sat.w16 cuda 0 32 --batch 8`
runs; `predict resnet18-int8-default-b2 …` is refused (no case gate of this digest).

```bash
M=artifacts/scaled_bridge_v2/implementations/1f75c9232c8a0202482fe9bce9a46360c7cc0999c5b0bcf7df5ebcc0446fc863
$M/run.sh root
artifacts/agent_orchestration/gpu_run.sh $M/run.sh predict mobilenet_v3_large-int8-default-b2 sat.w20 cuda 0 1000 --batch 8
```

`predict` (review 3, item 3) now refuses a case without a passing `gate.json` of its digest, and refuses a
saturating or float policy unless a policy gate of the same family passed on a case of the same model under that
digest. ResNet18 keeps running from archive `7c6344af…` (lane L8); MN-R shows the two digests give identical
ResNet18 records. The review-3 items taken along: the B2 gate requires all 32 retained predictions and
bit-identical replay inputs (item 4), the reviewer's whole-graph witness is now an engine witness
(`graph_witness.py`, adopted and extended to groups, control arm, the 2.2 operators; required by the case and
policy gates of B2 graphs; unit-tested on tiny MobileNet-like graphs on both backends), the `fc`-after-pool width
(item 6) is answered by the closure width, and the document items 2 and 5 are corrected above.

## Limits

- 32 images per case, 8 per policy: admission evidence only. Exact-store ties were not counted on real images.
- Only the `default` recipe. `fused_all` is defined by 2.2 and unit-tested, not measured; `quantize_input=false`
  stays simulator-only; shared-exponent formats, the log formats and NF4 are not admitted (design note in the
  contract).
- The B2 gate record's `scope` string still reads "contract 2.1" (text in `b2_replay.py`, not an engine source);
  the engine digest implements 2.2.
- L1 exports made by this lane with L1's command (8 new folders, 41 MB for V2 and 63 MB for V3 each, under
  `artifacts/experiment_b2/exports/`: `ff312f0f…` V2 fp6_e2m3, `3c5aa527…` V3 fp6_e2m3, `ae7f59fb…` V2 fp8_e4m3fn,
  `8a41571a…` V3 fp8_e4m3fn, `6e6775a9…` V2 int6, `6e115555…` V3 int6, `f999fd1c…` V2 fp7_e3m3, `33583e47…` V3
  fp7_e3m3). Each matched a pre-existing B2 configuration with 1000 retained predictions; L1's command rewrote
  those configuration records with identical content (md5 checked). One export attempt died with CUDA OOM and
  wrote nothing.

## Reproduce

```bash
.venv-b/bin/python -m tools.run.scaled_bridge_v2 seal-protocol MN                 # sealed once; prints the path
artifacts/scaled_bridge_v2/logs/mn-regress.sh                                     # MN-R
.venv-b/bin/python -m tools.run.scaled_bridge_v2 b2-export artifacts/experiment_b2/exports/<folder>
tools/run/scaled_bridge_v2_mn.sh case mobilenet_v2-int8-default-b2                # one case at a time
tools/run/scaled_bridge_v2_mn.sh policies mobilenet_v2-int8-default-b2 "sat.struct-0 sat.struct-4 fp16.x-9 f21"
.venv-b/bin/python -m tools.run.scaled_bridge_v2 report --milestone MN --run 1f75c9232c8a0202 --cutoff 1790890604.0241377
gpu_run.sh env SCALED_BRIDGE_V2_CUDA=1 .venv-b/bin/python -m pytest tests/unit/test_scaled_bridge_v2_*.py tests/conformance/test_scaled_bridge_v2_native.py -q   # 73 passed
```
