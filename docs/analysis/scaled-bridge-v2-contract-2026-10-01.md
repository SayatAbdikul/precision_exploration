# Scaled code-domain bridge v2: contract and development protocol

Frozen 2026-10-01, before any v2 panel was run. Machine-readable text:
`tools/scaled_bridge_v2/contract.py` (`CONTRACT`, version
`scaled-code-domain-bridge-2.0`, and `PROTOCOL_M1`). The v1 contract
(`docs/analysis/scaled-bridge-v1-contract-2026-09-28.md`) is unchanged and its
evidence is not relabelled. v2 keeps every v1 arithmetic definition and
restates only what it generalises.

## What v2 changes

1. **The recipe is data.** The engine consumes an export (schema
   `scaled-bridge-export-2`, documented at the top of
   `tools/scaled_bridge_v2/export.py`): per tensor the codebook, the scale and
   whether the boundary stores; per MAC node the weight grid integers,
   per-output-channel weight scales and the bias. A codebook is a list of
   ascending integer units on a 2^-shift grid with its codes and midpoint tie
   preferences; an unsigned activation range is simply a codebook without
   negative units. No kernel or engine code names a format or a recipe.
2. **Unstored (fused) boundaries.** A node exported with `store: null` passes
   its binary64 value on without rounding to a code. ReLU, max-pool, residual
   add and the output are defined on such values. A MAC or an average pool
   whose input is not a stored code tensor fails closed.
3. **Exact dot beyond 2^53.** The wide arm sums code-grid integer products
   exactly in a signed accumulator certified per output channel: 64-bit when
   max|a|·Σ|w| < 2^63, two 64-bit limbs when it is below 2^127. The integer
   is converted to binary64 once, round-to-nearest-even. Below 2^53 units this
   conversion is exact and the definition coincides with v1. Above it the
   conversion is one explicit rounding of the exact sum; nothing is
   approximated before it.
4. **Accumulator policies.** The dot accumulator is the only component an arm
   may change. `wide` (exact) and `control` (sequential binary32 FMA, v1's
   control arm) exist for every admitted case. The control arm is computed on
   grid integers; this equals code-level FMA because every operand is exact in
   binary32 and no overflow or subnormal can occur in the admitted range.
5. **Top-5 order is defined.** Descending value, ties by ascending class
   index. v1 used `torch.topk` on CPU binary64 and B used `torch.topk` on CUDA
   FP32; neither defines the order among equal logits. Low-bit codebooks
   quantise the logits too, so ties at the maximum are possible. v2 records
   the classes tied at the maximum on both sides and reports a tie-aware
   expected Top-1 next to the ordinary paired differences.
6. **Certificates.** Per MAC node and output channel: the v1 absolute bound,
   the tight prefix interval for the input codebook range, and the interval
   when the producer is a ReLU-class operator (input codes ≥ 0). The engine
   checks the structural input range on every call. The resulting signed
   widths per layer are a result in their own right.

## Unchanged from v1

Weight codes and scales are B's (FP32 scale bits promoted exactly). MAC
post-operations: `RNE64(RNE64(dot·s_in)·s_w[c])`, then an RNE64 add of the
binary64 bias. Residual: RNE64 reconstruction of each operand, RNE64 add.
ReLU and max-pool act on code levels before reconstruction. Global average
pool: exact code sum, RNE64 divide by the count, RNE64 multiply by the scale.
Store: RNE64 divide by the scale, nearest exact dyadic level, code-parity then
code-order midpoint ties, finite endpoint clipping, canonical +0. Nonfinite
values fail closed.

## Not admitted

`log4`, `log6`, `log8` and `nf4` have finite levels that are not dyadic
rationals, so no exact integer grid exists; they stay on the rational
fallback. Shared-exponent formats need the block contract (milestone M4).

## Witnesses required for admission

Per codebook: manifest-enumerated rational store oracle, exhaustive code-pair
products in both arms and both backends, independently indexed small
convolutions (grouped, depthwise, strided, dilated, padded), a long FMA
cancellation sequence, two-limb and RNE-conversion witnesses, rational
post-operation sequences, special-value failures. Per case: a first-image
whole-graph reference computed a different way (FX traversal, framework
binary64 convolution on 16-bit limbs, Python integer recombination), three
actual-node rational dot checks per MAC node in both arms, eight images with
identical CPU and CUDA layer codes, raw states, outputs, predictions and
diagnostics, and the B anchor replayed in its original batches of eight.
Agreement between backends alone is not a reference.

## Development protocol M1 (ResNet18 scalar formats)

Cases, fixed here: recipe family b1 (original B scales and weight codes) on
ResNet18; `maxabs` for int4, int5, int6, int8, q1_6, fp4_e2m1, fp5_e2m2,
fp6_e2m3, fp6_e3m2, fp7_e3m3, fp8_e4m3fn, fp8_e5m2, posit4_es0, posit6_es1,
posit8_es1, ternary, binary_pm1; `percentile_99_9` for int4, int5, int6, int8,
fp4_e2m1, fp5_e2m2.

Samples: the frozen ImageNet screen-1k list in ascending SHA256 order; prefix
8 for the CPU/CUDA gate, prefix 32 for the B replay panel, prefix 128 for at
most two cases (int8 maxabs, fp8_e4m3fn maxabs). No calibration is performed;
scales come from B's retained calibration-2k evidence. Nothing is tuned on
these images.

Statistics: paired Top-1/Top-5 differences wide−B, control−B, wide−FP32,
control−FP32 and control−wide using the v1 paired multinomial bootstrap
(10,000 resamples, seed 20260928, pointwise 95%); tie-aware expected Top-1;
first-divergence layer counts. Thirty-two images support a gate, not a
quality claim.

Stop rule: no extension beyond the stated prefixes. A case with any witness
failure is reported as failed and not admitted. No label-driven contract
repair. Every measurement invocation seals its wall-clock seconds in the run
ledger; GPU timings count only under the shared GPU lock.

## Contract 2.1: accumulator policies and repaired-recipe graphs

Frozen 2026-10-01, before any accumulator-policy panel and before any B2 graph was run. Machine-readable text:
`CONTRACT_2_1`, `PROTOCOL_MS` and `PROTOCOL_MB2` in `tools/scaled_bridge_v2/contract.py`, sealed at
`artifacts/scaled_bridge_v2/protocols/ms-4373badeebae72a0.json` and `mb2-e825eb2c19fb3c2e.json`. Every rule of
2.0 holds; 2.1 only adds.

### Accumulator policies

The policy name is the complete specification and labels all evidence.

| Name | Accumulator |
|---|---|
| `wide` | exact integer sum, certified lossless (2.0) |
| `control` | sequential binary32 fused multiply-add (2.0) |
| `sat.w<W>` | signed saturating integer, the same W bits at every MAC node |
| `sat.abs-<d>` | signed saturating integer; per node W = certified absolute width of the node minus d |
| `sat.struct-<d>` | the same with the certified structural width of the node |
| `fp16`, `fp16.x<e>` | IEEE binary16 |
| `f21`, `f21.x<e>` | 21-bit float: sign 1, exponent 8 (bias 127), fraction 12 |

**Saturating integer.** A signed two's-complement register of W bits (2 to 63) whose unit is the exact product
grid, 2^-(input shift + weight shift), zero point 0. From 0, in the reduction order of the control arm
(input channel, kernel row, kernel column; padded positions contribute a zero product), each exact integer
product is added exactly and the sum is clamped to [-2^(W-1), 2^(W-1)-1] after every add. It never wraps. The
final integer is converted like the wide dot: one RNE to binary64 (exact below 2^53 units) and an exact
power-of-two scaling. The retained scales, the bias add and the stores are the shared binary64 rules.
`sat.abs-0` and `sat.struct-0` are lossless by the certificate: they must reproduce the wide arm bit for bit
with no saturation event, and the gate checks this. The floor of a relative width is 2 bits.

**Float accumulators.** p significand bits (binary16: p = 11, exponents -14 to 15, subnormal spacing 2^-24;
21-bit: p = 13, exponents -126 to 127, subnormal spacing 2^-138; manifest
`tools/scaled_bridge_v2/manifests/fp21_e8m12_accumulator.json`). From +0, in the same order,
state = RNE(state + a·w·2^e): the product a·w of the two code levels is exact, the add is exact, and there is
exactly one round-to-nearest-even per multiply-add. Subnormals are gradual. Overflow gives ±infinity by the
IEEE rule, and an infinite state stays infinite. NaN is unreachable because every product is finite. The finite
result is multiplied by 2^-e exactly. Zero is canonical +0.

**Scale exponent e.** Zero unless the name carries `.x<e>` (|e| ≤ 64). Code levels are the manifest values, so
an integer format accumulates integers: two unsigned INT8 codes can multiply to 65,025, which is already at the
limit of binary16 (largest finite value 65,504). `fp16` therefore overflows on integer formats by construction.
`fp16.x-12` places the binary point twelve bits lower. The exponent is the designer's choice of binary point; it
costs nothing in hardware and is always part of the policy name, so no result can hide it.

**Non-finite results.** A MAC node with any infinite accumulator output is a recorded failure of that image.
The record names the node and counts the infinite outputs by sign. The image has no output and no prediction, is
scored as incorrect and is reported separately as a failure. Nothing is clamped or replaced. Other images in a
batch are unaffected.

**Record.** Per image and MAC node. Saturating: width, outputs, outputs with at least one clamp, number of adds
clamped at the high and at the low end (each capped at 65,535 per output). Float: outputs, infinite outputs,
reduction steps that ended infinite.

### Repaired-recipe (B2) graphs

- **Adapter.** A `b2-recipe-export-1` export is adapted without re-quantising. Weight units are the manifest
  levels of the exported weight codes. Weight scales, activation scales and the bias-corrected biases are the
  exported FP32 values promoted exactly to binary64. A boundary with `quantizes=false` becomes `store: null`.
  The unsigned variant becomes its own codebook (units 0 to 2^bits-1). The adapter verifies that level × scale
  in FP32 reproduces B2's reconstructed weights bit for bit.
- **Max-pool code pass-through.** A max-pool with `store: null` whose input is a stored code tensor forwards
  codes: the maximum code level of each window, padding ignored, is itself a level of the same codebook.
  Codes, codebook and scale pass on unchanged and nothing is rounded. Certificates look through such a node.
- **Unstored values.** An unstored value or output that is zero is canonical +0.
- **Validation.** The engine rejects operators and attributes it does not define: ceil-mode, dilated or
  over-padded max-pool, non-global average pool, convolution padding modes, bias or scale arrays that do not
  match the output channels, residual operands of different shapes.
- **Top-k.** Descending output value, ties by ascending class index (lowest class index first), as in 2.0. B and
  B2 used `torch.topk` on CUDA FP32 logits, which does not define the order among equal logits. Every paired
  difference against a simulator is therefore reported under both conventions for the simulator side (its
  retained order, and this rule applied to its logits), with the tie-aware expected accuracy next to both.
- **What differs from B2 by construction.** B2 reduces with framework FP32 convolutions on FP32 reconstructed
  operands and stores in FP32. The engine computes exact code-domain dots, factors the scales in binary64 and
  stores in binary64. Fused values (convolution into ReLU, residual add into ReLU) are binary64 here and FP32
  in B2. Bit identity with B2 is not expected and is not a gate.
- **What cannot run exactly.** An unquantised network input (`quantize_input=false`) has no code grid, so the
  first convolution cannot run in the code domain; the engine fails closed and that diagnostic stays
  simulator-only. Hard-swish, hard-sigmoid, ReLU6 and tensor multiply (`fused_all`, the MobileNets) are not
  defined in 2.1. Shared-exponent formats and formats without a dyadic grid are not admitted. An unquantised
  logit output (`quantize_logits=false`) does run: the output is then the unstored binary64 value.

### Width convention of the certificates

Certified width of a MAC node = 1 sign bit + bit_length( max over output channels c of max|a| · Σ_k |w[c,k]| ).

- a ranges over **every level of the input tensor's codebook** (the admitted code domain, not observed
  activations), as an integer in units of 2^-(input shift). The structural width uses a ≥ 0 behind a ReLU.
- w are the **exported weight integers of channel c**, in units of 2^-(weight shift), over all
  K = C_in/groups · kh · kw taps. The sum is per output channel; the node width is the maximum over channels;
  the network width is the maximum over nodes.
- The register unit is the product grid 2^-(input shift + weight shift): its least significant bit is the
  product of the smallest input step and the smallest weight step, so no product bit is dropped.
- Two's complement, sign bit included. The bound holds for every prefix in any order. The bias is not added in
  this register.

**Comparison with the published minifloat formula.** The formula relayed in the related-work audit ([Agg24],
citing [Ugu17]) is, for ExMy operands and n taps, (2^Ea + Ma) + (2^Ew + Mw) + ceil(log2 n) - 1. With n = 4,608
it gives 26, 32 and 34 bits for fp6_e2m3, fp6_e3m2 and fp7_e3m3; the certificates give 23, 29 and 31. Both count
the sign bit and both use the product grid. The three bits are:

| Step, fp6_e2m3, ResNet18 `layer4_1_conv1` (K = 4,608, the widest node) | Magnitude bound in product units | Signed bits |
|---|---:|---:|
| Published formula: product 2^12, taps 2^13, each rounded up to a power of two | 2^25 | 26 |
| Same worst case without the separate rounding: 4,608 · 56 · 56 | 14,450,688 < 2^24 | 25 |
| Certificate: actual weights of the worst channel, Σ|w| = 62,997 units instead of 4,608 · 56 = 258,048 | 3,527,832 < 2^22 | 23 |

One bit is the formula rounding the largest product (56 · 56 = 3,136, counted as 2^12) and the tap count
(4,608, counted as 2^13) up separately. Two bits are data: the certificate sums the trained weights of each
output channel, and the worst channel's Σ|w| is 24.4% of the all-maximum case. The same decomposition holds for
fp6_e3m2 (32 → 31 → 29; Σ|w| = 431,000 of 1,769,472) and fp7_e3m3 (34 → 33 → 31; 1,008,123 of 4,128,768).
The published number is a worst case over all weight tensors of that shape; the certificate is a proof for
this network's weights and for every input. The first two bits of the gap would differ for another network;
the two conventions agree on sign, grid and what is summed.

## Planned additions, not yet frozen

Block-scaled (shared-exponent) operands: design note at the end of section 2.2, not frozen. The MobileNet
operators are frozen in contract 2.2 below.

## Contract 2.2: MobileNet operators (frozen 2026-10-02 00:00, before any MobileNet panel)

Sealed with its protocol as `artifacts/scaled_bridge_v2/protocols/mn-9f55f6139786bdfe.json` (contract
`scaled-code-domain-bridge-2.2`, protocol `scaled-bridge-v2-mn-development-1`); the text there is the authority,
this section restates it. Every 2.1 rule holds unchanged. RNE64 is round-to-nearest-even to binary64; r of a state
is RNE64(code_level · scale) for stored codes and the binary64 value for an unstored value.

### Operators

| Operator | Definition | Rounding steps |
|---|---|---|
| ReLU6 | stored: max(level, +0), then RNE64(level · scale), then min(·, 6); unstored: min(max(v, +0), 6) | one (the scale product); the clamps are exact |
| hard-sigmoid | t = min(max(RNE64(r + 3), +0), 6); out = RNE64(t / 6) | two |
| hard-swish | t as above; out = RNE64(RNE64(r · t) / 6) (the order of `torch.nn.Hardswish`, x · relu6(x + 3) / 6) | three |
| tensor multiply (squeeze-excite) | both operands stored: p = u_a · u_b exactly (|p| < 2^53 certified), then RNE64(RNE64(p · 2^-(shift_a + shift_b) · s_a) · s_b), a = first input in node order; otherwise RNE64(r_a · r_b). Shapes equal, or (N, C, 1, 1) broadcast over (N, C, H, W) | two (stored) or one |

A zero result of any of these is canonical +0. The result is a binary64 value that the next boundary stores
with the 2.0 store rule (or leaves unstored when the boundary does not quantise).

### Certificates added in 2.2

- **Product certificate** (per multiply node): the largest |u_a · u_b| over the two codebook ranges, its signed
  width (1 + bit length) and the product shift; the engine refuses a bound of 2^53 or more.
- **Closure width** (per MAC node, `signed_bits_closure`): the same formula as the 2.1 widths, but the input
  range is obtained by interval propagation through the graph under the rules above (stored tensors carry unit
  bounds through their store rule, unstored values binary64 bounds; ReLU, ReLU6, hard-sigmoid and hard-swish map
  bounds through their definitions, hard-swish has its minimum -0.375 at -1.5; add adds bounds; multiply takes
  the corner products; pooling, identity and flatten keep bounds; convolutions, linear layers and the input are
  bounded by their codebook only). The interval is widened to contain zero. The engine checks every MAC input
  against its closure range on every call and fails closed. The 2.1 widths, and so `sat.struct-<d>`, are
  unchanged, so that evidence of earlier digests stays comparable. For `fc` behind the average pool of ResNet18
  the closure width uses a ≥ 0 (review 3, item 6); the structural width keeps the full signed range there.

### Graph validation added in 2.2

The B2 adapter requires the exported node rows to equal the operator rows of the independently loaded folded
FX graph, and refuses an in-place activation whose input tensor has another consumer. Average pooling must be
global (as in 2.1).

### What of the B2 MobileNet graphs runs exactly, and what does not

Boundary plan of the B2 default recipe (`fused_relu`, unsigned codes for non-negative integer tensors), read
from lane L1's exports: MobileNetV2 has 35 convolutions fused into ReLU6 (the ReLU6 output stores, unsigned for
integers), 17 convolutions stored signed (the linear bottleneck projections), 10 residual adds, a stored average
pool and the classifier. MobileNetV3-Large has 19 convolutions fused into ReLU, 43 convolutions stored, 21
hard-swish nodes (input = a stored convolution or linear output, output stored signed), 8 hard-sigmoid nodes
(input = the stored squeeze-excite expansion, output stored unsigned for integers), 8 multiplies (output stored;
signed in the 5 hard-swish blocks, unsigned in the 3 ReLU blocks) and 9 stored average pools. In the default
recipe no operator reads an unstored hard-swish or hard-sigmoid input, and both operands of every multiply are
stored, so every multiply takes the exact code-product path.

Not exact in the code domain, and why:
- `quantize_input=false`: the first convolution has no code grid (unchanged from 2.1).
- (Runs, but not measured here.) `fused_all` boundaries fuse a convolution or linear output into the
  hard-swish or hard-sigmoid that consumes it; the operator then reads the unstored binary64 value and its own
  output stores, so every MAC input is still a stored code tensor. 2.2 defines this case and the unit tests run
  it (tiny graphs, stored and fused variants), but protocol MN admits only the `default` recipe. Any graph in
  which a MAC would read an unstored value is refused (fail closed), never approximated.
- Shared-exponent (block-scaled) formats, the log formats and NF4 (no dyadic grid): not admitted.

### Design note, not frozen: shared-exponent (block-scaled) operands

A block-scaled dot is the sum over K-blocks of 2^(e_a + e_w) · (an integer block sum of element codes). Each
block sum is exact in int64 (bounded by block length · max|a| · max|w|). An exact accumulator then aligns the
block sums to the smallest block exponent of the dot: the aligned total needs (block-sum width + exponent
spread) bits, which a per-element runtime check can bound and which the two-limb path covers up to 127 bits;
beyond that the dot fails closed. The certificate would be per node: max block-sum width plus the largest
exponent spread the export's block scales allow (shared exponents are data, so the spread is known after
calibration for weights and bounded by the exponent field for activations). B's block semantics (activations
blocked along channels at stores, re-blocked along the reduction axis inside each convolution;
`tools/experiment_b_ext/shared.py`) must be matched exactly before a protocol is written. Nothing of this is
implemented.
