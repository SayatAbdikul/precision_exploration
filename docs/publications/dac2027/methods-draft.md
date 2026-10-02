# Methods (draft 1, about 1.5 pages)

<!-- Lane P2, 2026-10-02; revision 2 the same evening after independent review 1. Draft text for §3 of the manuscript. Every statement is checked against the cited
contract or analysis document (paths relative to the repository root). Bracketed items [..] are open. Numbers in
this section are definitions or counts, not results. Anonymous: no names, no affiliations. -->

## 3.1 Number formats

We study scalar formats of 4 to 8 bits from five families: two's-complement integers (INT4, INT5, INT6, INT8),
minifloats ExMy (FP4 E2M1, FP5 E2M2, FP6 E2M3, FP6 E3M2, FP7 E3M3, FP8 E4M3FN, FP8 E5M2), posits (posit4 es0,
posit6 es1, posit8 es1), logarithmic formats (log4, log6, log8) and the NF4 codebook, plus shared-exponent block
formats (MXFP4, MXFP6, MXFP8 and a 6-bit block fixed-point format, block size 32) and, below 4 bits, ternary and
binary. Each format is defined by a public manifest that enumerates its code levels and its rounding and
overflow rules; a format is therefore a finite, ascending set of levels with a tie preference, and quantisation
maps a scaled real value to the nearest level.
<!-- Manifests: public/formats/manifests/accepted/*.json (25 format manifests plus index.json: bfp6, binary_pm1,
fp4_e2m1 ... ternary; integers int4, int5, int6, int8, no int7).
Family list and block size 32: docs/analysis/b2-matrix-2026-10-01.md title (25 formats) and survey §3 N4C.4
("block size 32 only"). Codebook definition: docs/analysis/scaled-bridge-v2-contract-2026-10-01.md L13-18 ("A
codebook is a list of ascending integer units on a 2^-shift grid with its codes and midpoint tie preferences").
q1_6 is an output alias of signed INT8 (b2-matrix L46-47) and is not reported as a separate format. -->

Exact code-domain execution (§3.3) requires every level to be a dyadic rational. This holds for the integer,
minifloat and posit formats, which are therefore the subject of the accumulator study; the logarithmic formats
and NF4 have non-dyadic levels and the shared-exponent formats need a block contract, so they are evaluated in
simulation only.
<!-- contract L59-63 ("Not admitted"); contract 2.2 L281-282; engine L23-27 (all scalar formats with a dyadic grid
admitted on ResNet18). -->

## 3.2 Post-training quantisation recipe and data

Networks are the torchvision ImageNet checkpoints of ResNet18, MobileNetV2 and MobileNetV3-Large, and YOLOv8n for
detection [checkpoint identities: cite the frozen hashes]. All quantisation is post-training and uniform: every
quantised tensor uses the same format. Weights use one FP32 scale per output channel and activations one FP32 scale
per tensor; both scales are found by a mean-squared-error search over a grid derived from the format's own
codebook (ratios to the max-abs scale from 2^-6 up to the codebook's dynamic range, eight points per octave, then a
local refinement). Convolutions or additions feeding a ReLU/ReLU6 are requantised once, after the activation;
max-pooling forwards codes. Integer activations at non-negative boundaries use unsigned codes (non-integer
formats have no unsigned variant). Biases are corrected empirically on 256 calibration images. We call this
recipe "default"; a "minimal" variant (max-abs per-channel weight scales, no bias correction, signed codes) is
reported where the format order depends on it.
<!-- docs/analysis/b2-baseline-repair-2026-10-01.md §3 L117-135 (table of frozen recipes, scale grid from the
codebook) and L139-147 (hardware semantics of each switch). Recipe selection rule R_bits(6) (default for >= 6
bits, minimal below): b2-matrix L58-65. Detector recipe differs (no bias correction, SiLU fused, head
exemptions): docs/analysis/b2-detector-2026-10-01.md L97-101; the default/conformant choice is an OPEN owner
decision (L107-108). Mixed precision is out of scope: docs/decisions/owner-decisions-2026-10-01.md decision 3.
Checkpoint hashes: [TODO, from public/workloads/ manifests]. -->

Activation ranges are observed on the frozen 2,000-image calibration list drawn from the ImageNet training set
(and a 2,000-image COCO training list for the detector); bias correction uses its first 256 images. Calibration
and evaluation lists share no image by content hash.
<!-- b2-baseline-repair L453-454 ("256 stratified samples per image per node over 2000 images"; "bias correction
uses the first 256 calibration images"); confirmation-protocol-draft §4 and §7 (lists disjoint by sha256, 0
shared); evaluation-history-audit L91-103. -->

As a reconstruction baseline we reproduce AdaRound with its published settings and check it against the published
4-bit-weight ResNet18 result before using it.
<!-- docs/analysis/b2-reconstruction-baseline-2026-10-01.md L73-75 (faithfulness: drops 0.6, 1.3, 0.8 against
published 1.08; criterion written before measuring). -->

## 3.3 Exact code-domain execution

Every quantised tensor is stored as integer codes on a grid 2^-s together with its FP32 scale (promoted exactly to
binary64). A convolution or linear layer multiplies weight and activation codes exactly and sums the integer
products on the exact product grid 2^-(s_a+s_w); the "wide" reference sums them exactly in a signed integer whose
width is certified per output channel (64 bits, or two 64-bit limbs above 2^63). The sum is converted to binary64
once, with round-to-nearest-even, which is exact below 2^53 units; scales and bias are then applied in binary64
(RNE64(RNE64(dot·s_in)·s_w[c]) + b), and the result is stored by dividing by the next scale and rounding to the
nearest level of the next codebook. Residual adds, pooling and the MobileNet operators (ReLU6, hard-sigmoid,
hard-swish, the squeeze-excite product) have fixed binary64 definitions. Top-k is defined: descending value, ties
by ascending class index.
<!-- contract L10-46 (what v2 changes: recipe as data, exact dot beyond 2^53, top-5 order), L48-57 (post-ops and
store rule unchanged from v1), L223-241 (2.2 operators). In the two MobileNets every dot is below 2^53
product units (engine L522-523: widest 45 bits, no two-limb path). On ResNet18 the two-limb path is used by FP8 E5M2
(certificate 75 bits, "two 64-bit limbs, one RNE conversion", engine L68); posit8 (59 bits under the max-abs
recipe, engine L67) uses int64 with one RNE conversion. -->

Before any measurement, every admitted case passes a conformance gate: exhaustive code-pair products and a
rational store oracle per codebook, an independently computed whole-graph reference on the first image, rational
checks of three actual dot products per layer in each arm, and bit-identical CPU and CUDA records on eight images.
Agreement between the two backends alone is never accepted as a reference.
<!-- contract L65-77; engine L23-39 (23 of 23 ResNet18 cases; witness counts 2,474 to 268,666 per codebook),
L471-490 (10 of 10 MobileNet cases). A later fast implementation was accepted only after independent reviews found
no difference in about 97,000 image records (accumulator-sweep-mobilenet draft §2; S1-engine-review.md L468:
approve_use_with_conditions). The MobileNet sweep uses it for 8 of 10 cases from 17:11 on
(docs/analysis/accumulator-sweep-mobilenet-2026-10-02.md section 2, L48-59), so the paper must mention it. -->

## 3.4 Accumulator policies

Only the dot-product accumulator changes between arms; quantisation, scales and stores are shared. Products are
added in a fixed order (input channel, kernel row, kernel column; padded taps add a zero product).

- **wide**: the exact sum (reference).
- **control**: sequential binary32 fused multiply-add over the integer products.
- **sat.wW**: a W-bit two's-complement register (2 ≤ W ≤ 63) whose least significant bit is the product grid,
  clamped to [−2^(W−1), 2^(W−1)−1] after every add; it never wraps. The bias is added outside the register. The
  same W is used in every layer.
- **fp16.x e**, **f21.x e**: binary16 (11 significand bits) or a 21-bit float (1 sign, 8 exponent, 12 fraction
  bits) with one round-to-nearest-even per multiply-add and gradual underflow; the binary point is moved by 2^e,
  chosen per case (network and format) from the certificate (the largest prefix magnitude) so that the format's
  range covers it.
  Overflow to infinity is a recorded failure of the image.
<!-- contract L111-157 (policy table, saturating register L125-133, float accumulators L134-147, scale exponent
rule L142-147, non-finite results L148-155). The exponent rule: sweep L102-106 (protocol). -->

## 3.5 Certified lossless accumulator width

For a layer with weight codes w (in units of 2^-s_w) and an input codebook whose largest magnitude is max|a|
(units 2^-s_a), the certified width is

  W_cert = 1 + bitlength( max_c max|a| · Σ_k |w_{c,k}| ),

the maximum over output channels c and over all K = C_in/groups · k_h · k_w taps, counted with the sign bit in
product-grid units. It bounds every prefix of every dot for every input in the admitted code domain, in any order,
so a register of W_cert bits never saturates; a structural variant uses a ≥ 0 behind a ReLU. The network width is
the maximum over layers. For integer operands this is the per-channel weight-norm bound of A2Q [Col23], applied
after training; we apply it unchanged to the integer code grids of minifloat and posit formats. The published
closed-form width 2^Ea + Ma + 2^Eb + Mb + ⌈log2 n⌉ − 1 [Agg24, Dam24, citing Ugu17] uses the same unit (the
product of the two smallest subnormals) and counts one sign bit. We report three widths per format: the formula as
printed, the exact data-type bound bit_length(K · max|a| · max|w|) + 1 in our convention, and the certificate. On
ResNet18's widest layer the formula exceeds the exact data-type bound by one bit of form (it rounds the product
width and ⌈log2 n⌉ up separately) plus 0 to 2 bits of format convention (codes reserved for Inf/NaN, unsigned
activations); the certificate lies 2 bits below the exact data-type bound in every case, because the worst
channel's Σ|w| is 21 to 26 % of K · max|w|.
<!-- contract L185-198 (definition). docs/analysis/closest-prior-work-accumulators-2026-10-02.md (P1, STATUS: FINAL,
unreviewed): convention from [Dam24] eq. (1)-(2) L150-153 and L208-217; A2Q bound L156-161; table L238-249 (FP6
E2M3 26 / 25 / 25 / 23; FP7 34 / 33 / 33 / 31; FP8 E4M3FN 50 / 49 / 49 / 47; FP8 E5M2 80 / 79 / 77 / 75; INT8 30 /
28 / 29 / 27; INT6 26 / 24 / 25 / 23); worked example L225-234 (B2 Σ|w| = 63,683 of 258,048 units, 24.7 %; the
contract's 62,997 / 24.4 % is the B1 max-abs export, same widths). Do not write "3 to 5 bits tighter than the
published formula" (P1 L288). [Ugu17] itself not opened (P1 L154-155). -->

From each sweep we derive, per format: W_noevent (narrowest width with no clamp on any image, which reproduces the
exact arm bit for bit), W_acc(δ) (narrowest width whose top-1 is within δ = 0.5 or 1 pp of exact at that and every
wider measured width) and W_half (widest width at or below half of exact top-1). Widths were located on images
0-127 by a ladder fixed in the protocol (the certified width minus 2, then every 3 bits down until a rung falls to
half of exact or below), refined by a rule in the sweep code that adds the two widths below the narrowest
event-free width and the two above the widest collapsed width; this refinement was written into the protocol
record only after the measurements (addendum 3), and replaying it reproduces exactly the widths that were run.
The located widths were then measured on all 1,000 screen images, so the 1k numbers are not independent of the
location rule.
<!-- sweep L7-9 (addendum 3 written after review 1), L85-95 (ladder from protocol `location_phase`; refinement
`plan.refine` "that addendum 3 writes down"; replay artifacts/accumulator_sweep_v1/addendum3_check.json), L117-120
(derived widths), L352 (1k not independent of the location rule). Protocol:
public/experiments/configs/breadth-study/accumulator-sweep-protocol-v1.json + addenda 1-3; addendum 3
"missing_facts"; artifacts/agent_orchestration/handoffs/L8-accsweep-review.md L48, L77. -->

## 3.6 Hardware cost

Integer MACs (signed W × A multiplier, saturating (ACC+1)-bit adder, bias added at the end of a dot) are
synthesised from a generic RTL description with Yosys and ABC on the open ICsprout 55 nm standard-cell library
(typical corner, 1.2 V, 25 °C) and timed with OpenSTA at clock targets of 4 to 12 ns; we report core area at the
8 ns target that every configuration meets, and use only same-flow ratios. Every mapped netlist passes gate-level
simulation and formal equivalence against the RTL. This register differs from the evaluated `sat.wW` in two
respects that a join must state: the RTL adds the bias inside the register and clamps again, and its signed
activation port cannot carry the unsigned INT activation codes of the default recipe without one more bit; only
the signed-activation INT8 case matches it operand for operand. The area per accumulator bit also depends on
whether the synthesis merges the multiplier into the accumulate adder (Yosys `alumacc`); we report both.
[Non-integer MAC RTL: not yet available. Energy: not yet available. Aligned RTL variant: not yet available.]
<!-- docs/analysis/ics55-integer-accwidth-2026-10-01.md L5-13, L17-21, L221-228; review by lane P3,
docs/analysis/ics55-integer-accwidth-review-2026-10-02.md (Approve with corrections, L25-40; F1 blocking for the
join L31; part 4 semantics L10-14; flow sensitivity L15, F3 L33; join conditions L181-189);
docs/analysis/ics55-integer-isoclock-2026-10-01.md L7-17, L117-127. The RTL adds the bias inside the register and
uses signed activations; the engine adds the bias outside and uses unsigned INT activations in `default`
(sweep L265-279): the paper must state which accuracy rows match the RTL (INT8 `default_signed`). -->

## 3.7 Statistics and data roles

Top-1 is scored per image under two tie rules: lowest class index among the tied maxima, and expected credit (1/t
when the label is among t tied maxima). Quantised logits are frequently tied, so every difference is reported under
both rules. Each arm is compared with its reference on the same images: 95 % paired percentile-bootstrap intervals
of the difference (10,000 resamples), exact McNemar tests on the lowest-index outcomes, and no correction for
multiple comparisons (intervals are pointwise). At 1,000 images the half-width for near-equal accumulator arms is
about 0.8 pp and for near-equal format pairs about 1.3 pp. Detection uses COCO mAP50-95 with a fixed tie order and
an image bootstrap of the dataset-level AP.
<!-- sweep L109-116 (scoring and statistics; seed 20260927); gap L36-37 and L270-271 (ties); Q7 L30-52 (half-widths
0.78 and 1.31 pp; coverage 0.94-0.96 at n = 1,000, L61-62); detector-breadth L7-9 (2,000 draws, seed 310911). -->

All quality numbers in this version are measured on 1,000-image development screens (one image per ImageNet class;
1,000 COCO validation images) that were also used to select recipes and cases. A further 9,000 ImageNet validation
images and 4,000 COCO validation images are held out, untouched by any quantised configuration, for a single
confirmation of the frozen hypotheses [results to be added]; the FP32 detector baseline has already been scored on
all 5,000 COCO images and is excluded from confirmatory use.
<!-- owner-decisions-2026-10-01 decision 2; evaluation-history-audit L91-103; confirmation-protocol-draft §7 (1k
list: 1 per class). The confirmation run, its margins and contrasts are not yet frozen (protocol draft is
"PROPOSED - awaiting owner sign-off"). -->
