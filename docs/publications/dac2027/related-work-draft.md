# Related work (draft 2, about 0.7 page plus table T1)

<!-- Lane P2, 2026-10-02; revision 2 the same evening after independent review 1. Sources:
docs/analysis/related-work-audit-2026-10-01.md (the audit; sections 1, 2, 4, 7) and lane P1's
docs/analysis/closest-prior-work-accumulators-2026-10-02.md (first line "STATUS: FINAL", file time 17:59; no
independent review yet), which read the priority-1 papers in full text where an open copy existed. Every
characterisation below is limited to what those two documents record; [Des23] and [Ugu17] are characterised from
abstracts and secondary descriptions only (P1 L84-95, L154-155). Keys refer to refs.bib. No sentence presents
"complete cost changes the ranking" as new (audit L996-998). The draft is meant to be pasted into the manuscript:
it contains nothing about any other project of the authors. -->

**Low-bit formats and their arithmetic cost.** FP8 and integer inference have been compared on accuracy and
hardware cost [vB23, Kuz22]; the FP8 advantage observed under PTQ shrinks under QAT [Kuz22], and rankings of
quantisation methods change between academic and deployable settings [LiM21]. Below 8 bits, the closest prior work
[Agg24] evaluates minifloats against integers on ResNet18 and MobileNetV2 with several PTQ techniques, evaluates
accuracy with FP32 accumulators, sizes the accumulator with a closed-form exact width, and finds that integers keep
Pareto optimality on FPGAs once MAC cost is included; parallel accurate minifloat MACs on FPGAs cost more than
integer ones [Dam24]. At 8 bits, ASIC datapaths exist for exact FP8 and posit8 dot-product operators (16 nm
synthesis) [Des23], for expanding FP8-to-FP16 dot products in a training cluster (12 nm) [Ber22], for a fused FP8
four-way dot product with scaling and FP32 accumulation (5 nm synthesis) [Lut24], and for a narrow FP8
accumulation scheme evaluated post-layout in a 7 nm predictive kit [Nat25]. That complete arithmetic cost favours
integers over a multiplier-only view is therefore established; what is missing is evidence below 8 bits for
non-integer formats with quality attached.
<!-- audit C2 L964-1002; [Agg24] P1 L146-149 ("with accumulators in FP32"); [Dam24] P1 L150-153; [Des23] P1 L84-95
(abstract: TSMC 16FFC synthesis); [Ber22] P1 L97-110 (GF 12 nm synthesis and place-and-route, training, synthetic
dot products only); [Lut24] P1 L69-82 (5 nm synthesis at 3.6 GHz, no network accuracy); [Nat25] P1 L45-67 (ASAP7
post-layout, 0.7 V, 500 MHz). -->

**Accumulator width.** For integers, accumulator-aware quantisation guarantees overflow avoidance by bounding the
weights' L1 norm per output channel, in QAT with a post-training minimisation of the width [Col23, Col24a] and in
PTQ for language models [Col24b]; static interval analysis derives per-layer widths of scaled-integer FPGA
dataflow accelerators from the actual weights [Umu25]. Exact (Kulisch) accumulation of floating-point products has
a known width [Ugu17], which [Agg24, Dam24] use for minifloats. Narrower floating-point accumulators have been
studied for training [Wan18, Sak19] and for inference with fine-tuning: a 12-bit floating-point accumulator
matches FP32 accumulation on an FP8 ResNet18 after fine-tuning, while applied without fine-tuning to the
full-precision ResNet18 it loses about ten points [Blu24].
Wrap-around accumulation with a trained cyclic activation [Ni20], and narrow registers that spill to a wide one in
a statistically chosen order [Nat25], avoid the worst case altogether. Whether FP32 accumulation is adequate for
low-bit operands is assumed in [Agg24] and questioned for MX MACs [Cuy26]. All of these are 8-bit, training, or
both. We differ in two respects. First, our certificate is A2Q's per-channel weight-norm bound transcribed to the
integer code grids of minifloat and posit formats; it lies 2 bits below the exact data-type bound, and the rest of
the gap to the published formula is form and format convention. Second, for a saturating fixed-point register we
measure, under PTQ and exact execution, the width at which accuracy fails; binary16 and 21-bit float accumulators
show no failure in our range. We compare against accumulator-aware PTQ at equal width (§5).
<!-- audit C3 L1004-1047, §4 conditions L1201-1209. [Col23] Sec. 3.2 weight bound and "post-training minimization":
P1 L156-161. [Blu24]: P1 L127-142 (Table 3 fine-tuned 69.70 vs 69.90; Table 8 zero-shot M7E4 60.14 vs 69.75; floor
rounding, chunk 16). [Sak19] float accumulator, training: P1 L112-125. [Wan18] P1 L179-180. [Ni20] P1 L175-176.
[Nat25] P1 L53-59. [Cuy26] P1 L168-169. "Every priority-1 paper is 8-bit or training or both": P1 L23-27. Certificate
wording: P1 L268-274 (sentence 2) and Part C L238-262. The float accumulators have no failure point:
claims-ledger A5, gaps G3. "§5" = D4 in claims-ledger.md (Q8, reviewed with fixes applied, not re-reviewed). -->

**Simulation fidelity.** Quantisation studies, including [Agg24], usually rely on fake quantisation with FP32
accumulation; integer-only deployment paths define the arithmetic exactly [Jac18], and the gap between academic
simulation and hardware-deployable integer quantisation has been measured [LiM21]. Real FP8 and MX dot-product
hardware accumulates with specific semantics, such as one rounding per fused four-term sum into FP32 [Lut24] or a
partial result cut to a 16-bit mantissa [Cuy26], and the OCP MX specification is reported not to prescribe
accumulation precision or rounding [Isl26]. We give the non-integer, sub-8-bit version of this question with an
attribution: exact code-domain execution against the FP32 simulator, separated into tie order, accumulation,
scale factoring and store rounding.
<!-- audit C4 L1049-1082, esp. L1068-1071 ("C4 must be worded as the non-integer, sub-8-bit, attributed version of a
known concern"). P1 L36 and L299-301: no paper read measures network-level simulator fidelity. [Lut24] eq. 4 one
rounding: P1 L73-74. [Cuy26] 16-bit mantissa: P1 L168-169. OCP MX spec itself unverified (audit 7.5). -->

**Format selection.** Adaptive data types with accelerators [Guo22, Hu25] and per-layer INT/FP search [Dot24] choose
formats per tensor or layer; we keep one format per network (no mixed precision) and treat the accumulator, not the
format assignment, as the design variable.
<!-- audit C5 L1084-1104; owner decision 3 (no mixed precision). -->

## T1. Positioning (draft)

| Work | Formats (bits) | PTQ / QAT | Accumulation evaluated as | Accumulator width from | Hardware evidence | Networks |
|---|---|---|---|---|---|---|
| [Agg24] | minifloat, INT (3-8) | PTQ (best of four techniques) | FP32 fake quantisation | closed-form exact width | FPGA | ResNet18, MobileNetV2, ViT |
| [Dam24] | minifloat | — | exact (accurate MACs) | closed-form exact width, N parallel lanes | FPGA | — |
| [Des23] | FP8 E5M2/E4M3, posit8 | — | exact dot product | exact, rounded up to a power of two (secondary source) | ASIC synthesis (16 nm) | — |
| [Ber22] | FP8 → FP16 (expanding) | training | FP16, one rounding per two products | destination format | ASIC synthesis and P&R (12 nm) | none (synthetic dot products) |
| [Lut24] | FP8 E4M3/E5M2 | — | exact 4-term sum, one rounding into FP32 | fixed (FP32) | ASIC synthesis (5 nm) | none |
| [Nat25] | INT 5-8, FP8 E4M3 | no retraining | rounded FP8 products, 5-bit registers spilling to 32 bits | fixed narrow + overflow statistics | FPGA; ASIC post-layout (7 nm predictive) | ResNet-18, MobileNetV2, ViT-S |
| [Blu24] | FP32 or FP8 operands | fine-tuning (one zero-shot table) | 12-bit float, floor rounding, chunk 16 | chosen, then fine-tuned | analytical gate count | ResNet18/34/50 |
| [Sak19] | FP8 operands | training | float, predicted mantissa | variance analysis | analytical area model | ResNet 18, AlexNet, CIFAR ResNet |
| [Col23], [Col24a] | INT | QAT (+ post-training width minimisation) | integer | per-channel L1-norm bound | FPGA (FINN) | CIFAR-10 CNNs, ImageNet ResNet50 ([Col24a]) |
| [Col24b] | INT | PTQ | integer, tiled multi-stage | L1-norm guarantee | — | LLMs |
| [Umu25] | scaled INT | — | integer | static interval analysis | FPGA | dataflow networks |
| [LiM21] | INT | PTQ, QAT | simulated vs deployable | — | — | many |
| **This work** | INT, minifloat, posit (4-8; accumulator study 6-8) | PTQ | **exact code domain**; saturating fixed-point, binary16, 21-bit float, FP32 | **per-layer certificate + measured failure width** | ASIC synthesis, 55 nm (integer MACs today) | ResNet18, MobileNetV2/V3 (sweep INTERIM), YOLOv8n (simulator only) |

<!-- Rows from audit §1 entries and P1 Part A: [Dam24] P1 L150-153; [Des23] P1 L89-94 ("rounded up to a power of
two" is from [Lut24] Sec. V, secondary); [Ber22] P1 L100-110; [Lut24] P1 L72-82; [Nat25] P1 L50-64; [Blu24] P1
L130-140; [Sak19] P1 L115-124; [Col23] P1 L156-161 (CIFAR-10 MobileNetV1/ResNet18, BSD300 ESPCN/U-Net; FINN);
[Col24a] P1 L162-163 (ImageNet ResNet50, QAT); [Col24b] P1 L164-165 (tiles, LLMs only). The "This work" row must be
cut back to what exists at submission (non-integer MAC RTL: none today; MobileNet sweep not yet final). -->
