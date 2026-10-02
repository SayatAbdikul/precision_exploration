STATUS: DRAFT - NOT FINAL. Lane P1 was stopped on the owner's instruction on 2026-10-02 at 18:16, before its fix round. Review 1 (artifacts/agent_orchestration/handoffs/P1-closest-papers-review.md) found 4 blocking errors that are NOT corrected below: (1) Part B row 11 has no source ([Joh18] was misread); (2) the positioning is too wide for integer formats ([Nat25] Fig. 9 sweeps 5-8-bit operands against 8-20-bit accumulators with saturation, without retraining); (3) fp16 and f21 accumulators are within 1 point, not 'lossless'; (4) the stated search times are wrong. Read that review before using anything in this document. (Status line changed by the orchestrator; the text below is as the lane left it at 17:59.)

# Closest prior work on accumulators: full-text reading and positioning (lane P1, 2026-10-02)

Documents only. No model was run, no GPU was used, no held-out image or prediction file was read. Part C uses stored
certificates and manifests (CPU, `.venv/bin/python`). Reading notes with line references:
`artifacts/literature_v1/notes/papers.md`; Part C script and output: `artifacts/literature_v1/notes/partc_widths.py`,
`partc_widths_out.txt`; downloads and extracted text: `artifacts/literature_v1/downloads/` (20 MB). Citation keys are
those of `docs/analysis/related-work-audit-2026-10-01.md` (the audit). VERIFIED / UNVERIFIED are used as in the audit;
"full text" means the PDF was downloaded and searched locally with `pdftotext`, not summarised by a model.

## Bottom line (what changes for the paper)

1. **The published "3 to 5 bits" gap is mostly not substance.** For ResNet18 the closed form of [Agg24]/[Dam24] exceeds
   the certificate by 3 bits (5 for FP8 E5M2). Of this, 1 bit is the formula's form (it rounds the product width and
   ⌈log2 n⌉ up separately), 0 to 2 bits are format convention ([Agg24] formats have no NaN/Inf codes; the project's E5M2
   reserves the top binade, its integer activations are unsigned), and **2 bits in every case are substance**: the
   worst output channel's Σ|w| is 21 to 26 % of K·max|w|. Print "26 vs 23" only with this split (Part C).
2. **The weight-based per-channel width is prior art for integers.** A2Q [Col23] Sec. 3.2 states the bound
   P ≥ log2(‖w‖₁) + N − 1_signed(x) + 1 per channel and applies it after training ("post-training minimization");
   the project's integer certificate is that bound with the exact max|x|. The certificate is new only as the same
   bound on non-integer code grids (minifloat, posit) used as the lossless reference of a PTQ sweep. Narrow C3.
3. **Every priority-1 paper is 8-bit or training or both.** [Nat25] (FP8 E4M3, PTQ-like, post-layout 7 nm power),
   [Lut24] (FP8 DOT4 into FP32, 5 nm synthesis, no network), [Des23] (FP8/Posit8 exact operators, 16 nm, abstract only),
   [Ber22] (FP8 into FP16, 12 nm, training, no network), [Sak19] (float accumulator mantissa for training), [Blu24]
   (12-bit float accumulator, fine-tuned). None measures a failure width for sub-8-bit formats under PTQ. The headline
   survives, scoped to sub-8-bit formats, PTQ, CNNs and a measured failure width.
4. **[Blu24]'s zero-shot table is the strongest neighbour to cite:** a 12-bit float accumulator (M7E4, floor rounding,
   chunk 16) drops ResNet18 from 69.75 to 60.14 % without fine-tuning; with fine-tuning it reaches 70.06 %. The project's
   PTQ evidence on float accumulators stops at FP16 and a 21-bit float, which are lossless here.
5. **Design points a reviewer can ask for and the engine lacks** (Part B): wrap-around registers, chunked or two-stage
   accumulation, products rounded before accumulation, LSB truncation (two-sided fixed-point narrowing), float
   accumulators below 16 bits with floor rounding, fused k-term rounding. Wrap-around is the cheapest and most telling.
6. **The hardware half is the weak half against this set.** Four of the six priority-1 papers have ASIC numbers for
   FP8 datapaths (7, 5, 16, 12 nm); the project's ASIC join is integer-only and new RTL is on hold (owner, 2026-10-01).
7. C4 (simulator versus exact) is unchanged: no paper read here measures network-level simulator fidelity.

## Part A. Full-text reading

Format per entry: bibliography (status), source opened, formats, workloads, regime, accumulator, how the width is
chosen, hardware evidence and measured unit, headline numbers with location, what it does not do that this project does.

### Priority 1

**[Nat25] V. Natesh, H. T. Kung, D. Kong.** arXiv:2504.09072v1 (12 Apr 2025), "MGS: Markov Greedy Sums for Accurate
Low-Bitwidth Floating-Point Accumulation" (VERIFIED, arXiv API; no comment or DOI field). **Published version found in
this reading:** IPDPS 2026, pp. 557-569, DOI 10.1109/ipdps65963.2026.00054, title "MGS: Markov Greedy Sums for Low-Power
DNN Accumulation", same three authors (VERIFIED, Crossref; the published text was not opened, so the arXiv v1 is what
is characterised). Full text: arXiv PDF v1.
- Formats: INT 5-8 bits; FP8 E4M3 (Sec. 6.2.2). Workloads: MobileNetV2, ResNet-18, ViT(-Small) on CIFAR-10 and
  ImageNet-1K; image counts not stated. Regime: no retraining ("without retraining", Sec. 1); per-tensor uniform
  integer quantisation; FP8 recipe beyond "E4M3" not stated. Accuracy from their C++/CUDA emulation library (Sec. 6.3).
- Accumulator (FP8 dMAC, Sec. 6.3): "After multiplication and rounding" the product is an E4M3-range value; its 4-bit
  significand becomes a 5-bit two's-complement number and is added into one of **16 narrow 5-bit registers indexed by
  the exponent**; on overflow the register is shifted by its exponent into "a wide 32-bit accumulator"; at the end the
  16 registers are shifted and added, then "normalized, rounded". Products with |w·x| < 2^-9 are skipped. So: product
  rounded to FP8, then exact fixed-point summation with a narrow-first, spill-on-overflow schedule. Integer MGS: narrow
  register until overflow, then spill; accumulators 8 to 20 bits swept against A2Q, A2Q+, AGS and clipping (Fig. 9).
- Width choice: fixed narrow width (5 signed bits for FP8), statistics of overflow (Markov model), wide fallback.
- Hardware: FPGA (Virtex-7, Table 2) and ASIC **post-layout with the ASAP7 7 nm predictive PDK**, 0.7 V, 500 MHz,
  Genus/Innovus, gate-level activity from model traces, Voltus power (Sec. 6.4). Unit: one MAC.
- Headline: Table 1 ImageNet Top-1 (FP32 / INT8 / FP8 / dMAC): ResNet-18 70.58 / 68.45 / 70.12 / 70.11; MobileNetV2
  71.6 / 69.7 / 71.04 / 71.10; ViT-Small 80.08 / 79.17 / 80.02 / 80.07. Table 3 total power: FP8 MAC 97.37 µW, FP8 dMAC
  64.66 µW (−33.6 %) and 64.15 µW with skipping (−34.1 %); INT8 27.48 to 23.25 µW (−15.4 %), INT8 dMAC area +14.5 %.
- Does not: sub-8-bit floats, a lossless width, a single narrow register without fallback, saturation or wrap studies,
  area reduction (area grows), exact products (they are rounded to FP8). Related: AGS, Natesh & Kung, ISCAS 2025,
  DOI 10.1109/iscas56072.2025.11044071 (VERIFIED, Crossref; not opened), and PQS, arXiv:2504.09064 (abstract only).

**[Lut24] D. R. Lutz, A. Saini, M. Kroes, T. Elmer, H. Valsaraju.** "Fused FP8 4-Way Dot Product With Scaling and FP32
Accumulation." ARITH 2024, pp. 40-47, DOI 10.1109/ARITH61463.2024.00016 (VERIFIED, Crossref). Full text: the paper PDF
on the open ARITH 2024 conference site (ac.uma.es/arith2024/papers/…).
- Formats: FP8 E4M3 and E5M2, internally a 9-bit E5M3 bias-15 format (Sec. II). No workloads, no network accuracy.
- Operation (eq. 4): res = rnd(2^-sf × SoP + acc), SoP the exact sum of 4 products, acc FP32, 7-bit scale sf;
  **one rounding** (RNE only), subnormals supported. Late-accumulation design: products expanded to a 68-bit fixed-point
  number and summed "without any loss of information", then a modified FP32 adder. Early-accumulation design: addends
  aligned to an anchor = max{Pexp} + ⌈log2 N⌉ + 1 (eq. 5; 34 for N = 4), 94-bit adders. Also a 2-way parallel FP8 DOT2
  into FP16 (Table II).
- Width choice: exact for the 4-term SoP by construction; the running accumulator is FP32.
- Hardware: **synthesis, 5 nm, 3.6 GHz** (Table I). Unit: dot-product operator. FP8-DOT4-LA 1133 µm² (SoP 572 + reused
  FADD32 561), FP8-DOT4-EA 674 µm²; reference FADD32-with-FMA 512 µm². Table II DOT2/4: 1758 and 975 µm².
- Its Sec. V describes [Des23] (secondary description, used below).
- Does not: any accuracy; sub-8-bit; narrower than FP32 running accumulators (except DOT2 into FP16); PTQ.

**[Des23] O. Desrentes, B. Dupont de Dinechin, J. Le Maire.** "Exact Dot Product Accumulate Operators for 8-bit
Floating-Point Deep Learning." Euromicro DSD 2023, pp. 642-649, DOI 10.1109/DSD60849.2023.00093 (VERIFIED, Crossref and
HAL API record hal-04240816). **Full text NOT opened:** inria.hal.science and the CCSD mirror served an "Anubis" bot
check; it was not circumvented. Characterisation from the HAL abstract (marked A) and from [Lut24] Sec. V (marked L,
secondary, UNVERIFIED against Des23 itself).
- Formats (A): FP8 E5M2, E4M3 and Posit8 with several exponent sizes; baselines FP16 and INT8 multiplicands.
- Accumulator (A): products expanded to fixed point and summed into wide accumulators; back-ends round to FP32, then to
  an 8-bit format. (L): unit (i) is a 16-pair dot product into a two's-complement fixed-point accumulator whose width
  depends on the operand range, guard bits for about 4,000 products (12 bits), **rounded up to a power of two**;
  "In the E5M2 configuration the authors fixed the width of the accumulator at 128-bits".
- Hardware (A): synthesis, TSMC 16FFC, area and power. Unit: dot-product operator. Numbers: not available here.
- Does not (A): network accuracy, sub-8-bit, per-layer or weight-dependent widths, narrowing below exact.

**[Ber22] L. Bertaccini, G. Paulin, T. Fischer, S. Mach, L. Benini.** "MiniFloat-NN and ExSdotp: An ISA Extension and
a Modular Open Hardware Unit for Low-Precision Training on RISC-V cores." arXiv:2207.03192v1 (VERIFIED, arXiv API);
IEEE ARITH 2022, pp. 1-8, DOI 10.1109/ARITH54963.2022.00010 (VERIFIED, Crossref). Full text: arXiv PDF v1.
- Formats (Sec. III.A): FP8 = E5M2, FP8alt = E4M3, FP16, FP16alt (E8M7 with IEEE rounding and subnormals).
- Accumulator: ExSdotp = a×b + c×d + e with 8-bit inputs and a 16-bit (FP16 or FP16alt) accumulator (or 16 to 32);
  products exact, three addends sorted and aligned with sticky bits, **one normalisation and rounding per two products**
  (Sec. III.B, Fig. 4). Accumulation sequential over ExSdotp calls (Fig. 9).
- Width choice: the destination format (2w bits); no analysis of required width.
- Hardware: **synthesis (unit) and place-and-route (cluster), GlobalFoundries 12 nm** (Sec. V.A). ExSdotp has about
  30 % less area and 30 % shorter critical path than two cascaded expanding FMAs at 333 MHz (Fig. 7a); the ExSdotp SIMD
  block is 44.5 kGE of a 165 kGE FPU; cluster 4.3 MGE at 1.26 GHz; 575 GFLOPS/W on FP8-to-FP16 GEMM (Sec. V.C).
- Accuracy: synthetic Gaussian dot products only (Table IV): relative error against FP64 for FP8-to-FP16, ExSdotp
  5.9e-4 / 2.7e-3 / 3.9e-3 versus cascaded ExFMA 5.9e-4 / 8.2e-3 / 1.2e-2 at n = 500 / 1,000 / 2,000.
- Does not: network accuracy, inference PTQ, sub-8-bit, integer or narrower-than-16-bit accumulators.

**[Sak19] C. Sakr, N. Wang, C.-Y. Chen, J. Choi, A. Agrawal, N. Shanbhag, K. Gopalakrishnan.** "Accumulation Bit-Width
Scaling For Ultra-Low Precision Training Of Deep Networks." arXiv:1901.06588v1 (VERIFIED, arXiv API); ICLR 2019 (arXiv
comment and PDF header; venue page not opened, UNVERIFIED). Full text: arXiv PDF.
- **The accumulator is floating point** (this closes the audit's open question): (1, e, m) notation, the analysis
  predicts the accumulator mantissa m_acc from the variance retention ratio VRR(n, m_p, m_acc) under swamping (Sec. 3),
  with a two-level chunking corollary (chunk 64). Experiments use 6 exponent bits in accumulations, (1,5,2) FP8 inputs,
  final layer 16 bit, rounding of partial sums in a modified CUDA GEMM (Sec. 5).
- Workloads: CIFAR-10 ResNet 32, ImageNet ResNet 18 and AlexNet. Regime: training (FWD, BWD, GRAD GEMMs).
- Headline (Table 1, ResNet 18, predicted mantissa bits, normal / chunk-64): FWD conv0 (9,6), blocks 1-4 (7,5), (8,5),
  (8,5), (9,6); GRAD up to (15,10). With the predicted widths training converges within 0.5 % of the
  full-precision-accumulation baseline; one bit less degrades (Fig. 6).
- Hardware: analytical area model "underpinned by the hardware synthesis of low-precision FPUs" (Fig. 1b, node not
  stated); 1.5 to 2.2× FPU complexity reduction claimed.
- Does not: inference, PTQ, integer or fixed-point accumulators, a lossless width.

**[Blu24] Y. Blumenfeld, I. Hubara, D. Soudry.** "Towards Cheaper Inference in Deep Networks with Lower Bit-Width
Accumulators." arXiv:2401.14110v1 (VERIFIED, arXiv API); ICLR 2024 (VERIFIED by the audit on proceedings.iclr.cc). Full
text: arXiv PDF v1.
- Formats: full-precision weights and activations, or FP8 M4E3 with per-tensor flexible bias (qtorch); accumulator and
  product as M/E floats (eq. 2). Workloads: ResNet18/34/50 on ImageNet; an MNIST MLP for 8-bit accumulators.
- Accumulator (Sec. 3, Fig. 1): every FMA quantises the product (Q_prod) and the accumulator (Q_acc) with **floor**
  (round-to-nearest is ruled out as too costly), chunk-based accumulation with a constant **chunk of 16**, accumulator
  **M7E4 = 12 bits**, biases b_acc = 10, b_prod = 12; underflow flushes to zero.
- Regime: **fine-tuning** (5 epochs, then 1 epoch with underflow enabled).
- Headline: Table 3 ResNet18 FP8 W/A: FP32 accumulation 69.90 %, 12-bit 69.70 % (dual-stage); ResNet50 76.25 / 76.22 %.
  **Table 8 (Appendix B), zero-shot, no fine-tuning**, full-precision pretrained ResNet18 (baseline 69.75 %): M10E5
  69.50, M9E5 68.95, M8E5 66.70, M7E5 57.09, M6E5 20.49; M7E4 with b_acc, b_prod = 10, 12: 60.14 %.
- Hardware: gate-count estimate with the gate costs of [vB23] (App. E): 12-bit accumulation cuts "the gate count by
  63 % compared to 32 bit FP accumulation". Evidence: analytical.
- Does not: PTQ (the zero-shot point is the only one and it fails), sub-8-bit operands, fixed-point accumulators for
  float operands, certificates, synthesis.

### Priority 2 (accumulator sections only)

- **[Agg24]** arXiv:2311.12359v3; FPL 2024, DOI 10.1109/FPL64840.2024.00048 (VERIFIED, audit). Full text v3 PDF, Sec. IV-V.
  Formats ExMy with bias 2^(e-1)−1, subnormals, and "we deviate from IEEE conventions on the treatment of inf and NaN and
  represent neither"; signed formats only, zero point 0. Accuracy "with accumulators in FP32". Integer MAC width
  "ra + rb + ⌈log2 n⌉ + 1" cited to A2Q; minifloat "2^ea + ma + 2^eb + mb + ⌈log2 n⌉ − 1" cited to [Ugu17]; n = 4,608.
- **[Dam24]** TCAD 44(6), DOI 10.1109/TCAD.2024.3511343 (VERIFIED, audit). Full text: author version, Zenodo 14312383 (open,
  CC BY). Eq. (1) maps an operand to the integer (−1)^s·(1.m)·2^(c−1)·2^M, "we do not model NaNs"; eq. (2)
  "L_default = 2^Ea + Ma + 2^Eb + Mb + ⌈log2 N⌉ − 1" with N the number of parallel lanes; the accumulator is "a flattened
  two's complement number". This fixes the convention: the formula counts the sign bit and no guard bits.
- **[Ugu17]** HAL hal-01488916: not opened (same HAL bot check). The formula's convention is taken from [Dam24], which
  shares two authors with [Agg24].
- **[Col23] A2Q** arXiv:2308.13504v1; ICCV 2023 (VERIFIED, audit). Full text Sec. 3. Data-type bound (eq. 9-10):
  P ≥ α + φ(α) + 1, α = log2 K + N + M − 1 − 1_signed(x), φ(α) = log2(1 + 2^-α), with |x| ≤ 2^N assumed for unsigned
  inputs. **Weight bound (eq. 11-13): P ≥ β + φ(β) + 1, β = log2‖w‖₁ + N − 1_signed(x), per output channel**, and
  "post-training minimization (PTM) of P according to the final weight values" in the experiments. QAT; CIFAR-10
  MobileNetV1/ResNet18, BSD300 ESPCN/U-Net; FINN FPGA. [Agg24]'s integer formula is not A2Q's: for signed 8-bit
  operands and n = 4,608, A2Q gives 28 bits, [Agg24] 30.
- **[Col24a] A2Q+** arXiv:2401.10432v1; ICML 2024. Zero-centred weights give a bound on the input range rather than
  max|x| (Prop. 3.2); ImageNet ResNet50 keeps 95 % of baseline at a 12-bit accumulator (QAT).
- **[Col24b] AXE** arXiv:2409.17092v2 (2025-07-31). Accumulator-aware GPFQ and OPTQ (PTQ); multi-stage accumulation as
  tiles of size T with one inner width P_I, e.g. "accumulating at 16 bits in 64-element tiles"; v2 evaluates LLMs only.
- **[Umu25] SIRA** arXiv:2508.21493v1: not re-read; the audit's full-text check stands (constant-weight interval
  analysis, scaled integers, FPGA, 22 % narrower accumulators).
- **[Cuy26]** arXiv:2511.06313v1; ASP-DAC 2026. MX MAC; FP32 partial result cut to a **16-bit mantissa**, chosen by
  comparing addition error with MX re-quantisation error on synthetic data; synthesis, GF 22FDX, PrimeTime PX power.
- **[vB23]** arXiv:2303.17951v2. Kulisch accumulator sized "FX+12 ... to allow for 2^12 = 4096 extreme products": "For
  INT8, this means a 15+12 = 27-bit accumulator, and for e.g., FP8-E4, this would be INT37"; gate counts, analytical.
- **[Joh18]** arXiv:1811.01721v1. Kulisch accumulator spanning ±(fmax² + fmin²), but for energy "we restrict accumulator
  range to [fmin, fmax]": an exact accumulator deliberately narrowed at both ends; 28 nm synthesis; ResNet-50 within
  0.9 % top-1 without retraining.
- **[Ni20] WrapNet** arXiv:2007.13242v1; ICLR 2021. Two's-complement wrap-around accumulation with a cyclic activation and
  an overflow penalty; 8-bit accumulators; binary/ternary weights; training; 28 nm custom hardware; 2.4× CPU speed-up.
- **[Xie20]** arXiv:2005.13297v1 (IJCAI 2020, different title per audit). Trained per-tensor fixed-point bits so that a
  16-bit integer accumulator never overflows; CPU speed-up.
- **[Wan18]** arXiv:1812.08011v1; NeurIPS 2018 (comment). FP8 (1,5,2) products accumulated in **FP16 defined as (1,6,9)**,
  not IEEE binary16, with chunk-based accumulation and stochastic rounding; training.

## Part B. Design points a reviewer can ask the project to match

Project policies today (contract 2.1): `wide` (certified exact), `control` (sequential binary32 FMA, one RNE per MAC),
`sat.w<W>` (uniform signed saturating, product-grid LSB), `sat.struct-<d>` (per node, planned/deferred), `fp16[.x<e>]`
(IEEE binary16, one RNE per MAC), `f21[.x<e>]` (E8M12). One tap order (input channel, kernel row, kernel column).
ResNet18 numbers are from `docs/analysis/accumulator-sweep-2026-10-01.md` and `results/summaries/accumulator-sweep-v1/`
(1,000 development images). Q1 (MobileNet) and Q8 numbers are INTERIM (documents modified 17:14 and 17:30 today).

| # | Design point (source) | Exact semantics | Project evidence | Status | Run or policy that would answer it |
|---|---|---|---|---|---|
| 1 | FP8 products accumulated in FP32 ([Agg24] assumption, [Nat25] and [Blu24] baselines, [Lut24], [Des23] back-end) | sequential binary32, RNE per add | `control`: bit-identical to exact on 1,000 images for INT8, INT8-signed, INT6, FP6, FP7; differs in 162 / 7 / 28 outputs for FP8 E4M3 / posit8 / E5M2, within 1 pt | covered (sequential) | fused DOT4 then FP32 ([Lut24]): `fp32.dot4` (exact 4-term sum, one RNE into the FP32 state); low priority |
| 2 | FP8 into FP16 ([Ber22] ExSdotp, [Lut24] DOT2, [Nat25] text) | binary16, one rounding per 2 products (fused) | `fp16.x<e>`: one RNE per MAC, within 1 pt of exact in every ResNet18 case | partly | `fp16.dot2` (exact 2-term sum + acc, one RNE); FP16alt (E8M7, = bfloat16 with IEEE subnormals): `bf16[.x<e>]` |
| 3 | FP16 as (1,6,9) with chunk 64 ([Wan18], training) | 6-bit exponent, 9-bit fraction | none | not covered | `f16e6m9.chunk64`; only if a reviewer insists (training paper) |
| 4 | 12-bit float accumulator M7E4, **floor** rounding, chunk 16, product also quantised ([Blu24]) | Q_prod and Q_acc floor, underflow flush to 0, saturate at max | `f21` (E8M12) and `fp16` only; both lossless here | not covered | `f12.e4m7.floor.chunk16` with b_acc = 10, b_prod = 12, plus an RNE variant; [Blu24] Table 8 predicts failure under PTQ (60.14 % for FP32 W/A ResNet18) |
| 5 | Float accumulator mantissa sweep ([Blu24] M6E5…M10E5; [Sak19] m_acc with e = 6) | p-bit significand, fixed exponent | `fp16` (p = 11), `f21` (p = 13) | partly | `fe5m<k>` and `fe6m<k>` for k = 4 to 10, RNE and floor, with and without chunk 64: gives a float failure width to set beside `sat.w<W>` |
| 6 | Narrow integer register, saturating ([Nat25] clipping baseline, [Agg24] widths) | signed, clamp after every add | `sat.w<W>`: W_acc(1.0) INT8 19, INT6 15, FP6 17, FP7 23, E4M3 39, posit8 32, E5M2 > 63; cert − W_acc(1.0) = 6 to 8 (posit8 18) | covered (uniform) | per-node `sat.struct-<d>` (deferred); MobileNet (Q1, INTERIM) |
| 7 | **Wrap-around** two's complement ([Ni20], [Nat25] "transient overflow") | modulo 2^W, no clamp | none | not covered | `wrap.w<W>`. Exact whenever the final sum fits in W bits, whatever the prefixes do, so its lossless width is a final-sum bound, not a prefix bound; expected to beat `sat.w<W>` near W_noevent and to collapse abruptly below; cheap to add |
| 8 | Chunked / two-stage accumulation ([Sak19] chunk 64, [Blu24] chunk 16, [Col24b] tiles with inner P_I, [Nat25] narrow + 32-bit spill, [Ber22] 2-term fused) | inner register per tile of T taps, outer wide (or narrow) register | none; one tap order | not covered | `sat.w<W>.tile<T>` (inner W saturating, outer exact), T = 16, 64, and per-kernel-row; `fp16.tile<T>`; the certificate per tile is K = T instead of 4,608, so the lossless inner width drops by about log2(4608/T) bits |
| 9 | Product rounded before accumulation ([Nat25] FP8 product; [Blu24] Q_prod) | product RNE (or floor) to an 8-bit or M/E format | products exact in every policy | not covered | `prod.e4m3` (RNE) option combined with `wide`: isolates product rounding from accumulator narrowing |
| 10 | Exact Kulisch accumulator with closed-form width ([Agg24], [Dam24], [Ugu17], [Des23] 128-bit E5M2, [Lut24] 68-bit FX, [vB23] FX+12) | fixed point covering all products of the format | `wide` with certificate per node; Part C comparison | covered | print the Part C three-column table |
| 11 | Range-restricted exact accumulator, LSBs dropped ([Joh18] [fmin, fmax]; [Cuy26] 16-bit mantissa partial result) | fixed point with fewer fraction bits (truncate or round low bits of each product) | `sat.w<W>` narrows only from the MSB side; LSB is always the product grid | not covered | `fx.w<W>.f<F>`: W-bit register whose LSB is 2^-F of the output unit, RNE or truncation per product. This is the realistic narrow fixed-point accumulator for minifloats, whose product grid spans 2^-6 to 2^6 (FP6) |
| 12 | Weight-constrained integer accumulators (A2Q QAT, A2Q+ 12-bit, AXE PTQ) | weights changed so that ‖w‖₁ fits P | Q8 (INTERIM, 17:30): narrowest P within 1 pt, AXE INT8 25 / INT6 21 against `sat.wP` 19 / 15 | partly (integers, ResNet18) | AXE-style projection for minifloat weight grids (not in any paper) |
| 13 | Scale inside the dot product ([Lut24] 2^-sf before the FP32 add; MX shared exponents [Cuy26]) | scaled SoP added to a float state | scales applied after the exact dot in binary64; shared-exponent formats not admitted (contract 2.1) | not covered | needed only for MX/BFP arms; out of the current headline |

## Part C. The bit-counting convention

**The published formula, with its own definitions.** [Dam24] eq. (1) writes a normal operand as the integer
(−1)^s·(1.m)·2^(c−1)·2^M and a subnormal as (−1)^s·(0.m)·2^M, with c from 0 to 2^E − 1 and no NaN codes; [Agg24] uses
the same formats (bias 2^(e−1) − 1, subnormals, neither Inf nor NaN). In units of the smallest subnormal the largest
magnitude is (2^(M+1) − 1)·2^(2^E − 2), which has 2^E + M − 1 bits. A product of two operands therefore has at most
2^Ea + Ma + 2^Eb + Mb − 2 magnitude bits; n products add ⌈log2 n⌉ bits; the accumulator is "a flattened two's
complement number", which adds the sign bit. Sum: 2^Ea + Ma + 2^Eb + Mb + ⌈log2 n⌉ − 1. **The formula counts one sign
bit, no guard, round or sticky bits, and its LSB is the product of the two smallest subnormals** — the same register
unit and the same sign convention as the certificate. The integer formula of [Agg24], ra + rb + ⌈log2 n⌉ + 1, counts
both operand sign bits as magnitude and adds one more; A2Q's own data-type bound (which [Agg24] cites) is
⌈log2 n + ra + rb − 1 − 1_signed(x)⌉ + 1 (up to the φ term).

**The certificate** (`tools/scaled_bridge_v2/certificates.py`, `certify`): for every output channel c,
bound_c = max|a| · Σ_k |w[c,k]|, with a over every level of the input codebook in units of the input step and w the
exported weight integers in units of the weight step; width = bit_length(max_c bound_c) + 1. It bounds every prefix in
any order; the register unit is the product grid. The structural width replaces the input range by [0, max] behind a
ReLU and uses positive and negative weight sums separately.

**Worked example, FP6 E2M3, ResNet18 `layer4_1_conv1` (K = 4,608 = n), B2 export** (run `7c6344af…`, identical in
`1f75c923…`):
1. Formula: (4 + 3) + (4 + 3) + 13 − 1 = **26**.
2. Same worst case without the separate ceilings, [Agg24] format (max 7.5 = 60 units of 2^-3): 4,608 · 60 · 60 =
   16,588,800 < 2^24, so **25** bits. The lost bit is form: 3,600 is rounded to 2^12 and 4,608 to 2^13 separately.
3. Same worst case, project format (fp6_e2m3 reserves S.11.111 as NaN, max 7.0 = 56 units): 4,608 · 56 · 56 =
   14,450,688 < 2^24, still **25**. Convention, 0 bits here.
4. Certificate: max|a| = 56, worst channel Σ|w| = 63,683 units (24.7 % of 4,608 · 56 = 258,048): 3,566,248 < 2^22,
   **23** bits (structural 23). Substance: 2 bits. (The contract's table quotes Σ|w| = 62,997; that number is the B1
   maxabs export, `2a642161…/resnet18-fp6_e2m3-maxabs-b1`. Same widths.)

All ResNet18 B2 cases (network maximum, which is a K = 4,608 node in every case; `partc_widths_out.txt`):

| Case | Formula as printed | Exact worst case, [Agg24] convention | Exact worst case, project codebooks | Certificate abs / struct | Form | Convention | Substance (weights) |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP6 E2M3 | 26 | 25 | 25 | 23 / 23 | 1 | 0 | 2 |
| FP7 E3M3 | 34 | 33 | 33 | 31 / 30 | 1 | 0 | 2 |
| FP8 E4M3FN | 50 | 49 | 49 | 47 / 46 | 1 | 0 | 2 |
| FP8 E5M2 | 80 | 79 | 77 | 75 / 74 | 1 | 2 (IEEE Inf/NaN take the top binade of each operand) | 2 |
| INT8 (unsigned acts) | 30 | 28 (signed × signed) | 29 | 27 / 26 | – | 1 (formula counts both sign bits) | 2 |
| INT8, signed acts | 30 | 28 | 28 | 26 / 25 | – | 2 | 2 |
| INT6 (unsigned acts) | 26 | 24 | 25 | 23 / 22 | – | 1 | 2 |
| posit8 es1 | – | – | – | 50 / 49 | – | – | – |

(For integers the separate ceilings cost nothing at n = 4,608, so the whole formula excess is convention.) The worst-channel Σ|w| is 21 to 26 % of K·max|w| in every case; a factor of 4 is 2 bits.

**Which part is substance.** Only the last column: a bound computed from the actual weight codes against a worst case
over the format. It is real but it is the minifloat transcription of A2Q's weight bound ([Col23] eq. 13, β = log2‖w‖₁ +
N − 1_signed), which already includes per-channel ℓ1 norms and post-training minimisation for integers; and SIRA [Umu25]
does the same with interval arithmetic for scaled integers. The activation side of the certificate is still a worst
case over the codebook (max|a|); no data enter it. The structural width adds 0 to 1 bit (ReLU sign).

**How the paper should print it.** Three columns per format: the published closed form exactly as printed (with "[Agg24]
formats have no NaN/Inf codes"), the exact data-type bound in the project's convention (bit_length(K · max|a| · max|w|) +
1 per node), and the certificate. Claim only the second-to-third difference (2 bits for ResNet18) and attribute the idea
to A2Q for integers. Never write "the certificate is 3 bits tighter than the published formula". The more useful
comparison for the headline is the certificate against the measured W_acc(1.0) (6 to 8 bits, posit8 18), which no
published bound predicts.

## Part D. Positioning

### (i) Sentences the paper can defend, and sentences it must not write

Defensible (all ResNet18 development evidence today; MobileNet when Q1 completes):
1. "Under post-training quantization, a uniform saturating integer accumulator keeps ImageNet Top-1 within one point of
   exact execution down to 6 to 8 bits below the certified lossless width for FP6, FP7, FP8 E4M3 and 6- and 8-bit integer
   formats (18 bits for posit8), and fails within 2 to 3 further bits."
2. "We certify a lossless accumulator width per layer from the exported weight codes and the input code domain, the
   extension of A2Q's weight-norm bound to minifloat and posit grids; it lies 2 bits below the exact data-type bound for
   every ResNet18 case."
3. "Sequential binary32 accumulation is provably exact for FP6 E2M3 and INT6 on ResNet18 (certificate 23 bits ≤ 24),
   and is bit-identical on 1,000 images for INT8 and FP7 although not certified."
4. "Binary16 with a chosen binary point and a 21-bit float accumulator stay within one point of exact execution for
   every format studied (paired resolution about ±0.8 points on 1,000 images), extending the FP8-into-FP16 practice of [Ber22] and [Lut24] to sub-8-bit operands at network level."
5. "Prior narrow-accumulator results rely on training or fine-tuning ([Col23], [Col24a], [Blu24], [Ni20]), reorder or
   spill sums ([Nat25]), or target 8-bit operands ([Nat25], [Blu24], [Ber22], [Lut24], [Des23]); we measure the
   failure width under PTQ for sub-8-bit formats with exact execution."
6. "Without fine-tuning, a 12-bit floating-point accumulator is known to cost about 10 points on ResNet18 ([Blu24],
   Table 8)" — cite as their number, with their floor rounding and chunk 16.

Must not write:
- "first study of narrow accumulators for low-precision floating-point inference" ([Nat25], [Blu24], [Sak19], [Wan18]);
- "first lossless / overflow-free accumulator width" or "first weight-dependent width" ([Col23] eq. 13, [Umu25], Kulisch);
- "3 to 5 bits tighter than the published formula" (Part C: 2 bits are substance);
- "first ASIC cost of minifloat accumulation" ([Des23] 16 nm, [Ber22] 12 nm, [Lut24] 5 nm, [Nat25] ASAP7, [Joh18] 28 nm);
- "FP32 accumulation is required for FP8" or "12-bit accumulators suffice" as general statements;
- anything about MobileNet, detectors or held-out data before those runs exist.

### (ii) The audit's verdicts

- **C3: narrowed.** The audit already saw [Umu25] and the L1 nature of [Col23]. The full text adds: A2Q states the
  per-channel weight bound and post-training minimisation explicitly; [Dam24] fixes the formula's convention so that the
  3-bit gap splits 1 + 0 + 2; [Sak19] is a float-accumulator analysis. What remains is the non-integer transcription of a
  known integer bound, used as the lossless reference of a PTQ sweep; the FP32-exactness corollary stands.
- **C4: unchanged.** [Nat25] emulates its own unit bit-accurately and [Blu24] simulates FMAq; neither compares a
  framework simulator against exact execution at network level. New adjacent item: Khattak, Mikaitis, Graziani, "Accurate
  Models of AMD Matrix Cores", arXiv:2609.14845 (abstract only), which documents vendor accumulator semantics.
- **Section 4 recommendation: unchanged in direction, narrowed in wording.** Condition (1) of the audit's check holds and
  is now concrete: compare against the FP8 design points of Part B rows 1, 2, 4 and 8. Condition (2) is resolved by
  Part C (2 bits of substance, attributed to A2Q). New risk: the ASIC half rests on integer MACs while the closest papers
  have FP8 ASIC datapaths; either the non-integer MAC RTL comes back (owner hold) or the claim says "integer MAC cost,
  minifloat accumulator widths".

### (iii) Bounded search for closer work (2026-10-02, 18:40 to 19:00 +05)

- arXiv API, sorted by submission date, results from 2024-06 on inspected by title, abstracts of candidates read:
  `abs:"accumulator bit width" OR "accumulator bitwidth" OR "accumulation bit-width" OR "accumulator width" OR
  "accumulator precision"` (15 hits); `abs:accumulator AND (minifloat OR FP8 OR FP4 OR "low-precision floating")`
  (36); `abs:block AND abs:floating AND abs:accumulator` (16); `abs:posit AND (quire OR accumulat)` (15);
  `abs:accumulator AND abs:overflow AND abs:quantization` (13). Two queries failed and are not counted: `abs:accumulator
  AND abs:"post-training"` and `abs:logarithmic AND abs:accumulator` returned unrelated results (query parsing).
- Semantic Scholar citations API for [Blu24] (6 citing records), [Nat25] (1) and [Agg24] (10); one general web search.
- Crossref title queries for the venues of [Nat25] and its baselines.
- Found and relevant, none closer than the set above: [Nat25]'s IPDPS 2026 version; AGS (ISCAS 2025) and PQS
  (arXiv:2504.09064, Natesh & Kung: pruning, quantisation and sorting for low-bitwidth integer accumulation); TransDot
  (arXiv:2605.07245, FCCM 2026, FPGA trans-precision dot-product accumulation); Ten-Four (arXiv:2512.00053, fused dot
  product unit); MXDOTP/VMXDOTP (arXiv:2505.13159, 2603.04979, MX dot products on RISC-V); TREA (arXiv:2605.07321, edge
  accelerator with reduced accumulator width, under review); "Exploring Microscaling MX Minifloat Systolic Arrays on
  FPGAs" (2025, title only, citing [Agg24]). All abstract or title only.
- **Not found in this bounded search:** a paper that sweeps accumulator width to a failure point for sub-8-bit minifloat,
  posit, logarithmic or block-floating-point formats under PTQ on CNNs, or that certifies per-layer lossless widths
  for non-integer formats.

## Corrections to the audit (tracked file; not edited)

1. Entry [Sak19]: the accumulator **is** floating point (mantissa precision with 6 exponent bits); the open question in
   §1.7 and §7.5 is closed.
2. Entry [Nat25]: published as IPDPS 2026, pp. 557-569, DOI 10.1109/ipdps65963.2026.00054, title "MGS: Markov Greedy Sums
   for Low-Power DNN Accumulation" (Crossref). Its evidence level is post-layout (ASAP7 predictive 7 nm), not just
   "hardware power"; its FP8 products are rounded before an exact narrow-plus-32-bit accumulation.
3. Entry [Lut24]: the abstract and full text are now read from the open ARITH 2024 site (primary); synthesis is 5 nm at
   3.6 GHz; there is no network accuracy, so its overlap with C4 is about datapath semantics only.
4. Entry [Des23]: full text still unread (bot check); [Lut24] reports a 128-bit power-of-two accumulator for E5M2.
5. §1.1 worked example and §6: the 3-bit gap is 1 bit of formula form plus 2 bits of weights for FP6/FP7/FP8 E4M3;
   for E5M2 (5 bits) 2 more bits come from the project's IEEE Inf/NaN encoding. The formula counts the sign bit.
6. §1.1 [Agg24]: its integer formula is attributed to A2Q but is 2 bits looser than A2Q's data-type bound for signed
   operands (30 against 28 for INT8, n = 4,608).
7. §1.7 [Col23]: A2Q includes a per-channel weight-ℓ1 bound used after training (Sec. 3.2, "post-training
   minimization"), so the weight-dependent certificate is prior art for integers, not only for [Umu25].
8. §1.13 [Ber22]: its accuracy evidence is synthetic dot products against FP64 (Table IV); no network accuracy.
9. [Blu24]: its accumulator rounding is floor and chunk size 16; the zero-shot table (Appendix B) is a PTQ-like data
   point that the audit did not mention.

## Verification log

| Key | Source opened | Status |
|---|---|---|
| Nat25 | arXiv PDF v1; arXiv API; Crossref (IPDPS 2026, AGS) | full text (arXiv v1); published version not opened |
| Lut24 | ac.uma.es ARITH 2024 open PDF; Crossref | full text |
| Des23 | HAL API record; HAL and CCSD document links returned an Anubis bot check | abstract only (+ [Lut24] Sec. V) |
| Ber22 | arXiv PDF v1; arXiv API; Crossref | full text |
| Sak19 | arXiv PDF v1; arXiv API | full text |
| Blu24 | arXiv PDF v1; arXiv API | full text |
| Agg24 | arXiv PDF v3 | Sec. IV-V |
| Dam24 | Zenodo 14312383 author version (CC BY) | equations (1)-(2), Sec. III |
| Col23, Col24a, Col24b, Cuy26, vB23, Joh18, Ni20, Xie20, Wan18 | arXiv PDFs; arXiv API | targeted sections |
| Ugu17 | not attempted after the HAL bot check on Des23 | not opened |
| Umu25 | not re-read (audit's full-text check) | – |
| Search candidates | arXiv API abstracts; Semantic Scholar API; one web search | abstract or title only |

Quotations were matched against `pdftotext` output of the downloaded PDFs (layout and reading-order versions); line
references are in `artifacts/literature_v1/notes/papers.md`. Part C numbers were recomputed from the stored certificates
`artifacts/scaled_bridge_v2/runs/{7c6344af…,1f75c923…}/resnet18-*-default-b2/certificate.json` and the accepted format
manifests; the two runs agree. No paywall, login or bot check was bypassed.
