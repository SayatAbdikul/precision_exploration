# Related-work and novelty audit — 2026-10-01

Lane L3, part 1. Documents only; no experiment was run and no repository
evidence was changed. Scope: the public paper "Hardware-Cost-Aware Exploration
of Sub-8-Bit Numerical Formats for Edge CNN Inference"
([publication plan](../publications/README.md)), tested against candidate
claims C1–C5.

## 0. How this audit was verified, and its limits

- **VERIFIED** means the bibliographic fields and the characterisation come from
  a primary page opened during this audit: the arXiv abstract page, the arXiv
  API record (`export.arxiv.org/api/query?id_list=…`, which returns the arXiv
  title, authors, abstract, and the author-supplied comment and journal-ref
  fields), a proceedings page (PMLR), a publisher-deposited record (Zenodo for
  the TCAD paper), or an official conference programme or CFP page.
- **Venue (arXiv comment)** means the venue appears only in the author-supplied
  arXiv comment or journal-ref field. That is author metadata, not a publisher
  page.
- **UNVERIFIED** marks every field that rests only on a search-engine snippet
  or on a page that could not be opened. ACM DL, IEEE Xplore, OpenReview, CVF
  Open Access, dblp, MDPI, the OCP document server and one Harvard lab page
  returned HTTP 403, bot checks or empty bodies during this audit. DOIs for
  those papers are given as UNVERIFIED unless another primary page confirmed
  them.
- Characterisations are limited to what the abstract supports. Full text was
  read only for **Aggarwal et al. 2024 ("Shedding the Bits")**, through the
  arXiv HTML of v3, because it is the closest prior work. A small model
  summarised every page, so short quotations are reproduced as relayed and were
  not checked character by character against the PDF.
- Searches were run on 2026-10-01 with a general web search engine plus the
  arXiv API. Searches cannot prove a negative. "Apparently open" in Section 2
  means "not found in this audit", never "nobody has done it".
- The ImageNet 50k validation set is not available to this project for now
  (user statement for this wave). Any claim below that needs full-benchmark
  evidence is marked as blocked on that.
- **Independent check (lane L3, part 2).** A second agent re-checked every
  entry on 2026-10-01 and corrected this document in place. Changed statements
  carry the tag `[check]`. Section 7 lists what was checked, what changed and
  what is still UNVERIFIED. Extra sources used by the check: the Crossref REST
  API (publisher-deposited title, authors, venue, pages and DOI), the HAL
  archive API, official proceedings listings (NeurIPS, MLSys, ICLR, HPCA), and
  arXiv HTML full text downloaded and searched locally rather than summarised
  by a model.

Evidence-level vocabulary for hardware: **none**; **analytical** (closed-form or
model-based cost); **FPGA** (post-implementation LUT/DSP); **synthesis** (ASIC
logic synthesis); **post-layout**; **silicon**. "Not stated in abstract" means
the abstract does not say which.

## 1. Annotated bibliography

Each entry gives: authors; title; venue and year; identifier; opened source;
formats, models and bit widths; PTQ or QAT; simulation or bit-exact; hardware
evidence; headline (numbers only where the opened source states them); and
overlap with C1–C5.

### 1.1 Low-bit floating point versus integer inference

**[vB23] M. van Baalen, A. Kuzmin, S. S. Nair, Y. Ren, E. Mahurin, C. Patel,
S. Subramanian, S. Lee, M. Nagel, J. Soriaga, T. Blankevoort. "FP8 versus INT8
for efficient deep learning inference." arXiv:2303.17951 (2023; no venue on
arXiv).** VERIFIED (arXiv abstract page). FP8 versus INT8 for edge inference;
PTQ and QAT; the abstract does not say whether accuracy comes from simulation.
Hardware: the abstract states that "FP formats are somewhere between 50-180%
less efficient in terms of compute in dedicated hardware than the INT format".
Evidence level: **analytical** `[check]`. The full text (arXiv HTML v2) uses the
count of equivalent 2-input gates as "a first-order approximation" of area and
power, not synthesis, and it costs two accumulator types, "Kulisch and
floating-point accumulators". It also cites a synthesis-based estimate by
Rouhani et al. (2023) of a 40% performance decrease of FP8 against INT8. The
accumulator is therefore already part of this paper's cost comparison. Headline: INT8 is preferred for on-device inference, including for
FP8-trained networks converted to INT8. Overlap: C2 (a hardware-cost argument
against FP); C1 only in the sense that PTQ and QAT are both reported. Differs:
8-bit only; no sub-8-bit, exact execution or accumulator certificate in the
abstract.

**[Kuz22] A. Kuzmin, M. van Baalen, Y. Ren, M. Nagel, J. Peters,
T. Blankevoort. "FP8 Quantization: The Power of the Exponent."
arXiv:2208.09225 (2022).** VERIFIED (arXiv API). Venue NeurIPS 2022:
VERIFIED `[check]` (Main Conference Track listing on
proceedings.neurips.cc/paper_files/paper/2022). FP8 with varying exponent
bits versus INT8, across many networks; PTQ and QAT. Hardware: none stated.
Headline: for PTQ, FP8 beats INT8 in accuracy; the best exponent count depends
on outlier severity; under QAT "the difference in formats disappears"
(abstract) `[check]`. **Most direct prior evidence for
C1**: the recipe (PTQ versus QAT) changes how much the format matters. It does
not show a ranking reversal across PTQ recipes at sub-8 bits.

**[Shen24] H. Shen, N. Mellempudi, X. He, Q. Gao, C. Wang, M. Wang. "Efficient
Post-training Quantization with FP8 Formats." arXiv:2309.14592 (2023).**
VERIFIED (arXiv API). Venue MLSys 2024: VERIFIED `[check]` (listed on
proceedings.mlsys.org/paper_files/paper/2024). E5M2, E4M3 and E3M4 against INT8
over 75 architectures; PTQ. Hardware: none stated. Headline: FP8 workload
coverage of 92.64% versus 65.87% for INT8; E4M3 suits NLP and E3M4 suits
vision. Overlap: C1 (format ranking depends on domain). Differs: 8-bit only;
no cost model.

**[Zha23] Y. Zhang, L. Zhao, S. Cao, W. Wang, T. Cao, F. Yang, M. Yang,
S. Zhang, N. Xu. "Integer or Floating Point? New Outlooks for Low-Bit
Quantization on Large Language Models." arXiv:2305.12356 (2023).** VERIFIED
(arXiv API). Layer-wise selection between INT8/INT4 and FP8/FP4 (MoFQ) on
LLaMA; PTQ, weight-only and weight-activation. Hardware: none stated. Overlap:
C5 (per-layer format selection); C1 (the best format differs by layer). Differs:
LLM; no accumulator or complete cost.

**[Che25] M. Chen, M. Wu, H. Jin, Z. Yuan, J. Liu, C. Zhang, Y. Li, J. Huang,
J. Ma, Z. Xue, Z. Liu, X. Bin, P. Luo. "INT v.s. FP: A Comprehensive Study of
Fine-Grained Low-bit Quantization Formats." arXiv:2510.25602 (2025).**
VERIFIED (arXiv API). FP versus INT across granularities, including MX with
block size 32; LLMs; training and inference. Headline as relayed: for 8-bit
fine-grained formats "MXINT8 is superior", while 4-bit FP formats often win.
Hardware `[check]`: the abstract does claim that MXINT8 beats its FP
counterpart "in both algorithmic accuracy and hardware efficiency". The full
text (arXiv HTML v1) bases this on a hardware model of a matrix-multiply unit
(**analytical**; the word "synthesis" does not occur in the text) and states
that "MXINT8 and NVINT4 reduce energy by 37% and 38%, respectively, compared
with MXFP8 and NVFP4" (its Table 5). This replaces the earlier unverified
"20–40%" snippet. Overlap: C1/C2 (the INT-versus-FP ordering depends on bit width
and granularity). Differs: LLM; no edge CNN; no exact accumulator study.

**[Agg24] S. Aggarwal, H. J. Damsgaard, A. Pappalardo, G. Franco, T. B. Preußer,
M. Blott, T. Mitra. "Shedding the Bits: Pushing the Boundaries of Quantization
with Minifloats on FPGAs." arXiv:2311.12359 v3; FPL 2024, pp. 297–303, DOI
10.1109/FPL64840.2024.00048 (Crossref `[check]`).**
VERIFIED (arXiv abstract page and v3 HTML full text; the quotations and both
formulas below were re-checked against the downloaded HTML `[check]`). **Closest prior work to
this paper.** Models: ResNet-18, MobileNetV2 and ViT-B-32 on ImageNet-1K, with
1,000 calibration images. Formats: minifloats ExMy and integers, 3–8 bits for
weights and activations. PTQ with SmoothQuant, bias correction,
gradient-based learned rounding and GPTQ. Per the full text, the paper reports
"best-case results across all post-training optimization techniques" and does
not systematically study whether the technique changes the minifloat/integer
ranking. Accuracy is from Brevitas fake quantisation with FP32 accumulators
(simulation, not bit-exact). Hardware: FPGA, post-implementation LUT counts on
an AMD Versal VCK190. Accumulator widths come from closed-form worst-case
formulas: integer `ra+rb+⌈log2 n⌉+1`, and minifloat
`2^ea + ma + 2^eb + mb + ⌈log2 n⌉ − 1`, with n the largest dot-product length
(4,608 for ResNet-18). The paper notes that the minifloat width "grows
exponentially with the exponent bit-widths". Headline: "Minifloat quantization
typically outperforms integer quantization for bit-widths of four or more", but
"integer quantization often retains its Pareto optimality" once MAC resource
cost is included, "due to its slightly smaller hardware footprint than
minifloats at a given precision" `[check]`. These sentences are in the
introduction; the v3 abstract says only that minifloats "offer a promising
alternative for emerging workloads such as vision transformers". Overlap: **C2** (directly: complete MAC cost keeps integer on
the Pareto front); **C3** (closed-form lossless widths for minifloats exist);
**C1** (multiple PTQ recipes, but best-case only); **C4** (it relies on FP32
fake quantisation, which is the practice C4 questions). Differs: FPGA LUTs, not
an ASIC flow; worst-case formula, not a per-layer certificate from the admitted
code domain; no scale, requantisation or memory cost; no exact execution; no
paired confidence intervals.
Three further points from the full text `[check]`: (i) the paper states that
"our model accuracy assessments are conducted with accumulators in FP32" and
that, given "the limited precision of the minifloat formats under
consideration, this choice adequately ensures the validity of our results".
It asserts this and gives no certificate, which is the opening C3 and C4 use;
(ii) it omits "the hardware costs associated with converting the accumulated
values in the minifloat MAC back into the floating-point format", which is a
part of the complete cost in C2; (iii) the minifloat accumulator formula is
cited to Uguen and de Dinechin's Kulisch-accumulator report [Ugu17] (Section
1.13), and the integer formula to A2Q [Col23].
Worked example (this audit's arithmetic on the formula as relayed, not a number
from the paper): for E2M3 on both operands with n = 4,608 the formula gives
4 + 3 + 4 + 3 + 13 − 1 = 26 bits. The project's per-layer certificate for
ResNet-18 FP6 E2M3 gives a largest signed width of 23 bits, below the 24 bits
FP32 represents exactly ([scaled-bridge v1](scaled-bridge-v1-results-2026-09-28.md)).
This is the concrete gap between a generic bound and a code-domain certificate
that C3 can occupy. The width conventions must be matched exactly before this
comparison is printed. `[check]` The same arithmetic gives 32 bits for E3M2 and
34 bits for E3M3, against certified widths of 29 and 31 bits in the same
project document. The difference is 3 bits in all three cases. A constant
offset suggests that it may come from the width convention or from the
`⌈log2 n⌉` term rather than from format-specific structure. This must be
understood before C3 claims that the certificate is tighter.

**[Dam24] H. J. Damsgaard, K. J. Hossfeld, J. Nurmi, T. B. Preusser. "Parallel
Accurate Minifloat MACCs for Neural Network Inference on Versal FPGAs." IEEE
TCAD 44(6):2181–2194, June 2025 (online in 2024). DOI
10.1109/TCAD.2024.3511343.** VERIFIED (Zenodo record with abstract and DOI;
volume, issue, pages and the author spelling "Hoßfeld" from Crossref
`[check]`). "Accurate" (exact-sum) minifloat MACCs against integer MACCs
on AMD Versal; FPGA. Headline: custom compressor trees cut minifloat MACC area
by 17.7% and raise clock frequency by 16.2%. Minifloat MACCs use "20% to 180%
more resources" than same-size integer MACCs without conversion back to float,
and "60% to 300% more" including it. Overlap: **C2** (conversion support cost
changes the ranking gap) and C3 (exact accumulation for minifloats). Differs:
FPGA; no network-level quality, scale policy or memory.

**[Ger23] C. Gernigon, S.-I. Filip, O. Sentieys, C. Coggiola, M. Bruno.
"Low-Precision Floating-Point for Efficient On-Board Deep Neural Network
Processing." arXiv:2311.11172; EDHPC 2023 (arXiv comment).** VERIFIED (arXiv
abstract). 6-bit minifloat weights and activations; QAT; Thin U-Net
segmentation. Headline: 0.3% degradation against 0.5% for 6-bit integer.
Hardware: preliminary only, per the abstract. Overlap: weak C1/C2 (an edge
minifloat-versus-integer comparison).

**[Liu23] S. Liu, Z. Liu, X. Huang, P. Dong, K.-T. Cheng. "LLM-FP4: 4-Bit
Floating-Point Quantized Transformers." arXiv:2310.16836; EMNLP 2023 (arXiv
comment).** VERIFIED (arXiv API). FP4 PTQ for LLaMA-13B with exponent-bias
reparameterisation; no hardware. Peripheral (LLM).

**[Li23] J. Li, T. Zhang, I. E.-H. Yen, D. Xu. "FP8-BERT: Post-Training
Quantization for Transformer." arXiv:2312.05725 (2023).** VERIFIED (arXiv API).
FP8 PTQ beats INT8 PTQ on BERT. Peripheral.

**[ZhaM26] M. Zhang, J.-F. Li, Z. Sun, H. Bai, H.-L. Zhen, Z. Dong, X. Yu.
"Benchmarking Post-Training Quantization of Large Language Models under
Microscaling Floating Point Formats." arXiv:2601.09555 (2026).** VERIFIED
(arXiv API). 7 PTQ algorithms, 15 benchmarks, 3 LLM families; MXFP8 and MXFP4.
Headline: "PTQ effectiveness under MXFP depends strongly on format
compatibility". Overlap: **C1** (a recipe–format interaction is documented for
LLMs). Differs: LLM; no hardware.

**[Cim26] M. Cim, B. Topcu, M. T. Kandemir. "Diagnosing FP4 inference: a
layer-wise and block-wise sensitivity analysis of NVFP4 and MXFP4."
arXiv:2603.08747 (2026).** VERIFIED (arXiv API). Layer and block sensitivity
for Qwen2.5. Peripheral; relevant to per-layer selection (C5) only in a
diagnostic sense.

### 1.2 Shared-exponent, microscaling and block floating point

**[Rou23a] B. Rouhani, R. Zhao, V. Elango, R. Shafipour, M. Hall,
M. Mesmakhosroshahi, A. More, L. Melnick, M. Golub, G. Varatkar, L. Shao,
G. Kolhe, D. Melts, J. Klar, R. L'Heureux, M. Perry, D. Burger, E. Chung,
Z. Deng, S. Naghshineh, J. Park, M. Naumov. "With Shared Microexponents, A
Little Shifting Goes a Long Way." arXiv:2302.08007; ISCA 2023.** arXiv VERIFIED
(abstract page). ISCA 2023 venue and DOI 10.1145/3579371.3589351: VERIFIED
`[check]` (Crossref: Proceedings of the 50th Annual International Symposium on
Computer Architecture, pp. 1–13). Block Data Representations (BDR) framework;
MX formats with multi-level, fine-grained scaling. The abstract reports that MX
outperforms narrow FP and BFP; evaluated on generative pretraining, inference
and recommendation. The hardware evidence level is not stated in the abstract.
Overlap: C2 (format design space with a cost lens); C5 (format family
selection). Differs: no per-layer accumulator certificate; not edge CNN PTQ.

**[Rou23b] B. D. Rouhani et al. (33 authors, including R. Zhao, A. More,
M. Hall, … P. Micikevicius, … D. Burger, E. Chung). "Microscaling Data Formats
for Deep Learning." arXiv:2310.10537 (2023).** VERIFIED (arXiv abstract page;
the full author list is on that page). Per-block scaling with narrow FP and INT
elements; more than two dozen benchmarks; inference and training, including
sub-8-bit training of generative models. Hardware: not stated in the abstract.
Overlap: anchor format family (MXFP8/6/4, MXINT8) the project already uses.

**[OCP-MX] Open Compute Project. "OCP Microscaling Formats (MX) Specification
v1.0."** UNVERIFIED: the official PDF URL
(opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
returned HTTP 403. The project must cite the exact version it implements.
[Isl26] (below) states in its abstract that "the bit-wise numerical behavior of
OCP MX formats is not documented by GPU vendors, and the precision and
rounding are not prescribed by the OCP specification itself" `[check]`: the
earlier quotation was cut short and read as if the specification itself were
undocumented. The statement bears directly on C4, but it is a claim by
[Isl26] about the specification and was not checked against the
specification text.

**[Son18] Z. Song, Z. Liu, D. Wang. "Computation Error Analysis of Block
Floating Point Arithmetic Oriented Convolution Neural Network Accelerator
Design." arXiv:1709.07776; AAAI 2018 (arXiv comment).** VERIFIED (arXiv API).
BFP CNN accelerator error analysis on VGG16, ResNet-18, ResNet-50 and
GoogLeNet. Headline: an 8-bit mantissa costs under 0.3% accuracy; derives
noise-to-signal bounds. Overlap: C3 (analytical error bounds for a shared-
exponent format). Differs: 8-bit mantissa; no lossless accumulator
certificate; no ASIC cost per the abstract.

**[Fox21] S. Fox, S. Rasoulinezhad, J. Faraone, D. Boland, P. Leong. "A Block
Minifloat Representation for Training Deep Neural Networks." ICLR 2021.**
The title is listed among the ICLR 2021 papers on
iclr.cc/virtual/2021/papers.html `[check]`. Authors and abstract are still
only from mlanthology.org (secondary index), because OpenReview (id
6zaTwpNSsQ2) is blocked: UNVERIFIED. 4–8-bit block
minifloats with a learnable shared exponent bias, for training. Headline as
relayed: 6-bit BM ResNet on ImageNet with near-FP accuracy, and FMA units about
4.1× smaller and using 2.3× less energy than FP8. Hardware level: UNVERIFIED.
Overlap: C2 (format choice that yields "integer-like" FMA hardware).

**[Gil25] M. Gil, D. Ha, S. B. Harma, M. K. Yoon, B. Falsafi, W. W. Ro, Y. Oh.
"Avant-Garde: Empowering GPUs with Scaled Numeric Formats." ISCA 2025.**
VERIFIED: title, authors and session (1C) on the official ISCA 2025 programme
page. DOI 10.1145/3695053.3731100, pp. 153–165: VERIFIED `[check]` (Crossref).
Abstract content: secondary only (the Semantic Scholar copy of the abstract;
the ACM page returns 403). That copy describes hardware that flattens multi-level scaled formats into a
single-level internal format for a modified Tensor Core, with "up to 74% higher throughput and 44% lower execution
time". Overlap: C2 (scaling support cost handled in hardware). Differs:
GPU and large models.

**[Ram25] A. Ramachandran, S. Kundu, T. Krishna. "MicroScopiQ: Accelerating
Foundational Models through Outlier-Aware Microscaling Quantization."
arXiv:2411.05282; ISCA 2025 (arXiv comment, also listed on the ISCA 2025
programme page).** VERIFIED. Outlier-aware MX quantisation combined with
pruning; multi-precision INT PEs. The abstract reports up to 3× faster
inference and 2× lower energy. Peripheral (foundation models).

**[Hu26] W. Hu, Z. Zhang, H. Zhang, C. Zhang, C. Guo, Y. Feng, T. Hu, G. Li,
G. Hu, J. Wang, J. Leng. "M2XFP: A Metadata-Augmented Microscaling Data Format
for Efficient Low-bit Quantization." arXiv:2601.19213; ASPLOS 2026 (arXiv
comment).** VERIFIED (arXiv API). Metadata added to MXFP4; LLMs. The abstract
reports 70.63% less accuracy loss than MXFP4 and up to 1.91× speedup. Overlap:
C2 (metadata as part of the cost). LLM scope.

**[Cuy26] S. Cuyckens, X. Yi, R. Geens, J. Dumoulin, M. Wiesner, C. Fang,
M. Verhelst. "Precision-Scalable Microscaling Datapaths with Optimized
Reduction Tree for Efficient NPU Integration." arXiv:2511.06313; ASP-DAC 2026,
invited (arXiv comment).** VERIFIED (arXiv API). MX MAC datapaths in the SNAX
NPU. The abstract reports 657 GOPS/W (MXINT8), 1,438–1,675 GOPS/W (MXFP8/6)
and 4,065 GOPS/W (MXFP4). The implementation level is not stated in the
abstract. DOI 10.1109/ASP-DAC66049.2026.11420756, pp. 611–617 (Crossref
`[check]`). The abstract also says that in existing MX MACs "integer
accumulation requires expensive conversions from narrow floating-point
products, while FP32 accumulation suffers from quantization losses and costly
normalization", which bears on C3 and C4 `[check]`. Overlap: C2 (MX datapath cost including the reduction tree);
a natural hardware comparison point for the project's MX/BFP arms.

**[Che26] C.-T. Chen, D. Han, H. Mun, J. Hyun, A. Raha, A. Agarwal, M. Anders,
M. Abdelfattah, J.-s. Seo. "HBQ: Hierarchical Scaling Block Quantization with
Hardware-Efficiency-Aware Design for Accurate LLM Inference." arXiv:2609.00450;
MICRO 2026 (arXiv comment).** VERIFIED (arXiv API). Hierarchical block scaling
with a 28 nm ASIC accelerator. The abstract reports 2.3×/4.6× higher
area/energy efficiency than weight-only quantisation. Overlap: C2/C5 (scale
hierarchy chosen jointly with hardware). LLM scope. Shows that MICRO in 2026
still accepts scaled-format hardware papers.

### 1.3 Adaptive and custom float formats with hardware; the MANT name

**[Tam20] T. Tambe, E.-Y. Yang, Z. Wan, Y. Deng, V. J. Reddi, A. Rush,
D. Brooks, G.-Y. Wei. "AdaptivFloat: A Floating-point based Data Type for
Resilient Deep Learning Inference." arXiv:1909.13271.** arXiv VERIFIED. The DAC
2020 venue is VERIFIED `[check]` (Crossref): the same eight authors published
it as "Algorithm-Hardware Co-Design of Adaptive Floating-Point Encodings for
Resilient Deep Learning Inference", 57th ACM/IEEE DAC, 2020, DOI
10.1109/DAC18072.2020.9218516. Cite the DAC title.
Per-layer range-adaptive float; compared against BFP, uniform, IEEE-like float
and posit at ≤8 bits on sequence-transduction models. Hardware per the
abstract: 0.9× the energy efficiency and 1.14× the area of equivalent integer
accelerators (evidence level not stated). Overlap: C5 (per-layer range
adaptation); C2 (an FP-versus-INT hardware ratio).

**[Guo22] C. Guo, C. Zhang, J. Leng, Z. Liu, F. Yang, Y. Liu, M. Guo, Y. Zhu.
"ANT: Exploiting Adaptive Numerical Data Type for Low-bit Deep Neural Network
Quantization." arXiv:2208.14286; MICRO 2022 (arXiv comment).** VERIFIED (arXiv
abstract). Fixed-length adaptive type with a "flint" float/int hybrid and
per-tensor type selection from the distribution. The abstract reports 2.8×
speedup and 2.5× energy-efficiency improvement over quantisation accelerators.
Overlap: **C5** (per-tensor type selection with hardware). Differs: selection
by distribution, with no accumulator certificate or complete scale/memory
accounting per the abstract.

**[Hu25] W. Hu, H. Zhang, C. Guo, Y. Feng, R. Guan, Z. Hua, Z. Liu, Y. Guan,
M. Guo, J. Leng. "M-ANT: Efficient Low-bit Group Quantization for LLMs via
Mathematically Adaptive Numerical Type." arXiv:2502.18755 (cs.AR); HPCA 2025, pp. 1112–1126,
DOI 10.1109/HPCA61900.2025.00086.**
VERIFIED `[check]`, with one correction to the title. Three sources were
opened: (i) the arXiv record, titled "M-ANT: …"; (ii) the Crossref record of
the IEEE proceedings paper, also titled "M-ANT: …", with the same ten authors;
(iii) the official programme page hpca-conf.org/2025/main-program/, which
lists it in Session 8C, "Viva Las Learning Models – 1", under the title
**"MANT: Efficient Low-bit Group Quantization for LLMs via Mathematically
Adaptive Numerical Type"** with Shanghai Jiao Tong University affiliations.
So the programme prints "MANT" and the published title is "M-ANT". The
abstract itself says "we propose MANT, a mathematically adaptive numeric
type". Cite it as "M-ANT" with the DOI above.
Proposal: a group-wise adaptive numeric type for 4-bit weights and KV cache in
LLMs, with real-time KV-cache quantisation and a custom PE in a systolic array.
The abstract reports 2.99× (up to 4.46×) speedup and 2.81× (up to 4.10×) energy
reduction over LLM accelerators. **Name collision: confirmed** `[check]`. The paper names
its data type "MANT" in the abstract and the conference programme prints the
title with "MANT"; only the proceedings and arXiv titles add the hyphen. The
token is the same as the name this repository's README uses for the private
downstream architecture. Consequences:
(i) the public paper must cite M-ANT/MANT [Hu25] as related work on adaptive
types, and should not use the token "MANT" for anything of its own;
(ii) any later disclosure of the private architecture under the name "MANT"
will collide in search and citation indices with an HPCA paper on a closely
related topic (adaptive numeric types with accelerator hardware). Renaming the
private architecture, or always qualifying it, is advisable. This is a
recommendation, not a decision for this lane.

**[Ram24] A. Ramachandran, Z. Wan, G. Jeong, J. Gustafson, T. Krishna.
"Algorithm-Hardware Co-Design of Distribution-Aware Logarithmic-Posit Encodings
for Efficient DNN Inference." arXiv:2403.05465; DAC 2024 (arXiv comment; DOI
10.1145/3649329.3656544 from Crossref `[check]`).**
VERIFIED (arXiv API). Logarithmic posit (LP) type with genetic-algorithm
parameter search (LPQ) for CNNs and ViTs; under 1% accuracy drop. The abstract
reports 2× performance per area and 2.2× energy efficiency over quantisation
accelerators. Overlap: C5 (per-layer format parameter search plus hardware);
the posit/LNS families.

Other HPCA 2025 programme entries seen on the official programme page (titles
only; abstracts not opened): "LUT-DLA: Lookup Table as Efficient Extreme
Low-Bit Deep Learning Accelerator"; "Panacea: Novel DNN Accelerator using
Accuracy-Preserving Asymmetric Quantization and Energy-Saving Bit-Slice
Sparsity"; "BitMoD: Bit-serial Mixture-of-Datatype LLM Acceleration"; "FIGLUT:
An Energy-Efficient Accelerator Design for FP-INT GEMM Using Look-Up Tables".
The ISCA 2025 programme also lists "LUT Tensor Core: A Software-Hardware
Co-Design for LUT-Based Low-Bit LLM Inference". These show that the
MICRO/HPCA/ISCA bar for format papers is now a full accelerator with
LLM-scale workloads.

### 1.4 Posit inference and posit-versus-float hardware cost

**[Car19a] Z. Carmichael, H. F. Langroudi, C. Khazanov, J. Lillie,
J. L. Gustafson, D. Kudithipudi. "Deep Positron: A Deep Neural Network Using
the Posit Number System." arXiv:1812.01762; DATE 2019 (arXiv comment).**
VERIFIED (arXiv API). ≤8-bit posit inference; FPGA soft core for exact MAC
(Xilinx Virtex-7). The abstract reports that 8-bit posits outperform 8-bit
fixed and float. Overlap: C3 (an exact MAC for a non-integer format); C2 (FPGA
cost).

**[Car19b] Same authors. "Performance-Efficiency Trade-off of Low-Precision
Numerical Formats in Deep Neural Networks." arXiv:1903.10584; CoNGA 2019, DOI
10.1145/3316279.3316282 (arXiv comment and DOI fields; Crossref `[check]`).** VERIFIED (arXiv API). Fixed,
float and posit at ≤8 bits with exact MAC units on five classification tasks;
posits competitive in resources. Overlap: C2/C3 (exact-MAC cost comparison
across families).

**[Lan19] H. F. Langroudi, Z. Carmichael, D. Pastuch, D. Kudithipudi.
"Cheetah: Mixed Low-Precision Hardware & Software Co-Design Framework for DNNs
on the Edge." arXiv:1908.02386 (2019).** VERIFIED (arXiv API). Posit, float
and fixed at 5–8 bits for inference; MNIST, Fashion-MNIST and CIFAR-10.
Overlap: C2 (cost versus quality across families), but on small datasets.

**[Mal22] D. Mallasén, R. Murillo, A. A. Del Barrio, G. Botella, L. Piñuel,
M. Prieto. "PERCIVAL: Open-Source Posit RISC-V Core with Quire Capability."
arXiv:2111.15286; IEEE TETC 10(3):1241–1252, 2022, DOI
10.1109/TETC.2022.3187199 (Crossref `[check]`).** VERIFIED (arXiv API). A 32-bit
posit core with a quire; the abstract reports about 4 orders of magnitude
better matrix-multiply accuracy than FP32. The abstract also says that FPGA
and ASIC synthesis "highlight the significant overhead of including a quire
accumulator" `[check]`. Overlap: C3 (quire exact accumulation cost) and C2.

**[Mal26] D. Mallasén, P. D. Schiavone, A. A. Del Barrio, M. Prieto-Matias,
D. Atienza. "Increasing the Energy-Efficiency of Wearables Using Low-Precision
Posit Arithmetic with PHEE." arXiv:2501.18253; IEEE TCASAI 3(2):142–151, 2026
(arXiv journal-ref).** VERIFIED (arXiv API). 8–16-bit posits; post-synthesis
TSMC 16 nm. The abstract reports that the posit hardware "can be 38% smaller and
consume up to 42.3% less power at the functional unit level". That sentence
does not name the comparator; the context is narrow posits replacing wider
floats `[check]`. Overlap: C2 (synthesis-level
posit-versus-float cost). Not CNN-scale.

**[Jon25] A. A. Jonnalagadda, R. Thotli, J. L. Gustafson. "Closing the Gap
Between Float and Posit Hardware Efficiency." arXiv:2603.01615; CoNGA 2025
(arXiv comment).** VERIFIED (arXiv API). Bounded posits (b-posits). The
abstract reports 32-bit b-posit decoders with 79% less power and 71% less area
than standard posit decoders. Overlap: C2, decode cost as part of the complete
arithmetic cost (32-bit, not sub-8-bit).

### 1.5 Logarithmic number systems and additive powers of two

**[Miy16] D. Miyashita, E. H. Lee, B. Murmann. "Convolutional Neural Networks
using Logarithmic Data Representation." arXiv:1603.01025 (2016).** VERIFIED
(arXiv API). Base-2 log weights and activations; 3-bit with negligible loss;
multiplier-free. Hardware: none stated. Overlap: format family only.

**[Zha21] J. Zhao, S. Dai, R. Venkatesan, B. Zimmer, M. Ali, M.-Y. Liu,
B. Khailany, B. Dally, A. Anandkumar. "LNS-Madam: Low-Precision Training in
Logarithmic Number System using Multiplicative Weight Update."
arXiv:2106.13914; IEEE Trans. Computers 71(12):3179–3190, 2022 (Crossref
`[check]`).** VERIFIED (arXiv API). LNS training; the abstract reports
over 90% energy reduction against FP32 and 55% against FP8 from an LNS-to-
integer conversion datapath (evidence level not stated). Overlap: C2 (the
conversion datapath is part of the cost). Training, not PTQ.

**[Li20] Y. Li, X. Dong, W. Wang. "Additive Powers-of-Two Quantization: An
Efficient Non-uniform Discretization for Neural Networks." arXiv:1909.13144.**
VERIFIED (arXiv API). ICLR 2020 venue: VERIFIED `[check]` (title listed on
iclr.cc/virtual/2020/papers.html). APoT levels; QAT. The
abstract reports 76.6% top-1 for 4-bit ResNet-50 and 22% lower compute cost
than uniform quantisation. Overlap: format family (non-uniform with shift-add
hardware).

**[Joh18] J. Johnson. "Rethinking floating point for deep learning."
arXiv:1811.01721 (2018).** VERIFIED (arXiv API). 8-bit log float with log
multiply, linear add and **Kulisch accumulation**, plus posit-style tapering;
no retraining. The abstract reports results within 0.9% top-1 of FP32 on
ResNet-50, and **28 nm synthesis** at 0.96× power and 1.12× area of an
8/32-bit integer multiply-add. Overlap: **C3** (exact accumulation for a
non-integer 8-bit format, with synthesis cost); C2.

### 1.6 Codebook and non-uniform quantisation (NF4 and predecessors)

**[Han16] S. Han, H. Mao, W. J. Dally. "Deep Compression: Compressing Deep
Neural Networks with Pruning, Trained Quantization and Huffman Coding."
arXiv:1510.00149; ICLR 2016 oral (arXiv comment).** VERIFIED (arXiv API).
Trained weight sharing (codebook); 35–49× compression on AlexNet and VGG-16.
The earliest codebook anchor.

**[Zha18] D. Zhang, J. Yang, D. Ye, G. Hua. "LQ-Nets: Learned Quantization for
Highly Accurate and Compact Deep Neural Networks." arXiv:1807.10029; ECCV 2018
(arXiv comment).** VERIFIED (arXiv API). Learned non-uniform quantisers for
weights and activations; QAT.

**[Det23] T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer. "QLoRA:
Efficient Finetuning of Quantized LLMs." arXiv:2305.14314.** VERIFIED (arXiv
API; the arXiv comment says "Extended NeurIPS submission"). NeurIPS 2023
acceptance: VERIFIED `[check]` (Main Conference Track listing on
proceedings.neurips.cc/paper_files/paper/2023). Introduces 4-bit NormalFloat (NF4) and double
quantisation. Overlap: the source of the project's NF4 arm. NF4 was designed
for normally distributed weights (abstract). That it is used weight-only with
BF16 compute is not in the abstract: UNVERIFIED here `[check]`. The
project's NF4 weight-and-activation CNN arm is therefore an out-of-design use,
and the paper should say so.

### 1.7 Accumulator-aware and overflow-aware quantisation

**[Ni20] R. Ni, H.-m. Chu, O. Castañeda, P.-y. Chiang, C. Studer, T. Goldstein.
"WrapNet: Neural Net Inference with Ultra-Low-Resolution Arithmetic."
arXiv:2007.13242.** VERIFIED (arXiv API). The ICLR 2021 venue is VERIFIED `[check]`: iclr.cc/virtual/2021/papers.html
lists it as "WrapNet: Neural Net Inference with Ultra-Low-Precision
Arithmetic" (the arXiv title says "Ultra-Low-Resolution"). 8-bit accumulation
with a cyclic activation layer and an overflow penalty; training-based. The
abstract says only that "we demonstrate the efficacy of our approach on both
software and hardware platforms"; the word "validated" used earlier is not in
the abstract `[check]`. The abstract also states that high-resolution
accumulation "dominates the arithmetic complexity of inference when using
extreme quantization", which is the C2 premise for integers. Overlap: C3
from the opposite direction (tolerate overflow instead of certifying its
absence).

**[deB20] B. de Bruin, Z. Zivkovic, H. Corporaal. "Quantization of Deep Neural
Networks for Accumulator-constrained Processors." arXiv:2004.11783;
Microprocessors and Microsystems 72:102872, 2020 (arXiv journal-ref).**
VERIFIED (arXiv API). Bit-width choice constrained by a 16-bit accumulator;
within 1% of FP on CIFAR-10 and ILSVRC2012; about 2× speedup on ARM. Overlap:
C3/C5 (accumulator-constrained precision selection) for integers.

**[Xie20] H. Xie, S. Zhang, H. Ding, Y. Song, B. Shao, C. Hu, L. Cai, M. Li.
"Accelerating Neural Network Inference by Overflow Aware Quantization."
arXiv:2005.13297.** VERIFIED (arXiv API). The IJCAI 2020 venue is VERIFIED `[check]` (Crossref,
DOI 10.24963/ijcai.2020/121, pp. 868–875), but the published record has a
different title, "Overflow Aware Quantization: Accelerating Neural Network
Inference by Low-bit Multiply-Accumulate Operations", and four authors
(H. Xie, Y. Song, L. Cai, M. Li). Cite the IJCAI record. Trainable fixed-point
representation that avoids accumulation overflow; about 2× faster inference.
Integer only.

**[Sak19] C. Sakr, N. Wang, C.-Y. Chen, J. Choi, A. Agrawal, N. Shanbhag,
K. Gopalakrishnan. "Accumulation Bit-Width Scaling For Ultra-Low Precision
Training Of Deep Networks." arXiv:1901.06588; ICLR 2019 (arXiv comment).**
VERIFIED (arXiv API). Statistical (variance-based) prediction of the minimum
accumulator precision from accumulation length; training. Overlap: **C3**
(analytical accumulator precision as a function of accumulation length; the
abstract does not say that the accumulator is floating-point, and the full
text was not opened `[check]`). Differs:
statistical rather than certified; training; not per-layer exact.

**[Wan18] N. Wang, J. Choi, D. Brand, C.-Y. Chen, K. Gopalakrishnan. "Training
Deep Neural Networks with 8-bit Floating Point Numbers." arXiv:1812.08011;
NeurIPS 2018 (arXiv comment).** VERIFIED (arXiv API). Chunk-based accumulation
and stochastic rounding reduce accumulation from 32 to 16 bits. Training.
Overlap: C3 (FP16 accumulation as a design point). Relevant to the planned
FP16 accumulator arm.

**[Col23] I. Colbert, A. Pappalardo, J. Petri-Koenig. "A2Q:
Accumulator-Aware Quantization with Guaranteed Overflow Avoidance."
arXiv:2308.13504.** VERIFIED (arXiv abstract). The ICCV 2023 venue is
VERIFIED `[check]` (Crossref, DOI 10.1109/ICCV51070.2023.01558,
pp. 16943–16952). QAT; constrains
weight L1 norms to an accumulator bit width so that overflow is impossible;
FPGA-focused. The abstract reports up to 2.3× lower resource use than 32-bit
accumulators at 99.2% of FP accuracy. The arXiv page notes substantial overlap
with arXiv:2301.13376. Overlap: **C3** (guaranteed overflow-free accumulator
widths) for integers. Differs: integer only; QAT; it changes the weights to fit
the accumulator, whereas the project certifies the accumulator for fixed PTQ
weights.

**[Col24a] I. Colbert, A. Pappalardo, J. Petri-Koenig, Y. Umuroglu. "A2Q+:
Improving Accumulator-Aware Weight Quantization." ICML 2024, PMLR
235:9275–9291; arXiv:2401.10432.** VERIFIED (PMLR page and arXiv abstract). A
tighter overflow bound plus initialisation from pretrained checkpoints; QAT.
Overlap: C3 (integer).

**[Col24b] I. Colbert, G. Franco, F. Grob, J. Zhang, R. Saab.
"Accumulator-Aware Post-Training Quantization for Large Language Models"
(v1 title "Accumulator-Aware Post-Training Quantization").
arXiv:2409.17092.** VERIFIED (arXiv abstract). AXE: overflow-avoidance
guarantees added to GPFQ and OPTQ, i.e. **PTQ**; supports multi-stage
accumulation. The v1 abstract evaluates "image classification and language
generation models"; the v2 abstract reports only language generation
(Llama3 8B) `[check]`. The v2 abstract reports up to 98% of FP16 perplexity at 16-bit multi-stage
accumulation. Overlap: **C3/C5 for integers under PTQ**: the closest
accumulator-aware PTQ work. Differs: integer formats; it changes rounding to
meet a width. The project instead certifies widths for minifloats, BFP and
other non-integer formats, and joins them to MAC cost. The v2 abstract opens
with the premise that at narrow precisions "the cost of additions begins to
dominate that of multiplications in multiply-accumulate (MAC) units"
`[check]`. C2's starting point is therefore stated in prior work.

**[Blu24] Y. Blumenfeld, I. Hubara, D. Soudry. "Towards Cheaper Inference in
Deep Networks with Lower Bit-Width Accumulators." arXiv:2401.14110.** VERIFIED
(arXiv API). The ICLR 2024 venue is VERIFIED `[check]` (listed on
proceedings.iclr.cc/paper_files/paper/2024). The abstract reports training and
fine-tuning DNNs to use 12-bit accumulators "with no significant degradation in
accuracy". The phrase "building on FP8" used earlier is not in the abstract
`[check]`; the full text (arXiv HTML) defines an M/E floating-point
quantisation and a quantised FMA, so the accumulators are low-bit
floating-point. Overlap: **C3** (low-bit floating-point accumulators). Differs:
fine-tuning, not certified PTQ.

**[ElA25] E.-M. El Arar, S.-I. Filip, T. Mary, E. Riccietti. "Mixed precision
accumulation for neural network inference guided by componentwise forward
error analysis." arXiv:2503.15568.** VERIFIED (arXiv API). Error bound
proportional to the inner-product condition number; recompute high-condition
components in higher precision. Overlap: **C3/C5** (a rigorous error analysis
driving accumulator precision per component). Differs: a forward-error bound
rather than a lossless width; no hardware per the abstract.

**[Umu25] Y. Umuroglu, C. Berganski, F. Jentzsch, M. Danilowicz, T. Kryjak,
C. Bezaitis, M. Sjalander, I. Colbert, T. Preusser, J. Petri-Koenig, M. Blott.
"SIRA: Scaled-Integer Range Analysis for Optimizing FPGA Dataflow Neural
Network Accelerators." arXiv:2508.21493 (2025; submitted to ACM TRETS per the
arXiv page).** VERIFIED (arXiv abstract). **Static interval arithmetic** for
range, scale and bias of quantised tensors in FINN. The abstract reports 22%
smaller accumulator widths, 17% fewer LUTs and 66% fewer DSPs. Overlap: **C3**
(statically derived per-layer accumulator widths), but for scaled integers on
FPGA. The closest prior method to the project's certificate. `[check]` The
full text (arXiv HTML v1) contrasts the "datatype bound" of [Col23] with its
own analysis, which "exploits the case where weights are constant during
inference and yields smaller accumulators than those calculated via the
datatype bound". Deriving the width from the actual constant weights is thus
prior art for scaled integers. The words "minifloat" and "floating-point
format" do not occur in the text, so non-integer formats are not covered.

### 1.8 Exact (Kulisch) accumulation and the quire

Covered above: [Joh18] (Kulisch accumulation, 28 nm synthesis), [Car19a/b]
(exact MAC), [Mal22] (quire), [Dam24] (accurate minifloat MACCs) and [Agg24]
(closed-form minifloat accumulator widths).

**[Des23] O. Desrentes, B. Dupont de Dinechin, J. Le Maire. "Exact Dot Product
Accumulate Operators for 8-bit Floating-Point Deep Learning." Euromicro DSD
2023, pp. 642–649.** VERIFIED `[check]`: Crossref (DOI 10.1109/DSD60849.2023.00093,
pp. 642–649, same three authors) and the authors' HAL deposit hal-04240816,
which carries the abstract. Per that abstract: an architecture for exact dot
product accumulate operators, compared across E5M2, E4M3 and Posit8 formats
with different exponent sizes. The front-ends "expand their full-precision
products to fixed-point, and sum terms into wide accumulators"; the back-ends
round "first to FP32 and then to one of the 8-bit floating-point formats". The
operators are synthesised "targeting the TSMC 16FFC node" and their "area and
power" are compared "to a baseline of operators with FP16 and INT8
multiplicands". Evidence level: **synthesis (ASIC)**. Overlap: **C2 and C3 at
8 bits**: an ASIC comparison of complete exact dot-product operators for FP8
and posit against INT8 already exists. Differs: 8-bit only; operator level, no
network quality; no per-layer width.

### 1.9 PTQ baselines a reviewer will expect

All VERIFIED via the arXiv API unless noted.

- **[Nag19] M. Nagel, M. van Baalen, T. Blankevoort, M. Welling. "Data-Free
  Quantization Through Weight Equalization and Bias Correction."
  arXiv:1906.04721; ICCV 2019 (arXiv journal-ref).** Data-free 8-bit PTQ with
  equalisation and bias correction; state of the art on MobileNet. Directly
  relevant to the project's MobileNet INT8 maxabs collapse (4.10% in
  [B-stage balanced](b-stage-balanced-results-2026-09-27.md)). Reviewers will
  read that collapse as a known per-tensor-range failure that equalisation
  addresses.
- **[Nag20] M. Nagel, R. A. Amjad, M. van Baalen, C. Louizos, T. Blankevoort.
  "Up or Down? Adaptive Rounding for Post-Training Quantization." ICML 2020,
  PMLR 119:7197–7206; arXiv:2004.10568.** VERIFIED (PMLR page). AdaRound;
  ResNet18 and ResNet50 at 4-bit weights within 1% loss.
- **[Li21] Y. Li, R. Gong, X. Tan, Y. Yang, P. Hu, Q. Zhang, F. Yu, W. Wang,
  S. Gu. "BRECQ: Pushing the Limit of Post-Training Quantization by Block
  Reconstruction." arXiv:2102.05426.** The ICLR 2021 venue is VERIFIED `[check]` (title
  listed on iclr.cc/virtual/2021/papers.html). 4-bit
  ResNet and MobileNetV2 comparable to QAT; the first INT2 PTQ.
- **[Wei22] X. Wei, R. Gong, Y. Li, X. Liu, F. Yu. "QDrop: Randomly Dropping
  Quantization for Extremely Low-bit Post-Training Quantization."
  arXiv:2203.05740; ICLR 2022 (arXiv comment).** 2-bit activations for the
  first time; up to 51.49% improvement.
- **[Nag21] M. Nagel, M. Fournarakis, R. A. Amjad, Y. Bondarenko, M. van Baalen,
  T. Blankevoort. "A White Paper on Neural Network Quantization."
  arXiv:2106.08295.** PTQ and QAT pipelines; standard reference for
  per-channel ranges and batch-norm folding.
- **[Kri18] R. Krishnamoorthi. "Quantizing deep convolutional networks for
  efficient inference: A whitepaper." arXiv:1806.08342.** Per-channel weights
  and per-layer activations at 8 bits within 2% of FP; 4-bit weights with QAT
  at 2–10% loss.
- **[Jac18] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam,
  D. Kalenichenko. "Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference." arXiv:1712.05877.** The CVPR 2018 venue is
  VERIFIED `[check]` (Crossref, DOI 10.1109/CVPR.2018.00286, pp. 2704–2713). The integer-only inference reference,
  with fixed-point requantisation; relevant to the scale/requant part of C2.
- **[Wu20] H. Wu, P. Judd, X. Zhang, M. Isaev, P. Micikevicius. "Integer
  Quantization for Deep Learning Inference: Principles and Empirical
  Evaluation." arXiv:2004.09602.** 8-bit within 1% of FP, including
  MobileNets and BERT-large. A standard calibration baseline (max, percentile,
  entropy, MSE).

### 1.10 Mixed-precision and format search

- **[Wan19] K. Wang, Z. Liu, Y. Lin, J. Lin, S. Han. "HAQ: Hardware-Aware
  Automated Quantization with Mixed Precision." arXiv:1811.08886; CVPR 2019
  (arXiv comment).** VERIFIED (arXiv API). RL search over per-layer bit widths
  with hardware-simulator feedback; 1.4–1.95× latency reduction. Overlap: C5
  (hardware-in-the-loop per-layer selection), bit width only.
- **[Don19] Z. Dong, Z. Yao, A. Gholami, M. Mahoney, K. Keutzer. "HAWQ: Hessian
  AWare Quantization of Neural Networks with Mixed-Precision."
  arXiv:1905.03696; ICCV 2019 (arXiv comment).** VERIFIED (arXiv API).
  Hessian-guided per-layer precision. Overlap: C5 (bit width only).
- **[Yao21] Z. Yao, Z. Dong, Z. Zheng, A. Gholami, J. Yu, E. Tan, L. Wang,
  Q. Huang, Y. Wang, M. W. Mahoney, K. Keutzer. "HAWQ-V3: Dyadic Neural Network
  Quantization." arXiv:2011.10680; ICML 2021 (arXiv journal-ref as relayed).**
  VERIFIED (arXiv API). Integer-only inference with dyadic requantisation;
  INT4/INT8 mixed precision; deployment on T4 via TVM. Overlap: C2 (the cost of
  requantisation is designed in) and C4 (actually deployed integer inference
  rather than simulation).
- **[Dot24] J. Dotzel, G. Wu, A. Li, M. Umar, Y. Ni, M. S. Abdelfattah,
  Z. Zhang, L. Cheng, M. G. Dixon, N. P. Jouppi, Q. V. Le, S. Li. "FLIQS:
  One-Shot Mixed-Precision Floating-Point and Integer Quantization Search."
  arXiv:2308.03290; AutoML 2024 (arXiv comment).** VERIFIED (arXiv API). `[check]` The
  earlier description ("joint per-layer INT/FP search") went beyond the
  abstract. The abstract describes a one-shot mixed-precision search "in both
  integer and low-precision floating point models": integer results on
  ResNet-18 and ResNet-50, and a separate "mixed-precision floating-point
  search" that improves MobileNetV2 over FP8 models. It does not say that
  integer and floating-point layers are mixed within one model, and the search
  runs during training, so it is not PTQ. Overlap: **C5** (per-layer precision
  search for both families on the same CNNs). Differs: no accumulator
  certificate and no complete hardware cost per the abstract.
- **[Rus19] M. Rusci, A. Capotondi, L. Benini. "Memory-Driven Mixed Low
  Precision Quantization For Enabling Deep Network Inference On
  Microcontrollers." arXiv:1905.13082.** VERIFIED (arXiv API). Memory-
  constrained per-tensor 8/4/2-bit with integer-only deployment on an STM32H7;
  MobileNetV1 at 68% top-1. Overlap: C2 (memory as a cost) and C4 (deployed
  integer inference).
- **[Del25] K. Dellel, E. Trabes, A. Zayed, H. Faiedh, C. Valderrama.
  "Differentiable Selection of Bit-Width and Numeric Format for FPGA-Efficient
  Deep Networks." *Electronics* 14(18):3715, 2025, DOI
  10.3390/electronics14183715.** Bibliographic fields VERIFIED `[check]`
  (Crossref). Content UNVERIFIED (MDPI returns 403; the abstract was seen only
  in a search snippet). It reportedly learns fixed-versus-float per layer with
  QAT. Must be opened before it is cited.

### 1.11 Hardware-cost-aware format design-space exploration with RTL

- **[Pra25] K. Prabhu, J. Yu, X. A. Pan, Z. Xie, A. Aleshire, Z. Chen,
  A. A. Ratnani, P. Raina. "Voyager: An End-to-End Framework for Design-Space
  Exploration and Generation of DNN Accelerators." arXiv:2509.15205.**
  VERIFIED (arXiv API). HLS-generated accelerators supporting float, posit,
  integer and microscaling, with RTL generation per node and frequency. The
  full text (arXiv HTML v1) is now VERIFIED `[check]`: its accuracy table
  compares FP32, BF16, E4M3, Posit8, INT8 and MXINT8 on vision and language
  models including ResNet-18 on ImageNet, and it states that "microscaled
  integers achieve inference accuracy that is within 1% of 32-bit floating
  point, but with up to 23% lower area than low-precision floating point
  formats". Overlap: **C2** (multi-format RTL cost
  plus accuracy). Differs: 8-bit-centric; HLS; no accumulator certificate per
  the abstract. **A reviewer will ask why this framework was not used.**
- **[Inc22] A. Inci, S. G. Virupaksha, A. Jain, T.-W. Chin, V. V. Thallam,
  R. Ding, D. Marculescu. "QUIDAM: A Framework for Quantization-Aware DNN
  Accelerator and Model Co-Exploration." arXiv:2206.15463.** VERIFIED (arXiv
  API). Bit precision and PE-type co-exploration; performance per area varies
  more than 5× and energy more than 35× across designs. Overlap: C2/C5.
- **[Als23] G. Alsuhli, V. Sakellariou, H. Saleh, M. Al-Qutayri, B. Mohammad,
  T. Stouraitis. "Number Systems for Deep Neural Network Architectures: A
  Survey." arXiv:2307.05035.** VERIFIED (arXiv API). A survey to cite for the
  breadth of families.
- Also in this theme: [Agg24], [Dam24], [Joh18], [Tam20], [Fox21], [Cuy26],
  [Mal26] and [vB23].

### 1.12 Simulation-versus-deployment fidelity

- **[Zha19] T. Zhang, Z. Lin, G. Yang, C. De Sa. "QPyTorch: A Low-Precision
  Arithmetic Simulation Framework." arXiv:1910.04540; NeurIPS 2019 EMC²
  workshop (arXiv comment).** VERIFIED (arXiv API). A quantise-dequantise
  simulator for float, fixed and BFP. It does not claim bit-exactness for
  accumulation per the abstract. The sort of simulator C4 audits.
- **[Isl26] M. Islam, M. Mikaitis. "Simulation of Custom-Precision OCP MX Block
  Floating-Point Formats and Arithmetic." arXiv:2607.12915 (2026).** VERIFIED
  (arXiv API). MXsim (MATLAB, built on CPFloat), with configurable accumulator
  precision. States that "the bit-wise numerical behavior of OCP MX formats is not
  documented by GPU vendors, and the precision and rounding are not prescribed
  by the OCP specification itself" `[check]` (full quotation; the earlier one
  was cut short). Overlap: **C4** (a simulation of MX dot products with explicit
  accumulator precision). Differs: a library, not a network-level fidelity
  study.
- **[Kha25] F. A. Khattak, M. Mikaitis. "Accurate Models of NVIDIA Tensor
  Cores." arXiv:2512.07004.** VERIFIED (arXiv API). Bit-accurate software
  models of V100, A100, H100 and B200 matrix units in 8-, 16- and 19-bit
  formats, validated semi-exhaustively; covers rounding, accumulator width and
  normalisation. Overlap: **C4**, the hardware side. Shows that "the GPU's FP8
  path" is not FP32 quantise-dequantise semantics.
- **[Fas21] M. Fasi, N. J. Higham, M. Mikaitis, S. Pranesh. "Numerical
  behavior of NVIDIA tensor cores." PeerJ Computer Science 7:e330, 2021.**
  Bibliographic fields VERIFIED `[check]` (Crossref, DOI
  10.7717/peerj-cs.330, published 2021-02-10). Abstract read only in the
  Semantic Scholar copy (secondary): using V100, T4 and A100 cards it
  determines "what precision is used for the intermediate results, whether
  subnormal numbers are supported, what rounding mode is used, in which order
  the operations underlying the matrix multiplication are performed, and
  whether partial sums are normalized", and notes that these aspects "are not
  documented by NVIDIA".
- **[Mue25] L. Mueller, A. Garcia-Ortiz, A. Najafi, A. Fuks, L. Bamberg.
  "Rescaling-Aware Training for Efficient Deployment of Deep Learning Models on
  Full-Integer Hardware." arXiv:2510.11484 (submitted to IEEE ESL per the arXiv
  comment).** VERIFIED (arXiv API). The abstract reports full accuracy with 8×
  narrower rescaler multiplicands. Overlap: **C2** (requantisation width is a
  real cost knob) and C4 (rescaler precision in deployment).
- [Agg24] (FP32 accumulation inside Brevitas fake quantisation), [Jac18],
  [Yao21] and [Rus19] (deployed integer-only paths) bear on C4 indirectly. No
  study was found that measures, at network level, the top-1 gap between FP32
  quantise-dequantise simulation and bit-exact execution for sub-8-bit
  non-integer formats in CNNs, and attributes that gap to accumulator,
  scale-placement and rounding components (see C4 below).

### 1.13 Works added by the independent check `[check]`

These were not in the first version. Each was opened by the checking agent on
2026-10-01 through the source named. Characterisations are limited to the
abstract unless stated.

- **[Ugu17] Y. Uguen, F. de Dinechin. "Design-space exploration for the Kulisch
  accumulator." HAL report hal-01488916v2, 2017.** VERIFIED (HAL API record
  with abstract). Kulisch's exact accumulator revisited for reconfigurable
  computing with "smaller, more resource-efficient floating-point formats".
  [Agg24] cites it as the source of its minifloat accumulator-width formula.
  Overlap: **C3** (the origin of the closed-form exact width).
- **[Ber22] L. Bertaccini, G. Paulin, T. Fischer, S. Mach, L. Benini.
  "MiniFloat-NN and ExSdotp: An ISA Extension and a Modular Open Hardware Unit
  for Low-Precision Training on RISC-V Cores." arXiv:2207.03192; IEEE ARITH
  2022, DOI 10.1109/ARITH54963.2022.00010.** VERIFIED (arXiv API; Crossref).
  Two 8-bit and two 16-bit FP formats with expanding sum-of-dot-product
  instructions "that accumulate the result in a larger format". The fused unit
  saves "around 30% of the area and critical path" against two cascaded
  expanding FMAs; a cluster "implemented in 12 nm FinFET technology" reaches
  575 GFLOPS/W for FP8-to-FP16 GEMM. Evidence: ASIC implementation. Overlap:
  **C2/C3** (minifloat dot-product hardware with a wider accumulation format,
  in ASIC). Differs: training; FP16 accumulation, no lossless certificate.
- **[Lut24] D. R. Lutz, A. Saini, M. Kroes, T. Elmer, H. Valsaraju. "Fused FP8
  4-Way Dot Product With Scaling and FP32 Accumulation." IEEE ARITH 2024,
  pp. 40–47, DOI 10.1109/ARITH61463.2024.00016.** Bibliographic fields
  VERIFIED (Crossref). Abstract read only in the Semantic Scholar copy
  (secondary): two microarchitectures for "fused FP8 DOT4 accumulating to
  higher precision FP32 with scaling", one of which expands products to fixed
  point before accumulation; the designs are synthesised. Overlap: **C2, C3
  and C4**: an industrial FP8 dot-product datapath that accumulates in FP32
  and includes the scaling step. Open the paper before it is cited.
- **[Nat25] V. Natesh, H. T. Kung, D. Kong. "MGS: Markov Greedy Sums for
  Accurate Low-Bitwidth Floating-Point Accumulation." arXiv:2504.09072
  (2025).** VERIFIED (arXiv API). Reorders dot-product terms by exponent to
  avoid "swamping" in low-bitwidth floating-point accumulators at inference
  time for 8-bit floating point, with accuracy "on par with high-precision
  floating-point baselines" for image classification; its dMAC units "reduce
  power consumption by up to 34.1%". Overlap: **C3** and the recommended
  mechanism (a): narrow accumulators for low-bit floating-point inference with
  hardware power. Differs: 8-bit; error reduction rather than a lossless
  certificate.
- **[Col23a] I. Colbert, A. Pappalardo, J. Petri-Koenig. "Quantized Neural
  Networks for Low-Precision Accumulation with Guaranteed Overflow Avoidance."
  arXiv:2301.13376 (2023).** VERIFIED (arXiv API). The predecessor of A2Q:
  accumulator bit-width bounds derived for QAT with weight normalisation.
- **[Mic22] P. Micikevicius, D. Stosic, N. Burgess, M. Cornea, P. Dubey,
  R. Grisenthwaite, S. Ha, A. Heinecke, P. Judd, J. Kamalu, N. Mellempudi,
  S. Oberman, M. Shoeybi, M. Siu, H. Wu. "FP8 Formats for Deep Learning."
  arXiv:2209.05433 (2022).** VERIFIED (arXiv API). Defines E4M3 and E5M2. The
  reference for the FP8 anchors named in Section 3.
- **[Nou22] B. Noune, P. Jones, D. Justus, D. Masters, C. Luschi. "8-bit
  Numerical Formats for Deep Neural Networks." arXiv:2206.02915 (2022).**
  VERIFIED (arXiv API). A study of 8-bit floating-point formats for
  activations, weights and gradients.
- **[Rou20] B. Darvish Rouhani, D. Lo, R. Zhao, M. Liu, J. Fowers,
  K. Ovtcharov, et al. "Pushing the Limits of Narrow Precision Inferencing at
  Cloud Scale with Microsoft Floating Point." NeurIPS 2020.** Title and
  authors VERIFIED (proceedings.neurips.cc/paper_files/paper/2020 listing).
  Abstract not opened. The production block-floating-point predecessor of MX.
- **[Dai21] S. Dai, R. Venkatesan, H. Ren, B. Zimmer, W. J. Dally,
  B. Khailany. "VS-Quant: Per-vector Scaled Quantization for Accurate
  Low-Precision Neural Network Inference." arXiv:2102.04503 (2021).** VERIFIED
  (arXiv API); the MLSys 2021 venue was not checked. Per-vector scale factors
  with a two-level scheme, and a modified accelerator "to study the area and
  energy overheads of per-vector scaling support"; 4-bit weights and
  activations give "37% area saving and 24% energy saving" on ResNet50.
  Overlap: **C2** (the cost of scale support measured in hardware together
  with accuracy) and C5.
- **[Zha22] S. Q. Zhang, B. McDanel, H. T. Kung. "FAST: DNN Training Under
  Variable Precision Block Floating Point with Stochastic Rounding."
  arXiv:2110.15456 (2021).** VERIFIED (arXiv API); the HPCA 2022 venue was not
  checked. Block floating point with variable precision; training.
- **[Cam19] V. Camus, L. Mei, C. Enz, M. Verhelst. "Review and Benchmarking of
  Precision-Scalable Multiply-Accumulate Unit Architectures for Embedded
  Neural-Network Processing." IEEE JETCAS 9(4):697–711, 2019, DOI
  10.1109/JETCAS.2019.2950386.** Bibliographic fields VERIFIED (Crossref).
  Abstract not opened. A MAC-level benchmark a DAC reviewer may expect.
- **[LiM21] Y. Li, M. Shen, J. Ma, Y. Ren, M. Zhao, Q. Zhang, R. Gong, F. Yu,
  J. Yan. "MQBench: Towards Reproducible and Deployable Model Quantization
  Benchmark." arXiv:2111.03759; NeurIPS 2021 Datasets and Benchmarks (arXiv
  comment).** VERIFIED (arXiv API). Evaluates quantisation algorithms under
  hardware-deployable settings on several platforms. The abstract reports that
  algorithms perform about the same "on the conventional academic track",
  while "for the hardware-deployable quantization, there is a huge accuracy
  gap", and that "no existing algorithm wins every challenge". Overlap:
  **C4** (a network-level gap between academic simulation and deployable
  quantisation, for integers) and **C1** (algorithm ranking depends on the
  setting).
- **[Dai24] D. Dai, Y. Zhang, J. Zhang, Z. Hu, Y. Cai, Q. Sun, Z. Zhang.
  "Trainable Fixed-Point Quantization for Deep Learning Acceleration on
  FPGAs." arXiv:2401.17544 (2024).** VERIFIED (arXiv API). QFX emulates
  fixed-point arithmetic in PyTorch so that deployed HLS models produce "the
  same numerical results as their software counterparts". Overlap: C4 (a
  bit-accurate software model for fixed point).
- **[Mik23] M. Mikaitis. "Monotonicity of Multi-Term Floating-Point Adders."
  arXiv:2304.01407 (2023).** VERIFIED (arXiv API). Numerical properties of
  multi-term adders with a single normalisation and rounding. Overlap: C4
  (the semantics of hardware accumulation).
- **[LiX24] X. Li, A. Li, B. Fang, K. Swirydowicz, I. Laguna,
  G. Gopalakrishnan. "FTTN: Feature-Targeted Testing for Numerical Properties
  of NVIDIA & AMD Matrix Accelerators." arXiv:2403.00232 (2024).** VERIFIED
  (arXiv API). Tests for extra precision bits, accumulation order and
  subnormal handling, which "are not publicly documented". Overlap: C4.
- **[Gho21] A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney, K. Keutzer. "A
  Survey of Quantization Methods for Efficient Neural Network Inference."
  arXiv:2103.13630 (2021).** VERIFIED (arXiv API). The survey a reviewer will
  expect next to [Nag21].
- **SA-ANT (X. Geng, S. Liu, H. Wang, J. Han, H. Jiang, DATE 2026, DOI
  10.23919/DATE69613.2026.11539270)** and **ISQ (C. Zhang, X. Yang, S. Yu,
  R. Dou, L. Liu, "ISQ: Intermediate-Value Slip Quantization for
  Accumulator-Aware Training", IEEE Signal Processing Letters 32:976–980,
  2025, DOI 10.1109/LSP.2025.3539579).** Found as neighbouring Crossref hits;
  title, authors and venue only, abstracts not opened. SA-ANT is a further
  adaptive-type paper in the ANT line (C5); ISQ is a further accumulator-aware
  training method for integers (C3).

## 2. Claim-by-claim verdict

Project evidence referred to below: [B-stage balanced
results](b-stage-balanced-results-2026-09-27.md) (FP32 QDQ, 1,000-image
development panels) and the [scaled bridge v1
pilot](scaled-bridge-v1-results-2026-09-28.md) (exact code-domain execution,
128 images, ResNet-18, maxabs).

### C1. The PTQ recipe changes the cross-format ordering at ≤8 bits

**Verdict: partially covered. Confidence medium.**

- Prior evidence that format value depends on the recipe: [Kuz22] (the FP8
  advantage over INT8 under PTQ shrinks under QAT); [ZhaM26] (PTQ
  effectiveness under MXFP "depends strongly on format compatibility",
  LLMs); [Shen24] (best FP8 variant by domain); [Che25] (INT/FP ordering by bit
  width and granularity). [Agg24] ran four PTQ techniques on the same CNNs but
  reported the best case, without testing ranking stability.
- `[check]` [LiM21] (MQBench) adds that, for integer quantisation, algorithm
  rankings change between the academic setting and hardware-deployable
  settings ("no existing algorithm wins every challenge"). The general idea
  that conclusions depend on the recipe and the setting is therefore
  established; only the cross-family reversal at sub-8 bits is not.
- What remains: a **paired, uncertainty-quantified demonstration on edge CNNs
  at 4–8 bits that the cross-family order reverses between reasonable PTQ
  recipes**, with one faithful reconstruction-based PTQ baseline. The project
  already has candidate reversals (for example ResNet-18 INT6 versus BFP6:
  +17.1 pp under maxabs and +8.2 pp under percentile, so the order is
  preserved; MobileNetV2 FP6 E3M2 versus E2M3 at −25.4 pp under maxabs needs
  its percentile counterpart). As stated, C1 is vulnerable. A reviewer will
  answer that maxabs per-tensor calibration is a straw man; MobileNet INT8
  collapsing under maxabs is the failure that [Nag19] addresses.
- Differentiating experiment: for 3 models × 4–6 formats, compare reversal
  frequency across maxabs, percentile/MSE, per-channel ranges plus DFQ-style
  equalisation, and one reconstruction PTQ (AdaRound or BRECQ, reproduced
  faithfully). Report Kendall-τ of format ranks per recipe with bootstrap
  intervals. **C1 is defensible only if reversals survive among the strong
  recipes, not only between maxabs and the rest.** The 1,000-image development
  panels can establish this. The full 50k confirmation is blocked for now.

### C2. Complete arithmetic cost reorders a multiplier-only ranking

**Verdict: partially covered, substantially: at FPGA level for sub-8-bit
minifloats, and at ASIC or analytical level for 8-bit formats `[check]`.
Confidence medium-high.**

- [Agg24]: integer keeps Pareto optimality once MAC cost is included, despite
  higher minifloat accuracy (FPGA LUTs, worst-case accumulator widths).
  [Dam24]: minifloat MACCs use 20–180% more resources than integer, rising to
  60–300% with conversion (TCAD 2024, FPGA). [vB23]: FP 50–180% less
  efficient. [Fox21] and [Tam20] also give FP/INT hardware ratios. [Mue25]:
  requantisation width matters. [Cuy26], [Pra25] and [Che26]: MX and scaled
  datapath costs.
- `[check]` The first version understated the ASIC-level prior work. [Des23]
  synthesises complete exact dot-product operators for E5M2, E4M3 and Posit8
  at TSMC 16FFC and compares area and power with INT8 and FP16. [Joh18] gives
  28 nm synthesis of a log-float multiply-add with Kulisch accumulation
  against an integer multiply-add. [Ber22] implements expanding minifloat
  dot-product units in 12 nm. [vB23] counts gates for both Kulisch and
  floating-point accumulators. [Dai21] measures the area and energy overhead
  of per-vector scale support. [Che25] and [Pra25] report MXINT8 against
  FP formats with a hardware model and with generated RTL. The premise itself
  ("the cost of additions begins to dominate that of multiplications") is
  stated in [Col24b] and, for extreme quantisation, in [Ni20].
- What remains: (i) **ASIC evidence below 8 bits** `[check]`: an ASIC flow is not new
  in itself (see the bullet above), but no opened source gives ASIC cost for
  4–7-bit minifloat, BFP and the other families (the project has ICS55); (ii) explicit accounting for scale application and requantisation,
  decode, metadata and memory traffic per configuration, **with quality
  attached**; (iii) showing a reversal relative to a multiplier-only ranking
  for sub-8-bit families beyond INT/minifloat (BFP/MX, posit, LNS, codebook).
  The direction of the effect (complete cost favours integer) is already
  published, so the novelty is in breadth, sub-8-bit ASIC evidence and the
  uncertainty-aware Pareto, not the qualitative finding. `[check]` A paper
  that presents "complete cost changes the ranking" as a discovery would be
  contradicted by [Agg24], [Dam24], [Des23], [vB23] and [Col24b].
- Differentiating experiment: the same synthesised PE for each finalist
  (multiplier, decode, accumulator at the certified width, scale/requant unit,
  metadata), at one target frequency on ICS55, with workload activity. Plot the
  multiplier-only rank against the complete-PE rank at iso-quality.

### C3. Statically certified lossless per-layer accumulator widths for minifloats; FP32 accumulation exact for them

**Verdict: partially covered; the per-layer, code-domain certificate for
non-integer PTQ formats was not found. Confidence medium.**

- Closest prior work: [Agg24] (closed-form worst-case minifloat accumulator
  width, exponential in exponent bits); [Dam24] and [Joh18] (exact or Kulisch
  accumulation hardware for minifloats or log floats); [Car19a/b] and [Mal22]
  (exact MAC and quire for posits); [Umu25] (SIRA: static interval analysis of
  per-layer accumulator widths, **for scaled integers**); [Col23], [Col24a]
  and [Col24b] (overflow guarantees for integers, including PTQ in AXE);
  [Sak19], [Blu24] and [ElA25] (statistical, fine-tuned or error-bound
  accumulator precision); [Des23] (exact FP8 and Posit8 dot-product
  operators in ASIC, now VERIFIED).
- `[check]` Four points narrow the gap further than the first version said.
  (1) Deriving the width from the actual constant weights instead of the
  datatype is prior art for integers: [Umu25] says so explicitly, and the
  [Col23] bound is an L1-norm bound on the weights. (2) The exact-width formula
  for floating-point formats goes back to the Kulisch-accumulator literature
  ([Ugu17]). (3) Narrow accumulators for low-bit floating-point inference
  with hardware numbers exist: [Nat25] (MGS, 8-bit FP, dMAC power), [Blu24]
  (12-bit), [Ber22] (FP8 accumulated in FP16) and [Lut24] (FP8 accumulated in
  FP32). (4) [Agg24] already assumes, without proof, that FP32 accumulators
  are adequate for minifloats, and [Cuy26] states the opposite for MX MACs
  ("FP32 accumulation suffers from quantization losses"). The project's
  certificate would settle a point on which two published papers disagree in
  passing; that is a real but small contribution.
- What remains: a **per-layer certificate computed from the actual weight codes
  and the admitted activation code domain for minifloat, BFP/MX and other
  non-integer formats**, showing how much tighter it is than the generic bound,
  and the corollary that FP32 accumulation is provably exact for some formats
  (E2M3: all 21 ResNet-18 MAC layers under 2^24) but not provably exact for
  others (E3M2 at 29 bits and FP7 at 31 bits exceed 24 bits). `[check]` On the
  128-image pilot, the FP32 control still produced identical records for E3M2
  and no changed prediction for FP7 (57 of 128 images had a differing raw dot
  result), so the certificate currently shows where exactness is guaranteed,
  not where FP32 fails in practice. The arithmetic is elementary. The contribution
  is the per-layer, per-format map plus its use as the lossless reference for
  narrower accumulators. On its own this is a modest contribution. It becomes
  meaningful when joined to the failure point and to MAC area (Section 4).
- Differentiating experiment: for every finalist format and model, give the
  certified width per layer against the [Agg24]-style bound and the
  SIRA-style interval bound. Then sweep FP16, a custom 21-bit float, and
  narrower saturating fixed-point down to the measured failure width.

### C4. Fidelity of FP32 quantise-dequantise simulation against bit-exact execution for low-bit non-integer formats

**Verdict: apparently open in the specific form stated (non-integer
sub-8-bit formats, CNNs, network level, with attribution); the integer
analogue is published `[check]`. Confidence low-medium, because negatives
cannot be proven by search.**

- Related: simulators [Zha19] and [Isl26]; hardware numerics models [Kha25]
  and [Fas21] (UNVERIFIED); [Agg24] relies on fake quantisation with FP32
  accumulators; deployed integer-only paths in [Jac18], [Yao21] and [Rus19].
  No audited paper measured the network-level top-1 gap for sub-8-bit
  minifloats, or attributed it separately to accumulation, scale factoring and
  postop/store rounding.
- `[check]` Adjacent work found by the independent check: [LiM21] (MQBench)
  measures a network-level accuracy gap between academic simulation and
  hardware-deployable quantisation, for integers; [Dai24] provides a
  bit-accurate fixed-point software model; [Lut24], [Cuy26], [Mik23] and
  [LiX24] document that real FP8/MX dot-product hardware accumulates with
  specific, lossy semantics; [Isl26] says the OCP specification does not
  prescribe accumulation precision or rounding. C4 must be worded as the
  non-integer, sub-8-bit, attributed version of a known concern, not as the
  first observation that simulation and deployment differ.
- The project already has a non-obvious finding here. In scaled bridge v1, the
  exact code-domain contract changes ResNet-18 top-1 by −3.9 to −4.7 pp
  against B on 128 images (two of three paired intervals exclude zero). Yet
  replacing the exact accumulator with sequential FP32 changes no top-1 or
  top-5 outcome in any of the three cases `[check]` (records are identical
  for E2M3 and E3M2; for FP7 E3M3, 57 of 128 images have a differing raw dot
  result but no changed prediction), so on this panel the gap comes from the
  scale and postop contract rather than the accumulator. This is development evidence on 128 images; lane L5 is
  extending it to 1,000 images.
- Differentiating experiment: a factorial decomposition (accumulator ×
  scale-placement × postop rounding × store rounding) across formats and
  models, with paired intervals and first-divergence layer statistics.

### C5. A joint selection rule for format, scale policy and accumulator, possibly per layer

**Verdict: partially covered and crowded. Confidence medium.**

- Per-tensor or per-group adaptive type selection with accelerators: [Guo22]
  (ANT, MICRO'22), [Hu25] (M-ANT/MANT, HPCA'25) and [Ram24] (LP, DAC'24).
  Per-layer INT/FP selection: [Zha23]. Per-layer precision search for integer
  and for floating-point models on the same CNNs: [Dot24] (FLIQS; the abstract
  does not claim mixing the two families within one model `[check]`).
  Per-vector scale selection with hardware cost: [Dai21]. A further
  adaptive-type accelerator: SA-ANT (Section 1.13).
  Hardware-in-the-loop bit-width search: [Wan19] and [Don19]. Accumulator-aware
  PTQ: [Col24b]. Component-wise accumulation precision: [ElA25]. Scale
  hierarchies co-designed with hardware: [Che26].
- What remains: a selector that **includes the accumulator width (via the
  certificate) and the complete PE cost as decision variables**, validated on
  a held-out model. Without a held-out test and strong baselines (FLIQS-style
  search, HAQ/HAWQ), this would read as an incremental variant of ANT/FLIQS.
- Differentiating experiment: freeze the rule on two models, apply it to a
  held-out CNN, and compare against a uniform INT8/MXFP8 anchor and a FLIQS-like
  INT/FP per-layer search at iso-area.

## 3. Baselines and comparisons a reviewer will demand

### DAC / ICCAD reviewer (EDA and design focus), in priority order

1. **Strong, faithfully reproduced PTQ.** Per-channel weight ranges;
   percentile or MSE activation calibration; DFQ equalisation and bias
   correction for MobileNets [Nag19]; one reconstruction method (AdaRound
   [Nag20] or BRECQ [Li21]). The project's own rounding variant is
   "AdaRound-inspired" and will not be accepted as AdaRound.
2. **Standard anchors in the same flow:** INT8 per-channel, FP8 E4M3, MXFP8,
   MXINT8, and INT4 with a strong recipe, at matched bits and with matched
   first/last-layer policy.
3. **Real ASIC cost at the PE level:** synthesis, timing closure at a stated
   frequency and post-layout area for finalists on one PDK. Workload-derived
   switching activity for energy (vectorless is not acceptable as headline),
   per the [publication plan](../publications/README.md).
4. **Accumulator baselines:** a fixed FP32 accumulator; the [Agg24]
   closed-form width; A2Q/A2Q+ or AXE for integers at equal width; `[check]` for
   narrow floating-point accumulators, [Blu24] and MGS [Nat25]; for exact FP8
   operators in ASIC, [Des23]. Explain why the certificate is tighter, and
   when.
5. **Complete cost:** the scale/requantisation unit (cf. [Mue25], [Jac18]),
   decode, metadata (MX scales), and packed versus physical memory.
6. **Statistics and scale:** the full ImageNet validation set for headline
   numbers (currently blocked: the 50k set is not available) and at least three
   calibration seeds.
7. **Framework comparison:** why not Voyager [Pra25] or a FINN-style flow; a
   qualitative comparison with the published FPGA results of [Agg24] and
   [Dam24].

### MICRO / HPCA reviewer (architecture focus), in priority order

1. **A full accelerator, not a PE:** array, dataflow, buffers and DRAM traffic,
   with energy per inference and throughput at iso-area. Compare against
   adaptive-type accelerators (ANT [Guo22], M-ANT [Hu25], LP [Ram24],
   AdaptivFloat [Tam20]) and MX accelerators ([Cuy26], MicroScopiQ [Ram25],
   M2XFP [Hu26], Avant-Garde [Gil25]).
2. **Workload breadth:** recent MICRO/HPCA/ISCA format papers in this audit all
   evaluate LLMs or foundation models ([Hu25], [Ram25], [Hu26], [Che26],
   [Gil25]). A CNN-only study will be challenged unless edge CNNs are argued
   as the target domain, and at least one transformer should be included.
3. **A clear architectural mechanism** (a new datapath, a format or a
   hardware selector), not only a measurement study.
4. **Cycle-level performance plus post-layout physical evidence** for
   finalists.
5. All of the DAC items 1–2 and 6.

## 4. Recommendation: the one mechanism to commit to

**Commit to (a), the accumulator-width sweep to the failure point joined to
MAC area and energy, anchored by the per-layer certificate (C3) as the
lossless reference. Carry the simulator-versus-exact decomposition (C4) as the
methods section that justifies exact execution. Do not commit to (b) as the
headline.**

Reasons:

1. **It is the least-occupied gap that the evidence supports.** Accumulator
   reduction with guarantees is published for integers: QAT in [Col23] and
   [Col24a]; PTQ for LLMs in [Col24b]; FPGA static analysis in [Umu25].
   `[check]` Minifloat accumulator cost below 8 bits is published as
   worst-case formulas and FPGA LUTs ([Agg24], [Dam24]); at 8 bits it is also
   published in ASIC ([Des23], [Ber22], [Lut24], [Joh18]) and narrow
   floating-point accumulators have been studied for 8-bit inference
   ([Nat25], [Blu24]). No audited work gives, for non-integer
   sub-8-bit PTQ formats on edge CNNs, a per-layer certified lossless width,
   the measured quality-failure width under FP16, custom-float and saturating
   fixed-point accumulators, and the ASIC MAC area and energy at each width.
   That combination is a concrete design rule ("for format F, layer class L,
   use accumulator A; FP32 is exact for E2M3 but wasteful; saturating fixed
   at k bits fails at …") of the kind DAC and ICCAD accept.
2. **It reuses what exists:** the scaled bridge v1 certificates, the frozen
   sweep specification (FP16, custom21, narrower fixed-point with saturation),
   three admitted cases, and lane L4's integer MAC RTL and OpenSTA/OpenROAD
   flow. It can produce DAC-deadline evidence on 1,000-image panels without the
   50k set. Full-benchmark confirmation of the headline cases stays open until
   the 50k set is available.
3. **(b) is crowded and expensive.** Per-layer format selection is ANT, M-ANT,
   FLIQS, MoFQ and LP territory. A credible version needs a held-out model and
   baselines such as FLIQS and HAQ, which do not fit the calendar. It can become
   an extension if (a) shows per-layer heterogeneity in failure widths: the
   certificate-driven per-layer accumulator assignment then follows directly
   from (a)'s data.
4. **(c) alone suits MLSys better than DAC**, and the MLSys 2027 deadline
   (October 30, 2026) is too close. As a supporting section it is strong: the
   v1 result that the B-versus-exact gap (about −4 to −5 pp at 128 images)
   comes from the scale/postop contract and not from the accumulator is the
   reason exact execution is needed at all. It also protects C2/C3 from the
   objection that fake quantisation would have given the same answer.
5. **C1 and C2 should be supporting results, not the headline.** Their
   qualitative direction is already published ([Kuz22], [ZhaM26], [Agg24],
   [Dam24]). They gain weight from the project's breadth, its paired intervals
   and its ASIC numbers.

`[check]` Independent check of this recommendation: agreed, with three
conditions. (1) The claim must be scoped to sub-8-bit non-integer PTQ formats
with a measured failure width; at 8 bits the neighbouring results [Nat25],
[Blu24], [Des23], [Ber22] and [Lut24] already cover accumulator reduction and
ASIC cost, and they must be cited and compared. (2) The certificate must be
shown to be tighter than the [Agg24]/[Ugu17] formula for a reason other than
convention; the constant 3-bit difference in the worked example of Section 1.1
is a warning. (3) The headline must be the measured failure point joined to
ASIC area and energy, because the pilot shows no quality loss from FP32
accumulation even where the certificate exceeds 24 bits. If the sweep finds
no format-dependent failure width, option (c) (the C4 decomposition) is the
stronger remaining result and MLSys or a journal is the better venue.

Risks: the sweep may show that every format tolerates narrow accumulators
down to a common width. That is still a design rule, but a weaker one. ASIC
energy needs workload activity (lane L4). The failure-width measurement needs
exact execution throughput at 1,000 images per arm (lanes L2 and L5).

## 5. Venue fit and deadlines

Checked on 2026-10-01.

| Venue | Fit | Deadline | Verification |
| --- | --- | --- | --- |
| **DAC 2027** | **Best fit** for (a) plus an ASIC PE plus a design rule; 6 pages plus 1 page of references | Abstract **Nov 11, 2026**; manuscript **Nov 18, 2026** (5:00 pm US Pacific); conference begins Jul 11, 2027 | VERIFIED (re-checked `[check]`: the page says "up to 6 pages plus an additional 1 page for references only" and "5:00pm US Pacific Time"): dac.com/2027/program/research-manuscript-submissions |
| MLSys 2027 | Fit for C4 as a methods contribution | Paper **Oct 30, 2026**, 12:00 PM PDT (submissions open Oct 10); conference Jun 22–24, 2027 | VERIFIED (re-checked `[check]`; conference sessions Tue June 22 – Thu June 24, 2027, Bellevue, WA): mlsys.org/Conferences/2027/Dates |
| ISCA 2027 | Needs a full accelerator plus LLM workloads (Section 3) | Not found on an official page (iscaconf.org/isca2027 returned 404). A third-party site says the ISCA 2026 abstract deadline was Nov 10 | **UNVERIFIED** |
| HPCA 2027 | As ISCA | Abstract Jul 24, 2026; paper Jul 31, 2026 (AoE). **Passed** | VERIFIED (re-checked `[check]`): conf.researchr.org/track/hpca-2027/hpca-2027-main-conference |
| ASPLOS 2027 | Weak fit | April cycle (Apr 15, 2026) and September cycle (Sep 9, 2026). **Both passed**; the CFP lists no further cycle | VERIFIED (re-checked `[check]`; dates are AoE): asplos-conference.org/asplos2027/cfp |
| ICCAD 2026 | Good fit, but closed | Abstract April 7, 2026; regular paper April 14, 2026; conference November 8–12, 2026, San Jose. **Passed.** The "Sep 15, 2026" date seen earlier on HotCRP is not the submission deadline | VERIFIED `[check]`: iccad.com/2026, "Important Dates & Deadlines" |
| ICCAD 2027 | Good fit | Not announced (typically spring) | **UNVERIFIED** |
| DATE 2027 | Good fit, but closed | Abstract Sunday 13 September 2026 AoE; final paper Sunday 20 September 2026 AoE; notification 23 November 2026. **Passed** | VERIFIED `[check]`: date27.date-conference.com/call-for-papers |
| MICRO 2027 | As ISCA | No official page found (microarch.org/micro60 returned 404 on 2026-10-01) `[check]` | **UNVERIFIED** |
| IEEE TCAD / IEEE TC | Good fit for an extended version; [Dam24] is a TCAD precedent | Rolling submission | Not checked |
| ARITH, CoNGA | Fit for C3 as an arithmetic result alone | Not checked | **UNVERIFIED** |

Recommendation: target **DAC 2027** (abstract Nov 11, manuscript Nov 18,
2026). This matches the roadmap's November 8–22 submission window. Plan an
extended TCAD journal version that adds full-benchmark confirmation once the
50k set is available. ISCA/MICRO only become realistic with a full accelerator
and at least one transformer workload.

## 6. Open items for a follow-up agent

- `[check]` Closed by the independent check (details in Section 7): [Des23],
  the venues of [Kuz22], [Li20], [Ni20], [Li21], [Blu24], [Det23], [Tam20],
  [Jac18] and [Col23], and the full-text questions on [vB23], [Pra25] and
  [Umu25].
- Still open: the publisher pages for the abstracts of [Fas21], [Gil25],
  [Lut24], [Fox21], [Del25], [Rou20] and [Cam19] (bibliographic fields are
  verified; abstracts were read only in secondary copies or not at all), and
  the OCP MX v1.0 specification itself.
- Match the bit-width convention of the [Agg24] formula to the project
  certificate before printing the 26-versus-23-bit comparison.
- Re-check the deadlines for ISCA 2027, MICRO 2027 and ICCAD 2027 on official
  pages when they are published (DATE 2027 is now verified and has passed).
- `[check]` Open [Lut24], [Nat25], [Des23] and [Ber22] in full before the
  accumulator sweep is designed: they are the nearest results to mechanism
  (a) and the sweep must include or answer their design points.

## 7. Verification log (independent check, lane L3 part 2, 2026-10-01)

A second agent checked this document without relying on the first agent's
notes. No experiment was run. Changed statements in Sections 0–6 carry the tag
`[check]`.

### 7.1 Method

- **arXiv API**, re-fetched for all 67 arXiv identifiers cited in Sections
  1.1–1.12 and parsed locally: title, full author list, date, comment,
  journal-ref and DOI fields, and the abstract text. Numbers quoted from
  abstracts were compared against the raw abstract text, not a summary.
- **Crossref REST API** (publisher-deposited metadata) for venue, pages and
  DOI: [Dam24], [Rou23a], [Gil25], [Ram25], [Hu25], [Hu26], [Guo22], [Tam20],
  [Ram24], [Car19a], [Car19b], [Mal22], [Mal26], [Zha21], [Ger23], [Cuy26],
  [Agg24], [Col23], [Jac18], [Xie20], [Des23], [Fas21], [Del25], [Lut24],
  [Ber22], [Cam19].
- **Official listings**: proceedings.neurips.cc (2020, 2022, 2023),
  proceedings.mlsys.org (2024), proceedings.iclr.cc (2024), iclr.cc virtual
  pages (2020, 2021), proceedings.mlr.press (v119, v235), hpca-conf.org
  (2025 main programme), iscaconf.org (ISCA 2025 programme).
- **Full text, downloaded and searched locally**: [Agg24] v3, [vB23] v2,
  [Che25] v1, [Pra25] v1, [Umu25] v1, [Blu24] v1 (arXiv HTML).
- **HAL API**: [Des23] (abstract), [Ugu17].
- **Deadline pages** opened directly: DAC 2027, MLSys 2027, ASPLOS 2027,
  HPCA 2027 (conf.researchr.org), ICCAD 2026, DATE 2027. ISCA 2027 and
  MICRO 2027 pages returned 404.
- **Project numbers** quoted in Sections 1.1 and 2 were compared with
  `b-stage-balanced-results-2026-09-27.md` and
  `scaled-bridge-v1-results-2026-09-28.md`.
- Semantic Scholar was used only as a secondary copy of abstracts that
  publisher pages would not serve ([Gil25], [Fas21], [Lut24], and a re-check
  of [Dam24]); those are marked as secondary.

### 7.2 Result for the existing entries

No invented citation was found. Every arXiv identifier resolves to the stated
title and authors. The errors were in characterisation, venue status and one
truncated quotation.

Substantive corrections:

1. **[Hu25] MANT / M-ANT.** The published title (IEEE proceedings and arXiv) is
   "M-ANT: …"; only the HPCA 2025 programme page prints "MANT: …". The audit
   said the paper "is published as MANT". DOI and pages added. The name
   collision stands, because the abstract itself names the data type "MANT".
2. **[Isl26] quotation.** The quotation was cut after "not documented". The
   abstract says "not documented by GPU vendors, and the precision and
   rounding are not prescribed by the OCP specification itself". Corrected in
   Sections 1.2 and 1.12.
3. **[Dot24] FLIQS.** "Joint per-layer INT/FP search" and "per-layer
   format-family selection" went beyond the abstract, which describes separate
   integer and floating-point mixed-precision searches performed during
   training. Corrected in Sections 1.10 and 2 (C5).
4. **[Des23].** Was UNVERIFIED. Now verified, and it matters: it is an ASIC
   (TSMC 16FFC) area and power comparison of exact FP8 and Posit8 dot-product
   operators against INT8 and FP16. The C2 verdict and the Section 4 reasoning
   were changed, because "an ASIC flow rather than FPGA LUTs" is not new at
   8 bits.
5. **[vB23].** Evidence level was "not stated". It is analytical (2-input gate
   count), and it already includes Kulisch and floating-point accumulators.
6. **[Che25].** The audit said hardware was "not stated in the relayed
   abstract". The abstract does claim hardware efficiency. The unverified
   "20–40%" snippet was replaced by the verified "37% and 38%" energy figures
   from a hardware model.
7. **[Agg24].** Quotations and formulas confirmed. Added three facts from the
   full text: the integer advantage is described as "slightly smaller hardware
   footprint"; the paper asserts that FP32 accumulators are adequate; it omits
   the cost of converting the accumulator back to floating point. The quoted
   headline sentences are in the introduction, not in the v3 abstract.
8. **[Dam24].** Publication details corrected to TCAD 44(6):2181–2194, June
   2025.
9. **[Ni20].** The word "validated" was placed in quotation marks but is not
   in the abstract. The ICLR 2021 title differs from the arXiv title.
10. **[Blu24].** "Building on FP8" is not in the abstract.
11. **[Xie20].** The IJCAI 2020 record has a different title and four
    authors, not eight.
12. **[Mal26], [Det23], [Sak19], [Col24b].** Statements that went beyond the
    abstracts were narrowed (comparator of the 38%/42.3% figures; "BF16
    compute"; "floating-point accumulation"; which version evaluates image
    classification).
13. **C3 and C4 project evidence.** "Replacing the exact accumulator with
    sequential FP32 changes nothing" was narrowed: no top-1 or top-5 outcome
    changes, but for FP7 E3M3 the raw dot result differs on 57 of 128 images.
    The E3M2 and FP7 certificates exceed 24 bits, yet the pilot shows no
    quality effect.
14. **Worked example.** The [Agg24]-style formula gives 26, 32 and 34 bits for
    E2M3, E3M2 and E3M3; the project certificates give 23, 29 and 31. The
    constant 3-bit difference was added as a warning.
15. **Venue table.** ICCAD 2026: the official site gives April 7 (abstract)
    and April 14, 2026 (paper); the conflict is resolved. DATE 2027: 13 and 20
    September 2026 confirmed on the official call for papers.

Venue statements moved from UNVERIFIED to VERIFIED: [Kuz22] NeurIPS 2022,
[Shen24] MLSys 2024, [Rou23a] ISCA 2023, [Gil25] DOI, [Tam20] DAC 2020 (under
the DAC title), [Li20] ICLR 2020, [Det23] NeurIPS 2023, [Ni20] ICLR 2021,
[Li21] ICLR 2021, [Blu24] ICLR 2024, [Xie20] IJCAI 2020, [Col23] ICCV 2023,
[Jac18] CVPR 2018, [Pra25] full-text claim, [Fas21] and [Del25] bibliographic
fields. [Col24a] and [Nag20] PMLR page ranges were confirmed on
proceedings.mlr.press. The four HPCA 2025 titles and the ISCA 2025
"LUT Tensor Core" title were confirmed on the programme pages.

### 7.3 Verdicts

- **C1** unchanged (partially covered, medium). [LiM21] added as prior
  evidence that rankings depend on the setting.
- **C2** made less optimistic: prior coverage now reads "FPGA level for
  sub-8-bit minifloats, and ASIC or analytical level for 8-bit formats". The
  remaining gap is sub-8-bit ASIC evidence, breadth of families and the
  uncertainty-aware Pareto.
- **C3** label unchanged (partially covered, medium), but the text now says
  that weight-aware static bounds are prior art for integers ([Umu25],
  [Col23]), that the exact-width formula predates [Agg24] ([Ugu17]), and that
  narrow floating-point accumulators with hardware numbers exist at 8 bits
  ([Nat25], [Blu24], [Ber22], [Lut24]).
- **C4** narrowed from "apparently open at network level for CNNs" to
  "apparently open in the specific form stated; the integer analogue is
  published" ([LiM21]).
- **C5** unchanged (partially covered and crowded); the [Dot24] description
  was corrected and [Dai21] and SA-ANT were added.
- **Recommendation (Section 4)**: the checking agent agrees with mechanism
  (a), under the three conditions added there.

### 7.4 Omissions added

Section 1.13: [Ugu17], [Ber22], [Lut24], [Nat25], [Col23a], [Mic22], [Nou22],
[Rou20], [Dai21], [Zha22], [Cam19], [LiM21], [Dai24], [Mik23], [LiX24],
[Gho21], SA-ANT and ISQ. The omission search was bounded: three web searches
plus the checking agent's candidate list, each candidate then verified through
the arXiv API, Crossref, HAL or a proceedings listing. It is not a systematic
review.

### 7.5 Still UNVERIFIED after this check

- **OCP MX v1.0 specification**: the official PDF URL still returns HTTP 403.
  The statement that it does not prescribe accumulation precision or rounding
  rests on [Isl26] alone.
- **Abstracts read only in a secondary copy** (Semantic Scholar): [Gil25],
  [Fas21], [Lut24].
- **Abstracts not opened at all**: [Del25], [Rou20], [Cam19], SA-ANT, ISQ.
- **[Fox21]**: only the title is confirmed on the ICLR 2021 listing; authors,
  the 4.1× and 2.3× figures and the hardware level are still from a secondary
  index.
- **Venues that rest only on the arXiv comment or journal-ref field**:
  [Che26] (MICRO 2026), [Dot24] (AutoML 2024), [Wei22] (ICLR 2022),
  [Sak19] (ICLR 2019), [Wan18] (NeurIPS 2018), [Han16], [Zha18], [Son18],
  [Zha19], [Wan19], [Don19], [Nag19], [Yao21], [Liu23], [LiM21]. These are
  author-supplied fields; none was contradicted.
- **Venues not checked**: [Dai21] (MLSys 2021), [Zha22] (HPCA 2022), and the
  final venues, if any, of [Umu25], [Col24b], [ElA25], [Pra25], [Kha25],
  [Isl26], [Mue25], [Nat25], [Jon25].
- **Full texts not opened**: everything except the six listed in 7.1. In
  particular [Sak19] (whether its accumulator is floating-point), [Nat25],
  [Lut24], [Des23] and [Ber22] should be read before the accumulator sweep is
  designed.
- **Deadlines**: ISCA 2027, MICRO 2027 and ICCAD 2027 have no official page
  yet. The ISCA 2026 date in the table is still from a third-party site.
- **Short quotations** in entries that were not re-read in full text are as
  given in the abstracts; quotations in Section 1.1 for [Agg24] were matched
  against the downloaded HTML, with mathematical symbols compared by their
  LaTeX source.
