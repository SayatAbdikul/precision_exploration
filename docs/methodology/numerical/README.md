# Numerical methodology

## Research questions and hypothesis

The study asks:

1. Which representation preserves inference quality best at 4–8 bits?
2. What is the complete hardware cost when decode, multiplication, accumulation, rounding, scaling, conversion, and storage are included?
3. Under equal generic silicon/resource and quality constraints, which numerical architecture is Pareto-optimal?

The working hypothesis is that there may be no global winner. The preferred format can vary with workload, layer, tensor distribution, quality target, accumulator, scaling, and hardware budget.

## Workload suite

| Model | Reason | Dataset/metric | Plan priority |
|---|---|---|---|
| ResNet-18 | Canonical standard convolutions and residuals | ImageNet-1K Top-1/Top-5 | Mandatory |
| MobileNetV2 | Depthwise-separable mobile CNN; quantization sensitive | ImageNet-1K Top-1/Top-5 | Mandatory |
| MobileNetV3 | SE blocks and hard-swish | ImageNet-1K Top-1/Top-5 | Mandatory |
| EfficientNet-B0/Lite0 | Different depthwise/SE design | ImageNet-1K Top-1/Top-5 | Recommended; D3 decides main-suite inclusion |
| YOLOv8n or similar tiny detector | Practical edge detection | COCO mAP50–95/mAP50 | Recommended core detector |

For every model freeze a checkpoint, preprocessing, dataset/evaluation version, FP32 metrics, and deployment-graph hash. Report absolute quantized quality and delta from FP32.

## Candidate datatype universe

The exact D1 candidate set is not yet fixed. The initial universe covers all practically relevant families at 8 bits and below while avoiding the impossible claim that every custom encoding can be enumerated.

### A — integer and fixed point

- Signed INT8 through INT2; mandatory anchors include INT8, INT6, INT4.
- Unsigned UINT8 through UINT2 for non-negative post-activation tensors.
- Qm.n fixed-point variants with total width ≤8.
- Dynamic fixed point: shared scale plus integer mantissa.
- Power-of-two-scaled integer.

### B — conventional/custom floating point

| Total bits | Representative sign/exponent/mantissa layouts |
|---:|---|
| 8 | E5M2, E4M3, E3M4, E2M5 |
| 7 | E4M2, E3M3, E2M4 |
| 6 | E4M1, E3M2, E2M3 |
| 5 | E3M1, E2M2 |
| 4 | E2M1 |

Evaluate only explicit semantics: finite-only saturation versus IEEE-like behavior, subnormal/FTZ policy, exponent bias, signedness, overflow/underflow, and rounding. Standard FP8 E4M3/E5M2 uses published semantics; non-standard FP7/6/5/4 names never stand alone without a manifest.

### C — block floating point and microscaling

- BFP8/BFP6/BFP4-style variants: element width, shared exponent width, block size/axis.
- MXFP8 E4M3/E5M2, MXFP6 E3M2/E2M3, MXFP4 E2M1.
- MXINT8 and experimental MXINT6/5/4.
- Scaled FP5/6/8 with tensor/channel/block scale as Experiment B variants.

The initial MX/BFP baseline uses reduction dimension K as block axis, block size 32, independent W/A scales, valid-value scaling plus conceptual zero padding for the incomplete final block, with later block-size 16/32/64 sweeps.

### D — tapered precision

- Posit(n, es) for selected n=4…8 and es values using standard semantics.
- Posit-like custom encodings only when clearly distinct and hardware relevant.

### E — logarithmic and power of two

- LOG8 through LOG4 with explicit log-domain integer/fraction allocation.
- Pure sign plus quantized exponent/power-of-two representation.
- Hybrid log/linear formats with a near-zero linear region.

### F — codebook/non-uniform

- NF4 and generalized NF5/NF6-like codebooks.
- Learned k-means/optimized centroids.
- Arbitrary `2^n`-value LUT quantization as a flexible reference.

Decode, storage, and LUT/product cost are charged to the hardware configuration.

### G — binary, ternary, very low cardinality

- Binary ±1 and 0/1.
- Ternary {-1, 0, +1}.
- Four-level signed/unsigned 2-bit sets.

These are extreme-efficiency references and may be difficult under PTQ.

### H — derived/custom research formats

Custom bias/splits, shared-exponent/non-uniform mantissa, hybrid log/float, piecewise/codebook-assisted minifloats, W/A family asymmetry, or a new format derived from tensor and hardware Pareto evidence. Add only clearly specified, non-redundant candidates.

## Datatype manifest

Every format records:

- unique name, family, width, and encoding;
- signedness;
- exponent/mantissa, integer/fraction, regime, logarithmic, or codebook parameters;
- bias and zero/subnormal/NaN/infinity behavior;
- overflow, underflow, saturation, and rounding;
- intrinsic/external scale representation and granularity;
- MX/BFP block size/axis and shared-scale format;
- complete codebook values where applicable.

Example from the plan:

```json
{
  "name": "fp6_e3m2_finite",
  "family": "float",
  "bits": 6,
  "signed": true,
  "exp_bits": 3,
  "mantissa_bits": 2,
  "bias": 3,
  "subnormals": false,
  "nan": false,
  "infinity": false,
  "overflow": "saturate",
  "underflow": "flush_zero",
  "rounding": "rne"
}
```

## Quantization studies

### Experiment A — controlled comparison

- Static PTQ; same pretrained checkpoint; no retraining.
- Eval mode → BatchNorm fold → frozen deployment graph → calibration → PTQ.
- MSE calibration where intrinsic/necessary scale selection is required.
- Conventional INT: per-output-channel weights, per-tensor activations, symmetric signed mapping, zero point 0.
- No bias correction/reconstruction.
- Same fixed, disjoint calibration/evaluation subsets.
- Mainly uniform W=A in the broad screen, but independent W/A architecture.
- Model C and a sufficiently wide family-appropriate accumulator.
- Strict low-precision operator graph.
- No optional external scale for self-scaling formats under the later roadmap F6 refinement.

### Experiment B — reasonable best-achievable PTQ

Apply to promoted candidates rather than a full Cartesian product:

- general external scale where meaningful;
- power-of-two scale versus general/no-scale;
- per-tensor, per-channel, and block scaling;
- optimized clipping/calibration and family-specific settings;
- FP bias/range tuning;
- MX/BFP block sizes 16/32/64;
- signed versus unsigned activations where semantics permit;
- practical first/last-layer exceptions;
- selected W/A asymmetry.

Experiment A and B results are never merged without their labels.

## Weight/activation combinations

The plan explicitly proposes W4A8, W5A6, W6A6, W6A8, W8A6, W8A8 and selected family-specific combinations. Uniform results cannot eliminate a family before at least a small asymmetry study.

## Arithmetic and accumulation

The exact definitions live in `docs/contracts/arithmetic.md`. Experiment A uses Model C; promoted candidates compare:

- Model A: rounded product and native accumulation;
- Model B: rounded narrow product and wider accumulator;
- Model C: exact/full product entering the accumulator.

Sweep native and widened accumulators. Include INT32, FP16, quire, Kulisch/exact fixed accumulation, scaled wide partial sums, or hybrid log-multiply/linear-accumulate references where appropriate. The main reduction order is sequential; tree reduction is an ablation. Requantize only at operator output storage.

## Operator policy

The primary strict graph includes candidate-precision Conv2D, 1×1 Conv, depthwise Conv, Linear, residual add, elementwise multiply, ReLU/ReLU6, pooling, hard-swish decomposition, and datatype-specific LUTs for required nonlinear functions. BatchNorm is folded. Bias uses accumulator precision. YOLO NMS stays high-precision postprocessing. Practical higher-precision exceptions are secondary and include their hardware cost.

## Exact inference engine

### Stack

```text
PyTorch model/data frontend
    -> frozen quantized graph
    -> custom exact operators
    -> reference / C++ / CUDA datatype backend
```

PyTorch loads checkpoints/data, inspects/converts graphs, orchestrates calibration, and computes quality metrics. Numerically important operators use custom exact kernels.

### Storage and compute

Use one `uint8` container per 2–8-bit code initially. True datatype width—not host storage width—drives hardware/memory accounting. Storage and compute representations are separate:

```text
encoded code -> exact decoded temporary -> defined arithmetic -> explicit rounding
```

Maintain encoded-reference and optional predecoded-fast modes; outputs must be identical. Dense packing is a later runtime optimization.

### Convolution path

- Start normal convolution with im2col plus custom exact GEMM.
- Implement a dedicated depthwise kernel early.
- Start NCHW; consider NHWC only if profiling justifies conversion.
- Parallelize independent batch/channel/spatial outputs without changing per-output reduction order.

### Verification ladder

| Level | Requirement |
|---|---|
| Primitive | Exhaustive ADD/MUL/convert pairs where feasible |
| Dot product | Random vectors at K such as 3, 9, 27, 64, 128, 256, 1024 |
| Small Conv | Deterministic tensors against reference |
| Full layer | C++ versus CUDA bit for bit |
| Network | Layer outputs/logits on a fixed image set before sweeps |

## CUDA optimization sequence

1. Benchmark one representative FP6 standard Conv and one depthwise Conv using algorithmic, LUT, and predecoded execution.
2. Implement one logical serial accumulator per output for correctness.
3. Tile independent output work; use shared memory only for measured reuse.
4. Vectorize memory transactions; packed FP4/INT4 is later.
5. Make datatype operations branchless using masks, predication, integer arithmetic, and compile-time constants.
6. Compile specialized kernels by W/A/accumulator/MAC semantics instead of runtime switches.
7. Select LUT versus algorithmic execution by width/family. A one-byte output table is 256 B/1 KB/4 KB/16 KB/64 KB for 4/5/6/7/8 bits respectively.
8. Fuse cheap post-ops such as Conv+Bias+ReLU/ReLU6+requantize only after semantic stability.
9. Optimize only profiled dominant operators.

The engineering target proposed by the plan is approximately 1,000 ImageNet images in under 1–2 minutes per ResNet-18-sized configuration on a basic Colab NVIDIA GPU. This is a target, not an accepted requirement.

## Provisional first experiment

### Models

ResNet-18, MobileNetV2, MobileNetV3-Large. The roadmap additionally keeps one tiny detector in the core suite and defers EfficientNet main-suite inclusion to D3.

### Preliminary formats

```text
INT8 INT6 INT5 INT4
FP8 E4M3, FP8 E5M2
selected FP7
FP6 E3M2, FP6 E2M3
selected FP5
FP4 E2M1
MXFP8
MXFP6 E3M2, MXFP6 E2M3
MXFP4, MXINT8
Posit8, Posit6, Posit5
Log8, Log6, Log5
```

This list is provisional; D1 freezes the actual first candidate set.

### Accumulation and PTQ

- Plan pilot choices: native, 8-bit, 12-bit, 16-bit accumulation.
- Canonical broad Experiment A begins with Model C plus a sufficiently wide accumulator; accumulator squeezing is a promoted sweep.
- Controlled granularity: per-output-channel weights and per-tensor activations where mapping scales apply.
- Block-32 belongs to intrinsic MX/BFP semantics or the later Experiment B sweep, not a general Experiment A scale.

Run all valid configurations on the fixed 1k subset. Following the detailed roadmap, promote promising and uncertain results plus preservation candidates, then 5k/10k, then approximately 20–40 full-validation configurations.
