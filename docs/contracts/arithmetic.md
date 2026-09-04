# Arithmetic contract

Version: **1.0.0**
Status: **accepted for Phase 0**
F1–F4 accepted by project owner: **2026-09-04**

## Required format interface

Every datatype manifest and backend implements the same conceptual interface:

```text
NumberFormat
    encode(fp32)       -> low-precision code
    decode(code)       -> high-precision reference value
    add(a, b)          -> rounded datatype result
    mul(a, b)          -> rounded datatype result
    requantize(x)      -> target storage format
    convert(src, dst)  -> explicit format conversion
```

The manifest defines bit width/encoding, signedness, family parameters, bias, zero/subnormal/NaN/infinity behavior, overflow, underflow, rounding, external/intrinsic scale, block size/axis, and codebook values where applicable.

### Manifest semantic validation

JSON Schema validates structure; the Phase 1 manifest validator must additionally enforce these cross-field rules:

- Weight, activation, and output-storage manifests use at most 8 bits; accumulator manifests may be wider.
- A conventional signed float layout satisfies `bits = 1 + exp_bits + mantissa_bits`; an unsigned layout omits the sign bit.
- Fixed-point integer/fraction allocation plus any sign bit fits the declared width.
- A complete codebook contains exactly `2^bits` unique entries in code order.
- A `none` scale has `none` granularity; native MX/BFP shared scale has block granularity and all required block metadata.
- `exp_bits`, `mantissa_bits`, bias, special values, overflow, and underflow form one internally consistent encoding.
- Standard FP8/MX/Posit labels match their published semantics; a differing encoding receives a distinct custom name.
- Binary/ternary and logarithmic encodings enumerate all code meanings, including zero and invalid/reserved codes.
- An accumulator policy alias is resolved to a concrete manifest before execution.

## Independent format dimensions

Weight format, activation format, accumulator format, and output-storage format are independently configurable. A uniform W=A search may be used to control the first sweep, but the software architecture must not assume equality. The full numerical identity is at least:

```text
(weight format, activation format, accumulator format,
 output format, scaling policy, MAC model, reduction order)
```

## MAC models

Let `F` be the product/operand datatype and `A` the accumulator datatype.

### Model A — native arithmetic

```text
p_i = Round_F(x_i * w_i)
a_i = Round_F(a_(i-1) + p_i)
```

Product and accumulation both use operand precision. This represents the lowest-precision/lowest-complexity path and can accumulate large error at 4–6 bits.

### Model B — rounded product plus wider accumulator

```text
p_i = Round_F(x_i * w_i)
a_i = Round_A(a_(i-1) + Convert_A(p_i))
```

The product is rounded to the operand datatype before conversion into a wider accumulator.

### Model C — full-product/FMA-style accumulation

```text
a_i = Round_A(a_(i-1) + Exact(x_i * w_i))
```

The product enters the accumulator without first being rounded back to operand precision. The roadmap identifies Model C as the canonical controlled Experiment A baseline so premature product rounding does not dominate the format comparison. Models A/B/C are compared later for promoted/finalist configurations.

## Accumulator baseline and sweep

Experiment A uses a sufficiently wide, family-appropriate accumulator to measure representation quality before intentionally squeezing accumulation precision. Later sweeps include:

| Operand family | Candidate accumulators | Reference/notes |
|---|---|---|
| INT/fixed point | native, 8/10/12/16 where meaningful, INT24, INT32 | INT32 is the high-precision reference |
| FP/minifloat | native, widened FP, medium custom FP, FP16 | Candidate E/M layouts remain a pilot decision |
| Posit | native, wider Posit, quire | Quire must be considered |
| MX/BFP | wide partial sum plus scaled-block accumulation | Scale cost is included |
| Logarithmic | LNS accumulation or log multiply plus linear accumulator | Hybrid may be more practical |

The plan proposes an initial, non-frozen FP progression of FP6(E3M2) → FP8(E4M3) → FP12(E5M6) → FP16(E5M10).

## Product, conversion, and rounding points

Every configuration states:

- input and weight formats;
- exact or rounded product precision;
- accumulator format;
- conversion into accumulator domain;
- rounding after accumulation;
- overflow/underflow/saturation behavior at each point;
- bias representation and addition point;
- output activation/requantization datatype and rounding.

No implicit conversion is permitted. RNE is used where required by the format/baseline; alternative rounding modes are explicit later ablations.

## Accumulation order

The primary exact inference baseline uses deterministic sequential reduction:

```text
for k = 0, 1, ..., K - 1:
    acc = Add_A(acc, Product(x[k], w[k]))
```

Parallelism is across independent outputs, not within the reduction tree. Tree reduction is a separately identified architecture-sensitive ablation because rounded low-precision addition is not associative.

## Bias and requantization

Bias is added once after dot-product reduction, in the accumulator domain, before activation and output requantization:

```text
a      = sum_i x_i * w_i
y      = Round_A(a + bias)
output = Q_Fout(activation(y))
```

Operator results remain in the accumulator domain until they are stored in the output activation datatype. The convert/requantize rule is explicit.

## Worked Model C witness

This illustrative dot product uses the plan's FP6 E3M2 format for W/A/output and an explicit FP16 accumulator manifest. The selected values are exactly representable so the witness exposes domains and round points without depending on a later D1 candidate decision.

```text
x = [ 1.5, -0.5 ]          activation domain FP6
w = [ 0.5,  2.0 ]          weight domain FP6
b = 0.5                     accumulator domain FP16

acc_0 = FP16(0)

exact_product_0 = Exact(1.5 * 0.5) = 0.75
acc_1 = Round_FP16(acc_0 + exact_product_0) = 0.75

exact_product_1 = Exact(-0.5 * 2.0) = -1.0
acc_2 = Round_FP16(acc_1 + exact_product_1) = -0.25

biased = Round_FP16(acc_2 + Convert_FP16(b)) = 0.25
activated = ReLU(biased) = 0.25
output = Requantize_FP6(activated) = 0.25
```

The product is never rounded to FP6 before accumulation. Bias is added once after reduction. The only output-domain round occurs at operator storage. A conformance implementation must additionally record/check the encoded codes produced by the accepted manifest oracle.

## Resolving “family-appropriate wide”

`family_appropriate_wide` is a Phase 0 policy, not an implicit runtime datatype. Before any job executes, configuration expansion must resolve it to a concrete accumulator manifest and include that manifest hash in the executable configuration. The plan's high-quality reference anchors are:

- INT/fixed point: INT32;
- FP/minifloat: FP16 or an explicitly justified sufficiently wide custom FP;
- Posit: quire or explicitly widened Posit;
- MX/BFP: wide scaled partial-sum domain;
- logarithmic: defined LNS accumulator or log-product plus linear accumulator.

Phase 1 pilot work may refine the efficient sweep, but it cannot run an unresolved accumulator alias.

## Deliberately open numerical dimensions

Phase 0 does not choose the D1 format set, the winning accumulator, FP accumulator E/M layouts, Models A/B/C finalist trade-off, tree-reduction architecture, Experiment B scale policy, W/A combinations, signedness, block size, bias ablation, first/last-layer exception, or approximate-arithmetic branch.

## Exact oracle

High-precision host arithmetic may be used internally to calculate the correctly rounded target result. It is an oracle, not unrestricted FP32 accumulation. For every n-bit binary operation, exhaust all `2^n * 2^n` input pairs where feasible:

| Width | Codes | Pairs/operator |
|---:|---:|---:|
| 4 | 16 | 256 |
| 5 | 32 | 1,024 |
| 6 | 64 | 4,096 |
| 7 | 128 | 16,384 |
| 8 | 256 | 65,536 |

Optimized C++/CUDA and later RTL implementations are accepted only if they match the oracle bit for bit under the same manifest.
