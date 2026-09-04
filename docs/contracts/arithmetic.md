# Arithmetic contract

Status in supplied roadmap: **F1–F4 source-stated as frozen; project review not yet recorded**

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
