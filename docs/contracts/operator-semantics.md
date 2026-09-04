# Operator-semantics contract

Status in supplied roadmap: **required by N0.4–N0.5; project review not yet recorded**

## Execution regimes

| Regime | Definition |
|---|---|
| Strict low precision | Candidate precision throughout all reasonably quantizable operators; primary baseline |
| Practical low precision | A small, fixed set of higher-precision exceptions with both recovered quality and hardware overhead reported; secondary ablation |

## Operator table

| Operator/layer | Baseline behavior |
|---|---|
| Conv2D and 1×1 Conv | Candidate W/A formats plus configured accumulator/MAC model |
| Depthwise Conv | Candidate precision; dedicated exact kernel |
| Linear/FC | Candidate precision |
| BatchNorm | Fold into preceding convolution before calibration/PTQ |
| Bias | Represent and add in accumulator domain |
| Residual add | Explicit scale/domain alignment followed by defined quantized add and output rounding |
| Elementwise multiply | Candidate precision with explicit output requantization |
| ReLU | Exact clamp at zero in the quantized domain |
| ReLU6 | Exact clamp to `[0, 6]` in the quantized domain |
| MaxPool | Exact comparison in the datatype |
| AveragePool | Accumulator-precision sum, divide/scale, then requantize |
| Hard-swish | Bit-exact add, clamp, multiply, and scale decomposition |
| SiLU/sigmoid/tanh | Datatype-specific LUT |
| Softmax | Omit for Top-1/Top-5 when logits suffice |
| YOLO NMS | High-precision postprocessing outside the datatype hardware study |

## BatchNorm folding

For `y = W*x + b` followed by BatchNorm:

```text
alpha = gamma / sqrt(variance + epsilon)
W'    = alpha * W
b'    = alpha * (b - mean) + beta
```

Quantize `W'` and `b'` only after the deployment graph is frozen. Store the folded graph hash.

## Bias

Bias storage precision equals accumulator precision in the baseline because it contributes once per output channel and aggressive operand-width bias quantization adds avoidable error.

For integer quantization with `x = s_x*q_x` and `w = s_w*q_w`, use the product/accumulator scale:

```text
s_b[c] = s_x * s_w[c]
q_b[c] ~= b[c] / s_b[c]
```

Add bias once after reduction and before activation/requantization. Finalist-only ablations may compare operand-precision bias, accumulator-precision bias, and a high-precision quality upper bound.

## Residual alignment

Residual branches may have different numerical domains/scales. Convert both inputs into an explicitly selected output/add domain, perform the configured quantized addition, then round to the residual output datatype. Conversion and scale-alignment hardware cost is included in generic RTL/system analysis.

## Nonlinear LUTs

For inputs no wider than 8 bits, a full nonlinear table has at most 256 entries:

```text
LUT_F[code] = Q_F(f(decode_F(code)))
```

The manifest/operator version identifies LUT contents. Any generic hardware implementation includes LUT area, latency, and energy.

## First and last layers

The strict baseline applies the candidate datatype to first, internal, and last layers. A secondary practical ablation may use:

```text
first = 8 bit, internal = candidate precision, last = 8 bit
```

Both quality recovery and complete generic hardware overhead must be reported.

## Determinism and storage

Encoded values at 2–8 bits may initially occupy one `uint8` host container. This is a simulation representation only; hardware accounting uses true datatype width. Encoded and predecoded execution modes are permitted only when bit-identical. Initial tensor layout is NCHW; NHWC is considered only after profiling and remains numerically identical.
