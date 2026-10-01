# Exact FP64 post-operations: bounded component evidence

The separately versioned `tools/exact_execution_v3/postops.py` provides a finite,
unscaled FP6/FP7 FP64 output store and oracle-generated floating residual tables.
It does not change the frozen v1 executor. A new prepared graph can use
`FP64Stores.store(...)` when it returns a tensor and retain its original store
when it returns `None`. `FloatingResiduals.add(...)` handles the selected
unscaled FP6/FP7 FP64 adds; it delegates integers to the prepared v2 residual
path and unsupported/special inputs to the original scalar operator.

The store interprets native accumulator codes as FP64 bits, caches each bias
after a separate oracle FP64 encode, performs one FP64 RNE add, applies the
declared activation, and directly quantizes FP64 values to the low-bit format.
It never casts through FP32. It falls back for nonfinite accumulators/biases or
an overflowing bias result, unsupported encodings/scales, and other activations.
The residual tables retain both alignment quantization events and the final
sum event, with original exact pre-quantization values at every position.

Focused oracle comparisons cover every finite stored-code pair of FP6 E2M3,
FP6 E3M2 and FP7 E3M3; all three residual diagnostic events; every low-bit
finite value and nearest-even midpoint with adjacent FP64 neighbors; bias,
ReLU/ReLU6, saturation, subnormal inputs, and signed zero. NaN source codes
exercise the original residual fallback without emitting an extra event. The
post-operations and existing prepared-execution unit tests pass together:

```text
.venv/bin/python -m pytest tests/unit/test_exact_execution_v3_postops.py tests/unit/test_useful_quality.py tests/unit/test_exact_execution_v2.py -q
34 passed
```

An isolated synthetic CPU measurement on 16,384 FP6 E2M3 values gave a warm
store time of 0.0034 s versus 0.460 s for the scalar oracle, and a warm finite
residual time of 0.0036 s versus 0.657 s. The residual table's first use took
0.085 s. These are component timings under the contemporaneous workstation
load, not a full-image speedup claim. Full-image and diagnostic timing remain
part of the v3 runtime admission.

The independent useful-quality arithmetic audit now filters nonfinite codebook
entries before constructing finite-grid bounds. Actual nonfinite stored weights
or stored inputs reject the bound; dynamic MAC inputs carry an explicit
finiteness precondition. All 21 MAC nodes in each selected ResNet18 FP6/FP7
graph reproduced their previously observed finite-domain maximum prefix units:
419,160 (E2M3), 5,824,128 (E3M2), and 27,199,872 (FP7 E3M3).
