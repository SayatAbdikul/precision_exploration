# FP6 execution performance review — 2026-09-28

## Finding

The current FP6 experiments use the generic 4,096-bit rational MAC backend for
every convolution/linear operator. This is the principal optimization target.
The graphs request FP64 accumulation, while the compact native dispatch admits
only INT32, FP16 and FP32. Unscaled dyadic FP6 therefore falls into a much more
general implementation than its numeric domain requires.

This was a code and saved-evidence review. No running experiment, source guard,
native library, graph, recipe or prediction was changed; no new inference or
benchmark was launched. The timing snapshot and relevant source hashes are in
`results/summaries/fp6-performance-audit-2026-09-28.json`.

## Measured evidence

| Case and arm | Saved new images | Median execution time/image |
| --- | ---: | ---: |
| R18 FP6 E3M2 strict CPU | 7 | 3,989.07 s / 66.48 min |
| R18 FP6 E2M3 strict CPU | 8 | 3,227.05 s / 53.78 min |
| R18 FP6 E2M3 strict CUDA | 8 | 724.40 s / 12.07 min |
| R18 FP6 E2M3 matched FP32 control CPU | 6 | approximately 180 s / 3 min |

Input preparation is approximately 0.03 seconds/image. The recorded execution
time includes host operators, diagnostics and native reductions; it is not
kernel-only time. Runs share the workstation, so these are observed campaign
latencies rather than isolated hardware benchmarks. CPU/control comparisons
also change arithmetic and execution path, so their ratio is not a measured
speedup for a proposed exact implementation.

All 21 strict MAC operators in the saved FP6 records report
`rational_grid_4096`. E2M3 control reports
`matched_fp32_sequential_binary_predecoded`.

The first strict CPU image additionally runs the legacy executor before its
checkpoint is saved. This separate compatibility check took 3,261.81 seconds
for E2M3 and 4,011.62 seconds for E3M2. It explains approximately another hour of
first-image latency; it is not repeated for every image.

A host snapshot showed an RTX 4070 Ti and both current numerical workers using
roughly 3–4 CPU cores. GPU utilization at that instant was 28% while the workers
were in CPU stages. This does not establish utilization during the earlier
12-minute CUDA images or quantify contention from the concurrent physical job.

## Why the implementation is slow

1. `public/inference/native.py:265` excludes FP64 from `binary_admitted`.
   `flex_gemm` and `flex_conv2d` consequently select rational execution.
2. `public/inference/rational.py:35–58` sizes the grid for the full FP64 range,
   including the minimum subnormal quantum `2^-1074`, rather than the actual
   FP6 operand range. It chooses 128 32-bit limbs, or 4,096 bits.
3. `public/inference/cpp/rational_binary.h:89–108` reconstructs the accumulator
   in that wide grid, multiplies, adds/subtracts, normalizes and performs generic
   division/RNE for each MAC. ResNet18 performs **1,814,073,344 MACs/image**.
4. Each operand is packed as 128 limbs plus sign/kind: **520 bytes/value**.
   `native.py:305` also constructs convolution patches through Python/Fraction
   loops. The graph has 14,689,536 expanded patch entries, representing roughly
   **7.11 GiB of cumulative packed patch payload per image**, excluding weights.
   This is a representation-size calculation, not measured DRAM traffic or peak
   resident memory.
5. `public/cuda/kernels/exact_gemm.cu:109–134` assigns one thread to each output
   and performs its K reduction sequentially, with 32-thread blocks. Each call
   allocates buffers, copies operands, returns outputs synchronously to the host,
   then frees buffers. Large local arrays may cause occupancy/spill problems;
   compiler resource reports and profiling are needed to quantify them.
6. `public/inference/operators/dispatch.py:44–73` has no FP64 fast store. Bias,
   activation and output conversion use the scalar reference path over
   2,484,712 MAC output values/image. Bias is encoded repeatedly by position.
   `tools/exact_execution_v2/engine.py:96` also sends FP6 residual operations to
   the generic reference path.

CPU execution is already OpenMP-parallel over independent outputs
(`public/inference/cpp/backend.cpp:46`). Increasing threads alone leaves the
wide-arithmetic and packing costs in place. Both FP6 formats take the same broad
fallback; the precise reason for their CPU timing difference needs a per-layer
profile, rather than attribution to mantissa/exponent width alone.

## Recommended optimization sequence

### 1. Versioned native sequential FP64 reduction

Implement a gated CPU/CUDA path for operands exactly representable in FP64.
Use explicit correctly rounded FP64 FMA in the original reduction order, with
parallelism across independent outputs. Return the original FP64 accumulator
bit patterns and initially retain the existing bias/activation/output store.
Use direct convolution addressing and compact input/weight representations.

CPU `std::fma` and CUDA `__fma_rn(double, double, double)` are suitable building
blocks under verified RNE and compiler/runtime settings. NVIDIA documents the
CUDA intrinsic as one multiply-add with nearest-even rounding in its
[Double Precision Intrinsics reference](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html).

This preserves the FP64 arithmetic contract while removing the 4,096-bit
emulator from its finite dyadic domain. Keep a fallback for unsupported domains
and explicitly test NaN canonicalization, signed zeros and subnormal behavior.
Do not enable fast-math or silently move the separately rounded bias into FMA.

### 2. Certified integer-grid MACs for the frozen FP6/FP7 graphs

The static codebook/weight audit gives these conservative bounds:

| Format | Finite operand representation | Product quantum | Largest prefix/subset bound in product units, using frozen weights |
| --- | --- | --- | ---: |
| FP6 E2M3 | integer / 8; integer magnitude <=56 | 1/64 | 419,160 |
| FP6 E3M2 | integer / 16; integer magnitude <=384 | 1/256 | 5,824,128 |
| FP7 E3M3 | integer / 32; integer magnitude <=896 | 1/1024 | 27,199,872 |

Each graph has 21 MAC nodes and maximum K=4,608. The channel bound is
`maximum_activation_integer × sum(abs(stored_weight_integer))`; it covers every
prefix and every subset sum even with cancellation. All three actual-weight
bounds fit INT32 and are well below FP64's exact-integer limit. INT64 is a simple
wider first implementation. Both FP6 formats also have format-wide bounds below
INT32 at this K; FP7 needs its weight-specific certificate or wider accumulation.

Compute the exact integer sum and reconstruct the FP64 value with a power-of-two
scale. Because every original FP64 prefix is exact in these certified domains,
grouped/tree reduction can then be justified mathematically. This is a stronger
optimization than assuming an ordinary parallel floating reduction is harmless.
E2M3 scaled operands fit INT8; E3M2 and FP7 need wider operands or a separately
verified decomposition.

Existing proof building blocks are
`public/analysis/phase3/finite_fp64_nonmac.py:17` (`exact_grid`) and
`development/acceptance_proofs/dyadic_fp_mac_store.py:71–112` (quantum,
actual-weight prefix bound and bias/store threshold checks). The retained proof
summary reports 21 MACs and no pending channels for these three R18 graphs.
That supports implementing a new kernel; it does not certify an unwritten
implementation.

Admission must bind the actual graph, weights, manifests, geometry and unscaled
encodings. Reject/fallback on nonfinite codes and separately validate signed-zero
and padding behavior. Keep stored bias, activation, output quantization and
diagnostic intermediate values unchanged.

### 3. Accelerate post-operations and remove host/device overhead

- Cache oracle-encoded FP64 biases per channel.
- Add a verified FP64 store and low-bit quantizer; never copy the existing
  FP16/FP32 fast store's cast through FP32, which can introduce double rounding.
- Generate exhaustive oracle-based FP6 residual/alignment tables, preserving
  intermediate quantizer events and signed-zero/NaN policies. A 64×64 pair
  table is small.
- Keep compact immutable weights/decode tables on device and reuse workspaces.
  Later, keep activations device-resident once exact post-operations and sampled
  diagnostics have corresponding implementations.
- Tune tiling, memory access and block size after removing the wide fallback.
  Profile allocation/copy, native reduction, store, residual and diagnostics
  separately; do not optimize only a kernel microbenchmark.

These priorities are consistent with NVIDIA's guidance to minimize host/device
transfers and assess register/local-memory pressure in the
[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).

### 4. Enforce cost gates earlier in the experiment controller

`tools/breadth_study/useful_quality.py:466–485` uses eight images for normal-run
native admission; it does not first invoke the available one-image probe.
It waits for the full pilot batch, then runs admitted 32-image panels. The
24-worker-hour subtraction is applied only when allocating 128-image extensions.
Thus the initial pilots and 32-image stage can already exceed the advertised
total budget.

Use one-image timing/compatibility probes before launching eight-image gates,
reserve predicted costs before each subsequent stage, and schedule ready admitted
cases without waiting for every slow candidate's CPU pilot. Separate CPU and GPU
work queues within the resource/lease rules so CPU admission work does not leave
GPU capacity unused. Preserve the required eight-image gate; do not simply skip
CPU verification to appear faster.

At the observed E3M2 CPU rate, eight images plus its first-image legacy check
cost approximately **10 worker-hours** before its CUDA/control passes. At the
observed E2M3 CUDA rate, the strict arm alone costs approximately **6.4 hours for
32 images** or **25.8 hours for 128**. These are simple serial projections, not
new benchmarks. Optimizing before further expansion has high value.

## Validation and acceptance

1. Record per-layer host and CUDA-event timings on bounded saved/synthetic
   tensors, including preparation, kernel, transfers, store and diagnostics.
2. Check finite operand pairs, cancellation, overflow boundaries, rounding ties,
   signed zeros and nonfinite fallback against the unchanged rational oracle.
3. Test small convolution/GEMM shapes, padding, stride, dilation and groups;
   for integer reduction reproduce graph/weight-bound certificates.
4. Require identical FP64 accumulator bits and unchanged layer/output codes,
   top-five lists and diagnostic signatures, then complete the eight-image
   native compatibility gate under a new implementation identity.
5. Benchmark isolated and campaign-concurrent execution separately and report
   full per-image time and memory. Retained checkpoints remain tied to their
   original execution sources; migration/reuse needs explicit compatibility.

Ordinary FP32/TF32/framework convolution is not an automatic replacement for the
strict arm. FP7 lacks a global FP32 exact-prefix bound under the actual weights;
even FP6 bounds do not certify a particular tensor-core kernel's internal
arithmetic, special cases or stores. A specific implementation needs proof and
conformance before use.

No optimized speedup has yet been measured. The dispatch/representation
bottleneck is established, while its share of end-to-end time and the attainable
speedup require the bounded profiling and implementation above.

Separate analysis issue: `tools/analysis/useful_quality_e1.py:61` converts every
decoded code to `Fraction`, including the FP6 reserved NaN code. Its arithmetic
envelope helper needs finite-code filtering plus explicit nonfinite handling
before being used to certify these bounds.
