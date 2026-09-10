# Phase 2 — Build and validate the exact inference engine

Scope: **in repository**  
Criticality: **critical path for all quality sweeps**  
Starts after: **WP1B plus accepted operator/arithmetic contracts; workloads available for network checks**  
Status: **in progress — exact engine implemented; native classifier checks and D2/data evidence being completed**

## Implementation record

See `docs/architecture/exact-engine.md` and
`results/summaries/phase2-final-verification.json` for the implemented
interfaces and source-frozen evidence. The reference engine, C++/CUDA
rational and binary kernels, detector/DFL lowering, shared-scale convolution,
full calibration artifacts, resumable workload execution, all-format matrix,
and generic primitive pilots are implemented.

As of 2026-09-10, classifier synthetic checks match all 49/100/140 layers,
the detector matches all 176 layers on eight native-resolution images, and
the 25-format C++/CUDA matrix matches the rational oracle. COCO payloads and
FP32 mAP reproduction are complete. The complete frozen ImageNet 10k evaluation
and 1k screening payloads are restored; training calibration recovery remains
incomplete. The detector's zero-mAP FP6 result is diagnosed as a strict-format
range failure, with independent FP32 lowering parity verified. D3 retains
EfficientNet as optional. GPU counter restrictions prevent achieved-occupancy
and DRAM-bandwidth profiling for D2, so the exit criterion remains open.

The focused Phase 2 test suites currently pass 215 CPU tests and 93 CUDA
tests. The final verification report validates recovered image hashes, saved
native execution records and source identities, and lists remaining failures
explicitly. Frozen dataset and prediction expectations remain unchanged.

## Numerical-engine tasks

- **N2.1** — Implement exact quantized Conv2D, 1×1 Conv, Linear, residual add, and depthwise Conv.
- **N2.2** — Implement bias in the accumulator domain.
- **N2.3** — Implement ReLU/ReLU6, pooling, hard-swish, and required nonlinear LUT semantics.
- **N2.4** — Implement the strict low-precision baseline graph.

## Backend tasks

- **I2.1** — Correctness-first multithreaded C++ backend.
- **I2.2** — CUDA GEMM/Conv plus a dedicated depthwise kernel.
- **I2.3** — Benchmark algorithmic, LUT, and predecoded execution on representative FP6 plus another family.
- **I2.4** — Validate C++ and CUDA bit for bit.
- **I2.5** — Check network-level layer outputs/logits on a fixed image set.

The optimization sequence preserves deterministic sequential accumulation and specializes kernels by W/A/accumulator/MAC semantics. Host `uint8` storage does not change actual datatype-width accounting.

## Generic hardware pilots

- **H2.1** — Synthesize representative INT, FP, Posit, and log primitives to validate flow behavior.
- **H2.2** — Pilot timing constraints and 0–3 pipeline stages.
- **H2.3** — Pilot vectorless power for broad-pruning methodology only.

## External context only

The roadmap assigns these to a separate MANT repository:

- **M2.1** — Make no candidate-driven redesign yet.
- **M2.2** — Prepare logging for later quality delta, cycles, stalls, traffic, and utilization.

No MANT changes or logging are implemented here.

## Decision D2 — Fast backend per datatype/width

Choose algorithmic, LUT, predecoded, or mixed execution from measured latency/MAC rate/bandwidth/occupancy and exhaustive correctness. This may change runtime and packing but must not change scientific results.

## Decision D3 — Main workload breadth

After measuring engine speed and compute budget, decide whether EfficientNet joins the main suite or remains optional. ResNet-18, MobileNetV2, MobileNetV3, and one tiny detector remain the core target stated by the roadmap.

Accepted 2026-09-09: retain EfficientNet as optional. Measured native detector
execution projects to 29.81 CUDA hours for one 1k configuration, before other
pipeline costs; no additional mandatory workload budget has been allocated.
See `docs/decisions/d2-d3-phase2-runtime-and-breadth.md` for the decision and
the limits of that projection.

## Validation ladder

- Exhaustive primitive input pairs where feasible.
- Dot products at several K values such as 3, 9, 27, 64, 128, 256, and 1024.
- Tiny deterministic Conv tensors.
- Full-layer C++/CUDA equality.
- Fixed-image network output/logit checks.

## Exit criterion

- All tested primitive and layer outputs match the oracle/reference bit for bit.
- Reference and optimized modes are identical.
- FP32 baselines are reproduced.
- Scheduler resumes, deduplicates, and retains failures.
- Pilot backend and generic hardware-flow data support D2/D3.
