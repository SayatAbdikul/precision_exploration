# Phase 2 runtime strategy and workload breadth

Dates: D3 accepted 2026-09-09; D2 accepted 2026-09-11. Scope: in repository.

This record implements the owner's instruction to finish Phase 2. It keeps
the existing four core workloads and records an evidence-based optional-workload
decision; it does not assume a new deadline or a larger compute allocation.

## D3 — Accepted: EfficientNet remains optional

The main suite remains ResNet-18, MobileNetV2, MobileNetV3 Large and YOLOv8n.
EfficientNet is not added to the mandatory suite. This is a workload-scope
decision, not a claim about EfficientNet quality or performance.

The eight frozen native-resolution detector images have median execution times
of 107.32 seconds per image on CUDA and 325.58 seconds on C++. At the measured
FP6/FP32 configuration, a 1,000-image detector screen projects to 29.81 CUDA
hours before calibration, graph setup, preprocessing and scoring. Assuming
that same throughput for all 25 accepted candidates would take 745.28 hours
for the detector alone. That last number is a planning scenario: the other
formats have different execution costs and have not all been timed as networks.

No project-wide wall-clock allowance has been specified. The budget policy is
therefore to reserve no additional mandatory workload while the existing core
already carries this measured cost. This decision does not claim the core
suite fits a particular budget. It avoids requiring an additional workload
without an allocated increment. The 32×32 classifier integration timings are
not extrapolated to ImageNet resolution, and no EfficientNet timing is invented.

Alternatives considered were adding EfficientNet immediately, benchmarking it
before any core screening, and retaining it as optional. The last option
preserves the established scope and gives the current implementation a
concrete workload set. Reopen D3 when core screening throughput and an explicit
incremental compute allowance support staging and timing EfficientNet.

## D2 — Accepted: predecoded strategy for measured FP6/INT8 GEMM shapes

For both measured FP6 E3M2 and INT8 GEMM shapes, predecoded execution has the
lowest median latency on C++ and CUDA. The three strategies produce identical
output hashes. Predecoding has a setup cost; the generated evidence records
the number of repeated calls needed to amortize preparation of both operands.
It does not assume activation tensors can be reused across images.

These measurements support using the existing predecoded path for those
shapes. They do not establish the fastest strategy for every family and width:
there are only three samples per strategy, depthwise has no measured strategy
alternative, and the remaining-family matrix is a correctness/latency pilot.

The following required profiling evidence is now measured and validated:

- achieved occupancy: `sm__warps_active.avg.pct_of_peak_sustained_active`;
- DRAM read/write bytes: `dram__bytes_read.sum` and `dram__bytes_write.sum`;
- kernel duration: `gpu__time_duration.sum`, paired with those counters to
  derive kernel DRAM bandwidth.

The owner authenticated process-only profiling on 2026-09-11. The saved
`counter-attempt-zawqo23w` capture completed at 04:47:52 UTC. All six launches,
input/output identities, engine source, raw CSV units and artifact hashes pass
validation. No GPU counter policy was changed. Measured values are:

| Format | Strategy | Achieved occupancy | DRAM bandwidth (GB/s) |
| --- | --- | ---: | ---: |
| FP6 E3M2 | Predecoded | 20.17% | 42.79 |
| FP6 E3M2 | Algorithmic | 20.15% | 41.00 |
| FP6 E3M2 | Lookup | 20.14% | 41.53 |
| INT8 | Predecoded | 20.72% | 57.68 |
| INT8 | Algorithmic | 20.69% | 55.98 |
| INT8 | Lookup | 20.70% | 57.47 |

Bandwidth is total DRAM read/write bytes divided by the paired Nsight Compute
kernel duration, using decimal GB/s. These are one profiled launch per
family/strategy at GEMM M=196, N=64, K=288. They close the counter evidence gate;
the repeated unprofiled benchmarks determine the latency recommendation.
They do not establish a statistical ranking from profiled duration alone.

D2 accepts the existing predecoded path for the measured convolution/pointwise
GEMM shapes, accounting for operand preparation and reuse. Other accepted
families retain their correctness-validated implementations pending strategy
measurements. Depthwise has only one measured implementation. This decision
changes no arithmetic, datatype, accumulator or output semantics.

## Hardware and accumulator boundaries

The H2 pilot evidence is complete for its stated scope: four primitive families
at internal pipeline stages 0–3, 16 exhaustive simulation/synthesis runs, and
48 cell-delay timing/vectorless-power runs at 2, 5 and 10 ns. This is
post-synthesis flow validation, with no extracted parasitics or workload power.
Its artifact hashes are checked by the decision-evidence generator.

Phase 2 implements independently configured accumulator formats, stored bias
in that domain and sequential Model C. The configuration contract rejects an
unresolved accumulator policy alias. Phase 3 N3.2 must resolve and validate a
sufficiently wide family-appropriate accumulator for every Experiment A job.
The current FP32/INT32/quire/FP64 validation paths and finite-domain bounds do
not confer family-wide Experiment A acceptance. Mapped bias headroom and
native-image output sensitivity remain explicit checks before those sweeps.

## Reproduce and inspect the evidence

Run `.venv/bin/python -m tools.analysis.phase2_decision_evidence` to write
`results/summaries/phase2-decision-evidence.json`. It validates retained image
records, hardware reports and numerical equality between strategy outputs;
records hashes of its evidence files; and marks stale engine identities.

The native-image and final microbenchmark measurements currently identify
engine source SHA-256
`e0a8c05b4a9dbfbb58af6a236e495e7a3d0fac34162db4dace3119308dedadeb`.
The generator retains that provenance even if subsequent implementation
changes make the measurements historical.
