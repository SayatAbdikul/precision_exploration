# Phase 2 runtime strategy and workload breadth

Date: 2026-09-09. Scope: in repository.

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

## D2 — Open: provisional predecoded strategy

For both measured FP6 E3M2 and INT8 GEMM shapes, predecoded execution has the
lowest median latency on C++ and CUDA. The three strategies produce identical
output hashes. Predecoding has a setup cost; the generated evidence records
the number of repeated calls needed to amortize preparation of both operands.
It does not assume activation tensors can be reused across images.

These measurements support using the existing predecoded path for those
shapes. They do not establish the fastest strategy for every family and width:
there are only three samples per strategy, depthwise has no measured strategy
alternative, and the remaining-family matrix is a correctness/latency pilot.

The following profiling evidence is still required by Phase 2:

- achieved occupancy: `sm__warps_active.avg.pct_of_peak_sustained_active`;
- DRAM read/write bytes: `dram__bytes_read.sum` and `dram__bytes_write.sum`;
- kernel duration: `gpu__time_duration.sum`, paired with those counters to
  derive kernel DRAM bandwidth.

Nsight Systems records registers per thread and transfer timings. Neither
substitutes for those hardware counters. Nsight Compute returned
`ERR_NVGPUCTRPERM`, and the attempted privileged process required an interactive
password. No GPU counter policy was changed. The Systems trace was refreshed
on 2026-09-10: all six launches use the current engine source, and all five
retained artifact hashes verify. Collect the missing hardware counters against
that source before closing D2.

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
