# Exact inference engine — implementation version 2.0.0

Status: **Phase 2 complete — 2026-09-11**. This document specifies the implemented
execution interfaces and their limits. It supplements the accepted arithmetic
and operator contracts. D3 retains EfficientNet as optional; D2 is accepted
for the measured FP6/INT8 GEMM scope. Final verification has no remaining gates.
The Phase 0 arithmetic and scaling policies still apply.

## Execution and identities

`public/inference/tensor.py` stores shape, integer codes and an `Encoding`.
An encoding names a manifest and explicitly addresses scales per tensor,
per axis, or per K block. MX/BFP incomplete blocks own one scale for the valid
values. A block scale grid includes every non-K dimension; it is not a single
scale vector shared accidentally across rows. Tensor documents preserve scale
values as decimal strings. Encoded storage is limited to eight bits; it is not
the accumulator representation or a physical-memory size estimate.

`public/quantization/graph/executable.py` contains a small versioned graph IR.
Graph identities include constants, operation attributes, manifest hashes,
scale metadata, nonlinear tables and provenance. Runtime input tensors carry
their own shape and encoding. Unsupported operations, invalid edges and stale
manifest hashes fail explicitly. No PyTorch numerical fallback is installed.

`public/inference/conformance_job.py` adds an explicitly synthetic job schema
to the existing registry. It validates input/graph/source identities and runs
the real executor through lease-fenced scheduling. Its metrics cannot claim
ImageNet/COCO quality. `public/inference/workload_job.py` adds the separate
strict workload schema: it binds calibration artifacts, frozen payload hashes,
model identity, graph identity and runtime source identity, and writes
per-image resumable records. A failed attempt cannot overwrite a recovered
lease or a completed image record.

## Arithmetic

Finite decoded oracle values are represented as rational numbers. Model C
forms an exact product, adds it to the decoded accumulator, and rounds the
sum to the accumulator after every K step. Bias is encoded in the accumulator
domain and added exactly once after reduction. Conversion to the output
encoding is explicit. Reduction order is channel, kernel row, kernel column;
threads work on independent outputs.

For logarithmic operands, the rational reference represents the existing
oracle's 200-digit decoded approximation. It is not a mathematical claim of
exact irrational arithmetic. Nonfinite behavior and signed zero are explicit.

The new constant-memory accumulator encoder supports signed/unsigned integer,
fixed-point, and IEEE-like binary floating layouts without enumerating all
codes. Concrete manifests are provided for INT32, FP16, FP32, FP64 and a
64-bit fixed product accumulator for unscaled Posit8 ES1. That fixed format
has 24 fractional bits and saturates; it is not a NaR-capable standard quire.
These are available execution domains, **not an accepted family-wide wide
accumulator selection**. Worst-case bounds and workload error/overflow
evidence must establish the final Experiment A policies.

`tools.analysis.phase2_accumulator_bounds` records conservative finite,
unscaled dot-product grid bounds for all 25 accepted formats at eight K
values. It separates integer storage width from floating significand width
and identifies non-dyadic oracle grids. It excludes bias and cross-block
scale composition; it does not silently resolve the wide-accumulator policy.

## Operator rounding points

- Conv/Linear: sequential Model C, stored accumulator bias, optional fused
  identity/ReLU/ReLU6, then output encode. Integer mapping uses the product
  scale for each output channel.
- Padding is an exact mathematical zero, including for binary formats that
  cannot store zero. It is not the nearest encoded value to zero.
- Residual add: explicitly encode both branches into the alignment domain,
  add in the declared accumulator, and encode the output.
- Elementwise multiplication: exact product followed by output encoding.
  Broadcast multiplication has explicit compatible dimensions.
- ReLU/ReLU6: clamp decoded values, then encode the result.
- Hard-swish: accumulator-rounded add of 3, clamp to [0,6],
  accumulator-rounded multiply by the original input, accumulator-rounded
  division by 6, then output encode. Hard-sigmoid omits the input multiply.
- Average pooling: sum in the accumulator, then divide and requantize. The
  integer path carries the exact rational division into output encoding;
  the floating path rounds division to its declared accumulator. Padding
  inclusion is explicit. Global adaptive average pooling is implemented.
- SiLU/sigmoid/tanh: full input-code LUTs bound to scalar input/output domains.
  LUT contents are regenerated and checked against the declared function.
  A caller must define conversion into that input domain explicitly.

## Native backends

The C++ and CUDA implementations share `exact_binary.h`. A 384-bit integer
intermediate in units of 2^-192 preserves exact products and sums for the
admitted binary operand range. Explicit guard/sticky/tie logic rounds to
FP16, FP32, or saturating INT32. Native inputs are range-checked before entry.
NaN and infinity are supported by the floating accumulators.

The C++ backend parallelizes outputs with OpenMP. CUDA has independent-output
GEMM, general convolution, and a dedicated depthwise specialization. Both
support a separately measured algorithmic decoder, decode-LUT mode and
predecoded execution for initial INT/FP pilots. These LUTs decode operands;
they do not replace Model C with narrow multiplication tables.

Bulk table expansion avoids repeated Python scalar decoding for native
tensors. The direct-float parameter encoder is separately tested at every
rounding boundary and its immediate neighbours. Non-dyadic formats/scales
cannot silently pass through the binary native kernel. Bias, output encoding
and non-reduction operations currently use the common exact host operator
implementation. CUDA results are therefore measured with host preparation,
postprocessing, allocations and transfers, not described as GPU-only latency.

## Workload adapters and current limits

The FX importer handles the mandatory torchvision classifier operation set:
Conv, Linear, residuals, ReLU/ReLU6, hard-swish/hard-sigmoid, pooling, flatten,
and squeeze/excitation broadcast multiplication. It folds BN before encoding
weights. Mapping-format activation encodings must be supplied from frozen
calibration data by node name. It never learns scales from evaluation input.

The integration runs use real frozen checkpoints. C++/CUDA layer equality was
checked on eight images at each core model's native resolution: 49 ResNet18,
100 MobileNetV2, 140 MobileNetV3 Large, and 176 YOLO layers per image, for
3,720 matching layer comparisons. Classifier output tensors also match.
COCO's 2,000-image calibration and 5,000-image evaluation payloads are
restored and hash-verified. The complete frozen ImageNet evaluation (10,000)
and screening (1,000) payloads were restored on 2026-09-10. ImageNet training
recovery (2,000 images) and all eight core-model FP6 E3M2/INT8 calibration
artifacts also completed on 2026-09-10. The imported Phase 1 dataset export
under `data/data` matches all six frozen selections; its audit is recorded in
`results/summaries/phase2-phase1-import-audit.json`.

`tools.run.phase2_fixed_images` implements the classifier fixed-image gate
at each model's frozen native resolution. It selects the first eight images
by SHA256 from the frozen ImageNet screen list, verifies every payload,
checkpoint, source graph and preprocessing hash, then records FP32 logits
and compares all C++/CUDA layer hashes and outputs. All three classifier runs
completed and their reports were verified on 2026-09-10. Each completed image/backend is saved with a content hash
and a job identity under `artifacts/fixed_image_runs/`; rerunning the same command
verifies and reuses those checkpoints. The final report is published only after
all eight images match. A dataset root may be supplied explicitly; recovery
uses the original image identities.
The small conformance set does not reproduce the full Phase 1 quality baseline.

The original frozen YOLO prediction artifact supplied on 2026-09-11 matches
its Phase 1 SHA-256 exactly. All four baseline prediction artifacts now verify.
Official COCOeval rescoring also exactly reproduces the frozen mAP50–95
(0.3735112134743026) and mAP50 (0.5256168716683333); see
`results/summaries/phase2-phase1-artifact-import-audit.json`.
Kernel bandwidth and achieved occupancy are also measured and validated for
all six Phase 2 profiling launches. The owner authenticated process-only
Nsight Compute collection on 2026-09-11, closing D2 and the final Phase 2 gate.

The [remaining-work checklist](../roadmap/phases/phase-02-remaining-work.md)
records completion and zero remaining Phase 2 work. The final
verification report explicitly checks calibration coverage for all four models.

The zero-mAP FP6 detector probe is now diagnosed in
`results/summaries/phase2-detector-diagnosis.json`. Its unscaled format cannot
store pixel coordinates above 24, and probabilities below 0.03125 round to
zero. All 1,408 native layer comparisons agree, and independent FP32 lowering
matches the frozen model on all eight images. This is a strict candidate's
range failure. Adding output scaling or higher precision would require a
separately identified experimental policy.

Per-graph wide-accumulator acceptance, including mapped bias headroom, is
required before Phase 3 Experiment A sweeps. The explicit Phase 2 accumulator
validation paths do not grant that family-wide acceptance. The D2/D3 evidence
and this boundary are recorded in
`docs/decisions/d2-d3-phase2-runtime-and-breadth.md`.

## Measured implementation pilots (2026-09-09)

All three classifier architecture checks were refreshed against the same
engine source identity. Their 32x32 synthetic inputs matched at every layer:
ResNet-18 (49 nodes), MobileNetV2 (100), and MobileNetV3-Large (140).

The benchmark summaries now include convolution GEMM (196x64 outputs, K=288),
pointwise GEMM (784x32 outputs, K=32), and 32-channel 28x28 depthwise 3x3 Conv.
FP6 uses FP32 accumulation and INT8 uses INT32. Algorithmic, decode-LUT and
predecoded GEMM strategies agree bit for bit; C++/CUDA benchmark output hashes
also agree. Predecoded tensor preparation is outside the repeated timing,
and reported setup time makes that reuse explicit.

For the convolution GEMM case, three-repeat median times including bindings,
allocations and transfers were approximately 213 ms C++ / 9.7 ms CUDA for
cached predecoded FP6, and 221 ms / 8.0 ms for INT8 in the final benchmarks.
These layer pilots support D2 within the documented operand-reuse assumptions;
they are not network speed estimates.

Nsight Systems separately traced one launch of each family/strategy and
recorded kernel durations and transfer events. These single-launch timings
are not statistical rankings. Authenticated Nsight Compute collection on
2026-09-11 measured achieved occupancy of 20.14–20.72% and total DRAM bandwidth
of 41.00–57.68 GB/s across the six launches. Raw input/output/source identities,
metric units and artifact hashes validate. The GPU's counter-access policy
was not changed. See the D2 decision record for each strategy's measurements.

Hardware evidence covers four primitive families, 0–3 added pipeline stages,
and 10/5/2 ns clock constraints: 16 exhaustive RTL/synthesis pilots and 48
STA/vectorless-power runs. All timing runs meet the specified setup and hold
constraints. Extra output stages do not retime the primitive's combinational
logic, so this is a flow/pipeline-interface pilot. Area includes the primitive
and register wrapper. The iPA build reports zero net switching power; its
cell internal/leakage estimates remain limited vectorless flow evidence,
not workload energy or routed PPA.

## Reproduction

The local environment uses Python 3.12, torch 2.3.0+cpu and torchvision
0.18.0+cpu. CUDA is built independently with nvcc and a supported C++ host
compiler. The local GPU is an RTX 3060 Ti; GPU execution requires device
access in the process running the command.

The `engine` package extra declares NumPy for bulk native tensor preparation;
the `phase2` extra pins the model frameworks used by frozen graph restoration.
Use Python 3.12 for that pinned workload environment. CPU-only torch wheels
can be installed from PyTorch's CPU index; the CUDA engine uses nvcc directly
and does not depend on a CUDA-enabled torch build.

```sh
.venv/bin/python -m tools.setup.check_phase2_environment
.venv/bin/python -m tools.run.phase2_engine build --backend cpp
.venv/bin/python -m tools.run.phase2_engine build --backend cuda
OMP_NUM_THREADS=4 .venv/bin/python -m pytest tests/unit tests/integration/test_exact_graph.py tests/conformance/test_native_engine.py -q
PRECISION_TEST_BACKEND=cuda OMP_NUM_THREADS=4 .venv/bin/python -m pytest tests/conformance/test_native_engine.py -q
OMP_NUM_THREADS=4 .venv/bin/python -m tools.run.phase2_engine conformance --backend cuda
OMP_NUM_THREADS=4 .venv/bin/python -m tools.run.phase2_engine benchmark --backend cuda
OMP_NUM_THREADS=4 .venv/bin/python -m tools.run.phase2_workload_smoke --model resnet18 --input-size 32 --backends cpp cuda
.venv/bin/python -m tools.run.phase2_fixed_images --model resnet18 --preflight
.venv/bin/python -m tools.analysis.phase2_accumulator_bounds
```

Checkpoints and classifier/detector graph payloads can be restored with
`tools.setup.restore_phase2_checkpoints` and `tools.setup.restore_phase2_graphs`.
Those tools verify frozen hashes and do not refreeze model manifests. The
restored 725 primitive tables reproduce the existing tracked table index.
`tools.setup.restore_phase2_annotations` restored both official COCO annotation
files with their frozen hashes. COCO and ImageNet validation images are now
available. `tools.setup.restore_phase1_classifier_predictions` replays frozen
FP32 classifiers and publishes original artifacts only when their byte hashes
match; attempts that differ remain separate reproduction evidence.

All three classifier prediction files now reproduce their frozen byte hashes.
MobileNetV3 requires the native CPU convolution path for the original top-five
ordering. A second full COCO replay with native CPU kernels still differs
from the original prediction artifact, with an absolute mAP50–95 delta of
about 0.0000495; this replay remains separate historical evidence. The original
Phase 1 prediction JSON was subsequently supplied and hash-verified on
2026-09-11, closing the historical artifact gate.

The one-shot `tools.run.phase2_data_gate` continuation resumes official
ImageNet training recovery, verifies all frozen payloads, generates the six
classifier calibration artifacts, reruns CPU tests and refreshes verification.
Its current status is saved in `artifacts/dataset_indexes/phase2-data-gate.json`;
the run log is `artifacts/dataset_indexes/phase2-data-gate.log`.
This run completed on 2026-09-10 at 20:37 UTC; no recovery worker remains active.
It leaves failing tests and remaining Phase 2 gates explicit in the report.

GPU counter collection uses `.venv/bin/python -m tools.run.phase2_profile_counters`
(also available through `tools/run/phase2_profile_counters.sh`). Each attempt
retains separate CSV, process logs and launch inputs under
`artifacts/benchmarks/phase2/counter-attempt-*`; `counter-capture.json` records
the latest attempt, return code, tool version and artifact hashes. It preserves
the existing Nsight Systems inputs. Earlier attempts returned
`ERR_NVGPUCTRPERM` or required authentication. The owner authenticated the
successful `counter-attempt-zawqo23w` run on 2026-09-11; no attempt changed
driver policy.

An administrator can reproduce counter collection from their own terminal using:

```sh
sudo -v
.venv/bin/python -m tools.run.phase2_profile_counters --sudo
.venv/bin/python -m tools.analysis.phase2_profile_summary
.venv/bin/python -m tools.analysis.phase2_decision_evidence
.venv/bin/python -m tools.analysis.phase2_finalize
```

The importer validates all six launches, source/input/output identities,
kernel order, explicit units and all four requested metrics. It derives DRAM
bandwidth from the paired Nsight Compute duration and read/write counters.
Raw and details CSV layouts are supported; the raw layout was checked against
an installed Nsight Compute 2025.4.1 sample export. The later real Phase 2
capture supplies the measured evidence. D2 is closed for its measured
FP6/INT8 GEMM scope using current benchmarks and validated counter evidence.
Final verification rechecks the raw profiling and decision
artifact hashes, so changing a summary's missing-metric list cannot close it.

Hardware pilots are driven by `tools.run.phase2_hardware_pilot`. Their four
primitive families use exhaustive ROM baselines and 0–3 extra output stages
inside a two-register wrapper. They validate encoding, pipelining and flow
plumbing, not competitive arithmetic architecture or pipeline retiming.
The local ICsprout55 RVT Liberty and Yosys versions differ from Phase 1 and
are separately identified in the reports. STA/vectorless-power capability
is recorded separately from synthesis and never treated as routed PPA.

The timing/power runner is `tools.run.phase2_sta_power_pilot`; it requires
explicit `--ieda`, `--yosys`, and `--liberty` paths. Reports retain the tool,
Liberty, constraints, netlist, adapter and raw-report hashes. Source artifacts
are under `artifacts/ppa/phase2`; normalized summaries are tracked under
`results/summaries/phase2-{hardware,sta-power}-pilot.json`.

The source-frozen gate record is
`results/summaries/phase2-final-verification.json`.
The focused Phase 2 CPU suite is 253/253 passing. The full CPU suite has
275 passing tests and no failures after the original YOLO artifact was restored.
Retained CUDA native evidence is
93/93 passing against unchanged source identity
`e0a8c05b4a9dbfbb58af6a236e495e7a3d0fac34162db4dace3119308dedadeb`.
