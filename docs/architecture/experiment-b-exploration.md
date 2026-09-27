# Experiment B exploration: execution and evidence

The owner authorized starting the transition on 2026-09-25. The strict-A pilot
batch was stopped through its normal SIGTERM handler after both near-complete
MobileNetV3 pilots finished. Its final state was five completed pilots, four
interrupted and 83 pending. Image/backend checkpoints and the eight earlier
1k integer screens remain available. This does not mark original Phase 3/D4
complete. The transition receipt is
`artifacts/experiment_b/transition-20260925.json`.

## Main research campaign

B explores suitable PTQ recipes across all 25 accepted datatypes and all four
models. The exhaustive original strict-A recipe is now a source of selected
controls rather than a prerequisite for entering the new study. This revises
the previous proposal's requirement to complete all 100 strict-A screens first.

The initial matrix contains 200 planned configurations: 25 formats × four
models × two recipes. Start at 128 evaluation images per configuration, ordered
by SHA-256 within the frozen 1k list. These are diagnostic/development results;
128 images cannot support close rankings or eliminate a datatype family.
Extend at least one appropriate recipe per datatype/model pair to 1k, along
with meaningful method comparisons, then use separate confirmation data for
headline results. The 1k extension is an explicit CLI choice, not automatic
promotion based on a noisy 128-image score.

Two recipes are initially implemented:

- `maxabs`: maximum magnitude over every observed activation element and every
  weight element, divided by the format's positive finite maximum.
- `percentile_99_9`: 99.9th percentile of absolute magnitudes; activation
  thresholds use the declared stratified training samples, weight thresholds
  use all values within each output channel.

Both use external FP32 scales, per-output-channel weights and static per-node
activations. The scale, metadata and application costs belong in later hardware
comparisons. These recipes are simple initial comparators, not a claim of best
achievable PTQ. Reconstruction, adaptive rounding, MSE range search and
family-specific methods require separately identified implementations.

## Current implementation boundary

The initial scalar classifier implementation covered 21 formats × three
classifiers × two recipes = 126 configurations. Its inventory historically
records the 74 entries that were then blocked:

- 50 YOLO entries await validated detector-head observation and operator policy.
- 24 classifier MX/BFP entries await validated intrinsic shared-block execution
  along convolution reduction axes.

The versioned extension below completed these entries without substituting a
scalar approximation for a shared format.

## Versioned extension for the remaining 74

The separate `tools.experiment_b_ext` runner now implements those 74 entries
without changing the original runner's source identity or its 126 completed
records. Its inventory is the exact complement of the original blocked set.
The extension has its own evidence tree under `artifacts/experiment_b_ext/`.
The two trees together cover the 200 planned 128-image configurations only
after every extension summary has completed; the original B inventory remains
a historical record of what version 1 could execute.

Shared MX/BFP formats use their accepted E8M0 scale and 32-element block size.
Weights and convolution patches own blocks along flattened reduction K;
stored activations own blocks along channel. Each patch is requantized on K
before FP32 convolution. `maxabs` selects the smallest legal E8M0 scale that
covers the block maximum divided by the element codebook maximum;
`percentile_99_9` uses the analogous block percentile. Partial blocks use only
valid elements for scale selection and zero padding for absent K positions.
These dynamic intrinsic scales do not use the 2k classifier calibration
population. Neither technique is the strict-A exhaustive per-block MSE search.

YOLO uses the existing frozen lowering's named convolution, DFL, box decoding,
score, and concatenate operations, with original folded FP32 constants
substituted before B quantization. A two-probe graph check verifies the
unquantized output against the folded model. Scalar detector recipes calibrate
all 2k frozen COCO training images with a channel-covering observer. The
84-channel head joins hold separate 4-coordinate and 80-score scales; this
prevents box coordinate magnitudes from collapsing class probabilities. The
evaluation runs the pinned NMS/COCO category conversion and COCOeval on the
same 128 or 1k images for candidate and folded FP32 baseline. Detector mAP is
dataset-level and the 128-image values remain descriptive.

The extension keeps the original B distinction between exploratory FP32 QDQ
and exact sequential Model C/native acceptance. It does not silently relabel
either kind of evidence. Its provenance includes the original B source hash,
extension source hash, protocol, format manifest, model/checkpoint,
preprocessing, calibration, scales, evaluation list, and runtime.

After creating `.venv-b` with the original requirements, install the pinned
detector additions using `tools/experiment_b_ext/requirements-detector.txt`.
Run or resume the 74 extension configurations with:

```bash
cd /home/maveric/precision_exploration && .venv-b/bin/python -m tools.run.experiment_b_ext --run
```

Use `--prepare` for inventory, `--status` for progress, `--models` or
`--formats` to restrict a pilot, and `--images 1000` to explicitly extend a
selected configuration. Repeating the command reuses verified calibration,
per-image predictions and complete summaries. The extension holds the same
GPU/native pool locks as the original runner.

To resume both 128-image B components in order from one command, use:

```bash
cd /home/maveric/precision_exploration && bash tools/run/experiment_b_all.sh
```

The completed 128-image screen is 200/200 configurations: 126 original B
scalar-classifier records and 74 extension records. The extension's final
status and per-configuration summaries are in `artifacts/experiment_b_ext/`.
This is breadth-first development coverage, not 1k promotion, method
optimization, confirmation, native acceptance or the end of Phase 3.

The backend is **FP32 quantize/dequantize exploration**, independently identified
from the exact sequential Model C engine. Its declared behavior is:

- accepted-format finite values projected into FP32 codebooks;
- FP32 midpoint boundaries, normalized inputs and reconstructed values;
- tie preference based on the accepted manifest's encoding-code parity/order;
- finite endpoint clipping as an explicit B policy, including FP8 E5M2;
- ordinary deterministic FP32 framework convolution/reductions and biases;
- TF32 disabled;
- quantized inputs and arithmetic outputs, including first/last layers;
- identity, dropout-in-eval and flatten operations preserve their inputs.

Rounded log/NF levels, near-boundary decisions and reduction arithmetic can
differ from the high-precision oracle/exact engine. GPU/reference QDQ agreement
does not establish exact-A equivalence or native graph acceptance. Hardware
finite-accumulator claims need their own corresponding validation.

## Calibration and identity

Each classifier observes all 2,000 frozen training calibration images. Evaluation
images never set scales. A rotating channel-stratified observer collects up to
256 values per node per image and records the true maximum over all elements.
It covers every channel across the calibration population and avoids repeatedly
sampling only the same 16 flat positions. It is a declared approximation to the
full distribution, not exhaustive activation sampling.

Calibration batches of eight, each with verified payload identities, are saved
as hashed NPZ/checkpoint pairs. Resume reuses complete verified batches. Learned
scales, model/checkpoint/preprocessing identities, format hashes, code identity,
dataset lists, runtime and backend semantics determine configuration identity.
Changing a recipe or execution source cannot silently reuse old predictions.

Inference uses fixed batches of eight. Predictions are checkpointed per image.
If a batch is interrupted after some image records were written, the complete
original batch is recomputed and its retained predictions must agree. Promotion
from 128 to 1k reuses the identical prefix; it adds 872 images.

## Run and resume

The original `.venv` remains the exact-engine environment. The new `.venv-b`
contains PyTorch 2.3.0+cu121 and torchvision 0.18.0+cu121 with NumPy 1.26.4.
The matching CUDA wheel pair is documented in the
[official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/).

Run or resume the supported exploration entries with:

```bash
cd /home/maveric/precision_exploration && .venv-b/bin/python -m tools.run.experiment_b --run
```

Useful commands:

```bash
.venv-b/bin/python -m tools.run.experiment_b --prepare
.venv-b/bin/python -m tools.run.experiment_b --status
.venv-b/bin/python -m tools.run.experiment_b --run --models resnet18 --formats int8 --limit 1
.venv-b/bin/python -m tools.run.experiment_b --run --models resnet18 --formats int8 --images 1000
```

The entry point sets deterministic cuBLAS workspace configuration before CUDA
initialization. CUDA is the default; unavailable CUDA fails explicitly. `--device
cpu` is an explicitly different runtime identity for validation or diagnostics.

One process owns the B controller lock and the legacy native CPU/GPU locks,
preventing a conflicting A run. Normal SIGTERM/Ctrl+C retains completed evidence.
Do not edit B execution sources while the sweep is active; the runner rejects
source drift. No inference throughput is assumed from the old integer benchmark.

Evidence is under `artifacts/experiment_b/`: inventory, live status, calibration
checkpoints, numerical validation, configuration/scale records, per-image paired
predictions and summaries. An invocation finishing its supported subset is not
the same as completing the entire 200-configuration research matrix.
