# Continue Phase 3 on another device

This handoff preserves the existing campaign and per-image checkpoints. Phase 3
is still in progress: ResNet18 INT8 has one complete 1,000-image screen; the
current integer queue continues MobileNetV2 INT8, then ResNet18 INT6/5/4.
Copy-and-paste commands are in [phase3_other_device_commands.txt](../../phase3_other_device_commands.txt).

## 1. Preserve a consistent snapshot on the original PC

Finish or stop all experiment controllers, workers and benchmarks before the
final copy. Stop a foreground controller with Ctrl+C and confirm that its child
worker has exited. Completed image records survive; an unfinished image runs
again. Do not run the same queue independently on both devices: file locks only
coordinate processes on one filesystem, and divergent registries cannot simply
be combined.

Record `git rev-parse HEAD` and use exactly that commit on the new PC. Git carries
source, configurations, dataset manifests and curated summaries. It does **not**
carry the experiment payload. Separately copy these directories, keeping their
paths relative to the repository:

| Directory | Why it is needed |
| --- | --- |
| `artifacts/` | Weights, calibration, frozen evidence, predictions, saved images, continuation plan and logs |
| `data/raw/` | Frozen image datasets and annotations |
| `results/databases/` | Registry, including any SQLite WAL/SHM sidecars |
| `build/phase2/` | Original native libraries and their hash manifests |
| `cache/` and `results/raw/` | Supporting caches and retained raw results |

The current payload is approximately **11.7 GB**, excluding `.venv`. Allow extra
space for new results. Copy all registry files only while writers are stopped.
Use `rsync` without `--delete`; its checksum verification command is included in
the handoff file. Keep the original copy until the new device passes validation.
Do not copy `.venv`: recreate it for the target system.

## 2. Set up the new PC

The supplied commands assume **Linux x86_64 with an NVIDIA GPU**. Use a compatible
Linux installation, or validate a WSL2 installation separately. Native Windows
is not covered by these Bash/Linux commands.

Install Git, rsync, Python 3.12 with venv support, an NVIDIA driver, CUDA development
tools and a CUDA-compatible C++ host compiler. The source machine used Python
3.12.14, CUDA 13.1.115, `g++-14` for CUDA compilation and g++ 15.2 for C++ builds.
Follow NVIDIA's [Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
for the target distribution. Verify `nvidia-smi`, `nvcc --version` and
`g++-14 --version` before running the test script.

Clone the repository, check out the recorded commit and restore the payload.
The COCO image link currently points at the original PC's absolute path; the
handoff commands replace that symlink with the equivalent relative link.

Install the CPU wheels for **torch 2.3.0 / torchvision 0.18.0**, then the exported
[package versions](../../requirements-phase3-reproduction.txt). This is the
observed environment, not a tested cross-platform lockfile. Do not silently
upgrade packages if installation fails. The wheel pairing and index follow
[PyTorch's previous-version instructions](https://docs.pytorch.org/get-started/previous-versions/).
CPU-only PyTorch is intentional: candidate CUDA execution uses the project's
native CUDA library. `torch.cuda.is_available() == False` alone does not indicate
that this backend is unavailable.

## 3. Validate before resuming

First check the frozen engine, pipeline and continuation plan, and execute a tiny
exact reduction with both copied native libraries. Then run
`bash phase3_terminal_commands.txt`. Its default mode runs checks and writes logs;
leave `PHASE3_RUN_EXPERIMENTS` unset. The tests build temporary libraries and do
not replace the retained production builds.

Copied `.so` files are only usable if the target driver and Linux runtime are
compatible. If the native smoke check fails, preserve the error and the original
`build/phase2` files. Rebuilding over them changes hashes referenced by historical
evidence. A compatible build and its provenance need to be resolved before
continuing; do not bypass hash checks or regenerate the campaign.

Run a fresh two-image C++/CUDA diagnostic for each queued configuration using the
commands below. Existing completed pilot jobs would reuse checkpoints rather
than exercise the new hardware. The diagnostic actually executes the images,
compares all layer traces and final output hashes with the accepted pilot, and
records the new host and library identities. Any mismatch must be investigated
before resuming production. Its timings also show the target device's speed.

## 4. Resume the existing queue

Use the copied `artifacts/phase3/integer-continuation-plan.json`. The controller
rechecks identities, reuses completed work and resumes saved images. It already
passes `--recover-stale-seconds 300`; allow five minutes after the last old worker
heartbeat if the registry still says RUNNING. A copied lock file itself does not
mean another process holds its operating-system lock.

Record the migration boundary and device-check logs with the artifacts. Existing
per-image timings remain from the old PC; do not describe a combined screen's
average timing as a measurement of only the new GPU. Faster hardware does not
guarantee proportional speedup: the current thread experiment spends most time
in CPU output conversion, bias handling and quantization.

Completion of this integer queue does not close Phase 3. The remaining formats,
acceptance gates, diagnoses and D4 review remain in the
[remaining-work checklist](../roadmap/phases/phase-03-remaining-work.md).
