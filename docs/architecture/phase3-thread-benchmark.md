# Phase 3 CPU and CUDA thread benchmark

This diagnostic compares MobileNetV2 INT8/INT64 on the same first two frozen
screen images, at native resolution, with the production layer diagnostics.
It measures C++ and CUDA execution with 4, 8 and 16 CPU threads. It does not
add screening images or change an accepted configuration.

## Measured results, 2026-09-20

Ryzen 7 7800X3D and RTX 3060 Ti; mean wall seconds per image, including diagnostics:

| CPU threads | C++ backend | CUDA backend |
| ---: | ---: | ---: |
| 4 | 178.85 | 163.93 |
| 8 | 170.48 | 162.45 |
| 16 | 170.79 | 163.22 |

All 12 executions matched the accepted pilot's final outputs and complete layer
traces. Encoded input hashes also matched across all six settings for each image.
Evidence is in [the retained summary](../../results/summaries/phase3-thread-benchmark.json),
job `70b61dab665a1a157ce7e78054bba8ff7bd7497f68771bc535bac7d537ea24e9`.

CUDA with eight threads was the measured fastest setting. Its advantage over
CUDA with four threads was only 0.90%; two images do not establish that small
difference reliably. C++ with eight threads reduced wall time by 4.68% compared
with four. Sixteen threads did not improve overall time over eight on either
backend. At eight threads, CUDA reduced total time by 4.71% relative to C++.

The CUDA eight-thread run spent an average 137.03 seconds (84.4%) in output
storage/conversion, versus 1.29 seconds in native dispatch. CPU output handling
dominates this graph. Increasing GPU capacity or CPU thread counts alone will
not remove that bottleneck. Keep the accepted production queue unchanged;
substantial speedups need separately validated optimization of output handling.
In particular, the frozen loader calls `torch.set_num_threads(4)`, so changing
only the shell's `OMP_NUM_THREADS` does not reproduce this diagnostic override.

## Reproduction

Run when the CPU and CUDA screen/pilot workers have stopped:

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m tools.run.phase3_thread_benchmark \
  --model mobilenet_v2 --format int8 --images 2 --threads 4 8 16
```

The tool holds both existing native-worker locks for the comparison. Each
setting runs in a fresh subprocess, in a seeded shuffled order. It sets
`OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS`, disables dynamic
OpenMP team sizing, and explicitly sets PyTorch/OpenMP counts after the frozen
loader's four-thread default. Actual thread settings are verified and retained.
The override exists only in these diagnostic subprocesses; the production
runtime and screening source identities remain unchanged.

Each subprocess first executes a tiny exact native reduction to initialize its
backend context. The first full image has cold graph caches; the second can
reuse caches. Timing includes the original graph execution and sampled
diagnostics. Model loading, FP32 observation and input encoding are recorded
separately or excluded, matching the production per-image execution timer.

Every image must match all layer traces and the final output hash from the
accepted C++/CUDA pilot. A fast mismatching result fails the benchmark. Reports
also reject unequal image subsets and duplicate images within a setting.

## Evidence and timing interpretation

Per-image records and the job manifest are saved under
`artifacts/phase3/thread-benchmark/<job-sha>/`. A completed comparison publishes
`results/summaries/phase3-thread-benchmark.json`. The job retains machine,
native-library, campaign, engine, pipeline and benchmark implementation identity.

`inference_with_diagnostics_seconds` is the time to compare across settings.
The component measurements are explanatory:

- `native_dispatch_seconds` includes native reductions plus CPU admission and
  argument preparation; it is not a GPU-only kernel measurement.
- `store_seconds` covers conversion of accumulator results, prescribed bias
  storage/addition, activation and output quantization.
- `diagnostic_seconds` covers quantizer-event and layer-sample diagnostics.
  Some of this occurs inside output storage, so these timers overlap and must
  not be added as disjoint costs.
- `cpu_seconds` counts process CPU time across its threads; wall time and CPU
  time are distinct measurements.

Two images per setting identify large effects but cannot establish reliable
small percentage differences. Results apply to this graph and current host;
other formats and accumulator implementations require their own comparisons.
Normal desktop activity can add noise despite the experiment-worker locks.

The production queue can be resumed with its existing command after the
benchmark exits. Its completed image checkpoints are independent of these
diagnostic records.
