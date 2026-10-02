# Speed optimizations (lane S1), 2026-10-02

Protocol: `public/experiments/configs/breadth-study/speed-protocol-v1.json` (sha256 `45822ebc676aea0006439599a1fa14781d6c7c2cc358c283512af0639859b1b1`, file time 11:54:54 +05 (its own `written` field says 11:58), before any measurement of this lane). Addendum for the review follow-up: `speed-protocol-v1-addendum-1.json` (sha256 `128f64246f905394a28743a292467d0e2be6860b4e605eba8cb724bc1eb6a8d9`, 18:52 +05, before its measurements).
Evidence: development evidence only (ImageNet screen-1k, COCO screen-1k, the first 256 calibration images of the B2 input cache). No held-out image was read.
Summary files (each written once): `results/summaries/speed-v1/summary.json` (built by `tools/run/speed_summary.py`) and, for the review follow-up, `results/summaries/speed-v1/summary-addendum-1.json` (built by `tools/run/speed_summary_addendum.py`).
Revision: corrected 2026-10-02 after independent review 1 (`artifacts/agent_orchestration/handoffs/S1-speed-review.md`): finding B1 (shared-GPU throughput numbers) and the number slips of N2 fixed against the raw files; N1, N3 and N5 measured (section 5); N4, N6 and N7 stated where they apply. No identity result changed.

No result of the study changes. These fast paths give bit-identical records to the existing tools; only the time and the GPU memory needed to compute them change. No existing source was edited. Nothing was written under `artifacts/scaled_bridge_v2/`, `artifacts/experiment_b2*/` or any archive.

## 1. Numbers first

### 1.1 Exact engine (scaled bridge v2) on the GPU: `tools/scaled_bridge_fast`

Seconds per image of the predict loop (preprocessing included; process setup excluded, at 1.6-1.9 s for both paths). Both paths ran back to back on the same images inside one `gpu_run.sh --exclusive` call, after a one-batch warm-up. Accumulator: wide. Exclusive window 13:47:30-13:50:17 (2.8 min of the lane's 10-min budget). Conditions: one timed pass per batch size, on 96 (ResNet18) and 64 (MobileNetV3) images, wide accumulator only; the exclusive lock holds the GPU only, so the CPU load of other lanes was neither excluded nor recorded, and the archive is CPU-bound (host NumPy). The speed-ups below hold for these conditions only; repeat the passes (several per batch size, CPU load recorded) before a speed-up is quoted in the paper.

| case (images) | path | batch 8 | batch 32 | batch 64 | speed-up at batch 8 | best vs best |
|---|---|---|---|---|---|---|
| resnet18-int8-default-b2 (0-96) | archive 7c6344af | 0.1687 | 0.1709 | n/a | | |
| | fast | **0.0061** | 0.0066 | 0.0073 | **27.9x** | 27.9x (both best at batch 8) |
| mobilenet_v3_large-int8-default-b2 (0-64) | archive 1f75c923 | 0.2940 | 0.3239 | n/a | | |
| | fast | **0.0062** | 0.0062 | 0.0075 | **47.2x** | 47.5x (archive best 8, fast best 32) |

Items digest (every top-5, output hash, tie list) equal between the archive and the fast run and across all batch sizes in each pair (`artifacts/speed_v1/timing/*.json`).
Peak GPU memory of the fast process (torch max_memory_allocated, maximum over its batch sizes including 64): ResNet18 4.96 GB, MobileNetV3 8.56 GB. Batch 8 is the fastest setting and needs far less memory (section 3.1).

Throughput on the shared GPU (other lanes running; batch 8; ResNet18 int8 wide; 1,000 screen images, five passes per process; `artifacts/speed_v1/concurrency/`):

| setting | seconds per image per process: median of 5 passes (range) | images per second, total |
|---|---|---|
| one job, run B (loops 16:34:03-16:35:08) | 0.0126 (0.0122-0.0140) | about 79 |
| one job, run A (loops 16:36:42-16:37:43) | 0.0116 (0.0111-0.0136) | about 86 |
| two processes at once in one shared slot (loops fully overlapping, 16:54:09-16:55:41) | 0.0181 (0.0164-0.0195), 0.0185 (0.0160-0.0197) | about 109 |
| exclusive GPU, one process (section above; one pass, 96 images) | 0.0061 | 165 |

Two separately queued jobs of one lane did not overlap: `gpu_run.sh` balances slots between lanes and admitted them one after the other, so the pair was run as two processes inside one admitted job. Three concurrent jobs were not measured (lane cap of 2 jobs). MobileNetV3 int8 wide on the shared GPU, one job, two separate runs of 5 passes: medians 0.0077 (run B, 16:35:59-16:36:37, passes 0.0073-0.0082) and 0.0087 (run A, 16:39:20-16:40:08, passes 0.0085-0.0110), i.e. 0.0073-0.0110 s per image over the ten passes. Peak GPU memory at batch 8 (torch max_memory_allocated): ResNet18 0.67 GB, MobileNetV3 1.11 GB.

In practice a 1,000-image predict run of one (case, policy) drops from about 170 s (ResNet18) or 290-600 s (MobileNetV3, shared) to about 6-15 s. Process start is small for this engine: spawn to the end of the first 8-image batch took 2.1-2.3 s (interpreter 0.01 s, imports 0.12 s, engine context incl. CUDA initialisation 1.5-1.6 s, first batch 0.13 s; the archive's `run.sh root` digest check 0.05 s; shared GPU, `artifacts/speed_v1/setup-v1/setup.json`, section 5.2). The "30-40 s per process" of the lanes' brief (rule 7) is the orchestrator's general figure, not a measurement of this engine. What a separate process really costs on the shared GPU is the `gpu_run.sh` admission wait (10-16 min per admission at times today), so several policies or cases still belong in one admitted job.

Profile of the archives (`predict` loop, batch 8, 64 images, shared GPU, trace none; indicative shares, `artifacts/speed_v1/profile/`):

| stage (seconds for 64 images) | archive ResNet18 int8 wide | archive MobileNetV3 int8 wide | fast ResNet18 | fast MobileNetV3 |
|---|---|---|---|---|
| total engine time | 20.17 | 37.88 | 0.76 (uninstrumented) | |
| store quantization (codebook search, codes, diagnostics; host NumPy) | **11.60** | **24.58** | 0.18 (GPU) | 0.47 (GPU) |
| MAC post-operations (range checks, event statistics, scale, bias) | 2.14 | 4.25 | 0.21 (GPU) | 0.43 (GPU) |
| graph walk, record dicts, input casts, non-finite checks | 2.36 | 1.56 | 0.25 | 0.63 |
| native library call (total) | 1.74 | 2.40 | | |
| - reduction kernels alone (replayed on device pointers) | 0.55 | 0.41 | 0.36 | 0.32 |
| - transfers and allocation (library call minus kernels) | 1.19 | 1.99 | none (data stays on the GPU) | |
| - Python around the call (geometry, output arrays, checks) | 0.62 | 1.10 | | |
| element-wise operators (add, relu, hardswish, hardsigmoid, mul, pool) | 1.70 | 3.99 | 0.11 | 0.26 |
| output hashing and top-5 | 0.004 | 0.004 | | |
| seconds per image (shared) | 0.324 | 0.599 | 0.0149 | 0.0136 |

The archive spends 57-65 % of its engine time in host NumPy store quantization and only 2.7 % (ResNet18) and 1.1 % (MobileNetV3) in the reduction kernels. With sat.w20 instead of wide the archive takes 0.316 (ResNet18) and 0.596 (MobileNetV3) s per image and the fast path 0.0158 and 0.0149.

### 1.2 Block-format bias correction with less GPU memory: `tools/experiment_b2_fastblocks`

Lane L1's block cells re-run through `tools/run/speed_fastblocks.py` (outputs in `artifacts/speed_v1/fastblocks/`), full 1k-screen cells, shared GPU, `--min-free-mib 4000` (no `--heavy`):

| cell (default recipe) | bias-correction peak GPU memory, fast | peak from bias correction to end of cell, fast | bias correction, fast (s) | L1's preparation, original path (s, shared) | original path, peak during bias correction |
|---|---|---|---|---|---|
| resnet18 bfp6 | 0.59 GB | 0.75 GB | 35 | 24 | |
| mobilenet_v2 bfp6 | 1.63 GB | 2.16 GB | 124 | 128 | |
| mobilenet_v2 mxfp6_e3m2 | 1.63 GB | 2.16 GB | 130 | 135 | |
| mobilenet_v3_large bfp6 | 0.90 GB | 1.17 GB | 86 | 75 | |
| mobilenet_v2 mxfp8_e4m3 | 1.63 GB | 2.16 GB | 131 | 98 | **5.56 GB**, 83 s (original re-run, `--heavy 8000`, configuration identical to L1) |
| mobilenet_v3_large mxfp8_e4m3 | 0.90 GB | 1.17 GB | 108 | 55 | **3.84 GB**, 59 s (original re-run, `--heavy 8000`, configuration identical to L1) |

Memory is GB = 10^9 bytes of `torch.cuda.max_memory_allocated`, reset just before the bias-correction call. L1's preparation time also includes the weight quantization, so it is an upper bound for its bias correction; both are shared-GPU times. Memory: the bias correction needs 1.63 GB instead of 5.56 GB on mobilenet_v2 mxfp8_e4m3 (3.4x less) and 0.90 GB instead of 3.84 GB on mobilenet_v3_large mxfp8_e4m3 (4.3x less), and all six ran with `--min-free-mib 4000`, without `--heavy`. What other lanes see is the device-level footprint, which is larger than `max_memory_allocated` (CUDA context, cuBLAS workspace, the caching allocator's reserve): measured with nvidia-smi on the process's own pid, the fast path peaked at 3,088 MiB on mobilenet_v2 mxfp8_e4m3 (2,836 MiB during the bias correction, the rest in the evaluation; torch reserved 2.93 GB; section 5.1), and the reviewer measured 1,916 MiB on mobilenet_v3_large mxfp8_e4m3 (original path 4,954 MiB). So the fast cells stay below the rule-7 threshold of about 4 GB and run without `--heavy 8000` when admitted with `--min-free-mib 3600` or more; the original path stays a `--heavy 8000` job. Time: the host-memory path is not faster; against L1's shared preparation times it ranges from -4 % to +96 % per cell (L1's times also include the weight quantization), and against the original re-runs it took 131 s instead of 83 s (mobilenet_v2 mxfp8, +58 %) and 108 s instead of 59 s (mobilenet_v3_large mxfp8, +82 %); the originals ran in the heavy slot, i.e. beside fewer jobs. Pinned host memory (`pin=True`) was not measured. The gain is admission: heavy jobs wait for one cross-lane lock (this lane's original-path jobs waited over an hour for it; two timed out after 3,600 s), while these cells are admitted as ordinary jobs.

### 1.3 Detector bootstrap: `tools/analysis/fast_cocoeval`

CPU seconds per configuration (one process, `nice -n 10` beside other lanes' CPU work; `COCOeval.evaluate` runs once per configuration for both paths):

| configuration 04fd1b90...-1000-index | stats.py (pycocotools accumulate per draw) | fast_cocoeval |
|---|---|---|
| COCOeval.evaluate (shared, once) | 15.8 | 15.8 |
| per draw | 0.258 (2,000 draws measured) | 0.0171 (2,000 draws) |
| 2,000 resamples, total | 531 | 50.9 (incl. 0.9 s preparation) |
| 10,000 resamples, total | 2,592 (per-draw time x 10,000; not run) | 154.5 (137.8 measured for the draws) |

Speed-up of the resampling part: 15x per draw on this configuration; 10x for a whole 2,000-draw configuration including `evaluate`, 17x at 10,000 draws. Over the 8 configurations timed at 2,000 draws the fast path took 3.6-19.1 ms per draw (median 16.0); `evaluate` took 6.3-16.6 s on these 8 (2.7-23.9 s over all 142 validation runs). Two more 10,000-draw runs: 144.4 s (7221154...-1000-index) and 20.0 s (v1--int6--maxabs-1000-sealed).

## 2. Identity evidence (no tolerance anywhere)

### 2.1 Exact engine

Fast-path digest `edfecd5cdb0dc5adf06eff2807afa023b12d781cf0d08961a0b023da510a293b` (hashes of the package's numeric sources plus the base archive 1f75c923's engine digest). Every result below is sealed under `artifacts/speed_v1/validation/<kind>/edfecd5cdb0dc5ad/`.

1. MN-R regression (lane L2c's 82 record sets, `runs/1f75c923.../regress/*.json`): 82/82 sets, 656/656 full-trace records identical, `numerical()` equality (every layer's stored codes, state, raw and dot hashes, diagnostics, accumulator statistics and event counters, failure record, output hash, top-5 in contract tie order, tied classes), batch 8.
2. Full trace against the archives' own sealed batch-1 gate panels: 100/100 sets, 2,425/2,425 records identical (all 10 MobileNet cases x {wide 32, control 32, f21, fp16 rule, sat.struct-0/2/4: 8 images each} and all 8 ResNet18 B2 cases x every sealed panel), batch 8.
3. Predict items against the archives' predict items (sealed files of 7c6344af/1f75c923, read-only, or fresh archive runs redirected into `artifacts/speed_v1/archive-runs/<digest>/`): 20 cases (8 ResNet18 B2, 2 ResNet18 b1, 10 MobileNet) x 7 policy families (wide, control, f21, the fp16 rule exponent, sat.struct-<d>, sat.w with events, sat.w past collapse) x batches 1, 8 and 32, images 0-63: **453 (case, policy, batch) sets, 28,992 item comparisons, 0 different**, node constants equal in every set (20 cases x 7 policies x 3 batch sizes, plus posit8_es1 sat.w33 and the 10 MobileNet sat.w widths of item 5, each at 3 batch sizes). Three jobs (resnet18-fp8_e5m2, mobilenet_v2-int6, mobilenet_v3_large-fp7) died of CUDA out-of-memory at batch 32 beside other lanes and were completed with `--heavy 8000`. Every item compares top-5, output hash, tied classes, failure record and the non-zero per-node event counts; the file-level node constants (width or format, outputs per image) are compared too.
4. Batch invariance: the same 64 images give identical items at batch 1, 8 and 32 for every (case, policy) in item 3, and identical items digests at batch 8, 32 and 64 in the timing runs.
5. sat.w widths: the collapse widths give Top-1 0 with events on all 64 images in every case. ResNet18: the planned event widths have events on all 64 images (except posit8_es1 sat.w41, none; sat.w33 added: events 64/64, Top-1 unchanged, identical at batch 1, 8, 32). MobileNets: the planned widths (structural - 4) had no event, so a fast-path scan (`tools/run/speed_satw.py`, `artifacts/speed_v1/validation/satw-picks.json`) chose the widest of structural - 5 ... - 9 with events (V2 int8 w19, int6 w15, fp6 w16, fp7 w23, fp8 w39; V3 int8 w18, int6 w14, fp6 w16, fp7 w23, fp8 w39; events on 3-64 of 64 images, Top-1 within a few images of wide); fresh archive runs at these widths (redirected into `artifacts/speed_v1/archive-runs/`) match the fast path at batch 1, 8 and 32 on all 64 images.
6. GPU conformance test (`tests/conformance/test_scaled_bridge_fast.py`, full trace against sealed panels at batch 1 and 3; ResNet18 int8 wide and sat.w18, MobileNetV3 int8 wide and fp16.x-8, MobileNetV2 fp6 sat.struct-2): 10 passed.
7. The archived fast engine through its own `run.sh predict` (ResNet18 int8 wide, MobileNetV3 int8 sat.w14, MobileNetV2 fp6 sat.struct-2; images 0-63, batch 8): 64/64 items identical to the archive's items in each case.
8. Real-image batches in which only some images fail under the fp16 accumulator (review follow-up, section 5.3): four cases, 32-image windows with 1-4 mixed 8-image batches each, fast path against fresh archive runs at batches 1, 8 and 16.

How bit-identity is kept: the archived `kernel.h` (byte-identical copy, hash asserted) is compiled with the archive's flags (`nvcc -std=c++17 -O3 --ftz=false --fmad=false`), launched on device pointers; the host NumPy post-operations become torch CUDA int64/float64 element-wise kernels in the same order (separate kernels, so no fused multiply-add contraction; division always by a CUDA tensor, because `tensor / python_scalar` on CUDA multiplies by the reciprocal); the two-limb path keeps both 64-bit limbs; codes, hashing, certificates and policies come from the 1f75c923 archive package, imported unchanged.

### 2.2 Block bias correction

All 6 cells reproduce lane L1's records exactly: `configuration_sha256` equal (so the corrected biases, every weight hash and the bias report are equal), `logits_sha256` equal, all 8 readout arrays equal (`np.array_equal`), readout file sha256 equal, readout summary equal (mobilenet_v2 and mobilenet_v3_large x bfp6 and mxfp8_e4m3, mobilenet_v2 mxfp6_e3m2, resnet18 bfp6; default recipe; 1k screen). The fast version runs the same torch operations on the same tensors in the same per-node order (all FP32 chunks, then all quantized chunks; channel sums added chunk by chunk in the same order); only where the chunks live between operations differs.

### 2.3 fast_cocoeval

- Point estimates: 131/131 stored 1k detector configurations of lane L7 (`artifacts/experiment_b2_det/bootstrap/*-1000-*.npz`; index, reverse and random orders and the sealed v1 configurations) give mAP50-95 and mAP50 **identical** to pycocotools' accumulate (`stats.Evaluated.metrics`) and to the stored bootstrap point: maximum absolute difference 0.0.
- Bootstrap: on 8 configurations (6 '-1000-index', v1--bfp6--maxabs, v1--int6--maxabs) the 2,000-draw arrays with seed 310911 are identical to the stored stats.py arrays (maximum absolute difference 0.0) and the reported 95 % intervals are identical.
- Results: `artifacts/speed_v1/cocoeval/03fa4eaee3d836a7/` (keyed by the module's source hash).

## 3. How lanes use each fast path

### 3.1 Exact engine

Archive: `artifacts/speed_v1/implementations/edfecd5cdb0dc5adf06eff2807afa023b12d781cf0d08961a0b023da510a293b/` (`run.sh`, `manifest.json` with file hashes, `py/scaled_bridge_fast/`, `lib/cuda/fastdot.{so,json}`); run root `artifacts/speed_v1/runs/edfecd5c.../`. `run.sh` refuses to run if the live digest (package sources plus the 1f75c923 archive and its tracked dependencies) no longer resolves to its name.

```
GPU_LANE=<lane> artifacts/agent_orchestration/gpu_run.sh --min-free-mib 3000 \
  artifacts/speed_v1/implementations/edfecd5cdb0dc5adf06eff2807afa023b12d781cf0d08961a0b023da510a293b/run.sh \
  predict mobilenet_v3_large-int8-default-b2 sat.w16 cuda 0 1000 --batch 8
```

- Same CLI and record format as the archives' `predict` (compact items, `nodes`, the archive's field names), plus `execution_path`, `engine_sources` (the fast digest) and `base_engine_sources`. One sealed file per call: `runs/edfecd5c.../<case>/predictions/<policy>-cuda-<start>-<stop>.json`. Items where the archive has a sealed batch-1 panel are checked against it on the fly (`compared_with_sealed_batch1`).
- Gates follow the archives' rule: a case runs only with its fast-path case gate (`<case>/gate.json`: passing archive case gate, every full-trace set identical including wide and control, wide and control predict items identical on 64 images at batch 1, 8 and 32); a parameterised policy runs only with a passing fast-path policy gate of the same family on a case of the same model (`gate-<policy>.json`). Gated: all 20 cases (8 ResNet18 B2, 2 ResNet18 b1, 10 MobileNet) and 140 policy gates (each case x wide, control, f21, its fp16 rule, its sat.struct-<d>, two sat.w widths), so every policy family is admitted on all three models.
- Use batch 8 (fastest measured, about 1-2 GB): batch 32 on MobileNets and the two-limb fp8_e5m2 case went out of memory beside other lanes' jobs; a batch-32 job must be declared `--heavy 8000`.
- Even at batch 8 with `--min-free-mib 3000` a job can die of CUDA out-of-memory at start when other lanes fill the GPU between admission and allocation (it fails loudly before writing anything; seen by this lane and by the reviewer). A runner must check the exit code and retry such a job once with `--heavy 8000` (rule 7); the sealed output is only written at the end, so a retry never meets a partial file.
- Put several policies or cases in one admitted job where possible: the process start is about 2 s (section 5.2), but every `gpu_run.sh` admission can wait minutes on the shared GPU.
- A record of the fast path is the same evidence as the archive's record for the same case, policy and image (identity shown above); cite the fast digest and the base archive.
- Held-out confirmation runs: this lane's identity evidence is on development images (0-63 of the screen-1k for the bulk of the comparisons, plus the windows of section 5.3). A lane that runs the held-out confirmation through the fast path should keep a small archive cross-check per (case, policy family) on the confirmation images themselves, as lane Q1 does on 32-image ranges: run the archive's own `predict` on a short range of the same images, redirected into the lane's folder, and require item-by-item equality before using the fast records of that (case, family).

### 3.2 Block bias correction

In the lane's own process, before `prepare_blocks` runs:

```
import tools.experiment_b2.blocks as blocks
from tools.experiment_b2_fastblocks import bias_correct_blocks
blocks.bias_correct_blocks = bias_correct_blocks      # prepare_blocks looks the name up at call time
```

The configuration identity (`configuration_sha256`) is unchanged, because it hashes the configuration and the `tools/experiment_b2` sources, which are not edited. The record states the execution path in a sidecar next to it (`<model>--<format>--<recipe>.execution.json`, field `execution_path`), as `tools/run/speed_fastblocks.py` does. Redirect cell outputs into the lane's own folder (`matrix.MATRIX = ...`) and pass `build=False` to `data.cached_inputs`.

GPU admission: `GPU_LANE=<lane> artifacts/agent_orchestration/gpu_run.sh --min-free-mib 3600 .venv-b/bin/python -m <the lane's cell runner>` without `--heavy` (device-level peak measured at 1.3-3.1 GB on the cells above; section 5.1). Retry once with `--heavy 8000` if the job dies of CUDA out-of-memory (rule 7). A cell type not measured here (another model, or a recipe whose evaluation holds more activations) should be measured once at device level before it drops `--heavy` (`tools/run/speed_fastblocks_mem.py` shows how).

### 3.3 Detector bootstrap

```
from tools.experiment_b2_det.stats import Evaluated, RESAMPLES, SEED, CONFIDENCE
from tools.analysis.fast_cocoeval import FastAccumulate
fast = FastAccumulate(evaluated)                 # evaluated: a stats.Evaluated (COCOeval.evaluate ran once)
point = fast.metrics()                           # [mAP50-95, mAP50], identical to evaluated.metrics()
draws = fast.bootstrap(RESAMPLES, SEED, CONFIDENCE)   # (R, 2), identical to stats.py's draws for the same seed
```

Area 'all' and maxDets 100 only (what stats.py reports). Run it under `nice -n 10`.

## 4. Limits and open points

- The exact-engine fast path runs on CUDA only (no CPU backend); its identity is shown for the cases, policy families, images and batch sizes above, not for every width of every case. A new case needs its own gate (full-trace sets and 64-image comparisons, `tools/run/speed_engine_validate.py`, then `python -m tools.scaled_bridge_fast gates`).
- fp8_e5m2 (ResNet18): every admissible sat.w width (at most 63, the policy limit) collapses (structural width 74), so its "width with events" (w60) is also past collapse (Top-1 0 on the 64 images, events on all). posit8_es1: the gated sat.w41 has no event on the 64 images, so sat.w33 (events on all 64, Top-1 unchanged) was added.
- MobileNet sat.w: the planned widths (network structural width - 4) had no event on the 64 images; structural - 10 collapses. Widths with events and Top-1 near wide were chosen by a fast-path scan (`tools/run/speed_satw.py`) and then compared with fresh archive runs.
- Seconds per image are from short exclusive runs (64-96 images, one pass per batch size, wide only, CPU load of other lanes not controlled); the profile and the concurrency numbers are from the shared GPU and are indicative.
- The run-time exponent search of the block quantizer (`B2BlockQuantizer._chunk`, rule mse) was not changed: a different chunking changes the shapes of the per-block reductions, and identity of the chosen clipping would have to be shown again for every format; not attempted.
- Rule deviations, disclosed: some identity jobs at batch 32 ran without `--heavy` while their memory exceeded 4 GB; three of them died of CUDA out-of-memory and were retried with `--heavy 8000` (rule 7). One command of this lane was refused by the permission system (13:31, a `kill` of the lane's own helper queue) and not retried; no harm resulted.
- fast_cocoeval reproduces area 'all', maxDets 100 only.
- Disk (measured 19:08 with du): `artifacts/speed_v1` holds 423 MB. Of these, the fast engine's run root `artifacts/speed_v1/runs/edfecd5c.../` is 365 MB, almost all of it the 1,000-image predict files of lane Q1, which runs its sweep through the archived fast engine (206 of the 209 predict files; S1's own content there is the 160 gate files, 0.4 MB, and three 64-image smoke predicts, 0.8 MB); that growth belongs to the lane that runs the predicts. The reviewer's scratch is 12 MB. Lane S1's own outputs are about 47 MB (fresh archive runs 38 MB, validation 3 MB, the rest timing, profile, cocoeval, fastblocks cells, the archive 1.3 MB, the review follow-up under 1 MB), within the lane's 0.25 GB.
- Summary: `results/summaries/speed-v1/summary.json` (sha256 `01d0eb34cd15a408dae9594ff8b01d1bac134383889c71abc4b8cf909a1a4530`, file time 18:07:38, written once). Its raw values were correct; the review's number slips were in this document only, so it was not regenerated. The review follow-up is in `summary-addendum-1.json`.

## 5. Review 1 follow-up (protocol addendum 1)

All measured after `speed-protocol-v1-addendum-1.json` was written; development evidence; results in `results/summaries/speed-v1/summary-addendum-1.json` (sha256 `712199c1fbade9de32d22e11a94ed2c4cee5b0bd52265663b9cb96bbacfa69be`, written once), which also holds the recomputed numbers behind the corrections of B1 and N2.

### 5.1 Device-level GPU memory of a fast block cell (finding N1)

`tools/run/speed_fastblocks_mem.py` re-ran mobilenet_v2 mxfp8_e4m3 default (full 1k-screen cell, fast path) through the unchanged `tools/run/speed_fastblocks.py`, outputs redirected into `artifacts/speed_v1/fastblocks-mem-v1/`, with an in-process nvidia-smi sampler on its own pid (0.5 s, 1,054 samples, 0 errors). The cell again equals L1's record (configuration_sha256, logits_sha256, 8/8 readout arrays, readout file sha256, readout summary).

| measure | value |
|---|---|
| nvidia-smi process peak (device level) | **3,088 MiB** (at 500 s, in the evaluation after the bias correction) |
| nvidia-smi during the bias correction | up to 2,836 MiB |
| torch max_memory_allocated, whole process | 2.16 GB |
| torch max_memory_reserved, whole process | 2.93 GB |
| bias correction, peak allocated / time | 1.63 GB / 122 s (shared GPU) |

With the reviewer's device-level peaks (mobilenet_v3_large mxfp8_e4m3 1,916 MiB; mobilenet_v3_large mxfp4_e2m1 cum5_act_maxabs 2,814 MiB; resnet18 mxfp4_e2m1 cum5_act_maxabs 1,288 MiB) every measured fast cell stays below the 4 GB threshold of rule 7. Decision rule of the addendum: fast block cells run without `--heavy`, admitted with `--min-free-mib 3600` or more (section 3.2).

### 5.2 Process setup (finding N3)

`tools/run/speed_setup.py` (shared GPU, `artifacts/speed_v1/setup-v1/setup.json`): `run.sh root` (the archive's digest check) 0.05 s (3 runs); a fresh process from spawn to the end of its first 8-image batch 2.1-2.3 s for resnet18-int8 and mobilenet_v3_large-int8, wide (2 runs each): interpreter 0.01 s, `import tools.scaled_bridge_fast.worker` 0.12 s, engine context including CUDA initialisation 1.5-1.6 s, first batch 0.13 s. The earlier "30-40 s per process" was the brief's general figure; the real per-job cost on the shared GPU is the admission wait.

### 5.3 Real-image batches with some images failing under a float accumulator (finding N5)

`tools/run/speed_failmix.py scan` (fast path, chooses images only; `artifacts/speed_v1/validation/failmix-picks.json`) moved the fp16 scale exponent upward from each case's rule exponent. Mixed outcomes exist on real images at one exponent per case:

| case | rule | exponent with mixed outcomes | failed of 1,000 | mixed 8-image batches | window compared | failed in window | mixed batches in window |
|---|---|---|---|---|---|---|---|
| resnet18-int8-default-b2 | fp16.x-11 | fp16.x-3 | 827 | 97 | 0-31 | 27 | 4 |
| mobilenet_v2-int8-default-b2 | fp16.x-9 | fp16.x-2 | 346 | 118 | 16-47 | 11 | 4 |
| mobilenet_v3_large-int8-default-b2 | fp16.x-8 | fp16.x-2 | 2 | 2 | 40-71 | 1 | 1 |
| mobilenet_v3_large-fp7_e3m3-default-b2 | fp16.x-3 | fp16.x4 | 91 | 67 | 40-71 | 5 | 4 |

One exponent higher every one of the 1,000 images fails; one exponent lower none fails (on all 1,000 images for three cases; for mobilenet_v3_large-int8, fp16.x-3 was run on images 0-63 only, 0 failed). The reviewer's 24-image samples sat on either side of these steps. The archive's own `predict` ran on each window (redirected into `artifacts/speed_v1/archive-runs/`), and the fast path was compared item by item at batches 1, 8 and 16 (`validation/compare/edfecd5cdb0dc5ad/<case>--<policy>--<start>-<stop>--b<B>.json`):

**12 (case, batch) sets, 384 item comparisons, 0 different**, node constants equal in every set. The archive's own records show the same mixed outcomes in each window (failed images 27, 11, 1 and 5 of 32; first failing node conv1, features_14_conv_2, classifier_3, classifier_3; mixed 8-image batches 4, 4, 1, 4), so batches with failed and valid images side by side, including a batch of 16 that holds two such groups, reproduce the archive field by field (failure record, events, top-5, output hash of the valid images). The mobilenet_v3_large-int8 set died of CUDA out-of-memory at batch 8 beside other lanes after batch 1 had passed and was completed by its one retry with `--heavy 8000` (rule 7). This closes the gap the reviewer noted (that path had been covered only by a synthetic node test and code reading); it is still development evidence on four windows, not on every case.
