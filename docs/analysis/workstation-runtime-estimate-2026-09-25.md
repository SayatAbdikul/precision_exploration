# Experiment runtime estimate on the current workstation

Date: 2026-09-25. Scope: the proposed revised public numerical study, from
current progress, on the i9-12900K / RTX 4070 Ti / 32 GB workstation.
Coding, debugging, human review, waiting for acceptance implementation, and
dataset transfer time are excluded. No unmeasured future speedup is assumed.

This is a budget scenario, not a scheduled campaign ETA. The proposal does not
yet fix the complete experiment matrix. The inference subtotal below is
quantified; PTQ fitting, additional native pilots/diagnostic interventions, and
RTL/synthesis/P&R experiment runtimes remain additional unmeasured terms.

## Measured timing anchors

- Six workers produced 240.3 image executions/hour in the current-host,
  two-image-per-worker ResNet18 INT4 benchmark; four produced 162.0/hour.
  This is a small-sample, fast integer result, not an all-family measurement.
  [Resource profile](../../artifacts/phase3/controller/benchmarks/20260923T074159Z-3d5b3e9e/profile.json).
- Current four-worker pilot logs show CUDA image times around 240–263 seconds
  for ResNet18/MobileNetV2 posit4, and 501–518 seconds for MobileNetV3 Q1.6/FP7.
  Pilots also execute C++; production screens use one accepted backend.
- There is no measured current-host aggregate low-bit detector-screen rate.
  The retained historical YOLO INT8 C++ image took 12,059.85 seconds and cannot
  establish a current-host CUDA throughput. Detector rates below are assumptions.
- The prior 43.4-day illustration for 50k images at 75 seconds/image is
  **worker time**. Different configurations can overlap. At 240 aggregate
  images/hour, 50k image-equivalents take 8.7 days of workstation capacity,
  although an individual job remains limited by its own rate and the current
  controller does not shard one job across workers.

## Concrete experiment budget used for this estimate

The eight completed 1k screens are historical controls and are not rerun.
All extensions below reuse their own preceding-stage predictions, assuming
unchanged numerical identities. New PTQ/calibration recipes need new predictions.

| Remaining work | Classification evaluations | Detector evaluations |
| --- | ---: | ---: |
| 92 remaining A candidates at 256 images: 67 classifiers + 25 detectors | 17,152 | 6,400 |
| Additional PTQ discovery: 48 classifier and eight detector configurations at 256 images | 12,288 | 2,048 |
| Extend 30 classifier and ten detector configurations from 256 to 1k | 22,320 | 7,440 |
| Extend 24 classifier configurations from 1k to 10k | 216,000 | 0 |
| Final 4–6 recipes across three classifiers and one detector; extend 10k→50k / 1k→5k | 480,000–720,000 | 16,000–24,000 |
| Calibration robustness: eight additional full classifier runs and two detector runs | 400,000 | 10,000 |
| **Total** | **1,147,760–1,387,760** | **41,888–49,888** |

The robustness allocation corresponds to two additional calibration seeds for
two recipes on two classifier models, plus two additional seeds for one detector
recipe. It does not repeat every candidate with three seeds. The discovery and
promotion counts are explicit planning assumptions, not hard pruning limits.

Total: approximately **1.19–1.44 million additional model-image evaluations**.
Reconstruction optimization passes and calibration forward passes are not
silently treated as ordinary evaluation images in this total.

## Wall-clock scenarios, 24 hours/day

Use aggregate throughput that already includes concurrency:

```text
inference_days = (classification_images / classifier_images_per_hour
                + detector_images / detector_images_per_hour) / 24
```

Assume the workstation schedules classification and detection batches separately,
keeps a sufficient queue ready, and sustains the stated aggregate rates. Do not
divide these results by the worker count again. Queue tails, retries and mixed
resource contention can add time.

| Scenario | Classifier images/hour | Detector images/hour | Inference subtotal |
| --- | ---: | ---: | --- |
| Optimistic extension of fast-path behavior | 240 | 20 | 286.5–344.9 days; **9.4–11.3 months** |
| Illustrative mixed-workload budget | 80 | 10 | 772.3–930.7 days; **25.4–30.6 months** |
| Slow-family stress case | 40 | 2 | 2,068.2–2,484.9 days; **5.7–6.8 years** |

Only the 240 classifier anchor has a directly measured aggregate benchmark,
and only on ResNet18 INT4. The other aggregate rates are scenario inputs.
The slow-family case illustrates sensitivity, not a predicted outcome.
Unknown shared/log/nonlinear paths may be slower still. None is a guaranteed
lower or upper bound for every eventual recipe.

At the mixed-workload scenario, the subtotal breaks down as follows:

| Stage | Running time |
| --- | ---: |
| All 256-image exploration | 50.5 days |
| Extensions to 1k | 42.6 days |
| Classifier extensions to 10k | 112.5 days |
| Primary full validation | 316.7–475.0 days |
| Selected calibration-seed full evaluations | 250.0 days |

Dropping the seed-repeat evaluations from the optimistic scenario would yield
196.3–254.6 days (6.4–8.4 months), but that is a smaller scientific package.
Running only 12 hours/day doubles the calendar duration for the same throughput.

## Full experimental total and interpretation

```text
total experimental wall time
  = inference subtotal
  + PTQ fitting/calibration/native verification/diagnostic running time
  + hardware simulation/synthesis/P&R running time
  - overlap actually achieved between independent workloads
```

PTQ fitting and complete generic physical-flow runtimes have not been benchmarked
for the revised matrix. Treating them as zero would understate the full study;
assigning a precise duration now would invent evidence. Some hardware/CPU work
may overlap inference, but it competes for this workstation's CPU and RAM.

For planning, the numerical program is therefore a **roughly two-to-three-year
24/7 workload in the mixed scenario**, with an optimistic inference-only case
near nine-to-eleven months and slower cases lasting several years. This estimate
must be revised after representative detector/shared-family throughput and PTQ
fitting measurements. It excludes coding throughout and is not a claim that a
future optimized implementation would need the same time.
