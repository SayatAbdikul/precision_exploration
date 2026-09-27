# Exact execution validation and experiment readiness — 2026-09-26

The optimized exact engine passed 64 image comparisons: eight images for each of ResNet18 and MobileNetV2 INT4/5/6/8. Every saved layer code/hash, final output, execution strategy and quantizer-event signature matched the retained accepted pilot. Current implementation and historical execution-guard hashes still match the validated suite.

Four workers with four CPU threads each completed the suite in 254.64 seconds, or 904.8 images/hour including setup and validation overhead. This short compatibility suite is not a sustained campaign benchmark or a measurement of native low-bit hardware throughput.

On the separate two-image ResNet18 INT4 comparison, mean exact execution decreased from 68.540 to 7.277 seconds/image (9.42×). Both passes started with cold activation caches; the prepared pass ran first. Per-image time includes diagnostics and excludes setup. The changed implementation preserves candidate exact arithmetic; it does not replace candidate reductions with FP32.

| Model | Format | Mean seconds/image in four-worker suite |
| --- | --- | ---: |
| mobilenet_v2 | int4 | 11.035 |
| mobilenet_v2 | int5 | 11.362 |
| mobilenet_v2 | int6 | 12.377 |
| mobilenet_v2 | int8 | 17.707 |
| resnet18 | int4 | 7.441 |
| resnet18 | int5 | 7.721 |
| resnet18 | int6 | 7.969 |
| resnet18 | int8 | 10.142 |

## Resume and tests

Rerunning `.venv/bin/python -m tools.run.exact_execution` verified all eight completed jobs, reused the identical validation-file hashes and performed zero new inferences. This command runs compatibility diagnostics only; its maximum panel is eight images. It is not the launcher for the next quality experiments. Partial activation-cache replay is covered by the implementation unit tests.

The final implementation test run reported 58 passed in 3.07 seconds. The readiness audit also passed `git diff --check`. See the sealed summary for the test command and exact suite references.

## Next experiments

- **E1: arithmetic isolation.** Twelve model/datatype cases cover all ten datatype families and four models. Start with 32 images per arm, increasing to at most 128 development images. Compare exact reductions with FP32 reductions while holding graph, quantizers, weights, scales, images and preprocessing fixed. All twelve need the matched control; ten also need native acceptance. Integrate the prepared engine into a resumable quality runner. Existing B predictions are not a matched control.
- **E2: recipe diagnosis.** ResNet18 INT4 and MobileNetV2 INT4/5/6, three arms each: strict control, MSE clipping only, adaptive rounding only. Start with 128 images per arm; the frozen rule permits 256 for uncertain effects. Implement and version the two new recipes, then validate changed graphs before exact runs.
- **E3 B: broader evaluation.** Extend 171 selected recipe configurations from 128 to 1,000 images, preserving all 100 datatype/model pairs. Existing runners support this extension. Add a matrix-aware recipe selector first because current command-line filters select model and datatype, not the exact selected recipes. Reuse the first 128 only when full configuration identity remains unchanged: 872 additional candidate images per configuration, 149,112 total, plus required baseline work. No 1,000-image B summaries existed at this audit.
- **E4/E5: later confirmation and hardware study.** Repeat central comparisons with three frozen calibration seeds, audit confirmation-data exposure and evaluate complete-operation hardware costs. Larger exact runs require family-specific throughput evidence and an explicit compute budget.

All 200 existing B configurations and eight exact A quality screens remain preserved. E0 completion does not grant acceptance to other datatypes or models and does not complete Phase 3 D4. No new quality experiment was launched during this readiness audit.

## Evidence

- Sealed validation summary: `results/summaries/exact-execution-v2-validation-2026-09-26.json`
- Frozen matrix: `public/experiments/configs/breadth-study/comparison-matrix-v1.json`
- Saved prediction analysis: `docs/analysis/saved-prediction-study-v1.md`
