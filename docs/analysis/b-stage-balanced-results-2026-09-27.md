# B-stage balanced comparison results

2026-09-27. The twelve frozen existing configurations were extended from 128 to 1,000 evaluation images with the original numerical runners, unchanged identities and per-image checkpoints. The complete audited B ledger now has 183 of 200 configurations at 1,000 images; 17 remain at 128. Both recipes have 1,000 paired images for 83 datatype/model pairs, including 58 classifier pairs.

These are development FP32-QDQ results. The twelve arms were selected after inspecting their first 128 outcomes. Pointwise paired 95% bootstrap intervals describe this panel; they are not final confirmation. Accuracy differences below are percentage points (pp), with percentile 99.9 minus maxabs.

| Model | Format | Extended recipe | 128 top-1 % | New 872 top-1 % | Full 1,000 top-1 % | Recipe difference pp [95% interval] |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| mobilenet_v3_large | bfp6 | maxabs | 39.84 | 42.20 | 41.90 | +2.60 [+0.30, +4.80] |
| mobilenet_v2 | fp6_e2m3 | maxabs | 10.94 | 10.09 | 10.20 | +44.40 [+41.20, +47.50] |
| mobilenet_v2 | fp6_e3m2 | percentile_99_9 | 27.34 | 24.77 | 25.10 | -10.50 [-13.50, -7.50] |
| mobilenet_v2 | int6 | maxabs | 0.00 | 0.34 | 0.30 | +51.60 [+48.60, +54.70] |
| mobilenet_v3_large | fp6_e2m3 | maxabs | 0.00 | 0.11 | 0.10 | +44.20 [+41.20, +47.20] |
| mobilenet_v3_large | int6 | maxabs | 0.00 | 0.11 | 0.10 | +14.30 [+12.20, +16.50] |
| mobilenet_v3_large | int8 | maxabs | 3.12 | 4.24 | 4.10 | +64.80 [+61.80, +67.80] |
| resnet18 | fp6_e2m3 | percentile_99_9 | 63.28 | 56.42 | 57.30 | -8.90 [-11.70, -6.00] |
| resnet18 | fp6_e3m2 | percentile_99_9 | 52.34 | 54.01 | 53.80 | -11.50 [-14.30, -8.70] |
| resnet18 | int4 | maxabs | 0.00 | 0.23 | 0.20 | +8.30 [+6.50, +10.10] |
| resnet18 | int6 | maxabs | 50.00 | 51.72 | 51.50 | +9.10 [+5.80, +12.30] |
| resnet18 | nf4 | maxabs | 0.00 | 1.26 | 1.10 | +33.90 [+31.00, +36.90] |

The 128 and new-872 columns expose selection-panel versus extension-panel behavior; their different sample sizes and image identities should not be directly subtracted as a treatment effect.

## Selected cross-format comparisons

Difference is right minus left on the same 1,000 images and the same saved FP32 baseline. A change in format may also change scaling or block support policy.

| Model | Left | Right | Top-1 difference pp [95% interval] |
| --- | --- | --- | ---: |
| resnet18 | fp8_e4m3fn (maxabs) | fp7_e3m3 (maxabs) | +0.70 [-0.60, +2.00] |
| mobilenet_v2 | fp8_e4m3fn (maxabs) | fp7_e3m3 (maxabs) | -0.70 [-2.00, +0.60] |
| mobilenet_v3_large | fp8_e4m3fn (maxabs) | fp7_e3m3 (maxabs) | -0.20 [-1.80, +1.40] |
| resnet18 | int8 (maxabs) | bfp6 (maxabs) | -1.20 [-2.80, +0.40] |
| resnet18 | int8 (maxabs) | fp7_e3m3 (maxabs) | -0.10 [-1.90, +1.80] |
| resnet18 | int8 (maxabs) | mxfp8_e4m3 (maxabs) | -0.70 [-2.30, +1.00] |
| mobilenet_v2 | int8 (maxabs) | bfp6 (maxabs) | -6.20 [-8.60, -3.80] |
| mobilenet_v2 | int8 (maxabs) | fp7_e3m3 (maxabs) | -1.50 [-3.70, +0.80] |
| mobilenet_v2 | int8 (maxabs) | mxfp8_e4m3 (maxabs) | -0.70 [-3.00, +1.70] |
| resnet18 | int6 (maxabs) | bfp6 (maxabs) | +17.10 [+14.20, +19.90] |
| resnet18 | int6 (percentile_99_9) | bfp6 (percentile_99_9) | +8.20 [+5.60, +10.80] |
| resnet18 | fp6_e3m2 (maxabs) | fp6_e2m3 (maxabs) | +0.90 [-1.40, +3.20] |
| mobilenet_v2 | int6 (maxabs) | bfp6 (maxabs) | +60.00 [+57.00, +63.00] |
| mobilenet_v2 | int6 (percentile_99_9) | bfp6 (percentile_99_9) | +9.80 [+6.90, +12.60] |
| mobilenet_v2 | fp6_e3m2 (maxabs) | fp6_e2m3 (maxabs) | -25.40 [-28.50, -22.40] |
| mobilenet_v3_large | int6 (maxabs) | bfp6 (maxabs) | +41.80 [+38.80, +44.80] |
| mobilenet_v3_large | int6 (percentile_99_9) | bfp6 (percentile_99_9) | +30.10 [+26.90, +33.30] |
| mobilenet_v3_large | fp6_e3m2 (maxabs) | fp6_e2m3 (maxabs) | -35.20 [-38.10, -32.30] |
| resnet18 | int4 (maxabs) | nf4 (maxabs) | +0.90 [+0.30, +1.60] |
| resnet18 | int4 (maxabs) | mxfp4_e2m1 (maxabs) | +39.60 [+36.70, +42.60] |
| resnet18 | int4 (percentile_99_9) | nf4 (percentile_99_9) | +26.50 [+23.40, +29.60] |
| resnet18 | int4 (percentile_99_9) | mxfp4_e2m1 (percentile_99_9) | +33.00 [+29.80, +36.20] |

The three previously sealed detector paired-bootstrap results are retained by hash in the new evidence ledger. No detector predictions or detector bootstrap were rerun for this balance step.

## Reproduce

- Resume the exact twelve configurations: `.venv-b/bin/python -m tools.run.balanced_b run`
- Audit all saved predictions: `.venv/bin/python -m tools.analysis.b_stage_balanced_comparisons analyze`
- Rebuild this report and figures: `.venv/bin/python -m tools.analysis.b_stage_balanced_report`

Evidence: `results/summaries/b-stage-paired-1k-v2/analysis.json`, `balanced-evidence-ledger.json` in the same directory, and `results/figures/b-stage-balanced-recipe-effects-v2.png`.
