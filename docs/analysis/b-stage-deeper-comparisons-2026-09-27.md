# B-stage paired prediction analysis and deeper comparisons

2026-09-27. Read-only analysis of saved predictions; no new inference launched.

Audited all 200 configurations: 171 at 1,000 images and 29 at 128. All original 128-image prefixes and candidate/baseline seals, hashes, configuration identities and ordered sample contexts were verified. Classification metrics were recomputed. The three selected detector comparisons were independently recomputed with dataset-level COCO AP and 500 paired image-bootstrap draws.

**Pairing:** 71 datatype/model pairs have both recipes at 1,000 images (46 classifier, 25 detector). The other 29 recipe comparisons use only 128 shared images. Full-panel classifier comparisons also retain separate 128-image selection and 872-image extension statistics.

**Interpretation:** every result is B FP32-QDQ development evidence. Classifier intervals use 10,000 paired bootstrap draws; the equivalent three-outcome multinomial avoids repeated image allocations. Exact McNemar tests and Holm adjustment across the 75 classifier recipe comparisons are supplied as exploratory diagnostics. Neither adjustment nor the 872-image split turns adaptive development into independent final confirmation. Detector intervals are exploratory pointwise intervals from 500 draws; close claims need more resamples and confirmation.

## Main findings

- A single clipping recipe is not consistently best across architectures or formats. Inspect both top-1 and top-5: MobileNetV2 INT8 changes by -0.1 top-1 points but +4.3 top-5 points, with 283 changed top-1 predictions.
- BFP6 warrants deeper comparison with INT6/INT8. Its nominal six-bit payload excludes shared scale metadata; the cost comparison must include support arithmetic and memory.
- FP7 and FP8 E4M3 have small quality gaps under maxabs in all four models. This motivates uncertainty and hardware analysis, not a claim of equivalent quality.
- ResNet18 has useful four-bit contrasts (INT4, NF4, MXFP4), while several MobileNet low-bit variants collapse. These results motivate architecture/recipe diagnosis rather than selecting one universally best format.
- INT8 and Q1.6 have identical complete saved top-five lists/detections wherever both reach 1,000 images: seven model/recipe comparisons. The offline quantizer audit finds identical tie preferences and Q1.6 levels/boundaries exactly equal to INT8 divided by 64. External scaling can compensate for that factor. Preserve separate identities but avoid counting identical outputs as independent support; this is not a universal execution proof.
- Posit6/8, binary and ternary frequently have near-zero quality under these recipes. The offline Posit8 table has only three positive levels above 10% of its positive maximum, versus 115 for INT8. This motivates testing endpoint-based scaling versus reconstruction-driven scaling; it does not establish the cause of accuracy collapse. Scale/range, code-occupancy and layerwise diagnostics come before larger evaluation panels.

## Selected recipe comparisons

Difference = percentile 99.9 minus maxabs, in percentage points. Scores are top-1 %, except YOLO which uses mAP50–95 points.

| Model / format | Paired images | Maxabs | Percentile | Difference | Pointwise 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| resnet18 / int8 | 1000 | 69.80 | 62.40 | -7.40 | [-9.80, -5.00] |
| resnet18 / bfp6 | 1000 | 68.60 | 68.80 | +0.20 | [-1.10, +1.60] |
| resnet18 / mxfp4_e2m1 | 1000 | 39.80 | 41.50 | +1.70 | [-0.90, +4.30] |
| mobilenet_v2 / int8 | 1000 | 66.50 | 66.40 | -0.10 | [-2.70, +2.50] |
| mobilenet_v2 / fp6_e2m3 | 128 | 10.94 | 59.38 | +48.44 | [+39.06, +57.03] |
| mobilenet_v2 / bfp6 | 1000 | 60.30 | 61.70 | +1.40 | [-0.10, +2.90] |
| mobilenet_v3_large / int8 | 128 | 3.12 | 69.53 | +66.41 | [+58.59, +74.22] |
| mobilenet_v3_large / fp6_e2m3 | 128 | 0.00 | 41.41 | +41.41 | [+32.81, +50.00] |
| yolov8n / bfp6 | 1000 | 11.958 | 12.016 | +0.058 | [-0.201, +0.326] |
| yolov8n / int8 | 1000 | 31.591 | 2.905 | -28.685 | [-30.986, -27.707] |

The 128-image rows above are deliberately not mixed with their one-sided 1,000-image scores.

YOLO fp8_e4m3fn → fp7_e3m3 under maxabs: 15.975 → 15.705 mAP points; difference -0.271, paired interval [-0.443, +0.070] on 1,000 images.

## Selected same-recipe format comparisons

Difference = right minus left. The baseline model context and every paired FP32 top-five list were checked across original/extension runners. These comparisons change representation and sometimes its required scaling/block policy; they do not isolate intrinsic number representation alone.

| Model | Left | Right | Images | Left top-1 | Right top-1 | Difference | Pointwise 95% interval |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| resnet18 | fp8_e4m3fn (maxabs) | fp7_e3m3 (maxabs) | 1000 | 69.00 | 69.70 | +0.70 | [-0.60, +2.00] |
| mobilenet_v2 | fp8_e4m3fn (maxabs) | fp7_e3m3 (maxabs) | 1000 | 65.70 | 65.00 | -0.70 | [-2.00, +0.60] |
| mobilenet_v3_large | fp8_e4m3fn (maxabs) | fp7_e3m3 (maxabs) | 1000 | 68.60 | 68.40 | -0.20 | [-1.80, +1.40] |
| resnet18 | int8 (maxabs) | bfp6 (maxabs) | 1000 | 69.80 | 68.60 | -1.20 | [-2.80, +0.40] |
| resnet18 | int8 (maxabs) | fp7_e3m3 (maxabs) | 1000 | 69.80 | 69.70 | -0.10 | [-1.90, +1.80] |
| resnet18 | int8 (maxabs) | mxfp8_e4m3 (maxabs) | 1000 | 69.80 | 69.10 | -0.70 | [-2.30, +1.00] |
| mobilenet_v2 | int8 (maxabs) | bfp6 (maxabs) | 1000 | 66.50 | 60.30 | -6.20 | [-8.60, -3.80] |
| mobilenet_v2 | int8 (maxabs) | fp7_e3m3 (maxabs) | 1000 | 66.50 | 65.00 | -1.50 | [-3.70, +0.80] |
| mobilenet_v2 | int8 (maxabs) | mxfp8_e4m3 (maxabs) | 1000 | 66.50 | 65.80 | -0.70 | [-3.00, +1.70] |
| resnet18 | int6 (percentile_99_9) | bfp6 (percentile_99_9) | 1000 | 60.60 | 68.80 | +8.20 | [+5.60, +10.80] |
| mobilenet_v2 | int6 (percentile_99_9) | bfp6 (percentile_99_9) | 1000 | 51.90 | 61.70 | +9.80 | [+6.90, +12.60] |
| mobilenet_v3_large | int6 (percentile_99_9) | bfp6 (percentile_99_9) | 1000 | 14.40 | 44.50 | +30.10 | [+26.90, +33.30] |
| resnet18 | int4 (percentile_99_9) | nf4 (percentile_99_9) | 1000 | 8.50 | 35.00 | +26.50 | [+23.40, +29.60] |
| resnet18 | int4 (percentile_99_9) | mxfp4_e2m1 (percentile_99_9) | 1000 | 8.50 | 41.50 | +33.00 | [+29.80, +36.20] |
| resnet18 | fp6_e3m2 (maxabs) | fp6_e2m3 (maxabs) | 1000 | 65.30 | 66.20 | +0.90 | [-1.40, +3.20] |
| mobilenet_v2 | fp6_e3m2 (maxabs) | fp6_e2m3 (maxabs) | 128 | 40.62 | 10.94 | -29.69 | [-38.28, -21.09] |
| mobilenet_v3_large | fp6_e3m2 (maxabs) | fp6_e2m3 (maxabs) | 128 | 32.03 | 0.00 | -32.03 | [-39.84, -24.22] |

## Next enrollment: balance twelve recipe counterparts

Five core comparison groups retain 55 existing configuration identities. Twelve need their missing 872-image extension; the other 43 already have 1,000 images. This costs **10,464 new candidate-image evaluations**, with no changed calibration or numerical recipe. Once completed, the overall existing ledger would contain 183 configurations at 1,000 images and 83 full recipe pairs; 17 other recipes would remain exploratory at 128.

| Model | Format | Recipe to extend | Existing → target | New images |
| --- | --- | --- | --- | ---: |
| mobilenet_v2 | fp6_e2m3 | maxabs | 128 → 1000 | 872 |
| mobilenet_v2 | fp6_e3m2 | percentile_99_9 | 128 → 1000 | 872 |
| mobilenet_v2 | int6 | maxabs | 128 → 1000 | 872 |
| mobilenet_v3_large | bfp6 | maxabs | 128 → 1000 | 872 |
| mobilenet_v3_large | fp6_e2m3 | maxabs | 128 → 1000 | 872 |
| mobilenet_v3_large | int6 | maxabs | 128 → 1000 | 872 |
| mobilenet_v3_large | int8 | maxabs | 128 → 1000 | 872 |
| resnet18 | fp6_e2m3 | percentile_99_9 | 128 → 1000 | 872 |
| resnet18 | fp6_e3m2 | percentile_99_9 | 128 → 1000 | 872 |
| resnet18 | int4 | maxabs | 128 → 1000 | 872 |
| resnet18 | int6 | maxabs | 128 → 1000 | 872 |
| resnet18 | nf4 | maxabs | 128 → 1000 | 872 |

## Deeper comparisons selected

| Group | Scope | Purpose |
| --- | --- | --- |
| Model-dependent recipe effects | 8 configurations | Common conventional anchor across all models; inspect top1 and top5 separately and isolate detector head calibration. |
| Six-bit representation and recipe interactions | 30 configurations | Compare same nominal payload width under both recipes. Shared scale metadata, block grouping and support cost are additional factors. |
| Seven-bit near-baseline candidates | 8 configurations | Small observed FP7/FP8 gaps deserve paired uncertainty and cost analysis; similar scores do not prove equivalence. |
| Four-bit model/representation failure contrast | 6 configurations | INT4, NF4 and MXFP4 differ substantially on ResNet18. Compare recipes and diagnose before attributing the gap to datatype alone. |
| Detector shared-scale alternatives | 3 configurations | Retain shared-format detector evidence despite lower quality than INT8; evaluate head-domain sensitivity and support costs. |

After balancing, the proposed integer new-recipe pilots are ResNet18 INT4 and MobileNetV2 INT4/INT6, with MSE scale-only and a [BRECQ block-reconstruction baseline](https://openreview.net/forum?id=POWv6hDd9XH): six new arms × 128 images = 768 evaluations.

Also select three nonuniform scale-search pilots: ResNet18 Posit8, ResNet18 NF4 and MobileNetV2 FP6 E2M3, each at 128 images after an offline calibration reconstruction/code-occupancy audit. These add 384 evaluations and directly test scaling choices beyond integer formats. Keep each codebook, precision and graph fixed; optimize reconstruction using training calibration data only.

Together these are **nine proposed new-recipe arms × 128 images = 1,152 candidate-image evaluations**. Implementations, calibration data, search budgets, operator policy and promotion thresholds must be frozen first. Changed operator exceptions require additional matched controls outside that nominal count. The exact-A E2 adaptation is not a reproduction of a published B baseline, and integer methods cannot be assumed to apply unchanged to nonuniform grids.

Detector follow-up starts with YOLO INT8 head-domain/calibration diagnostics on 128 images. Existing B already separates score/box scale domains. Any new exemption or DFL/box policy gets a distinct configuration and matched control.

All ten datatype families remain represented through either core comparisons or diagnosis. LOG8 remains a hardware-support candidate; Posit, binary and ternary need failure diagnosis; Q1.6 remains an alias/control case. The full 200-configuration ledger is preserved. No 5k/full-validation enrollment was selected before calibration/confirmation roles and budget are frozen.

## Reproduction and artifacts

- Analysis: `results/summaries/b-stage-paired-1k-v1/analysis.json`
- Paired cross-format and detector statistics: `results/summaries/b-stage-paired-1k-v1/deeper-paired-statistics.json`
- Complete CSV tables and selected extensions: `results/summaries/b-stage-paired-1k-v1/`
- Selected companion protocol: `public/experiments/configs/breadth-study/b-deeper-comparisons-v1.json`
- Figure: `results/figures/b-stage-recipe-effects-1k-v1.png`

Run `.venv/bin/python -m tools.analysis.b_stage_deeper_comparisons analyze` to rebuild the audit. The detector-bootstrap subcommand records the three selected comparisons with seed 20260927 and 500 draws. Then run `.venv/bin/python -m tools.analysis.b_stage_shortlist`. Analysis uses saved data and CPU only. The existing inference matrix, runtime sources and controller enrollments remain unchanged; their resume command does not launch these twelve newly selected extensions.
