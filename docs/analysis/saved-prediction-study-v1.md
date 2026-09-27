# Saved-prediction analysis and frozen comparison matrix

Recomputed 200 saved B configurations (25 datatypes × 4 models × 2 recipes),
including dataset-level COCO AP, and verified paired predictions for eight exact-A screens.
No new quality inference was performed. All reported panels are development data.

Machine-readable evidence: `results/summaries/saved-prediction-study-v1/analysis.json`.
Frozen design: `public/experiments/configs/breadth-study/comparison-matrix-v1.json`.

## Paired recipe effects

Differences below are percentile minus maxabs, in percentage points. The classifier
intervals are pointwise paired image-bootstrap intervals (5,000 draws), not simultaneous
guarantees. Detector entries have point estimates only; AP is never averaged per image.

| Model | Percentile higher / tied / lower | Largest absolute recipe difference |
| --- | --- | --- |
| resnet18 | 10 / 3 / 12 | int5: +50.000 pp |
| mobilenet_v2 | 11 / 9 / 5 | int6: +50.000 pp |
| mobilenet_v3_large | 8 / 12 / 5 | int8: +66.406 pp |
| yolov8n | 11 / 1 / 13 | int8: -28.509 pp |

## Retained exact A

| Model / format | Exact top-1 (1k) | Loss vs paired FP32 |
| --- | --- | --- |
| mobilenet_v2 / int4 | 0.20% | -71.90 pp |
| resnet18 / int5 | 48.30% | -21.80 pp |
| resnet18 / int4 | 12.30% | -57.80 pp |
| resnet18 / int6 | 62.50% | -7.60 pp |
| mobilenet_v2 / int5 | 6.40% | -65.70 pp |
| mobilenet_v2 / int8 | 70.60% | -1.50 pp |
| resnet18 / int8 | 68.40% | -1.70 pp |
| mobilenet_v2 / int6 | 46.00% | -26.10 pp |

The JSON also contains A/B comparisons on the identical 128 image IDs.
These differences mix recipe and execution changes and cannot identify their causes.

## Frozen next comparisons

- E0: eight accepted integer graphs, eight-image exact compatibility each.
- E1: twelve family/model coverage cases, 32 images initially, up to 128; matched QDQ controls need implementation.
- E2: four severe-loss cases, 128-image controlled clipping/rounding ablations.
- E3 B: 171 existing QDQ configurations enrolled to 1k, preserving every datatype/model pair; not launched.
- Larger exact runs remain gated by compatibility, matching, measured throughput and an explicit compute budget.
- Final confirmation requires an exposure audit; previously evaluated images are not newly untouched data.

The matrix is frozen after observing the old development panel. It is prospective for new
runs, not a retrospective preregistration. All ten accepted datatype families and four models
are represented in E1; unavailable exact arms remain explicitly blocked.
