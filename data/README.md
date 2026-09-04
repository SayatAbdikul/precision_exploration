# Data inputs

Track identities, checksums, preprocessing versions, and fixed sample lists—not raw ImageNet/COCO payloads.

## Calibration manifests

`manifests/calibration/` contains:

- ImageNet 2,000-image class-balanced training list, two images per class;
- COCO approximately 1,000–2,000-image stratified training list;
- deterministic selection seed and algorithm;
- dataset version/root identity and per-list content hash;
- alternative fixed-seed lists used only for finalist robustness.

COCO stratification should preserve approximate category frequency, object-size distribution, and objects-per-image distribution.

## Evaluation manifests

`manifests/evaluation/` contains fixed, hashed lists for:

- ImageNet approximately 1k screen;
- ImageNet approximately 5k/10k validation;
- ImageNet full validation;
- COCO screening/full validation as selected by the accepted method;
- any fixed network-logit verification set;
- representative trace-generation image sets.

Every compared candidate within a stage uses the identical ordered list.

## Separation rules

- Calibration and reported evaluation inputs do not overlap.
- The 1k screening list is never reused as calibration data.
- Checkpoint, dataset, preprocessing, and list hashes enter the experiment metadata/config identity.
- Raw dataset payloads live under the ignored `raw/` mount/location and are never committed.
- Checkpoint payloads live in `artifacts/checkpoints/`; model manifests record their external origin and hash.

## Required baseline records

For each workload record FP32 Top-1/Top-5 or mAP, checkpoint hash, exact preprocessing, dataset/evaluator version, deployment-graph version, and evaluation script identity before quantized comparisons begin.
