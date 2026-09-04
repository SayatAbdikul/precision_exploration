# Experiment A contract

Version: **1.0.0**
Status: **accepted for Phase 0**
F5–F6 accepted by project owner: **2026-09-04**

## Purpose

Experiment A is the broad, controlled PTQ comparison. It is designed to expose intrinsic representation behavior before optional family-specific tuning. All candidates start from the same pretrained checkpoints; there is no retraining.

Experiment B is separate and asks for the best reasonable PTQ result per surviving family using optional scaling and other refinements.

## Canonical deployment graph

The required order is:

```text
FP32 checkpoint
    -> evaluation mode
    -> fold BatchNorm into preceding Conv
    -> freeze deployment graph
    -> collect calibration statistics
    -> PTQ
```

Folding after weight quantization is not allowed because it changes the deployed weight distribution and invalidates the calibrated scale. The frozen graph and checkpoint are content-addressed.

## Controlled PTQ rules

| Dimension | Experiment A baseline |
|---|---|
| Training | Static PTQ only; no retraining or QAT |
| Calibration objective | MSE where scale selection is required |
| Weights | Per-output-channel scaling for conventional INT mapping |
| Activations | Per-tensor scaling for conventional INT mapping |
| Signed INT | Symmetric, zero point 0 |
| Rounding | Nearest representable value; RNE where applicable |
| Bias correction/reconstruction | Disabled |
| Calibration inputs | Identical fixed subset for all datatype comparisons on a model |
| Evaluation inputs | Fixed list, disjoint from calibration |
| Weight/activation formats | Independently configurable; begin mainly with uniform W=A |
| MAC | Model C |
| Accumulator | Sufficiently wide and family appropriate |
| Operator regime | Strict low precision |

## Scaling rule and source reconciliation

The early plan proposed an MSE-optimized general external scale for all compatible ordinary formats. The later roadmap explicitly replaces that baseline. The canonical roadmap rule is:

- **No optional external scaling in Experiment A.**
- INT/fixed-point uses the scale necessary for its conventional value mapping.
- MX/BFP uses its native shared-scale semantics.
- FP/minifloat, Posit, logarithmic, and other self-scaling formats are evaluated directly unless their format definition intrinsically requires a scale.
- Each intrinsic/necessary scale is calibrated with the common MSE procedure where scale selection is part of that family.

The following move to Experiment B:

- general external scale for self-scaling formats;
- power-of-two scale ablation;
- scale granularity beyond the A baseline;
- exponent bias/range tuning;
- MX/BFP block-size sweeps;
- format-specific calibration improvements;
- practical first/last-layer exceptions;
- bias correction/reconstruction if later admitted by an explicit policy.

### Per-family Experiment A scaling table

| Family | Experiment A | Experiment B/later only |
|---|---|---|
| INT/fixed point | Conventional required mapping scale; MSE selection; signed symmetric zero point 0 | Power-of-two constraint, asymmetric activations, other granularities |
| FP/minifloat | Direct format; no optional external scale | General external scale, power-of-two scale, bias/range tuning |
| Posit | Direct standard/manifest semantics | General or power-of-two external scale |
| Log/power-of-two | Direct manifest semantics; only definition-required scale | General/power-of-two variants where meaningful |
| Codebook/non-uniform | Direct codebook; scale only when required by the format definition | Format-specific scale/codebook optimization |
| MX/BFP | Native shared scale; initial block axis K and block size 32 where defined by the selected manifest | Block sizes 16/32/64 and alternative scale/granularity policy |

The experiment config and datatype manifest jointly determine whether a scale is intrinsic/required; the backend may not infer an optional scale from observed tensor data.

## Scale search where the baseline requires one

For conventional INT/fixed-point mapping and other intrinsic/necessary scale choices:

1. evaluate approximately 100 coarse clipping/scale candidates;
2. locate the lowest-MSE region;
3. evaluate approximately 50 finer candidates around that region;
4. store the lowest-MSE scale with full configuration/provenance.

Candidates never share a forced numerical scale merely because they have the same bit width.

## Calibration datasets

### ImageNet

- Source: training split.
- Size: 2,000 images.
- Sampling: class balanced, two images per class.
- Selection: deterministic fixed seed and saved list/hash.
- Reuse: identical list across datatype comparisons for a model.
- Evaluation overlap: none.

### COCO

- Source: training split.
- Size: approximately 1,000–2,000 images.
- Sampling: fixed stratified subset preserving approximate category frequency, object-size distribution, and objects per image.
- Selection: deterministic fixed seed and saved list/hash.
- Evaluation overlap: none.

The approximately 1,000-image screening subset is an evaluation set, not a calibration set.

## Calibration robustness

Use one frozen calibration subset for the broad search. For approximately 5–10 finalists, repeat calibration using several alternative fixed seeds and report whether degradation and ranking remain stable.

## Experiment A output

For each configuration store:

- absolute Top-1/Top-5 or mAP and delta from the same-subset FP32 reference;
- per-image predictions/correctness and optional logit/confidence summary;
- per-layer distribution, SQNR, MSE, zero/outlier, overflow/underflow/sign-clipping statistics;
- configuration, manifest, checkpoint, graph, subset, backend, and code hashes;
- runtime and job status/error record.

The 1k stage labels configurations PROMISING, UNCERTAIN, or CATASTROPHIC/BROKEN. Promising and uncertain configurations advance; catastrophic results are diagnosed before pruning.
