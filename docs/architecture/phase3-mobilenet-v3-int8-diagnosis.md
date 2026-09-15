# MobileNetV3 INT8 pilot diagnosis

The strict eight-image MobileNetV3 Large INT8/FP64 CPU pilot loses four Top-1
correct predictions relative to FP32. Four selected one-layer interventions do
not reproduce that loss. The cause remains unresolved; this is diagnostic
evidence, not a catastrophic-format label or a D4 decision.

## Paired results on the same eight images

| Execution | Top-1 correct | Top-5 correct | Interpretation |
| --- | --- | --- | --- |
| Frozen FP32 | 6/8 | 8/8 | Paired baseline |
| Strict INT8 graph, FP64 accumulation | 2/8 | 6/8 | End-to-end pilot loss |
| Only `features_16_0` quantized | 6/8 | 8/8 | Does not reproduce the correctness loss alone |
| Only `classifier_3` quantized | 6/8 | 8/8 | Does not reproduce the correctness loss alone |
| Only residual `add_5` quantized | 6/8 | 8/8 | Does not reproduce the correctness loss alone |
| Only residual `add_6` quantized | 6/8 | 8/8 | Does not reproduce the correctness loss alone |

The MAC interventions use the strict candidate's input/output encodings, quantized
weights and declared accumulator for the selected MAC, then resume the folded
FP32 model. Residual interventions store each FP32 branch in its declared input
domain, align both branches, apply the declared accumulator and output store,
then resume FP32. Their fresh FP32 Top-5 predictions match the frozen baseline on all
eight images. Zero observed correctness loss in these studies does not prove
that either layer is harmless on other inputs or when combined with preceding
quantized layers.

The strict pilot's paired Top-1 delta is -0.50, with the frozen 95% paired
bootstrap interval [-0.875, -0.125]. Its Top-5 delta is -0.25, interval
[-0.625, 0]. These small-subset intervals are retained as `PILOT_ONLY`; they
cannot substitute for the fixed-1k screen.

## Layer evidence

Sampled MSE is highest at residuals `add_6` (10.6174), `add_5` (9.6593), and
the MAC `features_16_0` (9.6395). MSE depends on layer signal magnitude, so this
ordering is an inspection aid rather than a sensitivity ranking.

| Layer | Strict graph sampled MSE | One-layer sampled MSE |
| --- | --- | --- |
| `features_16_0` | 9.639487 | 0.032068 |
| `classifier_3` | 0.613934 | 0.000763 |
| `add_5` | 9.659257 | 0.015275 |
| `add_6` | 10.617414 | 0.011100 |

The much smaller isolated errors are consistent with substantial error arriving
from preceding layers in the strict graph. This is an inference from the four
interventions; it does not identify a unique cause or exclude interactions among
layers. The strict classifier's observed stores include one overflow event,
which is an output-quantizer event and does not establish accumulator overflow.

All 1,120 native layer shapes match the retained FP32 shapes. The local FP64
bounds and hard-activation checks provide additional evidence, but complete
native backend comparison and whole-graph accumulator sensitivity remain open.

## Evidence and next checks

- [Strict paired analysis](../../results/summaries/phase3-analysis-fcb112dfd35e.json)
- [Strict layer diagnostics](../../results/summaries/phase3-diagnostics-fcb112dfd35e.json)
- [Native shape comparisons](../../results/summaries/phase3-shape-check-fcb112dfd35e.json)
- [`features_16_0` intervention](../../results/summaries/phase3-sensitivity-193b3ed568c8.json)
- [`classifier_3` intervention](../../results/summaries/phase3-sensitivity-e803f46b77fb.json)
- [`add_5` intervention](../../results/summaries/phase3-sensitivity-dd8f557ffb55.json)
- [`add_6` intervention](../../results/summaries/phase3-sensitivity-3f3cfe6c9fba.json)

These reports retain hashes of their image records, configuration, graph and
implementation evidence. The configuration is
`a1b5c410eccdccab605427bcb12cdbc5fdfac2586fc8497ae4a06d4f41c53630`.

Next, validate the CUDA pilot and investigate earlier operations or cumulative
effects across successive blocks. The versioned residual runner now captures
both FP32 branch inputs and keeps separate evidence without changing the prior
one-MAC studies. Preserve the strict baseline while diagnosing cumulative
representation error, scale/alignment behavior and accumulator sensitivity.
