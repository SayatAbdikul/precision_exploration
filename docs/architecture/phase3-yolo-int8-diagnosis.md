# YOLOv8n INT8 output-store diagnosis

The strict INT8/FP64 one-image C++ pilot completed in 12,059.85 seconds
(about 3 hours 21 minutes), including diagnostics. It produced no detections;
FP32 mAP50–95 on that same image was 0.4621768427. All 176 native layer shapes
match the retained FP32 observations. This is a pilot observation, not a D4
classification or a complete screen.

## Isolated final-store experiment

An eight-image intervention applies only the prepared INT8 encoding of the
final `model_22` tensor to a freshly observed FP32 detector head. Every upstream
operation remains FP32. The unchanged postprocessor evaluates both the fresh
FP32 head and its quantized counterpart. Complete FP32 heads, stored codes,
predictions, original image shapes and source hashes are retained for replay.

| Metric | Fresh FP32 | Final store quantized | Paired delta, 95% interval |
| --- | ---: | ---: | --- |
| mAP50–95 | 0.432137 | 0 | -0.432137 [-0.608526, -0.375239] |
| mAP50 | 0.576866 | 0 | -0.576866 [-0.789918, -0.509393] |

Intervals use the frozen 2,000 joint image resamples and seed 310911, with
repeated-image multiplicity preserved. These eight selected diagnostic images
do not estimate the complete screen's uncertainty. The fresh FP32 prediction
lists differ from the original frozen lists on all eight images; the comparison
therefore uses the same freshly observed head on both sides. It does not replace
the frozen screen1k reference or reuse its metric as this study's comparator.

The retained output scale is approximately 0.21102811755426018. Its largest
positive INT8 value is 26.800570929391043. The FP32 output mixes four coordinate
channels in pixel units with 80 probability channels. Full-head inspection finds
28,916–31,793 of 33,600 coordinate elements above the representable maximum on
each image. FP32 coordinate maxima range from about 636.5 to 643.8. The store
also rounds 671,771–671,989 positive score elements to zero out of 672,000 score
elements per image. Some detections survive postprocessing, but none achieves
nonzero COCO AP on this subset.

Thus this final store alone is sufficient to reproduce zero mAP on the eight
diagnostic images. This does not prove it is the only failure in the strict
graph, establish a general INT8 limitation, or validate its pending accumulator
and native-backend gates.

## Frozen sampling coverage

The verified training2k observation cache uses
`16_evenly_spaced_flat_positions_per_node_per_image_v1`, implemented in
`public/quantization/calibration/observer.py`. For the final `[1,84,8400]` head,
only one of the 16 selected elements belongs to a coordinate channel: channel 0,
spatial position 0. Coordinate channels 1, 2 and 3 are never sampled. The other
15 positions belong to score channels. These positions repeat for every image.

Hash verification proves that the calibration used the frozen cache; it does
not establish representative sampling. The geometric omission is a concrete
calibration-coverage confound for this heterogeneous tensor. Even a larger
single scale would trade coordinate range against score resolution. No revised
scale has been selected from these evaluation images, and the strict campaign
and calibration artifacts remain unchanged.

Before attributing a detector failure to a representation, review coverage on
the frozen training set and, if a calibration correction is adopted, give that
experiment a separate version with preserved original evidence. Recheck all
affected INT encodings and their accumulator headroom. Any separate coordinate
and score storage proposal changes the graph policy and must be identified
explicitly; it cannot silently enter the uniform strict baseline.

## Reproduction and evidence

```bash
.venv/bin/python -m tools.run.phase3_head_store_sensitivity
```

The command verifies and reuses saved image evidence. Its summary is
`results/summaries/phase3-head-store-sensitivity.json`; the immutable study job
is `0a56a9e599a020c39b17f8b722af668281aa2b2f4718231a98b4c2b30807c3ef`.
The strict pilot is
`0a6b0b393f77555c11d267b908a633a8b505522c0d66573ae0c84e2884965a89`,
with paired, diagnostic and shape reports bearing its 12-character prefix.
The final-store study remains `DIAGNOSTIC_ONLY`; no detector format or family
has been pruned on this evidence.
