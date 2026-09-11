# Phase 2 completion checklist

Status: **complete — 2026-09-11**. Remaining Phase 2 work: **none**. Estimated time remaining: **0**. The [final verification report](../../../results/summaries/phase2-final-verification.json) has `status: complete` and `remaining_gates: []`.

This file began as the requested remaining-work list with time estimates. The completed gates and their evidence are retained below.

| Gate | Final result |
| --- | --- |
| Frozen dataset recovery | All ImageNet training/calibration 2k, screen 1k and evaluation 10k payloads verified; all COCO selections verified. The supplied Phase 1 dataset export independently matches all six frozen selections. |
| Calibration | All eight FP6 E3M2/INT8 artifacts validated across the four core models, each covering its frozen 2,000-image training population. |
| Native network conformance | Eight images per core model, with **3,720 matching C++/CUDA layer comparisons**; classifier output tensors also match. |
| Format conformance | All **25 accepted formats** match the rational oracle on C++/CUDA. |
| Frozen FP32 baselines | All four original prediction artifacts match their hashes. Official COCOeval rescoring of the restored YOLO JSON reproduces both frozen mAP metrics exactly. |
| Test evidence | **275 full CPU tests passed**, including the focused 253-test Phase 2 suite; retained **93 CUDA tests passed** against the unchanged engine source. |
| D2 runtime strategy | Accepted for measured FP6 E3M2/INT8 GEMM shapes after all six Nsight Compute launches validated. Predecoded strategy retained with preparation/reuse limits documented. |
| D3 workload breadth | Accepted: four core models retained; EfficientNet remains optional. |
| Generic hardware pilots | 16 exhaustive simulation/synthesis runs and 48 cell-delay timing/vectorless-power runs verified for the H2 pilot scope. |
| Final verification | Complete; no missing metrics, profiling/decision artifact errors, failing tests or remaining gates. |

## Final profiling evidence

The owner authenticated process-only collection on 2026-09-11. Capture
`artifacts/benchmarks/phase2/counter-attempt-zawqo23w/capture.json` completed at
04:47:52 UTC. Raw CSV SHA-256:
`6865230b1d209bb13c55e9c0495cb42c274d0c4206943631ed0cbaae5bb8df3e`.
All launch, input/output, source and artifact identities validate.

Across the six launches, achieved occupancy is **20.14–20.72%** and measured
DRAM bandwidth is **41.00–57.68 GB/s**. These are single-launch counter
measurements; the separate repeated benchmarks determine the runtime strategy.
No GPU driver policy was changed.

The earlier multi-hour recovery continuation completed on 2026-09-10 at
20:37 UTC, approximately 11 hours 45 minutes after launch. Recovery and counter
collection are finished; no background completion job is required.

## Scope retained for later phases

D2 acceptance is limited to the measured FP6/INT8 GEMM shapes and operand reuse
assumptions. It does not establish the fastest strategy for every accepted
family or an alternative depthwise strategy. Phase 3 Experiment A still needs
per-configuration sufficiently wide accumulator acceptance, including mapped
bias headroom. H2 results are post-synthesis pilot evidence, not routed or
workload power. These boundaries do not represent unfinished Phase 2 gates.

The strict unscaled FP6 detector's zero-mAP result remains a diagnosed candidate
range/precision failure; arithmetic semantics and frozen expectations were not
changed to make it pass.

## Evidence

- [Final verification](../../../results/summaries/phase2-final-verification.json)
- [GPU profiling](../../../results/summaries/phase2-cuda-profile.json)
- [D2/D3 evidence](../../../results/summaries/phase2-decision-evidence.json)
- [D2/D3 decision record](../../decisions/d2-d3-phase2-runtime-and-breadth.md)
- [Phase 1 dataset import audit](../../../results/summaries/phase2-phase1-import-audit.json)
- [Phase 1 artifact import and metric audit](../../../results/summaries/phase2-phase1-artifact-import-audit.json)
- [Phase 2 roadmap](phase-02-exact-engine.md)
- [Engine guide and reproduction commands](../../architecture/exact-engine.md)
