# Decision register

This register distinguishes the supplied documents' statements from decisions accepted by this project. F1–F8 were explicitly accepted without changes by the project owner on 2026-09-04.

Repository status values are **open/in-repo**, **open/external**, **accepted**, and **superseded**.

## F1–F8 source-stated starting decisions

| ID | Source-stated decision | Reason in roadmap | Still open | Repository status |
|---|---|---|---|---|
| F1 | Experiment A canonical MAC is Model C: full/exact product enters a sufficiently wide accumulator; the accumulator rounds in its own format | Prevent premature product rounding from dominating the first datatype comparison | Models A/B/C remain finalist comparisons | Accepted 2026-09-04 |
| F2 | Experiment A uses a sufficiently wide, family-appropriate accumulator | Measure representation quality before deliberately squeezing accumulation | Native, 8/10/12/16, INT32, FP, quire/Kulisch choices remain later dimensions | Accepted 2026-09-04 |
| F3 | Weight and activation formats are independently configurable | Avoid a rewrite for W≠A | Broad A may begin mainly with W=A; asymmetry is later | Accepted 2026-09-04 |
| F4 | Accumulator is independently configurable from W and A | Keep numerical architecture explicit | The winning accumulator remains open | Accepted 2026-09-04 |
| F5 | Static PTQ; no retraining; eval/folded BN; fixed calibration/evaluation subsets; MSE where scale selection applies; signed symmetric INT zero point 0; no bias correction/reconstruction in A | Reproducible controlled baseline | Family-specific PTQ improvements belong to B | Accepted 2026-09-04 |
| F6 | Experiment A has no optional external scale; only intrinsic/necessary family scaling | Reduce optimizer bias and expose intrinsic range behavior | General/power-of-two scale, bias/range, block size, and granularity move to B | Accepted 2026-09-04 |
| F7 | Public Stage A is architecture-independent with respect to MANT | Prevent current MANT assumptions from constraining the scientific search | Downstream MANT may retain or change its structure in another repository | Accepted 2026-09-04 |
| F8 | Export the public Pareto set plus a conservative guard set, not one winner | Protect against architecture interaction changing a near-Pareto ranking | Exact guard composition follows the public study | Accepted 2026-09-04 |

### F1–F8 acceptance record

- Decision: accept all eight roadmap defaults unchanged.
- Approved by: project owner.
- Date: 2026-09-04.
- Affected contracts: all files in `docs/contracts/` and the three Phase 0 schemas.
- Later gates remain open as listed below.

## D1–D11 in-repository gates

| Gate | Decision | Evidence required before decision | Main effect | Status |
|---|---|---|---|---|
| D1 | Initial candidate manifest set | Explicit definitions, relevance, implementability, redundancy review | Oracle tables, kernels, screening job count; preserve all major families | Accepted 2026-09-06; standard-semantics correction 2026-09-08 (see D1 record); set SHA-256 `986344a9dc6ef5b4c7a8194e4675964e170345c43d50f95bbb3cee0dfcd82c0b` |
| D2 | Fast backend strategy per family/width | Algorithmic/LUT/predecoded microbenchmarks plus exhaustive correctness and validated counters | Runtime/packing only; scientific result must remain unchanged | Accepted 2026-09-11 for measured FP6/INT8 GEMM scope |
| D3 | Include EfficientNet in the main full suite | Measured exact-engine speed and compute budget | Workload breadth and job count | Accepted 2026-09-09: retain as optional; four core workloads remain mandatory (see D3 record) |
| D4 | Promotion from 1k to 5k/10k | Paired statistics, confidence, failure diagnosis, family/hardware context | Numerical survival; uncertainty promotes | Open/in-repo |
| D5 | Which promoted configs receive full ablations | 1k/10k quality, layer sensitivity, family role, rough hardware cost | Controls W/A × accumulator × scale expansion | Open/in-repo |
| D6 | Reasonable Experiment B policy per family | Pilot benefit and support-hardware cost | Final optimized PTQ, scale RTL, metadata, memory | Open/in-repo |
| D7 | Approximately 8–15 generic RTL candidates | Full-dataset quality/confidence, family representation, hardware plausibility | Generic RTL/PPA effort | Open/in-repo |
| D8 | Small RTL Pareto set representing each numerical configuration | Architecture/pipeline/frequency synthesis sweep | Generic PE/tile/system modeling | Open/in-repo |
| D9 | Physical memory evidence level | Verified ICS55 macro/compiler/OpenRAM/inference support | Physical area/energy credibility and possible Pareto ordering | Open/in-repo |
| D10 | Final power methodology for 2–4 public physical finalists | Trace-convergence pilot and delay-aware flow capability | Energy fidelity and close-point ordering | Open/in-repo |
| D11 | Approximately 5–10 Pareto plus guard candidates to export | Complete public generic end-to-end Pareto front | External downstream candidate count; public conclusions unchanged | Open/in-repo |

## D12–D14 external downstream gates

These are preserved because they explain what the exported package enables. They must be decided in a separate MANT repository.

| Gate | External decision | Required evidence | Status |
|---|---|---|---|
| D12 | Current compatibility, small extension, or major redesign per candidate | Imported quality delta, mapping, cycle and conversion cost | Open/external |
| D13 | Final quality constraint and hardware objective | Complete downstream MANT Pareto | Open/external |
| D14 | Whether mixed precision/heterogeneous tiles are justified | Uniform-precision results and measured specialization/Pareto gap | Open/external |

## Decision record template

```text
ID:
Title:
Status: proposed | accepted | superseded
Owner:
Reviewers:
Date:
Scope: in-repository | external
Alternatives considered:
Evidence and artifact hashes:
Decision and rationale:
Affected contracts/configs/code:
Expected scientific impact:
Supersedes / superseded by:
```

No gate is closed merely because a provisional value appears in the source plan.

## D1 acceptance record

- Decision: accept the 25 candidates enumerated in `docs/decisions/d1-initial-candidate-set.md`.
- Approved by: project owner (the owner approved the proposed datatype list without changes).
- Date: 2026-09-06.
- Evidence: complete manifests under `public/formats/manifests/accepted/`, aggregate SHA-256 `986344a9dc6ef5b4c7a8194e4675964e170345c43d50f95bbb3cee0dfcd82c0b`, and the exhaustive table index under `public/formats/conformance/`.
- Scope: initial public candidate set only. D2–D11 were open at this acceptance date; later decisions are recorded separately below.

## D3 acceptance record

- Decision: retain EfficientNet as optional; keep ResNet-18, MobileNetV2, MobileNetV3 Large and YOLOv8n as the main suite.
- Authority: implementation decision under the owner's instruction to finish Phase 2; preserves the existing core scope.
- Date: 2026-09-09.
- Evidence and rationale: `docs/decisions/d2-d3-phase2-runtime-and-breadth.md` and the reproducible `results/summaries/phase2-decision-evidence.json` with artifact hashes. Eight native-resolution YOLO images measure 107.32 seconds/image on CUDA; a 1k screen at that one configuration projects to 29.81 hours before other pipeline costs.
- Budget policy: no additional mandatory workload is allocated while core execution is already expensive and no incremental compute allowance has been committed. The total available project budget remains unspecified; this decision does not invent one.
- Reopen when: measured core-screening throughput and an explicit incremental budget justify staging and timing EfficientNet.
- Scientific effect: no core workload or candidate family is removed; no claim is made about EfficientNet results. D2 was still open at this acceptance date.

## D2 acceptance record

- Decision: use the existing predecoded strategy for the measured FP6 E3M2 and INT8 convolution/pointwise GEMM shapes on C++/CUDA, accounting for operand preparation and reuse.
- Authority: implementation decision under the owner's instruction to finish Phase 2; the owner supplied the authenticated counter capture.
- Date: 2026-09-11.
- Evidence: current-source, identical-output strategy benchmarks plus six validated Nsight Compute launches measuring duration, DRAM bytes/bandwidth and achieved occupancy. See `results/summaries/phase2-cuda-profile.json`, `results/summaries/phase2-decision-evidence.json` and `docs/decisions/d2-d3-phase2-runtime-and-breadth.md`.
- Scope limits: three unprofiled latency samples per strategy; profiled counters are single-launch measurements. No alternative depthwise strategy or fastest implementation for other families is established.
- Scientific effect: runtime strategy only; numerical semantics and scientific outputs remain unchanged. Other families retain correctness-validated implementations until measured strategy comparisons are available.
