# Decision register

This register distinguishes the supplied documents' statements from decisions accepted by this project. “Source-frozen” means the roadmap labels the decision frozen; it does not substitute for a repository review record.

Repository status values are **source-frozen/unratified**, **open/in-repo**, **open/external**, **accepted**, and **superseded**.

## F1–F8 source-stated starting decisions

| ID | Source-stated decision | Reason in roadmap | Still open | Repository status |
|---|---|---|---|---|
| F1 | Experiment A canonical MAC is Model C: full/exact product enters a sufficiently wide accumulator; the accumulator rounds in its own format | Prevent premature product rounding from dominating the first datatype comparison | Models A/B/C remain finalist comparisons | Source-frozen/unratified |
| F2 | Experiment A uses a sufficiently wide, family-appropriate accumulator | Measure representation quality before deliberately squeezing accumulation | Native, 8/10/12/16, INT32, FP, quire/Kulisch choices remain later dimensions | Source-frozen/unratified |
| F3 | Weight and activation formats are independently configurable | Avoid a rewrite for W≠A | Broad A may begin mainly with W=A; asymmetry is later | Source-frozen/unratified |
| F4 | Accumulator is independently configurable from W and A | Keep numerical architecture explicit | The winning accumulator remains open | Source-frozen/unratified |
| F5 | Static PTQ; no retraining; eval/folded BN; fixed calibration/evaluation subsets; MSE where scale selection applies; signed symmetric INT zero point 0; no bias correction/reconstruction in A | Reproducible controlled baseline | Family-specific PTQ improvements belong to B | Source-frozen/unratified |
| F6 | Experiment A has no optional external scale; only intrinsic/necessary family scaling | Reduce optimizer bias and expose intrinsic range behavior | General/power-of-two scale, bias/range, block size, and granularity move to B | Source-frozen/unratified |
| F7 | Public Stage A is architecture-independent with respect to MANT | Prevent current MANT assumptions from constraining the scientific search | Downstream MANT may retain or change its structure in another repository | Source-frozen/unratified; reinforced by user scope |
| F8 | Export the public Pareto set plus a conservative guard set, not one winner | Protect against architecture interaction changing a near-Pareto ranking | Exact guard composition follows the public study | Source-frozen/unratified |

## D1–D11 in-repository gates

| Gate | Decision | Evidence required before decision | Main effect | Status |
|---|---|---|---|---|
| D1 | Initial candidate manifest set | Explicit definitions, relevance, implementability, redundancy review | Oracle tables, kernels, screening job count; preserve all major families | Open/in-repo |
| D2 | Fast backend strategy per family/width | Algorithmic/LUT/predecoded microbenchmarks plus exhaustive correctness | Runtime/packing only; scientific result must remain unchanged | Open/in-repo |
| D3 | Include EfficientNet in the main full suite | Measured exact-engine speed and compute budget | Workload breadth and job count | Open/in-repo |
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
