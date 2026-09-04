# Public precision-exploration publication plan

## Default strategy

The supplied plan recommends two papers by default:

1. a public numerical-format study produced from this repository;
2. a separate private MANT architecture paper produced elsewhere.

A combined co-design paper is considered only if a strong architecture–format interaction becomes a central result and the private work is separately approved for disclosure.

## Candidate public title

> Hardware-Cost-Aware Exploration of Sub-8-Bit Numerical Formats for Edge CNN Inference

## Public paper scope

- Systematic 4–8-bit study across INT/fixed, FP/minifloat, MX/BFP, Posit, logarithmic, codebook/non-uniform, and extreme low-cardinality families.
- ImageNet classification plus a tiny COCO detector.
- Controlled Experiment A and separately reported optimized Experiment B PTQ.
- Bit-exact arithmetic and network inference.
- W/A asymmetry, signedness, accumulator width/family, and MAC A/B/C sensitivity.
- Real generic RTL/PPA rather than proxy multiplier area alone.
- Scaling/conversion, metadata, memory, and generic PE/tile/system cost.
- Confidence-aware accuracy–energy–area–throughput Pareto fronts.
- Explicit comparison between raw arithmetic and all-in generic-system rankings.

## Required evidence before claims

| Claim class | Minimum evidence |
|---|---|
| Numerical quality | Frozen checkpoints/subsets, exact backend, full-validation results, paired confidence intervals |
| Datatype behavior | Public manifest, oracle/conformance, overflow/underflow/rounding definition |
| Optimized PTQ | Experiment B label plus support-hardware accounting |
| Accumulation | Exact accumulator/MAC semantics and matched configuration |
| Hardware area/timing | RTL/config/PDK/tool identity, common methodology, architecture Pareto search |
| Energy | Workload-derived activity for finalists; vectorless not used as headline result |
| Memory saving | Ideal, packed-logical, and physical evidence clearly separated |
| System efficiency | Generic normalized model with quality attached to every point |
| Pareto dominance | Per-workload and uncertainty-aware analysis, not one arbitrary weighted score |

## Core figures and tables

- Workload FP32 baselines and PTQ anchors.
- Candidate format/manifest summary.
- Controlled A versus optimized B quality.
- W/A and accumulator/MAC-model sensitivity.
- Layer sensitivity, SQNR/MSE, overflow/underflow, and calibration robustness.
- Quality versus energy/inference.
- Quality versus images/s/mm².
- Energy/inference versus area.
- Quality versus effective bits/value.
- Raw arithmetic versus all-in generic-system Pareto.
- Compute/memory/scaling/control energy breakdown.
- Raw/core versus all-in scaling/conversion area and energy.
- Ideal versus packed versus physical memory saving.
- Trace convergence and finalist physical-method limitations.

Generated figures and tables live in `results/figures/` and `results/tables/`; their generating code lives in `public/analysis/`.

## Publishability criteria from the plan

- Multiple representation families, not only FP splits.
- Bit-accurate numerical evaluation.
- Real RTL and physical evidence.
- ImageNet plus detection.
- Clear Pareto findings.
- At least one supported new insight or derived numerical/hardware design rule.

The plan suggests that hardware-aware automated search, a newly derived format/accumulation scheme, heterogeneous numerical formats, post-layout validation, and superiority to strong INT8/FP8/MX anchors would strengthen a top-tier architecture/EDA submission. Heterogeneous/MANT-specific conclusions remain outside this repository unless supported by the public generic study alone.

## Public reproduction package

- source-document provenance and accepted decision records;
- datatype manifests and conformance vectors;
- calibration/evaluation manifest hashes;
- checkpoint/preprocessing identities;
- PTQ and exact-inference code;
- canonical experiment configs and database schema;
- generic RTL and ICS55 flow scripts where releasable;
- summary tables, figure-generation code, and limitation statements;
- public Pareto-plus-guard candidate package schema.

No MANT ISA, tile, compiler, mapping, simulator, RTL, interconnect, floorplan, or proprietary result belongs in this paper or repository.
