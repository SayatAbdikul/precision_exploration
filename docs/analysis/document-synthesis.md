# Plan and roadmap synthesis

## Source roles

The comprehensive plan is the methodological source. It specifies the research questions, workload suite, datatype families, PTQ rules, arithmetic models, exact inference engine, scheduling, statistics, pruning, generic RTL and physical methodology, memory/scaling accounting, Pareto analysis, conformance, publication strategy, risks, and a provisional first experiment.

The roadmap is the sequencing source. It refines the methodology into phases 0–11, four work lanes, work packages WP0–WP11, source-stated starting decisions F1–F8, evidence gates D1–D14, exit criteria, and parallel work clusters.

For this repository, the documents are interpreted as follows:

- The public, architecture-independent path is implemented here.
- MANT-specific phases are documented only to define the downstream handoff and are not represented by implementation directories.
- A statement inside either source remains a plan item or source-stated decision unless the project records acceptance in the decision register.
- Later roadmap refinements resolve conflicts with earlier plan wording for the canonical sequence.

## Reconciled research scope

The public question is:

> Which representations at 8 bits and below provide the best quality–hardware-efficiency trade-offs for edge CNN inference under post-training quantization?

The study must jointly characterize:

- weight and activation formats, including asymmetric W/A combinations;
- accumulator format and MAC Models A/B/C;
- scaling kind and granularity, signedness, and block size where applicable;
- operator and deployment-graph semantics;
- exact end-to-end quality on classification and detection;
- primitive, MAC/FPU, scaling/conversion, memory, and generic PE/tile costs;
- synthesis/P&R, power, latency, II, throughput, and energy;
- Pareto fronts under different quality and resource budgets.

## Reconciliations required by the two sources

### Experiment A scaling

An earlier plan section describes an MSE-optimized general external scale for every compatible ordinary datatype. The later roadmap explicitly freezes a revised rule in F6 and N0.3: **Experiment A uses no optional external scaling**. Only family-intrinsic or necessary scaling is permitted:

- INT/fixed point uses its conventional mapping scale;
- MX/BFP uses its native shared scale;
- direct FP/minifloat/Posit/logarithmic formats use no extra scale unless their definition requires one.

General external scales, power-of-two scale ablations, exponent-bias/range tuning, scale-granularity sweeps, and MX/BFP block-size sweeps move to Experiment B. This later roadmap rule is the canonical plan recorded in `docs/contracts/experiment-a.md`.

### Early promotion

The provisional first-experiment text says to promote only the strongest candidates after the 1k screen. The detailed pruning section and roadmap D4 refine this: **promising and uncertain candidates both advance**, together with family representatives, near-Pareto points, expected hardware specialists, and cross-workload specialists. Only diagnosed broken or truly catastrophic configurations are removed early.

### Hardware hierarchy under repository scope

Early plan language uses primitive → MAC/FPU → MANT tile → whole chip. The later public/private boundary replaces the public upper levels with a generic normalized PE/tile/system model. This repository therefore evaluates:

1. primitive arithmetic;
2. complete MAC/FPU and accumulator;
3. scaling/conversion and memory;
4. generic PE/tile and normalized system.

MANT tile and chip evaluation is an external private follow-on.

### Provisional quantities

Candidate counts, frequencies, FP accumulator layouts, area budgets, trace lengths, pruning buffers, and final quality thresholds are not fixed. The source provides pilots or examples that become decisions only after the named evidence gate.

## Canonical public flow

```text
frozen checkpoint + deployment graph + dataset manifests
                         |
                         v
         datatype manifests + PTQ calibration
                         |
                         v
  exact oracle -> C++/CUDA inference -> paired statistics
                         |
                         v
         conservative 1k -> 10k -> full funnel
                         |
                         v
 generic arithmetic/scaling/memory RTL + ICS55 PPA/power
                         |
                         v
     confidence-aware generic end-to-end Pareto fronts
                         |
                         v
       versioned public Pareto + guard packages
                         |
                         v
             external private MANT repository
```

## Workload and validation core

- Mandatory classification workloads: ResNet-18, MobileNetV2, MobileNetV3.
- Recommended core detector: YOLOv8n or a similar tiny detector on COCO.
- EfficientNet-B0/Lite0 is recommended but remains subject to D3 after engine-speed evidence.
- ImageNet calibration: 2,000 class-balanced training images, two per class, fixed seed/list, no evaluation overlap.
- COCO calibration: approximately 1,000–2,000 stratified training images, fixed seed/list, no evaluation overlap.
- Screening and validation subsets are fixed and identical across compared candidates.
- Finalists repeat calibration with several alternative seeds to test ranking stability.

## Why the repository is component-oriented

Roadmap phases are temporary execution states; manifests, quantization, inference, RTL, flows, and analysis are durable components. Code is therefore organized under `public/` by owner, while phase Markdown links those components into a dependency sequence. Generated assets have distinct lifecycles:

- `data/` records immutable identities and sample lists;
- `artifacts/` stores reproducible heavyweight outputs;
- `cache/` stores deletable accelerators;
- `results/` stores run state and curated evidence.

## Main risks and controls

| Risk | Control required by the plan |
|---|---|
| Search space becomes unmanageable | Staged funnel; no full Cartesian expansion; synthesize only finalists |
| Quantizer quality hides datatype quality | Separate controlled Experiment A from optimized Experiment B |
| Backend or RTL semantics diverge | Shared manifests, exhaustive truth tables, bit-exact cross-backend tests |
| Multiplier-only comparison misleads | Include accumulation, scale/conversion, memory, control, and generic system cost |
| 1k sampling noise causes false pruning | Paired outcomes, bootstrap confidence intervals, uncertainty promotion |
| Approximate nonlinear behavior changes quality | Hardware-accurate LUT/decomposition semantics in exact inference |
| Vectorless or zero-delay power misleads | Vectorless only for broad pruning; real traces and delay-aware checks for finalists where supported |
| Nominal bit width overstates memory savings | Report ideal, packed-logical, and physical memory evidence separately |
| Exact inference is too slow | CUDA inner-kernel parallelism, CPU job parallelism, caching, staged datasets |
| MANT assumptions bias the public study | No MANT implementation or constants in this repository; export-only boundary |

## Scope outcome

Phases 0–7 and public release work are actionable here. Phases 8–10 remain in the roadmap documentation solely so the exported package has a clear downstream purpose. The removed MANT implementation tree must not be recreated in this repository.
