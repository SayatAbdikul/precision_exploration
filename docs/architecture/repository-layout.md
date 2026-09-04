# Repository layout and ownership

## Scope boundary

This repository ends at the architecture-independent Pareto frontier and the exported Pareto-plus-guard candidate package. There is deliberately no local MANT implementation directory. ISA, mapping, compiler, golden-model, cycle-simulator, tile, interconnect, RTL, and chip work belongs in a separate private repository.

## Directory tree

```text
precision_exploration/
├── docs/
│   ├── source/                 verbatim HTML planning inputs and hashes
│   ├── analysis/               reconciled interpretation and risks
│   ├── architecture/           repository and dependency rules
│   ├── contracts/              Phase 0 normative interfaces
│   ├── decisions/              F1–F8 assumptions and D1–D14 gates
│   ├── methodology/
│   │   ├── numerical/          formats, PTQ, arithmetic, operators, inference
│   │   ├── statistics/         subsets, uncertainty, promotion, scheduling
│   │   └── hardware/           RTL, PPA, power, memory, generic Pareto model
│   ├── roadmap/phases/         detailed phase records and scope labels
│   └── publications/           public paper/release plan
├── public/
│   ├── formats/
│   │   ├── manifests/          canonical machine-readable datatype definitions
│   │   ├── oracle/             high-precision encode/decode/arithmetic source
│   │   └── conformance/        generated/reference primitive vectors
│   ├── quantization/
│   │   ├── graph/              eval/folding/deployment-graph preparation
│   │   ├── calibration/        fixed-subset calibration and scale search
│   │   └── ptq/                controlled A and optimized B implementations
│   ├── inference/
│   │   ├── reference/          correctness-first exact evaluator
│   │   ├── cpp/                portable multithreaded exact backend
│   │   └── operators/          Conv, DWConv, Linear, residual, pooling, LUTs
│   ├── cuda/
│   │   ├── kernels/            exact specialized CUDA kernels
│   │   └── benchmarks/         algorithmic/LUT/predecoded comparisons
│   ├── workloads/
│   │   ├── models/             fixed checkpoint adapters and graph conversion
│   │   ├── datasets/           ImageNet/COCO loading and preprocessing
│   │   └── subsets/            validation of fixed calibration/evaluation lists
│   ├── experiments/
│   │   ├── configs/experiment_a/
│   │   ├── configs/experiment_b/
│   │   ├── scheduler/          lifecycle, resume, deduplication, distribution
│   │   └── registry/           SQLite schema and artifact references
│   ├── generic_rtl/
│   │   ├── arithmetic/         adders, multipliers, format datapaths
│   │   ├── accumulators/       native, widened, FMA, exact/Kulisch/quire paths
│   │   ├── scaling_conversion/ requantization, scale, conversion, metadata logic
│   │   ├── memory/             ideal, packed, and physical memory models
│   │   ├── pe_tile/            publishable normalized PE/tile composition
│   │   └── harness/            standard timing and trace-replay wrappers
│   ├── pdk_flow/
│   │   ├── common/             shared constraints, parsers, metric schema
│   │   ├── ics55/              primary synthesis/P&R flow
│   │   └── sky130/             optional mature-flow cross-check
│   ├── analysis/
│   │   ├── quality/            Top-1/Top-5/mAP and layer statistics
│   │   ├── statistics/         paired bootstrap and robustness
│   │   ├── hardware/           synthesis, power, memory, and trace analysis
│   │   └── pareto/             confidence-aware multi-level frontiers
│   └── package/
│       ├── schema/             versioned external handoff contract
│       └── export/             Pareto-plus-guard package builder
├── data/
│   ├── manifests/calibration/  fixed, hashed calibration lists
│   ├── manifests/evaluation/   fixed, hashed screening/full lists
│   └── raw/                    ignored local dataset mount
├── artifacts/                  ignored reproducible heavyweight outputs
├── cache/                      ignored and fully disposable acceleration data
├── results/
│   ├── raw/                    ignored run-level output
│   ├── databases/              ignored SQLite registries
│   ├── summaries/              tracked compact evidence
│   ├── tables/                 tracked review/publication tables
│   ├── figures/                tracked review/publication plots
│   └── candidate_packages/     versioned public handoff manifests/packages
├── tests/                      unit, conformance, integration, regression
└── tools/                      setup, run, and analysis entry points
```

## Single-source-of-truth rules

1. Datatype behavior is defined by `public/formats/manifests/` plus the arithmetic contract. No backend or RTL block may carry a private copy of those semantics.
2. A canonical experiment configuration completely determines its numerical result. Configuration identity is SHA-256 over deterministic serialization.
3. Checkpoint, folded graph, dataset subset, datatype manifest, kernel, RTL, tool, PDK, and flow identities are hashes or immutable version identifiers.
4. The run database stores metadata, status, metrics, error logs, and artifact hashes/paths; large payloads live in `artifacts/`.
5. Publication tables and figures are generated from tracked code and queryable results, never edited as the sole result source.

## Dependency direction

```text
docs/contracts + public/formats + data/manifests
                 |
                 +--> workloads --> quantization --> inference/cuda
                 |                                  |
                 |                                  v
                 |                         experiments/registry
                 |                                  |
                 +--> generic_rtl --> pdk_flow -----+
                                                    |
                                                    v
                                  quality/statistics/hardware/pareto
                                                    |
                                                    v
                                      public/package/export
                                                    |
                                                    v
                                external private MANT repository
```

No in-repository component may depend on unpublished MANT details or constants.

## Generated-data placement

| Material | Location |
|---|---|
| Dataset sample identities and hashes | `data/manifests/` |
| Raw ImageNet/COCO payloads | external mount surfaced at `data/raw/` |
| Checkpoints and folded graphs | `artifacts/checkpoints/`, `artifacts/folded_graphs/` |
| Calibration stats and quantized/encoded tensors | `artifacts/calibration_stats/`, `quantized_weights/`, `encoded_tensors/` |
| Truth tables and per-image predictions | `artifacts/datatype_truth_tables/`, `per_image_predictions/` |
| SAIF/VCD/operand traces | `artifacts/traces/` |
| Generated RTL/PPA products | `artifacts/rtl/`, `artifacts/ppa/` |
| Run lifecycle and metrics DB | `results/databases/` |
| Compact reports and publishable outputs | `results/summaries/`, `tables/`, `figures/` |
| Exported candidate package | `results/candidate_packages/` |
