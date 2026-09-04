# Methodology

This directory converts the complete supplied precision-exploration plan into three coordinated methods. Contractual rules are summarized here and defined normatively in `docs/contracts/`.

| Document | Coverage |
|---|---|
| `numerical/README.md` | Research questions, workload suite, datatype universe, manifests, Experiment A/B PTQ, arithmetic/accumulation, operators, exact inference, CUDA strategy, pilot experiment |
| `statistics/README.md` | Fixed subsets, paired metrics, bootstrap uncertainty, conservative promotion, configuration/format/family pruning, scheduler, database, caching, compute strategy, experiment-count control |
| `hardware/README.md` | Generic hierarchy, ICS55 methodology, arithmetic architecture DSE, power, memory, scaling/conversion, generic end-to-end model, Pareto fronts, baselines and conformance |

## Cross-method invariants

- All candidates use the same frozen checkpoint, deployment graph, preprocessing, and fixed comparison subsets for a workload.
- Datatype manifests are the shared semantic source for oracle, PTQ, exact inference, CUDA, and RTL.
- Weight, activation, accumulator, and output formats are independently configurable.
- Controlled Experiment A, optimized Experiment B, MAC/accumulator sweeps, and approximate arithmetic are reported separately.
- Quality remains attached to every hardware point.
- Screening uncertainty causes promotion, not elimination.
- Public analysis remains independent of all MANT-specific architecture details.
- Large outputs are content-addressed; canonical configs and result provenance are queryable.

## Method acceptance sequence

1. Accept Phase 0 contracts.
2. Reproduce FP32 and conventional INT8 baselines.
3. Prove primitive and backend conformance.
4. Run controlled screening and uncertainty analysis.
5. Add optimized numerical variants only for promoted candidates.
6. Freeze full-validation candidates before expensive hardware DSE.
7. Build generic Pareto fronts at increasing hardware realism.
8. Export the public Pareto plus guard set.
