# Execution roadmap

The supplied roadmap is phase-ordered rather than calendar-based. Code remains in stable component directories; the files under `phases/` preserve tasks, gates, exit criteria, parallelism, and scope.

## Scope interpretation

- **Implemented here:** Phases 0–7 and the public research/release work in Phase 11.
- **Documented for handoff only:** Phases 8–10 and MANT engineering work in Phase 11.
- **Not present here:** any MANT implementation directory.

The public study includes detailed P&R of 2–4 **generic public finalists** within Phases 6–7/D10. Roadmap Phase 10 is downstream MANT physical validation and remains external.

## Master critical path

```text
0 contracts
  -> 1 workloads/oracle/infrastructure
  -> D1 initial formats
  -> 2 exact engine and implementation pilots
  -> 3 controlled Experiment A
  -> D4 promotion
  -> 4 numerical depth and Experiment B
  -> 5 full validation / D7 RTL set
  -> 6 generic RTL, memory, scaling, power
  -> 7 public end-to-end Pareto / D11 export
  -> external phases 8–10
  -> 11 public research package + external MANT freeze
```

## Phase index

| Phase | Outcome | Scope | Key gates |
|---:|---|---|---|
| 0 | Freeze experiment contracts | In repository | Phase 0 exit |
| 1 | Build workloads, oracle/infra, and generic flow foundations | In repository | D1 |
| 2 | Bit-exact inference engine and generic implementation pilots | In repository | D2, D3 |
| 3 | Broad controlled Experiment A | In repository | D4 |
| 4 | W/A, accumulator/MAC, scaling/PTQ depth | In repository | D5, D6 |
| 5 | Full numerical validation and RTL shortlist | In repository | D7 |
| 6 | Generic RTL/PPA, scaling, memory, power DSE | In repository | D8, D9 |
| 7 | Generic end-to-end Pareto and candidate export | In repository | D10, D11 |
| 8 | Import/evaluate candidates on MANT | External context only | D12 |
| 9 | MANT tile/chip co-design | External context only | D13, D14 |
| 10 | MANT detailed physical validation | External context only | Evidence-chain exit |
| 11 | Research release and MANT architecture freeze | Public tasks here; MANT tasks external | Final outputs |

## Work-package registry

| WP | Main tasks | Starts after | Parallel with | Blocks | Scope |
|---|---|---|---|---|---|
| WP0 | Arithmetic, Experiment A, manifests/config, metrics, external boundary | Now | None | Everything | In repo |
| WP1A | Checkpoints, datasets, baselines, fixed subsets | WP0 | WP1B, WP1C | Experiment A | In repo |
| WP1B | Oracle, truth tables, scheduler, DB, artifacts, statistics | WP0 | WP1A, WP1C | Exact engine | In repo |
| WP1C | ICS55 bring-up, memory-support investigation, timing harness | WP0 | WP1A, WP1B, engine | Later generic hardware DSE | In repo |
| WP2 | C++/CUDA exact operators and bit-exact validation | WP1B + operator contract | Generic-flow pilots | Quality sweeps | In repo |
| WP3 | Uniform 1k→10k controlled Experiment A | WP1A + WP2 | Rough hardware estimates | Deep sweeps | In repo |
| WP4 | W/A, accumulators, MAC A/B/C, Experiment B | D4 | Branches A/B/C/D | Full validation | In repo |
| WP5 | 20–40 full numerical configs and seed robustness | WP4 | RTL requirements | D7 set | In repo |
| WP6 | Arithmetic/support/memory RTL, pipeline/frequency, PPA/power | D7 | By family/block | Public hardware Pareto | In repo |
| WP7 | Generic PE/system, iso-area, quality-attached Pareto, export | WP6 + memory/power method | Package preparation | D11 | In repo |
| WP8 | MANT importer/golden model/mapping/cycle simulation | D11 | Per candidate | D12 | External |
| WP9 | MANT tile variants, PPA, whole-chip DSE | D12 | Per candidate | Finalists | External |
| WP10 | MANT final P&R and switching power | D13/shortlist | Per candidate | MANT freeze | External |
| WP11 | Public reproducibility/paper; external MANT manual/spec | Stabilized results/WP10 | Writing can begin earlier | Final deliverables | Split |

## Parallel clusters

### Cluster A — after contracts

- checkpoint/workload preparation;
- datatype oracle and manifests;
- scheduler/database/artifact infrastructure;
- ICS55 flow bring-up.

### Cluster B — after D4

- W/A asymmetry and signedness;
- accumulator and MAC-model sweeps;
- Experiment B scale/PTQ/block-size sweeps;
- rough support/memory/hardware-cost modeling.

### Cluster C — after D7

- integer/significand multiplier architectures;
- FP add/mul/FMA and accumulator architectures;
- Posit/MX/log/codebook-specific blocks;
- scale/requantization/conversion RTL;
- memory organizations;
- power-trace and parsing infrastructure.

All parallel work consumes common manifests/contracts. Parallelism must not create divergent numerical semantics.

## Milestone outputs

```text
contracts/schemas
 -> reproducible FP32 baselines
 -> bit-exact format library
 -> exact C++/CUDA inference
 -> Experiment A dataset
 -> Experiment B + accumulation dataset
 -> full numerical finalists
 -> generic RTL/PPA database
 -> public generic Pareto frontier
 -> Pareto + guard candidate export
 -> external MANT evaluation
 -> public research package
```

## Final public package

- datatype manifests and conformance tests;
- controlled Experiment A results under minimal/canonical family semantics;
- separately labeled optimized Experiment B results;
- accumulator and MAC-model sensitivity;
- ImageNet plus detector quality with uncertainty;
- generic arithmetic/scaling/memory/PE RTL/PPA and Pareto fronts;
- explicit ideal-versus-realizable memory analysis;
- versioned candidate-package export;
- reproducibility instructions, configurations, and result tables.
