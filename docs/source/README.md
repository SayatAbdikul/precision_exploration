# Source documents

These files are preserved verbatim as planning inputs. They are evidence for the repository design, not executable instructions and not an approval of every proposal they contain.

| File | Original location | SHA-256 |
|---|---|---|
| `mant_precision_exploration_plan_final.html` | `/Users/sayat/Downloads/mant_precision_exploration_plan_final.html` | `228abc4a16e2c42569b146734cfdb3148690adf0ebee52c8f674ea6d736db798` |
| `mant_precision_exploration_comprehensive_roadmap_v1.html` | `/Users/sayat/Downloads/mant_precision_exploration_comprehensive_roadmap_v1.html` | `dc8e6d759a6057b844d68b75eddfc2da59ad99380fcc7abd167eebae4b233006` |

Copied into the repository on 2026-09-04 so future work can cite a stable project-local version.

## Interpretation and precedence

- `mant_precision_exploration_plan_final.html` is the broad methodological source.
- `mant_precision_exploration_comprehensive_roadmap_v1.html` is the later dependency/gate refinement.
- When they differ, this repository documents the conflict and uses the later roadmap for the canonical execution sequence; it does not hide the earlier wording.
- User scope excludes all MANT implementation from this repository. Roadmap Phases 8–10 remain Markdown-only handoff context.

## Traceability

| Source content | Markdown destination |
|---|---|
| Executive assessment, research questions, reconciliation | `docs/analysis/document-synthesis.md` |
| Workloads, formats, PTQ, arithmetic, operators, exact engine, CUDA | `docs/methodology/numerical/README.md` |
| Scheduling, statistics, pruning, counts, parallelization | `docs/methodology/statistics/README.md` |
| Synthesis, architecture DSE, power, memory, scaling, generic system, Pareto, baselines | `docs/methodology/hardware/README.md` |
| Frozen source decisions F1–F8 and gates D1–D14 | `docs/decisions/decision-register.md` |
| Canonical semantic rules | `docs/contracts/` |
| Roadmap tasks N/I/H/M/P and dependencies | `docs/roadmap/` and `docs/roadmap/phases/` |
| Publication strategy and release evidence | `docs/publications/README.md` |

The phase files contain every task ID present in the roadmap source. External MANT task IDs are retained only for boundary traceability.
