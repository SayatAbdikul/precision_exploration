# Phase 8 — Private MANT evaluation

Scope: **external handoff context only; no implementation in this repository**  
Starts after: **D11 candidate export**  
Status: **out of repository scope**

This phase is retained to document why the public package exists. It must be executed in a separate private MANT repository and must not retroactively alter public results.

## Source-roadmap import/golden-model tasks

- **M8.1** — Import public quantized model packages.
- **M8.2** — Extend/parameterize the private numerical golden model only as needed for selected candidates.
- **M8.3** — Measure `delta_MANT(F) = quality_MANT(F) - quality_generic(F)`.
- **M8.4** — Determine whether current MANT arithmetic changes the public ranking.

## Source-roadmap mapping/compiler tasks

- **M8.5** — Map representative CNN operators and full graphs.
- **M8.6** — Track tile memory, instruction memory, traffic, and synchronization.
- **M8.7** — Identify unsupported operations and conversion/requantization pressure.

## Source-roadmap cycle-simulation tasks

- **M8.8** — Measure cycles, stalls, bus traffic, tile utilization, and full-network latency.
- **M8.9** — Compare current one-format/current-word assumptions with candidate needs.

## Interpretation tasks

- **N8.1** — Separate generic numerical advantage from MANT-specific implementation advantage.
- **N8.2** — Attribute ranking changes to accumulator, scale path, memory width, conversion, communication, or operator support.

## Decision D12 — Compatibility strategy

External routes per candidate:

- A: compatible with current MANT; minimal changes;
- B: small extension such as independent accumulator width, conversion/requantization, parameters, or operator support;
- C: major redesign; continue only when benefit justifies architecture cost.

This is the earliest roadmap point for deciding whether MANT assumptions should change. No D12 work belongs in this repository.
