# Phase 7 — Build the public generic end-to-end Pareto frontier

Scope: **in repository; final public Stage A conclusion and external handoff**  
Starts after: **Phase 6 RTL/PPA plus accepted memory/power methodology**  
Status: **planned**

## Generic PE/tile/system tasks

- **H7.1** — Combine MAC/FMA, accumulator, scale/conversion, parameterized local memory, and generic control/interconnect overhead.
- **H7.2** — Sweep generic memory size, bandwidth, and area budgets.
- **H7.3** — Compare iso-frequency and architecture-optimized points.
- **H7.4** — Compare iso-area allocations: more smaller PEs versus fewer larger PEs.

Also report iso-tile-count, iso-power, iso-throughput, iso-energy, iso-memory-capacity, and iso-accuracy views where relevant. Do not insert MANT constants.

## Quality attachment

- **N7.1** — Retain the exact tuple: W format, A format, accumulator, scaling, MAC model, workload, and generic implementation.
- **N7.2** — Never report TOPS/W or another hardware metric without measured quality for the identical configuration.

## Pareto-analysis tasks

- **I7.1** — Quality versus energy/inference.
- **I7.2** — Quality versus images/s/mm².
- **I7.3** — Energy/inference versus area.
- **I7.4** — Raw arithmetic Pareto versus all-in generic-system Pareto.
- **I7.5** — Memory-versus-compute energy breakdown.

Also build quality-budget frontiers, knee-point analysis, per-workload and mean/worst-case fronts, effective-bits/value views, and confidence-aware dominance categories.

## Candidate-export preparation

The source roadmap calls these MANT-selection-preparation tasks; only their public/export side belongs here:

- **M7.1** — Define guard-set features such as family diversity, favorable/simple scaling, conversion, width, and arithmetic structure.
- **M7.2** — Prepare public package export for selected points.

No private importer or MANT compatibility measurement exists in this repository.

## Decision D10 — Final public physical power method

For the final 2–4 generic public physical candidates, freeze trace length, convergence tolerance, SAIF/VCD roles, and whether reliable delay-aware/glitch analysis is supported. If it is not reliable in ICS55, report activity-based limitations and optionally cross-check methodology in the mature secondary flow.

## Decision D11 — Pareto plus guard export set

Select approximately 5–10 candidates after the complete public end-to-end study. Include:

- highest quality;
- lowest energy;
- lowest area;
- highest images/J;
- highest images/s/mm²;
- two or three knee points;
- family alternatives;
- near-Pareto guard points with structurally interesting width/scaling/conversion/arithmetic behavior.

Export selection does not change or prune the reported public frontier.

## Outputs

- Confidence-aware public Pareto fronts at increasing hardware realism.
- Generic area and energy breakdowns.
- Complete PDK/RTL/power/memory/config provenance for final points.
- D10 physical-power policy.
- D11 versioned Pareto-plus-guard candidate package set.

## Public-study exit

Every exported point has an evidence chain from checkpoint → deployment graph/PTQ → exact inference → statistical quality → RTL semantics → generic physical implementation → quality-attached energy/area/throughput.
