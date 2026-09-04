# Phase 6 — Generic hardware design-space exploration

Scope: **in repository; public Stage A**  
Starts after: **D7 generic RTL shortlist**  
Parallelism: **by family and block using one metric/config schema**  
Status: **planned**

## Arithmetic RTL DSE

- **H6.1** — Integer/significand multipliers: array/Braun, Baugh-Wooley, shift-add, Booth, Wallace, Dadda, selected serial/LUT variants.
- **H6.2** — Final adders: ripple, CLA, carry-select, and selected prefix families.
- **H6.3** — Minifloat unpack, exponent/significand, normalize, round, and pack DSE.
- **H6.4** — FP adders and FMA/separate-MAC/narrow-product-wide-accumulator architectures.
- **H6.5** — Shortlisted Posit+quire, MX/BFP, and logarithmic paths.
- **H6.6** — Sweep approximately 0–3 internal pipeline stages and multiple frequency constraints.

The broader plan additionally enumerates recoding, compressor, prefix, serial/iterative, recursive, LUT, codebook, binary, ternary, and exact Kulisch options. Generate legal compositions and keep each datatype's area–delay–energy–throughput Pareto set.

Approximate arithmetic remains a separate optional study and cannot contaminate the primary exact comparison.

## Scaling and conversion RTL

- **H6.7** — INT requantization for arbitrary and power-of-two scales.
- **H6.8** — FP exponent-adjustment scale paths where applicable.
- **H6.9** — Dynamic MX/BFP scale generation and metadata handling.
- **H6.10** — Conversion/requantization between accumulator and output domains.

Report raw arithmetic and all-in support costs separately. Include metadata, generation, application, storage/read, conversion, and movement.

## Memory methodology

- **H6.11** — Level A ideal bits/value.
- **H6.12** — Level B packed logical memory with alignment/padding/ports/metadata.
- **H6.13** — Level C physical implementation using the best verified ICS55-supported option.
- **H6.14** — Measure/estimate area, read/write energy, bandwidth, latency, and effective bits/value.

Compare same stored-value count, same memory area, same bandwidth budget, and same working-set/locality conditions.

## Power and trace infrastructure

- **H6.15** — Use vectorless/default switching only for broad pruning.
- **I6.1** — Export representative quantized operand traces from exact inference.
- **I6.2** — Prepare SAIF as the primary average-activity flow and VCD for detailed analysis.
- **I6.3** — Include pipeline fill/drain and every iterative cycle.

Trace candidates include ResNet 3×3 Conv, MobileNet pointwise and depthwise Conv, plus optional detector/EfficientNet layers. Each datatype uses traces from its own quantized network.

## Physical-flow rules

- Primary: ICS55, RVT baseline.
- Optional: SKY130 cross-check.
- Standard launch-DFF → DUT → capture-DFF harness.
- Common frequency points plus architecture-optimized points.
- Broad synthesis, then placement/preliminary physical estimation for approximately 8–15, then detailed public P&R for 2–4 candidates where supported.
- Multiple placement seeds where practical.

## Decision D8 — RTL representatives

Each numerical configuration is represented by a small Pareto set, not one arbitrary implementation. Preserve meaningful combinational/simple, pipelined, and iterative/serial points across area, fmax, II, latency, energy, and throughput.

## Decision D9 — Physical memory evidence

After verifying ICS55 capability, choose and label one of:

1. characterized PDK macro/compiler;
2. validated generated SRAM;
3. synthesized local memory;
4. clearly separated analytical/packed model.

This decision can reorder the hardware Pareto set and must not claim unsupported signoff precision.

## Outputs

- RTL/config generators and conformance results.
- Synthesis/physical database across candidate architecture sets.
- Raw versus all-in scaling/conversion cost.
- Ideal, packed, and physical memory evidence.
- Datatype-specific trace assets and power-flow pilot data.
- D8 representative architecture sets and D9 memory policy.

## Blocks

The public generic end-to-end Pareto model in Phase 7.
