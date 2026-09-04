# Hardware-metrics contract

Status in supplied roadmap: **required by H0.1–H0.3; exact pilot-dependent values remain open**

## Public comparison hierarchy

This repository evaluates only architecture-independent generic hardware:

1. arithmetic primitive: multiplier, adder, decode/normalize;
2. complete MAC/FPU: decode, multiply, accumulation, normalize, round, scale/saturation;
3. support path: scaling, conversion, metadata, and memory;
4. generic PE/tile/system: parameterized compute, accumulator, local memory, generic control/interconnect and bandwidth assumptions.

MANT-specific tile and whole-chip metrics are outside this repository.

## Primary flow

- Primary process: ICS55/ICsprout55 at 55 nm.
- Baseline VT: RVT.
- Finalist sensitivity: HVT/RVT/LVT where libraries exist.
- Optional methodology cross-check: SKY130; other processes listed in the plan are not scaffolded unless a later decision adds them.
- If ICS55 lacks a trustworthy SRAM, extraction, or delay-aware capability, disclose the limitation and keep analytical/logical results separate from physical evidence.

## Standard timing harness

```text
launch DFF -> DUT -> capture DFF
```

Wrapper registers define launch/load conditions and are excluded from core area for a combinational DUT. Architecture-owned internal pipeline registers are included.

## Frequency and pipeline search

- Sweep progressively tighter timing constraints selected after an ICS55 pilot.
- Report iso-frequency points at common achievable clocks.
- Also report each architecture's own Pareto-optimal operating points.
- Sweep approximately 0–3 internal pipeline stages where meaningful.
- For serial/iterative designs distinguish latency from initiation interval.

```text
throughput = frequency / II
```

Do not compare architectures on frequency alone.

## Required metrics per implementation

- logic area and, separately, pipeline-register area;
- target clock, slack, achieved fmax;
- latency in cycles and time;
- initiation interval;
- operations/s or useful throughput;
- dynamic, leakage, clock, and total power where available;
- energy/op;
- EDP as an auxiliary measure;
- PDK/library/corner/VT, tool versions, RTL/config hashes.

Final physical candidates add routed area, cell/register counts, post-route timing, clock-tree contribution, placement seed, extraction/delay mode, trace identity, and switching-based power.

## Power rules

| Stage | Method | Allowed use |
|---|---|---|
| Broad sweep | Identical vectorless/default activity assumptions | Rough pruning only |
| Finalists | Datatype-specific representative CNN traces | Headline energy comparison |

SAIF is the primary average-activity input. VCD supports detailed/debug/peak/glitch analysis. Use delay-aware gate-level/post-route activity for the final 2–4 public physical candidates only where the flow is reliable; otherwise disclose non-glitch limitations.

Trace windows represent identical logical operations. Include fill/drain for pipelines and every cycle for iterative units. Test convergence across increasing samples such as 32, 64, 128, and 256 images; the final tolerance is set after the pilot, with approximately 1% suggested by the plan.

```text
energy/op = total power / throughput
leakage energy = leakage power * execution time
```

## Memory evidence levels

| Level | Evidence |
|---|---|
| A | Ideal `N * bit_width` information density |
| B | Packed logical words/banks including padding, alignment, ports, and metadata |
| C | Best verified physical option: PDK macro/compiler, validated generated SRAM, or clearly labeled synthesized/analytical fallback |

Report usable capacity, area, read/write energy, latency, delivered bandwidth, and effective bits/value. A datatype receives no physical SRAM saving that the selected memory organization cannot realize.

## Scaling/support accounting

Report raw and all-in datapath costs:

```text
A_all_in = A_raw + A_scale_generation + A_scale_application
         + A_conversion + A_metadata/control

E_all_in = E_MAC + E_scale_generation + E_scale_application
         + E_scale_read + E_conversion
```

For block scaling include metadata bits/value, scale generation, application, storage, and movement. For an 8-bit scale and block size B, metadata overhead is `8/B` bits/value.

## Generic end-to-end metrics

- images/s;
- images/s/mm²;
- images/J;
- energy/inference;
- TOPS/mm² and TOPS/W as supporting metrics;
- local-memory capacity and bandwidth;
- measured Top-1/Top-5 or mAP for the exact same configuration.

Each hardware point retains `(W format, A format, accumulator, scale policy, MAC model, workload)`.

## Pareto rule

Candidate B dominates A only when it is no worse in every included dimension and strictly better in at least one:

```text
quality_B >= quality_A
energy_B  <= energy_A
area_B    <= area_A
throughput_B >= throughput_A
```

Report confidence-aware fronts at primitive, MAC/accumulator, all-in datapath, and generic system levels. Do not use one arbitrary composite score as the primary research result.
