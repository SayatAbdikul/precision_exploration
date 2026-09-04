# Generic hardware methodology

## Scope and hierarchy

This repository evaluates publishable architecture-independent hardware only. The hierarchy is:

1. primitive multiplier/adder/decode/normalize;
2. complete MAC/FPU with accumulator, rounding, saturation, and format logic;
3. scaling, conversion, metadata, and memory;
4. generic normalized PE/tile/system with parameterized local memory, control, interconnect overhead, bandwidth, and area budgets.

There is no MANT tile or chip implementation here.

## PDK and synthesis policy

| PDK/library | Role in supplied plan |
|---|---|
| ICS55/ICsprout55 | Primary serious synthesis/P&R target |
| SKY130 | Secondary mature open-flow cross-check |
| GF180MCU or IHP SG13G2 | Optional alternatives only if later added |
| FreePDK45/Nangate45 | Academic/predictive comparison, not manufacturable evidence |
| ASAP7 | Optional scaling study, not fabrication evidence |

Use ICS55 RVT as baseline. Where available, repeat selected finalist runs with HVT/RVT/LVT. Do not imply signoff precision for unavailable models, SRAMs, extraction, or delay-aware features.

Sweep several progressively tighter clock constraints selected after a pilot. Report common iso-frequency points and each implementation's optimized Pareto points. Sweep approximately 0–3 internal pipeline stages where meaningful.

For iterative/serial units report latency, II, operations/s, energy/op, and area. A three-cycle fully pipelined `II=1` design is not equivalent to a three-cycle iterative `II=3` design.

## Physical depth by stage

| Candidate stage | Flow |
|---|---|
| Broad architecture search | Logic synthesis |
| Approximately 8–15 | Synthesis plus placement/preliminary physical estimate |
| Final 2–4 public physical candidates | Placement, CTS, routing, extraction, and detailed power where supported |

Use multiple placement seeds for finalists where practical and report median/representative results.

## Exact multiplier/adder search space

| Dimension | Options in plan |
|---|---|
| Basic multiplier | Array/Braun, Baugh-Wooley, shift-add |
| Recoding | None, Booth radix-2, modified Booth radix-4, radix-8, optional radix-16 |
| Reduction | Array, Wallace, Dadda, carry-save tree |
| Compressors | 3:2, 4:2, useful generalized counters |
| Final adder | Ripple, CLA, carry-skip, carry-select, conditional-sum |
| Prefix | Kogge-Stone, Brent-Kung, Sklansky, Han-Carlson, Ladner-Fischer |
| Serial/iterative | Bit-serial, digit-serial, shift-add, iterative Booth |
| Recursive | Karatsuba, Toom-Cook variants |
| Table based | Full product LUT/ROM, segmented LUT |
| Niche exact | Vedic/Urdhva-style and other distinct exact structures |

Represent each multiplier as recoding/partial-product generation + reduction + final adder + pipeline choice. Generate legal combinations rather than assuming large-width rankings apply at 2–8 bits.

Approximate/truncated/broken-array/Mitchell/approximate-compressor variants belong to a separate optional branch and do not enter the primary exact datatype comparison.

## Family-specific datapaths

### FP/minifloat multiply and add

Multiplier path: unpack → sign → exponent → significand multiply → normalize → round → pack. Search exponent adders, significand multiplier variants, staged/barrel/log shifts, leading detectors, RNE, required special-value support, and decode/multiply/normalize pipeline boundaries.

Adder search includes exponent compare, alignment, RCA/CLA/carry-select/prefix significand add, leading-zero detection/anticipation, single- versus dual-path, optional redundant representation, explicit rounding, and shallow pipelines.

### MAC/FMA/accumulator

- separate multiply plus add with product rounding;
- narrow product plus wider accumulator;
- fused full-product accumulation;
- carry-save/redundant accumulation;
- Kulisch/exact fixed accumulation followed by one rounding.

Kulisch/exact accumulation is a high-priority candidate for low-bit FP products.

### Posit

Ordinary decode/arithmetic/normalize/encode, wider-Posit accumulation, and Posit+quire dot product.

### MX/BFP

Shared-scale detection/storage/distribution, INT/minifloat element datapaths, scale-early versus scale-late, fixed/integer block partial sums, wide block accumulation, and final requantization.

### Logarithmic

Multiplication via log addition; compare full correction LUT, segmented LUT, piecewise linear, shift-add approximation only in the optional branch, and log-multiply plus linear accumulator.

### Codebook

Decode then conventional arithmetic, direct 2D product LUT, segmented/shared decode, and small LUT plus wider accumulator.

### Binary/ternary

XNOR-popcount/sign-select accumulation, ternary -x/0/+x with zero gating, and 2-bit multi-level shift/add or LUT.

## Scaling and conversion cost

Split support cost into metadata storage, scale generation, scale application, and scale movement.

- Static PTQ weight/activation scales are generated offline; hardware pays storage/read/application.
- Dynamic MX/BFP pays runtime magnitude/exponent detection, reduction, registers, and control.
- INT arbitrary requantization uses optimized multiplier/fixed-point/shift/round/clamp logic.
- INT power-of-two uses shift/round/saturate.
- FP power-of-two may use exponent adjustment; arbitrary scale may fold/reuse/combine with requantization.
- Posit general scale may require multiply; power-of-two may use regime/exponent manipulation.
- Log scale becomes log-domain addition.
- MX/BFP compares scale-early with amortized scale-late block partial sums.

For an 8-bit scale, block 16/32/64 metadata is 0.5/0.25/0.125 bits per value. Include scale traffic in memory and bandwidth results.

Report raw core versus all-in area/energy and stacked breakdowns for area, energy/op, critical-path contribution, and metadata bits/value.

## Memory methodology

### Level A — ideal

`C_ideal = N * b`; ideal savings versus 8 bits are 0%, 12.5%, 25%, 37.5%, 50%, 62.5%, and 75% for 8, 7, 6, 5, 4, 3, and 2 bits.

### Level B — packed logical

Model word/bank width, padding, alignment, ports, and metadata. Report nominal and effective bits/value. Example: four FP6 values in a 32-bit interface use 24/32 = 75% of the bits.

### Level C — physical

Preference order: characterized PDK macro/compiler, validated OpenRAM/technology-specific generated SRAM, then clearly labeled synthesized local memory or analytical fallback. CACTI/analytical estimates support exploration but cannot be the sole final silicon evidence.

Report capacity, area, read/write energy, latency, bandwidth, and effective bits/value. Compare same value count, physical area, bandwidth budget, and working-set/locality cases. Account separately for weights, activations, accumulators, and scale metadata.

```text
E_memory,inference = N_read * E_read + N_write * E_write
```

## Power methodology

Use identical vectorless assumptions only for broad pruning. Finalists use datatype-specific operand traces from exact inference, covering representative standard 3x3 Conv, pointwise Conv, depthwise Conv, and optional detector/EfficientNet layers.

- SAIF: primary average switching input.
- VCD: detailed waveform/debug/peak/glitch analysis.
- Delay-aware post-route activity: final 2–4 only when reliably supported.
- Count pipeline fill/drain, every iterative cycle, pipeline registers, and clock tree.
- Check trace convergence across approximately 32→64→128→256 images; freeze tolerance after pilot.

```text
energy/op = total_power / throughput
EDP       = energy/op * operation_latency
```

## Generic end-to-end model

For format/configuration F:

```text
A_tile(F) = A_compute + A_accumulator + A_scale/conversion
          + A_local-memory + A_control + A_local-interconnect

A_system = A_shared + N_F * A_tile(F)
N_F      = floor((A_budget - A_shared) / A_tile(F))
```

```text
throughput_peak      = N_F * useful_ops_per_cycle_F * frequency_F
throughput_effective = throughput_peak * utilization_F
```

Bandwidth bounds effective performance:

```text
T = min(T_compute, T_local_memory, T_interconnect, T_external_memory)
```

```text
E_inference = E_compute + E_local-memory + E_scaling/conversion
            + E_interconnect + E_off-chip + E_control + E_leakage

E_leakage = P_leakage * T_inference
```

The primary view is quality-constrained iso-area. Also report iso-tile-count, iso-power, iso-throughput, iso-energy, iso-memory-capacity, and iso-accuracy where useful. Example area budgets 1, 5, 10, and 25 mm² are exploratory, not fixed.

## Pareto analysis

Primary plots:

- quality versus energy/inference;
- quality versus images/s/mm²;
- energy/inference versus area;
- images/J versus quality;
- quality versus effective bits/value;
- memory-energy versus compute-energy breakdown;
- raw arithmetic versus all-in generic-system Pareto.

Use quality-budget frontiers such as:

```text
E*(delta) = min E(c) subject to quality_loss(c) <= delta
```

Evaluate several loss budgets (the plan suggests 0.25, 0.5, 1, 2, 3, and 5 percentage points), identify knee points without claiming uniqueness, build per-workload fronts before aggregate/worst-case views, and carry confidence intervals into dominance.

## Baselines and conformance

- Quality references: FP32; FP16 sanity; optional BF16.
- Standard low-bit anchors: INT8/6/4, standard FP8 E4M3/E5M2, standard MXFP8/6/4 and MXINT8 where relevant, ratified Posit configuration.
- Full-network sequence: reproduce FP32, conventional INT8 PTQ, and at least one standard FP8/MX point before trusting novel formats.
- Reproduce approximately 3–6 high-value external reference points spanning INT, FP8, MX, Posit, and log/non-uniform where feasible.
- Target primitive conformance is bit exact. Matching runtime quality should ideally be within approximately 0.1 percentage point; external-paper differences around 0.5 point may be acceptable only with documented preprocessing/calibration differences.
- RTL baselines: array multiplier + ripple adder, tool-inferred `*`, straightforward FP unpack/multiply/normalize/round, and best DSE-discovered architecture.

## Final public hardware package

For every final physical candidate report PDK/library/corner/VT, area/cells/registers, clock/timing, latency/II/throughput, power components, energy/op, memory/scaling support cost, exact quality/mAP, all config/RTL/tool/trace hashes, and limitations. Export candidate packages only after this evidence is attached.
