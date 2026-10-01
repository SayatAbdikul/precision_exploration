# Packed memory and bandwidth evidence, version 1.0.0

## Reproduction and inputs

```bash
/home/maveric/precision_exploration/.venv/bin/python -m pytest -q tests/unit/test_memory_model.py
/home/maveric/precision_exploration/.venv/bin/python -m tools.analysis.memory_model --observation-root /home/maveric/precision_exploration
```

The generator reads accepted manifests in this checkout and the retained, read-only `results/summaries/phase3-hardware-priors.json` under the supplied observation root. That Phase 3 summary was produced from prepared graphs and observed shapes. Its SHA-256 is `f16a0878d523db7f78df5aa8c25b452e2ba8bc22de4cc19ca9189c18afc39f97`. All 100 prepared graph files and four observed-shape files matched their recorded SHA-256 hashes in this run. The generator does not rerun inference, recalibration, graph preparation, or quality analysis. Without the observation summary, all-format tables still generate and graph coverage is explicitly reported missing. The `.venv` executable above is the verified project environment on this host; another environment needs the project's declared dependencies.

The reusable API is `public.analysis.memory_model.tensor_layout` and `throughput_roof`. Its manifest parser takes widths and intrinsic block semantics from the 25 accepted datatype manifests. Required-mapping scale counts are configuration inputs; their assumed 64-bit storage width is an analytical placeholder, not an implementation of arbitrary rational scales.

## Levels and assumptions

| Level | Calculation | Evidence boundary |
|---|---|---|
| A | Useful values × manifest code width | Ideal payload density only |
| B | Independent word-packed payload/metadata arrays, line alignment, bank rounding, port-limited word accesses | Analytical logical implementation; packing/unpacking, cross-word values, scale addressing and axis reorder are flagged but their area/energy is unmeasured |
| C | A technology-specific memory instance with area, latency and read/write energy | Unavailable here for ICS55 SRAM; physical fields remain null |

For each tensor, the model packs `ceil(line_values × code_bits / word_bits)` words per contiguous line, stripes words over the declared banks, and rounds each array to equal bank depth. Metadata is a separate packed array and restarts its stripe at bank zero. It also accepts bias words, accumulator words and a double-buffer multiplier. A non-last packing axis entails a gather or reorder; its cost is not awarded as free. Read/write cycles use the **maximum per-bank** payload-plus-metadata word load divided by that bank's port count, rounded up. Thus two one-word arrays both landing in bank zero take two cycles with one read port even if another bank is idle. This assumes available multiport structures; it does not characterize their physical cost. `whole_tensor_capacity_count` is an exact count of repeated, identically laid-out tensors under the chosen bank capacity, with no allocator fragmentation.

The common table uses a **32×33 weight matrix**, rows packed separately, two banks, one read and write port per bank, a 500 MHz assumed memory clock, and 16,384 bits of common capacity. It compares 32- and 64-bit words at the same stored-value count and capacity. Required-mapping formats use the manifest's per-tensor default of **one** scale, with an assumed 64-bit stored representation. A separate `memory-mapped-per-row.csv` shows 32 per-row scales as an illustrative policy present in retained prepared graphs. Intrinsic MX/BFP scale widths (8 bits), block sizes (32), reduction-axis grouping, and incomplete block counts come from the manifests. Alternate 16/64-element blocks are labeled hypothetical and do not inherit acceptance or quality.

The bandwidth roof uses the 64-bit-word layout, the same 8 GB/s combined W+A memory budget for every format, 64 MAC/cycle at 500 MHz, and four uses per fetched W and A value. Thus transfer bits per MAC are `W_read_traffic_bits_per_value/4 + A_read_traffic_bits_per_value/4`, and the rate is the minimum of the 32 GMAC/s compute roof and memory roof. Bank-capacity padding that is never read is excluded from traffic. This is an explicit reuse scenario, not observed accelerator traffic. Sequential one-pass read/write word counts are separately recorded in the format table. The model does not infer local tiling, broadcast network cost, cache hit rate, or energy from graph MAC counts.

## Selected findings

| Format | Nominal bits/value | Effective 64-bit logical bits/value | Metadata fraction | Padding fraction | 8 GB/s memory roof, GMAC/s |
|---|---:|---:|---:|---:|---:|
| INT8 | 8 | 9.82 | 0.6% | 17.9% | 13.12 |
| INT6 | 6 | 7.88 | 0.8% | 23.1% | 16.37 |
| INT5 | 5 | 5.94 | 1.0% | 14.8% | 21.77 |
| INT4 | 4 | 5.94 | 1.0% | 31.6% | 21.77 |
| FP7 E3M3 | 7 | 7.76 | 0% | 9.8% | 16.50 |
| FP6 E3M2 | 6 | 7.76 | 0% | 22.7% | 16.50 |
| FP4 E2M1 | 4 | 5.82 | 0% | 31.2% | 22.00 |
| BFP6 | 6 | 8.24 | 5.9% | 21.3% | 15.53 |
| MXFP4 E2M1 | 4 | 6.30 | 7.7% | 28.8% | 20.31 |

The 33-value row boundary makes INT4 and INT5 occupy the same 64-bit-word capacity in this setup; FP6 and FP7 likewise tie. This result is layout-specific, and the 32-bit-word table shows a different packing regime. INT4 with 32 mapped row scales rises to 7.76 effective bits/value in the separately labeled per-row scenario. Binary and ternary remain below wider formats in this scenario, but padding and mapped scale metadata take substantial fractions of their allocation. No physical area benefit follows from these logical bit counts.

## Retained graph examples

The graph table contains 25 format configurations for each of ResNet18, MobileNetV2, MobileNetV3-Large and YOLOv8n. It reads each prepared graph's actual constant shapes, scale count and scale axis. It reports **two** logical 64-bit weight layouts: flat packing per tensor and conservative last-dimension line alignment, each with separate scale words. Bias bits, the existing Phase 3 **byte-aligned** activation-liveness peak, a maximum one-accumulator-per-output bank, and Model C product counts are separate columns. The activation peak is not recomputed with this model's 64-bit bank organization. The layer table selects the largest-MAC layer per configuration and records observed output shape, reduction length, activation metadata and accumulator-bank alternatives. These are graph and schedule estimates. Views are materialized as copies under the existing Phase 3 liveness policy; internal nonlinear buffers and optional patches are not included in the peak.

| Model | INT4 flat weights, MiB | INT4 row-aligned weights, MiB | MXFP4 flat weights, MiB | INT4 peak live activations, MiB | Model C products, billions |
|---|---:|---:|---:|---:|
| ResNet18 | 5.61 | 29.55 | 5.92 | 0.77 | 1.814 |
| MobileNetV2 | 1.79 | 17.12 | 1.76 | 1.15 | 0.301 |
| MobileNetV3-Large | 2.75 | 23.25 | 2.77 | 0.77 | 0.217 |
| YOLOv8n | 1.54 | 11.92 | 1.59 | 1.56 | 4.372 |

Weight figures are per-tensor logical word allocations converted to MiB, not measured host memory. The large INT4 row-aligned values arise from imposing a new 64-bit boundary on every short innermost filter row. A real tiled/transposed weight layout could approach the flat case, but its gather and unpack costs need an implementation. MXFP prepared graphs can reshape block-convolution weights to `[output, reduction_k]`; its row layout therefore cannot be compared as if it shared an identical physical placement with the integer graph. Activation figures are the retained byte-packed graph schedule and are not combined with 64-bit bank allocations. Graph estimates do not establish actual accelerator traffic or an energy/inference value. All four graph/shape sets were available in this run; no model coverage was missing.

## ICS55 physical memory status

The repository's ICS55 availability inventory records RVT standard-cell Liberty/LEF collateral but no characterized SRAM/compiler and no validated technology-specific generated macro. No local SRAM Liberty macro appeared in the inspected project cache. A future synthesized register array must be named and measured as a **register array**, not relabeled SRAM. The Level C physical area, latency, read energy and write energy fields remain missing; no physical SRAM savings are awarded. Decision D9 remains open pending a verified memory option and its port, word-width, area and energy characterization.

## Artifacts

- `results/tables/memory-all-formats.csv`: all 25 formats at 32- and 64-bit words, same values and capacity.
- `results/tables/memory-throughput-roofs.csv`: same 8 GB/s, 4× reuse roof for every format.
- `results/tables/memory-graph-examples.csv` and `memory-layer-examples.csv`: all 100 retained graph configurations and largest-MAC layer examples.
- `results/tables/memory-mapped-per-row.csv` and `memory-hypothetical-blocks.csv`: explicit nondefault scale scenarios.
- `results/figures/memory-effective-bits.svg`: concise 64-bit-word visual comparison.
- `results/summaries/memory-model-1.0.0.json`: provenance, coverage, assumptions and evidence limits.

The full numerical-to-hardware identity, conversion RTL, workload-derived activity power, physical memory characterization and final candidate evaluation are later evidence gates. These tables do not close D7, D9 or the public Pareto study.
