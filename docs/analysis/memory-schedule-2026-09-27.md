# Comparable tiled memory schedule, version 1.0.0

## Reproduce

```bash
/home/maveric/precision_exploration/.venv/bin/python -m pytest -q tests/unit/test_memory_model.py tests/unit/test_memory_schedule.py
/home/maveric/precision_exploration/.venv/bin/python -m tools.analysis.memory_model --observation-root /home/maveric/precision_exploration
/home/maveric/precision_exploration/.venv/bin/python -m tools.analysis.memory_schedule --observation-root /home/maveric/precision_exploration
```

The observation root is read only. The scheduler verifies the prior summary and each selected graph and shape hash before using their dimensions. It reads no model weights or images and runs no inference. The source is [`public/analysis/memory_schedule.py`](../../public/analysis/memory_schedule.py), with the retained-shape evidence generator in [`tools/analysis/memory_schedule.py`](../../tools/analysis/memory_schedule.py). The all-25-format table remains [`memory-all-formats.csv`](../../results/tables/memory-all-formats.csv); the new 625-row schedule table is [`memory-scheduled-layers.csv`](../../results/tables/memory-scheduled-layers.csv). [`memory-schedule-1.0.0.json`](../../results/summaries/memory-schedule-1.0.0.json) records source and output hashes.

## Same layout and execution loop

Five fixed layer shapes are taken from the retained INT4 graphs: ResNet18 `conv1` (standard 7×7), MobileNetV2 `features_1_conv_0_0` (depthwise 3×3) and `features_1_conv_1` (pointwise 1×1), MobileNetV3-Large `features_1_block_1_0` (pointwise), and YOLOv8n `model_0_conv` (standard 3×3). These choices span all three requested convolution classes and four retained model families. Each of the 25 accepted formats is applied to **the same logical shape and schedule** for a given layer. This avoids attributing prepared-graph layout changes to a datatype. It also means these rows are hypothetical common-layout hardware cases, not claims of accepted numerical graph equivalence.

The base schedule has 8 output channels × 8 output positions × 32 reduction terms per tile. It uses four 64-bit banks, one read and one write port per bank, 8 KiB total bank capacity, 500 MHz, 64 MACs/cycle, and an 8 GB/s shared external budget. Input and weight tiles are double buffered. Weights are reloaded at each spatial tile; gathered input patches are reloaded at each output-channel tile. Standard and pointwise input patches broadcast across eight output channels; depthwise input patches are separate per channel. No reuse across tile boundaries or overlapping input halos is assumed. The model counts every reduction product, including boundary products implied by the retained shape counts.

Input, weight, output and scale tiles use the **same** `tensor_layout` word, line, bank and port model. Weight tiles are `[M,K]`; standard/pointwise gathered inputs are `[P,K]`; depthwise gathered inputs are `[M,P,K]`; outputs are `[M,P]`. The last dimension is independently word aligned in each row. Intrinsic metadata uses each manifest's 8-bit scale and 32-value block along the reduction/output row; this common grouping is a declared schedule assumption, not an assertion that every retained graph already has this axis. Required-mapping formats use one 64-bit analytical scale per W/A/O tensor, cached once at the external interface but read locally on each tile. Exact arbitrary-rational representation and conversion hardware are unresolved. Scales may require search or conversion work outside this traffic model.

Bias is assumed to be 32 bits per output channel, read once per spatial tile **after** the reduction. Output code and scale metadata are written once per tile. A 64-bit accumulator per output in the tile is held in registers and reported separately from bank occupancy; its register area and energy are not characterized. The optional spill scenario writes and rereads that accumulator tile at each reduction-tile boundary and charges both bank-port and external traffic. A format-specific accumulator implementation may differ, so this common 64-bit state is a comparison assumption. Tile-local peak input, weight, output-plus-bias, combined bank and accumulator-register occupancy are separate columns. Per-bank peak occupancy is also checked against each bank's declared depth. Full input/output activation-pair occupancy is calculated with the **same** bank/word organization, as a separate hypothetical resident-pair comparison; this pair generally does not fit the 8 KiB tile bank and is not used as the tile capacity test.

For each tile the model counts packed W/A bank reads and load writes, output-scale/bias reads, output writes, and any accumulator spill accesses. Each independent array starts bank striping at bank zero, so shared-bank collisions consume additional cycles. The local read and write limits are maximum per-bank load divided by declared ports. External transfer bits include W/A, output, metadata, bias and optional spill. Compute, local-read, local-write and external cycles are summed across the schedule. The reported rate divides Model C product count by the **largest** of these four totals, assuming ideal stage overlap; it is an analytical schedule upper bound, not measured accelerator throughput. If tile storage exceeds 8 KiB, throughput is null and the limiter is `capacity`.

## Selected results

Base scenario, 8 GB/s and 64 MACs/cycle at 500 MHz:

| Layer and format | External bits/MAC | Analytical rate (GMAC/s) | Limiter |
|---|---:|---:|---|
| ResNet18 standard, INT4 | 1.17 | 32.00 | compute |
| ResNet18 standard, INT5 / INT6 | 1.61 | 32.00 | compute |
| ResNet18 standard, INT8 | 2.15 | 29.77 | external |
| MobileNetV2 depthwise, INT4 / INT5 / INT6 | 9.33 | 6.86 | external |
| MobileNetV2 depthwise, INT8 | 17.33 | 3.69 | external |
| MobileNetV2 depthwise, MXFP4 E2M1 | 10.44 | 6.13 | external |
| MobileNetV2 pointwise, INT4 | 1.38 | 32.00 | compute |
| YOLOv8n standard, INT8 | 2.81 | 22.74 | external |

Depthwise convolution has little channel broadcast, so bits/MAC are much higher. The 8-position tile makes INT4/5/6 depthwise output and input rows each occupy a 64-bit word; nominal widths tie in this case. INT5 and INT6 tie in the 32-term reduction row too. This is a packing effect at the declared tile boundaries, not an intrinsic datatype ranking. The base INT4 ResNet18 tile allocates 6,144 bits in the local bank and 4,096 additional accumulator-register bits. Neither number is physical area.

For MobileNetV2 pointwise INT4, enlarging the tile from 8×8 to 16×16 cuts external bits/MAC from 1.38 to 0.69 under the assumed broadcast and reload schedule. A 1×1 tile raises it to 12.00 bits/MAC, with local reads limiting the rate to 2.67 GMAC/s. Changing 64-bit words to 32-bit words in the 8×8 tile yields 1.25 bits/MAC because of different line padding; the port and capacity assumptions remain explicit in the table. For ResNet18 INT4, spilling after each 32-term reduction tile raises bits/MAC from 1.17 to 4.65 and lowers this schedule's rate from 32.00 to 13.75 GMAC/s. Seven of the 625 rows, all 16×16 MobileNetV2 depthwise cases, exceed the assumed 8 KiB bank capacity and have no throughput value. This capacity comparison excludes the separately reported accumulator registers.

## Scope and evidence limits

The old one-pass [`memory-throughput-roofs.csv`](../../results/tables/memory-throughput-roofs.csv) remains an **analytical upper-bound scenario** with abstract reuse. The new table supplies a tile-loop reuse schedule and bank contention; neither is measured traffic. The earlier whole-graph activation-liveness peak uses byte-packed buffers; it remains separate and is not added to word/bank-packed weights. This report compares tile-local and full input/output-pair activation occupancy on the same word/bank basis as weights. The full pair uses common last-axis alignment and metadata grouping across formats; it does not claim to reproduce each retained graph's actual encoding layout. A full graph bank-aware liveness schedule would require a declared global allocator and buffer placement.

No ICS55 SRAM macro, characterized port/word option, SRAM area, access energy, or workload-power result is supplied by this model. Register accumulator cost, gather/reorder fabric, broadcast network, scale search/conversion, control, inter-tile synchronization, and buffer-to-bank mapping need implementation evidence. A physically synthesized register array must remain identified as such. These schedules do not close D9 or establish a physical area, energy or candidate ranking.

## Review correction

The reusable `tensor_layout` API now marks cross-word handling only if an actual value crosses a boundary **within a declared packing line**; a three-value INT5 row in a 32-bit word no longer receives a false crossing flag. Total padding fraction includes payload, metadata, bias and accumulator padding, using the same total allocation denominator. Explicit `block_size=0` is rejected rather than replaced by the manifest default. Focused regressions cover all three cases; the existing 50-row format and 100-row retained-graph tables were regenerated.
