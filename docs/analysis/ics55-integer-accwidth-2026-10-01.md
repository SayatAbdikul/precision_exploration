# ICsprout55 integer MAC: accumulator width at equal clock (2026-10-01)

Measures what the accumulator width of the existing integer MAC costs in area and clock period on the ICsprout55 library, for multiplier widths 4, 5, 6 and 8 and accumulator widths 16, 20, 24, 28, 32, 40, 48, 56 and 64, at six clock targets, and makes a first, preliminary join to the accuracy of the eight exact strict-A integer configurations. Method identical to stage 2 (`ics55-integer-isoclock-2026-10-01.md`); no new RTL, only parameter values of `public/generic_rtl/mac/integer_mac.sv`.

## Result in brief
- **All 36 configurations pass the existing RTL conformance check; all 216 points (36 x 6 targets) have zero max-transition and zero max-capacitance violations; 108 distinct mapped netlists; all 108 pass gate-level simulation on the conformance vectors and on every operand pair, and all 108 are formally equivalent to the RTL (register-cut ABC `cec`).** The 48 points of the eight stage-2 configurations (widths 4/5/6/8 at ACC 32 and 64, six targets) reproduce stage 2 exactly (same netlist hash and core area).
- **Area per accumulator bit** (slope of a straight-line fit of core area on accumulator width, 9 widths, R^2 0.99977 to 0.99996 at 12 and 10 ns): INT4 65.1, INT5 74.3, INT6 74.3, INT8 82.8 library area units per bit (core), i.e. 2.8 to 3.1 percent of the INT32 core per bit; at the 8 ns clock target 65.8, 75.8, 74.5, 84.8. Total area (core plus the neutral launch/capture wrapper) per bit: 80.5, 89.8, 89.7, 98.2 at 12 ns. Largest residual of the fit: 13.6 to 33.9 units at 12 ns (0.3 to 0.7 percent of the largest area; 89.9 units at 8 ns for INT8): the area is close to linear in accumulator width.
- **Area per multiplier-width step** at ACC 32 (OLS slope over W = 4, 5, 6, 8): 194.3 core units per multiplier bit (pairwise 255.4, 104.4, 220.5 per bit for 4->5, 5->6, 6->8; the pairs are not smooth, mapping noise); at ACC 64 314.1 (12 ns). One accumulator bit costs 0.34 to 0.43 of a multiplier step, i.e. one multiplier-width step buys 2.3 to 3.0 accumulator bits. INT8 minus INT4 at ACC 32 is 800.8 core units = 9.7 to 12.3 accumulator bits (12 ns).
- Smallest valid achieved period rises from 3.4 to 4.2 ns (ACC 16 to 32; 19 of those 20 configurations still meet the tightest target tried, 4 ns, so these are bounded by the sweep range, not floors) through 4.5 to 4.7 ns (ACC 40), 5.2 to 5.5 (48), 5.8 to 6.0 (56) to 6.1 to 6.7 ns (ACC 64). The 32-bit to 64-bit gap in smallest valid period is 2.3 to 2.7 ns.
- Targets all 36 configurations meet validly: 12, 10 and 8 ns (6, 5 and 4 ns are missed by the wider ones). The equal-clock tables and figure are therefore at 8 ns (binding) and 12 ns.
- Ratios reproduced from stage 2: INT4/INT8 at ACC 32 core 0.7268; INT8 ACC 64 over ACC 32 core 1.892 (12 ns, 10 ns), 1.948 (8 ns).
- **Preliminary join, eight exact strict-A integer configurations** (required width from the static bound dot_bound + stored bias, see below): required / covering synthesised width, core area at 8 ns, smallest valid period (ns): ResNet18 INT4 26/28, 1864, 3.75; INT5 28/28, 2066, 3.80; INT6 30/32, 2490, 3.89; INT8 34/40, 3558, 4.61; MobileNetV2 INT4 54/56, 3673, 5.84; INT5 56/56, 4194, 5.99; INT6 58/64, 4915, 6.26; INT8 62/64, 5711, 6.44. The requirement is set by the stored bias of the first layer, not by the dot product (dot-only: 18 to 28 bits).

## Accumulator behaviour (correction to the task text)
The task text says the existing accumulator wraps on overflow. The RTL does not wrap: `integer_mac.sv` computes an (ACC+1)-bit sum and clamps it with `saturate()` to `ACC_MAX`/`ACC_MIN` after every multiply-add and again after the bias add (its header comment says "saturating"; the oracle manifest says `overflow: saturate`; the conformance vectors include clamping cases). No accumulator logic was added or changed; the saturating accumulator is what was synthesised and measured. Cost of saturation is therefore included in every number below, and a wrapping accumulator would differ by an amount not measured here.

## Tools and provenance
- Image `openroad/orfs@sha256:6da005d1c3447799f7401215174196c75ba9ec7e28417624d0c20225f85d594c` via `tools/hardware/openroad_docker.sh` (sha256 `99c2ba0d13692b91de404caafe059f56d3715bda425b04935dc6969fdcf9d5a1`); `OpenSTA 3.1.0`, `OpenROAD unknown`; host Yosys `0.53+67 (git sha1 4f968c669)`; Icarus Verilog `13.0 (devel) (s20250103-36-gea26587b5)`.
- Library `ics55_LLSC_H7CR_typ_tt_1p2_25_nldm.lib` sha256 `38a9f9056a7e4417451e0aaa137efaf0461ee1fc574a8b2ad53622def34bb2b7`; other hashes in `results/summaries/ics55-integer-accwidth-v1.json`.
- `tools/hardware/integer_accwidth_sweep.py` imports `run_point`, the Tcl, the parsers and the validity rule from `tools/hardware/integer_isoclock_matrix.py` (not edited) and the conformance generator from `tools/hardware/integer_mac_baseline.py` (not edited). Same constraints as stage 2: clock uncertainty 0.10 ns, input and output delay 0.20 ns, 0.01 pF output load, ABC `-D target*1000-450` ps, `set_driving_cell BUFX2H7R`, `set_load 10` (fF), ideal clock, typical corner, no wires. Validity rule: a point is valid only with no pin on its worst path violating max transition, max capacitance or max fanout; here every point also has zero design-wide violations (the library has no max_fanout). The resizer would have run only on invalid points; none was needed.
- Summary logic differs from stage 2 in one way requested: a netlist is identified by its SHA-256 and periods are compared with a tolerance of 1e-4 ns (never exact float equality). `smallest_valid` lists every hash within tolerance of the minimum and whether the point where that netlist first appears meets its target. Unit tests: `tests/unit/test_integer_accwidth_sweep.py` (12 tests, including one netlist whose periods differ in the sixth decimal, 3.887763 and 3.887764, which must stay one netlist that meets its target).

## Conformance of the RTL at every width
The existing conformance tool (`tools/hardware/integer_mac_baseline.py run_one`: vectors from the oracle `model_c`, exhaustive pairs for W <= 6, sampled for W = 8, directed dots, bias extremes, clamping cases) ships accumulator manifests only for 32 and 64 bits, so its command line covers no other width. The driver in the sweep script installs a hook on the tool's own `format_named` that returns the shipped `int32_accumulator` manifest with only `name`, `bits` and `numeric.integer_bits` changed (checked equal to the shipped int32 and int64 manifests at 32 and 64 bits, apart from the free-text `notes`). All 36 configurations pass (`artifacts/ppa/ics55-integer-accwidth-v1/conformance-summary.json`). Limit: for ACC 16 to 28 the oracle manifest is a driver-supplied analogue, not a manifest accepted by the format registry; it follows the same rules (signed two's complement, RNE, saturate). For W = 8 the vectors are sampled, not exhaustive, and the INT8/INT32 saturation run exists only for ACC 32; saturation at the other widths is covered by the bias-extreme and clamping vectors and by the exhaustive-pair gate-level run below.

## Function of the mapped netlists
Scripts in `artifacts/ppa/ics55-integer-accwidth-v1/function/` (adapted from the stage-2 reviewer's `review/r2` scripts; bias values in the exhaustive stimulus are clamped to the accumulator range, which the original did not need):
- `gatesim.py`: for each of the 108 distinct netlists (one per hash) the mapped netlist with the PDK Verilog cell models is run (a) on the conformance vectors (e.g. 4745 cycles for INT8/ACC16) and (b) on every (weight, activation) pair, 2^(2W), shuffled into random dots with random and extreme biases against an independent Python model of the saturating MAC (e.g. 66891 cycles for INT8). Controls: the RTL passes both streams for all 36 configurations, and a one-cell mutant of each netlist fails the conformance vectors (108 of 108). Result: 108 of 108 netlists pass both (`function/gatesim-result.json`).
- `cec.py`: register-cut ABC `cec` of each netlist against the RTL elaborated with the matrix's own pre-mapping Yosys steps, registers matched one to one by name, all next-state and output bits. 108 of 108 equivalent, including all INT8 netlists (stage 2 left its INT8 netlists undecided within its time limits); 36 one-cell mutants (one per configuration) are non-equivalent (`function/cec-result.json`; `cec-result-raw-mislabelled-mutants.json` is the same run before a text-match bug that labelled the mutant verdicts "undecided" was fixed in the post-processing; the ABC outputs in `function/cec/*.out` are unchanged). Limit: the gold model rests on Yosys's reading of the RTL; the Icarus simulation is the independent check of that. Equivalence shows that mapping, buffering and sizing changed no logic; saturation behaviour is covered by both.

## Commands
```
cd /home/maveric/precision_exploration
.venv/bin/python tools/hardware/integer_accwidth_sweep.py conformance
.venv/bin/python tools/hardware/integer_accwidth_sweep.py synth            # 216 points, 8 workers, about 6 minutes
.venv/bin/python artifacts/ppa/ics55-integer-accwidth-v1/function/gatesim.py 8
.venv/bin/python artifacts/ppa/ics55-integer-accwidth-v1/function/cec.py 8 900
.venv/bin/python tools/hardware/integer_accwidth_sweep.py determinism      # points 6:40:5 and 8:64:8
.venv/bin/python tools/hardware/integer_accwidth_sweep.py summarize        # refuses to overwrite
.venv/bin/python -m pytest -q tests/unit/test_integer_accwidth_sweep.py
/home/maveric/precision_exploration/tools/hardware/openroad_docker.sh --version
```
Per point files (`synthesis.ys`, `abc.constr`, `netlist.v`, `yosys-stat.json`, `yosys.log`, `sta.tcl`, `opensta.txt`, `drv.txt`, `sta.log`) are in `artifacts/ppa/ics55-integer-accwidth-v1/points/int<W>-acc<A>/t<T>ns/`; all records in `records.json` there. Outputs: `results/summaries/ics55-integer-accwidth-v1.json`, `results/tables/ics55-integer-accwidth-v1.csv` (216 rows with netlist hash and `distinct_netlist_index`), `results/tables/ics55-integer-quality-cost-prelim-v1.csv`, `results/figures/ics55-integer-accwidth-v1.png/.pdf`.
Process note: the first `summarize` output (my own, a few minutes old, files under `results/`) was deleted and regenerated once because I added columns (tightest target met, dot-only width) to the join table; nothing else was removed or overwritten. The raw synthesis and function artifacts were not regenerated.

## Core and total area per configuration
Each cell: core / total area (library units, Yosys `stat -liberty`; total includes the launch/capture wrapper whose registers scale with accumulator width). `*` marks a target that the netlist misses (its achieved period is above the target); every cell is valid. Per-point numbers are in the CSV.

Target 12 ns: core / total area; '*' = target missed

| ACC | INT4 | INT5 | INT6 | INT8 |
|---|---|---|---|---|
| 16 | 1080 / 1420 | 1185 / 1540 | 1290 / 1660 | 1558 / 1960 |
| 20 | 1352 / 1753 | 1493 / 1910 | 1616 / 2048 | 1903 / 2366 |
| 24 | 1598 / 2061 | 1785 / 2264 | 1904 / 2398 | 2230 / 2755 |
| 28 | 1864 / 2389 | 2066 / 2606 | 2186 / 2741 | 2563 / 3150 |
| 32 | 2130 / 2717 | 2386 / 2988 | 2490 / 3107 | 2931 / 3579 |
| 40 | 2677 / 3387 | 2968 / 3694 | 3067 / 3807 | 3558 / 4330 |
| 48 | 3154 / 3987 | 3568 / 4416 | 3686 / 4551 | 4224 / 5120 |
| 56 | 3674 / 4631 | 4154 / 5126 | 4237 / 5225 | 4872 / 5891 |
| 64 | 4217 / 5298 | 4762 / 5858 | 4893 / 6005 | 5545 / 6688 |

Target 8 ns: core / total area; '*' = target missed

| ACC | INT4 | INT5 | INT6 | INT8 |
|---|---|---|---|---|
| 16 | 1080 / 1420 | 1185 / 1540 | 1290 / 1660 | 1558 / 1960 |
| 20 | 1352 / 1753 | 1493 / 1910 | 1616 / 2048 | 1903 / 2366 |
| 24 | 1598 / 2061 | 1785 / 2264 | 1904 / 2398 | 2230 / 2755 |
| 28 | 1864 / 2389 | 2066 / 2606 | 2186 / 2741 | 2563 / 3150 |
| 32 | 2130 / 2717 | 2386 / 2988 | 2490 / 3107 | 2931 / 3579 |
| 40 | 2677 / 3387 | 2968 / 3694 | 3067 / 3807 | 3558 / 4330 |
| 48 | 3153 / 3987 | 3632 / 4481 | 3686 / 4551 | 4224 / 5119 |
| 56 | 3673 / 4630 | 4194 / 5167 | 4236 / 5223 | 4872 / 5891 |
| 64 | 4279 / 5360 | 4827 / 5923 | 4915 / 6026 | 5711 / 6854 |

Target 6 ns: core / total area; '*' = target missed

| ACC | INT4 | INT5 | INT6 | INT8 |
|---|---|---|---|---|
| 16 | 1080 / 1420 | 1185 / 1540 | 1290 / 1660 | 1558 / 1960 |
| 20 | 1352 / 1753 | 1493 / 1910 | 1616 / 2048 | 1903 / 2366 |
| 24 | 1598 / 2061 | 1785 / 2264 | 1904 / 2398 | 2230 / 2755 |
| 28 | 1864 / 2389 | 2066 / 2606 | 2186 / 2741 | 2563 / 3150 |
| 32 | 2130 / 2717 | 2386 / 2988 | 2490 / 3107 | 2931 / 3579 |
| 40 | 2694 / 3404 | 3002 / 3728 | 3099 / 3840 | 3639 / 4411 |
| 48 | 3327 / 4161 | 3820 / 4668 | 3864 / 4729 | 4488 / 5383 |
| 56 | 4019 / 4976 | 4468 / 5440 | 4541 / 5529 | 5212 / 6231 |
| 64 | 4540 / 5621 * | 5167 / 6263 * | 5233 / 6345 * | 5996 / 7138 * |

Target 5 ns: core / total area; '*' = target missed

| ACC | INT4 | INT5 | INT6 | INT8 |
|---|---|---|---|---|
| 16 | 1080 / 1420 | 1185 / 1540 | 1290 / 1660 | 1558 / 1960 |
| 20 | 1352 / 1753 | 1493 / 1910 | 1616 / 2048 | 1903 / 2366 |
| 24 | 1598 / 2061 | 1785 / 2264 | 1904 / 2398 | 2230 / 2755 |
| 28 | 1863 / 2388 | 2066 / 2607 | 2187 / 2743 | 2639 / 3226 |
| 32 | 2161 / 2748 | 2409 / 3011 | 2552 / 3170 | 3041 / 3689 |
| 40 | 2843 / 3553 | 3165 / 3891 | 3300 / 4041 | 3817 / 4589 |
| 48 | 3384 / 4217 * | 3777 / 4626 * | 3929 / 4794 * | 4468 / 5363 * |
| 56 | 4019 / 4976 * | 4468 / 5440 * | 4541 / 5529 * | 5212 / 6231 * |
| 64 | 4540 / 5621 * | 5167 / 6263 * | 5233 / 6345 * | 5996 / 7138 * |

Target 4 ns: core / total area; '*' = target missed

| ACC | INT4 | INT5 | INT6 | INT8 |
|---|---|---|---|---|
| 16 | 1080 / 1420 | 1185 / 1540 | 1290 / 1660 | 1559 / 1960 |
| 20 | 1358 / 1759 | 1493 / 1910 | 1619 / 2051 | 1959 / 2423 |
| 24 | 1618 / 2081 | 1861 / 2339 | 1950 / 2444 | 2319 / 2844 |
| 28 | 1969 / 2494 | 2150 / 2690 | 2322 / 2878 | 2669 / 3256 |
| 32 | 2325 / 2912 | 2573 / 3175 | 2677 / 3294 | 3089 / 3737 * |
| 40 | 2878 / 3588 * | 3160 / 3886 * | 3228 / 3969 * | 3750 / 4522 * |
| 48 | 3384 / 4217 * | 3777 / 4626 * | 3929 / 4794 * | 4468 / 5363 * |
| 56 | 4019 / 4976 * | 4468 / 5440 * | 4541 / 5529 * | 5212 / 6231 * |
| 64 | 4540 / 5621 * | 5167 / 6263 * | 5233 / 6345 * | 5996 / 7138 * |

Area is not strictly monotone in the target (ABC is heuristic): for example INT4/ACC48 is 3154 at 12 and 10 ns and 3153 at 8 ns.

## Smallest valid achieved period
Achieved period = clock target minus worst setup slack; ideal clock, typical corner, no wires. The table gives the smallest over the six targets with, in brackets, the tightest target the design meets validly. Caveat: for ACC 16 to 28 (and ACC 32 except INT8) the design still meets 4 ns, the tightest target of this sweep, so the period shown is an upper bound set by the sweep range and by ABC's target, not a floor; for ACC 16 INT4 and INT5 have one single netlist at all six targets, so the period shown is that of the area-optimal mapping. The 40 bit and wider rows are floored by the mapping (identical netlists at tighter targets) and miss the tighter targets.

Smallest valid achieved period (ns), tightest target met

| ACC | INT4 | INT5 | INT6 | INT8 |
|---|---|---|---|---|
| 16 | 3.363 (4 ns) | 3.431 (4 ns) | 3.594 (4 ns) | 3.690 (4 ns) |
| 20 | 3.755 (4 ns) | 3.654 (4 ns) | 3.730 (4 ns) | 3.753 (4 ns) |
| 24 | 3.764 (4 ns) | 3.762 (4 ns) | 3.813 (4 ns) | 3.802 (4 ns) |
| 28 | 3.755 (4 ns) | 3.804 (4 ns) | 3.788 (4 ns) | 3.789 (4 ns) |
| 32 | 3.821 (4 ns) | 3.968 (4 ns) | 3.888 (4 ns) | 4.164 (5 ns) |
| 40 | 4.527 (5 ns) | 4.712 (5 ns) | 4.711 (5 ns) | 4.608 (5 ns) |
| 48 | 5.213 (6 ns) | 5.436 (6 ns) | 5.173 (6 ns) | 5.468 (6 ns) |
| 56 | 5.838 (6 ns) | 5.987 (6 ns) | 5.922 (6 ns) | 5.853 (6 ns) |
| 64 | 6.105 (8 ns) | 6.675 (8 ns) | 6.261 (8 ns) | 6.444 (8 ns) |

Number of distinct netlists per configuration: 1 to 4 (108 in total).

## Straight-line fits of core area against accumulator width
Least squares over the nine widths, per multiplier width. The fits at 12 and 10 ns are identical to within 0.1 unit (the 12 and 10 ns netlists differ only for INT8/ACC64); at 8 ns the 48, 56 and 64-bit netlists differ from the 12 ns ones for every multiplier width (tighter mapping), which gives larger residuals. Residuals are listed for ACC 16, 20, 24, 28, 32, 40, 48, 56, 64.

Fit at 12 ns, core area = intercept + slope*ACC

| mult | slope/bit | intercept | R2 | max abs resid | residuals (ACC 16..64) |
|---|---|---|---|---|---|
| int4 | 65.07 | 44.5 | 0.99984 | 29.3 | -5, +6, -8, -3, +3, +29, -14, -15, +8 |
| int5 | 74.33 | -1.5 | 0.99996 | 13.6 | -3, +8, +3, -14, +9, -3, +1, -8, +7 |
| int6 | 74.26 | 112.4 | 0.99977 | 33.9 | -10, +18, +10, -6, +1, -16, +9, -34, +28 |
| int8 | 82.79 | 247.8 | 0.99990 | 33.9 | -14, -1, -4, -3, +34, -1, +3, -12, -1 |

Total area slope: int4 80.52, int5 89.77, int6 89.71, int8 98.23
mult step (OLS over W=4,5,6,8) at ACC32 194.3, ACC64 314.1; pairwise ACC32: [255.4, 104.4, 220.5]
acc bit / mult step (ACC32): {'int4': 0.335, 'int5': 0.383, 'int6': 0.382, 'int8': 0.426} INT8-INT4 at ACC32 = 800.8 = acc bits {'int4': 12.3, 'int5': 10.8, 'int6': 10.8, 'int8': 9.7} INT4/INT8 0.7268 ACC64/32 1.8920
mult step by ACC: {'acc16': 120.0, 'acc20': 137.1, 'acc24': 155.4, 'acc28': 171.6, 'acc32': 194.3, 'acc40': 212.8, 'acc48': 255.1, 'acc56': 283.2, 'acc64': 314.1}

Fit at 10 ns, core area = intercept + slope*ACC

| mult | slope/bit | intercept | R2 | max abs resid | residuals (ACC 16..64) |
|---|---|---|---|---|---|
| int4 | 65.07 | 44.5 | 0.99984 | 29.3 | -5, +6, -8, -3, +3, +29, -14, -15, +8 |
| int5 | 74.33 | -1.5 | 0.99996 | 13.6 | -3, +8, +3, -14, +9, -3, +1, -8, +7 |
| int6 | 74.26 | 112.4 | 0.99977 | 33.9 | -10, +18, +10, -6, +1, -16, +9, -34, +28 |
| int8 | 82.79 | 247.9 | 0.99990 | 33.9 | -14, -1, -4, -3, +34, -1, +3, -12, -1 |

Total area slope: int4 80.52, int5 89.77, int6 89.71, int8 98.23
mult step (OLS over W=4,5,6,8) at ACC32 194.3, ACC64 314.0; pairwise ACC32: [255.4, 104.4, 220.5]
acc bit / mult step (ACC32): {'int4': 0.335, 'int5': 0.383, 'int6': 0.382, 'int8': 0.426} INT8-INT4 at ACC32 = 800.8 = acc bits {'int4': 12.3, 'int5': 10.8, 'int6': 10.8, 'int8': 9.7} INT4/INT8 0.7268 ACC64/32 1.8919
mult step by ACC: {'acc16': 120.0, 'acc20': 137.1, 'acc24': 155.4, 'acc28': 171.6, 'acc32': 194.3, 'acc40': 212.8, 'acc48': 255.1, 'acc56': 283.2, 'acc64': 314.0}

Fit at 8 ns, core area = intercept + slope*ACC

| mult | slope/bit | intercept | R2 | max abs resid | residuals (ACC 16..64) |
|---|---|---|---|---|---|
| int4 | 65.83 | 23.6 | 0.99951 | 42.3 | +3, +11, -6, -3, -0, +20, -30, -38, +42 |
| int5 | 75.83 | -37.2 | 0.99979 | 29.6 | +9, +14, +2, -20, -4, -28, +30, -15, +11 |
| int6 | 74.52 | 105.4 | 0.99964 | 42.5 | -7, +20, +10, -6, +0, -20, +4, -43, +40 |
| int8 | 84.83 | 191.6 | 0.99898 | 89.9 | +10, +14, +3, -4, +25, -27, -40, -71, +90 |

Total area slope: int4 81.27, int5 91.27, int6 89.96, int8 100.27
mult step (OLS over W=4,5,6,8) at ACC32 194.3, ACC64 339.4; pairwise ACC32: [255.4, 104.4, 220.5]
acc bit / mult step (ACC32): {'int4': 0.339, 'int5': 0.39, 'int6': 0.383, 'int8': 0.437} INT8-INT4 at ACC32 = 800.8 = acc bits {'int4': 12.2, 'int5': 10.6, 'int6': 10.7, 'int8': 9.4} INT4/INT8 0.7268 ACC64/32 1.9484
mult step by ACC: {'acc16': 120.0, 'acc20': 137.1, 'acc24': 155.4, 'acc28': 171.6, 'acc32': 194.3, 'acc40': 212.8, 'acc48': 249.4, 'acc56': 279.7, 'acc64': 339.4}
{'iverilog': 'Icarus Verilog version 13.0 (devel) (s20250103-36-gea26587b5)', 'openroad_wrapper': 'image openroad/orfs@sha256:6da005d1c3447799f7401215174196c75ba9ec7e28417624d0c20225f85d594c\nOpenSTA 3.1.0\nOpenROAD unknown', 'yosys': 'Yosys 0.53+67 (git sha1 4f968c669, clang++ 18.1.8 -fPIC -O3)'} 99c2ba0d1369 38a9f9056a7e

Reading: a bit of accumulator costs about 65 to 85 core units depending on the multiplier width (INT5 and INT6 are equal within the mapping noise), a bit of multiplier width about 194 at ACC 32 and 314 at ACC 64 (12 ns; 339 at 8 ns); the multiplier step therefore grows with the accumulator width (120 at ACC 16, 137, 155, 172, 194, 213, 255, 283, 314 at ACC 64). The fit describes this mapping only; the residual structure is not random for INT8 at 8 ns (negative at 40 to 56 bits, positive at 64), which fits timing pressure but was not isolated. Why the per-bit cost depends on the multiplier width was not investigated.

## Join to accuracy (PRELIMINARY)
**Do not quote as a result. This is a first join of two independent bodies of evidence.**

Rule used. For each exact strict-A integer configuration the audited per-graph accumulator bounds (`artifacts/phase3/configurations/<id>/accumulator-bounds.json`, referenced and hash-checked by `results/summaries/phase3-accumulator-audit.json`) give for every reduction a bound on the dot-product magnitude (`dot_bound`) and the largest stored bias in accumulator codes (`maximum_stored_bias_codes`); the file's own `remaining_headroom` equals 2^(bits-1)-1 minus their sum (checked for every reduction by the script). Required width = smallest signed width n with 2^(n-1)-1 >= max over reductions of (`dot_bound` + `maximum_stored_bias_codes`); the covering width is the smallest synthesised width >= n. For the four MobileNetV2 configurations and ResNet18 INT8 the evidence already records the INT64 resolution (the INT32 bounds were insufficient), consistent with this rule. Accuracy: 1,000-image paired screen analyses `results/summaries/phase3-analysis-<id>.json` (ids in `docs/architecture/phase3-eight-integer-review.md`; configuration hash checked against the audit).

| Model / format | Top-1 (95% paired change vs FP32, pts) | Top-5 | Accumulator in accuracy run | Required bits (static bound) | Dot-only bits | Covering width | Core area at 8 ns (12 ns) | Smallest valid period, ns (tightest target met) |
|---|---|---|---|---|---|---|---|---|
| ResNet18 INT4 | 12.3% (-60.9, -54.6) | 33.3% | INT32 | 26 | 20 | 28 | 1863.7 (1863.7) | 3.755 (4 ns) |
| ResNet18 INT5 | 48.3% (-24.7, -18.8) | 75.0% | INT32 | 28 | 22 | 28 | 2066.1 (2066.1) | 3.804 (4 ns) |
| ResNet18 INT6 | 62.5% (-10.0, -5.2) | 85.4% | INT32 | 30 | 24 | 32 | 2490.0 (2490.0) | 3.888 (4 ns) |
| ResNet18 INT8 | 68.4% (-3.0, -0.4) | 89.1% | INT64 | 34 | 28 | 40 | 3558.2 (3558.2) | 4.608 (5 ns) |
| MobileNetV2 INT4 | 0.2% (-74.7, -69.1) | 0.9% | INT64 | 54 | 18 | 56 | 3672.8 (3673.6) | 5.838 (6 ns) |
| MobileNetV2 INT5 | 6.4% (-68.7, -62.7) | 14.8% | INT64 | 56 | 20 | 56 | 4194.4 (4153.5) | 5.987 (6 ns) |
| MobileNetV2 INT6 | 46.0% (-29.1, -23.0) | 68.7% | INT64 | 58 | 22 | 64 | 4914.8 (4893.3) | 6.261 (8 ns) |
| MobileNetV2 INT8 | 70.6% (-3.0, -0.1) | 89.7% | INT64 | 62 | 26 | 64 | 5710.9 (5545.4) | 6.444 (8 ns) |

FP32 top-1 is 70.1% (ResNet18) and 72.1% (MobileNetV2). The CSV also holds the covering width and the 8 ns core area if only the dot-product bound were required (20, 24, 24, 28, 20, 20, 24, 28 bits: areas 1351.6, 1785.0, 1904.3, 2562.8, 1351.6, 1493.2, 1904.3, 2562.8); `not established` appears where a requirement cannot be derived (none here).

What it says, with care. Within the eight configurations the required width is set almost entirely by the stored bias of the first convolution (ResNet18 `conv1`, MobileNetV2 `features_0_0`): the product-domain bias code needs 26 to 62 bits, because it is the real bias divided by the tiny product of weight and activation scales; the dot product alone needs 18 to 28. MobileNetV2 needs 54 to 62 bits and therefore the widest, slowest MACs while its INT4 and INT5 accuracies (0.2% and 6.4%) are the worst; this is a join of two measurements, not a cause: the loss is attributed elsewhere (`phase3-eight-integer-review.md`) and the accuracy comes from the strict recipe with its own causes. The cost of a width requirement is that going from the INT8 requirement 34 bits (ResNet18, covering 40) to 62 bits (MobileNetV2, covering 64) costs 1.60 times the core area at 8 ns (5711 against 3558), at about the same accuracy retention (68.4% and 70.6%).

Caveats, all of them:
1. Preliminary; the accumulator width is a requirement derived from a static worst-case bound, not the smallest width that keeps top-1 (a narrower saturating accumulator might preserve accuracy; the exact engine was not run at narrower widths here). The numbers are an upper bound on need.
2. Strict-A recipe only; the simulator recipes are being repaired in another lane and these accuracies may change.
3. Development evidence on the 1,000-image screen with the `UNCERTAIN` label; no D4 decision exists.
4. MAC only: no scaling, requantisation, bias storage or memory cost; the bias term is counted in the width because the RTL takes the bias as an ACC_BITS input, but nothing else around the MAC is counted.
5. The synthesised accumulator saturates (it does not wrap).
6. Mapped netlists, no placement or wire parasitics, typical corner, no power or energy.
7. The accuracy runs used the INT32 or INT64 accumulator recorded in the table; the bound says it never overflows there, so the same accuracy is expected at the required width, but this was not re-run.
8. The covering width is limited to the nine synthesised widths; rounding up to the next one adds area (for example 58 to 64 bits).

## Determinism
Two points were rerun with the same script into `artifacts/ppa/ics55-integer-accwidth-v1/determinism-rerun/` (INT6/ACC40 at 5 ns and INT8/ACC64 at 8 ns): `netlist.v`, `yosys-stat.json`, `opensta.txt`, `drv.txt` and `abc.constr` are byte-identical to the original run for both (`determinism-result.json`). The 48 stage-2 points that overlap this sweep reproduce stage 2 (same netlist hash and core area), which is a further determinism check.

## Limits
- Mapped netlists only; ideal clock, 0.10 ns uncertainty, no wire parasitics, typical corner, no power.
- Yosys and ABC choose the architecture (ripple-style carry chains); absolute periods are conservative, only the same-flow comparisons are meaningful. The ABC overhead (450 ps) and output load (10 fF) are estimates as in stage 2.
- The sweep stops at a 4 ns target, so periods for accumulators up to 32 bits are bounded by it.
- Intermediate widths use a driver-supplied oracle manifest (see Conformance).
- The fit is of one heuristic mapping, with residuals up to 34 units at 12 ns and 90 at 8 ns.
- Out of scope and untouched: saturating or other new accumulator logic, non-integer formats, placement and routing, power, and anything from stages 1 and 2.
- Raw reports are under the git-ignored `artifacts/`; only hashes are in the summary.
