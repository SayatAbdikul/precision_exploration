> STATUS (added by the orchestrator): NOT FINAL. Lane P3 was stopped on the owner's instruction on 2026-10-02 at 18:17, during its fix round. Review 1 (artifacts/agent_orchestration/handoffs/P3-accwidth-hw-review-review.md) reproduced every number of substance, but its two blocking corrections (16 sampled points hold 15 distinct netlists, not 16; the clock times in this report are not real) and its must-fix items (definition of the +- on savings; the join table should lead with INT8 signed, the only case whose activation type the RTL carries) are NOT applied below. An unfinished 'round2' synthesis study of the fix round is under artifacts/ppa/ics55-integer-accwidth-review-v1/round2/ (stopped through its STOP file; incomplete).

# Review: ICsprout55 integer MAC accumulator-width study (lane P3, 2026-10-02)

Reviewed: `docs/analysis/ics55-integer-accwidth-2026-10-01.md` (commit 9488439), its handoff `artifacts/agent_orchestration/handoffs/L4-hardware-accwidth.md` and the kept evidence `artifacts/ppa/ics55-integer-accwidth-v1/`, `results/summaries/ics55-integer-accwidth-v1.json`, `results/tables/ics55-integer-accwidth-v1.csv`, `results/tables/ics55-integer-quality-cost-prelim-v1.csv`. Nothing in the study, in a tracked file or in another lane's folder was changed. Scripts and outputs: `artifacts/ppa/ics55-integer-accwidth-review-v1/`. Hardware numbers are synthesis-level (mapped netlists, ideal clock, typical corner, no wires). Accuracy numbers come from the reviewed ResNet18 accumulator sweep (`results/summaries/accumulator-sweep-v1/widths.csv`) and are development evidence on the ImageNet 1k screen. No GPU was used, no network was run and no held-out file was read.

## Numbers first

- **Part 1, tables:** every number in the study's tables and in its "Result in brief" was rebuilt from the per-point raw files of all 216 points. Areas were recomputed as cell counts times Liberty cell areas, and they equal the areas Yosys printed. Periods and slacks come from `opensta.txt` and violations from `drv.txt`. There are **0 mismatches** against `records.json`, the 216-row CSV, the summary JSON, the five area tables, the period table, the three fit tables and the join CSV. The join CSV's required widths were recomputed from the hash-checked bounds files. Slopes per accumulator bit at 12 ns are 65.07, 74.33, 74.26 and 82.79 core units for INT4, INT5, INT6 and INT8 (R^2 0.99977 to 0.99996). The multiplier step is 194.3 at ACC 32. The smallest valid period is 3.363 to 4.164 ns at ACC 16 to 32 and 6.105 to 6.675 ns at ACC 64. There are 108 distinct netlists.
- **Part 3, native reruns:** **all 216 points** were resynthesised with the native Yosys, which is the same build as the study's (0.53+67, 4f968c669). All 216 netlist SHA-256 values equal the recorded hashes, and all 216 `yosys-stat.json` files are byte-identical. Gate-level simulation of 16 netlists (all of them study hashes, 22k to 111k cycles each) shows **0 cycle differences** against the RTL. The RTL equals an independent model of the engine's saturation rule plus one bias clamp, and 3 of 3 mutants are detected. The register-cut ABC `cec` finds 8 of 8 netlists equivalent, and 2 of 2 mutants are not equivalent. OpenSTA could not be rerun because the container was removed, so timing was **only read** from the stored reports.
- **Missing evidence:** `netlist.v`, `yosys.log` and `sta.log` are **absent for all 216 points**, and also from `determinism-rerun/`. The study document says they are kept. The hashes are now confirmed by resynthesis, but the STA logs cannot be recovered.
- **Part 4, semantics:** the RTL at width W is **not** the engine's `sat.w<W>` for the B2 INT8 and INT6 `default` graphs.
  - In 20 of 21 MAC nodes the activations are unsigned codes (0..255 and 0..63). A signed W-bit port cannot carry them. It needs a W+1-bit signed port, which was not synthesised.
  - The RTL adds the bias **inside** the W-bit register and clamps a second time. The engine adds the bias outside, in binary64. With in-register biases, live `conv1` channels need up to 20 bits (INT8) and 16 bits (INT6), and 8 near-dead `conv1` channels need up to 34 and 30 bits. At W_acc(1.0), 2 live channels (INT8, 19 bits), 6 (INT8 signed, 18 bits) and 2 (INT6, 15 bits) exceed the register.
  - In 16 of 16 points checked, the STA worst path ends at the DUT's `result` register, so it runs through the bias adder and the second clamp.
  - Both sides agree on these points: the clamp after every add, the start from 0, the sign bit counted in the width, and the unit (all shifts are 0, so one unit is one product of integer codes). The tap order is not fixed by the RTL.
- **Part 2, flow sensitivity:** `synth -noalumacc` changes one synthesis option and keeps the multiplier out of the accumulate adder. With it the slope per accumulator bit falls to **44.2, 45.0 and 45.3** units (INT4, INT6, INT8 at 12 ns), against 65.1, 74.3 and 82.8. The multiplier step becomes about 140 to 147 per bit at every accumulator width. The W-dependent slope per bit is therefore an artefact of Yosys `alumacc` merging the multiplier into the (ACC+1)-bit `$macc`. This sensitivity measures area only.
- **Part 5, join (derived, preliminary, conditional on part 4):** core area at 8 ns in the study's flow.

  | Case | certified | no-event | within 1 point |
  |---|---:|---:|---:|
  | INT8 | 27 bits: 2,482 | 21 bits: 1,973 | 19 bits: 1,804 |
  | INT6 | 23 bits: 1,819 | 17 bits: 1,372 | 15 bits: 1,223 (extrapolated) |

  Narrowing from the certified width to the 1-point width saves 679 ± 68 (27.3 %) for INT8, 679 (28.3 %) for INT8 signed and 596 ± 36 (32.8 %) for INT6. With `-noalumacc` the savings are 376 (19.7 %), 376 (20.2 %) and 365 (26.3 %). Every width from 16 to 28 bits meets the 4 ns floor target, so **no period saving can be resolved**. There are no MobileNet cases: `results/summaries/accumulator-sweep-mn-v1/` does not exist.

## Verdict

**Approve with corrections.** The study's measurements are correct and reproducible: every table number, every netlist hash and every function claim that was checked holds. Its own join, against the phase-3 static bounds, is labelled preliminary and stays so. One finding (F1) blocks using the study as the hardware cost of the engine's `sat.w<W>` register. That is a limit of what the RTL is, not an error in the study.

| # | Finding | Blocking? |
|---|---|---|
| F1 | The RTL accumulator is not the engine's `sat.w<W>`. The differences are unsigned activations, the in-register bias add with a second clamp (which also sets the measured critical path), and the tap order (part 4). Any join of these costs to the software widths must state the conditions in part 5, or wait for an RTL variant. | **Blocking for the paper's join**; not for the study's own tables |
| F2 | The raw files are deleted: `netlist.v`, `yosys.log` and `sta.log` are absent in all 216 point folders and in `determinism-rerun/`. The document's Commands section says they are kept. The netlists are reproduced byte for byte by native Yosys; the STA logs cannot be recovered. | No (after correction C1) |
| F3 | The cost per accumulator bit depends on the flow. `alumacc` merges `$mul` into the accumulate `$add` (every rerun log: "merging $macc model for $mul … into $add"). With `-noalumacc` the slope is 44 to 47 units per bit for every W, and the area is 16 % to 35 % lower (INT8, ACC 20 to 64). The document's open question, "why the per-bit cost depends on the multiplier width", is answered by this. | No, but it must be stated (C3) |
| F4 | "One multiplier-width step buys 2.3 to 3.0 accumulator bits" holds only at ACC 32. The ratio is 1.4 to 1.8 bits at ACC 16 and 3.8 to 4.8 bits at ACC 64, and in this flow only. | No (C4) |
| F5 | Core area includes 2 flip-flops per accumulator bit (`acc_q`, `result`; 12.32 units per bit, which is 15 % to 19 % of the slope). The hardware-metrics contract asks for logic area and register area separately. | No (C5) |
| F6 | "Absolute periods are conservative" is one-sided. The ideal clock, no wires and the typical corner make them optimistic; ripple-style carry chains make them pessimistic. | No (C6) |
| F7 | A stray debug line (a Python dict of tool versions and two hash prefixes) is printed in the fits section. | No (C7) |
| F8 | The join evidence carries `bounds_status: pending_native_sensitivity` in the summary, and the document does not mention it. | No (C8) |
| F9 | An earlier, unfinished review left a folder `review/` inside the study's tree (`recompute.py`, `rerun*/`, `drv/`; 2026-10-01 17:34 to 17:41) with no report and no handoff. I did not use it. | No (C9) |
| F10 | An empty `drv.txt` (216 of 216) can no longer be told apart from a failed command, because `sta.log` is gone. The `opensta.txt` files of the same runs are complete, and the stage-2 calibration showed that this Tcl does report violators. | No |

## Corrections the original document needs (tracked file; not edited here)

- C1, Commands and Limits: state that `netlist.v`, `yosys.log` and `sta.log` were deleted after the run (all 216 points and `determinism-rerun/`). Cite this review's resynthesis: 216 of 216 netlist and stat hashes are identical.
- C2, Result in brief and the join: add that the MAC's accumulator includes an in-register bias add with a second clamp and takes signed W-bit activations. It is not the engine's `sat.w<W>` register (this review, part 4). The worst path ends at `result`.
- C3, Fits: say that the slope per bit and its dependence on W come from Yosys `alumacc` merging the multiplier into the accumulate adder. Under `-noalumacc` the slope is 44 to 47 units per bit and the multiplier step is about 140 to 147. Quote the per-bit cost as flow-specific.
- C4, Result in brief: "at ACC 32" after "one multiplier-width step buys 2.3 to 3.0 accumulator bits". Add the ACC 64 value (3.8 to 4.8).
- C5: report the register part of the core: 2 x DFFQX1H7R per accumulator bit (12.32 units) plus 1, and the rest as logic.
- C6, Limits: replace "absolute periods are conservative" with "absolute periods are not meaningful; ideal clock, no wires and the typical corner bias them low, the inferred carry chains bias them high".
- C7: delete the raw `{'iverilog': …} 99c2ba0d1369 38a9f9056a7e` line under "Fit at 8 ns".
- C8, Join caveats: the bounds records are `pending_native_sensitivity`.
- C9: mention or remove (owner's decision) the unfinished `review/` folder inside `artifacts/ppa/ics55-integer-accwidth-v1/`.
- C10, Join paragraph: "the widest, slowest MACs" should read "the widest MACs, slowest at synthesis level".

## Part 1: tables from raw reports

Script: `part1_recompute.py`. It is independent of the study's code and imports nothing from `tools/hardware`. Outputs are `part1-out.json` and `part1-mismatches.txt`, which is empty.

- **Inputs per point.** From `yosys-stat.json`: the cell counts of the `integer_mac` module and of the harness. The core area is the sum of count times Liberty area, the harness area is computed the same way, and the total is their sum. From `opensta.txt`: worst slack, TNS, arrival, setup, uncertainty and the path pins. The achieved period is the target minus the slack, and the script checks it against arrival + setup + 0.10 ns. The largest difference is 2e-6 ns (INT6/ACC64 at 12 ns), which is print rounding. From `drv.txt`: the violators, of which there are none in any of the 216 files.
- **Records.** For every one of the 216 points, the following agree with `records.json`: core, total and wrapper area (to 0.006), core and total cells, slack, TNS, period, path cells, worst path slew, validity, violation counts, the ABC `-D` value, and the SHA-256 of the report, stat and drv files. The summary's `records` equal `records.json`.
- **Netlist identity.** Points that share a recorded hash have identical cell counts, harness area and path delay. No set of identical cell counts carries two hashes. The count is 108 distinct, 1 to 4 per configuration. Part 3 confirms the hashes themselves.
- **Derived numbers.** The following equal the summary JSON to 1e-6:
  - the fits at 12, 10 and 8 ns: slope, intercept, R^2, residuals and largest residual, for core and total area;
  - the multiplier steps for all nine widths, with the pairwise steps;
  - the ratios: INT4/INT8 at ACC 32 = 0.7268, and INT8 ACC 64/32 = 1.8920, 1.8919 and 1.9484 at 12, 10 and 8 ns;
  - per configuration: the smallest valid period, its tied targets and the tightest target met;
  - the common targets, which are 12, 10 and 8 ns.
- **Document.** All 180 area cells (core and total, rounded), all 180 missed-target stars, the 36 period cells and the three fit tables, residuals included, agree with the raw files.
- **Result in brief.** Every number reproduces. The per-bit slope is 2.82 % to 3.12 % of the INT32 core. The largest residual is 13.6 to 33.9 units at 12 ns, which is 0.29 % to 0.69 % of the largest area. The multiplier step is 194.3, with pairwise steps of 255.4, 104.4 and 220.5. The accumulator-bit to multiplier-step ratio is 0.335 to 0.426. INT8 minus INT4 is 800.8 units, or 9.7 to 12.3 accumulator bits. The smallest-period ranges are 3.4 to 4.2, 4.5 to 4.7, 5.2 to 5.5, 5.8 to 6.0 and 6.1 to 6.7 ns. 19 of the 20 configurations up to ACC 32 meet 4 ns. The period gap from ACC 32 to ACC 64 is 2.28 to 2.71 ns. The 12 ns and 10 ns netlists differ only for INT8/ACC64. At 8 ns the 48, 56 and 64-bit netlists of every W differ from the 12 ns ones.
- **Join CSV.** The required widths were recomputed from the eight bounds files, whose SHA-256 values were checked: 26, 28, 30, 34, 54, 56, 58 and 62 bits, and 20, 22, 24, 28, 18, 20, 22 and 26 bits for the dot product alone. The binding node, the covering widths, the areas at 8 and 12 ns and the periods agree.

## Part 2: method

- **Constraints, checked in all 216 point scripts.**
  - Yosys: `chparam` gives W_BITS = A_BITS = W, as stated.
  - ABC: `-D` = target x 1000 - 450 ps, and `abc.constr` = `set_driving_cell BUFX2H7R` + `set_load 10`.
  - STA: period = target, clock uncertainty 0.1 ns, input and output delay 0.2 ns, output load 0.01 pF, no driving cell, so input slew is ideal.
  - `check_setup -verbose` is empty in all 216 reports, so every endpoint is constrained.
  - The Liberty file has no wire-load model, so "no wires" means zero wire capacitance, and it has no `max_fanout`. Its `default_max_transition` is 0.796 ns; the worst slew on a reported path is 0.090 to 0.442 ns.
- **Inherited ABC load caveat** (stage-2 review, finding B). The 10 fF load sits on every combinational output of each module. None of those nets carries the STA output load.
- **Where the worst path runs.** I mapped the start and end flip-flops of the stored reports onto the resynthesised netlists of 16 points. In every one, the path starts at a launch flop (weight, activation or `start_launch`) and **ends at the DUT `result` register**, after the multiplier, the accumulate adder, the first clamp, the bias adder and the second clamp. The 450 ps ABC overhead assumes the register-to-output class. For register-to-register paths the real overhead is about 0.23 ns (clock-to-Q 0.09 + setup 0.03 + 0.10), so ABC aims about 0.2 ns tighter than needed. Slack always comes from OpenSTA, so this is harmless.
- **Validity rule.** No pin on the worst path may violate max transition, max capacitance or max fanout. The rule is correctly applied and here it is vacuous: there is no violator anywhere (F10). Area is taken at the target, and the period is the smallest valid achieved period over the six targets. A netlist that misses its target can still be clocked at its achieved period. That is legitimate, and the document says so.
- **What "ideal clock, typical corner, no wires" allows.** It allows relative area and relative period within this flow and this RTL. It does not allow absolute fmax, sign-off timing, power or energy, or architecture-independent per-bit costs. These sentences go beyond synthesis-level evidence:
  - "Absolute periods are conservative" (C6).
  - "One multiplier-width step buys 2.3 to 3.0 accumulator bits", stated without the ACC 32 condition and the flow condition (C3, C4).
  - "The widest, slowest MACs" (C10).
  - The per-bit costs when quoted as a property of accumulator width rather than of this mapping (C3).

  The rest of the document is properly qualified.
- **Flow sensitivity** (`part2_flow_sensitivity.py`, `flow/flow-sensitivity.json`). It repeats the study's script for INT4, INT6 and INT8, all nine widths, at 12 and 8 ns, with only `synth -noalumacc` changed. The `study` arm reproduces the study's fits exactly.

  | Fit | slope per acc bit 12 ns (INT4 / INT6 / INT8) | 8 ns | multiplier step over W = 4, 6, 8 at ACC 16 / 32 / 64 (12 ns) |
  |---|---|---|---|
  | study flow | 65.07 / 74.26 / 82.79 | 65.83 / 74.52 / 84.83 | 119.6 / 200.2 / 332.0 |
  | `-noalumacc` | 44.21 / 44.96 / 45.32 | 44.42 / 45.67 / 47.00 | 142.0 / 139.9 / 147.1 |

  For INT8 at 12 ns the core area is 1,594, 2,145 and 3,585 under `-noalumacc`, against 1,903, 2,931 and 5,545, at ACC 20, 32 and 64. ABC's own delay estimate (`stime`) is not worse: 3,900 against 4,079 ps at ACC 20, and 9,050 against 9,626 at ACC 64. The variant netlists were not timed with OpenSTA and not function-checked, and they were deleted after reading. The per-bit cost and the interaction between W and ACC are therefore properties of Yosys's arithmetic extraction, not of the MAC.

## Part 3: native reruns

| What | How | Result |
|---|---|---|
| Resynthesis, all 216 points | `rerun_all_hashes.py`: stored `synthesis.ys` and `abc.constr` copied, only the three output paths changed, `yosys -s` with the cwd and TMPDIR in my folder; netlists deleted after hashing | 216 of 216 netlist SHA-256 = `records.json`; 216 of 216 stat files byte-identical; 108 distinct; ABC log shows `-D` and the 10 fF load; about 3 s per point |
| Resynthesis, 16 kept | `rerun_synth.py`: INT4 and INT8 at ACC 16 and 64 at 12 and 4 ns, plus 6:40:5, 8:64:8, 6:24:8, 5:32:6, 6:64:8, 8:20:8, 8:28:10 and 6:16:8 | identical; netlists kept in `rerun/` (4.6 MB) |
| Gate-level simulation | `part3_gatesim.py`, my own testbench and model: every operand pair as a singleton dot (65,536 for INT8), 3,000 random dots with extremes, valid gaps and resets, and long dots that drive into the high and low clamp and then reverse (where reachable: ACC 16 to 28); RTL and netlist in Icarus with the PDK cell models | 16 of 16: 0 cycle-trace differences; RTL results equal the model (`kernel.h` policy-2 rule, then one bias clamp) on every dot; mutants (one NAND2 swapped for NOR2) detected 3 of 3 (19,072, 50,814 and 50,422 cycles differ) |
| Equivalence | `part3_cec.py`: the study's register-cut ABC `cec` method, re-implemented (gold = RTL after the pre-mapping steps; gate = netlist with Liberty functions; latches matched by name) | 8 of 8 equivalent (INT4/ACC16 12 ns, INT4/ACC64 4 ns, INT8/ACC16 4 ns, INT8/ACC20 8 ns, INT6/ACC16 8 ns, INT8/ACC64 12 and 4 ns, INT6/ACC40 5 ns); 2 of 2 mutants NOT equivalent; about 5 min |

**Only read, not rerun:** all OpenSTA results (`opensta.txt`, `drv.txt`; no native OpenSTA, and the container is gone); the study's conformance run (`conformance-summary.json`); its `gatesim.py` and `cec.py` results; its determinism result; the figures, which were not regenerated.

## Part 4: is the hardware accumulator the software accumulator?

Sources: `public/generic_rtl/mac/integer_mac.sv` (lines 30 to 48), the contract 2.1 "Accumulator policies" and "Width convention", `tools/scaled_bridge_v2/kernel.h` (policy 2, lines 73 to 95), the certificates and exports of `resnet18-int8-default-b2`, `-int8-default_signed-b2` and `-int6-default-b2` (run 7c6344af, export 9ecfd612), and `part4_bias_units.py` → `part4-bias-units.json`.

| Aspect | RTL `integer_mac` | Engine `sat.w<W>` (B2 graphs) | Same? |
|---|---|---|---|
| Weights | signed W-bit, full range | int8 or int6 codes, signed | yes (RTL ⊇ engine) |
| Activations | signed A-bit, A = W in every synthesised point | `default`: `conv1` signed (int8 [-128, 127], int6 [-32, 31]); the other 20 nodes **unsigned** (int8.unsigned [0, 255], int6.unsigned [0, 63]). `default_signed`: all signed | INT8 signed: yes. INT8 and INT6 `default`: **no**, needs A = W+1 signed (9 or 7 bits) or an unsigned-mode port |
| Product | exact, W+A bits | exact (int64) | yes |
| Register unit | product of integer codes at scale 1 | product grid 2^-(input shift + weight shift); all shifts are 0 for these cases | yes |
| Width | ACC_BITS, sign included, [-2^(ACC-1), 2^(ACC-1)-1] | W, sign included, same range | yes |
| Start | `dot_start`: base 0 | from 0 | yes |
| Clamp | (ACC+1)-bit exact sum, clamped after every multiply-add | exact sum, clamped after every add | yes (also shown in simulation: RTL = kernel.h rule on all dots, including clamp-then-reverse) |
| Bias | added **inside** the ACC register at `dot_end` and **clamped again** | added outside, in binary64, after scaling; never clamped | **no** |
| Order | whatever the driver streams | input channel, kernel row, kernel column; padded taps add 0 | only if the driver uses the engine's order |
| Minimum width | ACC ≥ W+A-1 (elaboration) | 2 to 63 | for the widths used here (≥ 13), yes |
| Critical path | ends at `result`, after the bias stage | n/a | the measured period includes the bias stage |

**Bias in product units.** For the B2 exports, the stored bias code is bias / (s_in · s_w).
- INT8: live channels need up to 20 bits (`conv1`). Eight near-dead `conv1` channels (s_w < 1e-8, |bias| ≤ 3.5e-5) need up to 34 bits.
- INT8 signed: the same, 20 and 34 bits.
- INT6: 16 and 30 bits.

Live channels whose bias exceeds the register:

| Case | At W_acc(1.0) | At W_half |
|---|---|---|
| INT8 | 2 channels at 19 bits | 46 channels in 13 nodes at 17 bits |
| INT8 signed | 6 at 18 bits | 76 at 16 bits |
| INT6 | 2 at 15 bits | 112 at 13 bits |

On top of that, the RTL's second clamp acts whenever dot + bias leaves the range.

**Answer.**
- **INT8 `default`, INT6 `default`:** the cost of the RTL at width W is **not** the cost of `sat.w<W>`. The cost differs by the activation port (W+1 bits instead of W) and by the bias stage: a W-bit bias adder and clamp, plus the W-bit bias launch register in the wrapper. The numerics differ by the bias clamp.
- **INT8 `default_signed`:** the operand types match. Only the bias stage differs, together with the tap order of whatever driver is used.
- **RTL change that would close the gap** (not made; RTL work is on hold and `public/generic_rtl` is sealed):
  1. A parameter that removes the bias from the accumulator: no bias port, no bias adder and no second clamp, with `result <= mac_rounded` at `dot_end`. The bias moves to the output conversion at its own width.
  2. An activation-signedness parameter (unsigned codes zero-extended into a W+1-bit signed operand, or a sign-select input, since `conv1` is signed).
  3. Synthesis of W = 8 and 6 with A = 9 and 7.

  The driver must also stream taps in the engine's order. The flow sensitivity (part 2) shows that the synthesis recipe moves the per-bit cost by 32 % to 45 %. A variant should therefore be reported under both recipes, or the recipe should be fixed first.

## Part 5: the join (proposal; DERIVED and PRELIMINARY)

Script: `part5_join.py` → `part5-join.json` and `.csv`; `part5_flow_variant.py` → `part5-flow-variant.json`. Widths come from the reviewed ResNet18 sweep (B2, 1k screen, development evidence, one calibration draw). Areas are measured where the width was synthesised. Otherwise they come from the study's straight-line fit over the nine widths, with uncertainty given as the residual SD (8 ns: INT8 48.3, INT6 25.2) and the largest residual (89.9 and 42.5). The local interpolation between neighbouring widths agrees within 17 units. Periods are interpolated between the neighbouring measured widths.

| Case | Width | Bits | Core area 8 ns | Core area 12 ns | Smallest valid period (ns) |
|---|---|---:|---:|---:|---:|
| INT8 | certified (abs / struct) | 27 / 26 | 2,482 / 2,397 | 2,483 / 2,400 | 3.79 / 3.80 |
| INT8 | no event | 21 | 1,973 | 1,986 | 3.77 |
| INT8 | within 1 point | 19 | 1,804 | 1,821 | 3.74 |
| INT8 | half of exact (collapse) | 17 | 1,634 | 1,655 | 3.71 |
| INT8 signed | certified (abs / struct) | 26 / 25 | 2,397 / 2,313 | 2,400 / 2,318 | 3.80 / 3.80 |
| INT8 signed | no event / 1 point / half | 21 / 18 / 16 | 1,973 / 1,719 / 1,558.5 (measured) | 1,986 / 1,738 / 1,558.5 | 3.77 / 3.72 / 3.69 |
| INT6 | certified (abs / struct) | 23 / 22 | 1,819 / 1,745 | 1,821 / 1,746 | 3.79 / 3.77 |
| INT6 | no event | 17 | 1,372 | 1,375 | 3.63 |
| INT6 | within 1 point / half | 15 / 13 | 1,223 / 1,074 (**extrapolated** below 16) | 1,226 / 1,078 | ≤ 3.59 (16-bit value) |

**Savings from the certified (absolute) width:**

| Case | Bits saved | Saving at 8 ns, study flow | Saving at 8 ns, `-noalumacc` |
|---|---|---|---|
| INT8, to the 1-point width | 8 | 679 ± 68 (27.3 %); 662 ± 21 at 12 ns | 376 (19.7 %) |
| INT8, to the no-event width | 6 | 509 (20.5 %) | 282 (14.8 %) |
| INT8 signed, to the 1-point width | 8 | 679 (28.3 %) | 376 (20.2 %) |
| INT6, to the 1-point width | 8 | 596 ± 36 (32.8 %) | 365 (26.3 %) |
| INT6, to the no-event width | 6 | 447 (24.6 %) | 274 (19.7 %) |

**Period:** every width in this range meets the 4 ns target, the tightest one swept, so the 3.6 to 3.8 ns values are upper bounds. **No period saving can be claimed.**

**What the join depends on, plainly:**
1. The bias must be outside the accumulator, or zero. With the RTL's in-register bias, the arithmetic at W_acc(1.0) differs on 2 to 6 live `conv1` channels, and the critical path includes the bias stage.
2. For INT8 and INT6 `default`, unsigned activations need a W+1-bit signed port, which is not measured. Until that point is synthesised, only the INT8 `default_signed` row has matching operand types.
3. The hardware must stream taps in the engine's order, because saturation events depend on order.
4. The area per bit is specific to the flow (alumacc, part 2). The savings in percent move by 5 to 8 points between the two recipes.
5. These are a MAC core only (no scaling, conversion or memory), synthesis-level, at one corner, with no power.
6. The accuracy widths are 1k-screen development evidence for one network and one calibration draw.

MobileNet: `results/summaries/accumulator-sweep-mn-v1/` does not exist (checked 2026-10-02 19:20), so no MobileNet case is added.

## Commands

From `/home/maveric/precision_exploration`, with R=`artifacts/ppa/ics55-integer-accwidth-review-v1`:

```
nice -n 10 .venv/bin/python $R/part1_recompute.py
.venv/bin/python $R/rerun_synth.py 4            # 16 kept points -> $R/rerun/
.venv/bin/python $R/rerun_all_hashes.py 4       # 216 points, hashes only -> $R/rerun-all/hashes.json
cd $R && ../../../.venv/bin/python part3_gatesim.py 4    # -> gatesim/gatesim-result.json
cd $R && ../../../.venv/bin/python part3_cec.py 3        # -> cec/cec-result.json
cd $R && ../../../.venv/bin/python part2_flow_sensitivity.py 3   # -> flow/flow-sensitivity.json
.venv/bin/python $R/part4_bias_units.py         # -> part4-bias-units.json
.venv/bin/python $R/part5_join.py && .venv/bin/python $R/part5_flow_variant.py
```

Tools: `/home/maveric/workspace/oss-cad-suite/bin/` Yosys 0.53+67 (4f968c669) and Icarus 13.0 (devel), both identical to the study's version strings. SHA-256 of the RTL, harness, sweep script, matrix script, wrapper, Liberty and both LEFs all equal the summary's `hashes`.

## Limits

- OpenSTA was not rerun, so all timing numbers are checked only for internal consistency of the stored reports. `sta.log` is gone.
- Gate-level simulation covers 16 of 108 netlists and equivalence 8 of 108. The other netlists are covered by hash identity with the study's own 108/108 gate-level and `cec` results, which I did not rerun.
- Accumulator clamping by products is not reachable by simulation for ACC ≥ 32 (INT8) or ≥ 24 (INT4) within 70,000 cycles. There only the bias clamp is simulated, and `cec` covers the rest.
- The `-noalumacc` variant is area only. It has no STA, no function check, and its netlists were not kept.
- The part 4 bias analysis treats channels with s_w < 1e-8 as near-dead. Whether clamping their bias changes any prediction was not run.
- The join uses fits over 16 to 64 bits. INT6 at 15 and 13 bits is extrapolated.
- Disk: this review adds about 5 MB.
