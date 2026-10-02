# ICsprout55 integer MAC: timing-driven iso-clock matrix (2026-10-01)

Replaces the unconstrained-netlist timing of `ics55-integer-timing-2026-10-01.md` (whose critical paths all violate the library max-transition limit) with timing-driven netlists whose critical paths lie inside the characterised range of the library. No new or modified RTL: the same `public/generic_rtl/mac/integer_mac.sv` and `integer_mac_harness.sv` at their fixed repository paths.

Revision note: an independent review found that the first run of this matrix gave ABC an output load of 0.01 fF while the document said 0.01 pF (ABC's `set_load` is in fF). The matrix was rerun with the intended 10 fF (= 0.01 pF) and every number below comes from that rerun. The first run is kept, unchanged, in `artifacts/ppa/ics55-integer-isoclock-v1/superseded-abc-load-0.01fF/` as a sensitivity result.

## Result in brief
- All 64 points (8 configurations x 8 clock targets) have zero max-transition and zero max-capacitance violations anywhere in the design (the library defines no max_fanout, so that check is empty and means nothing); the worst slew on any critical path is 0.267 ns against the 0.796 ns limit. Every point is valid. The resizer was not needed (see Resizer).
- Smallest valid achieved period (clock target minus worst slack; ideal clock, typical corner, no wires). These are periods the netlist reaches while MISSING most tight targets, not targets it meets: INT4/ACC32 3.821 ns (first reached at the 4 ns target, slack +0.18, so it meets that target), INT5/ACC32 3.968 (4 ns target, slack +0.03, meets), INT6/ACC32 3.888 (the same netlist is first produced at the 4 ns target, which it meets with slack +0.11; tighter targets are missed), INT8/ACC32 4.164 (first produced at 4 ns, which it misses by 0.16; it meets 5 ns at 4.756), INT8/ACC64 6.444 (first produced at 6 ns, which it misses by 0.44; it meets 8 ns at 7.816). The stage-1 unconstrained values were 7.21, 6.93, 6.83, 7.21 and 11.93 ns, inflated by transition violations. The 32-bit versus 64-bit gap at INT8 is 2.3 ns here, not 4.7 ns.
- Below 3.8 to 4.2 ns (32-bit) and 6.4 ns (64-bit) ABC returns the same netlist whatever target it is given, so those periods are the limit of this mapping, not of the design.
- Targets that all five configurations meet validly: 12, 10 and 8 ns. At these targets the four 32-bit configurations are each one and the same netlist (achieved 5.5 to 5.8 ns, far below the target), so the multiplier ratio at 12, 10 and 8 ns is a ratio of area-optimal netlists, not of netlists squeezed to a clock. The only binding common target for the multiplier pair is 5 ns (INT4/ACC32 and INT8/ACC32 both meet it; 4 ns is missed by INT8); for the accumulator pair it is 8 ns (INT8/ACC64 misses 6 ns).
- Multiplier effect, INT4/ACC32 over INT8/ACC32: core area ratio 0.727 and total 0.759 at 12, 10 and 8 ns; 0.7266 / 0.7591 at 6 ns; 0.711 / 0.745 at 5 ns. That is -27.3 percent core, -24.1 percent total (-28.9 / -25.5 at the binding 5 ns target). Stage 1 implied 0.721 (-27.9 percent) core and 0.755 (-24.5 percent) total.
- Accumulator effect, INT8/ACC64 over INT8/ACC32: core 1.892 and total 1.869 at 12 ns (1.892 / 1.869 at 10 ns), core 1.948 and total 1.915 at the binding 8 ns target; i.e. +89.2 to +94.8 percent core and +86.9 to +91.5 percent total. Stage 1 implied +88.7 percent core (1.887) and +86.4 percent total (1.864).
- The accumulator effect is not the same at every operand width (supplementary): core 1.965 to 2.023 at INT4, INT5 and INT6, against 1.892 to 1.948 at INT8. It is about +89 to +102 percent across all of them.
- Area compared with the unconstrained stage-1 netlists: the timing-driven netlists at loose targets are different netlists and are 4.7 to 6.2 percent larger (core: INT4/ACC32 2130 vs 2006, INT8/ACC32 2931 vs 2782, INT8/ACC64 5545 vs 5249; total 2717 vs 2581, 3579 vs 3417, 6688 vs 6368). They are not the min-area netlist. The multiplier ratio and the 12/10 ns accumulator ratio agree with stage 1 to within about 1 point; the accumulator ratio at the binding 8 ns target does not (1.948 against 1.887).
- Sensitivity to the ABC output load (see Constraints): with the 0.01 fF actually used in the superseded first run the multiplier ratio is 0.7199 core / 0.7539 total and the accumulator ratio 1.8892 / 1.8657 at 12 ns and 1.9266 / 1.8961 at 8 ns (that run: summary in `superseded-abc-load-0.01fF/summary.json`). The conclusion is robust to this choice: multiplier 0.72 to 0.73 core, accumulator 1.89 to 1.95 core.

## Tools and provenance
- Image `openroad/orfs@sha256:6da005d1c3447799f7401215174196c75ba9ec7e28417624d0c20225f85d594c` (same digest as stage 1). Tool versions queried from the tool through the wrapper: `OpenSTA 3.1.0` (`sta::version`), `OpenROAD unknown` (`ord::openroad_version`, no git describe). Host Yosys `0.53+67 (git sha1 4f968c669, clang++ 18.1.8 -fPIC -O3)`, whose bundled ABC is invoked by `abc`.
- `tools/hardware/openroad_docker.sh` (sha256 `99c2ba0d13692b91de404caafe059f56d3715bda425b04935dc6969fdcf9d5a1`): runs openroad from the pinned image as the host user (uid/gid of the caller, `HOME=/tmp` inside the container), no network, repository mounted read-write and the PDK read-only at identical absolute paths, every argument forwarded unchanged (paths with spaces work), exit status propagated, version printed by `--version` from the tool. It writes nothing outside the repository (the `--version` temporary file is created in the working directory and removed). Tested earlier in this lane: a script path with a space, an environment variable with a space, `exit 7` returns 7, a missing script returns 1.
- `tools/hardware/integer_isoclock_matrix.py` (sha256 `7c4e45adcbc078e409d8521b41163d8afdf15504586e63b29b35ea641642c837`, recorded in the summary; any edit requires a rerun of the matrix) imports `RTL`, `TOP`, `digest` and `version` from `public/pdk_flow/ics55/validate_integer.py`. It does not use `run_flow.run`, because that function cannot pass a constraint file to ABC, and it does not use `opensta_matrix.tcl`, because the validity check needs `report_check_types`; its Tcl repeats the stage-1 constraints and the same `remove_from_collection` shim.
- Library `ics55_LLSC_H7CR_typ_tt_1p2_25_nldm.lib` sha256 `38a9f9056a7e4417451e0aaa137efaf0461ee1fc574a8b2ad53622def34bb2b7`; tech LEF `c33b8b7d4ce1d105735e461f31eaa80256c51cc29cc3fa2512ca514d125fe31f`; cells LEF `6e8a3b88db7427269f683f5a321bdf295c41092c36d8a770926751103be13a0c`. RTL hashes are in the summary JSON.
- Conformance gate: the script refuses to synthesise any configuration without a `pass` entry in `results/summaries/integer-mac-conformance-v1.json`. All eight configurations used here have one. That is a statement about the RTL; the mapped netlists are covered by the separate checks in the next section.

## Function of the mapped netlists (independent review checks)
The matrix tool itself does not check that a mapped netlist still computes the MAC. The independent reviewer did, twice, with scripts under `artifacts/ppa/ics55-integer-isoclock-v1/review/` (round 1 on the superseded 0.01 fF netlists, round 2 on the 10 fF netlists reported here: 29 distinct netlists among the 64 points, 4 per main configuration and 3 per supplementary one):
- Simulation: the existing conformance-harness vectors, run on the mapped netlists with the PDK Verilog cell models, pass for 29 of 29 (864 to 135818 cycles); a one-cell mutant of each fails; every operand pair (all 2^(2W)) at random dot lengths and random or extreme biases, against the reviewer's own Python model, passes for 29 of 29.
- Formal: register-cut ABC `cec` (registers matched one to one by name, every next-state and output bit) against a gold model built from the RTL with the matrix's own pre-mapping Yosys steps: 29 of 29 equivalent, including all 8 INT8 netlists and the INT4/5/6 ACC64 ones; four one-cell mutants are non-equivalent. This shows that technology mapping, buffering and sizing changed no logic relative to Yosys's generic netlist. Against a second gold (flatten before synth) 21 of 21 INT4/5/6 netlists are equivalent; two INT8 netlists tried were undecided and were stopped.
- Limits: both gold models rely on Yosys's reading of the RTL (the iverilog simulation is the independent check of that); ACC64 accumulator saturation is not reached in simulation but is covered by the formal check.

## Commands
```
cd /home/maveric/precision_exploration
.venv/bin/python /home/maveric/precision_exploration/tools/hardware/integer_isoclock_matrix.py --workers 8 \
  > artifacts/ppa/ics55-integer-isoclock-v1/run.log
# defaults: --out-dir artifacts/ppa/ics55-integer-isoclock-v1
#           --summary results/summaries/ics55-integer-isoclock-v1.json
#           --table   results/tables/ics55-integer-isoclock-v1.csv
# It refuses to overwrite an existing summary, table, or point directory; Yosys/ABC scratch files go to <out-dir>/tmp.
.venv/bin/python tools/hardware/integer_isoclock_matrix.py --workers 8 \
  --out-dir artifacts/ppa/ics55-integer-isoclock-v1/determinism-rerun/points \
  --summary artifacts/ppa/ics55-integer-isoclock-v1/determinism-rerun/summary.json \
  --table artifacts/ppa/ics55-integer-isoclock-v1/determinism-rerun/table.csv
.venv/bin/python artifacts/ppa/ics55-integer-isoclock-v1/determinism.py
.venv/bin/python -m pytest -q tests/unit/test_integer_isoclock_matrix.py
/home/maveric/precision_exploration/tools/hardware/openroad_docker.sh --version
```
Per point the script writes `synthesis.ys`, `abc.constr`, `netlist.v`, `yosys-stat.json`, `yosys.log`, `sta.tcl`, `opensta.txt`, `drv.txt`, `sta.log` under `artifacts/ppa/ics55-integer-isoclock-v1/int<W>-acc<A>/t<T>ns/`. The 64 points run in about 100 s on 8 workers.

## Constraints and units
- Synthesis: `read_verilog -sv` of the two RTL files by absolute path, `chparam W_BITS=A_BITS=<W>, ACC_BITS=<A>`, `hierarchy`, `proc`, `opt`, `synth -top integer_mac_harness`, `dfflibmap -liberty`, then `abc -liberty <lib> -constr abc.constr -D <ps>`, `opt_clean`. With `-constr` and `-D`, Yosys runs ABC `&nf -D <ps>; buffer; upsize -D <ps>; dnsize -D <ps>` (visible in each `yosys.log`). ABC's `-D` is in picoseconds; the liberty time unit is 1 ns and capacitance 1 pF.
- `abc.constr`: `set_driving_cell BUFX2H7R` (a mid-strength library buffer, a typical register-output strength) and `set_load 10`. ABC's `set_load` is in femtofarads, not in the liberty unit: ABC logs `Setting output load to be 10.000000` and shows `Cout = 10.0 ff` on output gates (verified by the reviewer by experiment). 10 fF equals the 0.01 pF STA output load as a number only: ABC puts the 10 fF on every combinational output of each module, and every one of those drives a register D pin (about 1 fF), while the harness outputs are driven directly by flops (DFFQX1H7R), which ABC does not size. So no net that ABC loads with 10 fF carries the 0.01 pF STA load. The effect is extra buffers (INT8/ACC32 142 against 78 at 12 ns, INT4/ACC32 119 against 54, INT8/ACC64 280 against 151) and 2.1 to 2.8 percent more total area than the 0.01 fF run; the ratios move little (multiplier core 0.7199 to 0.7268, accumulator core 1.8892 to 1.8920 at 12 ns and 1.9266 to 1.9484 at 8 ns). For the paper quote the range (multiplier 0.72 to 0.73, accumulator 1.89 to 1.95 core) rather than one value. The script checks that this log line is present.
- ABC delay target: `-D = target_ns * 1000 - 450` ps. ABC times only combinational logic; the clock period also pays clk-to-Q (about 0.13 ns), setup (about 0.04 ns), the 0.10 ns uncertainty and on output paths the 0.20 ns output delay. The worst class (register to output: 0.13 + 0.20 + 0.10) rounded up gives 450 ps; it is an estimate, and slack always comes from OpenSTA. For 12, 10, 8, 6, 5, 4, 3, 2 ns that is 11550, 9550, 7550, 5550, 4550, 3550, 2550, 1550 ps.
- STA (OpenSTA via openroad, as stage 1): clock `clk` with period equal to the target in ns, uncertainty 0.10 ns, input delay 0.20 ns on all inputs except `clk`, output delay 0.20 ns, 0.01 pF load on every output, ideal clock, no wire parasitics, typical corner (tt, 1.2 V, 25 C). The 84 data inputs have no driving cell or input transition (ideal slew), as in stage 1; the worst paths are register to register. `report_checks -path_delay max -group_path_count 1` gives the worst path; `report_check_types -max_slew -max_capacitance -max_fanout -violators` lists violations. `check_setup -verbose` is empty in all 64 reports and all endpoints are constrained.
- Achieved period = target minus worst setup slack (the path delay does not depend on the clock; confirmed against stage 1: 10 - 2.793602 = 7.206398).
- Calibration of the STA script: run on the stage-1 INT4/ACC32 netlist at 10 ns it reproduces slack +2.793602, 49 path cells, worst path slew 1.0386 ns, and 101 max-slew plus 4 max-capacitance violating pins (the stage-1 reviewer's numbers). Evidence: `artifacts/ppa/ics55-integer-isoclock-v1/calibration-stage1-int4-acc32-10ns/` (made with the first version of the script; the STA Tcl has not changed since).

## Validity rule
A point is valid only if no pin on its reported worst setup path appears in the `report_check_types` violator list for max transition, max capacitance or max fanout (so its delay lies within the characterised NLDM range). A point "meets timing validly" when it is valid and its worst slack is >= 0. The design-wide violation count is also recorded. Here the two coincide: no point has any violation. The max_fanout part of the rule is vacuous because the library has no max_fanout attribute. The reviewer recounted violations with separate Tcl on eight points (0 slew, 0 capacitance violators anywhere, worst slew 0.19 to 0.27 ns) and confirmed that the same Tcl finds 967 and 12 on the stage-1 INT8/ACC64 netlist.

## Table
Each cell: core area / total area (library units, Yosys `stat -liberty`; total includes the neutral launch/capture wrapper), achieved period, worst slack in ns. "miss" means the target is not met (the achieved period is then above the target). Raw rows: `results/tables/ics55-integer-isoclock-v1.csv`; full records and curves: `results/summaries/ics55-integer-isoclock-v1.json`.

| Config | 12 ns | 10 ns | 8 ns | 6 ns | 5 ns | 4 ns | 3 ns | 2 ns |
|---|---|---|---|---|---|---|---|---|
| INT4/ACC32 | 2130 / 2717<br>5.71 ns<br>+6.29 | 2130 / 2717<br>5.71 ns<br>+4.29 | 2130 / 2717<br>5.71 ns<br>+2.29 | 2130 / 2717<br>5.41 ns<br>+0.59 | 2161 / 2748<br>4.79 ns<br>+0.21 | 2325 / 2912<br>3.82 ns<br>+0.18 | 2325 / 2912<br>3.82 ns<br>-0.82 (miss) | 2325 / 2912<br>3.82 ns<br>-1.82 (miss) |
| INT5/ACC32 | 2386 / 2988<br>5.53 ns<br>+6.47 | 2386 / 2988<br>5.53 ns<br>+4.47 | 2386 / 2988<br>5.53 ns<br>+2.47 | 2386 / 2988<br>5.31 ns<br>+0.69 | 2409 / 3011<br>4.76 ns<br>+0.24 | 2573 / 3175<br>3.97 ns<br>+0.03 | 2573 / 3175<br>3.97 ns<br>-0.97 (miss) | 2573 / 3175<br>3.97 ns<br>-1.97 (miss) |
| INT6/ACC32 | 2490 / 3107<br>5.73 ns<br>+6.27 | 2490 / 3107<br>5.73 ns<br>+4.27 | 2490 / 3107<br>5.73 ns<br>+2.27 | 2490 / 3107<br>5.53 ns<br>+0.47 | 2552 / 3170<br>4.80 ns<br>+0.20 | 2677 / 3294<br>3.89 ns<br>+0.11 | 2677 / 3294<br>3.89 ns<br>-0.89 (miss) | 2677 / 3294<br>3.89 ns<br>-1.89 (miss) |
| INT8/ACC32 | 2931 / 3579<br>5.78 ns<br>+6.22 | 2931 / 3579<br>5.78 ns<br>+4.22 | 2931 / 3579<br>5.78 ns<br>+2.22 | 2931 / 3579<br>5.43 ns<br>+0.57 | 3041 / 3689<br>4.76 ns<br>+0.24 | 3089 / 3737<br>4.16 ns<br>-0.16 (miss) | 3089 / 3737<br>4.16 ns<br>-1.16 (miss) | 3089 / 3737<br>4.16 ns<br>-2.16 (miss) |
| INT8/ACC64 | 5545 / 6688<br>9.78 ns<br>+2.22 | 5545 / 6688<br>9.07 ns<br>+0.93 | 5711 / 6854<br>7.82 ns<br>+0.18 | 5996 / 7138<br>6.44 ns<br>-0.44 (miss) | 5996 / 7138<br>6.44 ns<br>-1.44 (miss) | 5996 / 7138<br>6.44 ns<br>-2.44 (miss) | 5996 / 7138<br>6.44 ns<br>-3.44 (miss) | 5996 / 7138<br>6.44 ns<br>-4.44 (miss) |

Area is not monotone in the target everywhere, because ABC is a heuristic and a looser target does not guarantee a smaller or equal netlist. Examples: INT4/ACC32 at 6 ns (2716.84) is smaller and faster than at 8 ns (2717.12); INT8/ACC64 at 12 ns (6688.08, 9.78 ns) is larger and slower than at 10 ns (6687.80, 9.07 ns); INT4/ACC32 at 4 ns equals its 3 ns netlist. These are mapping effects, not design effects, which is why the curves below are Pareto-filtered (a point is dropped if another point has no larger period and no larger area and is strictly better in one).

Area-delay curve per configuration, Pareto-filtered over valid points (lowest target at which that netlist is first produced -> achieved period in ns, total area); the unfiltered curve is in the `curves` field of the summary JSON. A point whose period exceeds its target is a missed target. (The summary JSON fields `smallest_valid_at_target_ns`, `smallest_valid_point_meets_its_target` and the duplicates in `pareto_front` were computed by exact comparison of periods printed to six decimals, so a one-in-a-million difference in the printout, for example 3.887763 against 3.887764 for one netlist, makes them name a later target and mark INT6/ACC32 as not meeting its target; the table here is by netlist and is correct. Fixing the script would need a full rerun and was not done; use the CSV with netlist hashes.)

| Config | Pareto points: target ns -> achieved period ns, total area |
|---|---|
| INT4/ACC32 | 4 -> 3.821, 2912.28; 5 -> 4.787, 2747.92; 6 -> 5.411, 2716.84 |
| INT5/ACC32 | 4 -> 3.968, 3174.92; 5 -> 4.762, 3010.56; 6 -> 5.308, 2987.60 |
| INT6/ACC32 | 4 -> 3.888, 3294.20; 5 -> 4.801, 3169.60; 6 -> 5.529, 3107.44 |
| INT8/ACC32 | 4 -> 4.164, 3736.88; 5 -> 4.756, 3689.28; 6 -> 5.426, 3579.24; 10 -> 5.785, 3578.96 |
| INT8/ACC64 | 6 -> 6.444, 7138.32; 8 -> 7.816, 6853.56; 10 -> 9.066, 6687.80 |
| INT4/ACC64 | 6 -> 6.105, 5621.28; 8 -> 7.756, 5360.04; 10 -> 9.009, 5298.16 |
| INT5/ACC64 | 6 -> 6.675, 6262.76; 8 -> 7.807, 5923.12; 10 -> 9.544, 5858.44 |
| INT6/ACC64 | 6 -> 6.261, 6344.80; 8 -> 7.758, 6026.44; 10 -> 9.343, 6004.88 |

Smallest valid achieved period per configuration is listed in the brief above, each labelled as met or missed. Differences of 0.1 to 0.3 ns between INT4, INT5, INT6 and INT8 at 32 bits are within mapping noise and are not a width trend; the accumulator width is the visible effect.

## Iso-clock ratios
Taken at targets where both configurations meet validly (all five meet at 12, 10, 8 ns).

| Ratio | 12 ns | 10 ns | 8 ns | 6 ns | 5 ns |
|---|---|---|---|---|---|
| INT4/ACC32 over INT8/ACC32, core | 0.7268 | 0.7268 | 0.7268 | 0.7266 | 0.7106 |
| INT4/ACC32 over INT8/ACC32, total | 0.7592 | 0.7592 | 0.7592 | 0.7591 | 0.7448 |
| INT8/ACC64 over INT8/ACC32, core | 1.8920 | 1.8919 | 1.9484 | - | - |
| INT8/ACC64 over INT8/ACC32, total | 1.8687 | 1.8686 | 1.9150 | - | - |

Stage 1 (unconstrained) for comparison, from its summary JSON: INT4/INT8 core 2006.48 / 2782.08 = 0.721, total 2581.04 / 3417.12 = 0.755; ACC64/ACC32 core 5249.16 / 2782.08 = 1.887, total 6368.04 / 3417.12 = 1.864. The "28 percent and +89 percent" of stage 1 are the core figures; at iso-clock they become -27.3 percent (core) / -24.1 percent (total) for the multiplier effect (-28.9 / -25.5 at the binding 5 ns target), and +89.2 to +94.8 percent (core) / +86.9 to +91.5 percent (total) for the accumulator effect, depending on the target. Total area includes a wrapper whose launch and capture registers scale with the accumulator width (bias and result), so the core ratio is the cleaner measure of the accumulator itself.

Supplementary, accumulator effect at other operand widths (conformance-verified INT4, INT5, INT6 with ACC64; not part of the headline), core ratio ACC64/ACC32 at 12 / 10 / 8 ns: INT4 1.980 / 1.980 / 2.009, INT5 1.996 / 1.996 / 2.023, INT6 1.965 / 1.965 / 1.974; total ratio 1.950 / 1.950 / 1.973, 1.961 / 1.961 / 1.983, 1.932 / 1.932 / 1.939. So the accumulator effect is about +96 to +102 percent core at INT4 to INT6 and +89 to +95 percent at INT8: it depends on the operand width. Achieved periods and areas:

| Config | 12 ns | 10 ns | 8 ns | 6 ns | 5 ns | 4 ns | 3 ns | 2 ns |
|---|---|---|---|---|---|---|---|---|
| INT4/ACC64 | 4217 / 5298<br>9.01 ns<br>+2.99 | 4217 / 5298<br>9.01 ns<br>+0.99 | 4279 / 5360<br>7.76 ns<br>+0.24 | 4540 / 5621<br>6.11 ns<br>-0.11 (miss) | 4540 / 5621<br>6.11 ns<br>-1.11 (miss) | 4540 / 5621<br>6.11 ns<br>-2.11 (miss) | 4540 / 5621<br>6.11 ns<br>-3.11 (miss) | 4540 / 5621<br>6.11 ns<br>-4.11 (miss) |
| INT5/ACC64 | 4762 / 5858<br>9.54 ns<br>+2.46 | 4762 / 5858<br>9.54 ns<br>+0.46 | 4827 / 5923<br>7.81 ns<br>+0.19 | 5167 / 6263<br>6.68 ns<br>-0.68 (miss) | 5167 / 6263<br>6.68 ns<br>-1.68 (miss) | 5167 / 6263<br>6.68 ns<br>-2.68 (miss) | 5167 / 6263<br>6.68 ns<br>-3.68 (miss) | 5167 / 6263<br>6.68 ns<br>-4.68 (miss) |
| INT6/ACC64 | 4893 / 6005<br>9.34 ns<br>+2.66 | 4893 / 6005<br>9.34 ns<br>+0.66 | 4915 / 6026<br>7.76 ns<br>+0.24 | 5233 / 6345<br>6.26 ns<br>-0.26 (miss) | 5233 / 6345<br>6.26 ns<br>-1.26 (miss) | 5233 / 6345<br>6.26 ns<br>-2.26 (miss) | 5233 / 6345<br>6.26 ns<br>-3.26 (miss) | 5233 / 6345<br>6.26 ns<br>-4.26 (miss) |

Intermediate accumulator widths (16 to 64 bits) were not run: the existing conformance tool only supports the `int32_accumulator` and `int64_accumulator` manifests (`public/formats/manifests/accumulators/`), so no other width can be verified without modifying existing files, and no unverified configuration was synthesised.

## Resizer
Not needed: ABC alone left no violation in any of the 64 points, so no resizer variant was run in the matrix (the summary field `resizer` is null throughout). Probe of what the resizer can do on an unplaced netlist, on the stage-1 INT4/ACC32 netlist at a 5 ns target (`artifacts/ppa/ics55-integer-isoclock-v1/probe-resizer-stage1-int4-acc32/`): `repair_design` runs without placement ("no estimated parasitics, using wire load models") and resized 3 instances, cutting max-slew violations from 101 to 0 but leaving 3 max-capacitance violations (2 on the critical path), so that variant is still invalid; its achieved period is 6.90 ns versus 7.21 ns. `repair_timing -setup` does not run: it stops with `RSZ-0089` because no wire RC value exists (the ICS55 collateral in this tree provides none) and a zero RC is rejected. The probe therefore labels that variant `abc+repair_design`. I did not invent an RC value.

## Determinism
The whole matrix was rerun into `artifacts/ppa/ics55-integer-isoclock-v1/determinism-rerun/` with the same script: for all 64 points `netlist.v`, `yosys-stat.json`, `opensta.txt`, `drv.txt` and `abc.constr` are byte-identical to the original run, and the summary JSON and CSV are byte-identical too (322 files compared, 0 differ; `determinism-result.json`). Only `synthesis.ys` and logs differ, because they embed the output directory. The RTL source paths are kept fixed at `/home/maveric/precision_exploration/public/generic_rtl/mac/`, since netlist bytes depend on them (see the stage-1 limits). The reviewer's independent rerun of the first-run matrix was also byte-identical.

## Limits
- Mapped netlists only: no placement, routing or clock tree; ideal clock with 0.10 ns uncertainty; no wire parasitics (so absolute delays are optimistic on that count); typical corner only; no power or energy.
- Yosys and ABC do not choose datapath architectures: the multiplier and accumulator are whatever `synth` infers (ripple-style carry chains), so absolute periods are conservative compared with a hand-built or Design-Compiler-style implementation. The usable results are the ratios, which compare the same flow on the same RTL.
- ABC sees each module separately and times only combinational logic between registers; the 450 ps overhead and the BUFX2H7R driving cell are fixed estimates, not a closed-loop result. The ABC output load (10 fF) is a choice; the sensitivity to it is given above.
- ABC's delay target hits a floor (identical netlists for tighter targets), so the matrix gives only 3 to 4 distinct netlists per configuration (29 in total), and the 12, 10 and 8 ns points of the 32-bit designs are one netlist each; ratios there compare area-optimal netlists.
- Inputs have ideal slew and no driving cell in STA.
- Validity concerns the reported worst path and the design-wide counts; it does not make the Liberty model of a short path accurate beyond what NLDM gives.
- Function of the reported mapped netlists was checked by the independent reviewer, not by the matrix tool (see the function section).
- Process notes: to reuse the v1 output names the first run (64 point directories, summary, CSV) was moved, unchanged, into `superseded-abc-load-0.01fF/`, so `ics55-integer-isoclock-v1.json/.csv` changed content under the same name (the review compared all 320 raw files with its own rerun: 0 differ). `yosys` is taken from PATH (/home/maveric/workspace/oss-cad-suite/bin/yosys, 0.53+67, recorded in the summary); the path is not pinned. `openroad_docker.sh --version` creates and removes a temporary file `.openroad_version.XXXXXX` in the caller's working directory.
- Raw reports sit under the git-ignored `artifacts/`; only hashes are in the summary.
- Out of scope and untouched: new RTL, placement and routing, power, and the three open findings of `hardware-followup-review-2026-09-27.md`.
