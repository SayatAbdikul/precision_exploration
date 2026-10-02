# MobileNet accumulator-width sweep (lane Q1), 2026-10-02

Development evidence only: ImageNet screen-1k list (images 0-999; location phase 0-127). No held-out image was read.
Protocol `public/experiments/configs/breadth-study/accumulator-sweep-mn-protocol-v1.json` (sha256 c7b76f02...) with addenda 1-5
(3: switch to lane S1's fast path, 17:09:27; 4: archive runs in slot jobs, 17:26:49; 5: K4 bookkeeping fix, 17:52:01).
Summaries (written once, 18:36): `results/summaries/accumulator-sweep-mn-v1/` (summary.json, widths, policies, node_events, kind_events,
location_128, simulator CSVs); figure `results/figures/accumulator-sweep-mn-top1-vs-width-v1.{png,pdf}`.

Revision r4 (2026-10-02, after review 1 in `artifacts/agent_orchestration/handoffs/Q1-mn-accsweep-review.md`): text corrections
only. The W_half definition, the per-kind slack ranges, the sat.struct-1 sentence, the engine time row, the simulator agreement counts
and four rounded differences are corrected; the [Agg24] formula and the trace_gates.py result are added; sections renumbered 1-6.
No summary, table value or prediction file changed (every number below was re-read from the written-once summaries).

Revision r5 (2026-10-02, after review 2 in the same file): text corrections only. The float-accumulator result is now stated under
both tie rules (two non-stress intervals exclude 0 under lowest index); the process-stopping sentence is corrected (agent r1 killed its
own runners twice, the owner stopped a third); the stress case MobileNetV3-L int6 is now left out of every across-case summary, as the
protocol's stress_cases entry says (its own values are given beside them); the reviewer record count, the fast minimum rate, the
first-saturating nodes of MobileNetV3-L fp8 and the archive's transient kernel error in section 6 are corrected or added. No summary,
table value or prediction file changed; nothing was rerun.

## 1. Numbers first (all 10 cases complete; values from results/summaries/accumulator-sweep-mn-v1/widths.csv)

Development evidence, ImageNet screen-1k (1,000 images), expected-credit Top-1. Widths in bits of a saturating two's-complement
register (`sat.w<W>`, uniform over all MAC nodes), sign bit included. Definitions as in the protocol (L8's `score.py derived`):
W_cert = statically certified lossless width (absolute, network maximum); W_noevent = narrowest measured width from which upward no
saturation event occurs on any image; W_acc(1.0) = narrowest width from which upward Top-1 is at least the exact (wide) arm minus
1.0 pp; W_half = **widest** width at which Top-1 is **at most half** of the exact arm (so W_half + 1 is above half; e.g.
MobileNetV2 int8: 17 bits 0.30 %, 18 bits 49.3 %). Slack = W_cert - W_acc(1.0).
[Agg24] formula = the published closed-form width as lane L8 applied it: integer 2r + ceil(log2 n) + 1, minifloat ExMy
2(2^e + m) + ceil(log2 n) - 1, with n = the largest fan-in K of the network (1,280, the classifier, in both MobileNets; the formula
grows with n only). Caveat (L8's, still open): we assume the formula counts the sign bit as the certificates do; whether the
paper's convention is the same, and whether its integer r includes a sign bit for unsigned activations, is not settled, so the
[Agg24] column may be one bit off. Under that assumption it is 3-4 bits above W_cert here (3-5 on ResNet18 at n = 4,608).

| case | exact Top-1 % | W_cert | W_noevent | W_acc(1.0) | W_half | slack | [Agg24] formula | per-node d at half |
|---|---|---|---|---|---|---|---|---|
| MobileNetV2 int8 | 72.55 | 25 | 20 | 19 | 17 | 6 | 28 | 2 |
| MobileNetV2 fp6_e2m3 | 66.33 | 21 | 17 | 15 | 14 | 6 | 24 | 2 |
| MobileNetV2 fp7_e3m3 | 66.77 | 29 | 24 | 23 | 21 | 6 | 32 | 2 |
| MobileNetV2 fp8_e4m3fn | 67.41 | 45 | 40 | 39 | 37 | 6 | 48 | 2 |
| MobileNetV3-L int8 | 72.00 | 24 | 20 | 19 | 17 | 5 | 28 | 3 |
| MobileNetV3-L int6 (stress, reduced grid; not in across-case summaries) | 20.50 | 20 | 16 | 14 | 12 | 6 | 24 | n/a |
| MobileNetV3-L fp6_e2m3 | 57.86 | 21 | 17 | 17 | 14 | 4 | 24 | 3 |
| MobileNetV3-L fp8_e4m3fn | 71.38 | 45 | 40 | 39 | 37 | 6 | 48 | 4 |
| MobileNetV2 int6 | 69.71 | 21 | 16 | 15 | 13 | 6 | 24 | 2 |
| MobileNetV3-L fp7_e3m3 | 72.23 | 29 | 24 | 23 | 21 | 6 | 32 | 4 |
| ResNet18 (lane L8) int8 / int6 / fp6 / fp7 / fp8 | | 27 / 23 / 23 / 31 / 47 | 21 / 17 / 19 / 26 / 42 | 19 / 15 / 17 / 23 / 39 | 17 / 13 / 14 / 21 / 37 | 8 / 8 / 6 / 8 / 8 | | |

Across-case statements below use the nine non-stress cases. The protocol's stress_cases entry leaves MobileNetV3-L int6 out of the
slack summary across cases; its node-level 'needed' widths rest on the reduced grid (15 and 13 bits unmeasured), so its node slack can
be one bit optimistic. Its own values are given in brackets.

- The transition is sharp on both MobileNets: from the first width with events (W_noevent - 1) to the widest width at or below half of
  the exact arm takes 2 bits in all nine non-stress cases (stress case: 3 bits on its reduced grid). E.g. MobileNetV2 int8 at 1k:
  20 bits no event; 19 bits events on 352 of 1,000 images, Top-1 unchanged at 72.55 % (2 class changes); 18 bits 49.3 %; 17 bits 0.3 %.
- Slack is 4-6 bits on MobileNets (6 in 7 of the 9 non-stress cases; stress case 6) against 6-8 on ResNet18: the certificate is closer
  to what MobileNets need.
- Per node (summary.json `governing`), the slack (certified node width minus the narrowest width without events at that node) follows
  fan-in: Spearman rho(node slack, log2 K) = 0.90-0.96 over the nine non-stress cases (stress case 0.84). Ranges over the nine cases,
  min-max (range of the per-case medians): depthwise (K = 9 or 25) 0-4 (0-2); stem 0-1; pointwise expand 1-5 (2-3); pointwise project
  0-9 (4-5); squeeze-excite reduce 2-8 (4.5-6) and expand 2-5 (3-4) on MobileNetV3 (4 non-stress cases); classifier (K = 1,280;
  MobileNetV3 also its K = 960 layer) 4-7 (4-6). Stress case, same order: 0-2, 0, 1-4, 2-7, 2-6, 1-4, 5-6; only its SE-expand minimum
  (1) lies outside the non-stress ranges. Node kind matters mainly through K. The needed width itself is only weakly related to the
  node's certified width: rho(needed, W_node) = -0.07 to 0.56 over the nine (MobileNetV2 int6 0.56, int8 0.45; all non-stress
  MobileNetV3 cases <= 0.26; stress case 0.14). A few nodes never saw an event in the measured range and are censored (one depthwise
  and one squeeze-excite expand node in each non-stress MobileNetV3 case; one SE-expand node in the stress case). The first nodes to
  saturate (events at W_noevent - 1) are the classifier and/or pointwise convolutions in the int8, int6 and fp6 cases and in
  MobileNetV3 fp8 (classifier and 2 pointwise expand), plus depthwise nodes in MobileNetV2 fp7/fp8 (3 each) and MobileNetV3 fp7
  (1, with the classifier and 3 pointwise expand). [Stress case: 5 nodes, with SE nodes and the stem.]
- Per-node rule sat.struct-<d> (every node at its own certified width minus d), 1k: d = 1 costs -0.13 to -2.89 pp on MobileNetV2
  (int8 -0.13 [-0.57, +0.27]; int6 -2.89 [-4.44, -1.34]; fp6 -2.14; fp7 -1.64; fp8 -2.59) with events on all 1,000 images, and 0.00 /
  -0.25 pp on MobileNetV3 int8 / fp6 with events on 308 / 434 images (d = 1 was not in the 1k set for MobileNetV3 fp7/fp8). d = 2
  halves MobileNetV2 (int8 28.5 %, others 7.9-10.2 %). MobileNetV3 holds longer: int8 -0.55 pp at d = 2 and 27.6 % at d = 3; fp6
  -4.2 pp at d = 2 and 4.9 % at d = 3; fp7 / fp8 -0.10 / +0.27 pp at d = 2, -1.03 / -0.43 pp at d = 3 and 18.0 / 16.3 % at d = 4.
- Float accumulators at the rule exponent (fp16 rule policy and f21) against wide, 18 comparisons in the nine non-stress cases.
  Expected credit: -0.97 to +0.81 pp, every interval includes 0. Lowest index: -1.5 to +1.3 pp, and 2 of 18 intervals exclude 0, in
  opposite directions, both the fp16.x-11 rule policy of an fp8 case: MobileNetV2 fp8 -1.5 [-2.8, -0.2] (McNemar p = 0.036; 15
  policy-only vs 30 wide-only correct) and MobileNetV3-L fp8 +1.3 [+0.2, +2.5] (p = 0.047; 25 vs 12); their expected-credit
  differences are -0.69 [-1.68, +0.27] and +0.81 [-0.16, +1.76]. No multiplicity correction is applied and no claim is made from these
  two. Stress case MobileNetV3-L int6 (20.5 %): fp16.x-4 +0.36 [-1.64, +2.31] expected / +0.2 [-2.1, +2.5] lowest index; f21 +1.36
  [+0.21, +2.53] expected / +1.8 [+0.3, +3.3] lowest index (p = 0.025; 38 vs 20). 0 failed images, 0 events; Top-1 class changes on 14-279 images (715 for the stress
  fp16 run). Control (binary32 FMA) = wide except MobileNetV3-L fp8 (3 Top-1 changes, -0.03 pp, 53 changed outputs) and MobileNetV2
  fp8 (+0.05 pp expected credit from ties, 0 lowest-index changes, 63 changed outputs).
- Exact engine against the B2 simulator at 1k (wide arm vs lane L1's sealed readout of the same configuration identity), exact - B2,
  pp [95 %], expected credit / lowest index (exact-only vs B2-only correct):
  V2 int8 +0.25 [-0.05, +0.60] / +0.4 [0.0, +0.9] (5 vs 1); int6 -0.39 [-1.10, +0.30] / -0.5 [-1.5, +0.5] (11 vs 16);
  fp6 +0.09 [-0.77, +0.95] / +0.4 [-0.8, +1.6] (21 vs 17); fp7 -0.86 [-1.65, -0.08] / -1.2 [-2.3, -0.1] (9 vs 21);
  fp8 -0.30 [-1.11, +0.55] / -0.6 [-1.7, +0.5] (14 vs 20);
  V3 int8 +0.40 [-0.70, +1.55] / +0.5 [-0.7, +1.7] (22 vs 17); int6 -0.50 [-1.66, +0.64] / +0.1 [-1.3, +1.6] (28 vs 27);
  fp6 -0.57 [-2.07, +0.93] / -0.8 [-2.7, +1.1] (41 vs 49); fp7 +1.11 [+0.15, +2.10] / +0.7 [-0.5, +1.9] (23 vs 16);
  fp8 -0.27 [-1.07, +0.52] / -0.7 [-1.8, +0.4] (13 vs 20).
  Same lowest-index class on 665 (V3 int6) to 989 (V2 int8) of 1,000 images. Stored-code divergence on the 32-image panels is 2-7 %
  on MobileNetV2 and 4-23 % on MobileNetV3; the three MobileNetV3 cases with the largest divergence (int8 17 %, fp7 20 %, fp8 23 %)
  agree on 912-934 images, more than fp6 (11.7 %, 806) and int6 (4.2 %, 665), so class agreement follows the case's accuracy and
  margins more than the code divergence. Under expected credit two of ten differences exclude 0, in opposite directions (V2 fp7,
  V3 fp7); under lowest index only V2 fp7. Code divergence therefore shows as scatter (class changes on 11-335 images, 6-90
  correctness-discordant images), not as a consistent bias.

## 2. Engines: which file came from which engine, and why the fast path is faster

Two implementations of the same exact engine produced this lane's files:

| | archive (reference) | fast path (lane S1) |
|---|---|---|
| path | `artifacts/scaled_bridge_v2/implementations/1f75c923.../run.sh` | `artifacts/speed_v1/implementations/edfecd5c.../run.sh` |
| engine_sources | 1f75c923... | edfecd5c... (base_engine_sources 1f75c923...) |
| run root | `artifacts/scaled_bridge_v2/runs/1f75c923.../` | `artifacts/speed_v1/runs/edfecd5c.../` |
| files of this lane | P1 wide/control 1k of all 10 cases (128/564/1000 tiles); MobileNetV2-int8 location runs made before 17:07; location of mobilenet_v2-int6 and mobilenet_v3_large-fp7_e3m3 (no fast gate until 18:07); K3 anchors | everything else from 17:11 on (the two late cases from 18:12), each run as one call (0-128 or 0-1000) |
| execution time of the same 20 wide/control 1k runs (files' own `execution_seconds`, shared GPU, batch 8) | 261-635 s (median 342 s; 0.26-0.64 s/image; MobileNetV2 int8 515 / 635 s as single calls, the other 18 summed over their 128/564/1000 tiles; MobileNetV3-L int8 wide over 0-64, 64-128 and the two larger tiles) | 8.5-13.5 s (median 10.2 s; the K2 calls) |
| same pairs, archive / fast | 24-60x (median 33x) | |
| wall time per fast call including process setup (logs/fast-jobs.jsonl) | | 1k calls 10.7-25.2 s (median 17.9 s, n = 129, all policies); 128-image calls 3.6-6.7 s |

Both columns ran on the same shared GPU with other lanes' work, so the times vary with contention; lane S1's controlled profile
(64 MobileNetV3-L int8 images) gives 0.599 s/image for the archive and 0.0136 s/image for the fast path (44x).

The fast path computes the same records; S1's engine review compared about 97,000 image records with 0 differences (about 74,000 by
the four reviewers: A about 12,000, B 38,415, C 10,648, D 12,712; and 23,616 by lane S1 itself), and this lane's own checks
K1-K4 (section 3) found none either. The arithmetic that defines the exact result, the integer reduction kernels, was never the
bottleneck: in S1's profile it takes 0.41 s of the archive's 37.88 s for 64 MobileNetV3 images (1.1 %). The archive spends the rest
on the host, around those kernels. The fast path is faster because of WHERE that surrounding work runs, not because of what is computed:

1. **Store quantisation moved from host NumPy to the GPU.** After every stored tensor the engine divides by the scale and finds the
   nearest codebook level (binary64 division, `searchsorted`, the codebook's tie rule). The archive does this in NumPy on the CPU, element
   by element for every activation of every image; lane S1's profile shows it is 57-65 % of the archive's time (24.6 s of 37.9 s for 64
   MobileNetV3 images). The fast path runs the same operations as torch CUDA float64/int64 kernels (0.47 s for the same images). Bit
   identity is kept by doing exactly the same IEEE operations: division by a 0-dim CUDA tensor (not torch's reciprocal shortcut for Python
   scalars, which differs in 26.6 % of cases), the same left-sided search on the same bounds, the same tie rule.
2. **The data stays on the GPU between nodes.** The archive copies the operands of every conv/linear node from host to fresh device
   buffers, runs the reduction kernel, and copies the results and the per-output event words back (about 2 s of 38 s, plus allocation
   and Python around each call). The fast path launches the same kernels on torch device pointers on torch's stream; no malloc, no copy.
3. **The MAC post-operations and element-wise operators moved to the GPU.** dot*scale*weight_scales+bias, range checks, event counting,
   add, ReLU/ReLU6, hard-swish/-sigmoid, the squeeze-excite multiply and pooling were NumPy (about 8 s of 38 s for MobileNetV3); they are
   now separate CUDA float64/int64 kernels in the same order (separate kernels, so no fused multiply-add contraction).
4. **Fail-closed checks no longer synchronise per node.** Range, closure and non-finite checks are collected as GPU flags and raised, with
   the archive's messages, once per batch before any record is built, instead of a host check after every node.
5. **Preprocessing overlaps the GPU.** The archive's own `image_batch` preprocessing (unchanged code) runs in one prefetch thread for the
   next batch while the GPU computes the current one.

What did NOT change: the reduction kernels (the archive's `kernel.h`, byte-identical, compiled with the archive's nvcc flags
`-O3 --ftz=false --fmad=false`; the PTX of all 8 kernel entry points is identical), the certificates, policies, codebooks, export loading,
hashing, top-5 and tie lists (imported from the 1f75c923 package unchanged), the record format and the sealed-file rules. Only
these are tied to this machine: torch 2.3.0+cu121 elementwise kernels, NumPy 1.26.4 and driver 595.91.07 (checked by K1 before every
runner start). Statements about the exact engine's speed in this document use the archive's numbers only.

## 3. Checks K1-K5 of the combined verdict (addendum 3)
- K1: `artifacts/accumulator_sweep_mn_v1/checks/k1-*.json` (manifest 14/14, run root edfecd5c, versions equal) before every runner start.
- K2 (17:11-17:15): fast wide and control 0-1000 against this lane's archive tiles, every field of every record and the node tables:
  `checks/k2-<case>.json`: 10 cases x 2 policies x 1,000 images identical, 0 differences.
- K3 (archive anchors, seeds of addendum 3): 21 of 21 anchors identical (8 + 2 late-gated cases: widest location width with events, one
  seeded grid policy, MobileNetV3-L int8 f21.x-8), `checks/k3-<case>-<policy>.json`.
- Cross-root (every (case, policy) with files in both roots, all overlapping records): 57 pairs, 23,446 records identical, 0 different
  (`checks/xroot-20261002-182310.json`).
- K4 (seeded 10 % repeat at batch 5 in a second process, scratch root `artifacts/accumulator_sweep_mn_v1/k4/speed`): round 1 11/11 files,
  11,000 records; round 2 12/12 files, 12,000 records; 0 differences, node tables equal.
- K5: 0 failed calls on either engine, no FAST-HALT; fast files only under the fast root; fast GPU time counted by job wall time
  (logs/fast-jobs.jsonl). Lane total since the protocol: 3.7 GPU job-hours of the 14-hour budget (reviewer's recount 3.76; nothing deferred).
- The two late cases (mobilenet_v2-int6, mobilenet_v3_large-fp7_e3m3): lane S1 wrote their fast gates at 18:07; K2 passed 18:12
  (1,000/1,000 each, wide and control). Their location tiles and some 1k tiles were made on the archive; their 1k grids are fast files.
  Reviewer D's read-only `trace_gates.py` was run over all fast gates by this lane's reviewer (review 1, 18:50; output
  `artifacts/accumulator_sweep_mn_v1/review-scratch/gates_trace.json`): 453 compare files, none with a difference, every reference
  written by the archive; all 16 gate files of the two late cases have status pass and identity ok.
- Independent review 1 (`handoffs/Q1-mn-accsweep-review.md`) also re-ran 8 (case, policy) x 64 images outside the K3 anchor ranges on
  the archive: 512/512 records identical to the fast 1k files.

## 4. Deviations, refusals, null results
- Addendum 2 tiles kept for archive calls only; fast runs are single calls. Addenda 3-5 change mechanics and accounting, not rules.
- The location ladder of MobileNetV2-int8 (and of the two late cases) was measured on the archive, the rest on the fast path; rows that
  mix roots list both file references (`policies[*].engine_files`). `location_128.csv` has no engine column; the engine of each
  128-image result follows from the file's root: archive for the location runs of mobilenet_v2-int6 (9 files) and
  mobilenet_v3_large-fp7_e3m3 (19) and for 15 of mobilenet_v2-int8's (its refinement rungs sat.w19 and sat.w17 are fast files),
  fast for the other seven cases.
- The MobileNetV3-L int8 wide 0-64 smoke file (11:33:21) predates the protocol freeze (11:39:23) and is part of that case's archive 1k
  assembly of the exact arm (disclosed in the protocol; identical to the fast 0-1000 file record for record).
- Addendum 5 (17:52:01) was written 21 s after the K4 round-1 CPU compare it records (bookkeeping of a failed result write, not a rule).
- Stopping runners: agent r1 killed its own runner processes twice before the 16:20 orchestrator restriction (PID 164297 at 12:05
  and the second runner at 12:39; `logs/run-main.log` and `logs/run-main-2.log` end on START lines without END; the calls they had
  launched kept running and sealed their files afterwards: V2 int8 control 0-1000 at 12:23:26, V2 int6 wide 564-1000 at 12:47:50;
  both are among the archive files K2 matched record for record); the project owner stopped runner PID 308537 at about 17:05 (`logs/run-main-3.log`
  ends at 17:01:08). From agent r3 on, runners were stopped with the STOP file only. No command of this lane was refused.
- The stress case MobileNetV3-L int6 was included in the across-case slack summaries of revisions r3/r4 against the protocol's
  stress_cases entry; revision r5 leaves it out (section 1). The written-once summaries hold per-case values only and are unchanged.
- K4 round 1's own result write failed on a relative path (results recovered by a CPU compare of the kept repeat files; addendum 5).
- Null results: control = wide in 8 of 10 cases; float accumulators at the rule exponent show no failed image and, under expected
  credit, no non-stress difference that excludes 0. Under lowest index 2 of 18 non-stress intervals exclude 0 in opposite directions
  (MobileNetV2 fp8 fp16.x-11 -1.5 pp, MobileNetV3-L fp8 fp16.x-11 +1.3 pp; section 1), and the stress case's f21 excludes 0 under both
  rules (+1.36 / +1.8 pp).
- Revision r4 corrected text errors found by review 1 (see the note at the top); the data, summaries and figure were not changed.

## 5. Limits
Two networks, one calibration draw (B2 default recipe), one tap order, 1,000 development images (location 128), ties scored by expected
credit and lowest index; a uniform register width is the same for all nodes (per-node widths were not searched; sat.struct-d is a fixed
rule, and its d grid is coarse). The stress case (MobileNetV3-L int6, 20.5 %) has a reduced grid: its W_acc(1.0) = 14 and W_half = 12 rest
on four measured widths (11, 12, 14, 16). The [Agg24] column depends on an unresolved sign-bit convention. The fast path's identity
rests on this machine's torch 2.3.0+cu121, NumPy 1.26.4 and driver 595.91.07.

## 6. Proposed held-out confirmation (proposal only, not run)
Per case: wide, W_noevent, W_acc(1.0), W_half, sat.struct-d_half and the rule fp16 policy on the 9,000 held-out images of the 10k list:
10 cases x 6 policies = 60 runs, 540,000 image evaluations. Fast path at the measured 0.0082-0.021 s/image (execution time of all
129 fast 1k files, 8.16-21.06 s) -> about 1.5-3.5 GPU-hours with setup; archive engine at the measured 0.26-0.64 s/image -> about
40-95 GPU-hours. Archive runs would need this lane's retry logic: the archive fails transiently with `bridge v2 kernel error 2` under
GPU memory pressure from other jobs (4 times in this lane's P1, once more in review 2's rerun). Both engines hard-code the screen-1k
list, so this needs an engine source change, i.e. a new digest and the re-review listed in the combined verdict (d) before any
held-out image is touched.
