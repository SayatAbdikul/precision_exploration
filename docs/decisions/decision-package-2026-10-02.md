PROPOSED - awaiting owner sign-off

# Decision package for the confirmation protocol (2026-10-02, revision 2)

Everything below is PROPOSED by lane P4; nothing is decided. Protocol: `docs/analysis/confirmation-protocol-draft2-2026-10-02.md`
(items appear there as [OC-n]); machine form `public/experiments/configs/breadth-study/confirmation-protocol-v1-draft2.json`.
GPU-hours at the new exact engine's shared rates unless marked "old" (arithmetic: `artifacts/confirmation_protocol_v1/design-v2.json`,
`design-v3.json`; protocol section 11). Recommended package: about 12-37 GPU-h plus 12-14 CPU-h, about 1-3 days on the shared GPU
(estimate), after about 1-2 days for the engine change and its review. INTERIM = rests on a lane not yet final or reviewed.

1. **Promoted set.** (a) 14 entries as the code applies rule (d); (b) 12, without NF4 and ternary. **Rec (b):** both are at chance on all three
   networks, and 12 is the other reading of the stress clause. Cost: (a) +6 simulator cells, 0.14-0.89 GPU-h; risk (b): chosen after the
   results, disclosed. Raised: `b2-matrix-2026-10-01.md:465-475,520-525`.
2. **ResNet18 exact grid; 9k event counts.** (a) 73 runs; (b) 56 runs. **Rec (a)**, event counts on the 9k a declared outcome. Reason: only
   (a) lets sequence I close the lower end of a bracket (INT8 W = 18 upper bound about -0.56 at z 3.14). Cost: (a) 2.3-3.0 GPU-h (old 40-64),
   1.4-1.8 GB uncompressed; (b) 1.8-2.3 (old 31-49), 1.1-1.4 GB. Raised: `accumulator-sweep-2026-10-01.md:360-392`; Q7 `:367-368`.
3. **Margins.** 1.0 pp for classifier contrasts; detector 0.5 mAP (8-bit, recipe differences), 1.0 mAP (6-bit). 0.5 pp accumulator claims:
   (i) secondary, pointwise, for the in-grid arms with screen discordance <= 3.64 % (22 ResNet18 + 10 MobileNet); (ii) confirmatory, a_case
   split 0.4/0.4/0.2 (5 named `sat.w` arms at C = 7, 7 at C = 15). **Rec (i)**, as lane Q7 measured. Cost 0 GPU; (ii) widens N and I
   (z 3.14 -> 3.20 at C = 7). Raised: `quality-metrics-resolution-2026-10-02.md:348-373`.
4. **Tie rules.** Classifiers: expected credit primary, lowest index secondary. Detector: (a) expected AP over 200 image orders primary;
   (b) fixed rule primary, expected AP beside every 6-bit number. **Rec (a):** the fixed rule sits at the 0-1st percentile for INT6. Cost:
   CPU minutes. Raised: `quality-metrics-resolution-2026-10-02.md:328-337`.
5. **Detector recipe.** (a) `default` (frozen); (b) `conformant` (the written freeze rule); (c) both, `conformant` primary. **Rec (c):** (b)
   follows the rule; (a) links to earlier tables. Cost: one recipe 12 configurations 0.7-3.2 GPU-h + 6.4-7.2 CPU-h; both 23: 1.4-6.1 +
   12.3-13.8 CPU-h. Raised: `b2-detector-2026-10-01.md:102-109,616-621`.
6. **Contrasts, criteria, alpha.** (6a) H1 R1: (i) B2 `minimal`, (ii) v1 max-abs, (iii) B2 without bias correction (lane Q2 interim: its 7
   non-chance interval reversals are all of this kind); (6b) criterion A (intervals exclude 0 oppositely, |D| >= delta) or B (beyond +-delta);
   (6c) H3 claim: raw width difference or slack form; (6d) M1 confirmatory or secondary; (6e) detector secondary or confirmatory; (6f)
   equal shares or H3 half; (6g) H1 pool promoted set only or any B2 format; caps 6 pairs, 2 per network (P4 proposals). **Rec:** (i)
   unless Q2's final file gives no pair under it, then (iii); A; slack; M1 confirmatory; detector secondary; equal; promoted only. Reason: a
   reversal between two reasonable recipes is not a weak-baseline effect, and raw widths differ by the product grid alone. Statistical
   cost of H1: joint 80 % power needs a true |D| of 1.19 / 1.73 / 2.13 pp at 7.1 / 15 / 22.6 % discordance. GPU: (ii) needs a v1 list change
   and review (about 1 day). Raised: draft 1 fact-check items 1-7 (`confirmation-protocol-draft-2026-10-01.md:101-107`); `owner-decisions-2026-10-01.md:41-42`.
7. **H2 while no non-integer cost exists (RTL on hold).** (a) freeze the cost criterion now; quality side as pointwise TOST (90 %, 1.0 pp);
   cost evaluated later without alpha; (b) remove H2 from this paper. **Rec (a):** the quality data cost nothing extra. Cost 0 GPU; risk:
   H2 untested here. Raised: `owner-decisions-2026-10-01.md:50-56`; draft 1 item 7.
8. **MobileNets in H3 and M1 (INTERIM, lane Q1 complete 18:23, not reviewed).** (a) ResNet18 only; (b) cases within 10 pp of FP32 (proposed
   floor) join H3 (8 of 10; C 7 -> 15: z 3.14 -> 3.37, bound 0.55 -> 0.60 pp at 2.8 %) and join M1 if projected TOST power >= 0.8 (5 of 8;
   FP7 on both networks fails, 1k intervals exclude 0); (c) descriptive. Grid: the item-2 rule (83 or 61 runs) or Q1's 60. **Rec (b)** with
   the item-2 rule if Q1's review is done before signing, else (c). Cost: 83 runs 2.9-3.2 GPU-h (old 107-132), 61: 2.2-2.4, 60: 2.1-2.4;
   83 runs 1.6-2.0 GB uncompressed. Raised: this lane's task; `accumulator-sweep-mobilenet-2026-10-02.md:117-122`.
9. **Statistical details.** Seed 20260927 (project functions) or 20261001 (draft 1); unstratified or class-stratified; detector interval
   basic for every contrast or percentile for every contrast. **Rec:** 20260927; unstratified; basic. Reason: the code exists; resampled mAP
   is biased upward. Stated, no choice: 200,000 resamples at adjusted levels; bounds against non-zero margins at nominal alpha / 1.4
   (1.5 at C = 15), checked exactly (`coverage-v2.json`, `coverage-v2a.json`). Cost 0. Raised: draft 1 `:24`; `b2_ties.py:28`;
   `b2-detector-breadth-2026-10-02.md:266-273` (INTERIM).
10. **Optional arms.** Per-node `sat.struct` (32 deferred runs) and the older-recipe B1 check (18): screen first 0.24-0.31 GPU-h, then 9k
    1.62-2.06; halving to d in {4, 8} (34 runs, 0.18-0.24 / 1.12-1.43) is a P4 proposal, not the sweep's plan. v1 gap rerun: 5.2-7.9 old
    GPU-h plus a v1 list change. AXE: 4 evaluation cells 0.09-0.59 (its fits are not in Q8's sources). **Rec:** per-node and B1 on the screen,
    secondary on the 9k only if done before signing; no v1 gap rerun (M1 covers it); AXE secondary (reviewer baseline). Raised:
    `accumulator-sweep-2026-10-01.md:297-300,390-391`; `scaled-bridge-gap-2026-10-01.md:273-285`; `related-work-audit-2026-10-01.md:1123`.
11. **Proposals P1, P3 and P5.** P1 B breadth-first, strict-A obligations open; P3 D4 stays open, promotion = entry into this protocol;
    P5 B2 results are D6 evidence only. (a) sign as written, P3 naming the 9k/4k sets; (b) amend; (c) leave open. **Rec (a):** otherwise
    promotion happens without a recorded gate. Cost 0. Raised: `proposed-decision-updates-2026-10-01.md:7-30,41-47`.
12. **Bias-correction form (lane Q3, INTERIM).** (a) keep B2's pass; (b) literal weights-only Appendix D; (c) local own-error; (d) guarded
    rule; (e) keep B2 frozen, one alternative form as a secondary held-out arm on FP8 E5M2, FP6 E3M2, LOG6 x ResNet18, MobileNetV2. **Rec (e)
    with (b)**, the cited method: (b)-(d) as the recipe reopen rule (c), the promoted set, exports and brackets. Cost: (e) 6 cells 0.14-0.89
    GPU-h; (b)-(d) days plus a new protocol. Raised: `b2-collapse-diagnosis-2026-10-02.md:546-566`.
13. **MobileNetV3 stem repair r3b (lane Q4, INTERIM).** (a) one secondary held-out cell; (b) MobileNetV3 recipe arm; (c) neither. **Rec (a):**
    chosen on the screen, the held-out cell tests exactly that. Cost: (a) 0.02-0.15 GPU-h; (b) re-export, days. Raised:
    `b2-activation-attribution-2026-10-02.md:358-377`.
14. **Historical budget ledgers.** (a) binding and closed; new confirmation ledger, ceiling 1.5 x upper estimate; (b) reopen ledger A;
    (c) drop. **Rec (a)**. Cost: (b) at least 6,443 s. Raised: `proposed-decision-updates-2026-10-01.md:32-39`; `owner-decisions-2026-10-01.md:42-43`.
15. **Scaled-bridge contract.** (a) F6 unamended, bridge = "B recipe executed with exact arithmetic", its `control` a B policy; (b) amend F6.
    **Rec (a)**. Cost 0. Raised: `proposed-decision-updates-2026-10-01.md:17-23`; `owner-decisions-2026-10-01.md:43-44`.
16. **Calibration seeds (lane Q5, INTERIM).** (a) one frozen draw, Q5 cited; (b) (a) plus Q5's subsets on the 9k for the H1 pair cells;
    (c) new disjoint 2k lists; (d) addable: Q5's accumulator bracket check on the screen (about 4 old GPU-h). **Rec (b):** reviewers expect
    three seeds; no new data. Cost: (b) up to 72 cells 1.68-10.70 GPU-h; (c) about 4,000 new images, weeks. Raised:
    `owner-decisions-2026-10-01.md:44`; `b2-calibration-seeds-2026-10-02.md:185-201,260-271`; `related-work-audit-2026-10-01.md:1130-1131`.
17. **Unsigned activation formats.** (a) labelled recipe switch, D1 unchanged; (b) extend D1. **Rec (a):** Q1.6 equals INT8 bit for bit.
    Cost 0. Raised: `proposed-decision-updates-2026-10-01.md:56`; `d1-initial-candidate-set.md:50` (deferred D1 dimension).
18. **Second PDK (D9).** (a) state the ICS55-only limit; (b) authorise one when RTL resumes. **Rec (a)**. Cost (b): weeks, disk. Raised:
    `proposed-decision-updates-2026-10-01.md:58`.
19. **ImageNet 50k payload.** (a) 9k/4k are this paper's confirmation, 40k later under its own protocol; (b) wait. **Rec (a).** 40k later:
    73 runs 10.1-12.8 GPU-h, 6.1-7.9 GB of records plus the payload, over the 3 GB cap. Raised: `owner-decisions-2026-10-01.md:45-46`.
20. **Exact-engine source change.** (a) list selection in the fast engine (base sources `1f75c923`), re-checks (d), independent review,
    owner acceptance; (b) change the archives (old rates); (c) no exact runs. **Rec (a)**. Cost: at most 2 GPU-h, about 1 day. Raised:
    `S1-engine-review.md:698,716-729`.
21. **Disk for held-out records.** (a) about 1 GB allowance, compressed records; (b) 3-4 GB uncompressed (over the cap); (c) slimmer records.
    **Rec (a):** gzip shrank 1k records 3.6-9.3 times. Cost: 73 + 83 runs 0.32-1.06 GB. Raised: protocol section 11.
