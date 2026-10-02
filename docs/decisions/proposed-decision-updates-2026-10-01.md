PROPOSED - awaiting owner sign-off

# Proposed decision updates, 2026-10-01

Drafted by lane L3 for the project owner. Nothing here is a decision. The register `docs/decisions/decision-register.md` is not edited. Records follow its template only loosely: P2 to P5 omit several fields, and no record gives artifact hashes or the "Expected scientific impact" field; these must be completed before any record is copied into the register. "Owner" and "Reviewers" are left for the owner. "Not recorded" means no repository document shows the owner approving the item. Lane handoff files under `artifacts/agent_orchestration/handoffs/` are work in progress, not evidence.

## P1. Breadth-first B exploration with the E1 and E2 studies

- ID: P1. Status: proposed. Scope: in-repository. Owner/Reviewers: blank. Date: 2026-10-01.
- Alternatives considered: finish all 100 strict-A 1k screens first (the earlier requirement); narrow study (`docs/roadmap/README.md` says the breadth-first proposal "supersedes the narrower study recommendation, not the accepted phase gates"; `docs/analysis/breadth-first-study-proposal-2026-09-25.md` itself says it "replaces the scope recommendation" of the focused five-week proposal).
- Authorisation on file: `docs/architecture/experiment-b-exploration.md` line 3: "The owner authorized starting the transition on 2026-09-25." `docs/roadmap/README.md`: "On 2026-09-25 the owner authorized starting the B-led exploration transition." `docs/analysis/breadth-first-study-proposal-2026-09-25.md` line 3: "the owner chose to start the B-led transition." For E1 and E2 the record is weaker: `docs/roadmap/README.md` calls the checkpoint's contents a "sequence discussed with the owner", and the checkpoint document that defines E1 and E2 (`docs/analysis/research-checkpoint-and-next-experiments-2026-09-25.md`) opens "Saved 2026-09-25 after the owner's request to preserve completed work and identify the additional experiments needed for the research. This records the direction agreed in the discussion". Who agreed is not named there. A written approval of E1/E2 as numbered studies: not recorded.
- Evidence: `artifacts/experiment_b/transition-20260925.json`; 200 configurations (183 at 1k, 17 at 128) per `docs/analysis/paper-roadmap-assessment-2026-09-27.md` and `results/summaries/b-stage-paired-1k-v2/analysis.json`; E1 and E2 definitions in the checkpoint document.
- Proposal: record as accepted-with-limits. B is FP32 quantize/dequantize; its results are development evidence, cannot complete original D4 (`docs/roadmap/README.md`), and the original strict-A obligations stay open, not relabelled. Selection used 128-image results, so 1k panels remain development data.
- Affected: `docs/architecture/experiment-b-exploration.md`, `tools/experiment_b*`.
- Supersedes: only the breadth-first proposal's "requirement for all 100 strict-A screens before B" (wording of that proposal's line 5). Does not supersede D4.

## P2. Scaled-bridge contract (external scales in exact execution) and F6

- ID: P2. Status: proposed. Scope: in-repository.
- Facts: F6 says "Experiment A has no optional external scale; only intrinsic/necessary family scaling" and moves "general/power-of-two scale ... to B" (register). The bridge applies B's external FP32 scales and weight codes inside exact code-domain arithmetic (`docs/analysis/scaled-bridge-v1-contract-2026-09-28.md`); v1 is sealed in `tools/scaled_bridge_v1/common.py`. v2 generalises it to recipe-as-data (`docs/analysis/scaled-bridge-v2-contract-2026-10-01.md`, which states "Frozen 2026-10-01, before any v2 panel was run"; the file is not yet committed, and milestones M2 to M4 are listed there as "Planned additions, not yet frozen". The implementation is in progress: lane L2 handoff `L2-exact-engine.md` read "Nothing verified yet", milestone M1 in progress, on 2026-10-01).
- Proposal: F6 is not amended. The bridge is classed "B recipe executed with exact arithmetic" (a B-family result with exact A-style arithmetic), never "Experiment A". Canonical A results (`docs/analysis/fp64-grid-v2-study-outcome-2026-09-28.md`) stay unscaled. The v1 doc states "No legacy unscaled certificate is extended or relabeled"; keep that rule.
- Authorisation: owner approval of the bridge as a study: not recorded. Neither contract document mentions the owner, and neither records who sealed it.
- Open: whether F1/F2 (Model C, wide accumulator) apply to the bridge's control arm (sequential binary32 FMA). Owner call.

## P3. Gate D4 (promotion from 1k to 5k/10k)

- ID: P3. Status: proposed (register: D4 Open/in-repo, unchanged).
- Facts: the 2026-09-25 redesign proposes splitting D4 into D4a (a candidate can enter depth safely) and D4b (coverage/promotion accounting complete), and "Completing a v2 gate would not retroactively complete the frozen v1 100-screen requirement" (`docs/analysis/research-redesign-2026-09-25.md` section 6). Owner acceptance of D4a/D4b: not recorded. Eight accepted 1k integer A screens exist (`paper-roadmap-assessment-2026-09-27.md`).
- Proposal: keep D4 open; adopt D4a/D4b only after sign-off; define promotion by the rule in `docs/analysis/confirmation-protocol-draft-2026-10-01.md` section 6, as promotion to confirmation, not to 10k development use. The 10k list and the COCO 5k list are not to be opened for candidate evaluation until then. Whether they are unopened today is not established: the evaluation-history audit is not done (protocol section 5). What is recorded: the FP32 classifier baselines used the 1k list (`results/summaries/phase1-fp32-baselines.json`, sample_count 1000), and YOLOv8n FP32 was scored on all 5,000 COCO images. Note that protocol sections 5 and 9 treat the 10k list as a development list that may be used; this sentence is stricter, and the owner should choose one.
- Evidence: `docs/roadmap/README.md` (B cannot declare D4 complete), `data/manifests/index.json`.

## P4. Lifetime budget ledgers: stay binding?

- ID: P4. Status: proposed.
- Ledger A, useful-quality E1 v2: 24 aggregate worker-hours, at most 12 for extensions (`docs/analysis/useful-quality-e1-v2-controller.md`); 0.27 h (975 s) left, no unreconciled reservations (`docs/analysis/fp64-grid-v2-study-outcome-2026-09-28.md`; `results/summaries/useful-quality-e1-v2/`). Controller state `budget_limited` (`docs/analysis/useful-quality-e1-v2.md`). Blocked: FP7 first-image rational comparison (6,443 s reservation) and the MobileNetV2 INT8 128-image extension.
- Ledger B, scaled bridge v1: 14,400 worker-seconds (`tools/scaled_bridge_v1/common.py`; contract line 57). Charged: 602.1 of 14,400 worker-seconds, with 0 failed or interrupted attempts (`docs/analysis/scaled-bridge-v1-results-2026-09-28.md`; the fact-check summed `charged_seconds` over the 102 records in `artifacts/scaled_bridge_v1/budget/attempts/` and got 602.1, newest record 2026-09-28). That leaves 13,797.9 worker-seconds.
- Origin of the figures: set in the sealed protocols. An owner instruction that fixed them: not recorded.
- Alternatives: (a) stay binding; the old ledgers remain closed, new studies get new ledgers (v1 contract states "The old E1 ledger is historical and is not reset or extended"). (b) Owner raises or reopens them. (c) Drop the ledger rule.
- Recommendation: (a). The ledgers protect resumability and honest accounting; they are not scientific limits. Missing panels stay labelled "budget-limited", not negative. New work is not under a capped ledger: the v2 contract only says each measurement "seals its wall-clock seconds in the run ledger" and states no ceiling, and the gap v1 protocol (`artifacts/scaled_bridge_gap_v1/protocol-1k-v1.json`) runs the sealed v1 engine on the 1k list with "v1 ledger, runs and reports are not written" and states no ceiling either. Under alternative (a) the owner would need to set ceilings for them. Reopening ledger A for FP7 and MobileNetV2 INT8 is cheap (the FP7 comparison alone is a 6,443 s reservation) and is an owner choice; those cases support certificate evidence for the accumulator headline, so I suggest it only if the owner wants FP7 in the paper.

## P5. B2 recipe work, D6 and F5

- ID: P5. Status: proposed. Scope: in-repository.
- Facts: B2 is a new versioned study (`public/experiments/configs/breadth-study/b2-baseline-repair-protocol-v1.json`, written "before any B2 quality measurement"; code `tools/experiment_b2/`; lane L1 handoff `L1-baseline.md`, design stage). Its switches include bias correction, MSE ranges, fused ReLU boundaries and an unsigned integer codebook. With all switches off it must reproduce v1 bit for bit.
- F5 forbids "bias correction/reconstruction in A" and says "Family-specific PTQ improvements belong to B". So B2 is within F5 only as B. D6 ("Reasonable Experiment B policy per family", open) is the gate B2 informs; the redesign asks D6 to "freeze the best recipe under a stated budget".
- Proposal: B2 results feed D6 as evidence only; D6 stays open until the frozen recipe passes the confirmation protocol's rule; the recipe is frozen before any complement is opened. Label B2's empirical bias correction a B recipe and do not present it as a published method. (The wording "AdaRound-inspired adaptation, not an AdaRound reproduction" in `docs/analysis/paper-roadmap-assessment-2026-09-27.md` item 4 is about the E2 rounding intervention, not B2; B2 has no rounding switch.)
- Owner approval of B2: not recorded.

## Questions only the owner can answer (priority order)

1. **Approve the confirmation protocol draft** (hypotheses, margins, data roles). The draft proposes that no confirmation run may start without sign-off; no such rule is recorded in the register today.
2. **Headline mechanism.** The audit recommends the accumulator-width sweep with a certified lossless width plus ASIC cost, with the simulator-versus-exact gap as methods (`docs/analysis/related-work-audit-2026-10-01.md` section 4). Accept, or choose another?
3. **ImageNet 50k payload.** Deferral is the owner's ("deferred_to_phase_5_by_owner", `data/manifests/index.json`; `docs/roadmap/phases/phase-01-foundation.md` line 219: "the 50,000-image ImageNet validation payload and full-list freeze are deferred to Phase 5"). When, and who provides it? Also: extra training images for new calibration seeds.
4. **Retraining/QAT.** Excluded by F5 and README ("Retraining or quantization-aware training; the supplied plan uses PTQ only"). In the audit, the integer work that reduces accumulators with guarantees is QAT-based in part ([Col23], [Col24a]), while [Col24b] is PTQ for LLMs and [Umu25] is static analysis; the work it names closest to this paper, [Agg24], uses PTQ (`docs/analysis/related-work-audit-2026-10-01.md` sections 1.1 and 4). Keep the exclusion and say so in the paper, or open a QAT arm as a separate study?
5. **Mixed precision** is external gate D14 ("Open/external"). Confirm it stays out of the public paper, noting the audit's option (b) would border on it.
6. **Unsigned activation formats** are outside the accepted D1 set (25 candidates, hash `986344a9...`; `docs/decisions/d1-initial-candidate-set.md` lists "unsigned variants" among the deferred later-phase dimensions). B2 defines an integer-only unsigned variant. Extend D1, or keep it as a labelled recipe switch?
7. **Transformer workload.** Redesign: do not claim beyond CNNs without "a properly audited transformer path". Add one, or state the CNN scope?
8. **Second PDK with memory macros.** D9 (physical memory evidence level) is open in the register. `docs/analysis/memory-model-2026-09-27.md` records "no characterized SRAM/compiler and no validated technology-specific generated macro" for ICS55; the physical area, latency and energy fields "remain missing" and "no physical SRAM savings are awarded". Authorise a second PDK, or state the limitation?
9. Whether to reopen ledger A (P4) and whether F1/F2 govern the bridge control arm (P2).

## Fact-check log (lane L3 part 4, 2026-10-01)

Independent check by a second agent. Every cited file was opened; numbers, dates, quotations and paths were compared with it. Nothing here is an owner decision.

Corrected in place (the original wording was wrong or unsupported):

1. Header: "Records use its template" was overstated; P2 to P5 omit template fields and no record has artifact hashes.
2. P1: the phrase "discussed with the owner" was attributed to the checkpoint document; it is in `docs/roadmap/README.md`. The checkpoint header is now quoted as written, and the breadth-first proposal's line 3 was added.
3. P1: "supersedes the narrower recommendation" was attributed to the breadth-first proposal; those words are in `docs/roadmap/README.md`.
4. P1: the text in quotation marks under "Supersedes" was not a quotation; replaced by the source wording.
5. P2: the v2 contract was called "work in progress"; the document says it is frozen. The implementation is what is in progress.
6. P2: "sealed by the implementing agents" is not recorded in any file; replaced by what the files show.
7. P3: "remain unopened" implied a verified state; the history audit is not done. Also flagged that P3 is stricter than protocol sections 5 and 9 about the 10k list.
8. P4: ledger B balance filled in (602.1 of 14,400 worker-seconds charged, recomputed from the attempt records).
9. P4: "New work (v2, gap v1) already runs under its own ledger" was wrong; neither has a stated ceiling.
10. P5: the AdaRound quotation concerns the E2 rounding intervention, not B2 bias correction, and comes from the roadmap assessment, not `docs/roadmap/README.md`.
11. Question 1: the "no confirmation run may start" rule is the draft's proposal, not a recorded rule.
12. Question 4: "the audit's closest integer prior work is QAT-based" was inaccurate; corrected from audit sections 1.1 and 4.
13. Question 6: source for "outside D1" added. Question 8: the unchecked memory statement replaced by what the memory-model document says.

Verified unchanged: F1/F2/F5/F6 wording, D4/D6/D9/D14 status and D1 count and hash (register); owner authorisation of the B transition on 2026-09-25 (`experiment-b-exploration.md` line 3, roadmap README); 200/183/17 coverage (`results/summaries/b-stage-paired-1k-v2/analysis.json`: configurations 200, at_1000 183, at_128 17); eight accepted 1k integer A screens; D4a/D4b text and the "would not retroactively complete" sentence (redesign section 6; the redesign's own status line is "proposal, not an adopted campaign or D4 decision"); ledger A figures 24 h, at most 12 h for extensions, 0.27 h (975 s) left, 6,443 s FP7 reservation, `budget_limited`; ledger B ceiling 14,400 (`common.py` `ceiling_worker_seconds`, contract line 57); both v1 contract quotations; B2 protocol quotation and switches (`tools/experiment_b2/recipe.py`); `deferred_to_phase_5_by_owner` and `phase-01-foundation.md` line 219 (the owner scope decision there is dated 2026-09-07); README QAT exclusion; redesign transformer wording; D6 "under a stated budget".

Not verified: who wrote or approved the bridge contracts; the current state of lanes L1, L2 and L5 beyond their handoff headers; whether any candidate run touched the 10k or COCO 5k lists (name search only).

Omissions the owner should know (not added to the proposals):

- P2: the D1 record (`docs/decisions/d1-initial-candidate-set.md` line 50) lists "external-scale minifloats" and "all accumulator sweeps" as deferred "later-phase dimensions". The bridge and the proposed accumulator sweep are exactly these, so their status may need a D1 or D5 note, not only an F6 note.
- P1: "accepted-with-limits" is not one of the register's status values (open/in-repo, open/external, accepted, superseded). D2 gives a precedent for "accepted" with a stated scope.
- P4: the gap v1 study runs the sealed v1 engine on 1,000 images outside the v1 ledger. That is allowed by its own protocol but means the 14,400 s ceiling does not bound all use of the v1 engine.
- Question 3: for this run the orchestrator relayed the owner's statement that the ImageNet 50k payload cannot be provided for now. That statement is not in a repository file; the timing question stays open.
