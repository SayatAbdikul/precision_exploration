PROPOSED - awaiting owner sign-off

# Confirmation protocol, pre-registration draft (2026-10-01)

Nothing here is frozen or approved. Hypotheses, margins and seeds are proposals. Companion: `docs/decisions/proposed-decision-updates-2026-10-01.md`. Other lanes' work (B2: `artifacts/agent_orchestration/handoffs/L1-baseline.md`; bridge v2: `L2-exact-engine.md`; 1k gap: `L5-bridge-gap.md`) is in progress and cited as such.

## 1. Scope and status of the evidence

Everything run so far (128, 256, 1k panels, E1/E2, bridge v1) is development evidence (`docs/analysis/paper-roadmap-assessment-2026-09-27.md`: "the larger panels remain development evidence"). Confirmation is the first use of data no candidate or recipe choice has touched. The protocol must be signed and hashed before any complement image is opened.

## 2. Hypotheses (falsifiable)

Base: redesign section 9 table (`docs/analysis/research-redesign-2026-09-25.md`) and audit section 4 (`docs/analysis/related-work-audit-2026-10-01.md`).

- **H1, recipe changes ordering.** For at least one predeclared format pair (X, Y) of equal nominal width on the same model, the paired top-1 difference X minus Y has the opposite sign under recipe R1 (v1 maxabs) and R2 (the frozen B2 recipe), each exceeding the margin delta in magnitude. Falsified if no pair reverses beyond delta. Motivation seen in development only: MNV3 INT8 4.10% (maxabs) versus 68.90% (percentile) at 1k (`paper-roadmap-assessment-2026-09-27.md`).
- **H2, complete cost reverses arithmetic-only ranking.** For at least one pair (X, Y) that is quality-equivalent within delta on the confirmation set, X has lower multiplier-only area and Y has lower complete-MAC area or energy (decode, scale, accumulate). Falsified if the order is the same at every pipeline setting evaluated. Quality part: confirmation data. Cost part: lane L4 flow (`docs/analysis/ics55-integer-timing-2026-10-01.md`; integer MACs only so far, so H2 is untested for non-integer formats; that document's Limits section says to treat its five minimum periods "as not valid timing points" and defers to `docs/analysis/ics55-integer-isoclock-2026-10-01.md`, which is the timing-driven matrix to cite). Cost has no sampling interval; report placement-seed spread. Both L4 documents are synthesis-level (the iso-clock one states "ideal clock, typical corner, no wires"), so no placed design exists yet from which a placement-seed spread could be reported.
- **H3, accumulator failure width is format dependent** (audit option (a)). For the promoted formats on ResNet18, define k*(F) as the smallest accumulator width (fixed point with saturation, FP16, custom float) whose top-1 stays within delta of the wide arm. Supported if k* differs by at least 2 bits between two formats. Falsified if all k* lie within 1 bit. Must also report the certificate width minus k*. Audit condition: "the pilot shows no quality loss from FP32 accumulation even where the certificate exceeds 24 bits".
- **Exploratory only, no confirmatory claim:** the third redesign hypothesis (operator structure predicts choices); the audit demotes per-layer selection (option b) to an extension.

## 3. Primary contrasts, margins, multiplicity

- Primary contrasts: at most 12, listed with the promoted set before unblinding. All paired by image id (redesign section 7). Each is a difference of per-image correctness (classifiers) or a COCOeval statistic recomputed on resampled images (detector; AP-small/medium/large retained).
- Margin delta (proposed, owner to set): 1.0 percentage point top-1; 0.5 mAP50-95 points for YOLOv8n. Illustration, not measurement: for discordance rate p, the paired standard error is about sqrt(p/n). At p = 5%, n = 40,000 gives 0.11 pp; n = 1,000 gives 0.7 pp. So 1 pp is resolvable only on the large set. Equivalence is claimed only if the whole interval lies inside the margin; non-inferiority is one-sided.
- Intervals: 10,000-resample paired image bootstrap (classifiers stratified by class; detector by image with COCOeval rerun), seed 20261001 (proposed, not data derived). Matches the bridge convention of paired bootstrap intervals (`scaled-bridge-v1-contract-2026-09-28.md`), but those were pointwise; here they are simultaneous.
- Multiplicity: family-wise alpha 0.05, split equally across H1, H2, H3 (Bonferroni), Holm inside each hypothesis, simultaneous (max-t bootstrap) bands within a hypothesis. Secondary contrasts are reported with unadjusted intervals marked descriptive. No adaptive peeking: one look per contrast set.
- Seeds: seed-to-seed variation reported separately from image variation.

## 4. Calibration seeds

Redesign section 7: three seeds for leading comparisons, five if "ranking is unstable and budget allows". Proposed: three; extend to five if any primary contrast changes sign across seeds or the seed standard deviation exceeds delta/2. Failed seeds are reported. Calibration sets are paired across recipes and fixed before fitting. Only one 2k set exists per dataset (`data/manifests/calibration/imagenet1k_train_2k.tsv`, `coco2017_train_2k.tsv`); verified disjoint from the 10k and 5k evaluation lists by content hash (0 shared). Two further disjoint sets need new train images, which are not on this machine (data/raw has 2k calibration only). Full-set evaluation of one seed is not a three-seed average (roadmap assessment item 7); seed robustness is run on a common subset.

## 5. Data roles and the audit before calling a complement untouched

| Role | ImageNet | COCO |
|---|---|---|
| Calibration | 2k train list | 2k train list |
| Development | 1k list, 10k list | 1k list |
| Confirmation | 40k complement of the 10k list (payload absent) | 4k complement of the 1k list within the 5k list |
| Reported also | full 50k | full val5k |

Facts: the COCO 5k list is the whole val2017 (`data/manifests/selection/coco2017_val_5k.json`: population_count 5000), and its payload is present (`data/raw/coco2017/val2017`). Complement "untouched" means untouched by this project's candidates and recipes, not by the world: pretrained weights were trained elsewhere and torchvision recipes may have been tuned on the public val set (not checked). A full-set score containing development images is not independent (redesign section 7).

**Known history that already breaks "untouched" for COCO:** YOLOv8n FP32 was evaluated on all 5,000 images (`results/summaries/phase2-yolov8n-fp32-cocoeval.json`: image_count 5000, map50_95 0.3736; `tools/phase3/baselines.py` names a val5k predictions file). The baseline outcome on the 4k complement is therefore already seen. Quantised-candidate use of the complement is not shown by this check; `b2-baseline-repair-protocol-v1.json` states the 10k list "is not touched by this study".

Audit required before each complement is used (all must pass and be hashed into the protocol):
1. Enumerate every artifact holding per-image predictions or per-image ids (`artifacts/**`, `results/**`) and compute the set of content hashes evaluated; require empty intersection with the complement for every candidate and recipe run. Not yet done: this draft only grepped for list names (hits in Phase 1/2 freeze summaries, `b-stage-paired-1k-v*/analysis.json`, `artifacts/phase3/baselines/yolov8n-*.json`) and did not open them.
2. Confirm no scale, range, recipe or promotion decision read an outcome on the complement.
3. Content-duplicate screen (section 7).
4. Record all FP32 baseline outcomes on the complement and label them as seen.

## 6. Rule for the promoted set (set left open)

The set depends on lane L1 and on lane L2 admissions, and is chosen only from development data. Proposed rule, frozen before opening confirmation: (i) every format-model pair must have a frozen recipe from the B2 exit test or a stated exception; (ii) include, by pre-declared strata, accuracy leaders, plausible hardware specialists and uncertain cases (roadmap: not top-N); (iii) size about 8 to 12 exact configurations for development, and a smaller central set (the roadmap's costing scenario is four classifiers and one or two detectors) for full confirmation; (iv) for H3, only cases with an admitted exact engine and a certificate; (v) the list, recipes, margins, contrasts and seeds are written to a hashed file before any complement image is read. Candidates failing a stated gate are reported as not promoted, with the reason, never as dominated.

## 7. Manifest audit (run 2026-10-01, read-only, content hashes)

Script run with python3 over `data/manifests/evaluation/*.tsv` (columns carry sha256 of image bytes):

| Check | Result |
|---|---|
| ImageNet 1k list rows / 10k rows | 1,000 / 10,000 |
| 1k sha256 values found in 10k | 1,000 of 1,000 (also equal by path and by source_name+sha pair); 1k unique 1,000, 10k unique 10,000 |
| COCO 1k rows / 5k rows | 1,000 / 5,000 |
| COCO 1k sha256 found in 5k | 1,000 of 1,000 (also by image_id) |
| Class balance | 10k: 10 per class in all 1,000 classes; 1k: 1 per class |
| Manifest file hashes vs `data/manifests/index.json` | all four equal |
| Calibration versus evaluation, by sha256 | ImageNet 2k vs 10k: 0 shared; COCO 2k vs 5k: 0 shared |
| Payload on disk | ImageNet `val_evaluation_10k` 10,000 files, `val_screen_1k` 1,000; COCO `val2017` 5,000 jpg plus txt |

Both nesting claims hold. Limit: ImageNet nesting is by content hash, so it was not tested against the original 50k.

**Enumerating the 40k complement once the payload exists.** (1) List the 50,000 ILSVRC2012 validation files by `source_name` (form `ILSVRC2012_val_00045880_n01440764.JPEG`; the 10k list has 10,000 distinct names, ids 1 to 49,998; the `_n01440764` suffix is how the pinned `ILSVRC/imagenet-1k` stream named files, so if the payload arrives with other file names, match on the eight-digit validation index; other namings were not checked here). Complement = all names minus the 10k names; expect exactly 40,000. Do not re-run the streaming shuffle (seed 20250904, buffer 2048, `tools/setup/download_imagenet_subset.py`): `stream_position` depends on it and the set difference does not. (2) Compute sha256 of each file. (3) Remove any whose sha256 is in the 10k list, and report the count. This is not hypothetical: the 10k selection dropped a byte-identical duplicate under label 77 whose twin kept under label 72 (`data/manifests/selection/imagenet1k_val_10k.json` content_deduplication; `0072/4a4ce1942e4c44d8361b.JPEG` is in the list). That label-77 file is outside the 10k yet, since the stream was the validation split, in the 50k (an inference: the 50k is not on disk, and the selection record keeps the dropped file's sha256 `4a4ce194...` but not its `source_name`, so it can be found only by content hash), so the complement as enumerated by name has at least one image whose content is already development data. Final confirmation size may thus be below 40,000; report n. (4) Labels from the standard ILSVRC2012 validation ground truth, mapped by the project's zero-based synset order (`public/workloads/datasets/imagenet1k.json`); verify the 10k labels reproduce first. (5) Hash the resulting list into the protocol. COCO: complement = `coco2017_val_5k.tsv` rows whose sha256 is not in `coco2017_val_1k.tsv`; expect 4,000 by the verified nesting.

## 8. Stop rules

- No peeking: a contrast set is evaluated once on confirmation data; an interim look is allowed only for crash or integrity checks, not outcomes.
- If no H1 pair reverses beyond delta, report H1 as not supported rather than widening the contrast list.
- Stop a configuration if its first 1,000 confirmation images show a failed integrity gate (engine disagreement with its control or oracle); failed gates are reported, not replaced.
- Stop extending seeds at five.
- Compute: stop at the pre-declared worker-hour ledger for the confirmation stage (to be sized from measured rates; the roadmap's planning scenario is 9.3 to 13.9 machine-days for four full classifier evaluations, a hypothetical rate). Budget-limited cases are labelled so.
- Any change to recipe, promoted set or margin after unblinding makes the result exploratory and requires a new protocol version.

## 9. Blocked on the ImageNet 50k payload

Blocked: the ImageNet 40k complement, the conventional 50k score, the audit steps 1 and 3 for ImageNet, the three-seed full-set claim, and any confirmatory ImageNet statement for H1 and H3. Not blocked: COCO complement (payload present, subject to the section 5 baseline caveat); H2 cost side; development-stage work (B2, bridge v2, 1k gap); preparing hashes, scripts and the frozen file. Until then, headline numbers are 1k and 10k development results and must be worded as such (no 10k candidate result was found in `results/summaries/` by a name search on 2026-10-01; the recorded FP32 classifier baselines are on the 1k list). The 10k list is on disk and is a development list (redesign section 7); any outcome read from it must be declared as development use.

## Fact-check log (lane L3 part 4, 2026-10-01)

Independent check by a second agent. Cited files were opened and compared. Nothing here is an owner decision; the protocol stays unsigned.

**Corrected in place.** (1) H2: the cited L4 timing document disowns its own minimum periods and is replaced by `ics55-integer-isoclock-2026-10-01.md`; both are synthesis-level, so "placement-seed spread" has nothing to report yet. (2) Section 7: the `source_name` form is that of the pinned stream, not necessarily of the payload the owner will supply. (3) Section 7: the label-77 duplicate being "in the 50k" is an inference, and its `source_name` is not recorded. (4) Section 9: no 10k candidate result was found, so "1k and 10k development results" describes roles, not existing numbers.

**Verified unchanged.** Quotations from the roadmap assessment, redesign sections 7 and 9, audit section 4, the v1 bridge contract and the B2 protocol; MNV3 INT8 4.10% versus 68.90%; the standard-error illustration (sqrt(0.05/40,000) = 0.11 pp, sqrt(0.05/1,000) = 0.71 pp); 9.3 to 13.9 machine-days; COCO selection record (image_count 5000 = population_count 5000); YOLOv8n FP32 at 5,000 images, map50_95 0.37356; `tools/phase3/baselines.py` val5k file name; download seed 20250904 and buffer 2048; zero-based synset mapping; duplicate twin `0072/4a4ce1942e4c44d8361b.JPEG` present under label 72 and `0077/680b4dfe2ad9387eecc6.JPEG` as the replacement.

**Manifest audit, rerun independently** (own script, kept outside the repository, read-only). Every row of the section 7 table was reproduced: ImageNet 1,000 and 10,000 rows, all sha256 unique, 1,000 of 1,000 nested (by sha256, by path, by name plus sha256, and by whole row); COCO 1,000 and 5,000 rows, 1,000 of 1,000 nested (by sha256, by image_id, by whole row); 10 per class in 1,000 classes and 1 per class; calibration against evaluation 0 shared for both datasets; payload 10,000, 1,000 and 5,000 files, every listed file present. Additional checks: all six manifest hashes and their selection-record hashes equal `data/manifests/index.json` (the draft checked four); the sha256 column equals the file bytes for 200 randomly chosen files in each of the 10k, 1k and COCO 5k payloads; the COCO complement has exactly 4,000 distinct hashes. No discrepancy.

**Data roles against redesign section 7.** They match: calibration 2k; development 256/1k/10k; ImageNet confirmation the 40k complement of the 10k list, with full 50k also reported; COCO confirmation the 4k complement of the 1k list, with full val5k also reported; paired image ids; three seeds, five where unstable. Two cautions. The redesign is itself marked "proposal, not an adopted campaign or D4 decision", so these roles are not yet an owner decision. The later roadmap assessment (item 6) words it the other way round: full 50k and 5k for full-benchmark claims, with "the audited unused complement" reported separately; the owner should say which is primary.

**Internal consistency of the statistical plan (not changed; for the owner and the next draft).**

1. Primary contrasts are few (at most 12) but not named. They are deferred to the promoted set. The plan cannot be hashed or signed in this state, and the split of the 12 among H1, H2 and H3 is not given.
2. H3 does not fit inside 12 contrasts as written. Finding k* needs one test per width per format (for example 4 formats by 6 widths is 24). Either declare a fixed-sequence test from wide to narrow, which needs no multiplicity adjustment, or raise the count.
3. H3 puts fixed point with saturation, FP16 and custom float on one "smallest width" axis. These are different accumulator types; "differs by at least 2 bits" is defined only within one type. The supported and falsified outcomes are otherwise exhaustive (integer widths: at most 1 bit apart, or at least 2).
4. H1 and H3 do not say whether "exceeding delta" and "within delta" refer to the point estimate or to the interval bound. Section 3 states it only for equivalence. Without it, H1 is not testable as written. Section 2 says "Falsified if no pair reverses beyond delta" while section 8 says "report H1 as not supported"; an interval that straddles delta is inconclusive, not a falsification, so the two sections should use one wording.
5. Multiplicity: section 3 names both Holm and max-t simultaneous bands inside a hypothesis. They are alternatives; state which one decides.
6. The detector has a margin (0.5 mAP50-95) but no hypothesis uses it: H1 needs recipe R2 from B2, and the B2 protocol lists YOLO as out of scope; H3 is ResNet18 only. No resolution estimate is given for 0.5 mAP at 4,000 images. The COCO role in confirmation is therefore undefined.
7. H2's cost side has no sampling error, yet H2 receives a third of alpha; only its equivalence part is a statistical test. Acceptable, but say so.
8. Section 4 runs seeds on a "common subset" without naming the data it comes from; if it is confirmation data it counts as a look. Section 9 lists a "three-seed full-set claim" as blocked although section 4 rules such a claim out.
9. Section 4 assumes new seeds need new disjoint 2k training sets. Seeds could also be subsamples of the existing 2k lists, which needs no new data. Owner choice.
10. Section 9 calls audit step 1 blocked for ImageNet. A useful part is not blocked: showing that every ImageNet image ever evaluated lies inside the 10k list by content hash. That can be done now.
11. With the 50k payload unavailable, the only ImageNet evaluation images on disk outside the 1k list are the 9,000 remaining in the 10k list (whether any run has already used them is not audited). Using them as an interim held-out set would depart from redesign section 7, where the 10k list is development data; decision draft P3 wants the 10k list unopened, while sections 5 and 9 here allow its use as development data. One rule is needed.

**Not verified.** Whether any candidate or recipe run touched the 10k list or the COCO 4k complement (name search only: the B-stage analysis cites the COCO 1k list, count 1,000, and names the 5k file only as its selection record); whether torchvision or Ultralytics recipes were tuned on the public validation sets; lane L1, L2, L4 and L5 results beyond the documents named above.
