# B2 format matrix: 25 formats, three classifiers, two frozen recipes (2026-10-01)

Lane L1, stage 2. Protocol: `public/experiments/configs/breadth-study/b2-matrix-protocol-v1.json` (written before
any cell of this stage was measured). Tables: `results/summaries/b2-matrix-v1/` (the document renders the 4-decimal set `*-d4.*` since revision 3). Figures:
`results/figures/b2-matrix-quality-v1.{pdf,png}`, `results/figures/b2-matrix-vs-int8-v2.{pdf,png}` (v2 replaces
`b2-matrix-vs-int8-v1` after review 1: it adds the max-abs activation-block arm of the shared-exponent formats and
marks off-scale estimates consistently; v1 is kept unchanged).

Revision 2 (2026-10-02, after independent review 1, `artifacts/agent_orchestration/handoffs/L1-matrix-review.md`):
corrected the occupancy image set (section 6), named the shared-exponent arm in every INT8 claim and added the
max-abs-arm INT8 table (sections 1, 3, 4.2), stated the operational form of the stress-configuration clause of rule
(d) (section 4.2), corrected the clipped-block statement (section 5), one interval bound, one family label and the
ordering-reversal table (chance-level pairs). No cell, summary table or rule outcome changed; nothing was re-run.

Revision 3 (2026-10-02, after independent review 2): the tables are now rendered from a 4-decimal version of the
summary tables (`results/summaries/b2-matrix-v1/*-d4.*`, written by `b2_matrix --digits 4 --tag d4` from the same
cell records; the 2-decimal v1 set is kept unchanged). Before, a 1-decimal value was the rounding of a 2-decimal
value: the review counted 13 of 372 displayed differences 0.1 off the rounding of the exact value; over all tables
76 of 3154 displayed numbers (estimates, interval bounds, occupancy columns) change, each by one unit of the last
displayed digit. Every class, ordering,
rule (c) score and proposal entry of the d4 set equals the v1 set, and no value moves by more than 0.005. Corrected
with it: section 5 now counts 5 (not 4) of 24 separated search-versus-max-abs pairs (MXFP8 E4M3 default on
MobileNetV3-Large, +0.9 [+0.005, +1.7], was hidden by the rounded lower bound; the table has a new column "interval
excludes 0"), and one bound in section 1 (+0.4, was +0.3). A displayed "-0.0" is a negative value smaller than 0.05.

**Evidence class: development evidence.** Every number below is measured on the frozen 1k ImageNet screen
(`imagenet_screen_1k`), the same list on which the two B2 recipes were selected in stage 1. Nothing here is a
held-out confirmation. No image of the 10k evaluation list beyond the 1k screen and no COCO image was opened.
Intervals are pointwise 95 percent paired image bootstrap intervals (10000 resamples, seed 20260927) without
multiplicity correction; with 1000 images, differences of about 1 to 2.5 points are not resolved.

## 1. Key results

Primary readout is expected-credit top-1 (section 7.1). "Separated" means the pointwise 95 percent paired interval
excludes zero. All 177 cell records exist (3 FP32 baselines, 150 matrix cells of which 42 are stage-1 cells reused
by identity, 24 intrinsic-arm cells of the shared-exponent formats).

- **8 bits.** Under the default recipe Posit8 (es1) is the only non-integer format within one point of INT8 on
  ResNet18 (+0.4 [-0.4, +1.2]) and MobileNetV2 (+0.1 [-0.8, +1.0]); on MobileNetV3-Large it is +1.8 [-0.2, +3.7]
  above INT8 (not resolved). FP8 E4M3, MXFP8 E4M3 (searched activation blocks, the literal recipe) and FP7 E3M3
  are 1.5 to 5.5 points below INT8 on ResNet18 and MobileNetV2 (separated) and within one point on
  MobileNetV3-Large, where INT8 itself is 3.4 [-5.2, -1.5] points below FP32. MXFP8 E4M3 with max-abs activation
  blocks (the arm that enters the proposal, section 4.2) is -1.5 [-2.8, -0.2], -5.7 [-7.7, -3.9] and
  -1.5 [-3.4, +0.4] against INT8 default: separated below on ResNet18 and MobileNetV2 and not resolved (not within
  one point) on MobileNetV3-Large (section 3). Log8 is separated below on MobileNetV2 and not resolved on the other two. FP8 E5M2 is separated below
  everywhere (-8 to -49 points). Q1.6 is an output alias of signed INT8 (section 6) and is not a separate result.
- **6 bits.** No format is within one point of INT8 on all three models under either recipe; the only 6-bit cell
  within one point is BFP6 under the minimal recipe on ResNet18 (-0.7 with searched activation blocks, -0.9 with
  max-abs activation blocks). On MobileNetV3-Large INT6 collapses (21.0 default, 2.5 minimal; the same in stage 1)
  while FP6 E2M3 keeps 58.4 and BFP6 57.1 (56.6 with max-abs activation blocks): against the integer of the same width
  every other 6-bit format is 22 to 37 points above INT6 there (default recipe), and on ResNet18 FP6 E2M3 and BFP6
  are within one point of INT6.
- **5 and 4 bits.** Plain PTQ does not hold at these widths on these models. The best 5-bit cell is 56.9 (FP5 E2M2
  minimal, ResNet18), the best on MobileNetV3-Large is 10.6. Under the default recipe every 4-bit cell is at most
  2.2; the minimal recipe recovers 15 to 42 on ResNet18 only (MXFP4 41.6 with max-abs activation blocks, NF4 34.4)
  and nothing on the MobileNets. Ternary and binary are at chance everywhere. (Reconstruction-based rounding is lane
  L6.)
- **Recipe rule (c).** The rule selected by the protocol is `R_bits(6)`: default recipe for formats of at least
  6 bits, minimal below. Its pick is separated-worse than the other recipe in 10 of 56 eligible cells (mean regret
  2.7 points), against 18 (6.8) for default everywhere and 16 (8.3) for minimal everywhere; the same rule wins under
  the lowest-index readout. No per-family exception met the adoption condition. Eight of the ten losses are
  wide-exponent formats at 6 to 8 bits (FP8 E5M2, FP6 E3M2, MXFP6 E3M2, Log6, each on ResNet18 and MobileNetV2),
  which prefer the minimal recipe there; the other two are the block fixed-point format BFP6 on ResNet18 and
  FP5 E2M2 on MobileNetV3-Large. Posit needs the default recipe: under the
  minimal recipe (max-abs weight scales) Posit8 collapses to 0.0 to 4.4 on all three models (as in stage 1).
- **Ordering (a).** The minimal recipe keeps most of the old (v1) ordering: Spearman 0.85 to 0.97 against the sealed
  v1 1k results, except 0.73 for MobileNetV3-Large against `v1_maxabs`, under which INT8 itself was at 4.1. The
  default recipe reorders it (Spearman 0.49 to 0.73), mostly because Posit moves from collapse to the top; of the
  separated reversals between a v1 ordering and the default ordering, 3 to 19 per comparison do not involve a posit
  format, and none of those is a pair of formats that are both at chance. Some reversal counts do include such
  pairs: 10 of the 27 reversals between the minimal and the default ordering of MobileNetV3-Large are between
  formats that are both below 1 percent in one of the two orderings (for example binary above Posit8 under the
  minimal recipe); the table in section 2 gives this count for every comparison.
- **Ties.** In the cells above 10 percent, 13 to 807 (median 181) of 1000 images have a tie for the largest logit.
  The expected-credit and lowest-index orderings agree closely (Spearman at least 0.955 per model and recipe; largest
  per-cell gap 1.8 points); the recipe rule and the proposal are identical under both readouts. Single-cell
  differences below about 2 points should not be read as format effects.
- **Exact-engine proposal (d), for the owner's sign-off.** Applying the protocol rule gives 2 anchors (FP32, INT8
  default), 5 candidates that pass the quality floor (Posit8 es1, FP8 E4M3, Log8, MXFP8 E4M3 with max-abs activation
  blocks, FP7 E3M3, all default recipe) and, because fewer than 6 passed, 7 flagged stress configurations: 14 in all,
  two more than the 8 to 12 the brief asked for. Two of the seven (NF4, ternary) come from the operational reading
  of the stress clause for families without a passing candidate; the other reading gives 12 (section 4.2).

## 2. The matrix and the ordering of formats (rule a)

Figure `results/figures/b2-matrix-quality-v1.png` shows the whole matrix. Expected-credit top-1 (FP32 in the first
row; rows "max-abs activation blocks" are the intrinsic arm of the shared-exponent formats, section 7.2):

{{T:matrix}}

Paired difference against FP32 of the same model, default recipe, with 95 percent interval:

{{T:vs-fp32-default}}

Minimal recipe:

{{T:vs-fp32-minimal}}

Orderings per model and recipe, by primary top-1 (`>` marks a separated step to the next format, `~` a tie). The
two v1 rows are the sealed old-recipe 1k results (readout `topk`, which carries the tie caveat; formats without a
v1 1k result are missing from those rows):

{{T:orderings}}

Rank correlation between orderings and the number of separated pairs whose order reverses (the last column counts
the reversed pairs in which both formats are below 1 percent in one of the two orderings, i.e. at chance):

{{T:ordering-changes}}

The reversed pairs (first format above the second in the "from" ordering, separated in both):

{{T:ordering-reversals}}

## 3. Per bit width against INT8 (rule b)

Figure `results/figures/b2-matrix-vs-int8-v2.png` shows every format against INT8 of the same recipe, with a
separate row for the max-abs activation-block arm of each shared-exponent format. Summary by
class (within one point; separated below; separated above; not resolved):

{{T:bit-width-classes}}

Against INT8, default recipe:

{{T:vs-int8-default}}

Against INT8, minimal recipe:

{{T:vs-int8-minimal}}

Against the integer of the same width (INT8, INT6, INT5, INT4), default recipe:

{{T:vs-same-width-default}}

Minimal recipe:

{{T:vs-same-width-minimal}}

The shared-exponent rows in the tables above are the searched arm (the literal recipe). The max-abs
activation-block arm, the one a shared-exponent format enters the proposal with, against INT8 of the recipe it
modifies (same classes):

{{T:vs-int8-intrinsic}}

The two arms differ in class in two of the 24 (format, recipe, model) places: MXFP8 E4M3 default on
MobileNetV3-Large (searched -0.7, within one point; max-abs -1.5, not resolved) and BFP6 default on ResNet18
(searched -2.4, separated below; max-abs -1.0, not resolved, the difference being -1.02).

## 4. Recipe rule (c) and the exact-engine proposal (d)

### 4.1 Which recipe, as a rule stated in advance

Default minus minimal per format and model, with interval (`n/e`: neither recipe reaches 10 percent, the cell is
not scored):

{{T:recipe-pairs}}

Scores of the candidate rules fixed in the protocol (primary readout):

{{T:recipe-rules}}

The same under the lowest-index readout:

{{T:recipe-rules-lowest-index}}

Cells in which each rule's pick is separated-worse:

{{T:recipe-rule-lost}}

`R_bits(6)` is selected. The rule was selected on the same screen on which it is scored, so it is development
evidence; it is simple enough to state in advance for a new model (default at 6 bits and more, minimal below), but
its known failure is the wide-exponent formats (E5M2, E3M2, Log6), for which the minimal recipe was better on two of
three models. Why the minimal recipe destroys Posit (max-abs weight scaling places the largest weight at `maxpos`,
so most weights fall into the coarse tapered region) is a plausible mechanism that this stage did not test.

### 4.2 Proposed configurations for the exact engine

**PROPOSAL, not a decision: it needs the project owner's sign-off.** Rule (d) of the protocol: anchors FP32 and
INT8; per (family, bit width) the best format by mean primary top-1 under the recipe of rule (c); quality floor mean
drop against FP32 at most 5.0 points and no model more than 10.0 below; shared-exponent formats judged by their
intrinsic arm (max-abs activation blocks; recipe `cum5_act_maxabs` is the default recipe with that one change); Q1.6
excluded as an alias of signed INT8; with fewer than 6 passing candidates, per family the best failing format of the
next lower bit width is added as a flagged stress configuration.

Operational form of the last clause (it is not spelled out in the protocol; this is how `tools/analysis/b2_matrix.py`
applies it, `propose()`): for a family with a passing candidate, "the next lower bit width" is the widest width below
its lowest passing width; for a family with no passing candidate, the family's widest failing candidate is taken (as
if the floor were above every width). The second case is the one the protocol does not define, and it decides three
entries: the integer family (INT8 is the anchor, not a candidate, so the family has no passing candidate and INT6 is
added), codebook (NF4, its only width) and binary/ternary (ternary; binary ties it at chance and has fewer bits).
Under the other natural reading (no stress entry for a family without a passing candidate, with the INT8 anchor
counting as the integer family's passing member) the stress set is FP6 E2M3, BFP6, INT6, Log6 and Posit6 es1, 12
entries in all. Neither reading was written down before the results; the table shows what the code applied.

MXFP8 E4M3 enters through its max-abs activation-block arm; its INT8 comparison is in the arm table of section 3
(-1.5, -5.7, -1.5 points; separated below on ResNet18 and MobileNetV2), not the searched-arm rows.

{{T:proposal}}

All candidates (best format per family and bit width):

{{T:candidates}}

Only 5 candidates pass, so the rule adds 7 stress configurations and the set has 14 entries. Two of them, NF4 and
ternary, are at or near chance on all three models (mean drop 60 and 72 points); an exact run of a configuration
that does not classify tells nothing about the format. Suggestion for the owner (made after seeing the results, so
not part of the rule): drop those two, which leaves 12 (2 anchors, 5 passing, 5 stress at 6 bits: FP6 E2M3, BFP6,
INT6, Log6, Posit6 es1); this is also the set the other reading of the stress clause gives (above). The proposal and the stress set are the same under the lowest-index readout.

## 5. Shared-exponent formats: search versus intrinsic arm, and scale metadata

Searched activation blocks (the literal recipe) against max-abs activation blocks (intrinsic arm), with the
median fraction of stored-activation blocks whose exponent the search lowered:

{{T:shared-arms}}

The activation block search changes little: it is separated on 5 of 24 pairs, in both directions (BFP6 default on
ResNet18 and MobileNetV2 lose 1.4 and 4.1 points with the search, BFP6 minimal on MobileNetV3-Large gains 5.1, MXFP8
default on MobileNetV3-Large gains 0.9 with a lower bound of +0.005, MXFP6 default on MobileNetV2 loses 1.2). For MXFP8 the search almost never lowers a block exponent: the median fraction of
stored-activation blocks lowered per tensor is at most 0.00003 in the six searched MXFP8 cells (the table rounds it
to 0.0), and the largest fraction in any one tensor is 0.0020 (MobileNetV3-Large, both recipes), as expected for a
format whose relative precision is the same in every binade (cell records, `occupancy.nodes[*].fraction_blocks_clipped`).

Scale metadata (8 bits per block of 32, section 7.2), in scale bits per element:

{{T:shared-metadata}}

## 6. Readout, ties and the stage-1 cells

Images with a tie for the largest logit per cell, and in brackets lowest-index minus expected-credit top-1:

{{T:ties}}

Sensitivity of the orderings to the readout rule:

{{T:readout-sensitivity}}

The 45 stage-1 cells (42 quantized, 3 FP32), executed again for the readout: each reproduced its sealed
configuration identity and 1000 of 1000 sealed top-5 lists; the 27 among them that the r3c tie table covers agree
with it. Sealed `topk` top-1 against the two readouts:

{{T:stage1}}

Q1.6 is INT8 scaled by 1/64. Its minimal-recipe readout is identical to INT8 minimal on all three models, and its
default-recipe top-5 lists equal the sealed INT8 `default_signed` top-5 lists on 1000 of 1000 images on all three
models (Q1.6 has no unsigned variant). It is reported as an alias and excluded from the proposal.

Code occupancy (the first 128 images of the 1k screen `imagenet_screen_1k`, the images the cell is evaluated on,
not calibration images; median SQNR over audited tensors, median entropy of the code histogram,
median fraction of levels used, largest fraction of codes at the two extreme levels), default recipe:

{{T:occupancy-default}}

Minimal recipe:

{{T:occupancy-minimal}}

In the MobileNetV3-Large `binary_pm1` default cell one audited FP32 tensor is identically zero while its binary
copy is not (binary has no zero code), so its SQNR is minus infinity; that tensor is excluded from the SQNR median and
marked in the cell record (`sqnr_db_nonfinite`), see section 8.

## 7. Method

### 7.1 Readout (fixed for all B2 reporting from this stage on)

Quantized logits tie, and `topk` breaks ties in an unspecified order. Two rules are fixed
(`tools/experiment_b2/readout.py`, version `b2-tie-readout-1`):

- primary, **expected credit**: top-1 credit `1/k` when the label is among the `k` classes that share the largest
  logit, else 0; top-5 credit `min(1, (5-g)/e)` when `g < 5`, with `g` classes strictly above the label logit and
  `e` classes equal to it (label included);
- secondary, **lowest class index**: classes ordered by (logit descending, class index ascending).

Per image the record keeps `label`, `greater`, `equal`, `equal_lower`, `tie_size` (uint16), the lowest-index
argmax and top-5 and the CUDA `topk` top-5 (int16), one compressed npz per cell (about 10 KB). Both rules for
top-1 and top-5 are functions of these counts. Batch size 8 is part of every configuration identity
(`PROTOCOL.inference_batch_size`); ResNet18 predictions depend slightly on it (stage 1, r3).

Stage-1 results post hoc: the sealed stage-1 records hold only `topk` lists, so each of the 42 stage-1 cells
(and the three FP32 baselines) was executed once more; a cell had to reproduce its sealed configuration identity
and 1000 of 1000 sealed top-5 lists before its readout record was written. All 45 did (section 6). The sealed v1
results (old recipes) keep their `topk` readout because no logits were kept; they carry the tie caveat.

### 7.2 Shared-exponent formats in B2

`bfp6`, `mxfp4_e2m1`, `mxfp6_e3m2`, `mxfp8_e4m3` use the block semantics of `tools/experiment_b_ext/shared.py`
unchanged (one E8M0 power-of-two scale per block of 32 elements; stored activations blocked along the channel
axis; convolution patches and weights blocked along the reduction axis K; FP32 accumulation), implemented in the
new module `tools/experiment_b2/blocks.py` under B2's rules:

- boundaries: the B2 plan (fused across ReLU) with one change: a max-pool output is re-blocked instead of
  forwarding codes, because its elements come from blocks with different exponents;
- range rules: `maxabs` and `percentile_99_9` are the extension's rules run by the extension's code; `mse` is the
  B2 scale search restricted to E8M0 scales: per block the exponent `e_maxabs - d`, `d = 0..6`, with the smallest
  block squared error (smallest `d` on ties);
- no unsigned variant (as for every non-integer format); bias correction is the B2 sequential empirical
  correction on the block engine (first 256 calibration images).

Two arms per shared-exponent cell. The matrix cell translates the recipe literally (`default`: searched
activation and weight blocks plus bias correction; `minimal`: searched activation blocks, max-abs weight blocks).
The activation block search is a run-time operation (seven trial encodings per block) that scalar formats do not
need, so every shared cell is also measured with activation blocks scaled by the intrinsic max-abs rule (recipes
`cum5_act_maxabs` beside `default` and `cum1_fused` beside `minimal`). A shared-exponent format may enter the
exact-engine proposal only through this intrinsic arm.

**Regression (B2 switches off = old extension).** With v1 boundaries, no bias correction and both rules `maxabs`
(or both `percentile_99_9`) the block engine gives bit-identical logits to `prepare_shared` on 1000 of 1000
screen images, bit-identical weights, and reproduces the sealed extension top-5 lists on 1000 of 1000 images, on
all six sampled cells (ResNet18 bfp6 maxabs, ResNet18 mxfp4_e2m1 percentile, MobileNetV2 mxfp6_e3m2 maxabs,
MobileNetV2 bfp6 percentile, MobileNetV3-Large mxfp8_e4m3 maxabs, MobileNetV3-Large mxfp6_e3m2 percentile);
records `artifacts/experiment_b2/matrix/regression/*--r2.json`. The first run of this regression passed five
cells and lost the sixth to CUDA out-of-memory in the vectorised path; `blocks.py` was then changed to process
blocks in slices and all six were re-run with tag `r2` (the five untagged records are kept, superseded).
The frozen B2 numerics are untouched: no existing file under `tools/experiment_b2/` was edited, the B2 source
identity is still `3966af3b...`, and the stage-1 regression tests pass (section 8).

**How block scale metadata is counted.** Eight scale bits per block of up to 32 elements; an incomplete block
still carries one scale. Reported per model as scale bits per weight element (blocks along K per output
channel), per stored activation element (blocks along the channel axis per spatial position) and per
convolution-patch element (transient MAC-operand blocks: these scales exist in the datapath and are not stored
with the feature map). Formats are grouped by their nominal element bits; effective bits are tabulated beside
them (section 5).

### 7.3 Statistics and rules

Per cell: paired difference against FP32 and against INT8 of the same recipe on the per-image expected credit
(`tools.analysis.b2_ties.paired`), top-5 likewise; under the lowest-index rule the same comparisons with
`tools.analysis.b_stage_balanced_comparisons.paired_outcomes` (binary outcomes, exact McNemar). The rules for
(a) ordering, (b) bit-width classes, (c) the recipe rule and (d) the exact-engine proposal were written into the
protocol before measuring (`analysis_rules_stated_in_advance`) and are applied by `tools/analysis/b2_matrix.py`
as written. One sentence of rule (c) needed an operational form, which was fixed while the campaign was
running, after the ResNet18 cells and part of the MobileNetV2 cells existed and before any quantized
MobileNetV3-Large cell existed: a per-family exception replaces the selected rule by another candidate rule for
one family when, on every format of that family where the two rules pick different recipes, the other rule's
pick is better with an interval excluding zero on all three models (`family_exceptions`). This is disclosed
because it was not fixed before the first results were seen.

## 8. Execution record

- Code added in this stage (no existing file edited): `tools/experiment_b2/{readout,blocks,matrix}.py`,
  `tools/experiment_b2/matrix_finite.py`, `tools/run/experiment_b2_matrix{,_finite}.py`,
  `tools/analysis/b2_matrix{,_figures,_report}.py`, `tools/analysis/b2_matrix_doc_template.md`,
  `tests/unit/test_experiment_b2_matrix{,_analysis,_finite,_report}.py` (`_report` added in revision 2, extended in revision 3).
- Tests: `.venv/bin/python -m pytest tests/unit/test_experiment_b2*.py tests/unit/test_experiment_b.py -q`
  (110 passed on 2026-10-01 at about 23:35, of which 27 are this stage's matrix, analysis and finite-SQNR tests; the glob also collects other lanes' `test_experiment_b2_*` files); the stage-1 tests among them prove that the frozen numerics are unchanged. At revision 2: 118 passed (29 of this stage), B2 source identity recomputed `3966af3b...`. At revision 3: 121 passed (32 of this stage), source identity again `3966af3b...`.
- Cells: `artifacts/experiment_b2/matrix/cells/` (177 sealed records: 3 FP32 baselines, 150 matrix cells, 24
  intrinsic-arm cells), readout records `matrix/readout/`, configuration pre-images `matrix/configurations/`.
- Commands: `campaign baseline`, `campaign regress`, `campaign matrix`, `campaign intrinsic` of
  `.venv-b/bin/python -m tools.run.experiment_b2_matrix` (every job under
  `artifacts/agent_orchestration/gpu_run.sh`); then `.venv/bin/python -m tools.analysis.b2_matrix`,
  `.venv/bin/python -m tools.analysis.b2_matrix_figures`,
  `.venv/bin/python -m tools.analysis.b2_matrix_report --assemble`. Revision 2 added only
  `.venv/bin/python -m tools.analysis.b2_matrix_figures --version v2 --only vs-int8` (new figure files; the v1 files
  are unchanged and the v1 code path still reproduces them byte for byte) and re-assembled this document; the
  summary tables are the v1 set, unchanged. Revision 3 added `--digits` to `b2_matrix` (default 2 reproduces every
  v1 csv byte for byte), wrote `.venv/bin/python -m tools.analysis.b2_matrix --digits 4 --tag d4` (new files
  `results/summaries/b2-matrix-v1/*-d4.*`) and assembles this document with
  `.venv/bin/python -m tools.analysis.b2_matrix_report --assemble --tag d4`.
- Context (review 1, finding 9; not this lane's change): the configuration pre-image
  `artifacts/experiment_b2/configurations/b7c945da...json` of the reused ResNet18 Posit8 default cell was re-sealed
  at 21:16 by lane L8's use of `export.py`; the digest of its payload still equals its name.
- Failures during execution, as they happened: the first job chain (`matrix/logs/chain-r1.sh`) lost 7 of its
  first 17 jobs to CUDA out-of-memory while other lanes filled the shared GPU; it was replaced by
  `chain-r2.sh` (same jobs, each waiting for 4500 MiB of free GPU memory, repeated passes over what is
  missing). chain-r2 ran from about 20:05 to 23:24 (+05): of its job runs, 97 exited 0 and 19 failed; 17 failures were CUDA out-of-memory and every one of those cells completed on a later pass; 2 were the same job, MobileNetV3-Large `binary_pm1`, failing deterministically at sealing (`ValueError: Out of range float values are not JSON compliant: -inf`, from a minus-infinity SQNR in the frozen occupancy audit). Its two cells were then run with the new wrapper `tools/run/experiment_b2_matrix_finite.py` (same job, non-finite SQNR stored as text, its own hash added to `own_sources`; configuration identity, readout and logits do not depend on it); their records list `matrix_finite.py` in `own_sources`. The chain's ledger also holds 5 exit-2 lines from running an empty job line after nothing was missing (harmless). The three FP32 baseline records were written before `blocks.py` and `matrix.py` reached their final versions, so they carry earlier hashes of those two files; FP32 does not use the block engine, and all three reproduced the sealed stage-1 baseline top-5 lists on 1000 of 1000 images. A failed job writes no cell record (the failed `binary_pm1` job had already written its readout and configuration files; the rerun compared its readout with them array by array and reused them); every cell record in the tables comes from a job that
  completed, and the campaign only runs cells that have no record.
- No seconds-per-image figure is reported from this stage (shared GPU slots).
- Disk added by this stage: 23 MB under `artifacts/experiment_b2/matrix/` (cell records 7.2 MB, readout records 3.7 MB, configuration pre-images 12 MB gzip), 160 KB of tables (plus 160 KB for the 4-decimal set of revision 3) and 792 KB of figures (v1 and the v2 replacement of one figure); the stage-1 input cache was neither enlarged nor changed.
