# Scaled-bridge contract 2.3 — DRAFT - not frozen

Lane E1, stage 1, 2026-10-02. Nothing in this draft has been run on an image, exported or gated. No engine source,
archive or run root was changed. It extends contracts 2.0, 2.1 and 2.2 (`docs/analysis/scaled-bridge-v2-contract-2026-10-01.md`). Every rule there
stays in force unless a section below replaces it. The draft becomes a contract only after the owner has made the
decisions listed at the end and it has been sealed before any measurement.

Reference models (part B): `tools/scaled_bridge_ext/` (pure Python integers and Fractions, CPU). Tests:
`.venv/bin/python -m pytest -q tests/unit/test_scaled_bridge_ext*.py` (96 passed, about 7 s; revised after reviews 1 and 2, see 'Revision r2', 'Revision r3' and 'Revision r4' at the end). The stored vectors of the
existing oracles are in `artifacts/scaled_bridge_ext_v1/vectors/existing-policies-v1.json.gz`: 601 cases, 85 KB, written
once, with the sha256 of the oracle sources recorded inside; a supplement (`existing-policies-v1-supplement-1.json.gz`, 53 cases: `sat.abs/struct` with 2.1 certificate fields, counters at and beyond the 65,535 cap) was added in r2. Where a choice depends on published design points, the
source is lane P1's `docs/analysis/closest-prior-work-accumulators-2026-10-02.md`, part B, rows #1 to #13. Since 18:17 that document reads 'DRAFT - NOT FINAL' with four uncorrected blocking errors (P1 review 1); row #11 has no source ([Joh18] narrows only the MSB side; [Cuy26] is a float partial result), so every choice that rests on row #11 is marked [PENDING P1].

## A0. Conventions

- **Product grid.** The unit is 2^-(s_a + s_w) code level. s_a and s_w are the input and weight codebook shifts, as in 2.1.
- **Term.** A term is one exact rational in product-grid units. For a scalar format a term is the product a·w of two code-grid integers. For a shared-exponent format a term is a block sum (A3).
- **Summation sequence.** The terms in the policy's order. The default order is 2.1's: input channel, then kernel row, then kernel column. A padded tap is a zero term.
- **Final conversion.** Unchanged from 2.1. The final register or accumulator value goes to binary64 with one RNE (exact below 2^53 units) and is then scaled by an exact power of two. The post-operations, stores and top-k are 2.0 to 2.2.
- **Non-finite results.** 2.1's failure rule now covers NaN as well as ±inf. NaN arises from +inf + (−inf): at a tree node, in a float or `wide` outer stage of `ch` (e.g. `ch2_flt.e2m1_wide`), or when a bias converted to −inf meets a +inf state. The image fails: it has no prediction, is scored wrong, and the record names the node and counts the outputs. Nothing is ever clamped or replaced.
- **Policy names.** A name is the complete specification. The grammar is in `tools/scaled_bridge_ext/policies.py`. Unknown or non-canonical names fail closed. The 2.0 and 2.1 names keep their meaning. `fp16` and `f21` are aliases of `flt.e5m10` and `flt.e8m12`, and `control` equals `flt.e8m23` under its 2.0 operand precondition. Tests show this on random vectors. Two further names are second spellings of one arithmetic, kept because the modifier matters elsewhere: `wide.elt` equals `wide` on a block node (both exact; `.elt` changes the chunks of `ch<C>_wide.elt_…`), and `sat.w<W>.rne`/`wrap.w<W>.rne` equal the plain names on scalar nodes (integer terms; `.rne` changes block terms). A record keeps the name as written (`test_a0_second_names_are_the_same_arithmetic`).

## A1. Image-list parameter

### Where the list is fixed

| Engine | File:line | What it does | Numeric source (in the digest)? |
|---|---|---|---|
| v2 (archives 7c6344af, 1f75c923) | `tools/scaled_bridge_v2/worker.py:30-32` (`context`) | `dataset('imagenet_screen_1k')`; refuses if the digest of the ordered rows ≠ `export.retained.ordered_samples_sha256` | yes (`tools/scaled_bridge_v2/common.py:40` `engine_sources()`: every `.py`, `.cpp`, `.cu` and `.h` file of the package except `NON_NUMERIC_FILES` = report, archive, b2_adapter, b2_replay; plus `manifests/*.json`, `tools/scaled_bridge_v1/common.py`, `public/inference/reference/arithmetic.py` and the fp16/fp32/fp64 accumulator manifests) |
| v2 | `worker.py:155` (`predict`) | refuses ranges outside `len(rows)` ("outside the development screen list") | yes |
| v2 | `worker.py:188` | record label `'list': 'imagenet_screen_1k (development)'` | yes |
| v2 exporters | `export.py:50`, `b2_adapter.py:207` | the retained list of an export (B anchor, B2 replay) | `export.py` is in `EXPORTER_FILES` (export identity) and, not being in `NON_NUMERIC_FILES`, also in the engine digest. `b2_adapter.py` is non-numeric. Neither needs to change. |
| fast (edfecd5c) | `tools/scaled_bridge_fast/worker.py:30-32` (`context`), `:132`, `:151` | same three roles | yes (`fast_sources()` takes every `.py/.cu/.h` file except `archive, profile, cli, __main__, validate, timing`; the fast identity is `fast_sources()` plus the v2 base digest `BASE_DIGEST` 1f75c923…, `tools/scaled_bridge_fast/common.py:38-44`) |
| fast | `profile.py:130` | profiling only | no |
| v2 gate paths | `replay.py:57`, `graph_witness.py:346` | B replay and the graph witness | yes; they stay on the screen list (gates are screen-only) |

Predict files are written to `runs/<digest>/<case>/predictions/<policy>-<backend>-<start>-<stop>.json`. The path has no list name in it.

### Smallest change (in a new package only; A1 and C1)

1. `context(case, policy, list_name='imagenet_screen_1k')` and `predict(..., list_name=...)`. The CLI gets
   `--list {imagenet_screen_1k, imagenet_heldout_9k}`, which is a closed set. `imagenet_heldout_9k` = the rows of
   `imagenet_evaluation_10k` record of `data/manifests/index.json` (its `path`, file sha256 and `logical_payload_root`
   `imagenet1k/val_evaluation_10k`; never the screen's root) whose sha256 is not in `imagenet_screen_1k`. Order: ascending
   sha256, the order `tools/experiment_b/common.py:100 dataset()` gives every list and every existing screen record
   (P4's draft 2, section 2 'Data roles', says manifest order: "rows in manifest order `03bb8753…`" and "Every held-out
   run iterates the complement rows in full-manifest order", lines 81 and 86 of the file as read in r4; the file is
   still being revised and its line numbers have moved twice, so this draft cites the sentences). One of the two orders
   must change: owner decision D2, recommended sha256 order.
   Identity check: P4's set digest, sha256 of the sorted sha256 column newline-joined with a trailing newline
   (`d47669bb…`), must equal the sealed value; each record also stores `digest(rows)` of the ordered rows. P4's
   manifest-order digest `03bb8753…` is P4's own function of the rows, not the engine's `digest(rows)`: D2 names one. The engine
   refuses this list unless the sealed protocol file exists and its sha256 matches a constant in the run command. This
   lane never reads that list.
2. Export check. The export's `retained.ordered_samples_sha256` must still equal the screen-list digest, since it
   identifies the export and is unchanged. The run's own list digest is checked separately, as in item 1.
3. Paths and labels. The screen list keeps today's path, so existing records stay addressable. Any other list writes under
   `<case>/predictions/<list_name>/` and labels the record `'list': '<name> (<role>)'`. The role is `development` for the
   screen and `held-out confirmation` otherwise.
4. The comparison with sealed batch-1 records (`compared_with_sealed_batch1`) is made on the screen list only.

Only `worker.py` (and `common.py`, if the identity gets a list field) changes. Both are numeric sources, so **the digest
changes**. Under S1-review verdict (d), a change confined to `worker.py` made in place would need re-checks 2 to 5 but
not check 1: S1's regress, trace and compare sets, then `gates` and `archive`; D's `check_archive.py` and
`trace_gates.py`; the sealed 1k comparisons; and the archive-referenced regimes. Stage 2a is not made in place: the new
package builds its library under its own root, so it also runs check 1 (kernel PTX identity and A's t1, about 2 GPU
minutes) although `kernel.h` and `fastdot.cu` are byte-identical (C1; C2 items 1 to 6). The list parameter still gets
its own digest (stage 2a), because 2b changes `kernel.h`, `fastdot.cu`, `native.py` and `engine.py`, and a review of the
list change alone is small.

### Gate on the screen list alone (no held-out image is read)

- **G-L1.** For every case and policy family of S1's compare sets, `--list imagenet_screen_1k` must give predict `images`
  and `nodes` byte-equal to the sealed edfecd5c predict files and to the archives' own (7c6344af/1f75c923), on images
  0–63 at batch 1, 8 and 32. It must also give 1,000-image files equal to the sealed 1k wide/control files of the 10 MobileNet
  cases and the 18 ResNet18 pairs.
- **G-L2.** A test-only list alias with the screen rows under another name gives identical `images` and a different path
  and label. This shows that nothing numeric depends on the list name.
- **G-L3.** Unit tests with the loader mocked. `imagenet_heldout_9k` is refused when the protocol file is missing or does
  not match, or when the rows digest differs. A range beyond the list length is refused. The held-out tsv is never opened
  in a test (the mock raises if it is).

## A2. Accumulator policies

All of them run through `tools/scaled_bridge_ext/reference.py` (`dot`, `dot_terms`). Events are counted per output,
and per node they are added up as in 2.1.

### A2.1 Wrap-around register: `wrap.w<W>`, `wrap.abs-<d>`, `wrap.struct-<d>`

The register is a signed two's-complement integer of W bits (2 ≤ W ≤ 127) on the product grid. It starts at 0 and, in
summation order, adds each term exactly. The exact sum s is then replaced by ((s + 2^(W−1)) mod 2^W) − 2^(W−1). There is
no clamp and no overflow flag in the datapath. Events: `high` and `low` count the adds whose exact sum lay above or below
the range (each wrap). The final value is converted as in 2.1.

- **Hardware.** A plain W-bit adder that drops its carry out. This is the cheapest register. WrapNet [Ni20] uses it,
  and so do [Nat25]'s "transient overflow" and P1 row #7.
- **No-event condition and certificate.** The no-event condition is the 2.1 prefix bound: the certified absolute and
  structural widths apply unchanged. `wrap.struct-0` must reproduce `wide` bit for bit with no event. Wrap differs from
  sat in one way: **the result is exact whenever the final sum fits in W bits, whatever the prefixes did.** The test
  `test_wrap_is_exact_when_the_final_sum_fits` shows this. An event is therefore not an error. The record adds
  `wrapped_outputs_exact`, the outputs with at least one wrap whose final value equals the exact one. The kernel computes
  this with a parallel exact sum, which costs one more 64-bit or two-limb register per thread (owner decision D11).
- **Operand limits.** One limb for W ≤ 63 and two 64-bit limbs for 64 ≤ W ≤ 127. Widths above 63 are needed to bracket
  FP8 E5M2, whose certified width is 75 (L8: "cannot be bracketed" at W ≤ 63).

### A2.2 Dropped low bits: `sat.w<W>.l<L>[.rne]`, `wrap.w<W>.l<L>[.rne]`

The register grid is 2^L product units (L ≥ 1). Each term t enters as the integer round(t / 2^L):

- default (no suffix): **floor**, toward −∞. This is what dropping L low bits of a two's-complement product does (an
  arithmetic shift), so it is the cheapest hardware.
- `.rne`: nearest, ties to the even integer.

The integer is added exactly, then the overflow rule (sat or wrap) applies. The final value is R·2^L units. Events:
`high`, `low`, and `inexact_terms` (terms that lost bits; this is not a failure).

- **Hardware.** The multiplier output is truncated or rounded to the accumulator's LSB, giving a fixed-point accumulator
  with fewer fraction bits. Examples are [Joh18]'s [fmin, fmax] window and [Cuy26]'s 16-bit partial result (P1 row #11).
  [PENDING P1: row #11 has no source after P1 review 1; the two examples above are withdrawn as hardware evidence and the policy stands as a design point of its own.] P1 suggests a register `fx.w<W>.f<F>` whose LSB is 2^-F code level. That is `.l<L>` with L = s_a + s_w − F on each node.
  This draft names the grid relative to the product grid, which is unambiguous per node. A code-level-relative alias is
  owner decision D3.
- **Certificate (no saturation event).** One definition for every integer stage, implemented in `tools/scaled_bridge_ext/certify.py`
  (its docstring is normative): per channel, [lo, hi] = the sum of the per-term ranges (2.1 abs or struct); entry onto
  g = 2^L gives [ceil(lo/g), floor(hi/g)] when every term is a multiple of g (scalar terms, L ≤ 0), else
  [floor(lo/g) − n + 1, floor(hi/g)] for floor and [ceil(lo/g − n/2), floor(hi/g + n/2)] for RNE (n = terms of the stage);
  W_cert = max over channels of the signed width of that range. A relative name `abs-<d>`/`struct-<d>` means
  max(2, W_cert − d) of ITS stage (with its L, chunk and bias), so `sat.abs-0.l4` is narrower than `sat.abs-0`; for a
  plain register W_cert equals the 2.1 field and the 2.1 certificate dict is accepted only for plain names. Low-bit loss has no certificate but
  a bound: when no overflow occurs, |result − exact| ≤ K·(2^L − 1) units for floor and ≤ K·2^(L−1) for RNE.
- **Operand limits.** As A2.1. −126 ≤ L ≤ 126, L ≠ 0 when written (refused otherwise); `.rne` without `.l<L>` is RNE entry at L = 0 (it matters only for shared-exponent terms).

### A2.3 Generic float accumulator: `flt.e<E>m<M>[.x<e>][.ftz][.ofsat][.nf][.rz][.pr-e<E'>m<M'>]`

- **Layout.** Sign bit; E exponent bits (2 ≤ E ≤ 11, bias 2^(E−1) − 1, top exponent code reserved for inf/NaN); M fraction
  bits (1 ≤ M ≤ 52); P = M + 1. Smallest subnormal 2^(2 − 2^(E−1) − M), largest finite (2^P − 1)·2^(2^(E−1) − P).
  `.x<e>` (|e| ≤ 64, nonzero when written) is 2.1's scale exponent with 2.1's sign: the accumulator holds the code-level
  value times 2^e (2.1: state = R(state + a·w·2^e)), and the finite result is multiplied by 2^−e exactly. The
  representable code-level values are therefore the format's numbers times 2^−e; `fp16.x-12` reaches 65,504·2^12. The kernel works in
  product-grid units with emin = 2 − 2^(E−1) − M + s − e and emax = 2^(E−1) − 1 + s − e, where s = s_a + s_w. This is the
  existing `SoftFloat` parameterisation, so (P, emin, emax) reach the kernel unchanged. A paper's exponent bias b
  (numbers 1.m·2^(code − b)) is the IEEE layout with e = b − (2^(E−1) − 1), provided the paper's value unit is the code level.
- **Step.** From +0, state = R(state + t) for every term: the exact product and **one rounding per multiply-add** (fused,
  model C). R is round-to-nearest-even with gradual subnormals. A value that rounds above the largest finite value becomes
  ±inf, and an infinite state stays infinite. Zero terms leave the state unchanged. Zero is canonical +0.
- **Variants** (each a modifier):
  - `.ftz`: flush to zero with tininess detected BEFORE rounding: an exact sum (or converted bias, or `.nf` product) whose
    magnitude is below the smallest normal becomes +0; every other value rounds to P bits (E4M3: 61/4096 → +0).
  - `.ofsat`: overflow gives the largest finite value of its sign.
  - `.nf`: not fused. The product is first rounded to the accumulator format, so there are two roundings per step (model B
    with the accumulator format).
  - `.rz`: every rounding is toward zero, and overflow then gives the largest finite value, as in IEEE (`.ofsat` is
    refused as redundant).
  - `.pr-e<E'>m<M'>[-x<e'>][-rz][-ftz]`: every product p (an exact code-level value) is first replaced by
    R'(p·2^e')·2^−e', where R' rounds to the IEEE layout E'/M' (top exponent code reserved, as for `flt`). The sign of e'
    is that of `.x<e>`, and e' is independent of the accumulator's `.x<e>`: the product is rounded in code-level units,
    never in accumulator units (200·100 under `flt.e5m10.x-12.pr-e4m3` becomes 240, the largest E4M3 value). R' is RNE,
    or toward zero with `-rz`; the accumulator's `.rz` does not act on it (`flt.e8m23.rz.pr-e2m1` rounds the product 1.75
    to 2.0, `…pr-e2m1-rz` to 1.5). Subnormals of the product format are gradual, or flushed with tininess before
    rounding under `-ftz` (as `.ftz` below). Overflow always saturates to the largest finite value of its sign. A product
    rounding counts in no event field. It models a rounded product (model B with a product format; P1 row #9) and is
    also allowed on `wide` and integer registers, not on `control` or an outer stage. A paper's product bias b maps to
    e' = b − (2^(E'−1) − 1), as for `.x<e>`.
- **Events** (normative definitions in the docstring of `reference.py`). Every sequential step, tree adder, lone tree root
  and bias-post add is one rounded add R(x + y). `nonfinite_steps`: results ±inf/NaN, including steps that carry a
  non-finite state (the 2.1 rule; zero terms count then); `overflow_steps`: finite operands whose unbounded rounding
  exceeds the largest finite value (inf, or saturation under `.ofsat`/`.rz`); `inexact_steps`: finite results ≠ exact
  sum; `absorbed_steps`: finite operands, result equal to one operand while the other is nonzero (swamping, both
  directions; under `.ftz` a nonzero term flushed to +0 against a +0 state counts). The term of a step is the product
  after `.nf`/`.pr-` rounding; in a sequential order a term that is zero (also one rounded to zero) is skipped with no
  add and no event while the state is finite. The skip is value-neutral because a sequential state is always a number
  of the format. **A tree has no skip** (r4, review 2 B11): a zero leaf (a padded tap or a zero product) enters its
  adder like any other leaf, and the adder rounds its output, R(x + 0) = R(x). Leaves are exact, unrounded products,
  so an adder with a zero operand is inexact whenever the other leaf is not a number of the format: `flt.e3m2.ord-tree`
  on leaves 9, 0, 1, 0 gives R(9 + 0) = 8, R(1 + 0) = 1, R(8 + 1) = 8, result 8 with `inexact_steps` 2 and
  `absorbed_steps` 1 (skipping the zeros would give R(9 + 1) = 10); under `.ftz` such an adder can flush
  (`flt.e4m3.ftz.ord-tree`: 61/4096 + 0 → +0, inexact and absorbed). This is what an RTL adder tree does: every adder
  rounds and none tests for zero. Padding therefore decides values under `.ord-tree` at border outputs. Tests
  `test_a25_tree_adder_with_a_zero_leaf_rounds`, `test_a23_tree_flush_against_a_zero_leaf_is_absorbed`. Product roundings and the bias conversion count in no field. Integer stages: `high`/`low` per fit
  (each add, tree leaf entry, tree adder, preloaded bias, bias-post add), `inexact_terms` per term that lost bits.
- **Certificate (lossless).** Suppose P ≥ W_cert − 1, the largest finite value is at least B units, where
  B = max over channels c of max(|lo_c|, hi_c) for the certified channel range [lo_c, hi_c] of A2.2 (the sum of the
  per-term abs or struct ranges, which contains every subset sum), and the subnormal
  spacing is no coarser than the product grid (emin ≤ 0 in grid units). Then every subset sum is an integer below 2^P
  units, so every step is exact and the result equals `wide` in every order. This holds for `.nf`, `.rz` and `.ofsat`;
  for `.ftz` it needs in addition the smallest normal ≤ 1 grid unit (review 1 counterexample: `flt.e4m7.ftz`, five
  1-unit products at shift 13, gives 0). It does not cover `.pr-` or a bias. Tests
  `test_lossless_condition_holds_for_every_variant` and `test_ftz_counterexample_of_review`.
- **Widths that bracket a failure.** These are proposed, not measured. E = 8 isolates precision: for every admitted format
  and x = 0, flt.e8m* has no overflow and no underflow. The FP8 E5M2 grid is 2^-32 code level against a subnormal spacing
  of at most 2^-127, and its bound is about 2^43 code level against 2^128. The proposed ladder is
  `flt.e8m{2,3,4,5,6,7,8,9,10,12}` per case. The upper anchor M_lossless = W_cert − 2 is a guarantee, not the failure
  point. P = 11 (fp16 rule) and 13 (f21) changed every output but moved Top-1 by at most 0.8 points on ResNet18 at
  1k, with one of 16 expected-credit intervals excluding 0 (L8; "within 1 point", not lossless, as P1 review 1 corrects).
  Binary32 (P = 24) is bit-identical to exact for INT8/INT6/FP6/FP7 at 1k. [Blu24]'s E4M7 fails under PTQ (P1 row #4), so failure is expected at M ≤ 7 for some formats.
  Then an exponent ladder `flt.e{4,5,6}m<M*+1>.x<rule>` at the located M* exposes range failures. Locate both on the
  screen-1k list first (development), as L8 did; they go to held-out confirmation only by the P4 protocol.

| Case (B2 default, ResNet18) | W_cert abs (L8) | M_lossless (guaranteed) | Proposed M ladder |
|---|---:|---:|---|
| INT8 | 27 | 25 | 2–10, 12 |
| INT6 | 23 | 21 | 2–10, 12 |
| FP6 E2M3 | 23 | 21 | 2–10, 12 |
| FP7 E3M3 | 31 | 29 | 2–10, 12 |
| FP8 E4M3FN | 47 | 45 | 2–10, 12 |
| posit8 es1 | 50 | 48 | 2–10, 12 |
| FP8 E5M2 | 75 | 73 | 2–10, 12 |

MobileNet cases use the same ladder, with W_cert taken from their certificates (lane Q1). M_lossless = 73 for FP8 E5M2
lies beyond the grammar's M ≤ 52: no float accumulator of 2.3 is guaranteed lossless there (E11M52 gives P = 53 < 74),
so only `wide` and registers of at least 75 bits are certified exact for that case.

- **Operand limits.** As 2.1. 2 ≤ P ≤ 53, −1000 ≤ emin ≤ 127, emin ≤ emax ≤ 1000 in grid units (the existing kernel check).
  Operands are below 2^32.

### A2.4 Chunked (two-stage) accumulation: `ch<C>_<inner>_<outer>`

The terms (taps on scalar nodes; block sums, or element products with `.elt`, on shared-exponent nodes) are put in the
inner policy's order and cut into consecutive chunks of C terms (the last one may be partial; padded taps count). Each
chunk is reduced by `<inner>` from zero. The chunk values are exact rationals, ±inf or NaN, and they are reduced by
`<outer>` in chunk order (sequential, or `.ord-tree`). Each chunk value v_j enters its outer add exactly, as a fused
operand: R(state + v_j). `<outer>` refuses `.ord-rev`/`.ord-hwc` and the term-level modifiers `.nf`, `.pr-` and `.elt`.
Before r4 `.nf` was accepted on `<outer>` and rounded every chunk value into the outer format before the add
(`ch2_wide_flt.e2m1.nf`: 0 overflow steps against 1 for the fused name). No design point of P1 part B needs a
rounded chunk result, and with a float inner stage of the outer format it would be a second name, so it is refused
(review 2 must-fix; D16).
A bias modifier may appear on `<outer>` only.
A float inner stage cannot feed an integer outer register (refused). `control` is not a stage.

- **Hardware and design points.**
  - `ch4_wide_flt.e8m23`: an exact 4-term dot product with one RNE into an FP32 state, the fused DOT4 of [Lut24]
    (P1 row #1).
  - `ch2_wide_flt.e5m10.x<e>`: FP8 into FP16, one rounding per two products, as in [Ber22] ExSdotp (row #2). With
    `flt.e8m7` the same is FP16alt/bfloat16.
  - `ch64_flt.e6m9_flt.e6m9`: [Wan18] (row #3).
  - `ch16_flt.e4m7.x3.ftz.rz.pr-e4m7-x5-rz-ftz_flt.e4m7.x3.ftz.rz`: [Blu24] (row #4: M7E4 accumulator, floor, chunk
    16, product quantised, underflow flushed, saturation at the maximum; b_acc = 10, b_prod = 12). The name parses and
    is canonical. `.rz` gives the saturation; `.x3` = 10 − 7 and `-x5` = 12 − 7 by the bias rule of A2.3.
    [PENDING owner/P1 detail, three assumptions: (i) the product layout is also E4M7 (P1 gives only b_prod);
    (ii) "floor" of a sign-magnitude float is truncation toward zero (`.rz`), not toward −∞; (iii) [Blu24]'s value
    unit is our code level and its top exponent code is reserved as in `flt` (otherwise x and the largest finite value
    change per case).] An RNE variant drops `.rz`/`-rz` and adds `.ofsat`.
  - `ch<T>_sat.w<P_I>_wide`: [Col24b] AXE multi-stage accumulation (row #8).
  - `ch<T>_sat.w<W>_sat.w32`: [Nat25]'s narrow register with a 32-bit spill.
- **Certificate** (`certify.py`, step 5). Inner: the A2.2 rule on each chunk's terms, maximised over chunks and channels
  (about log2(K/C) bits below the node width; P1 row #8). Outer: its terms are the chunk values, each bounded by its inner
  range (times g_in) when the inner register is at least that wide, else by the inner register's whole range; entry is
  exact when g_in is a multiple of g_out.
- **Events.** `inner_*` (summed over chunks), `outer_*`, and `chunks`.
- **Reference properties.** One chunk (C ≥ K) equals the plain inner policy (`test_one_chunk_equals_plain_...`). The hand
  examples include a saturating inner stage that recovers in the outer stage (−1 against plain −7), and +inf and −inf
  chunks meeting as NaN.

### A2.5 Summation orders: `.ord-rev`, `.ord-hwc`, `.ord-tree` (default `chw`)

- `chw` (the engine's order, unchanged): a sequential MAC whose loop nest has the input channel outermost; an NCHW
  im2col GEMM.
- `rev`: the same taps in descending order. This is a control for order sensitivity, not a hardware claim.
- `hwc`: kernel row, kernel column, input channel innermost. This is an NHWC (channel-last) sequential MAC, the usual
  order when input channels are vectorised.
- `tree`: a pairwise adder tree over the chw sequence. At every level elements (0,1), (2,3), … are added, and an odd last
  element moves up unchanged until one value is left. Leaves enter an integer register on its grid (rounded and
  overflow-checked). Float leaves are exact products (after `.nf`/`.pr-`), and every adder rounds its output once, also when one operand
  is zero: a padded tap is a zero leaf and is not skipped (A2.3). A reduction of
  exactly one term (K = 1, or a one-term chunk) is rounded once as R(+0 + leaf), the same step as sequential order. This models an RTL
  adder tree, the spatial dot-product unit of an array column or an MX dot unit. A tree over all K = 4,608 taps is an
  idealisation. The realistic RTL point is a tree inside a dot unit followed by sequential accumulation:
  `ch<T>_<inner>.ord-tree_<outer>`.
- **Which hardware uses which.** A sequential MAC (an output-stationary PE that accumulates over time) uses chw or hwc,
  depending on its loop nest. An adder-tree dot unit or a weight-stationary column reduction uses tree within T, then
  sequential.
- **Certificates.** Every tree node is a subset sum, so the 2.1 prefix bound covers every order. sat, wrap and `.l<L>`
  certificates and the float lossless condition are order-free. Below them, results depend on the order. The hand example
  is swamping: chw gives 4 and tree gives the exact 6.
- **Bias with tree.** Only `.bias-post`, added at the root; `.bias-pre` is refused.

### A2.6 Bias inside the register: `.bias-pre`, `.bias-post`

The node bias b_c (binary64, the export value) is put on the register grid exactly: B_c = RNE(b_c / (s_in·s_w[c]) ·
2^(s − L)) for integer stages (L = 0 without `.l<L>`), or rounded once into the float format (RNE, or toward zero under
`.rz`). With `.bias-pre` it is the initial register content (one fit: sat clamps, wrap wraps, counted as an event). With
`.bias-post` it is added after the last term through the same add. The post-operation becomes RNE64(RNE64(dot·s_in)·s_w[c])
with **no** binary64 bias add. On scalar nodes `wide.bias-pre` and `wide.bias-post` are the exact integer register with
the integer bias RNE(b_c units) on the product grid.

- **Shared-exponent nodes** (r4, review 2 B12). The bias in element-product units is b_c·2^(s_a + s_w) (no tensor
  scale), a dyadic rational. An integer stage receives RNE(b_c·2^(s_a + s_w) / 2^(λ + L)) on its anchored grid (A3.3).
  A `wide` stage has no fixed grid there (its exact sum aligns to the smallest live exponent, A3.2), so its bias is
  added **exactly** and the final conversion is the only rounding: raw = RNE64(D + b_c) in place of
  RNE64(RNE64(D) + b_c). This models a Kulisch accumulator wide enough for the bias's binary64 bits; the bias is one
  more term of the two-limb alignment check (A3.2). A float stage converts the bias once in code-level units. None of
  the three depends on the caller: before r4 the reference gave three values for `wide.bias-post` on one node (block
  term 1/8, bias 5/16: 0.125 with anchor 0, 0.625 with the certificate's anchor −1, exact 0.4375); now 0.4375 in every
  call (`test_a26_wide_bias_is_exact_on_block_nodes`).

- **Hardware.** An integer accelerator that preloads its accumulator with the int32 bias quantised on the scale
  s_in·s_w[c] (gemmlowp/TFLite style), or adds it in the accumulator at the end.
- **Certificate.** The register receives B_c = RNE(b_c / 2^L); the channel range becomes [lo + min(B_c, 0), hi + max(B_c, 0)]
  (`certify.py` step 4). A float stage converts the bias once into its format (rounding mode and flush rule of the
  stage; no event); `.bias-post` is then one rounded add, also when the state is ±inf (IEEE: +inf + −inf = NaN).
- **Reference.** `bias_units()` computes the exact rational, and the hand tests cover RNE onto the grid (2.5 → 2, 3.5 → 4)
  and saturation at preload.

### A2.7 Published design points (P1 part B) and the policy that answers each

"Status after 2.3" says what the exact engine could run, not that the study is first to measure it: P1 review 1
(error 2) found the integer positioning too wide ([Nat25] Fig. 9 already sweeps 5–8-bit operands against 8–20-bit
saturating accumulators without retraining). Every row rests on P1's draft, which is not final.

| P1 row | Design point | 2.3 name | Status after 2.3 |
|---|---|---|---|
| #1 | FP32 accumulation; [Lut24] fused DOT4 | `control`; `ch4_wide_flt.e8m23` | covered; DOT4 new |
| #2 | FP8 into FP16 ([Ber22], [Lut24] DOT2); bfloat16 | `flt.e5m10.x<e>`; `ch2_wide_flt.e5m10.x<e>`; `flt.e8m7` | new |
| #3 | (1,6,9) chunk 64 ([Wan18], training) | `ch64_flt.e6m9_flt.e6m9` | new (low priority) |
| #4 | [Blu24] E4M7, floor, chunk 16, product quantised, flush, saturate | `ch16_flt.e4m7.x3.ftz.rz.pr-e4m7-x5-rz-ftz_flt.e4m7.x3.ftz.rz` | new; [PENDING: product layout E4M7, floor = toward zero, and x from b_acc = 10 / b_prod = 12 assumed (A2.4)] |
| #5 | float mantissa sweep ([Blu24], [Sak19]) | `flt.e8m<M>` ladder; `flt.e5m<M>`, `flt.e6m<M>` with `.rz` | new |
| #6 | saturating register | `sat.w<W>`, `sat.struct-<d>` | covered (2.1) |
| #7 | wrap-around ([Ni20], [Nat25]) | `wrap.w<W>` | new |
| #8 | chunked / two-stage ([Sak19], [Blu24], [Col24b], [Nat25], [Ber22]) | `ch<T>_sat.w<W>_wide`, `ch<T>_flt…_…` | new |
| #9 | product rounded before accumulation | `.pr-e<E>m<M>` (also `wide.pr-e4m3`) | new |
| #10 | exact Kulisch with closed-form width | `wide` + certificate | covered |
| #11 | LSBs dropped ([Joh18], [Cuy26]) | `sat.w<W>.l<L>[.rne]`, `wrap…l<L>` | new; [PENDING P1: row #11 has no source after P1 review 1, so the policy is a design point of its own, not an answer to a published one] |
| #12 | weight-constrained integers (A2Q/AXE) | not an accumulator policy (lane Q8) | – |
| #13 | scale inside the dot (MX) | block terms (A3) | new |

## A3. Shared-exponent formats (MXFP4, MXFP6, MXFP8, BFP6)

Reference: `tools/scaled_bridge_ext/blocks.py`. Simulator: `tools/experiment_b_ext/shared.py` and
`tools/experiment_b2/blocks.py`, imported read-only. The tests compare against the simulator itself on CPU.

### A3.1 Element tables and the exponent rule (matched to the simulator)

- **Element table.** The simulator's binary32 levels (`shared.codebook`), which are dyadic. They are exact units on a
  2^-s grid: bfp6 s = 5, levels −32 … 31 (asymmetric, −1.0 to 31/32); mxfp4_e2m1 s = 1, |u| ≤ 12; mxfp6_e3m2 s = 4,
  |u| ≤ 448; mxfp8_e4m3 s = 9, |u| ≤ 229,376 (18 bits). The simulator's binary32 midpoints equal the exact midpoints for
  all four formats (tested), and its tie preferences are used as they are.
- **Exponent (E8M0).** One exponent e ∈ [−127, 127] per block of 32. `maxabs`: e is the smallest integer with
  max|v| ≤ L_max·2^e, clamped, where L_max is the largest (positive) level; an all-zero block has e = −127.
  `mse` (the B2 `default` recipe, a run-time search): for d = 0 … 6, e_d = clamp(e_maxabs − d), and the d with the
  smallest exact sum of squared errors wins, ties to the smallest d. The `percentile_99_9` block rule (b1 recipes only;
  binary32 top-two interpolation) is **not admitted**.
- **Element.** The nearest level to v/2^e, with exact midpoints and clipping to the end levels. A value exactly on a
  midpoint goes to the level whose code (the format's bit pattern, as `shared.codebook` numbers it) is even; if both
  codes have the same parity, to the smaller code. This is the simulator's rule (`ties` in `shared.codebook`), read from
  the table by `ElementBook.nearest`. Clipping (|v/2^e| above L_max) is reachable under `maxabs` only at the clamp
  e = 127; under `mse` it occurs whenever a d ≥ 1 wins, since every candidate d ≥ 1 clips at least the block's largest element (unless the
  clamp at −127 holds e).
- **Where blocks are formed.**
  - Weights: 32 consecutive taps along K = C_in/groups·kh·kw (chw) per output channel, with the last block zero-padded.
  - Every stored activation (every quantizing boundary of the B2 block plan, including max-pool outputs, which are
    re-blocked, and the network input): 32 consecutive channels at each (n, y, x).
  - Convolution: the patch of each output position (chw order, padded taps are zeros and belong to their block) is
    **re-blocked along K** with the activation rule and re-quantised. This is an exact store-like step inside the MAC node.
  - Linear: no re-blocking, because the input's channel blocks are its K blocks.
  - There is no per-tensor scale: values are u·2^(e − s).
- **Rule arithmetic.** The engine evaluates every rule exactly on its binary64 values, and the simulator evaluates it in
  binary32/binary64. For `maxabs` the two agree on identical binary32 inputs (the ratio test is exact in both). For
  `mse` agreement is **tested, not proven**: the simulator forms binary32 differences and a float64 sum of squares, so two
  candidates d whose exact errors differ by less than that rounding could be ordered differently. Tests (review 1 also
  found 0 differences in 3,600 full-precision binary32 blocks): maxabs on 40 random tensors of 1 to 99
  elements per format, mse on 30 tensors of one or two blocks per format with 0 disagreements on dyadic data, stored channel blocks, and a strided, padded, dilated convolution with a
  partial K block; all are equal to the simulator's own `BlockQuantizer`, `B2BlockQuantizer.core` and `block_conv2d`.
  In a whole graph the inputs already differ by construction (2.1: binary64 against FP32 stores). The gate therefore
  reports the share of blocks whose exponent differs from the simulator's rule applied to the simulator's own tensors, and
  that share is not a gate.

### A3.2 Exact block sums, alignment and the wide arm

For block b of a dot, S_b = Σ u_a·u_w is an exact integer: |S_b| ≤ 32·max|u_a|·max|u_w|, which is below 2^41 for mxfp8.
With E_b = e_a,b + e_w,b, the exact dot is D = Σ_b S_b·2^(E_b − s_a − s_w), and `wide` converts it with one RNE64. That
conversion cannot overflow or underflow: |D| < 2^310, and the smallest nonzero |D| is at least 2^(−254 − 2·9). The
post-operation is raw = RNE64(RNE64(D) + b_c): B2 block formats carry no tensor scale.

- **Alignment.** Exact accumulation aligns the nonzero block sums to E_min, the smallest E_b among them. A zero block (for
  example all padding, e = −127) is excluded, because otherwise one empty block would add up to 254 bits of spread. A
  two-limb implementation is exact when Σ_b |S_b|·2^(E_b − E_min) < 2^126 (`alignment()`: the per-output runtime check).
  Under `wide.bias-pre/post` the exact bias is one more aligned term m·2^e (odd significand m, exponent e, in
  element-product units; `alignment(..., bias_units=)`), so E_min can move down to the bias's last bit; the same check
  and the same fail-closed rule apply.
  When the check fails the output **fails closed** as `alignment_overflow`. It is never approximated. The stage-2 gate
  requires zero such failures on the gating images, or a wider exact register (4 limbs = 256 bits; owner decision D9).
- **Static certificate per node** (`static_width()`). This is the width of an exact fixed-point register anchored at the
  finest reachable grid min_b e_w,b + e_a,lo. With e_a,lo = −127 it runs to hundreds of bits, so the static bound is
  reported but not used for kernel choice. The per-output check above decides instead. e_a,hi comes from the 2.2 closure
  bound of the input tensor through the maxabs rule; e_a,lo is −127 unless proved otherwise.

### A3.3 How each accumulator policy acts on block sums

- **Granularity.**
  - Default `block`: one term per block, T_b = S_b·2^E_b. This models an MX dot unit whose inner 32-term sum is exact
    (the OCP MX dot product) and whose block results enter the accumulator.
  - `.elt` (a name modifier, after `.pr-` and before `.ord-`; allowed on a single stage or on the inner stage of `ch`,
    refused on the outer stage): one term per element product u_a·u_w·2^E_b in K order, with block padding excluded.
    K cannot be recovered from the blocks, so `.elt` requires it: `block_dot(..., length=K)` refuses a missing K and a
    K outside 32(n − 1) < K ≤ 32n for n blocks (review 2 must-fix; before r4 a missing K made the padding into terms).
    This models a dequantising scalar MAC. A name with `.elt` is refused on scalar nodes.
  - The granularity is read from the name only: `block_dot` takes no granularity argument (`block_terms` keeps one for
    the tests). Example names: `sat.abs-0.elt`, `ch32_flt.e8m10.elt_flt.e8m23`.
- **Floats** (`flt…`, `control`, chunked) are defined in code-level units as for scalar formats. The alignment is dynamic,
  as in a float adder.
- **Integer registers** (sat, wrap, `.l<L>`) need a fixed grid. Their LSB is 2^(λ + L) element product units, with the
  anchor λ = max_{c,b} e_w,c,b + e_a,hi: the grid of the largest reachable exponent pair, fixed per node. Terms from
  smaller exponents lose low bits on entry (floor, or RNE with `.rne`, also at L = 0). `.l-<n>` keeps n bits below the
  anchor. This models a fixed-window MX accumulator. λ belongs to the node and has no default: the reference refuses a
  policy with an integer stage on a block node unless the anchor or the node's `NodeCertificate.blocks` is given, and
  refuses a 2.1 certificate dict or a scalar certificate there (r4, B12). `wide` and float stages need no anchor: their
  values do not depend on it (tested). Certificate (`certify.NodeCertificate.blocks`, review-1 fix B2):
  per-term ranges ±β_{c,b}·2^(e_w,c,b + e_a,hi − λ), β_{c,b} = max|u_a|·Σ_{k∈b}|u_w,k| (per element with `.elt`), then
  the A2.2 rule with the INEXACT entry (+n − 1 terms for floor, ±n/2 for RNE). `struct-<d>` is refused on block nodes.
- **Orders.** `rev` and `tree` act on the term sequence. `hwc` is refused on block nodes (an NHWC MX engine would block
  along channels per tap, a different block structure, not an order). A name with both `.elt` and `.ord-hwc` fails at
  the parser; other hwc names are valid scalar names and are refused by the block dot.
- **Linear inputs.** A linear layer's input must be a stored block tensor of shape (N, C) or (N, C, 1, 1); otherwise
  the node is refused (the simulator then multiplies an unquantised input).
- **Reference property.** With equal exponents the block dot equals the scalar dot under every policy when the anchor
  equals the common exponent pair (`test_equal_exponents_equal_scalar_dot`). With differing exponents a float policy does
  not see the anchor (tested).

### A3.4 Store path and other operators with a shared exponent

- **Store.** Binary64 raw values → channel blocks → exponent by the rule (exact) → element codes → state (codes, units,
  one exponent per block). The reconstruction r = u·2^(e − s) is exact in binary64 (|u| < 2^18, exponent ≥ −136).
- **ReLU on stored codes.** max(u, 0) with the exponent kept.
- **Max-pool.** On r, then re-blocked (no code pass-through: B2's rule).
- **Residual add.** RNE64(r_a + r_b).
- **Global average pool.** Exact rational sum of r over the window, one RNE64, then RNE64(·/count). For blocks this
  replaces 2.1's code sum, because the elements of a channel sit on different grids.
- **Squeeze-excite multiply.** Both operands stored: u_a·u_b·2^(e_a + e_b − s_a − s_b), exact (≤ 36 bits).
- **Hard-swish and hard-sigmoid.** On r, as in 2.2.
- **Record.** Per stored tensor: exponent histogram and clipped elements. Per MAC node: max block-sum bits, spread
  E_b − E_min, and alignment failures.

### A3.5 Limits of the two-limb path

Operands: element units below 2^18 (32-bit operand class). Block sums: int64. Aligned total: below 2^126, which means a
spread of at most about 127 − 41 − log2(K/32) bits for mxfp8 (78 bits at K = 4,608) and about 107 for mxfp4. Beyond that,
fail-closed or 4 limbs (D9). Integer registers W ≤ 127 on the anchored grid. Not admitted: `percentile_99_9` block rule,
unsigned block codes, CLE, `quantize_input=false`, `quantize_logits=false` (B2 refuses all of them for block formats).

## A4. Logarithmic formats (log4, log6, log8) and NF4

Their levels are not dyadic rationals (log: ±2^(k/2^f); NF4: decimal constants). The binary32 level tables the simulator
uses, computed here:

| Format | grid shift | max \|u\| (bits) | product bits | fits the operand limit \|u\| < 2^32? |
|---|---:|---:|---:|---|
| log4 | 26 | 29 | 57 | yes |
| nf4 | 27 | 28 | 55 | yes |
| log6 | 28 | 34 | 68 | **no** |
| log8 | 32 | 44 | 88 | **no** |

- **(a) Not admitted.** The formats stay simulator-only, and the paper says so in one sentence per format.
  Cost: none. Scientific meaning: the exact-engine claims (certified widths, the accumulator failure width, the
  simulator-versus-exact gap) are made for dyadic formats only. The log formats keep their simulator accuracy, and the
  hardware cost model must not imply an exact width for them.
- **(b) Admitted through the binary32 tables.** The table becomes a dyadic codebook.
  - What it models: an LNS-to-linear hybrid MAC whose converter outputs a 24-bit binary32 significand into an exact
    linear accumulator. For NF4 it is the format as deployed: bitsandbytes stores the levels as float32, so (b) is
    faithful for NF4.
  - Cost: small for log4 and NF4 (codebook, conformance, case gates; no kernel change). Medium for log6 and log8: their
    products exceed 64 bits, so the kernel needs a third operand class (128-bit products from 32-bit limbs). log8's prefix
    bound is about 88 + 13 = 101 bits < 2^127, so `wide` fits two limbs.
  - Scientific caveat: their certified widths (about 70 to 100 bits) measure the binary32 table, not log arithmetic. A
    real LNS design would choose a converter precision F, which is (b) with a stated F, and that is a design study in
    itself.
- **(c) Exact algebraic accumulator.**
  - What it is: values in Z[2^(1/2^f)] with 2, 4 or 8 integer coordinates. Sums are exact. Stores compare algebraic
    numbers against irrational midpoints, which needs exact sign determination.
  - Cost: large (new kernel family, new store oracle, new witnesses).
  - Meaning: an exact real reference for log formats with no hardware counterpart. LNS hardware converts to linear with
    finite precision.
- **Recommendation: (a)** for the paper. Optionally add (b) for NF4 and log4 only, if the exact-engine table should
  include them; NF4 is at chance in 5 of 6 B2 cells (ResNet18 minimal: 34.4), so the value is low. Do not adopt (c). None of this is implemented here: it is an
  owner decision (D10).

## Part C. Plan for stage 2

### C1. Packaging (nothing edited in place)

New versioned packages, each copied from the **archive** copies, never from the live trees:

- Base: `artifacts/speed_v1/implementations/edfecd5c…/py/scaled_bridge_fast/`.
- Its imports: the 1f75c923 archive `py/scaled_bridge_v2/`.

The stages:

- **2a, `tools/scaled_bridge_list1/`** (list parameter only). Copy the fast package (`common.py, engine.py, native.py,
  worker.py, fastdot.cu, kernel.h, cli.py, __main__.py, archive.py, __init__.py`). Change only `worker.py` (A1) and the
  identity in `common.py` (a new LOGICAL prefix and run root `artifacts/scaled_bridge_ext_v1/runs/<digest>/`). The
  1f75c923 base is imported unchanged. `kernel.h` and `fastdot.cu` stay byte-identical, but the new package builds its
  library under its own root, so check 1 (PTX identity, A's t1, about 2 GPU minutes) is required.
- **2b, `tools/scaled_bridge_x1/`** (policies). Copy 2a, plus vendored copies of the base's `accumulators.py`,
  `certificates.py`, `oracles.py` and `conformance.py`.
  - `accumulators.py`: the parser of A0, delegating to `tools/scaled_bridge_ext/policies.py`.
  - `oracles.py`: kept unchanged as the manifest-driven gate of the 2.0/2.1 policies; `tools/scaled_bridge_ext/reference.py`
    is added beside it for the new policies. Every `tools/scaled_bridge_ext` file the engine imports (the parser decides
    kernel parameters; `certify.py` decides widths) is a numeric source and enters the digest.
  - `certificates.py`: the A2 certificates (wrap, `.l<L>`, per chunk, bias).
- **2c, blocks.** Adds a block exporter (a new file beside `tools/experiment_b2/export.py` that imports it, writing block
  weight codes as int8 and exponents as int8), a block adapter, the block store, and the block dot. This could be stage
  2b's digest or its own (D1).

**Kernel entries (2b).** The existing policies 0 to 3 keep their structs and instantiations byte-identical. New code goes
into new structs and new extern entry points:

- **Integer registers (sat/wrap).** One struct `IntReg` with W, L, the low-bit mode and the overflow rule. It runs with
  two limbs when W > 63. The existing `SatInt` is not modified.
- **Float accumulator.** `SoftFloat2` = the SoftFloat rounding core plus flags for ftz, ofsat, rz, nf and the product
  format. The 2.1 policies keep using `SoftFloat`. Three points differ from `SoftFloat` and need their own primitive
  gates: `.ftz` tests tininess on the exact sum BEFORE rounding (`rounding.FloatFormat`, D13); every rounded add returns
  an overflow flag (unbounded-exponent rounding above the largest finite value) for `overflow_steps`, which `SoftFloat`
  does not count; and the product format saturates on overflow while the accumulator gives ±inf unless `.ofsat`/`.rz`.
  Counters: the 2.1 `sat` word keeps its two 16-bit fields saturating at 65,535 (the supplement vectors cover the
  cap). The reference counts every other field without a cap. Per output no counter can exceed the number of adds
  (K + 2 at most, 4,610 on the admitted nodes), so 16-bit kernel counters are exact there; a cap for larger K must be
  written into the contract before it is frozen.
- **Orders.** Index permutations for rev and hwc in a new `reduce_stat_order<Order>`. The tree uses a streaming pairwise
  reducer whose equality with the level-wise tree of `orders.tree` (carry the odd element up) must be proven and
  witnessed for every K up to 4,608. Zero leaves enter the tree and every adder rounds (A2.3); a reducer that skips
  zero leaves fails the gate.
- **Two-stage.** `Chunked<Inner, Outer>`.
- **Bias.** A per-channel start value or final add: the RNE grid value of A2.6 for integer stages, one conversion for
  floats; on block nodes a `wide` stage adds it exactly as one more aligned term inside the two-limb check (A3.2).

**Block dot (2c).** A pre-pass computes patch blocks (exponents and element units, exact int64 on the GPU) at batch 8.
The kernel then sums each block exactly and applies the policy to the block terms. The alignment check runs per output.

### C2. Bit-identity with archive edfecd5c for every existing policy

Under each new digest, before any new policy is gated:

1. kernel.h/fastdot.cu: the PTX of the 8 existing entries must equal those in edfecd5c's `fastdot.so`, using the same
   nvcc command. A's t1 must pass: primitives, kernel outputs, division.
2. The 601 stored vectors (wide, control, sat.w, fp16[.x], f21[.x]) must give byte-equal values and event words through
   the new kernel.
3. S1's evidence under the new digest: regress 82 sets, trace 100 sets, compare 369 sets plus the two missing cases, then
   `gates` and `archive`.
4. D's `check_archive.py` and `trace_gates.py`.
5. The sealed 1k comparisons: B's 18 ResNet18 pairs, and the wide/control files of all 10 MobileNet cases.
6. The archive-referenced regimes: C's worklists (knee widths, sat.struct onset, fp16 failures, f21), B's references, and
   D's fp16 probe.

Stage 2a needs items 1 to 6 (identical kernel files, but a rebuilt library under a new root).

### C3. Gating each new policy and format against the reference models (full traces)

- **Primitive vectors.** For each family, compare the kernel's value and every event field with
  `tools/scaled_bridge_ext`. Use the tests' hand vectors plus at least 5,000 seeded random cases per family:
  - W in 2..127, L in 0..20, E in 2..11, M in 1..23, and every modifier;
  - orders: rev, hwc (with kh, kw > 1), tree, with tree K from 1 to 4,608;
  - the lone-root rule: K = 1 under `.ord-tree`, and one-term last chunks (K = 9, 25, 27 with `ch2`/`ch4` and an inner
    tree), each with and without `.bias-post`, against the sequential step R(+0 + leaf) value and events;
  - overflow: `overflow_steps` on every float variant (inf, `.ofsat`, `.rz`), on tree adders and on the bias-post
    add, plus `.ftz` cases on both sides of the smallest normal (tininess before rounding);
  - events: every field of A2.3 compared, including `absorbed_steps` with `.nf`/`.pr-` products that round to zero
    (skipped, no event) and counters at the 65,535 cap;
  - chunks: C ∈ {1, 2, 4, 16, 32, 64, K, K + 1};
  - bias: pre and post; on block nodes `wide` (exact), integer (anchored grid) and float;
  - tree adders with zero leaves (padded taps at border outputs, zero products), value and every event field (B11);
  - geometry: padded, strided, dilated and grouped.
- **Actual-node rational checks.** On the first image of every (case, policy), check four fixed outputs per MAC node and
  the first output with an event, recomputed by the reference from the traced input codes.
- **Full traces.** Eight images with full traces at batch 1 and 8: identical records and batch invariance.
- **Certificate equalities on 32 images.** `wrap.struct-0`, `sat.struct-0` and `flt.e8m<W_cert−2>` must equal `wide`
  bit for bit with no event (the float one on cases with W_cert ≤ 54 only, since M ≤ 52; also with `.nf` and `.rz`;
  for `.ftz` only where the smallest normal is at most one grid unit, A2.3).
- **Blocks.**
  - Every element and exponent of every stored tensor on 2 images, recomputed by `blocks.py` from the traced binary64 raw
    values.
  - The patch re-blocking and block terms of 4 outputs per node.
  - The alignment check: zero failures required.
  - An independent whole-graph witness: an FX traversal with Python big-integer block dots, extending `graph_witness.py`
    in a new file.
  - Exponent agreement with the simulator: reported, not a gate.

### C4. GPU time and disk (estimates; shared GPU, fast-path rates)

| Step | GPU | Disk |
|---|---|---|
| 2a re-checks C2.2–C2.6 | ≈ 1.5 job-hours (S1's sets about 1 h, the 1k comparisons 15 min, C's worklists 18 min) | < 50 MB of predict files |
| 2b primitives and policy gates (2 models) | ≈ 1 job-hour | < 30 MB |
| 2b development sweeps (survey items 4, 6, 8: about 150 + 56 + 24 runs at 15–40 s per 1k call) | ≈ 2–3 job-hours | ≈ 2 KB per image-run → about 0.5 GB for 230k image-runs; compact further or keep only events. Needs an owner disk allocation |
| 2c block exports (12 configurations; B2 block bias correction via the fastblocks path, 0.6–1.6 GB GPU, 35–130 s each) | ≈ 0.5 job-hour | int8 codes and exponents, compressed: about 3–12 MB per configuration, ≈ 80–100 MB. **Exceeds this lane's 0.05 GB**: owner allocation |
| 2c block gates (12 cases × 32-image full traces + replay) | ≈ 1–1.5 job-hours | ≈ 40 MB |

No step uses `--exclusive`. B2 block exports run `--heavy 8000` unless the fastblocks path is used.

### C5. What must not happen while lanes Q1 and S1 use the current archive

- No edit, rebuild or write in `tools/scaled_bridge_v2/**`, `tools/scaled_bridge_fast/**`,
  `artifacts/scaled_bridge_v2/**` (archives and runs) or `artifacts/speed_v1/**` (archives, runs, libraries).
- New packages build their libraries only under their own artifacts root.
- Build once before launching parallel jobs (the MB2 incident).
- A new package imports the archived base by path, never the live `tools/scaled_bridge_v2`.
- Never load two packages that both import a top-level `scaled_bridge_v2` in one process.
- Never write into `artifacts/experiment_b2/matrix/` or L1's export folders. Re-sealing L1's configuration records is
  forbidden (use `seal_configuration=False`).
- Never read or run the held-out list before the owner seals P4's protocol.
- At most one GPU job of the lane at a time, under 15 minutes each.

### C6. What an independent reviewer must recompute to trust a new digest

1. The digest from the files: the hash list of numeric sources (including every imported `tools/scaled_bridge_ext`
   file) plus the base digest.
2. A byte diff of the new package against the edfecd5c and 1f75c923 archive copies. Every changed line must be explained
   and unchanged files must be byte-identical.
3. PTX identity of the existing kernel entries (C2.1).
4. Their own run of the 601 stored vectors and of their own random vectors through the kernel against
   `tools/scaled_bridge_ext`. They should also check the reference itself with an independent implementation, for
   example NumPy float16 for `flt.e5m10`, plain int arithmetic for wrap, and a direct Fraction tree.
5. A sample of full traces recomputed with their own witness for each new family.
6. The certificates of two nodes per new family re-derived by an independent script.
7. Verdict (d) items 2–6 for the existing policies.
8. No write into Q1's or S1's roots: manifest hashes of edfecd5c, 1f75c923 and 7c6344af unchanged, and file listings with
   timestamps.
9. For 2a: screen-list identity on Q1's sealed files (G-L1), and the refusal of the held-out list without the sealed
   protocol (G-L3).

## Owner decisions this draft needs

| # | Decision | Recommendation |
|---|---|---|
| D1 | Stage the list parameter as its own digest (2a) before the policy work (2b/2c)? | yes: smaller review, unblocks the confirmation grid of existing policies |
| D2 | Held-out list definition (rows of the `imagenet_evaluation_10k` record whose sha256 is not in the screen), its iteration order (ascending sha256, as the engine's `dataset()` orders every list, against the manifest order of P4 draft 2 section 2: "Every held-out run iterates the complement rows in full-manifest order"), its identity (P4's set digest `d47669bb…` plus the engine's `digest(rows)` of the ordered rows; P4's manifest-order digest `03bb8753…` is a different function), and the authorization mechanism | sha256 order (P4 to change that sentence); both digests recorded; P4's protocol, sealed by the owner |
| D3 | Low-bit default: floor (bit drop) with `.rne` variant; LSB relative to the product grid, with or without a code-level alias; negative L (`.l-<n>`: n bits below the term grid, meaningful only for block terms); `.rne` without `.l<L>` = RNE entry at L = 0 (same arithmetic as the plain name on scalar nodes) | floor + `.rne`; product-grid relative; both forms as written |
| D4 | Allow register widths 64–127 (two limbs) to bracket FP8 E5M2 | yes |
| D5 | Float ladder (E8, M 2–10 and 12, then E 4–6) and which variants (`.ftz`, `.rz`, `.nf`, `.pr-`) are headline versus ablation | ladder headline; variants as ablation |
| D6 | Tap orders to run (hwc; tree inside `ch32`; whole-K tree as idealisation) | hwc and `ch32_….ord-tree_…` |
| D7 | Which chunked design points to include (P1 rows #1–#4, #8); the [Blu24] product layout | rows #1, #2, #4, #8 |
| D8 | Bias in the register: run it, and pre or post | pre, as one ablation |
| D9 | Block formats: activation rules (mse = B2 default, maxabs = intrinsic arm), default granularity `block`, integer anchor λ, fail-closed alignment or a 4-limb exact register, `.ord-hwc` refused on block nodes, a linear layer's input must be a stored block tensor (else the node is refused), `struct-<d>` refused on block nodes, disk allocation for block exports (~0.1 GB) | both rules; `block`; λ as A3.3; fail-closed first, measure the spread; the three refusals as written |
| D10 | Log formats and NF4: (a) / (b) / (c) | (a); (b) for NF4/log4 optional |
| D11 | Should wrap record "event but exact" with a parallel exact sum (one more register per thread)? | yes |
| D12 | Event definitions (A2.3, normative text in `reference.py`): `absorbed_steps` counts swamping in both directions (also a term flushed to +0 against +0); `overflow_steps` is a new field; a product that rounds to zero is a skipped term (no event); a lone tree root is one rounded add; a tree adder rounds also with a zero operand (no zero skip in a tree, r4); a float bias-post add meets ±inf by IEEE rules (+inf + −inf = NaN, a failure) | as written |
| D13 | `.ftz` (and `-ftz` of `.pr-`) detects tininess BEFORE rounding: an exact value below the smallest normal becomes +0 (E4M3: 61/4096 → +0), as flush-to-zero hardware does; the alternative (round on the subnormal grid, then flush) keeps 1/64 | before rounding |
| D14 | [Blu24] row #4 assumptions: product layout E4M7, "floor" = toward zero, x from the biases in code-level units (A2.4) | ask P1/the paper before it is run; run it as an ablation |
| D15 | Bias on shared-exponent nodes (A2.6): `wide` adds it exactly (one more aligned term of the two-limb check), integer stages round it onto the anchored grid 2^(λ + L), floats convert it once; alternative: refuse bias policies on block nodes | as written (r4) |
| D16 | Outer stage of `ch`: chunk values enter fused; `.nf` (a rounded chunk result before the outer add) refused, like `.pr-` and `.elt` | refuse; add a named modifier only if a design point needs it |

## Revision r2 (2026-10-02, after review 1)

Blocking B1-B10 and the must-fix items of `artifacts/agent_orchestration/handoffs/E1-engine-extension-review.md`
(Review 1) are addressed in the text above and in `tools/scaled_bridge_ext/` (new `certify.py`; `rounding`, `registers`,
`policies`, `reference`, `blocks` revised; tests in `tests/unit/test_scaled_bridge_ext_review1.py`, 80 tests in all).
Semantics changed on purpose: `.ftz` flushes before rounding (B7); `absorbed_steps` counts swamping in both directions
and `overflow_steps` is new (B8); a one-term tree is rounded once (B5); a float bias-post add meets a non-finite state by
IEEE rules (NaN for opposite infinities); relative widths follow `certify.py` (B3). Reviewer mutants rerun: all earlier
survivors killed except the mse search cut at d ≤ 3, which no block of the four tables exposes (d ≥ 4 never won in
12,000 random blocks). The text items r2 left open were completed in r3 (below).

## Revision r3 (2026-10-02, review 1 completed)

The text items r2 left open are done, and the r2 fixes were re-checked:

- **Re-checks.** 87 tests pass (r2's 80 plus 7 in `tests/unit/test_scaled_bridge_ext_r3.py`, one per sentence added
  below). The reviewer's probe (`review-scratch/adversarial.py`, sections 1 to 9) reproduces the fixed behaviour of B4,
  B5, B7, B8 and the NaN of A0; its last section uses the pre-r2 `Policy.width(dict)` API and was re-run through
  `certify.register_widths`: `sat.abs-0` 12, `sat.abs-0.l4` 9, `ch2_sat.abs-0_sat.abs-0` 10/12 bits on a 12-bit
  node (B3). The reviewer's 31 mutants against all three r2 test files: 30 killed. `pr_overflow_inf` crashed in the
  old plug and was re-created against the current API (killed by 2 tests). `mse_range_3` survives, as r2 documented.
- **A2.3.** `.x<e>` carries 2.1's sign (the accumulator holds the value times 2^e), and a paper's bias maps to
  e = b − (2^(E−1) − 1). `.pr-` is R'(p·2^e')·2^−e' in code-level units, with RNE under the accumulator's `.rz`, and
  subnormals or `-ftz` (B6). A product rounded to zero is a skipped term, and an ftz flush against +0 is absorbed (B8).
  The ladder note cites P1's correction ("within 1 point", not lossless), and E5M2's M_lossless = 73 lies beyond
  M ≤ 52.
- **A2.4 and A2.7.** The full [Blu24] name, with its three assumptions marked [PENDING] (D14). Row #11 is marked
  [PENDING P1]. The table is not a novelty claim (P1 error 2).
- **A3.1.** The tie rule in words (even code, then smaller code). Clipping under `mse`. Agreement for `mse` is tested,
  not proven.
- **A3.3.** `.elt` is a name modifier of a single or inner stage.
- **A1.** Stage 2a runs check 1 as well, which is consistent with C1/C2.
- **C1 and C3.** `SoftFloat2` gets ftz-before-rounding, the overflow flag, saturating product overflow and counter
  widths. C3 gains primitive gates for the lone root, `overflow_steps`, ftz on both sides of the smallest normal, and
  zero-rounded products. The float certificate equality is limited to W_cert ≤ 54.
- **Owner decisions.** D2 (order and identity), D3 (negative L, `.rne` at L = 0), D9 (three block refusals); new D12
  (event definitions), D13 (ftz before rounding), D14 ([Blu24] assumptions).
- **Docstrings.** `policies.py` (`.x`, `.pr-` units) and `reference.product_format` were reworded. The arithmetic is
  unchanged: the stored vectors and every earlier test pass unchanged.

## Revision r4 (2026-10-02, after review 2)

Blocking B11, B12 and the two must-fix items of Review 2 (`E1-engine-extension-review.md`) are resolved; the
non-blocking items are done as well.

- **B11 (A2.3, A2.5, C1, C3, D12).** The model was right and the text was wrong. A tree adder rounds its output
  also when one operand is zero, as an RTL adder tree does (no adder tests for zero). The sequential zero skip stays,
  because it cannot change a value. The text now says this and gives the 9, 0, 1, 0 example (8, not 10). The
  reviewer's surviving mutant (`tree_zero_leaf_skipped`, re-created in the session scratchpad) is killed by 2 new tests.
- **B12 (A2.6, A3.2, A3.3, D15).** On a shared-exponent node a `wide` bias is added exactly and is one more term of
  the alignment check (`blocks.alignment(..., bias_units=)`, `blocks.dyadic`). An integer stage needs the node anchor
  λ: `block_dot` refuses an integer policy without the anchor or the node's `NodeCertificate.blocks`, and refuses a
  2.1 dict or a scalar certificate. The reviewer's example now gives 0.4375 for every anchor and with the certificate.
  Code: `reference.Stage`/`reduce_terms`/`dot_terms` take `wide_grid` (False on block nodes, set by `block_dot`).
- **Must-fix: outer `.nf` (A2.4, D16).** Refused by `resolve`. Chunk values enter the outer add fused.
- **Must-fix: `.elt` length (A3.3).** `block_dot` and `block_terms(..., 'element')` require K and check
  32(n − 1) < K ≤ 32n.
- **Non-blocking.**
  - A1 cites P4's sentences instead of line numbers, and names P4's manifest-order digest `03bb8753…` as a different
    function from the engine's `digest(rows)`.
  - A1's table lists what `engine_sources()` hashes; `export.py` is in the engine digest; the fast identity includes
    `BASE_DIGEST`.
  - A2.3 defines B.
  - A0 names the two second spellings (`wide.elt`, `.rne` at L = 0 on scalar nodes).
  - `.elt` together with `.ord-hwc` fails at the parser.
- **Checks.**
  - `.venv/bin/python -m pytest -q tests/unit/test_scaled_bridge_ext*.py`: 96 passed, about 7 s (87 + 9 in
    `tests/unit/test_scaled_bridge_ext_r4.py`). Two earlier tests were adapted to the new refusals: the lossless test
    drops `.nf` from the outer name, and the block-grammar test passes `anchor=0`.
  - The stored vectors are unchanged (`-m tools.scaled_bridge_ext.vectors` is a no-op: 601 + 53 cases).
  - The reviewer's `compare2.py` was run as a scratchpad copy that passes `anchor=0` where it had used the old
    default. Seeds 11, 12 and 13 (100 trials each) give 0 value or event differences for sat, wrap, flt, wide and
    chunked scalar names. Every certified width (scalar and block) agrees. Every block value difference (35 + 23 + 31)
    is a `wide` stage with a bias on a block node, which is the deliberate B12 change. The reviewer's model rounds that
    bias onto the anchor grid.
  - `adversarial2.py`, run statement by statement: the B12, `.elt` and outer-`.nf` probes are now refused, and all
    other probes give the values quoted in review 2.
  - Nothing under `review-scratch/` was written.
