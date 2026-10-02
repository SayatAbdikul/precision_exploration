# Figures and tables plan (DAC 2027 draft 1)

Lane P2, 2026-10-02; revision 2 the same evening after independent review 1. Budget: 4 figures and 3-4 tables in 6 pages (see `outline.md`). "Exists" means the data
file is on disk today; it does not mean the data are final or reviewed (status per `claims-ledger.md`). Every
figure must carry the label "development evidence, 1k screen" until confirmation data exist. Generated figures
belong in `results/figures/`, their code in `public/analysis/` (`docs/publications/README.md` L60); new versions
get new file names (no overwrite).

## Main figure: F2, top-1 against accumulator register width, per format (the headline)

- **Shows.** For each format, top-1 (expected credit, with 95 % paired intervals against the exact arm) as the
  saturating register narrows; vertical markers at the certified width (abs), at W_noevent and at W_acc(1.0); the
  published closed-form width as a second marker. Panels: ResNet18 | MobileNetV2 | MobileNetV3-Large. Optional
  inset or second row: binary16 / f21 / FP32 control as points at the right edge.
- **Axes.** x: register width W (bits, sign included, product-grid units). Two alternatives to test in draft:
  (a) absolute W, (b) W − W_cert (aligns the curves; shows the near-constant 6-8 bit slack, ledger A2). Variant (b)
  is the likely main panel, (a) as a small inset, because (b) carries the design rule.
  y: top-1 % (0 to FP32).
- **Data.**
  - ResNet18: `results/summaries/accumulator-sweep-v1/policies.csv` (per case and policy: top-1 under both tie
    rules, paired difference and interval, event images), `widths.csv` (W_cert, W_noevent, W_acc, W_half, formula).
    **Exists** (reviewed).
  - MobileNetV2/V3: `results/summaries/accumulator-sweep-mn-v1/` **does not exist yet** (lane Q1 running; its
    document holds an INTERIM scratch table for 8 of 10 cases, `docs/analysis/accumulator-sweep-mobilenet-2026-10-02.md`
    L15-26, not to be plotted until the write-once summary exists).
- **Reuse.** `results/figures/accumulator-sweep-top1-vs-width-v1.{pdf,png}` (ResNet18, absolute W) is the starting
  point; it needs the MobileNet panels, the W − W_cert variant, and the formula marker.
- **Axis note.** On the INTERIM MobileNet data the absolute widths line up across networks (INT8 19, FP8 E4M3 39)
  while W − W_cert does not (slack 4-6 against 6-8); choose between (a) and (b) only after Q1 is final and reviewed
  (`outline.md` §Fallback).
- **Missing for the main figure:** (1) MobileNet panels (Q1, INTERIM); (2) any non-integer cost axis (would turn F2 into the
  joined figure F3); (3) held-out confirmation points at the frozen grid (sweep L360-392 proposes the grid; not
  run); (4) a float accumulator with a failure point (A5). Without (1) the figure is single-network.

## F1. Method pipeline (schematic)

- **Shows.** PTQ export (codebooks, scales, weight codes) → exact code-domain engine (per-node exact dot on the
  product grid, policies: wide, control, sat.wW, fp16.x e, f21) → certificate per node → readout with a fixed tie
  rule; beside it the FP32 fake-quant simulator path and where the two differ (accumulation, scale factoring,
  store rounding, top-k tie order).
- **Data.** None (drawn from `docs/analysis/scaled-bridge-v2-contract-2026-10-01.md` L10-46, L104-157).
- **Reuse.** None. To draw (inline SVG or TikZ-equivalent). Size: one column, ~5 cm.

## F3. MAC area against accumulator width, with the measured failure widths marked

- **Shows.** Core area (library units, 8 ns target) against accumulator width for each multiplier width, straight
  line fits; markers at the failure width (W_acc(1.0)) and certificate of each format whose MAC is synthesised;
  second axis or annotation: area per accumulator bit ÷ area per multiplier-width step.
- **Data.** `results/tables/ics55-integer-accwidth-v1.csv` and `results/summaries/ics55-integer-accwidth-v1.json`
  (**exist**, integer MACs INT4/5/6/8 × ACC 16-64; reviewed by lane P3, "Approve with corrections", with F1
  blocking for the join); join rows in `results/summaries/accumulator-sweep-v1/join.csv` (**exists**, INT8/INT6
  only) and P3's derived join (`docs/analysis/ics55-integer-accwidth-review-2026-10-02.md` part 5, L153-189, with
  `-noalumacc` variant; outputs `artifacts/ppa/ics55-integer-accwidth-review-v1/part5-join.{json,csv}` and
  `part5-flow-variant.json`, **exist**, not under `results/`).
- **Caveats to print on the figure.** The area per accumulator bit depends on the synthesis option (study flow
  versus `-noalumacc`, review F3 L33); the RTL register is not the engine's (F1 L31); only INT8 `default_signed`
  matches operand for operand.
- **Reuse.** `results/figures/ics55-integer-accwidth-v1.{pdf,png}` (integer area vs width).
- **Missing.** Non-integer MAC RTL (FP6 E2M3, FP7 E3M3, FP8 E4M3, posit8) at certified and failure widths, energy
  from workload activity, an RTL variant aligned with the engine (bias outside the register, unsigned activation
  port; sweep L265-279, P3 L181-189), and a fixed synthesis recipe. This is the second half of the headline
  (ledger A7).

## F4. Simulator against exact execution (methods)

- **Shows.** Paired difference exact − simulator per case under (i) the simulator's own top-k order, (ii) the common
  tie rule, (iii) expected credit; plus the fraction of tied images. ResNet18 original recipe (3 cases), ResNet18
  B2 (7 cases), MobileNets (10 cases at 1k, INTERIM).
- **Data.** `results/summaries/scaled-bridge-gap-v1/{summary.json,ties.json,diagnostics.json}` (**exist**,
  reviewed); `results/summaries/accumulator-sweep-v1/simulator.csv` (**exists**, B2 cases); MobileNet 1k
  comparison (INTERIM numbers in the Q1 document L41-47; summary file **missing** until Q1's write-once summary).
- **Reuse.** None found under `results/figures/` for the gap study (no `scaled-bridge-gap*` figure). New figure.
  May be replaced by table T4 if space is short.

## Tables

### T1. Positioning against the closest prior work (related work, §2)

- Columns: work; formats (and bits); PTQ/QAT; exact or simulated accumulation; accumulator width determined how
  (worst-case formula / static bound / measured failure); hardware evidence level (analytical / FPGA / synthesis
  / post-layout); networks.
- Data: `docs/analysis/related-work-audit-2026-10-01.md` §1 entries and lane P1's full-text reading
  (`docs/analysis/closest-prior-work-accumulators-2026-10-02.md` Part A); `related-work-draft.md` holds the draft.
  Exists ([Des23] and [Ugu17] abstract-level only).

### T2. Formats and certified widths

- Columns: format; code levels (definition source `public/formats/` manifest); admitted to exact engine (yes/no and
  why); per network (ResNet18, MobileNetV2, MobileNetV3): published formula as printed, exact data-type bound in
  the project's convention, certificate abs/struct (the three-column form P1 recommends, L257-260).
- Data: `widths.csv` (ResNet18), engine L500-511 (MobileNets, network maxima by node kind; per-node in
  `results/summaries/scaled-bridge-v2/MN-1f75c9232c8a0202/summary.json`), sweep L281-289 (formula), P1 Part C
  table L238-247 for ResNet18 (script and output `artifacts/literature_v1/notes/partc_widths.py`,
  `partc_widths_out.txt`). **Exists for ResNet18**; the exact data-type bound column for the MobileNets is
  **missing** (a CPU computation from the stored certificates and manifests, like P1's).

### T3. Derived widths per format (headline numbers)

- Columns: format; exact top-1; W_cert; formula; W_noevent; W_acc(1.0); W_half; cert − W_acc(1.0); FP32-control,
  fp16, f21 differences.
- Data: `results/summaries/accumulator-sweep-v1/widths.csv`, `policies.csv` (**exist**); MobileNet rows
  **missing** (Q1). This is sweep L36-43 plus L196-203 in one table.

### T4 (optional). Accumulator-aware PTQ at equal width

- Columns: case; certificate; narrowest P within 1 pp for AXE, naive projection, saturating register.
- Data: `results/summaries/b2-axe-v1/{summary.json,arms.csv}` (**exists**; Q8 review approve_with_fixes, fixes
  applied, not re-reviewed). Mark the `sat.wP` rows above the event-free width as derived.
- Reuse: `results/figures/b2-axe-top1-vs-P-v3.{pdf,png}` (v2 drew the derived `sat.wP` points like measured ones;
  Q8 document L6-8).

## Existing figures that are not planned for the 6-page paper

| Figure | Why not | Possible use |
|---|---|---|
| `results/figures/b2-matrix-quality-v1`, `b2-matrix-vs-int8-v2` | Format breadth under simulation; supporting C1 only | Journal version; one sentence in §4 |
| `results/figures/b2-recon-matrix--r2` | AdaRound baseline | One sentence (baseline faithfulness) |
| `results/figures/b2-detector-v1`, `b2-detector-breadth-v2` (v1 lists Q1.6 as a format; use v2) | Detector is simulator-only | One sentence; journal |
| `results/figures/b2-seeds-{cells,contrasts,size}-v1` | Calibration seeds (Q5, unreviewed) | One sentence in limitations; journal |
| `results/figures/b2-collapse-*--r2`, `b2-attrib-*-v1` | Diagnostics | Journal |
| `results/figures/quality-metrics-*-v1` | Methods detail (resolution) | One sentence in statistics |
| `results/figures/b-stage-*`, `saved-predictions-*` | Superseded original-recipe evidence | None |
| `results/figures/memory-effective-bits.svg` | Memory study not part of this paper | None |
