# Independent review of the hardware foundations

Reviewed on 2026-09-27. Task: `01a0e2e6-a9f3-7991-b22d-48ca887235d2`
(Hardware foundations: integer MAC, ICS55…). Implementation worktree:
`/home/maveric/.codex/worktrees/05f6/precision_exploration`.

## Assessment

The raw integer RTL and published analytical memory tables reproduce. The work
is a useful foundation, with specific correctness/provenance fixes below.
Assignment 2 remains partial: mapped synthesis is complete, but no new timing,
placement, CTS, routing, extracted timing, activity power, or physical SRAM
measurement has been demonstrated. Reports describe those limitations honestly.
The missing runtime/tool setup remains engineering work, distinct from missing
technology collateral. No full-network quality is attached to these raw MACs.

## Independent verification

- All ten MAC configurations reran: 164,489 cycles and 27,627 dot results passed.
  Saved vector, testbench, generated RTL hashes, and counts matched exactly.
  Review outputs: `/tmp/mac-recheck-0co66lo9`.
- 36 selected MAC/requirements, memory, hardware-parser, resource, and
  acceptance/storage tests passed. MAC outputs were redirected to
  `/tmp/precision-hardware-review-suite-8se5089i`; no original artifacts changed.
- All-format memory tables (50 rows), bandwidth roofs (25), per-row-scale and
  hypothetical-block tables regenerated identically. All 100 retained graph
  hashes and four shape hashes were verified. One graph from each model was
  independently recomputed from its actual constants and encodings.
- Flow-state failure/success fixtures reproduce the reporting defects below.
  Fixture outputs: `/tmp/ics55-review-stage-repro-3vudzr7g`. These fixtures are
  software tests, not physical measurements.
- INT4/INT32 mapped synthesis independently reran with a byte-identical netlist:
  2,006.48 core + 574.56 wrapper = 2,581.04 Liberty area units, matching saved
  evidence. Outputs: `/tmp/ics55-review-resynthesis-4i0o37lf`.

## Required corrections

1. **Verify accumulator-resolution content before using its identity.**
   `public/generic_rtl/mac/requirements.py:47` reads the resolution document but
   copies the indexed SHA without checking the bytes. A read-only test changing
   the ResNet18/INT8 resolution from INT64 to INT32 was accepted with the old
   resolution hash. Current eleven files match their hashes; saved arithmetic
   evidence is not invalidated. Add tamper rejection and verify all referenced
   numerical dependencies before emitting specifications.

2. **Make physical stage status reflect execution, and fail requested stages.**
   `public/pdk_flow/ics55/validate_integer.py:205` hard-codes STA/placement as
   unavailable even when supplied tools succeed. Conversely, the CLI returns
   success when all three explicitly requested STA runs fail, since its exit
   check at line 237 examines synthesis only. Preserve successful synthesis
   evidence when a later placement stage fails; represent each stage as not
   requested, unavailable, failed, or completed from actual execution. Test the
   entire success/failure matrix. Record driver/Tcl/constraint identities too.

3. **Complete binary and ternary requirements.**
   `public/generic_rtl/mac/requirements.py:42` emits null code mappings for these
   families. Include manifest-defined code meanings and reserved-code behavior,
   and verify them against oracle decoding. This is specification incompleteness,
   not an observed integer RTL defect.

4. **Correct three memory API edge cases.** In
   `public/analysis/memory_model.py`, line 155 flags cross-word handling from
   width divisibility alone: row-aligned shape [2,3], INT5, 32-bit words needs no
   crossing but is flagged. Line 165 omits bias/accumulator padding from the
   numerator while their allocation is in the denominator: shape [4] binary_pm1,
   64-bit words, three 32-bit biases and four 32-bit accumulators reports 18.75%
   instead of 28.75% total padding. Line 101 treats explicit block_size=0 as the
   default 32 through `or`; reject zero separately from None. Published tables
   do not exercise the affected cases.

5. **Make verification output locations configurable.** The MAC test currently
   writes to fixed repository artifacts. Use a caller-supplied output directory
   and pytest temporary directories so independent or concurrent validation does
   not overwrite recorded evidence. Preserve byte-identical existing vectors.

## Follow-up assignments

1. Fix the review findings, add regressions, and regenerate affected evidence.
2. Build or use a task-owned timing/physical runtime; reproduce ICS55 STA for
   the integer matrix and a placement pilot where supported. A read-only external
   editable-install lock error is a runtime/permission issue to resolve, not
   proof that the PDK cannot support placement. Keep actual RC/SRAM gaps explicit.
3. Implement a bounded, exactly specified integer scale/output-conversion block
   (power-of-two first), with RNE, saturation, signed boundary cases and an
   oracle-verified MAC integration. Unsupported general scales must be rejected
   or explicitly classified; do not inherit existing network quality scores.
4. Extend the memory analysis to comparable tiled layouts and a declared layer
   execution schedule, aligning activation allocation, bank contention, metadata
   reuse, bias/output/accumulator traffic, and external bandwidth assumptions.

Continue in the isolated worktree. Preserve the numerical campaign and its
running source/dependency identities. Final candidate selection, memory-policy
acceptance, workload energy and quality-linked Pareto rankings remain open.
