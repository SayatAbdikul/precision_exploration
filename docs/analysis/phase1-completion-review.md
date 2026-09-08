# Phase 1 completion review — 2026-09-08

Phase 1 has been reverified after the audit of the 2026-09-07 completion claim. The accepted 25-candidate scope and recorded ImageNet/EfficientNet/SKY130 scope decisions are unchanged.

| Audit gap | Completed correction | Verification |
|---|---|---|
| Incorrect OCP MX encodings, FP8 overflow and signed zero | Corrected MXFP4/MXFP6 manifests; oracle 1.1.0 rounds before overflow, preserves zero sign, isolates Decimal context and validates E8M0 scales | OCP vectors, boundary tests, independent PyTorch FP8 casts; regenerated 725 tables / 1,085,854 rows |
| Invalid or unresolved configurations could execute | Submission and execution both resolve and validate model/checkpoint/graph/preprocessing/list/manifest identities and supported semantic versions | Bad hashes, stale versions, strict-policy violations, invalid submissions and altered queued configurations rejected; real witness completes and deduplicates |
| Stale workers could finish reclaimed jobs | Renewable unique leases, fenced metrics/completion, retained failed attempts and separate retry attempts | Concurrent claims, heartbeat renewal, recovery and stale-token rejection |
| Module lists were mislabeled as frozen graphs | Actual Conv/BN folding and deterministic fixed-input ATen graph capture with edges, operation attributes and tensor hashes | Four graph hashes reproduced; no BatchNorm remains; eight real images per model pass fold parity and repeated inference |
| Malformed synthesis metrics were accepted | Parser 1.2.0 rejects NaN/Inf area, invalid per-cell counts and inconsistent totals, including Yosys submodule accounting | Malformed report tests and successful reparsing/idempotent ingestion of the real pilot |

Additional reliability work verifies dataset image and COCO annotation payloads, validates artifacts before registration, makes per-image batches atomic and identical retries idempotent, and checks exact dependency hashes before artifact reuse.

## Evidence

- `results/summaries/phase1-workload-verification.json`: all three full 1k classifier screens reproduce the frozen prediction files byte-for-byte. Eight real images per model verify folded/unfolded FP32 agreement, unchanged classifier Top-5, deterministic repeated folded inference, and reproducible deployment graph hashes.
- `results/summaries/phase1-completion-verification.json`: exit audit result and source/evidence hashes.
- `public/formats/conformance/truth-table-index.json`: 725 versioned tables, 1,085,854 rows, accepted-set hash `986344a9dc6ef5b4c7a8194e4675964e170345c43d50f95bbb3cee0dfcd82c0b`.
- `results/summaries/phase1-registry-export.json`: four baseline configurations/runs, 8,000 per-sample rows and four hardware records resolve to the rebuilt authoritative SQLite database.
- `public/experiments/configs/phase1_reference_witness.json`: executable validation/lifecycle integration witness with real frozen references and a concrete FP16 accumulator. It is not a network-level quantized inference result or an accumulator-quality selection.

Run `python3.10 -m pytest -q` and `python3.10 -m tools.analysis.verify_phase1_evidence` for the routine checks. The latter verifies all ten Phase 0 frozen hashes, every truth-table hash/row count, all 21,000 dataset-list payload references (nested lists count again), checkpoint/graph/preprocessing references, pilot reports, database integrity, artifact hashes and the deterministic database export. Workload reruns use `python3.10 -m tools.analysis.verify_phase1_workloads`.

## Evidence limits and handoff

Graph capture fixes the recorded input shape and FP32 preparation implementation. Phase 2 still owns low-precision operators, C++/CUDA execution and network conformance. The classifier baseline metrics remain the 1k screening references; the 10k list is staged and full ImageNet 50k validation is deferred to Phase 5. EfficientNet remains skipped with D3 open.

FP32 BN folding is algebraically equivalent but can change floating-point reduction rounding. Fold parity uses rtol 1e-4 / atol 1e-4 for logits/features. For YOLO decoded coordinates it uses rtol 1e-4 / atol 0.01 pixels, and for class probabilities rtol 1e-4 / atol 1e-5. Exact byte equality is required between repeated executions of the same folded graph. These tolerances do not relax the later bit-exact low-precision conformance requirement.

The full COCO baseline retains the original 5k predictions and official metric reproduction; this correction adds eight-image graph/folding/inference verification rather than claiming a second full 5k inference run. The ICS55 evidence remains the verified synthesis/STA pilot without placement, routing, extraction, power or characterized SRAM; D9 stays open. Phase 0 contract bytes are unchanged.
