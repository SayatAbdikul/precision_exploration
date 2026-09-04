# Tests

## `unit/`

- Manifest schema and semantic validation.
- Encode/decode, round, overflow, underflow, saturation, special-value, scale, and codebook behavior.
- Config canonicalization/hash stability.
- BN folding, bias-domain conversion, residual alignment, requantization, and LUT construction.
- Statistics, dominance, promotion, packing, and result-parser behavior.

## `conformance/`

- Exhaustive ADD/MUL/convert pairs for every feasible ≤8-bit format.
- Published standard vectors for FP8/MX/Posit where available.
- Reference versus C++ versus CUDA bit equality.
- Generic RTL versus oracle/backend equality on exhaustive vectors and representative traces.
- Straightforward RTL baseline versus generated DSE implementation semantic equality.

Primitive conformance target is bit exact.

## `integration/`

- Checkpoint → eval/BN fold → calibration → PTQ → exact inference → metrics.
- Fixed calibration/evaluation list separation and hashing.
- Scheduler lifecycle, atomic writes, resume, deduplication, and failure retention.
- C++/CUDA full-layer equality and network output/logit checks on a fixed image set.
- Trace generation → RTL replay → activity/power parsing.
- Candidate config → schema-validated public export package.

No private-importer integration test belongs in this repository.

## `regression/`

- Reproduced FP32 checkpoint quality.
- Conventional INT8 PTQ behavior.
- At least one standardized FP8/MX network reference where practical.
- Frozen format truth-table hashes.
- Deterministic experiment IDs and result-schema compatibility.
- Key final numerical and generic hardware evidence chains.

## Acceptance ladder before broad sweeps

1. Manifest and primitive oracle tests pass.
2. Optimized primitive results match exhaustive truth tables.
3. Dot products match at several reduction lengths.
4. Tiny deterministic Conv/DWConv cases match.
5. Full C++ and CUDA layers match bit for bit.
6. Fixed-image network outputs/logits match.
7. Scheduler resume/deduplication is proven.
