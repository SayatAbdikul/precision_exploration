# Phase 0 schema and identity fixtures

These fixtures prove the Phase 0 contracts are machine-expressible. They are illustrative and do not decide the D1 candidate set or report measured quality/hardware results.

## Expected-valid fixtures

- `manifests/valid/fp6_e3m2_finite.json` validates against the datatype schema.
- `experiments/valid/experiment_a_fp6_e3m2.json` validates against the experiment schema and demonstrates F1–F6.
- `packages/valid/fp6_e3m2_guard_package.json` validates against the export schema; all numeric results are explicitly illustrative.

## Expected-invalid fixtures

- Missing datatype rounding semantics.
- Floating-point family without float-layout/special-value semantics.
- Zero-bit integer format.
- Experiment A using optional general external scaling.
- Experiment A using native MAC instead of Model C/full product.
- Public package containing a private MANT field.

Each invalid fixture must fail its corresponding JSON Schema validation.

## Canonical identity witness

`canonical/experiment_a_fp6_e3m2.canonical.json` is the compact, recursively key-sorted, UTF-8/LF serialization of the valid experiment witness with no `experiment_id` field. Its SHA-256 is recorded in the adjacent `.sha256` file.

Phase 1 must reproduce the exact canonical bytes and hash before its config identity implementation is accepted.
