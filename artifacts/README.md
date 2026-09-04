# Generated artifacts

Large reproducible outputs live here and are ignored by Git. SQLite/result manifests reference each artifact by path and cryptographic hash.

| Directory | Plan-derived contents |
|---|---|
| `checkpoints/` | Frozen pretrained model payloads and integrity metadata |
| `folded_graphs/` | Eval-mode, BN-folded deployment graphs |
| `calibration_stats/` | Ranges, histograms, scale/clipping searches, layer statistics |
| `quantized_weights/` | Calibrated quantized weights keyed by graph/manifest/PTQ identity |
| `encoded_tensors/` | Low-precision encoded tensors and optional packed/predecoded forms |
| `datatype_truth_tables/` | Exhaustive ADD/MUL/convert tables and oracle metadata |
| `per_image_predictions/` | Paired FP32/quantized predictions, correctness, optional logits |
| `traces/` | Operand sequences, SAIF/VCD, trace manifests, convergence runs |
| `rtl/` | Generated/elaborated RTL and implementation-specific netlists/reports |
| `ppa/` | Synthesis, placement, route, timing, power, and extraction products |

## Required metadata

Every artifact records or is recoverable from:

- producer experiment/hardware-run ID;
- canonical configuration and manifest hashes;
- source checkpoint/graph/subset identity where relevant;
- producer code/RTL/tool version;
- content hash, size, and semantic type;
- dependencies and whether it is safe to reuse under changed downstream settings.

## Lifecycle rules

- Write atomically and register only complete artifacts.
- Do not put hand-authored source or the only copy of a scientific result here.
- Reuse calibration/weights only when upstream dependencies are identical.
- Treat encoded/predecoded/packed variants as different artifacts even if numerically equivalent.
- Keep curated small summaries in `results/`, not only in ignored artifact output.
- Cache data may be deleted freely; artifacts may be regenerated but are part of the evidence chain.
