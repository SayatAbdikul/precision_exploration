# D1 — Initial candidate manifest set

Status: **accepted**

Owner approval: **2026-09-06**

Frozen manifest-set SHA-256: `c859204f7e1ec3f1aa4a5b7381d811add705bf2ca6cb89dc1ee7bdadfd922e87`

## Decision

The initial public precision exploration contains 25 explicitly defined candidates:

- integer/fixed: INT8, INT6, INT5, INT4, and Q1.6;
- minifloat: FP8 E4M3FN, FP8 E5M2, FP7 E3M3, FP6 E3M2, FP6 E2M3, FP5 E2M2, and FP4 E2M1;
- block/shared-scale: BFP6, MXFP8 E4M3, MXFP6 E3M2, and MXFP4 E2M1, all with 32-value reduction-axis blocks and E8M0 shared scales;
- tapered: Posit(8,1), Posit(6,1), and Posit(4,0);
- logarithmic: LOG8, LOG6, and LOG4;
- non-uniform: NF4;
- very-low-cardinality: binary ±1 and ternary −1/0/+1 with one reserved code.

Every normative definition is in `public/formats/manifests/accepted/`; filenames are navigation only and identity is the canonical manifest content hash.

## Coverage and research value

| Family | Included widths | Reason retained |
|---|---|---|
| Integer/fixed | 4–8 | Mandatory efficiency anchors and one intrinsic fixed-point control |
| Float/minifloat | 4–8 | Standard FP8 anchors plus exponent/mantissa trade-offs below eight bits |
| BFP/MX | 4, 6, 8 | Tests native shared-scale behavior and its metadata/hardware cost |
| Posit | 4, 6, 8 | Represents tapered precision and enables later quire studies |
| Logarithmic | 4, 6, 8 | Represents power-friendly multiplication and non-linear spacing |
| Codebook | 4 | NF4 provides a widely used non-uniform reference |
| Binary/ternary | 1, 2 | Establishes extreme-efficiency bounds without implying expected quality |

No major roadmap family was removed. Custom variants are named by their complete semantics rather than borrowing a standard name.

## Redundancy and feasibility review

All 25 candidates have distinct width, scalar codebook, or scaling/block semantics. MX element encodings intentionally share scalar values with their corresponding minifloats at scale one, but they are not redundant experiments: an MX value is interpreted with an intrinsic E8M0 block scale and has different metadata, calibration, and hardware costs. BFP6 likewise remains distinct from fixed point because its exponent is shared per block. Binary and ternary require mapping scales and do not duplicate low-bit integer zero/code behavior.

The manifest parser and high-precision oracle load every accepted manifest. Exhaustive decode, every representable encode value, every adjacent-value rounding midpoint, ADD, MUL, and all ordered source-to-destination conversions are generated at scale one. Block families record the required 32-value context; block-scale selection is network/calibration work and is not mislabeled as a scalar truth-table property.

## Cost

The frozen build produces 25 exhaustive decode tables, 25 exhaustive encode-boundary tables, 25 ADD tables, 25 MUL tables, and 625 ordered conversion tables. Eight-bit binary operations have 65,536 pairs per operator; lower widths scale as `2^(2n)`. The tracked conformance index records every artifact hash and exact row count.

## Included, deferred, merged, and rejected

- Included: the 25 candidates above.
- Deferred: alternate biases, additional widths, unsigned variants, MXINT, other BFP block sizes, generalized NF codebooks, external-scale minifloats, and all accumulator sweeps. These remain valid later-phase dimensions.
- Merged: no approved candidate was merged; scalar-equal MX/minifloat pairs remain separate because their scaling semantics differ.
- Rejected: no additional unnamed or incompletely specified custom encoding may enter a sweep.

This decision establishes a broad starting set. It does not rank candidates or close D2–D11.
