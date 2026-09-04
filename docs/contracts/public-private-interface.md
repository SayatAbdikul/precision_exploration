# External candidate-package interface contract

Status in supplied roadmap: **required in Phase 0; schema finalized before the D11 handoff**  
Repository scope: **export only**

## Boundary

This repository contains no MANT-specific implementation. It discovers the architecture-independent public Pareto frontier and exports a conservative candidate set. Import, mapping, simulation, RTL, and chip evaluation happen in a separate private MANT repository.

Public code must not require or encode MANT ISA, tile, memory, interconnect, compiler, mapping, simulator, RTL, floorplan, or proprietary chip details.

## Export population

After the public generic end-to-end study, export approximately 5–10 candidates comprising:

- highest measured quality;
- lowest generic energy;
- lowest generic area;
- highest images/J;
- highest images/s/mm²;
- two or three Pareto knee points;
- at least one credible alternative datatype family;
- any near-Pareto guard candidate with potentially favorable downstream compatibility.

Selecting the package must not retroactively alter the public results.

## `QuantizedModelPackage`

Each candidate package contains or content-addresses:

- frozen deployment model graph and graph hash;
- encoded weights and tensor shapes/layouts;
- weight, activation, accumulator, and output datatype manifests and hashes;
- tensor scales, block axes/sizes, metadata formats, and clipping thresholds;
- operator metadata and exact operator semantics version;
- bias representation and addition point;
- MAC model, product precision, reduction order, rounding, overflow, and underflow rules;
- calibration and evaluation manifest identities;
- checkpoint and preprocessing identities;
- measured public quality and confidence intervals;
- generic hardware implementation identity and PPA/energy summary;
- package schema version and content hash.

Large payloads may remain content-addressed artifacts, but the exported manifest must be sufficient to verify their hashes and recover the complete configuration.

## Validation before export

- Package schema validation succeeds.
- All referenced hashes resolve.
- Datatype manifests pass conformance tests.
- Reference, C++, and CUDA outputs match for the frozen verification set.
- Public quality results correspond to the exact packaged configuration.
- No MANT-specific field, path, source, or constant is present.
- Package identity is deterministic under canonical serialization.

## External importer responsibility

The separate private repository is responsible for schema compatibility, unsupported-operator reporting, private numerical delta measurement, and any conversion into its internal representation. Those importer and downstream tasks are documented in roadmap Phases 8–10 for context but are not implemented here.

## Versioning

Schema changes are explicit and versioned. Backward-incompatible changes require a major version increment. Package hashes include the schema version and all semantically relevant fields.
