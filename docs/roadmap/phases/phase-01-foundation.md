# Phase 1 — Build the public foundation in parallel

Scope: **public precision-exploration repository only**
Starts after: **Phase 0 contracts — complete 2026-09-04**
Parallelism: **high across WP1A, WP1B, and WP1C**
Status: **complete — 2026-09-07**

## Goal

Build the reproducible inputs, exact datatype reference, experiment identity and execution infrastructure, and generic ICS55 pilot flow required by later public numerical and hardware work.

Phase 1 is complete when:

- the mandatory FP32 workloads reproduce from frozen checkpoints, preprocessing, datasets, graphs, and sample lists;
- the project owner has accepted D1, and every accepted candidate is represented by a complete, validated datatype manifest;
- the high-precision oracle and feasible exhaustive truth tables establish the reference semantics for those manifests;
- canonical identities, the scheduler, SQLite registry, artifact handling, and per-image result storage pass recovery and deduplication tests;
- the generic ICS55 register-to-register pilot flow produces normalized, traceable results and the available memory-evidence level is recorded.

No private architecture, simulator, compiler, mapping, ISA, tile, interconnect, or chip work is part of this plan.

## Completion record

Phase 1 passed all revised exit gates on 2026-09-07. The revision is limited to the owner's explicit decisions: the 50,000-image ImageNet validation payload is deferred to Phase 5, EfficientNet is intentionally skipped with D3 open, and SKY130 is a fallback only if the public ICsprout55 route is unavailable. ICsprout55 was usable, so SKY130 was not exercised.

| Gate | Result | Canonical evidence |
|---|---|---|
| P1-A workloads | Four pinned checkpoints/models, six frozen dataset lists, five selection records, three reproducible classifier screens, and one complete COCO validation baseline | `public/workloads/models/manifests/index.json`, `data/manifests/index.json`, `results/summaries/phase1-fp32-baselines.json`, `results/summaries/phase1-fp32-reproduction.json` |
| P1-B semantics | D1 accepted with 25 formats; 725 exhaustive tables and 1,085,846 rows cover decode, encode boundaries, ADD, MUL, and all 625 ordered conversions | manifest-set SHA-256 `c859204f7e1ec3f1aa4a5b7381d811add705bf2ca6cb89dc1ee7bdadfd922e87`; `public/formats/conformance/truth-table-index.json` |
| P1-C infrastructure | Canonical identities, lifecycle/retry/recovery, concurrent atomic claims, artifacts/dependencies, 8,000 per-sample rows, and idempotent hardware ingestion pass | `results/databases/phase1.sqlite`, `results/summaries/phase1-registry-export.json` |
| P1-D generic hardware | ICsprout55 v1.10.102 RVT synthesis repeated byte-identically; 10/5/2 ns post-synthesis STA sweep normalized; strongest memory evidence level 3 | `results/summaries/ics55-phase1-pilot.json`, `docs/analysis/icsprout55-availability.md` |
| P1-E integration | Real checkpoint/graph/list/manifest references resolve; altered hashes and private package fields fail; all tests pass | 45 tests passing under Python 3.10 |

Known evidence limits are explicit and do not block this phase: classifier metrics are the frozen 1k class-balanced screening references, the 10k ImageNet list is staged for Experiment A, the full 50k list is deferred, and ICsprout55 timing is post-synthesis cell-delay evidence without placement, routing, extraction, power, or SRAM characterization. D3 and D9 remain open.

## Fixed inputs from Phase 0

Phase 1 must implement, not reinterpret, the accepted contracts:

- `docs/contracts/arithmetic.md`;
- `docs/contracts/experiment-a.md`;
- `docs/contracts/operator-semantics.md`;
- `docs/contracts/config-and-identity.md`;
- `docs/contracts/hardware-metrics.md`;
- `docs/contracts/public-private-interface.md`;
- `public/formats/manifests/datatype-manifest.schema.json`;
- `public/experiments/configs/experiment.schema.json`;
- `public/package/schema/quantized-model-package.schema.json`;
- `public/generic_rtl/harness/README.md`;
- valid, invalid, and canonical witnesses under `tests/conformance/fixtures/`.

F1–F8 remain fixed. A Phase 1 implementation change that contradicts one of these inputs requires a versioned contract change rather than an undocumented exception.

## Scope boundaries

### Included

| Work package | Roadmap tasks | Purpose |
|---|---|---|
| WP1A — Workloads | N1.1–N1.6 | Freeze models, datasets, preprocessing, subsets, and FP32 baselines |
| WP1B — Oracle and infrastructure | I1.1–I1.6, D1 | Establish exact format semantics and reproducible experiment execution |
| WP1C — Generic PDK flow | H1.1–H1.5 | Validate the generic ICS55 flow, metric ingestion, and memory evidence |

### Excluded

- Phase 2 network operators, C++/CUDA exact-inference kernels, and backend optimization.
- Phase 3 Experiment A calibration and quantized quality sweeps.
- Phase 4 accumulator, W/A, scaling, and MAC-model ablations.
- Phase 6 candidate RTL architecture exploration and broad PPA DSE.
- Private implementation work of any kind.

Representative arithmetic RTL in H1.1 exists only to validate the generic flow and harness. It is not a Phase 6 candidate search and cannot support a format-ranking claim.

## Decisions and required external inputs

### D1 — Initial candidate manifest set

D1 is the only formal roadmap decision closed in Phase 1. It must be presented to the project owner after the manifest definitions and oracle feasibility evidence exist.

The D1 decision package must contain:

1. one complete proposed manifest per candidate;
2. a family/width/semantics matrix covering every major family in the roadmap;
3. a semantic-distinctness and redundancy analysis based on decoded values and behavior;
4. evidence that each candidate can be implemented by the oracle and later backends;
5. estimated truth-table and screening cost;
6. expected numerical and generic hardware research value;
7. a clear list of included, deferred, merged, and rejected candidates with rationale.

D1 chooses a broad, implementable starting set. It does not choose a winner, prune a major family, freeze later accumulator sweeps, or decide D2–D11.

After owner acceptance:

- update D1 in `docs/decisions/decision-register.md`;
- freeze the accepted manifest list and its aggregate content hash;
- regenerate affected conformance vectors if the accepted set differs from the proposal;
- use only the accepted set for Phase 1 truth-table completion and Phase 2 backend planning.

### Inputs that must be available during execution

These are operational prerequisites, not new research gates:

- readable ImageNet-1K and COCO dataset locations with identifiable versions;
- exact checkpoint sources and payloads for ResNet-18, MobileNetV2, MobileNetV3, and one selected tiny detector;
- the selected detector implementation/evaluator and its checkpoint;
- licenses or permissions needed to use the datasets and checkpoints locally;
- a runnable public-preview ICS55 tool/library setup, its RVT cells, supported corners, and tool versions;
- any available ICS55 SRAM/compiler/memory collateral needed for H1.4.

The optional choices are:

- whether to stage EfficientNet-B0 or EfficientNet-Lite0 under N1.2; D3 later decides main-suite inclusion;
- whether an optional SKY130 cross-check is useful under H1.3.

Neither optional choice may delay the mandatory Phase 1 outputs. The owner authorized SKY130 as a fallback on 2026-09-07 if ICsprout55 proved unavailable. ICsprout55 public RVT collateral was usable, so the fallback condition did not occur and SKY130 was not exercised.

## Execution order

```text
Phase 0 complete
       |
       +--> WP1A workloads -----------------------------+
       |                                                |
       +--> WP1B validator + oracle --> D1 --> tables ---+--> integration review
       |             |                                  |
       |             +--> identity --> scheduler/DB -----+
       |                                                |
       +--> WP1C generic ICS55 flow ---------------------+
                                                        |
                                                        +--> Phase 1 exit
```

WP1A, WP1B, and WP1C should proceed concurrently. Within WP1B, semantic validation and the oracle precede D1; D1 precedes completion of the full accepted truth-table set. Canonical identity precedes scheduler deduplication and artifact registration.

## Wave 0 — Readiness and immutable input inventory

Before implementation work expands:

1. verify that every Phase 0 schema and fixture still passes;
2. inventory dataset, checkpoint, evaluator, Python/toolchain, ICS55, and optional SKY130 availability;
3. record versions and content hashes for every input already available;
4. identify missing licensed or proprietary inputs without copying them into Git;
5. confirm that tracked paths contain no host-specific absolute dataset/PDK paths;
6. keep generated payloads in the existing `artifacts/`, `data/raw/`, `cache/`, and `results/` boundaries.

Wave 0 does not add a new scientific task. It prevents the three work packages from freezing incompatible input identities.

## WP1A — Numerical workloads

Primary locations:

- `public/workloads/models/`;
- `public/workloads/datasets/`;
- `public/workloads/subsets/`;
- `data/manifests/calibration/`;
- `data/manifests/evaluation/`;
- `artifacts/checkpoints/` and `artifacts/folded_graphs/` for generated payloads;
- `results/summaries/` and `results/tables/` for compact baseline evidence;
- `tests/integration/` and `tests/regression/` for reproducibility checks.

### N1.1 — Freeze mandatory checkpoints

Freeze one exact checkpoint for each mandatory workload:

- ResNet-18;
- MobileNetV2;
- MobileNetV3;
- one tiny detector selected from the roadmap's YOLOv8n-or-similar class.

For each model record:

- canonical model and variant name;
- framework/package and version;
- upstream source and checkpoint identifier;
- checkpoint byte size and SHA-256;
- license/provenance reference;
- architecture/config identity;
- state-dictionary/load procedure;
- evaluation-mode setting;
- deployment-graph preparation version and graph hash;
- expected input shape and output interpretation;
- applicable dataset, evaluator, and metric.

Checkpoint payloads are content-addressed artifacts and are not duplicated in tracked source. A model manifest must fail verification if its payload, graph, or declared hash differs.

### N1.2 — Optionally stage EfficientNet

If the optional staging work is accepted, prepare either EfficientNet-B0 or EfficientNet-Lite0 with the same manifest and verification fields as N1.1. Staging proves that the workload can be loaded and evaluated; it does not add EfficientNet to the mandatory suite or close D3.

If skipped, record that N1.2 was intentionally not exercised and leave D3 open.

### N1.4 — Freeze preprocessing and dataset identity

Define an explicit preprocessing record for each workload before reproducing metrics. It must include:

- input decoding and color convention;
- resize and crop algorithm and parameters;
- normalization constants and tensor layout;
- interpolation and antialias behavior where applicable;
- batching behavior that can affect results;
- label/category mapping;
- dataset release/version and evaluator version;
- preprocessing implementation/version hash.

Dataset manifests identify local payloads without embedding host-specific roots in canonical experiment configuration. Verification checks required files and their declared dataset identity while raw ImageNet/COCO data remains untracked.

### N1.5 — Freeze ImageNet calibration and evaluation lists

Generate deterministic ordered lists for:

- 2,000 class-balanced training images for calibration, two per class;
- a stratified 1,000-image screening list;
- a stratified 10,000-image evaluation list.

Record the selection algorithm, seed, dataset split/version, image identifiers, labels, ordering, normalization rules, count, and SHA-256 for every list.

The calibration list must be disjoint from every evaluation list. The 1k list is a fixed subset of the 10k list so paired stage-to-stage comparisons remain reproducible. Tests must detect duplicates, missing samples, count errors, class-balance errors, overlap with calibration, and list-hash drift.

Owner scope decision, 2026-09-07: the 50,000-image ImageNet validation payload and full-list freeze are deferred to Phase 5 and are not a Phase 1 exit requirement. The 10k class-balanced evaluation list is the largest ImageNet payload frozen here.

### N1.6 — Freeze COCO calibration and screening lists

Generate fixed ordered lists for:

- approximately 1,000–2,000 training images for calibration;
- a fixed detector screening list from the evaluation split;
- the full evaluation list required to reproduce the reference metric.

The selection record must preserve approximate category frequency, object-size distribution, and objects-per-image distribution, and must capture the exact selection method, seed, annotation version, evaluator version, image IDs, order, count, and SHA-256. Calibration and evaluation images must not overlap.

### N1.3 — Reproduce FP32 baselines

Run the frozen model, preprocessing, dataset, and evaluator combination for every mandatory workload:

- Top-1 and Top-5 for ImageNet classifiers;
- COCO mAP50–95 and mAP50 for the tiny detector.

Store the canonical run identity, environment metadata, result summary, and per-image/per-sample outputs needed to reproduce the aggregate. Compare the measurement with the checkpoint's declared upstream reference. A mismatch is not hidden by changing preprocessing: diagnose and record checkpoint, graph, transform, evaluator, nondeterminism, and software-version differences, then either reproduce or explicitly approve the measured baseline as the frozen project reference.

### WP1A acceptance

| Task | Completion evidence |
|---|---|
| N1.1 | Four mandatory checkpoint/model manifests resolve and reproduce their byte and graph hashes |
| N1.2 | Optional workload is staged with full identity, or the intentional skip is recorded without closing D3 |
| N1.3 | FP32 metrics and per-sample evidence reproduce from the frozen inputs |
| N1.4 | Preprocessing, dataset, label/category, and evaluator versions are explicit and hashable |
| N1.5 | ImageNet list counts, class balance, hashes, ordering, and calibration/evaluation disjointness pass tests |
| N1.6 | COCO list counts, stratification summary, hashes, ordering, and disjointness pass tests |

## WP1B — Exact arithmetic and experiment infrastructure

Primary locations:

- `public/formats/oracle/`;
- `public/formats/manifests/`;
- `public/formats/conformance/`;
- `public/experiments/configs/`;
- `public/experiments/scheduler/`;
- `public/experiments/registry/`;
- `artifacts/datatype_truth_tables/` and other existing artifact categories;
- `results/databases/`;
- `tests/unit/`, `tests/conformance/`, and `tests/integration/`.

### I1.2 — Implement manifest parsing and semantic validation

Implement versioned loading for the Phase 0 JSON Schema plus the cross-field rules in the arithmetic and identity contracts.

The validator must:

- reject unknown schema versions and unknown fields;
- enforce family-specific required fields and internally consistent bit allocations;
- enforce exact codebook cardinality and uniqueness;
- validate zero, reserved, subnormal, NaN, infinity, overflow, underflow, rounding, and saturation semantics;
- validate intrinsic/required/none scaling and MX/BFP block metadata;
- enforce role-aware widths: at most 8 bits for weight, activation, and output roles while permitting wider accumulator manifests;
- verify that standard names have standard semantics and require distinct names for variants;
- resolve name/hash references by canonical manifest content rather than filename alone;
- produce deterministic, actionable error records.

The parser returns one immutable normalized representation consumed by the oracle, truth-table generator, configuration resolver, and later backends. It may not infer unspecified numerical behavior.

### I1.1 — Implement the high-precision datatype oracle

Implement the accepted `NumberFormat` behavior:

```text
encode, decode, add, mul, requantize, convert
```

Use exact integer/rational or sufficient high-precision host arithmetic internally so the final code is the correctly rounded result required by the manifest. The host's native low-precision or unrestricted FP32 behavior is not the oracle.

For every implemented family, verify:

- complete encode/decode behavior for all codes;
- representable extrema, zero/sign-zero, reserved values, and special values;
- exact tie handling for the declared rounding mode;
- overflow, underflow, saturation, and flush/subnormal behavior;
- required scale and block semantics;
- conversion between accepted source and destination manifests;
- determinism across repeated runs and supported hosts.

The Phase 0 FP6 witness must reproduce exactly before additional D1 candidates are trusted.

### Decision D1 — Ratify the initial manifests

Prepare the D1 evidence package from fully explicit proposed manifests and oracle feasibility checks. Ask the project owner to accept or revise the candidate set. Do not continue to the complete accepted truth-table build until D1 is recorded.

The review must preserve representation from:

- integer/fixed point;
- conventional/custom minifloat;
- BFP/MX;
- Posit/tapered precision;
- logarithmic/power-of-two;
- codebook/non-uniform;
- binary/ternary/low-cardinality;
- clearly distinct custom research formats, if any are proposed.

Redundancy is determined from complete semantics, not names alone. Two manifests that decode to the same codebook but differ in unsupported or irrelevant labels should not create duplicate sweep work.

### I1.3 — Generate truth tables and conformance vectors

For each D1 manifest:

- enumerate all input codes for decode and encode/round boundary vectors;
- exhaust all ADD and MUL input-code pairs through 8 bits where the operation is defined and feasible;
- generate source-to-destination conversion vectors for accepted conversion pairs where feasible;
- include boundary, halfway, sign, zero, overflow, underflow, saturation, reserved, and special-value cases;
- include required scale/block context when a scalar table alone is insufficient;
- record manifest hash, oracle version, table schema/version, row count, generation command/config, content hash, and completeness status.

Truth tables are content-addressed artifacts. Compact metadata and conformance samples remain tracked. If a table is not exhaustive, its exact coverage and reason must be recorded; sampled evidence cannot be labeled exhaustive.

### I1.5 — Implement canonical identity and metadata capture

Implement the Phase 0 canonicalization algorithm exactly:

1. validate and semantically resolve the input;
2. remove only the derived identity field;
3. canonicalize as sorted, compact UTF-8 JSON with one terminal LF;
4. preserve array order and reject non-finite or ambiguous numeric identity values;
5. reject absolute, parent-relative, environment-expanded, or host-specific canonical paths;
6. compute lowercase SHA-256 over the canonical bytes.

The implementation must reproduce the recorded Phase 0 canonical byte stream and hash. It must also:

- canonicalize semantically equivalent JSON/YAML input identically after parsing;
- detect a one-field semantic mutation as a different identity;
- verify every referenced manifest, checkpoint, graph, dataset list, and artifact hash;
- verify calibration/evaluation disjointness;
- resolve `family_appropriate_wide` to a concrete accumulator manifest before executable experiment identity is generated;
- enforce Experiment A Model C, scale, bias, operator, and requantization cross-references;
- capture software revision and clean/dirty state plus runtime/tool metadata without confusing performance-only metadata with numerical identity.

### I1.4 — Build the scheduler, SQLite registry, artifact store, and lifecycle

Build the smallest infrastructure that satisfies the accepted lifecycle:

```text
PENDING -> RUNNING -> COMPLETED
                 \-> FAILED
                 \-> INVALID
```

The authoritative SQLite registry must represent:

- canonical configurations and experiment IDs;
- model, dataset/list, manifest, PTQ, arithmetic, runtime, and software provenance;
- run attempts, state transitions, timestamps, worker identity, and retained errors;
- metrics and per-image result references;
- content-addressed artifacts, hashes, sizes, semantic types, producers, and dependencies;
- generic hardware runs and normalized metric references required by H1.5.

Required behavior:

- atomically register canonical configurations;
- deduplicate completed work by experiment ID;
- safely claim one pending job once;
- reject invalid configurations before execution;
- retain failed attempts and logs;
- recover or reschedule interrupted RUNNING work by an explicit stale-job policy;
- publish artifacts only after complete atomic writes and verified hashes;
- distinguish disposable `cache/` contents from authoritative `artifacts/` and `results/databases/` records;
- enforce dependency-based artifact reuse from the Phase 0 invalidation matrix.

Phase 1 needs a reliable local scheduler and registry, not a distributed orchestration platform.

### I1.6 — Store per-image results for paired statistics

Store one result row per experiment and sample identity with:

- ordered image/sample ID;
- ground truth;
- FP32 prediction;
- candidate prediction when later available;
- FP32 correctness and candidate correctness;
- optional logits/confidence summaries when configured;
- model, subset, and experiment identities;
- row/artifact version and provenance.

Phase 1 proves the storage path with FP32 baseline evidence and deterministic fixtures. It does not run quantized Experiment A. Enforce uniqueness so retries cannot duplicate a sample and preserve enough ordering/identity information for later paired bootstrap and McNemar analysis without rerunning inference.

### WP1B acceptance

| Task | Completion evidence |
|---|---|
| I1.1 | Oracle unit/boundary tests and the Phase 0 FP6 witness pass deterministically |
| I1.2 | All Phase 0 valid fixtures pass; all invalid fixtures fail; new family/cross-reference tests pass |
| I1.3 | Every D1 manifest has hashed completeness metadata and all feasible exhaustive tables |
| I1.4 | Lifecycle, atomic claim, retry, failure retention, deduplication, and artifact-recovery integration tests pass |
| I1.5 | Golden canonical hash, semantic mutation, reference/hash, path, alias-resolution, and metadata tests pass |
| I1.6 | Per-image uniqueness, ordering, round-trip, and later paired-query fixtures pass |
| D1 | Project owner acceptance and evidence hashes are recorded in the decision register |

## WP1C — Generic hardware-flow foundation

Primary locations:

- `public/pdk_flow/common/`;
- `public/pdk_flow/ics55/`;
- `public/pdk_flow/sky130/` only if H1.3 is exercised;
- `public/generic_rtl/harness/`;
- representative smoke-test RTL under the existing generic RTL hierarchy;
- `artifacts/rtl/` and `artifacts/ppa/` for generated flow products;
- `results/databases/` and compact tracked summaries;
- `tests/integration/` and `tests/regression/`.

### H1.1 — Bring up the ICS55 pilot flow

Create a reproducible parameterized flow for simple representative arithmetic blocks using the accepted generic timing harness. The smoke set should be just large enough to exercise combinational and registered paths, such as a simple integer adder/multiplier and a registered arithmetic path; it must not expand into the Phase 6 architecture search.

The flow must:

- keep downloaded PDK/library release payloads outside Git even when publicly licensed;
- consume explicit tool, PDK, library, corner, VT, RTL, constraint, and harness identities;
- elaborate, synthesize, and run the available early physical stages;
- retain commands/configuration, logs, reports, failures, and hashes;
- separate neutral wrapper cost from DUT core and architecture-owned registers;
- emit enough evidence to reproduce the pilot on the same authorized environment.

### H1.2 — Validate the harness, corners, and RVT baseline

Apply the Phase 0 register-to-register harness contract and verify:

- launch/capture boundaries and reset/valid behavior;
- input/output constraints, drive, load, clock uncertainty, and library corner selection;
- ICS55 RVT as the baseline VT class;
- latency, initiation interval, target clock, achieved slack, and fmax reporting;
- core-only and all-in accounting boundaries;
- common progressively tighter pilot constraints without selecting final Phase 6 clocks;
- repeatability of the same RTL/configuration identity.

Any unavailable corner, model, extraction capability, or physical stage is labeled as unavailable rather than approximated silently.

### H1.3 — Optionally prepare a SKY130 cross-check

Exercise this task only if it provides a useful mature-flow sanity check for scripts or report normalization. Use the same logical representative RTL and metric schema, while assigning distinct PDK/library/corner/tool identities.

SKY130 results are cross-check evidence only. They do not replace ICS55, get numerically scaled into ICS55 claims, or establish cross-node format ranking.

### H1.4 — Record ICS55 SRAM and memory support

Investigate and record what is actually available for:

1. characterized ICS55 SRAM/compiler evidence;
2. validated technology-specific OpenRAM/generated memory evidence;
3. synthesized local-memory evidence;
4. analytical fallback evidence.

For each available route, record source, access constraints, supported widths/depths/ports, PVT/corner coverage, timing/power/area outputs, tool compatibility, validation status, and permitted publication level. Record missing collateral explicitly.

This task establishes the evidence level and informs later D9. It does not decide D9 or invent physical SRAM precision that Phase 1 cannot verify.

### H1.5 — Normalize and ingest synthesis results

Build a versioned parser/interface that converts tool reports into the Phase 0 metric vocabulary:

- PDK, library, corner, VT, tools, RTL, harness, constraints, pipeline, and target-clock identity;
- pass/fail and timing status;
- area with hierarchy and wrapper/core boundary;
- cell/register counts when available;
- critical path, slack, fmax, latency, and initiation interval;
- leakage, dynamic power, throughput, and energy/op only when supported by the evidence;
- evidence level, report paths/hashes, parser version, warnings, and missing fields.

Use explicit units and reject ambiguous or malformed reports. Ingest idempotently into the authoritative results interface so parsing the same reports twice does not create a second hardware run.

### WP1C acceptance

| Task | Completion evidence |
|---|---|
| H1.1 | At least one representative block completes the available ICS55 pilot stages with reproducible hashes/reports |
| H1.2 | RVT/corner/harness assumptions and latency/II/timing/accounting fields are verified |
| H1.3 | Optional cross-check is reproducible and clearly separated, or the intentional skip is recorded |
| H1.4 | ICS55 memory capability matrix states the strongest defensible evidence and all limitations |
| H1.5 | Golden report fixtures, malformed-report tests, unit conversion, missing-field, and idempotent-ingestion tests pass |

## Wave 3 — Cross-work-package integration

After the three work packages are individually ready, run the following end-to-end checks:

1. Load every D1 manifest through schema and semantic validation.
2. Reproduce its canonical manifest hash and resolve all manifest references by content.
3. Reproduce the Phase 0 canonical experiment hash exactly.
4. Resolve one valid configuration containing frozen model/checkpoint/graph, dataset lists, preprocessing, manifests, accumulator, arithmetic, and runtime metadata.
5. Reject the same configuration when one referenced hash, disjointness claim, role width, path, or Experiment A semantic field is invalid.
6. Register the valid configuration once, deduplicate a repeated submission, claim it atomically, and exercise COMPLETED, FAILED, INVALID, and interrupted-job recovery paths with fixtures.
7. Register and retrieve a content-addressed truth table and a per-image baseline result artifact.
8. Parse and ingest one ICS55 pilot result, preserving its complete hardware identity and evidence level.
9. Regenerate compact result exports from SQLite and confirm they resolve back to authoritative run/artifact identities.
10. Confirm that no tracked source, schema, configuration, result field, or dependency contains private implementation knowledge.

The integration fixture may use the existing FP6 witness and synthetic/local test data. It must not claim network-level quantized inference, which belongs to Phase 2 and Phase 3.

## Verification strategy

### Unit tests

- canonical JSON serialization and hashing;
- each manifest cross-field rule;
- every oracle primitive and rounding boundary;
- subset selection, normalization, balance, ordering, and disjointness helpers;
- scheduler transition and artifact-dependency logic;
- hardware report parsing and unit normalization.

### Conformance tests

- Phase 0 valid/invalid schema fixtures;
- all-code decode/encode checks;
- exhaustive feasible ADD/MUL/convert truth tables;
- per-family special-value and scaling vectors;
- canonical identity golden vector and mutation vectors.

### Integration tests

- model/checkpoint/preprocessing/list resolution;
- deterministic FP32 baseline rerun on a fixed small verification set;
- submit → claim → complete/fail/recover/deduplicate job lifecycle;
- atomic artifact publication and corruption/hash rejection;
- per-image result round trip;
- ICS55 report parse and idempotent database ingestion.

### Regression tests

- mandatory FP32 metric summaries remain tied to unchanged input identities;
- D1 manifest and truth-table hashes remain stable;
- schema/oracle/config semantic versions change explicitly when behavior changes;
- representative ICS55 pilot results remain parseable under the frozen parser fixture.

## Task dependency and completion matrix

| Task | Starts after | Blocks | Required final artifact/evidence |
|---|---|---|---|
| N1.1 | Phase 0; checkpoint access | N1.3, later graph/inference work | Frozen mandatory model/checkpoint manifests and hashes |
| N1.2 | Optional checkpoint access | D3 evidence only | Staged manifest/evidence or recorded skip |
| N1.3 | N1.1, N1.4–N1.6 | Experiment A trust | Reproduced FP32 metrics and per-sample evidence |
| N1.4 | Dataset/evaluator access | N1.3, config identity | Frozen preprocessing/dataset/evaluator manifests |
| N1.5 | ImageNet access and N1.4 identity | ImageNet calibration/screening | Hashed 2k/1k/10k ordered lists and tests; full 50k deferred by owner |
| N1.6 | COCO access and N1.4 identity | Detector calibration/screening | Hashed calibration/screening/full lists and tests |
| I1.1 | Arithmetic contract and parser interface | D1, I1.3, Phase 2 | Deterministic high-precision oracle |
| I1.2 | Phase 0 schema/contracts | D1, I1.1, I1.5 | Parser, semantic validator, normalized manifest object |
| D1 | Initial manifest definitions plus I1.1/I1.2 feasibility | I1.3 completion, Phase 2 family scope | Accepted decision record and manifest-set hash |
| I1.3 | D1 and oracle | Backend/RTL conformance | Hashed exhaustive-feasible tables and coverage records |
| I1.5 | Phase 0 canonical identity contract | I1.4 dedup/artifacts | Canonicalizer, resolver, metadata capture, golden tests |
| I1.4 | I1.5 identity | Reproducible sweeps | SQLite registry, scheduler, artifact lifecycle tests |
| I1.6 | I1.4 schema and N1.3 baseline shape | Paired statistics | Per-image schema/storage/query tests |
| H1.1 | ICS55 access and harness contract | H1.2, later hardware DSE | Reproducible representative ICS55 pilot run |
| H1.2 | H1.1 | Comparable generic results | Verified RVT/corner/harness/timing record |
| H1.3 | Optional usefulness decision | Nothing mandatory | SKY130 cross-check or recorded skip |
| H1.4 | ICS55 collateral investigation | Later D9 | Memory evidence capability matrix |
| H1.5 | Representative tool reports and I1.4 results interface | Later hardware analysis | Normalized parser, fixtures, idempotent ingestion |

## Review gates

### Gate P1-A — Workload freeze

Review model provenance, hashes, preprocessing, graph identity, dataset/list identities, subset separation, evaluator versions, baseline metrics, and per-image evidence. Any unresolved mismatch keeps the affected workload open.

### Gate P1-B — D1 and semantic foundation

Review manifest completeness, family coverage, redundancy, oracle feasibility, special values, scaling semantics, canonical identity, and table coverage. The project owner must explicitly close D1.

### Gate P1-C — Infrastructure reliability

Force duplicate submissions, invalid configs, failed jobs, interruption/recovery, artifact corruption, and repeated per-image/hardware ingestion. Phase 1 cannot pass on happy-path execution alone.

### Gate P1-D — Generic hardware-flow credibility

Review the ICS55 environment identity, RVT/corners, timing constraints, wrapper accounting, parser output, memory capability, and stated limitations. A successful command without traceable reports and normalized identity is insufficient.

### Gate P1-E — Cross-contract consistency

Numerical, reproducibility, and generic hardware reviewers confirm that one resolved configuration still has one expected numerical identity and that all evidence resolves by content hash across manifests, workloads, runs, artifacts, and hardware records.

## Risks and in-scope mitigations

| Risk | Consequence | Mitigation within Phase 1 |
|---|---|---|
| Dataset/checkpoint access or license ambiguity | Workload freeze cannot complete | Record exact missing input; do not substitute silently or commit restricted payloads |
| Upstream FP32 metric does not reproduce | Quantized deltas become unreliable | Diagnose checkpoint, graph, preprocessing, evaluator, and version differences before freezing |
| D1 candidate explosion | Excessive oracle/table/backend burden | Require redundancy and feasibility evidence while preserving all major families |
| Ambiguous custom format semantics | Different backends produce different answers | Reject incomplete manifests; never infer a missing rule |
| Host floating-point dependence | Non-portable oracle or hashes | Use exact/high-precision reference arithmetic and canonical bytes |
| SQLite concurrency or interrupted writes | Duplicate/corrupt experiments | Atomic claims, transactions, stale-job policy, content verification, and failure tests |
| Artifact/cache confusion | Invalid reuse or irreproducible evidence | Enforce authoritative artifact records and Phase 0 dependency invalidation rules |
| ICS55 collateral or corner gaps | Weak hardware evidence | Record capability/absence explicitly; keep evidence levels distinct |
| Tool-report format drift | Silent metric corruption | Versioned parser fixtures, strict units, missing-field errors, and raw report hashes |
| Optional work expands the critical path | Mandatory outputs stall | Keep N1.2 and H1.3 conditional and non-blocking |

## Phase 1 exit checklist

- [x] N1.1 mandatory checkpoint/model manifests are frozen and verified.
- [x] N1.2 EfficientNet staging is explicitly skipped without closing D3.
- [x] N1.3 mandatory FP32 Top-1/Top-5 and COCO mAP baselines reproduce.
- [x] N1.4 preprocessing, dataset, label/category, graph, and evaluator identities are frozen.
- [x] N1.5 ImageNet 2k calibration and 1k/10k evaluation lists pass identity and separation checks; full 50k is owner-deferred.
- [x] N1.6 COCO calibration/screening/full lists pass identity, stratification, and separation checks.
- [x] I1.1 the high-precision datatype oracle passes unit and witness tests.
- [x] I1.2 manifest schema plus semantic and cross-reference validation passes.
- [x] D1 is accepted by the project owner and the accepted manifest-set hash is recorded.
- [x] I1.3 all feasible exhaustive truth tables and coverage records are generated and hashed.
- [x] I1.4 scheduler, SQLite registry, artifact storage, lifecycle, recovery, and deduplication pass.
- [x] I1.5 canonical identity reproduces the Phase 0 golden vector and rejects invalid references/paths.
- [x] I1.6 per-image result storage supports deterministic paired queries without duplicate rows.
- [x] H1.1 the representative ICS55 pilot flow completes with reproducible evidence.
- [x] H1.2 timing harness, RVT, corner, timing, latency/II, and accounting assumptions are validated.
- [x] H1.3 the optional SKY130 cross-check is explicitly skipped because ICsprout55 is usable.
- [x] H1.4 the strongest available ICS55 memory evidence and its limitations are recorded.
- [x] H1.5 synthesis and post-synthesis STA reports parse into the standard metric interface and ingest idempotently.
- [x] Cross-work-package integration and regression checks pass.
- [x] No Phase 2/3/4/6 implementation or private implementation work has entered Phase 1.

## Outputs

- Reproducible FP32 baselines and per-sample evidence.
- Frozen dataset, checkpoint, graph, preprocessing, evaluator, calibration, and evaluation manifests.
- Accepted D1 candidate manifest set.
- Validated datatype manifests, high-precision oracle, truth tables, and conformance vectors.
- Deterministic configuration hashing, metadata capture, and reference resolution.
- Resumable, deduplicating scheduler; authoritative SQLite registry; content-addressed artifact handling; and per-image result storage.
- Working generic ICS55 pilot flow, validated timing harness/RVT/corners, normalized result ingestion, and recorded memory capability.
- Optional EfficientNet and SKY130 staging evidence only when exercised.

## Downstream handoff

- WP1A plus the later Phase 2 exact engine unblocks Phase 3 Experiment A.
- WP1B plus the accepted operator contract unblocks Phase 2 exact-engine implementation.
- WP1C unblocks later generic hardware DSE but is not on the Experiment A numerical critical path.
- D2–D11 remain open and require their own later evidence.
