# Phase 0 — Freeze the experiment contracts

Scope: **in repository**  
Criticality: **critical path**  
Status: **complete — 2026-09-04**

## Purpose

Remove semantic ambiguity before large-scale implementation. One experiment configuration must determine one expected numerical result, and public precision-exploration code must remain independent of MANT.

## Numerical/research tasks

- **N0.1** — Write the canonical arithmetic contract for Model C.
- **N0.2** — Define the accumulator-format interface and every conversion/rounding point.
- **N0.3** — Apply the revised Experiment A scaling rule: intrinsic/necessary scaling only; no optional external scale.
- **N0.4** — Freeze deployment-graph order: eval mode → BN folding → calibration → PTQ.
- **N0.5** — Define operator semantics and strict/practical low-precision modes.

Primary locations: `docs/contracts/arithmetic.md`, `experiment-a.md`, `operator-semantics.md`.

## Software/reproducibility tasks

- **I0.1** — Define the canonical experiment configuration schema.
- **I0.2** — Define the datatype-manifest schema.
- **I0.3** — Define content-hash identity rules for runs and artifacts.
- **I0.4** — Define the public candidate package exported to an external private MANT repository.

Primary locations: `docs/contracts/config-and-identity.md`, `public/formats/manifests/`, `public/package/schema/`.

## Generic hardware tasks

- **H0.1** — Freeze area, fmax, II, latency, throughput, leakage, dynamic power, and energy/op definitions.
- **H0.2** — Freeze public hierarchy: primitive → MAC/FPU → scaling/memory → generic PE/tile/system.
- **H0.3** — Define the standard register-to-register timing harness.

Primary locations: `docs/contracts/hardware-metrics.md`, `public/generic_rtl/harness/`.

## External-boundary tasks from the source roadmap

- **M0.1** — MANT ISA, tile, interconnect, mapping/compiler, and golden model remain private and outside this repository.
- **M0.2** — Define only the export/import boundary; do not redesign MANT during the public study.

This repository implements only the exporter/schema side.

## Source-stated baseline decisions in force for the plan

- F1: Experiment A uses Model C.
- F2: Experiment A uses a sufficiently wide family-appropriate accumulator.
- F3: W and A formats are independently configurable.
- F4: accumulator is independently configurable.
- F5: controlled static PTQ, folded BN, fixed subsets, no retraining, no bias correction/reconstruction.
- F6: no optional external scale in Experiment A.
- F7: public study is architecture independent from MANT.
- F8: export Pareto plus guard candidates rather than one winner.

Their repository approval status is tracked separately in `docs/decisions/decision-register.md`.

## Completion plan

Phase 0 should be completed in four ordered waves. Drafting may run in parallel inside a wave, but no contract is frozen until the cross-contract review in Wave 3 succeeds.

### Wave 1 — Ratify the baseline and vocabulary

#### Review F1–F8

Responsible role: project/research owner
Inputs: supplied plan, supplied roadmap, `docs/analysis/document-synthesis.md`
Output: updated status and rationale for F1–F8 in `docs/decisions/decision-register.md`

For each F decision:

1. confirm that the roadmap wording is the intended baseline;
2. accept it, revise it explicitly, or leave Phase 0 blocked;
3. record owner, reviewers, date, evidence/source, rationale, and affected contracts;
4. if revised, update every dependent contract rather than keeping two definitions.

F6 requires explicit attention because the earlier plan used general external scaling in Experiment A while the later roadmap moved optional scaling to Experiment B.

#### Freeze shared terminology

Responsible roles: numerical, infrastructure, and hardware reviewers
Output locations: all six files in `docs/contracts/`

Agree on unambiguous meanings for:

- datatype, format, family, configuration, and candidate;
- weight, activation, output-storage, product, and accumulator domains;
- intrinsic/necessary scale versus optional external scale;
- Model A, Model B, Model C, exact product, conversion, rounding, and requantization;
- strict versus practical low precision;
- primitive, MAC/FPU, support path, generic PE/tile/system;
- raw arithmetic versus all-in hardware cost;
- run, artifact, result, package, and content hash.

Completion check: the same term is not defined differently in two contract files.

### Wave 2 — Draft the three contract tracks in parallel

#### Track N — Numerical contract

Responsible role: numerical-method owner

1. **N0.1 — Finalize `arithmetic.md`.**
   - Define the common `NumberFormat` operations.
   - Define Models A/B/C mathematically.
   - State Model C as Experiment A's canonical MAC if F1 is accepted.
   - Define product precision, conversion, rounding, overflow/underflow, and exceptional behavior.
2. **N0.2 — Finalize the accumulator interface.**
   - Make W, A, accumulator, and output formats independent.
   - Define accumulator initialization, update, bias addition, conversion, and output requantization.
   - Freeze sequential reduction as the main baseline and label tree reduction an ablation.
   - State what “sufficiently wide, family appropriate” means as an interface requirement without selecting the later winning accumulator.
3. **N0.3 — Finalize `experiment-a.md`.**
   - Freeze controlled static PTQ and prohibited optimizations.
   - Write the per-family intrinsic/necessary scaling table.
   - Move general/power-of-two external scaling and family-specific optimization to Experiment B.
4. **N0.4 — Freeze the deployment graph.**
   - Eval mode → BN fold → graph freeze → calibration → PTQ.
   - Define checkpoint, folded-graph, preprocessing, and subset identity requirements.
5. **N0.5 — Finalize `operator-semantics.md`.**
   - Cover Conv/1×1/DWConv/Linear, bias, residual add, elementwise multiply, ReLU/ReLU6, pooling, hard-swish, nonlinear LUTs, softmax, and detector NMS.
   - Freeze the strict baseline and define practical exceptions only as later ablations.

Track N deliverables:

- accepted versions of the three numerical contract files;
- at least one worked dot-product example showing every arithmetic domain/rounding point;
- at least one worked Conv+BN-fold+bias+activation+requantization example;
- a table showing Experiment A scaling behavior for INT/fixed, FP/minifloat, MX/BFP, Posit, log/power-of-two, and codebook families;
- an explicit list of numerical dimensions intentionally left for later gates.

#### Track I — Configuration, identity, and package schemas

Responsible role: software/reproducibility owner

1. **I0.2 — Define the datatype-manifest schema first.**
   - Required common fields: name, family, bits, signedness, encoding, rounding, overflow, and underflow.
   - Conditional fields: exponent/mantissa/bias/special values, integer/fraction, regime/es, log allocation, codebook, scale, block size/axis, and shared-scale format.
   - Invalid combinations and schema-version behavior must be explicit.
2. **I0.1 — Define the canonical experiment schema.**
   - Include model/checkpoint/graph, dataset manifests, W/A/accumulator/output manifests, PTQ, arithmetic, operator-policy, runtime-semantic version, and seeds.
   - Distinguish numerical-identity fields from performance-only metadata.
3. **I0.3 — Freeze content identity.**
   - Specify deterministic serialization, path normalization, hash algorithm, and schema-version participation.
   - Define identities for runs, manifests, datasets, graphs, artifacts, and hardware runs.
   - Define when an artifact may be reused and when a change invalidates it.
4. **I0.4 — Define the external candidate-package schema.**
   - Include graph, encoded weights, scales/metadata, datatype manifests, operator/arithmetic semantics, public quality/confidence, and generic hardware evidence.
   - Define content-addressed payload references and package versioning.
   - Exclude all private MANT fields.

Track I deliverables:

- machine-readable datatype-manifest schema in `public/formats/manifests/`;
- machine-readable experiment schema in `public/experiments/configs/`;
- deterministic canonicalization/hash specification and golden examples;
- machine-readable candidate-package schema in `public/package/schema/`;
- one valid and several invalid illustrative manifests/configs drawn only from the supplied plan's fields;
- a field-level dependency table stating which changes invalidate calibration, encoded weights, predictions, traces, or hardware results.

Track I does **not** implement the parser, scheduler, database, or hash engine; those are Phase 1 tasks I1.2, I1.4, and I1.5. Phase 0 defines their contracts and fixtures.

#### Track H — Generic hardware measurement contract

Responsible role: generic-hardware owner

1. **H0.1 — Finalize `hardware-metrics.md`.**
   - Define area, fmax, latency, II, throughput, leakage, dynamic/clock/total power, energy/op, and energy/inference.
   - Define the numerical tuple that must accompany every point.
   - Define vectorless versus trace-driven evidence labels.
2. **H0.2 — Freeze the public hierarchy.**
   - Primitive → complete MAC/FPU → scaling/conversion/memory → generic PE/tile/system.
   - Define raw versus all-in cost boundaries and exclude MANT-specific constants.
3. **H0.3 — Define the standard timing harness.**
   - Launch DFF → DUT → capture DFF.
   - Specify which wrapper and internal registers count toward reported area/power.
   - Specify latency/II treatment for combinational, pipelined, and iterative units.

Track H deliverables:

- accepted hardware-metrics contract;
- metric names, units, formulas, and required provenance fields;
- hierarchy/accounting diagram or table;
- timing-harness interface and accounting rules in `public/generic_rtl/harness/`;
- explicit list of pilot-dependent values left for D8–D10, including frequencies, memory evidence level, trace length, and convergence tolerance.

Track H does **not** choose RTL architectures, final clocks, memory macros, or power methodology details that require Phase 1/2/6 evidence.

### Wave 3 — Cross-contract consistency and boundary review

#### Build the traceability matrix

Responsible roles: all three track owners

For every semantic field, record:

```text
contract definition
    -> datatype/experiment schema field
    -> oracle/backend consumer
    -> generic RTL/hardware-result field
    -> candidate-package export field, if applicable
```

Required checks:

- W, A, accumulator, output, scale, MAC, and rounding fields exist once and agree everywhere.
- Operator behavior has enough metadata to reproduce a result.
- Every hardware point can identify the exact quality configuration.
- Schema defaults do not create implicit numerical behavior.
- No field relies on an unpublished MANT assumption.

#### Walk through one complete witness configuration

Use the plan's illustrative FP6 E3M2 manifest only as a schema/contract witness, not as D1 candidate acceptance. Walk it through:

1. checkpoint and BN-folded graph identity;
2. Experiment A calibration/scaling policy;
3. W/A/accumulator and Model C arithmetic;
4. Conv reduction, bias, activation, and requantization;
5. canonical config serialization/hash;
6. expected artifact invalidation/reuse;
7. generic hardware tuple and metrics;
8. public candidate-package representation.

The walkthrough passes only if no unstated default or ambiguous conversion/rounding point remains.

#### Audit the external boundary

Responsible roles: repository owner and release reviewer

- Confirm there is no MANT implementation directory or dependency.
- Confirm the public package is export-only.
- Confirm public schemas contain no ISA, tile, topology, mapping, compiler, simulator, or proprietary chip fields.
- Confirm roadmap Phases 8–10 remain documentation-only external context.

### Wave 4 — Review, freeze, and record completion

#### Contract reviews

Perform four sign-offs:

| Review | Must approve |
|---|---|
| Numerical | Arithmetic, accumulator, Experiment A, graph, operator semantics |
| Reproducibility | Manifest/config/package schemas, identity, invalidation rules |
| Generic hardware | Metrics, hierarchy, timing harness, evidence labels |
| Scope/release | Architecture independence and external export boundary |

Any requested semantic change returns the affected contracts to Wave 2 and repeats the Wave 3 checks.

#### Freeze artifacts

For each accepted contract/schema:

- assign semantic version and status `accepted`;
- record owner, reviewers, acceptance date, and decision link;
- compute and record content hash;
- update F1–F8 statuses and rationale;
- record all deliberately open questions under their later D gate;
- publish the Phase 0 completion record in this file.

## Phase 0 deliverable matrix

| Task | Required deliverable | Acceptance evidence |
|---|---|---|
| N0.1 | Model C and A/B/C arithmetic contract | Worked examples; all round points explicit |
| N0.2 | Independent accumulator interface | W/A/acc/output conversions and bias/requantization explicit |
| N0.3 | Experiment A per-family scale policy | No optional scale leaks into A; B dimensions listed |
| N0.4 | Frozen deployment-graph order | BN-fold equations and identity fields complete |
| N0.5 | Strict/practical operator contract | All plan-listed operators classified |
| I0.1 | Canonical experiment schema | Full witness config validates without hidden defaults |
| I0.2 | Datatype-manifest schema | Valid/invalid family examples cover conditional fields |
| I0.3 | Run/artifact identity specification | Golden serialization/hash and invalidation matrix |
| I0.4 | External package schema | Complete export witness; private-field audit passes |
| H0.1 | Hardware metric dictionary | Names, units, equations, provenance, evidence labels |
| H0.2 | Generic hierarchy/accounting boundary | Raw/all-in components assigned exactly once |
| H0.3 | Timing-harness contract | Register accounting and latency/II rules explicit |
| M0.1 | Private-scope exclusion | No MANT implementation in repository/public schema |
| M0.2 | Export/import boundary definition | Export contract accepted; redesign remains external |

## Decisions that must remain open after Phase 0

Completing Phase 0 must not prematurely decide:

- D1 candidate manifest set;
- D2 algorithmic/LUT/predecoded backend choice;
- D3 EfficientNet main-suite inclusion;
- D4 numerical promotion thresholds beyond the conservative rule;
- D5 expensive ablation set;
- D6 final Experiment B policy per family;
- D7 RTL finalist set;
- D8 representative RTL architectures;
- D9 physical memory evidence level;
- D10 final trace/power methodology;
- D11 exported Pareto/guard composition;
- D12–D14 external MANT decisions.

Accumulator winners, exact FP accumulator layouts, frequency points, area budgets, trace lengths, and quality-loss thresholds also remain evidence-gated.

## Phase 0 acceptance gate

Phase 0 is complete only when all answers below are **yes**:

- [x] F1–F8 have explicit accepted records.
- [x] Every Phase 0 contract is versioned, reviewed, and marked accepted.
- [x] Datatype, experiment, and candidate-package schemas are defined.
- [x] One witness configuration traverses all contracts without ambiguity.
- [x] Every numerical field has a single source of truth and named consumers.
- [x] All conversions, rounding, bias, overflow/underflow, scaling, and requantization points are explicit.
- [x] Experiment A and Experiment B are separable from configuration alone.
- [x] Metric names, units, hierarchy, and timing-harness accounting are explicit.
- [x] Canonicalization, hashing, artifact reuse, and invalidation rules are deterministic.
- [x] Every hardware point can resolve the exact measured-quality configuration.
- [x] The export package is versioned and contains no private MANT details.
- [x] All deliberately unresolved questions are assigned to D1–D14 rather than hidden as TODOs.
- [x] Numerical, reproducibility, hardware, and scope cross-reviews are recorded.

## Completion record

### Approval and review

- F1–F8 accepted unchanged by the project owner on 2026-09-04.
- Contracts/schemas prepared and cross-checked by Codex on 2026-09-04.
- Numerical review passed: Model C, resolved accumulator, scaling, graph, bias, operator, and requantization semantics agree.
- Reproducibility review passed: schemas, canonical identity, artifact lifecycle, and invalidation rules agree.
- Generic hardware review passed: metric units/formulas, hierarchy, support-cost accounting, and timing-harness rules agree.
- Scope review passed: no MANT implementation directory exists; export schema contains no private fields and rejects an illustrative private-field fixture.

### Machine-readable deliverables

| Deliverable | Path | SHA-256 |
|---|---|---|
| Arithmetic contract | `docs/contracts/arithmetic.md` | `5aa06e06e8539a12ff52c789114bc399307533bd1139c0e4a725e8ba36be18b2` |
| Experiment A contract | `docs/contracts/experiment-a.md` | `695e5796a43d09cfbfae478e2df899a092f69e7d0b8063b79826b78b9e51e907` |
| Operator contract | `docs/contracts/operator-semantics.md` | `a8279a2eee8e9ef7f079e4d71e9b7700cc2e3d2f7ee51b27acb31ebe3413610f` |
| Config/identity contract | `docs/contracts/config-and-identity.md` | `ecbff714d0547db4e5e07c65421bde6a4d9e6a0be22f12098b790bbc7864d294` |
| Hardware-metrics contract | `docs/contracts/hardware-metrics.md` | `2af56e89ee7b04dc97a877d2c21e8535bc60b856c3e226b98408ef11f3ca6341` |
| External package contract | `docs/contracts/public-private-interface.md` | `4c928f336d660f2e1ad96a1acb2ae00e435bff05e7d0f50c1793c835eae48c24` |
| Datatype schema | `public/formats/manifests/datatype-manifest.schema.json` | `c0da1ee75f9fae79fd81982e05d11a073ff097108be221a5a136ee6df23842a0` |
| Experiment schema | `public/experiments/configs/experiment.schema.json` | `497e37882bb10aa82903f011139720c7b7266bbab75ea20910fd1944f35018ab` |
| Candidate-package schema | `public/package/schema/quantized-model-package.schema.json` | `4bc772a3e2cfb9eb3765badcc20f462ad7353a1273e1bfa9435b08504d0ccbe5` |
| Timing-harness contract | `public/generic_rtl/harness/README.md` | `d07682dd0f1f5412056a6827ef15d9b882debc907d3760106f3f8aeeee4d018b` |

### Witness walkthrough

The illustrative FP6 E3M2 witness covers:

1. a complete versioned datatype manifest;
2. a valid Experiment A config with disjoint dataset declarations, independent W/A/accumulator/output references, intrinsic/no optional scaling, Model C, wide explicit accumulator reference, sequential reduction, accumulator-domain bias, strict operators, and output-store requantization;
3. deterministic compact key-sorted serialization;
4. canonical SHA-256 `d7b6f2a3f8c4edfaaba7a2375f677d00f837cddef6f8213dd7c81dac3c597d2a`;
5. a schema-valid public guard-package witness with quality and generic-hardware evidence fields.

The witness is illustrative only. It does not close D1 or D11 and contains no measured scientific result.

### Validation evidence

- All schema and fixture JSON parses successfully.
- All three schemas pass Draft 2020-12 meta-schema validation.
- Valid datatype, Experiment A, and package fixtures pass their schemas.
- Invalid fixtures for missing rounding, missing float semantics, zero width, optional Experiment A external scaling, Experiment A native MAC, and a private MANT package field are rejected.
- The canonical experiment file reproduces its recorded SHA-256.
- Cross-contract traceability and artifact invalidation/reuse matrices are recorded in `docs/contracts/README.md` and `docs/contracts/config-and-identity.md`.

## Exit criterion

- **Passed:** one canonical config uniquely determines the numerical result.
- **Passed:** all datatype/operator/accumulator/scale semantics are versioned and testable.
- **Passed:** generic hardware comparison metrics and harness are unambiguous.
- **Passed:** the candidate-package boundary contains no private MANT knowledge.

## Blocks

Phase 0 no longer blocks implementation. Phase 1 may begin with D1 candidate-manifest review, workload/checkpoint preparation, oracle/schema-validator infrastructure, and generic ICS55 flow bring-up in parallel.
