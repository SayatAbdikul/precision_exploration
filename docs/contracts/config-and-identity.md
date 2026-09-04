# Configuration and identity contract

Version: **1.0.0**
Status: **accepted for Phase 0**
Accepted with the Phase 0 baseline: **2026-09-04**

## Reproducibility rule

The canonical machine-readable YAML/JSON configuration completely determines the numerical result. Serialize it deterministically and compute:

```text
experiment_id = SHA-256(canonical configuration)
```

Equivalent semantic configurations must canonicalize identically. Any semantically relevant change produces a new identity.

## Required configuration groups

```yaml
experiment_id: generated_from_config_hash

model:
  name: resnet18
  checkpoint_hash: "..."
  graph_version: bn_folded_v1
  graph_hash: "..."

dataset:
  calibration_list: imagenet_cal_2000_seed17.txt
  calibration_list_hash: "..."
  evaluation_list: imagenet_screen_1000_seed91.txt
  evaluation_list_hash: "..."
  preprocessing_version: "..."

formats:
  weight_manifest: fp6_e3m2_finite
  weight_manifest_hash: "..."
  activation_manifest: fp6_e3m2_finite
  activation_manifest_hash: "..."
  accumulator_manifest: fp12_e5m6
  accumulator_manifest_hash: "..."
  output_manifest: fp6_e3m2_finite

ptq:
  experiment: A
  method: mse
  weight_granularity: per_output_channel
  activation_granularity: per_tensor
  rounding: rne
  scaling_policy: intrinsic_or_required_only

arithmetic:
  mac_model: full_product
  product_precision: exact_to_accumulator
  reduction_order: sequential
  bias_domain: accumulator

runtime:
  backend: cuda
  kernel_version: "..."
  software_commit: "..."
  repository_state: clean

seeds:
  calibration: 17
  evaluation: 91
```

The actual schema may add fields but must not omit semantic identity.

## Identity inputs

- Model name, checkpoint hash, graph/folding version and graph hash.
- Dataset version, preprocessing, calibration/evaluation lists and hashes.
- All datatype manifests and hashes.
- PTQ method, search settings, granularity, clipping, scaling, block metadata, and rounding.
- MAC model, product precision, accumulator, conversion, reduction, bias, activation, and requantization semantics.
- Runtime backend and semantic kernel version.
- Seeds and any deterministic execution flags.

Host performance metadata such as GPU model may be stored without changing numerical identity unless it can change the result. Hardware runs receive their own identity including RTL, constraints, PDK/library/corner, tool versions, pipeline, and target clock.

## Canonical serialization

Version 1.0.0 uses the following exact rules:

1. Validate the object against its versioned schema before hashing.
2. Remove the derived identity field (`experiment_id` or `package_id`) from the hash preimage.
3. Represent the value as JSON encoded in UTF-8 without a byte-order mark.
4. Sort every object's keys lexicographically; preserve array order.
5. Use compact JSON with no insignificant whitespace.
6. Encode strings with JSON escaping, booleans as lowercase `true`/`false`, and null as `null`.
7. Permit only finite JSON numbers; configuration quantities whose exact textual identity matters use integers or strings with units rather than ambiguous host floats.
8. End the serialized file with exactly one LF byte.
9. Compute lowercase hexadecimal SHA-256 over those exact bytes.

Project paths in canonical configs use repository-relative POSIX form. Absolute paths, `.`/`..` segments, environment variables, and host-specific dataset roots do not enter canonical experiment identity. Dataset/checkpoint/artifact content hashes carry input identity.

The Phase 0 golden vector is:

```text
tests/conformance/fixtures/canonical/experiment_a_fp6_e3m2.canonical.json
SHA-256 = d7b6f2a3f8c4edfaaba7a2375f677d00f837cddef6f8213dd7c81dac3c597d2a
```

Phase 1's canonicalizer is accepted only if it reproduces the byte stream and hash.

## Identity types

| Identity | Preimage |
|---|---|
| Datatype manifest | Canonical validated manifest without any derived ID |
| Experiment | Canonical validated experiment config without `experiment_id` |
| Artifact | Raw artifact bytes; metadata separately records producer/dependencies |
| Dataset list | Exact normalized list bytes plus recorded dataset version |
| Folded graph | Serialized graph bytes plus graph-preparation semantic version |
| Hardware run | Numerical experiment ID, RTL hash, harness version, pipeline, constraints, PDK/library/corner/VT, tools, trace hash |
| Candidate package | Canonical package metadata without `package_id`; referenced payload hashes remain in the preimage |

## Invalidation and reuse matrix

| Changed input | Calibration stats | Quantized weights | Predictions/quality | Traces | Generic hardware result | Candidate package |
|---|---|---|---|---|---|---|
| Checkpoint or folded graph | Invalidate | Invalidate | Invalidate | Invalidate | Invalidate attached quality | Invalidate |
| Preprocessing or dataset version | Invalidate | Invalidate if calibration changes | Invalidate | Invalidate | Invalidate attached workload result | Invalidate |
| Calibration list/seed | Invalidate | Invalidate | Invalidate | Invalidate | Invalidate attached quality | Invalidate |
| Evaluation list/seed only | Reuse | Reuse | Invalidate | Invalidate | Invalidate attached quality | Invalidate |
| Weight format/scale/PTQ | Invalidate affected stats | Invalidate | Invalidate | Invalidate | Invalidate relevant RTL/system point | Invalidate |
| Activation format/scale/PTQ | Invalidate | Reuse weight artifact if independent | Invalidate | Invalidate | Invalidate relevant RTL/system point | Invalidate |
| Accumulator, MAC model, reduction order | Reuse | Reuse | Invalidate | Invalidate | Invalidate compute/system point | Invalidate |
| Operator semantics, bias, requantization | Invalidate affected stats | Invalidate if encoding changes | Invalidate | Invalidate | Invalidate support/system point | Invalidate |
| Runtime implementation only | Reuse | Reuse | Recompute execution artifact; expected numerical identity unchanged only after conformance | Recompute if encoding/order changes | Reuse only if trace/semantics unchanged | Rebuild evidence references if artifact hashes change |
| RTL architecture/pipeline/clock | Reuse | Reuse | Reuse | Reuse logical trace | Invalidate | Invalidate hardware evidence |
| PDK/library/corner/VT/tool/constraints | Reuse | Reuse | Reuse | Reuse | Invalidate | Invalidate hardware evidence |
| Statistical method/bootstrap seed | Reuse | Reuse | Recompute statistical summary only | Reuse | Reevaluate confidence-aware classification | Invalidate affected summary |

Reuse is allowed only when every dependency required by the producing artifact is identical by content/version.

## Results database

SQLite is the primary local store. It records:

| Group | Required metadata |
|---|---|
| Model | name, checkpoint hash, folded graph hash |
| Dataset | calibration/evaluation list names and hashes |
| Datatype | W/A/accumulator/output manifest hashes |
| PTQ | method, granularity, clipping/scale search, block configuration |
| Arithmetic | MAC model, product precision, reduction, rounding |
| Runtime | backend, kernel version, CPU/GPU, CUDA/compiler versions |
| Software | Git commit and clean/dirty state |
| Output | Top-1, Top-5, mAP, runtime, status, error log, artifact references |

Analysis-ready tables may be exported to CSV/Parquet, but the database remains the authoritative run registry.

## Artifact separation

Large files live in content-addressed artifact directories. Database records store path, hash, size, producer ID, and semantic type. Reusable categories are checkpoint, folded graph, calibration statistics, quantized weights, encoded tensors, truth tables, per-image predictions, and traces.

Do not repeat calibration when only accumulator/runtime semantics change and the cached calibration artifact remains valid under the accepted dependency rules.

## Job lifecycle

```text
PENDING -> RUNNING -> COMPLETED
                 \-> FAILED
                 \-> INVALID
```

- Writes are atomic.
- Completed jobs are deduplicated by experiment ID.
- Interrupted unfinished jobs may be rescheduled.
- Failures retain exceptions/logs for diagnosis.
- `INVALID` denotes a configuration rejected by schema/semantic checks, not a low-quality result.

## Per-image output

Store image ID, ground truth, FP32 prediction, quantized prediction, FP32 correctness, quantized correctness, and optional logits/confidence summaries so paired statistics do not require another inference pass.

## Phase 0 schema artifacts

- `public/formats/manifests/datatype-manifest.schema.json`
- `public/experiments/configs/experiment.schema.json`
- `public/package/schema/quantized-model-package.schema.json`
- `tests/conformance/fixtures/` valid, invalid, and canonical identity witnesses

## Phase 1 cross-reference checks required by this contract

The parser/validator must supplement JSON Schema with content-aware checks:

- Resolve every manifest name/hash pair and verify the bytes match the declared hash.
- Enforce ≤8-bit width for weight, activation, and output roles while allowing declared wider accumulators.
- Resolve `family_appropriate_wide` to the concrete accumulator manifest before generating the executable experiment ID.
- Verify calibration and evaluation lists are actually disjoint, not merely labeled disjoint.
- Verify Experiment A's manifest scale modes are only `none`, `required_mapping`, or `intrinsic_shared` as appropriate to the family.
- Verify Model C uses exact product to accumulator and that operator/bias/requantization versions match the contracts.
- Verify referenced checkpoint, graph, dataset lists, artifacts, and package payloads exist and match their hashes.
- Reject absolute/host-specific paths in canonical tracked configuration.
