# Configuration and identity contract

Status in supplied roadmap: **required by I0.1–I0.3; project review not yet recorded**

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
