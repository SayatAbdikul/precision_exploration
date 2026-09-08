# Project tools

These are thin user-facing orchestration entry points. Reusable implementation belongs under `public/`.

## `setup/`

Plan-derived responsibilities:

- verify Python/C++/CUDA and GPU information;
- verify synthesis/P&R tools, ICS55 libraries/corners, and optional SKY130 flow;
- verify dataset roots without copying payloads;
- report checkpoint/dataset/tool hashes and versions;
- check SQLite and artifact/cache locations;
- surface unsupported memory, extraction, or delay-aware capabilities explicitly.

## `run/`

Stable commands should cover:

- validate contracts, manifests, configs, and subset separation;
- reproduce FP32/INT8/standard anchors;
- generate truth tables and run conformance;
- calibrate/fold/quantize from canonical configs;
- launch/resume/deduplicate CPU or single/multi-GPU experiment jobs;
- launch staged 1k, 5k/10k, and full validation;
- generate operand traces;
- run generic RTL synthesis/P&R/power sweeps;
- export a schema-validated Pareto-plus-guard package.

CPU scheduling may run independent configs concurrently subject to memory bandwidth. A weak single GPU normally runs one well-batched experiment. Multiple machines distribute complete configurations.

## `analysis/`

Stable report generation should cover:

- absolute quality and delta from FP32;
- paired bootstrap intervals and disagreement tables;
- layer/class/calibration robustness;
- promotion and pruning audit reports;
- synthesis/PPA/power/memory parsing;
- raw versus all-in scaling/memory breakdowns;
- per-workload and aggregate confidence-aware Pareto fronts;
- publication tables/figures and D11 candidate selection.

Tools never embed MANT-specific assumptions or invoke a local MANT implementation.

## Phase 1 executable checks (Python 3.10)

With the frozen checkpoints, dataset payloads, and pilot reports present:

```sh
python3.10 -m pytest -q
python3.10 -m tools.analysis.verify_phase1_evidence
python3.10 -m tools.analysis.verify_phase1_workloads
```

The evidence verifier reads the authoritative SQLite database without modifying it. The workload verifier reruns all three 1k classifier screens, checks eight real images per folded model, and reproduces graph hashes. `--resume` rechecks already generated classifier prediction hashes before reusing them after an interrupted verification.

After a deliberate semantic or input change, regenerate the affected artifacts:

```sh
python3.10 -m tools.setup.freeze_phase1_inputs
python3.10 -m tools.setup.generate_truth_tables
python3.10 -m tools.analysis.finalize_phase1_evidence
```

The finalizer rebuilds the curated Phase 1 evidence database in a temporary file and replaces the authoritative database only after successful ingestion. The pre-correction database is retained as `results/databases/phase1-before-corrections.sqlite`; its obsolete identities are historical, not reusable current evidence.

Production submission validates every supported configuration before queueing and again before execution. Phase 1 supports the pinned FP32 evidence schema and the oracle 1.1.0 reference witness. C++/CUDA execution remains Phase 2 work. New executable schemas require an explicit validator. The local scheduler renews worker leases every 30 seconds; use a stale timeout longer than two heartbeat intervals. Recovery retains the failed attempt and creates a new attempt and lease. Metrics and completion require the current lease token.
