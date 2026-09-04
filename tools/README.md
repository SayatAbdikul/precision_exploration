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
