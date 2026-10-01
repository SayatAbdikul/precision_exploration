# Useful-quality E1 optimized study: development evidence

Controller: **budget_limited**. Enrollment: **sealed**.

| Canonical A case | Status | Completed paired images | Strict top-1 | Matched FP32 top-1 | FP32 top-1 | B top-1, different configuration |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| resnet18/int8 | completed_128 | 128 | 69.53% | 69.53% | 71.88% | 69.53% |
| mobilenet_v2/int8 | completed_32_extension_budget_limited | 32 | 71.88% | 71.88% | 71.88% | 68.75% |
| resnet18/fp6_e2m3 | completed_32 | 32 | 0.00% | 0.00% | 68.75% | 71.88% |
| resnet18/fp6_e3m2 | completed_32 | 32 | 0.00% | 0.00% | 68.75% | 59.38% |
| resnet18/fp7_e3m3 | native_gate_budget_limited | 0 | — | — | — | — |

Lifetime compute budget: 24.00 worker-hours; v1 saved/finished 20.67, v1 interrupted 1.44, v2 pre-enrollment 0.12, v2 completed 1.50, unreconciled reservations 0.00, remaining 0.27.

- resnet18/int8 (128 images): matched minus strict +0.00 percentage points, paired 95% interval [+0.00, +0.00]; full layer agreement 128/128, output agreement 128/128. Provenance: {'migrated_v1_original_provenance': 8, 'retained_verified_exact': 120} exact, {'migrated_v1_original_provenance': 8, 'new_verified_execution': 120} control.
  No output discordance observed; one-sided 95% upper bound 2.31% on this panel.
- mobilenet_v2/int8 (32 images): matched minus strict +0.00 percentage points, paired 95% interval [+0.00, +0.00]; full layer agreement 32/32, output agreement 32/32. Provenance: {'migrated_v1_original_provenance': 8, 'retained_verified_exact': 24} exact, {'migrated_v1_original_provenance': 8, 'new_verified_execution': 24} control.
  No output discordance observed; one-sided 95% upper bound 8.94% on this panel.
- resnet18/fp6_e2m3 (32 images): matched minus strict +0.00 percentage points, paired 95% interval [+0.00, +0.00]; full layer agreement 32/32, output agreement 32/32. Provenance: {'new_verified_execution': 32} exact, {'new_verified_execution': 32} control.
  No output discordance observed; one-sided 95% upper bound 8.94% on this panel.
- resnet18/fp6_e3m2 (32 images): matched minus strict +0.00 percentage points, paired 95% interval [+0.00, +0.00]; full layer agreement 32/32, output agreement 32/32. Provenance: {'new_verified_execution': 32} exact, {'new_verified_execution': 32} control.
  No output discordance observed; one-sided 95% upper bound 8.94% on this panel.
- resnet18/fp7_e3m3: native_gate_budget_limited; checkpoints: exact/cpp 0, exact/cuda 0, control/cpp 1, control/cuda 1.

The JSON ledger binds each result to its sealed v2 plan, optimized runtime certificate, native admission, source archive, and parent v1 plan/archive. Imported v1 predictions retain original execution provenance; their old compute cost is charged in the lifetime budget.

Resume experiments: `.venv/bin/python -m tools.run.useful_quality_v2 run`
Rebuild this report: `.venv/bin/python -m tools.analysis.useful_quality_e1_v2`

Development panels and promotions are exploratory; they are not final benchmark confirmation.
B maxabs is a different configuration and calibration; its gap to canonical A is not a pure arithmetic effect.
Migrated v1 and retained integer exact records keep their original source and timing provenance; zero new compute is not a throughput measurement.
A one- or eight-image native gate is not an accuracy panel. Incomplete or budget-limited cases have no panel quality claim.
Arithmetic effects are paired on the frozen samples; interval estimates describe sampling uncertainty, not implementation equivalence.
