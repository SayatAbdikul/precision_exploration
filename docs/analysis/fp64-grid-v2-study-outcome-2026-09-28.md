# Certified FP64 grid acceleration and useful-quality E1 outcome

The versioned FP64 grid runs the selected ResNet18 FP6/FP7 MACs as exact
integer sums, reconstructed at the declared FP64 scale. It admits only the
frozen graph, weight, encoding and geometry identities covered by its
per-channel prefix certificate; unsupported inputs fall back to the original
rational implementation. The old v1 executor, predictions and native library
remain archived under their original identities. The optimized FP64 store,
residual operations and matched-control path preserve the layer codes, output,
top-five list and quantizer diagnostic signature in retained full-image
comparisons.

| First image, frozen ResNet18 graph | Old execution | Optimized execution | Observed ratio |
| --- | ---: | ---: | ---: |
| FP6 E2M3 exact CPU | 3,245.49 s | 5.71 s | 568× |
| FP6 E2M3 exact CUDA | 724.66 s | 4.63 s | 157× |
| FP6 E3M2 exact CPU | 3,989.07 s | 5.96 s | 670× |
| FP6 E2M3 matched-control CUDA | 103.03 s | 5.55 s | 19× |

These are observed first-image execution ratios under contemporaneous
workstation load, excluding optimized engine setup (about 13–27 s). They are
not isolated hardware throughput or a confidence interval. All four listed
optimized images match their retained v1 oracle on every layer, output,
top-five prediction and diagnostic signature. The optimized E2M3 control CPU
image also matches v1, but its 94.19 s execution remains dominated by its
matched FP32 CPU MAC (89.71 s). The E2M3 exact CUDA full-image MAC phase is
0.47 s of 4.63 s; even zero-cost MACs would improve the whole image by at most
1.11×. The bounded Nsight sample found no kernel spills. Further native MAC
tiling or weight residency is therefore not the next material optimization.

The native/post-operation conformance suite passed 52 tests. The certificate
binds all three graph proofs, source and binary hashes, full-image retained
oracle evidence where available, CPU/CUDA FP7 first-image parity, and the
pre-enrollment benchmark charge. FP7's optimized first image took 5.34 s on
CPU and 4.71 s on CUDA and agreed on all recorded numerical fields. **FP7
has no retained whole-graph rational oracle**, so this is component and
cross-backend evidence only; it is not a native acceptance or quality panel.

The budgeted v2 study completed the eight-image CPU/CUDA native gates and
paired 32-image panels for ResNet18 INT8, MobileNetV2 INT8, ResNet18 FP6 E2M3,
and ResNet18 FP6 E3M2. The frozen promotion rule selected ResNet18 INT8 for a
paired 128-image extension. The independently audited results are:

| Canonical A case | Paired images | Exact/control top-1 | FP32 top-1 | Exact/control layer agreement |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 INT8 | 128 | 69.53% / 69.53% | 71.88% | 128/128 |
| MobileNetV2 INT8 | 32 | 71.88% / 71.88% | 71.88% | 32/32 |
| ResNet18 FP6 E2M3 | 32 | 0% / 0% | 68.75% | 32/32 |
| ResNet18 FP6 E3M2 | 32 | 0% / 0% | 68.75% | 32/32 |

All four panels also have complete output agreement between exact and matched
control. For ResNet18 INT8, zero output discordance across 128 images gives a
one-sided 95% upper bound of 2.31% for the output-discordance rate under this
sample model. The 32-image panels have a corresponding 8.94% upper bound; zero
observed difference does not establish universal equivalence. The two FP6
arms both collapse under the *frozen unscaled canonical A recipe*. This panel
does not isolate a general FP6 datatype effect: the B comparisons use different
calibration and graph recipes and must be reported separately.

The 24 aggregate worker-hour allowance includes 20.67 h of old v1 saved or
finished compute, 1.44 h of interrupted v1 work, 0.12 h of pre-enrollment
benchmark work and 1.50 h of v2 invocations. It has
0.27 h (975 s) remaining, with no unreconciled reservations. The mandatory
FP7 first-image rational comparison needs a 6,443 s reservation under the
frozen safety rule; its native gate and accuracy panel are therefore
budget-limited. MobileNetV2 INT8 qualified for extension by quality but its
measured reservation exceeded the remaining lifetime allowance. Neither
case should be labeled a negative result from a missing panel.

The one-line safe resume command is:

```bash
.venv/bin/python -m tools.run.useful_quality_v2 run
```

It rechecks sealed sources, checkpoints and resource leases, then schedules
only work permitted by the lifetime budget. Rebuild the audited result table
with `.venv/bin/python -m tools.analysis.useful_quality_e1_v2`. The detailed
machine-readable ledger is in `results/summaries/useful-quality-e1-v2/`, the
runtime certificate in
`artifacts/breadth_study/useful_quality_e1_v2/optimized-runtime-certificate.json`,
and the numerical checkpoints under its `jobs/` directory. These are
development panels and are not independent benchmark confirmation.
