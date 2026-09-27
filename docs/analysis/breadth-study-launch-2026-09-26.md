# Selected B extensions, matched E1, and exact E2

The frozen matrix is
[`comparison-matrix-v1.json`](../../public/experiments/configs/breadth-study/comparison-matrix-v1.json).
Its file SHA256 is `64e97fb3d673a6bb2d4444c334a09f96382ddc269149419d45437a33675bb091`.
The dated machine-readable snapshot is
[`breadth-study-launch-2026-09-26.json`](../../results/summaries/breadth-study-launch-2026-09-26.json).
The live queue state is in `artifacts/breadth_study/next_study_v1/status.json`.

## Completed evidence

* Selected B: **171/171 configurations at 1,000 images**, covering all 100
  datatype/model pairs and 71 additional recipe comparisons. Extending the
  retained 128-image panels required 149,112 new candidate-image evaluations.
  The selected run completed at 2026-09-26 11:18:08 UTC, about 97 minutes after
  launch. These are FP32 QDQ simulations under the declared B recipes; their
  speed is not native exact-A throughput.
* Initial E1: ResNet18 INT4 and MobileNetV2 INT4 completed their 32-image matched
  controls, including eight paired CPU/CUDA checks each. Both agreed with
  retained strict exact inference on all 32 final outputs and every layer.
  Their top-1 differences were **0 percentage points**. Absolute accuracy on
  this small development panel was 12.5% and 0%, respectively; arithmetic
  agreement does not mean acceptable task quality.
* E2 preparation: the original calibration observations and two 512-image
  calibration patch sets are saved. All four activation-scale refinement
  graphs passed the original integer arithmetic proof. The first real layer
  of each weight-rounding recipe completed its 1,000-step GPU fit; these
  optimizer checkpoints are reused by the full fitting stage.

## What E1 measures

The control uses the strict graph, stored weights, input codes, scales, bias
policy, reduction order, and output quantizers. It replaces reduction states
with sequential correctly rounded FP32 FMA/sums in the same operand domain,
then retains the original store interface. It is not an ordinary framework
convolution and is not the existing B simulation. Offline model loading and
FP32 diagnostic references are separate from candidate exact inference.

The initial integer graphs also have a conservative bound proving the integer
reduction prefixes fit exactly in FP32. Their zero observed arithmetic gap is
therefore expected. Changing only accumulation precision does not repair their
poor strict-recipe quality. That motivates the controlled E2 recipe ablations.

The expanded control is versioned separately in `matched_control_v2.py`. It
uses one FP32 rounding for a two-input residual sum, preserving cancellation
of rational operands. The initial completed control and its identities remain
unchanged.

## Running queue

**2026-09-27 scheduling update:** posit4 and Q1.6 have now completed all
32 matched images with complete layer/output agreement and zero top-1 difference.
Their absolute panel accuracies were 3.125% and 0%, respectively. FP6 completed
its native pilot and remains in its longer matched experiment. A checkpointed
scheduler restart reserves one worker for remaining E1 work and runs E2 fitting
alongside it; E2 exact inference then uses three workers while E1 is pending, or
four when no E1 work remains. This changes scheduling only. Saved image records,
numerical source identities, the comparison matrix and exact execution guard
are preserved. The original launch ordering below is superseded by this overlap.

1. E1 family expansion starts with ResNet18 posit4 and MobileNetV3-Large Q1.6,
   whose static graph proofs and saved eight-image native pilots reproduce.
   Each receives 32 exact and 32 matched-control CUDA images, with eight CPP
   checks for each mode. The first eight exact images must reproduce the
   retained native pilot layer/output hashes. Two workers use four threads each.
2. MobileNetV3-Large FP6 E3M2 has a static proof but still needs its paired
   native pilot. Its 32-image E1 study follows a successful pilot and admission.
3. E2 fits weight rounding for ResNet18 INT4 and MobileNetV2 INT4/5/6, then runs
   two new recipes per case on the same 128-image development population. Four
   workers with four threads each execute the eight new exact graphs. Every
   graph needs eight matching CPP/CUDA images; its first prepared CPP execution
   must also match the original uncompiled executor, including diagnostic events.
4. Compare each arm with the saved strict screen using paired bootstrap
   intervals (5,000 draws, seed 20260926). If either arm's interval contains
   +1 percentage point, extend both arms of that case to 256 images and reuse
   their first 128 checkpoints. These are exploratory pointwise intervals,
   not selection-adjusted confirmation claims.

Exact inference stays in `.venv` and uses the native C++/CUDA kernels. The
calibration-only optimizer runs in `.venv-b`, which has CUDA PyTorch. The parent
validates every folded weight tensor against the strict stored codes before
dispatch. It checks that returned weights contain only permitted floor/ceil
choices, retain original scales, and pass the integer graph proof before exact
evaluation. Fitting has separate source archives, runtime identities, Adam
state, RNG state, and sealed weight artifacts. No exact-runtime guard was relaxed
to accommodate the optimizer environment.

## E2 interventions and limitations

**Activation-scale refinement only:** strict A already used a 100-coarse /
50-fine MSE scale search. This arm adds a declared multi-start Lloyd-style MSE
refinement using the same retained calibration samples, with the original scale
included as a candidate. Weights are unchanged. This is a refinement of an
existing MSE recipe, not its first introduction.

**Adaptive weight rounding only:** an
[AdaRound-inspired](https://proceedings.mlr.press/v119/nagel20a.html) symmetric
linear-patch reconstruction adaptation. It uses 512 fixed training-calibration
images, two patches per convolution per image, 1,000 Adam steps, fixed scales,
and hard floor/ceil weight choices. It retains original layer codes if full
calibration-patch reconstruction error increases. It is not a reproduction of
the published method: the local objective omits the activation function, uses
symmetric inputs and sampled patches, and has a shorter optimization schedule.
Evaluation images and labels are not used to fit either intervention.

## Explicit unresolved cases

Seven selected E1 configurations still cannot enter native quality runs:

| Case | Remaining static proof gap |
| --- | --- |
| ResNet18 NF4 | Global average pool |
| MobileNetV2 log4 | Global average pool |
| MobileNetV3-Large binary / ternary | Global and squeeze/excitation average pools |
| MobileNetV2 MXFP4 E2M1 | Shared-scale MACs and average pool |
| YOLOv8n INT8 | DFL heads |
| YOLOv8n BFP6 | Shared-scale MACs, DFL and box decoding |

Their B results remain valid QDQ evidence. Neither B completion nor the new
study-scoped admissions change historical A acceptances or complete Phase 3/D4.
The seven cases remain visible in the queue inventory and final queue status.

**2026-09-27 proof update:** the new, separately archived
`development/acceptance_proofs/mapped_mean.py` bounds ordered FP64 summation and
division errors against the complete output-threshold lattice. It closes all
nine ternary pooling gates. Recomputing the mapped MAC and remaining non-MAC
proofs covers every operator in MobileNetV3-Large ternary. The separate queue
`artifacts/breadth_study/ternary_e1_v1/` waits for the main E1/E2 work, then runs
the eight-image paired native pilot and admits the 32-image E1 panel only if it
passes. Its process-local admission adapter has a distinct source identity;
the running inference implementation is unchanged.

For binary, five even-length pools have reproducible reference-operator
counterexamples: exact cancellation and sequential FP64 summation can select
different output codes. These cover unrestricted finite stored operands; they
are not claims that those particular tensors occur on evaluation images or are
reachable through preceding activations. Binary therefore still needs a valid
restricted-domain proof or a separately versioned arithmetic change. The
development evidence is saved under
`artifacts/phase3/proof-development/mapped-mean-v1/`. Six selected configurations
remain without a complete static proof; ternary now waits for native validation.

## Resume and inspect

From the project directory, restart the remaining queue with one command:

```bash
.venv/bin/python -m tools.run.study_next start
```

It starts/resumes the main queue and its gated ternary continuation, detaches,
keeps the main log in `artifacts/breadth_study/next_study_v1/run.log`, and
reuses validated image and optimizer checkpoints. A live controller lock rejects
duplicate launches. The queue requires the completed initial E1 status and
exclusive native resource leases; an invalid artifact or runtime change fails
closed. Earlier failed preflight attempts remain in the append-only log.

Inspect the current stage, saved image counts by mode/backend, completed panels,
and fitted layer counts with:

```bash
.venv/bin/python -m tools.run.study_next status
```

Source archives and content-addressed plans preserve the implementations used
for each job. Leave numerical source files unchanged during a run. The focused
test set passes 27 checks, including control arithmetic, the fitting bridge's
discrete-weight constraints, shape coverage, checkpoint resumption, and failure
reporting. Long-running family and E2 results must still pass their native gates;
the launch snapshot is not a completion certificate.
