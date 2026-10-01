# Useful-quality E1 v2 controller

The v2 controller reads the five sealed v1 plans and the original 128-image
population. It creates separate v2 plans only after receiving a reference to a
sealed optimized-runtime certificate. It never edits v1 checkpoints or assigns
them new execution times. The scientific promotion rule remains the v1 rule.

The worker handles one arithmetic arm and one backend per invocation. Each case
starts with a one-image timing and compatibility probe. A remaining seven-image
CPU/CUDA gate is costed from the measured probe before dispatch. A 32-image
CUDA panel can start as soon as its own eight-image gate passes; it need not
wait for another candidate's slow CPU pilot. The scheduler permits one CUDA
worker and a memory-dependent number of CPU workers while holding the original
controller and native resource leases. Extensions to 128 are decided in frozen
candidate order after the 32-image results and costed again before each arm.

The budget ledger charges v1 persisted inference, its first-image legacy
compatibility and completed invocation setup, all pre-enrollment full-image
benchmarks (including superseded probes), plus v2 completed invocations and
unreconciled reservations. The sealed benchmark artifacts provide a lower
bound on the certificate's pre-enrollment charge. Reservations are written
before dispatch. A crash
leaves a reservation charged until an explicit reconciliation; it cannot be
silently reused. The lifetime allowance is 24 aggregate worker-hours, with
at most 12 hours for extensions. If the original v1 work consumes the
allowance, v2 reports `budget_limited` and does not launch further images.
Each task also has a process timeout at its reserved duration. A timed-out
image is left unsealed and the controller stops; completed image checkpoints
remain valid.
If a v1 worker was interrupted between checkpoints, `audit-interruption`
acquires the native leases after quiescence and charges the full elapsed time
since its last saved image. Until that charge is sealed, remaining budget is
reported as zero.

INT8 strict records can come from the independently verified retained exact
screen. INT8 matched-control records may be imported from v1 only with their original
source, graph, sample, layer, output, prediction and diagnostic identities
checked. Migrated records retain their old seconds and carry a v1 reference;
their `new_compute_seconds` is zero. Floating strict and control records are newly
executed under optimized source identities and compared with their saved v1
arm oracles where present. Each floating arm must pass eight images on both CPU
and CUDA. The v2 gate compares layers, outputs, predictions and diagnostic
signatures; imported historical INT8 exact records keep the original
CPU/CUDA diagnostic-signature exception.

Before enrollment, v2 independently replays the frozen INT8 acceptance bounds
and all floating MAC/store and non-MAC proofs against the reconstructed legacy
source identity. It seals each replay once and checks that immutable result at
every worker stage. Floating controls additionally run the frozen FP32 native
conformance and the optimized store/residual conformance before admission.
When a retained v1 control image exists, the fresh optimized control is
compared to it. The uncompiled whole-graph first-image comparison applies to
new floating exact arms without a retained oracle, as specified by the frozen
protocol; unmatched controls use their exhaustive component conformance and
eight paired CPP/CUDA images.
At resume, the controller checks that every checkpoint is contiguous and
matches its plan, sample and source provenance, rechecks existing native
admissions, and recomputes saved 32/128-image summaries before scheduling.

The v2 legacy guard verifies every old guard field and recomputes the frozen
old source digest after excluding only the three additive native source files
`native_fp64_grid_v2.py`, `fp64_grid_v2.cpp`, and `fp64_grid_v2.cu`. The v2
source archive separately binds those files, the optimized implementation and
its new binary. This makes the old accepted library and checkpoints verifiable
after the additive code appears in the source tree.

After the optimized adapter and certificate are integrated, the one-line
resume command is `.venv/bin/python -m tools.run.useful_quality_v2 run`.
Initial enrollment requires the preceding preparation command:

```bash
.venv/bin/python -m tools.run.useful_quality_v2 audit-interruption
.venv/bin/python -m tools.run.useful_quality_v2 prepare --certificate artifacts/breadth_study/useful_quality_e1_v2/optimized-runtime-certificate.json
.venv/bin/python -m tools.run.useful_quality_v2 run
.venv/bin/python -m tools.run.useful_quality_v2 status
```

`run` resumes from sealed v2 records and allocations. No optimized panels have
been launched by this controller yet. Its frozen admission and budget logic
have focused unit tests; full runtime integration depends on the separately
validated optimized engine and its certificate.
