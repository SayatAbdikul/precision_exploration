# Phase 3 completion and resource-use plan

Prepared 2026-09-22 for `/home/maveric/precision_exploration` at `f91c1f3`.
Implementation update 2026-09-23: the resumable controller, exact integer store
extension, execution guard, resource benchmark and explicit D4 review path are
implemented. See [the operating guide](../../architecture/phase3-controller.md)
for the current command and guarantees. Capacity benchmarks were run; production
screens were not started by this implementation work. The frozen numerical
engine, campaign and historical image records remain unchanged. Acceptance proofs
for 92 configurations and the subsequent scientific work remain open.

Development update 2026-09-23: separate, hash-checked local proof records now
exist for all 92 blocked configurations. They bound 4,729 MAC nodes, including
1,515 ordinary FP64 nodes with exact finite MAC prefixes. The records do not
satisfy graph acceptance: non-MAC/output-code/nonfinite gates, native pilots,
detector calibration and D4 remain. Run
`.venv/bin/python -m tools.analysis.phase3_development_preflight --summary --test`
to verify the handoff before continuing the one-command controller. The
[operating guide](../../architecture/phase3-controller.md) gives the isolated
proof test command and acceptance boundary.

Further 2026-09-23 acceptance development: 35 graphs have conditional local
MAC output-code proofs, and 21 classifier graphs have their MAC/non-MAC
arithmetic recomputed end to end. None has eight-image paired native acceptance.
The dry-run pilot batch can prepare all 92 blocked graphs, but it cannot start
while the active controller owns the native locks. FP8 E4M3FN and posit bias
boundary checks retain concrete output-code discrepancies; those need a
versioned arithmetic decision or a stronger reachability argument before
acceptance. The full campaign and D4 remain open.

Final resource calibration selected six concurrent CUDA workers: 240.3 verified
image executions/hour on a two-image-per-worker ResNet18 INT4 diagnostic run,
including startup/verification. Four workers repeated at 162.0/hour (previously
164.0/hour). This is a measured workload-specific result, not an all-family ETA.

## Starting point and completion criteria

- 25 formats × four models: 100 configurations are calibrated and encoded.
- Five configurations have full graph acceptance: ResNet18 INT8/6/5/4 and
  MobileNetV2 INT8. ResNet18 INT8 has a completed 1,000-image screen.
  Direct inspection finds 851 saved MobileNetV2 INT8 screen images, leaving 149;
  the tracked progress summary's 487-image count is stale. Discover work from
  validated checkpoints and the registry, not this summary's counters.
- Remaining work includes 95 graph acceptances, 99 full screens or individually
  justified diagnosed dispositions, sensitivity/diagnosis work, and D4 review.
- Existing local bounds cover 4,221/5,025 MAC nodes and 5,871/6,600 non-MAC nodes.
  A local bound is not whole-graph acceptance or output-code equality.
- The current acceptance implementation calls `prove_integer_graph()` for every
  candidate. General floating, fixed/posit, logarithmic, codebook and shared
  format acceptance requires additional implementation and validated evidence.
- `tools/analysis/phase3_d4.py` now supports a verified explicit review path;
  missing screens and scientific reviews still prevent Phase 3 from closing.

Completion means every matrix entry has verified fixed-1k paired results or an
explicit, evidence-backed disposition permitted by the study; catastrophic
results have diagnoses; required sensitivity and preservation reviews are
recorded; and D4 has a versioned decision and promotion list. Missing evidence,
slow execution, and poor eight-image pilot quality cannot stand in for a result.
The 5k/10k promoted experiments are downstream of this Phase 3/D4 boundary.

## Hardware and measured bottleneck

| Resource | Observed on this device |
| --- | --- |
| CPU | Intel i9-12900K, 16 physical cores, 24 logical CPUs; hybrid core topology |
| RAM | Approximately 32 GB installed; approximately 23 GiB available at inspection |
| GPU | NVIDIA RTX 4070 Ti, approximately 12 GiB VRAM |
| Disk | Approximately 82 GiB free on the project filesystem |
| CUDA | 13.1 under `/usr/local/cuda/bin`; `g++-14` available |
| Python | Existing `.venv`; CPU PyTorch is intentional for this native CUDA engine |

The local ResNet18 INT4 benchmark used two images and four CPU threads. CUDA
averaged 104.16 s/image versus 197.89 s/image on C++; CUDA output storage/conversion
averaged 51.61 s and native dispatch 1.08 s. Total process CPU time was about
104.31 s/image, suggesting much of the remaining work is effectively serial CPU
work. Dispatch timing includes preparation and is not a GPU-kernel measurement.
The older MobileNetV2 benchmark describes another machine and another graph.

At this small-sample rate, one serial ResNet18 INT4 1k screen would take roughly
29 hours, excluding setup and analysis. This is a planning illustration, not a
campaign ETA; format costs differ greatly. Measure each family before forecasting.
Optimize verified images/hour and time to D4, rather than utilization percentages.

## Execution order

| Stage | Work | Exit condition / useful overlap |
| --- | --- | --- |
| 1. Preserve and validate | Retain current source/library/config hashes and SQLite-consistent state; validate saved images and the frozen continuation plan; perform fresh native migration checks for queued graphs | Resume identities match; accepted outputs reproduce on this host |
| 2. Resume useful existing work | Finish MobileNetV2 INT8, then ResNet18 INT6/5/4 using the existing queue; completed pilot evidence is reused | Five accepted configurations have complete screens; analysis and diagnostics verified |
| 3. Build resource-aware execution | Durable campaign controller, verified task reuse, bounded process concurrency, resource telemetry and graceful interruption | Restart/duplicate-writer/failure tests pass; validated throughput improves |
| 4. Remove dominant overhead | Exact native output conversion, bias/store/requantization, repeated encoding/preparation and shared-scale search costs | Bit-exact differential checks and fresh representative pilots pass |
| 5. Expand graph acceptance | Reuse CPU pilots, add missing CUDA evidence and family-specific accumulator/non-MAC checks; version any required numerical revisions | Each eligible configuration has explicit graph acceptance |
| 6. Screen and analyze continuously | Run accepted fixed-1k screens; overlap CPU statistics and diagnosis with GPU workers | All valid candidates screened; failures individually diagnosed |
| 7. Close D4 | Verify completeness, paired intervals, sensitivity, family/hardware preservation and individual dispositions | Versioned D4 decision and promotion list with no unexplained gaps |

Stages 3–5 can proceed while Stage 2 runs only in a separate source checkout and
separate artifact namespace. Do not edit hashed files or production libraries
under a running worker. Snapshot/copy SQLite consistently; do not independently
write two registries for the same campaign and attempt to merge them later.

## Scheduler design for this machine

1. Keep the numerical job identity separate from execution placement. Record
   host, library, worker/thread settings and implementation identity alongside
   each attempt. Preserve the frozen sequential MAC reduction order.
2. Initially parallelize different configurations. Keep one active owner per
   numerical job, a fenced registry lease, and an atomic lock per job. Replace
   the single global GPU lock with explicit capacity-limited device admission;
   do not simply remove it or launch duplicate `phase3_screen` processes.
3. Start the new scheduler's benchmark at two CUDA worker processes and one CPU
   proof/analysis worker. Trial one, two, four and, only if memory permits, six
   CUDA workers. Try small CPU thread budgets; reserve CPU capacity for GPU
   workers' Python/output stages. CPU-only native pilots run only if they improve
   combined throughput and unlock acceptance.
4. Use processes for independent Python-heavy work. Existing runtime calls
   `torch.set_num_threads(4)`, so changing `OMP_NUM_THREADS` alone is insufficient.
   Add an explicit execution-resource setting, record it, and validate identical
   output codes and complete layer traces. Map P/E cores from observed topology
   before testing affinity; do not assume CPU-number ranges identify core types.
5. Benchmark aggregate valid images/hour on the same fixed diagnostic images,
   with cold/warm runs recorded. Use at least several images and repeat the best
   two settings; tiny differences from a two-image run are not a reliable winner.
   Benchmark integer, floating/rational, shared-format and detector jobs separately.
6. Measure peak process RSS and CUDA allocation under load before admitting more
   workers. Proposed admission reserves are 6 GiB available system RAM, 2 GiB
   free VRAM, two physical cores of headroom and 20 GiB disk free. These are
   starting limits, not measured optima. Avoid swap; reduce admission on memory
   pressure or when additional workers reduce aggregate throughput.
7. Run statistics/diagnostics as a separate bounded CPU queue as screens finish.
   Cache invariant verified COCO/bootstrap inputs and avoid repeated identical
   analysis. Do not duplicate complete C++ screening when C++/CUDA acceptance
   already justifies a single chosen production backend.
8. Delay image-level sharding until configuration-level concurrency is measured.
   Sharding needs exclusive per-image ownership, deterministic final aggregation,
   crash tests and explicit migration of checkpoint schemas/identities.

The implemented controller now owns the legacy locks and delegates kernel-backed
leases to a bounded worker pool, retaining an exclusive lock for each job.
Standalone legacy runners retain their single-worker behavior.

## Performance and correctness work before broad expensive screens

- Profile output conversion and Python bookkeeping on this device. Move the
  repeated exact decode, bias rounding/addition, activation and output encoding
  loops to native code where worthwhile. Preserve ties, saturation, signed zero,
  special values and per-step Model C rounding; do not substitute approximate
  GEMM or reassociate reductions to obtain speed.
- Cache immutable encoded weights, decoded tables, native operand packing and
  per-graph metadata by full identity. Reuse allocated buffers when measurement
  demonstrates a benefit. Consider GPU residency only after the larger CPU
  costs are reduced; large changes need separate validation.
- Integrate shared-scale pruning/caching only after proving the same selected
  exponent, tie policy, output codes and diagnostics. Existing native BFP pilots
  are diagnostic evidence, not automatic authorization for canonical screens.
- Correct the omitted fixed-point intermediate shift in native rational-grid
  sizing, add boundary/saturation regression cases, and document the oracle's
  finite precision. Establish the relevant domains before using these paths.
- Include preprocessing dependencies in run identity. Verify completed output
  artifacts before treating a registry `COMPLETED` row as reusable. Neither
  issue requires discarding verified historical results, but both require
  explicit compatibility/version handling when the execution protocol changes.

Keep the current engine available for its existing queue. Source or pipeline
changes generate new identities and must never silently reuse old images.
An optimized successor requires primitive/reference/C++/CUDA checks, representative
full-layer comparisons, fresh eight-image pilots and the relevant graph proof.
Reuse across versions requires a recorded, reviewed compatibility/import
mechanism; if equivalence is not established, rerun the affected job. Retain
original images and provenance unchanged.

## Acceptance and diagnosis priority

1. Finish the already accepted integer screens and their paired analyses.
2. Reuse the existing eight-image MobileNetV2 INT6/5/4 CPU pilots, add missing
   CUDA records through the supported pilot-extension path, and attempt integer
   acceptance. Do not redo validated CPU images merely to change backend order.
3. Develop non-integer acceptance validators using the existing bound inventories.
   Prioritize ordinary classifier floating/fixed formats and configurations with
   substantial existing proof coverage. Resolve loose bounds and reachable
   nonfinite behavior explicitly. Native equality alone is insufficient.
4. Complete MobileNetV3 nonlinear/residual/bias and posit/quire validation using
   its existing pilots and sensitivity evidence. Maintain separately versioned
   accumulator revisions where prescribed arithmetic changes.
5. Complete shared MAC/non-MAC bounds and canonical native acceptance for BFP/MX.
   Close logarithmic/codebook boundary cases using precise reference checks.
6. Resolve the detector calibration-coverage confound before committing to large
   detector runs. Preserve the original calibration and failed evidence; evaluate
   representative coordinate/score sampling in an explicitly versioned revision.
   Any changed sampling/domain policy needs a documented Experiment A decision;
   do not mix revised and original results or infer that INT8 itself is broken.
7. Maintain family coverage and workload specialists in the ready queue. Schedule
   short jobs that unlock several configurations early, while giving long jobs
   fair progress. Do not prune families because runtime is high or a pilot is weak.

## Durable one-command controller requirements

Provide a single `tools/run/phase3_resume.py` entry point with automatic inventory
and measured resource settings. It should build a dependency graph over prepare,
proof, pilot, accept, screen, analysis, diagnosis and D4 tasks. A restart must:

- discover saved work and verify its identity/content before skipping it;
- submit only ready tasks and explain blocked dependencies;
- recover a stale lease only after 300 seconds and after excluding a live owner;
- checkpoint after each image/backend and retain useful partial work;
- propagate Ctrl+C/SIGTERM to child process groups, stop admission and retain
  completed records; retry unfinished work without duplicate writers;
- cap transient retries and retain deterministic failures without retry loops;
- continue independent tasks after a failure and return nonzero if work failed
  or remains blocked; zero must mean the requested scope is verified complete;
- write a status file/log with progress, per-stage counts, resources, blocked
  reasons and an ETA derived from measured remaining work;
- validate D4 evidence before declaring Phase 3 complete. Scientific revisions
  and recorded D4 review cannot be replaced with an automatic success flag.

The implemented interface is `.venv/bin/python -m tools.run.phase3_resume`.
It currently discovers ready pilots, acceptance, store validation, screens and
analysis; new family proof development, diagnosis and D4 review remain explicit
scientific work. It reports blocked configurations and does not invent an ETA
from timings measured on different machines.
Before release, test duplicate starts, interruption during an image, stale/live
leases, corrupted or missing completed outputs, changed preprocessing, memory
admission, worker failure, deterministic aggregation and restart with no lost
or duplicated completed image work. Then verify representative GPU execution.

## Current command

```bash
cd /home/maveric/precision_exploration && .venv/bin/python -m tools.run.phase3_resume
```

This uses the measured worker profile and resumes all currently provable integer
work. See the operating guide for status, calibration, exit codes and validation.

## Legacy serial command: frozen integer queue

```bash
cd /home/maveric/precision_exploration && env PATH="/usr/local/cuda/bin:$PATH" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_DYNAMIC=FALSE PYTHONUNBUFFERED=1 .venv/bin/python -m tools.run.phase3_continue_integer --plan artifacts/phase3/integer-continuation-plan.json
```

This is one line and uses existing code. It rechecks the saved ResNet18 INT8
analysis and ResNet18 INT6/5/4 pilots, resumes MobileNetV2 INT8 from its saved
images (currently 851, next image 852), then runs ResNet18 INT6/5/4 screens with
analysis and diagnostics. It
passes the 300-second stale-lease recovery threshold internally. An interrupted
image can be repeated; complete image/backend checkpoints are reused.

Run in a terminal that remains open, or an existing persistent terminal session.
The command neither installs software nor launches an unimplemented parallel
scheduler. Its scope is the frozen integer queue, not all 100 configurations.
Inspect per-task events/logs under `artifacts/phase3/integer-continuation/`:
the existing controller records failures and continues independent tasks, but
does not reliably communicate every task failure through its overall exit code.
Rerunning resumes/revalidates work; it does not imply D4 is complete.

## Evidence used

- `results/summaries/phase3-progress.json`
- `results/summaries/phase3-gate-inventory.json`
- `results/summaries/phase3-d4-readiness.json`
- `artifacts/phase3/integer-continuation-plan.json`
- `artifacts/phase3/thread-benchmark/6e2f898536a799536542be1909ce1bf763fb5cba3c01b85c9e8e3b9489b22118/summary.json`
- `tools/run/phase3_continue_integer.py`, `tools/phase3/jobs.py`,
  `tools/phase3/screening.py`, `tools/phase3/acceptance.py`,
  `tools/analysis/phase3_d4.py`
- `docs/roadmap/phases/phase-03-remaining-work.md`,
  `docs/architecture/phase3-device-migration.md`
