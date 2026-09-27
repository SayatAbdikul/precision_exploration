# Resumable Phase 3 controller

From a terminal, run:

```bash
cd /home/maveric/precision_exploration && .venv/bin/python -m tools.run.phase3_resume
```

Use the same command after interruption. The controller checks retained evidence,
reuses completed image/backend checkpoints, and restarts an unfinished image.
MobileNetV2 INT8 has 851 valid saved images at implementation time, so its screen
continues with image 852 after the execution compatibility check. ResNet18 INT8's
completed 1,000-image screen is verified and reused. Keep the terminal open.

## Scope and scientific gates

The initial inventory contains 29 execution/analysis tasks covering eight integer
configurations. Five already have acceptance; three MobileNetV2 pilots can reuse
their original CPU results and add CUDA evidence before acceptance. The controller
queues per-graph store validation, acceptance, 1,000-image screening, paired
statistics and layer diagnostics as their dependencies become ready.

**92 configurations still need graph/family acceptance. Phase 3 is not complete.**
They appear as blocked configurations, not rejected candidates. The controller
does not invent proofs or change the frozen detector calibration, accumulator
semantics or D4 methodology. Supported integer acceptance evidence is discovered
on the next invocation. Proof development, sensitivity experiments and scientific review
remain separate work; this is not a fully automated all-family research pipeline.

## Development-proof handoff (2026-09-23)

An isolated checkout contains reproducible local proof code for the 16 shared
graphs and the other 76 blocked graphs. The retained reports are
`artifacts/phase3/proof-development/shared-v1/summary.json` and
`artifacts/phase3/proof-development/remaining-v2/summary.json`. Together they
cover 4,729 MAC nodes. The ordinary FP64 analysis additionally proves exact
finite MAC prefixes for 1,515 nodes. These are local, conditional statements;
they have **not** been promoted to graph acceptance.

Further local MAC/output checks now cover Q1.6, six saturating FP formats,
Posit4/6/8 and scaled binary/ternary. Their versioned reports under
`artifacts/phase3/proof-development/` retain both passing cases and concrete
FP8 E4M3FN and posit bias-boundary counterexamples. A separate
`family-static-v2` report recomputes all MAC and non-MAC arithmetic for **21
classifier graphs**; each still requires native pilot evidence. Across the
new local checks, 35 configurations have conditional MAC output-code proofs.
None has been promoted to graph acceptance.

Verify the 92 hash-checked records, frozen campaign/engine identities and
current graph approvals, then run the 49 focused tests from archived proof
sources without touching the running controller:

```bash
.venv/bin/python -m tools.analysis.phase3_development_preflight --summary --test
```

The archived sources make this check independent of the isolated development
checkout. The checkout retains the resumable proof generators and their READMEs.

The pilot batch has a safe dry run for all 92 blocked graphs, prioritizing the
21 static classifier candidates. After the current controller exits, run:

```bash
.venv/bin/python -m development.acceptance_proofs.pilot_batch --run --workers 2
```

It uses the production controller's exclusive locks, verifies/reuses saved
pilot images, and executes eight C++ and eight CUDA images per graph. Adjust
`--workers` only after measuring memory and throughput for noninteger formats.
Pilots are diagnostic and may fail; this command does not issue acceptances.

The preflight currently finds eight accepted integer graphs and 92 still
blocked. Remaining acceptance work includes 729 non-MAC nodes, 225 posit bias
layers, output-code and nonfinite reasoning, and eight matching native pilot
images per graph. The frozen detector calibration and D4 reviews require their
own decisions. The local proof reports are deliberately excluded from the
controller's acceptance discovery; adding them as approvals would allow
unverified full screens. The current screening verifier also recomputes the
integer proof, so it **cannot yet screen noninteger families** even if approval
files are added. Develop and validate family-specific acceptance and verifier
dispatch in an isolated checkout, then integrate only when the live run has
stopped and the execution identities have been reviewed. The one-line resume
command above continues the currently accepted integer work; it cannot by
itself finish the full 100-configuration Phase 3 campaign today.

## Resource selection and progress

The resource profile is measured on this machine and verified against the current
controller, numerical engine, preprocessing, package versions, native libraries
and GPU identity. An absent profile defaults to one CUDA worker; a stale profile
fails with an explicit recalibration/override message.

```bash
.venv/bin/python -m tools.run.phase3_resume --calibrate --levels 1 2 4 6 --images 2
.venv/bin/python -m tools.run.phase3_resume --dry-run
.venv/bin/python -m tools.run.phase3_resume --status
```

`--workers N` explicitly overrides capacity (1–6); memory/disk admission still
applies. Each CUDA process uses four PyTorch/OpenMP/MKL threads and one OpenBLAS
thread. One separate CPU lane handles acceptance and analysis. Reserves are 6 GiB
available RAM, 2 GiB free VRAM and 20 GiB disk space, with a conservative 3 GiB RAM
reservation per worker including pending allocations. Admission pauses as slots
or reserves become unavailable. If no running work can free resources, the run
ends blocked so the user can address the constraint and restart.

The two-image ResNet18 INT4 benchmark measured the following, including startup
and validation:

| CUDA workers | Verified image executions/hour | Measurement |
| --- | ---: | --- |
| 1 | 42.5 | initial |
| 2 | 83.0 | initial |
| 4 | 164.0 | initial |
| 4 | 162.0 | final repeat |
| 6 | 240.3 | final; selected |

The six-worker run executed 12 diagnostic images in 179.8 seconds. Peak worker
RSS was about 1.01 GiB; minimum available RAM was 17.76 GiB and free VRAM was
9.70 GiB. Single-image execution fell from roughly 104 to 70 seconds in the
one-worker measurements. These are repeated diagnostic images, not new screening
results or a campaign-wide ETA. Other graphs and arithmetic families may have
substantially different costs, and throughput falls as the queue runs out of
independent ready configurations. The final measured profile is
`artifacts/phase3/controller/benchmarks/20260923T074159Z-3d5b3e9e/profile.json`.

Status is written every scheduling cycle to
`artifacts/phase3/controller/status.json`. Immutable run directories under
`artifacts/phase3/controller/runs/` retain inventory, tasks, logs and result
receipts. Resource measurements and compatibility checks have separate directories.
The status command requires at least one controller run; dry-run prints inventory
without starting or writing a campaign run. Historical image timings may span
machines and are deliberately not converted into a misleading current-host ETA.

## Resume and integrity guarantees

- The controller holds the legacy CPU/GPU locks for its lifetime. Workers inherit
  the appropriate kernel lock descriptor and acquire an additional per-job lock.
  Legacy runners, duplicate controllers and duplicate jobs cannot write together.
- Completed registry rows are reusable only after their output evidence verifies.
  Missing or corrupt evidence fails explicitly; historical records are not silently
  repaired. Pilot imports preserve source summary and image references.
- Ctrl+C/SIGTERM stops admission and terminates worker process groups. Completed
  checkpoints survive. Stale registry leases require 300 seconds without heartbeat;
  an immediate restart can report that the old lease is still running. Wait for
  that threshold and rerun. Live kernel locks are never bypassed.
- Deterministic failures block dependent tasks while independent work continues.
  A new invocation retries eligible unfinished work. There is no automatic
  infinite retry loop.
- Exit 1 means execution or integrity failure; exit 2 means scientific work or
  resources remain blocked, or execution finished but D4 review is still required.
  Signals return 128 plus the signal number. Dry-run/calibration/status return 0
  when their own operation succeeds; that does not indicate Phase 3 completion.

Do not edit execution sources, frozen inputs or libraries while workers run.
The execution guard refuses changed dependencies instead of silently rebasing
historical results. Source changes require explicit review and revalidation.

## Exact integer store extension

The frozen numerical source and pipeline identities remain unchanged. A separately
identified execution extension vectorizes the expensive integer output store.
It supports signed INT32/INT64 accumulators, signed integer outputs of at most
eight bits, identity/ReLU/ReLU6, and positive finite-decimal scales with numerator
and denominator bounded to 128 bits. Other cases use the original store.

Bias encoding uses the original oracle. Saturating bias addition clamps before
host integer addition to prevent INT64 wraparound. Exact rational half-step
thresholds and integer search implement round-to-nearest-even output codes.
The supported scale domain keeps original pre-division values within the frozen
200-digit oracle's exact representation; rational thresholds preserve ties and
non-tie code decisions. ReLU6 uses the original oracle's encoding of six.
Diagnostics observe the same exact pre-quantization values through a lazy sequence.
The native MAC order, saturation, weight encoding and graph structure are unchanged.

Before production use on each graph, eight original pilot images must reproduce
every layer hash, final output and quantizer-event counts. Certificates bind the
graph, accepted pilot, libraries, execution guard and controller identity. Each
new accelerated image records its certificate and implementation identity inside
the hashed checkpoint, so interrupted tasks retain execution provenance. Original
checkpoints remain unchanged. Capacity benchmarks alone cannot issue these
production certificates.

The known fixed-point scratch-width issue is guarded by rejection before unsafe
dispatch. A revised native kernel remains separate versioned work; the guard
does not claim the affected domain has been repaired or accepted.

## D4 review

The [2026-09-25 review of eight completed integer screens](phase3-eight-integer-review.md)
records their paired statistics, layer-diagnostic limits, four severe-loss
diagnosis gaps, and current sensitivity, hardware, family and calibration
findings. It is an interim review; the recomputed decision remains `D4_OPEN`.

`tools.analysis.phase3_d4` retains its default open/readiness behavior. It also
accepts `--review PATH` for an explicit attributed review. Review schema
`phase3-d4-review-1.0.0` requires `campaign_sha256`, `decision: approve_D4`, a
`reviewer`, timezone-aware `reviewed_at`, and a `configurations` list covering
each measured model/format exactly once with its exact analysis reference.
Entries may supply `preservation_signals` and a specific `diagnosis` reference.
Required `reviews` keys are `sensitivity`, `hardware_preservation`,
`family_preservation` and `calibration`; each requires completed status, reasoning
and hash-checked evidence. A diagnosis must name the configuration and analysis,
include evidence/reasoning, and conclude `catastrophic_configuration`.

Review cannot bypass missing full screens. All 100 configurations must satisfy
the existing conservative promotion policy before D4 is accepted. Decisions are
content-addressed under `artifacts/phase3/d4-decisions/`. No D4 approval or diagnosis
was fabricated as part of controller implementation.

## Implementation validation

On 2026-09-23, the complete test suite passed: **492 tests**. Added cases cover
kernel lock inheritance and legacy exclusion, job ownership, missing/corrupt
completed evidence, dependency failure propagation, memory admission, source
guards, interrupted checkpoint provenance, process-group termination (including
orphaned analysis children), exact integer rounding/saturation/diagnostics and
D4 review constraints. Real CUDA benchmarks compared every layer, output and
quantizer-event count against retained accepted evidence. The final dry-run
validated the 1,000 ResNet18 INT8 and 851 MobileNetV2 INT8 saved images.
An integration check through the production worker also replayed acceptance and
reused the completed ResNet18 INT8 screen successfully, without rerunning its
images. The final profile and all referenced benchmark evidence passed identity
and hash verification.
