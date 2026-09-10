# Phase 2 remaining work and time estimates

Snapshot: 2026-09-10, 15:14 UTC. Phase 2 is still in progress. Estimates below are wall-clock time on the current machine, not guarantees; access and artifact acquisition have no reliable deadline.

| Remaining work | Current evidence and completion condition | Estimated time remaining |
| --- | --- | --- |
| Restore frozen ImageNet training/calibration images | 1,090 / 2,000 image hashes verified; 910 missing. The original archive recovery worker remains alive with eight streams. Completion requires every frozen payload hash to match. | **5–8 hours**, provisionally. About 1,000 images recovered in six hours; class sizes, network throughput, and retries vary. Allow longer if the server slows down. |
| Generate and validate classifier calibration artifacts | After training recovery, calibrate ResNet18, MobileNetV2, and MobileNetV3 Large for FP6 E3M2 and INT8, then verify artifact identities. The continuation script runs these automatically. | **15–60 minutes**, a planning estimate, not yet measured for all three models. Depends on complete training data. |
| Close D2 profiling evidence | The collector and validated CSV importer are implemented. A fresh attempt returned `ERR_NVGPUCTRPERM`; process-only sudo still requires interactive administrator authentication. Raw counter and decision artifact hashes are now checked again by final verification. | **15–45 minutes** for profiling, export, and review once counter access is available. **Access wait unknown.** |
| Restore the original frozen YOLO FP32 prediction artifact | All three classifier artifacts now match their original hashes. YOLO replay reproduces mAP closely, but its JSON differs from the frozen SHA-256. Need the original artifact or a byte-identical reconstruction; do not replace the expected hash with new results. | **5–15 minutes** to verify and rerun the regression once the original artifact is available. **Artifact acquisition time unknown.** |
| Refresh final verification and close documentation | All four models' native-image reports remain verified. Repeat relevant checks after data/calibration completion and retain open gates. Latest focused CPU suite: 246 passed. Full suite: 264 passed, four failures tied to missing training payloads and the YOLO artifact. | **10–20 minutes** of review/update work. CPU test execution itself was approximately 13–15 seconds per suite. |

The compute/data path is approximately **5½–9½ hours from this snapshot**, assuming similar download throughput and no calibration failures. This is not an overall Phase 2 completion promise: administrator counter access and the historical YOLO artifact can extend the schedule indefinitely.

## Already completed

- ImageNet evaluation 10k and screen 1k payloads restored and hash verified; COCO payloads restored.
- C++/CUDA conformance for all 25 accepted number formats; the existing CUDA suite passed 93 tests for the current engine source.
- Eight native-resolution images per core model are complete and verified: 3,720 matching C++/CUDA layer comparisons across ResNet18, MobileNetV2, MobileNetV3 Large, and YOLO. All classifier output tensors also match.
- All three original classifier FP32 prediction artifacts restored byte for byte.
- YOLO's strict FP6 zero-mAP behavior diagnosed as candidate range/precision collapse; it is not an outstanding engine mismatch.
- D3 accepted: retain the four core models, with EfficientNet optional.

Wide-accumulator acceptance for Phase 3 Experiment A sweeps remains a later-phase prerequisite, rather than an additional Phase 2 completion claim.

## Active continuation and evidence

The one-shot continuation, `python -m tools.run.phase2_data_gate`, resumes verified training recovery, creates missing classifier calibrations, runs CPU checks, and refreshes the final report. Its persistent state and log are:

- [Worker state](../../../artifacts/dataset_indexes/phase2-data-gate.json)
- [Worker log](../../../artifacts/dataset_indexes/phase2-data-gate.log)
- [Final verification report](../../../results/summaries/phase2-final-verification.json)
- [Phase 2 roadmap](phase-02-exact-engine.md)
- [D2/D3 evidence and decisions](../../decisions/d2-d3-phase2-runtime-and-breadth.md)

Avoid launching a second recovery process while the existing worker is alive. A stopped or failed worker can be resumed; existing payload hashes and calibration artifacts are checked before reuse. The final report must retain open gates until their evidence actually passes.

## Work completed after creating this checklist

- Validated all three final classifier reports against their frozen samples, models, graphs, source identity, layers, and output tensors; native conformance is closed.
- Added explicit final-report gates for missing classifier calibration artifacts and incomplete/current-source profiling. Publication of the final report is now atomic.
- Latest verification after these changes: **218 focused CPU tests passed**; **236 full-suite tests passed, four failed** because training payloads and the original YOLO prediction artifact remain incomplete.

## Continuation at 15:14 UTC

- Kept the existing training worker (PID 69179, recovery child 69183) running;
  no duplicate recovery was launched.
- Implemented separate counter capture attempts and validation of raw/details
  CSV, all six launch identities, metric units, duration, occupancy and DRAM
  traffic. Tests cover incomplete, corrupt, stale and mismatched evidence,
  plus failed and successful collector execution with simulated profiler output.
- Checked the actual raw CSV layout using a bundled Nsight report. Fresh GPU
  collection remains blocked by counter permissions and sudo authentication.
- Refreshed profiling, D2/D3 and final verification reports. D2 remains open;
  all retained profiling and decision artifact hashes validate.
- **246 focused CPU tests passed; 264 full-suite tests passed, four failed.**
  No numerical engine sources changed, so the retained 93-test CUDA evidence
  remains current. Administrator collection commands are documented in
  [the engine guide](../../architecture/exact-engine.md).
