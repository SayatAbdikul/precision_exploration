# B recipe completion v1 run report

2026-09-28. The seven arms were selected using earlier development outcomes. These remain development FP32-QDQ results, not exact-arithmetic admission, hardware throughput, or final benchmark confirmation.

## Run status

The prepared one-hour run stopped after five of seven configurations completed because the extension runner attempted to write an existing validation record. Its write guard refused the rewrite at `artifacts/experiment_b_ext/validation/095d319b3087b139585312b504bdecbf2628a127bedf29953b7df47881931b02.json`. No resumption was attempted. The verifier passed for all seven retained prefixes and all five completed 1,000-image candidate/baseline pairs. The two incomplete cases remain at their original 128-image prefixes.

The verified ledger has 188/200 configurations at 1,000 images, 12 at 128 images, and 88 recipe pairs complete at 1,000 images. The five extensions added 4,360 candidate predictions (872 each).

## Paired outcomes

Deltas are percentile_99_9 minus maxabs, in percentage points, on identical ordered 1,000-image panels. Intervals are pointwise paired 95% intervals from the read-only `compare_records` / `paired_outcomes` helper (10,000 multinomial paired resamples, seed 20260927). Incomplete rows have only the retained 128-image prefix; no 1,000-image comparison is available.

| Model | Format | Completed images | Percentile Top-1 % | Maxabs Top-1 % | Top-1 Δ pp [paired 95% interval] | Percentile Top-5 % | Maxabs Top-5 % | Top-5 Δ pp [paired 95% interval] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet18 | mxfp8_e4m3 | 128 (prefix) | — | — | — | — | — | — |
| mobilenet_v2 | mxfp8_e4m3 | 128 (prefix) | — | — | — | — | — | — |
| resnet18 | fp7_e3m3 | 1,000 | 58.9 | 69.7 | -10.8 [-13.3, -8.3] | 86.8 | 88.5 | -1.7 [-3.0, -0.4] |
| resnet18 | fp8_e4m3fn | 1,000 | 59.7 | 69.0 | -9.3 [-11.8, -6.8] | 86.6 | 87.9 | -1.3 [-2.6, +0.0] |
| resnet18 | log8 | 1,000 | 58.0 | 66.4 | -8.4 [-11.2, -5.6] | 86.2 | 87.9 | -1.7 [-3.2, -0.3] |
| mobilenet_v3_large | fp8_e4m3fn | 1,000 | 61.2 | 68.6 | -7.4 [-9.9, -4.9] | 85.0 | 87.4 | -2.4 [-4.2, -0.6] |
| mobilenet_v3_large | log8 | 1,000 | 61.3 | 68.6 | -7.3 [-9.9, -4.7] | 87.2 | 88.8 | -1.6 [-3.4, +0.2] |

## Timing and resource observations

The runner was active for 133.6 seconds end to end. Recorded candidate inference time across completed extensions summed to 73.8 seconds, or 59.1 newly computed images/second during those inference intervals. End-to-end throughput was 32.6 new images/second. Inference timings exclude calibration/setup and are distinct from total wall time.

Sustained GPU utilization could not be measured: the bounded `nvidia-smi` monitor failed to communicate with the driver during execution. A later single sample showed 42% utilization, 861 MiB / 12,282 MiB VRAM and 45 °C; this is not sustained evidence and cannot establish whether the 50–65% soft goal was met.

## Preservation and scope

Launcher verification passed: seven original 128-image prefixes were checked; each completed 1,000-image candidate and baseline matched sealed digests and ordered sample identities; original configuration identities and 128-image summaries were preserved; and completed counterpart summaries were not rewritten. Actual coverage above is calculated from sealed results.

Selection used earlier development outcomes. These FP32-QDQ results do not establish exact arithmetic, hardware throughput, or final benchmark performance. No follow-on experiments were run.
