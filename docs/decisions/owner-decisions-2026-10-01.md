# Owner decisions — 2026-10-01

Recorded from the project owner's instruction in the working session of
2026-10-01: "go with your recommendations on all first three. On the name, we
will decide when we finish the whole work". The four questions were raised after
the related-work audit and the protocol drafts:

- `docs/analysis/related-work-audit-2026-10-01.md`
- `docs/decisions/proposed-decision-updates-2026-10-01.md`
- `docs/analysis/confirmation-protocol-draft-2026-10-01.md`

`docs/decisions/decision-register.md` has not been edited. These records are to
be folded into it when the owner chooses.

| # | Question | Decision | Status |
|---|---|---|---|
| 1 | Headline mechanism of the paper | The accumulator-width sweep down to the failure point, joined to ASIC MAC area and energy, with the certified lossless accumulator width as the reference. Simulator-versus-exact fidelity is the methods contribution. | Accepted by owner 2026-10-01 |
| 2 | Evaluation data while the ImageNet 50k payload is unavailable | The 9,000 images of the frozen ImageNet 10k evaluation list that lie outside the 1k screen are sealed as an interim confirmation set, as are COCO validation images outside the 1k screen. Development stays on the 1k screen lists. Runs on the reserved images wait until the hypotheses and the promoted set are frozen. | Accepted by owner 2026-10-01 |
| 3 | Scope | Post-training quantization only (no retraining or QAT, consistent with F5 and the README), CNN workloads only (no transformer workload), and no mixed precision (D14 stays an external gate). | Accepted by owner 2026-10-01 |
| 4 | Name collision between the private downstream architecture (MANT) and the data type of the HPCA 2025 paper "M-ANT" (DOI 10.1109/HPCA61900.2025.00086) | Deferred by the owner until the whole work is finished. | Open |

## Consequences for the work plan

- Decision 1 makes two experiments central: the software accumulator sweep
  (FP16, a 21-bit float and saturating narrowed integers, as specified at the
  end of `docs/analysis/scaled-bridge-v1-results-2026-09-28.md`) and MAC RTL for
  non-integer formats with certified and narrowed accumulators. The published
  accumulator-width formula (26, 32 and 34 bits for FP6 E2M3, FP6 E3M2 and FP7
  E3M3, against this project's certificates of 23, 29 and 31) must be reconciled.
- Decision 2 removes 10k development runs from the plan. Whether the reserved
  images are in fact untouched is being established by
  `docs/analysis/evaluation-history-audit-2026-10-01.md` (in progress). The
  YOLOv8n FP32 baseline was already scored on all 5,000 COCO images, so the COCO
  complement is not untouched for that baseline and this must be disclosed.
- Decision 3 removes the QAT anchor, the transformer workload and per-layer
  mixed-precision assignment from the plan. A held-out CNN for a transfer test
  remains possible.

## Owner questions still open

From the fact-check of the drafts: the named primary contrasts and their
decision criterion and margins; whether the historical budget ledgers stay
binding; the status of the scaled-bridge contract relative to F6; the source of
the calibration seeds; unsigned activation formats (extend D1 or keep as a
labelled recipe switch); a second PDK with memory macros (D9); and when the
ImageNet 50k payload will be supplied.

## Later the same day

- New RTL work (non-integer MACs, saturating accumulators, place and route) is
  on hold: the owner requires 50 GB to remain free on the workstation disk
  (instruction of 2026-10-01: "let's ignore the rtl work for now, because we
  need 50GB free in the disk"). The OpenROAD container image and the raw
  synthesis outputs were removed with the owner's approval to restore headroom;
  the result tables, documents, scripts and timing reports remain, and the runs
  regenerate deterministically.
