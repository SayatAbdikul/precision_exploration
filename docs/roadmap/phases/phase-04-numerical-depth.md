# Phase 4 — Deepen the promoted numerical search

Scope: **in repository**  
Starts after: **D4 promotion**  
Parallelism: **four evidence-sharing branches**  
Status: **planned**

## Branch A — Weight/activation asymmetry

- **N4A.1** — Test W4A8, W5A6, W6A6, W6A8, W8A6, W8A8, and selected family-specific combinations.
- **N4A.2** — Test signed versus unsigned activations where semantically appropriate.
- **N4A.3** — Preserve families that specialize in weights or activations.

## Branch B — Accumulation

- **N4B.1** — Sweep native and widened accumulator precisions.
- **N4B.2** — Compare MAC Models A/B/C on promoted/finalist formats.
- **N4B.3** — Include Posit quire, Kulisch, or exact fixed accumulation where relevant.
- **N4B.4** — Keep deterministic sequential reduction as the main baseline; tree reduction is a labeled ablation.

## Branch C — Experiment B PTQ

- **N4C.1** — Add general external scaling where meaningful.
- **N4C.2** — Compare power-of-two scaling.
- **N4C.3** — Sweep per-tensor, per-channel, and block granularity.
- **N4C.4** — Explore FP bias/range, MX/BFP block 16/32/64, and family-specific PTQ.
- **N4C.5** — Evaluate practical first/last-layer exceptions as a secondary ablation.

Experiment B remains clearly separated from the controlled Experiment A result.

## Branch D — Generic cost proxies

- **H4.1** — Model scale-metadata bits/value.
- **H4.2** — Estimate family-specific scale-generation and application complexity.
- **H4.3** — Track ideal versus packed-logical memory cost.
- **H4.4** — Preserve potentially hardware-efficient candidates even when quality is slightly lower.

## Decision D5 — Full expensive ablation set

Do not Cartesian-expand every W/A × accumulator × scaling × block-size combination. Give broader sweeps to strong/near-Pareto candidates; give at least one reasonable optimized configuration to a unique family; exclude clearly redundant combinations. Base the choice on 1k/10k quality, confidence, family role, layer sensitivity, and rough generic cost.

## Decision D6 — Reasonable Experiment B policy per family

Freeze only after pilot evidence:

- optional scale kind and granularity;
- block size and axis;
- family-specific calibration/clipping/range tuning;
- support metadata and scale hardware included with the benefit.

D6 defines “reasonable best-achievable PTQ” rather than intrinsic controlled behavior.

## Outputs

- W/A and signedness evidence.
- Accumulator/MAC-model sensitivity.
- Separately labeled optimized Experiment B results.
- Scale/memory/support cost proxies.
- D5 full-ablation list and D6 family policies.

## Blocks

Phase 5 full validation.
