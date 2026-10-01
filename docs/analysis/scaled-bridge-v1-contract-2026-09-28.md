# Scaled FP6/FP7 bridge: prospective contract and development protocol

This stage selects ResNet18/maxabs FP6 E2M3, FP6 E3M2 and FP7 E3M3 from earlier
development results. Its ordered samples are the existing SHA256-sorted
ImageNet development prefix. It is not independent final confirmation.
The machine-readable contract and promotion rule live in
`tools/scaled_bridge_v1/common.py` and are sealed before quality measurement.

The B anchor retains its exact FP32 folded parameters, external FP32 scales,
FP32 normalization and reconstructed operands, finite endpoint clipping,
manifest code-parity midpoint tie rule, quantization boundaries at every
original FX arithmetic node, deterministic CUDA runtime, and original batches
of eight. Replays use a read-only wrapper and never call B's writing runner.
Historical 1,000-image accuracy is distinct from each paired development panel.

The candidate uses B's weight codes and scale bits, with an explicit code-domain
dot product. The wide arm sums dyadic code products exactly in an INT64 grid
only after a per-channel bound proves every reduction prefix below 2^53 units.
The sum is converted exactly to binary64 code-level units. The control replaces
only that dot accumulator with sequential binary32 RNE fused multiply-add in
input-channel, kernel-row, kernel-column order, starting at positive zero.
Both arms then multiply by the activation scale in binary64, multiply by the
channel weight scale in binary64, and separately add the promoted folded FP32
bias in binary64. Those scale and bias operations each round to nearest even.
Scaling after the dot differs from B's FP32-rounded reconstructed operands.

Residual inputs reconstruct as binary64(code level × retained scale), add in
binary64 and quantize once to the output scale. ReLU and max-pool operate on
code values before reconstruction and output quantization; padded max-pool
positions are ignored. Global average pooling sums code-grid integers exactly,
converts the sum to binary64 code-level units, divides by the spatial count
with binary64 RNE, then multiplies by the input scale with binary64 RNE.
Identity and flatten preserve code and scale. Every other original B boundary
stores a new low-bit code by binary64 normalization, exact dyadic midpoint
comparison and the same manifest parity/order tie rule. Values outside finite
endpoints clip. Stored zeros are canonical positive zero. Nonfinite source or
raw arithmetic values fail closed; finite normalization overflow clips to an
endpoint. No legacy unscaled certificate is extended or relabeled.

Independent validation consists of rational primitive oracles, small convolution
geometry cases, scaled store/residual boundary tests, a first-image wide graph
reference using the original FX traversal and CPU binary64 framework convolution
on certified exact integer-grid operands, and fixed actual-node rational dot
checks in both arms. Each new arm then needs eight CPU/CUDA images with identical
layer codes, raw state hashes, outputs, predictions and new-contract diagnostic
signatures. Cross-backend agreement alone does not establish the reference.
The old unscaled FP7 acceptance remains absent.

All admitted cases run paired32 first. In the frozen format order, a case may
extend to128 if its wide top-1 is at least40%, is within20 percentage points of
its same-panel B anchor, all gates pass, and both arms fit the measured remaining
budget. Failed gates stop promotion; evaluation labels cannot redefine the
contract. Report wide-minus-B, control-minus-B, each-minus-FP32, and
control-minus-wide separately, using paired10,000-resample multinomial95%
intervals, seed20260928. These are pointwise development intervals.

A new ledger has a maximum of14,400 aggregate experiment worker-seconds,
including preparation, probes, conformance, failed and interrupted attempts,
both backends and both arms. Coding/compilation is separate. Each invocation
reserves conservative time before dispatch and is terminated at its reserved
limit; unreconciled attempts retain their full reservation. The old E1 ledger
is historical and is not reset or extended. Shared native resource locks are
required for this stage.
