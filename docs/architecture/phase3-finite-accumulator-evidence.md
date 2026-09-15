# Phase 3 finite accumulator evidence

Local evidence now covers all 4,221 ordinary MAC nodes in the current matrix.
The remaining 804 MAC nodes use shared block scaling. This is numerical
coverage, not 100-configuration acceptance: only ResNet18 INT8/INT64 and
MobileNetV2 INT8/INT64 currently have verified eight-image C++/CUDA acceptance.

`tools.analysis.phase3_gate_inventory` joins current graph identities and unions
node coverage across the integer, fixed and FP64 audits. It does not count
overlapping checks twice or issue acceptance. The snapshot covers 4,999 of
6,600 non-MAC nodes. Remaining shared-scale, nonlinear, nonfinite-reachability
and native gates stay explicit in `phase3-gate-inventory.json`.

## Broad FP64 MAC bounds

`tools.analysis.phase3_finite_fp64_bounds` evaluates all finite activation codes
against actual prepared weight rows, channel scales and original biases. It
includes a sequential prefix magnitude bound, one rounding per exact
multiply-add, the prescribed bias store and addition, and absolute subnormal
rounding error. Shared block MACs are explicitly pending.

All 3,133 ordinary FP64 MACs across 61 configurations exclude accumulator
overflow under the admitted finite input domain. This includes the five
previously revised integer-operand configurations; their 319 MACs are a subset.
The reported error is relative to exact sums of the oracle's finite values,
including its versioned irrational-format approximations. It is not an error
bound against ideal irrational numbers or a complete network.

Two ResNet18 FP8 E5M2 layers have conservative worst-case bounds above half the
smallest output spacing: `layer4_0_conv2` (about 0.532 spacings) and
`layer4_1_conv2` (about 1.823). Other bounds are smaller. A bound below half a
spacing still cannot establish output-code equality near every threshold.

`tools.analysis.phase3_fp64_spacing_sensitivity` checks all 366 retained FP64
first/longest-K native states against these bounds and recomputes exact-reference
output stores. All stay within their bounds and preserve the stored output
codes; the largest observed error is about 8.40e-9 minimum output spacings.
Six selected rows across the two loose-bound layers also match. These targeted
rows overlap previous coverage and must not be added as six unique new states.
They do not tighten the all-input bound or establish native image acceptance.

## Non-MAC arithmetic

`tools.analysis.phase3_finite_fp64_nonmac` checks 4,113 ordinary non-MAC nodes
across those 61 configurations: 1,088 have local output-code equality evidence,
2,789 perform no accumulator rounding, and 236 remain pending in that report.
Existing integer-domain proofs cover some of these conservative pending cases;
the gate inventory takes their union using exact current graph identities.

Dyadic lattice checks cover residual sums and box arithmetic where every
bounded intermediate fits FP64 exactly. Global pooling additionally checks
division errors against output thresholds and handles exact ties explicitly.
Hard activations enumerate every finite input code, preserving negative-zero
semantics. Non-dyadic residuals enumerate every unordered aligned-value pair.
These are local checks on identical stored inputs and fixed observed shapes.

## Logarithmic residual sensitivity

Finite log6 and log8 residual sums have concrete FP64 output-boundary
discrepancies: 26 of 2,080 unordered log6 value pairs and 134 of 32,896 log8
pairs differ from an unrounded sum followed by the same output store. The
floating error is tiny, but the stored output can move to an adjacent value
when the exact sum lies near a quantization threshold. Counterexamples retain
both operands, sums and output codes. This does not identify a native backend
disagreement: the declared FP64 Model C arithmetic itself rounds the sum.

An eight-image study of the first ResNet18 residual stores and aligns fresh
FP32 branches in each candidate domain. It retains both branch-code arrays and
complete exact/FP64 pair tables. Log6 differs at 45,774 of 1,605,632 positions
(about 2.85%); log8 differs at 24,721 positions (about 1.54%). These are encoded
FP32-branch frequencies, not strict-graph activation frequencies.

A second study verifies the retained codes against a fresh FP32 pass and
propagates each precision alternative through the same FP32 tail. Both log6
alternatives and both log8 alternatives preserve the fresh FP32 correctness:
Top-1 5/8 and Top-5 8/8. Ordered Top-5 lists agree between the precision
alternatives on 5/8 log6 images and 6/8 log8 images. Thus the differences can
change downstream rankings without changing correctness on this subset.
No complete-graph precision sufficiency or D4 conclusion follows.

```bash
.venv/bin/python -m tools.run.phase3_residual_rounding_frequency
.venv/bin/python -m tools.run.phase3_residual_rounding_propagation
```

The corresponding summaries are `phase3-residual-rounding-frequency.json` and
`phase3-residual-rounding-propagation.json`. Both use separately identified
diagnostic scopes, retain per-image checkpoints and archive implementation
references. Original strict configurations and running engine identities remain
unchanged.
