# Integer Model C baseline, version 1

`integer_mac.sv` is a reusable signed two's-complement dot-product core. Its
parameters `W_BITS`, `A_BITS`, and `ACC_BITS` are independent; verified
instances use W/A 4, 5, 6, or 8 bits and INT32 or INT64 accumulators, plus
INT4×INT8 and INT8×INT4 with INT32. Only signed raw integer codes at scale 1
are represented. The DUT multiplies into all `W_BITS+A_BITS` product bits,
adds each exact product to a signed saturating accumulator, then adds one
stored accumulator-width bias code after the final product. This matches
`public.inference.reference.arithmetic.model_c` for those declared domains.

Every `input_valid` edge accepts one product. `dot_start` discards the previous
sum and starts at zero; `dot_end` emits the post-bias accumulator code and
asserts `output_valid` after that edge. A singleton dot asserts both flags.
Invalid cycles hold the accumulator and produce no valid output. Synchronous
reset clears state and the output. Distinct dots can follow on consecutive
edges. `dot_start` and `dot_end` are meaningful only when `input_valid=1`;
the caller must send well-formed noninterleaved dots and supply the same stored
bias code on the end cycle. There is no backpressure: initiation interval is
one **product**, and a dot of length K uses K accepted cycles. Throughput in
complete dots depends on K; it is not one full dot per cycle except at K=1.

The multiplier, accumulator saturation, and feedback state are DUT owned. The
bias adder and result register are also DUT owned. There is no internal
arithmetic pipeline stage splitting the feedback path. `integer_mac_harness.sv`
adds neutral launch and capture registers; a result from a value launched at
edge N is captured after edge N+2. Those wrapper flops are excluded from DUT
area and reported separately by the physical flow.

The raw MAC output `result` is an accumulator code. Mapped W/A scales,
arbitrary scale multiplication, residual alignment, and other operator
functions require additional hardware. The actual integer model's bias code depends on its
product scale and graph; these scale and graph identities are not inferred
from the RTL. A raw MAC conformance result cannot be assigned a network
quality score without the exact complete numerical configuration.

`integer_output_convert.sv` adds a bounded output path for INT32/INT64
accumulator codes and INT4/5/6/8 stored output codes. It implements the exact
value `accumulator × 2^SCALE_EXP`, round to nearest even, saturate, and optional
ReLU, where `-ACC_BITS <= SCALE_EXP <= OUT_BITS`. The stored bias is included in
the MAC accumulator before this conversion. `integer_mac_scaled.sv` composes
the blocks; its output is valid one edge after the raw MAC output. The neutral
`integer_mac_scaled_harness.sv` adds launch and capture registers for a
comparable mapped-area experiment. Arbitrary mapped scales, scale selection,
residual alignment, and calibrated graph configuration remain outside this
block. Invalid RTL parameter combinations fail elaboration.

The generator in `requirements.py` projects all 25 accepted manifests and
graph accumulator-resolution references into
`results/summaries/hardware-requirements-v1.json`. It records unknown mapped
scale precision as unknown; the prior 64-bit storage width is analytical only.
For MX/BFP, a maximum detector cannot replace an accepted MSE block-scale
search without an equivalence proof or a new numerical configuration. A
parallel reduction or one-round exact accumulator is also a distinct numerical
configuration where sequential rounding changes the result.

## Reproduce

From the repository root, with the project's Python environment and Icarus
Verilog installed:

```bash
/home/maveric/precision_exploration/.venv/bin/python -m tools.hardware.integer_mac_baseline requirements
/home/maveric/precision_exploration/.venv/bin/python -m tools.hardware.integer_mac_baseline conformance
/home/maveric/precision_exploration/.venv/bin/python -m tools.hardware.integer_output_conversion conformance
```

The paths above identify the environment used for this run; another checkout
can use its own environment. The conformance command writes generated
testbenches and vector streams under ignored `artifacts/rtl/integer-mac-v1/`
and the compact, hash-linked result under
`results/summaries/integer-mac-conformance-v1.json`. INT4/5/6 primitive pairs
are exhaustive; INT8 uses boundary pairs plus seeded pairs. Directed sequences
exercise signed products, bias rounding before storage, positive/negative bias
clamps, valid gaps, reset, consecutive dots, multi-product dots, and an actual
INT8/INT32 accumulator overflow. INT64 internal overflow is not reachable
within a bounded practical vector stream; its bias saturation is exercised.
The 4×4/INT32 wrapper is independently checked for the two-edge capture
delay. Simulation of these dot products does not prove the scaling/conversion
or full-network implementation.

`integer_mac_baseline conformance` accepts `--output-dir` and `--summary` so
independent reruns and tests can use temporary output paths. The conversion
command accepts the same options. Its default evidence covers 80 static
conversion settings and three raw-MAC integration settings, with exact oracle
codes, signed boundaries, RNE ties, resets, and valid gaps. The area comparison
driver is `python -m tools.hardware.integer_output_synthesis --liberty PATH`;
it requires the same identified ICS55 RVT typical Liberty used by the raw
physical flow. Its result is cell-mapped Liberty area, without a timing or
power claim.
