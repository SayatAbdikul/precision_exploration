# Standard generic timing harness contract

Version: **1.0.0**  
Status: **accepted for Phase 0**  
Accepted: **2026-09-04**

## Scope

All public combinational, pipelined, and iterative arithmetic candidates are measured through the same register-to-register boundary:

```text
launch DFF -> DUT -> capture DFF
```

This contract defines comparison/accounting behavior. RTL implementation begins in later phases.

## Logical interface

The harness must expose:

- clock and synchronous reset;
- input operand code(s), format/config identity, and input-valid;
- output code/result and output-valid;
- ready/backpressure only when required by an iterative design;
- activity-trace points sufficient to replay identical logical operation sequences.

Widths and the number of operands are parameters of the tested block. Wrapper logic may adapt ports but may not change arithmetic semantics.

## Register accounting

- Launch/capture wrapper registers define realistic timing boundaries.
- Wrapper-register area is reported separately and excluded from combinational DUT core area.
- Architecture-owned internal pipeline/state registers are included in DUT area and power.
- Final pipelined comparisons include clock-tree/register power.
- Any adapter that performs decode, scaling, conversion, or buffering required by only one candidate belongs to that candidate's all-in cost rather than the neutral wrapper.

## Timing and throughput

Record target clock, achieved slack/fmax, latency in cycles, latency in time, and initiation interval.

```text
throughput = frequency / initiation_interval
```

- Combinational: typically one registered boundary stage; report the DUT combinational path separately from wrapper setup/clock-to-Q assumptions.
- Pipelined: internal stages count toward latency and register/clock cost; steady-state II is reported explicitly.
- Iterative/serial: every required cycle counts; an N-cycle latency does not imply II=1.

## Comparison conditions

- Same PDK, library, corner, VT class, synthesis/physical settings, and load/drive assumptions.
- Multiple progressively tighter timing constraints selected after the ICS55 pilot.
- Iso-frequency comparison at common achievable clocks plus architecture-optimized Pareto points.
- Same numerical manifest/config and identical logical operand sequence for semantically equivalent candidates.
- Vectorless activity is labeled preliminary; final energy uses registered representative traces.

## Required result identity

Every harness result retains datatype manifest hashes, arithmetic/MAC/accumulator configuration, RTL hash, pipeline choice, target clock, PDK/library/corner/VT, tool versions, constraints, trace hash, and metric/evidence level.

## Acceptance rule

A hardware point is invalid if the DUT fails oracle/backend conformance, wrapper logic hides candidate-specific support cost, latency/II is missing, or measured quality cannot be resolved to the exact same numerical configuration.
