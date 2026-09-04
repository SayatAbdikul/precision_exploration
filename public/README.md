# Public architecture-independent study

This tree is the implementation area for the complete precision-exploration study. It remains independent of MANT-specific ISA, topology, mapping, compiler, simulator, RTL, and chip details.

| Directory | Responsibility |
|---|---|
| `formats/` | Datatype manifests, high-precision oracle, and conformance vectors |
| `quantization/` | Deployment graph preparation, calibration, and controlled/optimized PTQ |
| `inference/` | Reference and C++ exact inference plus operator implementations |
| `cuda/` | Exact CUDA kernels and backend strategy benchmarks |
| `workloads/` | Model adapters, dataset interfaces, and fixed subset handling |
| `experiments/` | Machine-readable configs, scheduler, lifecycle, and run registry |
| `generic_rtl/` | Arithmetic, accumulation, scaling/conversion, memory, harness, PE/tile RTL |
| `pdk_flow/` | Shared physical-flow logic, primary ICS55 flow, optional SKY130 cross-check |
| `analysis/` | Quality, statistics, hardware parsing, and Pareto analysis |
| `package/` | Versioned Pareto-plus-guard package schema and external export logic |

## Required implementation order

1. Consume the accepted contracts and implement datatype manifests plus the high-precision oracle.
2. Freeze checkpoints, preprocessing, datasets, and disjoint calibration/evaluation manifests.
3. Build graph folding, controlled PTQ, exact operators, and C++/CUDA backends.
4. Prove bit-exactness from primitive truth tables through network checks.
5. Run Experiment A with conservative statistical promotion, then Experiment B and accumulation/W/A sweeps.
6. Freeze full-validation candidates before expensive generic RTL/PPA.
7. Build Pareto sets at increasing realism: primitive, MAC/FPU, scale/conversion, memory, generic PE/tile/system.
8. Export the public Pareto plus guard set.

## Boundary rule

Nothing in this tree may import MANT code or use MANT-specific constants. The only downstream interface is the schema and exporter under `package/`.
