# Disposable cache

This directory contains acceleration-only data. Everything must be safe to delete and recreate from tracked configurations/manifests plus registered artifacts.

Plan-derived cache candidates include:

- decoded copies of encoded weights;
- packed host-storage variants;
- compiled specialized CUDA kernels;
- LUTs copied into backend-specific layouts;
- im2col/intermediate tensor accelerators;
- query/report intermediates;
- local copies of already registered reusable outputs.

Cache keys include all inputs that affect the cached bytes. A cache hit must never substitute results from a different manifest, graph, accumulator, scale policy, kernel semantic version, or preprocessing identity.

Authoritative completed-run state belongs in `results/databases/`; durable generated evidence belongs in `artifacts/`. No scientific conclusion may depend on the cache being retained.
