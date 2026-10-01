"""Versioned exact graph adapter for the useful-quality FP6/FP7 study.

The frozen v1 prepared executor and rational oracle remain unchanged.  This
adapter installs the separately built, graph-bound grid backend and the
oracle-checked finite FP64 post-operations only for the selected graphs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from public.inference.native_fp64_grid_v2 import (
    CertifiedGridBackend, SELECTED_GRAPHS, library_path, verify_library,
)
from tools.exact_execution_v2.engine import PreparedGraph, PreparedOperators
from tools.exact_execution_v3.postops import FP64Stores, FloatingResiduals


ROOT = Path(__file__).resolve().parents[2]


class OptimizedPreparedOperators(PreparedOperators):
    """Keep all original operator domains while replacing proved hot paths."""

    def __init__(self, backend, library, constants, graph):
        super().__init__(backend, library, constants)
        self._fp64_stores = FP64Stores()
        if backend in {'cpp', 'cuda'}:
            self.native = CertifiedGridBackend(backend, graph, fallback_library=library)

    def _store(self, states, shape, accumulator, scales, bias, activation, output):
        result = self._fp64_stores.store(states, shape, accumulator, scales, bias, activation, output)
        return result if result is not None else super()._store(
            states, shape, accumulator, scales, bias, activation, output
        )


class OptimizedPreparedGraph(PreparedGraph):
    """Prepared graph with exact certified FP64 MAC, store and residual paths."""

    def __init__(self, document, *, backend='reference', library=None):
        super().__init__(document, backend=backend, library=library)
        if self.graph_sha256 in SELECTED_GRAPHS:
            self._operators = OptimizedPreparedOperators(backend, library, self._constants, document)
            self._residuals = FloatingResiduals()


def runtime_identity() -> dict[str, str]:
    """Return checked file identities that v2 archives before enrollment."""
    paths = [ROOT / 'tools/exact_execution_v2/optimized_engine.py',
             ROOT / 'tools/exact_execution_v3/postops.py',
             ROOT / 'public/inference/native_fp64_grid_v2.py',
             ROOT / 'public/inference/cpp/fp64_grid_v2.cpp',
             ROOT / 'public/cuda/kernels/fp64_grid_v2.cu']
    for backend in ('cpp', 'cuda'):
        binary = library_path(backend)
        verify_library(backend, binary)
        paths.extend((binary, binary.with_suffix('.json')))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}
