"""Run matrix cells whose code-occupancy audit contains a non-finite SQNR (stage 2, added 2026-10-01).

Why: ``engine.audit_summary`` (frozen) reports ``sqnr_db = 10 log10(signal / noise)`` and guards only
``noise == 0``.  When the FP32 reference of an audited tensor is identically zero but its quantized copy is not
(binary_pm1 has no zero code, so every zero maps to +-1), the SQNR is ``-inf`` dB.  The sealing JSON encoder
refuses non-finite floats, so ``matrix.run_cell`` fails after measuring the cell (seen on MobileNetV3-Large
binary_pm1 in chain-r2).  This module changes nothing numeric: inside ``patched()`` it

* replaces each non-finite ``sqnr_db`` of the audit rows by ``None`` and keeps the value as text in
  ``sqnr_db_nonfinite`` (``occupancy_summary`` already skips ``None``, so median and minimum SQNR are over the
  finite tensors; the record says which tensors were excluded), and
* adds this file's hash to the record's ``own_sources`` so that every cell run through it is identifiable.

The configuration identity, the readout and the logits do not depend on either change (``own_sources`` and the
audit are not inputs of the identity).  Use: ``python -m tools.run.experiment_b2_matrix_finite cells --model M
--format F --recipes default,minimal`` (same arguments as ``tools.run.experiment_b2_matrix cells``).
"""
from __future__ import annotations

from contextlib import contextmanager
import math

from tools.experiment_b.common import ROOT, file_hash

THIS = "tools/experiment_b2/matrix_finite.py"


def finite_rows(rows):
    """Copy of audit rows with non-finite ``sqnr_db`` replaced by ``None`` (value kept as text)."""
    out = {}
    for name, row in rows.items():
        value = row.get("sqnr_db")
        if value is not None and not math.isfinite(value):
            row = {**row, "sqnr_db": None, "sqnr_db_nonfinite": str(value)}
        out[name] = row
    return out


@contextmanager
def patched():
    from . import engine, matrix
    original_audit, original_own = engine.audit_summary, matrix.own_sources

    def audit_summary(interpreter):
        return finite_rows(original_audit(interpreter))

    def own_sources():
        return {**original_own(), THIS: file_hash(ROOT / THIS)}

    engine.audit_summary, matrix.own_sources = audit_summary, own_sources
    try:
        yield
    finally:
        engine.audit_summary, matrix.own_sources = original_audit, original_own


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "cells":
        raise SystemExit("only the 'cells' job is supported (campaigns use tools.run.experiment_b2_matrix)")
    from . import matrix
    sys.argv = ["experiment_b2_matrix_finite", *argv]
    with patched():
        return matrix.main()
