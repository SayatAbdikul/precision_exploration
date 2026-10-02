"""Paths, evidence helpers and source identities for scaled bridge v2.

v1 is imported, never edited. Exports are data: their identity depends on the
exporter sources only, so one export stays valid across engine revisions. Run
evidence is keyed by the engine source digest.
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from tools.experiment_b.common import ROOT, digest, file_hash, unseal, seal, atomic_json
from tools.phase3.common import reference, checked
from tools.scaled_bridge_v1.common import immutable, array_hash, save_npz

BASE = ROOT / 'artifacts/scaled_bridge_v2'
PACKAGE = Path(__file__).resolve().parent   # the live package or an archived copy of it
LOGICAL = 'tools/scaled_bridge_v2/'
EXPORT_SCHEMA = 'scaled-bridge-export-2'
EXPORTER_FILES = ('export.py', 'codebooks.py')
NON_NUMERIC_FILES = ('report.py', 'archive.py', 'b2_adapter.py', 'b2_replay.py')
GPU_LOCK = ROOT / 'artifacts/agent_orchestration/gpu.lock'


def logical(path):
    """Name of a source file in identities: package files keep their live name inside an archive."""
    path = Path(path).resolve()
    if path.is_relative_to(PACKAGE):
        return LOGICAL + str(path.relative_to(PACKAGE))
    return str(path.relative_to(ROOT.resolve()))


def _hashes(paths):
    return {logical(p): file_hash(p) for p in sorted(paths)}


def exporter_sources():
    return _hashes(PACKAGE / name for name in EXPORTER_FILES)


def engine_sources():
    files = [p for p in PACKAGE.glob('*') if p.suffix in {'.py', '.cpp', '.cu', '.h'}
             and p.name not in NON_NUMERIC_FILES]
    files += sorted((PACKAGE / 'manifests').glob('*.json'))
    files += [ROOT / 'tools/scaled_bridge_v1/common.py',
              ROOT / 'public/formats/manifests/accumulators/fp16_e5m10_accumulator.json',
              ROOT / 'public/inference/reference/arithmetic.py',
              ROOT / 'public/formats/manifests/accumulators/fp32_e8m23_accumulator.json',
              ROOT / 'public/formats/manifests/accumulators/fp64_e11m52_accumulator.json']
    return _hashes(files)


def run_root():
    return BASE / 'runs' / digest(engine_sources())


def case_id(model, fmt, recipe, family='b1'):
    return f'{model}-{fmt}-{recipe}-{family}'


def under_gpu_lock():
    """True when the launcher declares that it holds the shared GPU lock."""
    return os.environ.get('SCALED_BRIDGE_V2_GPU_LOCKED') == '1'


class Stopwatch:
    """Wall-clock accounting for one measurement invocation (auditable cost)."""
    def __init__(self, label, arguments):
        self.label, self.arguments, self.start = label, list(arguments), time.time()

    def close(self, **extra):
        record = {'label': self.label, 'arguments': self.arguments, 'started_epoch': self.start,
                  'wall_seconds': time.time() - self.start, 'gpu_lock_declared': under_gpu_lock(),
                  'pid': os.getpid(), **extra}
        folder = run_root() / 'ledger'
        folder.mkdir(parents=True, exist_ok=True)
        seal(folder / f'{int(self.start * 1000)}-{os.getpid()}.json', record)
        return record
