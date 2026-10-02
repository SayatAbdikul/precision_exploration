"""Paths and identities of the fast exact-engine path (lane S1, protocol speed-protocol-v1).

The fast path imports the archived scaled bridge v2 package of digest 1f75c923... unchanged (export loading,
codebooks, certificates, accumulator policies, hashing) and replaces only how the engine executes: the archived
reduction kernels run on device pointers and the host NumPy post-operations run as torch CUDA float64/int64
operations. Its evidence is keyed by its own digest: the hashes of this package's numeric sources plus the
digest of the base archive's engine sources. Nothing is ever written under artifacts/scaled_bridge_v2/.
"""
from __future__ import annotations
import sys
from pathlib import Path
from tools.experiment_b.common import ROOT, digest, file_hash, seal, unseal, atomic_json  # noqa: F401

BASE_DIGEST = '1f75c9232c8a0202482fe9bce9a46360c7cc0999c5b0bcf7df5ebcc0446fc863'
RESNET_DIGEST = '7c6344af732d1bf324c07b285921f63393150cdb3875356c86237d934fdc2f92'
BASE_ARCHIVE = ROOT / 'artifacts/scaled_bridge_v2/implementations' / BASE_DIGEST
V2 = ROOT / 'artifacts/scaled_bridge_v2'
SPEED = ROOT / 'artifacts/speed_v1'
PACKAGE = Path(__file__).resolve().parent      # the live package or an archived copy of it
LOGICAL = 'tools/scaled_bridge_fast/'
NON_NUMERIC_FILES = ('archive.py', 'profile.py', 'cli.py', '__main__.py', 'validate.py', 'timing.py')


def base():
    """The archived 1f75c923 package, imported as top-level `scaled_bridge_v2` (never the live sources)."""
    path = str(BASE_ARCHIVE / 'py')
    if 'scaled_bridge_v2' not in sys.modules:
        sys.path.insert(0, path)
    import scaled_bridge_v2
    import scaled_bridge_v2.common as c
    if Path(c.PACKAGE) != BASE_ARCHIVE / 'py/scaled_bridge_v2':
        raise ValueError(f'scaled_bridge_v2 resolves to {c.PACKAGE}, not the 1f75c923 archive')
    if digest(c.engine_sources()) != BASE_DIGEST:
        raise ValueError('the base archive no longer resolves to its digest (a tracked dependency changed)')
    return scaled_bridge_v2


def fast_sources():
    files = [p for p in sorted(PACKAGE.glob('*')) if p.suffix in {'.py', '.cu', '.h'} and p.name not in NON_NUMERIC_FILES]
    return {LOGICAL + p.name: file_hash(p) for p in files}


def identity():
    return {'fast_sources': fast_sources(), 'base_engine_sources_digest': BASE_DIGEST}


def run_root():
    return SPEED / 'runs' / digest(identity())


def reference_root(case):
    """Run root of the archive that admits a case: 7c6344af for ResNet18, 1f75c923 for the MobileNets."""
    return V2 / 'runs' / (RESNET_DIGEST if case.startswith('resnet18-') else BASE_DIGEST)


def immutable(path, payload):
    """Seal once; an existing file must hold the same payload."""
    path = Path(path)
    if path.exists():
        if unseal(path) != payload:
            raise ValueError(f'sealed evidence differs: {path}')
        return path
    if not path.resolve().is_relative_to(SPEED.resolve()):
        raise ValueError(f'the fast path writes only under {SPEED}: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    seal(path, payload)
    return path
