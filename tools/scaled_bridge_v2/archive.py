"""Archive the enrolled implementation by engine source digest (sources, built libraries, compiler manifests).

An archive is self-contained for the engine: run it from the repository root with
    artifacts/scaled_bridge_v2/implementations/<digest>/run.sh <command> [args]
It imports the archived package (as top-level `scaled_bridge_v2`), loads the archived libraries and reads and
writes the same run root as the live sources had at archive time, whatever the live sources are now.
run.sh refuses to run when that run root no longer resolves to the archived digest.
"""
from __future__ import annotations
import shutil
from .common import ROOT, BASE, PACKAGE, digest, file_hash, engine_sources, immutable, unseal
from .native import build

RUN = '''#!/usr/bin/env bash
# Runs the archived scaled bridge v2 implementation {name} from the repository root.
# Usage: run.sh <command> [args]   (same commands as python -m tools.run.scaled_bridge_v2)
HERE="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
cd -- "$HERE/../../../.." || exit 1
PY="${{SCALED_BRIDGE_V2_PYTHON:-.venv-b/bin/python}}"
# The run root also depends on a few tracked files outside the archive. If one of them changed, this archive
# would write under another digest: refuse instead.
NOW="$(PYTHONPATH="$HERE/py" "$PY" -m scaled_bridge_v2 root)" || exit 1
if [ "$(basename -- "$NOW")" != "{name}" ]; then
  echo "archive {name}: the run root resolves to $NOW (a tracked dependency changed); refusing to run" >&2
  exit 1
fi
PYTHONPATH="$HERE/py" exec "$PY" -m scaled_bridge_v2 "$@"
'''


def archive():
    name = digest(engine_sources()); folder = BASE / 'implementations' / name
    if (folder / 'manifest.json').exists():
        saved = unseal(folder / 'manifest.json')
        if any(file_hash(folder / k) != v for k, v in saved['files'].items()):
            raise ValueError('archived implementation drift')
        return folder
    files = [p for p in sorted(PACKAGE.glob('*')) if p.suffix in {'.py', '.cpp', '.cu', '.h'}]
    files += sorted((PACKAGE / 'manifests').glob('*.json'))
    for path in files:
        target = folder / 'py/scaled_bridge_v2' / path.relative_to(PACKAGE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    native = {}
    for backend in ('cpp', 'cuda'):
        library = build(backend); target = folder / 'lib' / backend
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(library, target / 'bridge2.so')
        shutil.copyfile(library.with_suffix('.json'), target / 'bridge2.json')
        native[backend] = unseal(target / 'bridge2.json')
    script = folder / 'run.sh'
    script.write_text(RUN.format(name=name)); script.chmod(0o755)
    listed = sorted(p for p in folder.rglob('*') if p.is_file())
    immutable(folder / 'manifest.json', {
        'engine_sources_digest': name, 'engine_sources': engine_sources(), 'native': native,
        'files': {str(p.relative_to(folder)): file_hash(p) for p in listed},
        'run_root': str((BASE / 'runs' / name).relative_to(ROOT)),
        'usage': 'from the repository root: artifacts/scaled_bridge_v2/implementations/<digest>/run.sh <command> '
                 '[args]; GPU commands go through artifacts/agent_orchestration/gpu_run.sh',
        'not_numeric': 'report.py, archive.py, b2_adapter.py and b2_replay.py are copied for convenience and are '
                       'not part of the engine digest'})
    return folder
