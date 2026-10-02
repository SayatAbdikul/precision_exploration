"""Run an ARCHIVED engine's own `predict` with its outputs redirected into lane S1's folder (read-only on the archive).

Usage (repository root, GPU through gpu_run.sh):
  .venv-b/bin/python -m tools.run.speed_archive_predict DIGEST CASE START STOP BATCH POLICY [POLICY...]
The archived package (artifacts/scaled_bridge_v2/implementations/DIGEST/py) is imported unchanged. Inside this
process only, its run_root() is replaced by artifacts/speed_v1/archive-runs/DIGEST (predictions, certificate and
ledger land there); the archive's gate requirement is checked first against its real run root (read-only).
These files are the archive's own records for comparisons with the fast path; they are never copied into the
archive's run root.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def main():
    digest_, case, start, stop, batch, *policies = sys.argv[1:]
    root = Path.cwd()
    sys.path.insert(0, str(root / 'artifacts/scaled_bridge_v2/implementations' / digest_ / 'py'))
    import scaled_bridge_v2.common as common
    import scaled_bridge_v2.worker as worker
    if common.digest(common.engine_sources()) != digest_ or Path(common.PACKAGE) != root / 'artifacts/scaled_bridge_v2/implementations' / digest_ / 'py/scaled_bridge_v2':
        raise SystemExit('archive does not resolve to its digest')
    target = root / 'artifacts/speed_v1/archive-runs' / digest_
    for policy in policies:
        if hasattr(worker, 'require_gates'):
            worker.require_gates(case, policy)          # real run root, read-only
    redirected = lambda: target                          # noqa: E731
    common.run_root = redirected
    worker.run_root = redirected
    if hasattr(worker, 'require_gates'):
        worker.require_gates = lambda case, policy: None
    for policy in policies:
        r = worker.predict(case, policy, 'cuda', int(start), int(stop), int(batch))
        print(json.dumps({k: r[k] for k in ('case', 'policy', 'start', 'stop', 'batch', 'execution_seconds')}), flush=True)


if __name__ == '__main__':
    main()
