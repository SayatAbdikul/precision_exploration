"""Read sealed `predict` files of the archived 1f75c923 run root (read-only) and assemble image ranges.

Wraps lane L8's tools/accumulator_sweep_v1/load.py (file naming, tiling) with this lane's digest and run root.
"""
import hashlib
import json
from pathlib import Path

from tools.accumulator_sweep_v1.load import prediction_files as _files
from tools.experiment_b.common import unseal

from .certs import RUN_ROOT, DIGEST

__all__ = ['covering', 'prediction_files', 'prefix_end', 'assemble']


def covering(files, start, stop):
    """Tiles of [start, stop) from sealed files [(a, b, path)], greedy from start preferring the file that reaches
    furthest; a file may start before `start` or end after `stop` (only its records inside [start, stop) are used,
    e.g. the first 128 images of a 1k file).  Returns [(a, b, path, file_start)] or None if a gap remains."""
    chosen, at = [], start
    while at < stop:
        options = [f for f in files if f[0] <= at < f[1]]
        if not options:
            return None
        best = max(options, key=lambda f: f[1])
        chosen.append((at, min(best[1], stop), best[2], best[0])); at = min(best[1], stop)
    return chosen


def prediction_files(case, policy, backend='cuda', root=RUN_ROOT):
    return _files(case, policy, backend, root)


def prefix_end(case, policy, stop, root=RUN_ROOT):
    """End of the longest contiguous tiling of sealed files from image 0 (at most stop)."""
    files = prediction_files(case, policy, root=root)
    best = 0
    for _, b, _ in files:
        if best < b <= stop and covering(files, 0, b) is not None:  # whole files only
            best = b
    return best


def assemble(case, policy, start, stop, backend='cuda', root=RUN_ROOT, check=True):
    """Image records [start, stop) of one case and policy, with the node constants and the file references."""
    tiles = covering(prediction_files(case, policy, backend, root), start, stop)
    if tiles is None:
        return None
    images, nodes, refs, seconds = [], None, [], 0.0
    for a, b, path, f0 in tiles:
        p = unseal(path) if check else json.load(open(path))['payload']
        if p['case'] != case or p['policy'] != policy or p['engine_sources'] != DIGEST or p['start'] != f0 or p['stop'] < b:
            raise ValueError(f'prediction file does not match its name {path}')
        if nodes is None or not nodes:
            nodes = p['nodes'] if nodes is None or p['nodes'] else nodes
        elif p['nodes'] and nodes != p['nodes']:
            raise ValueError('node constants differ between files')
        images.extend(p['images'][a - p['start']:b - p['start']]); seconds += p['execution_seconds']
        refs.append({'path': str(Path(path).relative_to(RUN_ROOT.parents[3])),
                     'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(), 'start': a, 'stop': b,
                     'execution_seconds': p['execution_seconds'], 'batch': p['batch']})
    return {'images': images, 'nodes': nodes, 'files': refs, 'execution_seconds': seconds}
