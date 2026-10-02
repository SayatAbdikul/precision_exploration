"""Read sealed `predict` files of the archived run root (read-only) and assemble image ranges."""
import hashlib
import json
import re
from pathlib import Path

from .certs import RUN_ROOT, DIGEST

NAME = re.compile(r'^(?P<policy>.+)-(?P<backend>cuda|cpp)-(?P<start>\d{5})-(?P<stop>\d{5})\.json$')


from tools.experiment_b.common import unseal  # the project's seal check (digest of the canonical payload)


def prediction_files(case, policy, backend='cuda', root=RUN_ROOT):
    folder = Path(root) / case / 'predictions'
    found = []
    for path in sorted(folder.glob('*.json')) if folder.exists() else []:
        m = NAME.match(path.name)
        if m and m['policy'] == policy and m['backend'] == backend:
            found.append((int(m['start']), int(m['stop']), path))
    return found


def covering(files, start, stop):
    """Choose files that tile [start, stop) exactly: greedy from start, preferring the longest file."""
    chosen, at = [], start
    while at < stop:
        options = [f for f in files if f[0] == at and f[1] <= stop]
        if not options:
            return None
        best = max(options, key=lambda f: f[1])
        chosen.append(best); at = best[1]
    return chosen


def assemble(case, policy, start, stop, backend='cuda', root=RUN_ROOT, check=True):
    """Image records [start, stop) of one case and policy, with the node constants and the file references."""
    tiles = covering(prediction_files(case, policy, backend, root), start, stop)
    if tiles is None:
        return None
    images, nodes, refs, seconds = [], None, [], 0.0
    for a, b, path in tiles:
        p = unseal(path) if check else json.load(open(path))['payload']
        if p['case'] != case or p['policy'] != policy or p['engine_sources'] != DIGEST or p['start'] != a or p['stop'] != b:
            raise ValueError(f'prediction file does not match its name {path}')
        if nodes is None:
            nodes = p['nodes']
        elif nodes != p['nodes']:
            raise ValueError('node constants differ between files')
        images.extend(p['images']); seconds += p['execution_seconds']
        refs.append({'path': str(path),
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'start': a, 'stop': b,
                     'execution_seconds': p['execution_seconds'], 'batch': p['batch']})
    return {'images': images, 'nodes': nodes, 'files': refs, 'execution_seconds': seconds}
