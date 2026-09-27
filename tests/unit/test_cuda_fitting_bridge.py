from pathlib import Path

import numpy as np
import pytest

from public.inference.tensor import Encoding, Tensor
from tools.breadth_study import cuda_fitting


def setup_fit(tmp_path, monkeypatch, codes):
    weights = Tensor((2, 2), (0, 0, 0, 0), Encoding('int4', (1, 1), axis=0))
    graph = {'constants': {'w': weights.document()}, 'provenance': {},
             'inputs': {'x': weights.encoding.document()}, 'nodes': [{'name': 'fc'}]}
    source = np.array([[.49, .49], [-.49, -.49]], dtype=np.float32)
    data, output = tmp_path / 'data.npz', tmp_path / 'codes.npy'
    np.savez(data, weights=source)
    np.save(output, codes, allow_pickle=False)
    ref = lambda p: {'path': str(p)}
    path = tmp_path / 'plan.json'
    layer = {'node': 'fc', 'constant': 'w', 'groups': 1, 'data': ref(data)}
    plan = {'bridge_sources': {}, 'guard_sha256': 'guard', 'strict_prepared': ref('prepared'),
            'strict_graph': ref('graph'), 'layers': [layer], 'calibration': {}}
    result = {'plan': ref(path), 'codes': ref(output), 'fit': {}}
    monkeypatch.setattr(cuda_fitting, 'unseal', lambda p: plan if Path(p) == path else result)
    monkeypatch.setattr(cuda_fitting, 'read', lambda p: graph if str(p) == 'graph' else {'graph': ref('graph')})
    monkeypatch.setattr(cuda_fitting, 'reference', ref)
    monkeypatch.setattr(cuda_fitting, 'checked', lambda r: Path(r['path']))
    monkeypatch.setattr(cuda_fitting, 'bridge_sources', lambda: {})
    monkeypatch.setattr(cuda_fitting, 'verify_guard', lambda: 'guard')
    monkeypatch.setattr(cuda_fitting, 'commit_graph', lambda plan, work, prepared, changed, fit: changed)
    return path, graph


def test_gpu_bridge_keeps_encoding_and_graph_and_accepts_only_discrete_choices(tmp_path, monkeypatch):
    codes = np.array([[[1, 0], [15, 0]]], dtype=np.uint8)
    path, graph = setup_fit(tmp_path, monkeypatch, codes)
    changed = cuda_fitting.finish(path)
    tensor = Tensor.from_document(changed['constants']['w'])
    assert tensor.codes == (1, 0, 15, 0)
    assert tensor.encoding == Tensor.from_document(graph['constants']['w']).encoding
    assert changed['inputs'] == graph['inputs'] and changed['nodes'] == graph['nodes']


@pytest.mark.parametrize('codes,reason', [
    (np.array([[[7, 0], [15, 0]]], dtype=np.uint8), 'floor/ceil'),
    (np.array([[[16, 0], [15, 0]]], dtype=np.uint8), 'geometry or dtype'),
    (np.array([[1, 0], [15, 0]], dtype=np.uint8), 'geometry or dtype'),
])
def test_gpu_bridge_rejects_invalid_fitted_weights(tmp_path, monkeypatch, codes, reason):
    path, _ = setup_fit(tmp_path, monkeypatch, codes)
    with pytest.raises(ValueError, match=reason):
        cuda_fitting.finish(path)
