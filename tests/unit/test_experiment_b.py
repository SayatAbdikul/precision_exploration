import json

import numpy as np
import pytest

from tools.experiment_b.common import inventory, seal, unseal
from tools.experiment_b.quantizer import numpy_qdq, table


def test_coverage_is_complete_and_unsupported_paths_are_explicit():
    rows = inventory()['configurations']
    assert len(rows) == 200
    assert len({(r['model'], r['format']) for r in rows}) == 100
    assert sum(r['blocked_reason'] is None for r in rows) == 126
    assert all(r['blocked_reason'] for r in rows if r['model'] == 'yolov8n')
    assert all(r['blocked_reason'] for r in rows if r['format'].startswith(('mx', 'bfp')))


def test_signed_integer_ties_and_clipping():
    x = np.array([-100, -1.5, -.5, 0, .5, 1.5, 2.5, 100], dtype=np.float32)
    np.testing.assert_array_equal(numpy_qdq(x, 'int4', 1), [-8, -2, 0, 0, 0, 2, 2, 7])


def test_codebooks_do_not_replace_shared_blocks_with_scalar_lookup():
    with pytest.raises(ValueError, match='shared-block'):
        table('mxfp4_e2m1')


def test_observer_covers_channels_missed_by_old_flat_sampler():
    torch = pytest.importorskip('torch')
    from tools.experiment_b.classifier import ObservingInterpreter, sample_indices
    channels = set()
    for image in range(8):
        ch, _ = sample_indices(1280, 49, image)
        channels.update(ch.tolist())
    assert channels == set(range(1280))
    # Detection-head coordinate channels 0..3 all receive observations.
    ch, _ = sample_indices(84, 8400, 0)
    assert {0, 1, 2, 3}.issubset(set(ch))
    graph = torch.fx.symbolic_trace(torch.nn.ReLU())
    # Functional ReLU isn't a production allowed node, but the observer's
    # channel sampler itself should retain a localized outlier in maxabs.
    collector = ObservingInterpreter(graph)
    x = torch.zeros(2, 1280, 7, 7)
    x[0, 1234, 2, 3] = 9000
    collector.run(x)
    assert max(collector.maxima.values()) == 9000


def test_corrupt_or_mismatched_checkpoint_cannot_resume(tmp_path):
    from tools.experiment_b.validation import verify_prediction
    p = tmp_path/'row.json'
    row = {'sha256': 'sample', 'label': '1'}
    seal(p, {'configuration_sha256': 'config', 'sample': row, 'top5': [1, 2, 3, 4, 5]})
    assert verify_prediction(p, 'config', row)['top5'][0] == 1
    with pytest.raises(ValueError, match='identity'):
        verify_prediction(p, 'another_recipe', row)
    document = json.loads(p.read_text())
    document['payload']['top5'][0] = 42
    p.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='integrity'):
        unseal(p)


def test_all_scalar_families_match_declared_cpu_reference():
    pytest.importorskip('torch')
    from tools.experiment_b.classifier import configure
    from tools.experiment_b.validation import validate_quantizers
    configure('cpu')
    result = validate_quantizers('cpu')
    assert result['status'] == 'passed'
    assert len(result['formats']) == 21


def test_prediction_resume_reuses_full_prefix_and_replays_partial_batch(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from tools.experiment_b import runner, classifier
    from tools.experiment_b.common import file_hash
    payload = tmp_path/'images'
    payload.mkdir()
    rows = []
    for i in range(128):
        p = payload/f'{i}.bin'
        p.write_bytes(str(i).encode())
        rows.append({'relative_path': p.name, 'sha256': file_hash(p), 'label': str(i)})
    monkeypatch.setattr(runner, 'BASE', tmp_path/'evidence')
    monkeypatch.setattr(runner, 'dataset', lambda name: ({}, rows, payload))
    monkeypatch.setattr(classifier, 'image_batch', lambda batch, *args: torch.tensor([int(r['label']) for r in batch]))
    class Model:
        def __init__(self):
            self.batches = []
        def __call__(self, labels):
            self.batches.append(labels.tolist())
            output = torch.zeros(len(labels), 1000)
            output[torch.arange(len(labels)), labels] = 100
            return output
    model = Model()
    result, timing = runner.evaluate(model, None, 'test', 'recipe1', 'cpu', 128, lambda x: None)
    assert timing['computed_images_this_invocation'] == 128
    assert len(model.batches) == 16
    again, timing = runner.evaluate(model, None, 'test', 'recipe1', 'cpu', 128, lambda x: None)
    assert again == result
    assert timing['computed_images_this_invocation'] == 0
    assert len(model.batches) == 16
    # Simulate a stop while committing image records in one batch.
    (runner.BASE/'predictions'/'recipe1'/(rows[9]['sha256']+'.json')).unlink()
    replay, timing = runner.evaluate(model, None, 'test', 'recipe1', 'cpu', 128, lambda x: None)
    assert timing['computed_images_this_invocation'] == 8
    assert model.batches[-1] == list(range(8, 16))
    assert [p['top5'] for p in replay] == [p['top5'] for p in result]
    _, timing = runner.evaluate(model, None, 'test', 'recipe2', 'cpu', 128, lambda x: None)
    assert timing['computed_images_this_invocation'] == 128
