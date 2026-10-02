"""Lane S1: the fast exact-engine path reproduces sealed archive records bit for bit (GPU; run through gpu_run.sh).

  GPU_LANE=S1 artifacts/agent_orchestration/gpu_run.sh --min-free-mib 3000 .venv-b/bin/python -m pytest -q tests/conformance/test_scaled_bridge_fast.py
Full trace (numerical(): every layer's code, state, raw and dot hash, accumulator statistics, failure, output, top-5,
ties) against the archives' sealed batch-1 gate panels, at batch 1 and 3 (batch invariance), read-only.
"""
import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs the CUDA environment (.venv-b)')
R7 = '7c6344af732d1bf324c07b285921f63393150cdb3875356c86237d934fdc2f92'
R1 = '1f75c9232c8a0202482fe9bce9a46360c7cc0999c5b0bcf7df5ebcc0446fc863'


@pytest.mark.parametrize('case,policy,root', [
    ('resnet18-int8-default-b2', 'wide', R7), ('resnet18-int8-default-b2', 'sat.w18', R7),
    ('mobilenet_v3_large-int8-default-b2', 'wide', R1), ('mobilenet_v3_large-int8-default-b2', 'fp16.x-8', R1),
    ('mobilenet_v2-fp6_e2m3-default-b2', 'sat.struct-2', R1)])
@pytest.mark.parametrize('batch', [1, 3])
def test_full_trace_records_equal_sealed(case, policy, root, batch):
    from tools.scaled_bridge_fast import worker
    from tools.scaled_bridge_fast.common import V2
    result = worker.regress(case, policy, 'cuda', 3, V2 / 'runs' / root, batch=batch)
    assert result['identical'] == 3 and not result['differences']
