from tools.run import accumulator_sweep_run as r


def test_wrapped_predict_matches_only_archived_predict_wrappers():
    base = ['bash', 'artifacts/agent_orchestration/gpu_run.sh', '--min-free-mib', '3000', '--wait', '7200']
    argv = base + [f'{r.ARCHIVE}/run.sh', 'predict', 'resnet18-int8-default-b2', 'sat.w17', 'cuda', '0', '1000']
    assert r.wrapped_predict(argv) == ('resnet18-int8-default-b2', 'sat.w17')
    assert r.wrapped_predict(base + [f'{r.ARCHIVE}/run.sh', 'panel', 'c', 'p']) is None
    assert r.wrapped_predict(base + ['other/run.sh', 'predict', 'c', 'p']) is None
    assert r.wrapped_predict(['.venv-b/bin/python', '-m', 'scaled_bridge_v2', 'predict', 'c', 'p']) is None
    assert r.wrapped_predict(argv[:-5]) is None  # truncated command line
