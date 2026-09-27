import pytest

from tools.breadth_study import prospective_execution, ternary_extension


def test_admission_adapter_tracks_sources_and_restores_process_hooks_on_failure(monkeypatch):
    original_sources = prospective_execution.sources
    original_verifier = prospective_execution.verify_admission
    monkeypatch.setattr(ternary_extension, 'extra_sources', lambda: {'new-proof.py': 'proof-digest'})
    with pytest.raises(RuntimeError, match='injected'):
        with ternary_extension.execution_adapter() as execution:
            assert execution.sources() == {**original_sources(), 'new-proof.py': 'proof-digest'}
            with pytest.raises(ValueError, match='unrelated'):
                execution.verify_admission({'purpose': 'E2', 'model': 'resnet18', 'format': 'int4'}, {})
            raise RuntimeError('injected')
    assert prospective_execution.sources is original_sources
    assert prospective_execution.verify_admission is original_verifier


def test_ternary_proof_rejects_other_cases_before_admission():
    with pytest.raises(ValueError, match='only the frozen selected ternary'):
        ternary_extension.static_proof({'model': 'mobilenet_v3_large', 'format': 'binary_pm1'})
