from __future__ import annotations

import json
from pathlib import Path

import pytest

from public.experiments.registry.identity import IdentityError, canonical_json_bytes, experiment_sha256, repository_state, validate_experiment


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "tests" / "conformance" / "fixtures" / "canonical" / "experiment_a_fp6_e3m2.canonical.json"
HASH_FILE = ROOT / "tests" / "conformance" / "fixtures" / "canonical" / "experiment_a_fp6_e3m2.sha256"


def test_phase0_golden_identity() -> None:
    configuration = json.loads(CANONICAL.read_text())
    assert canonical_json_bytes(configuration) == CANONICAL.read_bytes()
    assert experiment_sha256(configuration) == HASH_FILE.read_text().split()[0]


def test_key_order_does_not_change_identity() -> None:
    assert experiment_sha256({"b": 2, "a": 1}) == experiment_sha256({"a": 1, "b": 2})


def test_array_order_and_semantic_mutation_change_identity() -> None:
    assert experiment_sha256({"values": [1, 2]}) != experiment_sha256({"values": [2, 1]})
    assert experiment_sha256({"value": 1}) != experiment_sha256({"value": 2})


def test_nonportable_list_path_is_rejected() -> None:
    config_path = ROOT / "tests" / "conformance" / "fixtures" / "experiments" / "valid" / "experiment_a_fp6_e3m2.json"
    configuration = json.loads(config_path.read_text())
    configuration["dataset"]["calibration"]["path"] = "/host/imagenet.txt"
    with pytest.raises(IdentityError, match="portable"):
        validate_experiment(configuration, verify_lists=False)


def test_invalid_phase0_experiment_fixtures_are_rejected() -> None:
    folder = ROOT / "tests" / "conformance" / "fixtures" / "experiments" / "invalid"
    for path in folder.glob("*.json"):
        with pytest.raises(IdentityError):
            validate_experiment(json.loads(path.read_text()), verify_lists=False)


def test_repository_state_captures_revision_and_dirty_state() -> None:
    state = repository_state(ROOT)
    assert len(state["git_revision"]) == 40
    assert state["git_dirty"] in {"true", "false"}
    assert state["python"]
