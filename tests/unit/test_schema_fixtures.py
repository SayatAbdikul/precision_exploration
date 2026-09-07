from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]


def test_public_package_schema_accepts_public_fixture_and_rejects_private_field() -> None:
    schema = json.loads((ROOT / "public/package/schema/quantized-model-package.schema.json").read_text())
    validator = Draft202012Validator(schema)
    valid = json.loads(
        (ROOT / "tests/conformance/fixtures/packages/valid/fp6_e3m2_guard_package.json").read_text()
    )
    invalid = json.loads(
        (ROOT / "tests/conformance/fixtures/packages/invalid/package_with_private_mant_field.json").read_text()
    )
    assert list(validator.iter_errors(valid)) == []
    assert list(validator.iter_errors(invalid))
