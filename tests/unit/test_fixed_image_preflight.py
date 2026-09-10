import hashlib

import pytest

from tools.run.phase2_fixed_images import select_samples, verify_images


def test_selection_is_reproducible_and_contains_eight_frozen_hashes():
    selection, _ = select_samples()
    assert select_samples()[0] == selection
    hashes = [row["sha256"] for row in selection["samples"]]
    assert len(set(hashes)) == 8
    assert hashes == sorted(hashes)


def test_preflight_rejects_missing_changed_and_escaping_payloads(tmp_path):
    row = {"relative_path": "image.JPEG", "sha256": hashlib.sha256(b"frozen image").hexdigest()}
    with pytest.raises(FileNotFoundError):
        verify_images([row], tmp_path)
    path = tmp_path / "image.JPEG"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_images([row], tmp_path)
    path.write_bytes(b"frozen image")
    assert verify_images([row], tmp_path) == [path]
    with pytest.raises(ValueError, match="escapes"):
        verify_images([{**row, "relative_path": "../image.JPEG"}], tmp_path)
