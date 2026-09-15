from tools.phase3.common import checked
from tools.phase3.provenance import source_reference


def test_analysis_provenance_survives_later_source_edits(tmp_path):
    path = tmp_path / "analysis.py"
    path.write_text("VERSION = 1\n")
    original = source_reference(path, tmp_path)
    assert source_reference(path, tmp_path) == original
    path.write_text("VERSION = 2\n")
    updated = source_reference(path, tmp_path)
    assert updated["sha256"] != original["sha256"]
    assert checked(original, tmp_path).read_text() == "VERSION = 1\n"
    assert checked(updated, tmp_path).read_text() == "VERSION = 2\n"
