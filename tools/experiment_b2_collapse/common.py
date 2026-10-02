"""Paths and source identity of lane Q3."""
from __future__ import annotations

from tools.experiment_b.common import ROOT, file_hash

BASE = ROOT / "artifacts/experiment_b2_collapse"
PROTOCOL_FILE = ROOT / "public/experiments/configs/breadth-study/b2-collapse-protocol-v1.json"
SOURCES = ("correct.py", "build.py", "evaluate.py", "common.py")


def own_sources():
    folder = ROOT / "tools/experiment_b2_collapse"
    return {f"tools/experiment_b2_collapse/{name}": file_hash(folder / name) for name in SOURCES}
