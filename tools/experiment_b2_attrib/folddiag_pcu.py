"""Addendum-6 fold diagnostic rerun with ``pcu`` scales (lane Q4 r6, protocol addendum 8; CPU only).

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. nice -n 10 .venv-b/bin/python -m tools.experiment_b2_attrib.folddiag_pcu

Runs ``folddiag.main`` unchanged except that, inside this process only, the per-channel scales come from
``pcunbiased.channel_cache`` (unbiased random sample) instead of ``pcfull.channel_cache`` (border-column sample, review
finding B1), and the output names addendum 8.  Output (written once):
artifacts/experiment_b2_attrib/folddiag/fold-diagnostic-v2-pcu.json; the v1 file stays.
"""
from __future__ import annotations

import json
import types

from tools.experiment_b.common import ROOT, file_hash
from . import folddiag, pcunbiased

OUT = ROOT / "artifacts/experiment_b2_attrib/folddiag/fold-diagnostic-v2-pcu.json"
ADDENDUM = ROOT / "public/experiments/configs/breadth-study/b2-attrib-protocol-v1-addendum-8.json"


def main(argv=None):
    if OUT.exists():
        raise SystemExit(f"{OUT} exists: written once")
    folddiag.pcfull = types.SimpleNamespace(channel_cache=pcunbiased.channel_cache)
    folddiag.PROTOCOL = ADDENDUM
    code = folddiag.main(["--out", str(OUT)] + list(argv or []))
    data = json.loads(OUT.read_text())
    data["per_channel_scales"] = {"spec": pcunbiased.SPEC,
                                  "source_sha256": {"tools/experiment_b2_attrib/pcunbiased.py": file_hash(ROOT / "tools/experiment_b2_attrib/pcunbiased.py"),
                                                    "tools/experiment_b2_attrib/folddiag_pcu.py": file_hash(ROOT / "tools/experiment_b2_attrib/folddiag_pcu.py")},
                                  "note": "folddiag.py unchanged; pcfull.channel_cache replaced by pcunbiased.channel_cache in this process"}
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
