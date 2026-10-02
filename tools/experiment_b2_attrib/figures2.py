"""Lane Q4 v2 figures (agent r6, protocol addendum 8) from results/summaries/b2-attrib-v1/v2/ (written once).

    PYTHONPATH=. .venv/bin/python -m tools.experiment_b2_attrib.figures2 [--out DIR]

Redraws the two v1 figures that the review changed, with the v1 drawing code of ``figures.py``:
* b2-attrib-mbv3-repairs-v2: the ``pcf`` arms (border-column sample, review B1) are left out; the ``pcu`` arms are in;
* b2-attrib-mbv3-leave-one-out-v2: from the v2 leave-one-out (review B3: no ``+fixbc`` arm).
Writes results/figures/b2-attrib-<name>-v2.{png,pdf}; refuses to overwrite.  The v1 figures stay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.experiment_b.common import ROOT
from . import figures

SUMMARY = ROOT / "results/summaries/b2-attrib-v1/v2"
OUT = ROOT / "results/figures"


def save(fig, out, name):
    for ext in ("png", "pdf"):
        path = out / f"b2-attrib-{name}-v2.{ext}"
        if path.exists():
            raise SystemExit(f"{path} exists: figures are written once (use a new version)")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    figures.plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)
    s = {p.stem: json.loads(p.read_text()) for p in SUMMARY.glob("*.json")}
    s["b"] = {fmt: ({k: v for k, v in table.items() if "pcf:" not in k} if table else table) for fmt, table in s["b"].items()}
    figures.save = save
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figures.repairs_figure(s, out)
    figures.loo_figure(s, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
