"""Stage A2 selection rule of protocol b2-attrib-protocol-v1 (CPU): leave-one-out nodes per format.

Per format: rank the role groups by recovery R_G = top1(wide:role.G) - top1(ref_default) (ties by KL reduction),
take the top 2 groups with more than one node, list their nodes; above 40 nodes keep the 40 with the lowest stored
SQNR in the sealed default cell.  Prints one queue line per job (at most ``per_job`` arms each).
"""
from __future__ import annotations

import json
import sys

from tools.experiment_b.common import ROOT
from . import analysis

PROTOCOL = ROOT / "public/experiments/configs/breadth-study/b2-attrib-protocol-v1.json"


def select(model, fmt, top=2, cap=40):
    groups = json.loads(PROTOCOL.read_text())["groups"][model]["role"]
    result = analysis.attribution(model, fmt, groups)
    ranked = sorted(((g, r) for g, r in result["groups"].items() if r["nodes"] > 1),
                    key=lambda kv: (-kv[1]["recovery_pp"], -kv[1]["kl_reduction"]))[:top]
    sqnr = analysis.cell_sqnr(analysis.cells("default")[(model, fmt)])
    nodes = [n for g, _ in ranked for n in groups[g]]
    if len(nodes) > cap:
        nodes = sorted(nodes, key=lambda n: (sqnr.get(n) is None, sqnr.get(n) or 0.0))[:cap]
    return [g for g, _ in ranked], nodes


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    model, formats, per_job = argv[0], argv[1].split(","), int(argv[2]) if len(argv) > 2 else 30
    for fmt in formats:
        chosen, nodes = select(model, fmt)
        print(f"# {fmt}: groups {chosen}, {len(nodes)} nodes", file=sys.stderr)
        arms = [f"wide:node.{n}" for n in nodes]
        for i in range(0, len(arms), per_job):
            print(f"arms --model {model} --format {fmt} --stage A2 --arms {','.join(arms[i:i + per_job])}")


if __name__ == "__main__":
    main()
