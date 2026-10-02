"""Structural preconditions of the B2 boundary analysis.

``boundaries.analyze`` proves non-negativity from the graph structure and
looks only at FX-node operands of ``add`` and ``mul``.  A literal operand
(``x + (-3.0)``, ``x * -1``) is invisible to it, so such a node would be
marked non-negative wrongly and the unsigned quantizer, which maps a negative
input to code 0 without complaint, would clip silently.  None of the three
frozen classifiers contains such a node; this guard makes that an enforced
precondition before B2 is reused on another graph.

The guard lives outside the numeric sources (``common.NUMERIC_SOURCES``) on
purpose: it never changes a number, it only refuses graphs.
"""
from __future__ import annotations

import operator

import torch


def structural_problems(graph):
    """List the nodes for which the structural non-negativity proof would be unsound."""
    problems = []
    for node in graph.graph.nodes:
        if node.op == "call_function" and node.target in (operator.add, operator.mul):
            operands = list(node.args) + list(node.kwargs.values())
            if len(operands) != 2 or not all(isinstance(x, torch.fx.Node) for x in operands):
                problems.append(f"{node.name}: {node.target.__name__} with a non-node operand {node.args!r}")
    return problems


def check_graph(graph):
    """Raise ``ValueError`` when ``boundaries.analyze`` cannot be trusted on ``graph``."""
    problems = structural_problems(graph)
    if problems:
        raise ValueError("B2 non-negativity analysis is unsound for this graph: " + "; ".join(problems))
    return True
