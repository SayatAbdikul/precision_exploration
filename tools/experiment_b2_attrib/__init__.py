"""Experiment B2 activation attribution and uniform-precision activation repairs (lane Q4, 2026-10-02).

Diagnostic arms keep chosen activation boundaries wide (FP32) on top of the frozen B2 default; recipe arms
change the activation code of chosen boundaries (affine zero point, per-channel scales).  Nothing here edits
the frozen B2 sources; every arm is built from their functions and checked against sealed matrix cells.
"""
