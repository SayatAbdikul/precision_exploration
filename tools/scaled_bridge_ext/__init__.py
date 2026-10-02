"""Exact reference models for the scaled-bridge contract 2.3 draft (lane E1, stage 1).

Pure Python integers and Fractions, CPU only, slow on purpose: these are the oracles a later kernel is gated
against, written to be checked by reading. Nothing here is imported by, or changes, tools/scaled_bridge_v2 or
tools/scaled_bridge_fast. Contract text: docs/analysis/scaled-bridge-contract-2.3-draft-2026-10-02.md.

Modules
  rounding   integer rounding of rationals (floor = two's-complement bit drop, RNE) and a generic binary float
  registers  W-bit integer register: saturating or wrap-around, optional dropped low bits
  orders     tap orders (chw, rev, hwc) and the pairwise adder tree
  policies   policy names of contract 2.3 (parser and canonical form)
  reference  one dot product under a policy: sequential, ordered, tree, chunked (two-stage), bias in the register
  blocks     shared-exponent (MX/BFP) block semantics of the simulator, exact block sums and alignment widths
  vectors    stored test vectors of the existing oracles (written once under artifacts/scaled_bridge_ext_v1/)
"""
