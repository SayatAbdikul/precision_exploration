"""Certificate quantities of one case (read-only from the sealed certificate.json of the archived run root).

Every number here is a function of the certificate alone (weights and codebooks), never of measured images.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIGEST = '7c6344af732d1bf324c07b285921f63393150cdb3875356c86237d934fdc2f92'
RUN_ROOT = ROOT / 'artifacts/scaled_bridge_v2/runs' / DIGEST

# Float accumulators of contract 2.1: largest finite exponent (binary16: 65,504 < 2^16; f21: < 2^128) and the
# smallest normal exponent.
FLOATS = {'fp16': {'emax': 15, 'emin': -14}, 'f21': {'emax': 127, 'emin': -126}}
E_LIMIT = 64  # contract: |e| <= 64


def load_certificate(case, root=RUN_ROOT):
    with open(Path(root) / case / 'certificate.json') as handle:
        return json.load(handle)['payload']['certificates']


def node_table(certificates):
    """Per MAC node: absolute and structural certified widths, magnitude bound, product shift, taps."""
    rows = {}
    for name, c in certificates.items():
        rows[name] = {'abs': int(c['signed_bits_absolute']), 'struct': int(c['signed_bits_structural']),
                      'range': int(c['signed_bits_range']), 'bound_units': int(c['max_abs_prefix_units']),
                      'product_shift': int(c['product_shift']), 'K': int(c['K'])}
    return rows


def magnitude_exponent(rows):
    """ub: every prefix sum of every node satisfies |sum| < 2^ub in code-level units (a*w, no scales)."""
    return max(r['bound_units'].bit_length() - r['product_shift'] for r in rows.values())


def float_exponent(kind, rows):
    """Scale exponent e of the float accumulator, from the certificate only.

    Rule: if the plain format (e = 0) is proven overflow-free (ub <= emax) and every nonzero product and prefix sum
    is a normal number (smallest grid step 2^-shift >= 2^emin for every node), e = 0 (plain policy name: there is
    nothing to place).  Otherwise e = emax - ub, the largest binary-point shift for which the certificate proves
    that no prefix sum can overflow (|sum| * 2^e < 2^emax <= largest finite value).
    """
    spec = FLOATS[kind]; ub = magnitude_exponent(rows)
    smallest = -max(r['product_shift'] for r in rows.values())
    if ub <= spec['emax'] and smallest >= spec['emin']:
        return 0
    e = spec['emax'] - ub
    if abs(e) > E_LIMIT:
        raise ValueError('scale exponent outside the contract limit')
    return e


def float_policy(kind, rows):
    e = float_exponent(kind, rows)
    return kind if e == 0 else f'{kind}.x{e}'


def summary(case, root=RUN_ROOT):
    rows = node_table(load_certificate(case, root))
    return {'case': case, 'W_cert_abs': max(r['abs'] for r in rows.values()),
            'W_cert_struct': max(r['struct'] for r in rows.values()),
            'W_cert_range': max(r['range'] for r in rows.values()),
            'W_node_min_abs': min(r['abs'] for r in rows.values()),
            'ub': magnitude_exponent(rows), 'fp16_policy': float_policy('fp16', rows),
            'f21_policy': float_policy('f21', rows), 'nodes': rows}
