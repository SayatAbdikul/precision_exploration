"""Certificate quantities and node kinds of the MobileNet cases (read-only, archived run root 1f75c923...).

Every number here is a function of the sealed certificate.json and the adapted export (weights, codebooks, graph),
never of measured images.  The width rules are those of lane L8 (tools/accumulator_sweep_v1/certs.py), imported
and pointed at this run root.
"""
import json
import math
from pathlib import Path

from tools.accumulator_sweep_v1 import certs as v1

ROOT = v1.ROOT
DIGEST = '1f75c9232c8a0202482fe9bce9a46360c7cc0999c5b0bcf7df5ebcc0446fc863'
RUN_ROOT = ROOT / 'artifacts/scaled_bridge_v2/runs' / DIGEST
ARCHIVE = f'artifacts/scaled_bridge_v2/implementations/{DIGEST}'

NETWORKS = ('mobilenet_v2', 'mobilenet_v3_large')
FORMATS = ('int8', 'int6', 'fp6_e2m3', 'fp7_e3m3', 'fp8_e4m3fn')
CASES = tuple(f'{n}-{f}-default-b2' for n in NETWORKS for f in FORMATS)
STRESS = ('mobilenet_v3_large-int6-default-b2',)  # collapses in the B2 simulator itself (21 % Top-1 at 1k)
INTEGER = {'int8', 'int6'}

KINDS = ('stem', 'depthwise', 'pointwise_expand', 'pointwise_project', 'se_reduce', 'se_expand', 'classifier', 'conv')
ACTIVATIONS = {'relu', 'relu6', 'hardswish', 'hardsigmoid'}


def case_format(case):
    return case.split('-')[1]


def case_network(case):
    return case.split('-')[0]


def summary(case, root=RUN_ROOT):
    """L8's certificate summary (W_cert_abs/struct, ub, the fp16 and f21 rule policies, node table)."""
    return v1.summary(case, root=root)


def cli_fp16_policy(case, root=RUN_ROOT):
    """The archived engine's own `fp16-policy CASE` rule (cli.py), recomputed: fp16 if 15 - ub >= 0 else fp16.x<15-ub>."""
    rows = v1.node_table(v1.load_certificate(case, root))
    ub = v1.magnitude_exponent(rows)
    return 'fp16' if 15 - ub >= 0 else f'fp16.x{15 - ub}'


def load_export(case):
    paths = sorted((ROOT / 'artifacts/scaled_bridge_v2/exports').glob(f'*/{case}/export.json'))
    if len(paths) != 1:
        raise ValueError(f'{case}: expected one adapted export, found {len(paths)}')
    with open(paths[0]) as handle:
        return json.load(handle)['payload']


def mac_kind(nodes, name, cert):
    """Kind of a MAC node.  Lane L2's classification (tools/scaled_bridge_v2/report.py mac_kind: stem, depthwise,
    se_reduce, se_expand, pointwise, conv, classifier) with pointwise split by what consumes the output:
    pointwise_expand when the first non-identity consumer is an activation (ReLU, ReLU6, hard-swish; this includes
    the last 1x1 convolution before the pool), pointwise_project otherwise (linear bottleneck projection)."""
    by = {n['name']: n for n in nodes}
    node = by[name]
    if node['op'] == 'linear':
        return 'classifier'
    src = by[node['inputs'][0]]
    while src['op'] in ('identity', 'flatten'):
        src = by[src['inputs'][0]]
    if src['op'] == 'input':
        return 'stem'
    if cert['groups'] > 1:
        return 'depthwise'
    if src['op'] == 'avgpool':
        return 'se_reduce'
    if any(n['op'] == 'hardsigmoid' and name in n['inputs'] for n in nodes):
        return 'se_expand'
    if list(cert['shape'][2:]) != [1, 1]:
        return 'conv'
    consumers = [n for n in nodes if name in n['inputs']]
    while len(consumers) == 1 and consumers[0]['op'] == 'identity':
        consumers = [n for n in nodes if consumers[0]['name'] in n['inputs']]
    if consumers and all(n['op'] in ACTIVATIONS for n in consumers):
        return 'pointwise_expand'
    return 'pointwise_project'


def node_kinds(case, root=RUN_ROOT):
    """{MAC node: kind} for one case (certificate nodes, export graph)."""
    certificates = v1.load_certificate(case, root)
    nodes = load_export(case)['nodes']
    return {name: mac_kind(nodes, c['node'], c) for name, c in certificates.items()}


def widths_by_kind(case, root=RUN_ROOT):
    """Per kind: number of nodes, K values and the largest absolute / structural certified width."""
    rows = v1.node_table(v1.load_certificate(case, root))
    kinds = node_kinds(case, root)
    out = {}
    for name, r in rows.items():
        k = out.setdefault(kinds[name], {'nodes': 0, 'K': set(), 'abs': 0, 'struct': 0})
        k['nodes'] += 1; k['K'].add(r['K']); k['abs'] = max(k['abs'], r['abs']); k['struct'] = max(k['struct'], r['struct'])
    return {k: dict(v, K=sorted(v['K'])) for k, v in out.items()}


# [Agg24] closed-form accumulator width (related-work audit), as lane L8 used it: integer ra + rb + ceil(log2 n) + 1;
# minifloat ExMy 2^ea + ma + 2^eb + mb + ceil(log2 n) - 1.  The sign-bit convention is assumed (L8's caveat).
FORMULA_OPERANDS = {'int8': ('int', 8), 'int6': ('int', 6), 'fp6_e2m3': ('fp', 2, 3), 'fp7_e3m3': ('fp', 3, 3),
                    'fp8_e4m3fn': ('fp', 4, 3)}


def published_width(fmt, n):
    spec = FORMULA_OPERANDS.get(fmt)
    if spec is None:
        return None
    log = math.ceil(math.log2(n))
    if spec[0] == 'int':
        return 2 * spec[1] + log + 1
    return 2 * (2 ** spec[1] + spec[2]) + log - 1


def published_network_width(case, root=RUN_ROOT):
    """Largest per-node [Agg24] width over the MAC nodes (n = K of the node); the formula grows with n only."""
    rows = v1.node_table(v1.load_certificate(case, root))
    return published_width(case_format(case), max(r['K'] for r in rows.values()))
