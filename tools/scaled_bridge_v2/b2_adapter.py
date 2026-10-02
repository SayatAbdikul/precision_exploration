"""Adapter from lane L1's repaired-recipe export (`b2-recipe-export-1`) to `scaled-bridge-export-2`.

Nothing is re-quantised. The adapter translates data and verifies it:
  codebooks   FP32 levels -> integer units on the smallest dyadic grid. A signed table must be the accepted
              manifest codebook exactly (levels, codes, midpoint ties). The unsigned integer variant
              (`<format>:unsigned` in B2) becomes its own codebook `<format>.unsigned`.
  boundaries  quantizes=true -> store {codebook, FP32 scale promoted exactly}; quantizes=false -> store null
              (fused into the following ReLU, pass-through, or a max-pool that forwards codes).
  weights     manifest weight codes -> grid integers through the codebook; float32(level) * float32(scale) must
              reproduce B2's FP32 reconstructed weights bit for bit, signed zero included.
  biases      the exported FP32 bias (bias-corrected when the recipe says so) promoted exactly to binary64.
  graph       (contract 2.2) the node rows (name, operator, inputs) must equal the operator rows of the model's
              folded FX graph loaded independently through Experiment B's loader (`export.topology`, which also
              refuses non-global average pooling, ceil-mode max-pooling and unknown modules), and the folded
              graph text must equal the exported one. An in-place activation is accepted only when it is the only
              consumer of its input tensor (looking through identity and flatten aliases), so that in-place
              mutation cannot reach another consumer; its `inplace` attribute is then dropped.
  uniqueness  a case that already has an adapted export (written by any adapter revision) is not adapted again:
              the engine locates exports by case name and refuses two of one case.
This file is not an engine source: the engine reads the written export only.
"""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import numpy as np
from .common import (ROOT, BASE, PACKAGE, EXPORT_SCHEMA, digest, file_hash, unseal, reference, immutable, array_hash,
                     save_npz, logical, case_id)
from .codebooks import Codebook, scalar_codebook

UNSIGNED_VARIANT = 'b2_unsigned_integer_v1'
RETAINED_PREFIX = 128


def adapter_sources():
    return {logical(PACKAGE / name): file_hash(PACKAGE / name) for name in ('b2_adapter.py', 'export.py', 'codebooks.py')}


def fp32(hexdigits):
    """The FP32 value written as eight big-endian hexadecimal digits, as an exact Python float."""
    return float(np.frombuffer(bytes.fromhex(hexdigits), dtype='>f4')[0])


def codebook(entry):
    """Engine codebook data for one B2 codebook entry; fails unless the B2 table is reproduced exactly."""
    base = entry['base_format']
    accepted = scalar_codebook(base)
    if entry['signedness'] == 'signed':
        data = accepted
    else:
        if entry.get('variant') != UNSIGNED_VARIANT:
            raise ValueError('unknown unsigned codebook variant')
        values = [Fraction(float(v)) for v in entry['levels']]
        if any(q.denominator & (q.denominator - 1) for q in values):
            raise ValueError('unsigned levels are not dyadic')
        shift = max(q.denominator.bit_length() - 1 for q in values)
        data = {'id': base + '.unsigned', 'bits': accepted['bits'], 'shift': shift,
                'units': [int(q * (1 << shift)) for q in values], 'codes': [int(c) for c in entry['codes']],
                'choose_upper_tie': [bool(t) for t in entry['choose_upper_tie']],
                'variant': UNSIGNED_VARIANT, 'base_format': base,
                'origin': 'B2 unsigned integer variant: levels 0 .. 2^bits-1, code = level, midpoint ties to the even code',
                'B2_identity_sha256': entry['identity_sha256']}
    book = Codebook(data)
    same = (np.array_equal(book.levels, np.array(entry['levels'], dtype=np.float64))
            and [int(c) for c in book.codes] == [int(c) for c in entry['codes']]
            and [bool(t) for t in book.ties] == [bool(t) for t in entry['choose_upper_tie']]
            and np.array_equal(book.bounds, np.array(entry['boundaries'], dtype=np.float64)))
    if not same:
        raise ValueError(f'B2 codebook {base}/{entry["signedness"]} is not reproduced (levels, codes, ties, midpoints)')
    if entry['signedness'] == 'unsigned' and (
            [int(u) for u in book.units] != list(range(1 << data['bits'])) or data['codes'] != data['units']
            or data['choose_upper_tie'] != [(code + 1) % 2 == 0 for code in range((1 << data['bits']) - 1)]):
        raise ValueError('unsigned variant is not levels 0 .. 2^bits-1 with code = level and ties to the even code')
    return data


def retained_predictions(identity, rows):
    """B2's own sealed predictions for the leading rows, as far as they exist (read-only)."""
    folder = ROOT / 'artifacts/experiment_b2/predictions' / identity
    kept = []
    for index, row in enumerate(rows[:RETAINED_PREFIX]):
        path = folder / (row['sha256'] + '.json')
        if not path.exists():
            break
        record = unseal(path)
        if record['configuration_sha256'] != identity or record['sample'] != row or record['batch_start'] != 8 * (index // 8):
            raise ValueError('retained B2 prediction does not belong to this configuration, sample or batch')
        kept.append({'index': index, 'sha256': row['sha256'], 'top5': record['top5']})
    return kept


ACTIVATIONS = ('relu', 'relu6', 'hardswish', 'hardsigmoid')
ALIASES = ('identity', 'flatten')


def check_graph(payload):
    """Contract 2.2 graph checks: independent FX topology and single-consumer in-place activations."""
    from tools.experiment_b.classifier import configure, load_model
    from .export import topology
    configure('cpu')
    graph, _, original = load_model(payload['model'], 'cpu')
    del original
    rows = topology(graph)
    if [(r['name'], r['op'], r['inputs']) for r in rows] != [(r['name'], r['op'], list(r['inputs'])) for r in payload['nodes']]:
        raise ValueError('B2 node rows differ from the operator rows of the independently loaded folded FX graph')
    if str(graph.graph) != payload['folded_FX_topology']:
        raise ValueError('B2 folded FX topology text differs from the independently loaded graph')
    users = {}
    for row in payload['nodes']:
        for name in row['inputs']:
            users.setdefault(name, []).append(row['name'])
    ops = {row['name']: row['op'] for row in payload['nodes']}
    inputs = {row['name']: row['inputs'] for row in payload['nodes']}

    def consumers(name):
        """Non-alias consumers of the tensor named `name` and of every identity/flatten alias of it."""
        found = []
        for user in users.get(name, []):
            found.extend(consumers(user) if ops[user] in ALIASES else [user])
        return found

    inplace = 0
    for row in payload['nodes']:
        if row['op'] in ACTIVATIONS and row['attrs'].get('inplace'):
            source = row['inputs'][0]
            while ops[source] in ALIASES:
                source = inputs[source][0]
            if consumers(source) != [row['name']]:
                raise ValueError(f'{row["name"]}: in-place activation on a tensor with another consumer')
            inplace += 1
    return {'topology_rows': len(rows), 'inplace_activations_single_consumer': inplace,
            'check': 'node rows equal export.topology(load_model(model)); folded FX text equal; every in-place '
                     'activation is the only consumer of its input tensor and its aliases'}


def adapted_elsewhere(case):
    return sorted(p for p in (BASE / 'exports').glob(f'*/{case}/export.json'))


def export_b2(folder):
    """Adapt one `b2-recipe-export-1` folder. Returns the reference of the written (or existing) export."""
    from tools.experiment_b.common import dataset
    from tools.experiment_b2.export import load_export
    folder = Path(folder).resolve()
    payload, source = load_export(folder)
    recipe = payload['recipe']
    if not recipe['quantize_input']:
        raise ValueError('an unquantised network input has no code grid: not exact in the code domain (contract B2_not_exact)')
    case = case_id(payload['model'], payload['format'], payload['recipe_name'], 'b2')
    path = BASE / 'exports' / digest(adapter_sources()) / case / 'export.json'
    if path.exists():
        return reference(path)
    existing = adapted_elsewhere(case)
    if existing:
        raise ValueError(f'{case} already has an adapted export ({existing[0].parent}); a second one is refused')
    graph_check = check_graph(payload)
    books, ids = {}, {}
    for name, entry in payload['codebooks'].items():
        data = codebook(entry)
        books[data['id']] = data; ids[name] = data['id']
    nodes, arrays, checked_weights = [], {}, {}
    for row in payload['nodes']:
        key, op = row['name'], row['op']
        attrs = dict(row['attrs']); boundary = row.get('boundary')
        if op in ACTIVATIONS:
            attrs.pop('inplace', None)
        if op == 'maxpool' and attrs.pop('ceil_mode'):
            raise ValueError('ceil-mode max-pool is outside the contract')
        store = None
        if boundary is not None and boundary['quantizes']:
            scale = fp32(boundary['scale_fp32_hex'])
            if scale != boundary['scale'] or not scale > 0:
                raise ValueError(f'{key}: activation scale is not the stated FP32 value')
            store = {'codebook': ids[boundary['codebook']], 'scale': scale}
        node = {'name': key, 'op': op, 'inputs': list(row['inputs']), 'attrs': attrs, 'store': store,
                'fx_op': row['fx_op'], 'target': row['target'],
                'B2_boundary': None if boundary is None else {k: boundary[k] for k in ('quantizes', 'reason', 'signedness', 'nonnegative')}}
        if op in ('conv', 'linear'):
            info = payload['weights'][key]; book = Codebook(books[ids[info['codebook']]])
            codes = source[f'{key}.weight_codes']
            table = np.full(int(book.codes.max()) + 1, np.iinfo(np.int64).min, dtype=np.int64)
            table[book.codes] = book.units
            if int(codes.min()) < 0 or int(codes.max()) >= table.size:
                raise ValueError(f'{key}: weight code outside the codebook')
            units = table[codes]
            if (units == np.iinfo(np.int64).min).any():
                raise ValueError(f'{key}: weight code is not a codebook code')
            scales = source[f'{key}.weight_scales']
            if scales.dtype != np.float32 or scales.shape != (units.shape[0],) or list(units.shape) != info['shape']:
                raise ValueError(f'{key}: malformed weight export')
            shaped = scales.reshape((-1,) + (1,) * (units.ndim - 1))
            rebuilt = (np.ldexp(units.astype(np.float32), -book.shift) * shaped).astype(np.float32)
            rebuilt = np.where((units == 0) & np.signbit(source[f'{key}.original_weight']), np.float32(-0.), rebuilt)
            target = source[f'{key}.weight_reconstructed']
            if target.dtype != np.float32 or not np.array_equal(rebuilt.view('u4'), target.view('u4')):
                raise ValueError(f'{key}: weight codes do not reproduce the B2 FP32 reconstruction bit for bit')
            bias = source[f'{key}.bias']
            if bias.dtype != np.float32 or bias.shape != (units.shape[0],):
                raise ValueError(f'{key}: malformed bias export')
            arrays[key + '__weight_units'] = units.astype(np.int64)
            arrays[key + '__bias'] = bias.astype(np.float64)
            checked_weights[key] = array_hash(target)
            node['mac'] = {'weight_codebook': book.id, 'weight_units': key + '__weight_units',
                           'weight_scales': [float(s) for s in scales], 'bias': key + '__bias',
                           'bias_origin': 'B2 FP32 bias constant (bias-corrected when the recipe says so) promoted exactly to binary64',
                           'bias_changed_by_B2': bool(info['bias_changed'])}
        nodes.append(node)
    rows = dataset('imagenet_screen_1k')[1]
    identity = payload['configuration_sha256']
    logical_export = {
        'schema': EXPORT_SCHEMA, 'case': case, 'model': payload['model'], 'format': payload['format'],
        'recipe': payload['recipe_name'], 'recipe_family': 'b2 (repaired quantization recipe, lane L1)',
        'codebooks': books, 'nodes': nodes,
        'array_identities': {k: array_hash(v) for k, v in arrays.items()},
        'provenance': {
            'B2_export': reference(folder / 'export.json'), 'B2_export_version': payload['version'],
            'B2_configuration_sha256': identity, 'B2_recipe_name': payload['recipe_name'], 'B2_recipe': recipe,
            'B2_source_sha256': payload['source_sha256'], 'B2_runtime': payload['runtime'],
            'B2_codebook_ids': ids, 'model_context': payload['model_context'],
            'folded_FX_topology': payload['folded_FX_topology'],
            'B2_reconstructed_weight_sha256': checked_weights,
            'weight_check': 'manifest weight codes reproduce the B2 FP32 reconstructed weights bit for bit, including signed zero',
            'codebook_check': 'levels, codes, midpoint ties and FP32 midpoints of every B2 table equal the engine codebook',
            'graph_check': graph_check,
            'adapter_sources': adapter_sources()},
        'retained': {'ordered_samples_sha256': digest(rows), 'B2_configuration_sha256': identity,
                     'B2_prefix': retained_predictions(identity, rows)},
    }
    constants = save_npz(path.with_name('constants.npz'), arrays)
    immutable(path, {**logical_export, 'constants': constants})
    return reference(path)
