"""GPU execution of the archived scaled bridge v2 engine (1f75c923 semantics), bit for bit.

Every operation of `scaled_bridge_v2.engine.Engine.run` is reproduced on torch CUDA tensors:
int64 codes and units, binary64 values. Each NumPy expression becomes the same sequence of separately launched
elementwise kernels, so every binary64 operation is one correctly rounded IEEE operation (no FMA contraction
across kernels). Divisions always take a CUDA tensor divisor (torch replaces division by a CPU scalar with a
multiplication by its reciprocal). Multiplications by 2^-shift are exact (normal powers of two, operands exact).
Integer sums (average pool, event counts) are exact in int64. Fail-closed checks that depend on data are
collected as GPU flags and raised, with the archive's messages, after one synchronisation per batch; a failing
batch produces no record either way.
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn.functional as F
from .common import base
from .native import FastNative

_b = base()
from scaled_bridge_v2.common import array_hash, digest                                      # noqa: E402
from scaled_bridge_v2.codebooks import Codebook                                             # noqa: E402
from scaled_bridge_v2.certificates import certify, certify_products, PASSTHROUGH          # noqa: E402
from scaled_bridge_v2.accumulators import resolve                                          # noqa: E402
from scaled_bridge_v2.engine import validate, stable_top5, broadcast_shape, UNARY_2_2     # noqa: E402

DEVICE = torch.device('cuda')


class Book:
    """Codebook constants on the GPU."""
    def __init__(self, book):
        self.book, self.id, self.shift = book, book.id, book.shift
        self.step = float(2.0 ** -book.shift)            # exact power of two
        self.units = torch.from_numpy(book.units).to(DEVICE)
        self.levels = torch.from_numpy(book.levels).to(DEVICE)
        self.bounds = torch.from_numpy(book.bounds).to(DEVICE).contiguous()
        self.ties = torch.from_numpy(book.ties).to(DEVICE)
        self.lo_level, self.hi_level = self.levels[0], self.levels[-1]


class GCoded:
    def __init__(self, units, pos, scale, book):
        self.units, self.pos, self.scale, self.book = units, pos, scale, book


class GRaw:
    def __init__(self, values):
        self.values = values


_SCALARS = {}


def scalar(value):
    """A 0-dim CUDA binary64 constant (cached: creating one copies from the host and synchronises)."""
    if value not in _SCALARS:
        _SCALARS[value] = torch.tensor(value, dtype=torch.float64, device=DEVICE)
    return _SCALARS[value]


SIX = None


def reconstructed(units, scale, book):
    return units.to(torch.float64) * book.step * float(scale)


def real(state):
    return reconstructed(state.units, state.scale, state.book) if isinstance(state, GCoded) else state.values


def window_max(values, attrs):
    """max_pool2d with implicit -inf padding equals the archive's padded sliding-window maximum (2p <= k)."""
    return F.max_pool2d(values, tuple(attrs['kernel_size']), tuple(attrs['stride']), tuple(attrs['padding']))


def hard_gate(r):
    return torch.clamp_max(torch.clamp_min(r + 3.0, 0.0), 6.0)


class FastEngine:
    """Drop-in for `Engine(export, arrays, 'cuda', policy)`; run() returns the archive's records."""

    def __init__(self, export, arrays, policy):
        global SIX
        self.export, self.policy = export, resolve(policy)
        self.nodes = export['nodes']
        validate(self.nodes)
        self.books = {k: Codebook(v) for k, v in export['codebooks'].items()}
        self.gbooks = {k: Book(v) for k, v in self.books.items()}
        self.certificates = certify(export, arrays)
        self.products = certify_products(export)
        self.native = FastNative()
        SIX = scalar(6.0)
        self.mac = {}
        for node in self.nodes:
            if node['op'] not in ('conv', 'linear'):
                continue
            key = node['name']; cert = self.certificates[key]; spec = node['mac']
            self.policy.precondition(cert, self.books[cert['input_codebook']], self.books[cert['weight_codebook']])
            params, info = self.policy.node(cert['product_shift'], cert)
            kind = np.int32 if cert['kernel_operand_bits'] == 32 else np.int64
            w = arrays[spec['weight_units']]
            if node['op'] == 'linear':
                w = w[:, :, None, None]
            scales = np.asarray(spec['weight_scales'], dtype=np.float64)
            bias = np.asarray(arrays[spec['bias']], dtype=np.float64)
            if scales.shape != (w.shape[0],) or bias.shape != (w.shape[0],):
                raise ValueError(f'{key}: weight scale or bias shape does not match the output channels')
            if not (np.isfinite(scales).all() and (scales > 0).all() and np.isfinite(bias).all()):
                raise ValueError(f'{key}: invalid weight scale or bias')
            self.mac[key] = {'cert': cert, 'kind': torch.int32 if kind == np.int32 else torch.int64, 'params': params,
                             'info': info, 'w': torch.from_numpy(np.ascontiguousarray(w, dtype=kind)).to(DEVICE),
                             'scales': torch.from_numpy(scales[None, :, None, None].copy()).to(DEVICE),
                             'bias': torch.from_numpy(bias[None, :, None, None].copy()).to(DEVICE)}
        for node in self.nodes:
            store = node.get('store')
            if store is not None:
                s = np.asarray(store['scale'], dtype=np.float64)
                if not np.isfinite(s).all() or np.any(s <= 0):
                    raise ValueError('nonfinite bridge input/scale')
        self.scale_t = {n['name']: scalar(float(n['store']['scale'])) for n in self.nodes if n.get('store') is not None}

    # -- operators -------------------------------------------------------------------------------------------
    def quantize(self, raw, key, gb, diag):
        y = raw / self.scale_t[key]
        pos = torch.searchsorted(gb.bounds, y, right=False)
        adjacent = torch.clamp_max(pos, len(gb.book.bounds) - 1)
        tie = y == gb.bounds[adjacent]
        pos = pos + (tie & gb.ties[adjacent])
        units = gb.units[pos]
        counts = None
        if diag:
            n = raw.shape[0]
            counts = torch.stack([(y < gb.lo_level).reshape(n, -1).sum(1), (y > gb.hi_level).reshape(n, -1).sum(1),
                                  tie.reshape(n, -1).sum(1), (units == 0).reshape(n, -1).sum(1),
                                  (~torch.isfinite(y)).reshape(n, -1).sum(1)], 1)
        return units, pos, counts

    def _mac(self, node, state, checks):
        key = node['name']; m = self.mac[key]; cert = m['cert']
        if not isinstance(state, GCoded) or state.book.id != cert['input_codebook']:
            raise ValueError(f'{key}: MAC input is not the certified stored code tensor')
        x = state.units
        if node['op'] == 'linear':
            x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        lo1, hi1 = cert['structural_input_range_units']; lo2, hi2 = cert['closure_input_range_units']
        mn, mx = x.min(), x.max()
        checks.append(((mn < lo1) | (mx > hi1), f'{key}: input units outside the certified range'))
        checks.append(((mn < lo2) | (mx > hi2), f'{key}: input units outside the certified closure range'))
        dot, events, bad_native = self.native.run(x.to(m['kind']).contiguous(), m['w'], node['attrs'],
                                                  cert['product_shift'], self.policy, cert['kernel_operand_bits'], m['params'])
        checks.append((bad_native, 'nonfinite native result'))
        n = dot.shape[0]; stats = None; bad = None; used = dot
        if self.policy.kind == 'sat':
            e = events.reshape(n, -1).to(torch.int64) & 0xFFFFFFFF
            stats = torch.stack([(e != 0).sum(1), (e & 0xFFFF).sum(1), (e >> 16).sum(1)], 1)
        elif self.policy.kind == 'float':
            e = events.reshape(n, -1).to(torch.int64) & 0xFFFFFFFF
            finite = torch.isfinite(dot).reshape(n, -1)
            bad = ~finite.all(1)
            flat = dot.reshape(n, -1)
            stats = torch.stack([(~finite).sum(1), e.sum(1), torch.isposinf(flat).sum(1), torch.isneginf(flat).sum(1)], 1)
            used = torch.where(bad.reshape(n, 1, 1, 1), scalar(0.0), dot)
        raw = used * float(state.scale)
        raw = raw * m['scales']
        raw = raw + m['bias']
        if node['op'] == 'linear':
            raw = raw[:, :, 0, 0]
        return raw, dot, stats, bad

    def _value(self, node, args):
        op = node['op']; s = args[0] if args else None
        if op == 'add':
            a, b = real(args[0]), real(args[1])
            if a.shape != b.shape:
                raise ValueError(f'{node["name"]}: residual operands differ in shape')
            return a + b
        if op == 'relu':
            if isinstance(s, GCoded):
                return reconstructed(torch.clamp_min(s.units, 0), s.scale, s.book)
            return torch.clamp_min(s.values, 0.0) + 0.0
        if op in UNARY_2_2:
            if op == 'relu6':
                r = reconstructed(torch.clamp_min(s.units, 0), s.scale, s.book) if isinstance(s, GCoded) \
                    else torch.clamp_min(s.values, 0.0)
                return torch.clamp_max(r, 6.0) + 0.0
            r = real(s)
            if op == 'hardsigmoid':
                return hard_gate(r) / SIX + 0.0
            return (r * hard_gate(r)) / SIX + 0.0
        if op == 'mul':
            a, b = args
            cert = self.products[node['name']]
            shape = broadcast_shape(tuple(a.units.shape if isinstance(a, GCoded) else a.values.shape),
                                    tuple(b.units.shape if isinstance(b, GCoded) else b.values.shape), node['name'])
            if isinstance(a, GCoded) and isinstance(b, GCoded):
                if not cert['stored_operands'] or cert['input_codebooks'] != [a.book.id, b.book.id]:
                    raise ValueError(f'{node["name"]}: tensor-product operands are not the certified code tensors')
                product = (a.units * b.units).to(torch.float64)        # exact: |p| < 2^53 by the product certificate
                value = product * float(2.0 ** -(a.book.shift + b.book.shift)) * float(a.scale)
                value = value * float(b.scale)
            else:
                if cert['stored_operands']:
                    raise ValueError(f'{node["name"]}: tensor-product operands are not the certified code tensors')
                value = real(a) * real(b)
            if tuple(value.shape) != tuple(shape):
                raise ValueError(f'{node["name"]}: tensor-product shape')
            return value + 0.0
        if op == 'maxpool':
            if isinstance(s, GCoded):
                pooled = window_max(s.units.to(torch.float64), node['attrs']).to(torch.int64)
                return reconstructed(pooled, s.scale, s.book)
            return window_max(s.values, node['attrs']) + 0.0
        if op == 'avgpool':
            if not isinstance(s, GCoded):
                raise ValueError('average pooling of an unstored tensor is outside the contract')
            count = s.units.shape[-2] * s.units.shape[-1]
            if count * max(abs(s.book.book.lo), abs(s.book.book.hi)) >= 2**53:
                raise ValueError('average-pool code sum exceeds the exact binary64 grid')
            summed = s.units.sum(dim=(-2, -1), keepdim=True, dtype=torch.int64)
            raw = summed.to(torch.float64) * s.book.step
            raw = raw / scalar(float(count))
            return raw * float(s.scale)
        raise ValueError(f'operator {op} is not admitted by this contract revision')

    # -- one batch -------------------------------------------------------------------------------------------
    def run(self, inputs, *, trace='none'):
        """inputs: float32 array or CPU/CUDA tensor (n, 3, H, W). Returns the archive's records (trace none,
        codes or full); `timing` carries this path's own timings (not part of numerical())."""
        if trace not in ('full', 'codes', 'none'):
            raise ValueError('unknown trace level')
        start = time.perf_counter()
        x = torch.as_tensor(np.asarray(inputs) if not torch.is_tensor(inputs) else inputs)
        n = x.shape[0]
        values = {}; checks = []; macs = []; traced = []; output = None
        for node in self.nodes:
            key, op = node['name'], node['op']
            args = [values[name] for name in node['inputs']]
            if op == 'output':
                output = real(args[0]) + 0.0
                break
            passthrough = op in PASSTHROUGH
            pooled_codes = op == 'maxpool' and node.get('store') is None and isinstance(args[0], GCoded)
            if passthrough or pooled_codes:
                s = args[0]
                if pooled_codes:
                    units = window_max(s.units.to(torch.float64), node['attrs']).to(torch.int64)
                    state = GCoded(units, torch.searchsorted(s.book.units, units), s.scale, s.book)
                    diagnostics = {'code_passthrough': True}
                else:
                    shape = (n, -1) if op == 'flatten' else None
                    if isinstance(s, GCoded):
                        state = GCoded(s.units.reshape(shape) if shape else s.units,
                                       s.pos.reshape(shape) if shape else s.pos, s.scale, s.book)
                    else:
                        state = GRaw(s.values.reshape(shape) if shape else s.values)
                    diagnostics = {'skipped': True}
                values[key] = state
                if trace != 'none':
                    traced.append((key, 'pass', state, None, None, diagnostics))
                continue
            dot = None; bad = None; stats = None
            if op == 'input':
                raw = x.to(DEVICE).to(torch.float64)
            elif op in ('conv', 'linear'):
                raw, dot, stats, bad = self._mac(node, args[0], checks)
                macs.append((key, stats, bad, dot if bad is not None else None, int(dot[0].numel())))
            else:
                raw = self._value(node, args)
            checks.append((~torch.isfinite(raw).all(), f'{key}: nonfinite raw value'))
            store = node.get('store')
            if store is None:
                raw = raw + 0.0
                state = GRaw(raw); counts = None
            else:
                gb = self.gbooks[store['codebook']]
                units, pos, counts = self.quantize(raw, key, gb, trace != 'none')
                state = GCoded(units, pos, float(store['scale']), gb)
            values[key] = state
            if trace != 'none':
                traced.append((key, 'node', state, raw, dot, counts))
        if output is None:
            raise ValueError('missing or nonfinite output')
        # One synchronisation: the fail-closed checks in execution order, then the small host transfers.
        flags = torch.stack([f.reshape(()) for f, _ in checks]).cpu().numpy() if checks else np.zeros(0, bool)
        for flag, (_, message) in zip(flags, checks):
            if flag:
                raise ValueError(message)
        out = output.cpu().numpy()
        if out.ndim != 2 or not np.isfinite(out).all():
            raise ValueError('missing or nonfinite output')
        gpu_seconds = time.perf_counter() - start
        alive = np.ones(n, dtype=bool); failure = [None] * n
        accumulator = [{} for _ in range(n)]; failed_layers = [{} for _ in range(n)]
        alive_at = {}
        for key, stats, bad, dot, elements in macs:
            m = self.mac[key]
            if stats is not None:
                st = stats.cpu().numpy()
                for i in range(n):
                    if not alive[i]:
                        continue
                    if self.policy.kind == 'sat':
                        accumulator[i][key] = {**m['info'], 'elements': elements, 'saturated_elements': int(st[i, 0]),
                                               'high_clamps': int(st[i, 1]), 'low_clamps': int(st[i, 2])}
                    else:
                        accumulator[i][key] = {**m['info'], 'elements': elements, 'nonfinite_elements': int(st[i, 0]),
                                               'nonfinite_steps': int(st[i, 1])}
                if bad is not None:
                    badn = bad.cpu().numpy()
                    hit = np.flatnonzero(badn & alive)
                    if len(hit):
                        d = dot.cpu().numpy()
                    for i in hit:
                        failure[i] = {'kind': 'nonfinite accumulator result', 'node': key, **accumulator[i][key],
                                      'positive_infinite': int(st[i, 2]), 'negative_infinite': int(st[i, 3])}
                        failed_layers[i][key] = {'codes': None, 'state': None, 'raw': None, 'mac': array_hash(d[i:i + 1]),
                                                 'diagnostics': {'nonfinite_accumulator': True}}
                        alive[i] = False
            alive_at[key] = alive.copy()
        layers = [dict() for _ in range(n)]
        if trace != 'none':
            self._trace(traced, trace, n, alive_at, layers)
        for i in range(n):
            layers[i].update(failed_layers[i])
        timing = {'execution': time.perf_counter() - start, 'gpu_and_sync': gpu_seconds, 'batch_images': n}
        records = []
        for i in range(n):
            logits = out[i]
            record = {'layers': layers[i], 'output': array_hash(out[i:i + 1]), 'top5': stable_top5(logits),
                      'top1_tie_count': int(np.sum(logits == logits.max())),
                      'top1_tied_classes': [int(c) for c in np.flatnonzero(logits == logits.max())],
                      'diagnostic_signature': digest({k: v['diagnostics'] for k, v in layers[i].items()}),
                      'timing': timing, 'oracle_dot_checks': 0}
            if self.policy.parameterised:
                record['accumulator'] = accumulator[i]; record['failure'] = failure[i]
                if failure[i] is not None:
                    record.update(output=None, top5=None, top1_tie_count=None, top1_tied_classes=None)
            records.append(record)
        return records

    def _trace(self, traced, trace, n, alive_at, layers):
        """Layer records exactly as the archive's note(): hashes per image of codes, state, raw and dot; an image
        is noted at a node only while it is alive after that node (a failed MAC node is added by the caller)."""
        alive = np.ones(n, dtype=bool)

        def hashes(array):
            return [array_hash(array[i:i + 1]) for i in range(n)] if array is not None else None

        for key, kind, state, raw, dot, extra in traced:
            if key in alive_at:
                alive = alive_at[key]
            codes = state.book.book.codes[state.pos.cpu().numpy()] if isinstance(state, GCoded) else None
            if kind == 'pass':
                entry = {'codes': hashes(codes), 'diagnostics': extra}
                if trace == 'full':
                    entry.update(state=hashes(real(state).cpu().numpy()), raw=None, mac=None)
            else:
                if isinstance(state, GCoded):
                    c = extra.cpu().numpy(); size = int(np.prod(state.units.shape[1:]))
                    diagnostics = [{'elements': size, 'clipped_low': int(c[i, 0]), 'clipped_high': int(c[i, 1]),
                                    'ties': int(c[i, 2]), 'zeros': int(c[i, 3]), 'normalization_overflow': int(c[i, 4])}
                                   for i in range(n)]
                else:
                    diagnostics = {'unstored': True}
                entry = {'codes': hashes(codes), 'diagnostics': diagnostics}
                if trace == 'full':
                    entry.update(state=hashes(real(state).cpu().numpy()), raw=hashes(raw.cpu().numpy()),
                                 mac=hashes(dot.cpu().numpy() if dot is not None else None))
            for i in range(n):
                if alive[i]:
                    layers[i][key] = {k: (v[i] if isinstance(v, list) else v) for k, v in entry.items()}
