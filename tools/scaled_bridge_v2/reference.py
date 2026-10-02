"""Independent whole-graph witness for the exact (wide) arm.

It differs from the engine in every executed component: the original FX graph
is traversed by torch.fx.Interpreter, stores use torch.bucketize, and each
code-domain dot is computed by the framework's binary64 convolution on
operands split into 16-bit signed limbs. Every limb convolution is an exact
integer computation (all partial sums are below 2^53 in any order); the limb
results are recombined in Python integers and converted with Python's
correctly rounded int -> float. No native bridge kernel is involved.

Contract 2.1 graphs: a boundary with store=null carries a binary64 tensor
(fused boundary) and a max-pool with store=null over stored codes forwards
codes (framework max-pool on the level integers, looked up in the codebook).
"""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from .common import array_hash, digest
from .codebooks import Codebook
from .engine import stable_top5

LIMB = 16


def limbs(units):
    """Signed base-2^16 digits of an integer tensor, least significant first."""
    units = np.asarray(units, dtype=np.int64)
    sign = np.sign(units); magnitude = np.abs(units); parts = []
    while True:
        parts.append(torch.from_numpy((sign * (magnitude & ((1 << LIMB) - 1))).astype(np.float64)))
        magnitude = magnitude >> LIMB
        if not magnitude.any():
            return parts


def exact_dot(x_units, w_units, conv):
    """RNE64 of the exact integer convolution, as a binary64 numpy array."""
    xs, ws = limbs(x_units), limbs(w_units)
    k = int(np.prod(w_units.shape[1:]))
    if k * (1 << (2 * LIMB)) >= 2**53:
        raise ValueError('limb convolution would not be exact')
    if len(xs) == 1 and len(ws) == 1:
        return conv(xs[0], ws[0]).numpy()
    total = None
    for i, a in enumerate(xs):
        for j, b in enumerate(ws):
            part = conv(a, b).numpy()
            if not np.array_equal(part, np.rint(part)):
                raise ValueError('limb convolution is not integral')
            part = part.astype(np.int64).astype(object) * (1 << (LIMB * (i + j)))
            total = part if total is None else total + part
    return np.frompyfunc(float, 1, 1)(total).astype(np.float64)


class FXReference(torch.fx.Interpreter):
    """state[key]: (book, scale, units, codes) for stored codes, a binary64 tensor for an unstored value."""
    def __init__(self, graph, export, arrays):
        super().__init__(graph)
        self.ex, self.arrays = export, arrays
        self.nodes = {x['name']: x for x in export['nodes']}
        self.books = {k: Codebook(v) for k, v in export['codebooks'].items()}
        self.state = {}; self.trace = {}

    def stored(self, key):
        value = self.state[key]
        if not isinstance(value, tuple):
            raise ValueError('reference: this operator needs a stored code tensor')
        return value

    def real(self, key):
        value = self.state[key]
        if not isinstance(value, tuple):
            return value
        book, scale, units, _ = value
        return (units * float(2. ** -book.shift)) * float(scale)

    def note(self, key, raw, mac, diagnostics):
        value = self.state[key]
        self.trace[key] = {'codes': array_hash(value[3]) if isinstance(value, tuple) else None,
                           'state': array_hash(self.real(key).numpy()),
                           'raw': array_hash(raw.numpy()) if raw is not None else None,
                           'mac': array_hash(mac.numpy()) if mac is not None else None, 'diagnostics': diagnostics}

    def run_node(self, node):
        spec = self.nodes[node.name]; key = node.name; op = spec['op']; inputs = spec['inputs']; mac = None
        source = self.state[inputs[0]] if inputs and op != 'output' else None
        coded = isinstance(source, tuple)
        if op == 'input':
            raw = next(self.args_iter).double()
        elif op == 'output':
            return self.real(inputs[0]) + 0.
        elif op in ('identity', 'flatten'):
            if coded:
                book, scale, units, codes = source
                if op == 'flatten':
                    units = units.flatten(1); codes = codes.reshape(units.shape)
                self.state[key] = (book, scale, units, codes)
            else:
                self.state[key] = source.flatten(1) if op == 'flatten' else source
            self.note(key, None, None, {'skipped': True})
            return None
        elif op == 'maxpool' and spec['store'] is None and coded:
            book, scale, units, _ = source
            units = F.max_pool2d(units, **spec['attrs'])
            table = torch.from_numpy(book.units).double()
            index = torch.bucketize(units.contiguous(), table)
            if not torch.equal(table[index], units):
                raise ValueError('reference: pooled value is not a codebook level')
            self.state[key] = (book, scale, units, book.codes[index.numpy()])
            self.note(key, None, None, {'code_passthrough': True})
            return None
        elif op in ('conv', 'linear'):
            book, scale, units, _ = self.stored(inputs[0])
            w = self.arrays[spec['mac']['weight_units']]
            wbook = self.books[spec['mac']['weight_codebook']]
            x = units.numpy().astype(np.int64)
            if op == 'conv':
                dot = exact_dot(x, w, lambda a, b: F.conv2d(a, b, None, **spec['attrs']))
            else:
                dot = exact_dot(x, w, lambda a, b: F.linear(a, b, None))
            mac = torch.from_numpy(dot) * float(2. ** -(book.shift + wbook.shift))
            raw = mac * float(scale)
            shape = (1, -1, 1, 1) if op == 'conv' else (1, -1)
            raw = raw * torch.tensor(spec['mac']['weight_scales'], dtype=torch.float64).reshape(shape)
            raw = raw + torch.from_numpy(self.arrays[spec['mac']['bias']].astype(np.float64)).reshape(shape)
            if op == 'linear':
                mac = mac[:, :, None, None]
        elif op == 'add':
            raw = self.real(inputs[0]) + self.real(inputs[1])
        elif op == 'relu':
            if coded:
                book, scale, units, _ = source
                raw = (torch.relu(units) * float(2. ** -book.shift)) * float(scale)
            else:
                raw = torch.clamp_min(source, 0.) + 0.
        elif op == 'relu6':                     # contract 2.2
            if coded:
                book, scale, units, _ = source
                raw = torch.clamp_max((torch.relu(units) * float(2. ** -book.shift)) * float(scale), 6.) + 0.
            else:
                raw = torch.clamp(source, 0., 6.) + 0.
        elif op in ('hardsigmoid', 'hardswish'):
            r = self.real(inputs[0]); t = torch.clamp(r + 3., 0., 6.)
            raw = (t / 6. if op == 'hardsigmoid' else (r * t) / 6.) + 0.
        elif op == 'mul':
            a, b = self.state[inputs[0]], self.state[inputs[1]]
            if isinstance(a, tuple) and isinstance(b, tuple):
                product = a[2] * b[2]
                if product.abs().max().item() >= 2.**53:
                    raise ValueError('reference: code product is not exact in binary64')
                raw = ((product * float(2. ** -(a[0].shift + b[0].shift))) * float(a[1])) * float(b[1]) + 0.
            else:
                raw = self.real(inputs[0]) * self.real(inputs[1]) + 0.
        elif op == 'maxpool':
            if coded:
                book, scale, units, _ = source
                raw = (F.max_pool2d(units, **spec['attrs']) * float(2. ** -book.shift)) * float(scale)
            else:
                raw = F.max_pool2d(source, **spec['attrs']) + 0.
        elif op == 'avgpool':
            book, scale, units, _ = self.stored(inputs[0])
            raw = units.sum(dim=(-2, -1), keepdim=True) * float(2. ** -book.shift)
            raw = raw / float(units.shape[-2] * units.shape[-1]); raw = raw * float(scale)
        else:
            raise ValueError('reference encountered an unsupported FX node')
        store = spec['store']
        if store is None:
            raw = raw + 0.
            self.state[key] = raw
            self.note(key, raw, mac, {'unstored': True})
            return None
        book = self.books[store['codebook']]; scale = store['scale']
        y = raw / float(scale)
        mid = torch.from_numpy(book.bounds); ties = torch.from_numpy(book.ties)
        pos = torch.bucketize(y.contiguous(), mid, right=False); adj = pos.clamp_max(len(mid) - 1)
        tied = y == mid[adj]; pos = pos + (tied & ties[adj]).long()
        units = torch.from_numpy(book.units).double()[pos]
        codes = book.codes[pos.numpy()]
        diag = {'elements': raw.numel(), 'clipped_low': int((y < book.levels[0]).sum()),
                'clipped_high': int((y > book.levels[-1]).sum()), 'ties': int(tied.sum()),
                'zeros': int((units == 0).sum()), 'normalization_overflow': int((~torch.isfinite(y)).sum())}
        self.state[key] = (book, scale, units, codes)
        self.note(key, raw, mac, diag)
        return None

    def evaluate(self, inputs):
        with torch.inference_mode():
            output = self.run(torch.from_numpy(inputs).double()).numpy()
        return {'layers': self.trace, 'output': array_hash(output), 'top5': stable_top5(output[0]),
                'top1_tie_count': int(np.sum(output[0] == output[0].max())),
                'top1_tied_classes': [int(c) for c in np.flatnonzero(output[0] == output[0].max())],
                'diagnostic_signature': digest({k: v['diagnostics'] for k, v in self.trace.items()})}
