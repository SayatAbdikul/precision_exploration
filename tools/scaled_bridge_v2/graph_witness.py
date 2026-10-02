"""Policy-aware whole-graph witness for repaired-recipe (B2) graphs (contract 2.2).

Adopted from review 3 (artifacts/scaled_bridge_v2/review/b2_graph.py) and extended to grouped and depthwise
convolutions, the control arm, ReLU6, hard-sigmoid, hard-swish and tensor multiply. It is written from the
contract text and shares no code with the engine, the FX/limb reference, `Codebook` or the rational oracles:

  inputs      lane L1's RAW export (`b2-recipe-export-1`: node rows, boundaries, FP32 scales in hexadecimal, weight
              manifest codes) and the image tensor; never the adapted export.
  codebooks   signed: decoded from the accepted manifest through the public format oracle (lowest code per
              value, even-code-then-lower-code midpoint ties); unsigned variant: levels 0 .. 2^bits-1 from the
              bit count alone.
  reduction   NumPy int64 tap loop in contract order (input channel, kernel row, kernel column; a padded position
              adds a zero product), one output-channel group at a time. wide: exact integer sum. sat.*: clamp
              after every add, counters per output. control: binary32 state, each step the exact float64 sum of
              the binary32 state and the integer product (asserted below 2^53) rounded once to binary32, i.e. one
              fused multiply-add. fp16: binary16 state through NumPy's correctly rounded float64 -> float16
              conversion. f21: frexp/rint to 13 significant bits (normal range asserted).
  widths      sat.abs-/sat.struct- widths recomputed here from the raw weights and the 2.1 rules.
  other ops   binary64 NumPy in the order the contract states.

The witness compares, for one image, every stored code tensor (by the engine's hash convention), the output,
the contract Top-5, the accumulator counters and the failure record with the sealed engine record.
"""
from __future__ import annotations
import json
import re
import struct
import numpy as np
from .common import ROOT, digest, unseal, reference, checked, immutable, array_hash, run_root, engine_sources, Stopwatch

ALIASES = ('identity', 'flatten')
FLOAT_P = {'fp16': 11, 'f21': 13}


def manifest_book(name):
    from fractions import Fraction as Q
    from public.formats.oracle.number_format import NumberFormat
    fmt = NumberFormat(json.loads((ROOT / f'public/formats/manifests/accepted/{name}.json').read_text()))
    value = {}
    for code in range(1 << fmt.bits):
        v = fmt.decode(code)
        if v.is_finite():
            value[code] = Q(v)
    if any(q.denominator & (q.denominator - 1) for q in value.values()):
        raise ValueError('not a dyadic codebook')
    shift = max(q.denominator.bit_length() - 1 for q in value.values())
    best = {}
    for code, q in value.items():
        best[q] = min(code, best.get(q, code))
    levels = sorted(best)
    return {'bits': fmt.bits, 'shift': shift, 'units': [int(q * (1 << shift)) for q in levels],
            'codes': [best[q] for q in levels], 'unit_of_code': {c: int(q * (1 << shift)) for c, q in value.items()}}


def upper_ties(codes):
    return [(codes[i + 1] & 1, codes[i + 1]) < (codes[i] & 1, codes[i]) for i in range(len(codes) - 1)]


def signed_width(lo, hi):
    w = 1
    while not (-(1 << (w - 1)) <= lo and hi <= (1 << (w - 1)) - 1):
        w += 1
    return w


def parse(policy):
    if policy in ('wide', 'control'):
        return (policy,)
    m = re.fullmatch(r'sat\.(w|abs-|struct-)(\d+)', policy)
    if m:
        return ('sat', {'w': 'global', 'abs-': 'abs', 'struct-': 'struct'}[m.group(1)], int(m.group(2)))
    m = re.fullmatch(r'(fp16|f21)(?:\.x(-?\d+))?', policy)
    if not m:
        raise ValueError(f'witness: unknown policy {policy}')
    return ('float', m.group(1), int(m.group(2) or 0))


class Failed(Exception):
    pass


class Witness:
    def __init__(self, raw, arrays):
        self.raw, self.arrays, self.fmt = raw, arrays, raw['format']
        self.manifest = manifest_book(self.fmt)
        self.books = {}
        for key, entry in raw['codebooks'].items():
            if entry['signedness'] == 'signed':
                units, codes, shift = self.manifest['units'], self.manifest['codes'], self.manifest['shift']
            else:
                n = 1 << self.manifest['bits']; units, codes, shift = list(range(n)), list(range(n)), 0
            units = np.array(units, dtype=np.int64)
            level = np.ldexp(units.astype(np.float64), -shift)
            mid = (level[:-1] + level[1:]) / 2
            if any(float(m) * 2 != float(a) + float(b) for m, a, b in zip(mid, level[:-1], level[1:])):
                raise ValueError('witness: midpoint not exact')
            self.books[key] = {'units': units, 'level': level, 'mid': mid, 'shift': shift,
                               'up': np.array(upper_ties(codes) + [False]), 'codes': np.array(codes, dtype=np.int64),
                               'dtype': np.uint8 if max(codes) < 256 else np.uint16}
        self.lut = np.full(1 << self.manifest['bits'], np.iinfo(np.int64).min, dtype=np.int64)
        for c, u in self.manifest['unit_of_code'].items():
            self.lut[c] = u
        self.by = {n['name']: n for n in raw['nodes']}

    # -- states: {'idx', 'book', 'scale'} for stored codes, {'val'} for an unstored binary64 value
    def store(self, node, value):
        bd = node.get('boundary')
        if bd is None or not bd['quantizes']:
            return {'val': value + 0.0}
        book = self.books[bd['codebook']]
        scale = float(struct.unpack('>f', bytes.fromhex(bd['scale_fp32_hex']))[0])
        y = value / np.float64(scale)
        if np.isnan(y).any():
            raise ValueError('witness: NaN at a store')
        mid = book['mid']
        idx = np.searchsorted(mid, y, side='left')
        at = np.minimum(idx, len(mid) - 1)
        tie = (idx < len(mid)) & (mid[at] == y)
        idx = idx + (tie & book['up'][at])
        return {'idx': idx, 'book': book, 'scale': scale}

    @staticmethod
    def real(s):
        if 'val' in s:
            return s['val']
        return s['book']['level'][s['idx']] * np.float64(s['scale'])

    def source(self, name):
        node = self.by[name]
        while node['op'] in ALIASES or (node['op'] == 'maxpool' and not node['boundary']['quantizes']):
            node = self.by[node['inputs'][0]]
        return node

    def widths(self, src_node, book, w):
        flat = w.reshape(w.shape[0], -1); lo, hi = int(book['units'][0]), int(book['units'][-1])
        slo = 0 if (src_node['op'] in ('relu', 'relu6', 'hardsigmoid') and lo <= 0) else lo
        wa = ws = 0
        for row in flat:
            p = int(row[row > 0].sum()); n = int(row[row < 0].sum())
            wa = max(wa, (max(abs(lo), abs(hi)) * (p - n)).bit_length() + 1)
            ws = max(ws, signed_width(p * slo + n * hi, p * hi + n * slo))
        return wa, ws

    def mac(self, node, s, src_node, kind):
        key = node['name']
        if 'idx' not in s:
            raise ValueError('witness: MAC input is not stored')
        book = s['book']; x = book['units'][s['idx']]
        w = self.lut[self.arrays[f'{key}.weight_codes'].astype(np.int64)]
        if (w == np.iinfo(np.int64).min).any():
            raise ValueError('witness: weight code without a level')
        if node['op'] == 'linear':
            x = x.reshape(-1, 1, 1); w = w[:, :, None, None]; attrs = {}
        else:
            attrs = node['attrs']
        g = int(attrs.get('groups', 1))
        sh, sw = attrs.get('stride', [1, 1]); ph, pw = attrs.get('padding', [0, 0]); dh, dw = attrs.get('dilation', [1, 1])
        ci, h, wi = x.shape; co, cg, kh, kw = w.shape
        if cg * g != ci or co % g:
            raise ValueError('witness: group geometry')
        oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1; ow = (wi + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        xp = np.zeros((ci, h + 2 * ph, wi + 2 * pw), dtype=np.int64); xp[:, ph:ph + h, pw:pw + wi] = x
        shift = book['shift'] + self.manifest['shift']
        stats = {'elements': co * oh * ow}
        if kind[0] == 'sat':
            wa, ws = self.widths(src_node, book, w)
            W = kind[2] if kind[1] == 'global' else max(2, (wa if kind[1] == 'abs' else ws) - kind[2])
            hi = 2 ** (W - 1) - 1; lo = -2 ** (W - 1); stats['width'] = W
            up = np.zeros((co, oh, ow), dtype=np.int64); dn = np.zeros((co, oh, ow), dtype=np.int64)
        if kind[0] == 'float':
            e = kind[2]; factor = 2.0 ** (e - shift)
            state = np.zeros((co, oh, ow), dtype=np.float16 if kind[1] == 'fp16' else np.float64)
            steps = np.zeros((co, oh, ow), dtype=np.int64)
        elif kind[0] == 'control':
            state = np.zeros((co, oh, ow), dtype=np.float32)
        else:
            acc = np.zeros((co, oh, ow), dtype=np.int64)
        channel = (np.arange(co) // (co // g)) * cg
        with np.errstate(over='ignore'):
            for c in range(cg):
                rows = xp[c][None] if g == 1 else xp[channel + c]
                for ky in range(kh):
                    for kx in range(kw):
                        patch = rows[:, ky * dh: ky * dh + sh * (oh - 1) + 1: sh, kx * dw: kx * dw + sw * (ow - 1) + 1: sw]
                        term = w[:, c, ky, kx][:, None, None] * patch
                        if kind[0] == 'float':
                            total = state.astype(np.float64) + term.astype(np.float64) * factor
                            if kind[1] == 'fp16':
                                state = total.astype(np.float16)
                            else:
                                mant, ex = np.frexp(total)
                                state = np.ldexp(np.rint(np.ldexp(mant, FLOAT_P['f21'])), ex - FLOAT_P['f21'])
                                nz = np.abs(state[(state != 0) & np.isfinite(state)])
                                if nz.size and not (nz.max() < 2.0 ** 120 and nz.min() > 2.0 ** -120):
                                    raise ValueError('witness: f21 state outside the emulated normal range')
                            steps += np.isinf(state)
                        elif kind[0] == 'control':
                            wide = state.astype(np.float64) + term.astype(np.float64)
                            if np.abs(state).max(initial=0) + np.abs(term).max(initial=0) >= 2.0 ** 53:
                                raise ValueError('witness: control emulation outside the exact float64 domain')
                            state = wide.astype(np.float32)
                        else:
                            acc += term
                            if kind[0] == 'sat':
                                up += acc > hi; dn += acc < lo; np.clip(acc, lo, hi, out=acc)
        if kind[0] == 'float':
            bad = np.isinf(state)
            stats.update(nonfinite_elements=int(bad.sum()), nonfinite_steps=int(steps.sum()))
            if bad.any():
                raise Failed({'node': key, 'positive_infinite': int(np.isposinf(state).sum()),
                              'negative_infinite': int(np.isneginf(state).sum()), **stats})
            dot = state.astype(np.float64) * 2.0 ** (-e); dot[dot == 0] = 0.0
        elif kind[0] == 'control':
            dot = np.ldexp(state.astype(np.float64), -shift); dot[dot == 0] = 0.0
        else:
            if int(np.abs(acc).max(initial=0)) >= 2 ** 53:
                raise ValueError('witness: exact sum beyond the binary64 integers (not needed by these cases)')
            dot = np.ldexp(acc.astype(np.float64), -shift)
            if kind[0] == 'sat':
                stats.update(saturated_elements=int(((up > 0) | (dn > 0)).sum()),
                             high_clamps=int(np.minimum(up, 65535).sum()), low_clamps=int(np.minimum(dn, 65535).sum()))
        scales = self.arrays[f'{key}.weight_scales'].astype(np.float64)[:, None, None]
        bias = self.arrays[f'{key}.bias'].astype(np.float64)[:, None, None]
        value = dot * np.float64(s['scale']); value = value * scales; value = value + bias
        if node['op'] == 'linear':
            value = value[:, 0, 0]
        return value, stats

    def run(self, image, policy):
        kind = parse(policy); values = {}; out = {'codes': {}, 'acc': {}, 'failure': None}
        for node in self.raw['nodes']:
            key, op = node['name'], node['op']; args = [values[n] for n in node['inputs']]
            if op == 'output':
                out['output'] = self.real(args[0]) + 0.0
                break
            if op == 'identity':
                values[key] = args[0]; continue
            if op == 'flatten':
                s = dict(args[0]); k = 'idx' if 'idx' in s else 'val'; s[k] = s[k].reshape(-1); values[key] = s; continue
            if op == 'maxpool' and not node['boundary']['quantizes']:
                s = args[0]; at = node['attrs']
                if 'idx' not in s:
                    raise ValueError('witness: unstored max-pool input')
                (kh, kw), (sh, sw), (ph, pw) = at['kernel_size'], at['stride'], at['padding']
                c, h, wi = s['idx'].shape; oh = (h + 2 * ph - kh) // sh + 1; ow = (wi + 2 * pw - kw) // sw + 1
                pad = np.full((c, h + 2 * ph, wi + 2 * pw), -1, dtype=np.int64); pad[:, ph:ph + h, pw:pw + wi] = s['idx']
                best = np.full((c, oh, ow), -1, dtype=np.int64)
                for ky in range(kh):
                    for kx in range(kw):
                        best = np.maximum(best, pad[:, ky: ky + sh * (oh - 1) + 1: sh, kx: kx + sw * (ow - 1) + 1: sw])
                values[key] = {'idx': best, 'book': s['book'], 'scale': s['scale']}; continue
            if op == 'input':
                value = np.asarray(image, dtype=np.float64)
            elif op in ('conv', 'linear'):
                try:
                    value, stats = self.mac(node, args[0], self.source(node['inputs'][0]), kind)
                except Failed as failure:
                    out['failure'] = failure.args[0]; return out
                out['acc'][key] = stats
            elif op == 'add':
                a, b = self.real(args[0]), self.real(args[1])
                if a.shape != b.shape:
                    raise ValueError('witness: residual shapes')
                value = a + b
            elif op == 'relu':
                value = np.where(self.real(args[0]) > 0, self.real(args[0]), 0.0)
            elif op == 'relu6':
                value = np.minimum(np.where(self.real(args[0]) > 0, self.real(args[0]), 0.0), 6.0)
            elif op in ('hardsigmoid', 'hardswish'):
                r = self.real(args[0]); t = np.minimum(np.maximum(r + 3.0, 0.0), 6.0)
                value = t / 6.0 if op == 'hardsigmoid' else (r * t) / 6.0
            elif op == 'mul':
                a, b = args
                if 'idx' in a and 'idx' in b:
                    p = a['book']['units'][a['idx']][None] * b['book']['units'][b['idx']][None]
                    if np.abs(p).max(initial=0) >= 2 ** 53:
                        raise ValueError('witness: code product beyond the binary64 integers')
                    value = (np.ldexp(p.astype(np.float64), -(a['book']['shift'] + b['book']['shift']))
                             * np.float64(a['scale'])) * np.float64(b['scale'])
                    value = value[0]
                else:
                    value = self.real(a) * self.real(b)
            elif op == 'avgpool':
                s = args[0]
                if 'idx' not in s:
                    raise ValueError('witness: unstored average-pool input')
                units = s['book']['units'][s['idx']]
                total = units.sum(axis=(-2, -1), keepdims=True)
                value = np.ldexp(total.astype(np.float64), -s['book']['shift']) / np.float64(units.shape[-2] * units.shape[-1])
                value = value * np.float64(s['scale'])
            else:
                raise ValueError(f'witness: operator {op}')
            values[key] = self.store(node, value + 0.0)
            if 'idx' in values[key]:
                book = values[key]['book']
                out['codes'][key] = book['codes'][values[key]['idx']].astype(book['dtype'])
        return out


def compare(want, record, policy):
    """Problems between the witness result for one image and one sealed engine record."""
    problems = []
    kind = parse(policy)
    if want['failure'] is not None or record.get('failure') is not None:
        f = record.get('failure')
        if f is None or want['failure'] is None or any(f.get(k) != v for k, v in want['failure'].items()) \
                or record['top5'] is not None or record['output'] is not None:
            problems.append(('failure record', str(f)[:200], str(want['failure'])[:200]))
        return problems
    stored = {k for k, v in record['layers'].items() if v.get('codes') is not None
              and not (v['diagnostics'].get('code_passthrough') or v['diagnostics'].get('skipped'))}
    if set(want['codes']) != stored:
        problems.append(('storing node set', sorted(set(want['codes']) ^ stored)[:6]))
    for key, codes in want['codes'].items():
        if key in record['layers'] and array_hash(codes[None]) != record['layers'][key]['codes']:
            problems.append(('stored codes', key))
    output = want['output']
    if array_hash(output[None]) != record['output']:
        problems.append(('output hash',))
    top5 = sorted(range(output.size), key=lambda c: (-output[c], c))[:5]
    if top5 != record['top5']:
        problems.append(('top5', top5, record['top5']))
    if kind[0] in ('sat', 'float'):
        for key, stats in want['acc'].items():
            got = record['accumulator'].get(key, {})
            if any(got.get(k) != v for k, v in stats.items()):
                problems.append(('accumulator counters', key, {k: got.get(k) for k in stats}, stats))
    return problems


def witness(case, policy, backend='cpp', image=0):
    """Run the witness on one gating image and compare it with the sealed engine record; seal the result."""
    from tools.experiment_b.classifier import configure, load_model, image_batch
    from tools.experiment_b.common import dataset
    from tools.experiment_b2.export import load_export as load_b2
    from .export import load_export
    from .worker import locate
    watch = Stopwatch('graph-witness', [case, policy, backend, image])
    configure('cpu')
    ex, _, ref = load_export(case, locate(case))
    source = checked(ex['provenance']['B2_export'])
    raw, arrays = load_b2(source.parent)
    if raw['configuration_sha256'] != ex['provenance']['B2_configuration_sha256']:
        raise ValueError('witness: raw export belongs to another configuration')
    _, transform, original = load_model(ex['model'], 'cpu'); del original
    _, rows, payload = dataset('imagenet_screen_1k')
    if digest(rows) != ex['retained']['ordered_samples_sha256']:
        raise ValueError('witness: ordered development samples changed')
    path = run_root() / case / f'{policy}-{backend}' / f'{image:04d}.json'
    record = unseal(path)
    if record['export'] != ref or record['sample'] != rows[image]:
        raise ValueError('witness: sealed record belongs to another export or sample')
    inputs = image_batch(rows[image:image + 1], payload, transform, 'cpu').numpy()
    want = Witness(raw, arrays).run(inputs[0], policy)
    problems = compare(want, record, policy)
    result = {'status': 'pass' if not problems else 'fail', 'case': case, 'policy': policy, 'backend': backend,
              'image': image, 'sample': rows[image], 'record': reference(path), 'raw_export': reference(source),
              'stored_tensors_compared': len(want['codes']), 'accumulator_nodes_compared': len(want['acc']),
              'failure': want['failure'], 'problems': [str(p)[:400] for p in problems],
              'engine_sources': digest(engine_sources()),
              'basis': 'graph_witness.py: written from the contract text; raw lane-L1 export; manifest-decoded '
                       'codebooks; NumPy int64 tap loop per policy; no engine, reference or oracle code'}
    immutable(run_root() / case / f'graph-witness-{policy}-{backend}-{image:04d}.json', result)
    watch.close(problems=len(problems))
    print(f'{case} {policy}-{backend} image {image}: graph witness {result["status"]} '
          f'({len(want["codes"])} stored tensors, {len(want["acc"])} accumulator nodes, {len(problems)} problems)', flush=True)
    if problems:
        raise ValueError(f'graph witness mismatch {problems[:3]}')
    return result
