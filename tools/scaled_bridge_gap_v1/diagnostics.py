"""Step-4 diagnostic harness: ablate B-versus-contract differences one at a time.

DIAGNOSTIC, NOT BIT-CERTIFIED.  The harness re-implements both execution
semantics in torch on CUDA with per-node toggles.  Only its two endpoints are
checked against sealed evidence: the all-B configuration must reproduce the
B replay stored-code hashes at every node and the retained B Top-5, and the
all-X configuration must reproduce the sealed wide-arm stored-code hashes at
every node and its Top-5.  Mixed configurations are assembled from those same
parts but are not independently certified.

Toggles (B = simulator semantics, X = scaled contract semantics):
  in      input normalization  y = x/s in FP32 (B) or binary64 (X)
  mac     B   : FP32 cuDNN conv/linear of FP32-reconstructed operands with FP32 bias
          R64 : binary64 conv of the same FP32-reconstructed operands, binary64 bias
                (removes FP32 accumulation only; scales still applied before the dot)
          X   : exact code-grid dot, scales applied after in binary64, binary64 bias
  store   output normalization raw/s in FP32 (raw rounded to FP32 first) or binary64
  nonmac  residual add, ReLU, max/avg pooling reconstruction and arithmetic in FP32 or binary64
Zero canonicalisation is not a toggle: the B path keeps B's signed zeros only
implicitly (it reconstructs from codes), which is shown harmless by the all-B
endpoint reproducing every B stored code and Top-5.
"""
from __future__ import annotations
from collections import Counter
import json
import time
import numpy as np
from .common import *

DIAG_PROTOCOL = {
 'version': 'scaled-bridge-gap-diag-1',
 'date': '2026-10-01',
 'parent': 'scaled-bridge-gap-1k-1 (triggered: 1k wide-minus-B top-1 point estimate <= -1.0 pp for fp6_e3m2 (-1.9) and fp7_e3m3 (-1.4))',
 'evidence_level': 'development diagnostic; harness is not bit-certified except at its two checked endpoints',
 'cases': ['fp6_e3m2', 'fp7_e3m3', 'fp6_e2m3 (untriggered negative control, run last if GPU time allows)'],
 'panel': 'same frozen imagenet_screen_1k, all 1000 images, original batch-8 membership',
 'endpoint_gate': 'all-B must equal B replay stored-code hashes at every node and retained B Top-5 on every image; all-X must equal sealed '
                  'wide stored-code hashes at every node and Top-5 on every image; a failure stops the diagnostic for that case',
 'configurations': {
   'B': 'in=B mac=B store=B nonmac=B', 'X': 'in=X mac=X store=X nonmac=X',
   'B+in64': 'B with in=X', 'B+mac64acc': 'B with mac=R64', 'B+macX': 'B with mac=X', 'B+store64': 'B with store=X',
   'B+nonmac64': 'B with nonmac=X',
   'X-in32': 'X with in=B', 'X-macB': 'X with mac=B', 'X-store32': 'X with store=B', 'X-nonmac32': 'X with nonmac=B',
   'Xstage:<s>': 'B everywhere except all toggles X inside stage s in {stem, layer1, layer2, layer3, layer4, head}'},
 'measurements': {
   'accuracy': 'top-1/top-5 per configuration; paired_outcomes versus B and versus X (same statistics as parent)',
   'propagated_flips': 'per node: stored codes differing from the B configuration on the same image, split into up (larger code value) and down',
   'local_flips': 'single-toggle-from-B configurations only: per node, the toggled operation applied to the B configuration\'s own input '
                  'states (teacher forcing); stored codes differing from B output, split up/down and toward larger/smaller magnitude',
   'sign_bias': 'exact two-sided binomial test of up versus down local flips per configuration (all nodes pooled) and per node'},
 'stop_rule': 'one pass per case over 1000 images; no configuration is added after looking at accuracy; anything else is labelled exploratory',
}


STAGES = ('stem', 'layer1', 'layer2', 'layer3', 'layer4', 'head')


def stage(node):
    """Stage assignment AS RUN by scaled-bridge-gap-diag-1 (records under diagnostics/).

    Known defect (review 1, 2026-10-01): the eight residual adds (`add`, `add_1` .. `add_7`)
    do not start with `layerN`, so they fall into 'stem'.  The diag-1 'Xstage:stem' hybrid
    therefore also switched every residual add, and 'Xstage:layerN' did not switch its own adds.
    Kept unchanged so the sealed diag-1 records stay reproducible; use exec_stages() for
    reporting and for the corrected hybrids (scaled-bridge-gap-diag-1a, stagefix.py)."""
    for s in ('layer1', 'layer2', 'layer3', 'layer4'):
        if node.startswith(s):
            return s
    return 'head' if node in ('avgpool', 'flatten', 'fc', 'output') else 'stem'


def exec_stages(nodes):
    """Corrected stage of every node, from the graph in execution order.

    Nodes named layerN* belong to layerN; avgpool/flatten/fc/output to the head; a node with
    another name inherits the latest layer stage among its inputs (this places each residual add
    in the block that produces it); anything else before layer1 is the stem."""
    out = {}
    for n in nodes:
        name = n['name']
        s = next((t for t in ('layer1', 'layer2', 'layer3', 'layer4') if name.startswith(t)), None)
        if s is None and name in ('avgpool', 'flatten', 'fc', 'output'):
            s = 'head'
        if s is None:
            ins = [out[i] for i in n.get('inputs', []) if i in out]
            s = max(ins, key=STAGES.index) if ins else 'stem'
        out[name] = s
    return out


ALL_B = {'in': 'B', 'mac': 'B', 'store': 'B', 'nonmac': 'B'}
ALL_X = {'in': 'X', 'mac': 'X', 'store': 'X', 'nonmac': 'X'}


def configurations():
    c = {'B': (ALL_B, None), 'X': (ALL_X, None),
         'B+in64': ({**ALL_B, 'in': 'X'}, None), 'B+mac64acc': ({**ALL_B, 'mac': 'R64'}, None),
         'B+macX': ({**ALL_B, 'mac': 'X'}, None), 'B+store64': ({**ALL_B, 'store': 'X'}, None),
         'B+nonmac64': ({**ALL_B, 'nonmac': 'X'}, None),
         'X-in32': ({**ALL_X, 'in': 'B'}, None), 'X-macB': ({**ALL_X, 'mac': 'B'}, None),
         'X-store32': ({**ALL_X, 'store': 'B'}, None), 'X-nonmac32': ({**ALL_X, 'nonmac': 'B'}, None)}
    for s in ('stem', 'layer1', 'layer2', 'layer3', 'layer4', 'head'):
        c['Xstage:' + s] = (ALL_B, s)
    return c


SINGLE = ('B+in64', 'B+mac64acc', 'B+macX', 'B+store64', 'B+nonmac64')


class Harness:
    def __init__(self, ex, arrays, device):
        import torch
        from tools.scaled_bridge_v1.quantization import codebook
        self.t = torch; self.dev = device; self.ex = ex; self.nodes = ex['nodes']
        lev, codes, units, mid, ties, self.shift = codebook(ex['format'])
        self.unitbook = torch.tensor(units.astype(np.float64), device=device)
        self.unitkeys = units.astype(np.int64); self.codebook = codes
        self.mid = torch.tensor(mid, device=device); self.ties = torch.tensor(ties, device=device)
        self.mid32 = torch.tensor(mid.astype(np.float32), device=device)
        if not np.array_equal(mid.astype(np.float32).astype(np.float64), mid):
            raise ValueError('codebook midpoints not FP32-exact')
        self.ascale = ex['scales']['activation_scales']
        self.W = {}
        for n in self.nodes:
            if n['op'] in ('conv', 'linear'):
                k = n['name']
                self.W[k] = {'u': torch.tensor(arrays[k + '_units'].astype(np.float64), device=device),
                             'r32': torch.tensor(arrays[k + '_B_reconstructed'], device=device),
                             'b32': torch.tensor(arrays[k + '_bias'], device=device),
                             'ws': torch.tensor(ex['scales']['weight_scales'][k], dtype=torch.float64, device=device)}

    # ---- primitives -------------------------------------------------------
    def quant(self, y):
        t = self.t
        y64 = y.double()
        pos = t.bucketize(y64.contiguous(), self.mid, right=False); adj = pos.clamp_max(len(self.mid) - 1)
        pos = pos + ((y64 == self.mid[adj]) & self.ties[adj]).long()
        return self.unitbook[pos]

    def r32(self, u, s):
        t = self.t
        return (u * 2. ** -self.shift).float() * t.tensor(s, dtype=t.float32, device=self.dev)

    def r64(self, u, s):
        return (u * 2. ** -self.shift) * float(s)

    def store(self, raw, key, mode):
        t = self.t
        s = self.ascale[key]
        if mode == 'B':
            y = raw.float() / t.tensor(s, dtype=t.float32, device=self.dev)
        else:
            y = raw.double() / float(s)
        if not t.isfinite(y).all():
            raise ValueError('nonfinite normalized value')
        return self.quant(y)

    def node(self, n, args, x, cfg):
        """Return stored units for node n given input units `args` (list of (units, scale))."""
        F = self.t.nn.functional; op = n['op']; key = n['name']
        if op == 'input':
            raw = x if cfg['in'] == 'B' else x.double()
            return self.store(raw, key, cfg['in'])
        if op in ('conv', 'linear'):
            u, s = args[0]; W = self.W[key]; attrs = n['attrs']
            if cfg['mac'] == 'X':
                if op == 'conv':
                    dot = F.conv2d(u, W['u'], None, **attrs)
                    shape = (1, -1, 1, 1)
                else:
                    dot = F.linear(u, W['u'], None); shape = (1, -1)
                raw = dot * 2. ** (-2 * self.shift); raw = raw * float(s)
                raw = raw * W['ws'].reshape(shape); raw = raw + W['b32'].double().reshape(shape)
            else:
                a = self.r32(u, s); w = W['r32']; b = W['b32']
                if cfg['mac'] == 'R64':
                    a, w, b = a.double(), w.double(), b.double()
                raw = F.conv2d(a, w, b, **attrs) if op == 'conv' else F.linear(a, w, b)
            return self.store(raw, key, cfg['store'])
        if op == 'identity':
            return args[0][0]
        if op == 'flatten':
            return args[0][0].flatten(1)
        r = self.r32 if cfg['nonmac'] == 'B' else self.r64
        if op == 'add':
            raw = r(*args[0]) + r(*args[1])
        elif op == 'relu':
            raw = self.t.relu(r(*args[0]))
        elif op == 'maxpool':
            a = n['attrs']; raw = F.max_pool2d(r(*args[0]), a['kernel_size'], a['stride'], a['padding'], a['dilation'])
        elif op == 'avgpool':
            u, s = args[0]
            if cfg['nonmac'] == 'B':
                raw = F.adaptive_avg_pool2d(self.r32(u, s), (1, 1))
            else:
                raw = u.sum(dim=(-2, -1), keepdim=True) * 2. ** -self.shift
                raw = raw / float(u.shape[-2] * u.shape[-1]); raw = raw * float(s)
        else:
            raise ValueError('unsupported op ' + op)
        return self.store(raw, key, cfg['nonmac'])

    def forward(self, x, cfg, only_stage=None, teacher=None, stage_of=None):
        """Run the graph.  Returns (states dict key->units, logits, local dict key->units or None).

        stage_of: node name -> stage used for `only_stage`; default is the diag-1 as-run `stage`."""
        stage_of = stage_of or stage
        states = {}; scales = {}; local = {} if teacher is not None else None
        for n in self.nodes:
            key, op = n['name'], n['op']
            if op == 'output':
                u = states[n['inputs'][0]]; s = scales[n['inputs'][0]]
                c = cfg if only_stage is None else (ALL_X if stage_of(key) == only_stage else ALL_B)
                logits = self.r32(u, s) if c['nonmac'] == 'B' else self.r64(u, s)
                return states, logits, local
            c = cfg if only_stage is None else (ALL_X if stage_of(key) == only_stage else ALL_B)
            args = [(states[i], scales[i]) for i in n['inputs']]
            states[key] = self.node(n, args, x, c)
            scales[key] = scales[n['inputs'][0]] if op in ('identity', 'flatten') else self.ascale[key]
            if teacher is not None and op not in ('identity', 'flatten'):
                targs = [(teacher[i], scales[i]) for i in n['inputs']]
                local[key] = self.node(n, targs, x, c)
        raise ValueError('graph without output')

    def codes_hash(self, u):
        from tools.scaled_bridge_v1.common import array_hash
        a = u.cpu().numpy().astype(np.int64)
        idx = np.searchsorted(self.unitkeys, a)
        if not np.array_equal(self.unitkeys[idx], a):
            raise ValueError('non-grid unit')
        return [array_hash(self.codebook[idx][i:i + 1]) for i in range(a.shape[0])]


def topk(logits):
    """FP32 logits: B's batched CUDA topk; binary64 logits: the engine's per-image CPU topk (tie order matters)."""
    if logits.dtype == logits.new_zeros(()).float().dtype:
        return logits.topk(5, dim=1).indices.cpu().tolist()
    cpu = logits.cpu()
    return [cpu[i:i + 1].topk(5, dim=1).indices.tolist()[0] for i in range(cpu.shape[0])]


def diag_protocol_path():
    return BASE / 'protocol-diag-v1.json'


def run_case(name, start=0, stop=PANEL):
    import torch
    unseal_p = unseal(diag_protocol_path())
    if unseal_p != DIAG_PROTOCOL:
        raise ValueError('diagnostic protocol drift')
    from tools.experiment_b.classifier import configure, load_model, image_batch
    from tools.experiment_b.common import dataset
    from tools.scaled_bridge_v1.export import load_export
    started = time.perf_counter()
    configure('cuda')
    ex, arrays, ref = load_export(name)
    graph, transform, original = load_model('resnet18', 'cpu'); del original, graph
    _, rows, payload = dataset('imagenet_screen_1k')
    h = Harness(ex, arrays, 'cuda'); cfgs = configurations()
    out = BASE / 'diagnostics' / name; out.mkdir(parents=True, exist_ok=True)
    folder_B = arm_folder(name, 'B'); folder_W = arm_folder(name, 'wide')
    for s in range(start, stop, 8):
        path = out / f'{s:04d}.json'
        if path.exists():
            continue
        tick = time.perf_counter()
        x = image_batch(rows[s:s + 8], payload, transform, 'cuda')
        recB = [unseal(folder_B / f'{i:04d}.json') for i in range(s, s + 8)]
        recW = [unseal(folder_W / f'{i:04d}.json') for i in range(s, s + 8)]
        result = {'start': s, 'samples': [r['sample']['sha256'] for r in recB], 'configs': {}}
        with torch.inference_mode():
            base, logitsB, _ = h.forward(x, ALL_B)
            # endpoint gates
            topB = topk(logitsB)
            for key, u in base.items():
                hs = h.codes_hash(u)
                if any(hs[j] != recB[j]['layers'][key]['codes'] for j in range(8)):
                    raise ValueError(f'all-B endpoint does not reproduce B codes at {key}, batch {s}')
            if topB != [r['top5'] for r in recB]:
                raise ValueError(f'all-B endpoint Top-5 mismatch batch {s}')
            for label, (cfg, only) in cfgs.items():
                teacher = base if label in SINGLE else None
                st, lg, loc = (base, logitsB, None) if label == 'B' else h.forward(x, cfg, only, teacher)
                top = topk(lg)
                if label == 'X':
                    for key, u in st.items():
                        hs = h.codes_hash(u)
                        if any(hs[j] != recW[j]['layers'][key]['codes'] for j in range(8)):
                            raise ValueError(f'all-X endpoint does not reproduce wide codes at {key}, batch {s}')
                    if top != [r['top5'] for r in recW]:
                        raise ValueError(f'all-X endpoint Top-5 mismatch batch {s}')
                entry = {'top5': top}
                if label != 'B':
                    prop = {}
                    for key, u in st.items():
                        b = base[key]
                        if u.shape != b.shape:
                            continue
                        up = int((u > b).sum()); down = int((u < b).sum())
                        if up or down:
                            prop[key] = [up, down]
                    entry['propagated_flips'] = prop
                if loc is not None:
                    lf = {}
                    for key, u in loc.items():
                        b = base[key]
                        d = u != b
                        if d.any():
                            lf[key] = [int((u > b).sum()), int((u < b).sum()),
                                       int((u.abs() > b.abs()).sum()), int((u.abs() < b.abs()).sum()), int(b.numel())]
                    entry['local_flips'] = lf
                result['configs'][label] = entry
        result['seconds'] = time.perf_counter() - tick
        immutable(path, result)
        if (s // 8) % 25 == 0:
            print(f'{name} diag {s + 8}/{stop} {result["seconds"]:.2f}s/batch', flush=True)
    print(json.dumps({'job': 'diagnose', 'format': name, 'start': start, 'stop': stop,
                      'wall_seconds': time.perf_counter() - started}), flush=True)


def main(args):
    if args[0] == 'protocol':
        immutable(diag_protocol_path(), DIAG_PROTOCOL); print(diag_protocol_path())
    elif args[0] == 'run':
        run_case(args[1], int(args[2]), int(args[3]))
    elif args[0] == 'summarize':
        from .diag_summary import write
        write()
    else:
        raise ValueError('unknown diagnose action')
