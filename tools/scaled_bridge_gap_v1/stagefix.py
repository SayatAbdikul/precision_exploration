"""Corrected stage hybrids (scaled-bridge-gap-diag-1a).

Review 1 (2026-10-01) found that diag-1's stage assignment put the eight residual adds in
'stem' (diagnostics.stage).  This addendum reruns ONLY the six stage hybrids with the
execution-graph assignment diagnostics.exec_stages (each add in the block that produces it),
on the same 1000 images and batch-8 membership, with the same harness.  Everything else from
diag-1 (single switches, local flips, endpoints) is unaffected and is not rerun.

Per batch it checks the all-B endpoint exactly as diag-1 does (B replay codes at every node and
Top-5), and checks that the corrected head hybrid, whose node set is unchanged by the fix, gives
the same Top-5 as the diag-1 'Xstage:head' record.  It records, per corrected hybrid, the own-topk
Top-5, the common-rule Top-5 / top-1 tie set / Top-5 boundary tie (ties.common_order), the fc code
hash, and propagated code flips against all-B.
"""
from __future__ import annotations
import json
import time
from .common import *
from .diagnostics import (Harness, ALL_B, STAGES, exec_stages, stage, topk, diag_protocol_path, DIAG_PROTOCOL)

STAGEFIX_PROTOCOL = {
 'version': 'scaled-bridge-gap-diag-1a',
 'date': '2026-10-01',
 'parent': 'scaled-bridge-gap-diag-1 (correction addendum after independent review 1, finding nonblocking-2)',
 'reason': 'diag-1 assigned the residual adds add..add_7 to the stem (name-prefix rule); the stage hybrids were mislabelled: '
           'Xstage:stem also switched all eight adds, Xstage:layerN did not switch its own adds',
 'change': 'stage of every node from the execution graph: layerN* -> layerN; avgpool/flatten/fc/output -> head; other nodes inherit '
           'the latest layer stage among their inputs (each add joins the block producing it); remaining nodes before layer1 -> stem',
 'configurations': 'Xstage2:<s> for s in stem, layer1, layer2, layer3, layer4, head: B everywhere except all toggles X inside stage s',
 'panel': 'frozen imagenet_screen_1k, all 1000 images, original batch-8 membership; fp6_e2m3, fp6_e3m2, fp7_e3m3',
 'gates': 'all-B endpoint equals B replay stored-code hashes at every node and Top-5 on every image; Xstage2:head Top-5 equals the '
          'diag-1 Xstage:head record (node set unchanged); any failure stops the case',
 'measurements': 'top-1/top-5 measured (own topk) and under the common tie rule of scaled-bridge-gap-ties-1; paired_outcomes versus B; '
                 'propagated flips per corrected stage; images whose prediction differs from the as-run diag-1 hybrid',
 'evidence_level': 'development diagnostic; harness not bit-certified between endpoints; correction rerun written before it ran',
 'stop_rule': 'one pass per case over 1000 images; nothing added after looking at results',
}


def stagefix_protocol_path():
    return BASE / 'protocol-diag-1a-stagefix.json'


def labels_cfg():
    return ['Xstage2:' + s for s in STAGES]


def run_case(name, start=0, stop=PANEL):
    import torch
    from .ties import common_order
    if unseal(diag_protocol_path()) != DIAG_PROTOCOL:
        raise ValueError('diagnostic protocol drift')
    immutable(stagefix_protocol_path(), STAGEFIX_PROTOCOL)
    from tools.experiment_b.classifier import configure, load_model, image_batch
    from tools.experiment_b.common import dataset
    from tools.scaled_bridge_v1.export import load_export
    started = time.perf_counter()
    configure('cuda')
    ex, arrays, ref = load_export(name)
    graph, transform, original = load_model('resnet18', 'cpu'); del original, graph
    _, rows, payload = dataset('imagenet_screen_1k')
    h = Harness(ex, arrays, 'cuda')
    smap = exec_stages(h.nodes); stage_of = smap.__getitem__
    logit_node = [n for n in h.nodes if n['op'] == 'output'][0]['inputs'][0]
    out = BASE / 'diagnostics_stagefix' / name; out.mkdir(parents=True, exist_ok=True)
    diag = BASE / 'diagnostics' / name
    folder_B = arm_folder(name, 'B')
    for s in range(start, stop, 8):
        path = out / f'{s:04d}.json'
        if path.exists():
            continue
        tick = time.perf_counter()
        x = image_batch(rows[s:s + 8], payload, transform, 'cuda')
        recB = [unseal(folder_B / f'{i:04d}.json') for i in range(s, s + 8)]
        drec = unseal(diag / f'{s:04d}.json')
        result = {'start': s, 'samples': [r['sample']['sha256'] for r in recB], 'configs': {}}
        with torch.inference_mode():
            base, logitsB, _ = h.forward(x, ALL_B)
            for key, u in base.items():
                hs = h.codes_hash(u)
                if any(hs[j] != recB[j]['layers'][key]['codes'] for j in range(8)):
                    raise ValueError(f'all-B endpoint does not reproduce B codes at {key}, batch {s}')
            if topk(logitsB) != [r['top5'] for r in recB]:
                raise ValueError(f'all-B endpoint Top-5 mismatch batch {s}')
            for st_name in STAGES:
                label = 'Xstage2:' + st_name
                st, lg, _ = h.forward(x, ALL_B, st_name, None, stage_of)
                top = topk(lg)
                if st_name == 'head' and top != drec['configs']['Xstage:head']['top5']:
                    raise ValueError(f'corrected head hybrid differs from diag-1 Xstage:head, batch {s}')
                prop = {}
                for key, u in st.items():
                    b = base[key]
                    if u.shape != b.shape:
                        continue
                    up = int((u > b).sum()); down = int((u < b).sum())
                    if up or down:
                        prop[key] = [up, down]
                co = common_order(st[logit_node].reshape(8, -1).cpu().numpy())
                result['configs'][label] = {'top5': top, 'tie1': [c[0] for c in co], 'common_top5': [c[1] for c in co],
                                            'tie5': [c[2] for c in co], 'fc_hash': h.codes_hash(st[logit_node].reshape(8, -1)),
                                            'propagated_flips': prop}
        result['seconds'] = time.perf_counter() - tick
        immutable(path, result)
        if (s // 8) % 25 == 0:
            print(f'{name} stagefix {s + 8}/{stop} {result["seconds"]:.2f}s/batch', flush=True)
    print(json.dumps({'job': 'stagefix', 'format': name, 'start': start, 'stop': stop,
                      'wall_seconds': time.perf_counter() - started}), flush=True)


# ---- summary ---------------------------------------------------------------

def summarize_case(name):
    from tools.analysis.b_stage_balanced_comparisons import paired_outcomes
    from .analysis import correctness, directions
    from .ties import load_case
    folder = BASE / 'diagnostics_stagefix' / name
    recs = [unseal(p) for p in sorted(folder.glob('*.json'))]
    if [r['start'] for r in recs] != list(range(0, 8 * len(recs), 8)) or not recs:
        raise ValueError('noncontiguous stagefix batches')
    n = 8 * len(recs)
    t, labels, nt = load_case(name)
    if nt < n:
        raise ValueError('tie audit shorter than stagefix')
    labels = labels[:n]
    B = {k: t['B'][k][:n] for k in ('top5', 'common_top5', 'fc_hash')}
    out = {'format': name, 'images': n, 'seconds': sum(r['seconds'] for r in recs), 'hybrids': {}}
    for st_name in STAGES:
        label = 'Xstage2:' + st_name
        c = {k: sum((r['configs'][label][k] for r in recs), []) for k in ('top5', 'common_top5', 'tie1', 'fc_hash')}
        old = t['Xstage:' + st_name]
        row = {'top1_measured': 100 * sum(correctness(c['top5'], labels, 'top1')) / n,
               'top1_common_rule': 100 * sum(correctness(c['common_top5'], labels, 'top1')) / n,
               'fc_codes_differ_from_B_images': sum(a != b for a, b in zip(c['fc_hash'], B['fc_hash'])),
               'as_run_diag1': {'top1_measured': 100 * sum(correctness(old['top5'][:n], labels, 'top1')) / n,
                                'top1_common_rule': 100 * sum(correctness(old['common_top5'][:n], labels, 'top1')) / n,
                                'images_fc_hash_differs_from_corrected': sum(a != b for a, b in zip(old['fc_hash'][:n], c['fc_hash'])),
                                'images_top1_differs_from_corrected': sum(a[0] != b[0] for a, b in zip(old['top5'][:n], c['top5']))}}
        for rule, key in (('measured', 'top5'), ('common_rule', 'common_top5')):
            t1 = paired_outcomes(correctness(B[key], labels, 'top1'), correctness(c[key], labels, 'top1'))
            t5 = paired_outcomes(correctness(B[key], labels, 'top5'), correctness(c[key], labels, 'top5'))
            row['minus_B_' + rule] = {'top1_pp': t1['difference_pp'], 'top1_95': t1['pointwise_95_interval_pp'],
                                      'mcnemar_p': t1['mcnemar_exact_p'], 'top5_pp': t5['difference_pp'],
                                      'changes': directions(B[key], c[key], labels)}
        P = {}
        for r in recs:
            for k, v in r['configs'][label]['propagated_flips'].items():
                P.setdefault(k, [0, 0]); P[k][0] += v[0]; P[k][1] += v[1]
        row['propagated_flips_nodes'] = sorted(P)
        row['propagated_flips_total'] = [sum(v[0] for v in P.values()), sum(v[1] for v in P.values())]
        row['propagated_flips_fc'] = P.get('fc', [0, 0])
        out['hybrids'][label] = row
    return out


def write():
    if unseal(stagefix_protocol_path()) != STAGEFIX_PROTOCOL:
        raise ValueError('stagefix protocol drift')
    cases = [summarize_case(n) for n in FORMATS if (BASE / 'diagnostics_stagefix' / n).exists()]
    data = {'protocol': reference(stagefix_protocol_path()), 'evidence_level': STAGEFIX_PROTOCOL['evidence_level'],
            'certification': 'diagnostic harness; all-B endpoint and corrected head hybrid checked per batch', 'sources': sources(),
            'cases': cases}
    RESULTS.mkdir(parents=True, exist_ok=True)
    seal(RESULTS / 'stage_hybrids_corrected.json', data)
    for c in cases:
        print('##', c['format'], c['images'], f"{c['seconds']:.0f}s")
        for label, r in c['hybrids'].items():
            m, cr, o = r['minus_B_measured'], r['minus_B_common_rule'], r['as_run_diag1']
            print(f"{label:16s} meas {m['top1_pp']:+.1f} {m['top1_95']} common {cr['top1_pp']:+.1f} {cr['top1_95']} "
                  f"fc!=B {r['fc_codes_differ_from_B_images']} | as-run meas {o['top1_measured'] - c_b(c, r):+.1f} "
                  f"top1 differs {o['images_top1_differs_from_corrected']} fc differs {o['images_fc_hash_differs_from_corrected']}")
    return data


def c_b(case_row, row):
    """B measured top-1 % implied by a hybrid row (top1_measured - difference)."""
    return row['top1_measured'] - row['minus_B_measured']['top1_pp']


def main(args):
    if args[0] == 'protocol':
        immutable(stagefix_protocol_path(), STAGEFIX_PROTOCOL); print(stagefix_protocol_path())
    elif args[0] == 'run':
        run_case(args[1], int(args[2]), int(args[3]))
    elif args[0] == 'summarize':
        write()
    else:
        raise ValueError('unknown stagefix action')
