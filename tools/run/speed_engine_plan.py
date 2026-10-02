"""Lane S1 part 1c(ii): the (case, policy) test set of the fast exact-engine path and the archive runs it still needs.

  .venv/bin/python -m tools.run.speed_engine_plan [--write-queues]

Cases: every gated case of the two archives (the ResNet18 B2 cases and the two b1 cases on 7c6344af, the ten
MobileNet cases on 1f75c923). Policy families per case (protocol speed-protocol-v1): wide, control, f21, the fp16
rule exponent, sat.struct-<d>, sat.w at a width with saturation events and sat.w at a width past the collapse.
  ResNet18 B2: the archive's gated fp16 rule, sat.struct-<d> and sat.w (gate files of 7c6344af; every gated sat.w
    panel saturates); collapse = the narrowest sat.w with a sealed predict file over the first 64 images.
  ResNet18 b1 (no gate files in 7c6344af's root): the rule exponents of the accumulator sweep (int8 fp16.x-10,
    fp7 fp16.x-5), sat.struct-4, sat.w at structural width - 8 (events), narrowest sealed sat.w (collapse).
  MobileNets: the gated fp16 rule and sat.struct-<d> of 1f75c923 (int8: sat.struct-4), sat.w at the network
    structural width - 4 (events) and - 10 (collapse; checked on the records).
Coverage of the first 64 images by sealed archive predict files (read-only) or by earlier fresh archive runs under
artifacts/speed_v1/archive-runs/; the rest are fresh archive runs (tools.run.speed_archive_predict).
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from tools.experiment_b.common import ROOT, unseal

V2 = ROOT / 'artifacts/scaled_bridge_v2'
R7 = '7c6344af732d1bf324c07b285921f63393150cdb3875356c86237d934fdc2f92'
R1 = '1f75c9232c8a0202482fe9bce9a46360c7cc0999c5b0bcf7df5ebcc0446fc863'
SPEED = ROOT / 'artifacts/speed_v1'
IMAGES = 64
RESNET = ['resnet18-int8-default-b2', 'resnet18-int6-default-b2', 'resnet18-fp6_e2m3-default-b2',
          'resnet18-fp7_e3m3-default-b2', 'resnet18-fp8_e4m3fn-default-b2', 'resnet18-fp8_e5m2-default-b2',
          'resnet18-int8-default_signed-b2', 'resnet18-posit8_es1-default-b2',
          'resnet18-int8-maxabs-b1', 'resnet18-fp7_e3m3-maxabs-b1']
MOBILE = [f'{m}-{f}-default-b2' for m in ('mobilenet_v2', 'mobilenet_v3_large')
          for f in ('int8', 'int6', 'fp6_e2m3', 'fp7_e3m3', 'fp8_e4m3fn')]
B1 = {'resnet18-int8-maxabs-b1': ('fp16.x-10', 25), 'resnet18-fp7_e3m3-maxabs-b1': ('fp16.x-5', 30)}


def archive_of(case):
    return R7 if case.startswith('resnet18-') else R1


def covered(case, policy, start=0, stop=IMAGES):
    seen = set()
    for root in (V2 / 'runs' / archive_of(case), SPEED / 'archive-runs' / archive_of(case)):
        for path in (root / case / 'predictions').glob(f'{policy}-cuda-*.json'):
            a, b = (int(x) for x in path.stem.split('-')[-2:])
            seen.update(range(max(a, start), min(b, stop)))
    return len(seen) == stop - start


def gated(case, prefix):
    root = V2 / 'runs' / archive_of(case) / case
    names = sorted(p.stem[5:] for p in root.glob('gate-*.json') if unseal(p)['status'] == 'pass')
    return [n for n in names if n == prefix or n.startswith(prefix + '.') or n.startswith(prefix + '-')
            or (prefix == 'sat.w' and re.fullmatch(r'sat\.w\d+', n))]


def structural(case):
    certs = unseal(V2 / 'runs' / archive_of(case) / case / 'certificate.json')['certificates'].values()
    return max(c['signed_bits_structural'] for c in certs)


def narrowest_sealed(case):
    root = V2 / 'runs' / archive_of(case) / case / 'predictions'
    widths = sorted({int(m.group(1)) for p in root.glob('sat.w*-cuda-*.json')
                     if (m := re.match(r'sat\.w(\d+)-cuda', p.name))})
    return next(w for w in widths if covered(case, f'sat.w{w}'))


def policies(case):
    if case in B1:
        rule, width = B1[case]
        return {'wide': 'wide', 'control': 'control', 'f21': 'f21', 'fp16_rule': rule, 'sat_struct': 'sat.struct-4',
                'sat_w_events': f'sat.w{width - 8}', 'sat_w_collapse': f'sat.w{narrowest_sealed(case)}'}
    fp16 = [p for p in gated(case, 'fp16')]
    struct = [p for p in gated(case, 'sat.struct') if p != 'sat.struct-0']
    if len(fp16) != 1 or len(struct) != 1:
        raise ValueError(f'{case}: expected one gated fp16 rule and one sat.struct-<d>: {fp16} {struct}')
    out = {'wide': 'wide', 'control': 'control', 'f21': 'f21', 'fp16_rule': fp16[0], 'sat_struct': struct[0]}
    if case.startswith('resnet18-'):
        satw = gated(case, 'sat.w')
        out.update(sat_w_events=satw[0], sat_w_collapse=f'sat.w{narrowest_sealed(case)}')
    else:
        s = structural(case)
        out.update(sat_w_events=f'sat.w{s - 4}', sat_w_collapse=f'sat.w{s - 10}')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write-queues', action='store_true')
    args = p.parse_args()
    plan = {}
    for case in RESNET + MOBILE:
        pol = policies(case)
        plan[case] = {'archive': archive_of(case), 'policies': pol,
                      'missing': [v for v in pol.values() if not covered(case, v)]}
    print(json.dumps({c: {'policies': list(v['policies'].values()), 'missing': v['missing']} for c, v in plan.items()}, indent=0))
    path = SPEED / 'validation' / 'plan.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=1))
    if args.write_queues:
        lines = {'a': [], 'b': []}
        for i, (case, v) in enumerate((c, v) for c, v in plan.items() if v['missing']):
            lines['ab'[i % 2]].append(f'--min-free-mib 3000 .venv-b/bin/python -m tools.run.speed_archive_predict '
                                      f'{v["archive"]} {case} 0 {IMAGES} 8 {" ".join(v["missing"])}')
        for k, ls in lines.items():
            (SPEED / 'queues' / f'q-archive-{k}.txt').write_text('\n'.join(ls) + '\n')
            print(k, len(ls))


if __name__ == '__main__':
    main()
