"""Independently audit and summarize prospective useful-quality E1 evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path

from scipy.stats import binomtest

from tools.breadth_study.useful_quality import BASE, CASES, assert_plan, measured_cost, paired_stats, validate_record
from tools.experiment_b.common import unseal, seal
from tools.phase3.common import ROOT, checked, read, reference

OUT = ROOT/'results/summaries/useful-quality-e1-v1'
DOC = ROOT/'docs/analysis/useful-quality-e1-v1.md'


def accuracy(records, field, rank):
    correct = sum(row['ground_truth'] in row[field][:rank] for row in records)
    n = len(records)
    interval = binomtest(correct, n).proportion_ci(confidence_level=.95, method='exact')
    return {'correct': correct, 'images': n, 'percent': 100*correct/n,
            'pointwise_95_interval_percent': [100*interval.low, 100*interval.high]}


def check_pairing(records, baseline, samples):
    if len(records) != len(samples):
        raise ValueError('panel length does not match declared samples')
    for index, (record, sample) in enumerate(zip(records, samples)):
        expected = baseline['records'][index]
        if (record['sample'] != sample or expected['sample_sha256'] != sample['sha256']
                or record['ground_truth'] != expected['ground_truth']
                or record['fp32_prediction'] != expected['fp32_prediction']
                or int(sample['label']) != expected['ground_truth']):
            raise ValueError('saved prediction/baseline/sample pairing drift')
        if any(type(value) is not int or not 0 <= value < 1000 for value in record['prediction']):
            raise ValueError('saved class index is invalid')


def arithmetic_envelope(graph):
    """Conditional finite-domain MAC-prefix bounds, never a whole-graph proof."""
    from fractions import Fraction
    from decimal import Decimal
    import numpy as np
    from public.inference.reference.arithmetic import format_named
    from public.analysis.phase3.fixed_mac_bounds import quantum
    def finite(value):
        return not isinstance(value, Decimal) or value.is_finite()
    encodings = {**graph['inputs'], **{name: tensor['encoding'] for name, tensor in graph['constants'].items()}}
    rows = []
    for node in graph['nodes']:
        attrs = node['attrs']; left = encodings[node['inputs'][0]]
        encodings[node['name']] = attrs.get('output', left)
        if node['op'] not in {'conv2d', 'depthwise_conv2d', 'linear'}:
            continue
        weight = graph['constants'][node['inputs'][1]]
        acc = format_named(attrs['accumulator'])
        if acc.family != 'integer' and (left['scales'] != ['1'] or weight['encoding']['scales'] != ['1']):
            rows.append({'node': node['name'], 'status': 'outside_unscaled_bound'}); continue
        xf, wf = format_named(left['format']), format_named(weight['encoding']['format'])
        x_decoded = [xf.decode(i) for i in range(1 << xf.bits)]
        w_decoded = [wf.decode(i) for i in range(1 << wf.bits)]
        x_nonfinite = [i for i, value in enumerate(x_decoded) if not finite(value)]
        w_nonfinite = [i for i, value in enumerate(w_decoded) if not finite(value)]
        x = [Fraction(value) for value in x_decoded if finite(value)]
        w = [Fraction(value) for value in w_decoded if finite(value)]
        if not x or not w:
            rows.append({'node': node['name'], 'status': 'empty_finite_code_domain'}); continue
        if node['inputs'][0] in graph['constants']:
            stored_input = graph['constants'][node['inputs'][0]]['codes']
            if any(int(code) in x_nonfinite for code in stored_input):
                rows.append({'node': node['name'], 'status': 'nonfinite_stored_input'}); continue
        qx, qw = quantum(x), quantum(w)
        xu = max(abs(v/qx) for v in x)
        table = np.asarray([int(abs(Fraction(v)/qw)) if finite(v) else -1 for v in w_decoded], dtype=np.int64)
        weights = np.asarray(weight['codes'], dtype=np.int64).reshape(weight['shape'][0], -1)
        if any(int(code) in w_nonfinite for code in np.unique(weights)):
            rows.append({'node': node['name'], 'status': 'nonfinite_stored_weight',
                         'nonfinite_stored_weight_codes': sorted(set(map(int, np.unique(weights))) & set(w_nonfinite))}); continue
        units = int(xu)*int(table[weights].sum(axis=1).max())
        q = qx*qw
        sufficient = units <= 2**24 and q >= Fraction(1, 2**149) and units*q < 2**127
        rows.append({'node': node['name'], 'status': 'conditional_finite_inputs',
                     'excluded_nonfinite_input_codes': x_nonfinite,
                     'input_finiteness_precondition': 'validate every runtime input code to this MAC',
                     'maximum_absolute_prefix_quantum_units': units,
                     'product_quantum': str(q), 'FP32_numeric_prefix_exact_sufficient_bound': sufficient})
    return {'MAC_bounds': rows, 'all_MAC_numeric_prefix_bounds_fit_FP32': all(
                r.get('FP32_numeric_prefix_exact_sufficient_bound', False) for r in rows),
            'scope': 'conditional on runtime MAC inputs containing only finite codes; conservative numeric MAC prefixes using finite actual stored weights; excludes signed-zero code behavior, bias/output stores and non-MAC operators; not whole-graph admission'}


def analyze():
    enrollment = unseal(BASE/'enrollment.json')
    protocol = unseal(checked(enrollment['protocol']))
    promotions = unseal(BASE/'promotion.json') if (BASE/'promotion.json').exists() else {'decisions': []}
    decisions = {row['case']: row for row in promotions['decisions']}
    ledger = []
    for item in enrollment['cases']:
        row = {'case': item['case'], 'static_status': item['status'], 'status': 'blocked', 'images': 0}
        if 'plan' not in item:
            row['reason'] = item['reason']; ledger.append(row); continue
        path = checked(item['plan']); plan = unseal(path); assert_plan(plan)
        archive = unseal(checked(plan['source_archive']))
        if archive['implementation'] != plan['sources']:
            raise ValueError('source archive does not match the executing plan')
        for ref in archive['files'].values():
            checked(ref)
        row.update(plan=reference(path), graph=plan['graph'], configuration=plan['configuration'],
                   mapping=plan['mapping'], static_proof=plan['static_proof'], costs=measured_cost(path))
        row['counts'] = {f'{mode}/{backend}': len(list(path.parent.glob(f'[0-9][0-9][0-9][0-9]-{mode}-{backend}.json')))
                         for mode in ('exact', 'control') for backend in ('cpp', 'cuda')}
        admission = path.parent/'native-admission.json'
        if admission.exists():
            gate = unseal(admission)
            if gate['plan'] != reference(path) or gate['exact_and_control_cpp_cuda_images'] != 8:
                raise ValueError('native admission identity/count drift')
            for ref in gate['records']:
                checked(ref)
            for mode in ('exact', 'control'):
                for index in range(8):
                    pair = [unseal(path.parent/f'{index:04d}-{mode}-{b}.json') for b in ('cpp', 'cuda')]
                    for backend, record in zip(('cpp', 'cuda'), pair):
                        validate_record(record, plan, protocol['sample_rows'][index], mode, backend, index)
                    if any(pair[0][k] != pair[1][k] for k in ('layers', 'output_sha256', 'prediction')):
                        raise ValueError('native pilot pair does not agree')
            row['native_admission'] = reference(admission)
        else:
            row['status'] = 'native_pilot_pending'
        panels = [n for n in (32, 128) if (path.parent/f'summary-{n}.json').exists()]
        if not panels:
            ledger.append(row); continue
        n = max(panels); summary_path = path.parent/f'summary-{n}.json'; summary = unseal(summary_path)
        if not admission.exists() or summary['images'] != n or summary['plan'] != reference(path):
            raise ValueError('completed panel lacks matching native admission')
        paired = {}
        for mode in ('exact', 'control'):
            paired[mode] = []
            for index, ref in enumerate(summary['records'][mode]):
                record = unseal(checked(ref))
                validate_record(record, plan, protocol['sample_rows'][index], mode, 'cuda', index)
                paired[mode].append(record)
            if len(paired[mode]) != n:
                raise ValueError('panel has duplicate or missing image references')
            check_pairing(paired[mode], read(checked(plan['baseline'])), protocol['sample_rows'][:n])
        for rank in (1, 5):
            vectors = {mode: [r['ground_truth'] in r['prediction'][:rank] for r in records]
                       for mode, records in paired.items()}
            control_gap = paired_stats(vectors['exact'], vectors['control'])
            if control_gap != summary['statistics'][f'top{rank}']['control_minus_exact']:
                raise ValueError('paired arithmetic statistic does not reproduce')
        for a, b in zip(paired['exact'], paired['control']):
            if a['sample'] != b['sample'] or a['ground_truth'] != b['ground_truth'] or a['fp32_prediction'] != b['fp32_prediction']:
                raise ValueError('panel arms are not paired on identical inputs and full-precision reference')
        mapping = unseal(checked(plan['mapping'])); ordinary = mapping['ordinary_B']
        broot = ROOT/'artifacts'/ordinary['component']; bidentity = ordinary['configuration_sha256']
        b_records = []
        for original in paired['exact']:
            b = unseal(broot/'predictions'/bidentity/(original['sample']['sha256']+'.json'))
            if b['configuration_sha256'] != bidentity or b['sample'] != original['sample']:
                raise ValueError('ordinary B descriptive panel does not match image identities')
            b_records.append({'prediction': b['top5'], 'ground_truth': original['ground_truth']})
        row.update(status='completed', images=n, panels=panels, summary=reference(summary_path),
                   statistics=summary['statistics'], all_layer_agreement_images=summary['all_layer_agreement_images'],
                   output_agreement_images=summary['output_agreement_images'],
                   zero_output_discordance_upper_95_percent=100*(1-.05**(1/n)) if summary['output_agreement_images'] == n else None,
                   first_divergences=dict(Counter(r['first_layer'] for r in summary['divergences'] if r['first_layer'])),
                   arithmetic_envelope=arithmetic_envelope(read(checked(plan['graph']))),
                   promotion=decisions.get(item['case']), ordinary_B=ordinary['configuration'],
                   quality={mode: {f'top{k}': accuracy(records, 'prediction', k) for k in (1, 5)}
                            for mode, records in {**paired, 'ordinary_B_different_configuration': b_records}.items()})
        row['quality']['full_precision'] = {f'top{k}': accuracy(paired['exact'], 'fp32_prediction', k) for k in (1, 5)}
        ledger.append(row)
    if [row['case'] for row in ledger] != [m+'/'+f for m, f in CASES]:
        raise ValueError('candidate ledger lost or substituted a frozen case')
    result = {'version': 'useful-quality-e1-analysis-v1', 'analysis_source': reference(Path(__file__)),
              'protocol': enrollment['protocol'], 'enrollment': reference(BASE/'enrollment.json'), 'candidates': ledger,
              'limitations': ['Canonical A and ordinary B use different recipes and graph/operator policies; their gap is not a pure arithmetic effect.',
                              'The strict/control comparison fixes graph, codes, scales, recipe and sample order; only declared reductions change.',
                              'Panels and promotions are exploratory development evidence; no final benchmark or equivalence claim.',
                              'Strict integer images retain their original accepted execution provenance; their old timings are not new-run throughput.']}
    OUT.mkdir(parents=True, exist_ok=True)
    seal(OUT/'analysis.json', result)
    with (OUT/'status.csv').open('w', newline='') as stream:
        fields = ['case', 'status', 'images', 'exact_top1', 'control_top1', 'FP32_top1', 'B_top1_different_configuration',
                  'all_layer_agreement_images', 'output_agreement_images']
        writer = csv.DictWriter(stream, fields); writer.writeheader()
        for row in ledger:
            item = {k: row.get(k) for k in ('case', 'status', 'images', 'all_layer_agreement_images', 'output_agreement_images')}
            if row['images']:
                for field, mode in [('exact_top1', 'exact'), ('control_top1', 'control'), ('FP32_top1', 'full_precision'),
                                    ('B_top1_different_configuration', 'ordinary_B_different_configuration')]:
                    item[field] = row['quality'][mode]['top1']['percent']
            writer.writerow(item)
    write_report(result)
    if any(row['images'] for row in ledger):
        plot(ledger)
    print(json.dumps([{'case': r['case'], 'status': r['status'], 'images': r['images']} for r in ledger], indent=2))
    return result


def write_report(result):
    lines = ['# Useful-quality matched exact-arithmetic development study', '',
             'The five cases use their canonical exact-A graphs. Ordinary B maxabs results are shown separately because scales, residual alignment, bias policy and framework execution differ. No B score is inherited by an exact graph. Exact and matched sequential-FP32 arms share the same graph, stored codes, scales, recipe and ordered images.', '',
             '| Case | Admission/panel status | Images | Strict top-1 | Matched control top-1 | FP32 top-1 | B top-1 (different configuration) | Layer/output agreement |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in result['candidates']:
        if not row['images']:
            lines.append(f"| {row['case']} | {row['status']} | 0 | — | — | — | — | — |")
            continue
        q = row['quality']
        values = ' | '.join(f"{q[mode]['top1']['percent']:.2f}%" for mode in ('exact', 'control', 'full_precision', 'ordinary_B_different_configuration'))
        lines.append(f"| {row['case']} | {row['status']} | {row['images']} | {values} | {row['all_layer_agreement_images']}/{row['images']} layers; {row['output_agreement_images']}/{row['images']} outputs |")
    lines += ['', 'Paired effects are matched-control minus strict, in percentage points. Absolute quality and losses to the paired full-precision baseline must be considered alongside arithmetic agreement.', '']
    for row in result['candidates']:
        if not row['images']:
            continue
        gap = row['statistics']['top1']['control_minus_exact']; lo, hi = gap['pointwise_95_interval_pp']
        loss = row['statistics']['top1']['exact_minus_FP32']['difference_pp']
        lines.append(f"- {row['case']}: arithmetic effect {gap['difference_pp']:+.2f} pp, pointwise 95% paired interval [{lo:+.2f}, {hi:+.2f}]; strict minus FP32 {loss:+.2f} pp. Promotion: {(row.get('promotion') or {}).get('reason', 'not yet decided')}.")
        if row['zero_output_discordance_upper_95_percent'] is not None:
            lines.append(f"  No output disagreement observed; the one-sided 95% binomial upper bound is {row['zero_output_discordance_upper_95_percent']:.2f}% on this panel, so zero observations do not prove equivalence.")
    lines += ['', 'All five candidates remain in the evidence ledger. The protocol, graph/recipe mapping, static proofs, eight-image native admissions, source archive and image checkpoints live under `artifacts/breadth_study/useful_quality_e1_v1/`. The analysis JSON contains top-1/top-5 counts, absolute intervals, paired statistics, first-divergence locations and runtime costs.', '',
              'Resume: `.venv/bin/python -m tools.run.useful_quality run`', '',
              'Audit/rebuild: `.venv/bin/python -m tools.analysis.useful_quality_e1`', '',
              'A completed admitted panel does not amend historical A acceptance, translate a B recipe, or complete full-benchmark confirmation.', '']
    DOC.write_text('\n'.join(lines))


def plot(rows):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    complete = [r for r in rows if r['images']]
    fig, ax = plt.subplots(figsize=(11, 6))
    styles = [('exact', 'Strict canonical A', '#196785'), ('control', 'Matched sequential FP32', '#db8244'),
              ('full_precision', 'Full-precision baseline', '#404040'),
              ('ordinary_B_different_configuration', 'B maxabs (different configuration)', '#9b78b4')]
    x = np.arange(len(complete))
    for i, (mode, label, color) in enumerate(styles):
        values = [r['quality'][mode]['top1']['percent'] for r in complete]
        ax.bar(x+(i-1.5)*.2, values, width=.19, label=label, color=color,
               hatch='//' if mode.startswith('ordinary') else None)
    ax.set_xticks(x, [r['case'].replace('/', '\n')+f"\nn={r['images']}" for r in complete])
    ax.set_ylabel('Top-1 accuracy (%)'); ax.set_ylim(0, 100)
    ax.set_title('Useful-quality matched arithmetic: development panels', loc='left')
    ax.legend(loc='upper left', fontsize=9, frameon=False, ncol=2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.text(.06, .02, 'Strict/control share graph and recipe. B is a different configuration; B-to-A gaps are not pure arithmetic effects.', fontsize=9)
    fig.subplots_adjust(bottom=.24, top=.9, left=.08, right=.98)
    for suffix in ('png', 'pdf'):
        fig.savefig(ROOT/f'results/figures/useful-quality-e1-v1.{suffix}', dpi=170, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    analyze()
