"""Summaries of sealed v2 evidence (no measurement happens here).

Published summaries are written once. Every call computes the summary for an
explicit run root (engine source digest) and an explicit ledger cut-off, and
writes it as a new revision (`summary.json` is revision 1, then
`summary-r2.json`, ...) unless an existing revision already has exactly that
content. Nothing is ever replaced.
"""
from __future__ import annotations
from collections import Counter
import statistics
from .common import ROOT, BASE, digest, unseal, seal, reference, engine_sources, run_root, case_id

POLICY_ARMS = ('wide', 'control')
_ROOT = None


def root():
    """Run root the report reads: the current engine digest unless one was selected."""
    return _ROOT or run_root()

M1_CASES = [('resnet18', f, 'maxabs') for f in (
    'int4', 'int5', 'int6', 'int8', 'q1_6', 'fp4_e2m1', 'fp5_e2m2', 'fp6_e2m3', 'fp6_e3m2', 'fp7_e3m3', 'fp8_e4m3fn',
    'fp8_e5m2', 'posit4_es0', 'posit6_es1', 'posit8_es1', 'ternary', 'binary_pm1')] + [
    ('resnet18', f, 'percentile_99_9') for f in ('int4', 'int5', 'int6', 'int8', 'fp4_e2m1', 'fp5_e2m2')]
MILESTONES = {'M1': M1_CASES}
V1_FORMATS = ('fp6_e2m3', 'fp6_e3m2', 'fp7_e3m3')


def records(case, arm):
    return [unseal(p) for p in sorted((root() / case / arm).glob('[0-9]*.json'))]


def tie_aware(rows):
    """Expected Top-1 percent under uniform tie-breaking among classes tied at the maximum."""
    credit = [(int(r['sample']['label']) in r['top1_tied_classes']) / r['top1_tie_count'] for r in rows]
    return 100 * sum(credit) / len(credit)


def first_divergence(left, right, keys, order):
    """First differing node in export execution order (sealed JSON sorts keys alphabetically)."""
    first = Counter(); differing = 0
    for a, b in zip(left, right):
        for node in order:
            x = a['layers'][node]; y = b['layers'][node]
            if any(x.get(k) != y.get(k) for k in keys):
                first[node] += 1; differing += 1
                break
    return {'images_differing': differing, 'first_differing_node_images': dict(first)}


def v1_regression(fmt, wide, control):
    """v2 against the sealed v1 records: everything except the Top-5 tie order must be identical."""
    from tools.scaled_bridge_v1 import common as v1
    result = {}
    for arm, rows in (('wide', wide), ('control', control)):
        old = [unseal(p) for p in sorted((v1.run_root() / fmt / f'{arm}-cuda').glob('[0-9]*.json'))][:len(rows)]
        same = sum(a['layers'] == b['layers'] and a['output'] == b['output'] for a, b in zip(rows, old))
        result[arm] = {'compared_images': len(old), 'identical_layers_and_output': same,
                       'identical_top5': sum(a['top5'] == b['top5'] for a, b in zip(rows, old)),
                       'identical_top1': sum(a['top5'][0] == b['top5'][0] for a, b in zip(rows, old))}
    return result


def case_summary(model, fmt, recipe):
    from tools.scaled_bridge_v1.report import paired
    case = case_id(model, fmt, recipe); folder = root() / case
    row = {'case': case, 'model': model, 'format': fmt, 'recipe': recipe, 'gate': (folder / 'gate.json').exists()}
    if not (folder / 'certificate.json').exists():
        return row
    certificate = unseal(folder / 'certificate.json'); certs = certificate['certificates']
    export = unseal(ROOT / certificate['export']['path'])
    row['export'] = certificate['export']; row['historical_B'] = export['retained']['historical']
    row['layers'] = [{k: c[k] for k in ('node', 'K', 'signed_bits_absolute', 'signed_bits_range', 'signed_bits_structural',
                                         'binary64_exact_dot', 'exact_accumulator', 'fp32_all_prefix_exact_sufficient')}
                     for c in certs.values()]
    row['max_signed_bits'] = {k: max(c[k] for c in certs.values())
                              for k in ('signed_bits_absolute', 'signed_bits_range', 'signed_bits_structural')}
    row['layers_exact_in_binary64'] = sum(c['binary64_exact_dot'] for c in certs.values())
    row['layers_two_limb'] = sum(c['kernel_operand_bits'] == 64 for c in certs.values())
    row['mac_layers'] = len(certs)
    arms = {f'{p}-{b}': records(case, f'{p}-{b}') for p in POLICY_ARMS for b in ('cpp', 'cuda')}
    row['counts'] = {k: len(v) for k, v in arms.items()}
    row['timings'] = {k: {'median_execution_seconds': statistics.median(r['timing']['execution'] - r['timing']['oracle'] for r in v),
                          'median_mac_seconds': statistics.median(r['timing']['mac'] for r in v),
                          'all_under_gpu_lock': all(r['gpu_lock_declared'] for r in v)}
                      for k, v in arms.items() if v}
    wide, control = arms['wide-cuda'], arms['control-cuda']; n = min(len(wide), len(control))
    replayed = records(case, 'B'); row['B_replayed'] = len(replayed)
    order = [x['name'] for x in export['nodes'] if x['op'] != 'output']
    if n:
        wide, control = wide[:n], control[:n]; labels = [int(r['sample']['label']) for r in wide]
        predictions = {'wide': [r['top5'] for r in wide], 'control': [r['top5'] for r in control],
                       'B': [r['top5'] for r in export['retained']['B_prefix'][:n]],
                       'FP32': [r['top5'] for r in export['retained']['FP32_prefix'][:n]]}
        row['paired_images'] = n
        row['comparisons'] = {a + '_minus_' + b: paired(predictions[a], predictions[b], labels)
                              for a, b in (('wide', 'B'), ('control', 'B'), ('wide', 'FP32'), ('control', 'FP32'), ('control', 'wide'))}
        m = min(n, len(replayed))
        row['tie_aware'] = {'images': m, 'wide_top1': tie_aware(wide[:m]), 'control_top1': tie_aware(control[:m]),
                            'B_top1': tie_aware(replayed[:m]) if m else None,
                            'wide_images_with_tied_top1': sum(r['top1_tie_count'] > 1 for r in wide[:m]),
                            'B_images_with_tied_top1': sum(r['top1_tie_count'] > 1 for r in replayed[:m])} if m else None
        row['control_vs_wide'] = {'raw_dot': first_divergence(control, wide, ('mac',), order),
                                  'stored_codes': first_divergence(control, wide, ('codes',), order)}
        row['wide_vs_B_codes'] = first_divergence(wide[:m], replayed[:m], ('codes',), order)
        row['clipping'] = {'wide': sum(v['diagnostics'].get('clipped_low', 0) + v['diagnostics'].get('clipped_high', 0)
                                       for r in wide for v in r['layers'].values())}
        if model == 'resnet18' and recipe == 'maxabs' and fmt in V1_FORMATS:
            row['v1_regression'] = v1_regression(fmt, wide, control)
    return row


def revisions(folder, stem, suffix):
    """Existing revisions in order: stem.suffix is revision 1, stem-rN.suffix is revision N."""
    found = {}
    if (folder / f'{stem}.{suffix}').exists():
        found[1] = folder / f'{stem}.{suffix}'
    for path in folder.glob(f'{stem}-r*.{suffix}'):
        found[int(path.stem.rsplit('-r', 1)[1])] = path
    return dict(sorted(found.items()))


def publish(folder, data, table):
    """Write a new revision, or report the existing revision with identical content."""
    old = revisions(folder, 'summary', 'json')
    for number, path in old.items():
        if unseal(path) == data:
            print(f'unchanged: identical to revision {number} ({path})')
            return path
    number = max(old, default=0) + 1
    name = 'summary.json' if number == 1 else f'summary-r{number}.json'
    path = folder / name
    if path.exists():
        raise ValueError('refusing to replace a published summary')
    folder.mkdir(parents=True, exist_ok=True)
    seal(path, data)
    table_path = folder / ('table.txt' if number == 1 else f'table-r{number}.txt')
    if table_path.exists():
        raise ValueError('refusing to replace a published table')
    table_path.write_text(table)
    print(f'wrote revision {number}: {path}')
    return path


def cost(cutoff=None):
    """Ledger totals up to a recorded cut-off (epoch seconds of the last included invocation)."""
    ledger = [unseal(p) for p in sorted((root() / 'ledger').glob('*.json'))]
    if cutoff is None:
        cutoff = max((r['started_epoch'] for r in ledger), default=0.)
    kept = [r for r in ledger if r['started_epoch'] <= cutoff]
    return {'ledger_cutoff_epoch': cutoff, 'invocations': len(kept),
            'invocations_after_cutoff': len(ledger) - len(kept),
            'wall_seconds': sum(r['wall_seconds'] for r in kept),
            'wall_seconds_under_gpu_lock': sum(r['wall_seconds'] for r in kept if r['gpu_lock_declared']),
            'by_label': {k: sum(r['wall_seconds'] for r in kept if r['label'] == k)
                         for k in sorted({r['label'] for r in kept})}}


def conformance_checks():
    return {p.stem: unseal(p)['checks'] for p in sorted((root() / 'conformance').glob('*.json'))}


def source_list(milestone, name):
    """Engine source list of a run root: the live one if it has that digest, else the one published first."""
    if name == digest(engine_sources()):
        return engine_sources()
    first = ROOT / 'results/summaries/scaled-bridge-v2' / f'{milestone}-{name[:16]}' / 'summary.json'
    saved = unseal(first)['engine_sources'] if first.exists() else None
    if not isinstance(saved, dict) or digest(saved) != name:
        return 'the run root predates the current sources and no published revision lists them'
    return saved


def write_report(milestone, run=None, cutoff=None):
    global _ROOT
    if run is not None:
        matches = [p for p in (BASE / 'runs').glob(run + '*') if p.is_dir()]
        if len(matches) != 1:
            raise ValueError('run digest prefix does not select exactly one run root')
        _ROOT = matches[0]
    name = root().name
    cases = [case_summary(*c) for c in MILESTONES[milestone]]
    for c in cases:
        if 'comparisons' in c:
            changed = c['comparisons']['control_minus_wide']
            c['control_vs_wide']['top1_class_changed_images'] = len(changed['changed_top1_indices'])
            c['control_vs_wide']['ordered_top5_changed_images'] = len(changed['changed_top5_indices'])
    data = {'milestone': milestone, 'engine_sources_digest': name,
            'engine_sources': source_list(milestone, name),
            'cases': cases, 'gates_passed': sum(c['gate'] for c in cases), 'cases_total': len(cases),
            'primitive_conformance_checks': conformance_checks(), 'cost': cost(cutoff)}
    folder = ROOT / 'results/summaries/scaled-bridge-v2' / f'{milestone}-{name[:16]}'
    lines = ['case | gate | bits abs/range/struct | f64-exact layers | 2-limb | n | B top1 | wide top1 | wide-B pp [95%] | ctl-wide pp | ctl!=wide top1 class | tie-aware wide/B | CUDA s/img wide/ctl | CPU s/img wide']
    for c in cases:
        if 'comparisons' not in c:
            lines.append(f"{c['case']} | {c['gate']} | incomplete"); continue
        wb = c['comparisons']['wide_minus_B']['top1']; cw = c['comparisons']['control_minus_wide']['top1']
        b = c['max_signed_bits']; t = c['timings']; ta = c['tie_aware']
        lines.append(f"{c['case']} | {c['gate']} | {b['signed_bits_absolute']}/{b['signed_bits_range']}/{b['signed_bits_structural']} | "
                     f"{c['layers_exact_in_binary64']}/{c['mac_layers']} | {c['layers_two_limb']} | {c['paired_images']} | "
                     f"{wb['right_percent']:.2f} | {wb['left_percent']:.2f} | {wb['delta_pp']:+.2f} [{wb['paired_pointwise95_pp'][0]:+.2f},{wb['paired_pointwise95_pp'][1]:+.2f}] | "
                     f"{cw['delta_pp']:+.2f} | {c['control_vs_wide']['top1_class_changed_images']} | {ta['wide_top1']:.2f}/{ta['B_top1']:.2f} | "
                     f"{t['wide-cuda']['median_execution_seconds']:.3f}/{t['control-cuda']['median_execution_seconds']:.3f} | "
                     f"{t['wide-cpp']['median_execution_seconds']:.3f}")
    publish(folder, data, '\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return data


MS_CASES = ('resnet18-int8-maxabs-b1', 'resnet18-fp7_e3m3-maxabs-b1')
MS_POLICIES = ('sat.abs-0', 'sat.struct-0', 'sat.struct-4', 'sat.w16', 'fp16', 'fp16.x-12', 'f21')
# PROTOCOL_MS bounds. The MS run root is also the root an archived implementation writes into, so the summary
# reads the protocol prefixes only and the ledger up to the recorded start of the last MS invocation. Whatever
# another lane adds to that root later (longer panels, throughput runs, regressions) cannot change the summary.
MS_PANEL_IMAGES, MS_GATE_IMAGES = 32, 8
MS_RUN = '99de2758260f852ebaf5fafeb83e8bce48f84cbf8b7cefdb38b5f80f68bbb05b'
MS_LEDGER_CUTOFF = 1790861431.6161683
MS_REGRESSION = ('wide', 'control'), 'cuda', 8, '2a64216184ac39ea508407b941d53188d726b5dc30b5206a570b9fe105d2ecf3'


def policy_summary(case, policy, cuda_images=None, cpu_images=None):
    cuda_images = cuda_images or MS_PANEL_IMAGES; cpu_images = cpu_images or MS_GATE_IMAGES
    folder = root() / case
    row = {'case': case, 'policy': policy, 'gate': (folder / f'gate-{policy}.json').exists()}
    rows = records(case, f'{policy}-cuda')[:cuda_images]; wide = records(case, 'wide-cuda')[:len(rows)]
    if not rows:
        return row
    export = unseal(ROOT / rows[0]['export']['path'])
    order = [x['name'] for x in export['nodes'] if x['op'] != 'output']
    labels = [int(r['sample']['label']) for r in rows]
    def top1(items):
        return 100 * sum(r['top5'] is not None and r['top5'][0] == y for r, y in zip(items, labels)) / len(items)
    ok = [(r, w) for r, w in zip(rows, wide) if r['failure'] is None]
    nodes = {}
    for key in rows[0]['accumulator'] if rows[0]['accumulator'] else []:
        stats = [r['accumulator'][key] for r in rows if key in r['accumulator']]
        hit = 'saturated_elements' if 'saturated_elements' in stats[0] else 'nonfinite_elements'
        nodes[key] = {'width': stats[0].get('width'), 'images_reaching_node': len(stats),
                      'images_with_event': sum(v[hit] > 0 for v in stats),
                      'mean_fraction_of_outputs_with_event': sum(v[hit] / v['elements'] for v in stats) / len(stats)}
    cpu = records(case, f'{policy}-cpp')[:cpu_images]
    row.update({
        'images': len(rows), 'images_failed_nonfinite': len(rows) - len(ok),
        'failure_nodes': dict(Counter(r['failure']['node'] for r in rows if r['failure'] is not None)),
        'top1_percent_failures_incorrect': top1(rows), 'wide_top1_percent': top1(wide),
        'top1_class_differs_from_wide_images': sum(r['top5'] is None or r['top5'][0] != w['top5'][0] for r, w in zip(rows, wide)),
        'stored_codes_differ_from_wide': first_divergence([r for r, _ in ok], [w for _, w in ok], ('codes',), order),
        'raw_dot_differs_from_wide': first_divergence([r for r, _ in ok], [w for _, w in ok], ('mac',), order),
        'images_with_any_event': sum(any(v.get('saturated_elements', 0) + v.get('nonfinite_elements', 0) for v in r['accumulator'].values()) for r in rows),
        'nodes_with_event': {k: v for k, v in nodes.items() if v['images_with_event']},
        'widths': {k: v['width'] for k, v in nodes.items()} if policy.startswith('sat') else None,
        'median_execution_seconds': {'cuda': statistics.median(r['timing']['execution'] - r['timing']['oracle'] for r in rows),
                                     'cpp': statistics.median(r['timing']['execution'] - r['timing']['oracle'] for r in cpu) if cpu else None},
        'cpu_images': len(cpu)})
    return row


def write_policy_report(run=MS_RUN, cutoff=MS_LEDGER_CUTOFF):
    """MS summary, bounded to PROTOCOL_MS: prefix 32 (CUDA) and 8 (CPU), the four protocol regressions, ledger cut-off."""
    global _ROOT
    matches = [p for p in (BASE / 'runs').glob(run + '*') if p.is_dir()]
    if len(matches) != 1:
        raise ValueError('run digest prefix does not select exactly one run root')
    _ROOT = matches[0]
    name = root().name
    rows = [policy_summary(c, p) for c in MS_CASES for p in MS_POLICIES]
    arms, backend, images, old = MS_REGRESSION
    regress = [unseal(root() / 'regress' / f'{c}-{a}-{backend}-{images}-{old[:16]}.json') for c in MS_CASES for a in arms
               if (root() / 'regress' / f'{c}-{a}-{backend}-{images}-{old[:16]}.json').exists()]
    regress.sort(key=lambda r: (r['case'], r['policy']))
    spent = cost(cutoff); later = spent.pop('invocations_after_cutoff')
    print(f'ledger invocations after the cut-off (not part of MS, not in the summary): {later}')
    data = {'milestone': 'MS', 'engine_sources_digest': name, 'policies': rows,
            'gates_passed': sum(r['gate'] for r in rows), 'gates_total': len(rows),
            'policy_conformance': {k: unseal(root() / 'conformance' / 'policies.json')[k] for k in ('status', 'checks', 'by_class')},
            'regression_against_earlier_engine': [{k: r[k] for k in ('case', 'policy', 'backend', 'images_identical', 'old_engine_sources')} for r in regress],
            'cost': spent,
            'bounds': {'panel_images_cuda': MS_PANEL_IMAGES, 'gate_images_cpu': MS_GATE_IMAGES,
                       'ledger_cutoff_epoch': cutoff, 'regressions': 'wide and control, CUDA, 8 images, against ' + old[:16],
                       'note': 'records, ledger entries and regressions beyond these bounds are not read'}}
    lines = ['case | policy | gate | failed/32 | top1 (wide) | top1 class != wide | images with stored-code change | images with event | nodes with event | CUDA s/img | CPU s/img']
    for r in rows:
        if 'images' not in r:
            lines.append(f"{r['case']} | {r['policy']} | {r['gate']} | incomplete"); continue
        t = r['median_execution_seconds']
        lines.append(f"{r['case']} | {r['policy']} | {r['gate']} | {r['images_failed_nonfinite']}/{r['images']} | "
                     f"{r['top1_percent_failures_incorrect']:.2f} ({r['wide_top1_percent']:.2f}) | {r['top1_class_differs_from_wide_images']} | "
                     f"{r['stored_codes_differ_from_wide']['images_differing']} | {r['images_with_any_event']} | "
                     f"{len(r['nodes_with_event'])} | {t['cuda']:.3f} | {t['cpp'] if t['cpp'] is None else round(t['cpp'], 2)}")
    publish(ROOT / 'results/summaries/scaled-bridge-v2' / f'MS-{name[:16]}', data, '\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return data


MB2_CASES = tuple(f'resnet18-{f}-default-b2' for f in ('int8', 'int6', 'fp6_e2m3', 'fp7_e3m3', 'fp8_e4m3fn'))
MB2_PANEL_IMAGES, MB2_GATE_IMAGES = 32, 8      # PROTOCOL_MB2 prefixes; nothing beyond them is read
MB2_RUN = '7c6344af732d1bf324c07b285921f63393150cdb3875356c86237d934fdc2f92'
MB2_LEDGER_CUTOFF = 1790864059.7781281   # start of the last MB2 invocation (the final replay)


def b2_case_summary(case):
    from tools.scaled_bridge_v1.report import paired
    folder = root() / case
    row = {'case': case, 'gate': (folder / 'gate.json').exists()}
    if not (folder / 'certificate.json').exists():
        return row
    certificate = unseal(folder / 'certificate.json'); certs = certificate['certificates']
    export = unseal(ROOT / certificate['export']['path'])
    widths = ('signed_bits_absolute', 'signed_bits_range', 'signed_bits_structural')
    row.update({'export': certificate['export'], 'format': export['format'], 'recipe': export['recipe'],
                'B2_configuration_sha256': export['provenance']['B2_configuration_sha256'],
                'B2_recipe': export['provenance']['B2_recipe'], 'codebooks': sorted(export['codebooks']),
                'max_signed_bits': {k: max(c[k] for c in certs.values()) for k in widths},
                'widest_nodes': {k: [c['node'] for c in certs.values() if c[k] == max(d[k] for d in certs.values())] for k in widths},
                'layers': [{k: c[k] for k in ('node', 'K', 'input_codebook', 'weight_codebook', 'product_shift', 'input_range_units',
                                              'structural_input_range_units', 'max_abs_prefix_units') + widths
                            + ('binary64_exact_dot', 'exact_accumulator')} for c in certs.values()],
                'mac_layers': len(certs), 'layers_exact_in_binary64': sum(c['binary64_exact_dot'] for c in certs.values()),
                'layers_two_limb': sum(c['kernel_operand_bits'] == 64 for c in certs.values()),
                'boundaries': {'storing': sum(n.get('store') is not None for n in export['nodes']),
                               'unstored_arithmetic': sum(n.get('store') is None and n['op'] in ('conv', 'linear', 'add', 'relu') for n in export['nodes']),
                               'code_forwarding_maxpool': sum(n.get('store') is None and n['op'] == 'maxpool' for n in export['nodes'])}})
    arms = {f'{p}-{b}': records(case, f'{p}-{b}')[:MB2_PANEL_IMAGES if b == 'cuda' else MB2_GATE_IMAGES]
            for p in POLICY_ARMS for b in ('cpp', 'cuda')}
    row['counts'] = {k: len(v) for k, v in arms.items()}
    replayed = records(case, 'B2')[:MB2_PANEL_IMAGES]; row['B2_replayed'] = len(replayed)
    wide, control = arms['wide-cuda'], arms['control-cuda']
    n = min(len(wide), len(control), len(replayed))
    if not n:
        return row
    wide, control, replayed = wide[:n], control[:n], replayed[:n]
    labels = [int(r['sample']['label']) for r in wide]
    order = [x['name'] for x in export['nodes'] if x['op'] != 'output']
    stored = [x['name'] for x in export['nodes'] if x.get('store') is not None]
    top = {'wide': [r['top5'] for r in wide], 'control': [r['top5'] for r in control],
           'B2_retained_order': [r['top5'] for r in replayed], 'B2_lowest_index': [r['stable_top5'] for r in replayed]}
    row['paired_images'] = n
    row['comparisons'] = {
        'wide_minus_B2_retained_order': paired(top['wide'], top['B2_retained_order'], labels),
        'wide_minus_B2_lowest_index': paired(top['wide'], top['B2_lowest_index'], labels),
        'control_minus_wide': paired(top['control'], top['wide'], labels)}
    row['tie_aware'] = {'wide_top1': tie_aware(wide), 'control_top1': tie_aware(control), 'B2_top1': tie_aware(replayed),
                        'wide_images_with_tied_top1': sum(r['top1_tie_count'] > 1 for r in wide),
                        'B2_images_with_tied_top1': sum(r['top1_tie_count'] > 1 for r in replayed),
                        'B2_images_where_the_two_orders_give_another_top1': sum(r['top5'][0] != r['stable_top5'][0] for r in replayed),
                        'B2_images_where_the_two_orders_give_another_top5': sum(r['top5'] != r['stable_top5'] for r in replayed)}
    changed = {k: [sum(r['changed_codes_vs_wide'][k][0] for r in replayed), sum(r['changed_codes_vs_wide'][k][1] for r in replayed)]
               for k in stored}
    total = [sum(v[0] for v in changed.values()), sum(v[1] for v in changed.values())]
    row['wide_vs_B2'] = {
        'first_differing_boundary': first_divergence(wide, replayed, ('codes',), stored),
        'images_with_identical_codes_everywhere': sum(all(v[0] == 0 for v in r['changed_codes_vs_wide'].values()) for r in replayed),
        'changed_codes': total[0], 'compared_codes': total[1], 'changed_fraction': total[0] / total[1],
        'changed_fraction_by_boundary': {k: v[0] / v[1] for k, v in changed.items()},
        'images_with_identical_logit_codes': sum(r['changed_codes_vs_wide'][stored[-1]][0] == 0 for r in replayed),
        'max_abs_logit_difference': max(r['max_abs_logit_difference_vs_wide'] for r in replayed),
        'B2_retained_top5_reproduced_images': sum(r['retained_top5'] == r['top5'] for r in replayed),
        'inputs_bit_identical_images': sum(r['inputs_bit_identical_to_engine_inputs'] for r in replayed)}
    row['control_vs_wide'] = {'raw_dot': first_divergence(control, wide, ('mac',), order),
                              'stored_codes': first_divergence(control, wide, ('codes',), order),
                              'top1_class_changed_images': len(row['comparisons']['control_minus_wide']['changed_top1_indices']),
                              'ordered_top5_changed_images': len(row['comparisons']['control_minus_wide']['changed_top5_indices'])}
    row['clipping_wide'] = sum(v['diagnostics'].get('clipped_low', 0) + v['diagnostics'].get('clipped_high', 0)
                               for r in wide for v in r['layers'].values())
    row['oracle_dot_checks_first_image'] = {k: v[0]['oracle_dot_checks'] for k, v in arms.items() if k.endswith('cpp') and v}
    row['indicative_seconds_per_image'] = {k: statistics.median(r['timing']['execution'] - r['timing']['oracle'] for r in v)
                                           for k, v in arms.items() if v}
    return row


def write_b2_report(run=None, cutoff=None):
    """MB2 summary, bounded to PROTOCOL_MB2 (prefix 32 on CUDA and in the replay, 8 on CPU) and a ledger cut-off."""
    global _ROOT
    run = run or MB2_RUN; cutoff = cutoff or MB2_LEDGER_CUTOFF
    if run is None or cutoff is None:
        raise ValueError('the MB2 report needs a run digest and a ledger cut-off')
    matches = [p for p in (BASE / 'runs').glob(run + '*') if p.is_dir()]
    if len(matches) != 1:
        raise ValueError('run digest prefix does not select exactly one run root')
    _ROOT = matches[0]; name = root().name
    cases = [b2_case_summary(c) for c in MB2_CASES]
    books = sorted({b for c in cases for b in c.get('codebooks', [])})
    spent = cost(cutoff); later = spent.pop('invocations_after_cutoff')
    print(f'ledger invocations after the cut-off (not in the summary): {later}')
    data = {'milestone': 'MB2', 'engine_sources_digest': name, 'engine_sources': source_list('MB2', name),
            'cases': cases, 'gates_passed': sum(c['gate'] for c in cases), 'cases_total': len(cases),
            'primitive_conformance_checks': {b: unseal(root() / 'conformance' / f'{b}.json')['checks'] for b in books
                                             if (root() / 'conformance' / f'{b}.json').exists()},
            'tie_rule': 'contract order: descending value, lowest class index first. B2 retained order: torch.topk on '
                        'CUDA FP32 logits. Both are reported for the B2 side; the engine side always uses the contract order',
            'cost': spent,
            'bounds': {'panel_images_cuda': MB2_PANEL_IMAGES, 'gate_images_cpu': MB2_GATE_IMAGES, 'replay_images': MB2_PANEL_IMAGES,
                       'ledger_cutoff_epoch': cutoff, 'note': 'records and ledger entries beyond these bounds are not read'}}
    lines = ['case | gate | bits abs/range/struct | n | B2 top1 retained/lowest-index | wide top1 | wide-B2 pp [95%] retained order | '
             'wide-B2 pp [95%] lowest index | tie-aware wide/B2 | ctl-wide pp | ctl!=wide top1 class | images with all codes = B2 | '
             'changed codes | CUDA s/img wide (indicative)']
    for c in cases:
        if 'comparisons' not in c:
            lines.append(f"{c['case']} | {c['gate']} | incomplete"); continue
        a = c['comparisons']['wide_minus_B2_retained_order']['top1']; b = c['comparisons']['wide_minus_B2_lowest_index']['top1']
        cw = c['comparisons']['control_minus_wide']['top1']; w = c['max_signed_bits']; ta = c['tie_aware']; d = c['wide_vs_B2']
        lines.append(
            f"{c['case']} | {c['gate']} | {w['signed_bits_absolute']}/{w['signed_bits_range']}/{w['signed_bits_structural']} | "
            f"{c['paired_images']} | {a['right_percent']:.2f}/{b['right_percent']:.2f} | {a['left_percent']:.2f} | "
            f"{a['delta_pp']:+.2f} [{a['paired_pointwise95_pp'][0]:+.2f},{a['paired_pointwise95_pp'][1]:+.2f}] | "
            f"{b['delta_pp']:+.2f} [{b['paired_pointwise95_pp'][0]:+.2f},{b['paired_pointwise95_pp'][1]:+.2f}] | "
            f"{ta['wide_top1']:.2f}/{ta['B2_top1']:.2f} | {cw['delta_pp']:+.2f} | {c['control_vs_wide']['top1_class_changed_images']} | "
            f"{d['images_with_identical_codes_everywhere']} | {d['changed_codes']}/{d['compared_codes']} ({100 * d['changed_fraction']:.4f}%) | "
            f"{c['indicative_seconds_per_image']['wide-cuda']:.3f}")
    publish(ROOT / 'results/summaries/scaled-bridge-v2' / f'MB2-{name[:16]}', data, '\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return data


# ---------------------------------------------------------------------------------------------------------------
# PROTOCOL_MN (contract 2.2): MobileNetV2 and MobileNetV3-Large, repaired-recipe (B2 default) graphs.
MN_MODELS = ('mobilenet_v2', 'mobilenet_v3_large')
MN_FORMATS = ('int8', 'fp6_e2m3', 'fp8_e4m3fn', 'int6', 'fp7_e3m3')
MN_CASES = tuple(f'{m}-{f}-default-b2' for m in MN_MODELS for f in MN_FORMATS)
MN_POLICY_CASES = tuple(f'{m}-int8-default-b2' for m in MN_MODELS)
MN_POLICY_IMAGES = 8          # PROTOCOL_MN policy gate: eight images on CPU and on CUDA
MN_REGRESS_FROM = '7c6344af732d1bf3'
MN_RUN = '1f75c9232c8a0202482fe9bce9a46360c7cc0999c5b0bcf7df5ebcc0446fc863'
MN_LEDGER_CUTOFF = 1790890604.0241377   # start of the last MN invocation (replay of mobilenet_v3_large-fp7_e3m3)


def mac_kind(nodes, name, cert):
    """Kind of a MAC node for the width tables: stem, depthwise, se_reduce, se_expand, pointwise, conv, classifier."""
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
    return 'pointwise' if cert['shape'][2:] == [1, 1] else 'conv'


def mn_structure(export, certs):
    """Per-node widths split by kind, mul product certificates and boundary counts by operator."""
    from .certificates import certify_products
    nodes = export['nodes']; books = export['codebooks']
    widths = ('signed_bits_absolute', 'signed_bits_range', 'signed_bits_structural', 'signed_bits_closure')
    rows = []
    for c in certs.values():
        kind = mac_kind(nodes, c['node'], c)
        rows.append({'node': c['node'], 'kind': kind, 'K': c['K'], 'groups': c['groups'], 'kernel': c['shape'][2:],
                     'input_codebook': c['input_codebook'], 'weight_codebook': c['weight_codebook'],
                     'product_shift': c['product_shift'], 'closure_input_range_units': c['closure_input_range_units'],
                     **{k: c[k] for k in widths}, 'binary64_exact_dot': c['binary64_exact_dot'],
                     'two_limb': c['kernel_operand_bits'] == 64})
    by_kind = {}
    for kind in sorted({r['kind'] for r in rows}):
        sel = [r for r in rows if r['kind'] == kind]
        by_kind[kind] = {'nodes': len(sel), 'K': sorted({r['K'] for r in sel}),
                         **{f'max_{k}': max(r[k] for r in sel) for k in widths},
                         **{f'min_{k}': min(r[k] for r in sel) for k in widths}}
    products = certify_products(export)
    boundary = Counter()
    for n in nodes:
        if n['op'] in ('input', 'output', 'identity', 'flatten'):
            continue
        store = n.get('store')
        if store is None:
            boundary[(n['op'], 'unstored')] += 1
        else:
            book = books[store['codebook']]
            boundary[(n['op'], 'unsigned' if min(book['units']) >= 0 else 'signed')] += 1
    return {'mac_nodes': rows, 'widths_by_kind': by_kind,
            'max_widths': {k: max(r[k] for r in rows) for k in widths},
            'mul_product_certificates': [{k: v for k, v in p.items()} for p in products.values()],
            'boundaries_by_operator': {f'{op}:{kind}': n for (op, kind), n in sorted(boundary.items())}}


def write_mn_report(run=None, cutoff=None):
    """MN summary, bounded to PROTOCOL_MN (prefix 32 on CUDA and in the replay, 8 on CPU, 8 per policy panel), the
    7c6344af regressions and a ledger cut-off."""
    global _ROOT
    run = run or MN_RUN; cutoff = cutoff or MN_LEDGER_CUTOFF
    matches = [p for p in (BASE / 'runs').glob(run + '*') if p.is_dir()]
    if len(matches) != 1:
        raise ValueError('run digest prefix does not select exactly one run root')
    _ROOT = matches[0]; name = root().name
    cases = []
    for case in MN_CASES:
        row = b2_case_summary(case)
        if 'export' in row:
            certificate = unseal(root() / case / 'certificate.json')
            export = unseal(ROOT / certificate['export']['path'])
            row['structure'] = mn_structure(export, certificate['certificates'])
            row.pop('layers', None)           # superseded by structure.mac_nodes (adds kind and closure width)
            witness = root() / case / 'graph-witness-control-cpp-0000.json'
            row['control_graph_witness'] = ({k: unseal(witness)[k] for k in ('status', 'stored_tensors_compared', 'problems')}
                                            if witness.exists() else None)
        cases.append(row)
    policies = []
    for case in MN_POLICY_CASES:
        if not (root() / case / 'certificate.json').exists():
            continue
        for path in sorted((root() / case).glob('gate-*.json')):
            policy = path.stem[len('gate-'):]
            row = policy_summary(case, policy, MN_POLICY_IMAGES, MN_POLICY_IMAGES)
            witness = root() / case / f'graph-witness-{policy}-cpp-0000.json'
            row['graph_witness_first_image'] = unseal(witness)['status'] if witness.exists() else None
            row['lossless_identity'] = unseal(path)['lossless_identity']
            policies.append(row)
    regress = sorted((unseal(p) for p in (root() / 'regress').glob(f'*-{MN_REGRESS_FROM}.json')),
                     key=lambda r: (r['case'], r['policy'], r['backend']))
    regress_b1 = sorted((unseal(p) for p in (root() / 'regress').glob('*-b1-*.json')),
                        key=lambda r: (r['case'], r['policy'], r['backend']))
    keep = ('case', 'policy', 'backend', 'images_identical', 'old_engine_sources')
    books = sorted({b for c in cases for b in c.get('codebooks', [])})
    spent = cost(cutoff); later = spent.pop('invocations_after_cutoff')
    print(f'ledger invocations after the cut-off (not in the summary): {later}')
    conf = {b: {k: unseal(root() / 'conformance' / f'{b}.json').get(k) for k in ('checks', 'operator_checks_2_2')}
            for b in books if (root() / 'conformance' / f'{b}.json').exists()}
    data = {'milestone': 'MN', 'engine_sources_digest': name, 'engine_sources': source_list('MN', name),
            'evidence_label': 'development evidence on the screen-1k list (prefixes 8 and 32); not a quality claim',
            'cases': cases, 'gates_passed': sum(c['gate'] for c in cases),
            'cases_run': sum('export' in c for c in cases), 'cases_listed': len(cases),
            'policy_gates': policies, 'policy_gates_passed': sum(r['gate'] for r in policies),
            'regression_7c6344af': [{k: r[k] for k in keep} for r in regress],
            'regression_original_recipe': [{k: r[k] for k in keep} for r in regress_b1],
            'primitive_conformance': conf,
            'policy_conformance': ({k: unseal(root() / 'conformance' / 'policies.json')[k] for k in ('status', 'checks')}
                                   if (root() / 'conformance' / 'policies.json').exists() else None),
            'tie_rule': 'contract order: descending value, lowest class index first. B2 retained order: torch.topk on '
                        'CUDA FP32 logits. Both are reported for the B2 side; the engine side always uses the contract order',
            'cost': spent,
            'bounds': {'panel_images_cuda': MB2_PANEL_IMAGES, 'gate_images_cpu': MB2_GATE_IMAGES, 'replay_images': MB2_PANEL_IMAGES,
                       'policy_panel_images': MN_POLICY_IMAGES, 'ledger_cutoff_epoch': cutoff,
                       'note': 'records and ledger entries beyond these bounds are not read'}}
    lines = ['case | gate | bits abs/range/struct/closure (max) | depthwise struct/closure | SE reduce/expand closure | '
             'mul product bits | n | wide top1 | B2 top1 retained/lowest | wide-B2 pp [95%] retained | wide-B2 pp lowest | '
             'tie-aware wide/B2 | ctl-wide pp | ctl!=wide top1 | changed codes vs B2 | CUDA s/img wide (indicative)']
    for c in cases:
        if 'comparisons' not in c:
            lines.append(f"{c['case']} | {c['gate']} | {'not run' if 'export' not in c else 'incomplete'}"); continue
        s = c['structure']; w = s['max_widths']; k = s['widths_by_kind']
        dw = k.get('depthwise', {}); ser = k.get('se_reduce'); see = k.get('se_expand')
        se = f"{ser['max_signed_bits_closure']}/{see['max_signed_bits_closure']}" if ser and see else '-'
        mul = max((p.get('signed_bits_product', 0) for p in s['mul_product_certificates']), default=None)
        a = c['comparisons']['wide_minus_B2_retained_order']['top1']; b = c['comparisons']['wide_minus_B2_lowest_index']['top1']
        cw = c['comparisons']['control_minus_wide']['top1']; ta = c['tie_aware']; d = c['wide_vs_B2']
        lines.append(
            f"{c['case']} | {c['gate']} | {w['signed_bits_absolute']}/{w['signed_bits_range']}/{w['signed_bits_structural']}/"
            f"{w['signed_bits_closure']} | {dw.get('max_signed_bits_structural', '-')}/{dw.get('max_signed_bits_closure', '-')} | {se} | "
            f"{mul if mul is not None else '-'} | {c['paired_images']} | {a['left_percent']:.2f} | {a['right_percent']:.2f}/{b['right_percent']:.2f} | "
            f"{a['delta_pp']:+.2f} [{a['paired_pointwise95_pp'][0]:+.2f},{a['paired_pointwise95_pp'][1]:+.2f}] | {b['delta_pp']:+.2f} | "
            f"{ta['wide_top1']:.2f}/{ta['B2_top1']:.2f} | {cw['delta_pp']:+.2f} | {c['control_vs_wide']['top1_class_changed_images']} | "
            f"{d['changed_codes']}/{d['compared_codes']} ({100 * d['changed_fraction']:.4f}%) | {c['indicative_seconds_per_image']['wide-cuda']:.3f}")
    lines += ['', 'policy gates (8 images CPU = CUDA): case | policy | gate | lossless width check | failed/8 | top1 (wide) | '
              'top1 class != wide | images with event | nodes with event | graph witness']
    for r in policies:
        if 'images' not in r:
            lines.append(f"{r['case']} | {r['policy']} | {r['gate']} | incomplete"); continue
        lossless = 'identical to wide, 0 saturations' if r['lossless_identity'] is not None else '-'
        lines.append(f"{r['case']} | {r['policy']} | {r['gate']} | {lossless} | "
                     f"{r['images_failed_nonfinite']}/{r['images']} | {r['top1_percent_failures_incorrect']:.2f} ({r['wide_top1_percent']:.2f}) | "
                     f"{r['top1_class_differs_from_wide_images']} | {r['images_with_any_event']} | {len(r['nodes_with_event'])} | "
                     f"{r['graph_witness_first_image']}")
    lines += ['', f"regression against 7c6344af: {sum(r['images_identical'] == 8 for r in regress)}/{len(regress)} record sets identical "
              f"on 8 images; original recipe: {sum(r['images_identical'] == 8 for r in regress_b1)}/{len(regress_b1)}"]
    publish(ROOT / 'results/summaries/scaled-bridge-v2' / f'MN-{name[:16]}', data, '\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return data
