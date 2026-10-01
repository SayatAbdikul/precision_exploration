"""Read-only audit and report for the versioned useful-quality E1 study.

The v2 controller owns enrollment, allocations, gates, records and panels. This
module only reads those artifacts and writes separate analysis outputs. It
reports every frozen candidate, including unfinished and budget-limited cases,
without turning a pilot or incomplete panel into a quality result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import statistics

from tools.analysis.useful_quality_e1 import accuracy, check_pairing
from tools.breadth_study import useful_quality as v1
from tools.breadth_study import useful_quality_v2 as v2
from tools.breadth_study.useful_quality_v2_worker import validate_record
from tools.experiment_b.common import seal, unseal
from tools.phase3.common import ROOT, checked, digest, read, reference

BASE = v2.BASE
OUT = ROOT / 'results/summaries/useful-quality-e1-v2'
DOC = ROOT / 'docs/analysis/useful-quality-e1-v2.md'
MODES = ('exact', 'control')
BACKENDS = ('cpp', 'cuda')
PANELS = (32, 128)


def classify(*, counts, panels, admitted, promotion, blocked, controller_status, failed=False):
    """Keep scientific panel completion separate from resource disposition."""
    if failed:
        return 'failed'
    if 128 in panels:
        return 'completed_128'
    if 32 in panels:
        if promotion is None:
            return 'completed_32_promotion_pending'
        if not promotion.get('promote'):
            return 'completed_32' if promotion.get('quality_eligible') is False else 'completed_32_extension_budget_limited'
        if all(counts[f'{mode}/cuda'] >= 128 for mode in MODES):
            return 'extension_records_complete_summary_pending'
        if blocked:
            return 'completed_32_extension_budget_limited'
        return 'completed_32_extension_pending'
    if blocked:
        return 'panel_budget_limited' if admitted else 'native_gate_budget_limited'
    if admitted:
        return 'native_admitted_panel_pending'
    if any(counts.values()):
        return 'native_gate_in_progress'
    return 'enrolled_pending'


def _read_archive(ref, sources):
    archive = unseal(checked(ref))
    if archive['implementation'] != sources or set(archive['files']) != set(sources):
        raise ValueError('sealed source archive does not match plan source identity')
    for name, item in archive['files'].items():
        if item['sha256'] != sources[name]:
            raise ValueError('archived source hash differs from plan')
        checked(item)
    return archive


def _records(path, plan, samples):
    data, counts, provenance, costs, oracle_comparisons = {}, {}, {}, {}, {}
    for mode in MODES:
        for backend in BACKENDS:
            key = f'{mode}/{backend}'
            files = sorted(path.parent.glob(f'[0-9][0-9][0-9][0-9]-{mode}-{backend}.json'))
            if len(files) > (8 if backend == 'cpp' else 128):
                raise ValueError('v2 checkpoint count exceeds declared stage')
            arm = []
            verified_proofs = 0
            for index, file in enumerate(files):
                if file.name != f'{index:04d}-{mode}-{backend}.json':
                    raise ValueError('v2 checkpoint sequence has a gap')
                record = unseal(file)
                validate_record(record, plan, samples[index], mode, backend, index)
                proof_path = path.parent / f'{index:04d}-{mode}-{backend}-oracle.json'
                if proof_path.exists():
                    proof = unseal(proof_path)
                    fields = ('layers', 'output_sha256', 'prediction', 'diagnostic_signature')
                    oracle = unseal(checked(proof['v1_oracle']))
                    numerical = {field: record[field] for field in fields}
                    if (proof.get('matches') is not True or tuple(proof['fields']) != fields
                            or proof['v2_numerical_sha256'] != digest(numerical)
                            or any(oracle[field] != record[field] for field in fields)):
                        raise ValueError('v2 record differs from its retained v1 oracle comparison')
                    verified_proofs += 1
                arm.append(record)
            data[key] = arm
            counts[key] = len(arm)
            provenance[key] = dict(Counter(row['source_kind'] for row in arm))
            oracle_files = list(path.parent.glob(f'[0-9][0-9][0-9][0-9]-{mode}-{backend}-oracle.json'))
            if len(oracle_files) != verified_proofs:
                raise ValueError('orphaned v1 oracle comparison lacks a v2 checkpoint')
            oracle_comparisons[key] = verified_proofs
            new = [row['new_compute_seconds'] for row in arm if row['source_kind'] == 'new_verified_execution']
            costs[key] = {'new_images': len(new), 'new_compute_seconds': sum(new),
                          'median_new_compute_seconds_per_image': statistics.median(new) if new else None,
                          'maximum_recorded_peak_rss_kib': max((row.get('peak_rss_kib', 0) for row in arm), default=0)}
    return data, counts, provenance, costs, oracle_comparisons


def _gate(path, plan, records):
    admission_path = path.parent / 'native-admission.json'
    if not admission_path.exists():
        return None
    gate = unseal(admission_path)
    if (gate['plan'] != reference(path) or gate['images_per_mode_backend'] != 8
            or len(gate['records']) != 32):
        raise ValueError('v2 native admission identity or population differs')
    expected_records = [reference(path.parent / f'{i:04d}-{mode}-{backend}.json')
                        for mode in MODES for backend in BACKENDS for i in range(8)]
    if gate['records'] != expected_records:
        raise ValueError('v2 native admission is detached from the ordered checkpoint prefix')
    for ref in gate['conformance'] + gate['records']:
        checked(ref)
    for mode in MODES:
        for backend in BACKENDS:
            if len(records[f'{mode}/{backend}']) < 8:
                raise ValueError('native admission lacks its eight image records')
        for a, b in zip(records[f'{mode}/cpp'][:8], records[f'{mode}/cuda'][:8]):
            fields = ('layers', 'output_sha256', 'prediction')
            if mode != 'exact' or plan['format'] != 'int8':
                fields += ('diagnostic_signature',)
            if any(a[field] != b[field] for field in fields):
                raise ValueError('admitted CPP/CUDA records disagree')
    if plan['format'] != 'int8':
        if any(records[f'{mode}/{backend}'][i]['source_kind'] != 'new_verified_execution'
               for mode in MODES for backend in BACKENDS for i in range(8)):
            raise ValueError('floating v2 admission contains imported execution')
    return reference(admission_path)


def _panel(path, plan, panel, records, samples):
    summary_path = path.parent / f'summary-{panel}.json'
    summary = unseal(summary_path)
    if (summary['plan'] != reference(path) or summary['images'] != panel
            or not (path.parent / 'native-admission.json').exists()):
        raise ValueError('v2 summary identity/count or native admission missing')
    paired = {mode: records[f'{mode}/cuda'][:panel] for mode in MODES}
    if any(len(arm) != panel for arm in paired.values()):
        raise ValueError('v2 panel has insufficient CUDA records')
    for mode in MODES:
        expected_refs = [reference(path.parent / f'{i:04d}-{mode}-cuda.json') for i in range(panel)]
        if summary['records'][mode] != expected_refs:
            raise ValueError('v2 panel record references differ from checkpoint prefix')
        check_pairing(paired[mode], read(checked(plan['baseline'])), samples[:panel])
    for a, b in zip(paired['exact'], paired['control']):
        if (a['sample'], a['ground_truth'], a['fp32_prediction']) != (
                b['sample'], b['ground_truth'], b['fp32_prediction']):
            raise ValueError('v2 exact/control pairing differs')
    for rank in (1, 5):
        truth = [r['ground_truth'] in r['fp32_prediction'][:rank] for r in paired['exact']]
        exact = [r['ground_truth'] in r['prediction'][:rank] for r in paired['exact']]
        control = [r['ground_truth'] in r['prediction'][:rank] for r in paired['control']]
        expected = {'control_minus_exact': v1.paired_stats(exact, control),
                    'exact_minus_FP32': v1.paired_stats(truth, exact),
                    'control_minus_FP32': v1.paired_stats(truth, control)}
        if summary['statistics'][f'top{rank}'] != expected:
            raise ValueError('v2 paired statistic does not reproduce from saved predictions')
    order = plan['deployed_nodes']
    divergences = []
    for a, b in zip(paired['exact'], paired['control']):
        changed = [name for name in order if a['layers'][name] != b['layers'][name]]
        divergences.append({'index': a['index'], 'sample_sha256': a['sample']['sha256'],
                            'first_layer': changed[0] if changed else None,
                            'changed_layers': changed, 'output_matches': a['output_sha256'] == b['output_sha256']})
    if (summary['divergences'] != divergences
            or summary['all_layer_agreement_images'] != sum(not row['changed_layers'] for row in divergences)
            or summary['output_agreement_images'] != sum(row['output_matches'] for row in divergences)):
        raise ValueError('v2 layer/output divergence summary does not reproduce')
    mapping = unseal(checked(plan['mapping']))
    ordinary = mapping['ordinary_B']
    broot = ROOT / 'artifacts' / ordinary['component'] / 'predictions' / ordinary['configuration_sha256']
    b_rows = []
    for item in paired['exact']:
        b = unseal(broot / f"{item['sample']['sha256']}.json")
        if b['configuration_sha256'] != ordinary['configuration_sha256'] or b['sample'] != item['sample']:
            raise ValueError('descriptive B prediction is not on the same frozen image')
        b_rows.append({'prediction': b['top5'], 'ground_truth': item['ground_truth']})
    quality = {mode: {f'top{rank}': accuracy(rows, 'prediction', rank) for rank in (1, 5)}
               for mode, rows in {**paired, 'ordinary_B_different_configuration': b_rows}.items()}
    quality['full_precision'] = {f'top{rank}': accuracy(paired['exact'], 'fp32_prediction', rank)
                                 for rank in (1, 5)}
    return {'summary': reference(summary_path), 'images': panel,
            'statistics': summary['statistics'], 'quality': quality,
            'ordinary_B_configuration': ordinary['configuration'],
            'all_layer_agreement_images': summary['all_layer_agreement_images'],
            'output_agreement_images': summary['output_agreement_images'],
            'zero_output_discordance_upper_95_percent': (
                100 * (1 - .05 ** (1 / panel)) if summary['output_agreement_images'] == panel else None),
            'first_divergences': dict(Counter(row['first_layer'] for row in divergences if row['first_layer']))}


def _case(path, plan, sample_rows, controller, promotion):
    if plan['version'] != 'useful-quality-e1-case-v2' or plan['case'] not in v2.ORDER:
        raise ValueError('unexpected v2 plan/case')
    parent_path = checked(plan['parent_plan'])
    parent = unseal(parent_path)
    if (plan['protocol'] != parent['protocol'] or plan['graph'] != parent['graph']
            or plan['configuration'] != parent['configuration']
            or plan['baseline'] != parent['baseline'] or plan['mapping'] != parent['mapping']
            or plan['parent_sources'] != parent['sources']):
        raise ValueError('v2 plan changed frozen v1 scientific identity')
    _read_archive(plan['source_archive'], plan['sources'])
    _read_archive(parent['source_archive'], parent['sources'])
    checked(plan['optimized_certificate']); checked(plan['static_replay'])
    records, counts, provenance, costs, oracle_comparisons = _records(path, plan, sample_rows)
    admission = _gate(path, plan, records)
    panels = [n for n in PANELS if (path.parent / f'summary-{n}.json').exists()]
    if 128 in panels and 32 not in panels:
        raise ValueError('v2 extension panel lacks its initial 32-image panel')
    if panels and admission is None:
        raise ValueError('v2 completed panel has no native admission')
    result = _panel(path, plan, max(panels), records, sample_rows) if panels else None
    blocked = [item for item in controller.get('blocked', []) if item['case'] == plan['case']]
    blocked.extend(item for item in controller.get('extension_budget_limited', []) if item['case'] == plan['case'])
    failed = controller.get('status') == 'failed' and controller.get('failed_task', {}).get('case') == plan['case']
    disposition = classify(counts=counts, panels=panels, admitted=bool(admission),
                           promotion=promotion, blocked=blocked,
                           controller_status=controller.get('status', 'not_started'), failed=failed)
    row = {'case': plan['case'], 'status': disposition, 'images': result['images'] if result else 0,
           'counts': counts, 'panels': panels, 'native_admission': admission,
           'plan': reference(path), 'graph': plan['graph'], 'configuration': plan['configuration'],
           'mapping': plan['mapping'], 'static_proof': plan['static_proof'],
           'static_replay': plan['static_replay'], 'optimized_certificate': plan['optimized_certificate'],
           'v2_source_archive': plan['source_archive'], 'parent_v1_plan': plan['parent_plan'],
           'parent_v1_source_archive': parent['source_archive'],
           'v1_saved_checkpoint_counts': {
               f'{mode}/{backend}': len(list(parent_path.parent.glob(
                   f'[0-9][0-9][0-9][0-9]-{mode}-{backend}.json')))
               for mode in MODES for backend in BACKENDS},
           'source_provenance': provenance, 'retained_v1_oracle_comparisons': oracle_comparisons,
           'new_compute_costs': costs,
           'promotion': promotion, 'blocked_stages': blocked}
    if result:
        row.update(result)
    return row


def analyze(*, write=True):
    controller = read(BASE / 'status.json') if (BASE / 'status.json').exists() else {'status': 'not_started'}
    budget = v2.budget()
    enrollment_path = BASE / 'enrollment.json'
    if not enrollment_path.exists():
        rows = [{'case': case, 'status': 'not_enrolled', 'images': 0} for case in v2.ORDER]
        result = {'version': 'useful-quality-e1-analysis-v2', 'generated_at': datetime.now(timezone.utc).isoformat(),
                  'controller_status': controller.get('status'), 'enrollment_status': 'pending',
                  'budget': budget, 'candidates': rows,
                  'limitations': ['The optimized controller has not sealed an enrollment; no v2 image result is claimed.']}
    else:
        enrollment = unseal(enrollment_path)
        if (enrollment['version'] != v2.SCHEMA
                or tuple(item['case'] for item in enrollment['cases']) != v2.ORDER):
            raise ValueError('v2 enrollment candidate order or version changed')
        _read_archive(enrollment['source_archive'], unseal(checked(enrollment['cases'][0]['plan']))['sources'])
        protocol = unseal(checked(enrollment['scientific_protocol']))
        if len(protocol['sample_rows']) != 128:
            raise ValueError('frozen development image panel changed')
        promotions_path = BASE / 'promotion.json'
        promotions = unseal(promotions_path) if promotions_path.exists() else {'decisions': []}
        if promotions.get('scientific_protocol', enrollment['scientific_protocol']) != enrollment['scientific_protocol']:
            raise ValueError('v2 promotion is detached from frozen scientific protocol')
        choices = {row['case']: row for row in promotions['decisions']}
        rows = []
        for item in enrollment['cases']:
            path = checked(item['plan']); plan = unseal(path)
            if plan['case'] != item['case'] or plan['protocol'] != enrollment['scientific_protocol']:
                raise ValueError('v2 enrolled case differs from its plan/protocol')
            rows.append(_case(path, plan, protocol['sample_rows'], controller, choices.get(item['case'])))
        result = {'version': 'useful-quality-e1-analysis-v2', 'generated_at': datetime.now(timezone.utc).isoformat(),
                  'controller_status': controller.get('status'), 'enrollment_status': 'sealed',
                  'enrollment': reference(enrollment_path), 'scientific_protocol': enrollment['scientific_protocol'],
                  'optimized_certificate': enrollment['optimized_certificate'],
                  'v2_source_archive': enrollment['source_archive'],
                  'promotion': reference(promotions_path) if promotions_path.exists() else None,
                  'budget': budget, 'candidates': rows,
                  'limitations': [
                      'Development panels and promotions are exploratory; they are not final benchmark confirmation.',
                      'B maxabs is a different configuration and calibration; its gap to canonical A is not a pure arithmetic effect.',
                      'Migrated v1 and retained integer exact records keep their original source and timing provenance; zero new compute is not a throughput measurement.',
                      'A one- or eight-image native gate is not an accuracy panel. Incomplete or budget-limited cases have no panel quality claim.',
                      'Arithmetic effects are paired on the frozen samples; interval estimates describe sampling uncertainty, not implementation equivalence.']}
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        seal(OUT / 'analysis.json', result)
        with (OUT / 'status.csv').open('w', newline='') as stream:
            fields = ('case', 'status', 'images', 'exact_cuda_records', 'control_cuda_records',
                      'strict_top1_percent', 'control_top1_percent', 'FP32_top1_percent',
                      'B_top1_percent_different_configuration', 'all_layer_agreement_images',
                      'output_agreement_images')
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for row in result['candidates']:
                item = {'case': row['case'], 'status': row['status'], 'images': row['images'],
                        'exact_cuda_records': row.get('counts', {}).get('exact/cuda', 0),
                        'control_cuda_records': row.get('counts', {}).get('control/cuda', 0),
                        'all_layer_agreement_images': row.get('all_layer_agreement_images'),
                        'output_agreement_images': row.get('output_agreement_images')}
                if row['images']:
                    for field, mode in [('strict_top1_percent', 'exact'), ('control_top1_percent', 'control'),
                                        ('FP32_top1_percent', 'full_precision'),
                                        ('B_top1_percent_different_configuration', 'ordinary_B_different_configuration')]:
                        item[field] = row['quality'][mode]['top1']['percent']
                writer.writerow(item)
        DOC.parent.mkdir(parents=True, exist_ok=True)
        DOC.write_text(report(result))
    return result


def report(result):
    lines = ['# Useful-quality E1 optimized study: development evidence', '',
             f"Controller: **{result['controller_status']}**. Enrollment: **{result['enrollment_status']}**.", '',
             '| Canonical A case | Status | Completed paired images | Strict top-1 | Matched FP32 top-1 | FP32 top-1 | B top-1, different configuration |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in result['candidates']:
        if row['images']:
            q = row['quality']
            scores = [f"{q[mode]['top1']['percent']:.2f}%" for mode in
                      ('exact', 'control', 'full_precision', 'ordinary_B_different_configuration')]
        else:
            scores = ['—'] * 4
        lines.append(f"| {row['case']} | {row['status']} | {row['images']} | {' | '.join(scores)} |")
    budget = result['budget']
    lines += ['', f"Lifetime compute budget: {budget['limit_seconds']/3600:.2f} worker-hours; "
              f"v1 saved/finished {budget['v1_sunk_seconds']/3600:.2f}, v1 interrupted "
              f"{budget.get('v1_uncheckpointed_seconds', 0.0)/3600:.2f}, v2 pre-enrollment "
              f"{budget['v2_pre_enrollment_compute_seconds']/3600:.2f}, v2 completed "
              f"{budget['v2_spent_seconds']/3600:.2f}, unreconciled reservations "
              f"{budget['v2_reserved_seconds']/3600:.2f}, remaining {budget['remaining_seconds']/3600:.2f}.", '']
    for row in result['candidates']:
        if not row['images']:
            if row.get('counts'):
                counts = ', '.join(f'{arm} {n}' for arm, n in row['counts'].items())
                lines.append(f"- {row['case']}: {row['status']}; checkpoints: {counts}.")
            continue
        effect = row['statistics']['top1']['control_minus_exact']
        interval = effect['pointwise_95_interval_pp']
        lines.append(f"- {row['case']} ({row['images']} images): matched minus strict "
                     f"{effect['difference_pp']:+.2f} percentage points, paired 95% interval "
                     f"[{interval[0]:+.2f}, {interval[1]:+.2f}]; full layer agreement "
                     f"{row['all_layer_agreement_images']}/{row['images']}, output agreement "
                     f"{row['output_agreement_images']}/{row['images']}. "
                     f"Provenance: {row['source_provenance']['exact/cuda']} exact, "
                     f"{row['source_provenance']['control/cuda']} control.")
        if row['zero_output_discordance_upper_95_percent'] is not None:
            lines.append(f"  No output discordance observed; one-sided 95% upper bound "
                         f"{row['zero_output_discordance_upper_95_percent']:.2f}% on this panel.")
    lines += ['', 'The JSON ledger binds each result to its sealed v2 plan, optimized runtime certificate, native admission, source archive, and parent v1 plan/archive. Imported v1 predictions retain original execution provenance; their old compute cost is charged in the lifetime budget.', '',
              'Resume experiments: `.venv/bin/python -m tools.run.useful_quality_v2 run`',
              'Rebuild this report: `.venv/bin/python -m tools.analysis.useful_quality_e1_v2`', '',
              *result['limitations'], '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-write', action='store_true', help='audit and print without writing report files')
    args = parser.parse_args()
    result = analyze(write=not args.no_write)
    print(json.dumps({'controller_status': result['controller_status'],
                      'budget_remaining_hours': result['budget']['remaining_seconds'] / 3600,
                      'cases': [{'case': row['case'], 'status': row['status'], 'images': row['images']}
                                for row in result['candidates']]}, indent=2))


if __name__ == '__main__':
    main()
