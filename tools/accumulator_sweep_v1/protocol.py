"""Writes public/experiments/configs/breadth-study/accumulator-sweep-protocol-v1.json once (refuses to overwrite).

    .venv/bin/python -m tools.accumulator_sweep_v1.protocol CASE[:b1|b2][:integer] ... [--exclude CASE=reason ...]
"""
import argparse
import hashlib
import json
import time

from . import certs, plan

PATH = certs.ROOT / 'public/experiments/configs/breadth-study/accumulator-sweep-protocol-v1.json'


SENSITIVITY_CASES = ('resnet18-int8-default-b2', 'resnet18-fp7_e3m3-default-b2', 'resnet18-posit8_es1-default-b2')


def case_spec(case, recipe, integer):
    s = certs.summary(case)
    return {'case': case, 'recipe': recipe, 'integer': integer, 'enters_sweep': True,
            'float_sensitivity': case in SENSITIVITY_CASES,
            'W_cert_abs': s['W_cert_abs'], 'W_cert_struct': s['W_cert_struct'], 'W_node_min_abs': s['W_node_min_abs'],
            'ub': s['ub'], 'fp16_policy': s['fp16_policy'], 'f21_policy': s['f21_policy'],
            'location_ladder': plan.ladder(s['W_cert_abs'])}


def build(cases, excluded, estimate):
    return {
        'id': 'accumulator-sweep-protocol-v1',
        'lane': 'L8 (CTXMARK-L8-accsweep-r1)',
        'written': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'status': 'frozen before any sweep measurement (location or 1k); deviations go to addendum files',
        'evidence_class': 'development evidence: ImageNet screen-1k list (imagenet_screen_1k), first 128 images for '
                          'the location phase, images 0-1000 for the grid. No held-out image is run.',
        'engine': {'archive': f'artifacts/scaled_bridge_v2/implementations/{certs.DIGEST}', 'digest': certs.DIGEST,
                   'command': 'artifacts/agent_orchestration/gpu_run.sh --min-free-mib 3000 --wait 7200 $A/run.sh predict '
                              'CASE POLICY cuda START STOP --batch 8',
                   'contract': 'scaled bridge contract 2.1 (docs/analysis/scaled-bridge-v2-contract-2026-10-01.md); '
                               'policy names are complete specifications',
                   'reuse': 'a result on images [0, N) is assembled from contiguous sealed predict files of the same '
                            'case and policy (existing 32- and 64-image files and the 128-image location files '
                            'included); the engine is exact and batch invariant, checked in step 1 (64-image file '
                            'equals the 32-image file on its first 32 images).'},
        'cases': cases,
        'excluded_cases': excluded,
        'policies': {
            'wide': 'exact reference arm',
            'control': 'sequential binary32 FMA',
            'sat.w<W>': 'uniform family: signed saturating register of W bits at every MAC node',
            'sat.struct-<d>': 'per-node family: certified structural width of the node minus d',
            'float': 'fp16.x<e>, f21.x<e> (one rounding per multiply-add; binary point moved by e)'},
        'float_exponent_rule': {
            'ub': 'max over MAC nodes of bit_length(max_abs_prefix_units) - product_shift (certificate.json); '
                  'every prefix sum satisfies |sum| < 2^ub in code-level units',
            'rule': 'e = 0 (plain policy name) when the plain format is proven overflow-free (ub <= emax) and every '
                    'product step 2^-product_shift is a normal number (>= 2^emin); otherwise e = emax - ub, the '
                    'largest shift the certificate proves overflow-free. fp16: emax 15, emin -14; f21: 127, -126. '
                    'Implemented in tools/accumulator_sweep_v1/certs.py (unit tested).',
            'sensitivity_location_phase': 'cases int8-b2, fp7_e3m3-b2, posit8_es1-b2 (one integer, one minifloat, '
                                          'one posit; coarsened for the budget), 128 images: fp16.x<e+k> for k in %s; f21.x<15-ub> next to the '
                                          'rule policy (expected identical to plain f21: no overflow, no subnormal)'
                                          % list(plan.FP16_SENSITIVITY)},
        'location_phase': {
            'images': list(plan.LOCATION),
            'runs': 'every case: wide; sat.w ladder W_cert_abs-2, then every %d bits downward (at most %d rungs), '
                    'launched in descending order and stopped once a finished rung has expected Top-1 <= half of '
                    'wide (rungs already started are kept). B2 cases also: the rule fp16 and f21 policies and the '
                    'exponent sensitivity runs.' % (plan.LADDER_STEP, plan.LADDER_MAX_RUNGS),
            'bracket_rule': 'top = narrowest rung W such that no rung >= W has a saturation event on the 128 images '
                            '(W_cert_abs - 1 if the widest rung has one); bottom = (widest rung with expected Top-1 '
                            '<= half of wide) - %d (narrowest rung - %d if none reached half); clipped to '
                            '[2, W_cert_abs - 1] (tools/accumulator_sweep_v1/plan.py: bracket).'
                            % (plan.BOTTOM_MARGIN, plan.BOTTOM_MARGIN)},
        'screen_grid': {
            'images': list(plan.SCREEN),
            'per_case': {
                'priority_1': 'wide; sat.w<W> for every integer W in [bottom, top]',
                'priority_2': 'control; the rule fp16 and f21 policies (B2 cases)',
                'priority_3': 'integer cases: sat.w<W> for W in %s below the bracket and at or below W_cert_abs' %
                              list(plan.HARDWARE_WIDTHS),
                'priority_4': 'sat.struct-<d>, d in %s (B2 cases)' % list(plan.STRUCT_D_PRIMARY),
                'priority_5': 'original-recipe (b1) cases: wide and the uniform bracket (recipe-sensitivity check)',
                'priority_6': 'sat.struct-<d>, d in %s (B2 cases)' % list(plan.STRUCT_D_SECONDARY)},
            'extension_rules': [
                'top: while the 1k run at the widest bracket width has any saturation event, add the next wider width '
                '(below W_cert_abs)',
                'bottom: while fewer than %d bracket widths lie below W_half (1k), or no width reached half, add the '
                'next narrower width (down to 2)' % plan.BOTTOM_MARGIN],
            'hardware_widths_above_bracket': 'not run: a width above a width with no saturation event on the same '
                                             '1000 images gives the exact arm bit for bit (no prefix sum leaves the '
                                             'narrower register, so none leaves the wider one); reported as derived',
            'order': 'priority-major across cases (B2 cases in the listed order), at most 3 jobs of this lane at a '
                     'time, resumable runner tools/run/accumulator_sweep_run.py'},
        'definitions_uniform_family': {
            'W_cert': 'certified absolute width (network maximum)',
            'W_noevent': 'narrowest measured W from which upward no saturation event occurs on any of the 1000 images',
            'W_same': 'narrowest W from which upward all 1000 lowest-index Top-1 classes equal the exact arm '
                      '(a failed image counts as changed)',
            'W_acc(0.5), W_acc(1.0)': 'narrowest W from which upward the expected-credit Top-1 is at least the exact '
                                      'arm minus 0.5 / 1.0 points',
            'W_half': 'widest W at which expected-credit Top-1 is at most half of the exact arm',
            'from_upward': 'the condition holds at W and at every measured wider width of the bracket; widths above '
                           'the bracket top equal the exact arm when the top has no event'},
        'definitions_per_node_family': 'd_noevent, d_same, d_acc(0.5), d_acc(1.0) = largest d from which downward '
                                       '(d smaller) the condition holds over the measured d set (0 = lossless by '
                                       'the gate); d_half = smallest measured d at or below half. Coarse: only the '
                                       'measured d values.',
        'statistics': {
            'scores': 'expected credit (1/t when the label is among t tied maxima, failed image 0) and lowest-index '
                      'Top-1 (contract tie order)',
            'paired': 'every policy against wide of the same case on the same images: '
                      'tools.analysis.b2_ties.paired (expected credit) and '
                      'tools.analysis.b_stage_balanced_comparisons.paired_outcomes (lowest index), 95% paired '
                      'bootstrap intervals',
            'counts': 'images with a changed Top-1 class, with an accumulator event, failed; per-node event rates '
                      '(share of images, share of outputs) at the widths around the transition',
            'no_multiplicity_claims': 'descriptive development evidence; no hypothesis is confirmed here'},
        'by_product': 'exact wide arm against the B2 simulator\'s sealed 1k predictions (stage-1 per-image files or '
                      'the L1 matrix readout), both tie rules, every B2 case that has them',
        'join': 'integer cases: Top-1 at hardware widths 16-32 beside core area from '
                'results/tables/ics55-integer-accwidth-v1.csv at the 8 ns target (all configurations meet it); '
                'preliminary (hardware sweep unreviewed, RTL work on hold); semantic differences listed',
        'budget': estimate,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cases', nargs='+', help='CASE:RECIPE[:integer]')
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--estimate', required=True, help='JSON object with the budget estimate')
    args = ap.parse_args(argv)
    if PATH.exists():
        raise SystemExit(f'{PATH} exists; a protocol is written once (use an addendum)')
    cases = []
    for item in args.cases:
        parts = item.split(':')
        cases.append(case_spec(parts[0], parts[1], len(parts) > 2 and parts[2] == 'integer'))
    excluded = [dict(zip(('case', 'reason'), e.split('=', 1))) for e in args.exclude]
    doc = build(cases, excluded, json.loads(args.estimate))
    text = json.dumps(doc, indent=2, sort_keys=False) + '\n'
    PATH.write_text(text)
    print(PATH, hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main()
