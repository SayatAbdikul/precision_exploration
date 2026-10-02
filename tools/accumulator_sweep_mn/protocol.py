"""Writes public/experiments/configs/breadth-study/accumulator-sweep-mn-protocol-v1.json once (refuses to overwrite).

    .venv/bin/python -m tools.accumulator_sweep_mn.protocol
"""
import hashlib
import json
import sys
import time

from . import certs, plan

PATH = certs.ROOT / 'public/experiments/configs/breadth-study/accumulator-sweep-mn-protocol-v1.json'
L8_PROTOCOL = 'public/experiments/configs/breadth-study/accumulator-sweep-protocol-v1.json'


def case_spec(case):
    s = certs.summary(case)
    if s['fp16_policy'] != certs.cli_fp16_policy(case):
        raise ValueError(f'{case}: the L8 exponent rule and the engine fp16-policy rule disagree')
    e = certs.v1.float_exponent('fp16', s['nodes'])
    stress = case in certs.STRESS
    return {'case': case, 'network': certs.case_network(case), 'format': certs.case_format(case), 'recipe': 'b2',
            'integer': certs.case_format(case) in certs.INTEGER, 'stress_case': stress,
            'W_cert_abs': s['W_cert_abs'], 'W_cert_struct': s['W_cert_struct'], 'W_node_min_abs': s['W_node_min_abs'],
            'ub': s['ub'], 'fp16_policy': s['fp16_policy'], 'f21_policy': s['f21_policy'],
            'published_width_Agg24': certs.published_network_width(case),
            'widths_by_kind': certs.widths_by_kind(case),
            'location_ladder': plan.ladder(s['W_cert_abs']),
            'float_sensitivity': plan.sensitivity_policies(e, s['ub']) if case in plan.SENSITIVITY_CASES else []}


def build():
    cases = [case_spec(c) for c in certs.CASES]
    return {
        'id': 'accumulator-sweep-mn-protocol-v1',
        'lane': 'Q1 (CTXMARK-Q1-mn-accsweep-r1)',
        'written': time.strftime('%Y-%m-%d %H:%M:%S %z'),
        'status': 'frozen before any sweep measurement (wide/control 1k, location or grid); deviations go to '
                  'accumulator-sweep-mn-protocol-v1-addendum-<n>.json, written before the affected runs',
        'adapted_from': {'path': L8_PROTOCOL, 'sha256': hashlib.sha256((certs.ROOT / L8_PROTOCOL).read_bytes()).hexdigest(),
                         'changes': ['MobileNet cases on archive 1f75c923 instead of ResNet18 on 7c6344af',
                                     'ladder starts at W_cert_abs - 1 and steps by 2 (L8: W_cert_abs - 2, step 3)',
                                     'the location refinement rule (L8 addendum 3) is frozen here',
                                     'per-node family: a 128-image d ladder fixes the 1k d set (L8: fixed d, deferred)',
                                     'wide and control at 1k run first and serve as the location reference prefix',
                                     'fp16 rule and f21 are not run on 128 images separately (their 1k runs start at 0)',
                                     'no hardware widths below the bracket and no join (no MobileNet hardware sweep)',
                                     'GPU budget 14 job-hours, at most 2 concurrent jobs of this lane']},
        'evidence_class': 'development evidence: ImageNet screen-1k list (imagenet_screen_1k), images 0-127 for the '
                          'location phase, images 0-999 for the grid. No held-out image is run.',
        'engine': {'archive': certs.ARCHIVE, 'digest': certs.DIGEST,
                   'command': 'GPU_LANE=Q1 artifacts/agent_orchestration/gpu_run.sh --min-free-mib 3000 --wait 7200 '
                              '$M/run.sh predict CASE POLICY cuda START STOP --batch 8',
                   'contract': 'scaled bridge contract 2.1 + 2.2 (docs/analysis/scaled-bridge-v2-contract-2026-10-01.md); '
                               'policy names are complete specifications; Top-k ties by lowest class index',
                   'reuse': 'a result on images [0, N) is assembled from contiguous sealed predict files of the same case '
                            'and policy (existing 32- and 64-image files and the 128-image location files included); the '
                            'engine is exact and batch invariant (step 1: the new 64-image wide file of '
                            'mobilenet_v3_large-int8 equals the existing 32-image file on its first 32 images and the '
                            'sealed batch-1 panel).',
                   'gates': 'every case has its MN case gate; every case and policy family swept (saturating: '
                            'sat.struct-2 or the int8 gates sat.struct-0/-4; float: the rule fp16 policy and f21) has a '
                            'policy gate in the run root before its sweep (orchestrator ruling on the L2c review). '
                            'List: artifacts/accumulator_sweep_mn_v1/GATES-DONE.'},
        'cases': cases,
        'stress_cases': {c: 'MobileNetV3 INT6 collapses in the B2 simulator itself (21.1 % Top-1 at 1k, L1 sealed '
                            'predictions of configuration 5ddda738...); measured with a reduced grid and labelled as a '
                            'stress case; not used for the slack summary across cases' for c in certs.STRESS},
        'policies': {
            'wide': 'exact reference arm',
            'control': 'sequential binary32 FMA',
            'sat.w<W>': 'uniform family: signed saturating register of W bits at every MAC node',
            'sat.struct-<d>': 'per-node family: certified structural width of the node minus d',
            'float': 'fp16[.x<e>], f21[.x<e>] (one rounding per multiply-add; binary point moved by e)'},
        'float_exponent_rule': {
            'ub': 'max over MAC nodes of bit_length(max_abs_prefix_units) - product_shift (certificate.json)',
            'rule': 'L8 rule (tools/accumulator_sweep_v1/certs.py float_exponent): e = 0 when ub <= emax and every '
                    'product step is normal, else e = emax - ub. Equal to the archived engine command `fp16-policy CASE` '
                    'in all ten cases (checked when this file was written). f21: plain in all ten cases.',
            'sensitivity_location_phase': {'cases': list(plan.SENSITIVITY_CASES), 'images': list(plan.LOCATION),
                                           'runs': 'fp16.x<e+k>, k in %s, and f21.x<15-ub>' % list(plan.FP16_SENSITIVITY)}},
        'location_phase': {
            'images': list(plan.LOCATION),
            'reference': 'the first 128 images of the 1k wide run (priority 1)',
            'uniform_ladder': 'sat.w at W_cert_abs - 1, then every 2 bits down (at most 12 rungs), launched in '
                              'descending order; no new rung is launched once a finished rung has expected Top-1 <= half '
                              'of wide (rungs already started are kept)',
            'refinement': 'tools/accumulator_sweep_mn/plan.py refine(), evaluated after the ladder stopped and no sat.w '
                          'run of the case is running, again after every finished batch, until it asks for nothing: '
                          '(a) W0 = narrowest measured width with no event at it or any wider measured width; if a '
                          'narrower width was measured, ask for W0-1; if every measured width has an event, ask for every '
                          'width up to W_cert_abs-1; (b) Wh = widest measured width at or below half; if a wider width was '
                          'measured, ask for Wh+1. Widths >= W_cert_abs or < 2 are dropped. A call that fails 3 times is '
                          'not launched again and is reported.',
            'bracket_rule': 'top = narrowest measured W with no event at any measured W\' >= W (W_cert_abs - 1 if the '
                            'widest has one); bottom = (widest measured W at or below half of wide) - 2 (narrowest '
                            'measured - 2 if none reached half); clipped to [2, W_cert_abs - 1] (plan.bracket).',
            'per_node_ladder': 'sat.struct-d for d = 1, 2, ... 8 in ascending order (128 images), one at a time per '
                               'case; stops after the first d with expected Top-1 <= half of wide (plan.struct_next).',
            'per_node_1k_set': 'd_half(128) - 2 ... d_half(128) + 1, clipped to d >= 1; if no d reached half, the three '
                               'largest measured d (plan.struct_set). Not run for the stress case.'},
        'screen_grid': {
            'images': list(plan.SCREEN),
            'priority_1': 'wide and control, all ten cases (run before the location phase)',
            'priority_2': 'sat.w<W> for every integer W in [bottom, top] (stress case: top, then every second width, '
                          'and the bottom; plan.stress_widths)',
            'priority_3': 'the rule fp16 policy and f21, all ten cases',
            'priority_4': 'sat.struct-<d> for the location d set (not the stress case)',
            'extension_rules': [
                'top: while the 1k run at the widest bracket width has any saturation event, add the next wider width '
                '(below W_cert_abs)',
                'bottom: while fewer than 2 bracket widths lie below W_half (1k), or no width reached half, add the next '
                'narrower width (down to 2)'],
            'widths_above_bracket': 'not run: a width above one with no saturation event on the same images gives the '
                                    'exact arm bit for bit',
            'order': 'priority-major across cases, cases in the listed order; at most 2 jobs of this lane at a time; '
                     'resumable runner tools/run/accumulator_sweep_mn_run.py'},
        'definitions_uniform_family': {
            'W_cert': 'certified absolute width (network maximum)',
            'W_noevent': 'narrowest measured W from which upward no saturation event occurs on any of the 1000 images',
            'W_same': 'narrowest W from which upward all 1000 lowest-index Top-1 classes equal the exact arm (a failed '
                      'image counts as changed)',
            'W_acc(0.5), W_acc(1.0)': 'narrowest W from which upward the expected-credit Top-1 is at least the exact '
                                      'arm minus 0.5 / 1.0 points',
            'W_half': 'widest W at which expected-credit Top-1 is at most half of the exact arm',
            'slack': 'W_cert_abs - W_acc(1.0)',
            'implementation': 'tools/accumulator_sweep_v1/score.py derived (L8, imported)'},
        'definitions_per_node_family': 'd_noevent, d_same, d_acc(0.5), d_acc(1.0) = largest d from which downward the '
                                       'condition holds over the measured d set (d = 0 is lossless by the gate); d_half '
                                       '= smallest measured d at or below half. Coarse: only measured d values.',
        'statistics': {
            'scores': 'expected credit (1/t when the label is among t tied maxima, failed image 0) and lowest-index '
                      'Top-1 (contract tie order)',
            'paired': 'every policy against wide of the same case on the same images: tools.analysis.b2_ties.paired '
                      '(expected credit) and tools.analysis.b_stage_balanced_comparisons.paired_outcomes (lowest index, '
                      'exact McNemar p), 95 % paired bootstrap intervals',
            'counts': 'images with a changed Top-1 class, with an accumulator event, with a changed output, failed; per-node '
                      'event rates (share of images, share of outputs) grouped by node kind (stem, depthwise, '
                      'pointwise_expand, pointwise_project, se_reduce, se_expand, classifier; tools/accumulator_sweep_mn/'
                      'certs.py) at W_noevent-1 ... W_half and at the measured d values',
            'no_multiplicity_claims': 'descriptive development evidence; no hypothesis is confirmed here'},
        'simulator_comparison': 'exact wide arm (1k) against lane L1\'s sealed B2 readout of the same configuration '
                                'identity (artifacts/experiment_b2/matrix/readout/*.npz located through the matrix cell '
                                'whose configuration_sha256 equals the export\'s retained B2_configuration_sha256), under '
                                'the simulator\'s retained torch.topk order, the lowest-index rule and expected credit; '
                                'paired intervals and discordant counts; the code divergence of the 32-image MN panels '
                                'beside it',
        'budget': {
            'timing_basis': 'step 1 smoke: 64 images of mobilenet_v3_large-int8 wide in 16.5 s execution at batch 8, '
                            'shared GPU (0.26 s/image); MN document: 0.34-0.36 s/image shared. About 35 s setup per call.',
            'per_run_estimate_s': {'location_128_images': 75, 'screen_1000_images': 335, 'screen_872_images': 300},
            'estimated_runs': {'priority_1': 20, 'location': 'about 100 sat.w/sat.struct runs + 21 sensitivity runs',
                               'priority_2': 'about 65', 'priority_3': 20, 'priority_4': 'about 30'},
            'estimated_gpu_job_hours': {'priority_1': 1.9, 'location': 2.5, 'priority_2': 5.4, 'priority_3': 1.9,
                                        'priority_4': 2.5, 'total': 14.2},
            'rule': 'GPU job time = sum of the engine ledger wall_seconds of this lane\'s predict calls on the 1f75c923 '
                    'run root from this protocol on. Priority 1, the location phase and priority 2 (the 1k bracket) are '
                    'run in full. Priorities 3 and 4 are launched only while that sum plus the estimate of the runs '
                    'already running is below 14 hours; what is not run is listed as deferred. If the location phase '
                    'has to be coarsened, the sensitivity runs are dropped first (addendum).'},
    }


def main():
    if PATH.exists():
        sys.exit(f'{PATH} exists; the protocol is written once (use an addendum)')
    p = build()
    PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PATH, 'x') as handle:
        json.dump(p, handle, indent=1)
        handle.write('\n')
    print(PATH, hashlib.sha256(PATH.read_bytes()).hexdigest(), p['written'])


if __name__ == '__main__':
    main()
