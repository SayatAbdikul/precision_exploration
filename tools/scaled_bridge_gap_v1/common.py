"""Shared identity, protocol and sealed-record helpers for the 1k gap study.

The sealed v1 engine (tools/scaled_bridge_v1) is imported read-only and never
modified; its ledger and run directories are never written.  All new evidence
lives under artifacts/scaled_bridge_gap_v1.
"""
from __future__ import annotations
from pathlib import Path
from tools.experiment_b.common import ROOT, digest, file_hash, unseal, seal
from tools.phase3.common import reference, checked
from tools.scaled_bridge_v1 import common as v1

BASE = ROOT / 'artifacts/scaled_bridge_gap_v1'
RESULTS = ROOT / 'results/summaries/scaled-bridge-gap-v1'
FORMATS = v1.FORMATS
PANEL = 1000
V1_SOURCE_DIGEST = '5e11c33940783230a36ccda4ff624628be5137ca0bc345c8d2a5029f62154e51'

PROTOCOL = {
 'version': 'scaled-bridge-gap-1k-1',
 'date': '2026-10-01',
 'question': 'Is the 128-image wide-minus-B top-1 gap (-3.9 to -4.7 pp) of the sealed scaled bridge v1 real at n=1000, '
             'and is the FP32-accumulator control still indistinguishable from the exact (wide) arm?',
 'evidence_level': 'development: these three cases and the 1k screen were used in earlier selection; no confirmation claim',
 'engine': 'sealed tools/scaled_bridge_v1 exactly as enrolled (source digest ' + V1_SOURCE_DIGEST + '), called read-only; '
           'v1 ledger, runs and reports are not written',
 'cases': ['resnet18/' + f + '/maxabs' for f in FORMATS],
 'panel': 'frozen imagenet_screen_1k, all 1000 images, v1 order (ascending SHA-256 manifest order); first 128 = v1 sealed panel',
 'arms': {
   'wide': 'v1 Engine mode=wide, backend=cuda, batch 1, v1 record schema',
   'control': 'v1 Engine mode=control (sequential FP32 FMA dot), backend=cuda, batch 1, v1 record schema',
   'B': 'v1 TracedB replay of the original B QDQ interpreter, original CUDA runtime, original batch-8 membership; every Top-5 must equal '
        'the retained B and FP32 prediction records of artifacts/experiment_b/predictions (all 1000 images)',
   'FP32': 'retained FP32 baseline predictions (no rerun; checked again by the B replay)'},
 'reproduction_gate': 'images 0..127: wide/control numerical records (layers, output, top5, diagnostic signature) must equal the v1 sealed CUDA '
                      'records; B replay layers/top5/FP32_top5 must equal v1 sealed B traces. Any mismatch stops the study until explained.',
 'statistics': {
   'paired': 'tools/analysis/b_stage_balanced_comparisons.paired_outcomes (paired binary-outcome multinomial bootstrap, 10000 resamples, '
             'seed 20260927, pointwise 95% interval, exact McNemar p); top-1 primary, top-5 secondary',
   'comparisons': ['wide - B', 'control - wide', 'wide - FP32', 'control - FP32', 'B - FP32'],
   'direction_counts': 'per comparison: images correct only in left, only in right; for wide vs B also changed top-1 predictions '
                       'split by B-correct->wide-wrong, B-wrong->wide-correct, wrong->different-wrong; exact two-sided sign test on the asymmetry',
   'discordance_bound': 'control vs wide: Clopper-Pearson one-sided 95% upper bound on the per-image rate of changed top-1 predictions, '
                        'changed top-1 correctness, and changed ordered top-5 lists, at n=1000',
   'regression_to_mean': 'report all comparisons separately on images 0..127 (v1 panel) and 128..999 (new images)',
   'first_divergence': 'per image, first node in execution order whose stored-code hash differs (wide vs B, control vs wide)',
   'multiplicity': 'pointwise intervals only; three cases reported together; no simultaneous claim'},
 'stop_rule': 'fixed n=1000 per arm, no early stopping on accuracy; stop on any reproduction mismatch, nonfinite state, failed B Top-5 '
              'reproduction or source drift. Mechanism diagnostics (step 4) are run for a case only if its 1k wide-minus-B top-1 '
              'point estimate is <= -1.0 pp or its 95% interval excludes 0; a separate diagnostic protocol is written before they run.',
 'resources': 'one case and arm per job under artifacts/agent_orchestration/gpu.lock; wall-clock and locked seconds logged per job',
}


def sources():
    files = sorted((ROOT / 'tools/scaled_bridge_gap_v1').glob('*.py'))
    files += sorted((ROOT / 'tools/run').glob('scaled_bridge_gap*'))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if unseal(path) != value:
            raise ValueError(f'immutable gap evidence differs: {path}')
    else:
        seal(path, value)


def protocol_path():
    return BASE / 'protocol-1k-v1.json'


def write_protocol():
    immutable(protocol_path(), PROTOCOL)
    return protocol_path()


def require_protocol():
    if unseal(protocol_path()) != PROTOCOL:
        raise ValueError('protocol drift')


def v1_run_root():
    root = v1.run_root()
    if root.name != V1_SOURCE_DIGEST:
        raise ValueError('sealed v1 engine source drift')
    return root


def run_root():
    return BASE / 'runs' / V1_SOURCE_DIGEST


def arm_folder(name, arm):
    return run_root() / name / (arm if arm == 'B' else f'{arm}-cuda')
