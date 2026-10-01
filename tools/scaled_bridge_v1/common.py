from __future__ import annotations
import hashlib
import json
from pathlib import Path
from tools.experiment_b.common import ROOT, digest, file_hash, unseal, seal, atomic_json
from tools.phase3.common import reference, checked

BASE = ROOT / 'artifacts/scaled_bridge_v1'
FORMATS = ('fp6_e2m3', 'fp6_e3m2', 'fp7_e3m3')
CONTRACT = {
 'version': 'scaled-code-domain-bridge-1', 'model': 'resnet18', 'recipe': 'maxabs',
 'selection': 'development-informed; not independent confirmation',
 'weight_codes': 'original B FP32 normalization, finite levels and code-parity midpoint ties',
 'scales': 'retained positive finite FP32 bits, promoted exactly to binary64; per-node activation and per-output-channel weight',
 'input': 'original torchvision FP32 transform, then binary64 normalization and finite code RNE',
 'wide_dot': 'exact int64 sum of code-grid integer products; proved every prefix <2^53 units; convert exactly to binary64 code-level dot',
 'control_dot': 'same lexicographic input-channel/kernel-row/kernel-column order; FP32 RNE fused multiply-add of exact code-level products, +0 initial state',
 'control_implementation': 'power-of-two equivariant FP32 FMA on grid integers, then exact binary64 power-of-two conversion; no overflow/subnormal in admitted domain',
 'mac_scale_bias': 'RNE64(RNE64(dot * activation_scale) * weight_scale[channel]); then separate RNE64 add of exactly promoted folded FP32 bias',
 'residual': 'reconstruct each operand with RNE64(code_level * its scale), add RNE64, requantize once to output node scale',
 'relu': 'max(code_level, +0), RNE64 reconstruction, output requantization',
 'maxpool': 'maximum code level ignoring padding, RNE64 reconstruction, output requantization',
 'avgpool': 'exact code-grid sum, exact power-of-two conversion, RNE64 divide by spatial count, RNE64 multiply input scale, output requantization',
 'store': 'RNE64 divide by output FP32 scale promoted to binary64, nearest exact dyadic code level, manifest parity/order ties, finite endpoint clipping',
 'zeros_specials': 'canonical +0 at all candidate code stores and zero dot; nonfinite source or raw output fails closed; finite normalization overflow clips endpoint; no silent fallback',
 'identity_flatten': 'preserve code/scale state and reshape only',
 'top5': 'torch topk on final reconstructed binary64 logits; retained B topk remains its original FP32 batch-8 result',
 'diagnostics': 'per-node code and raw-state hashes, finite clipping/tie/zero/normalization-overflow counts; new contract diagnostics, no claim of legacy A diagnostic identity',
 'B_boundary': 'B reconstructs and normalizes in FP32 and uses framework FP32 reductions; factoring scales and using binary64 stores changes semantics',
}
PROTOCOL = {
 'version': 'scaled-bridge-paired-development-1', 'formats': list(FORMATS),
 'sample_order': 'existing ascending SHA256 ImageNet development manifest; prefix 8,32,128',
 'B_replay': 'original runtime and batch membership of eight; original artifact roots read-only',
 'gate': 'full independent FX/FP64 wide reference first image plus fixed actual-node rational checks in both arms; eight CPU/CUDA images per arm with exact state/code/output/prediction/diagnostic equality',
 'stages': [1,8,32,128], 'ceiling_worker_seconds': 14400,
 'promotion': 'after all admitted cases reach32: extend in frozen format order if wide top1>=40%, wide top1 is no more than20 percentage points below same-panel B, no unexplained gate mismatch, and both128 arms fit measured budget',
 'futility': 'stop a case on unexplained numerical mismatch, failed reference, nonfinite state, invalid bound or missed B replay; no label-driven contract repair',
 'statistics': 'paired binary-outcome multinomial bootstrap10000, seed20260928, pointwise95% intervals; changed top1/top5; no simultaneous or confirmation claim',
 'reservation': 'at least1.5x measured stage prediction plus setup allowance; timed-out attempts and unreconciled reservations remain charged',
}

def immutable(path, value):
    path = Path(path)
    if path.exists():
        if unseal(path) != value:
            raise ValueError(f'immutable bridge evidence differs: {path}')
    else:
        seal(path, value)

def sources():
    files = sorted((ROOT/'tools/scaled_bridge_v1').glob('*'))
    files += [ROOT/'tools/run/scaled_bridge.py']
    files=[p for p in files if p.suffix in {'.py','.cpp','.cu','.h'}]
    files += [ROOT/'public/inference/reference/arithmetic.py',
              ROOT/'public/formats/manifests/accumulators/fp32_e8m23_accumulator.json',
              ROOT/'public/formats/manifests/accumulators/fp64_e11m52_accumulator.json']
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}

def enrollment_path():
    return BASE/'enrollments'/(digest(sources())+'.json')

def run_root():
    return BASE/'runs'/digest(sources())

def array_hash(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    return digest({'dtype': a.dtype.str, 'shape': list(a.shape),
                   'bytes': hashlib.sha256(a.tobytes()).hexdigest()})

def save_npz(path, arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(array_hash(old[k]) != array_hash(v) for k,v in arrays.items()):
                raise ValueError('immutable bridge arrays differ')
    else:
        tmp = path.with_suffix('.partial')
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        tmp.replace(path)
    return reference(path)
