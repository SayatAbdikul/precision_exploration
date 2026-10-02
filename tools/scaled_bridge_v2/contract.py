"""Machine-readable contract and development protocols for scaled bridge v2."""
from __future__ import annotations
from .common import BASE, digest, immutable

CONTRACT = {
    'version': 'scaled-code-domain-bridge-2.0',
    'extends': 'scaled-code-domain-bridge-1 (tools/scaled_bridge_v1/common.py); v1 definitions hold unless restated',
    'recipe': 'data, not code: export schema scaled-bridge-export-2 names per tensor the codebook, the scale and '
              'whether the boundary stores; weight grid integers, per-output-channel weight scales and biases are arrays',
    'codebook': 'ascending integer units on a 2^-shift grid; levels and adjacent midpoints exact in binary64; '
                'stored code = lowest code of the level; midpoint ties prefer even code parity, then lower code',
    'scales': 'positive finite; consumed as binary64 values (B recipes: retained FP32 scale bits promoted exactly)',
    'input': 'original torchvision FP32 transform, promoted to binary64, then the store rule',
    'wide_dot': 'exact integer sum of code-grid products in a signed accumulator certified lossless per output channel '
                '(int64 when max|a|*sum|w| < 2^63, otherwise two 64-bit limbs when < 2^127); one RNE conversion of the '
                'integer to binary64, which is exact when the certified bound is below 2^53; then exact scaling by '
                '2^-(input shift + weight shift)',
    'control_dot': 'sequential binary32 RNE fused multiply-add of exact code-level products in input-channel, '
                   'kernel-row, kernel-column order from +0; padded positions contribute a zero operand; computed on '
                   'grid integers (power-of-two equivariant: operands exact in binary32, no overflow, no subnormal), '
                   'then exact binary64 conversion and power-of-two scaling',
    'accumulator_policy': 'the dot accumulator is the only thing an arm may change; every other rule is shared',
    'mac_scale_bias': 'RNE64(RNE64(dot * input_scale) * weight_scale[channel]); then RNE64 add of the binary64 bias',
    'store': 'RNE64 divide by the store scale; nearest exact dyadic level; midpoint ties by the codebook rule; '
             'values beyond the finite endpoints clip; stored zero is canonical +0',
    'unstored_boundary': 'a node whose export has store=null passes its binary64 value on unrounded further; '
                         'MAC and average-pool inputs must be stored code tensors, otherwise the engine fails closed',
    'residual': 'each operand is RNE64(code_level * its scale) if stored, else its binary64 value; RNE64 add',
    'relu': 'stored input: max(code_level, +0) then RNE64(code_level * scale); unstored input: max(value, +0)',
    'maxpool': 'maximum over the window ignoring padding, on code levels (stored) or values (unstored); '
               'stored input then RNE64(code_level * scale)',
    'avgpool': 'global: exact code-grid sum (certified below 2^53 units), exact power-of-two conversion, RNE64 divide by '
               'the spatial count, RNE64 multiply by the input scale',
    'identity_flatten': 'preserve the state (codes and scale, or the unstored value); reshape only',
    'output': 'RNE64(code_level * scale) of the final stored tensor, or the unstored value',
    'top5': 'descending output value; ties ordered by ascending class index. v1 used torch.topk, whose tie order is '
            'unspecified; B used CUDA FP32 torch.topk. Records carry the number and identity of classes tied at the '
            'maximum so that tie-aware accuracy can be computed for either side',
    'zeros_specials': 'canonical +0 at stores and zero dots; nonfinite input, scale, raw value or output fails closed; '
                      'finite normalization overflow clips to an endpoint; no silent fallback',
    'certificates': 'per MAC node and output channel: absolute prefix bound max|a|*sum|w| (v1 definition), the tight '
                    'prefix interval for the input codebook range, and the interval after intersecting the input range '
                    'with [0,inf) behind ReLU-class producers; the engine checks the structural input range on every call',
    'not_admitted': 'formats without a dyadic grid (log4, log6, log8, nf4) stay on the rational fallback; '
                    'shared-exponent formats need the block contract (milestone M4)',
    'B_boundary': 'B reconstructs and normalizes in FP32 and reduces with framework FP32 convolutions; factoring the '
                  'scales after an exact code-domain dot and storing in binary64 changes rounding-level semantics',
}

PROTOCOL_M1 = {
    'version': 'scaled-bridge-v2-m1-development-1',
    'purpose': 'engine admission for ResNet18 scalar formats; development evidence, not a quality claim',
    'cases': 'resnet18, recipe family b1 (original B scales and weight codes): maxabs for every scalar format with a '
             'dyadic grid (int4 int5 int6 int8 q1_6 fp4_e2m1 fp5_e2m2 fp6_e2m3 fp6_e3m2 fp7_e3m3 fp8_e4m3fn fp8_e5m2 '
             'posit4_es0 posit6_es1 posit8_es1 ternary binary_pm1) and percentile_99_9 for int4 int5 int6 int8 '
             'fp4_e2m1 fp5_e2m2; the case list is fixed here before any panel is run',
    'samples': 'frozen ImageNet screen-1k list in ascending SHA256 order; prefix 8 (CPU/CUDA gate), prefix 32 (B replay '
               'panel); prefix 128 for at most two cases (resnet18-int8-maxabs and resnet18-fp8_e4m3fn-maxabs)',
    'calibration': 'none performed here; scales and weight codes are the retained B recipe evidence (calibration-2k list)',
    'arms': 'wide (exact) and control (sequential FP32 FMA) under contract 2.0; B anchor replayed with the original '
            'runtime in batches of eight and required to reproduce every retained Top-5 list',
    'witnesses': 'primitive rational conformance per codebook; first-image independent FX/limb reference (wide); three '
                 'actual-node rational dot checks per MAC node in both arms; eight-image CPU/CUDA identity of layer '
                 'codes, raw states, outputs, predictions and diagnostics; v1 sealed records for fp6_e2m3, fp6_e3m2, '
                 'fp7_e3m3 must be reproduced except for the documented Top-5 tie order',
    'statistics': 'paired Top-1/Top-5 differences wide-B, control-B, wide-FP32, control-FP32, control-wide with the v1 '
                  'paired multinomial bootstrap (10000 resamples, seed 20260928, pointwise 95%); tie-aware expected '
                  'Top-1 (credit 1/t when the label is among t classes tied at the maximum) for wide, control and B; '
                  'first-divergence layer counts',
    'stop_rule': 'no extension beyond the stated prefixes; a case with any witness failure is not admitted and is '
                 'reported as failed; no label-driven contract repair; panels of 32 are too small for quality claims',
    'cost': 'wall-clock seconds of every measurement invocation are sealed in the run ledger; GPU timings count only '
            'when taken under the shared GPU lock',
}




# Contract 2.1 (frozen 2026-10-01 before any policy or B2 panel). Every 2.0 rule holds; these are additions.
CONTRACT_2_1 = {
    **CONTRACT,
    'version': 'scaled-code-domain-bridge-2.1',
    'extends_2_0': 'every rule of scaled-code-domain-bridge-2.0 holds unchanged; the keys below are additions',
    'accumulator_policy_names': 'wide | control | sat.w<W> | sat.abs-<d> | sat.struct-<d> | fp16[.x<e>] | f21[.x<e>]; '
                                'the name is the complete specification and labels all evidence',
    'saturating_dot': 'signed two\'s-complement integer of W bits (2 <= W <= 63) whose unit is the exact product grid '
                      '2^-(input shift + weight shift); zero point 0; from 0, in the reduction order of the control '
                      'arm, each exact integer product is added exactly and the sum is clamped to '
                      '[-2^(W-1), 2^(W-1)-1] after every add; it never wraps; the final integer is converted like '
                      'the wide dot (one RNE to binary64, exact below 2^53 units, exact power-of-two scaling); '
                      'scales, bias and stores are the shared binary64 rules',
    'saturating_width': 'sat.w<W>: the same W at every MAC node. sat.abs-<d>: per node, the certified absolute width '
                        'of the node (signed_bits_absolute, maximum over its output channels) minus d. '
                        'sat.struct-<d>: per node, the certified structural width minus d. Floor 2 bits. '
                        'sat.abs-0 and sat.struct-0 are lossless by the certificate and must reproduce the wide arm '
                        'bit for bit with no saturation event',
    'float_dot': 'binary floating-point accumulator with p significand bits: from +0, in the same reduction order, '
                 'state = RNE(state + a*w*2^e) where a*w is the exact code-level product and e the scale exponent; '
                 'exactly one rounding per multiply-add (fused); gradual subnormals; overflow to +-infinity by the '
                 'IEEE rule; an infinite state stays infinite; NaN is unreachable because every product is finite; '
                 'the finite result is multiplied by 2^-e exactly and is a binary64 value; zero is canonical +0',
    'float_formats': 'fp16: IEEE binary16 (p=11, emin=-14, emax=15; manifest '
                     'public/formats/manifests/accumulators/fp16_e5m10_accumulator.json). f21: sign 1, exponent 8 '
                     'bits with bias 127, fraction 12 bits (p=13, emin=-126, emax=127; manifest '
                     'tools/scaled_bridge_v2/manifests/fp21_e8m12_accumulator.json)',
    'float_scale_exponent': 'e = 0 unless the name carries .x<e> (|e| <= 64): the accumulator then holds code-level '
                            'values times 2^e. Code levels are the manifest values (integer codes for int formats), '
                            'so e is the designer\'s choice of binary point and is always stated in the policy name',
    'nonfinite_result': 'a MAC node with any infinite accumulator result is a recorded failure of that image: the '
                        'record names the node and counts the infinite outputs by sign; the image has no output and '
                        'no prediction, is scored as incorrect and is reported separately as a failure; nothing is '
                        'clamped or replaced; other images of a batch are unaffected',
    'accumulator_record': 'per image and MAC node. Saturating: width, outputs, outputs with at least one clamp, number '
                          'of adds clamped at the high end and at the low end (each capped at 65535 per output). '
                          'Float: outputs, infinite outputs, reduction steps that ended infinite',
    'maxpool_code_passthrough': 'a max-pool exported with store=null whose input is a stored code tensor forwards '
                                'codes: the maximum code level of each window (padding ignored) is itself a level '
                                'of the same codebook; codes, codebook and scale pass on unchanged and no rounding '
                                'happens. Certificates look through such a node to the producer of the codes',
    'unsigned_codebooks': 'a codebook whose units are all >= 0; store, clipping, ties and certificates are the general '
                          'rules applied to that level set',
    'unstored_zero': 'an unstored (binary64) value or output that is zero is canonical +0',
    'operator_validation': 'the engine rejects operators and attributes it does not define (ceil-mode or dilated or '
                           'over-padded max-pool, non-global average pool, convolution padding modes), bias or scale '
                           'arrays that do not match the output channels and residual operands of different shapes',
    'top_k': 'descending output value, ties by ascending class index (lowest class index first), as in 2.0. A paired '
             'difference against a simulator (B, B2) is reported under both conventions for the simulator side: its '
             'retained order (torch.topk on CUDA FP32, unspecified among equal logits) and this rule applied to its '
             'logits; tie-aware expected accuracy is reported next to both',
    'width_convention': 'certified width of a MAC node = 1 sign bit + bit_length(max over output channels c of '
                        'max|a| * sum_k |w[c,k]|). a ranges over every level of the input tensor\'s codebook (the '
                        'admitted code domain, not observed data), in integer units of 2^-(input shift); behind a '
                        'ReLU-class producer the structural width uses a >= 0; w are the exported weight integers '
                        'of channel c in units of 2^-(weight shift), all K = C_in/groups*kh*kw taps; the accumulator '
                        'unit is the product grid 2^-(input shift + weight shift); the bound holds for every prefix '
                        'in any order; the bias is not accumulated in this register. Reported per channel, per node '
                        '(maximum over channels) and per network (maximum over nodes)',
    'B2_recipe_graphs': 'exports of format b2-recipe-export-1 are adapted without re-quantising: weight units are '
                        'the manifest levels of the exported weight codes, weight scales, activation scales and '
                        '(bias-corrected) biases are the exported FP32 values promoted exactly to binary64, a '
                        'boundary with quantizes=false becomes store=null, the unsigned variant becomes its own '
                        'codebook. The adapter verifies that level*scale in FP32 reproduces the exported '
                        'reconstructed weights bit for bit',
    'B2_boundary': 'B2 reduces with framework FP32 convolutions on FP32 reconstructed operands and stores in FP32; '
                   'the engine computes exact code-domain dots, factors the scales in binary64 and stores in '
                   'binary64. Fused values (conv into ReLU, residual add into ReLU) are binary64 here and FP32 in B2',
    'B2_not_exact': 'an unquantised network input (quantize_input=false) has no code grid, so its first convolution '
                    'cannot run in the code domain and the engine fails closed; hard-swish, hard-sigmoid, ReLU6 and '
                    'tensor multiply (fused_all, MobileNets) are not defined in 2.1; shared-exponent formats and '
                    'formats without a dyadic grid are not admitted',
}

PROTOCOL_MS = {
    'version': 'scaled-bridge-v2-ms-development-1',
    'purpose': 'admission of the accumulator policies of contract 2.1; development evidence, not a quality claim',
    'cases': 'resnet18-int8-maxabs-b1 and resnet18-fp7_e3m3-maxabs-b1 (the M1 exports, unchanged)',
    'policies': 'sat.abs-0 sat.struct-0 sat.struct-4 sat.w16 fp16 fp16.x-12 f21; fixed here before any panel',
    'samples': 'frozen ImageNet screen-1k list in ascending SHA256 order; prefix 8 on CPU and CUDA, prefix 32 on CUDA',
    'witnesses': 'primitive rational policy witnesses on both backends (saturation at both end points, exact landing, '
                 'recovery, overflow to both infinities, sticky infinity, subnormal ties and underflow, scale '
                 'exponents, 64-bit operand products, geometry); unit tests against a from-scratch rational reference '
                 'and IEEE float16; on the first image of every policy and case, rational checks of four fixed '
                 'outputs per MAC node plus the first output that reports an event; eight images with identical CPU '
                 'and CUDA records including accumulator counts and failures; sat.abs-0 and sat.struct-0 identical '
                 'to the wide arm on all 32 images with zero saturation; wide and control regressed against the '
                 'sealed M1 records (8 images, CUDA) because the engine sources changed',
    'reported': 'per policy and case: images failed (non-finite), Top-1 on 32 images with failures counted as '
                'incorrect, per-layer saturation or non-finite frequency, images whose stored codes differ from the '
                'wide arm. Thirty-two images support a gate, not a quality claim',
    'stop_rule': 'no extension beyond the stated prefixes and policies; a policy with any witness failure is reported '
                 'as failed and not admitted; no label-driven repair',
    'cost': 'wall-clock seconds of every measurement invocation are sealed in the run ledger',
}

PROTOCOL_MB2 = {
    'version': 'scaled-bridge-v2-mb2-development-1',
    'purpose': 'admission of repaired-recipe (B2 default) graphs; development evidence, not a quality claim',
    'cases': 'resnet18, B2 recipe default (fused_relu boundaries, unsigned codes for non-negative integer tensors, '
             'MSE-searched scales, empirical bias correction) for int8, int6, fp6_e2m3, fp7_e3m3, fp8_e4m3fn; '
             'exports produced by lane L1\'s export command and adapted without change',
    'samples': 'frozen ImageNet screen-1k list in ascending SHA256 order; prefix 8 (CPU/CUDA gate), prefix 32 (replay)',
    'arms': 'wide and control. The accumulator policies are admitted by MS on the same engine and need no B2 panel',
    'witnesses': 'adapter check that the integer weight codes reproduce B2\'s FP32 reconstructed weights bit for bit; '
                 'primitive rational conformance of every codebook of the export, including the unsigned variants; '
                 'certificates and widths; first-image whole-graph reference computed a different way (wide arm); '
                 'actual-node rational dot checks in both arms; eight images with identical CPU and CUDA records',
    'replay': 'B2\'s own simulator (tools/experiment_b2 engine, original CUDA runtime, batches of eight) is run on the '
              'same 32 images from the same export configuration; its logits, retained top-5 and per-boundary codes '
              'are recorded; the engine is compared with it by first differing boundary, changed codes, Top-1 and '
              'Top-5 under both tie conventions and tie-aware accuracy; bit identity with B2 is not expected '
              '(contract key B2_boundary) and is not a gate; the gate is that the replay reproduces B2\'s own '
              'retained predictions where B2 retained them',
    'statistics': 'paired Top-1 differences wide-B2 and control-wide with the v1 paired multinomial bootstrap (10000 '
                  'resamples, seed 20260928, pointwise 95%), under both tie conventions, plus tie-aware expected Top-1',
    'stop_rule': 'no extension beyond 32 images; a case with any witness failure is reported as failed; no '
                 'label-driven contract repair',
    'cost': 'wall-clock seconds of every measurement invocation are sealed in the run ledger',
}


# Contract 2.2 (frozen 2026-10-01 before any MobileNet panel). Every 2.1 rule holds; these are additions.
CONTRACT_2_2 = {
    **CONTRACT_2_1,
    'version': 'scaled-code-domain-bridge-2.2',
    'extends_2_1': 'every rule of scaled-code-domain-bridge-2.1 holds unchanged; the keys below are additions',
    'real_operand': 'r of a state is RNE64(code_level * scale) for stored codes and the binary64 value for an '
                    'unstored value (the 2.0 residual rule)',
    'relu6': 'stored input: max(code_level, +0), then RNE64(code_level * scale), then min(., 6); unstored input: '
             'min(max(value, +0), 6); 6 is exact, so the clamps do not round',
    'hardsigmoid': 't = RNE64(r + 3); t = min(max(t, +0), 6); out = RNE64(t / 6)',
    'hardswish': 't as for hardsigmoid; out = RNE64(RNE64(r * t) / 6) (the order of torch.nn.Hardswish, '
                 'x * relu6(x + 3) / 6); a zero result is canonical +0',
    'mul': 'tensor product (squeeze-excite). Operand shapes are equal, or one operand is (N, C, 1, 1) and is '
           'broadcast over the spatial positions of the other (N, C, H, W). Both operands stored: the exact '
           'integer product p = u_a * u_b of their grid integers (certified |p| < 2^53 from the two codebook '
           'ranges, so its binary64 value is exact), then RNE64(RNE64(p * 2^-(shift_a + shift_b) * s_a) * s_b) '
           'with a the first input in node order (the MAC post-operation of a one-term dot). Otherwise '
           'RNE64(r_a * r_b). A zero result is canonical +0',
    'mul_certificate': 'per mul node: the largest |u_a * u_b| over the two codebook ranges, its signed width '
                       '(1 + bit_length) and the product shift; the engine refuses a product bound >= 2^53',
    'closure_certificate': 'in addition to the 2.1 widths (unchanged; sat.struct-<d> keeps the 2.1 structural '
                           'width so that evidence of earlier digests stays comparable), every MAC node reports '
                           'signed_bits_closure: the prefix interval for the input range obtained by interval '
                           'propagation through the graph under these rules. Stored tensors carry unit bounds, '
                           'unstored values binary64 bounds; a store maps bounds through the store rule (monotone); '
                           'relu, relu6, hardsigmoid and hardswish map bounds through their definitions (hardswish '
                           'has its minimum -0.375 at -1.5 and is monotone on either side; RNE64 is monotone); '
                           'add adds the bounds with RNE64; mul takes the extreme corner products; average and '
                           'max pooling, identity and flatten keep the bounds; conv, linear and the input are '
                           'bounded only by their codebook. The interval is widened to contain zero. The engine '
                           'checks every MAC input against its closure range on every call and fails closed',
    'graph_validation_2_2': 'the B2 adapter requires the exported node rows to equal the operator rows of the '
                            'independently loaded folded FX graph and refuses an in-place activation whose input '
                            'tensor has another consumer; average pooling must be global',
    'B2_not_exact': 'an unquantised network input (quantize_input=false) has no code grid, so its first convolution '
                    'cannot run in the code domain and the engine fails closed; shared-exponent formats and '
                    'formats without a dyadic grid are not admitted. Hard-swish, hard-sigmoid, ReLU6 and tensor '
                    'multiply are defined from 2.2 on, on stored and on unstored operands',
}

PROTOCOL_MN = {
    'version': 'scaled-bridge-v2-mn-development-1',
    'purpose': 'admission of MobileNetV2 and MobileNetV3-Large repaired-recipe (B2 default) graphs under contract '
               '2.2; development evidence on the screen-1k list, not a quality claim',
    'cases': 'mobilenet_v2 and mobilenet_v3_large, B2 recipe default (fused_relu boundaries, unsigned codes for '
             'non-negative integer tensors, MSE-searched scales, empirical bias correction): int8 (the existing '
             'lane-L1 exports), fp6_e2m3 and fp8_e4m3fn; int6 and fp7_e3m3 only while the lane stays inside its '
             'disk allowance (decided from measured disk use before each export, never from results); exports '
             'produced by lane L1\'s export command and adapted without change',
    'samples': 'frozen ImageNet screen-1k list in ascending SHA256 order; prefix 8 (CPU/CUDA gate), prefix 32 '
               '(CUDA arms and the B2 replay); no panel beyond 32 images; no held-out image is used',
    'arms': 'wide and control on every case',
    'witnesses': 'adapter checks (weights reproduce B2 FP32 reconstruction bit for bit; codebook tables; graph '
                 'rows equal the independently loaded FX graph; in-place activations single-consumer); primitive '
                 'rational conformance of every codebook of the export including the 2.2 operators (relu6, '
                 'hardsigmoid and hardswish on every code level at several scales and on directed binary64 '
                 'values; mul on code pairs within and across the codebooks of the export); certificates with '
                 'per-node widths (absolute, range, structural, closure) and mul product certificates; first-image '
                 'whole-graph FX/limb reference (wide); first-image whole-graph policy witness written from the '
                 'contract text and fed from lane L1\'s raw export (control arm); actual-node rational dot checks '
                 'in both arms; eight images with identical CPU and CUDA records',
    'replay': 'B2\'s own simulator rebuilt from the export configuration on the same 32 images in batches of eight; '
              'the gate requires that every one of the 32 images has a retained B2 prediction, that the replay '
              'reproduces each retained Top-5 and that the replay inputs are bit-identical to the engine inputs; '
              'bit identity with B2 is not expected and is not a gate',
    'policy_gate': 'one case per network (int8): sat.struct-0 (lossless by certificate: identical to the wide arm '
                   'with no saturation), sat.struct-4, fp16.x<e> with e = 15 - ub where ub is the largest '
                   'bit_length(max_abs_prefix_units) - product_shift over the MAC nodes (or plain fp16 when that '
                   'is not negative), and f21; eight images on CPU and on CUDA with identical records including '
                   'counters, first-image rational dot checks, and the whole-graph policy witness on the first image',
    'statistics': 'paired Top-1 differences wide-B2 (both tie conventions) and control-wide with the v1 paired '
                  'multinomial bootstrap (10000 resamples, seed 20260928, pointwise 95%), tie-aware expected Top-1',
    'regression': 'the engine sources change, so sealed record sets of digest 7c6344af (ResNet18: original-recipe '
                  'and B2 cases, wide, control and the accumulator policies) must be reproduced bit for bit on 8 '
                  'images each before any MobileNet evidence is used',
    'stop_rule': 'no extension beyond the stated prefixes; a case with any witness failure is reported as failed and '
                 'not admitted; no label-driven contract repair',
    'cost': 'wall-clock seconds of every measurement invocation are sealed in the run ledger',
}


def seal_protocol(name, protocol, contract=None):
    contract = contract or {'m1': CONTRACT, 'mn': CONTRACT_2_2}.get(name, CONTRACT_2_1)
    path = BASE / 'protocols' / f'{name}-{digest({"contract": contract, "protocol": protocol})[:16]}.json'
    immutable(path, {'contract': contract, 'protocol': protocol})
    return path
