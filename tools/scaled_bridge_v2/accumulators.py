"""Dot-product accumulator policies.

A policy names one kernel accumulator, its parameters for a MAC node and the
host-side precondition under which the kernel implementation equals its
definition. Graph, recipe and store code never depend on the policy.

Policy specifications (file-name safe strings):
  wide            exact integer sum (certified lossless), one RNE to binary64
  control         sequential binary32 fused multiply-add
  sat.w<W>        signed saturating integer of W bits on every node
  sat.abs-<d>     per node: W = certified absolute width of the node minus d
  sat.struct-<d>  per node: W = certified structural width of the node minus d
  fp16[.x<e>]     IEEE binary16 accumulator; optional scale exponent e
  f21[.x<e>]      21-bit float (sign 1, exponent 8 bias 127, fraction 12)
A float accumulator holds code-level values multiplied by 2^e (e = 0 unless
given); the result is divided by 2^e exactly. e is part of the policy name.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
import numpy as np
from .common import ROOT, PACKAGE

FLOATS = {
    'fp16': {'P': 11, 'emin': -24, 'emax': 15, 'bits': 16,
             'manifest': ROOT / 'public/formats/manifests/accumulators/fp16_e5m10_accumulator.json'},
    'f21': {'P': 13, 'emin': -138, 'emax': 127, 'bits': 21,
            'manifest': PACKAGE / 'manifests/fp21_e8m12_accumulator.json'},
}
MIN_SAT_WIDTH, MAX_SAT_WIDTH = 2, 63


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str          # wide | control | sat | float
    kernel_id: int
    definition: str
    reference: str = ''    # sat: global | abs | struct
    amount: int = 0        # sat: W (global) or the reduction d
    float_format: str = ''
    scale_exponent: int = 0

    @property
    def needs_float32_exact_operands(self):
        return self.kind == 'control'

    @property
    def parameterised(self):
        return self.kind in ('sat', 'float')

    def width(self, certificate=None):
        if self.reference == 'global':
            return self.amount
        if certificate is None:
            raise ValueError('a relative saturating width needs a certificate')
        base = certificate['signed_bits_absolute' if self.reference == 'abs' else 'signed_bits_structural']
        return max(MIN_SAT_WIDTH, base - self.amount)

    def node(self, shift, certificate=None):
        """Kernel parameters and their description for one reduction on a 2^-shift product grid."""
        if self.kind == 'sat':
            width = self.width(certificate)
            if not MIN_SAT_WIDTH <= width <= MAX_SAT_WIDTH:
                raise ValueError(f'saturating width {width} is outside {MIN_SAT_WIDTH}..{MAX_SAT_WIDTH}')
            return np.array([width, 0, 0, 0], dtype=np.int64), {'width': width}
        if self.kind == 'float':
            f = FLOATS[self.float_format]
            emin = f['emin'] + shift - self.scale_exponent
            emax = f['emax'] + shift - self.scale_exponent
            if not (-1000 <= emin <= 127 and -1000 <= emax <= 1000):
                raise ValueError('float accumulator exponent range is outside the kernel domain')
            return np.array([f['P'], emin, emax, 0], dtype=np.int64), {
                'format': self.float_format, 'scale_exponent': self.scale_exponent}
        return None, {}

    def precondition(self, certificate, input_book, weight_book):
        if certificate['max_abs_prefix_units'] >= 2**127:
            raise ValueError(f'{certificate["node"]}: prefix bound exceeds the two-limb exact accumulator')
        if self.needs_float32_exact_operands and not (input_book.float32_exact and weight_book.float32_exact):
            raise ValueError(f'{certificate["node"]}: operands are not exact binary32 values')
        if self.needs_float32_exact_operands and not (
                certificate['max_abs_prefix_units'] < 2**126 and certificate['product_shift'] <= 100):
            raise ValueError(f'{certificate["node"]}: outside the binary32 no-overflow/no-subnormal domain')
        self.node(certificate['product_shift'], certificate)


DEFINITIONS = {
    'wide': 'exact integer sum of code-grid products in a certified lossless signed accumulator (64-bit, or two '
            '64-bit limbs), then one RNE conversion to binary64 (exact below 2^53 units) and an exact '
            'power-of-two scaling to code-level units',
    'control': 'sequential binary32 RNE fused multiply-add of exact code-level products in input-channel, '
               'kernel-row, kernel-column order from +0; implemented on grid integers (power-of-two '
               'equivariant), then exact binary64 power-of-two conversion',
    'sat': 'signed two\'s-complement integer of W bits on the exact product grid (unit 2^-(input shift + weight '
           'shift)); from 0, in reduction order, each exact product is added exactly and the sum is clamped to '
           '[-2^(W-1), 2^(W-1)-1] after every add; it never wraps; then the wide conversion',
    'float': 'binary floating-point accumulator; from +0, in reduction order, state = RNE(state + exact '
             'product * 2^e) with one rounding per multiply-add, gradual subnormals, overflow to infinity; an '
             'infinite state stays infinite; the result is multiplied by 2^-e exactly; a non-finite result is '
             'recorded and never clamped',
}
POLICIES = {'wide': Policy('wide', 'wide', 0, DEFINITIONS['wide']),
            'control': Policy('control', 'control', 1, DEFINITIONS['control'])}
_SAT = re.compile(r'sat\.(?:w(\d+)|(abs|struct)-(\d+))')
_FLOAT = re.compile(r'(fp16|f21)(?:\.x(-?\d+))?')


def resolve(spec):
    """Policy object for a specification string; unknown specifications fail closed."""
    if isinstance(spec, Policy):
        return spec
    if spec in POLICIES:
        return POLICIES[spec]
    m = _SAT.fullmatch(spec)
    if m:
        if m.group(1):
            width = int(m.group(1))
            if not MIN_SAT_WIDTH <= width <= MAX_SAT_WIDTH or spec != f'sat.w{width}':
                raise ValueError('unsupported saturating width')
            return Policy(spec, 'sat', 2, DEFINITIONS['sat'], 'global', width)
        if spec != f'sat.{m.group(2)}-{int(m.group(3))}':
            raise ValueError('non-canonical policy name')
        return Policy(spec, 'sat', 2, DEFINITIONS['sat'], m.group(2), int(m.group(3)))
    m = _FLOAT.fullmatch(spec)
    if m:
        exponent = int(m.group(2) or 0)
        if abs(exponent) > 64 or (m.group(2) is not None and exponent == 0) or spec != m.group(1) + (f'.x{exponent}' if m.group(2) is not None else ''):
            raise ValueError('unsupported or non-canonical float accumulator name')
        return Policy(spec, 'float', 3, DEFINITIONS['float'], float_format=m.group(1), scale_exponent=exponent)
    raise ValueError(f'unknown accumulator policy {spec!r}')
