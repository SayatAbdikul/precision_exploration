"""Policy names of the contract 2.3 draft. A name is the complete specification; non-canonical names fail closed.

single  := base modifiers
base    := wide | control | sat.<width> | wrap.<width> | flt.e<E>m<M> | fp16 | f21
width   := w<W> (W = 2..127) | abs-<d> | struct-<d>
           abs/struct: the stage's certified width (certify.py: the 2.1 node width for a plain register, the
           widened width of A2.2/A2.4/A2.6 when the stage drops low bits, is a chunk stage or holds a bias) minus d,
           floor 2. `struct` is refused on shared-exponent nodes (no structural range is defined there).
modifiers, in this order, each at most once:
  .l<L>        integer registers: the register grid is 2^L term units, L in -126..126, L != 0 (L = 0 is written by
               leaving the modifier out; L < 0 keeps |L| bits below the term grid, which matters only for
               shared-exponent terms)
  .rne         integer registers: each term enters the grid rounded to nearest even (default: floor, the bit
               drop). Without .l<L> it means RNE entry at L = 0, which differs from floor only for terms that are
               not integers (shared-exponent nodes); on scalar nodes it is the same arithmetic as the plain name.
  .x<e>        floats: scale exponent, |e| <= 64, e != 0 (as in contract 2.1): the accumulator holds the code-level
               value times 2^e and the finite result is multiplied by 2^-e (representable code-level values are the
               format's numbers times 2^-e)
  .ftz         floats: flush to zero; an exact add result (or bias) below the smallest normal in magnitude
               becomes +0 (tininess before rounding)
  .ofsat       floats: overflow saturates to the largest finite value instead of infinity
  .nf          floats: not fused; the product is rounded to the accumulator format before the add
  .rz          floats: every rounding of the accumulator (adds, the bias, .nf product roundings) is toward zero;
               an overflow then gives the largest finite value (.ofsat is refused as redundant). It does NOT act
               on the product format of .pr-, which has its own -rz.
  .pr-e<E'>m<M'>[-x<e'>][-rz][-ftz]
               every product is first rounded to the IEEE layout E'/M' (top exponent code reserved, as for flt)
               before it enters the accumulator: the code-level product p becomes R'(p * 2^e') * 2^-e' (the sign of
               .x<e>; default e' = 0, |e'| <= 64, nonzero when written); e' is independent of the accumulator's
               .x<e>, the product is never rounded in accumulator units. Rounding RNE, or toward zero
               with -rz; gradual subnormals, or flush to zero (tininess before rounding) with -ftz; an overflow
               always saturates to the largest finite value of its sign. Allowed on wide, integer and float
               stages, not on control and not on the outer stage of ch.
  .elt         shared-exponent nodes: one term per element product u_a*u_w*2^(e_a+e_w) in K order (a
               dequantising scalar MAC) instead of one term per block sum. Refused on the outer stage of ch and
               together with .ord-hwc (hwc has no meaning on shared-exponent nodes); the engine refuses it on
               scalar nodes.
  .ord-<o>     summation order rev | hwc | tree (default chw, the engine's order). Integer and float stages only;
               on the outer stage of ch only .ord-tree; hwc needs a convolution's tap geometry and is refused on
               shared-exponent nodes.
  .bias-pre    the node bias, rounded onto the register grid, is the initial register content
  .bias-post   the same, added through the register after the last term (with .ord-tree: one more add at the
               root; .bias-pre is refused with .ord-tree)
chunked := ch<C>_<inner>_<outer>   two-stage: consecutive chunks of C TERMS (taps on scalar nodes, block sums or,
                                   with .elt, element products on shared-exponent nodes) of the inner order,
                                   each reduced by <inner> from zero; the chunk results are reduced by <outer>,
                                   each chunk value entering the outer add exactly (fused); a bias modifier is
                                   allowed on <outer> only; <outer> refuses the term-level modifiers .nf, .pr-
                                   and .elt and every order but .ord-tree
Aliases kept from contract 2.1: fp16 = flt.e5m10, f21 = flt.e8m12 (same arithmetic; canonical() gives the 2.3 name).
`control` (contract 2.0) is binary32 fused multiply-add = flt.e8m23 under its operand precondition.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import re
from .orders import ORDERS
from .registers import MIN_WIDTH, MAX_WIDTH, MAX_LOW_BITS

ALIASES = {'fp16': (5, 10), 'f21': (8, 12)}
MAX_SCALE = 64
_BASE = re.compile(r'(wide|control|fp16|f21|flt\.e(\d+)m(\d+)|(sat|wrap)\.(?:w(\d+)|(abs|struct)-(\d+)))((?:\.[a-z0-9-]+)*)')
_PRODUCT = re.compile(r'pr-e(\d+)m(\d+)(?:-x(-?\d+))?(-rz)?(-ftz)?')


@dataclass(frozen=True)
class Product:
    """Rounding format of every product (`.pr-`): IEEE layout E/M, units code level * 2^x."""
    E: int
    M: int
    x: int = 0
    rz: bool = False
    ftz: bool = False

    @property
    def token(self):
        return (f'pr-e{self.E}m{self.M}' + (f'-x{self.x}' if self.x else '') + ('-rz' if self.rz else '')
                + ('-ftz' if self.ftz else ''))


@dataclass(frozen=True)
class Policy:
    kind: str                    # wide | control | int | float | chunk
    overflow: str = ''           # int: sat | wrap
    reference: str = ''          # int: global | abs | struct
    amount: int = 0              # int: W (global) or d
    low_bits: int = 0
    low_mode: str = 'floor'
    E: int = 0
    M: int = 0
    alias: str = ''              # fp16 | f21 when written that way
    scale_exponent: int = 0
    ftz: bool = False
    ofsat: bool = False
    nonfused: bool = False
    rz: bool = False
    product: Product = None      # product rounding format, or None
    element: bool = False        # .elt (shared-exponent nodes: element-product terms)
    order: str = 'chw'
    bias: str = ''               # '' | pre | post
    chunk: int = 0
    inner: 'Policy' = None
    outer: 'Policy' = None

    def width(self, certified=None):
        """Register width: W for w<W>; for abs-<d>/struct-<d> the stage's certified width (an int, from
        certify.py) minus d, floor 2."""
        if self.reference == 'global':
            return self.amount
        if certified is None:
            raise ValueError('a relative register width needs the certified width of its stage (certify.py)')
        return max(MIN_WIDTH, int(certified) - self.amount)

    def name(self, canonical=False):
        if self.kind == 'chunk':
            return f'ch{self.chunk}_{self.inner.name(canonical)}_{self.outer.name(canonical)}'
        if self.kind in ('wide', 'control'):
            text = self.kind
        elif self.kind == 'int':
            text = self.overflow + '.' + (f'w{self.amount}' if self.reference == 'global' else f'{self.reference}-{self.amount}')
            text += (f'.l{self.low_bits}' if self.low_bits else '') + ('.rne' if self.low_mode == 'rne' else '')
        else:
            text = self.alias if self.alias and not canonical else f'flt.e{self.E}m{self.M}'
            text += (f'.x{self.scale_exponent}' if self.scale_exponent else '') + ('.ftz' if self.ftz else '')
            text += ('.ofsat' if self.ofsat else '') + ('.nf' if self.nonfused else '') + ('.rz' if self.rz else '')
        text += ('.' + self.product.token if self.product else '') + ('.elt' if self.element else '')
        text += (f'.ord-{self.order}' if self.order != 'chw' else '') + (f'.bias-{self.bias}' if self.bias else '')
        return text

    @property
    def canonical(self):
        return self.name(canonical=True)

    @property
    def stages(self):
        return (self.inner, self.outer) if self.kind == 'chunk' else (self,)

    @property
    def elementwise(self):
        return self.stages[0].element


def _layout(E, M, what):
    if not (2 <= E <= 11 and 1 <= M <= 52):
        raise ValueError(f'{what} layout outside E 2..11, M 1..52')


def _single(spec):
    m = _BASE.fullmatch(spec)
    if not m:
        raise ValueError(f'unknown accumulator policy {spec!r}')
    base, mods = m.group(1), m.group(8)
    if base in ('wide', 'control'):
        p = Policy(base)
    elif base in ALIASES or base.startswith('flt.'):
        E, M = ALIASES[base] if base in ALIASES else (int(m.group(2)), int(m.group(3)))
        _layout(E, M, 'float accumulator')
        p = Policy('float', E=E, M=M, alias=base if base in ALIASES else '')
    else:
        reference, amount = ('global', int(m.group(5))) if m.group(5) else (m.group(6), int(m.group(7)))
        if reference == 'global' and not MIN_WIDTH <= amount <= MAX_WIDTH:
            raise ValueError(f'register width outside {MIN_WIDTH}..{MAX_WIDTH}')
        p = Policy('int', overflow=m.group(4), reference=reference, amount=amount)
    for token in (mods.split('.')[1:] if mods else []):
        product = _PRODUCT.fullmatch(token)
        if p.kind == 'int' and re.fullmatch(r'l-?\d+', token):
            p = replace(p, low_bits=int(token[1:]))
        elif p.kind == 'int' and token == 'rne':
            p = replace(p, low_mode='rne')
        elif p.kind == 'float' and re.fullmatch(r'x-?\d+', token):
            p = replace(p, scale_exponent=int(token[1:]))
        elif p.kind == 'float' and token in ('ftz', 'ofsat', 'nf', 'rz'):
            p = replace(p, **{{'ftz': 'ftz', 'ofsat': 'ofsat', 'nf': 'nonfused', 'rz': 'rz'}[token]: True})
        elif p.kind in ('int', 'float', 'wide') and product:
            E, M = int(product.group(1)), int(product.group(2))
            _layout(E, M, 'product format')
            x = int(product.group(3)) if product.group(3) is not None else 0
            if abs(x) > MAX_SCALE or (product.group(3) is not None and x == 0):
                raise ValueError(f'product scale exponent outside -{MAX_SCALE}..{MAX_SCALE} or written as 0 in {spec!r}')
            p = replace(p, product=Product(E, M, x, bool(product.group(4)), bool(product.group(5))))
        elif token == 'elt':
            p = replace(p, element=True)
        elif token.startswith('ord-') and p.kind in ('int', 'float') and token[4:] in ORDERS and token[4:] != 'chw':
            p = replace(p, order=token[4:])
        elif token in ('bias-pre', 'bias-post') and p.kind != 'control':
            p = replace(p, bias=token[5:])
        else:
            raise ValueError(f'unknown or misplaced modifier {token!r} in {spec!r}')
    if abs(p.scale_exponent) > MAX_SCALE or abs(p.low_bits) > MAX_LOW_BITS or (p.order == 'tree' and p.bias == 'pre') \
            or (p.rz and p.ofsat) or (p.element and p.order == 'hwc'):
        raise ValueError(f'invalid modifiers in {spec!r}')
    if p.name() != spec:
        raise ValueError(f'non-canonical policy name {spec!r} (expected {p.name()!r})')
    return p


def resolve(spec):
    """Policy for a 2.3 name; unknown or non-canonical names fail closed."""
    if isinstance(spec, Policy):
        return spec
    m = re.fullmatch(r'ch(\d+)_([^_]+)_([^_]+)', spec)
    if m:
        chunk, inner, outer = int(m.group(1)), _single(m.group(2)), _single(m.group(3))
        if chunk < 1 or inner.bias or inner.kind == 'control' or outer.kind == 'control' or outer.product \
                or outer.element or outer.nonfused or outer.order not in ('chw', 'tree'):
            raise ValueError(f'invalid chunked policy {spec!r} (outer stage: no .nf, .pr-, .elt, or order other than tree)')
        if outer.kind == 'int' and inner.kind == 'float':
            raise ValueError('a float inner stage cannot feed an integer outer register')
        p = Policy('chunk', chunk=chunk, inner=inner, outer=outer)
        if p.name() != spec:
            raise ValueError(f'non-canonical policy name {spec!r}')
        return p
    if '_' in spec:
        raise ValueError(f'unknown accumulator policy {spec!r}')
    return _single(spec)
