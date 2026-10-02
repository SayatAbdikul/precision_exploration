"""One dot product under a contract 2.3 policy, in exact rationals.

Terms are exact rationals in product-grid units (2^-(shift_a + shift_w) of code level): for a scalar format the
integer products a*w of the code-grid integers, for a shared-exponent format the block sums or element products
(blocks.py). The value returned is the binary64 code-level dot that a kernel must return (one RNE conversion of the
final register or accumulator value, as in contract 2.1), or +-inf / nan for a non-finite float result.

Operations of a float stage (every one is a "rounded add" R(x + y) of two operands, counted as below):
  sequential step     x = state, y = term (after product rounding); a zero term is skipped (no add, no event)
                      while the state is finite; once the state is non-finite every later step, zero terms
                      included, counts one non-finite step and adds no rounding (the 2.1 rule)
  tree adder          x = left, y = right, at every internal node, zero operands included: an adder rounds
                      its output R(x + 0) = R(x) like any other, and a leaf is an exact (unrounded) term, so
                      an adder with a zero operand is inexact whenever the other leaf is not representable
                      (an RTL adder tree has no zero skip); the sequential zero skip is value-neutral only
                      because a sequential state is always representable
  lone root           a reduction of exactly one term under .ord-tree (K = 1, or a one-term chunk): R(+0 + leaf)
  bias                B = the bias converted once into the format (no event); .bias-pre: the initial state is B
                      (no add); .bias-post: R(state + B) after the last term (a rounded add)
The bias of an integer stage is RNE(b / 2^L) on its grid; of a `wide` stage RNE(b) on the term grid when the terms
are integers of a fixed grid (scalar nodes, wide_grid=True), and the exact rational b when they are not (shared-
exponent nodes, wide_grid=False: an exact accumulator there has no fixed grid, blocks.py).
Event fields of a float stage (per stage and output):
  nonfinite_steps  rounded adds and steps whose result is +-inf or nan (including those that only carry a
                   non-finite operand through)
  overflow_steps   rounded adds of two finite operands whose rounding with unbounded exponent exceeds the largest
                   finite value (the result is +-inf, or the largest finite value under .ofsat or .rz)
  inexact_steps    rounded adds with a finite result different from the exact sum
  absorbed_steps   rounded adds of two finite operands whose result equals one operand while the other is nonzero
                   (swamping; a term flushed to +0 against a zero state counts too)
Product roundings (.nf, .pr-) are not adds and count in no field.
Event fields of an integer stage: high / low = register fits (each sequential add, each tree leaf entry and tree
adder, the preloaded bias, the bias-post add) whose exact integer lay above / below the range; inexact_terms =
terms whose entry onto the grid lost bits (the bias is not a term).
"""
from __future__ import annotations
from fractions import Fraction as Q
import math
from .rounding import FloatFormat, INF, pow2, round_int
from .registers import Register
from .orders import permutation, tree
from .policies import resolve
from .certify import register_widths

BINARY64 = FloatFormat.ieee(11, 52)
NAN = math.nan
FLOAT_EVENTS = ('nonfinite_steps', 'overflow_steps', 'inexact_steps', 'absorbed_steps')


def _finite(x):
    return not isinstance(x, float)


def _add(x, y):
    """IEEE addition of exact values and infinities: inf + -inf = nan, nan is sticky."""
    if _finite(x) and _finite(y):
        return x + y
    if (isinstance(x, float) and math.isnan(x)) or (isinstance(y, float) and math.isnan(y)):
        return NAN
    if not _finite(x) and not _finite(y):
        return x if x == y else NAN
    return x if not _finite(x) else y


def product_format(product, shift):
    """The `.pr-` format in product-grid units: a code-level product p becomes R'(p * 2^x) * 2^-x (independent of `.x<e>`)."""
    fmt = FloatFormat.ieee(product.E, product.M, subnormals=not product.ftz, overflow='saturate',
                           mode='rz' if product.rz else 'rne')
    return fmt.scaled(product.x - shift)


class Stage:
    """One accumulator over one sequence of terms (product-grid units)."""

    def __init__(self, policy, shift, width=None, wide_grid=True):
        self.policy, self.kind, self.wide_grid = policy, policy.kind, wide_grid
        self.events = {}
        if self.kind == 'int':
            self.register = Register(width, policy.low_bits, policy.low_mode, policy.overflow)
        elif self.kind in ('float', 'control'):
            E, M = (8, 23) if self.kind == 'control' else (policy.E, policy.M)
            fmt = FloatFormat.ieee(E, M, subnormals=not policy.ftz, overflow='saturate' if policy.ofsat else 'inf',
                                   mode='rz' if policy.rz else 'rne')
            self.format = fmt.scaled(policy.scale_exponent - shift)        # the format in product-grid units
            self.events = dict.fromkeys(FLOAT_EVENTS, 0)
        elif self.kind != 'wide':
            raise ValueError('a stage is a single policy')

    # -- float: the one rounded add ---------------------------------------------------------------------------
    def _round_add(self, x, y):
        exact = _add(x, y)
        result, overflowed = self.format.round_overflow(exact)
        ev = self.events
        ev['overflow_steps'] += overflowed
        if not _finite(result):
            ev['nonfinite_steps'] += 1
        elif result != exact:
            ev['inexact_steps'] += 1
        if _finite(x) and _finite(y) and _finite(result) and ((y != 0 and result == x) or (x != 0 and result == y)):
            ev['absorbed_steps'] += 1
        return result

    # -- the operations -----------------------------------------------------------------------------------
    def bias(self, b):
        """The bias as the register receives it: the integer RNE(b / 2^L) of the grid (integer stages), the integer
        RNE(b) (wide on a fixed term grid) or b exactly (wide without one: shared-exponent nodes), or one
        conversion into the float format (its rounding mode and flush rule; a conversion, like a product rounding,
        counts in no event field)."""
        if self.kind == 'int':
            return round_int(Q(b) / pow2(self.register.low_bits), 'rne')
        if self.kind == 'wide':
            return Q(round_int(Q(b), 'rne')) if self.wide_grid else Q(b)
        return self.format.round(Q(b))

    def start(self, init=None):
        if init is None:
            return Q(0) if self.kind != 'int' else 0
        if self.kind == 'int':
            return self.register.fit(self.bias(init))                    # a fit: counts high/low
        return self.bias(init)

    def add_bias(self, state, b):
        if self.kind == 'int':
            return self.register.fit(state + self.bias(b))
        if self.kind == 'wide':
            return _add(state, self.bias(b))
        return self._round_add(state, self.bias(b))

    def step(self, state, term):
        if self.kind == 'int':
            return self.register.add(state, term)
        if self.kind == 'wide':
            return _add(state, term)
        if not _finite(state):
            self.events['nonfinite_steps'] += 1
            return _add(state, term)
        if term == 0:
            return state
        if self.policy.nonfused:
            term = self.format.round(term)
        return self._round_add(state, term)

    def leaf(self, term):
        if self.kind == 'int':
            return self.register.fit(self.register.enter(term))
        if self.kind in ('float', 'control') and self.policy.nonfused:
            return self.format.round(term)
        return term

    def combine(self, x, y):
        if self.kind == 'int':
            return self.register.fit(x + y)
        if self.kind == 'wide':
            return _add(x, y)
        return self._round_add(x, y)

    def finish(self, state):
        """Value of the final state in product-grid units (Fraction) or a non-finite float."""
        if self.kind == 'int':
            r = self.register
            self.events = {'high': r.high, 'low': r.low, 'inexact_terms': r.inexact}
            return r.value(state)
        return state


def reduce_terms(terms, policy, shift, width=None, bias_units=None, wide_grid=True):
    """(value in product-grid units, events) of one stage over terms already in summation order."""
    stage = Stage(policy, shift, width, wide_grid)
    if policy.bias and bias_units is None:
        raise ValueError('a bias policy needs the bias in product-grid units')
    if policy.order == 'tree':
        leaves = [stage.leaf(t) for t in terms]
        if not leaves:
            state = stage.start()
        elif len(leaves) == 1 and stage.kind in ('float', 'control'):
            state = stage._round_add(Q(0), leaves[0])                    # the lone root is rounded once
        else:
            state = tree(leaves, stage.combine)
        if policy.bias == 'post':
            state = stage.add_bias(state, bias_units)
    else:
        state = stage.start(bias_units if policy.bias == 'pre' else None)
        for t in terms:
            state = stage.step(state, t)
        if policy.bias == 'post':
            state = stage.add_bias(state, bias_units)
    value = stage.finish(state)
    return value, dict(stage.events)


def ordered(terms, policy, taps, shift):
    """Terms in the policy's summation order, each first rounded to the product format when the policy has one."""
    if policy.product:
        fmt = product_format(policy.product, shift)
        terms = [fmt.round(t) for t in terms]
    if policy.order == 'hwc' and taps is None:
        raise ValueError('.ord-hwc needs the convolution tap geometry (cg, kh, kw)')
    return [terms[k] for k in permutation(policy.order, taps if policy.order == 'hwc' else len(terms))]


def dot_terms(terms, spec, shift, certificate=None, taps=None, bias_units=None, wide_grid=True):
    """(value in product-grid units, events) for terms given in the engine's chw order.

    certificate: a certify.NodeCertificate (or, for a plain relative register only, the 2.1 certificate dict);
    needed only for abs-<d>/struct-<d> names. wide_grid: False on shared-exponent nodes (a `wide` bias is exact).
    """
    policy = resolve(spec)
    if taps is None and certificate is not None and not isinstance(certificate, dict):
        taps = certificate.taps
    widths = register_widths(policy, certificate)
    if policy.kind != 'chunk':
        return reduce_terms(ordered(terms, policy, taps, shift), policy, shift, widths[0], bias_units, wide_grid)
    inner, outer = policy.inner, policy.outer
    sequence = ordered(terms, inner, taps, shift)
    chunks, events = [], {}
    for i in range(0, len(sequence), policy.chunk):
        value, ev = reduce_terms(sequence[i:i + policy.chunk], inner, shift, widths[0], None, wide_grid)
        chunks.append(value)
        for key, count in ev.items():
            events['inner_' + key] = events.get('inner_' + key, 0) + count
    value, ev = reduce_terms(chunks, outer, shift, widths[1], bias_units, wide_grid)
    events.update({'outer_' + k: v for k, v in ev.items()})
    events['chunks'] = len(chunks)
    return value, events


def to_binary64(value, shift):
    """Code-level binary64 of a product-grid value: one RNE (exact below 2^53 units), infinities and nan kept."""
    if not _finite(value):
        return value
    return float(BINARY64.round(Q(value) / pow2(shift)))


def dot(pairs, spec, shift_a, shift_w, certificate=None, taps=None, bias_units=None):
    """(binary64 code-level dot, events) for (a, w) code-grid integer pairs in chw order (a scalar node)."""
    if resolve(spec).elementwise:
        raise ValueError('.elt is defined on shared-exponent nodes only')
    shift = shift_a + shift_w
    value, events = dot_terms([Q(a) * b for a, b in pairs], spec, shift, certificate, taps, bias_units)
    return to_binary64(value, shift), events


def legacy_word(policy, events):
    """The contract 2.1 event word of a saturating (sat, L = 0) or float policy."""
    policy = resolve(policy)
    if policy.kind == 'int':
        return min(events['high'], 65535) | (min(events['low'], 65535) << 16)
    if policy.kind == 'float':
        return events['nonfinite_steps']
    return None


def bias_units(bias, scale_in, scale_w, shift):
    """Node bias in product-grid units, exact rational: bias / (s_in * s_w[c]) * 2^shift (binary64 inputs)."""
    return Q(bias) / (Q(scale_in) * Q(scale_w)) * pow2(shift)
