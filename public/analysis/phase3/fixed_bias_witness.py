"""Construct finite activation witnesses for fixed-bias output discrepancies.

The search is exhaustive only over vectors with at most two nonzero entries.
Failure to find one is not an all-input precision proof. Returned vectors use
distinct actual reduction positions and are independently checked by callers.
"""
from fractions import Fraction

from public.analysis.phase3.fixed_bias_sensitivity import output_boundaries
from public.analysis.phase3.fixed_mac_bounds import quantum
from public.inference.reference.arithmetic import encode, format_named, round_integer


def discrepant_dot_targets(dot_quantum, maximum_dot, bias, step, output_name):
    """Enumerate every discrepant threshold-adjacent point on the dot lattice."""
    q, maximum, bias, step = map(Fraction, (dot_quantum, maximum_dot, bias, step))
    if q <= 0 or maximum < 0 or step <= 0 or (q / step).denominator != 1 or (maximum / q).denominator != 1:
        raise ValueError("requires a positive exact fixed-grid dot lattice and aligned bound")
    fmt = format_named(output_name)
    stored = round_integer(bias / step) * step
    if stored == bias:
        return []
    limit, targets = int(maximum / q), set()
    for threshold, *_ in output_boundaries(output_name):
        n = (threshold - bias) // q
        for index in (n, n + 1):
            if -limit <= index <= limit:
                dot = index * q
                if encode(fmt, dot + bias) != encode(fmt, dot + stored):
                    targets.add(dot)
    return sorted(targets, key=lambda value: (abs(value), value))


class SparseTwoSum:
    """Index the exact contributions of actual weight positions and codes."""

    def __init__(self, weights, activation_values):
        self.weights = tuple(map(Fraction, weights))
        self.values = {code: Fraction(value) for code, value in activation_values.items()}
        zeros = [code for code, value in self.values.items() if value == 0]
        if not self.weights or not zeros:
            raise ValueError("sparse witnesses need weights and an exact activation zero")
        self.zero_code = min(zeros)
        xq, wq = quantum(self.values.values()), quantum(self.weights)
        self.quantum = xq * wq
        self.contributions = {}
        if not self.quantum:
            return
        # Two distinct positions per weight value are enough: a queried pair
        # can forbid only one position. Keep the same property after collisions
        # between different weights/code products.
        positions = {}
        for index, weight in enumerate(self.weights):
            bucket = positions.setdefault(weight, [])
            if len(bucket) < 2:
                bucket.append(index)
        for weight, indices in positions.items():
            wi = int(weight / wq)
            for code, value in self.values.items():
                contribution = wi * int(value / xq)
                bucket = self.contributions.setdefault(contribution, [])
                for index in indices:
                    if len(bucket) < 2 and all(index != old[0] for old in bucket):
                        bucket.append((index, code))

    def find(self, target):
        target = Fraction(target)
        if target == 0:
            return [self.zero_code] * len(self.weights)
        if not self.quantum or (target / self.quantum).denominator != 1:
            return None
        goal = int(target / self.quantum)
        if goal in self.contributions:
            index, code = self.contributions[goal][0]
            result = [self.zero_code] * len(self.weights)
            result[index] = code
            return result
        for contribution, left in self.contributions.items():
            right = self.contributions.get(goal - contribution)
            if right is None:
                continue
            for i, a in left:
                for j, b in right:
                    if i != j:
                        result = [self.zero_code] * len(self.weights)
                        result[i], result[j] = a, b
                        return result
        return None
