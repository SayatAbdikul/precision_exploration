"""Sampled paired layer diagnostics; observed stores do not reveal overflow events.

Range violations against the FP32 reference are labelled separately from actual
quantizer events. Never infer overflow counts solely from saturated output codes.
"""
from __future__ import annotations

import numpy as np
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache

from public.inference.reference.arithmetic import format_named, real


def sample_indices(count, limit=4096):
    if type(count) is not int or count < 1 or type(limit) is not int or limit < 1:
        raise ValueError("positive diagnostic population/sample sizes required")
    return np.linspace(0, count-1, min(count, limit), dtype=np.int64)


def distribution(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("diagnostics require finite, nonempty samples")
    mean, variance = float(values.mean()), float(values.var())
    return {"count": len(values), "min": float(values.min()), "max": float(values.max()),
            "mean": mean, "variance": variance,
            "kurtosis": None if variance == 0 else float(np.mean((values-mean)**4)/variance**2),
            "percentiles": {str(p): float(np.percentile(values, p)) for p in (1, 5, 50, 95, 99)},
            "zero_fraction": float(np.mean(values == 0)),
            "outlier_fraction": float(np.mean(np.abs(values-mean) > 6*np.sqrt(variance)))}


def paired_layer(fp32, candidate, *, population_elements, sampled_indices):
    a, b = np.asarray(fp32, dtype=np.float64).ravel(), np.asarray(candidate, dtype=np.float64).ravel()
    if a.shape != b.shape or len(sampled_indices) != len(a):
        raise ValueError("paired diagnostic shape/index mismatch")
    left, right = distribution(a), distribution(b)
    mse, signal = float(np.mean((a-b)**2)), float(np.mean(a*a))
    sqnr = None if mse == 0 or signal == 0 else float(10*np.log10(signal/mse))
    return {"population_elements": population_elements, "sampled_elements": len(a),
            "sampling": "evenly_spaced_flat_v1", "fp32": left, "candidate": right, "mse": mse,
            "sqnr_db": sqnr, "sqnr_status": "exact" if mse == 0 else "zero_reference_signal" if signal == 0 else "finite",
            "nonzero_reference_stored_zero_fraction": float(np.mean((a != 0) & (b == 0))),
            "sign_reversal_fraction": float(np.mean(a*b < 0)),
            "quantizer_event_counts": None,
            "event_status": "requires_pre_store_instrumentation"}


class ReferenceSamples:
    def __init__(self, limit=4096):
        self.limit, self.records = limit, {}

    def record(self, name, tensor):
        values = tensor.detach().cpu().contiguous().numpy().ravel()
        selected = sample_indices(len(values), self.limit)
        self.records[name] = {"shape": tuple(tensor.shape), "indices": selected, "values": values[selected].copy(),
                              "population": len(values)}


class LayerDiagnostics:
    def __init__(self, references):
        self.references, self.records = references, {}
        self.events = {}
        self.current = None

    def begin_node(self, node):
        self.current = node["name"]
        self.events[self.current] = {"population_stores": 0, "sampled_stores": 0,
                                     "overflow": 0, "underflow_to_zero": 0, "sign_clipping": 0, "nonfinite_inputs": 0}

    def quantization(self, values, tensor):
        if self.current is None:
            return
        row = self.events[self.current]
        row["population_stores"] += len(tensor.codes)
        lo, hi, minimum_positive = finite_range(tensor.encoding.format)
        for offset in sample_indices(len(tensor.codes), self.references.limit):
            index = tuple(int(v) for v in np.unravel_index(int(offset), tensor.shape))
            scale = tensor.encoding.scale_at(tensor.shape, index)
            raw = values[int(offset)]
            value = real(raw.item() if isinstance(raw, np.generic) else raw)
            row["sampled_stores"] += 1
            if isinstance(value, Decimal) and not value.is_finite():
                row["nonfinite_inputs"] += 1
                continue
            value = Fraction(value)
            stored = real(tensor.value(index))
            row["overflow"] += int(value < lo*scale or value > hi*scale)
            row["sign_clipping"] += int(value < 0 and lo >= 0 or value > 0 and hi <= 0)
            row["underflow_to_zero"] += int(value != 0 and stored == 0 and minimum_positive is not None
                                             and abs(value) < minimum_positive*scale)

    def __call__(self, node, tensor):
        expected = self.references.records[node["name"]]
        if tuple(tensor.shape) != expected["shape"]:
            raise ValueError(f"FP32/candidate layer shape mismatch: {node['name']}")
        values = [float(tensor.value(tuple(int(i) for i in np.unravel_index(int(index), tensor.shape))))
                  for index in expected["indices"]]
        self.records[node["name"]] = paired_layer(expected["values"], values,
            population_elements=expected["population"], sampled_indices=expected["indices"])
        if self.events.get(node["name"], {}).get("sampled_stores"):
            self.records[node["name"]].update(quantizer_event_counts=self.events[node["name"]],
                event_status="sampled_actual_quantizer_stores_including_intermediate_alignment",
                event_sampling="up_to_4096_evenly_spaced_values_per_quantizer_call")
        else:
            self.records[node["name"]]["event_status"] = "no_instrumented_quantizer_calls"


@lru_cache(maxsize=64)
def finite_range(name):
    fmt = format_named(name)
    values = [real(fmt.decode(code)) for code in range(1 << fmt.bits)]
    values = [Fraction(v) for v in values if not isinstance(v, Decimal) or v.is_finite()]
    positives = [abs(v) for v in values if v]
    return min(values), max(values), min(positives) if positives else None
