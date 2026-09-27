"""Execution-semantics checks for B; these never issue exact-A acceptance."""
from __future__ import annotations

import numpy as np
import torch

from .common import formats, digest, seal, unseal
from .quantizer import Quantizer, numpy_qdq, table


def validate_quantizers(device):
    checked = []
    rng = np.random.default_rng(20260925)
    for entry in formats():
        if entry["family"] in {"mx_float", "bfp"}:
            continue
        name = entry["name"]
        levels, bounds, _ = table(name)
        probe = np.concatenate([levels, bounds, np.nextafter(bounds, np.float32(-np.inf)),
                                np.nextafter(bounds, np.float32(np.inf)),
                                rng.normal(0, float(levels[-1]), 2048).astype(np.float32)])
        q = Quantizer(name, device)
        for scale in (1.0, .037, 3.25):
            values = torch.tensor(probe, device=device)
            actual = q(values, torch.tensor(scale, dtype=torch.float32, device=device)).cpu().numpy()
            expected = numpy_qdq(probe, name, scale)
            np.testing.assert_array_equal(actual, expected)
        # Every finite level must round-trip in this surrogate's FP32 table.
        np.testing.assert_array_equal(numpy_qdq(levels, name, 1), levels)
        checked.append({"format": name, "probe_count": int(len(probe) * 3)})
    return {"status": "passed", "device": device, "formats": checked,
            "claim": "torch_matches_declared_numpy_FP32_QDQ_policy; not_Model_C_equivalence"}


def validate_fold(graph, original, inputs):
    with torch.inference_mode():
        expected = original(inputs)
        actual = graph(inputs)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    return {"status": "passed", "max_absolute_error": float((actual-expected).abs().max()),
            "rtol": 2e-4, "atol": 2e-4, "images": len(inputs)}


def verify_prediction(path, identity, row):
    record = unseal(path)
    if record["configuration_sha256"] != identity or record["sample"] != row:
        raise ValueError("prediction identity/sample mismatch")
    if (len(record["top5"]) != 5 or len(set(record["top5"])) != 5 or
            any(type(x) is not int or not 0 <= x < 1000 for x in record["top5"])):
        raise ValueError("invalid class predictions")
    return record
