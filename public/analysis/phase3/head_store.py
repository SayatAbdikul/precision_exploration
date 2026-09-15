"""Isolate a detector's final integer output store from upstream arithmetic."""
from fractions import Fraction

import numpy as np

from public.inference.reference.arithmetic import format_named
from public.quantization.ptq.encoding import float_tensor


def calibration_positions(shape, count=16):
    if len(shape) != 3 or shape[0] != 1 or shape[1] != 84 or shape[2] < 1:
        raise ValueError("head observation requires 1 x 84 x boxes")
    indices = np.linspace(0, int(np.prod(shape))-1, min(count, int(np.prod(shape))), dtype=np.int64)
    rows = [{"flat_index": int(i), "channel": int(i)//shape[2], "box_index": int(i)%shape[2]} for i in indices]
    return {"positions": rows, "coordinate_channels_sampled": sorted({r["channel"] for r in rows if r["channel"] < 4}),
            "coordinate_samples": sum(r["channel"] < 4 for r in rows), "score_samples": sum(r["channel"] >= 4 for r in rows)}


def store_head(array, encoding):
    fmt = format_named(encoding.format)
    if fmt.family != "integer" or encoding.axis is not None or encoding.block_size is not None:
        raise ValueError("head-store diagnosis requires scalar mapped integer encoding")
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3 or array.shape[:2] != (1, 84) or not np.isfinite(array).all():
        raise ValueError("head-store diagnosis requires finite 1 x 84 x boxes FP32 output")
    tensor = float_tensor(array, encoding)
    # Match detector_predictions' exact-value -> Python float -> FP32 conversion.
    table = np.asarray([float(Fraction(fmt.decode(c))*encoding.scales[0]) for c in range(1 << fmt.bits)], dtype=np.float32)
    codes = np.asarray(tensor.codes, dtype=np.uint8).reshape(array.shape)
    decoded = table[codes]
    maximum = ((1 << (fmt.bits-1))-1)*encoding.scales[0]
    minimum = -(1 << (fmt.bits-1))*encoding.scales[0]
    coords, scores = array[:, :4], array[:, 4:]
    return decoded, codes, {"shape": list(array.shape), "encoding": encoding.document(),
                           "maximum_representable": str(maximum), "minimum_representable": str(minimum),
                           "coordinate_elements": int(coords.size),
                           "coordinates_above_representable_max": int(np.count_nonzero(coords > float(maximum))),
                           "fp32_coordinate_max": float(coords.max()), "stored_coordinate_max": float(decoded[:, :4].max()),
                           "score_elements": int(scores.size), "positive_scores_rounded_to_zero": int(np.count_nonzero((scores > 0) & (decoded[:, 4:] == 0))),
                           "fp32_score_max": float(scores.max()), "stored_score_max": float(decoded[:, 4:].max())}


class DenseOutput:
    """Read-only array adapter for the unchanged detector postprocessor."""
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)
        self.shape = self.array.shape

    def values(self):
        return self.array.ravel()
