"""B2 identity, protocol constants and evidence locations.

Everything that v1 already provides (dataset identity, sealing, digests) is
imported from ``tools.experiment_b.common``; nothing there is modified.
"""
from __future__ import annotations

from tools.experiment_b import common as v1
from tools.experiment_b.common import ROOT, digest, file_hash

BASE = ROOT / "artifacts/experiment_b2"
MODELS = ("resnet18", "mobilenet_v2", "mobilenet_v3_large")
PANELS = (128, 512, 1000)
BIAS_CORRECTION_IMAGES = 256
# Files whose content decides the numbers a B2 configuration produces.  The
# runner, report, export and vendor modules are identified separately so that
# adding a report column does not relabel every measured configuration.  The
# recipe switches themselves are part of every configuration identity.
NUMERIC_SOURCES = ("__init__.py", "common.py", "codebook.py", "boundaries.py", "scales.py",
                   "engine.py", "data.py")
PROTOCOL = {
    "version": "experiment-b2-baseline-repair-1.0.0",
    "semantics": "b2_scalar_fp32_qdq_wideacc_v1",
    "inherits": v1.PROTOCOL["version"],
    "panel_order": "ascending_frozen_image_sha256",
    "panels": {"dev128": "first 128 of imagenet_screen_1k", "dev512": "first 512 of imagenet_screen_1k",
               "screen1k": "all 1000 of imagenet_screen_1k"},
    "calibration_observations": "v1 cache read-only: 2000 frozen calibration images, "
                                "rotating_channel_stratified_256_per_image_v1 samples and exact per-node maxima",
    "bias_correction_images": BIAS_CORRECTION_IMAGES,
    "bias_correction_list": "first 256 of imagenet_calibration_2k in ascending sha256 order",
    "inference_batch_size": v1.PROTOCOL["inference_batch_size"],
    "rounding": v1.PROTOCOL["rounding"],
    "overflow": v1.PROTOCOL["overflow"],
    "bias_and_reductions": v1.PROTOCOL["bias_and_reductions"],
    "interpretation": "development_evidence; recipe selected on dev512; screen1k contains dev512",
}


def numeric_sources():
    folder = ROOT / "tools/experiment_b2"
    return {f"tools/experiment_b2/{name}": file_hash(folder / name) for name in NUMERIC_SOURCES}


def source_identity():
    return digest({"prior_source": v1.source_identity(), "b2": numeric_sources()})
