"""Additional execution provenance without changing historical numerical job IDs."""
import importlib.metadata
import platform
import sys
from pathlib import Path

from public.inference.conformance_job import source_identity
from tools.phase3.common import ROOT, campaign, digest, file_hash, read
from tools.phase3.screening import pipeline_identity

GUARD_PATH = "public/experiments/configs/experiment_a/phase3-execution-guard-v1.json"
PACKAGES = ("torch", "torchvision", "numpy", "Pillow", "ultralytics", "opencv-python", "pycocotools")


def observed_guard(root=ROOT):
    sources = sorted((root / "public/workloads").rglob("*.py"))
    return {"schema_version": "phase3-execution-guard-1.0.0",
            "source_sha256": source_identity(), "pipeline_sha256": pipeline_identity(root),
            "campaign_sha256": digest(campaign(root)),
            "preprocessing_sources": {str(p.relative_to(root)): file_hash(p) for p in sources},
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
            "python_major_minor": list(sys.version_info[:2]),
            "libraries": {b: file_hash(root / f"build/phase2/{b}/libprecision_{b}.so") for b in ("cpp", "cuda")}}


def verify_guard(root=ROOT):
    expected = read(root / GUARD_PATH)
    actual = observed_guard(root)
    changes = [key for key in actual if actual[key] != expected.get(key)]
    if changes:
        raise ValueError("execution dependencies changed; explicit revalidation required: " + ", ".join(changes))
    return digest(expected)


def controller_identity(root=ROOT):
    paths = list((root / "tools/phase3").glob("*.py"))
    paths += [root / "tools/run/phase3_resume.py", root / GUARD_PATH]
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(paths)})


def machine_identity():
    cpu = Path("/proc/cpuinfo").read_text()
    model = next((line.split(":", 1)[1].strip() for line in cpu.splitlines() if line.startswith("model name")), "unknown")
    return {"host": platform.node(), "cpu": model, "platform": platform.platform()}
