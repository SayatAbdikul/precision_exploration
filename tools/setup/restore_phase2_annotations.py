"""Restore the official COCO annotations, verifying the frozen Phase 1 hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
# Official bucket endpoint preserves TLS verification; the dotted public
# image hostname currently fails certificate hostname validation.
SOURCE = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"


def main():
    manifest = json.loads((ROOT / "public/workloads/datasets/coco2017.json").read_text())
    output = ROOT / "data/raw/coco2017/annotations"
    output.mkdir(parents=True, exist_ok=True)
    missing = []
    for name, digest in manifest["annotation_sha256"].items():
        path = output / f"{name}.json"
        if path.is_file():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"refusing to overwrite mismatched annotation: {path}")
        else:
            missing.append(name)
    if missing:
        archive = ROOT / "cache/downloads/annotations_trainval2017.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.is_file():
            temporary = archive.with_suffix(".partial")
            print("Downloading official COCO annotation archive", flush=True)
            with urllib.request.urlopen(SOURCE, timeout=60) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as payload:
            for name in missing:
                content = payload.read(f"annotations/{name}.json")
                digest = hashlib.sha256(content).hexdigest()
                if digest != manifest["annotation_sha256"][name]:
                    raise ValueError(f"official annotation differs from frozen identity: {name}")
                temporary = output / f"{name}.partial"
                temporary.write_bytes(content)
                temporary.replace(output / f"{name}.json")
                print(f"{name}: frozen SHA256 verified", flush=True)


if __name__ == "__main__":
    main()
