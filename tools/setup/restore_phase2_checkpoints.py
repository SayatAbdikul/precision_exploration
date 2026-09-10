"""Restore pinned public checkpoints without changing frozen manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def main():
    index = json.loads((ROOT / "public/workloads/models/manifests/index.json").read_text())
    for row in index["manifests"]:
        manifest = json.loads((ROOT / row["path"]).read_text())
        path = ROOT / manifest["checkpoint_path"]
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("checkpoint path escapes repository")
        if path.is_file():
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["checkpoint_sha256"]:
                raise ValueError(f"existing checkpoint hash mismatch: {path}")
            print(f"{row['name']}: verified existing checkpoint", flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        digest = hashlib.sha256()
        request = urllib.request.Request(manifest["checkpoint_source"], headers={"User-Agent": "precision-exploration/0.1"})
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                output.write(block)
                digest.update(block)
        if digest.hexdigest() != manifest["checkpoint_sha256"]:
            raise ValueError(f"download hash mismatch: {row['name']}; partial file retained for diagnosis")
        temporary.replace(path)
        print(f"{row['name']}: restored and verified", flush=True)


if __name__ == "__main__":
    main()
