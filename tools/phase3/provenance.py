"""Preserve analysis implementation bytes as tools evolve during the campaign."""
from pathlib import Path
import hashlib
import os
import tempfile

from tools.phase3.common import ROOT, file_hash, reference


def source_reference(path, root=ROOT):
    path = Path(path).resolve()
    identity = file_hash(path)
    target = root / "artifacts/phase3/analysis-sources" / identity / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != identity:
        raise ValueError("analysis source changed while archiving")
    if target.exists() and target.read_bytes() != payload:
        raise ValueError("analysis source archive is immutable")
    if not target.exists():
        descriptor, temporary = tempfile.mkstemp(prefix=target.name+".", suffix=".partial", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    result = reference(target, root)
    if result["sha256"] != identity:
        raise ValueError("analysis source changed while archiving")
    return result
