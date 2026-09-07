"""Atomic content-addressed artifact publication."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class ArtifactRecord:
    sha256: str
    path: str
    size_bytes: int
    semantic_type: str


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def digest_bytes(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def _destination(self, digest: str) -> Path:
        return self.root / digest[:2] / digest[2:]

    def put_bytes(self, payload: bytes, *, semantic_type: str) -> ArtifactRecord:
        digest = self.digest_bytes(payload)
        destination = self._destination(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            descriptor, temporary_name = tempfile.mkstemp(prefix="artifact-", dir=destination.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                Path(temporary_name).replace(destination)
            finally:
                temporary = Path(temporary_name)
                if temporary.exists():
                    temporary.unlink()
        self.verify(digest)
        return ArtifactRecord(digest, str(destination), len(payload), semantic_type)

    def put_stream(self, source: BinaryIO, *, semantic_type: str) -> ArtifactRecord:
        return self.put_bytes(source.read(), semantic_type=semantic_type)

    def get(self, digest: str) -> bytes:
        self.verify(digest)
        return self._destination(digest).read_bytes()

    def verify(self, digest: str) -> None:
        path = self._destination(digest)
        if not path.is_file():
            raise FileNotFoundError(f"artifact is missing: {digest}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"artifact hash mismatch: expected {digest}, got {actual}")
