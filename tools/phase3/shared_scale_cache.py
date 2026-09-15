"""Durable memoization of the unchanged exhaustive shared-scale oracle.

This cache is installed only in a graph-preparation process. It stores complete
oracle responses per finite input block; it does not approximate or shorten the
255-scale search. The numerical engine files and runtime are unchanged.
"""
from contextlib import contextmanager
from fractions import Fraction
import json
from pathlib import Path
import sqlite3

from public.formats.oracle.manifest import manifest_sha256
from public.inference.reference.arithmetic import format_named, real
from tools.phase3.common import digest


class SharedScaleCache:
    def __init__(self, path, *, source_sha256, oracle, progress=None, fast_fp32=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=60, isolation_level=None)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS scales (identity TEXT PRIMARY KEY, request TEXT NOT NULL, response TEXT NOT NULL, response_sha256 TEXT NOT NULL)")
        self.source_sha256, self.oracle, self.progress = source_sha256, oracle, progress
        self.fast_fp32 = fast_fp32
        self.hits, self.misses = 0, 0

    def close(self):
        self.connection.close()

    def stats(self):
        return {"hits": self.hits, "misses": self.misses, "source_sha256": self.source_sha256,
                "exact_fp32_acceleration": self.fast_fp32,
                "policy": "unchanged_exhaustive_oracle_responses_per_block"}

    def __call__(self, values, format_name, *, coarse_candidates=100, fine_candidates=50):
        values = tuple(values)
        fmt = format_named(format_name)
        if fmt.manifest["scaling"]["mode"] != "intrinsic_shared":
            return self.oracle(values, format_name, coarse_candidates=coarse_candidates, fine_candidates=fine_candidates)
        normalized = tuple(real(v) for v in values)
        if (not normalized or len(normalized) > fmt.manifest["block"]["block_size"]
                or any(not isinstance(v, Fraction) and not v.is_finite() for v in normalized)
                or any(type(n) is not int or n < 2 for n in (coarse_candidates, fine_candidates))):
            # Preserve the oracle's own validation/error behavior.
            return self.oracle(values, format_name, coarse_candidates=coarse_candidates, fine_candidates=fine_candidates)
        request = {"schema_version": "phase3-shared-scale-cache-1.0.0", "source_sha256": self.source_sha256,
                   "format": format_name, "manifest_sha256": manifest_sha256(fmt.manifest),
                   "values": [str(Fraction(v)) for v in normalized],
                   "coarse_candidates": coarse_candidates, "fine_candidates": fine_candidates}
        identity = digest(request)
        row = self.connection.execute("SELECT request,response,response_sha256 FROM scales WHERE identity=?", (identity,)).fetchone()
        if row is not None:
            response = self._read(row, request)
            self.hits += 1
        else:
            if self.fast_fp32:
                from tools.phase3.shared_scale_fast import mse_scale_fp32
                response = mse_scale_fp32(values, format_name, coarse_candidates=coarse_candidates,
                                          fine_candidates=fine_candidates, fallback=self.oracle)
            else:
                response = self.oracle(values, format_name, coarse_candidates=coarse_candidates, fine_candidates=fine_candidates)
            encoded_request, encoded_response = json.dumps(request, sort_keys=True), json.dumps(response, sort_keys=True)
            self.connection.execute("INSERT OR IGNORE INTO scales VALUES(?,?,?,?)",
                                    (identity, encoded_request, encoded_response, digest(response)))
            # Concurrent preparers may have computed the same block. Verify
            # the winning committed response instead of overwriting it.
            row = self.connection.execute("SELECT request,response,response_sha256 FROM scales WHERE identity=?", (identity,)).fetchone()
            if self._read(row, request) != response:
                raise ValueError("conflicting shared-scale oracle cache response")
            self.misses += 1
        if self.progress and (self.hits+self.misses) % 256 == 0:
            self.progress(self.stats())
        return response

    @staticmethod
    def _read(row, request):
        stored_request, response = json.loads(row[0]), json.loads(row[1])
        if stored_request != request or digest(response) != row[2]:
            raise ValueError("shared-scale cache identity/content mismatch")
        if (response.get("format") != request["format"] or response.get("candidate_count") != 255
                or response.get("tie_break") != "smallest_scale"):
            raise ValueError("shared-scale cache response does not describe the exhaustive oracle")
        return response

    @contextmanager
    def installed(self):
        from public.quantization.calibration import mse
        original = mse.mse_scale
        if original is not self.oracle:
            raise ValueError("shared-scale oracle was already replaced in this process")
        mse.mse_scale = self
        try:
            yield self
        finally:
            mse.mse_scale = original
