"""Minimal local scheduler built on the authoritative SQLite registry."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Mapping

from public.experiments.registry.database import ExperimentRegistry


class LocalScheduler:
    def __init__(self, registry: ExperimentRegistry, *, worker_id: str):
        self.registry = registry
        self.worker_id = worker_id

    def run_once(self, executor: Callable[[Mapping[str, Any]], Mapping[str, tuple[float, str]]]) -> bool:
        claimed = self.registry.claim_next(self.worker_id)
        if claimed is None:
            return False
        run_id = int(claimed["run_id"])
        configuration = json.loads(bytes(claimed["canonical_json"]).decode("utf-8"))
        try:
            metrics = executor(configuration)
            self.registry.add_metrics(run_id, metrics)
            self.registry.finish(run_id, "COMPLETED")
        except Exception as exc:
            self.registry.finish(run_id, "FAILED", error=f"{type(exc).__name__}: {exc}")
        return True
