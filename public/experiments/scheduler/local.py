"""Local execution with validation and renewable worker leases."""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any, Mapping

from public.experiments.registry.database import ExperimentRegistry, RegistryError
from public.experiments.registry.identity import experiment_sha256


class LocalScheduler:
    def __init__(self, registry: ExperimentRegistry, *, worker_id: str, heartbeat_seconds: float = 30):
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        self.registry = registry
        self.worker_id = worker_id
        self.heartbeat_seconds = heartbeat_seconds

    def run_once(self, executor: Callable[[Mapping[str, Any]], Mapping[str, tuple[float, str]]]) -> bool:
        claimed = self.registry.claim_next(self.worker_id)
        if claimed is None:
            return False
        run_id, token = int(claimed["run_id"]), claimed["lease_token"]
        configuration = json.loads(bytes(claimed["canonical_json"]).decode("utf-8"))
        try:
            configuration = self.registry.validator(configuration)
            if experiment_sha256(configuration) != claimed["experiment_id"]:
                raise ValueError("stored configuration identity mismatch")
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.registry.finish(run_id, "INVALID", lease_token=token, error=str(exc))
            return True
        stopped = threading.Event()

        def renew():
            with ExperimentRegistry(self.registry.path, validator=self.registry.validator) as connection:
                while not stopped.wait(self.heartbeat_seconds):
                    try:
                        connection.heartbeat(run_id, self.worker_id, lease_token=token)
                    except RegistryError:
                        return

        heartbeat = threading.Thread(target=renew, daemon=True)
        heartbeat.start()
        try:
            metrics = executor(configuration)
            self.registry.finish(run_id, "COMPLETED", lease_token=token, metrics=metrics)
        except RegistryError:
            # A recovered attempt owns a different lease. Its results cannot be overwritten.
            row = self.registry.connection.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row["status"] == "RUNNING":
                self.registry.finish(run_id, "FAILED", lease_token=token, error="invalid execution metrics")
        except Exception as exc:
            try:
                self.registry.finish(run_id, "FAILED", lease_token=token, error=f"{type(exc).__name__}: {exc}")
            except RegistryError:
                pass
        finally:
            stopped.set()
            heartbeat.join()
        return True
