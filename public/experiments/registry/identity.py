"""Canonical serialization and content-aware experiment validation."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

from public.formats.oracle.manifest import manifest_sha256, validate_manifest


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_SCHEMA_PATH = REPO_ROOT / "public" / "experiments" / "configs" / "experiment.schema.json"


class IdentityError(ValueError):
    """Raised when configuration identity cannot be established safely."""


def load_configuration(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        value = yaml.safe_load(text) if source.suffix.lower() in {".yaml", ".yml"} else json.loads(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise IdentityError(f"cannot parse {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise IdentityError("configuration root must be an object")
    return value


def _check_finite(value: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise IdentityError(f"non-finite JSON number at {'/'.join(path) or '<root>'}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _check_finite(child, path + (str(key),))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _check_finite(child, path + (str(index),))


def canonical_json_bytes(value: Mapping[str, Any], *, derived_field: str | None = None) -> bytes:
    normalized = copy.deepcopy(dict(value))
    if derived_field:
        normalized.pop(derived_field, None)
    _check_finite(normalized)
    try:
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise IdentityError(f"configuration is not canonical JSON data: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def experiment_sha256(configuration: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(configuration, derived_field="experiment_id")).hexdigest()


def content_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_relative_path(raw: str) -> PurePosixPath:
    if not raw or "\\" in raw or "$" in raw or "~" in raw:
        raise IdentityError(f"canonical path is not portable: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in raw.split("/")):
        raise IdentityError(f"canonical path must be portable normalized repository-relative POSIX: {raw!r}")
    return path


def _verify_list(reference: Mapping[str, Any], repository_root: Path) -> tuple[str, ...]:
    relative = _portable_relative_path(reference["path"])
    path = repository_root.joinpath(*relative.parts)
    try:
        path.resolve().relative_to(repository_root.resolve())
    except ValueError as exc:
        raise IdentityError(f"list escapes repository root: {reference['path']}") from exc
    if not path.is_file():
        raise IdentityError(f"referenced list does not exist: {reference['path']}")
    actual = content_sha256(path)
    if actual != reference["sha256"]:
        raise IdentityError(
            f"list hash mismatch for {reference['path']}: declared {reference['sha256']}, actual {actual}"
        )
    if path.suffix.lower() == ".tsv":
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        identity_field = next((field for field in ("sha256", "image_id", "sample_id", "relative_path")
                               if rows and field in rows[0]), None)
        if identity_field is None:
            raise IdentityError(f"TSV list has no stable sample identity field: {reference['path']}")
        entries = tuple(str(row[identity_field]) for row in rows)
    else:
        entries = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if len(entries) != len(set(entries)):
        raise IdentityError(f"duplicate entries in list: {reference['path']}")
    return entries


def resolve_model(name: str, repository_root: Path) -> dict[str, Any]:
    from public.workloads.models.identity import verify_model_manifest
    index = json.loads((repository_root / "public/workloads/models/manifests/index.json").read_text())
    matches = [row for row in index["manifests"] if row["name"] == name]
    if len(matches) != 1:
        raise IdentityError(f"unresolved model: {name}")
    relative = _portable_relative_path(matches[0]["path"])
    try:
        return verify_model_manifest(repository_root / relative, repository_root=repository_root)
    except (ValueError, OSError, KeyError) as exc:
        raise IdentityError(f"model reference verification failed: {exc}") from exc


def validate_baseline(configuration: Mapping[str, Any], *, repository_root: Path) -> dict[str, Any]:
    """The explicitly supported FP32 evidence schema, separate from PTQ jobs."""
    value = copy.deepcopy(dict(configuration))
    if set(value) != {"schema_version", "model", "dataset", "preprocessing_sha256", "evaluator", "precision", "seed"}:
        raise IdentityError("invalid FP32 baseline fields")
    if value["precision"] != "fp32" or type(value["seed"]) is not int:
        raise IdentityError("invalid FP32 precision or seed")
    model = resolve_model(value["model"].get("name"), repository_root)
    model_path = repository_root / f"public/workloads/models/manifests/{model['name']}.json"
    expected_model = {"name": model["name"], "manifest_sha256": content_sha256(model_path),
                      "checkpoint_sha256": model["checkpoint_sha256"],
                      "deployment_graph_sha256": model["deployment_graph_sha256"]}
    if value["model"] != expected_model:
        raise IdentityError("baseline model identity mismatch")
    if value["preprocessing_sha256"] != model["preprocessing_sha256"] or value["evaluator"] != model["evaluator"]:
        raise IdentityError("baseline preprocessing/evaluator mismatch")
    records = json.loads((repository_root / "data/manifests/index.json").read_text())["records"]
    reference = value["dataset"]
    record = records.get(reference.get("name"))
    if record is None or reference != {"name": reference["name"], "list_sha256": record["sha256"], "count": record["count"]}:
        raise IdentityError("baseline dataset identity mismatch")
    _verify_list(record, repository_root)
    from public.workloads.datasets.identity import verify_payload_record
    verify_payload_record(repository_root, record)
    _check_finite(value)
    return value


def validate_submission(configuration: Mapping[str, Any], *, repository_root: str | Path = REPO_ROOT,
                        manifests: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    root = Path(repository_root)
    try:
        if configuration.get("schema_version") == "phase2-workload-2.0.0":
            from public.inference.workload_job import validate_job
            return validate_job(configuration,repository_root=root)
        if configuration.get("schema_version") == "phase2-conformance-2.0.0":
            from public.inference.conformance_job import validate_job
            return validate_job(configuration)
        if configuration.get("schema_version") == "phase1-fp32-baseline-1.0.0":
            return validate_baseline(configuration, repository_root=root)
        return validate_experiment(configuration, manifests=manifests, repository_root=root)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise IdentityError(str(exc)) from exc


def validate_experiment(
    configuration: Mapping[str, Any],
    *,
    manifests: Mapping[str, Mapping[str, Any]] | None = None,
    repository_root: str | Path | None = None,
    verify_lists: bool = True,
) -> dict[str, Any]:
    """Validate schema and all resolvable Phase 1 cross-references."""

    value = copy.deepcopy(dict(configuration))
    schema = json.loads(EXPERIMENT_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda item: list(item.absolute_path))
    if errors:
        detail = "; ".join(
            f"{'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise IdentityError(f"experiment schema validation failed: {detail}")

    _check_finite(value)
    for list_ref in (value["dataset"]["calibration"], value["dataset"]["evaluation"]):
        _portable_relative_path(list_ref["path"])

    root = Path(repository_root) if repository_root is not None else REPO_ROOT
    model = resolve_model(value["model"]["name"], root)
    for field, frozen in (("checkpoint_sha256", "checkpoint_sha256"), ("graph_sha256", "deployment_graph_sha256"),
                          ("graph_version", "deployment_graph_version"), ("preprocessing_version", "preprocessing_version")):
        if value["model"][field] != model[frozen]:
            raise IdentityError(f"model {field} mismatch")
    if value["operators"]["semantics_version"] != "1.0.0" or value["runtime"]["semantic_version"] != "1.1.0":
        raise IdentityError("unsupported operator/runtime semantic version")
    if value["runtime"]["backend"] != "reference" or value["runtime"]["kernel_version"] != "oracle-1.1.0":
        raise IdentityError("unsupported Phase 1 backend/kernel version")
    if value["operators"]["policy"] == "strict" and value["operators"].get("higher_precision_exceptions"):
        raise IdentityError("strict operator policy forbids higher precision exceptions")
    if value["ptq"]["experiment"] == "A" and value["ptq"]["rounding"] != "rne":
        raise IdentityError("Experiment A requires rne")
    if manifests is None:
        paths = list((root / "public/formats/manifests/accepted").glob("*.json"))
        paths += list((root / "public/formats/manifests/accumulators").glob("*.json"))
        manifests = {item["name"]: item for path in paths if (item := json.loads(path.read_text())).get("name")}
    if manifests is not None:
        resolved: dict[str, dict[str, Any]] = {}
        for role, reference in value["formats"].items():
            name = reference["name"]
            if name not in manifests:
                raise IdentityError(f"unresolved {role} manifest: {name}")
            manifest = validate_manifest(manifests[name], role=role)
            actual = manifest_sha256(manifest)
            if actual != reference["sha256"]:
                raise IdentityError(
                    f"{role} manifest hash mismatch for {name}: declared {reference['sha256']}, actual {actual}"
                )
            resolved[role] = manifest

        if value["ptq"]["experiment"] == "A":
            permitted = {"none", "required_mapping", "intrinsic_shared"}
            for role in ("weight", "activation", "output"):
                mode = resolved[role]["scaling"]["mode"]
                if mode not in permitted:
                    raise IdentityError(f"Experiment A forbids {mode} scaling for {role}")
        if value["arithmetic"]["accumulator_policy"] == "family_appropriate_wide":
            accumulator = resolved["accumulator"]
            if accumulator["bits"] <= max(resolved["weight"]["bits"], resolved["activation"]["bits"]):
                raise IdentityError("family_appropriate_wide did not resolve to a wider accumulator manifest")

    if verify_lists:
        root = Path(repository_root) if repository_root is not None else REPO_ROOT
        from public.workloads.datasets.identity import verify_payload_record
        records = json.loads((root / "data/manifests/index.json").read_text())["records"]
        for reference in (value["dataset"]["calibration"], value["dataset"]["evaluation"]):
            record = records.get(reference["name"])
            if record is None or any(reference[key] != record[key] for key in ("path", "sha256")):
                raise IdentityError("list hash mismatch or unresolved frozen list")
            verify_payload_record(root, record)
        calibration = set(_verify_list(value["dataset"]["calibration"], root))
        evaluation = set(_verify_list(value["dataset"]["evaluation"], root))
        overlap = calibration & evaluation
        if overlap:
            example = sorted(overlap)[0]
            raise IdentityError(f"calibration/evaluation lists overlap; example: {example}")

    expected = value.get("experiment_id")
    actual_id = experiment_sha256(value)
    if expected is not None and expected != actual_id:
        raise IdentityError(f"experiment_id mismatch: declared {expected}, actual {actual_id}")
    return value


def repository_state(path: str | Path = REPO_ROOT) -> dict[str, str]:
    """Capture stable environment metadata without making it numerical identity."""

    root = Path(path)
    state = {
        "repository": root.name,
        "git_present": str((root / ".git").exists()).lower(),
        "platform": os.uname().sysname,
        "platform_release": platform.release(),
        "python": platform.python_version(),
    }
    if (root / ".git").exists():
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True
        ).stdout
        state.update({"git_revision": revision, "git_dirty": str(bool(porcelain)).lower()})
    return state
