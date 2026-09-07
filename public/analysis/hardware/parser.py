"""Strict normalization of generic synthesis pilot reports."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from public.experiments.registry.identity import canonical_json_bytes


PARSER_VERSION = "1.1.0"


class HardwareParseError(ValueError):
    """Raised when a hardware report is incomplete or ambiguous."""


def hardware_run_id(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HardwareParseError(f"missing or invalid non-negative integer field: {key}")
    return value


def _validate_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(identity)
    required = {
        "rtl_sha256",
        "harness_version",
        "pdk",
        "library",
        "corner",
        "vt",
        "tool",
        "tool_version",
        "target_clock",
        "pipeline_stages",
    }
    missing = sorted(required - normalized.keys())
    if missing:
        raise HardwareParseError(f"missing hardware identity fields: {', '.join(missing)}")
    return normalized


def parse_yosys_stat(
    report: str | Path | Mapping[str, Any],
    *,
    top: str,
    identity: Mapping[str, Any],
    evidence_level: str = "logic_synthesis",
) -> dict[str, Any]:
    report_sha256 = None
    if isinstance(report, Mapping):
        value = dict(report)
    else:
        path = Path(report)
        try:
            payload = path.read_bytes()
            report_sha256 = hashlib.sha256(payload).hexdigest()
            value = json.loads(payload)
        except (OSError, json.JSONDecodeError) as exc:
            raise HardwareParseError(f"cannot read Yosys JSON report: {exc}") from exc

    modules = value.get("modules")
    design = value.get("design")
    if not isinstance(modules, Mapping) or not isinstance(design, Mapping):
        raise HardwareParseError("Yosys report requires modules and design objects")
    module = modules.get(top) or modules.get(f"\\{top}")
    if not isinstance(module, Mapping):
        raise HardwareParseError(f"top module not found in Yosys report: {top}")

    cell_types = module.get("num_cells_by_type", {})
    if not isinstance(cell_types, Mapping):
        raise HardwareParseError("num_cells_by_type must be an object")
    design_cell_types = design.get("num_cells_by_type", cell_types)
    if not isinstance(design_cell_types, Mapping):
        raise HardwareParseError("design.num_cells_by_type must be an object")
    metrics: dict[str, dict[str, int | float | str]] = {
        "cell_count": {"value": _integer(module, "num_cells"), "unit": "count"},
        "design_cell_count": {"value": _integer(design, "num_cells"), "unit": "count"},
        "leaf_cell_count": {
            "value": sum(int(count) for cell, count in design_cell_types.items() if not str(cell).startswith("$")),
            "unit": "count",
        },
        "wire_count": {"value": _integer(module, "num_wires"), "unit": "count"},
        "wire_bits": {"value": _integer(module, "num_wire_bits"), "unit": "bits"},
        "public_wire_bits": {"value": _integer(module, "num_pub_wire_bits"), "unit": "bits"},
    }
    area = module.get("area", design.get("area"))
    if area is not None:
        if isinstance(area, bool) or not isinstance(area, (int, float)) or area < 0:
            raise HardwareParseError("area must be a non-negative number")
        metrics["cell_area"] = {"value": float(area), "unit": "library_area_units"}
    register_count = sum(
        int(count) for cell, count in design_cell_types.items()
        if str(cell).lstrip("\\$_").upper().startswith(("DFF", "SDFF"))
    )
    metrics["register_count"] = {"value": register_count, "unit": "count"}

    core_name = identity.get("core_module")
    if core_name is not None:
        core = modules.get(core_name) or modules.get(f"\\{core_name}")
        if not isinstance(core, Mapping):
            raise HardwareParseError(f"core module not found in Yosys report: {core_name}")
        metrics["core_cell_count"] = {"value": _integer(core, "num_cells"), "unit": "count"}
        metrics["wrapper_leaf_cell_count"] = {
            "value": metrics["leaf_cell_count"]["value"] - _integer(core, "num_cells"),
            "unit": "count",
        }
        core_area = core.get("area")
        if not isinstance(core_area, (int, float)) or isinstance(core_area, bool) or core_area < 0:
            raise HardwareParseError("core area is missing or invalid")
        metrics["core_cell_area"] = {"value": float(core_area), "unit": "library_area_units"}
        if area is not None:
            metrics["wrapper_cell_area"] = {
                "value": round(float(area) - float(core_area), 12),
                "unit": "library_area_units",
            }

    normalized_identity = _validate_identity(identity)
    return {
        "schema_version": "1.0.0",
        "hardware_run_id": hardware_run_id(normalized_identity),
        "identity": normalized_identity,
        "metrics": metrics,
        "cell_types": {str(key): int(item) for key, item in sorted(cell_types.items())},
        "evidence_level": evidence_level,
        "parser_version": PARSER_VERSION,
        "warnings": [],
        "source_report_sha256": report_sha256,
    }


def _single_float(pattern: str, text: str, field: str) -> float:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise HardwareParseError(f"OpenSTA report requires exactly one {field} field")
    value = float(matches[0])
    if not math.isfinite(value):
        raise HardwareParseError(f"OpenSTA {field} must be finite")
    return value


def parse_opensta_report(
    report: str | Path,
    *,
    identity: Mapping[str, Any],
    evidence_level: str = "post_synthesis_sta_no_interconnect",
) -> dict[str, Any]:
    """Normalize the explicit summary emitted by ``opensta_pilot.tcl``.

    OpenSTA's default time unit is fixed to nanoseconds by the pilot identity.
    The parser rejects missing/duplicate summary fields and verifies the
    reported frequency against the reported minimum period.
    """

    path = Path(report)
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise HardwareParseError(f"cannot read OpenSTA report: {exc}") from exc
    normalized_identity = _validate_identity(identity)
    if normalized_identity.get("time_unit") != "ns":
        raise HardwareParseError("OpenSTA pilot identity requires time_unit='ns'")

    target_match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*ns", str(normalized_identity["target_clock"]))
    if target_match is None:
        raise HardwareParseError("target_clock must be an explicit value in ns")
    target_period = float(target_match.group(1))
    worst_slack = _single_float(r"^worst slack max\s+([-+0-9.eE]+)\s*$", text, "worst slack")
    total_negative_slack = _single_float(r"^tns max\s+([-+0-9.eE]+)\s*$", text, "total negative slack")
    minimum_period = _single_float(r"^\S+ period_min =\s*([-+0-9.eE]+)\s+fmax", text, "minimum period")
    reported_fmax = _single_float(r"^\S+ period_min =\s*[-+0-9.eE]+\s+fmax =\s*([-+0-9.eE]+)\s*$", text, "fmax")
    if minimum_period <= 0 or reported_fmax <= 0:
        raise HardwareParseError("OpenSTA minimum period and fmax must be positive")
    calculated_fmax = 1000.0 / minimum_period
    if not math.isclose(reported_fmax, calculated_fmax, rel_tol=0.005, abs_tol=0.05):
        raise HardwareParseError("OpenSTA fmax is inconsistent with the minimum period")

    return {
        "schema_version": "1.0.0",
        "hardware_run_id": hardware_run_id(normalized_identity),
        "identity": normalized_identity,
        "metrics": {
            "target_period": {"value": target_period, "unit": "ns"},
            "worst_slack": {"value": worst_slack, "unit": "ns"},
            "total_negative_slack": {"value": total_negative_slack, "unit": "ns"},
            "minimum_clock_period": {"value": minimum_period, "unit": "ns"},
            "fmax": {"value": reported_fmax, "unit": "MHz"},
            "timing_met": {"value": int(worst_slack >= 0), "unit": "boolean"},
        },
        "evidence_level": evidence_level,
        "parser_version": PARSER_VERSION,
        "warnings": [
            "post-synthesis STA uses ideal clocks and Liberty cell delay only",
            "no placement, routing, extracted parasitics, or power evidence is claimed",
        ],
        "missing_metrics": ["routed_area", "wire_delay", "leakage_power", "dynamic_power", "energy_per_operation"],
        "source_report_sha256": hashlib.sha256(payload).hexdigest(),
    }
