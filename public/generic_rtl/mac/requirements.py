"""Versioned hardware requirement projection of accepted numerical manifests.

No arithmetic policy is redefined here: this is an inventory of the frozen
manifest and operator contracts, with unresolved mappings kept explicit.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from public.formats.oracle.manifest import load_manifest, manifest_sha256
from public.formats.oracle.number_format import NumberFormat

ROOT = Path(__file__).resolve().parents[3]
ACCEPTED = ROOT / "public/formats/manifests/accepted"
RESOLUTIONS = ROOT / "public/experiments/configs/experiment_a/phase3-accumulator-resolutions-v1.json"
CAMPAIGN = ROOT / "public/experiments/configs/experiment_a/phase3-screen-v1.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accumulator_info(name: str) -> dict:
    path = ROOT / "public/formats/manifests/accumulators" / f"{name}.json"
    manifest = load_manifest(path)
    return {"name": name, "bits": manifest["bits"], "family": manifest["family"],
            "encoding": manifest["encoding"], "rounding": manifest["rounding"],
            "overflow": manifest["overflow"], "manifest_sha256": manifest_sha256(manifest)}


def indexed_resolution(root: Path, reference: dict, model: str, name: str) -> dict:
    """Read a resolution only after checking its indexed content identity."""
    relative = Path(reference["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"resolution path is not repository relative: {relative}")
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    if file_sha256(path) != reference["sha256"]:
        raise ValueError(f"accumulator resolution hash mismatch: {model}/{name}")
    resolution = json.loads(path.read_text())
    if resolution.get("model") != model or resolution.get("format") != name:
        raise ValueError(f"accumulator resolution identity mismatch: {model}/{name}")
    return resolution


def discrete_code_mapping(manifest: dict) -> dict:
    """Project every binary/ternary code through the accepted oracle."""
    fmt = NumberFormat(manifest)
    values = []
    reserved = []
    for code in range(1 << manifest["bits"]):
        decoded = fmt.decode(code)
        if decoded.is_nan():
            reserved.append(code)
        values.append({"code": code, "value": str(decoded),
                       "classification": "reserved" if decoded.is_nan() else "finite"})
    return {"encoding": manifest["encoding"], "manifest_zero": manifest["zero"],
            "code_values": values, "reserved_codes": reserved}


def requirements() -> dict:
    index = json.loads((ACCEPTED / "index.json").read_text())
    resolutions = json.loads(RESOLUTIONS.read_text())["resolutions"]
    campaign = json.loads(CAMPAIGN.read_text())
    format_reference = campaign["inputs"]["formats"]
    if format_reference["path"] != str((ACCEPTED / "index.json").relative_to(ROOT)):
        raise ValueError("campaign accepted-format index path changed")
    if file_sha256(ACCEPTED / "index.json") != format_reference["sha256"]:
        raise ValueError("campaign accepted-format index hash mismatch")
    rows = []
    for entry in index["manifests"]:
        m = load_manifest(ROOT / entry["path"])
        if manifest_sha256(m) != entry["sha256"]:
            raise ValueError(f"manifest identity changed: {entry['name']}")
        name, family, scaling = m["name"], m["family"], m["scaling"]
        code_mapping = (discrete_code_mapping(m) if family in {"binary", "ternary"} else
                        m.get("numeric", m.get("float", m.get("codebook", m.get("posit", m.get("logarithmic"))))))
        selected = {}
        for model in campaign["models"]:
            resolution = resolutions.get(f"{model}/{name}")
            if resolution:
                r = indexed_resolution(ROOT, resolution, model, name)
                selected[model] = {"accumulator": accumulator_info(r["accumulator"]), "status": r["status"],
                                   "resolution_sha256": resolution["sha256"]}
            else:
                selected[model] = {"accumulator": None, "status": "graph_resolution_not_in_index"}
        intrinsic = scaling["mode"] == "intrinsic_shared"
        integer = family == "integer"
        rows.append({
            "format": name, "family": family, "manifest_sha256": entry["sha256"],
            "weight": {"format": name, "bits": m["bits"], "signed": m["signed"],
                       "encoding": m["encoding"], "code_mapping": code_mapping},
            "activation": {"format": name, "bits": m["bits"], "signed": m["signed"],
                           "encoding": m["encoding"], "code_mapping": code_mapping},
            "product": {"model": "C", "precision": "exact_unrounded_product_to_accumulator",
                        "integer_bits_if_raw_integer": 2 * m["bits"] if integer else None,
                        "general_hardware_width": None if not integer else 2 * m["bits"]},
            "accumulator": {"policy": "per_graph_evidence_required", "by_model": selected,
                            "round_after_each_product_add": True,
                            "integer_baseline_options": [accumulator_info(n) for n in ("int32_accumulator", "int64_accumulator")] if integer else []},
            "reduction_order": "sequential_k_ascending",
            "rounding": m["rounding"], "operand_overflow": m["overflow"],
            "accumulator_overflow": "from_resolved_accumulator_manifest",
            "bias": {"representation": "stored_accumulator_code", "point": "once_after_reduction_before_activation_and_requantization",
                     "scale": "product_domain_for_integer; resolved_domain_for_other_families"},
            "scale": {"kind": scaling["mode"], "granularity": scaling["granularity"],
                      "selection": "static_offline_mse_mapping" if scaling["mode"] == "required_mapping" else
                                   "intrinsic_block_mse_search" if intrinsic else "none",
                      "precision_bits": scaling.get("scale_bits") if intrinsic else None,
                      "precision_status": "manifest_exact" if intrinsic else "unresolved_mapping_precision" if scaling["mode"] != "none" else "not_applicable",
                      "block": m.get("block"), "mapped_scale_bits_64": "analytical_storage_assumption_only" if scaling["mode"] == "required_mapping" else None},
            "output": {"storage_format": name, "conversion": "activation_then_explicit_requantization_at_output_store",
                       "rounding": m["rounding"], "overflow": m["overflow"], "underflow": m["underflow"],
                       "conversion_rtl_status": "bounded_exact_power_of_two_integer_only" if integer else "missing",
                       "external_scale_mapping_status": "missing" if not intrinsic else "intrinsic_block_scale_specified"},
            "hardware_status": {"raw_integer_model_c_mac": "implemented_scale_1" if integer else "not_implemented",
                                "full_operator": "requirements_only",
                                "mapping": "needs_calibrated_scale_mapping; exact_power_of_two_conversion_only" if integer else "needs_family_datapath_and_conversion_rtl",
                                "dynamic_scale_selection": "requires_oracle_equivalent_mse_policy" if intrinsic else "not_intrinsic"},
            "operator_identity": {"path": "docs/contracts/operator-semantics.md", "version": "1.0.0",
                                  "sha256": file_sha256(ROOT / "docs/contracts/operator-semantics.md")},
            "campaign_identity": "public/experiments/configs/experiment_a/phase3-screen-v1.json",
            "numerical_change_required_for": ["parallel_tree_reduction", "single_round_exact_kulisch_accumulation",
                                               "maximum_based_shared_scale_in_place_of_mse_search"] if intrinsic else
                                             ["parallel_tree_reduction", "single_round_exact_kulisch_accumulation"],
        })
    return {"schema_version": "hardware-requirements-1.0.0", "accepted_index_aggregate_sha256": index["aggregate_sha256"],
            "arithmetic_contract": "docs/contracts/arithmetic.md#1.0.0",
            "operator_contract": "docs/contracts/operator-semantics.md#1.0.0",
            "campaign_arithmetic": campaign["arithmetic"], "campaign_ptq": campaign["ptq"],
            "campaign_sha256": file_sha256(CAMPAIGN),
            "formats": rows,
            "limits": ["uniform W=A=output campaign projection; independent formats remain allowed by arithmetic contract",
                       "network configuration identity also requires graph, calibration, scales, operators and resolved accumulator",
                       "raw scale=1 integer MAC is not a calibrated full operator or network quality point"]}
