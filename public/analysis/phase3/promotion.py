"""Conservative D4 preservation without a top-N cutoff or implicit family pruning."""


def promote(records, expected_pairs):
    expected_pairs = set(expected_pairs)
    indexed = {}
    for row in records:
        key = (row["model"], row["format"])
        if key in indexed or key not in expected_pairs:
            raise ValueError("duplicate or unexpected screening configuration")
        if row.get("scope") != "screen" or row.get("images") != 1000:
            raise ValueError("D4 requires complete fixed-1k evidence, not native pilots")
        if row["label"] not in {"PROMISING", "UNCERTAIN", "CATASTROPHIC/BROKEN"}:
            raise ValueError("unknown screening classification")
        indexed[key] = row
    missing = sorted(expected_pairs - set(indexed))
    promotions, removals, pending = [], [], []
    preservation_reasons = {"family_representative", "near_pareto", "hardware_efficiency", "workload_specialist", "uncertainty_buffer"}
    for key, row in sorted(indexed.items()):
        signals = set(row.get("preservation_signals", []))
        if not signals <= preservation_reasons:
            raise ValueError("unrecognized preservation signal")
        reasons = sorted(signals)
        if row["label"] in {"PROMISING", "UNCERTAIN"}:
            reasons.append(row["label"].lower())
        if row["label"] == "CATASTROPHIC/BROKEN" and not row.get("diagnosis_verified"):
            pending.append({"model": key[0], "format": key[1], "reason": "catastrophic result needs verified diagnosis"})
            reasons.append("diagnosis_pending_preserve")
        item = {"model": key[0], "format": key[1], "family": row["family"]}
        if reasons:
            promotions.append({**item, "reasons": reasons})
        else:
            removals.append({**item, "reason": "diagnosed catastrophic configuration only"})
    # If every measured representative of a family would be removed, require
    # explicit family review rather than turning configuration failures into it.
    represented = {row["family"] for row in promotions}
    families = {row["family"] for row in records}
    for family in sorted(families - represented):
        pending.append({"family": family, "reason": "no promoted family representative; further reasonable configurations need review"})
    return {"status": "ready_for_D4_review" if not missing and not pending else "incomplete",
            "missing_configurations": [{"model": model, "format": name} for model, name in missing],
            "pending_review": pending, "promote": promotions, "configuration_removals": removals,
            "family_eliminations": [], "hard_top_n": None}
