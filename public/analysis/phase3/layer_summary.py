"""Pool sampled layer diagnostics without treating missing events as zeros."""
import math


def summarize(records, backend):
    if not records:
        raise ValueError("layer summary needs image evidence")
    names = set(records[0]["backends"][backend]["diagnostics"])
    if any(set(record["backends"][backend]["diagnostics"]) != names for record in records):
        raise ValueError("diagnostic layer coverage differs between images")
    output = []
    for name in sorted(names):
        rows = [record["backends"][backend]["diagnostics"][name] for record in records]
        samples = sum(row["sampled_elements"] for row in rows)
        if samples <= 0:
            raise ValueError("diagnostics have no sampled elements")
        mse = sum(row["mse"]*row["sampled_elements"] for row in rows)/samples
        power = sum((row["fp32"]["variance"]+row["fp32"]["mean"]**2)*row["sampled_elements"] for row in rows)/samples
        observed = [row["quantizer_event_counts"] for row in rows if row["quantizer_event_counts"] is not None]
        events = {key: sum(row[key] for row in observed) for key in observed[0]} if observed else None
        output.append({"node": name, "images": len(rows), "sampled_elements": samples,
                       "population_elements": sum(row["population_elements"] for row in rows), "sample_mse": mse,
                       "sample_sqnr_db": 10*math.log10(power/mse) if power > 0 and mse > 0 else None,
                       "sqnr_status": "exact" if mse == 0 else "zero_reference_signal" if power == 0 else "finite",
                       "sample_zero_fraction": sum(row["candidate"]["zero_fraction"]*row["sampled_elements"] for row in rows)/samples,
                       "sample_outlier_fraction": sum(row["candidate"]["outlier_fraction"]*row["sampled_elements"] for row in rows)/samples,
                       "images_with_observed_quantizer_calls": len(observed), "quantizer_event_counts": events,
                       "event_statuses": sorted({row["event_status"] for row in rows})})
    return {"backend": backend, "images": len(records), "layers": output,
            "inspection_order_by_sample_mse": [row["node"] for row in sorted(output, key=lambda row: (-row["sample_mse"], row["node"]))],
            "limits": ["sampled descriptive evidence, not a causal layer-sensitivity experiment",
                       "event counts describe observed quantizer calls, including alignment and table generation",
                       "cached operations may have no quantizer calls; missing observations are not zero events",
                       "sample indices repeat between images; no independent-element confidence interval is implied"]}
