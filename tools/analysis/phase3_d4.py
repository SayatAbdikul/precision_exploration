"""Publish conservative D4 readiness; missing experiments never become pruning."""
from public.analysis.phase3.promotion import promote
from public.analysis.phase3.statistics import paired_classification, screening_label
from public.inference.reference.arithmetic import format_named
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.evidence import verify_complete


def main():
    plan = campaign()
    expected = [(model, name) for model in plan["models"] for name in plan["formats"]]
    records = []
    pending = []
    for path in sorted((ROOT / "results/summaries").glob("phase3-analysis-*.json")):
        analysis = read(path)
        if analysis["scope"] != "screen":
            continue
        if analysis["campaign_sha256"] != digest(plan):
            raise ValueError("D4 analysis belongs to another campaign")
        checked(analysis["analysis_implementation"])
        checked(analysis["statistics_implementation"])
        for dependency in analysis.get("statistics_dependencies", []):
            checked(dependency)
        summary_path = checked(analysis["evidence"][0])
        job, summary, (_, config, _, _, _, _) = verify_complete(summary_path)
        if (summary["scope"] != "screen" or summary["images"] != 1000 or analysis["job_sha256"] != summary["job_sha256"]
                or analysis["configuration_sha256"] != summary["configuration_sha256"]):
            raise ValueError("D4 requires matching fixed-1k execution evidence")
        policy = plan["statistics"]
        statistics = analysis["statistics"]
        if any(statistics[key] != policy[key] for key in ("confidence", "seed")) or statistics["images"] != 1000:
            raise ValueError("statistical settings differ from the frozen campaign")
        model, name = config["model"], config["formats"]["activation"]["name"]
        if model != "yolov8n":
            paired = read(checked(summary["paired"]))
            reproduced = paired_classification(paired, [row["sample_id"] for row in paired],
                confidence=policy["confidence"], seed=policy["seed"], resamples=policy["classification_resamples"])
            if reproduced != statistics:
                raise ValueError("paired classification statistics do not reproduce")
        elif statistics["resamples"] != policy["detector_resamples"]:
            raise ValueError("detector resample count differs from frozen policy")
        label = screening_label(statistics, policy)
        if label["diagnosis_required"]:
            pending.append({"model": model, "format": name, "reason": "catastrophic quality loss still needs verified diagnosis; preserve"})
        records.append({"model": model, "format": name, "family": format_named(name).family,
                        "scope": "screen", "images": 1000, "label": label["label"],
                        "analysis": reference(path), "metrics": statistics["metrics"]})
    result = promote(records, expected)
    indexed = {(row["model"], row["format"]): row for row in records}
    for row in result["promote"]:
        evidence = indexed[(row["model"], row["format"])]
        row.update(analysis=evidence["analysis"], metrics=evidence["metrics"])
    result["pending_review"].extend(pending)
    # This report is a readiness artifact, not an automatic acceptance of D4.
    result["pending_review"].extend([
        {"reason": "review selected layer-sensitivity evidence and remaining catastrophic diagnoses"},
        {"reason": "family, near-Pareto, workload-specialist and hardware preservation review still required"}])
    result["status"] = "incomplete"
    result["sensitivity_evidence"] = [reference(path) for path in sorted((ROOT / "results/summaries").glob("phase3-sensitivity-*.json"))]
    result.update(schema_version="phase3-d4-readiness-1.0.0", campaign_sha256=digest(plan),
                  completed_screen_analyses=len(records), decision="D4_OPEN")
    write(ROOT / "results/summaries/phase3-d4-readiness.json", result)
    print({"decision": result["decision"], "completed_screen_analyses": len(records), "missing_configurations": len(result["missing_configurations"])})


if __name__ == "__main__":
    main()
