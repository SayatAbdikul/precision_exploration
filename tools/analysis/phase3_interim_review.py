"""Record an evidence-checked interim review of the eight completed integer screens.

This is not a D4 approval or a diagnosis of the four catastrophic-loss cases.
The native pilot controller can continue independently while this runs.
"""
from datetime import datetime, timezone

from public.analysis.phase3.statistics import screening_label
from tools.phase3.acceptance import verify_acceptance
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write


INTEGER_RUN = "20260923T081200Z-91dced18"
KEYS = {f"{model}/{fmt}" for model in ("resnet18", "mobilenet_v2")
        for fmt in ("int4", "int5", "int6", "int8")}


def review(root=ROOT):
    plan = campaign(root)
    campaign_sha = digest(plan)
    run_path = root / "artifacts/phase3/controller/runs" / INTEGER_RUN / "status.json"
    run = read(run_path)
    if (run["campaign_sha256"] != campaign_sha or run.get("status") != "blocked" or
            run["counts"] != {"pending": 0, "running": 0, "completed": 29,
                              "failed": 0, "blocked": 0, "interrupted": 0}):
        raise ValueError("integer controller did not finish its 29 tasks cleanly")
    readiness_path = root / "results/summaries/phase3-d4-readiness.json"
    readiness = read(readiness_path)
    if (readiness["campaign_sha256"] != campaign_sha or readiness["decision"] != "D4_OPEN" or
            readiness["completed_screen_analyses"] != 8 or len(readiness["missing_configurations"]) != 92):
        raise ValueError("D4 readiness must be recomputed for eight complete screens")

    rows = []
    for task in run["tasks"].values():
        if task["task"]["kind"] != "analyze":
            continue
        if task["status"] != "completed":
            raise ValueError("analysis task was not completed")
        receipt = read(checked(task["result"], root))
        outputs = receipt["outputs"]
        analysis = read(checked(outputs["analysis"], root))
        diagnostics = read(checked(outputs["diagnostics"], root))
        key = f"{analysis['model']}/{analysis['format']}"
        if (key not in KEYS or analysis["scope"] != "screen" or analysis["images"] != 1000 or
                diagnostics["model"] != analysis["model"] or diagnostics["format"] != analysis["format"] or
                diagnostics["scope"] != "screen" or diagnostics["images"] != 1000 or
                diagnostics["job_sha256"] != analysis["job_sha256"] or
                diagnostics["evidence"] != analysis["evidence"][0] or
                analysis["campaign_sha256"] != campaign_sha or
                analysis["classification"] != screening_label(analysis["statistics"], plan["statistics"])):
            raise ValueError("screen, analysis or diagnostics identity mismatch")
        summary_path = checked(analysis["evidence"][0], root)
        summary = read(summary_path)
        job = read(summary_path.parent / "job.json")
        prepared = read(checked(job["prepared"], root))
        verify_acceptance(prepared, root)
        if (summary["status"] != "completed" or summary["images"] != 1000 or
                summary["job_sha256"] != analysis["job_sha256"] or
                prepared["configuration_sha256"] != analysis["configuration_sha256"] or
                len(diagnostics["layers"]) == 0 or
                any(layer["images"] != 1000 for layer in diagnostics["layers"])):
            raise ValueError("review requires complete matching native evidence")
        highest = sorted((layer for layer in diagnostics["layers"]
                          if isinstance(layer.get("sample_mse"), (int, float))),
                         key=lambda layer: layer["sample_mse"], reverse=True)[:3]
        rows.append({
            "configuration": key,
            "analysis": outputs["analysis"], "diagnostics": outputs["diagnostics"],
            "screen": analysis["evidence"][0], "acceptance": prepared["screen_acceptance"],
            "top1": analysis["statistics"]["metrics"]["top1"],
            "top5": analysis["statistics"]["metrics"]["top5"],
            "screening_label": analysis["classification"],
            "review_status": ("quality_loss_confirmed_cause_unresolved_preserve" if
                              analysis["classification"]["diagnosis_required"] else
                              "retain_uncertain_pending_full_campaign"),
            "highest_sampled_mse_nodes": [{"node": layer["node"], "sample_mse": layer["sample_mse"]}
                                          for layer in highest],
            "nonfinite_quantizer_events": sum((layer.get("quantizer_event_counts") or {}).get("nonfinite_inputs", 0)
                                              for layer in diagnostics["layers"]),
        })
    if {row["configuration"] for row in rows} != KEYS or len(rows) != len(KEYS):
        raise ValueError("review does not cover exactly eight integer configurations")
    rows.sort(key=lambda row: row["configuration"])
    diagnosis_pending = [row["configuration"] for row in rows
                         if row["screening_label"]["diagnosis_required"]]
    if diagnosis_pending != ["mobilenet_v2/int4", "mobilenet_v2/int5", "mobilenet_v2/int6", "resnet18/int4"]:
        raise ValueError("catastrophic-loss population changed")

    sensitivity_paths = [root / "results/summaries" / f"phase3-sensitivity-{suffix}.json"
                         for suffix in ("260aa67dd4eb", "2669f173ebf3", "4d80322df03d", "a4e4f8669412")]
    for path in sensitivity_paths:
        study = read(path)
        if study["model"] != "resnet18" or study["format"] != "int4" or study["images"] != 8:
            raise ValueError("retained one-layer sensitivity study changed")
    hardware_path = root / "results/summaries/phase3-hardware-priors.json"
    hardware = read(hardware_path)
    if (hardware["campaign_sha256"] != campaign_sha or
            len(hardware["prepared_graph_storage"]) != 100):
        raise ValueError("hardware prior coverage changed")
    gate_path = root / "results/summaries/phase3-gate-inventory.json"
    gate = read(gate_path)
    if gate["campaign_sha256"] != campaign_sha or len(gate["records"]) != 100:
        raise ValueError("calibration/preparation gate inventory changed")
    document = {
        "schema_version": "phase3-interim-eight-screen-review-1.0.0",
        "campaign_sha256": campaign_sha,
        "reviewer": "Codex evidence review", "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "eight completed integer classifier fixed-1k screens",
        "decision": "D4_OPEN", "campaign_review_complete": False,
        "configuration_reviews": rows, "diagnoses_pending": diagnosis_pending,
        "missing_screen_configurations": len(readiness["missing_configurations"]),
        "configuration_removals": [], "family_eliminations": [],
        "review_dimensions": {
            "sensitivity": {
                "status": "reviewed_current_evidence_causal_diagnoses_pending",
                "finding": "Four ResNet18 INT4 one-operation studies do not reproduce the strict full-graph loss; the three MobileNetV2 severe cases lack causal interventions. Sampled layer error is descriptive, not causal.",
                "evidence": [reference(path, root) for path in sensitivity_paths],
            },
            "hardware_preservation": {
                "status": "reviewed_priors_measurements_pending",
                "finding": "All 100 prepared graphs have storage/operator priors, but no routed area, measured energy or complete-system PPA. No hardware-based removal is supported.",
                "evidence": [reference(hardware_path, root)],
            },
            "family_preservation": {
                "status": "reviewed_current_evidence_other_families_pending",
                "finding": "Only eight integer classifier screens are measured. Preserve the integer family and every unmeasured family; no top-N or family elimination is justified.",
                "evidence": [reference(readiness_path, root)],
            },
            "calibration": {
                "status": "reviewed_frozen_inputs_representativeness_pending",
                "finding": "The eight screens share the frozen campaign and calibration definitions. Their losses do not identify calibration as the cause. Detector calibration coverage is separately unresolved.",
                "evidence": [reference(gate_path, root),
                             reference(root / "docs/architecture/phase3-yolo-int8-diagnosis.md", root)],
            },
        },
        "evidence": [reference(run_path, root), reference(readiness_path, root)],
        "limits": ["not an approval of D4", "not a causal diagnosis of any severe configuration",
                   "sampled layer MSE and quantizer events cannot identify a sole failing operator",
                   "native pilots are not fixed-1k screens"],
    }
    directory = root / "artifacts/phase3/reviews/interim-eight-integer"
    path = directory / f"{digest(document)}.json"
    if path.exists() and read(path) != document:
        raise ValueError("review artifact hash collision")
    write(path, document)
    return path


if __name__ == "__main__":
    print(review())
