"""Explicit per-graph compatibility evidence for the optional exact integer store."""
import argparse
from pathlib import Path
import resource

from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.evidence import verify_complete
from tools.phase3.execution_guard import verify_guard, controller_identity
from tools.phase3.provenance import source_reference


def trial(plan_path, ordinal, root=ROOT):
    from public.analysis.phase3.diagnostics import LayerDiagnostics
    from tools.phase3.integer_store import accelerated_stores
    from tools.run.phase3_thread_benchmark import trial as original_trial
    plan = read(plan_path)
    _, _, (_, _, graph, _, records, _) = verify_complete(checked(plan["accepted_pilot"], root), root, current_execution=True)
    original = LayerDiagnostics.__call__
    image_index, compared = 0, 0
    def compare(self, node, tensor):
        nonlocal image_index, compared
        original(self, node, tensor)
        expected = records[image_index]["backends"]["cuda"]["diagnostics"][node["name"]]
        if self.records[node["name"]].get("quantizer_event_counts") != expected.get("quantizer_event_counts"):
            raise ValueError("accelerated quantizer events differ from the retained reference")
        compared += 1
        if node["name"] == graph["nodes"][-1]["name"]:
            image_index += 1
    guard, implementation = verify_guard(root), controller_identity(root)
    LayerDiagnostics.__call__ = compare
    try:
        with accelerated_stores():
            original_trial(plan_path, ordinal)
    finally:
        LayerDiagnostics.__call__ = original
    if image_index != plan["images"] or compared != len(graph["nodes"]) * image_index:
        raise ValueError("incomplete accelerated layer comparison")
    if verify_guard(root) != guard or controller_identity(root) != implementation:
        raise ValueError("execution extension changed during validation")
    write(plan_path.parent / f"{ordinal:02d}-compatibility.json", {
        "guard_sha256": guard, "controller_sha256": implementation,
        "images": image_index, "layer_event_comparisons": compared,
        "matches_original_layer_codes_outputs_and_quantizer_events": True,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "native_evidence": reference(plan_path.parent / f"{ordinal:02d}-completed.json", root)})


def validation_plan(task, root=ROOT, *, images=8):
    approval = read(root / "artifacts/phase3/acceptance" / task["configuration_sha256"] / "acceptance.json")
    return {"schema_version": "phase3-thread-benchmark-1.0.0", "scope": "exact_integer_store_compatibility",
            "model": task["model"], "format": task["format"], "images": images,
            "settings": [{"backend": "cuda", "threads": 4}],
            "source_sha256": task["source_sha256"], "pipeline_sha256": task["pipeline_sha256"],
            "campaign_sha256": digest(campaign(root)), "prepared": task["prepared"],
            "accepted_pilot": approval["pilot"],
            "implementation": source_reference(root / "tools/run/phase3_thread_benchmark.py", root),
            "extension": source_reference(root / "tools/phase3/integer_store.py", root),
            "libraries": {b: reference(root / f"build/phase2/{b}/libprecision_{b}.so", root) for b in ("cpp", "cuda")}}


def certificate_path(task, root=ROOT):
    return root / "artifacts/phase3/controller/compatibility" / task["configuration_sha256"] / controller_identity(root) / "certificate.json"


def verify_certificate(task, root=ROOT):
    path = certificate_path(task, root)
    certificate = read(path)
    if (certificate["configuration_sha256"] != task["configuration_sha256"] or certificate["images"] != 8 or
            certificate["guard_sha256"] != verify_guard(root) or certificate["controller_sha256"] != controller_identity(root)):
        raise ValueError("invalid exact-store compatibility certificate")
    plan = read(checked(certificate["plan"], root))
    expected = validation_plan(task, root)
    if plan != expected:
        raise ValueError("compatibility certificate belongs to another execution definition")
    evidence = read(checked(certificate["compatibility"], root))
    if (evidence["images"] != 8 or not evidence["matches_original_layer_codes_outputs_and_quantizer_events"] or
            any(evidence[k] != certificate[k] for k in ("guard_sha256", "controller_sha256"))):
        raise ValueError("incomplete compatibility evidence")
    checked(evidence["native_evidence"], root)
    return reference(path, root)


def validate_extension(task, root=ROOT):
    path = certificate_path(task, root)
    if path.exists():
        return verify_certificate(task, root)
    plan = validation_plan(task, root)
    write(path.parent / "job.json", plan)
    trial(path.parent / "job.json", 0, root)
    evidence = read(path.parent / "00-compatibility.json")
    write(path, {"schema_version": "phase3-exact-store-certificate-1.0.0",
                 "configuration_sha256": task["configuration_sha256"], "images": 8,
                 "guard_sha256": verify_guard(root), "controller_sha256": controller_identity(root),
                 "plan": reference(path.parent / "job.json", root),
                 "compatibility": reference(path.parent / "00-compatibility.json", root),
                 "scope": "bit-exact native outputs, every layer hash and quantizer event counts versus frozen eight-image pilot",
                 "admission": "signed INT32/64 to signed <=8-bit integer; bounded finite-decimal scales; original fallback otherwise"})
    return verify_certificate(task, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    args = parser.parse_args()
    trial(args.trial, args.ordinal)


if __name__ == "__main__":
    main()
