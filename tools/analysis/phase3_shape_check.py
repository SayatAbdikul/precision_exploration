"""Cross-check retained FP32 shape observations against a completed native pilot."""
import argparse
from pathlib import Path

from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.evidence import verify_complete
from tools.phase3.provenance import source_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    args = parser.parse_args()
    _, summary, (_, config, graph, _, records, _) = verify_complete(args.pilot, current_execution=True)
    if summary["scope"] != "pilot":
        raise ValueError("shape cross-check requires a native pilot")
    inventory = read(ROOT / "results/summaries/phase3-shapes.json")
    if inventory["campaign_sha256"] != digest(campaign()):
        raise ValueError("shape inventory campaign mismatch")
    shape_ref = inventory["models"][config["model"]]
    observed = read(checked(shape_ref))
    expected = {node["name"]: observed["shapes"][node["name"]] for node in graph["nodes"]}
    checks = 0
    for record in records:
        for backend, execution in record["backends"].items():
            if {name: trace["shape"] for name, trace in execution["layers"].items()} != expected:
                raise ValueError(f"{backend} native shapes differ from FP32 observations")
            checks += len(expected)
    report = {"schema_version": "phase3-shape-check-1.0.0", "status": "verified", "model": config["model"],
              "pilot": reference(args.pilot), "shape_evidence": shape_ref,
              "implementation": source_reference(__file__), "images": len(records), "layer_shape_comparisons": checks,
              "scope": "native versus FP32 shape agreement for this pilot only"}
    write(ROOT / f"results/summaries/phase3-shape-check-{summary['job_sha256'][:12]}.json", report)
    print({"status": report["status"], "layer_shape_comparisons": checks})


if __name__ == "__main__":
    main()
