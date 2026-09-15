"""Prepare frozen Phase 3 graphs; retain per-configuration preparation errors."""
import argparse
from pathlib import Path
from tools.phase3.common import ROOT, campaign, read, write
from tools.phase3.graphs import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--formats", nargs="+")
    parser.add_argument("--report", type=Path, default=ROOT / "results/summaries/phase3-preparation.json",
                        help="independent workers must use distinct report paths")
    args = parser.parse_args()
    plan = campaign()
    path = args.report
    report = read(path) if path.exists() else {"schema_version": "phase3-preparation-1.0.0", "records": {}}
    for model in args.models or plan["models"]:
        for format_name in args.formats or plan["formats"]:
            key = f"{model}/{format_name}"
            print(f"preparing {key}", flush=True)
            try:
                report["records"][key] = build(model, format_name, plan)
            except Exception as error:
                report["records"][key] = {"status": "preparation_failed", "error": f"{type(error).__name__}: {error}"}
            write(path, report)
            print(key, report["records"][key]["status"], flush=True)


if __name__ == "__main__":
    main()
