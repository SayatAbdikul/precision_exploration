"""Combine independently checkpointed preparation workers without shared writes."""
from public.inference.conformance_job import source_identity
from tools.phase3.common import ROOT, campaign, checked, digest, read


def preparation_records(root=ROOT):
    paths = [root / "results/summaries/phase3-preparation.json"]
    paths += sorted((root / "artifacts/phase3/preparation").glob("*.json"))
    records, sources = {}, {}
    current = source_identity()
    for path in paths:
        if not path.exists():
            continue
        report = read(path)
        if report["schema_version"] != "phase3-preparation-1.0.0":
            raise ValueError("unknown preparation inventory schema")
        for key, row in report["records"].items():
            source = None
            if "configuration" in row:
                config = read(checked(row["configuration"], root))
                if (key != f"{row['model']}/{row['format']}" or digest(config) != row["configuration_sha256"]
                        or config["model"] != row["model"] or config["formats"]["activation"]["name"] != row["format"]):
                    raise ValueError("preparation key differs from its configuration")
                source = config["runtime"]["source_sha256"]
            if key not in records or source == current and sources[key] != current or source is not None and sources[key] is None:
                records[key], sources[key] = row, source
            elif source == current and sources[key] == current and row != records[key]:
                # A versioned accumulator resolution can supersede another
                # same-source configuration; choose the active definition.
                if row["configuration_sha256"] == records[key]["configuration_sha256"]:
                    raise ValueError("conflicting preparation evidence for the same configuration")
                from tools.phase3.graphs import configuration
                active = digest(configuration(row["model"], row["format"], campaign(root), root))
                if row["configuration_sha256"] == active:
                    records[key] = row
    return records
