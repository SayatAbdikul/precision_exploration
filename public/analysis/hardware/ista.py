"""Strict reader for the local iSTA/iPA pilot's timing and power summaries."""
from __future__ import annotations

import math
from pathlib import Path
import re


def parse_reports(timing: Path, power: Path):
    text = timing.read_text()
    version = re.search(r"GitVersion: ([0-9a-f]{40})", text)
    paths, tns = {"max": [], "min": []}, {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 8 and cells[2] in paths:
            paths[cells[2]].append(float(cells[6]))
        elif len(cells) == 3 and cells[1] in paths:
            if cells[1] in tns:
                raise ValueError("pilot expects exactly one clock group")
            tns[cells[1]] = float(cells[2])
    if not version or not all(paths.values()) or set(tns) != set(paths):
        raise ValueError("incomplete iSTA timing summary")
    metrics = {}
    for kind, label in (("max", "setup"), ("min", "hold")):
        metrics[label + "_worst_slack"] = {"value": min(paths[kind]), "unit": "ns"}
        metrics[label + "_tns"] = {"value": tns[kind], "unit": "ns"}
    text = power.read_text()
    for label, key in (("Net Switch Power", "net_switch_power"),
                       ("Cell Internal Power", "cell_internal_power"),
                       ("Cell Leakage Power", "cell_leakage_power"),
                       ("Total Power", "total_power")):
        matches = re.findall(r"^" + label + r"\s+==\s+([+0-9.eE-]+)(.*)$", text, re.MULTILINE)
        if len(matches) != 1 or key == "total_power" and matches[0][1].strip() != "W":
            raise ValueError("missing, duplicate or unsupported iPA power units")
        value = float(matches[0][0])
        if value < 0:
            raise ValueError("negative power")
        metrics[key] = {"value": value, "unit": "W"}
    if not all(math.isfinite(row["value"]) for row in metrics.values()):
        raise ValueError("nonfinite hardware metric")
    components = sum(metrics[key]["value"] for key in ("net_switch_power", "cell_internal_power", "cell_leakage_power"))
    if not math.isclose(components, metrics["total_power"]["value"], rel_tol=0.002, abs_tol=1e-12):
        raise ValueError("power components do not sum to total within report precision")
    return {"metrics": metrics, "report_git_version": version[1], "parser_version": "2.0.0",
            "timing_met": all(metrics[key + "_worst_slack"]["value"] >= 0 for key in ("setup", "hold")),
            "limitations": ["ideal clock, no placement/routing or extracted parasitics",
                            "vectorless activity, not workload power or energy",
                            "net switching power may be absent in this tool build"]}
