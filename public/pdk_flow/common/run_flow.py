"""Reproducible Yosys/OpenROAD pilot driver for representative arithmetic RTL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path, environment: dict[str, str], log: Path) -> None:
    result = subprocess.run(command, cwd=cwd, env=environment, text=True, capture_output=True, check=False)
    log.write_text(
        f"command: {shlex.join(command)}\nreturncode: {result.returncode}\n\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(f"flow command failed with exit {result.returncode}; see {log}")


def run(
    *,
    config_path: str | Path,
    work_dir: str | Path,
    liberty: str | Path | None = None,
    tech_lef: str | Path | None = None,
    cells_lef: str | Path | None = None,
    yosys: str = "yosys",
    openroad: str | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    repository_root = Path(__file__).resolve().parents[3]
    rtl = (repository_root / config["rtl"]).resolve()
    rtl.relative_to(repository_root)
    output = Path(work_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    top = config["top"]

    yosys_script = output / "synthesis.ys"
    commands = [
        f"read_verilog -sv {rtl}",
        f"hierarchy -check -top {top}",
        "proc",
        "opt",
        f"synth -top {top}",
    ]
    if liberty is not None:
        liberty_path = Path(liberty).resolve()
        commands.extend(
            [
                f"dfflibmap -liberty {liberty_path}",
                f"abc -liberty {liberty_path}",
                "opt_clean",
            ]
        )
    stat_command = "stat -json"
    if liberty is not None:
        stat_command += f" -liberty {Path(liberty).resolve()}"
    commands.extend(
        [
            f"write_verilog -noattr {output / 'netlist.v'}",
            f"tee -o {output / 'yosys-stat.json'} {stat_command}",
        ]
    )
    yosys_script.write_text("\n".join(commands) + "\n", encoding="utf-8")
    environment = dict(os.environ)
    _run([yosys, "-s", str(yosys_script)], cwd=output, environment=environment, log=output / "yosys.log")

    external_hashes: dict[str, str] = {}
    if liberty is not None:
        external_hashes["liberty_sha256"] = _sha256(Path(liberty))
    if openroad is not None:
        required = {"liberty": liberty, "tech_lef": tech_lef, "cells_lef": cells_lef}
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"OpenROAD run requires: {', '.join(missing)}")
        environment.update(
            {
                "TOP": top,
                "NETLIST": str(output / "netlist.v"),
                "LIBERTY": str(Path(liberty).resolve()),
                "TECH_LEF": str(Path(tech_lef).resolve()),
                "CELLS_LEF": str(Path(cells_lef).resolve()),
                "OUTPUT_DEF": str(output / "placed.def"),
            }
        )
        script = Path(__file__).with_name("openroad_pilot.tcl")
        _run([openroad, str(script)], cwd=output, environment=environment, log=output / "openroad.log")
        external_hashes["tech_lef_sha256"] = _sha256(Path(tech_lef))
        external_hashes["cells_lef_sha256"] = _sha256(Path(cells_lef))

    expected_outputs = {"synthesis.ys", "netlist.v", "yosys-stat.json", "yosys.log"}
    if openroad is not None:
        expected_outputs.update({"placed.def", "openroad.log"})
    manifest = {
        "schema_version": "1.0.0",
        "config": config,
        "config_sha256": _sha256(config_file),
        "rtl_sha256": _sha256(rtl),
        "external_resource_hashes": external_hashes,
        "outputs": {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in sorted(output.iterdir())
            if path.is_file() and path.name in expected_outputs
        },
    }
    (output / "run-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--liberty")
    parser.add_argument("--tech-lef")
    parser.add_argument("--cells-lef")
    parser.add_argument("--yosys", default="yosys")
    parser.add_argument("--openroad")
    args = parser.parse_args()
    result = run(
        config_path=args.config,
        work_dir=args.work_dir,
        liberty=args.liberty,
        tech_lef=args.tech_lef,
        cells_lef=args.cells_lef,
        yosys=args.yosys,
        openroad=args.openroad,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
