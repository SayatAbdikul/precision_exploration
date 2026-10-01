"""Prepare the task-owned, reproducible ICS55 ECC INT4 pilot project."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT = ROOT / "artifacts/ppa/ics55-hardware-foundation/integer4-ecc-readonly-runtime"
PDK_ROOT = "/home/maveric/nursultan/texer.ai/ecc/chipcompiler/thirdparty/icsprout55-pdk"

ECC_TOML = '''[design]
name = "integer_mac_harness"
top = "integer_mac_harness"
rtl = ["sources.f"]
clock_port = "clk"
frequency_mhz = 100.0

[pdk]
name = "ics55"
root = "{pdk_root}"

[flow]
preset = "rtl2gds"
run = "default"
'''
RTL = ("public/generic_rtl/mac/integer_mac.sv",
       "public/generic_rtl/mac/integer_mac_harness.sv")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--pdk-root", type=Path, default=Path(PDK_ROOT))
    args = parser.parse_args(argv)
    project = args.project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    (project / "ecc.toml").write_text(ECC_TOML.format(pdk_root=args.pdk_root.resolve()))
    (project / "sources.f").write_text("\n".join(os.path.relpath(ROOT / name, project) for name in RTL) + "\n")
    print(project)


if __name__ == "__main__":
    main()
