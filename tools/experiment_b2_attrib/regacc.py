"""Lane Q4 v2.1 correction (agent r7): accumulator fields with MODEL-MATCHED certificates.

    PYTHONPATH=. .venv/bin/python -m tools.experiment_b2_attrib.regacc [--write]

Why: ``summarize2.accumulator_v2`` reads ``summarize2.CERTS``, which holds only the MobileNetV3 certificates, and
``summarize2.build`` applies it to the MobileNetV2 and ResNet18 INT8 regression rows as well. In
``results/summaries/b2-attrib-v1/v2/b_regression.json`` the fields ``network_max_bits_arm_check``,
``network_max_bits_presubtracted``, ``certificate_matches_layer_defaults``, ``presubtracted_increase_layers`` and
``presubtracted_widest_layers`` of those two networks therefore compare against MobileNetV3's layers. The MobileNetV3
rows of ``v2/b.json`` are right. This module recomputes the same fields with each network's own sealed certificate
(scaled-bridge run 1f75c923..., the run the MobileNetV3 rows use) and writes
``results/summaries/b2-attrib-v1/v2.1/accumulator_model_matched.json`` once. It re-derives the MobileNetV3 rows as a
check (they must equal ``v2/b.json``). No measurement, no GPU: it reads the sealed arm records only.
"""
from __future__ import annotations

import argparse
import json

from tools.experiment_b.common import ROOT, file_hash, unseal
from . import analysis, summarize

RUN = ROOT / "artifacts/scaled_bridge_v2/runs/1f75c9232c8a0202482fe9bce9a46360c7cc0999c5b0bcf7df5ebcc0446fc863"
OUT = ROOT / "results/summaries/b2-attrib-v1/v2.1"
V2 = ROOT / "results/summaries/b2-attrib-v1/v2"
CELLS = [("mobilenet_v3_large", "int8"), ("mobilenet_v3_large", "int6"), ("mobilenet_v2", "int8"), ("resnet18", "int8")]
FIELDS = ("network_max_bits_arm_check", "network_max_bits_presubtracted", "certificate_matches_layer_defaults",
          "presubtracted_increase_layers", "presubtracted_widest_layers")


def certificate_path(model, fmt):
    return RUN / f"{model}-{fmt}-default-b2" / "certificate.json"


def certificate_bits(model, fmt):
    path = certificate_path(model, fmt)
    return {name: row["signed_bits_absolute"] for name, row in unseal(path)["certificates"].items()}


def accumulator_matched(meta, cert):
    """Same arithmetic as ``summarize2.accumulator_v2``, with the caller's certificate."""
    acc = summarize.accumulator(meta)
    if acc is None:
        return None
    layers = meta["accumulator"]["layers"]
    unchanged = max((b for name, b in cert.items() if name not in layers), default=0)
    arm = max([unchanged] + [v["bits_arm"] for v in layers.values()])
    pre = max([unchanged] + [v.get("bits_arm_presubtracted_operand", v["bits_arm"]) for v in layers.values()])
    acc.update(network_max_bits_certificate_default=max(cert.values()),
               network_max_bits_arm_check=arm, network_max_bits_presubtracted=pre,
               certificate_matches_layer_defaults=all(cert.get(n) == v["bits_default"] for n, v in layers.items()),
               layers_missing_from_certificate=sorted(n for n in layers if n not in cert),
               presubtracted_increase_layers=sorted(n for n, v in layers.items()
                                                    if v.get("bits_arm_presubtracted_operand", 0) > v["bits_default"]),
               presubtracted_widest_layers=sorted(n for n, v in layers.items()
                                                  if v.get("bits_arm_presubtracted_operand", v["bits_arm"]) == pre),
               zero_points=sorted({v["zero_point"] for v in layers.values() if "zero_point" in v}))
    return acc


def build():
    v2 = {"mobilenet_v3_large": json.loads((V2 / "b.json").read_text()),
          "regression": json.loads((V2 / "b_regression.json").read_text())}
    out = {"cells": {}, "v2_mismatches": [], "certificates": {}}
    for model, fmt in CELLS:
        cert = certificate_bits(model, fmt)
        out["certificates"][f"{model}/{fmt}"] = {"path": str(certificate_path(model, fmt).relative_to(ROOT)),
                                                 "sha256": file_hash(certificate_path(model, fmt)),
                                                 "network_max_bits": max(cert.values()), "layers": len(cert)}
        records = analysis.arm_records(model, fmt)
        rows = {}
        for name, record in sorted(records.items()):
            if record["arm_kind"] != "recipe" and name != "ref_default":
                continue
            acc = accumulator_matched(record["meta"], cert)
            if acc is None:
                continue
            rows[name] = acc
            old = (v2["mobilenet_v3_large"].get(fmt, {}) if model == "mobilenet_v3_large"
                   else v2["regression"].get(model, {})).get(name, {}).get("accumulator") or {}
            diff = {k: [old.get(k), acc.get(k)] for k in FIELDS if old.get(k) != acc.get(k)}
            if diff:
                out["v2_mismatches"].append({"cell": f"{model}/{fmt}", "arm": name, "v2_vs_matched": diff})
        out["cells"][f"{model}/{fmt}"] = rows
    out["note"] = ("v2.1 correction of the accumulator fields of v2/b_regression.json (MobileNetV3 certificate was "
                   "applied to MobileNetV2/ResNet18). Development evidence; no new measurement.")
    out["source"] = {"module": "tools/experiment_b2_attrib/regacc.py",
                     "module_sha256": file_hash(ROOT / "tools/experiment_b2_attrib/regacc.py")}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    out = build()
    if args.write:
        if OUT.exists():
            raise SystemExit(f"{OUT} exists: summaries are written once")
        OUT.mkdir(parents=True)
        (OUT / "accumulator_model_matched.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for cell, rows in out["cells"].items():
        for name, acc in rows.items():
            print(cell, name, {k: acc.get(k) for k in ("network_max_bits_default", "network_max_bits_certificate_default",
                                                       "network_max_bits_arm_check", "network_max_bits_presubtracted",
                                                       "layers_with_increase", "certificate_matches_layer_defaults")})
    print("v2 mismatches:", len(out["v2_mismatches"]),
          sorted({m["cell"] for m in out["v2_mismatches"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
