"""Determinism check and storage of weight integers (protocol v1 storage rule; addendum-2 procedure).

    .venv/bin/python -m tools.experiment_b2_axe.codes --case resnet18-int8 --rerun <scratch folder>

For every arm of ``<scratch folder>/<case>`` that has a ``.codes.npz`` (a re-run with ``--save-codes``): the digest of
the stored integers must equal the re-run record's ``weight_codes_sha256`` and the sealed ``evals-v2`` record's, and
the re-run's per-image readouts must equal the sealed ones array for array.  Then the codes files are copied to
``artifacts/experiment_b2_axe/codes-v2/<case>/`` with a sealed manifest (written once; refuses to overwrite).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from tools.experiment_b.common import ROOT, file_hash, seal, unseal

EVALS = ROOT / "artifacts/experiment_b2_axe/evals-v2"
STORE = ROOT / "artifacts/experiment_b2_axe/codes-v2"


def digest_npz(path):
    """``build.codes_digest`` computed from a stored codes file (names sorted, little-endian int16 bytes)."""
    h = hashlib.sha256()
    with np.load(path) as z:
        for name in sorted(z.files):
            h.update(name.encode())
            h.update(np.ascontiguousarray(z[name].astype("<i2")).tobytes())
    return h.hexdigest()


def same_arrays(a, b):
    with np.load(a) as x, np.load(b) as y:
        return sorted(x.files) == sorted(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)


def check(case, rerun):
    rows = []
    for path in sorted((Path(rerun) / case).glob("*.codes.npz")):
        arm = path.name[:-len(".codes.npz")]
        fresh = unseal(path.with_name(f"{arm}.json"))
        sealed_path = EVALS / case / f"{arm}.json"
        sealed = unseal(sealed_path)
        source = sealed.get("identical_to") or arm
        row = {"arm": arm, "codes_digest": digest_npz(path),
               "rerun_weight_codes_sha256": fresh["weight_codes_sha256"],
               "sealed_weight_codes_sha256": sealed["weight_codes_sha256"],
               "sealed_record": str(sealed_path.relative_to(ROOT)), "sealed_record_sha256": file_hash(sealed_path),
               "readouts_identical": same_arrays(path.with_name(f"{arm}.npz"), EVALS / case / f"{source}.npz"),
               "readout_equal": fresh["readout"] == sealed["readout"]}
        row["deterministic"] = (row["codes_digest"] == row["rerun_weight_codes_sha256"] == row["sealed_weight_codes_sha256"]
                                and row["readouts_identical"] and row["readout_equal"])
        rows.append((path, row))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--rerun", required=True)
    parser.add_argument("--store", default="", help="testing only: another destination")
    args = parser.parse_args(argv)
    rows = check(args.case, args.rerun)
    if not rows:
        raise SystemExit("no codes files found")
    bad = [row["arm"] for _, row in rows if not row["deterministic"]]
    print(json.dumps([row for _, row in rows], indent=1))
    if bad:
        raise SystemExit(f"not deterministic: {bad}; nothing copied")
    folder = (Path(args.store) if args.store else STORE) / args.case
    if (folder / "manifest.json").exists():
        raise SystemExit(f"{folder}/manifest.json exists; written once")
    folder.mkdir(parents=True, exist_ok=True)
    for path, row in rows:
        target = folder / path.name
        shutil.copyfile(path, target)
        row["file"] = target.name
        row["file_sha256"] = file_hash(target)
        row["bytes"] = target.stat().st_size
    seal(folder / "manifest.json", {"case": args.case, "protocol": "b2-axe-protocol-v1 storage rule; addendum-2 procedure",
                                    "dtype": "int8 per MAC node, [C, K] in PyTorch weight order (C, Cin*kh*kw)",
                                    "arms": [row for _, row in rows]})


if __name__ == "__main__":
    main()
