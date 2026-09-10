"""One launch per strategy/family for Nsight Compute; no wall-time benchmark."""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
import random

from public.experiments.registry.identity import canonical_json_bytes
from public.inference.conformance_job import source_identity
from public.inference.native import NativeBackend, prepare_tensor
from public.inference.reference.arithmetic import format_named
from public.inference.tensor import Encoding, Tensor

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/phase2/profile-inputs.json")
    args = parser.parse_args()
    source = source_identity()
    native, rng, records = NativeBackend("cuda"), random.Random(201), []
    for name in ("fp6_e3m2", "int8"):
        fmt = format_named(name)
        finite = [c for c in range(1 << fmt.bits) if fmt.decode(c).is_finite()]
        batch, channels, k = 196, 64, 288
        left = tuple(rng.choice(finite) for _ in range(batch*k))
        right = tuple(rng.choice(finite) for _ in range(channels*k))
        kernel_arguments = dict(batch=batch, channels=channels, k=k,
                    accumulator="int32_accumulator" if name == "int8" else "fp32_e8m23_accumulator")
        expected = None
        for strategy in ("predecoded", "algorithmic", "lookup"):
            if strategy == "predecoded":
                inputs = prepare_tensor(Tensor((batch,k), left, Encoding(name)))
                weights = prepare_tensor(Tensor((channels,k), right, Encoding(name)))
                result = native.gemm(inputs, weights, **kernel_arguments)
            else:
                result = native.encoded_gemm(left, right, activation_format=name, weight_format=name,
                                             strategy=strategy, **kernel_arguments)
            if expected is not None and result != expected:
                raise RuntimeError("profiled strategy changes numerical results")
            expected = result
            records.append({"format": name, "strategy": strategy, "launch_index": len(records), **kernel_arguments,
                            "output_sha256": hashlib.sha256(canonical_json_bytes({"codes": list(result)})).hexdigest()})
    if source_identity() != source:
        raise RuntimeError("engine source changed during profiling")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    temporary.write_text(json.dumps({"source_sha256": source, "launches": records}, indent=2) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
