"""Measure certified native strategies for every accepted operand format."""
import argparse
import hashlib
import json
import random
import statistics
import time
from fractions import Fraction
from pathlib import Path

from public.inference.conformance_job import source_identity
from public.inference.native import NativeBackend
from public.inference.reference.arithmetic import format_named,real,model_c,pow2,binary
from public.experiments.registry.identity import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends",nargs="+",default=["cpp","cuda"])
    parser.add_argument("--repeats",type=int,default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    source = source_identity()
    backends = {name:NativeBackend(name) for name in args.backends}
    accepted = json.loads((ROOT/"public/formats/manifests/accepted/index.json").read_text())
    records = []
    for spec in accepted["manifests"]:
        fmt = format_named(spec["name"])
        choices = [real(fmt.decode(c)) for c in range(1<<fmt.bits) if fmt.decode(c).is_finite()]
        mode = fmt.manifest["scaling"]["mode"]
        accumulator = "int32_accumulator" if fmt.family == "integer" else "posit8_es1_quire64_accumulator" if fmt.family == "posit" else "fp64_e11m52_accumulator"
        acc = format_named(accumulator)
        rng = random.Random(2718)
        batch,channels,k = 4,8,128
        def operands(count,weight=False):
            values = []
            for i in range(count):
                scale = pow2((-127,0,127,5)[(i%k)//32]) if mode == "intrinsic_shared" else Fraction(73 if weight else 37,100) if mode == "required_mapping" and fmt.family != "integer" else Fraction(1)
                values.append(binary(real(choices[rng.randrange(len(choices))]),scale,"mul"))
            return values
        left,right = operands(batch*k),operands(channels*k,True)
        expected = tuple(model_c(zip(left[n*k:(n+1)*k],right[c*k:(c+1)*k]),acc) for n in range(batch) for c in range(channels))
        measured = {}
        for name,backend in backends.items():
            durations = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                actual = backend.flex_gemm(left,right,batch=batch,channels=channels,k=k,accumulator=accumulator)
                durations.append(time.perf_counter()-start)
                if actual != expected:
                    raise ValueError(f"reference mismatch for {fmt.name}/{name}")
            median = statistics.median(durations)
            measured[name] = {"strategy":backend.last_strategy,"median_seconds":median,"samples_seconds":durations,
                              "mac_per_second":batch*channels*k/median}
        records.append({"format":fmt.name,"manifest_sha256":spec["sha256"],"batch":batch,"channels":channels,"k":k,
                        "accumulator":accumulator,"scale_policy":"full E8M0 extreme blocks" if mode == "intrinsic_shared" else "INT raw accumulator units" if fmt.family=="integer" else "fixed independent 0.37/0.73 mapping scales" if mode=="required_mapping" else "unscaled",
                        "output_sha256":hashlib.sha256(canonical_json_bytes({"codes":list(expected)})).hexdigest(),"backends":measured})
        print(f"{fmt.name}: reference matched on {', '.join(measured)}",flush=True)
    if source_identity() != source:
        raise ValueError("sources changed during family benchmark")
    report = {"schema_version":"2.0.0","source_sha256":source,"candidate_set_sha256":accepted["aggregate_sha256"],"records":records,
              "scope":"small 4x8 K128 strategy/dispatch pilots, packing and transfers included; not network throughput or an accepted accumulator policy"}
    (ROOT/"results/summaries/phase2-family-matrix.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
