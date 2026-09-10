"""Conservative accumulator candidates with explicit finite-domain bounds.

This evidence does not turn a width-only check into a family-wide acceptance.
Mapped integer bias and observed output sensitivity remain per-graph gates.
"""
import json
from fractions import Fraction
from pathlib import Path

from public.inference.reference.arithmetic import format_named,pow2
from public.formats.oracle.manifest import manifest_sha256
from tools.analysis.phase2_accumulator_bounds import bounds

ROOT = Path(__file__).resolve().parents[2]


def main():
    accepted = json.loads((ROOT/"public/formats/manifests/accepted/index.json").read_text())
    records = []
    k = 4608
    for row in accepted["manifests"]:
        fmt = format_named(row["name"])
        grid = bounds(fmt.name,k)
        mode = fmt.manifest["scaling"]["mode"]
        if fmt.family == "integer":
            name = "int32_accumulator"
            raw_bound = Fraction(grid["maximum_sum_magnitude"])
            record = {"reduction_domain":"raw integer operand codes; bias scale is sx*sw[channel]",
                      "finite_dot_exact":raw_bound <= (1<<31)-1,"raw_dot_bound":str(raw_bound),
                      "remaining_bias_headroom":str((1<<31)-1-raw_bound),
                      "required_runtime_check":"abs(round(bias/(sx*sw[channel]))) plus the dot bound must fit signed INT32"}
        elif fmt.family == "posit":
            name = "posit8_es1_quire64_accumulator"
            raw_bound = Fraction(grid["maximum_sum_magnitude"])
            maximum = ((1<<63)-1)*pow2(-24)
            quantum = Fraction(grid["product_grid_quantum"])
            record = {"reduction_domain":"finite unscaled posit products, fixed 64-bit accumulator with 24 fractional bits",
                      "finite_dot_exact":raw_bound <= maximum and (quantum/pow2(-24)).denominator==1,
                      "maximum_stored_magnitude":str(maximum),"remaining_bias_headroom":str(maximum-raw_bound),
                      "maximum_bias_rounding_error":str(pow2(-25)),
                      "required_runtime_check":"finite inputs and stored bias within remaining headroom; NaR is an explicit failed execution, not a standard quire value"}
        else:
            name = "fp64_e11m52_accumulator"
            u = pow2(-53)
            gamma = k*u/(1-k*u)
            maximum = Fraction(grid["maximum_sum_magnitude"])
            if mode == "intrinsic_shared":
                maximum *= pow2(254)
            record = {"reduction_domain":"decoded real operand values; sequential exact-product/FP64-round Model C",
                      "finite_unscaled_dot_exact":grid.get("fp64_covers_finite_unscaled_product_grid",False),
                      "rounding_error_bound":"gamma_K * sum(abs(exact products)), excluding underflow and separately stored bias",
                      "gamma_K":str(gamma),"gamma_K_float":float(gamma),
                      "maximum_dot_magnitude":str(maximum) if mode != "required_mapping" else "depends on the two frozen mapping scales",
                      "required_runtime_check":"per-graph mapped scale range, finite outputs, stored-bias rounding and output-boundary sensitivity"}
            if mode == "intrinsic_shared":
                record["shared_scope"] = "independent E8M0 scales from 2^-127 through 2^127 across all K blocks; FP64 covers exponent range, not every exact sum bit"
        acc = format_named(name)
        records.append({"format":fmt.name,"format_sha256":row["sha256"],"candidate_accumulator":name,
                        "accumulator_sha256":manifest_sha256(acc.manifest),"k_bound":k,**record})
    output = {"schema_version":"2.0.0","status":"candidate_policies_with_bounds; per-graph acceptance still open",
              "scope":"homogeneous W=A, K<=4608; mixed families require an explicit separate check",
              "records":records,"remaining":["mapped integer bias headroom checks on all core workload graphs",
              "wide-accumulator output sensitivity on frozen native-resolution images",
              "NaR-capable standard posit/quire policy if nonfinite posit execution is required"]}
    (ROOT/"results/summaries/phase2-wide-policy-candidates.json").write_text(json.dumps(output,indent=2)+"\n")


if __name__ == "__main__":
    main()
