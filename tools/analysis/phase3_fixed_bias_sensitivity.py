"""Check fixed bias storage against output thresholds of every current posit MAC."""
from fractions import Fraction
from math import prod

from public.analysis.phase3.fixed_bias_sensitivity import bias_boundary_check
from public.analysis.phase3.fixed_mac_bounds import quantum
from public.inference.conformance_job import source_identity
from public.inference.reference.arithmetic import format_named, real
from public.inference.tensor import parse_encoding
from tools.phase3.common import ROOT, campaign, checked, digest, read, reference, write
from tools.phase3.graphs import configuration
from tools.phase3.preparation_inventory import preparation_records
from tools.phase3.provenance import source_reference


def main():
    plan, source = campaign(), source_identity()
    proof_ref = read(ROOT / "results/summaries/phase3-fixed-mac-bounds.json")["immutable_evidence"]
    proofs = read(checked(proof_ref))
    if proofs["source_sha256"] != source or proofs["campaign_sha256"] != digest(plan):
        raise ValueError("fixed bias checks require current exact MAC proofs")
    inventory, records = preparation_records(), []
    for proof in proofs["records"]:
        prepared = inventory[proof["configuration"]]
        config = read(checked(prepared["configuration"]))
        if (config != configuration(prepared["model"], prepared["format"], plan) or config["runtime"]["source_sha256"] != source
                or proof["graph"] != prepared["graph"]):
            raise ValueError("fixed bias proof refers to an old configuration")
        graph = read(checked(prepared["graph"]))
        nodes, layers = {n["name"]: n for n in graph["nodes"]}, []
        for bound in proof["records"]:
            if bound["status"] != "exact_products_and_sums_after_bias_store":
                raise ValueError("bias boundary check needs the exact product-grid and prefix headroom proof")
            node = nodes[bound["node"]]
            weight = graph["constants"][node["inputs"][1]]
            w, output = parse_encoding(weight["encoding"]), parse_encoding(node["attrs"]["output"])
            if output.axis is not None or output.scales != (Fraction(1),) or w.axis is not None or w.scales != (Fraction(1),):
                raise ValueError("posit bias proof requires intrinsic ordinary unit scales")
            wf = format_named(w.format)
            decoded = {c: Fraction(wf.decode(c)) for c in set(weight["codes"])}
            k = prod(weight["shape"][1:])
            step, x_quantum = Fraction(bound["accumulator_step"]), Fraction(bound["activation_quantum"])
            checks = []
            for channel in bound["channels"]:
                c = channel["channel"]
                values = [decoded[code] for code in set(weight["codes"][c*k:(c+1)*k])]
                q = x_quantum*quantum(values)
                bias = real(node["attrs"]["bias"][c]) if "bias" in node["attrs"] else Fraction(0)
                checks.append({"channel": c, **bias_boundary_check(q, Fraction(channel["maximum_absolute_dot_codes"])*step,
                                                                 bias, step, output.format)})
            layers.append({"node": node["name"], "channels": checks,
                           "status": "exact_output_codes" if all(c["status"] == "exact_output_codes" for c in checks) else "pending"})
        records.append({"configuration": proof["configuration"], "configuration_artifact": prepared["configuration"], "graph": prepared["graph"], "layers": layers})
        print(proof["configuration"], sum(l["status"] == "exact_output_codes" for l in layers), "/", len(layers), "MAC output boundaries covered", flush=True)
    if source_identity() != source:
        raise ValueError("engine changed during fixed bias checks")
    report = {"schema_version": "phase3-fixed-bias-sensitivity-1.0.0", "source_sha256": source, "campaign_sha256": digest(plan),
              "status": "local_bias_boundary_evidence_only", "exact_mac_proofs": proof_ref, "records": records,
              "implementations": [source_reference(ROOT / p) for p in (
                  "tools/analysis/phase3_fixed_bias_sensitivity.py", "public/analysis/phase3/fixed_bias_sensitivity.py",
                  "public/analysis/phase3/fixed_mac_bounds.py")],
              "limits": ["finite stored activations and actual retained weights with previously proved exact prefixes/headroom",
                         "compares the original frozen bias with its prescribed fixed-grid store before identical posit output quantization",
                         "the conservative lattice contains all achievable sums but may contain unreachable ones",
                         "pending lattice discrepancies need witnesses or sensitivity checks, not automatic precision revisions or pruning",
                         "no native image acceptance or network-quality claim"]}
    path = ROOT / "artifacts/phase3/accumulator-probes" / f"fixed-bias-{digest(report)}.json"
    write(path, report)
    write(ROOT / "results/summaries/phase3-fixed-bias-sensitivity.json", {**report, "immutable_evidence": reference(path)})


if __name__ == "__main__":
    main()
