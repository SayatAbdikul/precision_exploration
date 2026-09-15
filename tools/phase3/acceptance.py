"""Accept only proven finite integer graphs with eight-image native evidence.

FP64, quire, nonlinear integer domains and other unproved operations remain
pending. A pending proof is an engineering gate, not a candidate rejection.
"""
from math import prod
from fractions import Fraction

from public.inference.conformance_job import source_identity
from public.inference.reference.arithmetic import format_named
from public.quantization.graph.executable import graph_sha256
from tools.phase3.common import ROOT, read, checked, reference, write, digest, campaign
from tools.phase3.graphs import accumulator_bounds
from tools.phase3.baselines import screen_rows
from tools.phase3.evidence import verify_complete


def verify_pilot(prepared, pilot_ref, root=ROOT):
    verify_complete(checked(pilot_ref, root), root, current_execution=True)
    config = read(checked(prepared["configuration"], root))
    graph = read(checked(prepared["graph"], root))
    pilot = read(checked(pilot_ref, root))
    if (pilot["status"] != "completed" or pilot["scope"] != "pilot" or pilot["images"] != 8
            or pilot["configuration_sha256"] != digest(config) or pilot["source_sha256"] != source_identity()
            or not pilot["backend_equality"] or not pilot["diagnostics_complete"]):
        raise ValueError("eight current-source native images with diagnostics are required")
    rows, _, _ = screen_rows(config["model"], campaign(root), root, verify_images=False)
    if len(pilot["image_records"]) != 8:
        raise ValueError("pilot record count mismatch")
    expected_layers = {node["name"] for node in graph["nodes"]}
    graph_identity = graph_sha256(graph)
    records = []
    for image_ref, expected in zip(pilot["image_records"], rows[:8]):
        document = read(checked(image_ref, root))
        if digest({k: v for k, v in document.items() if k != "record_sha256"}) != document["record_sha256"]:
            raise ValueError("pilot image record content hash mismatch")
        if (document["sample"] != expected or document["job_sha256"] != pilot["job_sha256"]
                or document["graph_sha256"] != graph_identity):
            raise ValueError("pilot image identity mismatch")
        a, b = document["backends"]["cpp"], document["backends"]["cuda"]
        if a["layers"] != b["layers"] or a["output_sha256"] != b["output_sha256"] or set(a["layers"]) != expected_layers:
            raise ValueError("pilot native conformance is incomplete")
        if set(a["diagnostics"]) != expected_layers or set(b["diagnostics"]) != expected_layers:
            raise ValueError("pilot diagnostics are incomplete")
        records.append(document)
    return graph, config, pilot, records


def prove_integer_graph(graph, input_shapes, layer_shapes):
    bounds = accumulator_bounds(graph)
    pending = []
    if not bounds["reductions"] or any(row["status"] != "exact_mac_bound" for row in bounds["reductions"]):
        pending.append("not every MAC has an exact mapped integer headroom proof")
    shapes = {**input_shapes, **layer_shapes}
    encodings = dict(graph["inputs"])
    rows = []
    for node in graph["nodes"]:
        attrs, operation = node["attrs"], node["op"]
        source = node["inputs"][0]
        encoding = encodings[source]
        encodings[node["name"]] = attrs.get("output", encoding)
        acc = format_named(attrs["accumulator"]) if "accumulator" in attrs else None
        if acc is not None and acc.family != "integer":
            pending.append(f"{node['name']}: non-integer accumulator needs a separate proof")
        if operation in {"linear", "conv2d", "depthwise_conv2d"}:
            continue
        if operation == "elementwise" and attrs.get("operation") == "add":
            fmt = format_named(attrs["alignment"]["format"])
            if fmt.family != "integer":
                pending.append(f"{node['name']}: residual alignment is not an integer grid")
                continue
            count = 2
        elif operation == "adaptive_average_pool2d" or operation == "pool2d" and attrs["kind"] == "average":
            if source not in shapes:
                raise ValueError("pool proof needs the observed native input shape")
            fmt = format_named(encoding["format"])
            if fmt.family != "integer" or encoding.get("axis") is not None:
                pending.append(f"{node['name']}: pool input is not a scalar integer grid")
                continue
            if operation == "adaptive_average_pool2d":
                count = prod(shapes[source][-2:])
            else:
                kernel = attrs["kernel_size"]
                count = kernel*kernel if type(kernel) is int else prod(kernel)
        elif (operation in {"flatten", "reshape"} or operation == "pool2d" and attrs["kind"] == "max"
              or operation == "activation" and attrs["function"] in {"identity", "relu", "relu6"}):
            continue
        else:
            pending.append(f"{node['name']}: {operation} is outside the proven integer operator set")
            continue
        maximum = max(abs(Fraction(fmt.decode(code))) for code in range(1 << fmt.bits))
        bound = count*maximum
        if acc is None or acc.family != "integer" or bound > (1 << (acc.bits-1))-1:
            pending.append(f"{node['name']}: non-MAC accumulation has no sufficient integer headroom proof")
        rows.append({"node": node["name"], "summands": count, "maximum_abs_sum_codes": str(bound)})
    return {"status": "pending" if pending else "accepted", "reasons": pending,
            "graph_sha256": graph_sha256(graph), "mac_bounds": bounds["reductions"], "other_sum_bounds": rows,
            "scope": "finite integer codes; exact sums within each declared integer accumulator, prescribed product-scale bias RNE and exact mean division before output store"}


def accept(prepared, pilot_ref, root=ROOT):
    graph, config, pilot, records = verify_pilot(prepared, pilot_ref, root)
    manifest = read(checked(campaign(root)["inputs"][config["model"]], root))
    input_shapes = {name: manifest["input_shape"] for name in graph["inputs"]}
    shapes = {name: row["shape"] for name, row in records[0]["backends"]["cpp"]["layers"].items()}
    for record in records[1:]:
        if {name: row["shape"] for name, row in record["backends"]["cpp"]["layers"].items()} != shapes:
            raise ValueError("native graph shapes vary across acceptance images")
    proof = prove_integer_graph(graph, input_shapes, shapes)
    proof.update(source_sha256=source_identity(), configuration_sha256=digest(config), pilot=pilot_ref,
                 proof_implementation=reference(__file__, root))
    directory = root / "artifacts/phase3/acceptance" / digest(config)
    write(directory / "proof.json", proof)
    if proof["status"] != "accepted":
        return proof
    acceptance = {"status": "accepted", "source_sha256": source_identity(), "configuration_sha256": digest(config),
                  "proof": reference(directory / "proof.json", root), "pilot": pilot_ref}
    write(directory / "acceptance.json", acceptance)
    # Do not mutate the prepared record referenced by the completed pilot.
    approved = {**prepared, "screen_acceptance": reference(directory / "acceptance.json", root)}
    write(directory / "prepared-accepted.json", approved)
    return {**acceptance, "prepared": reference(directory / "prepared-accepted.json", root)}


def verify_acceptance(prepared, root=ROOT):
    acceptance_ref = prepared.get("screen_acceptance")
    if acceptance_ref is None:
        raise ValueError("full screen blocked: accumulator, native pilot and diagnostics acceptance missing")
    acceptance = read(checked(acceptance_ref, root))
    if (acceptance.get("status") != "accepted" or acceptance.get("source_sha256") != source_identity()
            or acceptance.get("configuration_sha256") != prepared["configuration_sha256"]):
        raise ValueError("full screen acceptance is stale or belongs to another configuration")
    graph, config, pilot, records = verify_pilot(prepared, acceptance["pilot"], root)
    proof = read(checked(acceptance["proof"], root))
    checked(proof["proof_implementation"], root)
    if (proof["proof_implementation"] != reference(__file__, root) or proof["pilot"] != acceptance["pilot"]
            or proof["source_sha256"] != source_identity() or proof["configuration_sha256"] != digest(config)):
        raise ValueError("accumulator proof provenance is stale or mismatched")
    manifest = read(checked(campaign(root)["inputs"][config["model"]], root))
    shapes = {name: row["shape"] for name, row in records[0]["backends"]["cpp"]["layers"].items()}
    for record in records[1:]:
        if {name: row["shape"] for name, row in record["backends"]["cpp"]["layers"].items()} != shapes:
            raise ValueError("native graph shapes vary across acceptance images")
    recomputed = prove_integer_graph(graph, {name: manifest["input_shape"] for name in graph["inputs"]}, shapes)
    if recomputed["status"] != "accepted" or any(proof.get(key) != value for key, value in recomputed.items()):
        raise ValueError("retained accumulator proof does not reproduce")
    return acceptance
