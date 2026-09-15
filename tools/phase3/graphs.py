"""Prepared graph identities and conservative per-node accumulator bounds."""
from fractions import Fraction
from math import prod

from public.formats.oracle.manifest import manifest_sha256
from public.inference.conformance_job import source_identity
from public.inference.reference.arithmetic import format_named, real, round_integer
from public.inference.tensor import parse_encoding, SharedEncoding
from public.inference.workload_job import prepare, SCHEMA
from public.quantization.graph.executable import graph_sha256
from tools.phase3.common import ROOT, read, write, checked, reference, digest


_AUTO_RESOLUTION = object()


def configuration(model, format_name, plan, root=ROOT, *, accumulator_resolution=_AUTO_RESOLUTION):
    if model not in plan["models"] or format_name not in plan["formats"]:
        raise ValueError("configuration outside frozen Phase 3 matrix")
    inventory = read(root / f"artifacts/phase3/calibration/{model}.json")
    if inventory["campaign_sha256"] != digest(plan):
        raise ValueError("calibration inventory belongs to a different campaign")
    calibration = next(row["calibration"] for row in inventory["records"] if row["format"] == format_name)
    checked(calibration, root)
    policies = read(checked(plan["inputs"]["accumulator_candidates"], root))["records"]
    policy = next(row for row in policies if row["format"] == format_name)
    if accumulator_resolution is _AUTO_RESOLUTION:
        index = root / "public/experiments/configs/experiment_a/phase3-accumulator-resolutions-v1.json"
        accumulator_resolution = read(index)["resolutions"].get(f"{model}/{format_name}") if index.exists() else None
    accumulator_name = policy["candidate_accumulator"]
    if accumulator_resolution is not None:
        resolution = read(checked(accumulator_resolution, root))
        if (resolution["campaign_sha256"] != digest(plan) or resolution["model"] != model
                or resolution["format"] != format_name or resolution["replaces"] != accumulator_name):
            raise ValueError("accumulator resolution identity mismatch")
        for item in resolution["evidence"]:
            checked(item, root)
        accumulator_name = resolution["accumulator"]
    names = {"weight": format_name, "activation": format_name, "output": format_name,
             "accumulator": accumulator_name}
    dataset_name = "coco_screen_1k" if model == "yolov8n" else "imagenet_screen_1k"
    dataset = read(checked(plan["inputs"]["datasets"], root))["records"][dataset_name]
    result = {"schema_version": "phase3-configuration-1.0.0", "campaign_sha256": digest(plan), "model": model,
            "calibration": calibration, "formats": {role: {"name": name, "sha256": manifest_sha256(format_named(name).manifest)}
                                                       for role, name in names.items()},
            "evaluation": {"name": dataset_name, "sha256": dataset["sha256"], "selection": "all"},
            "runtime": {"backend": "cuda", "compare_backend": None, "semantic_version": "2.0.0", "source_sha256": source_identity()}}
    if accumulator_resolution is not None:
        result["accumulator_resolution"] = accumulator_resolution
    return result


def prepare_configuration(config, root=ROOT):
    # Reuse the Phase 2 preparation path, but retain a distinct campaign/job ID.
    workload = {key: value for key, value in config.items() if key not in {"campaign_sha256", "accumulator_resolution"}}
    workload.update(schema_version=SCHEMA, purpose="strict_workload_validation")
    return prepare(workload, root)


def accumulator_bounds(graph):
    """Static checks supplement, never replace, native sensitivity evidence."""
    encodings = {name: parse_encoding(value) for name, value in graph["inputs"].items()}
    records, unresolved = [], []
    for node in graph["nodes"]:
        attrs = node["attrs"]
        output = parse_encoding(attrs["output"]) if "output" in attrs else encodings[node["inputs"][0]]
        encodings[node["name"]] = output
        if node["op"] not in {"linear", "conv2d", "depthwise_conv2d", "block_conv2d"}:
            if "accumulator" in attrs:
                unresolved.append({"node": node["name"], "reason": "non-MAC accumulator sensitivity requires validation"})
            continue
        weight = graph["constants"][node["inputs"][1]]
        k = prod(weight["shape"][1:])
        acc = format_named(attrs["accumulator"])
        row = {"node": node["name"], "k": k, "accumulator": acc.name, "status": "requires_sensitivity"}
        x, w = encodings[node["inputs"][0]], parse_encoding(weight["encoding"])
        if acc.family == "integer":
            if isinstance(x, SharedEncoding) or isinstance(w, SharedEncoding) or x.axis is not None or w.axis not in {None, 0}:
                raise ValueError("integer accumulator has unsupported scale addressing")
            xf, wf = format_named(x.format), format_named(w.format)
            if xf.family != "integer" or wf.family != "integer":
                raise ValueError("integer MAC bounds require integer operands")
            maximum_x = max(abs(Fraction(xf.decode(c))) for c in range(1 << xf.bits))
            maximum_w = max(abs(Fraction(wf.decode(c))) for c in range(1 << wf.bits))
            dot_bound = k * maximum_x * maximum_w
            biases = attrs.get("bias", ["0"] * weight["shape"][0])
            if len(biases) != weight["shape"][0]:
                raise ValueError("invalid bias channel count")
            bias_bound = max(abs(round_integer(real(bias)/(x.scales[0]*w.scales[channel if w.axis == 0 else 0])))
                             for channel, bias in enumerate(biases))
            headroom = (1 << (acc.bits-1))-1-dot_bound-bias_bound
            row.update(dot_bound=str(dot_bound), maximum_stored_bias_codes=bias_bound, remaining_headroom=str(headroom),
                       status="exact_mac_bound" if headroom >= 0 else "insufficient_headroom")
        else:
            row["reason"] = "decoded scale/product range and output-boundary sensitivity need acceptance"
        records.append(row)
    return {"graph_sha256": graph_sha256(graph), "status": "rejected" if any(row["status"] == "insufficient_headroom" for row in records)
            else "pending_native_sensitivity", "reductions": records, "non_mac_checks": unresolved,
            "accepted": False}


def build(model, format_name, plan, root=ROOT):
    config = configuration(model, format_name, plan, root)
    identity = digest(config)
    work = root / "artifacts/phase3/configurations" / identity
    if (work / "prepared.json").exists():
        saved = read(work / "prepared.json")
        for key in ("configuration", "graph", "accumulator_bounds"):
            checked(saved[key], root)
        return saved
    graph, *_ = prepare_configuration(config, root)
    if config["runtime"]["source_sha256"] != source_identity():
        raise ValueError("engine source changed during graph preparation; rerun to prepare a current configuration")
    write(work / "configuration.json", config)
    write(work / "graph.json", graph)
    write(work / "accumulator-bounds.json", accumulator_bounds(graph))
    saved = {"configuration_sha256": identity, "model": model, "format": format_name,
             "configuration": reference(work / "configuration.json", root), "graph": reference(work / "graph.json", root),
             "accumulator_bounds": reference(work / "accumulator-bounds.json", root),
             "status": "prepared_pending_accumulator_and_native_gates"}
    write(work / "prepared.json", saved)
    return saved
