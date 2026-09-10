"""Content-addressed PTQ evidence bound to frozen models and training subsets."""
import hashlib
import json
from pathlib import Path

from public.experiments.registry.identity import canonical_json_bytes,resolve_model
from public.workloads.datasets.identity import sha256,load_tsv,verify_payload_record
from public.inference.tensor import Encoding,SharedEncoding,parse_encoding
from public.inference.reference.arithmetic import format_named
from public.quantization.calibration.numpy_mse import mse_scale_numpy,POLICY
from public.quantization.calibration.observer import SAMPLING

SCHEMA = "phase2-calibration-2.0.0"


def frozen_context(root,model_name,*,verify_payloads=True):
    root = Path(root)
    model = resolve_model(model_name,root)
    records = json.loads((root/"data/manifests/index.json").read_text())["records"]
    prefix = "coco" if model_name == "yolov8n" else "imagenet"
    cal_name,eval_name = prefix+"_calibration_2k",prefix+"_screen_1k"
    cal,ev = records[cal_name],records[eval_name]
    lists = []
    for record in (cal,ev):
        if sha256(root/record["path"]) != record["sha256"]:
            raise ValueError("frozen calibration/evaluation list mismatch")
        rows = load_tsv(root/record["path"])
        if len(rows) != record["count"]:
            raise ValueError("frozen sample count mismatch")
        lists.append({row["sha256"] for row in rows})
    if lists[0]&lists[1]:
        raise ValueError("calibration/evaluation content overlap")
    if verify_payloads:
        verify_payload_record(root,cal)
    return {"model":model_name,"model_manifest_sha256":sha256(root/f"public/workloads/models/manifests/{model_name}.json"),
            "checkpoint_sha256":model["checkpoint_sha256"],"source_graph_sha256":model["deployment_graph_sha256"],
            "preprocessing_sha256":model["preprocessing_sha256"],"calibration_list":{ "name":cal_name,"sha256":cal["sha256"],"count":cal["count"]},
            "evaluation_list_sha256":ev["sha256"]}


def create(context,arrays,observations,format_name):
    if observations["image_count"] != context["calibration_list"]["count"] or observations["sampling"] != SAMPLING:
        raise ValueError("calibration must observe every frozen training sample")
    if set(arrays) != set(observations["nodes"]):
        raise ValueError("observation node coverage mismatch")
    fmt = format_named(format_name)
    encodings,scores = {},{}
    for name,values in sorted(arrays.items()):
        if hashlib.sha256(values.astype("<f4").tobytes()).hexdigest() != observations["nodes"][name]["sample_sha256"]:
            raise ValueError("calibration observation hash mismatch")
        mode = fmt.manifest["scaling"]["mode"]
        if mode == "required_mapping":
            scores[name] = mse_scale_numpy(values,format_name)
            encoding = Encoding(format_name,(scores[name]["scale"],))
        elif mode == "intrinsic_shared":
            encoding = SharedEncoding(format_name,axis=1)
        elif mode == "none":
            encoding = Encoding(format_name)
        else:
            raise ValueError("Experiment A forbids optional external scaling")
        encodings[name] = encoding.document()
    return {"schema_version":SCHEMA,"context":context,"observations":observations,"format":format_name,
            "mapping_policy":POLICY,"encodings":encodings,"scores":scores}


def validate(document,root,*,verify_payloads=True):
    if set(document) != {"schema_version","context","observations","format","mapping_policy","encodings","scores"} or document["schema_version"] != SCHEMA:
        raise ValueError("invalid calibration artifact schema")
    if document["mapping_policy"] != POLICY:
        raise ValueError("unsupported frozen mapping calibration policy")
    if frozen_context(root,document["context"]["model"],verify_payloads=verify_payloads) != document["context"]:
        raise ValueError("stale calibration/model/data identity")
    observation = document["observations"]
    if observation["image_count"] != document["context"]["calibration_list"]["count"] or observation["sampling"] != SAMPLING:
        raise ValueError("incomplete calibration population")
    if not document["encodings"] or set(document["encodings"]) != set(observation["nodes"]):
        raise ValueError("calibration encoding coverage mismatch")
    mapped = format_named(document["format"]).manifest["scaling"]["mode"] == "required_mapping"
    if set(document["scores"]) != (set(document["encodings"]) if mapped else set()):
        raise ValueError("calibration MSE score coverage mismatch")
    for name,value in document["encodings"].items():
        encoding = parse_encoding(value)
        if encoding.format != document["format"]:
            raise ValueError("calibration format mismatch")
        node = observation["nodes"][name]
        if (type(node["sample_count"]) is not int or type(node["observed_elements"]) is not int
                or not observation["image_count"] <= node["sample_count"] <= 16*observation["image_count"]
                or node["observed_elements"] < node["sample_count"]):
            raise ValueError("invalid calibration observation counts")
        if mapped:
            score = document["scores"][name]
            if (encoding.axis is not None or score["policy"] != POLICY or score["candidate_count"] not in {1,150}
                    or Encoding(document["format"],(score["scale"],)) != encoding or score["mse"] < 0):
                raise ValueError("calibration mapping encoding/MSE mismatch")
    canonical_json_bytes(document)
    return document


def artifact_sha256(document):
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()
