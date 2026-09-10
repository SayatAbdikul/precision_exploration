"""Strict workload validation through the lease-fenced experiment registry.

These jobs select an explicit accumulator manifest. They do not label it an
accepted family-wide Experiment A policy; that decision has separate evidence.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import time

from public.experiments.registry.identity import canonical_json_bytes,experiment_sha256
from public.inference.conformance_job import source_identity
from public.inference.tensor import parse_encoding,SharedEncoding,Tensor
from public.inference.reference.arithmetic import format_named,Accumulator
from public.quantization.calibration.artifact import validate as validate_calibration
from public.quantization.graph.executable import execute,graph_sha256
from public.workloads.datasets.identity import sha256,load_tsv,verify_payload_record

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "phase2-workload-2.0.0"


def artifact_path(root,reference):
    if set(reference) != {"path","sha256"}:
        raise ValueError("invalid artifact reference")
    path = (root/reference["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256(path) != reference["sha256"]:
        raise ValueError("workload artifact path/hash mismatch")
    return path


def validate_job(configuration,*,repository_root=ROOT):
    root = Path(repository_root)
    value = copy.deepcopy(configuration)
    if set(value) != {"schema_version","purpose","model","calibration","formats","evaluation","runtime"}:
        raise ValueError("invalid Phase 2 workload job fields")
    if value["schema_version"] != SCHEMA or value["purpose"] != "strict_workload_validation":
        raise ValueError("unsupported workload execution policy")
    runtime = value["runtime"]
    if (set(runtime) != {"backend","compare_backend","semantic_version","source_sha256"}
            or runtime["semantic_version"] != "2.0.0" or runtime["source_sha256"] != source_identity()
            or runtime["backend"] not in {"reference","cpp","cuda"}
            or runtime["compare_backend"] not in {None,"reference","cpp","cuda"}):
        raise ValueError("workload runtime/source identity mismatch")
    cal = validate_calibration(json.loads(artifact_path(root,value["calibration"]).read_text()),root)
    if cal["context"]["model"] != value["model"]:
        raise ValueError("workload/calibration model mismatch")
    if set(value["formats"]) != {"weight","activation","accumulator"}:
        raise ValueError("weight, activation and accumulator must be independently specified")
    from public.formats.oracle.manifest import manifest_sha256
    accepted = {row["name"] for row in json.loads((root/"public/formats/manifests/accepted/index.json").read_text())["manifests"]}
    for role,reference in value["formats"].items():
        fmt = format_named(reference["name"])
        if reference != {"name":fmt.name,"sha256":manifest_sha256(fmt.manifest)}:
            raise ValueError("workload format manifest mismatch")
        if role == "accumulator":
            if not isinstance(fmt,Accumulator):
                raise ValueError("workload requires a concrete accumulator manifest")
        elif fmt.name not in accepted:
            raise ValueError("workload operand is outside the accepted candidate set")
    if cal["format"] != value["formats"]["activation"]["name"]:
        raise ValueError("activation calibration format mismatch")
    evaluation = value["evaluation"]
    if set(evaluation) != {"name","sha256","selection"} or evaluation["selection"] not in {"all","first_eight_by_sha256_v1"}:
        raise ValueError("unsupported evaluation selection")
    records = json.loads((root/"data/manifests/index.json").read_text())["records"]
    allowed = {"coco_screen_1k","coco_evaluation_5k"} if value["model"] == "yolov8n" else {"imagenet_screen_1k","imagenet_evaluation_10k"}
    if evaluation["name"] not in allowed or records[evaluation["name"]]["sha256"] != evaluation["sha256"]:
        raise ValueError("workload frozen evaluation list mismatch")
    record = records[evaluation["name"]]
    verify_payload_record(root,record)
    training = records[cal["context"]["calibration_list"]["name"]]
    if {r["sha256"] for r in load_tsv(root/record["path"])} & {r["sha256"] for r in load_tsv(root/training["path"])}:
        raise ValueError("workload calibration/evaluation overlap")
    canonical_json_bytes(value)
    return value


def prepare(value,root=ROOT):
    """Regenerate the strict graph from the verified checkpoint and calibration."""
    os.environ.setdefault("YOLO_CONFIG_DIR",str(root/"cache/ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR",str(root/"cache/matplotlib"))
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if torch.__version__.split("+")[0] != "2.3.0":
        raise ValueError("workload runtime requires frozen torch 2.3.0")
    manifest = json.loads((root/f"public/workloads/models/manifests/{value['model']}.json").read_text())
    cal = json.loads(artifact_path(root,value["calibration"]).read_text())
    encodings = {name:parse_encoding(document) for name,document in cal["encodings"].items()}
    provenance = {"kind":"strict_workload_validation","model_manifest_sha256":cal["context"]["model_manifest_sha256"],
                  "source_graph_sha256":manifest["deployment_graph_sha256"],"calibration_sha256":value["calibration"]["sha256"]}
    if value["model"] == "yolov8n":
        import cv2
        import ultralytics
        from ultralytics.data.augment import LetterBox
        from public.quantization.graph.yolo import lower
        if ultralytics.__version__ != manifest["framework_version"]:
            raise ValueError("detector framework version mismatch")
        model = ultralytics.YOLO(root/manifest["checkpoint_path"]).model.eval()
        letterbox = LetterBox(new_shape=(640,640),auto=False,stride=32)
        def sample(path):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"cannot decode image: {path}")
            shape = image.shape[:2]
            image = letterbox(image=image)[:,:,::-1].transpose(2,0,1)
            return torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0)/255,shape
        input_encoding = encodings["images"]
    else:
        import torchvision
        from torchvision import models
        from PIL import Image
        from public.quantization.graph.torchvision import lower
        from public.workloads.models.torchvision_eval import MODEL_SPECS
        if torchvision.__version__.split("+")[0] != manifest["framework_version"]:
            raise ValueError("classifier framework version mismatch")
        _,weights_class,weights_name = MODEL_SPECS[value["model"]]
        transform = getattr(getattr(models,weights_class),weights_name).transforms()
        model = getattr(models,value["model"])(weights=None)
        model.load_state_dict(torch.load(root/manifest["checkpoint_path"],map_location="cpu",weights_only=True))
        model.eval()
        def sample(path):
            with Image.open(path) as image:
                return transform(image.convert("RGB")).unsqueeze(0),None
        placeholders = [node.name for node in torch.fx.symbolic_trace(model).graph.nodes if node.op == "placeholder"]
        if len(placeholders) != 1:
            raise ValueError("workload requires one model input")
        input_encoding = encodings[placeholders[0]]
    graph = lower(model,input_encoding=input_encoding,weight_format=value["formats"]["weight"]["name"],
                  accumulator=value["formats"]["accumulator"]["name"],provenance=provenance,
                  activation_encodings=encodings,weight_calibration_policy=cal["mapping_policy"])
    return graph,input_encoding,sample,model,manifest


def detector_predictions(output,original_shape,image_id):
    import numpy as np
    import torch
    from ultralytics.utils import ops
    tensor = torch.from_numpy(np.asarray([float(v) for v in output.values()],dtype=np.float32).reshape(output.shape))
    if not torch.isfinite(tensor).all():
        raise ValueError("detector produced nonfinite predictions")
    predictions = ops.non_max_suppression(tensor,conf_thres=.001,iou_thres=.7,multi_label=True,max_det=300)[0]
    ops.scale_boxes((640,640),predictions[:,:4],original_shape)
    boxes = ops.xyxy2xywh(predictions[:,:4])
    boxes[:,:2] -= boxes[:,2:]/2
    from ultralytics.data.converter import coco80_to_coco91_class
    categories = coco80_to_coco91_class()
    return [{"image_id":int(image_id),"category_id":categories[int(row[5])],"bbox":[round(float(v),3) for v in box],"score":round(float(row[4]),5)}
            for row,box in zip(predictions,boxes)]


def run_job(configuration,*,repository_root=ROOT):
    root = Path(repository_root)
    value = validate_job(configuration,repository_root=root)
    identity = experiment_sha256(value)
    work = root/"artifacts/workload_runs"/identity
    work.mkdir(parents=True,exist_ok=True)
    (work/"configuration.json").write_bytes(canonical_json_bytes(value))
    graph,encoding,load_image,model,manifest = prepare(value,root)
    graph_identity = graph_sha256(graph)
    (work/"graph.json").write_bytes(canonical_json_bytes(graph))
    record = json.loads((root/"data/manifests/index.json").read_text())["records"][value["evaluation"]["name"]]
    rows = sorted(load_tsv(root/record["path"]),key=lambda row:row["sha256"])
    if value["evaluation"]["selection"] == "first_eight_by_sha256_v1":
        rows = rows[:8]
    payload = root/"data/raw"/record["logical_payload_root"]
    completed = []
    import numpy as np
    import torch
    from public.quantization.ptq.encoding import float_tensor
    for index,row in enumerate(rows,1):
        path = work/f"{row['sha256']}.json"
        if path.exists():
            stored = json.loads(path.read_text())
            check = stored.pop("record_sha256")
            if hashlib.sha256(canonical_json_bytes(stored)).hexdigest() != check or stored["sample"] != row or stored["graph_sha256"] != graph_identity or stored["job_sha256"] != identity:
                raise ValueError("resume record identity mismatch")
            completed.append(stored)
            continue
        fp32,original_shape = load_image(payload/row.get("relative_path",row.get("file_name","")))
        if list(fp32.shape) != manifest["input_shape"]:
            raise ValueError("preprocessed input shape differs from frozen native resolution")
        if isinstance(encoding,SharedEncoding):
            tensor = Tensor.quantize(fp32.numpy().ravel().tolist(),tuple(fp32.shape),encoding)
        else:
            tensor = float_tensor(fp32.numpy(),encoding)
        inputs = {next(iter(graph["inputs"])):tensor}
        backends = {}
        result = None
        for backend in dict.fromkeys(b for b in (value["runtime"]["backend"],value["runtime"]["compare_backend"]) if b):
            start = time.perf_counter()
            actual = execute(graph,inputs,backend=backend)
            seconds = time.perf_counter()-start
            if result is not None and (actual["layers"] != result["layers"] or actual["outputs"] != result["outputs"]):
                mismatch = next((name for name in actual["layers"] if actual["layers"][name] != result["layers"][name]),"outputs")
                raise ValueError(f"first backend mismatch on {row['sha256']}: {mismatch}")
            result = actual
            backends[backend] = {"seconds":seconds,"layers":actual["layers"],"execution_modes":actual["execution_modes"]}
            print(f"{value['model']} image {index}/{len(rows)} {backend}: {len(actual['layers'])} layers, {seconds:.2f}s",flush=True)
        output = next(iter(result["outputs"].values()))
        decoded = output.values()
        if any(not np.isfinite(float(v)) for v in decoded):
            raise ValueError("nonfinite workload outputs")
        with torch.inference_mode():
            baseline = model(fp32)
            if isinstance(baseline,tuple):
                baseline = baseline[0]
            baseline = baseline.cpu().contiguous().numpy()
        if not np.isfinite(baseline).all():
            raise ValueError("FP32 reference produced nonfinite outputs")
        prediction = detector_predictions(output,original_shape,row["image_id"]) if value["model"] == "yolov8n" else sorted(range(len(decoded)),key=lambda c:(-decoded[c],c))[:5]
        stored = {"job_sha256":identity,"graph_sha256":graph_identity,"sample":row,"prediction":prediction,"backends":backends,
                  "output_sha256":hashlib.sha256(canonical_json_bytes(output.document())).hexdigest(),
                  "fp32_output_sha256":hashlib.sha256(baseline.tobytes()).hexdigest(),"fp32_output_shape":list(baseline.shape)}
        stored["record_sha256"] = hashlib.sha256(canonical_json_bytes(stored)).hexdigest()
        temporary = path.with_suffix(".partial")
        temporary.write_bytes(canonical_json_bytes(stored))
        temporary.replace(path)
        completed.append(stored)
    if source_identity() != value["runtime"]["source_sha256"]:
        raise ValueError("engine sources changed during workload execution")
    metrics = {"image_count":(len(completed),"count")}
    if value["runtime"]["compare_backend"]:
        metrics["bit_exact"] = (1,"boolean")
    if value["model"] == "yolov8n":
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        annotations = root/"data/raw/coco2017/annotations/instances_val2017.json"
        frozen = json.loads((root/"public/workloads/datasets/coco2017.json").read_text())
        if sha256(annotations) != frozen["annotation_sha256"]["instances_val2017"]:
            raise ValueError("COCO annotation hash mismatch")
        predictions = [prediction for row in completed for prediction in row["prediction"]]
        (work/"predictions.json").write_text(json.dumps(predictions,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
        truth = COCO(str(annotations))
        if predictions:
            detected = truth.loadRes(predictions)
        else:
            detected = COCO()
            detected.dataset = {**truth.dataset,"annotations":[]}
            detected.createIndex()
        evaluator = COCOeval(truth,detected,"bbox")
        evaluator.params.imgIds = [int(row["sample"]["image_id"]) for row in completed]
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
        metrics.update(map50_95=(float(evaluator.stats[0]),"fraction"),map50=(float(evaluator.stats[1]),"fraction"))
    else:
        metrics.update(top1=(sum(int(row["sample"]["label"]) == row["prediction"][0] for row in completed)/len(completed),"fraction"),
                       top5=(sum(int(row["sample"]["label"]) in row["prediction"] for row in completed)/len(completed),"fraction"))
    (work/"summary.json").write_bytes(canonical_json_bytes({"job_sha256":identity,"graph_sha256":graph_identity,"metrics":metrics,
          "scope":value["evaluation"],"accumulator_policy":"explicit_manifest_validation; family-wide Experiment A acceptance is separate"}))
    return metrics
