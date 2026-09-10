"""Observe the frozen training population and freeze reusable PTQ encodings."""
import argparse
import json
import os
from pathlib import Path

from public.experiments.registry.identity import canonical_json_bytes
from public.quantization.calibration.artifact import frozen_context,create,artifact_sha256
from public.quantization.calibration.observer import Observer,detector_observe
from public.workloads.datasets.identity import load_tsv

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True,choices=("resnet18","mobilenet_v2","mobilenet_v3_large","yolov8n"))
    parser.add_argument("--formats",nargs="+",default=["fp6_e3m2","int8"])
    parser.add_argument("--observations",type=Path,help="Reuse a verified observations JSON and its adjacent NPZ")
    parser.add_argument("--available-only",action="store_true",help="Cache classifier observations for present frozen images without publishing calibration artifacts")
    args = parser.parse_args()
    if args.available_only and (args.observations or args.model == "yolov8n"):
        parser.error("--available-only requires a classifier and cannot be combined with --observations")
    os.environ.setdefault("YOLO_CONFIG_DIR",str(ROOT/"cache/ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"cache/matplotlib"))
    import numpy as np
    context = frozen_context(ROOT,args.model,verify_payloads=not args.available_only)
    destination = ROOT/"artifacts/calibration"
    destination.mkdir(parents=True,exist_ok=True)
    if args.observations:
        saved = json.loads(args.observations.read_text())
        if saved["context"] != context:
            raise ValueError("observation provenance mismatch")
        observations = saved["observations"]
        with np.load(args.observations.with_suffix(".npz"),allow_pickle=False) as archive:
            arrays = {name:archive[name] for name in archive.files}
    else:
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        observer = Observer()
        manifest = json.loads((ROOT/f"public/workloads/models/manifests/{args.model}.json").read_text())
        records = json.loads((ROOT/"data/manifests/index.json").read_text())["records"]
        record = records[context["calibration_list"]["name"]]
        rows = sorted(load_tsv(ROOT/record["path"]),key=lambda row:row["sha256"])
        payload = ROOT/"data/raw"/record["logical_payload_root"]
        if args.model == "yolov8n":
            import cv2
            from ultralytics import YOLO
            from ultralytics.data.augment import LetterBox
            from public.quantization.graph.yolo import lower
            from public.inference.tensor import Encoding
            model = YOLO(ROOT/manifest["checkpoint_path"]).model.eval()
            constants = {}
            graph = lower(model,input_encoding=Encoding("fp6_e3m2"),weight_format="fp6_e3m2",accumulator="fp32_e8m23_accumulator",
                          provenance={"kind":"fp32_calibration_observation_plan"},constant_observer=lambda name,value:constants.__setitem__(name,value))
            letterbox = LetterBox(new_shape=(640,640),auto=False,stride=32)
            def sample(path):
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"cannot decode calibration image: {path}")
                image = letterbox(image=image)[:,:,::-1].transpose(2,0,1)
                return torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0)/255
            run = lambda value:detector_observe(graph,constants,value,observer)
            with torch.inference_mode():
                for index,row in enumerate(rows,1):
                    run(sample(payload/row.get("relative_path",row.get("file_name",""))))
                    observer.finish_image()
                    if index%100 == 0:
                        print(f"observed {index}/{len(rows)} frozen training images",flush=True)
            observations,arrays = observer.summary(),observer.arrays()
        else:
            from tools.run.phase2_calibration_cache import observe_classifier
            result = observe_classifier(ROOT,context,rows,payload,available_only=args.available_only)
            if args.available_only:
                return
            observations,arrays = result
        saved = {"context":context,"observations":observations}
        identity = artifact_sha256(saved)
        observation_path = destination/f"{args.model}-observations-{identity}.json"
        observation_path.write_bytes(canonical_json_bytes(saved))
        np.savez_compressed(observation_path.with_suffix(".npz"),**arrays)
        print(str(observation_path),flush=True)
    for name in args.formats:
        document = create(context,arrays,observations,name)
        path = destination/f"{args.model}-{name}-{artifact_sha256(document)}.json"
        path.write_bytes(canonical_json_bytes(document))
        print(str(path),flush=True)


if __name__ == "__main__":
    main()
