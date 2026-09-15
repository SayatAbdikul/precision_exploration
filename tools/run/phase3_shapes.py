"""Capture FP32 graph shapes on one frozen image per model for hardware priors."""
from tools.phase3.baselines import screen_rows
from tools.phase3.common import ROOT, campaign, checked, digest, file_hash, read, reference, write
from tools.phase3.provenance import source_reference
from tools.phase3.runtime import load_runtime


class Shapes:
    def __init__(self):
        self.shapes = {}

    def record(self, name, tensor):
        shape = list(tensor.shape)
        if name in self.shapes or not shape or any(size <= 0 for size in shape):
            raise ValueError("duplicate or empty FP32 shape observation")
        self.shapes[name] = shape


def main():
    plan = campaign()
    implementations = [source_reference(ROOT / path) for path in (
        "tools/run/phase3_shapes.py", "tools/phase3/runtime.py",
        "public/quantization/calibration/observer.py")]
    report = {"schema_version": "phase3-shapes-1.0.0", "campaign_sha256": digest(plan), "models": {}}
    for model in plan["models"]:
        print(f"observing {model} shapes", flush=True)
        rows, image_root, population = screen_rows(model, plan, verify_images=False)
        row = rows[0]
        image = image_root / row.get("relative_path", row.get("file_name", ""))
        if file_hash(image) != row["sha256"]:
            raise ValueError("shape observation image hash mismatch")
        sample, observe, manifest = load_runtime(model, plan)
        inputs, _ = sample(image)
        if list(inputs.shape) != manifest["input_shape"]:
            raise ValueError("shape observation input differs from frozen manifest")
        shapes = Shapes()
        observe(inputs, shapes)
        result = {"schema_version": "phase3-model-shapes-1.0.0", "model": model,
                  "campaign_sha256": digest(plan), "model_manifest": plan["inputs"][model],
                  "population_sha256": population["sha256"], "sample": row,
                  "implementations": implementations, "shapes": shapes.shapes,
                  "scope": "FP32 observation of fixed-resolution graph shapes; not native numerical acceptance"}
        path = ROOT / "artifacts/phase3/shapes" / f"{model}-{digest(result)}.json"
        write(path, result)
        report["models"][model] = reference(path)
        write(ROOT / "results/summaries/phase3-shapes.json", report)
        print(f"{model}: {len(shapes.shapes)} tensor shapes retained", flush=True)


if __name__ == "__main__":
    main()
