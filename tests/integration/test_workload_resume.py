"""Exercise the real workload executor's per-image recovery using tiny tensors.

Frozen external-input validation is tested separately; this fixture replaces
checkpoint loading so interruption/recovery can run without licensed images.
"""
import copy
import hashlib
import json
from functools import partial

import pytest

from public.experiments.registry import ExperimentRegistry
from public.experiments.scheduler import LocalScheduler
from public.inference import workload_job
from public.inference.conformance_job import source_identity
from public.inference.tensor import Tensor,Encoding
from public.quantization.graph.executable import freeze_graph


def test_real_workload_failure_resume_dedup_and_corrupt_record(tmp_path,monkeypatch):
    torch = pytest.importorskip("torch")
    encoding = Encoding("fp6_e3m2")
    graph = freeze_graph(inputs={"x":encoding.document()},constants={"w":Tensor.quantize([1,0,0,1],(2,2),encoding)},
        nodes=[{"name":"fc","op":"linear","inputs":["x","w"],"attrs":{"accumulator":"fp32_e8m23_accumulator","output":encoding.document()}}],
        outputs=["fc"],provenance={"kind":"synthetic_conformance"})
    folder = tmp_path/"data/manifests"
    folder.mkdir(parents=True)
    rows = [{"relative_path":str(i),"sha256":str(i)*64,"label":"1"} for i in (1,2)]
    (folder/"images.tsv").write_text("relative_path\tsha256\tlabel\n"+"".join(f"{r['relative_path']}\t{r['sha256']}\t{r['label']}\n" for r in rows))
    (folder/"index.json").write_text(json.dumps({"records":{"fixture":{"path":"data/manifests/images.tsv","logical_payload_root":"fixture"}}}))
    fail = [True]
    loaded = []
    def load(path):
        loaded.append(path.name)
        if path.name == "2" and fail[0]:
            raise RuntimeError("injected image decode failure")
        return torch.tensor([[1.,2.]]),None
    monkeypatch.setattr(workload_job,"validate_job",lambda value,**kw:copy.deepcopy(value))
    monkeypatch.setattr(workload_job,"prepare",lambda *a:(graph,encoding,load,torch.nn.Identity(),{"input_shape":[1,2]}))
    config = {"schema_version":"test-only-workload-resume","model":"fixture","evaluation":{"name":"fixture","selection":"all"},
              "runtime":{"source_sha256":source_identity(),"backend":"reference","compare_backend":None}}
    with ExperimentRegistry(tmp_path/"runs.sqlite",validator=lambda value:value) as registry:
        first,_ = registry.submit(config)
        scheduler = LocalScheduler(registry,worker_id="resume-test")
        assert scheduler.run_once(partial(workload_job.run_job,repository_root=tmp_path))
        failed = registry.connection.execute("SELECT status,error FROM runs WHERE run_id=?",(first,)).fetchone()
        assert failed["status"] == "FAILED" and "injected image decode failure" in failed["error"]
        fail[0] = False
        second,_ = registry.submit(config)
        assert second != first
        assert scheduler.run_once(partial(workload_job.run_job,repository_root=tmp_path))
        assert registry.submit(config) == (second,"COMPLETED")
        assert loaded == ["1","2","2"]  # completed first image was reused
        assert registry.connection.execute("SELECT status FROM runs WHERE run_id=?",(first,)).fetchone()[0] == "FAILED"
        path = next((tmp_path/"artifacts/workload_runs").glob("*/"+"1"*64+".json"))
        corrupted = json.loads(path.read_text())
        corrupted["prediction"] = [0]
        path.write_text(json.dumps(corrupted))
        with pytest.raises(ValueError,match="resume record identity"):
            workload_job.run_job(config,repository_root=tmp_path)
