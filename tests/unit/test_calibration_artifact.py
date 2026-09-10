import copy
import hashlib
import json
from pathlib import Path
from fractions import Fraction

import numpy as np
import pytest

from public.inference.tensor import Encoding,Tensor
from public.inference.reference.arithmetic import format_named,decode
from public.quantization.ptq.encoding import float_tensor
from public.quantization.calibration import artifact
from public.quantization.calibration.numpy_mse import mse_scale_numpy
from public.quantization.calibration.observer import Observer,SAMPLING

ACCEPTED = json.loads((Path(__file__).resolve().parents[2]/"public/formats/manifests/accepted/index.json").read_text())["manifests"]


@pytest.mark.parametrize("name",[r["name"] for r in ACCEPTED if r["family"] not in {"mx_float","bfp"}])
def test_bulk_checkpoint_encoder_matches_boundary_neighbors(name):
    fmt = format_named(name)
    enc = Encoding(name,("0.37",)) if fmt.manifest["scaling"]["mode"] == "required_mapping" else Encoding(name)
    values = [decode(fmt,c,enc.scales[0]) for c in range(1<<fmt.bits)]
    finite = sorted(set(Fraction(v) for v in values if isinstance(v,Fraction) or v.is_finite()))
    centers = np.array([float((a+b)/2) for a,b in zip(finite,finite[1:])])
    samples = np.concatenate((np.nextafter(centers,-np.inf),centers,np.nextafter(centers,np.inf),[-0.,0.]))
    assert float_tensor(samples,enc) == Tensor.quantize(samples.tolist(),samples.shape,enc)


def test_observer_population_coverage_and_artifact_binding(monkeypatch):
    torch = pytest.importorskip("torch")
    observer = Observer()
    for image in range(2):
        observer.record("x",torch.tensor([[-2.,-1.,0.,1.,2.]]))
        observer.finish_image()
    context = {"model":"fixture","calibration_list":{"count":2}}
    doc = artifact.create(context,observer.arrays(),observer.summary(),"int4")
    monkeypatch.setattr(artifact,"frozen_context",lambda *a,**kw:context)
    assert artifact.validate(doc,Path(".")) == doc
    changed = copy.deepcopy(doc)
    changed["observations"]["image_count"] = 1
    with pytest.raises(ValueError,match="population"):
        artifact.validate(changed,Path("."))
    altered = observer.arrays()
    altered["x"][0] = 12
    with pytest.raises(ValueError,match="hash"):
        artifact.create(context,altered,observer.summary(),"int4")
    observer.record("different",torch.ones(1))
    with pytest.raises(ValueError,match="coverage"):
        observer.finish_image()


def test_numpy_mse_is_explicit_and_deterministic():
    values = [-3,-2,-1,0,1,2,3]
    result = mse_scale_numpy(values,"int4")
    assert result == mse_scale_numpy(values,"int4")
    assert result["candidate_count"] == 150
    assert result["mse"] < .03
    with pytest.raises(ValueError,match="required_mapping"):
        mse_scale_numpy(values,"fp6_e3m2")
