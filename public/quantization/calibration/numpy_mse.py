"""Versioned FP64 MSE scale search for static mapping calibration.

This optimizer uses floating MSE, while deployment encodings and inference
use exact scalar boundaries and Model C. It is distinct from mse_scale's
rational scoring policy and its identity must be retained in artifacts.
"""
import numpy as np

from public.inference.tensor import Encoding
from public.inference.reference.arithmetic import format_named
from public.quantization.ptq.encoding import scalar_float_codes

POLICY = "mse_numpy_f64_100coarse_50fine_v1"


def mse_scale_numpy(values,format_name):
    values = np.asarray(values,dtype=np.float64).ravel()
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("calibration requires nonempty finite values")
    fmt = format_named(format_name)
    if fmt.manifest["scaling"]["mode"] != "required_mapping":
        raise ValueError("static mapping optimizer only supports required_mapping formats")
    decoded = np.array([float(fmt.decode(c)) for c in range(1<<fmt.bits)])
    decoded = np.unique(decoded[np.isfinite(decoded)])
    maximum = decoded[-1]
    boundaries = (decoded[:-1]+decoded[1:])/2
    def score(scale):
        positions = np.searchsorted(boundaries,values/scale,side="left")
        return float(np.mean(np.square(values-decoded[positions]*scale)))
    span = float(np.max(np.abs(values)))
    if span == 0:
        return {"scale":"1","mse":0.0,"candidate_count":1,"policy":POLICY}
    step = span/maximum/100
    coarse = np.arange(1,101,dtype=np.float64)*step
    scored = [(score(s),float(s)) for s in coarse]
    best = min(scored)[1]
    lower,upper = max(step/50,best-step),best+step
    fine = lower+(upper-lower)*np.arange(50,dtype=np.float64)/49
    scored.extend((score(s),float(s)) for s in fine)
    error,best = min(scored)
    return {"scale":str(best),"mse":error,"candidate_count":150,"policy":POLICY}
