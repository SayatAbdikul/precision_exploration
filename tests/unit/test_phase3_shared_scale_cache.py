from fractions import Fraction
import json

import pytest

from public.inference.tensor import SharedEncoding, Tensor
from public.quantization.calibration.mse import mse_scale
from tools.phase3.shared_scale_cache import SharedScaleCache


@pytest.mark.parametrize("name", ["bfp6", "mxfp8_e4m3", "mxfp6_e3m2", "mxfp4_e2m1"])
def test_cached_tensor_codes_scales_and_partial_blocks_match_unchanged_oracle(tmp_path, name):
    values = [Fraction(i-3, 16) for i in range(7)]
    policy = SharedEncoding(name, axis=1)
    expected = Tensor.quantize(values, (1, 7), policy)
    cache = SharedScaleCache(tmp_path / "cache.sqlite", source_sha256="source", oracle=mse_scale)
    try:
        with cache.installed():
            assert Tensor.quantize(values, (1, 7), policy).document() == expected.document()
            assert Tensor.quantize(values, (1, 7), policy).document() == expected.document()
        assert cache.stats()["misses"] == 1 and cache.stats()["hits"] == 1
    finally:
        cache.close()


def test_saved_blocks_survive_interruption_and_new_engine_cannot_reuse_them(tmp_path):
    path = tmp_path / "cache.sqlite"
    calls = []
    def interrupted(values, name, **kwargs):
        calls.append(values)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return mse_scale(values, name, **kwargs)
    cache = SharedScaleCache(path, source_sha256="old", oracle=interrupted)
    expected = cache([0, 1], "bfp6")
    with pytest.raises(RuntimeError, match="interruption"):
        cache([0, 2], "bfp6")
    cache.close()
    cache = SharedScaleCache(path, source_sha256="old", oracle=mse_scale)
    assert cache([0, 1], "bfp6") == expected
    cache([0, 2], "bfp6")
    assert cache.stats()["hits"] == cache.stats()["misses"] == 1
    cache.close()
    cache = SharedScaleCache(path, source_sha256="new", oracle=mse_scale)
    cache([0, 1], "bfp6")
    assert cache.stats()["hits"] == 0 and cache.stats()["misses"] == 1
    cache.close()


def test_corrupt_cache_fails_closed_and_zero_tie_is_preserved(tmp_path):
    cache = SharedScaleCache(tmp_path / "cache.sqlite", source_sha256="source", oracle=mse_scale)
    expected = mse_scale([0]*32, "mxfp4_e2m1")
    assert cache([0]*32, "mxfp4_e2m1") == expected
    row = cache.connection.execute("SELECT identity,response FROM scales").fetchone()
    bad = json.loads(row[1]); bad["scale"] = "1"
    cache.connection.execute("UPDATE scales SET response=? WHERE identity=?", (json.dumps(bad), row[0]))
    with pytest.raises(ValueError, match="content mismatch"):
        cache([0]*32, "mxfp4_e2m1")
    with pytest.raises(ValueError):
        cache([0]*32, "mxfp4_e2m1", coarse_candidates=1)
    cache.close()


def test_accelerated_and_original_workers_share_identical_cache_entries(tmp_path):
    path = tmp_path / "cache.sqlite"
    block = [Fraction(i-7, 32) for i in range(16)]
    original = SharedScaleCache(path, source_sha256="source", oracle=mse_scale)
    expected = original(block, "mxfp6_e3m2")
    fast = SharedScaleCache(path, source_sha256="source", oracle=mse_scale, fast_fp32=True)
    assert fast(block, "mxfp6_e3m2") == expected
    other = list(reversed(block))
    with fast.installed():
        result = Tensor.quantize(other, (1, 16), SharedEncoding("mxfp6_e3m2", axis=1))
    assert result == Tensor.quantize(other, (1, 16), SharedEncoding("mxfp6_e3m2", axis=1))
    assert original(other, "mxfp6_e3m2") == mse_scale(other, "mxfp6_e3m2")
    assert original.hits == fast.hits == 1
    original.close(); fast.close()
