"""Lane Q2 cells: the B2 matrix cell machinery redirected into ``artifacts/experiment_b2_rank/`` (read-only use).

Everything here is a patch of module attributes inside this process only; no existing file is edited.

* ``redirected(part)``: ``matrix.MATRIX`` -> ``artifacts/experiment_b2_rank/<part>``; ``matrix_finite.patched()``
  (non-finite audit SQNR stored as text); this lane's module hashes added to ``own_sources``.
* ``memoized()``: one ``runner.model_setup`` and one ``data.v1_calibration`` per model and process (deterministic
  loads; every caller gets a fresh shallow copy of the setup dict).  The GPU reproduction check runs through it.
* ``unsigned_variant(format)``: the unsigned sibling codebook (``unsigned.py``) at the nodes the default recipe
  proves non-negative; scalar formats through ``codebook.supports_unsigned`` / ``codebook.unsigned_table``, block
  formats through ``blocks_unsigned.prepare_blocks``.  The configuration gets the key ``rank_unsigned_variant``
  (so the identity differs from every signed cell), and no stage-1 record is looked up for it.
* Four recipes that complete the factorial and are not frozen recipes (registered here, in this process only).
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import replace
import json
import time

from tools.experiment_b.common import ROOT, digest, file_hash
from tools.experiment_b2 import codebook, data, matrix, runner
from tools.experiment_b2 import frozen  # noqa: F401  (registers the frozen recipes)
from tools.experiment_b2.matrix_finite import patched as finite_patched
from tools.experiment_b2.recipe import NAMED, register

RANK = ROOT / "artifacts/experiment_b2_rank"
L1_CELLS = ROOT / "artifacts/experiment_b2/matrix/cells"
OWN = ("tools/experiment_b2_rank/cells.py", "tools/experiment_b2_rank/unsigned.py",
       "tools/experiment_b2_rank/blocks_unsigned.py")

NEW_RECIPES = {
    # integer cube corners that no existing named arm covers (U = unsigned, W = weight range, B = bias correction)
    "rank_signed_no_bias_correction": replace(NAMED["default"], unsigned=False, bias_correction="none"),
    "rank_signed_weight_maxabs": replace(NAMED["default"], unsigned=False, weight_range="maxabs"),
    # intrinsic (max-abs activation block) arm of the shared-exponent formats: the two middle corners
    "rank_intrinsic_no_bias_correction": replace(NAMED["cum5_act_maxabs"], bias_correction="none"),
    "rank_intrinsic_weight_maxabs": replace(NAMED["cum5_act_maxabs"], weight_range="maxabs"),
}
for _name, _recipe in NEW_RECIPES.items():
    if any(r == _recipe for k, r in NAMED.items() if k != _name):
        raise RuntimeError(f"{_name} duplicates an existing named recipe")
    register(_name, _recipe)


def own_hashes():
    return {name: file_hash(ROOT / name) for name in OWN}


def l1_cell(model, format_name, recipe_name):
    found = sorted(L1_CELLS.glob(f"{model}--{format_name}--{recipe_name}--1000--*.json"))
    return found[0] if found else None


def own_cell(part, model, format_name, recipe_name, variant="plain"):
    tag = f"{recipe_name}--unsigned" if variant == "unsigned" else recipe_name
    found = sorted((RANK / part / "cells").glob(f"{model}--{format_name}--{tag}--1000--*.json"))
    return found[0] if found else None


@contextmanager
def redirected(part):
    original_root, original_own = matrix.MATRIX, matrix.own_sources
    folder = part if hasattr(part, "mkdir") else RANK / part
    matrix.MATRIX = folder
    with finite_patched():
        inner_own = matrix.own_sources
        matrix.own_sources = lambda: {**inner_own(), **own_hashes()}
        try:
            yield folder
        finally:
            matrix.own_sources = inner_own
    matrix.MATRIX, matrix.own_sources = original_root, original_own


@contextmanager
def memoized():
    setups, calibrations = {}, {}
    original_setup, original_calibration = runner.model_setup, data.v1_calibration

    def model_setup(model, device):
        if (model, device) not in setups:
            setups[(model, device)] = original_setup(model, device)
        return dict(setups[(model, device)])

    def v1_calibration(model):
        if model not in calibrations:
            calibrations[model] = original_calibration(model)
        return calibrations[model]

    original_inputs = data.cached_inputs

    def cached_inputs(*args, **kwargs):
        # Rule of this campaign: the B2 input cache is never built or enlarged (as matrix.main does).
        return original_inputs(*args, **{**kwargs, "build": False})

    runner.model_setup, data.v1_calibration, data.cached_inputs = model_setup, v1_calibration, cached_inputs
    try:
        yield
    finally:
        runner.model_setup, data.v1_calibration, data.cached_inputs = original_setup, original_calibration, original_inputs


def _with_variant(configuration, format_name):
    from .unsigned import describe
    configuration = {**configuration, "rank_unsigned_variant": {"codebook": describe(format_name),
                                                                "sources": own_hashes()}}
    return configuration, digest(configuration)


@contextmanager
def unsigned_variant(format_name):
    from tools.experiment_b2 import blocks
    from . import blocks_unsigned
    from .unsigned import BLOCK, SCALAR, sibling_table
    if format_name not in SCALAR + BLOCK:
        raise ValueError(f"no unsigned variant for {format_name}")
    saved = {"supports": codebook.supports_unsigned, "table": codebook.unsigned_table, "build": runner.build,
             "build_shared": matrix.build_shared, "prepare": blocks.prepare_blocks,
             "stage1": matrix.stage1_record, "cell_path": matrix.cell_path, "occupancy": matrix.shared_occupancy}

    def supports_unsigned(name):
        # scalar formats only: a block format's boundary plan must stay signed (blocks.block_plan refuses unsigned
        # rows); its unsigned activation blocks come from blocks_unsigned.prepare_blocks via plan["nonnegative"]
        return (name == format_name and format_name in SCALAR) or saved["supports"](name)

    def unsigned_table(name):
        return sibling_table(name) if name == format_name else saved["table"](name)

    def build(args, **kwargs):
        setup, recipe, engine, scales, configuration, _ = saved["build"](args, **kwargs)
        configuration, identity = _with_variant(configuration, args.format)
        return setup, recipe, engine, scales, configuration, identity

    def build_shared(args):
        setup, recipe, engine, scales, configuration, _ = saved["build_shared"](args)
        configuration, identity = _with_variant(configuration, args.format)
        return setup, recipe, engine, scales, configuration, identity

    def cell_path(model, name, recipe_name, identity):
        return matrix.MATRIX / "cells" / f"{model}--{name}--{recipe_name}--unsigned--{matrix.IMAGES}--{identity[:12]}.json"

    def shared_occupancy(engine, inputs, device):
        return blocks_unsigned.occupancy(engine, inputs, device, matrix.AUDIT_IMAGES)

    codebook.supports_unsigned, codebook.unsigned_table = supports_unsigned, unsigned_table
    runner.build, matrix.build_shared = build, build_shared
    blocks.prepare_blocks = blocks_unsigned.prepare_blocks
    matrix.stage1_record = lambda *a, **k: None
    matrix.cell_path, matrix.shared_occupancy = cell_path, shared_occupancy
    try:
        yield
    finally:
        codebook.supports_unsigned, codebook.unsigned_table = saved["supports"], saved["table"]
        runner.build, matrix.build_shared = saved["build"], saved["build_shared"]
        blocks.prepare_blocks, matrix.stage1_record = saved["prepare"], saved["stage1"]
        matrix.cell_path, matrix.shared_occupancy = saved["cell_path"], saved["occupancy"]


def run_jobs(model, jobs, device="cuda", part="factorial", reuse_l1=True):
    """Run ``jobs`` = [(format, recipe, variant[, part])] for one model; one JSON line per cell on stdout.

    An item without its own part goes into ``part``.  Every item is redirected into its part's folder (the model
    setup and calibration data are loaded once per process for all parts).
    """
    import torch
    with ExitStack() as stack:
        stack.enter_context(memoized())
        for item in jobs:
            format_name, recipe_name, variant = item[:3]
            item_part = item[3] if len(item) > 3 else part
            tick = time.monotonic()
            line = {"model": model, "format": format_name, "recipe": recipe_name, "variant": variant, "part": item_part}
            existing = None if variant == "unsigned" else (l1_cell(model, format_name, recipe_name) if reuse_l1 else None)
            existing = existing or own_cell(item_part, model, format_name, recipe_name, variant)
            if existing is not None:
                print(json.dumps({**line, "status": "reused", "cell": str(existing.relative_to(ROOT))}), flush=True)
                continue
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with ExitStack() as inner:
                root = inner.enter_context(redirected(item_part))
                if variant == "unsigned":
                    inner.enter_context(unsigned_variant(format_name))
                elif variant != "plain":
                    raise ValueError(variant)
                record = matrix.run_cell(model, format_name, recipe_name, device)
            brief = matrix.brief(record)
            peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
            print(json.dumps({**line, **brief, "peak_mib": peak, "seconds": round(time.monotonic() - tick, 1),
                              "folder": str(root.relative_to(ROOT))}), flush=True)
    return 0


def reproduce(model, format_name, recipe_name, folder, device="cuda", engine="matrix"):
    """Run one L1 cell redirected into ``folder`` and compare it with L1's sealed record (bit for bit).

    ``engine="blocks_unsigned_off"`` builds a shared-exponent cell with this lane's block engine and no unsigned
    node (``blocks_unsigned.prepare_blocks(..., unsigned=False)``) instead of ``blocks.prepare_blocks``.
    """
    import functools
    import numpy as np
    from tools.experiment_b.common import unseal
    from tools.experiment_b2 import blocks, readout
    from . import blocks_unsigned
    reference = unseal(l1_cell(model, format_name, recipe_name))
    with ExitStack() as stack:
        stack.enter_context(memoized())
        stack.enter_context(redirected(folder))
        if engine == "blocks_unsigned_off":
            original = blocks.prepare_blocks
            blocks.prepare_blocks = functools.partial(blocks_unsigned.prepare_blocks, unsigned=False)
            stack.callback(setattr, blocks, "prepare_blocks", original)
        record = matrix.run_cell(model, format_name, recipe_name, device)
    mine, theirs = readout.load(ROOT / record["readout_file"]), readout.load(ROOT / reference["readout_file"])
    arrays = {key: bool(np.array_equal(mine[key], theirs[key])) for key in readout.DTYPES}
    return {"model": model, "format": format_name, "recipe": recipe_name, "engine": engine,
            "l1_configuration_sha256": reference["configuration_sha256"],
            "configuration_sha256": record["configuration_sha256"],
            "identity_equal": record["configuration_sha256"] == reference["configuration_sha256"],
            "logits_sha256_equal": record["logits_sha256"] == reference["logits_sha256"],
            "readout_arrays_equal": arrays, "all_equal": all(arrays.values())
            and record["logits_sha256"] == reference["logits_sha256"]
            and record["configuration_sha256"] == reference["configuration_sha256"]}
