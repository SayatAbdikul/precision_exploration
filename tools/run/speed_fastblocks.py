"""Lane S1 part 2: re-run lane L1 block-format cells with the host-memory bias correction, into lane S1's folder.

  .venv-b/bin/python -m tools.run.speed_fastblocks MODEL FORMAT RECIPE [--path fast|original] [--prepare-only]

Inside this process only: tools.experiment_b2.matrix.MATRIX -> artifacts/speed_v1/fastblocks/<path>/matrix (cell
record, readout and configuration land there, never in artifacts/experiment_b2/matrix/); data.cached_inputs is
called with build=False (the shared B2 input cache is read, never extended); with --path fast,
blocks.bias_correct_blocks is tools.experiment_b2_fastblocks.bias_correct_blocks. The bias-correction call is
timed and its peak GPU memory (torch.cuda.max_memory_allocated, reset before the call) recorded.
--prepare-only stops after the configuration (bias correction included) and compares its identity with L1's cell.
A full cell is compared with L1's record: configuration_sha256, logits_sha256 and every readout array.
Result: artifacts/speed_v1/fastblocks/<path>/<model>--<format>--<recipe>[--prepare].execution.json (written once).
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # as tools/run/experiment_b2_matrix.py, before CUDA starts
import argparse
import functools
import json
import time
import numpy as np
from tools.experiment_b.common import ROOT, unseal, file_hash

SPEED = ROOT / 'artifacts/speed_v1/fastblocks'
L1 = ROOT / 'artifacts/experiment_b2/matrix'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model'); p.add_argument('format'); p.add_argument('recipe')
    p.add_argument('--path', choices=('fast', 'original'), default='fast'); p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    import torch
    import tools.experiment_b2.matrix as M
    import tools.experiment_b2.frozen  # noqa: F401  (registers the frozen recipes, as matrix.main does)
    import tools.experiment_b2.blocks as B
    from tools.experiment_b2 import data
    out = SPEED / args.path
    target = out / f'{args.model}--{args.format}--{args.recipe}{"--prepare" if args.prepare_only else ""}.execution.json'
    if target.exists():
        raise SystemExit(f'{target} exists')
    M.MATRIX = out / 'matrix'
    for sub in ('cells', 'readout', 'configurations'):
        (M.MATRIX / sub).mkdir(parents=True, exist_ok=True)
    data.cached_inputs = functools.partial(data.cached_inputs, build=False)
    implementation = B.bias_correct_blocks
    if args.path == 'fast':
        from tools.experiment_b2_fastblocks import bias_correct_blocks as implementation
    measured = {}

    def timed(*a, **k):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated(); tick = time.monotonic()
        report = implementation(*a, **k)
        torch.cuda.synchronize()
        measured.update(bias_correction_seconds=time.monotonic() - tick, gpu_bytes_before=before,
                        bias_correction_peak_gpu_bytes=torch.cuda.max_memory_allocated())
        return report
    B.bias_correct_blocks = timed
    reference = sorted((L1 / 'cells').glob(f'{args.model}--{args.format}--{args.recipe}--1000--*.json'))
    if len(reference) != 1:
        raise SystemExit(f'expected one L1 cell, found {len(reference)}')
    ref = unseal(reference[0])
    tick = time.monotonic()
    result = {'model': args.model, 'format': args.format, 'recipe': args.recipe, 'path': args.path,
              'execution_path': ('tools.experiment_b2_fastblocks.bias_correct_blocks (per-chunk activations in host memory, '
                                 'one chunk on the GPU)' if args.path == 'fast' else 'tools.experiment_b2.blocks.bias_correct_blocks'),
              'l1_cell': str(reference[0].relative_to(ROOT)), 'l1_cell_sha256': file_hash(reference[0])}
    if args.prepare_only:
        from types import SimpleNamespace
        build = M.build_shared(SimpleNamespace(model=args.model, format=args.format, recipe=args.recipe, device='cuda', images=M.IMAGES))
        identity = build[-1]
        result.update(configuration_sha256=identity, configuration_identical=identity == ref['configuration_sha256'])
    else:
        record = M.run_cell(args.model, args.format, args.recipe, 'cuda')
        mine = np.load(ROOT / record['readout_file']); theirs = np.load(ROOT / ref['readout_file'])
        arrays = sorted(set(mine.files) | set(theirs.files))
        same = [k for k in arrays if k in mine.files and k in theirs.files and np.array_equal(mine[k], theirs[k])]
        result.update(cell=str(M.cell_path(args.model, args.format, args.recipe, record['configuration_sha256']).relative_to(ROOT)),
                      configuration_sha256=record['configuration_sha256'],
                      configuration_identical=record['configuration_sha256'] == ref['configuration_sha256'],
                      logits_sha256_identical=record['logits_sha256'] == ref['logits_sha256'],
                      readout_arrays=len(arrays), readout_arrays_identical=len(same),
                      readout_file_sha256_identical=record['readout_file_sha256'] == ref['readout_file_sha256'],
                      readout_summary_identical=record['readout'] == ref['readout'],
                      cell_cost=record.get('cost'))
    result.update(measured, wall_seconds=time.monotonic() - tick, process_peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                  written=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    target.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k not in ('cell_cost',)}), flush=True)


if __name__ == '__main__':
    main()
