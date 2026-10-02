"""Lane S1 part 3: fast_cocoeval against pycocotools (stats.Evaluated) on the stored 1k detector configurations.

  .venv/bin/python -m tools.run.speed_cocoeval_validate FILE_STEM... [--draws N] [--time-reference N] [--big 10000]
FILE_STEM: <id>-1000-<tag> of artifacts/experiment_b2_det/detections/*.npz (the same inputs as runner._bootstrap_job).
Per file: point estimate fast vs pycocotools (and vs the stored bootstrap point); with --draws N the first N stored
bootstrap draws (seed 310911) recomputed fast and compared with the stored array; --time-reference N times N
pycocotools draws; --big R times R fast draws. One JSON file per stem under artifacts/speed_v1/cocoeval/ (kept).
"""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
from tools.experiment_b.common import ROOT, dataset

BASE = ROOT / 'artifacts/experiment_b2_det'
OUT = ROOT / 'artifacts/speed_v1/cocoeval'
SOURCE = ROOT / 'tools/analysis/fast_cocoeval/__init__.py'


def out_dir():
    """Results are keyed by the fast_cocoeval source hash (the first implementation's results stay in OUT)."""
    from tools.experiment_b.common import file_hash
    return OUT / file_hash(SOURCE)[:16]


def evaluated_for(stem, truth):
    from tools.experiment_b2_det.post import records
    from tools.experiment_b2_det.runner import load_detections
    from tools.experiment_b2_det.stats import Evaluated
    from tools.experiment_b2_det.runner import sealed_v1
    identity, images, tag = stem.rsplit('-', 2)
    _, rows, _ = dataset('coco_screen_1k')
    rows = rows[:int(images)]
    if tag == 'sealed':                      # runner._bootstrap_job mode 'sealed': v1--<format>--<recipe>, index order
        _, name, recipe = identity.split('--')
        arrays = sealed_v1(name, recipe, rows)[1]
        tag = 'index'
    else:
        arrays = load_detections(BASE / 'detections' / f'{stem}.npz')
    ids = [int(r['image_id']) for r in rows]
    order = list(range(len(rows)))
    if tag == 'reverse':
        order = order[::-1]
    elif tag.startswith('random'):
        order = [int(i) for i in np.random.default_rng(7919 + int(tag[6:])).permutation(len(rows))]
    return Evaluated(truth, records(arrays, rows), [ids[i] for i in order])


def main():
    from tools.experiment_b2_det.stats import annotations, RESAMPLES, SEED, CONFIDENCE, interval
    from tools.analysis.fast_cocoeval import FastAccumulate
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stems', nargs='+'); p.add_argument('--draws', type=int, default=0)
    p.add_argument('--time-reference', type=int, default=0); p.add_argument('--big', type=int, default=0)
    args = p.parse_args()
    truth = annotations()
    out = out_dir(); out.mkdir(parents=True, exist_ok=True)
    for stem in args.stems:
        target = out / f'{stem}--d{args.draws}-r{args.time_reference}-b{args.big}.json'
        if target.exists():
            continue
        tick = time.process_time(); ev = evaluated_for(stem, truth); setup_cpu = time.process_time() - tick
        tick = time.process_time(); fast = FastAccumulate(ev); prep_cpu = time.process_time() - tick
        reference = ev.metrics()
        point = fast.metrics()
        from tools.experiment_b.common import file_hash
        result = {'stem': stem, 'fast_source_sha256': file_hash(SOURCE), 'point_reference': reference.tolist(), 'point_fast': point.tolist(),
                  'point_identical': bool(np.array_equal(reference, point)),
                  'point_max_abs_difference': float(np.abs(reference - point).max()),
                  'evaluate_cpu_seconds': setup_cpu, 'fast_preparation_cpu_seconds': prep_cpu}
        stored_path = BASE / 'bootstrap' / f'{stem}.npz'
        if stored_path.exists():
            with np.load(stored_path) as z:
                stored_point, stored_draws = z['point'], z['draws']
            result['stored_point_identical'] = bool(np.array_equal(stored_point, point))
            if args.draws and len(stored_draws):
                tick = time.process_time(); draws = fast.bootstrap(RESAMPLES, SEED, CONFIDENCE)[:args.draws] if args.draws == RESAMPLES \
                    else _first(fast, args.draws, SEED, CONFIDENCE, RESAMPLES)
                result['fast_cpu_seconds_per_draw'] = (time.process_time() - tick) / args.draws
                ref = stored_draws[:args.draws]
                result.update(draws_compared=args.draws, draws_identical=bool(np.array_equal(ref, draws)),
                              draws_max_abs_difference=float(np.abs(ref - draws).max()))
                if args.draws == len(stored_draws):
                    result['intervals_identical'] = all(interval(draws[:, i]) == interval(stored_draws[:, i]) for i in range(2))
        if args.time_reference:
            from public.analysis.phase3.statistics import settings
            rng = settings(CONFIDENCE, RESAMPLES, SEED); tick = time.process_time()
            for _ in range(args.time_reference):
                ev.metrics((rng.integers(0, ev.images, size=ev.images) + 1).tolist())
            result['reference_cpu_seconds_per_draw'] = (time.process_time() - tick) / args.time_reference
        if args.big:
            tick = time.process_time(); fast.bootstrap(args.big, SEED, CONFIDENCE)
            result[f'fast_cpu_seconds_{args.big}_draws'] = time.process_time() - tick
        target.write_text(json.dumps(result, indent=1, sort_keys=True))
        print(json.dumps({k: v for k, v in result.items() if not k.startswith('point_') or k == 'point_identical'}), flush=True)


def _first(fast, count, seed, confidence, resamples):
    from public.analysis.phase3.statistics import settings
    rng = settings(confidence, resamples, seed)
    return np.array([fast.metrics((rng.integers(0, fast.n, size=fast.n) + 1).tolist()) for _ in range(count)])


if __name__ == '__main__':
    main()
