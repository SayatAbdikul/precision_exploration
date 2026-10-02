"""Lane S1, protocol speed-protocol-v1-addendum-1 (review 1, finding N1): device-level GPU memory of a block cell
run through the host-memory bias correction.

  .venv-b/bin/python -m tools.run.speed_fastblocks_mem MODEL FORMAT RECIPE     (GPU through gpu_run.sh)

Runs tools/run/speed_fastblocks.py (unchanged, --path fast) inside this process with its output folder redirected to
artifacts/speed_v1/fastblocks-mem-v1/ (the cell, readout and configuration land there; identity against L1's record
is checked by speed_fastblocks as before). Added here, in this process only:
  - a sampler thread: nvidia-smi --query-compute-apps used_memory of this process's pid, every 0.5 s;
  - torch.cuda.max_memory_allocated / max_memory_reserved over the WHOLE process (speed_fastblocks resets the peak
    before the bias correction; the peaks are read before every reset and at the end).
Result: artifacts/speed_v1/fastblocks-mem-v1/fast/<model>--<format>--<recipe>.memory.json (written once).
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # as speed_fastblocks / experiment_b2_matrix
import json
import subprocess
import sys
import threading
import time

from tools.experiment_b.common import ROOT

OUT = ROOT / 'artifacts/speed_v1/fastblocks-mem-v1'


class Sampler(threading.Thread):
    def __init__(self, pid, period=0.5):
        super().__init__(daemon=True)
        self.pid, self.period, self.stop = str(pid), period, threading.Event()
        self.samples, self.peak_mib, self.peak_at, self.errors = 0, 0, None, 0
        self.marks = []                       # (label, epoch) of phases, for the timeline
        self.timeline = []                    # (seconds since start, MiB) when the value changes by >= 64 MiB
        self.t0 = time.time()

    def run(self):
        last = -10 ** 9
        while not self.stop.is_set():
            try:
                out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True, timeout=10).stdout
                mib = next((int(line.split(',')[1]) for line in out.splitlines() if line.split(',')[0].strip() == self.pid), 0)
                self.samples += 1
                if mib > self.peak_mib:
                    self.peak_mib, self.peak_at = mib, time.time() - self.t0
                if abs(mib - last) >= 64:
                    self.timeline.append((round(time.time() - self.t0, 1), mib)); last = mib
            except Exception:                 # noqa: BLE001  (a missed sample is counted, not fatal)
                self.errors += 1
            self.stop.wait(self.period)


def main():
    model, fmt, recipe = sys.argv[1:4]
    target = OUT / 'fast' / f'{model}--{fmt}--{recipe}.memory.json'
    if target.exists():
        raise SystemExit(f'{target} exists')
    import torch
    import tools.run.speed_fastblocks as SF
    SF.SPEED = OUT                            # this process only: outputs into fastblocks-mem-v1/
    sampler = Sampler(os.getpid()); sampler.start()
    peaks = {'allocated': [], 'reserved': []}
    reset = torch.cuda.reset_peak_memory_stats

    def recording_reset(*a, **k):
        if torch.cuda.is_initialized():
            peaks['allocated'].append(torch.cuda.max_memory_allocated()); peaks['reserved'].append(torch.cuda.max_memory_reserved())
            sampler.marks.append(('reset_peak_memory_stats (start of bias correction)', round(time.time() - sampler.t0, 1)))
        return reset(*a, **k)
    torch.cuda.reset_peak_memory_stats = recording_reset
    sys.argv = ['speed_fastblocks', model, fmt, recipe, '--path', 'fast']
    tick = time.time()
    try:
        SF.main()
    finally:
        sampler.stop.set(); sampler.join(timeout=15)
    peaks['allocated'].append(torch.cuda.max_memory_allocated()); peaks['reserved'].append(torch.cuda.max_memory_reserved())
    execution = OUT / 'fast' / f'{model}--{fmt}--{recipe}.execution.json'
    record = json.loads(execution.read_text())
    result = {'protocol': 'speed-protocol-v1-addendum-1 (n1_device_memory_of_fast_block_cells)',
              'model': model, 'format': fmt, 'recipe': recipe, 'execution_record': str(execution.relative_to(ROOT)),
              'identity': {k: record.get(k) for k in ('configuration_identical', 'logits_sha256_identical', 'readout_arrays',
                                                     'readout_arrays_identical', 'readout_file_sha256_identical',
                                                     'readout_summary_identical')},
              'nvidia_smi_process_peak_mib': sampler.peak_mib, 'nvidia_smi_peak_at_seconds': sampler.peak_at,
              'nvidia_smi_samples': sampler.samples, 'nvidia_smi_errors': sampler.errors, 'sample_period_seconds': sampler.period,
              'timeline_mib': sampler.timeline, 'marks': sampler.marks,
              'torch_max_memory_allocated_whole_process_bytes': max(peaks['allocated']),
              'torch_max_memory_reserved_whole_process_bytes': max(peaks['reserved']),
              'torch_peaks_per_phase': peaks,
              'bias_correction_peak_gpu_bytes': record.get('bias_correction_peak_gpu_bytes'),
              'bias_correction_seconds': record.get('bias_correction_seconds'),
              'process_wall_seconds': time.time() - tick, 'written': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    target.write_text(json.dumps(result, indent=1))
    print('MEM ' + json.dumps({k: v for k, v in result.items() if k not in ('timeline_mib', 'torch_peaks_per_phase')}), flush=True)


if __name__ == '__main__':
    main()
