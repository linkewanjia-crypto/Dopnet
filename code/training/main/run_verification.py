#!/usr/bin/env python3
"""Correct references and measure full digital receiver latency without training."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def parser():
    code = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-results', type=Path, required=True,
                   help='Existing dopnet_additions directory (or its cells directory); neural models are reused')
    p.add_argument('--project', type=Path, default=code / 'frozen_reference',
                   help='Unmodified frozen source supplied in the server ZIP')
    p.add_argument('--output', type=Path, default=code / 'verification_results')
    p.add_argument('--suite', choices=('core', 'nominal', 'manuscript', 'all'), default='manuscript',
                   help='core=99; nominal=236; manuscript=269; all=636 original conditions')
    p.add_argument('--stage', choices=('all', 'preflight', 'prepare', 'benchmark', 'calibrate',
                                      'benchmark-selected', 'evaluate', 'latency', 'report'), default='all')
    p.add_argument('--cpu-threads', type=int, nargs='+', default=(1, 4, 8))
    p.add_argument('--q-grid', type=int, nargs='+', default=(9, 15, 21))
    p.add_argument('--ridge-grid', type=float, nargs='+', default=(.01, .1, 1, 10, 100))
    p.add_argument('--noise-grid', type=float, nargs='+', default=(.01, .1, 1, 10, 100))
    p.add_argument('--statistical-calibration-extra', type=int, default=256,
                   help='Extra independent calibration frames for CP-specific covariance fitting; frozen test frames do not change')
    p.add_argument('--batch-sizes', type=int, nargs='+', default=(1, 8, 32))
    p.add_argument('--benchmark-warmups', type=int, default=2)
    p.add_argument('--benchmark-repetitions', type=int, default=5)
    p.add_argument('--latency-warmups', type=int, default=5)
    p.add_argument('--latency-repetitions', type=int, default=30)
    p.add_argument('--bootstrap', type=int, default=2000)
    return p


def main():
    args = parser().parse_args()
    args.output = args.output.expanduser().resolve()
    args.project = args.project.expanduser().resolve()
    from verification_extension.protocol import result_root, writer_lock
    args.reference_results = result_root(args.reference_results)
    if any(value < 1 for value in list(args.cpu_threads) + list(args.batch_sizes)):
        raise ValueError('Threads and batch sizes must be positive')
    if args.benchmark_repetitions < 3 or args.latency_repetitions < 10:
        raise ValueError('Use at least three verification-cost repetitions and ten receiver-latency repetitions')
    if min(args.benchmark_warmups, args.latency_warmups) < 1 or args.bootstrap < 1000:
        raise ValueError('Use positive warmups and at least 1000 bootstrap replicates')
    if args.statistical_calibration_extra < 0 or args.statistical_calibration_extra % 32:
        raise ValueError('Independent calibration extra must be a nonnegative multiple of 32')
    if any(q < 1 or q % 2 == 0 for q in args.q_grid):
        raise ValueError('BEM basis counts must be positive odd integers')
    if any(not math.isfinite(value) or value <= 0 for value in list(args.ridge_grid) + list(args.noise_grid)):
        raise ValueError('Calibration scales must be positive and finite')
    if not set(args.batch_sizes) <= {1, 8, 32}:
        raise ValueError('Receiver timing supports batch sizes 1, 8 and 32')
    for values in (args.cpu_threads, args.batch_sizes, args.q_grid, args.ridge_grid, args.noise_grid):
        if len(values) != len(set(values)):
            raise ValueError('Grid values must be unique')
    # Independent workers prevent frozen replay and public model modules mixing.
    if args.stage == 'all':
        with writer_lock(args.output, '.controller.lock'):
            run_all(args)
        return
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    from verification_extension import workflow
    with writer_lock(args.output):
        if args.stage == 'benchmark-selected':
            workflow.benchmark(args, selected=True)
        else:
            getattr(workflow, args.stage)(args)


def run_all(args):
    stages = ('preflight', 'prepare', 'benchmark', 'calibrate', 'benchmark-selected',
              'latency', 'evaluate', 'report')
    command = [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]
    # An explicitly supplied --stage all must not survive as a duplicate option.
    if '--stage' in command:
        index = command.index('--stage')
        del command[index:index + 2]
    args.output.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        stage_command = command + ['--stage', stage]
        print('\n>>> ' + ' '.join(stage_command), flush=True)
        log = args.output / 'logs' / (stage + '.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a', encoding='utf-8') as handle:
            handle.write('\nRun started %.6f\n' % time.time()); handle.flush()
            process = subprocess.Popen(stage_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                print(line, end='', flush=True); handle.write(line); handle.flush()
            if process.wait():
                raise RuntimeError('Stage failed: %s. Existing checkpoints/scores are unchanged; see %s' % (stage, log))
    print('Completed corrected scores, CPU/GPU policies, digital latency, and tables; no model was trained.', flush=True)


if __name__ == '__main__':
    main()
