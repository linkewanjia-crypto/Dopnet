#!/usr/bin/env python3
"""Reuse completed calibration; use GPU waveform generation and GPU-only receiver timing."""
from pathlib import Path
import argparse
import os
import subprocess
import sys


def parser():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-results', type=Path, required=True)
    p.add_argument('--project', type=Path, default=root / 'frozen_reference')
    p.add_argument('--output', type=Path, default=root / 'verification_results')
    p.add_argument('--suite', choices=('core', 'nominal', 'manuscript', 'all'), default='manuscript')
    p.add_argument('--stage', choices=('resume', 'latency-gpu', 'evaluate', 'report-gpu'), default='resume')
    p.add_argument('--from-stage', choices=('latency-gpu', 'evaluate', 'report-gpu'), default='latency-gpu')
    p.add_argument('--host-threads', type=int, default=8, help='Host execution threads; does not enable CPU model timing')
    p.add_argument('--batch-sizes', type=int, nargs='+', default=(1, 8, 32))
    p.add_argument('--latency-warmups', type=int, default=5)
    p.add_argument('--latency-repetitions', type=int, default=30)
    p.add_argument('--bootstrap', type=int, default=2000)
    return p


def worker_command(stage):
    args = list(sys.argv[1:])
    for flag in ('--stage',):
        if flag in args:
            i = args.index(flag)
            del args[i:i + 2]
        args = [arg for arg in args if not arg.startswith(flag + '=')]
    return [sys.executable, str(Path(__file__).resolve()), *args, '--stage', stage]


def main():
    args = parser().parse_args()
    args.cpu_threads = (args.host_threads,)
    from verification_extension import protocol as P
    args.output = args.output.expanduser().resolve()
    args.project = args.project.expanduser().resolve()
    args.reference_results = P.result_root(args.reference_results)
    if (args.latency_repetitions < 10 or args.latency_warmups < 1 or args.bootstrap < 1000
            or not args.batch_sizes or len(args.batch_sizes) != len(set(args.batch_sizes))
            or not set(args.batch_sizes) <= {1, 8, 32} or args.host_threads < 1):
        raise ValueError('Use positive threads/warmups, latency repetitions>=10, bootstrap>=1000')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    lock = P.signed_json(args.output / 'calibration_lock.json')
    if lock.get('code_sha256') != P.source_hashes() or lock.get('status') != 'locked':
        raise RuntimeError('Original calibrated implementation changed. Apply the hotfix without editing the locked files.')
    if args.stage == 'resume':
        stages = ('latency-gpu', 'evaluate', 'report-gpu')
        stages = stages[stages.index(args.from_stage):]
        with P.writer_lock(args.output, '.controller.lock'):
            for stage in stages:
                command = worker_command(stage)
                print('\n>>> ' + ' '.join(command), flush=True)
                log = args.output / 'logs' / (stage + '.log')
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open('a', encoding='utf-8') as stream:
                    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, encoding='utf-8', errors='replace')
                    for line in proc.stdout:
                        print(line, end='', flush=True); stream.write(line); stream.flush()
                    if proc.wait():
                        raise RuntimeError('Failed stage: %s; see %s' % (stage, log))
        print('Completed GPU-only receiver timing, corrected scores and report; existing calibration reused.', flush=True)
        return
    with P.writer_lock(args.output):
        if args.stage == 'latency-gpu':
            from verification_gpu.latency import benchmark
            benchmark(args)
        elif args.stage == 'evaluate':
            from verification_gpu.execution import select_generation_backend
            select_generation_backend(args)
            from verification_extension.workflow import evaluate
            evaluate(args)
        elif args.stage == 'report-gpu':
            from verification_gpu.reporting import report
            report(args)


if __name__ == '__main__':
    main()
