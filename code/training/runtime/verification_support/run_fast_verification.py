#!/usr/bin/env python3
"""Verified public-design caching; original frozen GPU data, GPU-only latency."""
from pathlib import Path
import argparse
from contextlib import contextmanager
import json
import os
import subprocess
import sys
import time


@contextmanager
def evaluation_provenance(args):
    """Bind each newly written score to its verified execution adapter."""
    import numpy as np
    from verification_extension import protocol as P, workflow as W
    from verification_gpu.speed_profile import hashes
    policy_sha = P.file_hash(args.output / 'execution_optimization.json')
    sources = hashes()
    old_npz, old_json = W.atomic_npz, W.write_json
    def save_npz(path, arrays):
        payload = dict(arrays)
        if Path(path).parent.name == 'cells':
            payload['execution_optimization_sha256'] = np.asarray(policy_sha)
            payload['execution_adapter_sha256'] = np.asarray(json.dumps(sources, sort_keys=True))
        return old_npz(path, payload)
    def save_json(path, value):
        if Path(path).name.endswith('.COMPLETE.json') or Path(path).name == 'COMPLETE.json':
            value = {**value, 'execution_optimization_sha256': policy_sha,
                     'execution_adapter_sha256': sources}
        return old_json(path, value)
    W.atomic_npz, W.write_json = save_npz, save_json
    try:
        yield
    finally:
        W.atomic_npz, W.write_json = old_npz, old_json


def parser():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-results', type=Path, required=True)
    p.add_argument('--project', type=Path, default=root / 'frozen_reference')
    p.add_argument('--output', type=Path, default=root / 'verification_results')
    p.add_argument('--suite', choices=('core', 'nominal', 'manuscript', 'all'), default='manuscript')
    stages = ('all', 'preflight', 'prepare', 'speed-profile', 'benchmark', 'calibrate',
              'benchmark-selected', 'latency-gpu', 'evaluate', 'report-gpu')
    p.add_argument('--stage', choices=stages, default='all')
    p.add_argument('--host-threads', type=int, default=8)
    p.add_argument('--cpu-threads', type=int, nargs='+', default=(1, 4, 8))
    p.add_argument('--cache-mib', type=int, default=1024)
    p.add_argument('--batch-sizes', type=int, nargs='+', default=(1, 8, 32))
    p.add_argument('--latency-warmups', type=int, default=5)
    p.add_argument('--latency-repetitions', type=int, default=30)
    p.add_argument('--bootstrap', type=int, default=2000)
    return p


def command(stage):
    argv = list(sys.argv[1:])
    if '--stage' in argv:
        i = argv.index('--stage'); del argv[i:i + 2]
    argv = [a for a in argv if not a.startswith('--stage=')]
    return [sys.executable, str(Path(__file__).resolve()), *argv, '--stage', stage]


def main():
    args = parser().parse_args()
    from verification_extension import protocol as P
    args.reference_results = P.result_root(args.reference_results)
    args.output = args.output.expanduser().resolve()
    args.project = args.project.expanduser().resolve()
    if (args.host_threads < 1 or not args.cpu_threads or min(args.cpu_threads) < 1
            or args.cache_mib < 0 or not args.batch_sizes or not set(args.batch_sizes) <= {1, 8, 32}
            or args.latency_warmups < 1 or args.latency_repetitions < 10 or args.bootstrap < 1000):
        raise ValueError('Invalid thread/cache/batch/timing settings')
    args.q_grid, args.ridge_grid, args.noise_grid = (9, 15, 21), (.01, .1, 1, 10, 100), (.01, .1, 1, 10, 100)
    args.statistical_calibration_extra = 256
    args.benchmark_warmups, args.benchmark_repetitions = 2, 5
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if args.stage == 'all':
        stages = ['preflight', 'prepare', 'speed-profile', 'benchmark', 'calibrate',
                  'benchmark-selected', 'latency-gpu', 'evaluate', 'report-gpu']
        # Reuse the original signed calibration rather than regenerate it.
        if (args.output / 'calibration_lock.json').is_file():
            stages.remove('benchmark'); stages.remove('calibrate')
        if (args.output / 'execution_optimization.json').is_file():
            from verification_gpu.speed_profile import load_policy
            try:
                policy = load_policy(args)
            except RuntimeError:
                policy = None
            if policy is not None and policy['cache_mib'] == args.cache_mib:
                stages.remove('speed-profile')
        with P.writer_lock(args.output, '.controller.lock'):
            for stage in stages:
                print('\n>>> ' + ' '.join(command(stage)), flush=True)
                path = args.output / 'logs' / ('fast_' + stage + '.log')
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a', encoding='utf-8') as log:
                    proc = subprocess.Popen(command(stage), stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True,
                                            encoding='utf-8', errors='replace')
                    for line in proc.stdout:
                        print(line, end='', flush=True); log.write(line); log.flush()
                    if proc.wait():
                        raise RuntimeError('Failed stage %s; see %s' % (stage, path))
        print('Completed verified faster execution, corrected scores, GPU latency and paper package.', flush=True)
        return
    from verification_extension import workflow as W
    with P.writer_lock(args.output):
        if args.stage in ('preflight', 'prepare'):
            getattr(W, args.stage)(args)
        elif args.stage == 'speed-profile':
            from verification_gpu.speed_profile import profile
            profile(args)
        elif args.stage in ('benchmark', 'calibrate', 'benchmark-selected', 'evaluate'):
            from verification_gpu.speed_profile import load_policy, hashes
            from verification_gpu.accelerate import installed, statistics
            policy = load_policy(args)
            started = time.perf_counter()
            with installed(policy['cache_mib'], policy['native_chunks']):
                if args.stage == 'benchmark-selected':
                    W.benchmark(args, selected=True)
                elif args.stage == 'evaluate':
                    from verification_gpu.execution import select_generation_backend
                    select_generation_backend(args)
                    from verification_gpu.schema import installed as historical_score_schema
                    with historical_score_schema(), evaluation_provenance(args):
                        W.evaluate(args)
                else:
                    getattr(W, args.stage)(args)
                counts = statistics()
            P.write_json(args.output / ('execution_' + args.stage + '.json'), {
                'stage': args.stage, 'complete': True, 'adapter_sha256': hashes(),
                'optimization_policy_sha256': P.file_hash(args.output / 'execution_optimization.json'),
                'core_sha256': P.source_hashes(), 'counts': counts,
                'elapsed_seconds': time.perf_counter() - started,
                'original_frozen_waveform_gate_retained': True,
                'historical_bits_loading': 'original bits read only for declared BER cells; no synthetic bit totals',
                'precision_and_estimators_unchanged': True})
        elif args.stage == 'latency-gpu':
            from verification_gpu.latency import benchmark
            benchmark(args)
        elif args.stage == 'report-gpu':
            from verification_gpu.reporting import report
            report(args)


if __name__ == '__main__':
    main()
