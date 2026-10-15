"""Validate and time equivalent execution on independent calibration only."""
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from verification_extension import classical as C, protocol as P, workflow as W
from verification_gpu import accelerate as A


def hashes():
    root = Path(__file__).resolve().parents[1]
    names = ('verification_gpu/accelerate.py', 'verification_gpu/speed_profile.py',
             'verification_gpu/schema.py', 'run_fast_verification.py')
    return {name: P.file_hash(root / name) for name in names}


def timed(fn, device, repetitions=3):
    fn()
    values = []
    for _ in range(repetitions):
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter(); fn()
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        values.append(time.perf_counter() - start)
    return statistics.median(values)


def load_policy(args):
    value = P.signed_json(Path(args.output) / 'execution_optimization.json')
    if (not value.get('complete') or not value.get('equivalence_passed')
            or value['adapter_sha256'] != hashes() or value['core_sha256'] != P.source_hashes()):
        raise RuntimeError('Execution optimization is unverified or its source changed; rerun speed-profile')
    from verification_extension.calibration import _hardware
    if value['hardware'] != _hardware(torch):
        raise RuntimeError('Execution speed profile belongs to another server; rerun speed-profile')
    return value


def profile(args):
    from verification_extension.calibration import _envelope, _hardware
    arrays, manifest = W.calibration_arrays(args)
    all_arrays = W.statistical_arrays(args, arrays, manifest)
    groups = C.group_calibration_batches(all_arrays, manifest)
    if not torch.cuda.is_available():
        raise RuntimeError('Original GPU generator and GPU receiver timing require CUDA')
    torch.set_num_threads(int(args.host_threads))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    candidates, gates, performance = {}, [], []
    group_items = sorted(groups.items(), key=lambda x: abs(x[1]['geometry'].cp_wave_samples - 128))
    for device in ('cpu', 'cuda:0'):
        A.configure(args.cache_mib)
        for gi, (key, group) in enumerate(group_items):
            g = group['geometry']; arr = group['batches'][0]
            xp, yp = arr['x_pilot'][0], arr['y_pilot'][0]
            yd, labels, nv = arr['rx_data'][0], arr['tx_label'][0:1], float(arr['noise_var'][0])
            metadata = json.loads(str(np.asarray(arr['metadata']).item()))
            modulation = metadata['condition']['modulation']
            grid = [(q, start, stop) for q in (9, 15, 21)
                    for start, stop in ((-4, 28), (-8, 40))] if gi == 0 else [(21, -8, 40)]
            for method in C.BEM_METHODS:
                for q, start, stop in grid:
                    settings = C.BEMSettings(method=method, q=q, delay_start=start, delay_stop=stop,
                                            training_doppler_envelope_hz=_envelope(manifest))
                    baseline = A.OriginalJointBEM(g, settings, xp.shape[0], device=device)
                    fast = A.CachedJointBEM(g, settings, xp.shape[0], device=device)
                    with torch.inference_mode():
                        original_design = baseline.design_matrix(xp)
                        fast.design_matrix(xp)
                        cached_design = fast.design_matrix(xp)
                        if not torch.equal(original_design, cached_design):
                            raise RuntimeError('Cached design differs from original: ' + method)
                        # Exercise recombined public occasions, not a repeated
                        # complete-frame lookup. No recombined frame is scored.
                        if len(arr['noise_var']) > 1:
                            other = arr['x_pilot'][1]
                            fast.design_matrix(other)
                            mixed = xp.copy(); mixed[:, 1::2] = other[:, 1::2]
                            if not torch.equal(fast.design_matrix(mixed), baseline.design_matrix(mixed)):
                                raise RuntimeError('Public-design block separability check failed')
                        for rho in (.01, .1, 1., 10., 100.):
                            c1, i1 = baseline.fit(xp, yp, nv, design=original_design, relative_ridge=rho)
                            c2, i2 = fast.fit(xp, yp, nv, design=cached_design, relative_ridge=rho)
                            if not torch.equal(c1, c2) or i1 != i2:
                                raise RuntimeError('Cached ridge fit changed coefficients or solver gates')
                        gates.append({'device': device, 'geometry': key, 'settings': asdict(settings),
                                      'design_bitwise_equal': True, 'all_five_ridges_bitwise_equal': True})
                        print('[speed-gate] %s %s CP=%d Q=%d support=[%d,%d) PASS' %
                              (device, method, g.cp_wave_samples, q, start, stop), flush=True)
                        if (q, start, stop) != (21, -8, 40):
                            continue
                        def ridge_work(op):
                            d = op.design_matrix(xp)
                            return [op.fit(xp, yp, nv, design=d, relative_ridge=rho)
                                    for rho in (.01, .1, 1., 10., 100.)]
                        old = timed(lambda: ridge_work(baseline), device)
                        new = timed(lambda: ridge_work(fast), device)
                        performance.append({'device': device, 'method': method, 'geometry': key,
                                            'workload': 'five_ridges_same_public_frame',
                                            'original_seconds': old, 'cached_seconds': new,
                                            'speedup': old / new, 'cache_state': 'warm_public_design_blocks'})
                        c, _ = baseline.fit(xp, yp, nv)
                        reference, _ = baseline.native_mmse(c, yd, nv, symbol_batch_size=4)
                        eb, bb = C.frame_bit_counts(reference[None], labels, modulation)
                        for chunk in sorted({4, 8, len(g.data_visible)}):
                            z, info = baseline.native_mmse(c, yd, nv, symbol_batch_size=chunk)
                            torch.testing.assert_close(z, reference, rtol=5e-11, atol=5e-12)
                            e, b = C.frame_bit_counts(z[None], labels, modulation)
                            if not torch.equal(e, eb) or not torch.equal(b, bb):
                                raise RuntimeError('Native symbol batching changed decoded bit counts')
                            if info['relative_residual_max'] > 1e-6:
                                raise RuntimeError('Native symbol batching failed the original residual gate')
                            elapsed = timed(lambda: baseline.native_mmse(c, yd, nv,
                                              symbol_batch_size=chunk), device)
                            candidates.setdefault(device + '/' + method, {}).setdefault(chunk, []).append(elapsed)
                    del baseline, fast, original_design, cached_design
                    if device.startswith('cuda'):
                        torch.cuda.empty_cache()
    native = {key: min(values, key=lambda chunk: sum(values[chunk]) / len(values[chunk]))
              for key, values in candidates.items()}
    result = {'schema_version': 1, 'complete': True, 'equivalence_passed': True,
              'scope': 'independent calibration algebra and traditional compute; no neural CPU timing or generation-backend comparison',
              'cache_mib': int(args.cache_mib), 'native_chunks': native,
              'core_sha256': P.source_hashes(), 'adapter_sha256': hashes(), 'hardware': _hardware(torch),
              'gates': gates, 'performance': performance,
              'native_chunk_seconds': {key: {str(k): v for k, v in values.items()}
                                       for key, values in candidates.items()},
              'generator_modified': False, 'training_performed': False,
              'precision': 'complex128', 'fit_and_detector_tolerance': 1e-6,
              'cache_contents': 'public pilot design blocks on CPU; per-frame Gram reused across existing ridges',
              'benchmark_note': 'Warm repeated-frame speedups apply to the displayed compute workloads, not the entire test suite'}
    result['signature'] = P.digest(result)
    P.write_json(Path(args.output) / 'execution_optimization.json', result)
    print('Equivalent execution verified. Native symbol chunks: ' + str(native), flush=True)
    for row in performance:
        print('[speed] %s %s five-ridge workload: %.2fx' %
              (row['device'], row['method'], row['speedup']), flush=True)
    return result
