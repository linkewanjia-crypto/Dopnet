"""Server-only synchronized fp32 timing of channel estimation plus equalization.

The synthetic received batch is generated once for each batch size and then
reused by every checkpoint, warmup, and timed repetition.  Neither waveform
generation nor pilot/ADD preprocessing is inside the timed region.  This is
not an end-to-end or real-time receiver benchmark.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch

from data.generator import BatchChannelGenerator
from experiment_protocol import file_hash, write_json
from models.receiver import (
    SharedUserChannelEstimator,
    banded_multiuser_equalize,
    ici_bands_from_dopnet_aux,
    linear_multiuser_equalize,
    select_data_csi,
    select_data_ici_bands,
)
from train import model_inputs
from utils.checkpoints import load_checkpoint


METHODS = ('dopnet', 'dopnet_lite', 'slow_time', 'fortitran')
SEEDS = (42, 43, 44)
INPUT_SEED = 920001
CONDITION = {
    'channel_type': 'doubly_selective',
    'modulation': 'qpsk',
    'delay_spread_ns': 300.0,
    'doppler_hz': 1200.0,
    'cp_len': 16,
    'pilot_spacing': 4,
}
SNR_DB = 20.0
INPUT_KEYS = (
    'input', 'add_input', 'add_input_slow',
    'add_input_pilot_only_fast_time', 'noise_var', 'ls_grid', 'rx_data',
)
PCG_MAX_ITER = 96
PCG_TOL = 1e-6
MAX_ACCEPTED_RESIDUAL = 1e-4


def _canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _shared_input_generation_identity(cfg):
    """Physical generation policy, excluding an optional derived feature output."""
    pipeline = dict(cfg['data_pipeline'])
    # This flag only emits an extra pilot-only ADD tensor. None of the four
    # receivers timed here consumes it; old reused configs omit the flag.
    pipeline.pop('emit_pilot_only_fast_time_add', None)
    return cfg['system'], pipeline


def _batch_digest(batch):
    """Content hash of the actual tensors consumed by the profiled receiver."""
    hasher = hashlib.sha256()
    for key in INPUT_KEYS:
        if key not in batch:
            continue
        tensor = batch[key].detach().contiguous().cpu()
        hasher.update(key.encode('utf-8'))
        hasher.update(str(tensor.dtype).encode('ascii'))
        hasher.update(_canonical_json(list(tensor.shape)).encode('ascii'))
        hasher.update(tensor.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _cache_metadata(manifest, cfg, batch_sizes):
    generation = {
        'system': cfg['system'],
        'data_pipeline': cfg['data_pipeline'],
        'condition': CONDITION,
        'snr_db': SNR_DB,
        'input_seed': INPUT_SEED,
        'batch_sizes': list(batch_sizes),
    }
    return {
        'format_version': 1,
        'experiment_signature': manifest['signature'],
        'generation_sha256': hashlib.sha256(
            _canonical_json(generation).encode('utf-8')).hexdigest(),
        'condition': CONDITION,
        'snr_db': SNR_DB,
        'input_seed': INPUT_SEED,
        'batch_sizes': list(batch_sizes),
    }


def _load_or_make_inputs(cache_path, manifest, cfg, device, batch_sizes):
    """Persist pre-generated, CPU-stored inputs; transfer occurs before timing."""
    expected = _cache_metadata(manifest, cfg, batch_sizes)
    if cache_path.exists():
        cache = load_checkpoint(cache_path, map_location='cpu')
        if cache.get('metadata') != expected:
            raise RuntimeError(
                f'Input cache protocol differs: {cache_path}. Use a new --input-cache path.'
            )
        raw_batches = cache['batches']
        expected_hashes = cache['hashes']
    else:
        generator = BatchChannelGenerator(cfg, device, INPUT_SEED)
        raw_batches = {}
        for B in batch_sizes:
            rng = torch.Generator(device=device).manual_seed(INPUT_SEED + 1009 * B)
            simulated = generator.simulate(
                B, CONDITION, torch.full((B,), SNR_DB, device=device),
                rng, INPUT_SEED + 2003 * B,
            )
            raw_batches[B] = {
                key: simulated[key].detach().contiguous().cpu()
                for key in INPUT_KEYS if key in simulated
            }
            del simulated
        expected_hashes = {B: _batch_digest(raw_batches[B]) for B in batch_sizes}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + '.tmp')
        torch.save({
            'metadata': expected, 'batches': raw_batches,
            'hashes': expected_hashes,
        }, temporary)
        temporary.replace(cache_path)
        del generator
    batches = {}
    hashes = {}
    for B in batch_sizes:
        source = raw_batches[B]
        if not {'input', 'add_input', 'add_input_slow', 'noise_var', 'rx_data'} <= set(source):
            raise RuntimeError(f'Input cache is incomplete for B={B}')
        if source['input'].shape[0] != B or source['rx_data'].shape[0] != B:
            raise RuntimeError(f'Input cache has wrong batch size for B={B}')
        hashes[B] = _batch_digest(source)
        if hashes[B] != expected_hashes[B]:
            raise RuntimeError(f'Input cache content hash differs for B={B}: {cache_path}')
        batches[B] = {key: value.to(device) for key, value in source.items()}
    return batches, hashes, expected


def _native_available(model):
    # The architecture, not a runtime result, determines this protocol.
    return model.estimator_name == 'dopnet' and model.base.use_add_branch


def _receiver_step(model, cfg, batch, mode):
    kw = model_inputs(batch)
    ce_input = batch['input']
    if mode == 'common_diagonal':
        hhat = model(ce_input, **kw)
        hdata = select_data_csi(hhat, cfg)
        symbols, _ = linear_multiuser_equalize(
            hdata, batch['rx_data'], batch['noise_var'], method='mmse'
        )
        return symbols, None
    if mode != 'native_own_ici':
        raise ValueError(mode)
    if not _native_available(model):
        raise RuntimeError('Native ADD-to-ICI receiver is unavailable for this model')
    hhat, aux = model(ce_input, return_aux=True, **kw)
    if 'add_fused_pred' not in aux:
        raise RuntimeError('Expected native ADD representation is absent')
    bands = ici_bands_from_dopnet_aux(
        model, ce_input, hhat, aux, half_band=2, replace_center=True
    )
    bands = select_data_ici_bands(bands, cfg)
    symbols, _, info = banded_multiuser_equalize(
        bands, batch['rx_data'], batch['noise_var'],
        solver='pcg', max_iter=PCG_MAX_ITER, tol=PCG_TOL, return_info=True,
    )
    fallback = (
        not np.isfinite(info['relative_residual_max'])
        or info['relative_residual_max'] > MAX_ACCEPTED_RESIDUAL
    )
    pcg_iterations = info['iterations']
    if fallback:
        symbols, _, info = banded_multiuser_equalize(
            bands, batch['rx_data'], batch['noise_var'],
            solver='direct', return_info=True,
        )
    if (
        not np.isfinite(info['relative_residual_max'])
        or info['relative_residual_max'] > MAX_ACCEPTED_RESIDUAL
    ):
        raise RuntimeError(f'Native ICI equalizer failed residual check: {info}')
    return symbols, {
        'direct_fallback': bool(fallback),
        'pcg_iterations': int(pcg_iterations),
        'relative_residual_max': float(info['relative_residual_max']),
    }


@torch.inference_mode()
def _time_one(model, cfg, batch, mode, warmups, repetitions):
    for _ in range(warmups):
        symbols, _ = _receiver_step(model, cfg, batch, mode)
        del symbols
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident_before = torch.cuda.memory_allocated()
    milliseconds = []
    native_info = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter()
        symbols, info = _receiver_step(model, cfg, batch, mode)
        torch.cuda.synchronize()
        milliseconds.append((time.perf_counter() - start) * 1000.0)
        if info is not None:
            native_info.append(info)
        del symbols
    peak = torch.cuda.max_memory_allocated()
    mean = statistics.mean(milliseconds)
    result = {
        'latency_ms_mean': mean,
        'latency_ms_p50': float(np.quantile(milliseconds, 0.50)),
        'latency_ms_p95': float(np.quantile(milliseconds, 0.95)),
        'latency_ms_std': statistics.stdev(milliseconds),
        'frames_per_second': 1000.0 * batch['input'].shape[0] / mean,
        'memory_allocated_before_mb': resident_before / 2**20,
        'memory_peak_mb': peak / 2**20,
        'memory_incremental_peak_mb': max(0, peak - resident_before) / 2**20,
        'direct_fallback_repetitions': sum(x['direct_fallback'] for x in native_info),
        'pcg_iterations_mean': (
            statistics.mean(x['pcg_iterations'] for x in native_info)
            if native_info else None
        ),
        'native_residual_max': (
            max(x['relative_residual_max'] for x in native_info)
            if native_info else None
        ),
    }
    return result


def _summarize_seeds(rows):
    """Report model-seed variation without treating timed repeats as independent models."""
    groups = {}
    for row in rows:
        if row['status'] != 'measured':
            continue
        key = (row['method'], row['batch_size'], row['receiver_mode'])
        groups.setdefault(key, []).append(row)
    out = []
    for (method, batch_size, mode), group in sorted(groups.items()):
        values = [row['latency_ms_mean'] for row in group]
        out.append({
            'method': method,
            'batch_size': batch_size,
            'receiver_mode': mode,
            'n_model_seeds': len(group),
            'latency_ms_mean_of_seed_means': statistics.mean(values),
            'latency_ms_std_across_seeds': (
                statistics.stdev(values) if len(values) > 1 else None
            ),
            'latency_ms_min_seed_mean': min(values),
            'latency_ms_max_seed_mean': max(values),
            'direct_fallback_repetitions_total': sum(
                row['direct_fallback_repetitions'] for row in group
            ),
        })
    return out


def profile(output, *, seeds=SEEDS, batch_sizes=(1, 8),
            modes=('common_diagonal', 'native_own_ici'), warmups=5,
            repetitions=30, input_cache=None, device_name='cuda:0'):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA server required; this benchmark must not run on the laptop')
    if warmups < 1 or repetitions < 10:
        raise ValueError('Use at least one warmup and ten timed repetitions')
    device = torch.device(device_name)
    if device.type != 'cuda':
        raise ValueError('--device must select a CUDA GPU')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output = Path(output).resolve()
    manifest = json.loads((output / 'experiment.json').read_text(encoding='utf-8'))
    names = [f'{method}_seed{seed}' for method in METHODS for seed in seeds]
    missing = [name for name in names
               if name not in manifest['configs']
               or not (output / 'checkpoints' / name / 'model.pt').exists()]
    if missing:
        raise FileNotFoundError(f'Missing trained checkpoint/config: {missing}')
    # All methods must share receiver geometry and physical generation policy.
    # An optional, unused derived feature can differ in archived configs.
    reference = manifest['configs'][names[0]]
    for name in names[1:]:
        cfg = manifest['configs'][name]
        if _shared_input_generation_identity(cfg) != _shared_input_generation_identity(reference):
            raise RuntimeError(f'Nonshared input generation config: {name}')
    if input_cache is None:
        input_cache = output / 'receiver_latency_inputs.pt'
    batches, input_hashes, input_metadata = _load_or_make_inputs(
        Path(input_cache).resolve(), manifest, reference, device, batch_sizes
    )
    result_path = output / 'receiver_latency.json'
    report = {
        'complete': False,
        'protocol': {
            'scope': 'channel estimator forward plus equalizer',
            'common_diagonal': 'all models; identical data-symbol diagonal MMSE',
            'native_own_ici': (
                'only models that produce ADD coefficients; centre CFR plus '
                'four ICI side bands, PCG 96 iterations/tolerance 1e-6; '
                'direct solve on residual >1e-4, as in evaluate.py'
            ),
            'methods': list(METHODS),
            'seeds': list(seeds),
            'batch_sizes': list(batch_sizes),
            'receiver_modes': list(modes),
            'precision': 'fp32',
            'execution': 'eager, torch.inference_mode, TF32 disabled',
            'timing': 'host wall clock with CUDA synchronization before and after every repetition',
            'warmups': warmups,
            'repetitions': repetitions,
            'input_generation': 'one fixed synthetic doubly_selective batch per batch size',
            'condition': CONDITION,
            'snr_db': SNR_DB,
            'input_cache': str(Path(input_cache).resolve()),
            'input_cache_metadata': input_metadata,
            'input_sha256_by_batch_size': {str(k): v for k, v in input_hashes.items()},
            'experiment_signature': manifest['signature'],
            'profiling_script_sha256': file_hash(Path(__file__).resolve()),
        },
        'hardware': {
            'gpu': torch.cuda.get_device_name(device),
            'cuda_capability': list(torch.cuda.get_device_capability(device)),
            'total_gpu_memory_mb': torch.cuda.get_device_properties(device).total_memory / 2**20,
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'python': platform.python_version(),
            'platform': platform.platform(),
        },
        'limitations': [
            'Excludes waveform acquisition, channel simulation, pilot LS/interpolation, '
            'fast/slow-time ADD construction, host-to-device transfer, demapping and BER scoring.',
            'Requires an otherwise idle GPU; concurrent jobs and clock/thermal variation alter timing.',
            'The common diagonal and native ICI protocols answer different receiver questions '
            'and must not be merged into one speed ranking.',
            'These measurements do not establish over-the-air or real-time deployment latency.',
        ],
        'rows': [],
        'seed_summary': [],
    }
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding='utf-8'))
        if existing.get('protocol') != report['protocol']:
            raise RuntimeError(
                f'Existing latency report has a different protocol: {result_path}. '
                'Archive it before running a new protocol.'
            )
    write_json(result_path, report)
    for method in METHODS:
        for seed in seeds:
            name = f'{method}_seed{seed}'
            path = output / 'checkpoints' / name / 'model.pt'
            ck = load_checkpoint(path, map_location='cpu')
            cfg = ck['cfg']
            if cfg != manifest['configs'][name] or ck.get('selection_metric') != 'val_nmse_db':
                raise RuntimeError(f'Checkpoint provenance mismatch: {name}')
            model = SharedUserChannelEstimator(cfg).to(device).eval()
            model.load_state_dict(ck['model'], strict=True)
            if any(p.dtype != torch.float32 for p in model.parameters()):
                raise RuntimeError(f'Non-fp32 checkpoint model: {name}')
            parameter_count = sum(p.numel() for p in model.parameters())
            checkpoint_sha256 = file_hash(path)
            for B in batch_sizes:
                batch = batches[B]
                for mode in modes:
                    if mode == 'native_own_ici' and not _native_available(model):
                        report['rows'].append({
                            'method': method, 'seed': seed, 'batch_size': B,
                            'receiver_mode': mode, 'status': 'not_applicable',
                            'reason': 'No native ADD-to-ICI coefficients',
                            'checkpoint_sha256': checkpoint_sha256,
                            'input_sha256': input_hashes[B],
                        })
                        continue
                    print(f'[receiver-latency] {name} B={B} {mode}', flush=True)
                    stats = _time_one(model, cfg, batch, mode, warmups, repetitions)
                    report['rows'].append({
                        'method': method, 'seed': seed, 'batch_size': B,
                        'receiver_mode': mode, 'status': 'measured',
                        'checkpoint_sha256': checkpoint_sha256,
                        'input_sha256': input_hashes[B],
                        'parameters_total': parameter_count,
                        **stats,
                    })
                    # Completed rows are visible for a long server benchmark;
                    # only complete=true qualifies for a paper table.
                    write_json(result_path, report)
            del model, ck
            gc.collect()
            torch.cuda.empty_cache()
    report['seed_summary'] = _summarize_seeds(report['rows'])
    report['complete'] = True
    write_json(result_path, report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='results/dopnet')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 8])
    ap.add_argument('--modes', nargs='+',
                    choices=['common_diagonal', 'native_own_ici'],
                    default=['common_diagonal', 'native_own_ici'])
    ap.add_argument('--warmups', type=int, default=5)
    ap.add_argument('--repetitions', type=int, default=30)
    ap.add_argument('--input-cache', type=Path, default=None)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(SEEDS):
        ap.error(f'--seeds must be unique members of {SEEDS}')
    if len(set(args.batch_sizes)) != len(args.batch_sizes) or not set(args.batch_sizes) <= {1, 8}:
        ap.error('--batch-sizes must be unique members of {1, 8}')
    if len(set(args.modes)) != len(args.modes):
        ap.error('--modes must be unique')
    profile(args.output, seeds=tuple(args.seeds), batch_sizes=tuple(args.batch_sizes),
            modes=tuple(args.modes), warmups=args.warmups, repetitions=args.repetitions,
            input_cache=args.input_cache, device_name=args.device)


if __name__ == '__main__':
    main()
