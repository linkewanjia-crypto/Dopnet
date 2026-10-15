#!/usr/bin/env python3
"""Original PyTorch Table 5 diagnostic with immutable, relocatable inputs.

Help, dry plans and --verify-saved require only the standard library. New CPU
inference requires --execute and a new output directory outside code and data.
No training, download, NumPy model reimplementation or pickle fallback is used.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

MODELS = tuple(f'{family}_seed{seed}' for family in ('dopnet', 'cfr_derived') for seed in (42, 43, 44))
BATCHES = tuple(f'batch_{i:04d}.npz' for i in range(0, 128, 16))
CALIBRATION = Path('reproduction_inputs/support_diagnostics_20261007/raw_observations/calibration')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_child(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('An input identity escapes its declared root')
    return path


def verify_rows(root, rows):
    for row in rows:
        path = checked_child(root, row['relative'])
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Input identity differs: ' + row['relative'])
    return len(rows)


def summarize(records):
    result = {}
    for family in ('dopnet', 'cfr_derived'):
        result[family] = {}
        for group in ('all', 'static', 'dynamic'):
            selected = [r for r in records if r['model'].startswith(family + '_seed')
                        and (group == 'all' or (r['doppler'] == 0 if group == 'static' else r['doppler'] > 0))]
            if not selected:
                continue
            mean = lambda key: statistics.mean(r[key] for r in selected)
            result[family][group] = {
                'model_observation_pairs': len(selected),
                'beta_weighted': mean('beta_wmean'), 'beta_unweighted': mean('beta_mean'),
                'add_branch_power_fraction': mean('add_share'),
                'normal_nmse_linear': mean('nmse_out'),
                'normal_nmse_db': 10 * math.log10(mean('nmse_out')),
                'beta1_nmse_linear': mean('nmse_beta1'),
                'beta1_nmse_db': 10 * math.log10(mean('nmse_beta1'))}
    return result


def table5(summary):
    result = {}
    for group, values in summary.get('dopnet', {}).items():
        result[group] = [f"{values['beta_weighted']:.3f}", f"{values['beta_unweighted']:.3f}",
                         f"{100 * values['add_branch_power_fraction']:.1f}",
                         f"{values['normal_nmse_db']:.2f}", f"{values['beta1_nmse_db']:.2f}"]
    return result


def verify_saved(data_root, manifest):
    verify_rows(data_root, manifest['authoritative_saved_result'])
    row_path = data_root / manifest['authoritative_saved_result'][1]['relative']
    records = [json.loads(line) for line in row_path.read_text(encoding='utf8').splitlines() if line.strip()]
    expected_keys = {(name, batch, frame, user) for name in MODELS for batch in BATCHES
                     for frame in range(4) for user in range(2)}
    actual_keys = {(r['model'], r['file'], r['frame'], r['user']) for r in records}
    if len(records) != 384 or actual_keys != expected_keys:
        raise ValueError('The frozen 384-pair matrix is incomplete or duplicated')
    current = table5(summarize(records))
    if current != manifest['expected_table5']:
        raise ValueError('Frozen records no longer reproduce the current Table 5 rounding')
    return {'status': 'PASS_SAVED_384_PAIR_MATRIX_AND_TABLE5', 'records': len(records),
            'unique_physical_frames': 32, 'unique_user_observations': 64, 'trained_models': 6,
            'table5': current, 'new_inference_executed': False}


def arr(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)

def complex_ri(t):
    a = arr(t)
    return a[:, 0] + 1j * a[:, 1]

def trusted_inputs(z, cfg):
    """Rebuild independently using original PyTorch waveform routines."""
    from data.waveform import _pilot_ls_and_interp, _time_features, _add_backprojection
    from data.frame_layout import make_layout
    from data.add_geometry import delay_window_indices, add_delay_bins, delay_guard_bins
    xp = torch.from_numpy(np.array(z['x_pilot'], copy=True)).to(torch.complex64)
    yp = torch.from_numpy(np.array(z['y_pilot'], copy=True)).to(torch.complex64).permute(0, 2, 1, 3)
    batch, users, npilots, nf = xp.shape
    antennas = yp.shape[1]
    symbols = int(cfg['system']['frame_num_symbols'])
    layout = make_layout(symbols, int(cfg['system']['pilot_symbol_period']))
    pb = torch.tensor(layout.pilot_blocks, dtype=torch.long)
    y = torch.zeros((batch, antennas, layout.num_blocks, nf), dtype=torch.complex64)
    y.index_copy_(2, pb, yp)
    frame = torch.zeros((batch, users, layout.num_blocks, nf), dtype=torch.complex64)
    frame.index_copy_(2, pb, xp)
    nz = xp.abs() > 0
    nper = int(nz[0, 0, 0].sum())
    assert torch.all(nz.sum(-1) == nper)
    spacing = nf // nper
    pidx = nz.nonzero(as_tuple=False)[:, -1].reshape(batch, users, npilots, nper)
    pval = xp.gather(-1, pidx)
    coarse, _ = _pilot_ls_and_interp(y, layout, pidx, pval, users, antennas, nf, spacing)
    mask = torch.zeros((batch, users, symbols, nf), dtype=torch.float32)
    for ip, symbol in enumerate(layout.pilot_visible):
        mask[:, :, symbol, :].scatter_(-1, pidx[:, :, ip, :], 1.0)
    cp = int(z['cp_wave_samples'])
    age, horizon, *_ = _time_features(cfg, batch, users, symbols, antennas, nf, cp, torch.device('cpu'))
    x = torch.stack([coarse.real, coarse.imag,
                     mask[:, :, :, None, :].expand(batch, users, symbols, antennas, nf),
                     age, horizon], dim=2).float()
    add = _add_backprojection(cfg, y, frame, layout, cp,
                             delay_window_indices(nf, add_delay_bins(cfg), delay_guard_bins(cfg), device=torch.device('cpu')),
                             fast_time=True, pilot_only_received=False)
    return x, add, spacing

def torch_forward(model, x, add, beta_one=False):
    captured = {}
    def aft_hook(_module, _args, out):
        captured['aft_residual'] = out.detach()
    def add_pre(_module, args):
        captured['add_input'] = args[0].detach()
    handles = [model.aft_decoder.register_forward_hook(aft_hook),
               model.add_encoder.register_forward_pre_hook(add_pre)]
    if beta_one:
        handles.append(model.add_fusion_gate.register_forward_hook(
            lambda _module, _args, out: torch.ones_like(out)))
    try:
        out, aux = model(x, add_input=add, return_aux=True)
        coarse = torch.complex(x[:, 0], x[:, 1])
        aft = coarse + torch.complex(captured['aft_residual'][:, 0], captured['aft_residual'][:, 1])
        aft_coeff = model._aft_to_add(aft, model.physics.symbol_times_from_input(x))
        return {'coarse': coarse, 'aft_est': aft, 'aft_add': aft_coeff,
                'add_in': complex_ri(captured['add_input']),
                'add_est': complex_ri(aux['add_branch_pred']),
                'beta': aux['domain_beta_add'], 'fused': complex_ri(aux['add_fused_pred']),
                'off': aux['doppler_offset_bins'], 'phys': complex_ri(aux['physics_h']),
                'gamma': aux['domain_gamma_aft'], 'out': complex_ri(out)}
    finally:
        for handle in handles:
            handle.remove()


def run(args, code_root, data_root, output, manifest):
    # Set resource limits before importing scientific packages. Never use GPU.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '2'
    global np, torch
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.set_default_dtype(torch.float32)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code_root / 'training/main'))
    from models.dopnet import DOPNetEstimator

    output.mkdir(parents=True, exist_ok=False)
    models = MODELS[:1] if args.smoke else MODELS
    batches = BATCHES[:1] if args.smoke else BATCHES
    report = {
        'schema': 'tmlcn_portable_original_pytorch_gate_v1', 'status': 'RUNNING',
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'smoke_only': args.smoke, 'models': list(models), 'batches': list(batches),
        'runtime': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                    'device': 'cpu', 'threads': 2, 'cuda_visible_devices': '-1'},
        'inputs_verified_against_manifest': True,
        'input_manifest_sha256': sha(Path(__file__).with_name('gate_inputs.json')),
        'checkpoint_load': 'torch.load(weights_only=True,map_location=cpu); strict=True',
        'training_executed': False, 'numpy_model_executed': False,
        'scientific_source_or_frozen_results_modified': False,
        'definitions': {
            'beta_weighted': 'Per observation: sum(beta*abs(fused)^2)/sum(abs(fused)^2).',
            'add_branch_power_fraction': 'Per observation: sum(abs((1-beta)*ADD)^2)/(sum(abs(beta*AFT_ADD)^2)+sum(abs((1-beta)*ADD)^2)); this excludes the cross term.',
            'nmse': 'Per observation: sum(abs(pred-target)^2)/sum(abs(target)^2) on data symbols.',
            'aggregation': manifest['table5_authoritative_aggregation'],
            'beta1': 'Force coefficient fusion gate beta=1; retain ADD-driven offset and downstream gate. This does not isolate an exclusive causal mechanism.'},
        'batch_timings': []}
    def save():
        (output / 'REPORT.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    save()
    records = []
    try:
        for name in models:
            checkpoint = torch.load(data_root / 'weights/selected' / (name + '.pt'),
                                    map_location='cpu', weights_only=True)
            cfg = checkpoint['cfg']
            model = DOPNetEstimator(cfg).cpu().eval()
            state = {key.removeprefix('base.'): value for key, value in checkpoint['model'].items()}
            model.load_state_dict(state, strict=True)
            for batchname in batches:
                started = time.perf_counter()
                with np.load(data_root / CALIBRATION / batchname, allow_pickle=False) as archive:
                    z = {key: archive[key] for key in archive.files}
                meta = json.loads(str(z['metadata']))
                x, add, spacing = trusted_inputs(z, cfg)
                batch, users = x.shape[:2]
                if (batch, users) != (4, 2):
                    raise ValueError('The fixed calibration matrix requires four frames and two users per batch')
                xx = x.reshape(batch * users, *x.shape[2:])
                aa = add.reshape(batch * users, *add.shape[2:])
                data_symbols = [s for s in range(int(cfg['system']['frame_num_symbols']))
                                if s % int(cfg['system']['pilot_symbol_period'])]
                target = np.asarray(z['target_h']).reshape(batch * users, len(data_symbols),
                    int(cfg['system']['n_ant']), int(cfg['system']['n_subcarriers']))
                with torch.inference_mode():
                    normal = torch_forward(model, xx, aa)
                    forced = torch_forward(model, xx, aa, beta_one=True)
                for i in range(batch * users):
                    def nmse(value):
                        prediction = arr(value)[i, data_symbols]
                        return float(np.sum(np.abs(prediction - target[i]) ** 2)
                                     / np.sum(np.abs(target[i]) ** 2))
                    beta = arr(normal['beta'])[i]
                    aft = arr(normal['aft_add'])[i]
                    add_est = arr(normal['add_est'])[i]
                    fused = arr(normal['fused'])[i]
                    aft_power = np.abs(beta * aft) ** 2
                    add_power = np.abs((1 - beta) * add_est) ** 2
                    fused_power = np.abs(fused) ** 2
                    row = {
                        'model': name, 'file': batchname, 'frame': i // users, 'user': i % users,
                        'doppler': meta['condition']['doppler_hz'], 'channel': meta['condition']['channel_type'],
                        'spacing': spacing, 'snr': meta['snr_db'][i // users],
                        'nmse_out': nmse(normal['out']), 'nmse_beta1': nmse(forced['out']),
                        'nmse_coarse': nmse(normal['coarse']), 'nmse_phys': nmse(normal['phys']),
                        'nmse_aft': nmse(normal['aft_est']),
                        'beta_mean': float(beta.mean()),
                        'beta_wmean': float((beta * fused_power).sum() / fused_power.sum()),
                        'beta_gt09': float((beta > 0.9).mean()), 'beta_lt01': float((beta < 0.1).mean()),
                        'add_share': float(add_power.sum() / (aft_power.sum() + add_power.sum())),
                        'gamma_mean': float(arr(normal['gamma'])[i].mean()),
                        'gamma_dmean': float(arr(normal['gamma'])[i][data_symbols].mean())}
                    if not all(math.isfinite(v) for v in row.values() if isinstance(v, float)):
                        raise ValueError('A diagnostic metric is not finite')
                    records.append(row)
                    with (output / 'PER_OBSERVATION.jsonl').open('a', encoding='utf8') as stream:
                        stream.write(json.dumps(row) + '\n')
                elapsed = time.perf_counter() - started
                report['batch_timings'].append({'model': name, 'batch': batchname, 'seconds': elapsed})
                report['observations_complete'] = len(records)
                save()
                print(json.dumps({'model': name, 'batch': batchname, 'pairs_complete': len(records),
                                  'seconds': elapsed}), flush=True)
        report['aggregates'] = summarize(records)
        report['table5'] = table5(report['aggregates'])
        report['unique_physical_frames'] = 4 if args.smoke else 32
        report['unique_user_observations'] = 8 if args.smoke else 64
        report['table5_full_scope'] = not args.smoke
        report['table5_rounding_matches_saved'] = None if args.smoke else report['table5'] == manifest['expected_table5']
        report['status'] = ('COMPLETE_SMOKE_8_PAIRS_NOT_FULL_TABLE5' if args.smoke
                            else 'COMPLETE_FULL_384_PAIRS_ORIGINAL_PYTORCH')
        report['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (output / 'TABLE5_RECOMPUTED.json').write_text(json.dumps(report['table5'], indent=2) + '\n', encoding='utf8')
        save()
        print(json.dumps({'status': report['status'], 'pairs': len(records),
                          'table5_rounding_matches_saved': report['table5_rounding_matches_saved']}), flush=True)
    except Exception as error:
        report['status'] = 'FAILED_NO_SUCCESS_CLAIM'
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=None,
                        help='Data directory; then DOPNET_DATA_ROOT; default sibling ../data')
    parser.add_argument('--code-root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Code root; default directory containing scripts/')
    parser.add_argument('--output', type=Path, help='New external directory for new inference outputs')
    parser.add_argument('--smoke', action='store_true', help='Only dopnet_seed42 / batch_0000: four frames, eight pairs')
    parser.add_argument('--execute', action='store_true', help='Actually run fixed CPU inference; never training')
    parser.add_argument('--verify-saved', action='store_true',
                        help='Read-only SHA, 384-pair matrix and Table 5 aggregation verification; standard library only')
    args = parser.parse_args(argv)
    if args.verify_saved and (args.execute or args.smoke or args.output):
        parser.error('--verify-saved is read-only and cannot be combined with --execute, --smoke or --output')
    code_root = args.code_root.expanduser().resolve()
    data_root = (args.data_root or Path(os.environ.get('DOPNET_DATA_ROOT', str(code_root.parent / 'data')))).expanduser().resolve()
    manifest = json.loads(Path(__file__).with_name('gate_inputs.json').read_text(encoding='utf8'))
    if args.verify_saved:
        print(json.dumps(verify_saved(data_root, manifest), indent=2))
        return 0
    if args.output is None:
        parser.error('--output is required for a dry inference plan or --execute')
    output = args.output.expanduser().resolve()
    if output.exists() or any(output == root or output.is_relative_to(root) for root in (code_root, data_root)):
        raise ValueError('Use a new output directory outside both code and data')
    sources_verified = verify_rows(code_root, manifest['sources_relative_to_code_root'])
    inputs_verified = verify_rows(data_root, manifest['inputs_relative_to_data_root'])
    if not args.execute:
        print(json.dumps({'execute': False, 'code_root': str(code_root), 'data_root': str(data_root),
                          'output': str(output), 'source_identities_verified': sources_verified,
                          'data_identities_verified': inputs_verified,
                          'models': list(MODELS[:1] if args.smoke else MODELS),
                          'batches': list(BATCHES[:1] if args.smoke else BATCHES),
                          'model_observation_pairs': 8 if args.smoke else 384,
                          'device': 'cpu', 'threads': 2, 'training': False}, indent=2))
        return 0
    run(args, code_root, data_root, output, manifest)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ImportError, RuntimeError, KeyError) as error:
        print('GATE_ENTRY_ERROR: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
