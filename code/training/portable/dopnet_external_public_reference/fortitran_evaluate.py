"""Score fixed 40-epoch equal-budget checkpoints on unchanged frozen test frames.

Only the new predictions are computed. Existing DOP-Net and FortiTran scores are
copied with source-file hashes, for paired comparisons and three-seed intervals.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

if not __package__:
    _package_dir = Path(__file__).resolve().parent
    sys.path[:] = [value for value in sys.path if Path(value or '.').resolve() != _package_dir]
    sys.path.insert(0, str(_package_dir.parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', type=Path, required=True)
    ap.add_argument('--reference-results', type=Path, required=True)
    ap.add_argument('--training-output', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--suite', choices=('manuscript', 'all'), default='manuscript')
    ap.add_argument('--bootstrap', type=int, default=2000)
    args = ap.parse_args()
    sys.path.insert(0, str(args.project.resolve()))
    import numpy as np
    import torch
    from verification_extension import protocol as P
    from reference_replay import update_waveform_digest
    from dopnet_public_reference import ReferenceResults, check_reference
    from data.generator import BatchChannelGenerator
    from models.receiver import SharedUserChannelEstimator, select_data_csi, linear_multiuser_equalize
    from utils.metrics import nmse_per_sample
    from utils.checkpoints import load_checkpoint
    from train import model_inputs
    from evaluate import frame_bit_counts
    from experiment_protocol import TEST_SEED, experiment_cells, digest
    from aggregate_results import aggregate

    if not torch.cuda.is_available():
        raise RuntimeError('Run scoring on the CUDA server, never the local laptop')
    reference_root, experiment, old_evaluation = check_reference(args.reference_results)
    reference = ReferenceResults(reference_root)
    from budget_matched_baselines.recipe import signed_metadata, validate_manifest
    from budget_matched_baselines.verify_author_delivery import (
        check_budget_marker, check_history, checkpoint_metadata, CLAIM, MAX_EPOCHS, TERMINATION,
    )
    recipe_path = args.training_output / 'recipe_manifest.json'
    recipe = signed_metadata(recipe_path)
    if (recipe['reference_experiment_signature'] != experiment['signature']
            or recipe['reference_evaluation_signature'] != old_evaluation['signature']):
        raise RuntimeError('Recipe is bound to another historical experiment')
    # These are the entire original generator observations, including tensors
    # that FortiTran does not consume, because all original hashes must match.
    cfg = copy.deepcopy(experiment['configs']['dopnet_seed42'])
    cfg['model']['ablation_profile'] = 'full'
    cells, skipped = experiment_cells(cfg)
    reference.validate_protocol(experiment, cfg, cells, skipped)
    cells = P.select_suite(cells, args.suite)
    old_indices = [i for i, route in enumerate(old_evaluation['routes'])
                   if route['method'] in ('dopnet', 'fortitran')]
    old_routes = [old_evaluation['routes'][i] for i in old_indices]
    old_indices = reference.project_indices(old_indices)
    for method in ('dopnet', 'fortitran'):
        if [r['training_seed'] for r in old_routes if r['method'] == method] != [42, 43, 44]:
            raise RuntimeError('Historical comparison checkpoints must contain seeds 42,43,44')
    routes = old_routes + [dict(method='fortitran_budget_matched', training_seed=s, trained=True)
                           for s in (42, 43, 44)]
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    models, model_cfgs, checkpoint_hashes = [], [], {}
    for seed in (42, 43, 44):
        directory = args.training_output / 'checkpoints' / f'fortitran_budget_matched_seed{seed}'
        marker = P.read_json(directory / 'complete.json')
        check_budget_marker(marker, seed, recipe['signature'])
        if marker.get('last_checkpoint_sha256') != P.file_hash(directory / 'last.pt'):
            raise RuntimeError('Fixed-budget terminal checkpoint SHA differs from completion marker')
        path = directory / 'model.pt'
        checkpoint = load_checkpoint(path)
        model_cfg = checkpoint['cfg']
        _, prepared_cfg, _ = validate_manifest(recipe_path, args.project, seed)
        if model_cfg != prepared_cfg:
            raise RuntimeError('Checkpoint configuration differs from its prepared recipe')
        if marker.get('selected_checkpoint_sha256') != P.file_hash(path):
            raise RuntimeError('Best checkpoint SHA differs from completion marker')
        if checkpoint.get('recipe_signature') != recipe['signature']:
            raise RuntimeError('Checkpoint is bound to another training recipe')
        if model_cfg['model']['estimator'] != 'fortitran' or checkpoint.get('selection_metric') != 'val_mse':
            raise RuntimeError('Source-guided equal-budget checkpoint must be selected by validation MSE')
        if int(model_cfg['seed']) != seed or model_cfg['system'] != cfg['system']:
            raise RuntimeError('Checkpoint seed or physical geometry differs')
        if checkpoint.get('validation_cache_sha256') != recipe['validation']['sha256']:
            raise RuntimeError('Training validation cache binding differs')
        with (directory / 'history.csv').open(encoding='utf-8-sig', newline='') as stream:
            history = list(csv.DictReader(stream))
        last = checkpoint_metadata((directory / 'last.pt').read_bytes(), directory.name + '.last.pt')
        check_history(history, checkpoint, last, marker, directory.name)
        model = SharedUserChannelEstimator(model_cfg).to(device).eval()
        model.load_state_dict(checkpoint['model'], strict=True)
        models.append(model)
        model_cfgs.append(model_cfg)
        checkpoint_hashes[str(seed)] = P.file_hash(path)
    payload = dict(routes=routes, cells=cells, suite=args.suite,
                   reference_evaluation_signature=old_evaluation['signature'],
                   reference_experiment_signature=experiment['signature'],
                   fixed_test_reference=reference.provenance(),
                   training_recipe_sha256=P.file_hash(args.training_output / 'recipe_manifest.json'),
                   checkpoint_sha256=checkpoint_hashes,
                   training_claim=CLAIM, training_termination=TERMINATION, training_max_epochs=MAX_EPOCHS,
                   evaluator_sha256=P.file_hash(Path(__file__)),
                   original_evaluator_sha256=P.file_hash(Path(__import__('budget_matched_baselines.evaluate_author', fromlist=['__file__']).__file__)),
                   portable_evaluation_schema='dopnet_external_public_reference_v1',
                   public_reference_view=reference.view_provenance(),
                   samples_per_cell=512, batch_size=8,
                   test_seed=TEST_SEED, receiver_protocols=['common_diagonal', 'native_own_ici_else_diagonal'])
    signature = digest(payload)
    out = args.output.resolve()
    protected = (reference_root.resolve(), args.training_output.resolve(), Path(__file__).resolve().parents[4])
    if any(out == path or path in out.parents for path in protected):
        raise RuntimeError('Use an independent external evaluation output')
    out.mkdir(parents=True, exist_ok=True)
    with P.writer_lock(out, '.budget_matched_evaluation.lock'):
        metadata = dict(**payload, signature=signature)
        if (out / 'evaluation.json').exists() and P.read_json(out / 'evaluation.json') != metadata:
            raise RuntimeError('Evaluation identity changed; select another output directory')
        P.write_json(out / 'evaluation.json', metadata)
        generator = BatchChannelGenerator(cfg, device, TEST_SEED)
        for number, cell in enumerate(cells, 1):
            path = out / 'cells' / (cell['cell_id'] + '.npz')
            done_path = path.with_suffix('.COMPLETE.json')
            source = reference_root / 'cells' / path.name
            source_hash = reference.source_score_sha256(cell)
            public_source_hash = reference.actual_score_sha256(cell)
            if path.exists() and done_path.exists():
                done = P.read_json(done_path)
                if (done.get('signature') != signature or done.get('source_score_sha256') != source_hash
                        or done.get('public_reference_score_sha256') != public_source_hash
                        or done.get('score_sha256') != P.file_hash(path)):
                    raise RuntimeError('Stale or damaged new score cell')
                with np.load(path, allow_pickle=False) as z:
                    if z['nmse'].shape != (len(routes), 512) or z['signature'].item() != signature:
                        raise RuntimeError('Invalid saved score layout')
                    reference.assert_fingerprint(cell, z['realization_sha256'].item())
                continue
            started = time.perf_counter()
            with np.load(source, allow_pickle=False) as z:
                if z['signature'].item() != reference.score_signature:
                    raise RuntimeError('Source score identity differs')
                arrays = {'nmse': np.empty((len(routes), 512), dtype=np.float64)}
                arrays['nmse'][:len(old_routes)] = z['nmse'][old_indices]
                if cell['ber']:
                    for key in ('errors_diagonal', 'errors_native', 'bits', 'native_residual_max', 'native_direct_fallback'):
                        arrays[key] = np.zeros((len(routes), 512), dtype=z[key].dtype)
                        arrays[key][:len(old_routes)] = z[key][old_indices]
                    for key in ('errors_noise_minus3db', 'errors_noise_plus3db'):
                        if key in z:
                            arrays[key] = np.zeros((len(routes), 512), dtype=z[key].dtype)
                            arrays[key][:len(old_routes)] = z[key][old_indices]
                arrays['native_available'] = np.zeros(len(routes), dtype=np.int8)
                arrays['native_available'][:len(old_routes)] = z['native_available'][old_indices]
                expected_bits = z['bits'][old_indices[0]].copy() if cell['ber'] else None
            waveform_hash = hashlib.sha256()
            with torch.inference_mode():
                for start in range(0, 512, 8):
                    seed = TEST_SEED + int(digest(dict(cell=cell['cell_id'], offset=start))[:15], 16)
                    rng = torch.Generator(device=device).manual_seed(seed)
                    cond = dict(channel_type=cell['channel'], modulation=cell['modulation'],
                                delay_spread_ns=cell['delay_spread_ns'], doppler_hz=cell['doppler_hz'],
                                cp_len=cell['cp_len'], pilot_spacing=cell['pilot_spacing'])
                    batch = generator.simulate(8, cond, torch.full((8,), cell['snr_db'], device=device), rng, seed + 1000000001)
                    update_waveform_digest(waveform_hash, batch)
                    target = select_data_csi(batch['target_h'], cfg)
                    for index, (model, model_cfg) in enumerate(zip(models, model_cfgs), len(old_routes)):
                        prediction, _ = model(batch['input'], return_aux=True, **model_inputs(batch))
                        prediction = select_data_csi(prediction, model_cfg)
                        values = nmse_per_sample(prediction, target).cpu().numpy()
                        if not np.isfinite(values).all():
                            raise RuntimeError('Nonfinite source-guided equal-budget NMSE')
                        arrays['nmse'][index, start:start+8] = values
                        if cell['ber']:
                            symbols, _ = linear_multiuser_equalize(prediction, batch['rx_data'], batch['noise_var'], method='mmse')
                            if not torch.isfinite(symbols).all():
                                raise RuntimeError('Nonfinite equalizer output')
                            errors, bits = frame_bit_counts(symbols, batch['tx_label'], cell['modulation'])
                            if not np.array_equal(bits, expected_bits[start:start+8]):
                                raise RuntimeError('Frozen BER denominator differs')
                            arrays['errors_diagonal'][index, start:start+8] = errors
                            arrays['errors_native'][index, start:start+8] = errors
                            arrays['bits'][index, start:start+8] = bits
                            for db, key in ((-3, 'errors_noise_minus3db'), (3, 'errors_noise_plus3db')):
                                if key in arrays:
                                    z, _ = linear_multiuser_equalize(prediction, batch['rx_data'], batch['noise_var'] * 10**(db/10), method='mmse')
                                    if not torch.isfinite(z).all():
                                        raise RuntimeError('Nonfinite noise sensitivity output')
                                    arrays[key][index, start:start+8] = frame_bit_counts(z, batch['tx_label'], cell['modulation'])[0]
                    del batch, target, prediction
            reference.assert_fingerprint(cell, waveform_hash.hexdigest())
            arrays.update(cluster_id=np.arange(512)//8, signature=np.array(signature),
                          meta=np.array(json.dumps(cell)), realization_sha256=np.array(waveform_hash.hexdigest()),
                          source_score_sha256=np.array(source_hash),
                          public_reference_score_sha256=np.array(public_source_hash))
            P.atomic_npz(path, arrays)
            P.write_json(done_path, dict(signature=signature, source_score_sha256=source_hash,
                                        public_reference_score_sha256=public_source_hash,
                                        score_sha256=P.file_hash(path), realization_sha256=waveform_hash.hexdigest()))
            print(f'[{number}/{len(cells)}] frozen SHA PASS {cell["cell_id"]}; {time.perf_counter()-started:.1f}s', flush=True)
        P.write_json(out / 'evaluation_complete.json', dict(signature=signature, cells=len(cells)))
        aggregate(out, args.bootstrap)
        P.write_json(out / 'COMPLETE.json', dict(status='complete', signature=signature,
                                               cells=len(cells), checkpoint_sha256=checkpoint_hashes))


if __name__ == '__main__':
    main()
