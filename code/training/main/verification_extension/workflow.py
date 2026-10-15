"""No-retraining server workflow, with streamed frozen-test observations."""
from __future__ import annotations

from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import numpy as np

from .protocol import (METHODS, atomic_npz, check_reference, digest, file_hash,
                       read_json, select_suite, signed_json, source_hashes, write_json)


def preflight(args):
    results, experiment, evaluation = check_reference(args.reference_results)
    selected = select_suite(evaluation['cells'], args.suite)
    project = Path(args.project).resolve()
    if not (project / 'reference_replay.py').is_file():
        raise FileNotFoundError('Missing frozen_reference directory: ' + str(project))
    names = [f'{method}_seed{seed}' for method in ('dopnet', 'dopnet_lite', 'slow_time', 'fortitran')
             for seed in (42, 43, 44)]
    missing = [name for name in names if not (results / 'checkpoints' / name / 'model.pt').is_file()]
    if missing:
        raise FileNotFoundError('Latency/recheck needs the existing model.pt files: ' + ', '.join(missing))
    for cell in selected:
        with np.load(results / 'cells' / (cell['cell_id'] + '.npz'), allow_pickle=False) as z:
            if (str(z['signature'].item()) != evaluation['signature']
                or json.loads(str(z['meta'].item())) != cell
                or z['nmse'].shape != (len(evaluation['routes']), 512)
                or not np.array_equal(z['cluster_id'], np.arange(512) // 8)):
                raise RuntimeError('Invalid frozen reference score: ' + cell['cell_id'])
    report = {'status': 'metadata_pass', 'suite': args.suite, 'selected_cells': len(selected),
              'frames_per_cell': 512, 'raw_observation_storage': 'streamed; no complete test waveform archive needed',
              'source_results': str(results), 'existing_checkpoints_required': names,
              'reference_evaluation_signature': evaluation['signature'],
              'reference_experiment_signature': experiment['signature'],
              'training_performed': False, 'code_sha256': source_hashes()}
    write_json(Path(args.output) / 'preflight.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def prepare(args):
    from .replay import FrozenReplay
    results, _, evaluation = check_reference(args.reference_results)
    cells = select_suite(evaluation['cells'], args.suite)
    replay = FrozenReplay(args.project, results, cells, args.suite)
    replay.prepare_calibration(Path(args.output) / 'observation_cache')
    replay.prepare_statistical_extra(Path(args.output) / 'observation_cache', args.statistical_calibration_extra)


def calibration_arrays(args):
    from classical_extension.run_classical_cpu import read_export, read_complete, load_batch
    cache = Path(args.output) / 'observation_cache'
    manifest = read_export(cache)
    manifest['required_cp_wave_samples'] = sorted({
        int(cell['cp_len']) * int(manifest['geometry']['waveform_fft_size']) //
        int(manifest['geometry']['n_subcarriers']) for cell in manifest['cells']})
    if args.statistical_calibration_extra:
        manifest['statistical_extra_seed'] = 810000002
        manifest['statistical_extra_count'] = args.statistical_calibration_extra
    marker, _ = read_complete(cache, manifest, 'calibration')
    return [load_batch(cache, record) for record in marker['batches']], manifest


def benchmark(args, selected=False):
    from .calibration import benchmark_classical
    arrays, manifest = calibration_arrays(args)
    statistical = statistical_arrays(args, arrays, manifest)
    lock = signed_json(Path(args.output) / 'calibration_lock.json') if selected else None
    benchmark_classical(arrays, manifest, Path(args.output), cpu_threads=tuple(args.cpu_threads),
                        warmups=args.benchmark_warmups, repetitions=args.benchmark_repetitions,
                        settings=lock['settings'] if lock else None,
                        state_directory=Path(args.output) if selected else None,
                        benchmark_arrays_list=statistical)


def statistical_arrays(args, arrays, manifest):
    """Load additional independent calibration, never original test targets."""
    cache = Path(args.output) / 'observation_cache'
    extra = []
    if args.statistical_calibration_extra:
        marker = read_json(cache / 'calibration_extra/COMPLETE.json')
        if (marker['samples'] != args.statistical_calibration_extra or marker['seed'] != 810000002
                or marker['role'] != 'statistical_calibration'
                or marker['export_signature'] != manifest['export_signature']):
            raise RuntimeError('Additional independent calibration contract differs')
        offset, seen = 0, set()
        for record in marker['batches']:
            path = (cache / record['path']).resolve()
            path.relative_to(cache.resolve())
            if path in seen or record['offset'] != offset or record['samples'] <= 0:
                raise RuntimeError('Additional independent calibration chunks are duplicated or unordered')
            seen.add(path)
            if file_hash(path) != record['sha256']:
                raise RuntimeError('Additional independent calibration changed')
            with np.load(path, allow_pickle=False) as z:
                chunk_arrays = {key: z[key] for key in z.files}
            meta = json.loads(str(chunk_arrays['metadata'].item()))
            if (meta['role'] != marker['role'] or meta['calibration_seed'] != marker['seed']
                    or meta['offset'] != offset or meta['samples'] != record['samples']
                    or len(chunk_arrays['noise_var']) != record['samples']
                    or chunk_arrays['x_pilot'].shape[0] != record['samples']):
                raise RuntimeError('Additional independent calibration chunk metadata differs')
            extra.append(chunk_arrays)
            offset += record['samples']
        if offset != marker['samples']:
            raise RuntimeError('Additional independent calibration is incomplete')
        manifest['statistical_calibration_sources'] = {
            'base_seed': 810000001, 'base_samples': 128,
            'extra_seed': marker['seed'], 'extra_samples': marker['samples'],
            'extra_complete_sha256': file_hash(cache / 'calibration_extra/COMPLETE.json'),
            'extra_batches': marker['batches']}
    return arrays + extra


def calibrate(args):
    from .calibration import calibrate_classical
    arrays, manifest = calibration_arrays(args)
    statistical = statistical_arrays(args, arrays, manifest)
    policy = read_json(Path(args.output) / 'classical_device_policy.json')
    calibrate_classical(arrays, manifest, Path(args.output), policy,
                        q_grid=tuple(args.q_grid), delay_windows=((-4, 28), (-8, 40)),
                        ridge_grid=tuple(args.ridge_grid), noise_grid=tuple(args.noise_grid),
                        statistical_arrays_list=statistical)


def evaluate(args):
    # Frozen generation is imported before any public model/generator module.
    from .replay import FrozenReplay
    results, experiment, reference_eval = check_reference(args.reference_results)
    cells = select_suite(reference_eval['cells'], args.suite)
    replay = FrozenReplay(args.project, results, cells, args.suite)
    import torch
    from .classical import (BEMSettings, TorchJointBEM, geometry_from_export, geometry_key,
                            diagonal_mmse, nmse_per_frame, frame_bit_counts)
    from .calibration import load_calibrated_bank, resolve_classical_device
    out = Path(args.output)
    lock = signed_json(out / 'calibration_lock.json')
    if lock.get('status') != 'locked' or lock.get('methods') != list(METHODS):
        raise RuntimeError('All corrected references must have a complete independent calibration lock')
    if lock['code_sha256'] != source_hashes():
        raise RuntimeError('Correction implementation changed after calibration; use a new output directory')
    lock_sha = file_hash(out / 'calibration_lock.json')
    preferred = {method: resolve_classical_device(out / 'classical_device_policy.json', method)
                 for method in METHODS}
    calibration = read_json(out / 'observation_cache/export_manifest.json')
    if lock.get('export_signature') != calibration['export_signature']:
        raise RuntimeError('Calibration lock does not belong to this frozen observation protocol')
    evaluation = {'schema_version': 2, 'methods': list(METHODS), 'suite': args.suite,
                  'cells': cells, 'calibration_lock_sha256': lock_sha,
                  'reference_evaluation_signature': reference_eval['signature'],
                  'reference_experiment_signature': experiment['signature'],
                  'code_sha256': source_hashes(), 'streamed_frozen_replay': True,
                  'source_score_sha256': 'each cell binds its original saved score NPZ',
                  'common_detector': 'same diagonal multiuser MMSE for all estimators',
                  'native_detector': 'BEM: fitted full occupied-band ICI MMSE; statistical reference: unavailable',
                  'precision': 'complex128', 'training_performed': False}
    evaluation['signature'] = digest(evaluation)
    if (out / 'evaluation.json').exists() and read_json(out / 'evaluation.json') != evaluation:
        raise RuntimeError('Output already contains an incompatible corrected evaluation')
    write_json(out / 'evaluation.json', evaluation)
    banks, operators, complete = {}, {}, {}

    def estimate(method, arrays, index, selected_device):
        row = preferred[method]
        threads = int(row.get('cpu_threads') or args.cpu_threads[0])
        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
        g = geometry_from_export(replay.manifest, arrays, index)
        nv = float(arrays['noise_var'][index])
        xp, yp = arrays['x_pilot'][index], arrays['y_pilot'][index]
        if method == METHODS[0]:
            if selected_device not in banks:
                banks[selected_device] = load_calibrated_bank(out, selected_device, dtype='complex128')
            key = geometry_key(g)
            if key not in banks[selected_device]:
                raise RuntimeError('Missing independently calibrated prior for public geometry ' + key)
            item = banks[selected_device][key]
            cfr, info = item['prior'].estimate(xp, yp, nv, g, noise_scale=item['noise_scale'])
            operator = coeff = None
        else:
            settings = BEMSettings(**lock['settings'][method])
            op_key = (method, geometry_key(g), selected_device)
            if op_key not in operators:
                # Keep per-device geometry caches small during long all-suite runs.
                if len(operators) >= 8:
                    operators.clear(); gc.collect()
                operators[op_key] = TorchJointBEM(g, settings, xp.shape[0], device=selected_device, dtype='complex128')
            operator = operators[op_key]
            coeff, info = operator.fit(xp, yp, nv)
            cfr = operator.diagonal_cfr(coeff)
        return cfr, info, operator, coeff

    for number, cell in enumerate(cells, 1):
        cid = cell['cell_id']
        score_path = results / 'cells' / (cid + '.npz')
        source_sha = file_hash(score_path)
        with np.load(score_path, allow_pickle=False) as original:
            expected_realization = str(original['realization_sha256'].item())
            old_bits = original['bits'][next(i for i, route in enumerate(reference_eval['routes'])
                                            if route['method'] == 'dopnet' and route['training_seed'] == 42)].copy()
        dest = out / 'cells' / (cid + '.npz')
        marker_path = dest.with_suffix('.COMPLETE.json')
        if marker_path.exists():
            marker = read_json(marker_path)
            if (marker.get('signature') != evaluation['signature'] or marker.get('lock_sha256') != lock_sha
                or marker.get('source_score_sha256') != source_sha
                or marker.get('realization_sha256') != expected_realization
                or not dest.is_file() or marker.get('npz_sha256') != file_hash(dest)):
                raise RuntimeError('Incompatible/corrupt completed cell: ' + cid)
            complete[cid] = marker['npz_sha256']
            print('[%d/%d] %s checksum-verified scores reused' % (number, len(cells), cid), flush=True)
            continue
        started = time.perf_counter()
        score = {key: np.zeros((3, 512), dtype=np.float64)
                 for key in ('nmse', 'fit_relative_residual', 'native_relative_residual', 'ridge_lambda')}
        score.update({key: np.zeros((3, 512), dtype=np.int64)
                      for key in ('errors_diagonal', 'errors_native', 'bits', 'fit_iterations', 'native_iterations')})
        score['errors_native'][0] = -1
        used_backends, fallback_count = {}, 0
        # Exhausting this iterator performs the complete original 512-frame SHA check.
        for offset, arrays in replay.test_batches(cell):
            with torch.inference_mode():
                for mi, method in enumerate(METHODS):
                    device = preferred[method]['preferred_device']
                    for bi in range(len(arrays['noise_var'])):
                        frame = offset + bi
                        try:
                            cfr, info, op, coeff = estimate(method, arrays, bi, device)
                        except RuntimeError as error:
                            if 'out of memory' not in str(error).lower() or device == 'cpu':
                                raise
                            # Only execution changes; precision, priors and fitted basis stay fixed.
                            torch.cuda.empty_cache(); fallback_count += 1
                            device = 'cpu'
                            cfr, info, op, coeff = estimate(method, arrays, bi, device)
                        used_backends[method + '@' + device] = used_backends.get(method + '@' + device, 0) + 1
                        score['nmse'][mi, frame] = float(nmse_per_frame(cfr[None], arrays['target_h'][bi:bi + 1])[0])
                        score['fit_relative_residual'][mi, frame] = max(
                            info['relative_residual_max'], info.get('recovered_normal_relative_residual_max', 0))
                        score['fit_iterations'][mi, frame] = int(info.get('iterations', 0))
                        score['ridge_lambda'][mi, frame] = float(info.get('ridge_lambda', 0))
                        if cell['ber']:
                            nv = float(arrays['noise_var'][bi])
                            yd, labels = arrays['rx_data'][bi], arrays['tx_label'][bi:bi + 1]
                            common = diagonal_mmse(cfr, yd, nv)
                            errors, bits = frame_bit_counts(common[None], labels, cell['modulation'])
                            score['errors_diagonal'][mi, frame], score['bits'][mi, frame] = int(errors[0]), int(bits[0])
                            if int(bits[0]) != int(old_bits[frame]):
                                raise RuntimeError('Original and corrected BER denominators differ')
                            if op is not None:
                                native, native_info = op.native_mmse(coeff, yd, nv, symbol_batch_size=4)
                                ne, nb = frame_bit_counts(native[None], labels, cell['modulation'])
                                if int(nb[0]) != int(bits[0]):
                                    raise RuntimeError('Detector paths have different bit totals')
                                score['errors_native'][mi, frame] = int(ne[0])
                                score['native_relative_residual'][mi, frame] = native_info['relative_residual_max']
                                score['native_iterations'][mi, frame] = int(native_info.get('iterations', 0))
            if offset % 64 == 0:
                print('[%d/%d] %s %d/512 frames; %.1fs' %
                      (number, len(cells), cid, offset + 8, time.perf_counter() - started), flush=True)
        score.update(methods=np.asarray(METHODS), native_available=np.asarray([False, True, True]),
                     frame_index=np.arange(512, dtype=np.int64), cluster_ids=np.arange(512, dtype=np.int64) // 8,
                     cell_id=np.asarray(cid), meta=np.asarray(json.dumps(cell, sort_keys=True)),
                     realization_sha256=np.asarray(expected_realization), source_score_sha256=np.asarray(source_sha),
                     lock_sha256=np.asarray(lock_sha), signature=np.asarray(evaluation['signature']))
        # No file is committed before every generated input in this cell matches.
        atomic_npz(dest, score)
        marker = {'status': 'complete', 'schema_version': 2, 'cell_id': cid, 'meta': cell, 'samples': 512,
                  'methods': list(METHODS), 'signature': evaluation['signature'], 'lock_sha256': lock_sha,
                  'source_score_sha256': source_sha, 'realization_sha256': expected_realization,
                  'npz_sha256': file_hash(dest), 'backend_frame_counts': used_backends,
                  'oom_cpu_fallback_frames': fallback_count, 'frozen_waveform_sha256_pass': True,
                  'elapsed_seconds': time.perf_counter() - started}
        write_json(marker_path, marker)
        complete[cid] = marker['npz_sha256']
    write_json(out / 'COMPLETE.json', {'status': 'complete', 'schema_version': 2,
               'signature': evaluation['signature'], 'methods': list(METHODS),
               'suite': args.suite, 'cell_files': complete, 'calibration_lock_sha256': lock_sha,
               'source_fingerprints_verified': True, 'training_performed': False})


def latency(args):
    from .latency import benchmark
    benchmark(Path(args.output) / 'observation_cache', args.reference_results,
              Path(args.output) / 'latency', seeds=(42, 43, 44),
              batch_sizes=tuple(args.batch_sizes), warmups=args.latency_warmups,
              repetitions=args.latency_repetitions, cpu_threads=max(args.cpu_threads),
              cpu_thread_candidates=tuple(args.cpu_threads),
              latency_capture=Path(args.output) / 'observation_cache/latency_capture.npz')


def report(args):
    from .report import report
    from .bundle import bundle_results
    statistics = report(Path(args.output), args.reference_results, bootstrap=args.bootstrap,
                        latency_output=Path(args.output) / 'latency')
    bundle_results(args.output)
    return statistics
