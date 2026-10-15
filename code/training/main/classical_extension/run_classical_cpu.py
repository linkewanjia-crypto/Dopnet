#!/usr/bin/env python
"""CPU benchmark, independent-validation calibration, and frozen-frame scoring.

Example sequence:
  python run_classical_cpu.py --data raw_frames --output classical_results --benchmark
  python run_classical_cpu.py --data raw_frames --output classical_results --calibrate
  python run_classical_cpu.py --data raw_frames --output classical_results --evaluate

No simulator, GPU framework, neural checkpoint or network connection is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import re
import time


SCHEMA = 1
REQUIRED = ("y_pilot", "x_pilot", "target_h", "rx_data", "tx_label", "noise_var",
            "pilot_visible", "data_visible", "cp_wave_samples", "waveform_fft_size",
            "subcarrier_spacing_hz")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                              allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def create_json_once(path: Path, value):
    """Exclusive creation: no second calibration can replace a frozen choice."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                          allow_nan=False) + '\n').encode('utf-8')
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def output_writer_lock(output: Path):
    """OS-released single-writer lock; a killed process leaves no stale lock."""
    handle = (output / '.cpu_run.lock').open('a+b')
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    acquired = False
    try:
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Another CPU process is writing this output folder') from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('Another CPU process is writing this output folder') from exc
        acquired = True
        yield
    finally:
        if acquired:
            if os.name == 'nt':
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def code_hashes() -> dict:
    root = Path(__file__).resolve().parent
    return {name: file_hash(root / name) for name in (
        'run_classical_cpu.py', 'dopnet_classical/bem.py', 'dopnet_classical/metrics.py')}


def within(root: Path, relative: str) -> Path:
    p = (root / relative).resolve()
    try:
        p.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError('Batch path leaves exported-data folder') from exc
    return p


def read_export(data: Path):
    manifest = read_json(data / 'export_manifest.json')
    if manifest.get('schema_version') != SCHEMA or not manifest.get('export_signature'):
        raise RuntimeError('Unknown or unsigned raw-frame export')
    if not manifest.get('reference_provenance') or not isinstance(manifest.get('cells'), list):
        raise RuntimeError('Missing reference provenance or selected cell list')
    ids = [c['cell_id'] for c in manifest['cells']]
    if len(ids) != len(set(ids)):
        raise RuntimeError('Repeated exported cell ID')
    if any(not re.fullmatch(r'[0-9a-f]{16}', cid) for cid in ids):
        raise RuntimeError('Invalid reference cell ID')
    if manifest.get('batch_size') != 8 or manifest.get('samples_per_cell') != 512:
        raise RuntimeError('The historical test protocol requires 512 frames in original batches of 8')
    return manifest


def read_complete(data: Path, manifest: dict, role: str, cell_id=None):
    path = data / ('calibration' if role == 'calibration' else f'cells/{cell_id}') / 'COMPLETE.json'
    marker = read_json(path)
    if marker.get('schema_version') != SCHEMA or marker.get('role') != role:
        raise RuntimeError(f'Invalid completed export: {path}')
    if marker.get('export_signature') != manifest['export_signature']:
        raise RuntimeError(f'Export signature mismatch: {path}')
    if role == 'test':
        if marker.get('cell_id') != cell_id or marker.get('samples') != manifest['samples_per_cell']:
            raise RuntimeError(f'Wrong test-cell identity/count: {path}')
        if not re.fullmatch(r'[0-9a-f]{64}', marker.get('realization_sha256', '')):
            raise RuntimeError('Missing verified original realization fingerprint')
        cell = next(c for c in manifest['cells'] if c['cell_id'] == cell_id)
        if marker.get('meta') != cell:
            raise RuntimeError('Cell metadata differs from the root export manifest')
    else:
        if (marker.get('samples') != 128 or manifest.get('calibration_seed') != 810000001
            or marker.get('seed') != manifest['calibration_seed']):
            raise RuntimeError('Independent training-mixture calibration needs 128 frames and seed 810000001')
    cursor = 0
    for b in marker['batches']:
        if b['offset'] != cursor or b['samples'] < 1:
            raise RuntimeError('Exported batch offsets are not contiguous')
        if role == 'test' and b['samples'] != 8:
            raise RuntimeError('Every original test-generation cluster must retain 8 frames')
        p = within(data, b['path'])
        if not p.is_file() or file_hash(p) != b['sha256']:
            raise RuntimeError(f'Raw-frame batch checksum mismatch: {p}')
        cursor += b['samples']
    if cursor != marker['samples']:
        raise RuntimeError('Completed export has an incomplete frame count')
    return marker, path


def load_batch(data: Path, batch: dict):
    import numpy as np
    path = within(data, batch['path'])
    # Check again immediately before read; a checksum is not merely informational.
    if file_hash(path) != batch['sha256']:
        raise RuntimeError(f'Raw batch changed during processing: {path}')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    if any(k not in arrays for k in REQUIRED):
        raise RuntimeError('Raw batch is missing receiver/score tensors')
    b = batch['samples']
    x, y, h, yd, labels = (arrays[k] for k in ('x_pilot', 'y_pilot', 'target_h', 'rx_data', 'tx_label'))
    if x.ndim != 4 or y.ndim != 4 or h.ndim != 5 or yd.ndim != 4 or labels.ndim != 4:
        raise RuntimeError('Raw tensors have the wrong dimensions')
    if any(a.shape[0] != b for a in (x, y, h, yd, labels)):
        raise RuntimeError('Raw batch frame count mismatch')
    u, p, n = x.shape[1:]
    r = y.shape[2]
    d = h.shape[2]
    if y.shape != (b, p, r, n) or h.shape != (b, u, d, r, n):
        raise RuntimeError('Incompatible raw pilot/target shapes')
    if yd.shape != (b, d, r, n) or labels.shape != (b, u, d, n):
        raise RuntimeError('Incompatible received-data/label shapes')
    if arrays['noise_var'].shape != (b,) or np.any(arrays['noise_var'] < 0):
        raise RuntimeError('Invalid noise variance')
    if len(arrays['pilot_visible']) != p or len(arrays['data_visible']) != d:
        raise RuntimeError('Wrong visible-symbol index count')
    for a in (x, y, h, yd):
        if not np.iscomplexobj(a) or not np.all(np.isfinite(a)):
            raise RuntimeError('Raw receiver/target tensors must be finite complex arrays')
    if not np.issubdtype(labels.dtype, np.integer) or not np.all(np.isfinite(arrays['noise_var'])):
        raise RuntimeError('Invalid score labels or nonfinite noise')
    return arrays


def scalar_for_frame(a, frame: int, batch_size: int):
    import numpy as np
    v = np.asarray(a)
    if v.size == 1:
        return v.reshape(-1)[0].item()
    if v.shape == (batch_size,):
        return v[frame].item()
    raise RuntimeError('Numerology must be scalar or one value per frame')


def frame_geometry(manifest: dict, arrays: dict, frame: int):
    from dopnet_classical.bem import WaveformGeometry
    g = manifest['geometry']
    b, _, _, n = arrays['x_pilot'].shape
    return WaveformGeometry(
        nfft=int(scalar_for_frame(arrays['waveform_fft_size'], frame, b)),
        n_active=n,
        cp_wave_samples=int(scalar_for_frame(arrays['cp_wave_samples'], frame, b)),
        subcarrier_spacing_hz=float(scalar_for_frame(arrays['subcarrier_spacing_hz'], frame, b)),
        num_symbols=int(g['frame_num_symbols']),
        pilot_visible=tuple(int(v) for v in arrays['pilot_visible']),
        data_visible=tuple(int(v) for v in arrays['data_visible']),
        active_signed_bins=tuple(g['active_signed_bins']))


def training_envelope(manifest: dict) -> float:
    s = manifest['cfg_system']
    choices = [float(v) for key in ('doppler_hz_choices', 'threegpp_doppler_hz_choices')
               for v in s.get(key, [])]
    if not choices or max(choices) <= 0:
        raise RuntimeError('No declared training Doppler envelope in cfg_system')
    return max(choices)


def base_settings(args, manifest):
    from dopnet_classical.bem import BEMSettings
    return BEMSettings(q=args.q, delay_start=args.delay_start, delay_stop=args.delay_stop,
        training_doppler_envelope_hz=training_envelope(manifest),
        fit_max_iter=args.fit_max_iter, fit_tolerance=args.fit_tolerance,
        detector_max_iter=args.detector_max_iter, detector_tolerance=args.detector_tolerance)


def benchmark(args, manifest):
    from dopnet_classical.bem import METHODS, JointBEMOperator, diagonal_mmse, fit_joint_bem, native_mmse
    marker, _ = read_complete(args.data, manifest, 'calibration')
    arrays = load_batch(args.data, marker['batches'][0])
    g = frame_geometry(manifest, arrays, 0)
    settings = base_settings(args, manifest)
    rows = []
    for method in METHODS:
        s = replace(settings, method=method, relative_ridge=args.benchmark_ridge)
        started = time.perf_counter()
        op = JointBEMOperator(g, s, arrays['x_pilot'].shape[1])
        coeff, fit_info = fit_joint_bem(op, arrays['x_pilot'][0], arrays['y_pilot'][0], float(arrays['noise_var'][0]))
        fitted_at = time.perf_counter()
        cfr = op.diagonal_cfr(coeff)
        diagonal_mmse(cfr, arrays['rx_data'][0], float(arrays['noise_var'][0]))
        diagonal_at = time.perf_counter()
        _, native_info = native_mmse(op, coeff, arrays['rx_data'][0], float(arrays['noise_var'][0]))
        ended = time.perf_counter()
        row = {'method': method, 'fit_seconds': fitted_at - started,
               'cfr_and_diagonal_detector_seconds': diagonal_at - fitted_at,
               'native_detector_seconds': ended - diagonal_at,
               'total_seconds': ended - started, 'fit': fit_info, 'native': native_info}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    frames = len(manifest['cells']) * int(manifest['samples_per_cell'])
    report = {'schema_version': SCHEMA, 'role': 'benchmark_only', 'methods': rows,
              'threads': args.threads, 'settings': asdict(settings),
              'benchmark_relative_ridge': args.benchmark_ridge,
              'selected_cells': len(manifest['cells']), 'selected_test_frames': frames,
              'single_frame_linear_projection_hours': sum(r['total_seconds'] for r in rows) * frames / 3600,
              'note': 'Projection from one independent calibration frame only. Actual geometry, convergence, I/O and CPU throttling can change cost; this is not a completion-time guarantee.',
              'code_sha256': code_hashes()}
    atomic_json(args.output / 'benchmark.json', report)
    print('Benchmark only: no test scores or calibration lock were created.', flush=True)


def calibrate(args, manifest):
    import numpy as np
    from dopnet_classical.bem import METHODS, JointBEMOperator, SolverConvergenceError, fit_joint_bem
    from dopnet_classical.metrics import nmse_per_frame
    marker, marker_path = read_complete(args.data, manifest, 'calibration')
    settings = base_settings(args, manifest)
    grid = sorted(set(float(x) for x in args.ridge_grid))
    if not grid or any(x <= 0 or not np.isfinite(x) for x in grid):
        raise RuntimeError('Calibration ridge candidates must be positive and finite')
    lock_path = args.lock or args.output / 'calibration_lock.json'
    if lock_path.exists():
        raise RuntimeError(f'A calibration lock already exists: {lock_path}. Use it for evaluation or select a new output folder; existing choices are not overwritten.')
    results = {method: [{'relative_ridge': rho, 'frame_nmse': [], 'status': 'running'} for rho in grid]
               for method in METHODS}
    count, started = 0, time.perf_counter()
    for batch in marker['batches']:
        arrays = load_batch(args.data, batch)
        for bi in range(batch['samples']):
            g = frame_geometry(manifest, arrays, bi)
            for method in METHODS:
                op = JointBEMOperator(g, replace(settings, method=method), arrays['x_pilot'].shape[1])
                design = op.design_matrix(arrays['x_pilot'][bi])
                for candidate in results[method]:
                    if candidate['status'] == 'failed':
                        continue
                    try:
                        coeff, info = fit_joint_bem(op, arrays['x_pilot'][bi], arrays['y_pilot'][bi],
                            float(arrays['noise_var'][bi]), design=design, relative_ridge=candidate['relative_ridge'])
                    except SolverConvergenceError as exc:
                        candidate.update(status='failed', failed_frame_offset=batch['offset'] + bi,
                                         failure=str(exc))
                        print(f"Calibration candidate rejected: {method}, ridge={candidate['relative_ridge']}: {exc}", flush=True)
                        continue
                    score = float(nmse_per_frame(op.diagonal_cfr(coeff)[None], arrays['target_h'][bi:bi + 1])[0])
                    candidate['frame_nmse'].append(score)
                    candidate['max_relative_residual'] = max(candidate.get('max_relative_residual', 0), info['relative_residual_max'])
                    candidate['max_iterations'] = max(candidate.get('max_iterations', 0), info['iterations'])
            count += 1
            if count % 8 == 0:
                print(f'Calibration {count}/{marker["samples"]} independent frames, elapsed={time.perf_counter()-started:.1f}s', flush=True)
    chosen = {}
    for method in METHODS:
        eligible = []
        for candidate in results[method]:
            candidate['completed_frames'] = len(candidate['frame_nmse'])
            if candidate['status'] == 'running' and candidate['completed_frames'] == marker['samples']:
                candidate['status'] = 'complete'
                candidate['mean_linear_nmse'] = float(np.mean(candidate['frame_nmse']))
                eligible.append(candidate)
        if not eligible:
            atomic_json(args.output / 'calibration_failed.json', {'method': method, 'candidates': results[method]})
            raise RuntimeError(f'No {method} candidate converged on every calibration frame. No lock/results accepted. Increase fit solver budget in a new calibration output, then retry.')
        selected = min(eligible, key=lambda c: (c['mean_linear_nmse'], c['relative_ridge']))
        chosen[method] = asdict(replace(settings, method=method, relative_ridge=selected['relative_ridge']))
    lock = {'schema_version': SCHEMA, 'status': 'locked', 'locked': True,
            'methods': list(METHODS), 'settings': chosen, 'candidates': results,
            'criterion': 'mean linear per-frame payload CFR NMSE over all 128 independent training-mixture calibration frames',
            'ridge_rule': 'lambda = relative_ridge * noise_variance * mean(diag(D^H D)); unit-energy payload constellations',
            'calibration_samples': marker['samples'], 'calibration_seed': manifest['calibration_seed'],
            'calibration_complete_sha256': file_hash(marker_path),
            'calibration_batch_sha256': {b['path']: b['sha256'] for b in marker['batches']},
            'export_signature': manifest['export_signature'],
            'reference_provenance': manifest['reference_provenance'],
            'cfg_system_sha256': object_hash(manifest['cfg_system']),
            'geometry_sha256': object_hash(manifest['geometry']),
            'code_sha256': code_hashes(), 'elapsed_seconds': time.perf_counter() - started,
            'interpretation': 'These are conventional joint BEM ridge estimators adapted to the existing public pilot waveform; no original-paper pilot design or channel-statistics oracle is claimed.'}
    lock['signature'] = object_hash(lock)
    create_json_once(lock_path, lock)
    print(f'Locked independent-calibration choices: {lock_path}', flush=True)
    for method, s in chosen.items():
        print(f'{method}: relative_ridge={s["relative_ridge"]}', flush=True)


def validated_lock(args, manifest):
    lock_path = args.lock or args.output / 'calibration_lock.json'
    lock = read_json(lock_path)
    unsigned = {k: v for k, v in lock.items() if k != 'signature'}
    if lock.get('schema_version') != SCHEMA or lock.get('status') != 'locked' or lock.get('signature') != object_hash(unsigned):
        raise RuntimeError('Invalid calibration lock')
    if lock.get('code_sha256') != code_hashes():
        raise RuntimeError('Estimator/runner/scoring code changed after calibration')
    if lock.get('export_signature') != manifest['export_signature'] or lock.get('reference_provenance') != manifest['reference_provenance']:
        raise RuntimeError('Calibration lock belongs to a different reference export')
    if lock.get('cfg_system_sha256') != object_hash(manifest['cfg_system']) or lock.get('geometry_sha256') != object_hash(manifest['geometry']):
        raise RuntimeError('Training distribution/geometry changed since calibration')
    _, marker_path = read_complete(args.data, manifest, 'calibration')
    if file_hash(marker_path) != lock['calibration_complete_sha256']:
        raise RuntimeError('Calibration dataset changed after lock')
    return lock, lock_path


def evaluate(args, manifest):
    import numpy as np
    from dopnet_classical.bem import METHODS, BEMSettings, JointBEMOperator, diagonal_mmse, fit_joint_bem, native_mmse
    from dopnet_classical.metrics import frame_bit_counts, nmse_per_frame
    lock, lock_path = validated_lock(args, manifest)
    if lock['methods'] != list(METHODS):
        raise RuntimeError('Both predeclared classical methods must be locked')
    lock_sha = file_hash(lock_path)
    # Reporting reads the canonical output-local lock. Preserve its exact bytes
    # when --lock points elsewhere so every existing SHA binding stays valid.
    canonical_lock = args.output / 'calibration_lock.json'
    if canonical_lock.resolve() != lock_path.resolve():
        if canonical_lock.exists():
            if file_hash(canonical_lock) != lock_sha:
                raise RuntimeError('Output-local calibration lock differs from --lock')
        else:
            fd = os.open(str(canonical_lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(lock_path.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
    settings = {method: BEMSettings(**lock['settings'][method]) for method in METHODS}
    cells = list(manifest['cells'])
    if args.cell_id:
        requested = set(args.cell_id)
        missing = requested - {c['cell_id'] for c in cells}
        if missing:
            raise RuntimeError(f'Unknown requested cell IDs: {sorted(missing)}')
        cells = [c for c in cells if c['cell_id'] in requested]
    args.output.mkdir(parents=True, exist_ok=True)
    result_dir = args.output / 'cells'
    result_dir.mkdir(exist_ok=True)
    evaluation = {'schema_version': SCHEMA, 'methods': list(METHODS),
                  'calibration_lock_sha256': lock_sha, 'export_signature': manifest['export_signature'],
                  'reference_provenance': manifest['reference_provenance'],
                  'native_detector': 'full occupied-band joint MMSE from fitted BEM; no oracle or center replacement',
                  'code_sha256': code_hashes()}
    evaluation['signature'] = object_hash(evaluation)
    path = args.output / 'evaluation.json'
    if path.exists() and read_json(path) != evaluation:
        raise RuntimeError('An incompatible evaluation already occupies this output directory')
    atomic_json(path, evaluation)
    cell_files = {}
    for number, meta in enumerate(cells, 1):
        cid = meta['cell_id']
        marker, marker_path = read_complete(args.data, manifest, 'test', cid)
        source_sha = file_hash(marker_path)
        dest = result_dir / f'{cid}.npz'
        done_path = result_dir / f'{cid}.COMPLETE.json'
        if dest.exists() and not done_path.exists():
            # A kill after atomic NPZ commit but before the completion marker is
            # recoverable. Retain that unaccepted file for traceability, then
            # recompute the cell rather than treating it as a valid result.
            quarantine = dest.with_name(f'{cid}.uncommitted.{file_hash(dest)[:12]}.{time.time_ns()}.npz')
            dest.resolve().relative_to(args.output.resolve())
            quarantine.resolve().relative_to(args.output.resolve())
            dest.rename(quarantine)
            print(f'{cid}: preserved uncommitted result as {quarantine.name}; cell will be recomputed', flush=True)
        if dest.exists() or done_path.exists():
            if not (dest.exists() and done_path.exists()):
                raise RuntimeError(f'Uncommitted cell result exists: {cid}. It is never treated as complete.')
            done = read_json(done_path)
            if (done.get('signature') != evaluation['signature'] or done.get('source_complete_sha256') != source_sha
                or done.get('lock_sha256') != lock_sha or done.get('npz_sha256') != file_hash(dest)):
                raise RuntimeError(f'Incompatible/corrupt completed CPU cell: {cid}')
            cell_files[cid] = file_hash(dest)
            print(f'[{number}/{len(cells)}] {cid}: checksum-verified complete cell reused', flush=True)
            continue
        started = time.perf_counter()
        m, f = len(METHODS), marker['samples']
        result = {k: np.empty((m, f), dtype=np.float64) for k in ('nmse', 'fit_relative_residual', 'ridge_lambda')}
        for k in ('errors_diagonal', 'errors_native', 'bits', 'fit_iterations', 'native_iterations'):
            result[k] = np.zeros((m, f), dtype=np.int64)
        result['native_relative_residual'] = np.zeros((m, f), dtype=np.float64)
        for batch in marker['batches']:
            arrays = load_batch(args.data, batch)
            for bi in range(batch['samples']):
                offset = batch['offset'] + bi
                g = frame_geometry(manifest, arrays, bi)
                for mi, method in enumerate(METHODS):
                    op = JointBEMOperator(g, settings[method], arrays['x_pilot'].shape[1])
                    coeff, info = fit_joint_bem(op, arrays['x_pilot'][bi], arrays['y_pilot'][bi], float(arrays['noise_var'][bi]))
                    cfr = op.diagonal_cfr(coeff)
                    result['nmse'][mi, offset] = nmse_per_frame(cfr[None], arrays['target_h'][bi:bi + 1])[0]
                    result['fit_relative_residual'][mi, offset] = info['relative_residual_max']
                    result['fit_iterations'][mi, offset] = info['iterations']
                    result['ridge_lambda'][mi, offset] = info['ridge_lambda']
                    if meta['ber']:
                        common = diagonal_mmse(cfr, arrays['rx_data'][bi], float(arrays['noise_var'][bi]))
                        native, native_info = native_mmse(op, coeff, arrays['rx_data'][bi], float(arrays['noise_var'][bi]))
                        labels = arrays['tx_label'][bi:bi + 1]
                        errors, bits = frame_bit_counts(common[None], labels, meta['modulation'])
                        native_errors, native_bits = frame_bit_counts(native[None], labels, meta['modulation'])
                        if int(bits[0]) != int(native_bits[0]):
                            raise RuntimeError('Inconsistent bit totals between detector paths')
                        result['errors_diagonal'][mi, offset] = errors[0]
                        result['errors_native'][mi, offset] = native_errors[0]
                        result['bits'][mi, offset] = bits[0]
                        result['native_relative_residual'][mi, offset] = native_info['relative_residual_max']
                        result['native_iterations'][mi, offset] = native_info['iterations']
            completed = batch['offset'] + batch['samples']
            print(f'[{number}/{len(cells)}] {cid}: {completed}/{f} frames, elapsed={time.perf_counter()-started:.1f}s', flush=True)
        result.update(methods=np.asarray(METHODS), frame_index=np.arange(f, dtype=np.int64),
                      cluster_ids=np.arange(f, dtype=np.int64) // int(manifest['batch_size']),
                      realization_sha256=np.asarray(marker['realization_sha256']),
                      cell_id=np.asarray(cid), lock_sha256=np.asarray(lock_sha),
                      source_complete_sha256=np.asarray(source_sha),
                      signature=np.asarray(evaluation['signature']))
        tmp = dest.with_name(dest.name + '.tmp')
        with tmp.open('wb') as stream:
            np.savez_compressed(stream, **result)
        tmp.replace(dest)
        done = {'schema_version': SCHEMA, 'status': 'complete', 'cell_id': cid,
                'samples': f, 'methods': list(METHODS), 'signature': evaluation['signature'],
                'realization_sha256': marker['realization_sha256'], 'lock_sha256': lock_sha,
                'source_complete_sha256': source_sha, 'npz_sha256': file_hash(dest),
                'elapsed_seconds': time.perf_counter() - started}
        atomic_json(done_path, done)
        cell_files[cid] = done['npz_sha256']
    # A root completion marker covers all cells currently declared in the export,
    # never a requested subset or an interrupted partial matrix.
    for meta in manifest['cells']:
        cid = meta['cell_id']
        dest, done_path = result_dir / f'{cid}.npz', result_dir / f'{cid}.COMPLETE.json'
        if not (dest.exists() and done_path.exists()):
            print('Selected cells finished. Full exported matrix is incomplete; no root COMPLETE.json or paper-ready assertion created.', flush=True)
            return
        done = read_json(done_path)
        marker, marker_path = read_complete(args.data, manifest, 'test', cid)
        if (done.get('signature') != evaluation['signature'] or done.get('lock_sha256') != lock_sha
            or done.get('source_complete_sha256') != file_hash(marker_path)
            or done.get('realization_sha256') != marker['realization_sha256']
            or done.get('npz_sha256') != file_hash(dest)):
            raise RuntimeError('A cell needed for root completion is incompatible/corrupt')
        cell_files[cid] = done['npz_sha256']
    atomic_json(args.output / 'COMPLETE.json', {
        'schema_version': SCHEMA, 'status': 'complete', 'methods': list(METHODS),
        'signature': evaluation['signature'], 'cell_files': cell_files,
        'calibration_lock_sha256': lock_sha,
        'export_manifest_sha256': file_hash(args.data / 'export_manifest.json'),
        'export_signature': manifest['export_signature'], 'cells': len(cell_files),
        'samples_per_cell': manifest['samples_per_cell']})
    print(f'Full exported matrix completed and bound to calibration lock: {args.output}', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', type=Path, required=True, help='Verified server raw-frame export directory')
    p.add_argument('--output', type=Path, required=True)
    stage = p.add_mutually_exclusive_group(required=True)
    stage.add_argument('--benchmark', action='store_true')
    stage.add_argument('--calibrate', action='store_true')
    stage.add_argument('--evaluate', action='store_true')
    p.add_argument('--lock', type=Path, help='Existing lock for evaluation; new lock path for calibration')
    p.add_argument('--threads', type=int, default=2, help='Process-local BLAS/OpenMP thread cap')
    p.add_argument('--q', type=int, default=15)
    p.add_argument('--delay-start', type=int, default=-8)
    p.add_argument('--delay-stop', type=int, default=40, help='Exclusive delay support end')
    p.add_argument('--ridge-grid', type=float, nargs='+', default=[0.01, 0.1, 1, 10, 100])
    p.add_argument('--benchmark-ridge', type=float, default=1.0)
    p.add_argument('--fit-max-iter', type=int, default=256)
    p.add_argument('--fit-tolerance', type=float, default=1e-6)
    p.add_argument('--detector-max-iter', type=int, default=128)
    p.add_argument('--detector-tolerance', type=float, default=1e-6)
    p.add_argument('--cell-id', action='append', help='Evaluation-only work selection; does not relax complete-matrix reporting')
    return p


def main():
    args = parser().parse_args()
    if args.threads < 1:
        raise RuntimeError('--threads must be positive')
    # Set before importing NumPy/SciPy. This affects only this process.
    for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(args.threads)
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    if args.data == args.output:
        raise RuntimeError('Result output must differ from the immutable raw export directory')
    args.output.mkdir(parents=True, exist_ok=True)
    with output_writer_lock(args.output):
        manifest = read_export(args.data)
        if args.benchmark:
            benchmark(args, manifest)
        elif args.calibrate:
            calibrate(args, manifest)
        else:
            evaluate(args, manifest)


if __name__ == '__main__':
    main()
