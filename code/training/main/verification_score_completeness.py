"""Supplementary read-only checks of corrected classical score arrays.

Use after verification_gpu.verify_delivery succeeds. This independent local
checker reads the final ZIP and original score files only: no torch, waveform
generation, inference, calibration or recomputation of statistics. It follows
the exact verification_extension.workflow.evaluate NPZ writer dialect.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import unittest
from zipfile import ZipFile

import numpy as np


METHODS = ('pilot_covariance_lmmse', 'oce_bem_joint_cholesky', 'dpss_bem_joint_cholesky')
FLOAT_KEYS = ('nmse', 'fit_relative_residual', 'native_relative_residual', 'ridge_lambda')
INT_KEYS = ('errors_diagonal', 'errors_native', 'bits', 'fit_iterations', 'native_iterations')
IDENTITY_KEYS = ('methods', 'native_available', 'frame_index', 'cluster_ids', 'cell_id',
                 'meta', 'realization_sha256', 'source_score_sha256', 'lock_sha256', 'signature')
EXACT_KEYS = set(FLOAT_KEYS + INT_KEYS + IDENTITY_KEYS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def signed(value, label):
    require(value.get('signature') == digest({k: v for k, v in value.items() if k != 'signature'}),
            'Invalid signed metadata: ' + label)
    return value


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def selected_cells(cells):
    dynamic = {'doubly_selective', 'tdl_b', 'tdl_c', 'tdl_d', 'cdl_b', 'cdl_c',
               'cdl_d', 'tdl_a', 'tdl_e', 'cdl_a', 'cdl_e'}
    result = []
    for m in cells:
        nominal = m['cp_len'] == 16 and m['pilot_spacing'] == 4 and m['delay_spread_ns'] == 300.
        main = nominal and m['modulation'] == 'qpsk' and 'nmse' in m['studies']
        core = (nominal and m['channel'] in dynamic and m['doppler_hz'] in (800., 1200., 1600.)
                and m['modulation'] == '16qam' and m['snr_db'] == 20. and 'receiver' in m['studies'])
        if main or core:
            result.append(m)
    require(len(result) == 269 and len({m['cell_id'] for m in result}) == 269,
            'Need exactly 269 distinct frozen manuscript conditions')
    require(sum(m['ber'] is True for m in result) == 33
            and sum(m['ber'] is False for m in result) == 236,
            'Expected 33 scored BER conditions and 236 NMSE-only conditions')
    return result


def expected_bits(system, modulation):
    """Public layout formula, matching labels [B,U,data-symbols,subcarriers]."""
    for key in ('n_users', 'n_subcarriers', 'frame_num_symbols', 'pilot_symbol_period'):
        value = system[key]
        require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
                'Invalid public frame geometry: ' + key)
    visible, period = system['frame_num_symbols'], system['pilot_symbol_period']
    require(visible >= 2 and period >= 2 and (visible - 1) % period == 0,
            'Public visible frame must begin/end with pilot occasions')
    pilots = len(range(0, visible, period))
    bps = {'bpsk': 1, 'qpsk': 2, '16qam': 4, '64qam': 6}[modulation]
    return system['n_users'] * (visible - pilots) * system['n_subcarriers'] * bps


def matrix(z, key, dtype, shape=(3, 512)):
    require(key in z, 'Missing score array: ' + key)
    value = np.asarray(z[key])
    require(value.shape == shape, 'Incorrect dimensions: ' + key)
    require(value.dtype == np.dtype(dtype), 'Incorrect writer dtype: ' + key)
    return value


def validate_arrays(new, old, meta, reference_route_count, dopnet42_index,
                    bit_total, settings):
    """Array checks only; also used with tiny synthesized score fixtures."""
    require(set(new) == EXACT_KEYS, 'Unknown/missing corrected NPZ fields')
    shape = np.asarray(new['nmse']).shape
    require(shape == (3, 512), 'Incorrect corrected NMSE dimensions')
    floats = {k: matrix(new, k, 'float64') for k in FLOAT_KEYS}
    ints = {k: matrix(new, k, 'int64') for k in INT_KEYS}
    for key, value in floats.items():
        require(np.isfinite(value).all() and (value >= 0).all(),
                'Nonfinite or negative floating diagnostic: ' + key)
    require(np.asarray(new['methods']).shape == (3,)
            and np.asarray(new['methods']).dtype.kind in 'US'
            and np.asarray(new['methods']).tolist() == list(METHODS), 'Incorrect method order')
    native = np.asarray(new['native_available'])
    require(native.dtype == np.dtype(bool) and native.shape == (3,)
            and np.array_equal(native, [False, True, True]), 'Incorrect native interface availability')
    for key, value in (('frame_index', np.arange(512)), ('cluster_ids', np.arange(512) // 8)):
        require(np.asarray(new[key]).dtype == np.dtype('int64')
                and np.array_equal(new[key], value), 'Incorrect frame/cluster metadata: ' + key)
    require(np.asarray(old['nmse']).shape == (reference_route_count, 512), 'Original NMSE dimensions differ')
    require(np.asarray(old['cluster_id']).dtype.kind in 'iu'
            and np.array_equal(old['cluster_id'], np.arange(512) // 8), 'Original cluster metadata differ')
    for key in ('fit_iterations', 'native_iterations'):
        # The frozen writer records direct Cholesky iterations=0. Refinement
        # counts are a different quantity and must not be inferred from this.
        require((ints[key] == 0).all(), 'Direct-solver iteration field differs: ' + key)
    require((floats['ridge_lambda'][0] == 0).all(), 'Statistical method has no BEM ridge field')
    require((floats['ridge_lambda'][1:] > 0).all(), 'BEM ridge regularization must be positive')
    for mi, method in enumerate(METHODS):
        fit_tol = 1e-6 if mi == 0 else settings[method]['fit_tolerance']
        require(math.isfinite(fit_tol) and 0 < fit_tol <= 1e-6, 'Invalid frozen fit tolerance')
        require((floats['fit_relative_residual'][mi] <= fit_tol).all(),
                'Fit residual exceeds frozen tolerance: ' + method)
    require((floats['native_relative_residual'][0] == 0).all(),
            'Unavailable statistical native receiver must have unscored diagnostics')
    require((ints['errors_native'][0] == -1).all(),
            'Unavailable statistical native BER must use -1, never a scored zero')
    require(type(meta['ber']) is bool, 'Cell BER eligibility must be a Boolean')
    if meta['ber']:
        require(bit_total > 0, 'Scored BER must have a positive public-layout denominator')
        source_bits = np.asarray(old['bits'])
        require(source_bits.shape == (reference_route_count, 512)
                and source_bits.dtype.kind in 'iu' and (source_bits == bit_total).all(),
                'Original BER denominators differ from the public geometry')
        require((ints['bits'] == source_bits[dopnet42_index]).all()
                and (ints['bits'] == bit_total).all(), 'Corrected/source BER denominators differ')
        require((ints['errors_diagonal'] >= 0).all()
                and (ints['errors_diagonal'] <= ints['bits']).all(), 'Invalid diagonal bit-error counts')
        require((ints['errors_native'][1:] >= 0).all()
                and (ints['errors_native'][1:] <= ints['bits'][1:]).all(), 'Invalid BEM native bit-error counts')
        for mi, method in enumerate(METHODS[1:], 1):
            limit = settings[method]['detector_tolerance']
            require(math.isfinite(limit) and 0 < limit <= 1e-6, 'Invalid frozen detector tolerance')
            require((floats['native_relative_residual'][mi] <= limit).all(),
                    'Native residual exceeds frozen tolerance: ' + method)
    else:
        # These fields exist because the corrected writer initializes arrays
        # for every cell. They are unscored placeholders, not zero BER scores.
        require('bits' not in old and 'errors_diagonal' not in old and 'errors_native' not in old,
                'Historical NMSE-only score unexpectedly contains BER fields')
        require((ints['bits'] == 0).all() and (ints['errors_diagonal'] == 0).all()
                and (ints['errors_native'][1:] == 0).all()
                and (floats['native_relative_residual'] == 0).all(),
                'NMSE-only BER fields must remain unscored writer placeholders')


def scalar(z, key):
    value = np.asarray(z[key])
    require(value.shape == () and value.dtype.kind in 'US', 'Incorrect text scalar: ' + key)
    return str(value.item())


def verify(archive, reference_results):
    reference = Path(reference_results)
    if reference.name == 'cells':
        reference = reference.parent
    load = lambda name: json.loads((reference / name).read_text(encoding='utf-8-sig'))
    experiment = signed(load('experiment.json'), 'original experiment')
    original_eval = signed(load('evaluation.json'), 'original evaluation')
    require(original_eval['experiment_signature'] == experiment['signature'], 'Original provenance differs')
    require(len(original_eval['cells']) == 636, 'Need original full 636-cell experiment metadata')
    expected = {m['cell_id']: m for m in selected_cells(original_eval['cells'])}
    routes = original_eval['routes']
    indices = [i for i, route in enumerate(routes)
               if route['method'] == 'dopnet' and route['training_seed'] == 42]
    require(len(indices) == 1, 'Need exactly one original DOP-Net seed42 route')
    system = experiment['configs']['dopnet_seed42']['system']
    with ZipFile(archive) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        require(len(names) == len(set(names)) and z.testzip() is None, 'ZIP names/CRC validation failed')
        manifests = [n for n in names if n.endswith('/return_package_manifest.json')]
        require(len(manifests) == 1, 'Need exactly one corrected return manifest')
        prefix = manifests[0][:-len('return_package_manifest.json')]
        require(all(n.startswith(prefix) for n in names), 'Files outside expected archive root')
        read = lambda name: z.read(prefix + name)
        read_json = lambda name: json.loads(read(name).decode('utf-8-sig'))
        manifest = read_json('return_package_manifest.json')
        require(manifest['paper_ready'] is True and manifest['pending'] == [], 'Package is not paper ready')
        require(set(manifest['files']) == {n[len(prefix):] for n in names} - {'return_package_manifest.json'},
                'Return manifest does not bind every file')
        for name, sha in manifest['files'].items():
            path = PurePosixPath(name)
            require(not path.is_absolute() and '..' not in path.parts and '\\' not in name, 'Unsafe manifest path')
            require(hashlib.sha256(read(name)).hexdigest() == sha, 'Internal SHA differs: ' + name)
        evaluation = signed(read_json('evaluation.json'), 'corrected evaluation')
        require(evaluation['suite'] == 'manuscript' and evaluation['methods'] == list(METHODS), 'Wrong evaluation scope')
        require(len(evaluation['cells']) == 269 and {m['cell_id']: m for m in evaluation['cells']} == expected,
                'Corrected conditions differ from original manuscript suite')
        require(evaluation['reference_evaluation_signature'] == original_eval['signature']
                and evaluation['reference_experiment_signature'] == experiment['signature'], 'Reference provenance differs')
        lock = signed(read_json('calibration_lock.json'), 'calibration lock')
        lock_sha = manifest['files']['calibration_lock.json']
        require(evaluation['calibration_lock_sha256'] == lock_sha and lock['status'] == 'locked', 'Calibration binding differs')
        complete = read_json('COMPLETE.json')
        require(complete['status'] == 'complete' and complete['signature'] == evaluation['signature']
                and set(complete['cell_files']) == set(expected), 'Root completion binding differs')
        scores = {n.removeprefix('cells/').removesuffix('.npz') for n in manifest['files']
                  if n.startswith('cells/') and n.endswith('.npz')}
        require(scores == set(expected), 'Missing or unexpected corrected score cell')
        for cid, meta in expected.items():
            name = 'cells/' + cid + '.npz'
            sha = manifest['files'][name]
            require(complete['cell_files'][cid] == sha, 'Root score binding differs: ' + cid)
            source_sha = file_hash(reference / name)
            marker = read_json('cells/' + cid + '.COMPLETE.json')
            require(marker['status'] == 'complete' and marker['samples'] == 512
                    and marker['meta'] == meta and marker['methods'] == list(METHODS)
                    and marker['frozen_waveform_sha256_pass'] is True, 'Invalid score completion marker: ' + cid)
            bindings = {'signature': evaluation['signature'], 'lock_sha256': lock_sha,
                        'source_score_sha256': source_sha, 'npz_sha256': sha}
            require(all(marker.get(k) == v for k, v in bindings.items()), 'Score marker SHA differs: ' + cid)
            with np.load(reference / name, allow_pickle=False) as old, np.load(io.BytesIO(read(name)), allow_pickle=False) as new:
                require(scalar(old, 'signature') == original_eval['signature']
                        and json.loads(scalar(old, 'meta')) == meta, 'Original score identity differs: ' + cid)
                require(scalar(new, 'cell_id') == cid and json.loads(scalar(new, 'meta')) == meta,
                        'Corrected score identity differs: ' + cid)
                for key in ('signature', 'lock_sha256', 'source_score_sha256'):
                    require(scalar(new, key) == bindings[key], 'Corrected score SHA differs: ' + cid + '.' + key)
                require(scalar(old, 'realization_sha256') == scalar(new, 'realization_sha256')
                        == marker['realization_sha256'], 'Frozen frame identity differs: ' + cid)
                validate_arrays(new, old, meta, len(routes), indices[0],
                                expected_bits(system, meta['modulation']), lock['settings'])
    return {'status': 'pass', 'archive_sha256': file_hash(archive), 'cells_checked': 269,
            'methods': list(METHODS), 'frames_per_cell': 512, 'scored_ber_cells': 33,
            'nmse_only_cells': 236, 'finite_nmse_and_diagnostics': True,
            'integer_error_bit_counts_and_source_denominators': True,
            'scope': 'Supplementary pure-file array checks; run full verify_delivery separately; no models or statistics run'}


def fixture(ber):
    # A synthetic writer-shaped score fixture, never a channel waveform/model.
    new = {k: np.ones((3, 512), dtype=np.float64) * 1e-8 for k in FLOAT_KEYS}
    new['ridge_lambda'][0] = 0
    new['native_relative_residual'][0] = 0
    new.update({k: np.zeros((3, 512), dtype=np.int64) for k in INT_KEYS})
    new['errors_native'][0] = -1
    new.update(methods=np.asarray(METHODS), native_available=np.asarray([False, True, True]),
               frame_index=np.arange(512, dtype=np.int64), cluster_ids=np.arange(512, dtype=np.int64) // 8)
    new.update({k: np.asarray('synthetic') for k in IDENTITY_KEYS if k not in new})
    old = {'nmse': np.zeros((2, 512)), 'cluster_id': np.arange(512) // 8}
    if ber:
        new['bits'][:] = 12
        old['bits'] = np.full((2, 512), 12, dtype=np.int64)
    else:
        new['native_relative_residual'][:] = 0
    settings = {m: {'fit_tolerance': 1e-6, 'detector_tolerance': 1e-6} for m in METHODS[1:]}
    return new, old, {'ber': ber}, 2, 0, 12, settings


class MetadataTests(unittest.TestCase):
    def test_ber_writer_dialect(self):
        validate_arrays(*fixture(True))

    def test_nmse_only_placeholders(self):
        validate_arrays(*fixture(False))

    def test_nonfinite_or_negative_nmse(self):
        for bad in (math.nan, math.inf, -1.):
            args = fixture(True)
            args[0]['nmse'][1, 1] = bad
            with self.assertRaises(ValueError):
                validate_arrays(*args)

    def test_missing_and_wrong_shape(self):
        args = fixture(True)
        del args[0]['bits']
        with self.assertRaises(ValueError):
            validate_arrays(*args)
        args = fixture(True)
        args[0]['errors_diagonal'] = np.zeros((3, 511), dtype=np.int64)
        with self.assertRaises(ValueError):
            validate_arrays(*args)

    def test_float_counts_rejected(self):
        args = fixture(True)
        args[0]['bits'] = args[0]['bits'].astype(np.float64)
        with self.assertRaises(ValueError):
            validate_arrays(*args)

    def test_ber_denominator_and_error_range(self):
        for key, value in (('bits', 0), ('bits', 13), ('errors_diagonal', -1),
                           ('errors_diagonal', 13), ('errors_native', 13)):
            args = fixture(True)
            args[0][key][1, 0] = value
            with self.assertRaises(ValueError):
                validate_arrays(*args)

    def test_unavailable_native_cannot_be_zero(self):
        args = fixture(True)
        args[0]['errors_native'][0, 0] = 0
        with self.assertRaises(ValueError):
            validate_arrays(*args)

    def test_nmse_only_cannot_carry_scored_ber(self):
        args = fixture(False)
        args[0]['bits'][1, 0] = 12
        with self.assertRaises(ValueError):
            validate_arrays(*args)
        args = fixture(False)
        args[1]['bits'] = np.zeros((2, 512), dtype=np.int64)
        with self.assertRaises(ValueError):
            validate_arrays(*args)

    def test_residual_and_ridge_validity(self):
        for key, value in (('fit_relative_residual', 1.1e-6),
                           ('native_relative_residual', 1.1e-6), ('ridge_lambda', 0.)):
            args = fixture(True)
            args[0][key][1, 0] = value
            with self.assertRaises(ValueError):
                validate_arrays(*args)

    def test_public_geometry_denominator(self):
        system = {'n_users': 2, 'n_subcarriers': 128,
                  'frame_num_symbols': 25, 'pilot_symbol_period': 4}
        self.assertEqual(expected_bits(system, '16qam'), 18432)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--reference-results', type=Path,
                        default=Path(__file__).resolve().parent.parent / '实验结果及数据/dopnet_additions')
    parser.add_argument('--self-test', action='store_true', help='Tiny synthetic score metadata only')
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MetadataTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.archive is None:
        parser.error('--archive is required unless --self-test is selected')
    try:
        result = verify(args.archive, args.reference_results)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps({'status': 'fail', 'error': str(error)}, indent=2), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
