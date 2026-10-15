"""Synthetic saved-score contract tests; no scientific data or model execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import report as r
from .protocol import METHODS, atomic_npz, file_hash, write_json


class ReportingContracts(unittest.TestCase):
    def test_complete_scores_do_not_claim_missing_hardware_results(self):
        state = r.readiness(True, {'status': 'not_available'}, [], {'status': 'preliminary_settings_only'}, [])
        self.assertTrue(state['score_assets_ready'])
        self.assertFalse(state['digital_latency_ready'])
        self.assertFalse(state['classical_verification_cost_ready'])
        self.assertFalse(state['paper_ready'])
        self.assertEqual(state['status'], 'scores_complete_pending_hardware')
        self.assertEqual(len(state['pending']), 3)
        full = r.readiness(True, {'status': 'complete'}, ['corrected_digital_receiver_latency.tex',
                           'corrected_digital_receiver_throughput.tex'],
                           {'status': 'complete'}, ['corrected_classical_backend_cost.tex'])
        self.assertTrue(full['paper_ready'])
        self.assertEqual(full['status'], 'complete')
        self.assertEqual(full['pending'], [])
        batch_only = r.readiness(True, {'status': 'complete'}, ['corrected_digital_receiver_throughput.tex'],
                                 {'status': 'complete'}, ['corrected_classical_backend_cost.tex'])
        self.assertTrue(batch_only['throughput_assets_ready'])
        self.assertFalse(batch_only['single_frame_latency_ready'])
        self.assertFalse(batch_only['digital_latency_ready'])
        self.assertFalse(batch_only['paper_ready'])
        self.assertEqual(batch_only['pending'], ['single-frame digital receiver CPU/CUDA latency'])

    def test_bit_counts_bootstrap_uses_ratio_of_totals(self):
        cluster = np.arange(512) // 8
        ids = np.arange(64)
        bits = np.tile(np.arange(1, 65).repeat(8)[None], (1, 1)) * 10
        errors = np.ones((1, 512))
        weights = np.eye(64, dtype=int) * 64
        point, boot = r._old.metric_samples(errors, bits, cluster, weights, ids)
        self.assertAlmostEqual(float(point[0]), float(errors.sum() / bits.sum()))
        np.testing.assert_allclose(boot[:, 0], 1 / (np.arange(1, 65) * 10))

    def test_common_weights_and_seed_draws_produce_zero_identical_paired_ci(self):
        rng = np.random.default_rng(14)
        point = np.asarray([.2, .21, .19])
        boot = rng.uniform(.15, .25, size=(1000, 3))
        draws = rng.integers(0, 3, size=(1000, 3))
        row = r.paired_row({}, 'slow_time', 'nmse', point, boot, point, boot, draws)
        for name in ('paired_difference', 'ci95_lo', 'ci95_hi', 'paired_gap_db',
                     'paired_gap_ci95_lo_db', 'paired_gap_ci95_hi_db'):
            self.assertEqual(row[name], 0.)

    def test_macro_equal_channel_weight_and_incomplete_group_is_omitted(self):
        key = ('f', 's', 'nmse', 'dopnet')
        one = (np.asarray([1., 1., 1.]), np.ones((1000, 3)))
        three = (np.asarray([3., 3., 3.]), np.full((1000, 3), 3.))
        buckets = {key: {'a': {'a1': one}, 'b': {'b1': three, 'b2': three, 'b3': three}}}
        expected = {key[:3]: {'a1', 'b1', 'b2', 'b3'}}
        result = r.aggregate_complete_groups(buckets, expected)
        np.testing.assert_equal(result[key][0], [2., 2., 2.])
        del buckets[key]['b']['b3']
        self.assertEqual(r.aggregate_complete_groups(buckets, expected), {})

    def fixture_cell(self, root):
        cid = '0000000000000001'
        meta = {'cell_id': cid, 'channel': 'doubly_selective', 'doppler_hz': 1200.,
                'snr_db': 20., 'modulation': '16qam', 'cp_len': 16, 'pilot_spacing': 4,
                'delay_spread_ns': 300., 'studies': ['receiver'], 'ber': True}
        evaluation = {'signature': 'a' * 64, '_reference_routes': [
            {'method': 'dopnet', 'training_seed': 42}, {'method': 'dopnet', 'training_seed': 43},
            {'method': 'dopnet', 'training_seed': 44}]}
        original = {'frame_index': np.arange(512), 'cluster_id': np.arange(512) // 8,
                    'realization_sha256': np.asarray('b' * 64), 'bits': np.full((3, 512), 1024)}
        arrays = {'nmse': np.full((3, 512), .25), 'errors_diagonal': np.full((3, 512), 20, dtype=np.int64),
                  'errors_native': np.asarray([np.full(512, -1), np.full(512, 12), np.full(512, 13)]),
                  'bits': np.full((3, 512), 1024, dtype=np.int64), 'native_available': np.asarray([False, True, True]),
                  'fit_relative_residual': np.full((3, 512), 1e-12), 'native_relative_residual': np.full((3, 512), 1e-12),
                  'fit_iterations': np.zeros((3, 512), dtype=np.int64), 'native_iterations': np.zeros((3, 512), dtype=np.int64),
                  'frame_index': np.arange(512), 'cluster_ids': np.arange(512) // 8,
                  'signature': np.asarray('a' * 64), 'lock_sha256': np.asarray('c' * 64),
                  'source_score_sha256': np.asarray('d' * 64), 'realization_sha256': np.asarray('b' * 64),
                  'methods': np.asarray(METHODS)}
        path = root / 'cells' / (cid + '.npz')
        atomic_npz(path, arrays)
        sha = file_hash(path)
        marker = {'status': 'complete', 'meta': meta, 'signature': 'a' * 64, 'lock_sha256': 'c' * 64,
                  'source_score_sha256': 'd' * 64, 'realization_sha256': 'b' * 64, 'npz_sha256': sha}
        write_json(path.with_suffix('.COMPLETE.json'), marker)
        return cid, meta, original, evaluation, {cid: sha}, arrays, marker

    def test_fake_corrected_npz_is_bound_and_native_sentinel_is_not_zero_ber(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cid, meta, original, evaluation, bindings, arrays, marker = self.fixture_cell(root)
            result = r.verify_corrected_cell(root, cid, meta, original, 'd' * 64, evaluation, 'c' * 64, bindings)
            np.testing.assert_equal(result[1], [False, True, True])
            self.assertTrue(np.all(result[0]['errors_native'][0] == -1))
            native_methods = [m for m, available in zip(METHODS, result[1]) if available]
            self.assertNotIn('pilot_covariance_lmmse', native_methods)

    def test_fake_corrected_npz_rejects_wrong_source_order_and_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cid, meta, original, evaluation, bindings, arrays, marker = self.fixture_cell(root)
            with self.assertRaises(ValueError):
                r.verify_corrected_cell(root, cid, meta, original, 'e' * 64, evaluation, 'c' * 64, bindings)
            original['cluster_id'] = original['cluster_id'][::-1]
            with self.assertRaises(ValueError):
                r.verify_corrected_cell(root, cid, meta, original, 'd' * 64, evaluation, 'c' * 64, bindings)
            original['cluster_id'] = np.arange(512) // 8
            original['bits'] += 1
            with self.assertRaises(ValueError):
                r.verify_corrected_cell(root, cid, meta, original, 'd' * 64, evaluation, 'c' * 64, bindings)

    def test_failed_low_ridge_candidate_cannot_be_used_as_calibration_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / 'prior.npz'
            atomic_npz(state, {'frequency_covariance': np.eye(2), 'temporal_correlation': np.eye(2)})
            settings = {m: {'method': m, 'relative_ridge': 1.} for m in METHODS[1:]}
            lock = {'calibration_seed': 810000001, 'calibration_samples': 128, 'settings': settings,
                    'candidates': {m: [{'status': 'complete', 'completed_frames': 128,
                                       'frame_nmse': [1.] * 128, 'mean_linear_nmse': 1., 'settings': settings[m],
                                       'solver_residual_max': 1e-12}] for m in METHODS[1:]},
                    'statistical_priors': {'fake': {'fit_frames': 96, 'holdout_frames': 32, 'all_frames': 128,
                        'noise_scale': 1., 'npz_relative_path': 'prior.npz', 'sha256': file_hash(state),
                        'noise_candidates': [{'status': 'complete', 'frames': 32, 'frame_nmse': [1.] * 32,
                                              'mean_linear_nmse': 1., 'solver_residual_max': 1e-12, 'noise_scale': 1.}]}}}
            r.validate_calibration_lock(lock, root)
            bad = copy.deepcopy(lock)
            bad['candidates'][METHODS[1]].append({'status': 'failed', 'completed_frames': 0,
                                                  'frame_nmse': [], 'mean_linear_nmse': -1., 'settings': {'relative_ridge': .01}})
            with self.assertRaises(ValueError):
                r.validate_calibration_lock(bad, root)

    def test_measured_latency_rejects_unequal_experiment_and_unqualified_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'latency'
            source.mkdir()
            dest = root / 'summary'
            dest.mkdir()
            timing = {'complete': True, 'protocol': {'experiment_signature': 'f' * 64},
                      'hardware': {'cpu': 'synthetic fixture'}, 'device_recommendations': [], 'rows': []}
            for device, passed in [('cpu', True), ('cuda:0', False)]:
                timing['rows'].append({'method': 'dopnet', 'seed': 42, 'batch_size': 1,
                    'receiver_mode': 'common_diagonal', 'device': device, 'source_name': 'calibration',
                    'input_boundary': 'received_occupied_ofdm_grid', 'status': 'measured',
                    'numerical_equivalence': {'passed': passed}, 'timing': {'host_origin_total': {
                        'latency_ms_p50': 2., 'latency_ms_p95': 3., 'latency_ms_mean': 2.5}}})
            write_json(source / 'digital_receiver_latency.json', timing)
            assets, status = r.latency_assets(dest, source, 'f' * 64)
            self.assertEqual(status['status'], 'complete')
            text = (dest / 'corrected_digital_receiver_latency.tex').read_text(encoding='utf-8')
            self.assertIn('2.000 / 3.000', text)
            self.assertIn('--', text)
            with self.assertRaises(ValueError):
                r.latency_assets(dest, source, '0' * 64)

    def test_batch_eight_only_insertion_references_existing_throughput_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'latency'
            source.mkdir()
            dest = root / 'summary'
            dest.mkdir()
            timing = {'complete': True, 'protocol': {'experiment_signature': 'f' * 64},
                      'hardware': {}, 'device_recommendations': [], 'rows': [{
                        'method': 'dopnet', 'seed': 42, 'batch_size': 8, 'receiver_mode': 'common_diagonal',
                        'device': 'cpu', 'source_name': 'calibration', 'input_boundary': 'received_occupied_ofdm_grid',
                        'status': 'measured', 'numerical_equivalence': {'passed': True}, 'timing': {
                          'host_origin_total': {'latency_ms_p50': 2., 'latency_ms_p95': 3., 'latency_ms_mean': 2.5}}}]}
            write_json(source / 'digital_receiver_latency.json', timing)
            r.latency_assets(dest, source, 'f' * 64)
            r.write_insertion(dest, [], [], 'nominal', True)
            text = (dest / 'corrected_paper_insertion.tex').read_text(encoding='utf-8')
            self.assertIn('corrected_digital_receiver_throughput.tex', text)
            self.assertNotIn('corrected_digital_receiver_latency.tex', text)

    def test_unavailable_optional_cp_does_not_block_a_covered_core_suite(self):
        lock = {'statistical_priors': {'available': {'geometry': {'nfft': 16, 'n_active': 8, 'cp_wave_samples': 2}}},
                'unavailable_statistical_priors': {'optional': {'frames': 12,
                      'geometry': {'nfft': 16, 'n_active': 8, 'cp_wave_samples': 4}}}}
        r.validate_required_priors(lock, [{'cp_len': 1}])
        with self.assertRaises(ValueError):
            r.validate_required_priors(lock, [{'cp_len': 1}, {'cp_len': 2}])

    def test_latency_table_uses_complete_recommended_cpu_thread_not_mixed_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, dest = root / 'latency', root / 'summary'
            source.mkdir(); dest.mkdir()
            recommendation = {'method': 'dopnet', 'batch_size': 1, 'receiver_mode': 'common_diagonal',
                              'source_name': 'calibration', 'best_cpu_threads_by_device': {'cpu': 4}}
            timing = {'complete': True, 'protocol': {'experiment_signature': 'f' * 64,
                      'seeds': [42, 43, 44], 'cpu_threads': 8}, 'hardware': {},
                      'device_recommendations': [recommendation], 'rows': []}
            for threads, median in [(1, 10.), (4, 2.)]:
                for seed in (42, 43, 44):
                    timing['rows'].append({'method': 'dopnet', 'seed': seed, 'cpu_threads': threads,
                        'batch_size': 1, 'receiver_mode': 'common_diagonal', 'device': 'cpu',
                        'source_name': 'calibration', 'input_boundary': 'received_occupied_ofdm_grid',
                        'status': 'measured', 'numerical_equivalence': {'passed': True},
                        'timing': {'host_origin_total': {'latency_ms_p50': median, 'latency_ms_p95': median + 1,
                                                        'latency_ms_mean': median}}})
            # An incomplete fast GPU option must not be reported as eligible.
            timing['rows'].append({**timing['rows'][-1], 'device': 'cuda:0', 'seed': 42, 'cpu_threads': 8})
            write_json(source / 'digital_receiver_latency.json', timing)
            r.latency_assets(dest, source, 'f' * 64)
            text = (dest / 'corrected_digital_receiver_latency.tex').read_text(encoding='utf-8')
            self.assertIn('2.000 / 3.000 ($t=4$)', text)
            self.assertNotIn('6.000 / 7.000', text)
            self.assertIn(' & --', text)
            csv_text = (dest / 'corrected_digital_receiver_latency.csv').read_text(encoding='utf-8-sig')
            self.assertIn('cpu_threads', csv_text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
