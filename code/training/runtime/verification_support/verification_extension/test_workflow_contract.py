"""Small fake-receiver tests: no Torch, neural model, or real data execution."""
from contextlib import nullcontext
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from verification_extension import protocol as P
from verification_extension import workflow as W


class Prior:
    def estimate(self, *_args, **_kwargs):
        return np.ones((2, 1, 2, 4), dtype=np.complex128), {'relative_residual_max': 1e-9}


class Operator:
    def __init__(self, *_args, **_kwargs):
        pass

    def fit(self, *_args, **_kwargs):
        return None, {'relative_residual_max': 1e-9,
                      'recovered_normal_relative_residual_max': 2e-9}

    def diagonal_cfr(self, _):
        return np.ones((2, 1, 2, 4), dtype=np.complex128)

    def native_mmse(self, *_args, **_kwargs):
        return np.ones((2, 1, 4), dtype=np.complex128), {'relative_residual_max': 1e-9}


class WorkflowCommitTest(unittest.TestCase):
    def test_extra_chunks_preserve_base_calibration(self):
        self.check_extra_chunks(False)

    def test_repeated_extra_offset_is_rejected(self):
        self.check_extra_chunks(True)

    def check_extra_chunks(self, duplicate):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); cache = root / 'observation_cache'
            records = []
            for offset in (0, 4):
                path = cache / ('calibration_extra/batch_%04d.npz' % offset)
                P.atomic_npz(path, {
                    'metadata': np.asarray(json.dumps({'role': 'statistical_calibration',
                        'calibration_seed': 810000002, 'offset': offset, 'samples': 4})),
                    'noise_var': np.ones(4), 'x_pilot': np.ones((4, 2, 1, 4))})
                records.append({'path': path.relative_to(cache).as_posix(), 'offset': offset,
                                'samples': 4, 'sha256': P.file_hash(path)})
            if duplicate:
                records[1]['offset'] = 0
            P.write_json(cache / 'calibration_extra/COMPLETE.json', {
                'samples': 8, 'seed': 810000002, 'role': 'statistical_calibration',
                'export_signature': 'toy-export', 'batches': records})
            base = [{'noise_var': np.zeros(4)}]
            args = SimpleNamespace(output=root, statistical_calibration_extra=8)
            if duplicate:
                with self.assertRaisesRegex(RuntimeError, 'duplicated or unordered'):
                    W.statistical_arrays(args, base, {'export_signature': 'toy-export'})
            else:
                loaded = W.statistical_arrays(args, base, {'export_signature': 'toy-export'})
                self.assertEqual(len(loaded), 3)
                self.assertIs(loaded[0], base[0])

    def run_fake(self, fail_waveform):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); ref = root / 'reference'; out = root / 'output'
            (ref / 'cells').mkdir(parents=True)
            (out / 'observation_cache').mkdir(parents=True)
            cell = {'cell_id': '0123456789abcdef', 'ber': True, 'modulation': 'qpsk'}
            wave = 'a' * 64
            P.atomic_npz(ref / 'cells/0123456789abcdef.npz', {
                'realization_sha256': np.asarray(wave), 'bits': np.full((1, 512), 16, dtype=np.int64)})
            lock = {'status': 'locked', 'methods': list(P.METHODS), 'code_sha256': P.source_hashes(),
                    'export_signature': 'toy-export',
                    'settings': {m: {'method': m} for m in P.METHODS[1:]}}
            lock['signature'] = P.digest(lock)
            P.write_json(out / 'calibration_lock.json', lock)
            P.write_json(out / 'observation_cache/export_manifest.json', {'export_signature': 'toy-export'})
            policy = {'precision': 'complex128', 'recommendations': [
                {'method': m, 'preferred_device': 'cpu', 'cpu_threads': 1, 'precision': 'complex128'}
                for m in P.METHODS]}
            policy['signature'] = P.digest(policy)
            P.write_json(out / 'classical_device_policy.json', policy)
            fake_torch = ModuleType('torch')
            fake_torch.inference_mode = nullcontext
            fake_torch.get_num_threads = lambda: 1
            fake_torch.set_num_threads = lambda _: None
            fake_classical = ModuleType('verification_extension.classical')
            fake_classical.BEMSettings = lambda **kw: kw
            fake_classical.TorchJointBEM = Operator
            fake_classical.geometry_from_export = lambda *_: 'toy-geometry'
            fake_classical.geometry_key = lambda _: 'toy-key'
            fake_classical.diagonal_mmse = lambda *_: np.ones((2, 1, 4), dtype=np.complex128)
            fake_classical.nmse_per_frame = lambda *_: np.asarray([1.0])
            fake_classical.frame_bit_counts = lambda *_: (np.asarray([1]), np.asarray([16]))
            fake_calibration = ModuleType('verification_extension.calibration')
            fake_calibration.load_calibrated_bank = lambda *_args, **_kw: {
                'toy-key': {'prior': Prior(), 'noise_scale': 1.0}}
            fake_calibration.resolve_classical_device = lambda _path, method: next(
                row for row in policy['recommendations'] if row['method'] == method)
            fake_replay = ModuleType('verification_extension.replay')
            arrays = {'noise_var': np.ones(8), 'x_pilot': np.ones((8, 2, 1, 4)),
                      'y_pilot': np.ones((8, 1, 2, 4)), 'target_h': np.ones((8, 2, 1, 2, 4)),
                      'rx_data': np.ones((8, 1, 2, 4)), 'tx_label': np.zeros((8, 2, 1, 4), dtype=np.int64)}

            class Replay:
                def __init__(self, *_):
                    self.manifest = {}

                def test_batches(self, _):
                    for offset in range(0, 512, 8):
                        yield offset, arrays
                    if fail_waveform:
                        raise RuntimeError('Frozen waveform mismatch')

            fake_replay.FrozenReplay = Replay
            ref_eval = {'signature': 'toy-reference-evaluation', 'cells': [cell],
                        'routes': [{'method': 'dopnet', 'training_seed': 42}]}
            args = SimpleNamespace(reference_results=ref, project=root, output=out,
                                   suite='core', cpu_threads=(1,))
            modules = {'torch': fake_torch, 'verification_extension.classical': fake_classical,
                       'verification_extension.calibration': fake_calibration,
                       'verification_extension.replay': fake_replay}
            with patch.dict('sys.modules', modules), patch.object(W, 'check_reference',
                    return_value=(ref, {'signature': 'toy-experiment'}, ref_eval)), \
                    patch.object(W, 'select_suite', return_value=[cell]):
                if fail_waveform:
                    with self.assertRaisesRegex(RuntimeError, 'waveform mismatch'):
                        W.evaluate(args)
                    self.assertFalse((out / 'cells/0123456789abcdef.npz').exists())
                    self.assertFalse((out / 'COMPLETE.json').exists())
                else:
                    W.evaluate(args)
                    marker = P.read_json(out / 'cells/0123456789abcdef.COMPLETE.json')
                    self.assertEqual(marker['meta'], cell)
                    self.assertTrue(marker['frozen_waveform_sha256_pass'])
                    with np.load(out / 'cells/0123456789abcdef.npz') as z:
                        self.assertTrue(np.all(z['fit_relative_residual'][1:] == 2e-9))
                        self.assertTrue(np.all(z['errors_native'][0] == -1))
                        self.assertTrue(np.array_equal(z['cluster_ids'], np.arange(512) // 8))

    def test_bad_waveform_never_commits_scores(self):
        self.run_fake(True)

    def test_complete_waveform_commits_bound_scores(self):
        self.run_fake(False)


if __name__ == '__main__':
    unittest.main()
