"""Synthetic reporting-contract tests; fixtures are deleted after each run."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import report_classical as report


class ReportingContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dopnet_synthetic_report_')
        self.root = Path(self.temp.name)
        self.raw = self.root / 'raw'
        self.output = self.root / 'scores'
        self.reference = self.root / 'reference'
        (self.output / 'cells').mkdir(parents=True)
        (self.reference / 'cells').mkdir(parents=True)
        self.raw.mkdir()
        self.cells = []
        self.routes = [dict(method=m, training_seed=s, trained=True)
                       for m in report.LEARNED for s in report.SEEDS]
        self.routes += [dict(method=m, training_seed=None, trained=False) for m in report.OLD_CLASSICAL]
        for ch in ('doubly_selective',) + report.SEEN + report.HELDOUT:
            for fd in (800.0, 1200.0, 1600.0):
                for mod, snr in (('qpsk', 15.0), ('qpsk', 25.0), ('16qam', 20.0)):
                    cell_id = hashlib.sha256(repr((ch, fd, mod, snr)).encode()).hexdigest()[:16]
                    self.cells.append(dict(cell_id=cell_id, channel=ch, doppler_hz=fd,
                                           modulation=mod, snr_db=snr, cp_len=16,
                                           pilot_spacing=4, delay_spread_ns=300.0,
                                           studies=['receiver'] if mod == '16qam' else ['nmse'],
                                           ber=mod == '16qam'))
        experiment = dict(fixture='synthetic only')
        experiment['signature'] = report.object_hash(experiment)
        evaluation = dict(cells=self.cells, routes=self.routes,
                          experiment_signature=experiment['signature'])
        evaluation['signature'] = report.object_hash(evaluation)
        fingerprints = {m['cell_id']: hashlib.sha256(('synthetic:' + m['cell_id']).encode()).hexdigest()
                        for m in self.cells}
        provenance = dict(reference_evaluation_signature=evaluation['signature'],
                          reference_experiment_signature=experiment['signature'], reference_cells=99,
                          reference_samples_per_cell=512, reference_batch_size=8)
        provenance['reference_realizations_sha256'] = report.object_hash(dict(
            evaluation_signature=evaluation['signature'], experiment_signature=experiment['signature'],
            waveform_keys=('input', 'add_input', 'add_input_slow', 'rx_data', 'target_h', 'tx_label'),
            realization_sha256=fingerprints))
        manifest = dict(cells=self.cells, suites={'core': [m['cell_id'] for m in self.cells]},
                        samples_per_cell=512, batch_size=8, schema_version=1,
                        export_signature='synthetic-export', calibration_seed=810000001,
                        cfg_system={'fixture': 'synthetic'}, geometry={'fixture': 'synthetic'},
                        reference_provenance=provenance,
                        fixture='synthetic unit test, never research evidence')
        report.write_json(self.raw / 'export_manifest.json', manifest)
        report.write_json(self.reference / 'evaluation.json', evaluation)
        report.write_json(self.reference / 'experiment.json', experiment)
        calib_batch = self.raw / 'calibration' / 'synthetic.bin'
        calib_batch.parent.mkdir()
        calib_batch.write_bytes(b'synthetic calibration fixture')
        report.write_json(self.raw / 'calibration' / 'COMPLETE.json', dict(
            schema_version=1, role='calibration', samples=128, seed=810000001,
            export_signature='synthetic-export',
            batches=[dict(path='calibration/synthetic.bin', offset=0, samples=128,
                          sha256=report.sha256(calib_batch))]))
        settings = {m: dict(method=m, relative_ridge=1., fit_tolerance=1e-6, detector_tolerance=1e-6,
                            fit_max_iter=256, detector_max_iter=128) for m in report.NEW_CLASSICAL}
        lock = dict(schema_version=1, status='locked', locked=True, methods=list(report.NEW_CLASSICAL),
                    export_signature='synthetic-export', reference_provenance=provenance,
                    cfg_system_sha256=report.object_hash(manifest['cfg_system']),
                    geometry_sha256=report.object_hash(manifest['geometry']),
                    code_sha256=report.code_hashes(),
                    calibration_seed=810000001, calibration_samples=128,
                    calibration_complete_sha256=report.sha256(self.raw / 'calibration' / 'COMPLETE.json'),
                    settings=settings, candidates={m: [dict(status='complete', completed_frames=128,
                                                            relative_ridge=1., frame_nmse=[.4] * 128,
                                                            mean_linear_nmse=.4)] for m in report.NEW_CLASSICAL})
        lock['signature'] = report.object_hash(lock)
        report.write_json(self.output / 'calibration_lock.json', lock)
        lock_hash = report.sha256(self.output / 'calibration_lock.json')
        cpu_evaluation = dict(schema_version=1, methods=list(report.NEW_CLASSICAL),
                              calibration_lock_sha256=lock_hash, export_signature='synthetic-export',
                              reference_provenance=provenance, code_sha256=report.code_hashes())
        cpu_evaluation['signature'] = report.object_hash(cpu_evaluation)
        report.write_json(self.output / 'evaluation.json', cpu_evaluation)
        bindings = {}
        for index, meta in enumerate(self.cells):
            cid = meta['cell_id']
            fingerprint = hashlib.sha256(('synthetic:' + cid).encode()).hexdigest()
            raw_complete = self.raw / 'cells' / cid / 'COMPLETE.json'
            batch = raw_complete.parent / 'synthetic.bin'
            batch.parent.mkdir(parents=True)
            batch.write_bytes(b'synthetic receiver fixture')
            report.write_json(raw_complete, dict(schema_version=1, role='test', samples=512,
                                                 cell_id=cid, meta=meta, export_signature='synthetic-export',
                                                 realization_sha256=fingerprint,
                                                 batches=[dict(path=f'cells/{cid}/synthetic.bin', offset=offset,
                                                               samples=8, sha256=report.sha256(batch))
                                                          for offset in range(0, 512, 8)]))
            nmse = np.full((len(self.routes), 512), .2 + index / 1000.0)
            nmse[:3] *= np.array([.9, 1., 1.1])[:, None]
            errors = np.full_like(nmse, 10, dtype=np.int64)
            bits = np.full_like(errors, 100)
            avail = np.ones(len(self.routes), dtype=np.int8)
            avail[-3:] = 0
            np.savez(self.reference / 'cells' / (cid + '.npz'),
                     nmse=nmse, errors_diagonal=errors, errors_native=errors,
                     bits=bits, native_available=avail,
                     cluster_id=np.arange(512) // 8,
                     signature=np.array(evaluation['signature']),
                     meta=np.array(json.dumps(meta)), realization_sha256=np.array(fingerprint))
            score_file = self.output / 'cells' / (cid + '.npz')
            np.savez(score_file, nmse=np.full((2, 512), .3 + index / 1000.0),
                     errors_diagonal=np.full((2, 512), 15, dtype=np.int64),
                     errors_native=np.full((2, 512), 12, dtype=np.int64),
                     bits=np.full((2, 512), 100, dtype=np.int64),
                     frame_index=np.arange(512), cluster_ids=np.arange(512) // 8,
                     methods=np.array(report.NEW_CLASSICAL), cell_id=np.array(cid),
                     realization_sha256=np.array(fingerprint), lock_sha256=np.array(lock_hash),
                     source_complete_sha256=np.array(report.sha256(raw_complete)),
                     signature=np.array(cpu_evaluation['signature']),
                     fit_relative_residual=np.full((2, 512), 1e-8),
                     native_relative_residual=np.full((2, 512), 1e-8),
                     fit_iterations=np.full((2, 512), 20), native_iterations=np.full((2, 512), 20))
            bindings[cid] = report.sha256(score_file)
            report.write_json(score_file.with_suffix('.COMPLETE.json'), dict(
                status='complete', signature=cpu_evaluation['signature'],
                source_complete_sha256=report.sha256(raw_complete), lock_sha256=lock_hash,
                npz_sha256=bindings[cid]))
        report.write_json(self.output / 'COMPLETE.json',
                          dict(cell_files=bindings, calibration_lock_sha256=lock_hash,
                               export_manifest_sha256=report.sha256(self.raw / 'export_manifest.json'),
                               status='complete', methods=list(report.NEW_CLASSICAL),
                               signature=cpu_evaluation['signature'], export_signature='synthetic-export'))
        self.args = argparse.Namespace(classical_output=str(self.output),
                                       reference_results=str(self.reference),
                                       export_manifest=str(self.raw / 'export_manifest.json'),
                                       suite='core', destination=None, bootstrap=1000,
                                       bootstrap_seed=1234)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_then_partial(self):
        with patch.object(report, 'make_plots', return_value=[]):
            self.assertTrue(report.report(self.args))
        dest = self.output / 'summary'
        stats = report.read_json(dest / 'statistics.json')
        self.assertTrue(stats['paper_ready'])
        self.assertEqual(stats['completed_verified_cells'], 99)
        self.assertEqual(stats['classical_training_seeds'], [])
        insertion = (dest / 'paper_insertion.tex').read_text(encoding='utf8')
        self.assertIn('66 conditions', insertion)
        self.assertIn('33 conditions', insertion)
        self.assertNotIn('classical_results/classical_native_ber.tex', insertion)
        self.assertTrue((dest / 'native_receiver_insertion.tex').is_file())
        # A missing hard case does not become an implicitly smaller headline grid.
        missing = self.output / 'cells' / (self.cells[-1]['cell_id'] + '.npz')
        missing.unlink()
        self.assertFalse(report.report(self.args))
        self.assertFalse((dest / 'paper_insertion.tex').exists())
        self.assertEqual(report.read_json(dest / 'statistics.json')['missing_cell_ids'],
                         [self.cells[-1]['cell_id']])

    def test_raw_fingerprint_mismatch(self):
        cid = self.cells[0]['cell_id']
        path = self.raw / 'cells' / cid / 'COMPLETE.json'
        marker = report.read_json(path)
        marker['realization_sha256'] = '0' * 64
        report.write_json(path, marker)
        with self.assertRaisesRegex(ValueError, 'waveform fingerprint'):
            report.report(self.args)


if __name__ == '__main__':
    unittest.main()
