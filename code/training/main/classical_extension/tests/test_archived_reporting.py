"""Integrity checks for score-only reproduction, using disposable tiny fixtures."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import report_classical as report


class ArchivedReporting(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = dict(schema_version=1, cfg_system={'fixture': True},
                             geometry={'fixture': True}, reference_provenance={'fixture': True})
        self.manifest['export_signature'] = report.object_hash(self.manifest)
        self.manifest.update(cells=[{'cell_id': 'synthetic'}], suites={'core': ['synthetic']})
        report.write_json(self.root / 'export_manifest.json', self.manifest)
        self.lock = dict(schema_version=1, status='locked', code_sha256=report.code_hashes(),
                         cfg_system_sha256=report.object_hash(self.manifest['cfg_system']),
                         geometry_sha256=report.object_hash(self.manifest['geometry']),
                         export_signature=self.manifest['export_signature'],
                         reference_provenance=self.manifest['reference_provenance'])
        self.lock['signature'] = report.object_hash(self.lock)
        report.write_json(self.root / 'calibration_lock.json', self.lock)
        (self.root / 'cells').mkdir()
        self.score = self.root / 'cells/synthetic.npz'
        self.score.write_bytes(b'disposable score hash fixture, not experiment data')
        self.evidence = dict(paper_ready=True, status='complete', suite='core', expected_cells=1,
                             completed_verified_cells=1, missing_cell_ids=[],
                             calibration_lock_sha256=report.sha256(self.root / 'calibration_lock.json'),
                             export_manifest_sha256=report.sha256(self.root / 'export_manifest.json'),
                             cell_files={'synthetic': report.sha256(self.score)}, report_asset_sha256={})
        self.save_evidence()

    def save_evidence(self):
        report.write_json(self.root / 'server_report/statistics.json', self.evidence)

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_completed_archive(self):
        lock, path = report.archive_lock(self.root, self.manifest, 'core')
        self.assertEqual(lock, self.lock)
        self.assertEqual(path, self.root / 'calibration_lock.json')

    def test_changed_score_rejected(self):
        self.score.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'score changed'):
            report.archive_lock(self.root, self.manifest, 'core')

    def test_partial_archive_rejected(self):
        self.evidence['completed_verified_cells'] = 0
        self.save_evidence()
        with self.assertRaisesRegex(ValueError, 'missing or inconsistent'):
            report.archive_lock(self.root, self.manifest, 'core')

    def test_changed_manifest_rejected(self):
        self.manifest['geometry']['changed'] = True
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            report.archive_lock(self.root, self.manifest, 'core')


if __name__ == '__main__':
    unittest.main()
