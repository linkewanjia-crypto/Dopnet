"""CPU-only tests for the fixed-test contract; no waveform simulation."""
import copy
from pathlib import Path
import unittest

import yaml

from experiment_protocol import experiment_cells, make_config
from reference_replay import ReferenceResults, EXPECTED_BATCH, EXPECTED_CELLS, EXPECTED_SAMPLES


ROOT = Path(__file__).resolve().parents[1]


class FixedTestProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = yaml.safe_load((ROOT / 'configs' / 'dopnet.yaml').read_text(encoding='utf-8'))
        cls.cfg = make_config(base, 'dopnet', 42)
        cls.cells, cls.skipped = experiment_cells(cls.cfg)
        cls.reference = ReferenceResults.__new__(ReferenceResults)
        cls.reference.evaluation = {'cells': cls.cells, 'skipped_infeasible': cls.skipped}
        cls.reference.experiment = {'configs': {'dopnet_seed42': copy.deepcopy(cls.cfg)}}
        cls.reference.experiment['configs']['dopnet_seed42']['data_pipeline'].pop(
            'emit_pilot_only_fast_time_add', None)
        cls.reference.cell_fingerprints = {cell['cell_id']: 'a' * 64 for cell in cls.cells}
        cls.manifest = {'samples_per_cell': EXPECTED_SAMPLES, 'batch_size': EXPECTED_BATCH}

    def test_exact_grid_and_optional_pilot_feature_are_accepted(self):
        self.assertEqual(len(self.cells), EXPECTED_CELLS)
        self.reference.validate_protocol(self.manifest, self.cfg, self.cells, self.skipped)
        self.reference.assert_fingerprint(self.cells[0], 'a' * 64)

    def test_changed_draw_or_geometry_is_rejected(self):
        for changed in ({'samples_per_cell': 256, 'batch_size': 8},
                        {'samples_per_cell': 512, 'batch_size': 16}):
            with self.subTest(changed=changed), self.assertRaisesRegex(RuntimeError, '512 samples'):
                self.reference.validate_protocol(changed, self.cfg, self.cells, self.skipped)
        altered = copy.deepcopy(self.cells)
        altered[0]['snr_db'] += 1
        with self.assertRaisesRegex(RuntimeError, 'cell specifications'):
            self.reference.validate_protocol(self.manifest, self.cfg, altered, self.skipped)
        cfg = copy.deepcopy(self.cfg)
        cfg['system']['carrier_frequency_hz'] += 1
        with self.assertRaisesRegex(RuntimeError, 'system configuration'):
            self.reference.validate_protocol(self.manifest, cfg, self.cells, self.skipped)

    def test_changed_waveform_digest_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'waveform mismatch'):
            self.reference.assert_fingerprint(self.cells[0], 'b' * 64)


if __name__ == '__main__':
    unittest.main()
