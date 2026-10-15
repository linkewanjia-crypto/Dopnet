"""Run on the server; no waveform generation or training in these tests."""
import copy
import json
from pathlib import Path
import unittest
import yaml
from data.channels import validate_channel_config
from data.generator import BatchChannelGenerator
from epoch_cache import identity
from experiment_protocol import VARIANTS, make_config

ROOT=Path(__file__).resolve().parents[1]

class ConfigurationContract(unittest.TestCase):
    def setUp(self):
        self.base=yaml.safe_load((ROOT/'configs/dopnet.yaml').read_text())

    def test_actual_eleven_channel_validation_is_idempotent(self):
        system=copy.deepcopy(self.base['system'])
        self.assertEqual(len(system['channel_type_weights']),11)
        validate_channel_config(system)
        expected=copy.deepcopy(system)
        for _ in range(20):
            validate_channel_config(system)
            self.assertEqual(system,expected)

    def test_non_unit_weights_are_normalized_once(self):
        system=copy.deepcopy(self.base['system'])
        system['channel_type_weights']=[w*100 for w in system['channel_type_weights']]
        validate_channel_config(system)
        expected=copy.deepcopy(system)
        self.assertAlmostEqual(sum(system['channel_type_weights']),1.)
        validate_channel_config(system)
        self.assertEqual(system,expected)

    def test_manifest_cache_and_training_paths_match(self):
        reference=identity(make_config(self.base,'dopnet',42),1)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                published=make_config(self.base,variant,42)
                self.assertEqual(published['train']['batch_size'],32)
                self.assertEqual(published['system']['threegpp_link_batch_size'],8)
                cfg=yaml.safe_load(yaml.safe_dump(published))
                before=copy.deepcopy(cfg)
                recorded=json.loads(json.dumps(identity(cfg,1)))
                self.assertEqual(cfg,before)  # identity must not mutate
                validate_channel_config(cfg['system'])  # train.py entry
                self.assertEqual(cfg,published)  # checkpoint vs manifest
                generator=BatchChannelGenerator(cfg,'cuda',seed=10043)
                self.assertEqual(cfg,published)  # constructor must not mutate
                self.assertEqual(recorded,identity(cfg,1))
                self.assertEqual(recorded,reference)
                generator.cfg['system']['channel_type_weights'][0]=0.99
                self.assertEqual(cfg,published)  # no shared nested aliases

    def test_material_data_changes_still_rejected(self):
        cfg=make_config(self.base,'dopnet',42)
        reference=identity(cfg,1)
        changed=copy.deepcopy(cfg)
        changed['system']['snr_db_min']+=1
        self.assertNotEqual(reference,identity(changed,1))
        changed=copy.deepcopy(cfg)
        changed['system']['channel_type_weights'][0]+=.1
        self.assertNotEqual(reference,identity(changed,1))

if __name__=='__main__': unittest.main()
