"""CPU-only protocol check for matched optimizer steps in data-efficiency runs."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from epoch_cache import read_batches
from experiment_protocol import file_hash


class EpochCacheReplayTest(unittest.TestCase):
    def test_replays_unique_prefix_without_extra_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/'batch_000000.pt'
            torch.save({'input':torch.arange(8).reshape(8,1),
                        'noise_var':torch.arange(8).reshape(8,1).float()},path)
            manifest={'frames':8,'batches':[{'file':path.name,'sha256':file_hash(path)}]}
            cfg={'train':{'train_samples':8,'unique_epoch_samples':2,'batch_size':3}}
            with patch('epoch_cache.check_identity',return_value=manifest):
                batches=list(read_batches(root,cfg,1,torch.device('cpu')))
            self.assertEqual([int(b['input'].shape[0]) for b in batches],[3,3,2])
            self.assertEqual(torch.cat([b['input'] for b in batches]).flatten().tolist(),
                             [0,1,0,1,0,1,0,1])
            self.assertEqual(torch.cat([b['noise_var'] for b in batches]).flatten().tolist(),
                             [0.,1.,0.,1.,0.,1.,0.,1.])

    def test_rejects_nonintegral_replay_budget(self):
        with patch('epoch_cache.check_identity',return_value={'frames':8,'batches':[]}):
            with self.assertRaisesRegex(RuntimeError,'unique/shared'):
                list(read_batches('.',{'train':{'train_samples':8,
                    'unique_epoch_samples':3,'batch_size':3}},1,torch.device('cpu')))


if __name__=='__main__':
    unittest.main()
