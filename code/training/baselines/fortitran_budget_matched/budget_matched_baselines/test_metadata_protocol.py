"""Restricted checkpoint metadata tests only; no Torch or model execution."""
from __future__ import annotations

import hashlib
import pickle
import random
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from budget_matched_baselines.migrate_prefix import _validate_checkpoint, PARENT_CLAIM, PRESERVED_STATE
from budget_matched_baselines.recipe import VALIDATION_SHA256
from budget_matched_baselines.verify_author_delivery import (
    MetadataUnpickler, checkpoint_metadata, _safe_latin1_encode,
)


class MetadataProtocolTests(unittest.TestCase):
    def test_protocol2_numpy_rng_preserves_scalar_resume_contract(self):
        cfg = {'train': {'gradient_microbatch_size': 32}}
        parent = {'signature': 'parent-signature', 'source_files_sha256': {'runtime.py': 'same'},
                  'implementation_sha256': {'recipe.py': 'same'}}
        state = dict(
            cfg=cfg, epoch=1, recipe_signature=parent['signature'], recipe_claim=PARENT_CLAIM,
            selection_metric='val_mse', validation_cache_sha256=VALIDATION_SHA256,
            source_files_sha256=parent['source_files_sha256'], implementation_sha256=parent['implementation_sha256'],
            optimizer_name='adam', scheduler_name='exponential', effective_batch_frames=512,
            physical_gradient_microbatch_frames=32, model={'toy_weight': np.array([0.], dtype=np.float32)},
            optimizer={'param_groups': [{'lr': .001 * .995}]}, scheduler={'gamma': .995, 'last_epoch': 1},
            rng_python=random.Random(42).getstate(), rng_numpy=np.random.RandomState(42).get_state(),
            rng_torch=b'inert_test_placeholder', rng_cuda=[], history=[],
        )
        self.assertTrue(set(PRESERVED_STATE) <= set(state))
        payload = pickle.dumps(state, protocol=2)
        self.assertIn(b'_codecs', payload)
        self.assertIn(b'encode', payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'toy_metadata_only.pt'
            with ZipFile(path, 'w') as archive:
                archive.writestr('archive/data.pkl', payload)
            data = path.read_bytes()
            metadata = checkpoint_metadata(data, 'protocol2_numpy_rng')
            sidecar = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                           recipe_signature=parent['signature'], epoch=1)
            _validate_checkpoint(metadata, cfg, parent, sidecar, path)
            self.assertEqual(metadata['rng_numpy'][0], 'MT19937')
            self.assertEqual(metadata['rng_numpy'][2:], state['rng_numpy'][2:])
        self.assertNotIn('torch', sys.modules)

    def test_protocol2_byte_literal_allows_only_bounded_strict_latin1(self):
        self.assertEqual(_safe_latin1_encode('\x00\xff', 'latin1'), b'\x00\xff')
        for args in ((b'bytes', 'latin1'), ('text', 'utf-8'), ('text', 'latin1', 'ignore'),
                     ('x' * (32 * 1024 * 1024 + 1), 'latin1')):
            with self.subTest(kind=type(args[0]).__name__, encoding=args[1]):
                with self.assertRaises(ValueError):
                    _safe_latin1_encode(*args)
        reader = MetadataUnpickler.__new__(MetadataUnpickler)
        with self.assertRaises(ValueError):
            reader.find_class('_codecs', 'lookup')


if __name__ == '__main__':
    unittest.main()
