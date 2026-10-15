"""Metadata/AST/scalar checks only. Never imports torch or runs any model."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

from . import prepare_wimae as P
from . import train_wimae as T


def fixture(seed=42):
    return {"seed": seed, "device": "auto", "model": {"estimator": "wimae_ce"},
            "receiver": {"unchanged": True},
            "system": {"n_users": 2, "n_ant": 8, "n_subcarriers": 128, "frame_num_symbols": 25,
                       "threegpp_link_batch_size": 8},
            "data_pipeline": {"generation_batch_size": 32, "shuffle_buffer": False},
            "train": {"train_samples": 12000, "val_samples": 2000, "validation_seed": 610000001}}


class JSONCheckpoints:
    """Stand-in storage for scalar metadata; no tensors or framework imports."""
    @staticmethod
    def save_checkpoint(value, path):
        Path(path).write_text(json.dumps(value))

    @staticmethod
    def load_checkpoint(path, location):
        return json.loads(Path(path).read_text())


class MetadataTests(unittest.TestCase):
    def test_source_bytes_and_ast(self):
        self.assertEqual(P.verify_upstream()["commit"], P.UPSTREAM_COMMIT)
        for path in P.HERE.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    def test_config_preserves_physics_and_total_budget(self):
        original = fixture()
        copied = copy.deepcopy(original)
        cfg = P.build_config(original, 42)
        self.assertEqual(original, copied)
        for name in ("system", "data_pipeline", "model", "receiver"):
            self.assertEqual(original[name], cfg[name])
        spec = cfg["wimae_reproduction"]
        self.assertEqual(spec["ssl_epochs"] + spec["ce_epochs"], 40)
        self.assertEqual(12000 * spec["total_epochs"], 480000)
        self.assertEqual(spec["slices_per_frame"], 50)
        self.assertFalse(spec["early_stopping"])

    def test_token_geometry_and_paired_mask_rounding(self):
        self.assertEqual(2 * (32 // 1) * (32 // 16), 128)
        self.assertEqual(2 * (8 // 1) * (128 // 16), 128)
        kept = int(64 * (1 - 0.6))
        self.assertEqual(kept * 2, 50)
        self.assertEqual((64 - kept) * 2, 78)
        # Scalar parameter count for the exact upstream FFN128 topology.
        layer = (3 * 64 * 64 + 3 * 64) + (64 * 64 + 64) + (64 * 128 + 128) + (128 * 64 + 64) + 4 * 64
        encoder = 16 * 64 + 64 + 128 * 64 + 12 * layer
        decoder = 64 + 128 * 64 + 4 * layer + 64 * 16 + 16
        self.assertEqual(encoder + decoder, 554128)

    def test_sample_weighted_microbatch_identity(self):
        samples = [float(i) / 7 for i in range(32)]
        expected = sum(samples) / 32
        for micro in (1, 2, 4, 8, 16, 32):
            accumulated = sum((sum(samples[i:i + micro]) / len(samples[i:i + micro]))
                              * len(samples[i:i + micro]) / 32 for i in range(0, 32, micro))
            self.assertAlmostEqual(expected, accumulated)
        self.assertEqual(12000 // 32, 375)

    def test_best_copy_recovery_uses_committed_selected_epoch(self):
        manifest = {"signature": "metadata_fixture"}
        cfg = P.build_config(fixture(), 42)
        rows = [{"epoch": i, "phase": "ssl", "phase_epoch": i, "train_frames": 12000,
                 "optimizer_steps": 375, "val_loss": 2.0 / i} for i in range(1, 4)]
        state = {"cfg": cfg, "recipe_signature": manifest["signature"], "epoch": 3,
                 "phase": "ssl", "history": rows, "val": {"loss": rows[-1]["val_loss"]},
                 "selection_metric": "val_mse",
                 "validation_cache_sha256": P.VALIDATION_SHA256}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            T.save_state(JSONCheckpoints, folder / "last.pt", state, manifest)
            T.recover_best_pointers(JSONCheckpoints, folder, manifest, cfg, state)
            recovered = T.load_state(JSONCheckpoints, folder / "ssl_best.pt", manifest, cfg)
            self.assertEqual(recovered, state)
            # An earlier best whose actual weights were lost is not inventable.
            (folder / "ssl_best.pt").unlink()
            state["history"][-1]["val_loss"] = 3.0
            with self.assertRaises(RuntimeError):
                T.recover_best_pointers(JSONCheckpoints, folder, manifest, cfg, state)

    def test_training_does_not_use_no_grad_wrappers(self):
        source = (P.HERE / "wimae_author.py").read_text()
        tree = ast.parse(source)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        for call in calls:
            if call.func.attr in ("encode", "decode"):
                self.fail("No-grad upstream wrapper used by the training adapter")
        train_tree = ast.parse((P.HERE / "train_wimae.py").read_text())
        generator_calls = [n for n in ast.walk(train_tree) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Attribute) and n.func.attr == "iter_batches"]
        self.assertEqual(len(generator_calls), 1)
        self.assertIsInstance(generator_calls[0].args[1], ast.Name)
        self.assertEqual(generator_calls[0].args[1].id, "epoch")


if __name__ == "__main__":
    unittest.main()
