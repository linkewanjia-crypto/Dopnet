"""Scalar/metadata/AST tests only; no Torch, models, generation or GPU work."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys
import unittest

from budget_matched_baselines.recipe import build_config, EarlyStopping, MAX_EPOCHS, window_size
from budget_matched_baselines.migrate_prefix import assert_config_transition, check_committed_prefix


def fixture():
    return {"seed": 42, "device": "cuda",
            "system": {"threegpp_link_batch_size": 8, "n_users": 2, "n_ant": 8,
                       "n_subcarriers": 128, "frame_num_symbols": 25},
            "model": {"estimator": "fortitran", "ablation_profile": "full"},
            "receiver": {"neural_detector": False},
            "train": {"epochs": 40, "train_samples": 12000, "val_samples": 2000,
                      "validation_seed": 610000001, "optimizer": "adamw"},
            "data_pipeline": {"generation_batch_size": 32, "shuffle_buffer": False}}


def prefix_fixture(values=(4.0, 3.0, 5.0, 6.0)):
    rows, best, stale, selected = [], None, 0, None
    for epoch, mse in enumerate(values, 1):
        if best is None or mse < best:
            best, stale, selected = mse, 0, epoch
        else:
            stale += 1
        rows.append({"epoch": epoch, "selection_metric": "val_mse", "train_frames": 12000,
                     "optimizer_steps": 24, "lr": .001 * .995 ** (epoch - 1),
                     "next_lr": .001 * .995 ** epoch, "train_loss": 2.0,
                     "val_loss": mse, "val_nmse_db": -epoch, "stale_epochs": stale})
    last = {"epoch": len(rows), "best_val_loss": best,
            "early_stopping": {"patience": 3, "best": best, "stale": stale},
            "minimum_observed_val_nmse_db": -len(rows), "training_complete": stale >= 3,
            "val": {"loss": values[-1]}, "history": copy.deepcopy(rows)}
    model = {"epoch": selected, "best_val_loss": best,
             "early_stopping": {"patience": 3, "best": best, "stale": 0},
             "training_complete": False, "val": {"loss": values[selected - 1]},
             "history": copy.deepcopy(rows[:selected])}
    return [{key: str(value) for key, value in row.items()} for row in rows], model, last


def parent_config(source, seed):
    """A synthetic parent metadata fixture, independent of the old package."""
    cfg = build_config(source, seed)
    cfg["train"].update(epochs=1000, early_stopping_patience=3)
    metadata = cfg["author_recipe"]
    for key in ("early_stopping_enabled", "termination", "max_epochs", "original_max_epochs",
                "original_early_stopping", "original_repository_patience"):
        metadata.pop(key)
    metadata.update(identifier="fortitran_task_adapted_author_recipe_v1",
                    claim="task_adapted_author_recipe", early_stopping_patience=3,
                    patience_provenance="official_repository_default; paper gives no numerical patience",
                    required_engine="author_baseline_reproduction/train_author.py",
                    warning="train.py alone does not implement this accumulation, MSE selection, or early stopping")
    return cfg


class BudgetRecipeTests(unittest.TestCase):
    def test_fixed_budget_keeps_geometry_and_author_optimizer(self):
        source = fixture()
        before = copy.deepcopy(source)
        cfg = build_config(source, 42)
        self.assertEqual(source, before)
        for key in ("system", "model", "receiver", "data_pipeline"):
            self.assertEqual(cfg[key], source[key])
        self.assertEqual(cfg["train"]["epochs"], 40)
        self.assertEqual(cfg["train"]["batch_size"], 512)
        self.assertEqual(cfg["train"]["optimizer"], "adam")
        self.assertEqual(cfg["train"]["scheduler_gamma"], .995)
        self.assertEqual(cfg["train"]["selection_metric"], "val_mse")
        self.assertEqual(cfg["train"]["amp"], "none")
        self.assertEqual(cfg["train"]["early_stopping_patience"], 41)
        self.assertFalse(cfg["author_recipe"]["early_stopping_enabled"])
        self.assertEqual(cfg["author_recipe"]["claim"], "source_guided_equal_budget")

    def test_migration_only_allows_termination_and_claim_transition(self):
        source = fixture()
        old, new = parent_config(source, 42), build_config(source, 42)
        assert_config_transition(old, new)
        for part, key, value in (("system", "n_ant", 4), ("train", "lr", .002),
                                 ("train", "gradient_microbatch_size", 8)):
            altered = copy.deepcopy(new)
            altered[part][key] = value
            with self.assertRaises(RuntimeError):
                assert_config_transition(old, altered)

    def test_tracker_preserves_best_stale_and_never_stops_within40(self):
        tracker = EarlyStopping(best=1.0, stale=2)
        self.assertEqual(tracker.stale, 2)
        for _ in range(MAX_EPOCHS - 2):
            improved, stop = tracker.update(1.5)
            self.assertFalse(improved)
            self.assertFalse(stop)
        self.assertEqual(tracker.best, 1.0)
        self.assertEqual(tracker.stale, 40)

    def test_committed_prefix_has_exact_mse_best_and_consecutive_epochs(self):
        rows, model, last = prefix_fixture()
        proof = check_committed_prefix(rows, model, last)
        self.assertEqual(proof["completed_epoch"], 4)
        self.assertEqual(proof["selected_epoch"], 2)
        self.assertEqual(proof["stale_epochs"], 2)
        self.assertFalse(proof["parent_naturally_terminal"])
        for field, value in (("train_frames", "11999"), ("optimizer_steps", "23"),
                              ("epoch", "7"), ("stale_epochs", "0")):
            changed = copy.deepcopy(rows)
            changed[-1][field] = value
            with self.assertRaises(RuntimeError):
                check_committed_prefix(changed, model, last)
        model["epoch"] = 1
        with self.assertRaises(RuntimeError):
            check_committed_prefix(rows, model, last)

    def test_parent_early_stop_terminal_does_not_imply_new_completion(self):
        rows, model, last = prefix_fixture((4.0, 3.0, 5.0, 6.0, 7.0))
        proof = check_committed_prefix(rows, model, last)
        self.assertTrue(proof["parent_naturally_terminal"])
        self.assertLess(proof["completed_epoch"], 40)
        altered = copy.deepcopy(last)
        altered["training_complete"] = False
        with self.assertRaises(RuntimeError):
            check_committed_prefix(rows, model, altered)

    def test_effective512_short_tail_exactly12000(self):
        seen, sizes = 0, []
        while seen < 12000:
            size = window_size(12000, seen, 512)
            sizes.append(size)
            seen += size
        self.assertEqual(sizes, [512] * 23 + [224])

    def test_training_ast_has_no_early_stop_branch_and_imports_no_torch(self):
        path = Path(__file__).with_name("train_author.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bad = [node for node in ast.walk(tree) if isinstance(node, ast.If)
               and isinstance(node.test, ast.Name) and node.test.id == "should_stop"]
        self.assertEqual(bad, [])
        direct_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                          and ((isinstance(node, ast.ImportFrom) and node.module == "torch")
                               or (isinstance(node, ast.Import) and any(a.name == "torch" for a in node.names)))]
        self.assertEqual(direct_imports, [])
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
