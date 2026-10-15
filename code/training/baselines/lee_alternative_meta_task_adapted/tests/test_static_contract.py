"""Run locally with standard library only: budget, leakage and chain-rule gates.

These checks do not execute Torch, a model, data generation or GPU timings.
"""
import ast
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lee_meta_equal40 import protocol as P


class StaticContractTests(unittest.TestCase):
    def test_total_budget_and_phases(self):
        self.assertEqual(P.META_EPOCHS + P.ONLINE_EPOCHS, 40)
        self.assertEqual(P.FRESH_FRAMES * P.EPOCHS, 480000)
        self.assertEqual(P.EPISODE_FRAMES * P.UPDATES_PER_EPOCH, 12000)
        self.assertEqual(P.phase_for_epoch(1), "add_meta")
        self.assertEqual(P.phase_for_epoch(2), "joint_online")
        self.assertEqual(P.phase_for_epoch(40), "joint_online")
        for bad in (0, 41, True, 1.0):
            with self.assertRaises(ValueError):
                P.phase_for_epoch(bad)

    def test_support_query_partition_is_a_bijection(self):
        for labels in ([800] * 32, [1620, 0, 800, 1200] * 8):
            support, query = P.episode_partition(labels)
            self.assertEqual((len(support), len(query)), (16, 16))
            self.assertFalse(set(support) & set(query))
            self.assertEqual(sorted(support + query), list(range(32)))
            self.assertLessEqual(max(labels[i] for i in support), min(labels[i] for i in query))
        self.assertEqual(P.episode_partition([0] * 32), (list(range(16)), list(range(16, 32))))

    def test_bad_metadata_is_not_silently_dropped(self):
        for labels in ([0] * 31, [0] * 31 + [math.nan], [0] * 31 + [-1]):
            with self.assertRaises(ValueError):
                P.episode_partition(labels)

    def test_ledger_rejects_incomplete_or_restarted_data(self):
        rows = [dict(global_epoch=e, data_epoch=e, phase=P.phase_for_epoch(e),
                     fresh_frames=12000, optimizer_updates=375,
                     support_frames=6000 if e == 1 else 0,
                     query_frames=6000 if e == 1 else 0) for e in range(1, 41)]
        self.assertEqual(P.check_terminal_ledger(rows)["fresh_frames"], 480000)
        self.assertFalse(P.check_terminal_ledger(rows)["training_delivery_qualified"])
        for bad in (rows[:-1], [dict(r) for r in rows]):
            if len(bad) == 40:
                bad[1]["data_epoch"] = 1
            with self.assertRaises(ValueError):
                P.check_terminal_ledger(bad)
        for key, value in (("fresh_frames", 12001), ("optimizer_updates", 374),
                           ("support_frames", 6001), ("global_epoch", True)):
            bad = [dict(r) for r in rows]
            bad[0][key] = value
            with self.assertRaises(ValueError):
                P.check_terminal_ledger(bad)

    def test_full_chain_rule_differs_from_first_order(self):
        # Analytic one-dimensional MAML, no tensors or training execution.
        theta, support_label, query_label, alpha = 0.7, 1.2, -0.5, 0.2
        phi = theta - alpha * (theta - support_label)
        full = (phi - query_label) * (1 - alpha)
        first_order = phi - query_label
        def objective(value):
            adapted = value - alpha * (value - support_label)
            return (adapted - query_label) ** 2 / 2
        epsilon = 1e-6
        finite_difference = (objective(theta + epsilon) - objective(theta - epsilon)) / (2 * epsilon)
        self.assertAlmostEqual(full, finite_difference, places=8)
        self.assertGreater(abs(full - first_order), 0.1)

    def test_ast_preserves_second_order_base_parameter_gradient(self):
        tree = ast.parse((ROOT / "lee_meta_equal40/meta_updates.py").read_text(encoding="utf-8"))
        grad_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute) and n.func.attr == "grad"]
        self.assertEqual(len(grad_calls), 2)
        self.assertIs(next(k.value.value for k in grad_calls[0].keywords if k.arg == "create_graph"), True)
        self.assertEqual(ast.unparse(grad_calls[1].args[1]), "tuple(theta.values())")
        adapted_assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                                  and any(isinstance(t, ast.Name) and t.id == "adapted" for t in n.targets))
        self.assertNotIn("detach", ast.unparse(adapted_assignment))
        self.assertNotIn("sft", (ROOT / "lee_meta_equal40/meta_updates.py").read_text(encoding="utf-8"))

    def test_ast_parallel_input_and_offline_role_isolation(self):
        tree = ast.parse((ROOT / "lee_meta_equal40/model.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LeeAlternativeMeta")
        functions = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
        forward = ast.unparse(functions["forward"])
        self.assertIn("self.sft(ls) + self._add_from_ls(ls)", forward)
        self.assertNotIn("sft", ast.unparse(functions["add_prediction"]))
        self.assertIn("x[:, :, :2]", ast.unparse(functions["_public_ls"]))
        self.assertNotIn("target", ast.unparse(functions["forward"]))
        self.assertNotIn("doppler", ast.unparse(functions["forward"]))
        self.assertIn("norm='backward'", ast.unparse(functions["_add_from_ls"]))

    def test_recipe_truthfulness_and_deadline(self):
        recipe = json.loads((ROOT / "lee_meta_equal40/recipe.json").read_text(encoding="utf-8"))
        self.assertEqual(recipe["paper_sha256"], "8521e13f8693ee73fe6e8c15f650e78c0225c3ece4e89074647f2f812d12a6e1")
        self.assertEqual(recipe["deadline_utc"], P.DEADLINE_UTC)
        self.assertEqual(recipe["seeds"], [42, 43, 44])
        self.assertEqual(recipe["extra_pretraining_or_support_frames"], 0)
        self.assertEqual(recipe["test_adaptation_steps"], 0)
        for key in ("server_release_qualified", "gpu_probe_started", "training_started",
                    "original_complete_training_recipe_disclosed", "author_code_available"):
            self.assertIs(recipe[key], False)


if __name__ == "__main__":
    unittest.main()
