"""Metadata-only regression cases for completion, selection and seed intervals."""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

from . import prepare_wimae as P
from . import train_wimae as T
from .queue_wimae import select_probe
from .test_wimae_metadata import fixture, JSONCheckpoints
from .verify_wimae import ASSET_NAMES, check_assets, check_paired_intervals, check_seed_intervals
from .wimae_protocol import check_marker, check_training, core_routes


class PipelineMetadataTests(unittest.TestCase):
    def test_completion_requires_all40(self):
        with self.assertRaises(RuntimeError):
            check_marker({"status": "complete", "training_complete": True, "seed": 42,
                          "recipe_signature": "fixture", "completed_epoch": 39}, 42, "fixture")

    def test_route_names_keep_the_two_WiMAE_runs_distinct(self):
        routes = [{"method": method, "training_seed": s, "trained": True} for method in
                  ("dopnet", "dopnet_lite", "lee3ddual", "wimae_ce", "fortitran") for s in (42, 43, 44)]
        indices, selected = core_routes(routes)
        self.assertEqual(indices, list(range(15)))
        self.assertEqual(selected, routes)
        routes[-1]["training_seed"] = 42
        with self.assertRaises(RuntimeError):
            core_routes(routes)

    def test_probe_ranking_accounts_for_both_phases_and_memory(self):
        def candidate(micro, speed, reserved=1):
            return {"status": "passed", "scope": "synthetic_shape_compute_only", "parameters": 554128,
                    "cuda_memory_cap_enforced": True, "gpu_cap_gib": 6, "real_waveforms_generated": False,
                    "waveform_generation_concurrency_certified": False, "gpu_total_gib": 47.4,
                    "gpu_free_before_probe_gib": 46, "probe_override_microbatch": micro,
                    "phases": [{"phase": phase, "microbatch_frames": micro,
                                "expanded_channel_slices_per_microbatch": 50 * micro,
                                "completed_optimizer_steps": 1, "peak_reserved_gib": reserved,
                                "synthetic_frames_per_second": fps} for phase, fps in zip(("ssl", "ce"), speed)]}
        chosen = select_probe([candidate(1, (10, 10)), candidate(2, (80, 5)), candidate(4, (18, 18)),
                               candidate(8, (30, 30), 12), {"status": "failed"}])
        self.assertEqual(chosen["selected_microbatch"], 4)
        self.assertFalse(chosen["waveform_generation_concurrency_certified"])
        with self.assertRaises(RuntimeError):
            select_probe([candidate(1, (float("nan"), 10))])

    def test_completed_marker_is_byte_stable_and_authenticates_ledger(self):
        cfg = P.build_config(fixture(), 42)
        manifest = {"signature": "tiny_metadata_fixture", "source_files_sha256": {}, "implementation_sha256": {}}
        rows = [{"epoch": e, "phase": "ssl" if e <= 20 else "ce", "phase_epoch": e if e <= 20 else e - 20,
                 "train_frames": 12000, "optimizer_steps": 375, "train_loss": 1.0,
                 "val_loss": 1 / e if e <= 20 else abs(e - 30) + .25,
                 "val_nmse_db": None if e <= 20 else -1.0} for e in range(1, 41)]
        def state(epoch, parent=None):
            return {"cfg": cfg, "recipe_signature": manifest["signature"], "source_files_sha256": {},
                    "implementation_sha256": {}, "epoch": epoch, "phase": rows[epoch - 1]["phase"],
                    "task": "ssl" if epoch <= 20 else "ce", "history": copy.deepcopy(rows[:epoch]),
                    "val": {"loss": rows[epoch - 1]["val_loss"]}, "best_ssl_mse": .05,
                    "best_ce_mse": .25 if epoch >= 30 else None, "ssl_parent": parent,
                    "selection_metric": "val_mse", "validation_cache_sha256": P.VALIDATION_SHA256,
                    "training_complete": epoch == 40, "estimator": "wimae_author_slice_ce",
                    "environment": {"torch": "inert_metadata_fixture"},
                    **{k: "inert_placeholder" for k in ("rng_python", "rng_numpy", "rng_torch", "rng_cuda", "rng_private_training_mask")}}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            T.save_state(JSONCheckpoints, folder / "ssl_best.pt", state(20), manifest)
            parent = {"checkpoint_sha256": P.file_hash(folder / "ssl_best.pt"), "epoch": 20, "val_mse": .05,
                      "global_transition_epoch": 21, "optimizer_scheduler_reset": True, "global_training_RNG_preserved": True}
            T.save_state(JSONCheckpoints, folder / "model.pt", state(30, parent), manifest)
            T.save_state(JSONCheckpoints, folder / "last.pt", state(40, parent), manifest)
            deadline = datetime(2050, 1, 1, tzinfo=timezone.utc)
            T.finish(JSONCheckpoints, folder, manifest, cfg, deadline, "first_commit")
            original = (folder / "complete.json").read_bytes()
            T.finish(JSONCheckpoints, folder, manifest, cfg, deadline, "recovered_complete")
            self.assertEqual(original, (folder / "complete.json").read_bytes())
            _, hashes = check_training(folder, manifest, cfg, lambda p: JSONCheckpoints.load_checkpoint(p, "cpu"))
            self.assertEqual(len(hashes), 5)
            rows[0]["train_frames"] = 11999
            T.write_history(folder / "history.csv", rows)
            with self.assertRaises(RuntimeError):
                check_training(folder, manifest, cfg, lambda p: JSONCheckpoints.load_checkpoint(p, "cpu"))

    def test_seed_interval_scalar_formula(self):
        values = (1.0, 2.0, 3.0)
        half = 4.302652729696142 / math.sqrt(3)
        row = dict(metric="nmse", n_training_seeds=3, seed42=1, seed43=2, seed44=3,
                   mean=2, mean_db=10 * math.log10(2), seed_sd=1,
                   seed_ci95_lo_unbounded=2 - half, seed_ci95_hi_unbounded=2 + half)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scalar.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            check_seed_intervals(path)
            row["mean_db"] = sum(10 * math.log10(v) for v in values) / 3
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaises(RuntimeError):
                check_seed_intervals(path)

    def test_no_framework_was_imported(self):
        self.assertNotIn("torch", sys.modules)

    def test_missing_paper_asset_cannot_pass_final_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            ready = dict(status="complete", paper_ready=True, pending=[], evaluation_signature="ev",
                         statistics_signature="stats", assets_sha256={name: "missing" for name in ASSET_NAMES})
            ready["signature"] = P.digest(ready)
            P.write_json(folder / "PAPER_READY.json", ready)
            with self.assertRaises(RuntimeError):
                check_assets(folder, {"signature": "ev"}, {"signature": "stats"})

    def test_paired_summary_requires_all_comparators_and_same_seed_pairs(self):
        summary = [dict(cell_id="fixture", metric="ber_diagonal", method=m, seed42=i / 100,
                        seed43=i / 100, seed44=i / 100) for i, m in enumerate(
                            ("dopnet", "dopnet_lite", "lee3ddual", "wimae_ce", "fortitran", "fortitran_budget_matched", "wimae_author"))]
        pairs = [dict(cell_id="fixture", metric="ber_diagonal", comparator=r["method"], reference="dopnet",
                      paired_difference=r["seed42"], paired_seed_ci95_lo=r["seed42"], paired_seed_ci95_hi=r["seed42"],
                      paired_test_cluster_ci95_lo=r["seed42"], paired_test_cluster_ci95_hi=r["seed42"]) for r in summary[1:]]
        with tempfile.TemporaryDirectory() as temporary:
            def store(name, rows):
                path = Path(temporary) / name
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                return path
            a, b = store("summary.csv", summary), store("paired.csv", pairs)
            check_paired_intervals(a, b)
            store("paired.csv", pairs[:-1])
            with self.assertRaises(RuntimeError):
                check_paired_intervals(a, b)
            replacement = dict(pairs[-1], comparator="dopnet", paired_difference=0,
                               paired_seed_ci95_lo=0, paired_seed_ci95_hi=0,
                               paired_test_cluster_ci95_lo=0, paired_test_cluster_ci95_hi=0)
            store("paired.csv", pairs[:-1] + [replacement])
            with self.assertRaises(RuntimeError):
                check_paired_intervals(a, b)


if __name__ == "__main__":
    unittest.main()
