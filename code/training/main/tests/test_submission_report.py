"""CPU-only contract tests; synthetic rows are never presented as study results."""
import math
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_protocol import CLASSICAL, SEEDS, VARIANTS, digest, file_hash as real_file_hash
import submission_report
from submission_report import (MECHANISM, PROFILE_METHODS, PUBLIC_METHODS,
                               ROBUST, _verify_signed_json, build_figures,
                               build_tables, macro, validate_new_run,
                               validate_receiver_latency)


def synthetic_fixture():
    rows, pairs, profiles, operators = [], [], [], []

    def row(method, scope="all", study="nmse", metric="nmse", modulation="qpsk",
            doppler_hz="all", snr_db="all"):
        mean = (0.12 + 0.001 * len(rows)) if metric == "nmse" else 0.02
        trained = method in VARIANTS
        item = dict(method=method, scope=scope, study=study, metric=metric,
                    modulation=modulation, doppler_hz=doppler_hz, snr_db=snr_db,
                    n_training_seeds=3 if trained else 0, mean=mean,
                    mean_db=10 * math.log10(mean),
                    seed_ci95_lo_db=10 * math.log10(mean * .95),
                    seed_ci95_hi_db=10 * math.log10(mean * 1.05),
                    seed_ci95_lo=mean * .95, seed_ci95_hi=mean * 1.05)
        if trained:
            item.update(seed42=mean * .99, seed43=mean, seed44=mean * 1.01)
        rows.append(item)

    for method in PUBLIC_METHODS:
        for scope in ("all", "gpp_heldout", "gpp_high_mobility"):
            row(method, scope=scope)
        for scope in ("dynamic_all", "gpp_heldout"):
            for metric in ("ber_diagonal", "ber_native"):
                row(method, scope=scope, study="receiver", metric=metric,
                    modulation="16qam")
        if method in ROBUST:
            for study in ("unseen_doppler", "delay_shift", "pilot_cp"):
                row(method, study=study)
    for method in MECHANISM:
        for fd in (0, 400, 800, 1200, 1600):
            row(method, scope="gpp_all", doppler_hz=str(float(fd)), snr_db="15.0")
    for method in ROBUST:
        for fd in (0, 400, 800, 1200, 1600):
            row(method, scope="gpp_heldout", study="receiver",
                metric="ber_diagonal", modulation="16qam",
                doppler_hz=str(float(fd)), snr_db="20.0")

    for method in MECHANISM[1:]:
        for scope in ("all", "gpp_high_mobility", "gpp_heldout"):
            pairs.append(dict(comparator=method, reference="dopnet", scope=scope,
                              study="nmse", metric="nmse", modulation="qpsk",
                              doppler_hz="all", snr_db="all", paired_difference=.01,
                              paired_seed_ci95_lo=.005, paired_seed_ci95_hi=.015,
                              mean_paired_gap_db=.3,
                              paired_gap_db_seed_ci95_lo=.1,
                              paired_gap_db_seed_ci95_hi=.5))
        pairs.append(dict(comparator=method, reference="dopnet", scope="dynamic_all",
                          study="receiver", metric="ber_diagonal", modulation="16qam",
                          doppler_hz="all", snr_db="all", paired_difference=.01,
                          paired_seed_ci95_lo=.005, paired_seed_ci95_hi=.015))

    for method in VARIANTS:
        for batch in (1, 8):
            profiles.append(dict(method=method, seed=42, batch_size=batch,
                                 precision="fp32", execution="eager", warmups=10,
                                 repetitions=100, latency_ms_mean=5.0 * batch,
                                 latency_ms_p50=4.9 * batch,
                                 latency_ms_p95=5.2 * batch,
                                 incremental_peak_mb=100.0,
                                 parameters_trainable=10000))
    for spacing in (2, 4, 8):
        for fast in (True, False):
            operators.append(dict(cp=16, pilot_spacing=spacing, fast_time=fast,
                                  effective_rank_rtol_1e6=100,
                                  retained_singular_condition=1000,
                                  energy_fraction_outside_joint_pilot_tones=0.0))
    return rows, pairs, profiles, operators


def synthetic_receiver_latency():
    methods = list(submission_report.RECEIVER_LATENCY_METHODS)
    modes = list(submission_report.RECEIVER_LATENCY_MODES)
    experiment = {"signature": "e" * 64}
    evaluation = {"checkpoint_sha256": {
        f"{method}_seed{seed}": "c" * 64
        for method in methods for seed in SEEDS}}
    protocol = dict(experiment_signature=experiment["signature"],
                    methods=methods, seeds=list(SEEDS), batch_sizes=[1, 8],
                    receiver_modes=modes,
                    scope="channel estimator forward plus equalizer",
                    precision="fp32", execution="eager, torch.inference_mode, TF32 disabled",
                    timing="host wall clock with CUDA synchronization",
                    warmups=5, repetitions=30,
                    condition=submission_report.RECEIVER_TIMING_CONDITION,
                    input_generation="one fixed synthetic doubly_selective batch per batch size",
                    snr_db=20.0, input_sha256_by_batch_size={"1": "a" * 64, "8": "b" * 64},
                    input_cache_metadata=dict(experiment_signature=experiment["signature"],
                                              batch_sizes=[1, 8],
                                              condition=submission_report.RECEIVER_TIMING_CONDITION,
                                              snr_db=20.0, input_seed=920001,
                                              generation_sha256="d" * 64))
    rows = []
    summaries = []
    for method in methods:
        for batch in (1, 8):
            for mode in modes:
                active = mode == "common_diagonal" or method in submission_report.RECEIVER_LATENCY_NATIVE
                for seed in SEEDS:
                    item = dict(method=method, seed=seed, batch_size=batch,
                                receiver_mode=mode, status="measured" if active else "not_applicable",
                                checkpoint_sha256="c" * 64,
                                input_sha256=protocol["input_sha256_by_batch_size"][str(batch)])
                    if active:
                        item.update(latency_ms_mean=5.0 * batch,
                                    latency_ms_p50=4.9 * batch,
                                    latency_ms_p95=5.2 * batch,
                                    latency_ms_std=.1,
                                    frames_per_second=1000 * batch / (5.0 * batch),
                                    memory_allocated_before_mb=100,
                                    memory_peak_mb=120,
                                    memory_incremental_peak_mb=20,
                                    direct_fallback_repetitions=0,
                                    pcg_iterations_mean=10 if mode == "native_own_ici" else None,
                                    native_residual_max=1e-5 if mode == "native_own_ici" else None)
                    else:
                        item["reason"] = "No native ADD-to-ICI coefficients"
                    rows.append(item)
                if active:
                    summaries.append(dict(method=method, batch_size=batch,
                                          receiver_mode=mode, n_model_seeds=3,
                                          latency_ms_mean_of_seed_means=5.0 * batch,
                                          latency_ms_std_across_seeds=0,
                                          latency_ms_min_seed_mean=5.0 * batch,
                                          latency_ms_max_seed_mean=5.0 * batch,
                                          direct_fallback_repetitions_total=0))
    report = dict(complete=True, protocol=protocol, rows=rows,
                  seed_summary=summaries, hardware={"gpu": "Synthetic GPU"},
                  limitations=["Excludes waveform acquisition and host-to-device transfer; "
                               "not real-time deployment latency."])
    return report, experiment, evaluation


class SubmissionReportTests(unittest.TestCase):
    def test_requires_signed_manifest(self):
        with self.assertRaisesRegex(RuntimeError, "signature"):
            _verify_signed_json({"signature": "not-a-valid-signature"}, "fixture")

    def test_requires_all_three_seeds(self):
        rows, _, _, _ = synthetic_fixture()
        target = next(item for item in rows if item["method"] == "dopnet")
        target.pop("seed44")
        with self.assertRaisesRegex(ValueError, "seed44"):
            macro(rows, "dopnet")

    def test_missing_control_fails_closed(self):
        rows, pairs, profiles, operators = synthetic_fixture()
        rows = [row for row in rows if row["method"] != "pilot_only_fast_time"]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_tables(rows, pairs, profiles, operators)

    def test_all_tables_and_figures_from_synthetic_rows(self):
        rows, pairs, profiles, operators = synthetic_fixture()
        tables = build_tables(rows, pairs, profiles, operators)
        self.assertEqual(len(tables), 7)
        self.assertIn("conditional on the fixed test set", tables["table_mechanism.tex"])
        self.assertIn("model-stage", tables["table_model_stage_profile.tex"].lower())
        with tempfile.TemporaryDirectory() as temp:
            outputs = build_figures(rows, profiles, Path(temp))
            self.assertEqual(len(outputs), 8)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0
                                for path in outputs))

    def test_complete_run_checks_every_synthetic_cell(self):
        """The fixture has all 636 metadata rows but no simulated channel data."""
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "cells").mkdir()
            (out / "summary").mkdir()
            source = out / "synthetic-reference.zip"
            source.write_bytes(b"fixture")
            cells = [dict(cell_id=f"{index:016x}", ber=False)
                     for index in range(636)]
            fingerprints = {cell["cell_id"]: "a" * 64 for cell in cells}
            provenance = {"reference_evaluation_signature": "fixture"}
            reference = SimpleNamespace(
                provenance=lambda: provenance,
                evaluation=dict(cells=cells, skipped_infeasible=[]),
                cell_fingerprints=fingerprints,
                fingerprint="b" * 64,
            )
            configs = {}
            for method in VARIANTS:
                for seed in SEEDS:
                    unique_budget = 12000
                    if method.endswith("_25pct"):
                        unique_budget = 3000
                    elif method.endswith("_50pct"):
                        unique_budget = 6000
                    configs[f"{method}_seed{seed}"] = {"train": {
                        "train_samples": 12000, "shared_epoch_samples": 12000,
                        "unique_epoch_samples": unique_budget}}
            experiment = dict(reference_archive_sha256=submission_report.REFERENCE_SHA256,
                              reference_results=provenance, configs=configs,
                              samples_per_cell=512, batch_size=8)
            experiment["signature"] = digest(experiment)
            routes = ([dict(method=method, training_seed=seed, trained=True)
                       for method in VARIANTS for seed in SEEDS]
                      + [dict(method=method, training_seed=None, trained=False)
                         for method in CLASSICAL + ("perfect_diagonal_csi",)])
            evaluation = dict(fixed_test_reference=provenance,
                              experiment_signature=experiment["signature"],
                              cells=cells, skipped_infeasible=[], routes=routes,
                              checkpoint_sha256={key: "c" * 64 for key in configs})
            evaluation["signature"] = digest(evaluation)
            (out / "experiment.json").write_text(json.dumps(experiment), encoding="utf-8")
            (out / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
            (out / "evaluation_complete.json").write_text(json.dumps(
                dict(signature=evaluation["signature"], cells=636)), encoding="utf-8")
            cluster = np.arange(512) // 8
            nmse = np.zeros((len(routes), 512), dtype=np.float64)
            for meta in cells:
                np.savez_compressed(out / "cells" / (meta["cell_id"] + ".npz"),
                                    signature=np.array(evaluation["signature"]),
                                    meta=np.array(json.dumps(meta)), nmse=nmse,
                                    cluster_id=cluster,
                                    realization_sha256=np.array("a" * 64))
            for name in submission_report.SUMMARY_CSV_NAMES:
                (out / "summary" / name).write_text("fixture,value\n", encoding="utf-8")
            statistics = dict(evaluation_signature=evaluation["signature"],
                              bootstrap=2000, training_seeds=list(SEEDS),
                              data_binding_version=1,
                              cell_file_sha256={meta["cell_id"]: real_file_hash(
                                  out / "cells" / (meta["cell_id"] + ".npz")) for meta in cells},
                              summary_csv_sha256={name: real_file_hash(out / "summary" / name)
                                                  for name in submission_report.SUMMARY_CSV_NAMES})
            statistics["signature"] = digest(statistics)
            (out / "summary" / "statistics.json").write_text(json.dumps(statistics),
                                                                 encoding="utf-8")
            def fixture_file_hash(path):
                return (submission_report.REFERENCE_SHA256 if Path(path) == source
                        else real_file_hash(path))
            with patch.object(submission_report, "file_hash",
                              side_effect=fixture_file_hash), \
                 patch.object(submission_report, "ReferenceResults",
                              return_value=reference):
                validate_new_run(out, source)
                macro_path = out / "summary" / "macro_summary.csv"
                original_csv = macro_path.read_bytes()
                macro_path.write_bytes(original_csv + b"tampered\n")
                with self.assertRaisesRegex(RuntimeError, "Summary CSV differs"):
                    validate_new_run(out, source)
                macro_path.write_bytes(original_csv)
                cell_path = out / "cells" / (cells[0]["cell_id"] + ".npz")
                original_cell = cell_path.read_bytes()
                cell_path.write_bytes(original_cell + b"tampered")
                with self.assertRaisesRegex(RuntimeError, "Raw evaluated cell differs"):
                    validate_new_run(out, source)
                cell_path.write_bytes(original_cell)
                (out / "cells" / (cells[0]["cell_id"] + ".npz")).unlink()
                with self.assertRaisesRegex(RuntimeError, "Raw evaluated cells"):
                    validate_new_run(out, source)

    def test_optional_receiver_latency_requires_complete_three_seed_rows(self):
        report, experiment, evaluation = synthetic_receiver_latency()
        validate_receiver_latency(report, experiment, evaluation)
        rows, pairs, profiles, operators = synthetic_fixture()
        tables = build_tables(rows, pairs, profiles, operators, report)
        self.assertEqual(len(tables), 8)
        self.assertIn("not confidence intervals", tables["table_receiver_stage_latency.tex"])
        report["rows"].pop()
        with self.assertRaisesRegex(RuntimeError, "missing measurement rows"):
            validate_receiver_latency(report, experiment, evaluation)

    def test_partial_receiver_latency_fails(self):
        report, experiment, evaluation = synthetic_receiver_latency()
        report["complete"] = False
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            validate_receiver_latency(report, experiment, evaluation)

    def test_intentionally_common_only_b1_receiver_latency_is_allowed(self):
        report, experiment, evaluation = synthetic_receiver_latency()
        report["protocol"]["batch_sizes"] = [1]
        report["protocol"]["receiver_modes"] = ["common_diagonal"]
        report["protocol"]["input_cache_metadata"]["batch_sizes"] = [1]
        report["protocol"]["input_sha256_by_batch_size"].pop("8")
        report["rows"] = [row for row in report["rows"]
                          if row["batch_size"] == 1 and row["receiver_mode"] == "common_diagonal"]
        report["seed_summary"] = [row for row in report["seed_summary"]
                                  if row["batch_size"] == 1
                                  and row["receiver_mode"] == "common_diagonal"]
        validate_receiver_latency(report, experiment, evaluation)
        rows, pairs, profiles, operators = synthetic_fixture()
        table = build_tables(rows, pairs, profiles, operators, report)[
            "table_receiver_stage_latency.tex"]
        self.assertIn("n/a", table)

    def test_nonpositive_linear_nmse_lower_bound_is_not_fabricated_in_db(self):
        rows, pairs, profiles, operators = synthetic_fixture()
        data_row = next(row for row in rows if row["method"] == "dopnet_25pct"
                        and row["scope"] == "all" and row["study"] == "nmse")
        data_row["seed_ci95_lo"] = 0.0
        data_row["seed_ci95_lo_unbounded"] = -.01
        data_row["seed_ci95_lo_db"] = ""
        mechanism_row = next(row for row in rows if row["method"] == "slow_time"
                             and row["scope"] == "gpp_all" and row["doppler_hz"] == "800.0")
        mechanism_row["seed_ci95_lo"] = 0.0
        mechanism_row["seed_ci95_lo_unbounded"] = -.01
        mechanism_row["seed_ci95_lo_db"] = ""
        table = build_tables(rows, pairs, profiles, operators)["table_data_efficiency.tex"]
        self.assertIn("\\infty", table)
        with tempfile.TemporaryDirectory() as temp:
            outputs = build_figures(rows, profiles, Path(temp))
            self.assertEqual(len(outputs), 8)
            self.assertTrue(all(path.stat().st_size > 0 for path in outputs))

        data_row["seed_ci95_lo_db"] = -30.0
        with self.assertRaisesRegex(ValueError, "finite dB"):
            macro(rows, "dopnet_25pct")


if __name__ == "__main__":
    unittest.main()
