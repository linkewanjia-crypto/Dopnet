"""Build a conservative, CPU-only TWC submission report from completed results.

The fixed public archive supplies test-realization provenance, never model scores.
Every number in the report comes from the *new* run's summary or profile files.
No model, channel generator, CUDA runtime, or training code is invoked here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np

from experiment_protocol import CLASSICAL, SEEDS, VARIANTS, digest, file_hash
from reference_replay import ReferenceResults


REFERENCE_SHA256 = "3ff13781fe02e2637f8a5654285b883e2bdf2a1aa1889dc05cdecf116ae9a9a4"
PUBLIC_METHODS = tuple(VARIANTS) + tuple(CLASSICAL) + ("perfect_diagonal_csi",)
RETRAIN_MODE = f"retrain_{len(VARIANTS)*len(SEEDS)}"
REUSE_MODE = f"reuse_27_train_{len(VARIANTS)*len(SEEDS)-27}"
EXPECTED_CELLS = 636
EXPECTED_SAMPLES = 512
EXPECTED_BATCH = 8
SUMMARY_CSV_NAMES = ("cell_summary.csv", "cell_paired_differences.csv",
                     "macro_summary.csv", "macro_paired_differences.csv",
                     "receiver_diagnostics.csv")

NAMES = {
    "dopnet": "DOP-Net",
    "dopnet_lite": "DOP-Net-Lite",
    "slow_time": "Slow-time control",
    "pilot_only_fast_time": "Pilot-only fast-time",
    "cfr_derived": "CFR-derived control",
    "aft_only": "AFT-only",
    "no_fusion": "No physics-aware fusion",
    "no_offgrid": "No off-grid refinement",
    "lee3ddual": "Lee (supervised)",
    "wimae_ce": "WiMAE-style CE adapter",
    "fortitran": "FortiTran adapter",
    "ls_interp": "LS interpolation",
    "lmmse_wiener": "Wiener LMMSE",
    "dpsbem_ls": "DPS-BEM LS",
    "pwb_linear": "Linear PWB",
    "perfect_diagonal_csi": "Perfect diagonal CSI",
}

MECHANISM = ("dopnet", "slow_time", "pilot_only_fast_time", "cfr_derived")
COMPARATORS = ("dopnet", "fortitran", "lee3ddual", "wimae_ce", "dopnet_lite",
               "slow_time", "ls_interp", "lmmse_wiener", "dpsbem_ls", "pwb_linear")
ROBUST = ("dopnet", "slow_time", "pilot_only_fast_time",
          "fortitran", "dopnet_lite")
PROFILE_METHODS = ("dopnet", "dopnet_lite", "fortitran", "slow_time",
                   "pilot_only_fast_time", "lee3ddual", "wimae_ce")
RECEIVER_LATENCY_METHODS = ("dopnet", "dopnet_lite", "slow_time", "fortitran")
RECEIVER_LATENCY_NATIVE = frozenset(("dopnet", "dopnet_lite", "slow_time"))
RECEIVER_LATENCY_MODES = ("common_diagonal", "native_own_ici")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
RECEIVER_TIMING_CONDITION = {
    "channel_type": "doubly_selective", "modulation": "qpsk",
    "delay_spread_ns": 300.0, "doppler_hz": 1200.0,
    "cp_len": 16, "pilot_spacing": 4,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def numeric(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Missing/non-numeric {key}: {row}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key}: {row}")
    return value


def _one(rows, **fields):
    matches = [row for row in rows if all(str(row.get(key)) == str(value)
                                       for key, value in fields.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one result, found {len(matches)}: {fields}")
    return matches[0]


def macro(rows, method, scope="all", study="nmse", metric="nmse", modulation="qpsk",
          doppler_hz="all", snr_db="all"):
    row = _one(rows, method=method, scope=scope, study=study, metric=metric,
               modulation=modulation, doppler_hz=doppler_hz, snr_db=snr_db)
    trained = method in VARIANTS
    if int(row["n_training_seeds"]) != (3 if trained else 0):
        raise ValueError(f"Incorrect seed count for {method}: {row}")
    mean = numeric(row, "mean")
    if metric == "nmse":
        if mean <= 0 or not math.isclose(numeric(row, "mean_db"),
                                          10 * math.log10(mean),
                                          rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Inconsistent linear/dB NMSE: {method}")
    elif metric.startswith("ber") and not 0 <= mean <= 1:
        raise ValueError(f"BER outside [0, 1]: {method}")
    if trained:
        values = [numeric(row, f"seed{seed}") for seed in SEEDS]
        if not math.isclose(sum(values) / 3, numeric(row, "mean"), rel_tol=1e-9,
                            abs_tol=1e-10):
            raise ValueError(f"Three-seed mean mismatch: {method}")
        for key in ("seed_ci95_lo", "seed_ci95_hi"):
            numeric(row, key)
        if metric == "nmse":
            low = numeric(row, "seed_ci95_lo")
            high = numeric(row, "seed_ci95_hi")
            if high <= 0:
                raise ValueError(f"Nonpositive NMSE upper confidence bound: {method}")
            if low == 0:
                # aggregate_results.py clips a nonpositive linear t-bound to
                # zero. Its dB transform is -infinity, intentionally blank in
                # CSV, never a fabricated finite value.
                if row.get("seed_ci95_lo_db") not in (None, ""):
                    raise ValueError(f"Nonpositive NMSE lower bound has finite dB value: {method}")
                if numeric(row, "seed_ci95_lo_unbounded") > 0:
                    raise ValueError(f"Inconsistent unclipped NMSE lower bound: {method}")
            elif low < 0 or not math.isclose(numeric(row, "seed_ci95_lo_db"),
                                             10 * math.log10(low),
                                             rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Inconsistent confidence interval dB: {method}")
            if not math.isclose(numeric(row, "seed_ci95_hi_db"),
                                10 * math.log10(high),
                                rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Inconsistent confidence interval dB: {method}")
    return row


def paired(rows, method, scope="all", study="nmse", metric="nmse",
           modulation="qpsk", doppler_hz="all", snr_db="all"):
    if method == "dopnet":
        raise ValueError("DOP-Net is the paired reference, not a comparator")
    row = _one(rows, comparator=method, reference="dopnet", scope=scope,
               study=study, metric=metric, modulation=modulation,
               doppler_hz=doppler_hz, snr_db=snr_db)
    for key in ("paired_difference", "paired_seed_ci95_lo", "paired_seed_ci95_hi"):
        numeric(row, key)
    if metric == "nmse":
        for key in ("mean_paired_gap_db", "paired_gap_db_seed_ci95_lo",
                    "paired_gap_db_seed_ci95_hi"):
            numeric(row, key)
    return row


def _verify_signed_json(obj, label):
    if obj.get("signature") != digest({key: value for key, value in obj.items()
                                        if key != "signature"}):
        raise RuntimeError(f"Invalid {label} signature")


def validate_new_run(out, reference_archive):
    """Read every new cell's provenance; refuse partial or mixed experiments."""
    out = Path(out)
    reference_archive = Path(reference_archive)
    if not reference_archive.is_file() or file_hash(reference_archive) != REFERENCE_SHA256:
        raise RuntimeError("Use the supplied, immutable DOPNet_Results.zip")
    reference = ReferenceResults(reference_archive)
    expected_reference = reference.provenance()
    experiment = read_json(out / "experiment.json")
    evaluation = read_json(out / "evaluation.json")
    done = read_json(out / "evaluation_complete.json")
    stats = read_json(out / "summary" / "statistics.json")
    _verify_signed_json(experiment, "experiment")
    _verify_signed_json(evaluation, "evaluation")
    if stats.get("data_binding_version") != 1:
        raise RuntimeError("Summary lacks versioned raw/CSV SHA-256 bindings; rerun aggregation")
    _verify_signed_json(stats, "summary statistics")
    if experiment.get("reference_archive_sha256") != REFERENCE_SHA256:
        raise RuntimeError("The new experiment is not bound to the supplied reference ZIP")
    if experiment.get("reference_results") != expected_reference:
        raise RuntimeError("Reference provenance differs in experiment.json")
    if evaluation.get("fixed_test_reference") != expected_reference:
        raise RuntimeError("Evaluation did not use the fixed public test realization")
    if evaluation.get("experiment_signature") != experiment["signature"]:
        raise RuntimeError("Evaluation and experiment signatures disagree")
    if done != {"signature": evaluation["signature"], "cells": EXPECTED_CELLS}:
        raise RuntimeError("Evaluation is incomplete or its marker is stale")
    if stats.get("evaluation_signature") != evaluation["signature"]:
        raise RuntimeError("Summary statistics are stale or absent")
    if int(stats.get("bootstrap", 0)) < 1000 or stats.get("training_seeds") != list(SEEDS):
        raise RuntimeError("Summary uncertainty protocol differs")
    summary_hashes = stats.get("summary_csv_sha256", {})
    if set(summary_hashes) != set(SUMMARY_CSV_NAMES):
        raise RuntimeError("Summary CSV SHA-256 manifest is incomplete")
    for name in SUMMARY_CSV_NAMES:
        path = out / "summary" / name
        if not path.is_file() or file_hash(path) != summary_hashes[name]:
            raise RuntimeError(f"Summary CSV differs from aggregation manifest: {name}")
    if experiment.get("samples_per_cell") != EXPECTED_SAMPLES or experiment.get("batch_size") != EXPECTED_BATCH:
        raise RuntimeError("The fixed test requires exactly 512 frames in batches of 8")
    if evaluation.get("cells") != reference.evaluation["cells"]:
        raise RuntimeError("New evaluation changed the fixed 636-cell test grid")
    if evaluation.get("skipped_infeasible") != reference.evaluation["skipped_infeasible"]:
        raise RuntimeError("Infeasible-grid accounting differs from reference")
    if len(evaluation["cells"]) != EXPECTED_CELLS:
        raise RuntimeError("Expected 636 evaluation cells")

    expected_names = {f"{method}_seed{seed}" for method in VARIANTS for seed in SEEDS}
    if set(experiment.get("configs", {})) != expected_names:
        raise RuntimeError(f"The experiment must have all {len(expected_names)} learned configurations")
    if set(evaluation.get("checkpoint_sha256", {})) != expected_names:
        raise RuntimeError("Evaluation checkpoint manifest is incomplete")
    mode = experiment.get("training_mode", RETRAIN_MODE)
    if mode not in (RETRAIN_MODE, REUSE_MODE):
        raise RuntimeError("Unknown training provenance mode")
    if mode == REUSE_MODE:
        from historical_reuse import OLD_VARIANTS
        old_names = {f"{method}_seed{seed}" for method in OLD_VARIANTS for seed in SEEDS}
        expected_hashes = reference.evaluation["checkpoint_sha256"]
        if experiment.get("reused_checkpoint_sha256") != expected_hashes:
            raise RuntimeError("Reused checkpoint source list differs from the immutable reference")
        for name in old_names:
            marker = read_json(out / "checkpoints" / name / "complete.json")
            if (marker.get("signature") != experiment["signature"]
                    or marker.get("source_checkpoint_sha256") != expected_hashes[name]
                    or marker.get("normalized_checkpoint_sha256")
                    != evaluation["checkpoint_sha256"][name]
                    or file_hash(out / "checkpoints" / name / "model.pt")
                    != marker["normalized_checkpoint_sha256"]):
                raise RuntimeError(f"Reused checkpoint provenance differs: {name}")
    if experiment.get("historical_data_comparison_required") is not True:
        raise RuntimeError("New run lacks the strict historical-data comparison contract")
    if not experiment.get("historical_training_manifest_signature"):
        raise RuntimeError("Historical training-data manifest provenance is missing")
    historical_manifest_hashes = {}
    for seed in SEEDS:
        for epoch in range(1, 41):
            proof_path = out / "historical_data_match" / f"seed{seed}_epoch{epoch:03d}.json"
            proof = read_json(proof_path)
            _verify_signed_json(proof, "historical training-data proof")
            old_sha = proof.get("historical_manifest_sha256")
            if not isinstance(old_sha, str) or not HEX64.fullmatch(old_sha):
                raise RuntimeError("Historical training-data parent digest is invalid")
            historical_manifest_hashes[f"{seed}_{epoch}"] = old_sha
            if (proof.get("seed") != seed or proof.get("epoch") != epoch
                    or proof.get("matched_batches") != 375
                    or proof.get("matched_frames") != 12000):
                raise RuntimeError("Incomplete historical training-data proof")
    if digest(historical_manifest_hashes) != experiment["historical_training_manifest_signature"]:
        raise RuntimeError("Historical training-data proof parent set differs from experiment")
    expected_routes = ([{"method": method, "training_seed": seed, "trained": True}
                        for method in VARIANTS for seed in SEEDS]
                       + [{"method": method, "training_seed": None, "trained": False}
                          for method in CLASSICAL + ("perfect_diagonal_csi",)])
    if evaluation.get("routes") != expected_routes:
        raise RuntimeError(f"Expected exactly {len(expected_names)} learned routes and five fixed references")

    # The data-efficiency axis varies unique realizations, while optimizer
    # steps and the shared generated stream remain fixed across every variant.
    configs = experiment["configs"]
    for seed in SEEDS:
        full = int(configs[f"dopnet_seed{seed}"]["train"]["train_samples"])
        for method in VARIANTS:
            train = configs[f"{method}_seed{seed}"]["train"]
            if (int(train["train_samples"]) != full
                    or int(train.get("shared_epoch_samples", full)) != full):
                raise RuntimeError(f"Optimization/shared stream budget differs: {method}, seed {seed}")
            expected_unique = (full // 4 if method.endswith("_25pct") else
                               full // 2 if method.endswith("_50pct") else full)
            if int(train.get("unique_epoch_samples", full)) != expected_unique:
                raise RuntimeError(f"Incorrect unique training realization budget: {method}, seed {seed}")

    cell_dir = out / "cells"
    expected_files = {f"{meta['cell_id']}.npz" for meta in evaluation["cells"]}
    if {path.name for path in cell_dir.glob("*.npz")} != expected_files:
        raise RuntimeError("Raw evaluated cells are missing or extra cells are present")
    cell_hashes = stats.get("cell_file_sha256", {})
    if set(cell_hashes) != {meta["cell_id"] for meta in evaluation["cells"]}:
        raise RuntimeError("Raw-cell SHA-256 manifest is incomplete")
    expected_cluster = np.arange(EXPECTED_SAMPLES) // EXPECTED_BATCH
    for meta in evaluation["cells"]:
        cell_id = meta["cell_id"]
        path = cell_dir / f"{cell_id}.npz"
        if file_hash(path) != cell_hashes[cell_id]:
            raise RuntimeError(f"Raw evaluated cell differs from aggregation manifest: {cell_id}")
        with np.load(path, allow_pickle=False) as cell:
            if str(cell["signature"].item()) != evaluation["signature"]:
                raise RuntimeError(f"Stale cell: {cell_id}")
            if json.loads(str(cell["meta"].item())) != meta:
                raise RuntimeError(f"Cell metadata differs: {cell_id}")
            if cell["nmse"].shape != (len(expected_routes), EXPECTED_SAMPLES):
                raise RuntimeError(f"Incomplete NMSE tensor: {cell_id}")
            if not np.isfinite(cell["nmse"]).all() or (cell["nmse"] < 0).any():
                raise RuntimeError(f"Invalid NMSE values: {cell_id}")
            if not np.array_equal(cell["cluster_id"], expected_cluster):
                raise RuntimeError(f"Generation clusters differ: {cell_id}")
            if str(cell["realization_sha256"].item()) != reference.cell_fingerprints[cell_id]:
                raise RuntimeError(f"Fixed test waveform differs: {cell_id}")
            if meta["ber"]:
                for name in ("bits", "errors_diagonal", "errors_native"):
                    if cell[name].shape != (len(expected_routes), EXPECTED_SAMPLES):
                        raise RuntimeError(f"Incomplete {name}: {cell_id}")
                bits = cell["bits"]
                if (bits <= 0).any():
                    raise RuntimeError(f"Invalid bit denominator: {cell_id}")
                for name in ("errors_diagonal", "errors_native"):
                    errors = cell[name]
                    if (errors < 0).any() or (errors > bits).any():
                        raise RuntimeError(f"Invalid bit-error counts in {name}: {cell_id}")
    return experiment, evaluation, stats, reference


def validate_profile(profile):
    if not profile.get("gpu") or "model-stage" not in profile.get("note", "").lower():
        raise RuntimeError("Missing GPU identity or model-stage latency limitation")
    rows = profile.get("rows", [])
    for method in VARIANTS:
        for batch in (1, 8):
            row = _one(rows, method=method, seed=42, batch_size=batch)
            if row.get("precision") != "fp32" or row.get("execution") != "eager":
                raise RuntimeError(f"Latency protocol differs for {method}, B={batch}")
            if int(row.get("warmups", 0)) < 10 or int(row.get("repetitions", 0)) < 100:
                raise RuntimeError("Insufficient latency repetitions")
            for key in ("latency_ms_mean", "latency_ms_p50", "latency_ms_p95",
                        "incremental_peak_mb", "parameters_trainable"):
                if numeric(row, key) < 0:
                    raise RuntimeError("Invalid latency/profile value")
    return rows


def validate_receiver_latency(report, experiment, evaluation):
    """Reject a partial or mixed receiver-stage timing artifact if it exists."""
    if report.get("complete") is not True:
        raise RuntimeError("receiver_latency.json is present but incomplete")
    protocol = report.get("protocol", {})
    if protocol.get("experiment_signature") != experiment["signature"]:
        raise RuntimeError("Receiver timing belongs to another experiment")
    batches = protocol.get("batch_sizes")
    modes = protocol.get("receiver_modes")
    if (protocol.get("methods") != list(RECEIVER_LATENCY_METHODS)
            or protocol.get("seeds") != list(SEEDS)
            or batches not in ([1], [1, 8])
            or modes not in (["common_diagonal"], list(RECEIVER_LATENCY_MODES))):
        raise RuntimeError("Receiver timing is missing methods, seeds, batch sizes, or modes")
    if (protocol.get("scope") != "channel estimator forward plus equalizer"
            or protocol.get("precision") != "fp32"
            or "eager" not in protocol.get("execution", "")
            or "synchronization" not in protocol.get("timing", "")
            or protocol.get("condition") != RECEIVER_TIMING_CONDITION
            or float(protocol.get("snr_db", -1)) != 20.0
            or protocol.get("input_generation") != "one fixed synthetic doubly_selective batch per batch size"
            or int(protocol.get("warmups", 0)) < 1
            or int(protocol.get("repetitions", 0)) < 10):
        raise RuntimeError("Receiver timing protocol is incomplete or has changed")
    metadata = protocol.get("input_cache_metadata", {})
    if (metadata.get("experiment_signature") != experiment["signature"]
            or metadata.get("batch_sizes") != batches
            or metadata.get("condition") != protocol.get("condition")
            or metadata.get("snr_db") != protocol.get("snr_db")
            or metadata.get("input_seed") != 920001
            or not HEX64.fullmatch(str(metadata.get("generation_sha256", "")))):
        raise RuntimeError("Receiver timing input-cache provenance differs")
    hashes = protocol.get("input_sha256_by_batch_size", {})
    if set(hashes) != {str(batch) for batch in batches} or any(
            not HEX64.fullmatch(str(value)) for value in hashes.values()):
        raise RuntimeError("Receiver timing input hashes are missing or malformed")
    if 8 in batches and hashes["1"] == hashes["8"]:
        raise RuntimeError("Distinct receiver batch sizes have the same input hash")
    if not report.get("hardware", {}).get("gpu"):
        raise RuntimeError("Receiver timing GPU identity is missing")
    limitations = " ".join(report.get("limitations", [])).lower()
    if not all(term in limitations for term in ("excludes", "host-to-device", "real-time")):
        raise RuntimeError("Receiver timing limitations are missing")

    expected = {(method, seed, batch, mode)
                for method in RECEIVER_LATENCY_METHODS for seed in SEEDS
                for batch in batches for mode in modes}
    indexed = {}
    for row in report.get("rows", []):
        key = (row.get("method"), row.get("seed"), row.get("batch_size"),
               row.get("receiver_mode"))
        if key not in expected or key in indexed:
            raise RuntimeError(f"Unexpected/duplicate receiver timing row: {key}")
        indexed[key] = row
        method, seed, batch, mode = key
        if row.get("input_sha256") != hashes[str(batch)]:
            raise RuntimeError(f"Receiver input hash mismatch: {key}")
        checkpoint = evaluation["checkpoint_sha256"][f"{method}_seed{seed}"]
        if row.get("checkpoint_sha256") != checkpoint:
            raise RuntimeError(f"Receiver checkpoint hash mismatch: {key}")
        should_measure = mode == "common_diagonal" or method in RECEIVER_LATENCY_NATIVE
        if row.get("status") != ("measured" if should_measure else "not_applicable"):
            raise RuntimeError(f"Receiver timing row was not completed: {key}")
        if should_measure:
            for field in ("latency_ms_mean", "latency_ms_p50", "latency_ms_p95",
                          "frames_per_second"):
                if numeric(row, field) <= 0:
                    raise RuntimeError(f"Invalid receiver timing: {key}")
            for field in ("latency_ms_std", "memory_allocated_before_mb",
                          "memory_peak_mb", "memory_incremental_peak_mb"):
                if numeric(row, field) < 0:
                    raise RuntimeError(f"Invalid receiver profiling value: {key}")
            fallback = int(row.get("direct_fallback_repetitions", -1))
            if not 0 <= fallback <= int(protocol["repetitions"]):
                raise RuntimeError(f"Invalid native direct-fallback count: {key}")
            if mode == "common_diagonal":
                if fallback != 0 or row.get("pcg_iterations_mean") is not None:
                    raise RuntimeError(f"Common receiver mixed with native ICI protocol: {key}")
            else:
                if (numeric(row, "pcg_iterations_mean") < 1
                        or not 0 <= numeric(row, "native_residual_max") <= 1e-4):
                    raise RuntimeError(f"Native ICI solve diagnostics failed: {key}")
        elif not row.get("reason"):
            raise RuntimeError(f"Missing native-unavailable reason: {key}")
    if set(indexed) != expected:
        raise RuntimeError("receiver_latency.json has missing measurement rows")

    summaries = {}
    for item in report.get("seed_summary", []):
        key = (item.get("method"), item.get("batch_size"), item.get("receiver_mode"))
        if key in summaries:
            raise RuntimeError(f"Duplicate receiver seed summary: {key}")
        summaries[key] = item
    expected_summaries = {(method, batch, mode)
                          for method in RECEIVER_LATENCY_METHODS for batch in batches
                          for mode in modes
                          if mode == "common_diagonal" or method in RECEIVER_LATENCY_NATIVE}
    if set(summaries) != expected_summaries:
        raise RuntimeError("Receiver seed summaries are incomplete")
    for (method, batch, mode), item in summaries.items():
        if int(item.get("n_model_seeds", -1)) != 3:
            raise RuntimeError("Receiver timing must summarize all three model seeds")
        records = [indexed[(method, seed, batch, mode)] for seed in SEEDS]
        values = [numeric(row, "latency_ms_mean") for row in records]
        checks = (("latency_ms_mean_of_seed_means", sum(values) / 3),
                  ("latency_ms_std_across_seeds", statistics.stdev(values)),
                  ("latency_ms_min_seed_mean", min(values)),
                  ("latency_ms_max_seed_mean", max(values)))
        for field, expected_value in checks:
            if not math.isclose(numeric(item, field), expected_value,
                                rel_tol=1e-9, abs_tol=1e-9):
                raise RuntimeError(f"Receiver seed summary disagrees with rows: {method}, {field}")
        if int(item.get("direct_fallback_repetitions_total", -1)) != sum(
                int(row["direct_fallback_repetitions"]) for row in records):
            raise RuntimeError(f"Receiver fallback summary disagrees: {method}")
    return report


def _db_ci(row):
    lower = (None if numeric(row, "seed_ci95_lo") == 0
             else numeric(row, "seed_ci95_lo_db"))
    return (numeric(row, "mean_db"), lower, numeric(row, "seed_ci95_hi_db"))


def _fmt_db_ci(row):
    mid, lo, hi = _db_ci(row)
    if lo is None:
        return f"{mid:.2f} \\((-\\infty, {hi:.2f}]\\)"
    return f"{mid:.2f} [{lo:.2f}, {hi:.2f}]"


def _fmt_ber(row):
    mid = 100 * numeric(row, "mean")
    if int(row["n_training_seeds"]) == 3:
        lo = 100 * numeric(row, "seed_ci95_lo")
        hi = 100 * numeric(row, "seed_ci95_hi")
        return f"{mid:.3f} [{lo:.3f}, {hi:.3f}]"
    return f"{mid:.3f}"


def _fmt_gap(row):
    return (f"{numeric(row, 'mean_paired_gap_db'):.2f} "
            f"[{numeric(row, 'paired_gap_db_seed_ci95_lo'):.2f}, "
            f"{numeric(row, 'paired_gap_db_seed_ci95_hi'):.2f}]")


def _fmt_ber_gap(row):
    return (f"{100 * numeric(row, 'paired_difference'):.3f} "
            f"[{100 * numeric(row, 'paired_seed_ci95_lo'):.3f}, "
            f"{100 * numeric(row, 'paired_seed_ci95_hi'):.3f}]")


def _tex_table(label, caption, columns, rows, alignment):
    lines = ["\\begin{table*}[t]", "\\centering", "\\footnotesize", f"\\caption{{{caption}}}",
             f"\\label{{tab:{label}}}", f"\\begin{{tabular}}{{{alignment}}}",
             "\\toprule", " & ".join(columns) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(value) for value in row) + " \\\\"
              for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def build_tables(rows, pairs, profile_rows, operator_rows, receiver_latency=None):
    """Demand every selected row before returning any manuscript table."""
    tables = {}

    result = []
    for method in COMPARATORS:
        all_row = macro(rows, method)
        held = macro(rows, method, scope="gpp_heldout")
        high = macro(rows, method, scope="gpp_high_mobility")
        ber = macro(rows, method, scope="dynamic_all", study="receiver",
                    metric="ber_diagonal", modulation="16qam")
        result.append((NAMES[method], f"{numeric(all_row, 'mean_db'):.2f}",
                       f"{numeric(held, 'mean_db'):.2f}",
                       f"{numeric(high, 'mean_db'):.2f}",
                       f"{100 * numeric(ber, 'mean'):.3f}"))
    tables["table_comparison.tex"] = _tex_table(
        "comparison",
        "Task-matched comparison on the historical fixed 636-cell test set, "
        "previously inspected and not a fresh blind holdout. NMSE is in dB; "
        "16-QAM BER is a percentage under the common diagonal detector. Each learned "
        "entry averages training seeds 42--44. Held-out denotes TDL-A/E and CDL-A/E; "
        "high mobility denotes 3GPP profiles with maximum Doppler at least 800 Hz. "
        "Held-out refers to profile types excluded from training, not a new blind "
        "test realization. Classical methods have no training-seed uncertainty.",
        ("Estimator", "All NMSE", "Held-out NMSE", "High-mob. NMSE", "16-QAM BER (\\%)"),
        result, "lrrrr")

    result = []
    for method in MECHANISM[1:]:
        gaps = [_fmt_gap(paired(pairs, method, scope=scope))
                for scope in ("all", "gpp_high_mobility", "gpp_heldout")]
        ber = _fmt_ber_gap(paired(pairs, method, scope="dynamic_all", study="receiver",
                                  metric="ber_diagonal", modulation="16qam"))
        result.append((NAMES[method], *gaps, ber))
    tables["table_mechanism.tex"] = _tex_table(
        "mechanism",
        "Paired control differences relative to DOP-Net on identical test realizations "
        "and matched training seeds. Positive NMSE gap (dB) or BER gap (percentage points) "
        "means the control is worse. Brackets are pointwise 95\\% training-seed "
        "\\(t\\) intervals with two degrees of freedom, conditional on the fixed test set; "
        "these are exploratory, not multiplicity-corrected claims. The historical "
        "fixed test realization was previously inspected.",
        ("Control", "All NMSE gap", "High-mob. gap", "Held-out gap", "16-QAM BER gap (pp)"),
        result, "lcccc")

    result = []
    for family, ids in (("DOP-Net", ("dopnet_25pct", "dopnet_50pct", "dopnet")),
                        ("Slow-time control", ("slow_time_25pct", "slow_time_50pct", "slow_time"))):
        for percent, method in zip((25, 50, 100), ids):
            all_row = macro(rows, method)
            held = macro(rows, method, scope="gpp_heldout")
            high = macro(rows, method, scope="gpp_high_mobility")
            result.append((family, str(percent), _fmt_db_ci(all_row),
                           _fmt_db_ci(held), _fmt_db_ci(high)))
    tables["table_data_efficiency.tex"] = _tex_table(
        "data-efficiency",
        "Training-data efficiency at 25, 50, and 100\\% unique per-epoch "
        "realizations, with the same 12,000-frame optimizer input and update count "
        "for all budgets; reduced-unique variants repeat a deterministic shared-cache "
        "prefix. NMSE is in dB, with pointwise 95\\% training-seed "
        "\\(t\\) intervals (three seeds, two degrees of freedom), conditional on "
        "the historical fixed test realizations. Models are trained separately at each "
        "budget. If a lower linear NMSE bound is nonpositive, its dB bound is "
        "\\((-\\infty,u]\\); no finite lower dB limit is imputed.",
        ("Estimator", "Unique data (\\%)", "All NMSE [95\\% CI]",
         "Held-out [95\\% CI]", "High-mob. [95\\% CI]"), result, "lcccc")

    result = []
    for method in ROBUST:
        values = [_fmt_db_ci(macro(rows, method, scope="all", study=study))
                  for study in ("unseen_doppler", "delay_shift", "pilot_cp")]
        result.append((NAMES[method], *values))
    tables["table_robustness.tex"] = _tex_table(
        "robustness",
        "Additional fixed-grid robustness studies. Entries are NMSE in dB with pointwise "
        "95\\% three-training-seed intervals conditional on the test set. The Doppler "
        "interpolation, delay-shift, and CP/pilot grids contain only physically feasible "
        "cells. The historical fixed test realization was previously inspected; a "
        "nonpositive lower linear NMSE bound is shown as \\((-\\infty,u]\\).",
        ("Estimator", "Unseen Doppler [95\\% CI]", "Delay shift [95\\% CI]",
         "CP / pilot [95\\% CI]"), result, "lccc")

    result = []
    for method in ROBUST + ("perfect_diagonal_csi",):
        diagonal = macro(rows, method, scope="dynamic_all", study="receiver",
                         metric="ber_diagonal", modulation="16qam")
        native = macro(rows, method, scope="dynamic_all", study="receiver",
                       metric="ber_native", modulation="16qam")
        held = macro(rows, method, scope="gpp_heldout", study="receiver",
                     metric="ber_diagonal", modulation="16qam")
        result.append((NAMES[method], _fmt_ber(diagonal), _fmt_ber(native),
                       _fmt_ber(held)))
    tables["table_receiver.tex"] = _tex_table(
        "receiver",
        "16-QAM BER percentages. The common diagonal detector is the primary "
        "cross-method comparison. The native column uses each method's own available "
        "ICI information, otherwise the diagonal detector, and therefore must not be "
        "read as a controlled equalizer comparison. Brackets on learned methods are "
        "pointwise 95\\% training-seed intervals, conditional on the test set. "
        "Perfect diagonal CSI is a diagonal reference, not a full-ICI oracle.",
        ("Estimator", "Common diagonal [95\\% CI]", "Native [95\\% CI]",
         "Held-out diagonal [95\\% CI]"), result, "lccc")

    result = []
    for method in PROFILE_METHODS:
        one = _one(profile_rows, method=method, seed=42, batch_size=1)
        eight = _one(profile_rows, method=method, seed=42, batch_size=8)
        result.append((NAMES[method], f"{int(numeric(one, 'parameters_trainable')):,}",
                       f"{numeric(one, 'latency_ms_mean'):.2f}",
                       f"{numeric(one, 'latency_ms_p95'):.2f}",
                       f"{numeric(eight, 'latency_ms_mean'):.2f}",
                       f"{numeric(one, 'incremental_peak_mb'):.1f}"))
    tables["table_model_stage_profile.tex"] = _tex_table(
        "model-stage-profile",
        "Model-stage-only profiling on the recorded GPU, seed-42 checkpoint, "
        "FP32 eager execution, ten warmups, and 100 timed repetitions. Latency excludes "
        "waveform acquisition, LS/interpolation, ADD preprocessing, equalization, and "
        "host--device transfer. Times are descriptive repeats, not training-seed intervals "
        "or end-to-end receiver latency.",
        ("Estimator", "Parameters", "\\(B=1\\) mean ms", "\\(B=1\\) p95 ms",
         "\\(B=8\\) mean ms", "Incremental peak MiB"), result, "lrrrrr")

    result = []
    for spacing in (2, 4, 8):
        for fast in (True, False):
            item = _one(operator_rows, cp=16, pilot_spacing=spacing, fast_time=fast)
            result.append((str(spacing), "Fast-time ADD" if fast else "Slow-time ADD",
                           str(int(numeric(item, "effective_rank_rtol_1e6"))),
                           f"{numeric(item, 'retained_singular_condition'):.2e}",
                           f"{numeric(item, 'energy_fraction_outside_joint_pilot_tones'):.3f}"))
    tables["table_operator.tex"] = _tex_table(
        "operator",
        "Linear-map audit from received occupied pilot-symbol bins to the user-0 "
        "auxiliary delay--Doppler representation at CP=16. The rank threshold is "
        "\\(10^{-6}\\). Both maps are deterministic transforms of the same received "
        "frame; the table does not claim additional raw measurements or a channel "
        "forward-model rank advantage.",
        ("Pilot spacing", "Map", "Effective rank", "Retained condition", "Outside joint pilots"),
        result, "ccrrr")

    if receiver_latency is not None:
        summaries = receiver_latency["seed_summary"]
        timing_protocol = receiver_latency["protocol"]
        repetitions = int(timing_protocol["repetitions"])
        def interval(method, batch, mode):
            item = _one(summaries, method=method, batch_size=batch, receiver_mode=mode)
            return (f"{numeric(item, 'latency_ms_mean_of_seed_means'):.2f} "
                    f"[{numeric(item, 'latency_ms_min_seed_mean'):.2f}, "
                    f"{numeric(item, 'latency_ms_max_seed_mean'):.2f}]")
        result = []
        for method in RECEIVER_LATENCY_METHODS:
            common_one = interval(method, 1, "common_diagonal")
            common_eight = (interval(method, 8, "common_diagonal")
                            if 8 in timing_protocol["batch_sizes"] else "n/a")
            if (method in RECEIVER_LATENCY_NATIVE
                    and "native_own_ici" in timing_protocol["receiver_modes"]):
                native = interval(method, 1, "native_own_ici")
                item = _one(summaries, method=method, batch_size=1,
                            receiver_mode="native_own_ici")
                fallback = (f"{int(item['direct_fallback_repetitions_total'])}/"
                            f"{3 * repetitions}")
            else:
                native = "n/a"
                fallback = "n/a"
            result.append((NAMES[method], common_one, common_eight, native, fallback))
        gpu = receiver_latency["hardware"]["gpu"].replace("_", "\\_")
        tables["table_receiver_stage_latency.tex"] = _tex_table(
            "receiver-stage-latency",
            "Synchronized FP32 eager timing on " + gpu + ". The comparable common "
            "diagonal column times estimator forward plus the same diagonal MMSE equalizer. "
            "Each entry is the mean of three checkpoint-seed mean times, with the "
            "seed-mean minimum and maximum in brackets; these are descriptive ranges, "
            "not confidence intervals. \\(B=8\\), where measured, is total batch time. "
            "Native ADD-to-ICI timing, where measured, uses a separate banded equalizer "
            "and is not directly comparable with "
            "the common diagonal protocol. Timing excludes waveform acquisition and "
            "simulation, pilot LS/interpolation, fast/slow-time ADD construction, "
            "host--device transfer, demapping, and BER scoring; it is not end-to-end "
            "or over-the-air latency.",
            ("Estimator", "Common \\(B=1\\) ms", "Common \\(B=8\\) ms",
             "Native \\(B=1\\) ms", "Direct fallbacks"), result, "lcccc")
    return tables


def _nmse_display_floor(rows, methods, scope, study, modulation, snr_db):
    """Visual clipping floor only; never a reported confidence limit."""
    finite = []
    for method in methods:
        for fd in (0, 400, 800, 1200, 1600):
            item = macro(rows, method, scope=scope, study=study, metric="nmse",
                         modulation=modulation, doppler_hz=str(float(fd)),
                         snr_db=str(float(snr_db)))
            finite.extend(value for value in _db_ci(item) if value is not None)
    return min(finite) - 2.0


def _draw_band(ax, rows, method, scope, study, metric, modulation, snr_db, color=None,
               marker="o", as_percent=False, nmse_display_floor=None):
    selected = []
    for fd in (0, 400, 800, 1200, 1600):
        row = macro(rows, method, scope=scope, study=study, metric=metric,
                    modulation=modulation, doppler_hz=str(float(fd)), snr_db=str(float(snr_db)))
        selected.append(row)
    x = np.array((0, 400, 800, 1200, 1600), dtype=float)
    if metric == "nmse":
        triplets = [_db_ci(row) for row in selected]
        y = np.array([item[0] for item in triplets])
        high = np.array([item[2] for item in triplets])
        open_lower = np.array([item[1] is None for item in triplets])
        if open_lower.any() and nmse_display_floor is None:
            raise ValueError("Unbounded dB CI requires a labeled display clipping floor")
        low = np.array([nmse_display_floor if item[1] is None else item[1]
                        for item in triplets])
    else:
        scale = 100 if as_percent else 1
        y, low, high = [scale * np.array([numeric(row, key) for row in selected])
                        for key in ("mean", "seed_ci95_lo", "seed_ci95_hi")]
        open_lower = np.zeros(len(x), dtype=bool)
    line, = ax.plot(x, y, marker=marker, label=NAMES[method], color=color)
    ax.fill_between(x, low, high, color=line.get_color(), alpha=.13)
    if open_lower.any():
        ax.scatter(x[open_lower], low[open_lower], marker="v", s=55,
                   facecolors="none", edgecolors=line.get_color(), zorder=4)
    return bool(open_lower.any())


def build_figures(rows, profile_rows, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    outputs = []

    fig, ax = plt.subplots(figsize=(7.2, 4.3), layout="constrained")
    floor = _nmse_display_floor(rows, MECHANISM, "gpp_all", "nmse", "qpsk", 15)
    has_open_lower = False
    for method in MECHANISM:
        has_open_lower |= _draw_band(ax, rows, method, "gpp_all", "nmse", "nmse",
                                     "qpsk", 15, nmse_display_floor=floor)
    ax.set(xlabel="Maximum Doppler (Hz)", ylabel="NMSE (dB)",
           title="Mechanism controls · 3GPP channels · QPSK · 15 dB\nHistorical fixed test set")
    if has_open_lower:
        ax.text(.01, .02, "Hollow ▽: lower CI extends to −∞ dB; clipped for display",
                transform=ax.transAxes, fontsize=7,
                bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    outputs += _save_figure(fig, folder / "fig_mechanism_doppler")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), layout="constrained", sharex=True)
    for ax, scope, title in zip(axes, ("all", "gpp_heldout"),
                                ("All 15 channels", "Held-out 3GPP")):
        curves = []
        for label, ids in (("DOP-Net", ("dopnet_25pct", "dopnet_50pct", "dopnet")),
                           ("Slow-time control", ("slow_time_25pct", "slow_time_50pct", "slow_time"))):
            triplets = [_db_ci(macro(rows, method, scope=scope)) for method in ids]
            curves.append((label, triplets))
        finite = [value for _, triplets in curves for triplet in triplets
                  for value in triplet if value is not None]
        floor = min(finite) - 2.0
        has_open_lower = False
        for label, triplets in curves:
            y = np.array([item[0] for item in triplets])
            upper = np.array([item[2] for item in triplets])
            open_lower = np.array([item[1] is None for item in triplets])
            lower = np.array([floor if item[1] is None else item[1]
                              for item in triplets])
            x = np.array((25, 50, 100))
            line, = ax.plot(x, y, "o-", label=label)
            ax.fill_between(x, lower, upper, alpha=.16, color=line.get_color())
            if open_lower.any():
                ax.scatter(x[open_lower], lower[open_lower], marker="v", s=55,
                           facecolors="none", edgecolors=line.get_color(), zorder=4)
                has_open_lower = True
        ax.set(title=title + " · historical fixed test set",
               xlabel="Unique training realizations (%); fixed optimizer steps",
               ylabel="NMSE (dB)")
        if has_open_lower:
            ax.text(.01, .02, "Hollow ▽: CI to −∞ dB; display clipped",
                    transform=ax.transAxes, fontsize=7,
                    bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        ax.set_xticks((25, 50, 100)); ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    outputs += _save_figure(fig, folder / "fig_data_efficiency")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.3), layout="constrained")
    for method in ROBUST:
        _draw_band(ax, rows, method, "gpp_heldout", "receiver", "ber_diagonal",
                   "16qam", 20, as_percent=True)
    ax.axvspan(800, 1600, alpha=.07, color="grey")
    ax.set(xlabel="Maximum Doppler (Hz)", ylabel="16-QAM BER (%)",
           title="Held-out 3GPP · common diagonal detector · 20 dB\nHistorical fixed test set")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    outputs += _save_figure(fig, folder / "fig_heldout_receiver")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.5), layout="constrained")
    for method in PROFILE_METHODS:
        profile = _one(profile_rows, method=method, seed=42, batch_size=1)
        score = macro(rows, method)
        x = numeric(profile, "latency_ms_mean")
        y = numeric(score, "mean_db")
        ax.scatter([x], [y], s=55, label=NAMES[method])
    ax.set(xlabel="Model-stage latency, B=1 (ms; seed-42 checkpoint)",
           ylabel="All-channel NMSE (dB; three-seed mean)",
           title="Accuracy and model-stage cost · recorded GPU, FP32 eager\nHistorical fixed test set")
    ax.grid(alpha=.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    outputs += _save_figure(fig, folder / "fig_accuracy_latency")
    plt.close(fig)
    return outputs


def _save_figure(fig, stem):
    paths = []
    for ext in ("pdf", "png"):
        path = stem.with_suffix("." + ext)
        fig.savefig(path, dpi=220)
        paths.append(path)
    return paths


def generate(out, reference_archive, report_dir=None):
    out = Path(out).expanduser().resolve()
    reference_archive = Path(reference_archive).expanduser().resolve()
    experiment, evaluation, stats, reference = validate_new_run(out, reference_archive)
    macro_rows = read_csv(out / "summary" / "macro_summary.csv")
    paired_rows = read_csv(out / "summary" / "macro_paired_differences.csv")
    profile = read_json(out / "profile.json")
    profile_rows = validate_profile(profile)
    operator = read_json(out / "operator_audit.json")
    if not operator.get("rows"):
        raise RuntimeError("Operator audit is missing")
    receiver_latency_path = out / "receiver_latency.json"
    receiver_latency = (validate_receiver_latency(read_json(receiver_latency_path),
                                                  experiment, evaluation)
                        if receiver_latency_path.exists() else None)
    tables = build_tables(macro_rows, paired_rows, profile_rows, operator["rows"],
                          receiver_latency)
    destination = Path(report_dir).expanduser().resolve() if report_dir else out / "submission_report"
    if destination == out or destination == reference_archive.parent:
        raise ValueError("Choose a dedicated report directory")
    (destination / "tables").mkdir(parents=True, exist_ok=True)
    for name, contents in tables.items():
        (destination / "tables" / name).write_text(contents, encoding="utf-8")
    figures = build_figures(macro_rows, profile_rows, destination / "figures")
    native_note = (
        "Native ADD-to-ICI timing is shown in a separate column because it uses a "
        "different equalizer. "
        if receiver_latency is not None
        and "native_own_ici" in receiver_latency["protocol"]["receiver_modes"] else
        "Native ADD-to-ICI timing was not measured. "
    )
    receiver_note = (
        "The optional receiver-stage benchmark times estimator forward plus common "
        "diagonal MMSE equalization over three checkpoint seeds, on one fixed synthetic "
        f"input per batch size. {native_note}Brackets are ranges across seed means, not confidence "
        "intervals. This still excludes waveform acquisition, pilot/ADD preprocessing, "
        "transfer, and demapping; it is not end-to-end latency.\n\n"
        if receiver_latency is not None else
        "Receiver-stage timing was not supplied; its table is omitted.\n\n"
    )
    provenance_note = (
        "This incremental run reused 27 archived checkpoints for the nine existing "
        f"configurations and trained {len(VARIANTS)*len(SEEDS)-27} checkpoints for {len(VARIANTS)-9} retained additions. Every reused "
        "checkpoint matched its archived SHA-256, configuration and model state; all "
        "120 shared training epochs matched the historical per-batch tensor hashes. "
        if experiment.get("training_mode") == REUSE_MODE else
        f"All {len(VARIANTS)*len(SEEDS)} checkpoints were trained in this new run. "
        "Before model updates, all 120 shared epochs matched every original legacy training-tensor hash. "
    )
    notes = (
        "# DOP-Net TWC-oriented result report\n\n"
        "All numerical result tables and plots were generated by this **completed** "
        "evaluation. The supplied DOPNet_Results.zip fixes the 636-cell test realization; "
        "no archived model score entered these outputs. "
        f"{provenance_note}\n\n"
        "The fixed historical test realization was previously inspected during model "
        "development. Reusing it gives paired comparability but **does not constitute a "
        "fresh blind holdout**. The `gpp_heldout` label means those channel profile types "
        "were excluded from training, not that the test realization was unseen. New claims "
        "should be confirmed on an independently generated test set before submission.\n\n"
        f"The {len(tables)} self-contained LaTeX tables in `tables/` require the `booktabs` package; "
        "four figures are supplied in PDF and PNG formats.\n\n"
        f"The {len(VARIANTS)*len(SEEDS)} learned routes are {len(VARIANTS)} configurations trained at seeds 42, 43, and 44. "
        "The five fixed routes are classical or perfect-diagonal-CSI references. "
        "The data-efficiency variants keep 12,000 optimizer input frames per epoch and "
        "repeat a deterministic prefix of 3,000 or 6,000 unique generated frames, so "
        "the comparison varies unique realization diversity at fixed optimizer steps. "
        "NMSE is averaged in the linear domain before conversion to dB. Macro results "
        "weight cells equally within each channel and channels equally within each scope.\n\n"
        "Intervals in tables and figures are pointwise 95% Student-t intervals across three "
        "training seeds (df=2), conditional on this shared fixed test set. The raw aggregation "
        "also stores test-cluster bootstrap intervals conditional on the trained checkpoints. "
        "These are different uncertainty sources, and neither supports a population-wide "
        "or multiplicity-adjusted significance claim. If a linear NMSE t-interval has a "
        "nonpositive lower bound, its dB lower bound is negative infinity. Tables show "
        "`(-∞, upper]`; figures use hollow downward triangles at a labeled visual "
        "clipping floor. That floor is not a numerical confidence bound.\n\n"
        "BER comparisons use the common diagonal detector as the controlled primary outcome. "
        "Native receiver BER may mix different ICI reconstruction capabilities. Latency is "
        "model-stage only on the GPU recorded in profile.json; it is not complete receiver "
        "latency and does not establish real-time deployment.\n\n"
        f"{receiver_note}"
        f"New experiment signature: `{experiment['signature']}`.  "
        f"New evaluation signature: `{evaluation['signature']}`.  "
        f"Reference ZIP SHA-256: `{REFERENCE_SHA256}`.  "
        f"Reference waveform fingerprint: `{reference.fingerprint}`.\n"
    )
    (destination / "README.md").write_text(notes, encoding="utf-8")
    source_files = [out / name for name in ("experiment.json", "evaluation.json",
                   "evaluation_complete.json", "summary/statistics.json",
                   "summary/macro_summary.csv", "summary/macro_paired_differences.csv",
                   "profile.json", "operator_audit.json")]
    if receiver_latency is not None:
        source_files.append(receiver_latency_path)
    output_files = sorted((destination / "tables").glob("*.tex")) + figures + [destination / "README.md"]
    report = {
        "purpose": "TWC-oriented descriptive analysis of a completed new experiment",
        "new_experiment_signature": experiment["signature"],
        "new_evaluation_signature": evaluation["signature"],
        "reference_archive_sha256": REFERENCE_SHA256,
        "reference_realizations_sha256": reference.fingerprint,
        "training_seeds": list(SEEDS),
        "test_cells": EXPECTED_CELLS,
        "frames_per_cell": EXPECTED_SAMPLES,
        "test_generation_clusters_per_cell": EXPECTED_SAMPLES // EXPECTED_BATCH,
        "bootstrap_repetitions": stats["bootstrap"],
        "latency_gpu": profile["gpu"],
        "receiver_stage_latency_included": receiver_latency is not None,
        "receiver_stage_gpu": (receiver_latency["hardware"]["gpu"]
                               if receiver_latency is not None else None),
        "analysis_script_sha256": file_hash(Path(__file__).resolve()),
        "inputs_sha256": {str(path.relative_to(out)).replace("\\", "/"): file_hash(path)
                          for path in source_files},
        "outputs_sha256": {str(path.relative_to(destination)).replace("\\", "/"): file_hash(path)
                           for path in output_files},
        "limits": ["Training-seed confidence intervals conditional on fixed test set",
                   "Test-cluster bootstrap confidence intervals conditional on checkpoints",
                   "Pointwise intervals only; no multiple-comparison adjustment",
                   "Historical fixed test set previously inspected; not a fresh blind holdout",
                   "Nonpositive linear NMSE CI lower bound maps to -infinity dB; figure clipping is display-only",
                   "Model-stage-only latency; not full receiver latency"],
    }
    (destination / "report_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report verified and written to {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Completed *new* experiment output directory")
    parser.add_argument("--reference-results", type=Path, required=True,
                        help="Supplied immutable DOPNet_Results.zip")
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()
    generate(args.output, args.reference_results, args.report_dir)


if __name__ == "__main__":
    main()
