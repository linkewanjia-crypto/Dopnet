"""Read-only directory/ZIP check: metadata, score arrays, SHA and CRC only.

No Torch import, simulator, model, training or inference is used. Tensor objects
inside checkpoints are replaced by inert placeholders by the existing reader.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tempfile
from zipfile import ZipFile

from .prepare_wimae import (HERE, UPSTREAM_COMMIT, VALIDATION_SHA256, digest,
                            RECIPE, build_config, file_hash, signed_metadata)
from .wimae_protocol import (CLAIM, FORTI_METHOD, LABELS, NEW_METHOD, SEEDS, check_evaluation_metadata,
                             check_training, core_routes, read_json, require)

SUMMARY_NAMES = {"cell_summary.csv", "cell_paired_differences.csv", "macro_summary.csv",
                 "macro_paired_differences.csv", "receiver_diagnostics.csv"}
ASSET_NAMES = ("equal_budget_dl_comparison.tex", "equal_budget_dl_comparison.pdf", "equal_budget_dl_comparison.png")


def check_delivery_patch(source_root):
    """Bind this explicit delivery patch to the untouched training/scorer bytes."""
    source_root = Path(source_root)
    patch = signed_metadata(source_root / "delivery_patch.json")
    parent = signed_metadata(source_root / "PARENT_SOURCE_RELEASE_MANIFEST.json")
    require(patch.get("version") == "1.0.1" and patch.get("changes") == ["verify_wimae.py", "report_wimae.py"]
            and parent["signature"] == patch["parent_source_signature"] ==
            "8152135ee98377d6aada3bf8d90a8a61d98250c1c70bf9f2e23836b66ccd6bca",
            "Unknown delivery-only repair provenance")
    require(file_hash(source_root / "parent_verify_wimae.txt") ==
            parent["files"]["dl_baseline_expansion/verify_wimae.py"]["sha256"], "Parent checker bytes differ")
    for name, expected in patch["bound_delivery_source_sha256"].items():
        require(safe_name(name) and file_hash(source_root / name) == expected,
                "Delivery patch source changed: " + name)
        if name not in patch["changes"]:
            require(expected == parent["files"]["dl_baseline_expansion/" + name]["sha256"],
                    "A frozen science source changed in a delivery patch")
    return patch


def safe_name(name):
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name and ":" not in name and path.as_posix() == name


def check_seed_intervals(path, t_critical=4.302652729749464):
    """Scalar arithmetic; preserves the original linear-NMSE aggregation."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if "mean" not in row or not row.get("mean"):
            continue
        require(row.get("n_training_seeds") == "3", "Each neural method needs three training seeds")
        values = [float(row["seed" + str(s)]) for s in SEEDS]
        require(all(math.isfinite(v) and v >= 0 for v in values), "Nonfinite seed point estimate")
        mean = sum(values) / 3
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / 2)
        half = t_critical * sd / math.sqrt(3)
        for key, expected in (("mean", mean), ("seed_sd", sd),
                              ("seed_ci95_lo_unbounded", mean - half), ("seed_ci95_hi_unbounded", mean + half)):
            require(math.isclose(float(row[key]), expected, rel_tol=1e-10, abs_tol=1e-14), "Three-seed interval differs: " + key)
        if row["metric"] == "nmse" and mean > 0:
            require(math.isclose(float(row["mean_db"]), 10 * math.log10(mean), rel_tol=1e-10, abs_tol=1e-12),
                    "NMSE dB was averaged before linear NMSE")
    require(rows, "Empty result summary")


def check_paired_intervals(summary_path, paired_path, macro=False, t_critical=4.302652729749464):
    with Path(summary_path).open(encoding="utf-8-sig", newline="") as stream:
        summaries = list(csv.DictReader(stream))
    with Path(paired_path).open(encoding="utf-8-sig", newline="") as stream:
        differences = list(csv.DictReader(stream))
    context = ("study", "scope", "modulation", "doppler_hz", "snr_db", "metric") if macro else ("cell_id", "metric")
    index = {tuple(row[k] for k in context) + (row["method"],): row for row in summaries}
    require(len(index) == len(summaries), "Duplicate summary groups")
    groups = {}
    for identity in index:
        groups.setdefault(identity[:-1], set()).add(identity[-1])
    require(groups and all(methods == set(LABELS) for methods in groups.values()),
            "Summary group lacks exactly the seven expected methods")
    expected_pairs = {identity for identity in index if identity[-1] != "dopnet"}
    require(len(differences) == len(summaries) * 6 // 7, "Paired summary lacks a method/group")
    seen = set()
    for row in differences:
        key = tuple(row[k] for k in context)
        identity = key + (row["comparator"],)
        require(identity in expected_pairs and identity not in seen and row["reference"] == "dopnet",
                "Duplicate/missing/self paired comparison")
        seen.add(identity)
        comparator, reference = index[identity], index[key + ("dopnet",)]
        deltas = [float(comparator["seed" + str(s)]) - float(reference["seed" + str(s)]) for s in SEEDS]
        def check_interval(values, mean_key, lo_key, hi_key):
            mean = sum(values) / 3
            sd = math.sqrt(sum((v - mean) ** 2 for v in values) / 2)
            half = t_critical * sd / math.sqrt(3)
            for field, expected in ((mean_key, mean), (lo_key, mean - half), (hi_key, mean + half)):
                require(math.isclose(float(row[field]), expected, rel_tol=1e-10, abs_tol=1e-14), "Paired seed interval differs")
        check_interval(deltas, "paired_difference", "paired_seed_ci95_lo", "paired_seed_ci95_hi")
        lo, hi = float(row["paired_test_cluster_ci95_lo"]), float(row["paired_test_cluster_ci95_hi"])
        require(math.isfinite(lo) and math.isfinite(hi) and lo <= hi, "Invalid paired cluster-bootstrap interval")
        if row["metric"] == "nmse" and all(float(x["seed" + str(s)]) > 0 for x in (comparator, reference) for s in SEEDS):
            gaps = [10 * math.log10(float(comparator["seed" + str(s)]) / float(reference["seed" + str(s)])) for s in SEEDS]
            check_interval(gaps, "mean_paired_gap_db", "paired_gap_db_seed_ci95_lo", "paired_gap_db_seed_ci95_hi")
    require(seen == expected_pairs, "Paired comparisons lack an expected comparator/group")


def check_assets(output, ev, statistics):
    ready = signed_metadata(output / "PAPER_READY.json")
    require(ready.get("status") == "complete" and ready.get("paper_ready") is True and ready.get("pending") == []
            and ready.get("evaluation_signature") == ev["signature"]
            and ready.get("statistics_signature") == statistics["signature"]
            and set(ready["assets_sha256"]) == set(ASSET_NAMES), "Paper asset completion binding differs")
    for name in ASSET_NAMES:
        path = output / "paper_assets" / name
        require(path.is_file() and path.stat().st_size > 100 and file_hash(path) == ready["assets_sha256"][name],
                "Paper asset absent/empty/changed: " + name)
        head = path.read_bytes()[:8]
        if name.endswith(".pdf"):
            require(head.startswith(b"%PDF-"), "Invalid PDF asset")
        elif name.endswith(".png"):
            require(head == b"\x89PNG\r\n\x1a\n", "Invalid PNG asset")
        else:
            require(b"\\begin{table" in path.read_bytes(), "Invalid TeX table asset")


def verify_directory(output, training_output, reference_results, forti_evaluation, forti_code_root, source_root=None, require_assets=True):
    import numpy as np
    import yaml
    from budget_matched_baselines.verify_author_delivery import checkpoint_metadata
    from .evaluate_wimae import check_forti_reference
    output, training, reference = map(lambda p: Path(p).resolve(), (output, training_output, reference_results))
    source_root = Path(source_root or HERE).resolve()
    patch = check_delivery_patch(source_root)
    ev, statistics = check_evaluation_metadata(output)
    done = read_json(output / "COMPLETE.json")
    require(done.get("paper_ready") is False and done.get("pending") == ["paper_assets"],
            "Scoring completion and final paper readiness must be distinct")
    recipe_path = training / "wimae_manifest.json"
    recipe = signed_metadata(recipe_path)
    experiment = signed_metadata(reference / "experiment.json")
    old_ev = signed_metadata(reference / "evaluation.json")
    require(ev["training_recipe_sha256"] == file_hash(recipe_path) and ev["training_recipe_signature"] == recipe["signature"]
            and ev["reference_experiment_signature"] == experiment["signature"] == recipe["reference_experiment_signature"]
            and ev["reference_evaluation_signature"] == old_ev["signature"] == recipe["reference_evaluation_signature"]
            and old_ev["experiment_signature"] == experiment["signature"], "WiMAE train/test/reference binding differs")
    require(recipe["validation_cache_sha256"] == VALIDATION_SHA256 and recipe["training_frame_exposures_per_seed"] == 480000
            and recipe["recipe"] == RECIPE and recipe["claim"] == CLAIM and recipe["seeds"] == list(SEEDS)
            and recipe["reference_runtime_code_sha256"] == experiment["code_sha256"]
            == digest(recipe["source_files_sha256"]), "WiMAE budget/validation/runtime plan differs")
    for name, expected in recipe["implementation_sha256"].items():
        require(safe_name(name) and file_hash(source_root / name) == expected, "WiMAE implementation source differs: " + name)
    for name, expected in ev["scoring_sources_sha256"].items():
        require(safe_name(name) and file_hash(source_root / name) == expected, "Scoring source differs: " + name)
    upstream = read_json(source_root / "vendor" / "wimae_upstream" / "UPSTREAM.json")
    require(upstream["commit"] == UPSTREAM_COMMIT and upstream["license"] == "MIT", "WiMAE upstream differs")
    for name, expected in upstream["files"].items():
        path = source_root / "vendor" / "wimae_upstream" / name
        require(safe_name(name) and file_hash(path) == expected["sha256"] and path.stat().st_size == expected["bytes"],
                "Upstream source/license bytes differ")
    require(len(old_ev["cells"]) == 636 and experiment["samples_per_cell"] == 512 and experiment["batch_size"] == 8,
            "Historical test contract differs")
    old_indices, old_routes = core_routes(old_ev["routes"])
    forti_ev, forti_indices, _, forti_archive_sha, forti_hashes = check_forti_reference(
        forti_evaluation, reference, experiment, old_ev, ev["cells"], forti_code_root)
    require(ev["forti_archive_sha256"] == forti_archive_sha and ev["forti_evaluation_signature"] == forti_ev["signature"]
            and ev["routes"][:15] == old_routes and ev["routes"][15:18] == [forti_ev["routes"][i] for i in forti_indices],
            "Reused Forti or historical route identity differs")
    for seed in SEEDS:
        name = f"wimae_author_seed{seed}"
        cfg_record = recipe["configs"][name]
        cfg_path = training / cfg_record["file"]
        require(file_hash(cfg_path) == cfg_record["sha256"], "Prepared WiMAE config bytes differ")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        require(digest(cfg) == cfg_record["config_sha256"] and cfg["seed"] == seed, "WiMAE config identity differs")
        spec = cfg["wimae_reproduction"]
        require(cfg == build_config(experiment["configs"][f"wimae_ce_seed{seed}"], seed,
                                     spec["gradient_microbatch_frames"], spec["effective_batch_frames"], spec["amp"]),
                "Prepared config differs from the reviewed20+20 adaptation")
        reader = lambda path: checkpoint_metadata(path.read_bytes(), path.name)
        _, hashes = check_training(training / "checkpoints" / name, recipe, cfg, reader)
        require(hashes == ev["training_artifacts_sha256"][str(seed)]
                and hashes["model.pt"] == ev["checkpoint_sha256"][str(seed)] == done["checkpoint_sha256"][str(seed)],
                "Training evidence changed after scoring")
    fingerprints = {}
    for cell in old_ev["cells"]:
        path = reference / "cells" / (cell["cell_id"] + ".npz")
        with np.load(path, allow_pickle=False) as z:
            require(z["signature"].item() == old_ev["signature"] and json.loads(z["meta"].item()) == cell
                    and z["nmse"].shape == (len(old_ev["routes"]), 512)
                    and np.array_equal(z["cluster_id"], np.arange(512) // 8), "Historical636 score identity differs")
            fingerprints[cell["cell_id"]] = z["realization_sha256"].item()
    fixed = ev["fixed_test_reference"]
    require(fixed["reference_realizations_sha256"] == digest(dict(evaluation_signature=old_ev["signature"],
            experiment_signature=experiment["signature"], waveform_keys=tuple(fixed["reference_waveform_keys"]),
            realization_sha256=fingerprints)), "Full historical waveform digest differs")
    expected_keys = ("input", "add_input", "add_input_slow", "rx_data", "target_h", "tx_label")
    require(tuple(fixed["reference_waveform_keys"]) == expected_keys and fixed["reference_samples_per_cell"] == 512
            and fixed["reference_batch_size"] == 8 and fixed["reference_cells"] == 636, "Waveform hash tensor contract differs")
    expected_diagnostics = {}
    for cell in ev["cells"]:
        cid = cell["cell_id"]
        path, old_path, forti_path = (output / "cells" / (cid + ".npz"), reference / "cells" / (cid + ".npz"),
                                     Path(forti_evaluation) / "cells" / (cid + ".npz"))
        marker = read_json(path.with_suffix(".COMPLETE.json"))
        old_sha, forti_sha = file_hash(old_path), file_hash(forti_path)
        require(marker["source_score_sha256"] == old_sha and marker["forti_score_sha256"] == forti_sha
                == forti_hashes["evaluation/cells/" + cid + ".npz"]
                and marker["realization_sha256"] == fingerprints[cid], "New cell source/frozen waveform binding differs")
        with np.load(path, allow_pickle=False) as z, np.load(old_path, allow_pickle=False) as old, np.load(forti_path, allow_pickle=False) as forti:
            require(z["signature"].item() == ev["signature"] and json.loads(z["meta"].item()) == cell
                    and z["source_score_sha256"].item() == old_sha and z["forti_score_sha256"].item() == forti_sha
                    and z["realization_sha256"].item() == fingerprints[cid]
                    and np.array_equal(z["cluster_id"], np.arange(512) // 8), "New cell metadata differs")
            require(z["nmse"].shape == (21, 512) and np.isfinite(z["nmse"]).all() and np.all(z["nmse"] >= 0)
                    and np.array_equal(z["nmse"][:15], old["nmse"][old_indices])
                    and np.array_equal(z["nmse"][15:18], forti["nmse"][forti_indices]), "NMSE reused columns differ")
            require(z["native_available"].shape == (21,) and np.array_equal(z["native_available"][:15], old["native_available"][old_indices])
                    and np.array_equal(z["native_available"][15:18], forti["native_available"][forti_indices])
                    and np.all(z["native_available"][18:] == 0), "Native receiver availability differs")
            if cell["ber"]:
                bits = z["bits"]
                require(bits.shape == (21, 512) and np.all(bits > 0) and np.all(bits == old["bits"][old_indices[0]]), "BER denominators differ")
                for key in ("errors_diagonal", "errors_native", "bits", "native_residual_max", "native_direct_fallback",
                            "errors_noise_minus3db", "errors_noise_plus3db"):
                    if key not in old:
                        require(key not in z, "New receiver field not in reference")
                        continue
                    require(z[key].shape == (21, 512) and np.isfinite(z[key]).all()
                            and np.array_equal(z[key][:15], old[key][old_indices])
                            and np.array_equal(z[key][15:18], forti[key][forti_indices]), "Receiver reused columns differ: " + key)
                    if key.startswith("errors_"):
                        require(np.issubdtype(z[key].dtype, np.integer) and np.all(z[key] >= 0) and np.all(z[key] <= bits), "Invalid BER numerator")
                require(np.array_equal(z["errors_native"][18:], z["errors_diagonal"][18:])
                        and np.all(z["native_residual_max"][18:] == 0) and np.all(z["native_direct_fallback"][18:] == 0),
                        "WiMAE diagonal fallback diagnostics differ")
                for route_index, route in enumerate(ev["routes"]):
                    expected_diagnostics[(cid, route["method"], str(route["training_seed"]))] = {
                        "native_ici_available": int(z["native_available"][route_index]),
                        "residual_max": float(z["native_residual_max"][route_index].max()),
                        "direct_fallback_frames": int(z["native_direct_fallback"][route_index].sum()),
                        "bit_errors_native": int(z["errors_native"][route_index].sum()),
                        "bit_total": int(z["bits"][route_index].sum())}
            else:
                require("bits" not in z and not any(k.startswith("errors_") for k in z.files), "NMSE-only cell received fictitious bits")
    require(statistics["bootstrap"] >= 1000 and math.isclose(statistics["t_critical_df2"], math.sqrt(2 * .95**2 / (1 - .95**2)), rel_tol=1e-10),
            "Statistical interval protocol differs")
    require(set(statistics["summary_csv_sha256"]) == SUMMARY_NAMES, "Summary lacks paired or receiver diagnostics CSV")
    for name in ("cell_summary.csv", "macro_summary.csv"):
        check_seed_intervals(output / "summary" / name, statistics["t_critical_df2"])
    check_paired_intervals(output / "summary/cell_summary.csv", output / "summary/cell_paired_differences.csv", t_critical=statistics["t_critical_df2"])
    check_paired_intervals(output / "summary/macro_summary.csv", output / "summary/macro_paired_differences.csv", True, statistics["t_critical_df2"])
    with (output / "summary/receiver_diagnostics.csv").open(encoding="utf-8-sig", newline="") as stream:
        diagnostic_rows = list(csv.DictReader(stream))
    require(len(diagnostic_rows) == len(expected_diagnostics), "Receiver diagnostic matrix is incomplete")
    seen = set()
    for row in diagnostic_rows:
        key = (row["cell_id"], row["method"], row["training_seed"])
        require(key in expected_diagnostics and key not in seen, "Unknown/duplicate receiver diagnostic row")
        seen.add(key)
        for field, value in expected_diagnostics[key].items():
            require(math.isclose(float(row[field]), value, rel_tol=1e-12, abs_tol=1e-14), "Receiver diagnostic differs from score data")
    if require_assets:
        check_assets(output, ev, statistics)
    return {"status": "passed", "paper_ready": bool(require_assets), "pending": [] if require_assets else ["paper_assets"], "cells_verified": 269,
            "methods": 7, "routes": 21, "seeds": list(SEEDS), "training_epochs_per_seed": 40,
            "validation_cache_sha256": VALIDATION_SHA256, "evaluation_signature": ev["signature"],
            "training_recipe_signature": recipe["signature"], "forti_archive_sha256": forti_archive_sha,
            "real_data_generated": False, "model_execution": False,
            "delivery_patch_signature": patch["signature"], "delivery_patch_version": "1.0.1"}


def verify_archive(archive, reference_results, forti_code_root, implementation_root=None):
    source_root = Path(implementation_root or HERE)
    with ZipFile(archive) as z, tempfile.TemporaryDirectory(prefix="dopnet_wimae_metadata_") as temporary:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        require(len(names) == len(set(names)) and all(safe_name(n) for n in names), "Unsafe/duplicate ZIP paths")
        require(z.testzip() is None, "ZIP CRC failed")
        manifest = json.loads(z.read("DELIVERY_MANIFEST.json"))
        require(manifest.get("signature") == digest({k: v for k, v in manifest.items() if k != "signature"})
                and manifest.get("status") == "complete" and manifest.get("paper_ready") is True
                and manifest.get("pending") == [], "Incomplete or unsigned delivery manifest")
        require(set(names) == set(manifest["files"]) | {"DELIVERY_MANIFEST.json"}, "ZIP membership differs from manifest")
        root = Path(temporary)
        for name, expected in manifest["files"].items():
            data = z.read(name)
            require(hashlib.sha256(data).hexdigest() == expected, "ZIP internal file SHA differs: " + name)
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        require(set(("source/wimae/delivery_patch.json", "source/wimae/parent_verify_wimae.txt",
                     "source/wimae/PARENT_SOURCE_RELEASE_MANIFEST.json")).issubset(manifest["files"]),
                "Delivery ZIP lacks explicit patch/parent provenance")
        patch = check_delivery_patch(root / "source" / "wimae")
        require(patch == check_delivery_patch(source_root), "Bundled and reviewed patch identities differ")
        recipe = signed_metadata(root / "training" / "wimae_manifest.json")
        for name, expected in recipe["implementation_sha256"].items():
            require(file_hash(root / "source" / "wimae" / name) == expected
                    and file_hash(source_root / name) == expected, "Local/archived WiMAE implementation differs")
        for name in ("wimae_protocol.py", "evaluate_wimae.py", "report_wimae.py", "verify_wimae.py", "queue_wimae.py"):
            require(file_hash(root / "source" / "wimae" / name) == file_hash(source_root / name),
                    "Local/archived reviewed scoring/delivery implementation differs: " + name)
        for name, expected in recipe["source_files_sha256"].items():
            require(safe_name(name) and file_hash(root / "source" / "archive_runtime" / name) == expected,
                    "Archived original generator/runtime source differs")
        forti_archive = root / "evidence" / "fortitran_budget_matched_results.zip"
        from budget_matched_baselines.verify_author_delivery import verify as verify_forti
        verify_forti(forti_archive, reference_results, forti_code_root)
        forti_root = root / "forti_directory"
        with ZipFile(forti_archive) as fz:
            for item in fz.infolist():
                if item.filename.startswith("evaluation/") and not item.is_dir():
                    target = forti_root / item.filename[len("evaluation/"):]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(fz.read(item))
        forti_root.mkdir(parents=True, exist_ok=True)
        (forti_root / "budget_matched_results.zip").write_bytes(forti_archive.read_bytes())
        proof = verify_directory(root / "evaluation", root / "training", reference_results, forti_root,
                                 forti_code_root, root / "source" / "wimae")
        require(proof["evaluation_signature"] == manifest["evaluation_signature"]
                and proof["training_recipe_signature"] == manifest["training_recipe_signature"], "Delivery identity differs")
    proof["zip_crc"] = "passed"
    proof["archive_sha256"] = file_hash(Path(archive))
    return proof


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--archive", type=Path)
    mode.add_argument("--directory", type=Path)
    ap.add_argument("--training-output", type=Path)
    ap.add_argument("--forti-evaluation", type=Path)
    ap.add_argument("--reference-results", type=Path, required=True)
    ap.add_argument("--forti-code-root", type=Path, required=True)
    ap.add_argument("--implementation-root", type=Path)
    args = ap.parse_args()
    if args.archive:
        proof = verify_archive(args.archive, args.reference_results, args.forti_code_root, args.implementation_root)
    else:
        require(args.training_output and args.forti_evaluation, "Directory check needs training-output and forti-evaluation")
        proof = verify_directory(args.directory, args.training_output, args.reference_results,
                                 args.forti_evaluation, args.forti_code_root, args.implementation_root)
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
