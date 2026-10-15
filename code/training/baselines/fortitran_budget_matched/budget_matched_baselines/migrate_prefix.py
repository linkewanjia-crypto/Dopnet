"""Server-only audited continuation of a committed FortiTran training prefix.

The parent run is never changed. Model/optimizer/scheduler/RNG/history are
retained exactly; only the independent recipe identity and fixed-budget
termination metadata change. Metadata helpers can be tested without Torch.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

if __package__ in (None, ""):
    _package_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _package_dir]
    sys.path.insert(0, str(_package_dir.parent))

from budget_matched_baselines.recipe import (
    CLAIM, MAX_EPOCHS, RECIPE, SEEDS, TERMINATION, TRACKER_PATIENCE,
    VALIDATION_SHA256, digest, file_hash, runtime_sources, signed_metadata,
    validate_manifest, write_json,
)

PARENT_RECIPE = "fortitran_task_adapted_author_recipe_v1"
PARENT_CLAIM = "task_adapted_author_recipe"
PARENT_IMPLEMENTATION = ("recipe.py", "prepare.py", "train_author.py")
PRESERVED_STATE = ("model", "optimizer", "scheduler", "rng_python", "rng_numpy",
                   "rng_torch", "rng_cuda", "history")


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def assert_config_transition(parent, child):
    """Allow only declared recipe naming/claim/termination metadata changes."""
    expected = copy.deepcopy(parent)
    _require(parent["train"]["epochs"] == 1000
             and parent["train"]["early_stopping_patience"] == 3,
             "Parent is not the original max1000/patience3 recipe")
    _require(parent["author_recipe"]["identifier"] == PARENT_RECIPE
             and parent["author_recipe"]["claim"] == PARENT_CLAIM,
             "Parent recipe identity differs")
    expected["train"].update(epochs=MAX_EPOCHS, early_stopping_patience=TRACKER_PATIENCE)
    expected["author_recipe"].update(
        identifier=RECIPE, claim=CLAIM,
        early_stopping_patience=TRACKER_PATIENCE,
        early_stopping_enabled=False, termination=TERMINATION, max_epochs=MAX_EPOCHS,
        original_max_epochs=1000, original_early_stopping=True,
        original_repository_patience=3,
        patience_provenance="best/stale bookkeeping only; never terminates this fixed-40-epoch run",
        required_engine="budget_matched_baselines/train_author.py",
        warning="source-guided Adam/MSE/effective512/gamma.995; original1000/early-stop3 replaced by fixed40 to match DOP-Net",
    )
    _require(child == expected,
             "Migration changed configuration outside epochs/termination/claim/engine naming metadata")


def _same_history(stored, rows, label):
    _require(len(stored) == len(rows), f"{label} history length differs")
    for obj, row in zip(stored, rows):
        _require(set(obj) == set(row), f"{label} history columns differ")
        for key, value in obj.items():
            _require(value == row[key] if isinstance(value, str) else float(value) == float(row[key]),
                     f"{label} history differs at {key}")


def check_committed_prefix(rows, selected, last):
    """Authenticate whole-epoch history and strict validation-MSE selection."""
    epoch = last["epoch"]
    _require(type(epoch) is int and 1 <= epoch <= MAX_EPOCHS and len(rows) == epoch,
             "Prefix must have 1..40 committed, consecutive epochs")
    best, stale, best_epoch = None, 0, None
    minimum_nmse = None
    for index, row in enumerate(rows, 1):
        _require(int(row["epoch"]) == index and row["selection_metric"] == "val_mse"
                 and int(row["train_frames"]) == 12000 and int(row["optimizer_steps"]) == 24,
                 "Prefix contains an incomplete epoch or another optimizer budget")
        values = {key: float(row[key]) for key in
                  ("lr", "next_lr", "val_loss", "train_loss", "val_nmse_db")}
        _require(all(math.isfinite(value) for value in values.values())
                 and values["val_loss"] >= 0 and values["train_loss"] >= 0,
                 "Nonfinite/invalid prefix metrics")
        _require(math.isclose(values["lr"], .001 * .995 ** (index - 1), rel_tol=1e-10, abs_tol=1e-15)
                 and math.isclose(values["next_lr"], values["lr"] * .995, rel_tol=1e-10, abs_tol=1e-15),
                 "Parent learning-rate trajectory differs")
        if best is None or values["val_loss"] < best:
            best, stale, best_epoch = values["val_loss"], 0, index
        else:
            stale += 1
        _require(int(row["stale_epochs"]) == stale
                 and (stale < 3 or index == epoch),
                 "Parent strict-improvement/early-stop history differs")
        minimum_nmse = values["val_nmse_db"] if minimum_nmse is None else min(minimum_nmse, values["val_nmse_db"])
    _require(type(selected["epoch"]) is int and selected["epoch"] == best_epoch
             and selected["epoch"] <= epoch
             and selected["best_val_loss"] == last["best_val_loss"] == best,
             "Selected checkpoint is not the strict minimum validation-MSE prefix checkpoint")
    _require(last["early_stopping"] == {"patience": 3, "best": best, "stale": stale}
             and last["minimum_observed_val_nmse_db"] == minimum_nmse,
             "Last prefix tracker or validation diagnostics differ")
    _require(last.get("training_complete") is (stale >= 3),
             "Parent checkpoint completion is inconsistent with its original stopping rule")
    _require(selected["early_stopping"] == {"patience": 3, "best": best, "stale": 0}
             and selected.get("training_complete") is False,
             "Selected parent checkpoint tracker/completion differs")
    _require(last["val"]["loss"] == float(rows[-1]["val_loss"])
             and selected["val"]["loss"] == float(rows[best_epoch - 1]["val_loss"]),
             "Checkpoint validation value disagrees with committed history")
    _same_history(last["history"], rows, "last")
    _same_history(selected["history"], rows[:best_epoch], "selected")
    return {"completed_epoch": epoch, "selected_epoch": best_epoch,
            "best_val_mse": best, "stale_epochs": stale,
            "parent_naturally_terminal": stale >= 3}


def _validate_checkpoint(state, cfg, parent, sidecar, path):
    _require(sidecar["sha256"] == file_hash(path) and sidecar["bytes"] == path.stat().st_size
             and sidecar["recipe_signature"] == parent["signature"]
             and sidecar["epoch"] == state["epoch"], "Parent checkpoint sidecar SHA/epoch differs")
    _require(state["cfg"] == cfg and state["recipe_signature"] == parent["signature"]
             and state["recipe_claim"] == PARENT_CLAIM
             and state["selection_metric"] == "val_mse"
             and state["validation_cache_sha256"] == VALIDATION_SHA256
             and state["source_files_sha256"] == parent["source_files_sha256"]
             and state["implementation_sha256"] == parent["implementation_sha256"]
             and state["optimizer_name"] == "adam" and state["scheduler_name"] == "exponential"
             and state["effective_batch_frames"] == 512
             and state["physical_gradient_microbatch_frames"] == cfg["train"]["gradient_microbatch_size"],
             "Parent checkpoint configuration/source/optimizer contract differs")
    _require(all(key in state for key in PRESERVED_STATE)
             and isinstance(state["model"], dict) and bool(state["model"])
             and isinstance(state["optimizer"], dict) and isinstance(state["scheduler"], dict),
             "Parent checkpoint lacks resumable model/optimizer/scheduler/RNG/history")
    _require(state["scheduler"].get("gamma") == .995
             and state["scheduler"].get("last_epoch") == state["epoch"],
             "Parent scheduler state differs from its committed epoch")
    groups = state["optimizer"].get("param_groups", [])
    _require(bool(groups) and all(math.isclose(group["lr"], .001 * .995 ** state["epoch"],
                rel_tol=1e-10, abs_tol=1e-15) for group in groups), "Parent optimizer next learning rate differs")


def _assert_equal(torch, parent, child, label):
    """Exact loaded-state equality, including tensor buffers and NumPy RNG."""
    import numpy as np
    if isinstance(parent, torch.Tensor):
        _require(isinstance(child, torch.Tensor) and parent.dtype == child.dtype
                 and parent.shape == child.shape and torch.equal(parent, child), f"Preserved tensor differs: {label}")
    elif isinstance(parent, np.ndarray):
        _require(isinstance(child, np.ndarray) and parent.dtype == child.dtype
                 and parent.shape == child.shape and np.array_equal(parent, child), f"Preserved ndarray differs: {label}")
    elif isinstance(parent, dict):
        _require(isinstance(child, dict) and set(parent) == set(child), f"Preserved mapping differs: {label}")
        for key in parent:
            _assert_equal(torch, parent[key], child[key], f"{label}.{key}")
    elif isinstance(parent, (list, tuple)):
        _require(type(parent) is type(child) and len(parent) == len(child), f"Preserved sequence differs: {label}")
        for index, (before, after) in enumerate(zip(parent, child)):
            _assert_equal(torch, before, after, f"{label}[{index}]")
    else:
        _require(type(parent) is type(child) and parent == child, f"Preserved scalar differs: {label}")


def _copy_bound(source, destination, expected):
    _require(file_hash(source) == expected, f"Parent source changed before snapshot: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    _require(file_hash(source) == expected == file_hash(destination),
             f"Parent source changed while snapshotting: {source}")


def migrate(project, parent_manifest_path, child_manifest_path, *, seeds=(42,), parent_implementation=None):
    # Must precede every Torch/runtime import. Local metadata tests call only
    # the pure helpers above, never this server migration entry point.
    _require(os.name == "posix", "Checkpoint migration is server-only; do not run it on the local laptop")
    project = Path(project).resolve()
    parent_manifest_path, child_manifest_path = Path(parent_manifest_path).resolve(), Path(child_manifest_path).resolve()
    parent_root, output = parent_manifest_path.parent, child_manifest_path.parent
    parent_manifest_sha = file_hash(parent_manifest_path)
    child_manifest_sha = file_hash(child_manifest_path)
    _require(output != parent_root and parent_root not in output.parents,
             "New output must be independent of the parent run")
    _require(bool(seeds) and len(set(seeds)) == len(seeds) and set(seeds) <= set(SEEDS), "Unsupported/duplicate migration seeds")
    parent = signed_metadata(parent_manifest_path)
    _require(parent.get("recipe") == PARENT_RECIPE and parent.get("claim") == PARENT_CLAIM
             and parent.get("seeds") == list(SEEDS), "Parent manifest is not the source-guided original recipe")
    sources = runtime_sources(project)
    _require(sources == parent["source_files_sha256"]
             and digest(sources) == parent["reference_runtime_code_sha256"], "Parent archived runtime no longer matches its original signature")
    parent_implementation = Path(parent_implementation or Path(__file__).resolve().parents[1] / "author_baseline_reproduction").resolve()
    engines = {name: file_hash(parent_implementation / name) for name in PARENT_IMPLEMENTATION}
    _require(engines == parent["implementation_sha256"], "Parent recipe engine source differs from its original manifest")
    # Use the inert metadata reader before any tensor deserialization.
    from budget_matched_baselines.verify_author_delivery import checkpoint_metadata
    import yaml
    plans = []
    for seed in seeds:
        child, cfg, name = validate_manifest(child_manifest_path, project, seed)
        _require(child["reference_experiment_signature"] == parent["reference_experiment_signature"]
                 and child["reference_evaluation_signature"] == parent["reference_evaluation_signature"]
                 and child["source_files_sha256"] == sources
                 and child["validation_cache_sha256"] == parent["validation_cache_sha256"] == VALIDATION_SHA256,
                 "Parent and child do not belong to the same frozen experiment/validation")
        oldname = f"fortitran_author_recipe_seed{seed}"
        cfg_record = parent["configs"][oldname]
        cfg_path = parent_root / cfg_record["file"]
        _require(file_hash(cfg_path) == cfg_record["sha256"], "Parent YAML SHA differs")
        oldcfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        _require(digest(oldcfg) == cfg_record["config_sha256"], "Parent YAML config digest differs")
        assert_config_transition(oldcfg, cfg)
        folder, destination = parent_root / "checkpoints" / oldname, output / "checkpoints" / name
        _require(not destination.exists(), f"Destination already exists; migration never overwrites a run: {destination}")
        paths = {key: folder / key for key in ("model.pt", "last.pt", "history.csv", "model.pt.sha256.json", "last.pt.sha256.json")}
        snapshots = {key: file_hash(path) for key, path in paths.items()}
        metadata = {}
        for key in ("model.pt", "last.pt"):
            sidecar = json.loads(paths[key + ".sha256.json"].read_text(encoding="utf-8"))
            metadata[key] = checkpoint_metadata(paths[key].read_bytes(), key)
            _validate_checkpoint(metadata[key], oldcfg, parent, sidecar, paths[key])
        with paths["history.csv"].open(encoding="utf-8-sig", newline="") as stream:
            history_rows = list(csv.DictReader(stream))
        prefix = check_committed_prefix(history_rows, metadata["model.pt"], metadata["last.pt"])
        marker_path = folder / "complete.json"
        if marker_path.exists():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            _require(marker.get("recipe_signature") == parent["signature"]
                     and type(marker.get("completed_epoch")) is int
                     and 0 <= marker["completed_epoch"] <= prefix["completed_epoch"], "Parent marker identity/epoch differs")
            if marker.get("status") == "complete":
                _require(prefix["parent_naturally_terminal"]
                         and marker["completed_epoch"] == prefix["completed_epoch"]
                         and marker.get("selected_epoch") == prefix["selected_epoch"]
                         and marker.get("selected_checkpoint_sha256") == snapshots["model.pt"]
                         and marker.get("last_checkpoint_sha256") == snapshots["last.pt"],
                         "Parent completion marker is unsupported by its committed prefix")
        _require(all(file_hash(path) == snapshots[key] for key, path in paths.items()), "Parent checkpoint commit changed during inspection; retry after a whole-epoch commit")
        plans.append((seed, child, cfg, name, folder, destination, paths, snapshots, prefix, cfg_path))
    torch = importlib.import_module("torch")
    _require(torch.cuda.is_available(), "Run migration on the connected CUDA server")
    receipts = []
    for seed, child, cfg, name, folder, destination, paths, snapshots, prefix, cfg_path in plans:
        proof_folder = output / "prefix_provenance" / f"seed{seed}"
        _require(not proof_folder.exists(), "Prefix provenance already exists; do not overwrite an audit snapshot")
        proof_folder.mkdir(parents=True)
        _copy_bound(parent_manifest_path, proof_folder / "parent_recipe_manifest.json", parent_manifest_sha)
        _copy_bound(cfg_path, proof_folder / "parent_config.yaml", parent["configs"][f"fortitran_author_recipe_seed{seed}"]["sha256"])
        for key, path in paths.items():
            _copy_bound(path, proof_folder / "parent_checkpoints" / key, snapshots[key])
        for relative, sha in engines.items():
            _copy_bound(parent_implementation / relative, proof_folder / "parent_engine" / relative, sha)
        for relative, sha in sources.items():
            _copy_bound(project / relative, proof_folder / "parent_runtime" / relative, sha)
        for optional in ("complete.json", "progress.json"):
            path = folder / optional
            if path.exists():
                _copy_bound(path, proof_folder / "parent_checkpoints" / optional, file_hash(path))
        proof = {"schema": "fortitran_equal_budget_prefix_provenance_v1", "status": "authenticated_prefix",
                 "parent_manifest_signature": parent["signature"], "child_manifest_signature": child["signature"],
                 "parent_manifest_sha256": parent_manifest_sha, "child_manifest_sha256": child_manifest_sha,
                 "parent_config_sha256": parent["configs"][f"fortitran_author_recipe_seed{seed}"]["sha256"],
                 "parent_checkpoint_files_sha256": snapshots, "parent_implementation_sha256": engines,
                 "parent_runtime_files_sha256": sources, "seed": seed, **prefix,
                 "claim": CLAIM, "termination": TERMINATION, "state_preservation": list(PRESERVED_STATE),
                 "state_changes": ["cfg recipe identity/claim/termination", "recipe_signature", "recipe_claim",
                                   "implementation_sha256", "early_stopping.patience:3→41 (best/stale unchanged)",
                                   "training_complete:epoch==40", "termination", "prefix_provenance"],
                 "parent_output": str(parent_root), "new_output": str(output),
                 "created_utc": datetime.now(timezone.utc).isoformat()}
        proof["signature"] = digest(proof)
        write_json(proof_folder / "provenance.json", proof)
        staging = destination.with_name(destination.name + f".migration-{os.getpid()}")
        _require(not staging.exists(), f"Migration staging already exists: {staging}")
        staging.mkdir(parents=True)
        for key in ("model.pt", "last.pt"):
            state = torch.load(proof_folder / "parent_checkpoints" / key, map_location="cpu", weights_only=False)
            migrated = dict(state)
            migrated.update(cfg=cfg, recipe_signature=child["signature"], recipe_claim=CLAIM,
                            implementation_sha256=child["implementation_sha256"],
                            early_stopping={**state["early_stopping"], "patience": TRACKER_PATIENCE},
                            training_complete=state["epoch"] == MAX_EPOCHS, termination=TERMINATION,
                            prefix_provenance={"signature": proof["signature"], "parent_checkpoint_sha256": snapshots[key],
                                               "relative_path": f"prefix_provenance/seed{seed}/provenance.json"})
            target = staging / key
            torch.save(migrated, target)
            loaded = torch.load(target, map_location="cpu", weights_only=False)
            for preserved in PRESERVED_STATE:
                _assert_equal(torch, state[preserved], loaded[preserved], preserved)
            changed = {"cfg", "recipe_signature", "recipe_claim", "implementation_sha256", "early_stopping", "training_complete", "termination", "prefix_provenance"}
            for unchanged in set(state) - changed:
                _assert_equal(torch, state[unchanged], loaded[unchanged], unchanged)
            _require(loaded["early_stopping"]["best"] == state["early_stopping"]["best"]
                     and loaded["early_stopping"]["stale"] == state["early_stopping"]["stale"], "Best/stale bookkeeping changed")
            write_json(target.with_suffix(target.suffix + ".sha256.json"),
                       {"sha256": file_hash(target), "bytes": target.stat().st_size,
                        "recipe_signature": child["signature"], "epoch": state["epoch"]})
        shutil.copyfile(proof_folder / "parent_checkpoints/history.csv", staging / "history.csv")
        _require(file_hash(staging / "history.csv") == snapshots["history.csv"], "History bytes changed")
        complete = prefix["completed_epoch"] == MAX_EPOCHS
        marker = {"status": "complete" if complete else "incomplete_migrated_prefix", "training_complete": complete,
                  "reason": "authenticated_terminal_checkpoint" if complete else "authenticated_same_trajectory_prefix",
                  "recipe_signature": child["signature"], "seed": seed,
                  "completed_epoch": prefix["completed_epoch"], "selected_epoch": prefix["selected_epoch"],
                  "max_epochs": MAX_EPOCHS, "termination": TERMINATION, "early_stopping_enabled": False,
                  "best_val_mse": prefix["best_val_mse"], "selection_metric": "val_mse", "claim": CLAIM,
                  "validation_cache_sha256": VALIDATION_SHA256,
                  "effective_batch_frames": 512, "gradient_microbatch_frames": cfg["train"]["gradient_microbatch_size"],
                  "exact_original_dataset_reproduction": False,
                  "selected_checkpoint_sha256": file_hash(staging / "model.pt"),
                  "model_sha256": file_hash(staging / "model.pt"), "last_checkpoint_sha256": file_hash(staging / "last.pt"),
                  "prefix_provenance_signature": proof["signature"], "finished_utc": datetime.now(timezone.utc).isoformat()}
        write_json(staging / "complete.json", marker)
        write_json(staging / "progress.json", marker)
        _require(all(file_hash(path) == snapshots[key] for key, path in paths.items())
                 and file_hash(parent_manifest_path) == parent_manifest_sha
                 and file_hash(child_manifest_path) == child_manifest_sha,
                 "Parent commit or manifest changed during migration; no new run published")
        staging.rename(destination)
        migrated_hashes = {key: file_hash(destination / key)
                           for key in ("model.pt", "last.pt", "history.csv")}
        for key in ("model.pt", "last.pt", "history.csv", "model.pt.sha256.json", "last.pt.sha256.json"):
            _copy_bound(destination / key, proof_folder / "migrated_checkpoints" / key,
                        migrated_hashes.get(key, file_hash(destination / key)))
        receipt = {"status": "migrated", "seed": seed, "run": name, "prefix_provenance_signature": proof["signature"],
                   "parent_checkpoint_files_sha256": snapshots, "new_checkpoint_files_sha256": migrated_hashes,
                   "completed_epoch": prefix["completed_epoch"], "remaining_epochs": MAX_EPOCHS - prefix["completed_epoch"],
                   "old_run_unchanged": True, "state_equality_checked": list(PRESERVED_STATE),
                   "child_manifest_signature": child["signature"], "created_utc": datetime.now(timezone.utc).isoformat()}
        receipt["signature"] = digest(receipt)
        write_json(proof_folder / "migration_receipt.json", receipt)
        receipts.append(receipt)
        print(f"PREFIX_MIGRATED {name} epoch={prefix['completed_epoch']} remaining={receipt['remaining_epochs']}; parent preserved", flush=True)
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], choices=SEEDS)
    parser.add_argument("--parent-implementation", type=Path)
    args = parser.parse_args()
    migrate(args.project, args.parent_manifest, args.manifest, seeds=tuple(args.seeds),
            parent_implementation=args.parent_implementation)


if __name__ == "__main__":
    main()
