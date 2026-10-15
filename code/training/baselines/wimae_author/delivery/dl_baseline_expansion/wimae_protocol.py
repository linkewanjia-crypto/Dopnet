"""Framework-free integrity checks for the independent WiMAE experiment."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .prepare_wimae import VALIDATION_SHA256, digest, file_hash, signed_metadata

SEEDS = (42, 43, 44)
OLD_METHODS = ("dopnet", "dopnet_lite", "lee3ddual", "wimae_ce", "fortitran")
NEW_METHOD = "wimae_author"
FORTI_METHOD = "fortitran_budget_matched"
CLAIM = "task_adapted_equal_frame_budget"
LABELS = {"dopnet": "DOP-Net", "dopnet_lite": "DOP-Net-Lite",
          "lee3ddual": "Dual3D (supervised adapter)",
          "wimae_ce": "WiMAE (supervised adapter)",
          "fortitran": "FortiTran (common recipe)",
          FORTI_METHOD: "FortiTran (equal budget)",
          NEW_METHOD: "WiMAE (20 MAE + 20 CE)"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def core_routes(routes):
    indices = []
    for method in OLD_METHODS:
        found = [(i, route) for i, route in enumerate(routes) if route["method"] == method]
        require([r["training_seed"] for _, r in found] == list(SEEDS),
                "Historical core method lacks ordered42/43/44 seeds: " + method)
        require(all(r["trained"] is True for _, r in found), "Core route is not trained")
        indices.extend(i for i, _ in found)
    return indices, [routes[i] for i in indices]


def check_marker(marker, seed, signature):
    expected = {"status": "complete", "training_complete": True, "seed": seed,
                "recipe_signature": signature, "completed_epoch": 40, "total_epochs": 40,
                "max_epochs": 40, "ssl_epochs": 20, "ce_epochs": 20,
                "effective_batch_frames": 32, "training_frame_exposures": 480000,
                "selection_metric": "val_mse", "claim": CLAIM,
                "original_CE_recipe_reproduction": False,
                "validation_cache_sha256": VALIDATION_SHA256}
    for key, value in expected.items():
        require(marker.get(key) == value, "WiMAE completion contract differs: " + key)
    require(21 <= marker.get("selected_epoch", 0) <= 40, "WiMAE CE selection is not a CE epoch")


def _read_state(path, manifest, cfg, reader):
    path = Path(path)
    sidecar = read_json(path.with_suffix(".pt.sha256.json"))
    sha = file_hash(path)
    require(sidecar.get("sha256") == sha and sidecar.get("bytes") == path.stat().st_size
            and sidecar.get("recipe_signature") == manifest["signature"], "Checkpoint sidecar differs: " + path.name)
    state = reader(path)
    require(state.get("cfg") == cfg and state.get("recipe_signature") == manifest["signature"]
            and state.get("source_files_sha256") == manifest["source_files_sha256"]
            and state.get("implementation_sha256") == manifest["implementation_sha256"]
            and state.get("validation_cache_sha256") == VALIDATION_SHA256
            and state.get("selection_metric") == "val_mse"
            and state.get("estimator") == "wimae_author_slice_ce", "Checkpoint recipe/model/source differs")
    epoch, phase = state.get("epoch"), state.get("phase")
    require(type(epoch) is int and 1 <= epoch <= 40 and phase == ("ssl" if epoch <= 20 else "ce"),
            "Checkpoint global epoch/phase differs")
    require(sidecar.get("epoch") == epoch and sidecar.get("phase") == phase, "Checkpoint sidecar epoch differs")
    rows = state.get("history", [])
    require([r["epoch"] for r in rows] == list(range(1, epoch + 1)), "Checkpoint ledger has missing epochs")
    for row in rows:
        e = row["epoch"]
        require(row["phase"] == ("ssl" if e <= 20 else "ce") and row["phase_epoch"] == (e if e <= 20 else e - 20)
                and row["train_frames"] == 12000 and row["optimizer_steps"] == 375,
                "Checkpoint does not satisfy fresh12000/effective32/20+20 ledger")
        require(math.isfinite(float(row["train_loss"])) and math.isfinite(float(row["val_loss"]))
                and float(row["val_loss"]) >= 0, "Nonfinite ledger loss")
    for key in ("rng_python", "rng_numpy", "rng_torch", "rng_cuda", "rng_private_training_mask"):
        require(key in state, "Checkpoint lacks its exact-resume state: " + key)
    require(state["val"]["loss"] == rows[-1]["val_loss"], "Checkpoint validation scalar differs from ledger")
    require(type(state["environment"]["torch"]) is str, "Torch version must be plain metadata text")
    return state, sha


def check_training(folder, manifest, cfg, reader):
    """Reader may be restricted metadata deserialization or trusted server load."""
    folder = Path(folder)
    marker = read_json(folder / "complete.json")
    check_marker(marker, cfg["seed"], manifest["signature"])
    last, last_sha = _read_state(folder / "last.pt", manifest, cfg, reader)
    selected, selected_sha = _read_state(folder / "model.pt", manifest, cfg, reader)
    ssl, ssl_sha = _read_state(folder / "ssl_best.pt", manifest, cfg, reader)
    require(last["epoch"] == 40 and last["training_complete"] is True and last["task"] == "ce",
            "Terminal checkpoint is not the committed40th epoch")
    require(selected["phase"] == "ce" and selected["task"] == "ce" and ssl["phase"] == "ssl",
            "Wrong phase/task in selected checkpoints")
    require(marker["last_checkpoint_sha256"] == last_sha and marker["selected_checkpoint_sha256"] == selected_sha
            and marker["model_sha256"] == selected_sha, "Completion checkpoint hashes differ")
    require(marker["gradient_microbatch_frames"] == cfg["wimae_reproduction"]["gradient_microbatch_frames"],
            "Completion microbatch differs from prepared recipe")
    for phase, state in (("ssl", ssl), ("ce", selected)):
        rows = [r for r in last["history"] if r["phase"] == phase]
        best = min(rows, key=lambda r: r["val_loss"])
        require(state["epoch"] == best["epoch"] and state["val"]["loss"] == best["val_loss"]
                and state["history"] == last["history"][:state["epoch"]], "Selected checkpoint is not phase validation best")
    require(last["best_ssl_mse"] == ssl["val"]["loss"] and last["best_ce_mse"] == selected["val"]["loss"]
            and marker["selected_epoch"] == selected["epoch"] and marker["best_val_mse"] == selected["val"]["loss"],
            "Selected validation losses differ")
    parent = last["ssl_parent"]
    require(parent == selected["ssl_parent"] == marker["ssl_parent"] and parent["checkpoint_sha256"] == ssl_sha
            and parent["epoch"] == ssl["epoch"] and parent["val_mse"] == ssl["val"]["loss"]
            and parent["global_transition_epoch"] == 21 and parent["optimizer_scheduler_reset"] is True
            and parent["global_training_RNG_preserved"] is True, "CE initialization provenance differs")
    with (folder / "history.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == 40, "History CSV lacks40epochs")
    for actual, expected in zip(csv_rows, last["history"]):
        require(set(actual) == set(expected), "History CSV columns differ")
        for key, value in expected.items():
            require(actual[key] == ("" if value is None else str(value)), "History CSV differs from committed ledger")
    hashes = {name: file_hash(folder / name) for name in ("complete.json", "model.pt", "last.pt", "ssl_best.pt", "history.csv")}
    return selected, hashes


def check_evaluation_metadata(output):
    output = Path(output)
    ev = signed_metadata(output / "evaluation.json")
    statistics = signed_metadata(output / "summary" / "statistics.json")
    done = read_json(output / "COMPLETE.json")
    require(done.get("status") == "complete" and done.get("signature") == ev["signature"]
            and done.get("cells") == 269 and len(ev["cells"]) == 269
            and read_json(output / "evaluation_complete.json") == {"signature": ev["signature"], "cells": 269}
            and statistics["evaluation_signature"] == ev["signature"], "Incomplete or unbound WiMAE evaluation")
    require(ev.get("samples_per_cell") == 512 and ev.get("batch_size") == 8
            and len(ev["routes"]) == 21 and ev.get("training_claim") == CLAIM
            and statistics.get("training_seeds") == list(SEEDS), "WiMAE evaluation protocol differs")
    require([r["method"] for r in ev["routes"]] == [m for m in (*OLD_METHODS, FORTI_METHOD, NEW_METHOD) for _ in SEEDS]
            and [r["training_seed"] for r in ev["routes"]] == list(SEEDS) * 7,
            "Seven-method/three-seed route layout differs")
    require(set(statistics["cell_file_sha256"]) == {c["cell_id"] for c in ev["cells"]}, "Statistics lacks cells")
    for cell in ev["cells"]:
        path = output / "cells" / (cell["cell_id"] + ".npz")
        marker = read_json(path.with_suffix(".COMPLETE.json"))
        require(marker.get("signature") == ev["signature"] and marker.get("score_sha256") == file_hash(path)
                == statistics["cell_file_sha256"][cell["cell_id"]], "Cell changed after scoring/statistics")
    for name, expected in statistics["summary_csv_sha256"].items():
        require(file_hash(output / "summary" / name) == expected, "Summary CSV changed after statistics")
    return ev, statistics
