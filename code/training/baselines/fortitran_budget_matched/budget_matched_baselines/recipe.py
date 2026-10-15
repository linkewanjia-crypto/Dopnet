"""Metadata-only source-guided FortiTran recipe at the DOP-Net training budget.

The original waveform, project architecture adapter, and training-mixture draws
are retained. This is not a reproduction of the original SISO dataset.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path

SEEDS = (42, 43, 44)
RECIPE = "fortitran_source_guided_equal_budget_v1"
CLAIM = "source_guided_equal_budget"
MAX_EPOCHS = 40
TERMINATION = "fixed_budget_40_epochs"
TRACKER_PATIENCE = MAX_EPOCHS + 1
VALIDATION_SHA256 = "83d0e03524628b617671c42a73d3c458cc3395c31965a4e22295e8389f56e597"
VALIDATION_BYTES = 4_682_930_065
VALIDATION_NAME = "validation_895338a79e73ad3f.pt"
PAPER_SETTINGS = {
    "optimizer": "adam", "loss": "real_imag_concatenated_mse",
    "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0,
    "batch_size": 512, "lr": 0.001, "lr_decay_per_epoch": 0.995,
    "max_epochs": 1000, "early_stopping": True,
    "training_frames": 100000, "validation_frames": 10000,
    "early_stopping_patience_in_paper": None,
}
SOURCES = {
    "paper": "https://arxiv.org/html/2505.09076v1",
    "official_repository": "https://github.com/BerkIGuler/AdaFortiTran",
    "official_trainer": "https://raw.githubusercontent.com/BerkIGuler/AdaFortiTran/master/src/main/trainer.py",
    "official_patience_default": "https://raw.githubusercontent.com/BerkIGuler/AdaFortiTran/master/src/main/parser.py",
    "official_early_stopping": "https://raw.githubusercontent.com/BerkIGuler/AdaFortiTran/master/src/utils.py",
    "license": "MIT (official repository; no third-party implementation is vendored here)",
}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def runtime_sources(project):
    """Exactly the file selection used by the archived code_hash contract."""
    project = Path(project).resolve()
    protocol_path = project / "experiment_protocol.py"
    if not protocol_path.is_file():
        raise RuntimeError("Training project has no archived experiment_protocol.py")
    tree = ast.parse(protocol_path.read_text(encoding="utf-8-sig"), filename=str(protocol_path))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "code_hash"]
    if len(functions) != 1:
        raise RuntimeError("Archived source must define exactly one code_hash function")
    assignments = [node for node in functions[0].body
                   if isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Name) and target.id == "runtime"
                       for target in node.targets)]
    if len(assignments) != 1:
        raise RuntimeError("Archived code_hash must contain one literal runtime file set")
    runtime = ast.literal_eval(assignments[0].value)
    if (not isinstance(runtime, set) or not runtime or any(
            not isinstance(name, str) or not name.endswith(".py")
            or Path(name).name != name or "/" in name or "\\" in name
            or name in (".", "..") for name in runtime)):
        raise RuntimeError("Archived runtime selection is not a set of Python file basenames")
    files = [p for p in project.rglob("*.py")
             if (p.parent == project and p.name in runtime)
             or (p.parent.parent == project and p.parent.name in ("data", "models", "utils"))]
    required = ("train.py", "fixed_validation.py", "experiment_protocol.py",
                "data/generator.py", "models/baselines.py", "models/receiver.py",
                "utils/checkpoints.py", "utils/metrics.py")
    missing = [name for name in required if not (project / name).is_file()]
    if missing:
        raise RuntimeError(f"Training project source is incomplete: {missing}")
    return {p.relative_to(project).as_posix(): file_hash(p) for p in sorted(files)}


def implementation_sources():
    root = Path(__file__).resolve().parent
    return {name: file_hash(root / name)
            for name in ("recipe.py", "prepare.py", "train_author.py", "migrate_prefix.py")}


def signed_metadata(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    signature = data.get("signature")
    if not isinstance(signature, str) or signature != digest(
            {key: value for key, value in data.items() if key != "signature"}):
        raise RuntimeError(f"Invalid reference metadata signature: {path}")
    return data


def build_config(archived_config, seed, *, microbatch=32):
    """Change training recipe only; sampling and observation fields stay exact."""
    if seed not in SEEDS:
        raise ValueError(f"Supported training seeds are {SEEDS}")
    if not isinstance(microbatch, int) or not 0 < microbatch <= 32:
        raise ValueError("Gradient microbatch must be between 1 and 32 frames")
    cfg = copy.deepcopy(archived_config)
    if cfg.get("model", {}).get("estimator") != "fortitran":
        raise RuntimeError("Reference config must describe the archived FortiTran adapter")
    if int(cfg.get("train", {}).get("train_samples", 0)) != 12000:
        raise RuntimeError("Expected the archived 12,000 fresh training frames per epoch")
    if int(cfg["train"].get("epochs", 0)) != MAX_EPOCHS:
        raise RuntimeError("The archived comparison must use the same 40-epoch training budget")
    if int(cfg["train"].get("val_samples", 0)) != 2000:
        raise RuntimeError("Expected the archived 2,000-frame validation artifact")
    if int(cfg["train"].get("validation_seed", 0)) != 610000001:
        raise RuntimeError("Archived validation generation seed differs")
    if int(cfg.get("system", {}).get("threegpp_link_batch_size", 0)) != 8:
        raise RuntimeError("Do not alter the archived Sionna link chunk of 8")
    geometry = {key: int(cfg["system"].get(key, 0)) for key in
                ("n_ant", "n_subcarriers", "frame_num_symbols", "n_users")}
    if geometry != {"n_ant": 8, "n_subcarriers": 128, "frame_num_symbols": 25, "n_users": 2}:
        raise RuntimeError("Expected Nr=8, Nf=128, T=25 and Nu=2")
    if int(cfg.get("data_pipeline", {}).get("generation_batch_size", 0)) != 32:
        raise RuntimeError("Do not alter the archived generation batch of 32")
    if bool(cfg["data_pipeline"].get("shuffle_buffer", True)):
        raise RuntimeError("Expected the archived nonshuffled generation stream")
    cfg["seed"] = seed
    cfg["device"] = "cuda"
    cfg["train"].update(
        task="ce", batch_size=512, gradient_microbatch_size=microbatch,
        validation_microbatch_size=microbatch, epochs=MAX_EPOCHS,
        optimizer="adam", lr=0.001, weight_decay=0.0,
        betas=[0.9, 0.999], eps=1e-8, amsgrad=False,
        scheduler="exponential", scheduler_gamma=0.995,
        warmup_epochs=0, min_lr=0.0,
        baseline_loss_mode="paper", nmse_weight=0.0,
        symbol_loss_weight=0.0, grad_clip=None,
        amp="none", tf32=False, cudnn_benchmark=False, fused_optimizer=False,
        selection_metric="val_mse", early_stopping_patience=TRACKER_PATIENCE,
        early_stopping_min_delta=0.0,
    )
    cfg["author_recipe"] = {
        "identifier": RECIPE,
        "claim": CLAIM,
        "exact_original_dataset_reproduction": False,
        "architecture": "archived_source_guided_fortitran_geometry_adapter",
        "effective_batch_unit": "received_multiuser_frames",
        "optimizer_batch_frames": 512,
        "original_batch_unit": "SISO_frames",
        "gradient_accumulation": "sample_weighted_mean; exact short-tail denominator",
        "loss_region": "data_OFDM_symbols (the archived task region)",
        "original_loss_region": "full_SISO_grid",
        "precision": "float32; TF32 disabled",
        "fresh_training_frames_per_epoch": 12000,
        "original_training_frames": 100000,
        "historical_validation_frames": 2000,
        "original_validation_frames": 10000,
        "early_stopping_patience": TRACKER_PATIENCE,
        "early_stopping_enabled": False,
        "termination": TERMINATION,
        "max_epochs": MAX_EPOCHS,
        "original_max_epochs": 1000,
        "original_early_stopping": True,
        "original_repository_patience": 3,
        "patience_provenance": "best/stale bookkeeping only; never terminates this fixed-40-epoch run",
        "oracle_channel_statistics": False,
        "geometry_adaptations": {
            "original_ofdm_grid": [120, 14], "task_ofdm_grid": [128, 25],
            "original_pilot_lattice": [40, 2], "task_pilot_pool": [64, 7],
            "original_patch": [3, 2], "task_patch": [4, 5],
            "task_users": 2, "task_receive_branches": 8,
        },
        "required_engine": "budget_matched_baselines/train_author.py",
        "warning": "source-guided Adam/MSE/effective512/gamma.995; original1000/early-stop3 replaced by fixed40 to match DOP-Net",
    }
    if cfg["system"] != archived_config["system"] or cfg["data_pipeline"] != archived_config["data_pipeline"]:
        raise AssertionError("Source sampling or physical observations changed")
    if cfg["model"] != archived_config["model"] or cfg["receiver"] != archived_config["receiver"]:
        raise AssertionError("The archived estimator geometry changed")
    return cfg


class EarlyStopping:
    """Strict MSE best/stale tracker; fixed-budget engine ignores the stop flag."""
    def __init__(self, patience=TRACKER_PATIENCE, best=None, stale=0):
        if patience < 1 or stale < 0 or stale > patience:
            raise ValueError("Invalid early-stopping state")
        self.patience = int(patience)
        self.best = best
        self.stale = int(stale)

    def update(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite validation MSE")
        improved = self.best is None or value < self.best
        if improved:
            self.best, self.stale = value, 0
        else:
            self.stale += 1
        return improved, self.stale >= self.patience

    def state_dict(self):
        return {"patience": self.patience, "best": self.best, "stale": self.stale}


def window_size(total_frames, emitted_frames, effective_batch=512):
    remaining = int(total_frames) - int(emitted_frames)
    if remaining <= 0 or effective_batch <= 0:
        raise ValueError("No valid optimizer window remains")
    return min(int(effective_batch), remaining)


def validate_manifest(manifest_path, project, seed):
    """No tensor deserialization or model calculation is needed for this check."""
    path = Path(manifest_path).resolve()
    manifest = signed_metadata(path)
    if manifest.get("recipe") != RECIPE or manifest.get("seeds") != list(SEEDS):
        raise RuntimeError("Unexpected author-recipe manifest")
    sources = runtime_sources(project)
    if sources != manifest["source_files_sha256"]:
        raise RuntimeError("Frozen training source changed; do not resume this experiment")
    if digest(sources) != manifest["reference_runtime_code_sha256"]:
        raise RuntimeError("Project source no longer matches the archived runtime hash")
    if implementation_sources() != manifest["implementation_sha256"]:
        raise RuntimeError("Author-recipe engine changed; generate a new independent manifest")
    if seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}")
    name = f"fortitran_budget_matched_seed{seed}"
    record = manifest["configs"][name]
    cfg_path = path.parent / record["file"]
    if file_hash(cfg_path) != record["sha256"]:
        raise RuntimeError("Author-recipe YAML differs from the frozen plan")
    # YAML may contain other user structures, so import only the safe loader.
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if digest(cfg) != record["config_sha256"]:
        raise RuntimeError("Author-recipe configuration digest differs")
    if cfg["author_recipe"]["identifier"] != RECIPE:
        raise RuntimeError("Configuration is not the declared author recipe")
    return manifest, cfg, name
