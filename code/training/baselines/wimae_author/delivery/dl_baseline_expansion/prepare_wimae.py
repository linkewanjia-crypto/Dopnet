"""Prepare an independent 20+20 WiMAE experiment; metadata only."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

SEEDS = (42, 43, 44)
RECIPE = "wimae_slice_ce_equal_frames_20plus20_v1"
UPSTREAM_COMMIT = "c0909cf37c7d04f9598005da7f18d1501b8daa6c"
VALIDATION_SHA256 = "83d0e03524628b617671c42a73d3c458cc3395c31965a4e22295e8389f56e597"
VALIDATION_BYTES = 4_682_930_065
HERE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def signed_metadata(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if value.get("signature") != digest({k: v for k, v in value.items() if k != "signature"}):
        raise RuntimeError(f"Invalid metadata signature: {path}")
    return value


def runtime_sources(project):
    project = Path(project).resolve()
    tree = ast.parse((project / "experiment_protocol.py").read_text(encoding="utf-8-sig"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "code_hash"]
    assignments = [n for n in functions[0].body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == "runtime" for t in n.targets)] if len(functions) == 1 else []
    if len(assignments) != 1:
        raise RuntimeError("Archived code_hash must declare one literal runtime file set")
    names = ast.literal_eval(assignments[0].value)
    if not isinstance(names, set) or not names or any(
            not isinstance(n, str) or not n.endswith(".py") or Path(n).name != n
            or "/" in n or "\\" in n for n in names):
        raise RuntimeError("Invalid archived runtime filename set")
    files = [p for p in project.rglob("*.py") if
             (p.parent == project and p.name in names) or
             (p.parent.parent == project and p.parent.name in ("data", "models", "utils"))]
    return {p.relative_to(project).as_posix(): file_hash(p) for p in sorted(files)}


def implementation_sources():
    files = [HERE / n for n in ("prepare_wimae.py", "wimae_author.py", "train_wimae.py")]
    files += list((HERE / "vendor" / "wimae_upstream").rglob("*.py"))
    files += [HERE / "vendor" / "__init__.py", HERE / "vendor" / "wimae_upstream" / "LICENSE",
              HERE / "vendor" / "wimae_upstream" / "UPSTREAM.json"]
    return {p.relative_to(HERE).as_posix(): file_hash(p) for p in sorted(set(files))}


def verify_upstream():
    root = HERE / "vendor" / "wimae_upstream"
    metadata = json.loads((root / "UPSTREAM.json").read_text(encoding="utf-8-sig"))
    if metadata.get("commit") != UPSTREAM_COMMIT or metadata.get("license") != "MIT":
        raise RuntimeError("Unrecognized WiMAE upstream snapshot")
    required = {"models/wimae.py", "models/modules/encoder.py", "models/modules/decoder.py",
                "models/modules/masking.py", "models/modules/patching.py",
                "models/modules/pos_encodings.py", "LICENSE"}
    if set(metadata["files"]) != required:
        raise RuntimeError("Incomplete or unexpected upstream snapshot")
    for name, record in metadata["files"].items():
        path = root / name
        if file_hash(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Vendored upstream bytes changed: {name}")
    return metadata


def build_config(archived, seed, microbatch=1, effective_batch=32, amp="bf16"):
    if seed not in SEEDS or microbatch not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Unsupported seed or frame microbatch")
    if effective_batch != 32 or microbatch > effective_batch or amp not in ("none", "bf16"):
        raise ValueError("Effective frame batch is fixed32; precision is none/bf16")
    cfg = copy.deepcopy(archived)
    s, tr, dp = cfg["system"], cfg["train"], cfg["data_pipeline"]
    expected = {"n_users": 2, "n_ant": 8, "n_subcarriers": 128, "frame_num_symbols": 25,
                "threegpp_link_batch_size": 8}
    if any(s.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Frozen physical geometry/link chunk differs")
    if (tr.get("train_samples") != 12000 or tr.get("val_samples") != 2000
            or tr.get("validation_seed") != 610000001
            or dp.get("generation_batch_size") != 32 or dp.get("shuffle_buffer") is not False):
        raise RuntimeError("Frozen data stream or historical validation differs")
    cfg["seed"], cfg["device"] = seed, "cuda"
    # Keep archived model/receiver fields for generator and validation identity.
    # The independent model reads wimae_reproduction, never that registry entry.
    tr.update(epochs=40, batch_size=32, gradient_microbatch_size=microbatch,
              validation_microbatch_size=microbatch, amp=amp, tf32=False,
              cudnn_benchmark=False, selection_metric="val_mse", optimizer="adam",
              lr=3e-4, weight_decay=0.001, betas=[0.9, 0.999], eps=1e-8,
              scheduler="phase_cosine", warmup_epochs=0, min_lr=3e-6,
              grad_clip=1.0, early_stopping=False, nmse_weight=0.0,
              symbol_loss_weight=0.0, baseline_loss_mode="paper")
    cfg["wimae_reproduction"] = {
        "identifier": RECIPE, "architecture": "official_WiMAE_joint_antenna_frequency_time_slices",
        "claim": "task_adapted_equal_frame_budget", "original_CE_recipe_reproduction": False,
        "upstream_commit": UPSTREAM_COMMIT, "patch_size": [1, 16], "encoder_dim": 64,
        "encoder_layers": 12, "encoder_nhead": 16, "decoder_layers": 4,
        "decoder_nhead": 8, "mask_ratio": 0.6, "tokens_per_slice": 128,
        "slices_per_frame": 50, "ssl_epochs": 20, "ce_epochs": 20, "total_epochs": 40,
        "fresh_frames_per_global_epoch": 12000, "training_seed_offset": 10001,
        "effective_batch_frames": 32, "gradient_microbatch_frames": microbatch,
        "normalization": {"kind": "observed_pilot_complex_rms_per_frame_user",
                          "epsilon": 1e-8, "minimum_scale": 1e-6, "center": False,
                          "inverse_uses_target": False},
        "optimizer": {"type": "adam", "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.001, "ssl_lr": 3e-4, "ce_lr": 1e-4},
        "scheduler": {"type": "cosine", "T_max_per_phase": 20, "eta_min": 3e-6,
                      "reset_at_phase_boundary": True},
        "gradient_clip": 1.0, "amp": amp, "early_stopping": False,
        "ssl_selection": "lowest_fixed_mask_historical_validation_reconstruction_mse",
        "ce_selection": "lowest_raw_data_symbol_historical_validation_mse",
        "train_mask_seed": seed + 930000001,
        "validation_mask_seed": 940000001,
        "phase_transition": "best_SSL_weights; reset_optimizer_scheduler_once; preserve_global_epoch_RNG",
        "original_pretraining": {"epochs": 3000, "batch_channel_matrices": 3072,
                                 "lr": 3e-4, "eta_min": 3e-6, "CE_recipe": "not_defined"},
        "adaptations": ["8x128 instead of32x32", "per_symbol_shared_weights_no_temporal_attention",
                        "20masked_reconstruction_plus20supervised_CE", "32multiuser_frames_optimizer_batch",
                        "observed_pilot_RMS_instead_of_global_training_RI_statistics",
                        "phase_cosine_Tmax20_no_early_stop", "bf16_if_configured", "raw_CE_loss"],
    }
    if cfg["system"] != archived["system"] or cfg["data_pipeline"] != archived["data_pipeline"]:
        raise AssertionError("Physical generation contract changed")
    return cfg


def validate_manifest(path, project, seed):
    path = Path(path).resolve()
    manifest = signed_metadata(path)
    if manifest.get("recipe") != RECIPE or manifest.get("seeds") != list(SEEDS):
        raise RuntimeError("Wrong WiMAE experiment manifest")
    if runtime_sources(project) != manifest["source_files_sha256"]:
        raise RuntimeError("Archived waveform/runtime source changed")
    if implementation_sources() != manifest["implementation_sha256"]:
        raise RuntimeError("WiMAE engine/adapter/upstream snapshot changed")
    verify_upstream()
    if seed not in SEEDS:
        raise ValueError("Unsupported seed")
    name = f"wimae_author_seed{seed}"
    record = manifest["configs"][name]
    cfg_path = path.parent / record["file"]
    if file_hash(cfg_path) != record["sha256"]:
        raise RuntimeError("Prepared WiMAE configuration bytes changed")
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if digest(cfg) != record["config_sha256"]:
        raise RuntimeError("Prepared WiMAE configuration identity differs")
    return manifest, cfg, name


def prepare(project, reference_results, validation_cache, output, microbatch=1, amp="bf16"):
    import yaml
    project, reference, output = map(lambda p: Path(p).resolve(),
                                     (project, reference_results, output))
    if output == reference or reference in output.parents or output == project or project in output.parents:
        raise RuntimeError("Use an independent output outside the archived project/reference")
    experiment = signed_metadata(reference / "experiment.json")
    evaluation = signed_metadata(reference / "evaluation.json")
    if (evaluation.get("experiment_signature") != experiment["signature"]
            or experiment.get("validation_cache_sha256") != VALIDATION_SHA256):
        raise RuntimeError("Historical experiment/evaluation/validation identity differs")
    sources = runtime_sources(project)
    if digest(sources) != experiment["code_sha256"]:
        raise RuntimeError("Use the exact archived experiment source tree")
    upstream = verify_upstream()
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for seed in SEEDS:
        cfg = build_config(experiment["configs"][f"wimae_ce_seed{seed}"], seed,
                           microbatch=microbatch, amp=amp)
        name = f"wimae_author_seed{seed}"
        relative = f"configs/{name}.yaml"
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        if path.exists() and path.read_text(encoding="utf-8") != contents:
            raise RuntimeError("Do not overwrite a different WiMAE plan")
        path.write_text(contents, encoding="utf-8", newline="\n")
        records[name] = {"file": relative, "sha256": file_hash(path), "config_sha256": digest(cfg)}
    manifest = {
        "schema": "dopnet_wimae_equal_frames_v1", "recipe": RECIPE, "seeds": list(SEEDS),
        "reference_experiment_signature": experiment["signature"],
        "reference_evaluation_signature": evaluation["signature"],
        "reference_runtime_code_sha256": experiment["code_sha256"],
        "source_files_sha256": sources, "implementation_sha256": implementation_sources(),
        "upstream": upstream, "validation_cache_sha256": VALIDATION_SHA256,
        "validation": {"path": str(Path(validation_cache).resolve()), "sha256": VALIDATION_SHA256,
                       "bytes": VALIDATION_BYTES, "frames": 2000, "training_use": False},
        "configs": records, "completion": "all40epochs;20SSL+20CE;12000fresh_frames_each",
        "training_frame_exposures_per_seed": 480000, "test_data_used_for_training": False,
        "claim": "task_adapted_equal_frame_budget", "time_budget": "external_explicit_deadline_required",
    }
    manifest["signature"] = digest(manifest)
    path = output / "wimae_manifest.json"
    if path.exists() and signed_metadata(path) != manifest:
        raise RuntimeError("Output contains another frozen plan")
    write_json(path, manifest)
    print(f"WIMAE_PREPARED {path}; no model execution or data generation", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "reference-results", "validation-cache", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8, 16, 32), default=1)
    parser.add_argument("--amp", choices=("none", "bf16"), default="bf16")
    args = parser.parse_args()
    prepare(args.project, args.reference_results, args.validation_cache,
            args.output, args.microbatch, args.amp)


if __name__ == "__main__":
    main()
