"""Prepare three hash-bound FortiTran fixed-40-epoch plans; no computation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    _package_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _package_dir]
    sys.path.insert(0, str(_package_dir.parent))

from budget_matched_baselines.recipe import (
    SEEDS, RECIPE, CLAIM, MAX_EPOCHS, TERMINATION, PAPER_SETTINGS, SOURCES, VALIDATION_SHA256, VALIDATION_BYTES,
    build_config, digest, file_hash, implementation_sources, runtime_sources,
    signed_metadata, write_json,
)


def prepare(project, reference_results, validation_cache, output, microbatch=32):
    import yaml
    project = Path(project).resolve()
    reference = Path(reference_results).resolve()
    output = Path(output).resolve()
    source = signed_metadata(reference / "experiment.json")
    evaluation = signed_metadata(reference / "evaluation.json")
    if evaluation.get("experiment_signature") != source["signature"]:
        raise RuntimeError("Archived experiment and evaluation signatures differ")
    if source.get("validation_cache_sha256") != VALIDATION_SHA256:
        raise RuntimeError("Reference did not use the expected historical validation artifact")
    for seed in SEEDS:
        dopnet = source["configs"][f"dopnet_seed{seed}"]["train"]
        if int(dopnet.get("epochs", 0)) != MAX_EPOCHS or int(dopnet.get("train_samples", 0)) != 12000:
            raise RuntimeError("DOP-Net reference must use the same 40×12000-frame budget")
    sources = runtime_sources(project)
    if digest(sources) != source["code_sha256"]:
        raise RuntimeError(
            "Training project does not match the archived runtime code SHA-256. "
            "Use the original DOPNet_Submission_Experiments source tree; do not bypass this gate.")
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for seed in SEEDS:
        archived = source["configs"][f"fortitran_seed{seed}"]
        cfg = build_config(archived, seed, microbatch=microbatch)
        name = f"fortitran_budget_matched_seed{seed}"
        relative = f"configs/{name}.yaml"
        cfg_path = output / relative
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        contents = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        if cfg_path.exists() and cfg_path.read_text(encoding="utf-8") != contents:
            raise RuntimeError(f"Do not overwrite a different training configuration: {cfg_path}")
        cfg_path.write_text(contents, encoding="utf-8", newline="\n")
        records[name] = {"file": relative, "sha256": file_hash(cfg_path),
                         "config_sha256": digest(cfg)}
    manifest = {
        "schema": "dopnet_budget_matched_plan_v1", "recipe": RECIPE,
        "claim": CLAIM, "seeds": list(SEEDS),
        "paper_settings": PAPER_SETTINGS, "sources": SOURCES,
        "reference_experiment_signature": source["signature"],
        "reference_evaluation_signature": evaluation["signature"],
        "reference_runtime_code_sha256": source["code_sha256"],
        "source_files_sha256": sources,
        "implementation_sha256": implementation_sources(),
        "validation_cache_sha256": VALIDATION_SHA256,
        "validation": {"path": str(Path(validation_cache).resolve()),
                       "sha256": VALIDATION_SHA256, "bytes": VALIDATION_BYTES,
                       "frames": 2000, "generation_seed": 610000001,
                       "usage": "checkpoint_selection_only; never optimizer input"},
        "configs": records,
        "initialization": "independently seeded weights; authenticated same-trajectory prefix migration is separately attested",
        "training_draws": "unchanged archived generator; seed+10001; 12000 new frames/epoch",
        "test_data": "not consumed by this training engine",
        "termination": TERMINATION, "max_epochs": MAX_EPOCHS,
        "early_stopping_enabled": False,
        "completion": "exactly40 committed epochs; deadline is incomplete, never successful completion",
        "comparison_budget": {"fresh_frames_per_epoch": 12000, "epochs": MAX_EPOCHS,
                              "total_fresh_training_frames": 480000},
    }
    manifest["signature"] = digest(manifest)
    target = output / "recipe_manifest.json"
    if target.exists() and signed_metadata(target) != manifest:
        raise RuntimeError("Output already contains a different manifest; choose a new output directory")
    write_json(target, manifest)
    print(f"BUDGET_MATCHED_PREPARED {target}; metadata only; no model or data calculation", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--reference-results", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--microbatch-size", type=int, default=32)
    args = parser.parse_args()
    prepare(args.project, args.reference_results, args.validation_cache,
            args.output, args.microbatch_size)


if __name__ == "__main__":
    main()
