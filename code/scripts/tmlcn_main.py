#!/usr/bin/env python3
"""Explicit future full-training orchestration for the 33 retained main routes.

This adds portable path bindings and route selection around the unchanged
scientific functions. No numerical function is reimplemented. --help imports
no scientific framework. Called only by the opt-in tmlcn.py facade.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

METHODS = (
    "dopnet", "cfr_derived", "slow_time", "pilot_only_fast_time", "aft_only",
    "no_fusion", "no_offgrid", "lee3ddual", "wimae_ce", "dopnet_lite", "fortitran",
)
SEEDS = (42, 43, 44)
REFERENCE_REL = "data/results/base_scores/original_retained"
OLD_TRAINING_MANIFEST_SHA = "1e1e781e2ca0f07d8aeb6b798968362199e999bb29b7489eb020c6455025420f"
VALIDATION_SHA = "83d0e03524628b617671c42a73d3c458cc3395c31965a4e22295e8389f56e597"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=("train", "generate", "evaluate"), required=True)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--epoch", type=int, choices=range(1, 41))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--validation-cache", type=Path,
                   help="Real fixed cache with a genuine adjacent reconstruction attestation")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    root = args.root.resolve()
    source = root / "code/training/main"
    out = args.output.expanduser().resolve()
    if out == root or out.is_relative_to(root):
        raise ValueError("Use an output outside the immutable package")
    if args.stage != "evaluate" and (args.method is None or args.seed is None):
        raise ValueError("Select one retained --method and --seed")
    if args.stage == "generate" and args.epoch is None:
        raise ValueError("Generation requires one explicit --epoch")
    # Original experiment_lock uses Linux fcntl, and the physics runtime needs
    # CUDA. Refuse before importing torch on unsupported local execution hosts.
    if os.name != "posix":
        raise RuntimeError("Future scientific execution requires a compatible Linux CUDA host")
    validation = (args.validation_cache.resolve() if args.validation_cache else
                  root / "data/reproduction_inputs/validation_895338a79e73ad3f.pt")
    if args.stage == "train":
        proof = validation.with_name(validation.stem + "_verified.json")
        if not proof.is_file():
            raise RuntimeError("Missing genuine fixed-validation reconstruction attestation. "
                               "Run the explicit verify-validation operator first, then pass its cache via --validation-cache.")
    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(root / "code/training/portable"))
    from experiment_protocol import code_hash, digest, file_hash, write_json
    from experiment_lock import experiment_lock
    from dopnet_public_reference import ReferenceResults
    from historical_reuse import HistoricalTrainingReference
    reference_path = root / REFERENCE_REL
    reference = ReferenceResults(reference_path)
    old_training_path = root / "data/training_history/original_selected/experiment.json"
    if file_hash(old_training_path) != OLD_TRAINING_MANIFEST_SHA:
        raise RuntimeError("Original a720 training metadata bytes differ")
    old_training = json.loads(old_training_path.read_text(encoding="utf-8"))
    if old_training.get("signature") != digest({k: v for k, v in old_training.items() if k != "signature"}):
        raise RuntimeError("Original training metadata signature differs")
    # Training receipts retain their actual a720 identity; fixed-test scores use
    # the separate corrected 0af lineage and new 38-route representation.
    historical = HistoricalTrainingReference(root / "code/frozen_base/metadata_reference",
                                            SimpleNamespace(experiment=old_training))
    configs = {}
    for method in METHODS:
        for seed in SEEDS:
            name = f"{method}_seed{seed}"
            cfg = json.loads((root / "code/config/training/main" / (name + ".json")).read_text(encoding="utf-8-sig"))
            if cfg["seed"] != seed or cfg["train"]["epochs"] != 40 or cfg["train"]["train_samples"] != 12000:
                raise RuntimeError("Registered training budget/config differs: " + name)
            configs[name] = cfg
    payload = dict(code_sha256=code_hash(source), configs=configs,
                   training_mode="tmlcn_fresh_33_explicit_method_seed_invocations",
                   historical_data_comparison_required=True,
                   historical_training_manifest_signature=historical.manifest_set_signature,
                   samples_per_cell=512, batch_size=8,
                   validation_source="historical_fixed_cache", validation_cache_sha256=VALIDATION_SHA,
                   reference_results=reference.view_provenance(),
                   historical_training_source_sha256=OLD_TRAINING_MANIFEST_SHA)
    manifest = dict(**payload, signature=digest(payload))

    def run(command, name):
        log = out / "logs" / (name + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as stream:
            result = subprocess.run(list(map(str, command)), cwd=source,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Scientific subprocess failed; inspect " + str(log))

    with experiment_lock(out):
        manifest_path = out / "experiment.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Output contains a different immutable experiment plan")
        if not manifest_path.exists():
            write_json(manifest_path, manifest)
        for name, cfg in configs.items():
            path = out / "configs" / (name + ".yaml")
            # JSON is valid YAML. Keep all original scientific configuration
            # values; no dependency import is needed for serialization.
            if path.exists() and json.loads(path.read_text(encoding="utf-8")) != cfg:
                raise RuntimeError("Existing route config differs: " + name)
            if not path.exists():
                write_json(path, cfg)
        if args.stage == "generate":
            cache = out / "epoch_cache" / f"seed{args.seed}_epoch{args.epoch:03d}"
            run([sys.executable, "-B", "epoch_cache.py", "--config",
                 out / "configs" / f"dopnet_seed{args.seed}.yaml", "--epoch", args.epoch,
                 "--output", cache], f"cache_{args.seed}_{args.epoch}")
            data = json.loads((cache / "complete.json").read_text(encoding="utf-8"))
            # Check every archived tensor hash before any optimizer operation.
            proof = historical.check_epoch(data, args.seed, args.epoch)
            write_json(out / "historical_data_match" / f"seed{args.seed}_epoch{args.epoch:03d}.json", proof)
            write_json(out / "data_manifests" / f"seed{args.seed}_epoch{args.epoch:03d}.json", data)
        elif args.stage == "train":
            from fixed_validation import (assert_historical_identity, assert_reconstruction_attestation,
                                          verify_historical_cache_file)
            cfg = configs[f"{args.method}_seed{args.seed}"]
            assert_historical_identity(cfg)
            verify_historical_cache_file(validation, VALIDATION_SHA)
            assert_reconstruction_attestation(cfg, validation)
            # The original runner is unchanged on disk. Only its orchestration
            # seed selection is narrowed for this explicit invocation.
            import round_robin
            round_robin.SEEDS = (args.seed,)
            round_robin.train_rounds(out, configs, manifest, run,
                fixed_validation_cache=validation, fixed_validation_sha256=VALIDATION_SHA,
                train_variants=(args.method,), historical_training_reference=historical)
            # The original progress filename now describes a selected route.
            # Make that scope explicit rather than implying all 33 are done.
            completed = []
            for name in configs:
                marker = out / "checkpoints" / name / "complete.json"
                if marker.is_file():
                    value = json.loads(marker.read_text(encoding="utf-8"))
                    if value.get("signature") == manifest["signature"]:
                        completed.append(name)
            write_json(out / "training_progress.json", dict(
                requested_route=f"{args.method}_seed{args.seed}",
                requested_route_completed_epochs=40, total_epochs_per_route=40,
                completed_routes=completed, expected_routes=len(configs),
                all_retained_routes_complete=len(completed) == len(configs)))
        else:
            historical.verify_proofs(out)
            from dopnet_public_reference import main_evaluate as evaluate
            evaluate.VARIANTS = {name: evaluate.VARIANTS[name] for name in METHODS}
            evaluate.run(out, str(reference_path))
            run([sys.executable, "-B", "aggregate_results.py", "--output", out,
                 "--bootstrap", "2000"], "aggregation")
    print(json.dumps({"stage": args.stage, "output": str(out), "science_source_bytes_changed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
