#!/usr/bin/env python3
"""Portable TMLCN entry point. Help, list and plans use only Python's stdlib.

No scientific process starts unless --execute is explicitly supplied. Original
scientific files and gates are preserved; this facade only binds local paths.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

MAIN_METHODS = (
    "dopnet", "cfr_derived", "slow_time", "pilot_only_fast_time", "aft_only",
    "no_fusion", "no_offgrid", "lee3ddual", "wimae_ce", "dopnet_lite", "fortitran",
)
EXTERNAL = {
    "fortitran_budget_matched": {
        "cwd": "code/training/baselines/fortitran_budget_matched",
        "prepare": "budget_matched_baselines.prepare",
        "train": "budget_matched_baselines.train_author",
        "evaluate": "budget_matched_baselines.evaluate_author",
        "manifest": "recipe_manifest.json",
    },
    "wimae_author": {
        "cwd": "code/training/baselines/wimae_author",
        "prepare": "dl_baseline_expansion.prepare_wimae",
        "train": "dl_baseline_expansion.train_wimae",
        "evaluate": "dl_baseline_expansion.evaluate_wimae",
        "manifest": "wimae_manifest.json",
    },
    "lee_meta": {
        "cwd": "code/training/baselines/lee_alternative_meta_task_adapted",
        "prepare": "lee_meta_equal40.prepare",
        "manifest": "lee_manifest.json",
    },
}
SEEDS = (42, 43, 44)
REFERENCE_REL = "data/results/base_scores/original_retained"
VALIDATION_REL = "data/reproduction_inputs/validation_895338a79e73ad3f.pt"
VALIDATION_SHA = "83d0e03524628b617671c42a73d3c458cc3395c31965a4e22295e8389f56e597"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def local(root, relative):
    p = (root / relative).resolve()
    if not p.is_relative_to(root):
        raise ValueError("Package binding escapes root: " + relative)
    return p


def output_path(root, value):
    p = Path(value).expanduser().resolve()
    if p == root or p.is_relative_to(root):
        raise ValueError("Use a new output outside the immutable candidate package")
    return p


def future_deadline(value):
    if not value:
        raise ValueError("An explicit future --deadline-utc is required")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError("--deadline-utc must include an explicit UTC offset")
    if stamp <= datetime.now(timezone.utc):
        raise ValueError("--deadline-utc has expired")


def root_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                   help="Candidate package root; defaults to the package containing code/scripts")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("list", help="List exactly 33 main and 9 independent training recipes")
    plan = sub.add_parser("plan", help="Show one future training command; never execute")
    plan.add_argument("--method", required=True, choices=MAIN_METHODS + tuple(EXTERNAL))
    plan.add_argument("--seed", required=True, type=int, choices=SEEDS)
    plan.add_argument("--output", required=True)
    plan.add_argument("--manifest")
    plan.add_argument("--deadline-utc")
    plan.add_argument("--authorization")
    plan.add_argument("--authorization-text")
    plan.add_argument("--acknowledge-new-run", action="store_true")
    plan.add_argument("--validation-cache", type=Path)
    plan.add_argument("--resume", action="store_true")
    train = sub.add_parser("train", help="Plan or explicitly start one method/seed full recipe")
    train.add_argument("--method", required=True, choices=MAIN_METHODS + tuple(EXTERNAL))
    train.add_argument("--seed", required=True, type=int, choices=SEEDS)
    train.add_argument("--output", required=True)
    train.add_argument("--manifest", help="New portable prepared manifest for independent method")
    train.add_argument("--deadline-utc")
    train.add_argument("--authorization", help="Real fresh authorization; never historical evidence")
    train.add_argument("--authorization-text", help="Actual operator authorization text for a new Lee run")
    train.add_argument("--acknowledge-new-run", action="store_true")
    train.add_argument("--validation-cache", type=Path)
    train.add_argument("--resume", action="store_true", help="Resume the same new Lee run ID; never import historical authorization")
    train.add_argument("--execute", action="store_true")
    gen = sub.add_parser("generate", help="Plan or generate and verify one retained training epoch")
    gen.add_argument("--method", required=True, choices=MAIN_METHODS)
    gen.add_argument("--seed", type=int, required=True, choices=SEEDS)
    gen.add_argument("--epoch", type=int, required=True, choices=range(1, 41))
    gen.add_argument("--output", required=True)
    gen.add_argument("--execute", action="store_true")
    prep = sub.add_parser("prepare", help="Bind independent recipe to copied local inputs")
    prep.add_argument("--method", required=True, choices=tuple(EXTERNAL))
    prep.add_argument("--reference-dir", type=Path)
    prep.add_argument("--output", required=True)
    prep.add_argument("--validation-cache", type=Path,
                      help="Real external working cache produced by verify-validation")
    prep.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8, 16, 32))
    prep.add_argument("--execute", action="store_true")
    prep.add_argument("--acknowledge-new-run", action="store_true")
    prep.add_argument("--authorization-text")
    ev = sub.add_parser("evaluate", help="Plan or evaluate full retained main routes / independent method")
    ev.add_argument("--method", choices=("main",) + tuple(EXTERNAL), default="main")
    ev.add_argument("--output", required=True)
    ev.add_argument("--training-output")
    ev.add_argument("--reference-dir", type=Path)
    ev.add_argument("--deadline-utc")
    ev.add_argument("--authorization")
    ev.add_argument("--manifest", help="Actual new prepared Lee manifest")
    ev.add_argument("--authorization-text")
    ev.add_argument("--acknowledge-new-run", action="store_true")
    ev.add_argument("--parent-archive", type=Path,
                    help="Scientific parent view; defaults to the package's new Lee parent view, never permission")
    ev.add_argument("--forti-evaluation", type=Path, help="Actual completed independent FortiTran evaluation for WiMAE lineage")
    ev.add_argument("--execute", action="store_true")
    pl = sub.add_parser("plot", help="Plan or render saved main summary with original plotter")
    pl.add_argument("--output", required=True, help="Existing fresh experiment output")
    pl.add_argument("--execute", action="store_true")
    ext = sub.add_parser("extract-reference", help="SHA-check and copy the actual retained 38-route public reference to a new directory")
    ext.add_argument("--output", required=True)
    ext.add_argument("--execute", action="store_true")
    validation = sub.add_parser("verify-validation", help="Copy cache outside package and perform original CUDA reconstruction verifier")
    validation.add_argument("--output", required=True, help="New working directory outside package")
    validation.add_argument("--execute", action="store_true")
    mechanism = sub.add_parser("mechanism", help="Plan scalar support-mechanism replay of five small saved inputs")
    mechanism.add_argument("--output", required=True)
    mechanism.add_argument("--render-tables", action="store_true")
    mechanism.add_argument("--execute", action="store_true")
    return p


def recipes(root):
    for method in MAIN_METHODS:
        for seed in SEEDS:
            rel = f"code/config/training/main/{method}_seed{seed}.json"
            yield {"method": method, "seed": seed, "config": rel,
                   "config_present": local(root, rel).is_file(), "style": "main_exact_shared_epochs"}
    for method, spec in EXTERNAL.items():
        stem = "lee_meta" if method == "lee_meta" else method
        suffix = ".json" if method == "lee_meta" else ".yaml"
        for seed in SEEDS:
            rel = spec["cwd"] + f"/recorded_configuration/training/configs/{stem}_seed{seed}{suffix}"
            recorded = rel
            if method == "lee_meta":
                rel = f"code/configs/portable/lee_seed{seed}.json"
            yield {"method": method, "seed": seed, "config": rel,
                   "config_present": local(root, rel).is_file(),
                   "recorded_config": recorded,
                   "style": "new_Lee_orchestration_preserved_science" if method == "lee_meta" else "original_independent_recipe",
                   "historical_gate_retained": method == "lee_meta"}


def command_for(root, args):
    cmd = args.command
    main_dir = local(root, "code/training/main")
    main_worker = local(root, "code/scripts/tmlcn_main.py")
    py = [sys.executable, "-B"]
    if cmd == "plan":
        args.command = "train"
        return command_for(root, args)
    if cmd in ("train", "generate") and args.method in MAIN_METHODS:
        argv = py + [str(main_worker), "--root", str(root), "--stage", cmd,
                     "--method", args.method, "--seed", str(args.seed), "--output", args.output]
        if cmd == "generate":
            argv += ["--epoch", str(args.epoch)]
        if getattr(args, "validation_cache", None):
            argv += ["--validation-cache", str(args.validation_cache.resolve())]
        return argv, main_dir, []
    if cmd == "evaluate" and args.method == "main":
        return py + [str(main_worker), "--root", str(root), "--stage", "evaluate",
                     "--output", args.output], main_dir, []
    if cmd == "plot":
        return py + [str(local(root, "code/scripts/tmlcn_plot.py")), "--output", args.output], main_dir, []
    if cmd == "mechanism":
        producer = local(root, "code/frozen_base/analysis/support_diagnostics_20261007/reproduce_mechanism.py")
        argv = py + [str(producer), "--data-root", str(local(root, "data/results/base_analysis/support_diagnostics_20261007")),
                     "--output-dir", args.output, "--skip-plot" if args.render_tables else "--verify-only"]
        return argv, producer.parent, []
    method = args.method
    spec = EXTERNAL[method]
    cwd = local(root, spec["cwd"])
    project = local(root, "code/training/runtime/original_project")
    support = local(root, "code/training/runtime/verification_support")
    paths = [cwd, support, local(root, "code/training/portable")]
    if getattr(args, "reference_dir", None) is None:
        args.reference_dir = local(root, REFERENCE_REL)
    if method == "wimae_author":
        paths += [local(root, EXTERNAL["fortitran_budget_matched"]["cwd"])]
    if cmd == "prepare":
        mb = args.microbatch
        if method == "lee_meta":
            if mb is not None:
                raise ValueError("New Lee prepare retains recorded scientific configurations; microbatch is not overridden")
            argv = py + [str(local(root, "code/scripts/lee_portable.py")), "prepare", "--project", str(project),
                         "--verification-root", str(support), "--reference-results", str(args.reference_dir.resolve()),
                         "--configs-dir", str(local(root, "code/configs/portable")),
                         "--validation-cache", str(args.validation_cache.resolve() if args.validation_cache else local(root, VALIDATION_REL)), "--output", args.output]
            if args.acknowledge_new_run:
                argv += ["--acknowledge-new-run"]
            if args.authorization_text:
                argv += ["--authorization-text", args.authorization_text]
            return argv, cwd, paths
        argv = py + ["-m", spec["prepare"], "--project", str(project),
                     "--reference-results", str(args.reference_dir.resolve()),
                     "--validation-cache", str(args.validation_cache.resolve() if args.validation_cache else local(root, VALIDATION_REL)), "--output", args.output]
        if method == "fortitran_budget_matched":
            argv += ["--microbatch-size", str(mb or 32)]
        elif method == "wimae_author":
            argv += ["--microbatch", str(mb or 1), "--amp", "bf16"]
        return argv, cwd, paths
    if method == "lee_meta":
        # Gate worker supplies the separate reviewable portable entry. Historical
        # signed parent evidence is not repurposed as permission for a new run.
        gate = local(root, "code/scripts/lee_portable.py")
        argv = py + [str(gate), cmd, "--project", str(project), "--verification-root", str(support),
                     "--checkpoint-reader", str(local(root, "code/training/baselines/fortitran_budget_matched/budget_matched_baselines/verify_author_delivery.py"))]
        if cmd == "train":
            if not args.manifest:
                raise ValueError("New Lee training requires --manifest from portable prepare")
            argv += ["--seed", str(args.seed)]
            if args.resume:
                argv += ["--resume"]
        if cmd == "evaluate":
            if args.parent_archive is None:
                args.parent_archive = local(root, "data/reproduction_inputs/lee_parent_scientific_view.zip")
            if not args.training_output or not args.reference_dir or not args.parent_archive or not args.manifest:
                raise ValueError("Lee evaluation requires --manifest, --training-output and actual --parent-archive scientific scores")
            argv += [
                     "--training-output", str(Path(args.training_output).resolve()),
                     "--reference-results", str(args.reference_dir.resolve()),
                     "--parent-archive", str(args.parent_archive.resolve())]
        if getattr(args, "manifest", None):
            argv += ["--manifest", str(Path(args.manifest).resolve())]
        if getattr(args, "deadline_utc", None):
            argv += ["--deadline-utc", args.deadline_utc]
        if getattr(args, "acknowledge_new_run", False):
            argv += ["--acknowledge-new-run"]
        if getattr(args, "authorization_text", None):
            argv += ["--authorization-text", args.authorization_text]
        argv += ["--output", args.output]
        return argv, cwd, paths
    if cmd == "train":
        if not args.manifest:
            raise ValueError("Independent training requires --manifest from a new local prepare")
        if Path(args.manifest).expanduser().resolve().parent != Path(args.output):
            raise ValueError("Independent --output must be the new prepared manifest's parent directory")
        argv = py + ["-m", spec["train"], "--project", str(project),
                     "--manifest", str(Path(args.manifest).resolve()), "--seed", str(args.seed)]
        if args.deadline_utc:
            argv += ["--deadline-utc", args.deadline_utc]
        return argv, cwd, paths
    if cmd == "evaluate":
        if not args.training_output or not args.reference_dir:
            raise ValueError("Independent evaluation needs --training-output and --reference-dir")
        mode = "fortitran" if method == "fortitran_budget_matched" else "wimae"
        argv = py + [str(local(root, "code/scripts/tmlcn_external.py")), mode, "--project", str(support),
                     "--reference-results", str(args.reference_dir.resolve()),
                     "--training-output", str(Path(args.training_output).resolve()), "--output", args.output]
        if method == "fortitran_budget_matched":
            argv += ["--suite", "manuscript", "--bootstrap", "2000"]
        else:
            if not args.forti_evaluation:
                raise ValueError("WiMAE evaluation requires actual --forti-evaluation scientific input")
            if args.deadline_utc:
                argv += ["--deadline-utc", args.deadline_utc]
            argv += ["--inference-microbatch", "1", "--bootstrap", "2000",
                     "--forti-evaluation", str(args.forti_evaluation.resolve()),
                     "--forti-code-root", str(local(root, EXTERNAL["fortitran_budget_matched"]["cwd"]))]
        return argv, cwd, paths
    raise ValueError("Unsupported command")


def extract_reference(root, destination):
    sys.path.insert(0, str(local(root, "code/training/portable")))
    from dopnet_public_reference.reader import copy_reference
    print(json.dumps(copy_reference(local(root, REFERENCE_REL), destination)))


def verify_validation(root, destination):
    source = local(root, VALIDATION_REL)
    if source.stat().st_size != 4682930065 or file_hash(source) != VALIDATION_SHA:
        raise ValueError("Immutable validation source bytes differ")
    if destination.exists():
        raise ValueError("Validation verification requires a new working directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < source.stat().st_size + 600 * 1024 * 1024:
        raise ValueError("Insufficient space for independent working cache and genuine sidecar")
    destination.mkdir()
    working = destination / source.name
    shutil.copyfile(source, working)
    if file_hash(working) != VALIDATION_SHA:
        raise ValueError("Working validation copy SHA differs")
    code = local(root, "code/training/main")
    result = subprocess.call([sys.executable, "-B", str(code / "verify_historical_validation_reconstruction.py"),
                              "--cache", str(working), "--sha256", VALIDATION_SHA, "--batch-size", "8"], cwd=code)
    if result:
        raise RuntimeError("Original validation reconstruction failed; no success is claimed")
    print(json.dumps({"working_validation_cache": str(working), "source_sha256": VALIDATION_SHA,
                      "original_verifier_exit_code": result}))


def main(argv=None):
    parser = root_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    root = args.root.expanduser().resolve()
    if args.command == "list":
        print(json.dumps({"main_routes": 33, "independent_routes": 9, "seeds": list(SEEDS),
                          "recipes": list(recipes(root))}, indent=2))
        return 0
    original_command = args.command
    args.output = str(output_path(root, args.output))
    execute = bool(getattr(args, "execute", False))
    if args.command == "extract-reference":
        if execute:
            extract_reference(root, Path(args.output))
        else:
            print(json.dumps({"execute": False, "source": REFERENCE_REL,
                              "identity_manifest": REFERENCE_REL + "/REFERENCE_VIEW.json",
                              "projected_route_count": 38, "source_route_count": 53,
                              "destination": args.output, "action": "sha_bound_public_reference_copy"}, indent=2))
        return 0
    if args.command == "verify-validation":
        if execute:
            if os.name != "posix":
                raise ValueError("Future validation reconstruction requires a compatible Linux CUDA host")
            verify_validation(root, Path(args.output))
        else:
            print(json.dumps({"execute": False, "source": VALIDATION_REL, "sha256": VALIDATION_SHA,
                              "destination": args.output, "action": "working_copy_and_original_cuda_verifier",
                              "source_bytes": 4682930065, "generated_attestation_here": False}, indent=2))
        return 0
    command, cwd, paths = command_for(root, args)
    plan = {"execute": execute, "requested_action": original_command, "command": command,
            "cwd": str(cwd), "pythonpath": [str(p) for p in paths],
            "full_training_verified_here": False, "original_algorithm_files_modified": False,
            "immutable_reference": REFERENCE_REL, "immutable_validation": VALIDATION_REL,
            "validation_sha256": VALIDATION_SHA}
    if not execute:
        print(json.dumps(plan, indent=2))
        return 0
    if getattr(args, "method", None) == "lee_meta":
        if not getattr(args, "acknowledge_new_run", False) or not getattr(args, "authorization_text", None):
            raise ValueError("New Lee execution requires real --acknowledge-new-run and --authorization-text; historical controls are not reused")
    if getattr(args, "method", None) == "wimae_author" and args.command in ("train", "evaluate"):
        future_deadline(args.deadline_utc)
    env = os.environ.copy()
    if paths:
        env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    return subprocess.call(command, cwd=cwd, env=env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print("TMLCN_ENTRY_ERROR: " + str(error), file=sys.stderr)
        raise SystemExit(2)
