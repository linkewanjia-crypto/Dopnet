"""Independent source-guided FortiTran engine at the fixed DOP-Net budget.

No original training or verification source is patched. No test observations or
historical validation frame is ever used for an optimizer update.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

if __package__ in (None, ""):
    _package_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _package_dir]
    sys.path.insert(0, str(_package_dir.parent))

from budget_matched_baselines.recipe import (
    EarlyStopping, RECIPE, CLAIM, MAX_EPOCHS, TERMINATION, TRACKER_PATIENCE, VALIDATION_SHA256, digest, file_hash,
    validate_manifest, window_size, write_json,
)


def _utc_now():
    return datetime.now(timezone.utc)


def _deadline(value):
    if value is None:
        return None
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("--deadline-utc must include a UTC offset or Z")
    return result.astimezone(timezone.utc)


def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(42, 43, 44))
    parser.add_argument("--resume", choices=("auto", "none"), default="auto")
    parser.add_argument("--deadline-utc", help="Global queue deadline; incomplete runs remain incomplete")
    parser.add_argument("--probe-only", action="store_true",
                        help="Synthetic-shape compute/memory probe; no dataset access or checkpoints")
    parser.add_argument("--probe-seconds", type=float, default=90.0)
    parser.add_argument("--probe-output", default=None)
    return parser


def _load_runtime(project):
    project = Path(project).resolve()
    sys.path.insert(0, str(project))
    # Configure the project's allocator before importing PyTorch.
    allocator = importlib.import_module("utils.cuda_memory")
    allocator.configure_cuda_allocator()
    torch = importlib.import_module("torch")
    trainer = importlib.import_module("train")
    if Path(trainer.__file__).resolve().parent != project:
        raise RuntimeError("Python imported train.py from a different source tree")
    receiver = importlib.import_module("models.receiver")
    if Path(receiver.__file__).resolve().parent.parent != project:
        raise RuntimeError("Python imported model code from a different source tree")
    metrics = importlib.import_module("utils.metrics")
    checkpoints = importlib.import_module("utils.checkpoints")
    return torch, trainer, receiver, metrics, checkpoints


def _seed_runtime(torch, seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _set_process_cap(torch, device, env_name):
    raw = os.environ.get(env_name)
    if raw is None:
        return None
    cap = float(raw)
    total = int(torch.cuda.get_device_properties(device).total_memory)
    fraction = cap * 2**30 / total
    if not math.isfinite(cap) or not 0 < fraction <= 1:
        raise ValueError(f"Invalid {env_name}: {raw}")
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    return cap


def _model(torch, trainer, receiver, cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("Author-recipe model computation is permitted only on the CUDA server")
    device = torch.device("cuda:0")
    trainer.setup_backend(cfg, device)
    model = receiver.SharedUserChannelEstimator(cfg).to(device)
    if model.estimator_name != "fortitran":
        raise RuntimeError("Only the declared FortiTran adapter is allowed")
    return model, device


def _optimizer(torch, model, cfg):
    tr = cfg["train"]
    return torch.optim.Adam(model.parameters(), lr=tr["lr"],
                            betas=tuple(tr["betas"]), eps=tr["eps"],
                            weight_decay=tr["weight_decay"], amsgrad=False,
                            foreach=True)


def _mse(torch, receiver, model, batch, cfg):
    # FortiTran consumes only the public input. It cannot see target_h, ADD
    # labels, true channel parameters, or any historical validation labels.
    prediction = receiver.select_data_csi(model(batch["input"]), cfg)
    target = receiver.select_data_csi(batch["target_h"].float(), cfg)
    return torch.nn.functional.mse_loss(prediction.float(), target), prediction, target


def _finite_gradients(torch, model):
    flags = [torch.isfinite(parameter.grad).all() for parameter in model.parameters()
             if parameter.grad is not None]
    if not flags or not bool(torch.stack(flags).all()):
        raise RuntimeError("Nonfinite or missing training gradients")


def _probe(torch, trainer, receiver, cfg, args, report_path):
    seconds = float(args.probe_seconds)
    if not math.isfinite(seconds) or not 1 <= seconds <= 300:
        raise ValueError("--probe-seconds must be between 1 and 300")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA server required for the synthetic probe")
    device = torch.device("cuda:0")
    cap = _set_process_cap(torch, device, "DOPNET_PROBE_GPU_CAP_GIB")
    _seed_runtime(torch, 970000001 + args.seed)
    model, device = _model(torch, trainer, receiver, cfg)
    optimizer = _optimizer(torch, model, cfg)
    model.train()
    system = cfg["system"]
    micro = int(cfg["train"]["gradient_microbatch_size"])
    effective = int(cfg["train"]["batch_size"])
    shape = (micro, int(system["n_users"]), 5,
             int(system["frame_num_symbols"]), int(system["n_ant"]),
             int(system["n_subcarriers"]))
    generator = torch.Generator(device=device).manual_seed(980000001 + args.seed)
    # These tensors have no connection to simulation, test cells or validation.
    inputs = torch.randn(shape, device=device, generator=generator)
    inputs[:, :, 2].zero_()
    pilot_times = list(range(0, shape[3], int(system["pilot_symbol_period"])))
    for user in range(shape[1]):
        inputs[:, user, 2, pilot_times, :, user::4] = 1.0
    target = torch.randn((micro, shape[1], 2, *shape[3:]),
                         device=device, generator=generator)
    batch = {"input": inputs, "target_h": target}
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total = torch.cuda.mem_get_info(device)
    started = time.perf_counter()
    iterations = optimizer_steps = accumulated = 0
    optimizer.zero_grad(set_to_none=True)
    while time.perf_counter() - started < seconds:
        loss, prediction, truth = _mse(torch, receiver, model, batch, cfg)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite synthetic probe loss")
        (loss * (micro / effective)).backward()
        accumulated += micro
        iterations += 1
        if accumulated >= effective:
            _finite_gradients(torch, model)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            accumulated = 0
        del loss, prediction, truth
        # Synchronization makes the timed interval a completed-work measurement.
        torch.cuda.synchronize(device)
    # The probe is not a training run. Never persist its random weights.
    elapsed = time.perf_counter() - started
    if optimizer_steps < 1:
        raise RuntimeError("Synthetic probe interval did not complete an optimizer step")
    report = {
        "schema": "fortitran_concurrent_compute_probe_v1", "status": "passed",
        "success": True, "scope": "synthetic_shape_only",
        "validation_cache_opened": False, "real_waveforms_generated": False,
        "checkpoint_saved": False, "scientific_score_generated": False,
        "model_training_only_probe": True,
        "waveform_generation_concurrency_certified": False,
        "seed": args.seed, "seconds": elapsed, "microbatch_frames": micro,
        "effective_batch_frames": effective, "completed_microbatches": iterations,
        "completed_optimizer_steps": optimizer_steps,
        "optimizer_steps": optimizer_steps,
        "synthetic_frames_per_second": iterations * micro / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "gpu_cap_gib": cap, "cuda_memory_cap_enforced": cap is not None,
        "free_before_gib": free_before / 2**30,
        "device_total_gib": total / 2**30,
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "requires_separate_generation_memory_gate": True,
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return report


def _load_validation(trainer, cfg, manifest, output):
    fixed = importlib.import_module("fixed_validation")
    path = Path(manifest["validation"]["path"])
    if manifest["validation_cache_sha256"] != VALIDATION_SHA256:
        raise RuntimeError("Unexpected frozen validation SHA-256")
    fixed.assert_historical_identity(cfg)
    before = fixed.historical_file_metadata(path)
    fixed.verify_historical_cache_file(path, VALIDATION_SHA256)
    if fixed.historical_file_metadata(path) != before:
        raise RuntimeError("Historical validation changed during SHA-256 verification")
    proof = dict(before, status="passed", cache_sha256=VALIDATION_SHA256,
                 use="checkpoint_selection_only; never training optimizer input")
    data = fixed.load_historical_core(cfg, path, VALIDATION_SHA256, verified_proof=proof)
    write_json(output / "historical_validation_attestation.json", proof)
    return data


def _rng_state(torch):
    import numpy as np
    return {"rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all()}


def _restore_rng(torch, state):
    import numpy as np
    random.setstate(state["rng_python"])
    np.random.set_state(state["rng_numpy"])
    torch.set_rng_state(state["rng_torch"])
    torch.cuda.set_rng_state_all(state["rng_cuda"])


def _history(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_state(checkpoints, path, state, manifest):
    checkpoints.save_checkpoint(state, path)
    write_json(path.with_suffix(path.suffix + ".sha256.json"), {
        "sha256": file_hash(path), "bytes": path.stat().st_size,
        "recipe_signature": manifest["signature"], "epoch": state["epoch"],
    })


def _load_state(checkpoints, path, manifest, cfg):
    sidecar = json.loads(path.with_suffix(path.suffix + ".sha256.json").read_text(encoding="utf-8"))
    if (sidecar["recipe_signature"] != manifest["signature"]
            or sidecar["sha256"] != file_hash(path)
            or sidecar["bytes"] != path.stat().st_size):
        raise RuntimeError("Checkpoint SHA-256 or recipe signature differs")
    state = checkpoints.load_checkpoint(path, "cpu")
    if (state["cfg"] != cfg or state["recipe_signature"] != manifest["signature"]
            or state["selection_metric"] != "val_mse"
            or state["validation_cache_sha256"] != VALIDATION_SHA256
            or sidecar["epoch"] != state["epoch"]
            or state.get("recipe_claim") != CLAIM
            or state.get("source_files_sha256") != manifest["source_files_sha256"]
            or state.get("implementation_sha256") != manifest["implementation_sha256"]
            or state.get("termination") != TERMINATION
            or state.get("early_stopping", {}).get("patience") != TRACKER_PATIENCE
            or not 1 <= state["epoch"] <= MAX_EPOCHS
            or state.get("training_complete") is not (state["epoch"] == MAX_EPOCHS)):
        raise RuntimeError("Checkpoint training, source or validation contract differs")
    if "prefix_provenance" in state:
        reference = state["prefix_provenance"]
        output = path.resolve().parents[2]
        proof_path = (output / reference["relative_path"]).resolve()
        proof_path.relative_to(output)
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        if (proof.get("signature") != digest({key: value for key, value in proof.items() if key != "signature"})
                or proof["signature"] != reference["signature"]
                or proof["child_manifest_signature"] != manifest["signature"]
                or reference["parent_checkpoint_sha256"] not in proof["parent_checkpoint_files_sha256"].values()):
            raise RuntimeError("Migrated prefix is not bound to its original signed provenance")
    return state


class _DeadlineReached(RuntimeError):
    pass


def _check_deadline(deadline):
    if deadline is not None and _utc_now() >= deadline:
        raise _DeadlineReached("Global deadline reached; discard uncommitted epoch")


def _write_result(checkpoints, folder, manifest, cfg, args, deadline, status,
                  reason, completed, best_val_mse):
    if status == "complete" and completed != MAX_EPOCHS:
        raise RuntimeError("Only exactly40 committed epochs complete the matched budget")
    result = {
        "status": status, "training_complete": status == "complete", "reason": reason,
        "recipe_signature": manifest["signature"], "seed": args.seed,
        "completed_epoch": completed, "max_epochs": MAX_EPOCHS,
        "termination": TERMINATION, "early_stopping_enabled": False,
        "best_val_mse": best_val_mse, "selection_metric": "val_mse",
        "validation_cache_sha256": VALIDATION_SHA256,
        "effective_batch_frames": 512,
        "gradient_microbatch_frames": cfg["train"]["gradient_microbatch_size"],
        "claim": CLAIM, "exact_original_dataset_reproduction": False,
        "deadline_utc": deadline.isoformat(), "finished_utc": _utc_now().isoformat(),
    }
    if (folder / "model.pt").is_file():
        selected = _load_state(checkpoints, folder / "model.pt", manifest, cfg)
        result["selected_checkpoint_sha256"] = file_hash(folder / "model.pt")
        result["model_sha256"] = result["selected_checkpoint_sha256"]
        result["last_checkpoint_sha256"] = file_hash(folder / "last.pt")
        result["selected_epoch"] = selected["epoch"]
    write_json(folder / "complete.json", result)
    write_json(folder / "progress.json", result)
    print(f"AUTHOR_TRAINING_{status.upper()} {folder.name} epoch={completed} reason={reason}", flush=True)
    return result


def _train_epoch(torch, receiver, model, optimizer, generator, cfg, epoch, deadline):
    total = int(cfg["train"]["train_samples"])
    effective = int(cfg["train"]["batch_size"])
    micro = int(cfg["train"]["gradient_microbatch_size"])
    if effective != 512:
        raise RuntimeError("Author-recipe effective batch must remain 512 frames")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    seen = in_window = optimizer_steps = 0
    window = window_size(total, seen, effective)
    weighted_loss = 0.0
    started = time.perf_counter()
    # The generator sees its archived batch of 32. Changing optimizer grouping
    # never changes generation buffers, microchunks, simulator seeds or draws.
    generation_batch = int(cfg["data_pipeline"]["generation_batch_size"])
    for generated in generator.iter_batches(total, epoch, generation_batch):
        _check_deadline(deadline)
        count = int(generated["input"].shape[0])
        for start in range(0, count, micro):
            stop = min(count, start + micro)
            cursor = start
            while cursor < stop:
                _check_deadline(deadline)
                take = min(stop - cursor, window - in_window)
                batch = {"input": generated["input"][cursor:cursor + take],
                         "target_h": generated["target_h"][cursor:cursor + take]}
                loss, prediction, truth = _mse(torch, receiver, model, batch, cfg)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite real training loss")
                # Each optimizer window computes a frame-weighted mean. The
                # final 224-frame window is divided by 224, never by 512.
                (loss * (take / window)).backward()
                weighted_loss += float(loss.detach()) * take
                seen += take
                in_window += take
                cursor += take
                del loss, prediction, truth, batch
                if in_window == window:
                    _finite_gradients(torch, model)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    in_window = 0
                    if seen < total:
                        window = window_size(total, seen, effective)
                    print(f"author epoch={epoch} frames={seen}/{total} optimizer_steps={optimizer_steps}",
                          flush=True)
    torch.cuda.synchronize()
    if seen != total or in_window != 0 or optimizer_steps != math.ceil(total / effective):
        raise RuntimeError("Optimizer accumulation did not cover exactly one training epoch")
    return {"train_loss": weighted_loss / total, "train_frames": seen,
            "optimizer_steps": optimizer_steps,
            "train_seconds": time.perf_counter() - started}


def _validate(torch, receiver, metrics, model, data, cfg, device, deadline):
    import numpy as np
    model.eval()
    n = int(cfg["train"]["val_samples"])
    micro = int(cfg["train"]["validation_microbatch_size"])
    losses = 0.0
    nmse = []
    with torch.no_grad():
        for start in range(0, n, micro):
            _check_deadline(deadline)
            stop = min(n, start + micro)
            batch = {"input": data["input"][start:stop].to(device),
                     "target_h": data["target_h"][start:stop].to(device)}
            loss, prediction, truth = _mse(torch, receiver, model, batch, cfg)
            losses += float(loss) * (stop - start)
            nmse.extend(metrics.nmse_per_sample(prediction, truth).detach().cpu().numpy().tolist())
            del batch, loss, prediction, truth
    torch.cuda.synchronize(device)
    result = {"loss": losses / n, "nmse_db": float(metrics.nmse_summary_db(np.asarray(nmse))[0])}
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("Nonfinite historical-validation metrics")
    return result


def _train(torch, trainer, receiver, metrics, checkpoints, cfg, manifest, name, args, output):
    deadline = _deadline(args.deadline_utc)
    if deadline is None:
        raise ValueError("Actual training requires a global --deadline-utc; the controller owns the shared time budget")
    folder = output / "checkpoints" / name
    folder.mkdir(parents=True, exist_ok=True)
    complete_path = folder / "complete.json"
    if complete_path.exists():
        old = json.loads(complete_path.read_text(encoding="utf-8"))
        if old.get("recipe_signature") != manifest["signature"]:
            raise RuntimeError("Existing run belongs to a different author recipe")
        if old.get("status") == "complete":
            # Recheck the final checkpoint instead of silently trusting a flag.
            _load_state(checkpoints, folder / "model.pt", manifest, cfg)
            recorded = _load_state(checkpoints, folder / "last.pt", manifest, cfg)
            terminal = (recorded["epoch"] == MAX_EPOCHS and recorded.get("training_complete") is True
                        and recorded.get("termination") == TERMINATION)
            if (not terminal or old.get("completed_epoch") != recorded["epoch"]
                    or old.get("model_sha256") != file_hash(folder / "model.pt")
                    or old.get("last_checkpoint_sha256") != file_hash(folder / "last.pt")):
                raise RuntimeError("Completion marker is not supported by its terminal checkpoints")
            print(f"AUTHOR_TRAINING_ALREADY_COMPLETE {name}", flush=True)
            return old
    if _utc_now() >= deadline:
        recorded = (_load_state(checkpoints, folder / "last.pt", manifest, cfg)
                    if (folder / "last.pt").is_file() else None)
        terminal = bool(recorded and recorded["epoch"] == MAX_EPOCHS
                        and recorded.get("training_complete") is True and recorded.get("termination") == TERMINATION)
        return _write_result(
            checkpoints, folder, manifest, cfg, args, deadline,
            "complete" if terminal else "incomplete_time_limit",
            "authenticated_terminal_checkpoint" if terminal else "global_deadline_before_training_start",
            int(recorded["epoch"]) if recorded else 0,
            recorded["best_val_loss"] if recorded else None)
    _set_process_cap(torch, torch.device("cuda:0"), "DOPNET_TRAIN_GPU_CAP_GIB")
    _seed_runtime(torch, args.seed)
    validation = _load_validation(trainer, cfg, manifest, output)
    model, device = _model(torch, trainer, receiver, cfg)
    optimizer = _optimizer(torch, model, cfg)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    generator = trainer.BatchChannelGenerator(cfg, device=device, seed=args.seed + 10001)
    stopper = EarlyStopping(patience=TRACKER_PATIENCE)
    history = []
    start_epoch = 1
    best_nmse = None
    prefix_provenance = None
    last = folder / "last.pt"
    if last.exists():
        if args.resume == "none":
            raise RuntimeError("Existing checkpoints cannot be overwritten by a fresh run")
        state = _load_state(checkpoints, last, manifest, cfg)
        prefix_provenance = state.get("prefix_provenance")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        stopper = EarlyStopping(**state["early_stopping"])
        history = state["history"]
        if (len(history) != state["epoch"] or any(int(row["epoch"]) != index
                or int(row["train_frames"]) != 12000 or int(row["optimizer_steps"]) != 24
                for index, row in enumerate(history, 1))):
            raise RuntimeError("Resume state lacks consecutive complete training epochs")
        start_epoch = int(state["epoch"]) + 1
        best_nmse = state["minimum_observed_val_nmse_db"]
        _restore_rng(torch, state)
        print(f"AUTHOR_RESUMED {name} epoch={state['epoch']}", flush=True)
    else:
        if (folder / "model.pt").exists() or (folder / "history.csv").exists():
            raise RuntimeError("Partial run has no authenticated last.pt; use a new run directory")
    torch.cuda.reset_peak_memory_stats(device)
    status, reason = "running", None
    completed = start_epoch - 1
    # A process can exit after last.pt commits the stopping epoch but before it
    # writes complete.json. Never train an extra epoch on recovery.
    terminal_checkpoint = completed == MAX_EPOCHS
    epoch_range = (() if terminal_checkpoint else
                   range(start_epoch, int(cfg["train"]["epochs"]) + 1))
    for epoch in epoch_range:
        if _utc_now() >= deadline:
            status, reason = "incomplete_time_limit", "global_deadline_at_epoch_boundary"
            break
        lr = float(optimizer.param_groups[0]["lr"])
        write_json(folder / "progress.json", {
            "status": "running", "phase": "training", "epoch": epoch,
            "recipe_signature": manifest["signature"], "seed": args.seed,
            "deadline_utc": deadline.isoformat(), "training_complete": False,
        })
        try:
            _check_deadline(deadline)
            train_stats = _train_epoch(torch, receiver, model, optimizer, generator, cfg, epoch, deadline)
            val = _validate(torch, receiver, metrics, model, validation, cfg, device, deadline)
            _check_deadline(deadline)
        except _DeadlineReached:
            status, reason = "incomplete_time_limit", "global_deadline_uncommitted_epoch_discarded"
            print(f"AUTHOR_EPOCH_DISCARDED {name} epoch={epoch}; authenticated last.pt remains the resume point",
                  flush=True)
            break
        improved, should_stop = stopper.update(val["loss"])
        best_nmse = val["nmse_db"] if best_nmse is None else min(best_nmse, val["nmse_db"])
        scheduler.step()
        row = {"epoch": epoch, "lr": lr, "next_lr": float(optimizer.param_groups[0]["lr"]),
               **train_stats, "val_loss": val["loss"], "val_nmse_db": val["nmse_db"],
               "selection_metric": "val_mse", "stale_epochs": stopper.stale,
               "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20}
        history.append(row)
        completed = epoch
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "cfg": cfg, "epoch": epoch,
            "task": "ce", "estimator": "fortitran", "ablation_profile": cfg["model"]["ablation_profile"],
            "selection_metric": "val_mse", "best_val_loss": stopper.best,
            "val": val, "minimum_observed_val_nmse_db": best_nmse,
            "optimizer_name": "adam", "scheduler_name": "exponential",
            "validation_cache_sha256": VALIDATION_SHA256,
            "recipe_signature": manifest["signature"],
            "recipe_claim": CLAIM,
            "source_files_sha256": manifest["source_files_sha256"],
            "implementation_sha256": manifest["implementation_sha256"],
            "early_stopping": stopper.state_dict(), "history": history,
            "effective_batch_frames": 512,
            "physical_gradient_microbatch_frames": cfg["train"]["gradient_microbatch_size"],
            "training_complete": epoch == MAX_EPOCHS,
            "termination": TERMINATION,
            **_rng_state(torch),
        }
        if prefix_provenance is not None:
            state["prefix_provenance"] = prefix_provenance
        if improved:
            _save_state(checkpoints, folder / "model.pt", state, manifest)
        _save_state(checkpoints, last, state, manifest)
        _history(folder / "history.csv", history)
        print(f"budget {name} epoch={epoch}/{MAX_EPOCHS} train_mse={row['train_loss']:.6g} "
              f"val_mse={val['loss']:.6g} val_nmse_db={val['nmse_db']:.4f} "
              f"lr={lr:.6g} stale_epochs={stopper.stale} early_stopping=disabled", flush=True)
        if epoch == MAX_EPOCHS:
            status, reason = "complete", TERMINATION
    if status == "running":
        # Resume can encounter a last checkpoint committed at the stopping
        # epoch before complete.json was written.
        if completed == MAX_EPOCHS:
            status, reason = "complete", "authenticated_terminal_checkpoint"
        else:
            status, reason = "incomplete_time_limit", "no_terminal_condition_reached"
    return _write_result(checkpoints, folder, manifest, cfg, args, deadline,
                         status, reason, completed, stopper.best)


def main():
    args = build_argparser().parse_args()
    output = Path(args.manifest).resolve().parent
    name = f"fortitran_budget_matched_seed{args.seed}"
    report_path = (Path(args.probe_output).resolve() if args.probe_output else
                   output / f"probe_seed{args.seed}.json") if args.probe_only else (
                       output / "checkpoints" / name / "failure.json")
    stage = "source_contract"
    try:
        manifest, cfg, name = validate_manifest(args.manifest, args.project, args.seed)
        stage = "runtime_import"
        torch, trainer, receiver, metrics, checkpoints = _load_runtime(args.project)
        if args.probe_only:
            stage = "synthetic_compute_probe"
            _probe(torch, trainer, receiver, cfg, args, report_path)
        else:
            stage = "author_recipe_training"
            result = _train(torch, trainer, receiver, metrics, checkpoints,
                            cfg, manifest, name, args, output)
            if result["status"] != "complete":
                return 75
        return 0
    except Exception as error:
        failure = {
            "status": "failed", "success": False, "stage": stage,
            "seed": args.seed, "probe_only": bool(args.probe_only),
            "scope": "synthetic_shape_only" if args.probe_only else CLAIM,
            "training_complete": False, "exception": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
            "finished_utc": _utc_now().isoformat(),
        }
        write_json(report_path, failure)
        print(json.dumps(failure, ensure_ascii=False, allow_nan=False), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
