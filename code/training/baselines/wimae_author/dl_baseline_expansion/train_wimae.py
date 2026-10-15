"""CUDA-only, resumable 20-MAE+20-CE training on the frozen DOP-Net stream.

Use python -m dl_baseline_expansion.train_wimae. No original project source is
edited, and no test frames enter training or checkpoint selection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
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

from .prepare_wimae import (VALIDATION_SHA256, digest, file_hash,
                            validate_manifest, write_json)


def utc_now():
    return datetime.now(timezone.utc)


def deadline_from(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Explicit deadline must include its UTC offset")
    return result.astimezone(timezone.utc)


class DeadlineReached(RuntimeError):
    pass


def check_deadline(deadline):
    if utc_now() >= deadline:
        raise DeadlineReached("Global deadline reached; uncommitted epoch is discarded")


@contextmanager
def seed_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "worker.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        old = json.loads(path.read_text())
        try:
            os.kill(int(old["pid"]), 0)
        except ProcessLookupError:
            path.unlink()
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        else:
            raise RuntimeError("A worker owns this seed output; do not launch a duplicate")
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"pid": os.getpid(), "started_utc": utc_now().isoformat()}, stream)
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text()).get("pid") == os.getpid():
            path.unlink()


def load_runtime(project):
    project = Path(project).resolve()
    sys.path.insert(0, str(project))
    importlib.import_module("utils.cuda_memory").configure_cuda_allocator()
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this engine on the CUDA server, never the local laptop")
    trainer = importlib.import_module("train")
    receiver = importlib.import_module("models.receiver")
    if (Path(trainer.__file__).resolve().parent != project
            or Path(receiver.__file__).resolve().parent.parent != project):
        raise RuntimeError("Wrong archived runtime was imported")
    checkpoints = importlib.import_module("utils.checkpoints")
    metrics = importlib.import_module("utils.metrics")
    model_module = importlib.import_module("dl_baseline_expansion.wimae_author")
    return torch, trainer, receiver, checkpoints, metrics, model_module.WiMAESliceCE


def seed_runtime(torch, seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state(torch, model):
    import numpy as np
    return {"rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
            "rng_private_training_mask": model.mask_rng_state()}


def restore_rng(torch, model, state):
    import numpy as np
    random.setstate(state["rng_python"])
    np.random.set_state(state["rng_numpy"])
    torch.set_rng_state(state["rng_torch"])
    torch.cuda.set_rng_state_all(state["rng_cuda"])
    model.load_mask_rng_state(state["rng_private_training_mask"])


def memory_cap(torch, name):
    value = os.environ.get(name)
    if value is None:
        return None
    cap = float(value)
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = cap * 2**30 / total
    if not math.isfinite(cap) or not 0 < fraction <= 1:
        raise ValueError("Invalid CUDA process-memory cap")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return cap


def optimizer_for(torch, model, cfg, phase):
    spec = cfg["wimae_reproduction"]
    settings = spec["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=settings[phase + "_lr"],
                                 betas=tuple(settings["betas"]), eps=settings["eps"],
                                 weight_decay=settings["weight_decay"], foreach=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=spec["scheduler"]["eta_min"])
    return optimizer, scheduler


def autocast(torch, cfg):
    return torch.amp.autocast("cuda", enabled=cfg["wimae_reproduction"]["amp"] == "bf16",
                              dtype=torch.bfloat16)


def loss_for(torch, receiver, model, batch, cfg, phase):
    if phase == "ssl":
        return model.ssl_loss(batch["target_h"], batch["input"]), None, None
    prediction = receiver.select_data_csi(model(batch["input"]), cfg)
    target = receiver.select_data_csi(batch["target_h"].float(), cfg)
    return torch.nn.functional.mse_loss(prediction.float(), target), prediction, target


def step_optimizer(torch, model, optimizer, cfg):
    flags = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not flags or not bool(torch.stack(flags).all()):
        raise RuntimeError("Nonfinite or absent parameter gradients")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["wimae_reproduction"]["gradient_clip"])
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("Nonfinite gradient norm")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def train_epoch(torch, receiver, model, optimizer, generator, cfg, epoch, phase, deadline):
    spec = cfg["wimae_reproduction"]
    total, effective, micro = 12000, 32, int(spec["gradient_microbatch_frames"])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    seen = in_window = steps = 0
    window = min(effective, total)
    weighted_loss = 0.0
    started = time.perf_counter()
    # Data batching never follows the model's frame microbatch. It stays32/8.
    for generated in generator.iter_batches(total, epoch, 32):
        check_deadline(deadline)
        for start in range(0, int(generated["input"].shape[0]), micro):
            check_deadline(deadline)
            take = min(micro, int(generated["input"].shape[0]) - start, window - in_window)
            batch = {key: generated[key][start:start + take] for key in ("input", "target_h")}
            with autocast(torch, cfg):
                loss, prediction, target = loss_for(torch, receiver, model, batch, cfg, phase)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite training objective")
            (loss * (take / window)).backward()
            weighted_loss += float(loss.detach()) * take
            seen += take
            in_window += take
            del batch, loss, prediction, target
            if in_window == window:
                step_optimizer(torch, model, optimizer, cfg)
                steps += 1
                in_window = 0
                window = min(effective, total - seen) if seen < total else 0
                if steps == 1 or steps % 50 == 0:
                    print(f"wimae phase={phase} epoch={epoch} frames={seen}/12000 updates={steps}", flush=True)
    torch.cuda.synchronize()
    if seen != total or in_window or steps != math.ceil(total / effective):
        raise RuntimeError("A training epoch did not cover exactly12000 frames")
    return {"train_loss": weighted_loss / total, "train_frames": seen,
            "optimizer_steps": steps, "train_seconds": time.perf_counter() - started}


def validate(torch, receiver, metrics, model, validation, cfg, phase, deadline):
    import numpy as np
    model.eval()
    micro = int(cfg["train"]["validation_microbatch_size"])
    total_loss = 0.0
    nmse = []
    private_state = model.mask_rng_state()
    if phase == "ssl":
        model.seed_mask_rng(cfg["wimae_reproduction"]["validation_mask_seed"])
    try:
        with torch.no_grad():
            for start in range(0, 2000, micro):
                check_deadline(deadline)
                stop = min(start + micro, 2000)
                batch = {key: validation[key][start:stop].to("cuda:0") for key in ("input", "target_h")}
                with autocast(torch, cfg):
                    loss, prediction, target = loss_for(torch, receiver, model, batch, cfg, phase)
                total_loss += float(loss) * (stop - start)
                if phase == "ce":
                    nmse.extend(metrics.nmse_per_sample(prediction, target).cpu().numpy().tolist())
                del batch, loss, prediction, target
    finally:
        model.load_mask_rng_state(private_state)
    result = {"loss": total_loss / 2000,
              "nmse_db": float(metrics.nmse_summary_db(np.asarray(nmse))[0]) if nmse else None}
    if not math.isfinite(result["loss"]) or (nmse and not math.isfinite(result["nmse_db"])):
        raise RuntimeError("Nonfinite historical validation metric")
    return result


def load_validation(cfg, manifest, output):
    fixed = importlib.import_module("fixed_validation")
    path = Path(manifest["validation"]["path"])
    if manifest["validation_cache_sha256"] != VALIDATION_SHA256:
        raise RuntimeError("Unexpected historical validation artifact")
    fixed.assert_historical_identity(cfg)
    before = fixed.historical_file_metadata(path)
    fixed.verify_historical_cache_file(path, VALIDATION_SHA256)
    if fixed.historical_file_metadata(path) != before:
        raise RuntimeError("Historical cache changed during verification")
    proof = dict(before, status="passed", cache_sha256=VALIDATION_SHA256,
                 use="phase_checkpoint_selection_only; never_optimizer_input")
    data = fixed.load_historical_core(cfg, path, VALIDATION_SHA256, verified_proof=proof)
    write_json(output / "historical_validation_attestation.json", proof)
    return data


def save_state(checkpoints, path, state, manifest):
    checkpoints.save_checkpoint(state, path)
    write_json(path.with_suffix(".pt.sha256.json"), {"sha256": file_hash(path),
               "bytes": path.stat().st_size, "recipe_signature": manifest["signature"],
               "epoch": state["epoch"], "phase": state["phase"]})


def load_state(checkpoints, path, manifest, cfg):
    record = json.loads(path.with_suffix(".pt.sha256.json").read_text())
    if (record["recipe_signature"] != manifest["signature"]
            or record["sha256"] != file_hash(path) or record["bytes"] != path.stat().st_size):
        raise RuntimeError("Checkpoint sidecar identity/SHA differs")
    state = checkpoints.load_checkpoint(path, "cpu")
    if not 1 <= int(state["epoch"]) <= 40:
        raise RuntimeError("Checkpoint epoch is outside the fixed budget")
    if (state["cfg"] != cfg or state["recipe_signature"] != manifest["signature"]
            or state["validation_cache_sha256"] != VALIDATION_SHA256
            or state.get("source_files_sha256") != manifest.get("source_files_sha256")
            or state.get("implementation_sha256") != manifest.get("implementation_sha256")
            or state.get("selection_metric") != "val_mse"
            or record["epoch"] != state["epoch"] or record["phase"] != state["phase"]):
        raise RuntimeError("Checkpoint training/validation contract differs")
    if state["phase"] != ("ssl" if state["epoch"] <= 20 else "ce"):
        raise RuntimeError("Checkpoint phase/global epoch differs")
    rows = state["history"]
    if ([r["epoch"] for r in rows] != list(range(1, state["epoch"] + 1))
            or any(r["train_frames"] != 12000 or r["optimizer_steps"] != 375
                   or r["phase"] != ("ssl" if r["epoch"] <= 20 else "ce") for r in rows)):
        raise RuntimeError("Committed epoch ledger is incomplete")
    if state["phase"] == "ce":
        parent = state.get("ssl_parent")
        if (not isinstance(parent, dict) or parent.get("global_transition_epoch") != 21
                or parent.get("optimizer_scheduler_reset") is not True
                or parent.get("global_training_RNG_preserved") is not True):
            raise RuntimeError("CE checkpoint lacks its authenticated phase transition")
    return state


def write_history(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def recover_best_pointers(checkpoints, folder, manifest, cfg, last):
    """Recover a crash between committed last and publication of its best copy.

    Last is published before best; an interrupted best write can be repaired
    only from that same authenticated selected epoch. Earlier missing best
    artifacts cannot be reconstructed from another epoch and cause a failure.
    """
    if last is None:
        return
    for phase, filename in (("ssl", "ssl_best.pt"), ("ce", "model.pt")):
        rows = [r for r in last["history"] if r["phase"] == phase]
        if not rows:
            continue
        selected_row = min(rows, key=lambda r: r["val_loss"])
        path = folder / filename
        valid = False
        try:
            selected = load_state(checkpoints, path, manifest, cfg)
            valid = (selected["phase"] == phase and selected["epoch"] == selected_row["epoch"]
                     and selected["val"]["loss"] == selected_row["val_loss"])
        except (OSError, ValueError, RuntimeError, KeyError):
            pass
        if not valid:
            if last["phase"] != phase or last["epoch"] != selected_row["epoch"]:
                raise RuntimeError("Selected earlier checkpoint is absent/damaged; cannot invent its weights")
            save_state(checkpoints, path, last, manifest)
            print(f"WIMAE_RECOVERED_BEST_COPY {filename} epoch={last['epoch']}", flush=True)
    if last["phase"] == "ce" and file_hash(folder / "ssl_best.pt") != last["ssl_parent"]["checkpoint_sha256"]:
        raise RuntimeError("SSL parent bytes differ from the recorded CE initialization")
    # History is a derived view of the authenticated epoch ledger.
    write_history(folder / "history.csv", last["history"])


def finish(checkpoints, folder, manifest, cfg, deadline, reason):
    last = load_state(checkpoints, folder / "last.pt", manifest, cfg) if (folder / "last.pt").exists() else None
    recover_best_pointers(checkpoints, folder, manifest, cfg, last)
    complete = bool(last and last["epoch"] == 40 and len(last["history"]) == 40)
    result = {"status": "complete" if complete else "incomplete_time_limit",
              "training_complete": complete, "recipe_signature": manifest["signature"],
              "seed": cfg["seed"], "completed_epoch": last["epoch"] if last else 0,
              "total_epochs": 40, "ssl_epochs": 20, "ce_epochs": 20,
              "max_epochs": 40, "effective_batch_frames": 32,
              "gradient_microbatch_frames": cfg["wimae_reproduction"]["gradient_microbatch_frames"],
              "training_frame_exposures": last["epoch"] * 12000 if last else 0,
              "reason": reason, "deadline_utc": deadline.isoformat(),
              "finished_utc": utc_now().isoformat(), "selection_metric": "val_mse",
              "claim": "task_adapted_equal_frame_budget", "original_CE_recipe_reproduction": False,
              "validation_cache_sha256": VALIDATION_SHA256}
    if last:
        result["last_checkpoint_sha256"] = file_hash(folder / "last.pt")
    if (folder / "model.pt").exists():
        selected = load_state(checkpoints, folder / "model.pt", manifest, cfg)
        if selected["phase"] != "ce":
            raise RuntimeError("Final CE model selected from the SSL phase")
        if last is None or selected["epoch"] > last["epoch"]:
            raise RuntimeError("Best CE checkpoint is ahead of committed last")
        ce_rows = [r for r in last["history"] if r["phase"] == "ce"]
        best_row = min(ce_rows, key=lambda r: r["val_loss"])
        if (selected["epoch"] != best_row["epoch"] or selected["val"]["loss"] != best_row["val_loss"]):
            raise RuntimeError("Best CE checkpoint does not match validation ledger")
        result.update(selected_checkpoint_sha256=file_hash(folder / "model.pt"),
                      model_sha256=file_hash(folder / "model.pt"),
                      selected_epoch=selected["epoch"], best_val_mse=selected["val"]["loss"],
                      ssl_parent=last["ssl_parent"])
    elif complete:
        raise RuntimeError("All epochs were recorded but final CE selection is absent")
    marker_path = folder / "complete.json"
    if complete and marker_path.exists():
        previous = json.loads(marker_path.read_text(encoding="utf-8"))
        if previous.get("status") == "complete":
            # The logical terminal state is reauthenticated above. Preserve
            # its first completion bytes, because evaluation binds this file.
            identity_keys = set(result) - {"reason", "finished_utc"}
            if any(previous.get(key) != result[key] for key in identity_keys):
                raise RuntimeError("Existing completion marker differs from authenticated terminal state")
            return previous
    write_json(marker_path, result)
    write_json(folder / "progress.json", result)
    print("WIMAE_TRAINING_" + result["status"].upper() + " " + folder.name, flush=True)
    return result


def train(runtime, cfg, manifest, args):
    torch, trainer, receiver, checkpoints, metrics, Model = runtime
    deadline = deadline_from(args.deadline_utc)
    output = Path(args.manifest).resolve().parent
    folder = output / "checkpoints" / f"wimae_author_seed{args.seed}"
    with seed_lock(folder):
        state = load_state(checkpoints, folder / "last.pt", manifest, cfg) if (folder / "last.pt").exists() else None
        recover_best_pointers(checkpoints, folder, manifest, cfg, state)
        if state and state["epoch"] == 40:
            return finish(checkpoints, folder, manifest, cfg, deadline, "authenticated_all40_committed")
        if args.resume == "none" and (state or (folder / "ssl_best.pt").exists() or (folder / "model.pt").exists()):
            raise RuntimeError("Use a new output for a fresh initialization")
        if state is None and ((folder / "ssl_best.pt").exists() or (folder / "model.pt").exists()):
            raise RuntimeError("Partial artifacts without authenticated last; do not silently restart")
        if utc_now() >= deadline:
            return finish(checkpoints, folder, manifest, cfg, deadline, "deadline_before_next_epoch")
        memory_cap(torch, "DOPNET_TRAIN_GPU_CAP_GIB")
        seed_runtime(torch, args.seed)
        trainer.setup_backend(cfg, torch.device("cuda:0"))
        model = Model(cfg, device="cuda:0").to("cuda:0")
        phase = state["phase"] if state else "ssl"
        optimizer, scheduler = optimizer_for(torch, model, cfg, phase)
        history, best_ssl, best_ce, ssl_parent = [], None, None, None
        start = 1
        if state:
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            history = state["history"]
            best_ssl, best_ce, ssl_parent = state["best_ssl_mse"], state["best_ce_mse"], state["ssl_parent"]
            start = state["epoch"] + 1
            if state["environment"]["torch"] != str(torch.__version__):
                raise RuntimeError("PyTorch version changed across exact resume")
        validation = load_validation(cfg, manifest, folder)
        generator = trainer.BatchChannelGenerator(cfg, device="cuda:0", seed=args.seed + 10001)
        if state:
            restore_rng(torch, model, state)
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(start, 41):
            try:
                check_deadline(deadline)
                if epoch == 21:
                    if phase != "ssl" or ssl_parent is not None:
                        raise RuntimeError("Invalid or duplicated phase transition")
                    parent_path = folder / "ssl_best.pt"
                    parent = load_state(checkpoints, parent_path, manifest, cfg)
                    if parent["phase"] != "ssl" or parent["val"]["loss"] != best_ssl:
                        raise RuntimeError("SSL initialization is not its selected validation best")
                    model.load_state_dict(parent["model"], strict=True)
                    phase = "ce"
                    optimizer, scheduler = optimizer_for(torch, model, cfg, phase)
                    ssl_parent = {"checkpoint_sha256": file_hash(parent_path), "epoch": parent["epoch"],
                                  "val_mse": best_ssl, "global_transition_epoch": 21,
                                  "optimizer_scheduler_reset": True, "global_training_RNG_preserved": True}
                    print("WIMAE_PHASE_TRANSITION " + json.dumps(ssl_parent), flush=True)
                expected_phase = "ssl" if epoch <= 20 else "ce"
                if phase != expected_phase:
                    raise RuntimeError("Global epoch/phase mismatch")
                lr = float(optimizer.param_groups[0]["lr"])
                write_json(folder / "progress.json", {"status": "running", "training_complete": False,
                           "phase": phase, "epoch": epoch, "seed": args.seed,
                           "recipe_signature": manifest["signature"], "deadline_utc": deadline.isoformat()})
                stats = train_epoch(torch, receiver, model, optimizer, generator, cfg, epoch, phase, deadline)
                val = validate(torch, receiver, metrics, model, validation, cfg, phase, deadline)
                check_deadline(deadline)
            except DeadlineReached:
                return finish(checkpoints, folder, manifest, cfg, deadline, "uncommitted_epoch_discarded")
            current_best = best_ssl if phase == "ssl" else best_ce
            improved = current_best is None or val["loss"] < current_best
            if improved:
                if phase == "ssl":
                    best_ssl = val["loss"]
                else:
                    best_ce = val["loss"]
            scheduler.step()
            history.append({"epoch": epoch, "phase": phase, "phase_epoch": epoch if phase == "ssl" else epoch - 20,
                            "lr": lr, "next_lr": float(optimizer.param_groups[0]["lr"]),
                            **stats, "val_loss": val["loss"], "val_nmse_db": val["nmse_db"],
                            "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20})
            committed = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "cfg": cfg, "epoch": epoch,
                         "phase": phase, "history": history, "val": val,
                         "best_ssl_mse": best_ssl, "best_ce_mse": best_ce, "ssl_parent": ssl_parent,
                         "selection_metric": "val_mse", "training_complete": epoch == 40,
                         "estimator": "wimae_author_slice_ce", "task": "ssl" if phase == "ssl" else "ce",
                         "validation_cache_sha256": VALIDATION_SHA256,
                         "recipe_signature": manifest["signature"],
                         "source_files_sha256": manifest["source_files_sha256"],
                         "implementation_sha256": manifest["implementation_sha256"],
                         "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                                         "gpu": torch.cuda.get_device_name(0), "amp": cfg["wimae_reproduction"]["amp"]},
                         **rng_state(torch, model)}
            # Commit last before publishing its selected copy. If interrupted,
            # recover_best_pointers can recover exactly this committed state.
            save_state(checkpoints, folder / "last.pt", committed, manifest)
            if improved:
                save_state(checkpoints, folder / ("ssl_best.pt" if phase == "ssl" else "model.pt"), committed, manifest)
            write_history(folder / "history.csv", history)
            print(f"wimae seed={args.seed} phase={phase} epoch={epoch}/40 train_mse={stats['train_loss']:.6g} "
                  f"val_mse={val['loss']:.6g} lr={lr:.6g}", flush=True)
        return finish(checkpoints, folder, manifest, cfg, deadline, "all40_epochs_committed")


def probe(runtime, cfg, manifest, args):
    torch, trainer, receiver, _, _, Model = runtime
    if not 2 <= args.probe_seconds <= 300 or args.probe_microbatch not in (1, 2, 4, 8):
        raise ValueError("Probe needs2..300 seconds/phase and microbatch1/2/4/8")
    deadline = deadline_from(args.deadline_utc)
    cap = memory_cap(torch, "DOPNET_PROBE_GPU_CAP_GIB")
    trainer.setup_backend(cfg, torch.device("cuda:0"))
    free_before, total_before = torch.cuda.mem_get_info()
    results = []
    for phase in ("ssl", "ce"):
        check_deadline(deadline)
        seed_runtime(torch, args.seed + 950000001)
        model = Model(cfg, device="cuda:0").train()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != 554128:
            raise RuntimeError("Actual model parameter count differs from the upstream topology")
        optimizer, _ = optimizer_for(torch, model, cfg, phase)
        micro = args.probe_microbatch
        rng = torch.Generator(device="cuda:0").manual_seed(args.seed + 960000001)
        inputs = torch.randn((micro, 2, 5, 25, 8, 128), device="cuda:0", generator=rng)
        inputs[:, :, 2].zero_()
        for user in range(2):
            inputs[:, user, 2, 0::4, :, user::4] = 1
        target = torch.randn((micro, 2, 2, 25, 8, 128), device="cuda:0", generator=rng)
        batch = {"input": inputs, "target_h": target}
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        iterations = steps = accumulated = 0
        while time.perf_counter() - started < args.probe_seconds:
            check_deadline(deadline)
            with autocast(torch, cfg):
                loss, prediction, truth = loss_for(torch, receiver, model, batch, cfg, phase)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite synthetic probe loss")
            (loss * (micro / 32)).backward()
            accumulated += micro
            iterations += 1
            if accumulated == 32:
                step_optimizer(torch, model, optimizer, cfg)
                steps += 1
                accumulated = 0
            del loss, prediction, truth
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if steps < 1:
            raise RuntimeError("Probe must include a complete optimizer update")
        results.append({"phase": phase, "microbatch_frames": micro,
                        "expanded_channel_slices_per_microbatch": micro * 50,
                        "tokens_per_channel_slice": 128, "effective_batch_frames": 32,
                        "completed_microbatches": iterations, "completed_optimizer_steps": steps,
                        "elapsed_seconds": elapsed, "synthetic_frames_per_second": micro * iterations / elapsed,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
        del model, optimizer, batch, inputs, target
        torch.cuda.empty_cache()
    report = {"status": "passed", "scope": "synthetic_shape_compute_only", "phases": results,
              "recipe_signature": manifest["signature"], "probe_override_microbatch": args.probe_microbatch,
              "implementation_sha256": manifest["implementation_sha256"],
              "amp": cfg["wimae_reproduction"]["amp"], "gpu": torch.cuda.get_device_name(0),
              "torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu_cap_gib": cap,
              "gpu_total_gib": total_before / 2**30,
              "gpu_free_before_probe_gib": free_before / 2**30,
              "cuda_memory_cap_enforced": cap is not None, "real_waveforms_generated": False,
              "validation_cache_opened": False, "checkpoint_saved": False,
              "scientific_score_generated": False, "waveform_generation_concurrency_certified": False,
              "parameters": parameter_count, "parameter_count_source": "server_model_parameter_count"}
    target = Path(args.probe_output) if args.probe_output else Path(args.manifest).parent / f"probe_micro{args.probe_microbatch}.json"
    write_json(target, report)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--resume", choices=("auto", "none"), default="auto")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-seconds", type=float, default=45)
    parser.add_argument("--probe-microbatch", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--probe-output")
    args = parser.parse_args()
    folder = Path(args.manifest).resolve().parent / "checkpoints" / f"wimae_author_seed{args.seed}"
    try:
        manifest, cfg, _ = validate_manifest(args.manifest, args.project, args.seed)
        runtime = load_runtime(args.project)
        if args.probe_only:
            probe(runtime, cfg, manifest, args)
            return 0
        result = train(runtime, cfg, manifest, args)
        return 0 if result["training_complete"] else 75
    except DeadlineReached as error:
        record = {"status": "incomplete_time_limit", "training_complete": False,
                  "scope": "synthetic_compute_probe" if args.probe_only else "training",
                  "seed": args.seed, "error": str(error), "finished_utc": utc_now().isoformat()}
        target = Path(args.probe_output) if args.probe_only and args.probe_output else folder / "incomplete.json"
        write_json(target, record)
        return 75
    except Exception as error:
        record = {"status": "failed", "training_complete": False, "seed": args.seed,
                  "probe_only": args.probe_only, "exception": type(error).__name__,
                  "recipe_signature": manifest["signature"] if "manifest" in locals() else None,
                  "implementation_sha256": manifest["implementation_sha256"] if "manifest" in locals() else None,
                  "probe_override_microbatch": args.probe_microbatch if args.probe_only else None,
                  "error": str(error), "traceback": traceback.format_exc(),
                  "finished_utc": utc_now().isoformat()}
        target = Path(args.probe_output) if args.probe_only and args.probe_output else folder / "failure.json"
        write_json(target, record)
        print(json.dumps(record), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
