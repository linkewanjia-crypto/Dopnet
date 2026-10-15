"""Serial WiMAE qualification/training/scoring after fixed40 Forti fully exits.

All children share one explicitly supplied deadline. A synthetic compute probe
does not certify the real waveform path, and incomplete training cannot score.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .prepare_wimae import digest, file_hash, prepare, signed_metadata, validate_manifest, write_json
from .train_wimae import DeadlineReached, check_deadline, deadline_from, seed_lock
from .wimae_protocol import (SEEDS, check_marker, check_training, check_evaluation_metadata,
                             read_json, require)


def process_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        if os.name == "posix":
            text = (Path("/proc") / str(pid) / "stat").read_text()
            if text[text.rfind(")") + 2:].split()[0] == "Z":
                return False
        return True
    except (ProcessLookupError, FileNotFoundError):
        return False


def no_forti_workers(args, state):
    records = list(state.get("tasks", {}).values())
    if any(process_alive(r.get("pid")) for r in records):
        return False
    lock = args.forti_queue_state.parent / "queue.lock"
    if lock.exists() and process_alive(read_json(lock).get("pid")):
        return False
    controller_pid = args.forti_training_output / "queue_controller.pid"
    if controller_pid.exists() and process_alive(int(controller_pid.read_text().strip())):
        return False
    if os.name == "posix":
        tokens = ("budget_matched_baselines.train_author", "budget_matched_baselines.evaluate_author",
                  "budget_matched_baselines.report_author", "budget_matched_baselines.queue",
                  "budget_matched_baselines.start")
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit() and int(entry.name) != os.getpid():
                try:
                    command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                    if str(args.forti_training_output.resolve()) in command and any(t in command for t in tokens):
                        if process_alive(int(entry.name)):
                            return False
                except OSError:
                    pass
    return True


def select_probe(records, generation_headroom=34.5, extra_margin=3.0):
    """Rank the measured two-phase workload only after memory qualification."""
    qualified = []
    for record in records:
        if record.get("status") != "passed" or record.get("scope") != "synthetic_shape_compute_only":
            continue
        if (record.get("cuda_memory_cap_enforced") is not True or record.get("gpu_cap_gib") != 6
                or record.get("parameters") != 554128 or record.get("real_waveforms_generated") is not False
                or record.get("waveform_generation_concurrency_certified") is not False):
            continue
        phases = record.get("phases", [])
        if [p.get("phase") for p in phases] != ["ssl", "ce"]:
            continue
        micro = record.get("probe_override_microbatch")
        if micro not in (1, 2, 4, 8) or any(p.get("microbatch_frames") != micro
                or p.get("expanded_channel_slices_per_microbatch") != 50 * micro
                or p.get("completed_optimizer_steps", 0) < 1 for p in phases):
            continue
        reserved = max(p["peak_reserved_gib"] for p in phases)
        if not all(math.isfinite(float(v)) and float(v) >= 0 for v in
                   (reserved, record["gpu_total_gib"], record["gpu_free_before_probe_gib"])):
            continue
        if (reserved + generation_headroom + extra_margin > record["gpu_total_gib"]
                or reserved + generation_headroom + extra_margin > record["gpu_free_before_probe_gib"]):
            continue
        fps = [p["synthetic_frames_per_second"] for p in phases]
        if any(not math.isfinite(float(v)) or v <= 0 for v in fps):
            continue
        # Both phases contain the same frame exposure; compare their combined
        # time, rather than selecting a microbatch from only the easier phase.
        combined = 2 / (1 / fps[0] + 1 / fps[1])
        qualified.append((combined, micro, record))
    require(qualified, "No memory-qualified two-phase probe; retain generation32/link8 and stop")
    qualified.sort(key=lambda item: (-item[0], item[1]))
    return {"status": "qualified_compute_only", "selected_microbatch": qualified[0][1],
            "combined_synthetic_frames_per_second": qualified[0][0],
            "generation_headroom_gib": generation_headroom, "additional_margin_gib": extra_margin,
            "waveform_generation_concurrency_certified": False,
            "claim": "requires_first_real_training_batch_full_path_observation",
            "candidate_summary": [{"microbatch": micro, "combined_frames_per_second": speed}
                                  for speed, micro, _ in qualified]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "reference-results", "validation-cache", "output", "forti-queue-state",
                 "forti-training-output", "forti-evaluation", "forti-code-root", "support-code-root"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--deadline-utc", required=True)
    ap.add_argument("--probe-seconds", type=float, default=45)
    ap.add_argument("--poll-seconds", type=float, default=30)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--generation-headroom-gib", type=float, default=34.5)
    ap.add_argument("--additional-margin-gib", type=float, default=3.0)
    args = ap.parse_args()
    require(0 < args.poll_seconds <= 60 and 2 <= args.probe_seconds <= 300, "Invalid bounded poll/probe duration")
    require(args.generation_headroom_gib >= 34.5 and args.additional_margin_gib >= 3,
            "Do not weaken the measured waveform generation memory reserve")
    deadline = deadline_from(args.deadline_utc)
    output = args.output.resolve()
    protected = [args.project.resolve(), args.reference_results.resolve(), args.forti_training_output.resolve()]
    require(not any(output == p or p in output.parents for p in protected), "Use a separate expansion output")
    output.mkdir(parents=True, exist_ok=True)
    package_root = Path(__file__).resolve().parents[1]
    dependencies = [package_root, args.forti_code_root.resolve().parent, args.support_code_root.resolve()]
    for folder in dependencies:
        sys.path.insert(0, str(folder))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in dependencies) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    identity = dict(project=str(args.project.resolve()), reference=str(args.reference_results.resolve()),
                    validation=str(args.validation_cache.resolve()), deadline_utc=deadline.isoformat(),
                    forti_state=str(args.forti_queue_state.resolve()), forti_evaluation=str(args.forti_evaluation.resolve()),
                    generation_headroom_gib=args.generation_headroom_gib, additional_margin_gib=args.additional_margin_gib,
                    queue_source_sha256=file_hash(Path(__file__)), bootstrap=args.bootstrap,
                    probe_seconds=args.probe_seconds, amp="bf16")
    state_path = output / "queue_state.json"
    state = read_json(state_path) if state_path.exists() else {"identity": identity, "tasks": {}, "status": "starting"}
    require(state["identity"] == identity, "Queue identity/deadline differs; never extend a frozen run silently")

    def publish(status):
        state["status"] = status
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)

    def run(name, module, options, child_env=None, allow_failed=False, completion_checker=None):
        check_deadline(deadline)
        previous = state["tasks"].get(name, {})
        require(not process_alive(previous.get("pid")), "A previously launched worker is still active")
        command = [sys.executable, "-u", "-m", module] + [str(v) for v in options]
        require(not previous or previous.get("command") == command, "Resume task argv differs")
        if previous.get("status") == "complete" and completion_checker:
            current_hashes = completion_checker()
            require(previous.get("output_sha256") == current_hashes, "Completed task evidence changed: " + name)
            print("WIMAE_QUEUE_REUSE_AUTHENTICATED " + name, flush=True)
            return 0
        record = dict(command=command, status="running", started_utc=datetime.now(timezone.utc).isoformat())
        state["tasks"][name] = record
        publish("running_" + name)
        log = output / "logs" / (name + ".log")
        log.parent.mkdir(exist_ok=True)
        with log.open("ab") as stream:
            child = subprocess.Popen(command, cwd=package_root, env=dict(env, **(child_env or {})),
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            record["pid"] = child.pid
            write_json(state_path, state)
            code = child.wait()
        record.update(returncode=code, status="complete" if code == 0 else "incomplete_time_limit" if code == 75 else "failed")
        write_json(state_path, state)
        if code == 75:
            raise DeadlineReached("Child stopped at the unchanged deadline: " + name)
        if code and not allow_failed:
            raise RuntimeError("Stage failed; see " + str(log))
        if code == 0 and completion_checker:
            record["output_sha256"] = completion_checker()
            write_json(state_path, state)
        return code

    with seed_lock(output / "queue_owner"):
        try:
            from budget_matched_baselines import verify_author_delivery as forti_checker
            require(Path(forti_checker.__file__).resolve().parent == args.forti_code_root.resolve(),
                    "Wrong deployed immutable Forti integrity package")
            while True:
                check_deadline(deadline)
                forti_state = read_json(args.forti_queue_state) if args.forti_queue_state.exists() else {}
                if forti_state.get("status") == "complete" and no_forti_workers(args, forti_state):
                    proof = forti_checker.verify(args.forti_evaluation / "budget_matched_results.zip",
                                                 args.reference_results, args.forti_code_root)
                    state["forti_delivery"] = proof
                    break
                publish("waiting_for_complete_Forti_and_all_workers_exit")
                time.sleep(args.poll_seconds)
            publish("serial_compute_qualification")
            probe_plan = output / "probe_plan"
            if not (probe_plan / "wimae_manifest.json").exists():
                prepare(args.project, args.reference_results, args.validation_cache, probe_plan, 1, "bf16")
            probes = []
            for micro in (1, 2, 4, 8):
                target = output / "probes" / f"micro{micro}.json"
                if not target.exists():
                    run(f"probe_micro{micro}", "dl_baseline_expansion.train_wimae",
                        ["--project", args.project, "--manifest", probe_plan / "wimae_manifest.json", "--seed", 42,
                         "--deadline-utc", deadline.isoformat(), "--probe-only", "--probe-seconds", args.probe_seconds,
                         "--probe-microbatch", micro, "--probe-output", target],
                        {"DOPNET_PROBE_GPU_CAP_GIB": "6"}, allow_failed=True)
                record = read_json(target)
                probe_manifest, _, _ = validate_manifest(probe_plan / "wimae_manifest.json", args.project, 42)
                require(record.get("recipe_signature") == probe_manifest["signature"]
                        and record.get("implementation_sha256") == probe_manifest["implementation_sha256"]
                        and record.get("probe_override_microbatch") == micro
                        and record.get("status") in ("passed", "failed"),
                        "Cached probe belongs to another recipe/source/candidate or is incomplete")
                probes.append(record)
            selection_path = output / "microbatch_selection.json"
            selection = select_probe(probes, args.generation_headroom_gib, args.additional_margin_gib)
            if selection_path.exists():
                require(read_json(selection_path) == selection, "Qualified microbatch selection changed")
            write_json(selection_path, selection)
            training = output / "wimae_training"
            if not (training / "wimae_manifest.json").exists():
                prepare(args.project, args.reference_results, args.validation_cache, training,
                        selection["selected_microbatch"], "bf16")
            recipe = signed_metadata(training / "wimae_manifest.json")
            for seed in SEEDS:
                def completed_training(s=seed):
                    from budget_matched_baselines.verify_author_delivery import checkpoint_metadata
                    manifest, cfg, name = validate_manifest(training / "wimae_manifest.json", args.project, s)
                    _, hashes = check_training(training / "checkpoints" / name, manifest, cfg,
                                               lambda p: checkpoint_metadata(p.read_bytes(), p.name))
                    return hashes
                run(f"train_seed{seed}", "dl_baseline_expansion.train_wimae",
                    ["--project", args.project, "--manifest", training / "wimae_manifest.json", "--seed", seed,
                     "--deadline-utc", deadline.isoformat(), "--resume", "auto"], completion_checker=completed_training)
                check_marker(read_json(training / "checkpoints" / f"wimae_author_seed{seed}" / "complete.json"), seed, recipe["signature"])
            evaluation = output / "evaluation"
            def completed_scoring():
                from .verify_wimae import verify_directory
                verify_directory(evaluation, training, args.reference_results, args.forti_evaluation,
                                 args.forti_code_root, require_assets=False)
                return {n: file_hash(evaluation / n) for n in ("COMPLETE.json", "evaluation.json", "summary/statistics.json")}
            run("frozen_three_seed_scoring", "dl_baseline_expansion.evaluate_wimae",
                ["--project", args.project, "--reference-results", args.reference_results, "--training-output", training,
                 "--forti-evaluation", args.forti_evaluation, "--forti-code-root", args.forti_code_root,
                 "--output", evaluation, "--deadline-utc", deadline.isoformat(), "--bootstrap", args.bootstrap,
                 "--inference-microbatch", 1], completion_checker=completed_scoring)
            def completed_report():
                from .verify_wimae import verify_archive
                verify_archive(evaluation / "wimae_results_for_paper.zip", args.reference_results, args.forti_code_root)
                return {n: file_hash(evaluation / n) for n in ("wimae_results_for_paper.zip", "wimae_results_for_paper.zip.sha256")}
            run("seven_method_report", "dl_baseline_expansion.report_wimae",
                ["--project", args.project, "--reference-results", args.reference_results, "--training-output", training,
                 "--forti-evaluation", args.forti_evaluation, "--forti-code-root", args.forti_code_root,
                 "--output", evaluation, "--deadline-utc", deadline.isoformat()], completion_checker=completed_report)
            publish("complete")
            return 0
        except DeadlineReached as error:
            state["reason"] = str(error)
            publish("incomplete_time_limit")
            return 75
        except Exception as error:
            state["reason"] = str(error)
            publish("failed")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
