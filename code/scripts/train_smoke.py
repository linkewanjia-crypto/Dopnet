#!/usr/bin/env python3
"""Run three native training batches and one fresh validation frame.

This standard-library launcher selects a bounded CUDA FP32 configuration and
an external output location.
It starts train.py in a separate interpreter: no imports, hooks, cached tensors,
selected weights, private acceptance worker or historical attestation bypass.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def identity(path):
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("dopnet", "dopnet_lite"), default="dopnet")
    parser.add_argument("--output", required=True, type=Path,
                        help="New external run directory; must not already exist.")
    parser.add_argument("--execute", action="store_true",
                        help="Run the native trainer; otherwise only print the plan.")
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    for protected in (code, (code.parent / "data").resolve()):
        if output == protected or protected in output.parents:
            parser.error("--output must be outside the supplied code/data trees")
    if output.exists():
        parser.error("--output already exists; choose a new run directory")
    train = code / "training/main/train.py"
    config = code / "config/training/main" / (args.method + "_seed42.json")
    run_name = args.method + "_three_batch_smoke"
    command = [
        sys.executable, "-B", str(train), "--config", str(config),
        "--run-name", run_name, "--seed", "42", "--device", "cuda",
        "--train-samples", "3", "--val-samples", "1", "--batch-size", "1",
        "--generation-batch-size", "1", "--microbatch-size", "1",
        "--stop-after-epoch", "1", "--amp", "none",
        "--set", "train.validation_cache=false",
        "--set", "system.sionna_device=cuda",
        "--set", "data_pipeline.require_cuda=true",
        "--set", "data_pipeline.synthetic_microbatch_size=1",
        "--set", "data_pipeline.threegpp_microbatch_size=1",
    ]
    report = {
        "schema": "dopnet_public_native_training_smoke_v1", "status": "PLAN",
        "method": args.method, "command": command, "working_directory": str(output),
        "source": {"training/main/train.py": identity(train),
                   str(config.relative_to(code)): identity(config)},
        "budget": {"training_samples": 3, "batch_size": 1,
                   "optimizer_steps_in_native_loop": 3, "validation_samples": 1,
                   "stop_after_epoch": 1, "scheduler_horizon_epochs": 40},
        "fresh_generator_inputs": True, "historical_reconstruction": False,
        "private_worker_or_hooks_used": False,
        "checkpoint_restore_tested_by_this_command": False,
    }
    print(json.dumps(report, indent=2), flush=True)
    if not args.execute:
        return 0
    output.mkdir(parents=True)
    receipt = output / "SMOKE_REPORT.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    report["status"] = "RUNNING"
    receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    result = subprocess.run(command, cwd=output, env=env, check=False)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["native_returncode"] = result.returncode
    report["status"] = "FAIL_NATIVE_TRAINER"
    if result.returncode == 0:
        try:
            run = output / "checkpoints" / run_name
            with (run / "history.csv").open(encoding="utf-8", newline="") as stream:
                history = list(csv.DictReader(stream))
            valid = (len(history) == 1 and int(history[0]["epoch"]) == 1
                     and all(math.isfinite(float(history[0][key]))
                             for key in ("train_loss", "val_loss", "val_nmse_db")))
            checkpoints = {name: identity(run / name) for name in ("model.pt", "last.pt")}
            valid = valid and all(row["bytes"] > 0 for row in checkpoints.values())
            report.update(history=history, checkpoints=checkpoints,
                          status="PASS_NATIVE_THREE_BATCH_SMOKE" if valid else "FAIL_OUTPUT_CHECK")
        except (OSError, ValueError, KeyError, IndexError) as error:
            report.update(status="FAIL_OUTPUT_CHECK", output_error=str(error))
    receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("smoke result:", report["status"], "| receipt:", receipt, flush=True)
    return 0 if report["status"] == "PASS_NATIVE_THREE_BATCH_SMOKE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
