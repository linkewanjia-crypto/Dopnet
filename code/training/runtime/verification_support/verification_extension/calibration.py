"""Independent calibration and measured CPU/CUDA selection for classical receivers.

No simulator, neural model, checkpoint training or test target is used here.
The caller supplies completed, independently generated calibration chunks.
Covariance holdout retains whole chunks; every BEM candidate scores every one
of the declared calibration frames with a residual-checked complex128 solver.
"""
from __future__ import annotations

from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
import uuid
from collections import Counter

import numpy as np

from . import protocol as P

PRECISION = "complex128"
REFERENCE_RELATIVE_L2_TOLERANCE = 5e-7


def _runtime():
    import torch
    from . import classical
    return torch, classical


def _metadata(arrays):
    value = arrays.get("metadata")
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return json.loads(str(np.asarray(value).item()))


def _calibration_identity(arrays_list, manifest):
    count = sum(int(arrays["x_pilot"].shape[0]) for arrays in arrays_list)
    if manifest.get("calibration_seed") != 810000001:
        raise ValueError("Need the declared independent calibration seed 810000001")
    expected = int(manifest.get("calibration_samples", 128))
    if count != expected or expected != 128:
        raise ValueError(f"Complete original calibration requires exactly 128 frames; supplied {count}")
    fingerprints = []
    for arrays in arrays_list:
        meta = _metadata(arrays)
        if meta and (meta.get("role") != "calibration" or meta.get("calibration_seed", 810000001) != 810000001):
            raise ValueError("Do not use test observations/targets for calibration")
        import hashlib
        h = hashlib.sha256()
        for key in ("x_pilot", "y_pilot", "target_h", "rx_data", "tx_label", "noise_var",
                    "pilot_visible", "data_visible", "cp_wave_samples", "waveform_fft_size",
                    "subcarrier_spacing_hz"):
            value = np.ascontiguousarray(arrays[key])
            h.update(key.encode("ascii")); h.update(str(value.dtype).encode("ascii"))
            h.update(json.dumps(list(value.shape)).encode("ascii")); h.update(value.tobytes())
        fingerprints.append({"frames": int(arrays["x_pilot"].shape[0]), "sha256": h.hexdigest()})
    return {"calibration_seed": 810000001, "calibration_samples": count,
            "export_signature": manifest["export_signature"], "chunks": fingerprints,
            "calibration_content_sha256": P.digest(fingerprints)}


def _statistical_identity(arrays_list, manifest, base_identity):
    """Additional independent chunks affect priors only, never the BEM ensemble."""
    extra_seed = manifest.get("statistical_extra_seed")
    extra_count = int(manifest.get("statistical_extra_count", 0))
    expected = base_identity["calibration_samples"] + extra_count
    counts, fingerprints = {}, []
    import hashlib
    for arrays in arrays_list:
        meta = _metadata(arrays)
        role = meta.get("role", "calibration")
        seed = meta.get("calibration_seed", meta.get("statistical_calibration_seed"))
        if role == "calibration":
            seed = 810000001 if seed is None else seed
            if seed != 810000001:
                raise ValueError("Original prior chunks have an undeclared calibration seed")
        elif role == "statistical_calibration":
            if extra_seed is None or seed != extra_seed or seed == 810000001:
                raise ValueError("Additional prior chunks need their distinct declared independent seed")
        else:
            raise ValueError("Only original or additional independent calibration targets may fit priors")
        count = int(arrays["x_pilot"].shape[0]); counts[str(seed)] = counts.get(str(seed), 0) + count
        h = hashlib.sha256()
        for key in ("x_pilot", "y_pilot", "target_h", "rx_data", "tx_label", "noise_var",
                    "pilot_visible", "data_visible", "cp_wave_samples", "waveform_fft_size",
                    "subcarrier_spacing_hz"):
            value = np.ascontiguousarray(arrays[key]); h.update(key.encode("ascii"))
            h.update(str(value.dtype).encode("ascii")); h.update(json.dumps(list(value.shape)).encode("ascii")); h.update(value.tobytes())
        fingerprints.append({"frames": count, "seed": seed, "role": role, "sha256": h.hexdigest()})
    if sum(counts.values()) != expected or counts.get("810000001", 0) != 128:
        raise ValueError("Statistical calibration does not contain the complete base128 plus declared extra independent frames")
    if extra_count and counts.get(str(extra_seed), 0) != extra_count:
        raise ValueError("Additional statistical calibration frame count differs from manifest")
    if [row["sha256"] for row in fingerprints if row["seed"] == 810000001] != [row["sha256"] for row in base_identity["chunks"]]:
        raise ValueError("Original128 statistical frames differ from the BEM calibration ensemble")
    return {"frames": expected, "seed_frame_counts": counts, "chunks": fingerprints,
            "extra_seed": extra_seed, "extra_frames": extra_count,
            "exported_source_proof": manifest.get("statistical_calibration_sources", {}),
            "content_sha256": P.digest(fingerprints)}


def _envelope(manifest):
    system = manifest["cfg_system"]
    values = [float(v) for key in ("doppler_hz_choices", "threegpp_doppler_hz_choices")
              for v in system.get(key, [])]
    if not values or max(values) <= 0:
        raise ValueError("Missing positive declared training Doppler envelope")
    return max(values)


def _setting(method, manifest, overrides=None):
    _, C = _runtime()
    values = {"method": method, "training_doppler_envelope_hz": _envelope(manifest)}
    if overrides:
        values.update(asdict(overrides) if hasattr(overrides, "__dataclass_fields__") else overrides)
    values["method"] = method
    return C.BEMSettings(**values)


def _geometry_restore(values):
    _, C = _runtime()
    values = dict(values)
    for key in ("pilot_visible", "data_visible", "active_signed_bins"):
        values[key] = tuple(values[key])
    return C.WaveformGeometry(**values)


def _hardware(torch):
    return {"platform": platform.platform(), "cpu": platform.processor(),
            "logical_cpu_count": os.cpu_count(), "python": platform.python_version(),
            "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "gpus": [{"device": f"cuda:{index}", "name": torch.cuda.get_device_name(index),
                      "total_memory_mb": torch.cuda.get_device_properties(index).total_memory / 2**20}
                     for index in range(torch.cuda.device_count())] if torch.cuda.is_available() else []}


def _sync(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _frame_noise(arrays, index):
    return float(np.asarray(arrays["noise_var"])[index])


def _modulation(arrays):
    if "condition" in arrays:
        return arrays["condition"]["modulation"]
    return _metadata(arrays)["condition"]["modulation"]


def _maximum_residual(info):
    return max(float(info.get("relative_residual_max", 0)),
               float(info.get("recovered_normal_relative_residual_max", 0)))


def _run_frame(arrays, frame, geometry, method, op_or_prior, *, noise_scale=1.0,
               diagnostic=False):
    """Resident ensemble/basis; actual per-frame design/fit and scalar scoring."""
    torch, C = _runtime()
    device, dtype = op_or_prior.device, op_or_prior.dtype
    # Transfers are inside this function. No target enters estimate()/fit().
    xp, yp, rx, truth = (torch.as_tensor(arrays[key][frame], device=device, dtype=dtype)
                         for key in ("x_pilot", "y_pilot", "rx_data", "target_h"))
    labels = torch.as_tensor(arrays["tx_label"][frame], device=device, dtype=torch.int64)
    noise = _frame_noise(arrays, frame)
    if method == C.STATISTICAL_METHOD:
        cfr, fit_info = op_or_prior.estimate(xp, yp, noise, geometry, noise_scale=noise_scale)
        coefficients = None
    else:
        coefficients, fit_info = op_or_prior.fit(xp, yp, noise)
        cfr = op_or_prior.diagonal_cfr(coefficients)
    common, common_info = C.diagonal_mmse(cfr, rx, noise, return_info=True)
    nmse = C.nmse_per_frame(cfr[None], truth[None])[0]
    errors, bits = C.frame_bit_counts(common[None], labels[None], _modulation(arrays))
    scores = [nmse, errors[0].to(torch.float64), bits[0].to(torch.float64)]
    native, native_info = None, None
    if coefficients is not None:
        native, native_info = op_or_prior.native_mmse(coefficients, rx, noise)
        native_errors, native_bits = C.frame_bit_counts(native[None], labels[None], _modulation(arrays))
        scores += [native_errors[0].to(torch.float64), native_bits[0].to(torch.float64)]
    scalar_scores = torch.stack(scores).detach().cpu().numpy()
    infos = {"fit": fit_info, "common_detector": common_info, "native_detector": native_info}
    maximum = max(_maximum_residual(info) for info in infos.values() if info is not None)
    if not np.all(np.isfinite(scalar_scores)) or maximum > 1e-6:
        raise C.SolverConvergenceError("Classical verification timing failed scalar/residual gate")
    result = {"scores": scalar_scores, "solver_residual_max": maximum, "solver_info": infos}
    if diagnostic:
        result.update(cfr=cfr.detach().cpu().numpy(), common_symbols=common.detach().cpu().numpy(),
                      native_symbols=native.detach().cpu().numpy() if native is not None else None)
    return result


def compare_classical_outputs(reference, candidate):
    checks = {}
    for key in ("cfr", "common_symbols", "native_symbols"):
        if reference[key] is None:
            continue
        a, b = np.asarray(reference[key]), np.asarray(candidate[key])
        relative = float(np.linalg.norm((b - a).ravel()) / max(np.linalg.norm(a.ravel()), 1e-12))
        checks[key] = {"relative_l2": relative, "max_abs": float(np.max(np.abs(b-a))),
                       "passed": bool(np.all(np.isfinite(b)) and relative <= REFERENCE_RELATIVE_L2_TOLERANCE)}
    # NMSE may differ by roundoff. Integer BER numerator/denominator must agree.
    count_equal = bool(np.array_equal(reference["scores"][1:], candidate["scores"][1:]))
    passed = all(value["passed"] for value in checks.values()) and count_equal
    return {"passed": passed, "relative_l2_tolerance": REFERENCE_RELATIVE_L2_TOLERANCE,
            "ber_counts_equal": count_equal, "tensor_checks": checks}


def choose_backend_rows(rows, methods):
    """Do not select unavailable, failed or numerically unequal backends."""
    choices = []
    for method in methods:
        candidates = [row for row in rows if row["method"] == method and row["status"] == "measured"
                      and row["numerical_equivalence"]["passed"]]
        if not candidates:
            raise RuntimeError(f"No qualified backend for {method}")
        best = min(candidates, key=lambda row: row["latency_ms_p50"])
        choices.append({"method": method, "preferred_device": best["device"],
                        "cpu_threads": best["cpu_threads"], "precision": PRECISION,
                        "latency_ms_p50": best["latency_ms_p50"],
                        "frames_per_second": best["frames_per_second"],
                        "scope": "host calibration arrays to fit+CFR+common/native detection and scalar NMSE/BER on host",
                        "selection_basis": "measured wall-clock median, same independent frames and complex128; CFR/symbol/residual/count gates passed"})
    return choices


def _statistical_banks(arrays_list, manifest, device):
    _, C = _runtime()
    groups = C.group_calibration_batches(arrays_list, manifest)
    return {key: {"prior": C.PilotCovarianceLMMSE.from_calibration(group["batches"], manifest,
                            device=device, dtype=PRECISION), "noise_scale": 1.0}
            for key, group in groups.items()}


def benchmark_classical(arrays_list, manifest, output, *, cpu_threads=(1, 4, 8),
                        devices=("cpu", "cuda:0"), warmups=2, repetitions=5,
                        settings=None, state_directory=None, benchmark_arrays_list=None):
    """Measure CPU thread candidates and CUDA before selecting verification backend."""
    torch, C = _runtime()
    arrays_list = list(arrays_list)
    identity = _calibration_identity(arrays_list, manifest)
    benchmark_arrays = arrays_list if benchmark_arrays_list is None else list(benchmark_arrays_list)
    benchmark_identity = identity if benchmark_arrays_list is None else _statistical_identity(benchmark_arrays, manifest, identity)
    if warmups < 1 or repetitions < 3 or not cpu_threads or any(value < 1 for value in cpu_threads):
        raise ValueError("Use warmups>=1, repetitions>=3, positive CPU thread candidates")
    if "cpu" not in devices or len(set(devices)) != len(devices):
        raise ValueError("Include CPU once as the numerical reference")
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    groups = C.group_calibration_batches(benchmark_arrays, manifest)
    required_cps = manifest.get("required_cp_wave_samples")
    frames, missing_workloads = [], []
    public_cells = manifest.get("cells", [])
    for key, group in sorted(groups.items()):
        geometry = group["geometry"]
        if required_cps is not None and geometry.cp_wave_samples not in required_cps:
            continue
        desired_spacings = sorted({int(cell["pilot_spacing"]) for cell in public_cells
            if int(cell["cp_len"]) * geometry.nfft // geometry.n_active == geometry.cp_wave_samples})
        if not desired_spacings:
            frames.append((key, geometry, group["batches"][0], 0))
            continue
        for spacing in desired_spacings:
            matches = [arrays for arrays in group["batches"]
                       if np.count_nonzero(arrays["x_pilot"][0, 0, 0]) * spacing == geometry.n_active]
            if matches:
                frames.append((key, geometry, matches[0], 0))
            else:
                missing_workloads.append({"cp_wave_samples": geometry.cp_wave_samples,
                                          "pilot_spacing": spacing,
                                          "reason": "original128 calibration contains no matching public pilot density"})
        if not any(item[0] == key for item in frames):
            frames.append((key, geometry, group["batches"][0], 0))
    if not frames:
        raise ValueError("No original calibration frame has a required suite CP")
    options = [(torch.device("cpu"), int(threads)) for threads in cpu_threads]
    unavailable = []
    for name in devices:
        if name == "cpu":
            continue
        device = torch.device(name)
        if device.type != "cuda":
            raise ValueError("Only CPU/CUDA devices supported")
        if not torch.cuda.is_available() or device.index is not None and device.index >= torch.cuda.device_count():
            unavailable.append({"device": name, "reason": "CUDA unavailable"})
        else:
            options.append((device, int(cpu_threads[0])))
    setting_map = {method: _setting(method, manifest, (settings or {}).get(method)) for method in C.BEM_METHODS}
    if state_directory is not None:
        lock = P.signed_json(Path(state_directory) / "calibration_lock.json")
        if lock["calibration_content_sha256"] != identity["calibration_content_sha256"]:
            raise RuntimeError("Selected-settings benchmark has different independent calibration inputs")
        setting_map = {method: _setting(method, manifest, lock["settings"][method]) for method in C.BEM_METHODS}
    report = {"complete": False, "schema_version": 1, "precision": PRECISION,
              "hardware": _hardware(torch), "code_sha256": P.source_hashes(),
              "calibration": identity, "benchmark_calibration": benchmark_identity,
              "settings": {key: asdict(value) for key, value in setting_map.items()},
              "protocol": {"scope": "verification throughput, not RF/ADC deployment latency",
                           "timed": "per-frame input H2D, design/fit, CFR, common MMSE and BEM native ICI-MMSE, NMSE/BER and scalar D2H",
                           "excluded": "offline covariance calibration, model/channel training, disk IO, cached public basis construction",
                           "frames_per_repetition": len(frames), "selected_frames": [{"geometry_key": key,
                               "original_chunk_index": next(index for index, candidate in enumerate(benchmark_arrays) if candidate is arrays),
                               "pilot_spacing": int(geometry.n_active // np.count_nonzero(arrays["x_pilot"][0, 0, 0])),
                               "selection": "first matching existing independent chunk", "frame": frame}
                               for key, geometry, arrays, frame in frames],
                           "public_workload_selection": "first existing independent chunk per required CP/pilot spacing; selection uses no channel targets or test performance",
                           "missing_public_workloads": missing_workloads,
                           "warmups": warmups, "repetitions": repetitions, "cpu_threads": list(cpu_threads),
                           "timing": "perf_counter wall clock, CUDA synchronization before/after every complete repetition",
                           "state_directory": str(Path(state_directory).resolve()) if state_directory is not None else None},
              "unavailable_devices": unavailable, "rows": []}
    path = output / "classical_backend_benchmark.json"
    P.write_json(path, report)
    references, previous_threads = {}, torch.get_num_threads()
    try:
        with torch.inference_mode():
            for device, threads in options:
                torch.set_num_threads(threads)
                if device.type == "cuda":
                    torch.cuda.set_device(device)
                try:
                    bank = (load_calibrated_bank(state_directory, device, PRECISION) if state_directory is not None
                            else _statistical_banks(benchmark_arrays, manifest, device))
                except (torch.cuda.OutOfMemoryError, C.SolverConvergenceError) as exc:
                    report["rows"].extend({"method": method, "device": str(device), "cpu_threads": threads,
                        "status": "failed", "stage": "offline resident bank setup", "error": str(exc),
                        "numerical_equivalence": {"passed": False}} for method in C.METHODS)
                    P.write_json(path, report); gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                for method in C.METHODS:
                    print(f"[classical-benchmark] {method} {device} CPU-threads={threads}", flush=True)
                    try:
                        method_frames = frames if method != C.STATISTICAL_METHOD else [item for item in frames if item[0] in bank]
                        if not method_frames:
                            raise RuntimeError("No required-CP statistical prior available for the selected-settings benchmark")
                        operators = {key: bank[key]["prior"] if method == C.STATISTICAL_METHOD else
                                     C.TorchJointBEM(geometry, setting_map[method], arrays["x_pilot"].shape[1], device=device, dtype=PRECISION)
                                     for key, geometry, arrays, _ in method_frames}
                        def one(diagnostic=False):
                            return [_run_frame(arrays, frame, geometry, method, operators[key],
                                noise_scale=bank.get(key, {}).get("noise_scale", 1.0), diagnostic=diagnostic)
                                for key, geometry, arrays, frame in method_frames]
                        diagnostics = one(True)
                        if method not in references and device.type == "cpu":
                            references[method] = diagnostics
                        if method not in references:
                            raise C.SolverConvergenceError("No successful CPU reference exists for same-input backend qualification")
                        checks = [compare_classical_outputs(a, b) for a, b in zip(references[method], diagnostics)]
                        for _ in range(warmups):
                            one()
                        _sync(torch, device)
                        peak_before = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        samples, residuals, native_chunks = [], [], []
                        for _ in range(repetitions):
                            _sync(torch, device); start = time.perf_counter()
                            measured = one()
                            _sync(torch, device)
                            samples.append((time.perf_counter() - start) * 1000)
                            residuals.extend(record["solver_residual_max"] for record in measured)
                            native_chunks.extend(record["solver_info"]["native_detector"]["symbol_batch_size"]
                                for record in measured if record["solver_info"]["native_detector"] is not None)
                        row = {"method": method, "device": str(device), "cpu_threads": threads,
                               "status": "measured", "numerical_equivalence": {"passed": all(check["passed"] for check in checks), "frames": checks},
                               "latency_ms_mean": float(np.mean(samples)), "latency_ms_p50": float(np.quantile(samples, .5)),
                               "latency_ms_p95": float(np.quantile(samples, .95)), "repetition_latency_ms": samples,
                               "frames_per_repetition": len(method_frames),
                               "frames_per_second": 1000 * len(method_frames) / float(np.mean(samples)),
                               "solver_residual_max": max(residuals),
                               "native_detector_symbol_batch_sizes": sorted(set(native_chunks)),
                               "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                               "gpu_incremental_peak_mb": (torch.cuda.max_memory_allocated(device) - peak_before) / 2**20 if device.type == "cuda" else None}
                    except (torch.cuda.OutOfMemoryError, C.SolverConvergenceError) as exc:
                        row = {"method": method, "device": str(device), "cpu_threads": threads,
                               "status": "failed", "error": str(exc), "numerical_equivalence": {"passed": False}}
                    report["rows"].append(row); P.write_json(path, report)
                    operators = None; one = None; diagnostics = None; measured = None
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                del bank
        report["complete"] = True
        recommendations = choose_backend_rows(report["rows"], C.METHODS)
        P.write_json(path, report)
        policy = {"schema_version": 1, "status": "qualified", "complete": True,
                  "precision": PRECISION, "scope": report["protocol"]["scope"],
                  "hardware": report["hardware"], "code_sha256": report["code_sha256"],
                  "calibration": identity, "settings": report["settings"],
                  "benchmark_calibration": benchmark_identity,
                  "benchmark_sha256": P.file_hash(path), "recommendations": recommendations,
                  "qualified_backends": [{key: row[key] for key in ("method", "device", "cpu_threads",
                        "latency_ms_p50", "frames_per_second")}
                      for row in report["rows"] if row["status"] == "measured" and row["numerical_equivalence"]["passed"]],
                  "excluded_backends": [row for row in report["rows"] if row["status"] != "measured" or not row["numerical_equivalence"]["passed"]]}
        policy["signature"] = P.digest(policy)
        P.write_json(output / "classical_device_policy.json", policy)
        return policy
    finally:
        torch.set_num_threads(previous_threads)


def _policy(policy, method):
    policy = P.signed_json(policy) if isinstance(policy, (str, Path)) else policy
    if policy.get("signature") != P.digest({k: v for k, v in policy.items() if k != "signature"}):
        raise RuntimeError("Unsigned/changed classical device policy")
    if not policy.get("complete") or policy.get("precision") != PRECISION:
        raise RuntimeError("Device policy is incomplete or has another precision")
    torch, _ = _runtime()
    if policy.get("code_sha256") != P.source_hashes():
        raise RuntimeError("Correction implementation differs from measured backend policy; rerun benchmark")
    hardware = policy.get("hardware", {})
    if (hardware.get("torch") != str(torch.__version__) or hardware.get("python") != platform.python_version()
            or hardware.get("logical_cpu_count") != os.cpu_count()):
        raise RuntimeError("Server runtime/CPU geometry differs from backend benchmark; remeasure here")
    selected = [row for row in policy["recommendations"] if row["method"] == method]
    if len(selected) != 1:
        raise RuntimeError(f"No unique qualified device recommendation for {method}")
    device = torch.device(selected[0]["preferred_device"])
    if device.type == "cuda":
        if not torch.cuda.is_available() or device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError("Measured CUDA backend unavailable; rerun CPU-only benchmark")
        name = next((row["name"] for row in hardware["gpus"] if row["device"] == str(device)), None)
        if name != torch.cuda.get_device_name(device):
            raise RuntimeError("CUDA hardware differs from backend benchmark; remeasure here")
    return selected[0]


def resolve_classical_device(policy_path, method):
    """Validate measured policy and select its qualified CPU fallback if needed."""
    path = Path(policy_path).resolve(); policy = P.signed_json(path)
    report_path = path.parent / "classical_backend_benchmark.json"
    if not report_path.is_file() or P.file_hash(report_path) != policy.get("benchmark_sha256"):
        raise RuntimeError("Classical device policy does not match its completed backend benchmark")
    torch, _ = _runtime()
    if not policy.get("complete") or policy.get("precision") != PRECISION or policy.get("code_sha256") != P.source_hashes():
        raise RuntimeError("Classical backend policy is incomplete or has changed code/precision")
    hardware = policy["hardware"]
    if (hardware["torch"] != str(torch.__version__) or hardware["python"] != platform.python_version()
            or hardware["logical_cpu_count"] != os.cpu_count()):
        raise RuntimeError("Server runtime/CPU geometry differs from measured classical policy; remeasure here")
    matches = [row for row in policy["recommendations"] if row["method"] == method]
    if len(matches) != 1:
        raise RuntimeError(f"No unique measured backend for {method}")
    selected = dict(matches[0]); device = torch.device(selected["preferred_device"])
    if device.type == "cuda":
        unavailable = not torch.cuda.is_available() or device.index is not None and device.index >= torch.cuda.device_count()
        if unavailable:
            report = P.read_json(report_path)
            candidates = [row for row in report["rows"] if row["method"] == method and row["device"] == "cpu"
                          and row["status"] == "measured" and row["numerical_equivalence"]["passed"]]
            if not candidates:
                raise RuntimeError("CUDA unavailable and no numerically qualified CPU fallback exists")
            best = min(candidates, key=lambda row: row["latency_ms_p50"])
            selected.update(preferred_device="cpu", cpu_threads=best["cpu_threads"],
                            latency_ms_p50=best["latency_ms_p50"], frames_per_second=best["frames_per_second"],
                            runtime_fallback="measured CUDA device currently unavailable")
        else:
            name = next((row["name"] for row in hardware["gpus"] if row["device"] == str(device)), None)
            if name != torch.cuda.get_device_name(device):
                raise RuntimeError("GPU hardware differs from measured classical policy; remeasure here")
    selected["calibration"] = policy["calibration"]
    selected["settings"] = policy["settings"].get(method)
    selected["benchmark_sha256"] = policy["benchmark_sha256"]
    return selected


def _validated_complete_candidate(record, contract, count):
    if record.get("status") != "complete" or record.get("contract") != contract:
        raise RuntimeError("Candidate has a changed input/code/grid protocol")
    if record.get("frames") != count or record.get("completed_frames") != count or len(record.get("frame_nmse", [])) != count:
        raise RuntimeError("A partial candidate cannot enter calibration selection")
    values = np.asarray(record["frame_nmse"], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise RuntimeError("Invalid complete candidate scores")
    if not np.isclose(record["mean_linear_nmse"], float(values.mean()), rtol=1e-12, atol=1e-15):
        raise RuntimeError("Complete candidate mean does not match its frame scores")
    residual = float(record.get("solver_residual_max", math.inf))
    if not math.isfinite(residual) or residual > 1e-6:
        raise RuntimeError("Unconverged candidate cannot enter selection")
    return record


def _qualified_cpu_fallback(policy, method, output):
    value = P.signed_json(policy) if isinstance(policy, (str, Path)) else policy
    if value.get("signature") != P.digest({key: item for key, item in value.items() if key != "signature"}):
        raise RuntimeError("Changed backend policy cannot authorize a numerical CPU fallback")
    candidates = [row for row in value.get("qualified_backends", []) if row["method"] == method and row["device"] == "cpu"]
    if not candidates:
        path = Path(output) / "classical_backend_benchmark.json"
        if not path.is_file() or P.file_hash(path) != value["benchmark_sha256"]:
            raise RuntimeError("No verified same-precision CPU backup exists")
        report = P.read_json(path)
        candidates = [row for row in report["rows"] if row["method"] == method and row["device"] == "cpu"
                      and row["status"] == "measured" and row["numerical_equivalence"]["passed"]]
    if not candidates:
        raise RuntimeError(f"No numerically qualified CPU fallback for {method}")
    row = min(candidates, key=lambda candidate: candidate["latency_ms_p50"])
    return {"preferred_device": "cpu", "cpu_threads": row["cpu_threads"], "precision": PRECISION}


def _statistical_geometry_result(split, manifest, noise_grid, device, output, key, contract):
    """Whole-geometry unit of work, allowing an OOM retry without partial choices."""
    torch, C = _runtime()
    prior = C.PilotCovarianceLMMSE.from_calibration(split["fit_batches"], manifest, device=device, dtype=PRECISION)
    candidates = []
    for scale in noise_grid:
        values, residual_max = [], 0.0
        for arrays in split["holdout_batches"]:
            for frame in range(arrays["x_pilot"].shape[0]):
                h, info = prior.estimate(arrays["x_pilot"][frame], arrays["y_pilot"][frame],
                    _frame_noise(arrays, frame), split["geometry"], noise_scale=float(scale))
                error = C.nmse_per_frame(h[None], torch.as_tensor(arrays["target_h"][frame], device=prior.device, dtype=prior.dtype)[None])
                values.append(float(error[0].cpu())); residual_max = max(residual_max, _maximum_residual(info))
        candidates.append({"noise_scale": float(scale), "frames": len(values),
                           "mean_linear_nmse": float(np.mean(values)), "frame_nmse": values,
                           "solver_residual_max": residual_max, "status": "complete"})
    selected = min(candidates, key=lambda candidate: candidate["mean_linear_nmse"])
    refit = C.PilotCovarianceLMMSE.from_calibration(split["all_batches"], manifest, device=device, dtype=PRECISION)
    relative = f"covariance/{key}.npz"; P.atomic_npz(Path(output) / relative, refit.state_arrays())
    return {"status": "complete", "contract": contract, "geometry": asdict(split["geometry"]),
            "noise_scale": selected["noise_scale"], "fit_frames": split["fit_frames"],
            "holdout_frames": split["holdout_frames"], "all_frames": split["fit_frames"] + split["holdout_frames"],
            "split_rule": split["split_rule"], "npz_relative_path": relative,
            "sha256": P.file_hash(Path(output) / relative), "calibration_info": refit.calibration_info,
            "noise_candidates": candidates, "fit_device": str(device)}


def _bem_frame_ridges(operator, arrays, frame, settings):
    """One frame's design shared by all ridges; no partial result is returned."""
    torch, C = _runtime()
    design = operator.design_matrix(arrays["x_pilot"][frame])
    truth = torch.as_tensor(arrays["target_h"][frame], device=operator.device, dtype=operator.dtype)
    values, residuals = [], []
    for setting in settings:
        coefficients, info = operator.fit(arrays["x_pilot"][frame], arrays["y_pilot"][frame],
            _frame_noise(arrays, frame), design=design, relative_ridge=setting.relative_ridge)
        cfr = operator.diagonal_cfr(coefficients)
        values.append(float(C.nmse_per_frame(cfr[None], truth[None])[0].cpu()))
        residuals.append(_maximum_residual(info))
    if not np.all(np.isfinite(values)) or max(residuals) > 1e-6:
        raise C.SolverConvergenceError("A candidate failed its declared score/residual gate")
    return values, residuals


def _exclusive_signed_json(path, value):
    """A lock is published once. Existing locks are never overwritten."""
    path = Path(path)
    unsigned = {key: item for key, item in value.items() if key != "signature"}
    value = {**unsigned, "signature": P.digest(unsigned)}
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".publish.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        # An atomic exclusive hard-link publication cannot expose a partial
        # lock, and refuses to replace a lock published by another process.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def calibrate_classical(arrays_list, manifest, output, policy, *, q_grid=(9, 15, 21),
                        delay_windows=((-4, 28), (-8, 40)),
                        ridge_grid=(.01, .1, 1, 10, 100), noise_grid=(.01, .1, 1, 10, 100),
                        statistical_arrays_list=None):
    """Calibrate every declared candidate, then atomically publish an immutable lock."""
    torch, C = _runtime()
    arrays_list = list(arrays_list); identity = _calibration_identity(arrays_list, manifest)
    statistical_arrays_list = arrays_list if statistical_arrays_list is None else list(statistical_arrays_list)
    statistical_identity = _statistical_identity(statistical_arrays_list, manifest, identity)
    q_grid = tuple(q_grid); delay_windows = tuple(tuple(value) for value in delay_windows)
    ridge_grid = tuple(ridge_grid); noise_grid = tuple(noise_grid)
    if (not q_grid or not delay_windows or not ridge_grid or not noise_grid
            or len(set(q_grid)) != len(q_grid) or len(set(delay_windows)) != len(delay_windows)
            or len(set(ridge_grid)) != len(ridge_grid) or len(set(noise_grid)) != len(noise_grid)
            or any(not math.isfinite(float(value)) or value <= 0 for value in tuple(ridge_grid) + tuple(noise_grid))):
        raise ValueError("Declare unique nonempty grids with positive finite ridge/noise values")
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    codes = P.source_hashes()
    grid = {"q_grid": list(q_grid), "delay_windows": [list(value) for value in delay_windows],
            "ridge_grid": list(map(float, ridge_grid)), "noise_grid": list(map(float, noise_grid))}
    contract = {"calibration": identity, "statistical_calibration": statistical_identity,
                "required_cp_wave_samples": manifest.get("required_cp_wave_samples"),
                "grid": grid, "code_sha256": codes, "precision": PRECISION}
    lock_path = output / "calibration_lock.json"
    if lock_path.exists():
        lock = P.signed_json(lock_path)
        if lock.get("status") != "locked" or lock.get("contract") != contract:
            raise RuntimeError("Existing immutable calibration lock belongs to another protocol")
        # Loading verifies every saved covariance checksum as well.
        load_calibrated_bank(output, "cpu", PRECISION)
        return lock
    groups = C.group_calibration_batches(statistical_arrays_list, manifest)
    required_cps = manifest.get("required_cp_wave_samples")
    splits, unavailable_priors = {}, {}
    for key, group in groups.items():
        try:
            splits.update(C.split_calibration_groups(group["batches"], manifest,
                                                     min_fit_frames=16, min_holdout_frames=8))
        except ValueError as exc:
            if required_cps is None or group["geometry"].cp_wave_samples in required_cps:
                raise
            unavailable_priors[key] = {"geometry": asdict(group["geometry"]), "frames": group["frames"],
                                      "reason": str(exc), "required_for_selected_suite": False}
    if required_cps is not None and not set(required_cps) <= set(split["geometry"].cp_wave_samples for split in splits.values()):
        raise ValueError("A required suite CP has no independent statistical prior; add declared independent statistical calibration")
    candidate_dir = output / "calibration_candidates"; candidate_dir.mkdir(exist_ok=True)
    bank_dir = output / "covariance"; bank_dir.mkdir(exist_ok=True)
    prior_records, candidate_records = {}, {method: [] for method in C.BEM_METHODS}
    previous_threads = torch.get_num_threads()
    try:
        with torch.inference_mode():
            choice = _policy(policy, C.STATISTICAL_METHOD)
            torch.set_num_threads(int(choice["cpu_threads"])); stat_device = choice["preferred_device"]
            for key, split in sorted(splits.items()):
                record_path = candidate_dir / f"statistical_{key}.json"
                if record_path.exists():
                    record = P.read_json(record_path)
                    if record.get("status") != "complete" or record.get("contract") != contract:
                        raise RuntimeError("Changed statistical calibration protocol")
                    file_path = output / record["npz_relative_path"]
                    if not file_path.is_file() or P.file_hash(file_path) != record["sha256"]:
                        raise RuntimeError("Stored covariance state checksum mismatch")
                    prior_records[key] = record
                    continue
                print(f"[covariance-calibration] geometry={key} fit={split['fit_frames']} holdout={split['holdout_frames']}", flush=True)
                try:
                    record = _statistical_geometry_result(split, manifest, noise_grid, stat_device, output, key, contract)
                except torch.cuda.OutOfMemoryError as exc:
                    if not str(stat_device).startswith("cuda"):
                        raise
                    previous_device = stat_device
                    backup = _qualified_cpu_fallback(policy, C.STATISTICAL_METHOD, output)
                    stat_device = "cpu"; torch.set_num_threads(int(backup["cpu_threads"]))
                    gc.collect(); torch.cuda.empty_cache()
                    record = _statistical_geometry_result(split, manifest, noise_grid, stat_device, output, key, contract)
                    record["execution_fallback"] = {"from": previous_device, "to": "cpu", "reason": "CUDA OOM",
                                                     "precision_and_candidates_unchanged": True,
                                                     "complete_geometry_recomputed": True, "error": str(exc)}
                P.write_json(record_path, record); prior_records[key] = record
                gc.collect()
                if str(stat_device).startswith("cuda"):
                    torch.cuda.empty_cache()
            flattened = [(arrays, frame, C.geometry_from_export(manifest, arrays, frame))
                         for arrays in arrays_list for frame in range(arrays["x_pilot"].shape[0])]
            for method in C.BEM_METHODS:
                choice = _policy(policy, method)
                torch.set_num_threads(int(choice["cpu_threads"])); device = choice["preferred_device"]
                for q in q_grid:
                    for delay_start, delay_stop in delay_windows:
                        settings = [_setting(method, manifest, {"q": int(q), "delay_start": int(delay_start),
                                      "delay_stop": int(delay_stop), "relative_ridge": float(rho)}) for rho in ridge_grid]
                        paths = [candidate_dir / f"{method}_{P.digest(asdict(setting))[:16]}.json" for setting in settings]
                        if all(path.exists() for path in paths):
                            candidate_records[method].extend(_validated_complete_candidate(P.read_json(path), contract, len(flattened)) for path in paths)
                            continue
                        block_id = P.digest([asdict(value) for value in settings])[:16]
                        progress_path = candidate_dir / f"{method}_{block_id}_progress.json"
                        progress = P.read_json(progress_path) if progress_path.exists() else {
                            "status": "in_progress", "contract": contract,
                            "settings": [asdict(value) for value in settings], "processed_frames": 0,
                            "frame_nmse": [[] for _ in settings], "solver_residual_max": [0.0 for _ in settings],
                            "frame_backends": [], "execution_fallbacks": []}
                        if progress.get("contract") != contract or progress.get("settings") != [asdict(value) for value in settings]:
                            raise RuntimeError("Candidate progress belongs to changed input/code/grid")
                        offset = int(progress["processed_frames"])
                        complete_chunk_boundaries = {0, *np.cumsum([arrays["x_pilot"].shape[0] for arrays in arrays_list]).tolist()}
                        if (offset not in complete_chunk_boundaries or offset > len(flattened)
                                or len(progress["frame_nmse"]) != len(settings)
                                or len(progress["solver_residual_max"]) != len(settings)
                                or len(progress.get("frame_backends", [])) != offset
                                or any(len(values) != offset or not np.all(np.isfinite(values)) for values in progress["frame_nmse"])):
                            raise RuntimeError("Incomplete/corrupt candidate progress")
                        print(f"[BEM-calibration] {method} Q={q} delay=[{delay_start},{delay_stop}) frames={offset}/128", flush=True)
                        operators = {}
                        try:
                            for index in range(offset, len(flattened)):
                                arrays, frame, geometry = flattened[index]
                                key = C.geometry_key(geometry)
                                # At most one operator per CP. One frame's design is
                                # shared by all five ridges, then released.
                                try:
                                    if key not in operators:
                                        operators[key] = C.TorchJointBEM(geometry, settings[0], arrays["x_pilot"].shape[1], device=device, dtype=PRECISION)
                                    operator = operators[key]
                                    values, residuals = _bem_frame_ridges(operator, arrays, frame, settings)
                                except torch.cuda.OutOfMemoryError as exc:
                                    if not str(device).startswith("cuda"):
                                        raise
                                    previous_device = device
                                    backup = _qualified_cpu_fallback(policy, method, output)
                                    device = "cpu"; torch.set_num_threads(int(backup["cpu_threads"]))
                                    operators.clear(); operator = None; gc.collect(); torch.cuda.empty_cache()
                                    operator = C.TorchJointBEM(geometry, settings[0], arrays["x_pilot"].shape[1], device=device, dtype=PRECISION)
                                    operators[key] = operator
                                    values, residuals = _bem_frame_ridges(operator, arrays, frame, settings)
                                    progress.setdefault("execution_fallbacks", []).append({"frame_index": index,
                                        "from": previous_device, "to": "cpu", "reason": "CUDA OOM", "error": str(exc),
                                        "precision_and_candidates_unchanged": True, "all_frame_ridges_recomputed": True})
                                for ri, value in enumerate(values):
                                    progress["frame_nmse"][ri].append(value)
                                    progress["solver_residual_max"][ri] = max(progress["solver_residual_max"][ri], residuals[ri])
                                progress["processed_frames"] = index + 1
                                progress["frame_backends"].append(str(device))
                                # Only completed original chunks become resumable.
                                if frame + 1 == arrays["x_pilot"].shape[0]:
                                    P.write_json(progress_path, progress)
                        except (C.SolverConvergenceError, torch.cuda.OutOfMemoryError) as exc:
                            P.write_json(candidate_dir / f"{method}_{block_id}_failure.json",
                                         {"status": "failed", "contract": contract, "settings": progress["settings"],
                                          "processed_frames": progress["processed_frames"], "error": str(exc),
                                          "selection_permitted": False})
                            raise
                        for ri, (setting, path) in enumerate(zip(settings, paths)):
                            values = progress["frame_nmse"][ri]
                            record = {"status": "complete", "contract": contract, "settings": asdict(setting),
                                      "frames": len(values), "completed_frames": len(values),
                                      "frame_nmse": values, "mean_linear_nmse": float(np.mean(values)),
                                      "solver_residual_max": progress["solver_residual_max"][ri], "calibration_device": device,
                                      "backend_frame_counts": dict(Counter(progress["frame_backends"])),
                                      "execution_fallbacks": progress.get("execution_fallbacks", [])}
                            _validated_complete_candidate(record, contract, len(flattened))
                            if path.exists():
                                record = _validated_complete_candidate(P.read_json(path), contract, len(flattened))
                            else:
                                P.write_json(path, record)
                            candidate_records[method].append(record)
                        del operators; gc.collect()
                        if str(device).startswith("cuda"):
                            torch.cuda.empty_cache()
            selected_settings = {method: min(records, key=lambda value: value["mean_linear_nmse"])["settings"]
                                 for method, records in candidate_records.items()}
            lock = {"schema_version": 1, "status": "locked", "methods": list(C.METHODS),
                    "precision": PRECISION, "code_sha256": codes, **identity,
                    "calibration_seed": identity["calibration_seed"], "calibration_count": identity["calibration_samples"],
                    "statistical_calibration_samples": statistical_identity["frames"],
                    "statistical_calibration_sources": statistical_identity,
                    "contract": contract, "settings": selected_settings,
                    "statistical_priors": prior_records, "unavailable_statistical_priors": unavailable_priors,
                    "candidates": candidate_records,
                    "selection_rule": "minimum mean linear per-frame NMSE on complete independent calibration only; deterministic declared grid order breaks ties",
                    "statistical_selection_rule": "whole-generation-chunk holdout per public numerology; noise scale selected on holdout, prior refitted on all same-numerology calibration frames"}
            return _exclusive_signed_json(lock_path, lock)
    finally:
        torch.set_num_threads(previous_threads)


def load_calibrated_bank(output, device, dtype=PRECISION):
    """Verify locked covariance files and restore a prior for each public geometry."""
    _, C = _runtime()
    output = Path(output).resolve(); lock = P.signed_json(output / "calibration_lock.json")
    if lock.get("status") != "locked" or lock.get("precision") != dtype:
        raise RuntimeError("Calibration bank is not locked at the requested precision")
    if lock.get("code_sha256") != P.source_hashes():
        raise RuntimeError("Correction source changed after immutable prior calibration")
    bank = {}
    for key, record in lock["statistical_priors"].items():
        path = (output / record["npz_relative_path"]).resolve(); path.relative_to(output)
        if P.file_hash(path) != record["sha256"]:
            raise RuntimeError(f"Locked covariance checksum mismatch: {key}")
        geometry = _geometry_restore(record["geometry"])
        if C.geometry_key(geometry) != key:
            raise RuntimeError("Covariance geometry key mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            prior = C.PilotCovarianceLMMSE(arrays["frequency_covariance"], arrays["temporal_correlation"],
                geometry, device=device, dtype=dtype, calibration_info=record["calibration_info"])
        bank[key] = {"prior": prior, "noise_scale": float(record["noise_scale"])}
    return bank
