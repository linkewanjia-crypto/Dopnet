"""Server benchmark of existing checkpoints from received observations to bits.

The existing frozen export has occupied-band frequency-domain pilots and data,
not ADC samples. An optional separately verified time-domain capture adds real
CP removal/full-band FFT/active-bin selection. Every row states its boundary.
The benchmark includes pilot LS/interpolation, the particular
ADD observation consumed by each model, channel estimation, equalization and
hard demapping. Host-origin timing additionally includes H2D and decoded-bit
D2H transfers. It never generates a channel or trains a model.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

METHODS = ("dopnet", "dopnet_lite", "slow_time", "fortitran")
SEEDS = (42, 43, 44)
RAW_KEYS = ("y_pilot", "x_pilot", "rx_data", "noise_var")
NUMERICAL_RELATIVE_L2_TOL = 5e-4
MAX_BIT_DISAGREEMENT = 1e-4
PCG_MAX_ITER = 96
PCG_TOL = 1e-6
MAX_ACCEPTED_RESIDUAL = 1e-4


def _torch():
    import torch
    return torch


def _runtime_import_path():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def implementation_hashes():
    """Bind an automatic device decision to the public receiver implementation."""
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve()]
    for folder in ("models", "data", "utils"):
        files.extend((root / folder).rglob("*.py"))
    return {path.relative_to(root).as_posix(): _file_hash(path) for path in sorted(set(files))}


def _validate_policy_report(policy, report, current_implementation_hashes):
    """Only newly measured, implementation-bound policies enable automatic use."""
    if not report.get("complete"):
        raise RuntimeError("Latency report is incomplete")
    protocol = report["protocol"]
    if (policy.get("experiment_signature") != protocol.get("experiment_signature")
            or policy.get("source_export_signature") != protocol.get("input_export_signature")
            or policy.get("recommendations") != report.get("device_recommendations")):
        raise RuntimeError("Inference policy identities or recommendations differ from its measured report")
    recorded = protocol.get("implementation_sha256")
    if recorded is None:
        raise RuntimeError("Historical latency report lacks implementation binding; remeasure before automatic device selection")
    if recorded != current_implementation_hashes:
        raise RuntimeError("Receiver implementation changed after latency timing; remeasure on this server")


def _safe_child(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def _physics_identity(cfg):
    """Inference geometry, independent of data-generator emission/performance flags."""
    system = dict(cfg["system"])
    # These set simulator working-memory policy, not an observed-frame geometry.
    for key in ("threegpp_link_batch_size", "threegpp_delay_support_margin"):
        system.pop(key, None)
    return system


def _load_sources(observations, results, cell_ids, latency_capture=None):
    """Load only completed exported observations; checksum before using each shard."""
    observations, results = Path(observations).resolve(), Path(results).resolve()
    export = _json(observations / "export_manifest.json")
    experiment = _json(results / "experiment.json")
    if export.get("schema_version") != 1:
        raise RuntimeError("Unsupported frozen-observation export schema")
    provenance = export.get("reference_provenance", {})
    if provenance.get("reference_experiment_signature") != experiment["signature"]:
        raise RuntimeError("Observations and model results originate from different experiments")
    entries = []
    if cell_ids:
        cells = {cell["cell_id"]: cell for cell in export["cells"]}
        for cid in cell_ids:
            if cid not in cells:
                raise ValueError(f"Cell is absent from frozen export: {cid}")
            marker_path = _safe_child(observations, f"cells/{cid}/COMPLETE.json")
            entries.append((cid, marker_path, "test", cells[cid]))
    else:
        entries.append(("calibration", observations / "calibration" / "COMPLETE.json",
                        "calibration", None))
    sources = []
    for label, marker_path, role, expected_meta in entries:
        if not marker_path.is_file():
            raise FileNotFoundError(
                f"Need completed existing observations: {marker_path}. "
                "The cells/*.npz score files alone are not receiver inputs.")
        marker = _json(marker_path)
        if marker.get("role") != role or marker.get("export_signature") != export["export_signature"]:
            raise RuntimeError(f"Invalid observation completion marker: {marker_path}")
        if role == "test" and marker.get("meta") != expected_meta:
            raise RuntimeError(f"Cell metadata mismatch: {label}")
        if role == "calibration" and marker.get("seed") != export.get("calibration_seed"):
            raise RuntimeError("Calibration seed mismatch")
        if not marker.get("batches"):
            raise RuntimeError("Completed export has no observation shards")
        # A fixed existing shard is enough for latency. Larger batches repeat its
        # frames, explicitly recorded, and are not used as statistical samples.
        record = marker["batches"][0]
        path = _safe_child(observations, record["path"])
        if _file_hash(path) != record["sha256"]:
            raise RuntimeError(f"Frozen observation checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            wanted = RAW_KEYS + ("pilot_visible", "data_visible", "cp_wave_samples",
                                 "waveform_fft_size", "subcarrier_spacing_hz", "metadata")
            arrays = {key: archive[key] for key in wanted}
        metadata = json.loads(str(arrays.pop("metadata").item()))
        arrays["condition"] = metadata["condition"]
        _validate_arrays(arrays, export["geometry"])
        sources.append({"name": label, "role": role, "arrays": arrays,
                        "path": str(path), "sha256": record["sha256"],
                        "source_samples": int(record["samples"]),
                        "condition": metadata["condition"],
                        "realization_sha256": marker.get("realization_sha256")})
    if latency_capture is not None:
        path = Path(latency_capture).resolve()
        with np.load(path, allow_pickle=False) as archive:
            keys = RAW_KEYS + ("pilot_visible", "data_visible", "cp_wave_samples",
                               "waveform_fft_size", "subcarrier_spacing_hz", "metadata",
                               "rx_time", "rx_time_backend", "rx_time_l_min", "rx_time_num_blocks")
            arrays = {key: archive[key] for key in keys}
        metadata = json.loads(str(arrays.pop("metadata").item()))
        arrays["condition"] = metadata["condition"]
        sidecar = _json(path.with_suffix(".json")) if path.with_suffix(".json").is_file() else {}
        signature = (metadata.get("reference_experiment_signature") or
                     sidecar.get("reference_experiment_signature"))
        if signature != experiment["signature"]:
            raise RuntimeError("Time-domain capture has no matching experiment signature")
        checksum = _file_hash(path)
        if sidecar.get("capture_sha256") and sidecar["capture_sha256"] != checksum:
            raise RuntimeError("Time-domain capture sidecar checksum mismatch")
        _validate_arrays(arrays, export["geometry"])
        time_input = arrays["rx_time"]
        if time_input.ndim != 3 or time_input.dtype != np.complex64 or not np.all(np.isfinite(time_input)):
            raise ValueError("Time-domain input must contain actual finite [B,R,sample] complex64 observations")
        if time_input.shape[:2] != (arrays["y_pilot"].shape[0], arrays["y_pilot"].shape[2]):
            raise ValueError("Time-domain and OFDM-grid captures have different frame/antenna dimensions")
        backend = str(np.asarray(arrays["rx_time_backend"]).item())
        if backend not in ("synthetic", "sionna"):
            raise ValueError("Unknown original time-domain demodulator backend")
        required = int(np.asarray(arrays["rx_time_num_blocks"]).item()) * (
            int(np.asarray(arrays["waveform_fft_size"]).item()) + int(np.asarray(arrays["cp_wave_samples"]).item()))
        if time_input.shape[-1] < required:
            raise ValueError("Time-domain capture does not contain every visible OFDM block")
        sources.append({"name": "time_capture", "role": metadata.get("role", "latency_capture"),
                        "arrays": arrays, "path": str(path), "sha256": checksum,
                        "source_samples": int(time_input.shape[0]), "condition": metadata["condition"],
                        "realization_sha256": metadata.get("reference_realization_sha256")})
    return export, experiment, sources


def _validate_arrays(arrays, geometry):
    x, y, data, nv = (arrays[key] for key in ("x_pilot", "y_pilot", "rx_data", "noise_var"))
    if x.ndim != 4 or y.ndim != 4 or data.ndim != 4 or nv.ndim != 1:
        raise ValueError("Expected exported batched occupied-grid observations")
    b, u, p, n = x.shape
    r, d = int(geometry["n_ant"]), len(arrays["data_visible"])
    if u != int(geometry["n_users"]) or n != int(geometry["n_subcarriers"]):
        raise ValueError("Observation user/carrier geometry mismatch")
    if y.shape != (b, p, r, n) or data.shape != (b, d, r, n) or nv.shape != (b,):
        raise ValueError("Incompatible exported receiver-grid shapes")
    if not all(np.all(np.isfinite(value)) for value in (x, y, data, nv)) or np.any(nv < 0):
        raise ValueError("Receiver observations must be finite with nonnegative noise variance")
    if any(value.dtype != np.complex64 for value in (x, y, data)) or nv.dtype != np.float32:
        raise ValueError("Benchmark inputs must preserve complex64/fp32 precision")
    spacing = int(arrays["condition"]["pilot_spacing"])
    if n % spacing or not np.all(np.count_nonzero(x, axis=-1) == n // spacing):
        raise ValueError("Known transmitted pilots do not have the declared comb density")
    indices = np.sort(np.where(x != 0, np.arange(n), n), axis=-1)[..., :n // spacing]
    if np.any(np.diff(indices, axis=-1) != spacing):
        raise ValueError("Known pilot grid is not a uniform user comb")
    visible = sorted(list(map(int, arrays["pilot_visible"])) + list(map(int, arrays["data_visible"])))
    if visible != list(range(int(geometry["frame_num_symbols"]))):
        raise ValueError("Pilot/data grids must cover every visible OFDM symbol exactly once")


def _batch_raw(source, batch_size):
    arrays = source["arrays"]
    count = arrays["y_pilot"].shape[0]
    indices = np.arange(batch_size) % count
    torch = _torch()
    keys = ("rx_time", "x_pilot", "noise_var") if "rx_time" in arrays else RAW_KEYS
    tensors = {key: torch.from_numpy(np.ascontiguousarray(arrays[key][indices])) for key in keys}
    h = hashlib.sha256()
    for key in keys:
        a = tensors[key].numpy()
        h.update(key.encode("ascii"))
        h.update(str(a.dtype).encode("ascii"))
        h.update(json.dumps(list(a.shape)).encode("ascii"))
        h.update(a.tobytes())
    return tensors, h.hexdigest(), min(count, batch_size)


def _to_device(raw, device):
    return {key: value.to(device=device, non_blocking=False) for key, value in raw.items()}


def demodulate_time_capture(raw, cfg, arrays):
    """Original CP removal/full FFT/active-bin selection on actual received samples."""
    if "rx_time" not in raw:
        return raw
    torch = _torch()
    from data.channel_backend import active_sionna_grid_indices
    from data.frame_layout import make_layout
    from data.generator import _ofdm_demodulate_torch
    nfft = int(np.asarray(arrays["waveform_fft_size"]).item())
    cp = int(np.asarray(arrays["cp_wave_samples"]).item())
    blocks = int(np.asarray(arrays["rx_time_num_blocks"]).item())
    n = int(cfg["system"]["n_subcarriers"])
    backend = str(np.asarray(arrays["rx_time_backend"]).item())
    samples = raw["rx_time"]
    if backend == "synthetic":
        grid = _ofdm_demodulate_torch(samples, blocks, nfft, cp, n)
    elif backend == "sionna":
        useful = samples[..., :blocks * (nfft + cp)].reshape(*samples.shape[:-1], blocks, nfft + cp)[..., cp:]
        transformed = torch.fft.fft(useful, dim=-1, norm="ortho")
        l_min = int(np.asarray(arrays["rx_time_l_min"]).item())
        phase = -2.0 * math.pi * float(l_min) / float(nfft) * torch.arange(nfft, dtype=torch.float32, device=samples.device)
        transformed = transformed * torch.exp(torch.complex(torch.zeros_like(phase), phase))
        transformed = torch.fft.fftshift(transformed, dim=-1)
        indices = torch.as_tensor(active_sionna_grid_indices(nfft, n), device=samples.device)
        grid = transformed.index_select(-1, indices)
    else:
        raise ValueError("Unknown time-domain capture backend")
    layout = make_layout(int(cfg["system"]["frame_num_symbols"]), int(cfg["system"]["pilot_symbol_period"]))
    pilot_blocks = torch.as_tensor(layout.pilot_blocks, device=samples.device)
    data_blocks = torch.as_tensor(layout.data_blocks, device=samples.device)
    return {"y_pilot": grid.index_select(2, pilot_blocks).permute(0, 2, 1, 3).contiguous(),
            "rx_data": grid.index_select(2, data_blocks).permute(0, 2, 1, 3).contiguous(),
            "x_pilot": raw["x_pilot"], "noise_var": raw["noise_var"]}


def _check_time_demodulation(raw, cfg, arrays, batch_size):
    if "rx_time" not in raw:
        return None
    rebuilt = demodulate_time_capture(raw, cfg, arrays)
    idx = np.arange(batch_size) % arrays["y_pilot"].shape[0]
    checks = {}
    for key in ("y_pilot", "rx_data"):
        value = rebuilt[key].detach().cpu().numpy()
        original = arrays[key][idx]
        discrepancy = float(np.max(np.abs(value - original)))
        passed = bool(np.allclose(value, original, rtol=3e-5, atol=3e-5))
        checks[key] = {"passed": passed, "maximum_absolute_discrepancy": discrepancy}
        if not passed:
            raise RuntimeError(f"Actual-sample demodulation cannot reconstruct saved {key}: {discrepancy:g}")
    return checks


def preprocess_observed_grid(raw, cfg, arrays):
    """Rebuild exactly the observable inference frontend, without channel targets."""
    torch = _torch()
    from data.add_geometry import add_delay_bins, delay_guard_bins, delay_window_indices
    from data.frame_layout import make_layout
    from data.waveform import _pilot_ls_and_interp, _time_features, _add_backprojection
    raw = demodulate_time_capture(raw, cfg, arrays)
    system = cfg["system"]
    b, u, p, n = raw["x_pilot"].shape
    r = raw["y_pilot"].shape[2]
    device = raw["x_pilot"].device
    layout = make_layout(int(system["frame_num_symbols"]), int(system["pilot_symbol_period"]))
    if tuple(map(int, arrays["pilot_visible"])) != layout.pilot_visible or tuple(map(int, arrays["data_visible"])) != layout.data_visible:
        raise RuntimeError("Checkpoint and observed symbol layouts differ")
    if int(np.asarray(arrays["waveform_fft_size"]).item()) != int(system["waveform_fft_size"]) or float(np.asarray(arrays["subcarrier_spacing_hz"]).item()) != float(system["subcarrier_spacing_hz"]):
        raise RuntimeError("Observed numerology differs from trained model")
    spacing = int(arrays["condition"]["pilot_spacing"])
    cp_wave = int(np.asarray(arrays["cp_wave_samples"]).item())
    pb = torch.as_tensor(layout.pilot_blocks, device=device, dtype=torch.long)
    db = torch.as_tensor(layout.data_blocks, device=device, dtype=torch.long)
    pv = torch.as_tensor(layout.pilot_visible, device=device, dtype=torch.long)
    received = raw["y_pilot"].new_zeros((b, r, layout.num_blocks, n))
    received.index_copy_(2, pb, raw["y_pilot"].permute(0, 2, 1, 3))
    received.index_copy_(2, db, raw["rx_data"].permute(0, 2, 1, 3))
    # No unknown transmitted payload symbols are used by this frontend.
    frame = raw["x_pilot"].new_zeros((b, u, layout.num_blocks, n))
    frame.index_copy_(2, pb, raw["x_pilot"])
    bins = torch.arange(n, device=device).view(1, 1, 1, n).expand(b, u, p, n)
    indices = bins.masked_fill(raw["x_pilot"] == 0, n).sort(dim=-1).values[..., :n // spacing]
    values = raw["x_pilot"].gather(-1, indices)
    coarse, _ = _pilot_ls_and_interp(received, layout, indices, values, u, r, n,
                                     spacing, want_ls_grid=False)
    pilot_mask = torch.zeros((b, u, layout.visible_symbols, n), device=device)
    pilot_mask.index_copy_(2, pv, raw["x_pilot"].ne(0).float())
    age, horizon, *_ = _time_features(cfg, b, u, layout.visible_symbols, r, n, cp_wave, device)
    mask = pilot_mask[:, :, :, None, :].expand(b, u, layout.visible_symbols, r, n)
    batch = {"input": torch.stack((coarse.real, coarse.imag, mask, age, horizon), dim=2).float(),
             "noise_var": raw["noise_var"],
             "rx_data": torch.stack((raw["rx_data"].real, raw["rx_data"].imag), dim=1).float()}
    if str(cfg["model"].get("estimator", "dopnet")).lower() == "dopnet":
        profile = str(cfg["model"].get("ablation_profile", "full"))
        if profile != "aft_only":
            indices = delay_window_indices(n, add_delay_bins(cfg), delay_guard_bins(cfg), device=device)
            slow = profile == "no_fast_time_add"
            pilot_only = profile == "pilot_only_fast_time"
            key = "add_input_slow" if slow else ("add_input_pilot_only_fast_time" if pilot_only else "add_input")
            batch[key] = _add_backprojection(cfg, received, frame, layout, cp_wave, indices,
                                            per_user=False, fast_time=not slow,
                                            pilot_only_received=pilot_only)
    return batch


def _estimate(model, cfg, batch, mode):
    from models.receiver import ici_bands_from_dopnet_aux, select_data_csi, select_data_ici_bands
    kwargs = {key: batch.get(key) for key in ("add_input", "add_input_slow",
              "add_input_pilot_only_fast_time", "noise_var", "ls_grid")}
    if mode == "common_diagonal":
        h = model(batch["input"], **kwargs)
        return h, select_data_csi(h, cfg)
    h, aux = model(batch["input"], return_aux=True, **kwargs)
    bands = ici_bands_from_dopnet_aux(model, batch["input"], h, aux,
                                     half_band=2, replace_center=True)
    return h, select_data_ici_bands(bands, cfg)


def _equalize(estimate, batch, mode):
    from models.receiver import linear_multiuser_equalize, banded_multiuser_equalize
    if mode == "common_diagonal":
        z, _ = linear_multiuser_equalize(estimate, batch["rx_data"], batch["noise_var"], method="mmse")
        return z, None
    z, _, info = banded_multiuser_equalize(estimate, batch["rx_data"], batch["noise_var"],
                                         solver="pcg", max_iter=PCG_MAX_ITER,
                                         tol=PCG_TOL, return_info=True)
    pcg_iterations = info["iterations"]
    residual = info["relative_residual_max"]
    fallback = not np.isfinite(residual) or residual > MAX_ACCEPTED_RESIDUAL
    if fallback:
        z, _, info = banded_multiuser_equalize(estimate, batch["rx_data"], batch["noise_var"],
                                             solver="direct", return_info=True)
    if not np.isfinite(info["relative_residual_max"]) or info["relative_residual_max"] > MAX_ACCEPTED_RESIDUAL:
        raise RuntimeError(f"ICI equalizer failed its actual residual check: {info}")
    return z, {**info, "direct_fallback": bool(fallback), "pcg_iterations": pcg_iterations}


def _demap(symbols, constellation, shifts):
    torch = _torch()
    distances = (symbols[..., None] - constellation).abs().square()
    labels = distances.argmin(dim=-1)
    bits = ((labels[..., None] >> shifts) & 1).to(torch.uint8)
    return labels, bits


def _constellation(modulation, device):
    torch = _torch()
    from utils.modem import labels_to_symbols, bits_per_symbol, modulation_order
    points = labels_to_symbols(np.arange(modulation_order(modulation)), modulation)
    constellation = torch.as_tensor(points, dtype=torch.complex64, device=device)
    shifts = torch.arange(bits_per_symbol(modulation) - 1, -1, -1, device=device)
    return constellation, shifts


def _chain(model, cfg, raw, arrays, mode, constellation, shifts, host_output=False,
           diagnostic=False):
    batch = preprocess_observed_grid(raw, cfg, arrays)
    h, estimate = _estimate(model, cfg, batch, mode)
    symbols, info = _equalize(estimate, batch, mode)
    labels, bits = _demap(symbols, constellation, shifts)
    if diagnostic:
        return {"cfr": h.detach().cpu().numpy(), "symbols": symbols.detach().cpu().numpy(),
                "labels": labels.detach().cpu().numpy(), "bits": bits.detach().cpu().numpy(),
                "equalizer": info}
    if host_output:
        bits = bits.cpu()
    return bits, info


def _synchronize(device):
    torch = _torch()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cpu_memory():
    memory = {}
    try:
        import psutil
        memory["process_rss_mb"] = psutil.Process().memory_info().rss / 2**20
    except ImportError:
        memory["process_rss_mb"] = None
    if sys.platform != "win32":
        import resource
        # Linux reports KiB; macOS reports bytes. This is a process lifetime
        # high-water mark, not a tensor-allocation peak for the measured stage.
        scale = 2**20 if sys.platform == "darwin" else 1024
        memory["process_lifetime_max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale
    return memory


def summarize_times(milliseconds, batch_size):
    values = np.asarray(milliseconds, dtype=np.float64)
    if values.size < 2 or np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("Need at least two finite positive timed repetitions")
    mean = float(values.mean())
    return {"latency_ms_mean": mean, "latency_ms_p50": float(np.quantile(values, .50)),
            "latency_ms_p95": float(np.quantile(values, .95)),
            "latency_ms_std": float(values.std(ddof=1)),
            "frames_per_second": 1000.0 * batch_size / mean,
            "amortized_ms_per_frame": mean / batch_size,
            "repetition_latency_ms": values.tolist()}


def _time_operation(fn, device, batch_size, warmups, repetitions, collect_equalizer=False):
    torch = _torch()
    for _ in range(warmups):
        result = fn()
        del result
    _synchronize(device)
    before = _cpu_memory()
    resident = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        resident = torch.cuda.memory_allocated(device)
    times, solver_info = [], []
    for _ in range(repetitions):
        _synchronize(device)
        started = time.perf_counter()
        result = fn()
        _synchronize(device)
        times.append((time.perf_counter() - started) * 1000.0)
        if collect_equalizer and result[1] is not None:
            solver_info.append(result[1])
        del result
    stats = summarize_times(times, batch_size)
    stats["cpu_memory_before"] = before
    stats["cpu_memory_after"] = _cpu_memory()
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        stats.update({"cuda_allocated_before_mb": resident / 2**20,
                      "cuda_peak_allocated_mb": peak / 2**20,
                      "cuda_incremental_peak_mb": max(0, peak - resident) / 2**20})
    if collect_equalizer:
        stats.update({"direct_fallback_repetitions": sum(item["direct_fallback"] for item in solver_info),
                      "pcg_iterations_mean": statistics.mean(item["pcg_iterations"] for item in solver_info) if solver_info else None,
                      "native_residual_max": max(item["relative_residual_max"] for item in solver_info) if solver_info else None})
    return stats


def compare_outputs(reference, candidate):
    """Per-frame numerical checks; repeats are not independent validation samples."""
    rows = []
    b = reference["cfr"].shape[0]
    for frame in range(b):
        row = {"frame_index": frame}
        for key in ("cfr", "symbols"):
            a, v = reference[key][frame], candidate[key][frame]
            row[f"{key}_relative_l2"] = float(np.linalg.norm((v - a).ravel()) /
                                               max(np.linalg.norm(a.ravel()), 1e-12))
            row[f"{key}_max_abs"] = float(np.max(np.abs(v - a)))
        row["decoded_bit_disagreement_rate"] = float(np.mean(reference["bits"][frame] != candidate["bits"][frame]))
        row["finite"] = bool(np.all(np.isfinite(candidate["cfr"][frame])) and np.all(np.isfinite(candidate["symbols"][frame])))
        row["passed"] = bool(row["finite"] and row["cfr_relative_l2"] <= NUMERICAL_RELATIVE_L2_TOL
                             and row["symbols_relative_l2"] <= NUMERICAL_RELATIVE_L2_TOL
                             and row["decoded_bit_disagreement_rate"] <= MAX_BIT_DISAGREEMENT)
        rows.append(row)
    return {"passed": all(row["passed"] for row in rows), "per_frame": rows,
            "relative_l2_tolerance": NUMERICAL_RELATIVE_L2_TOL,
            "maximum_bit_disagreement_rate": MAX_BIT_DISAGREEMENT}


def recommend_devices(rows, expected_seeds=None):
    """Choose separately for each workload using host-origin wall-clock median."""
    groups, attempted = {}, {}
    for row in rows:
        key = (row["method"], row["batch_size"], row["receiver_mode"], row["source_name"])
        if row.get("status") != "not_applicable":
            attempted.setdefault(key, set()).add(row["seed"])
        if row.get("status") != "measured" or not row.get("numerical_equivalence", {}).get("passed"):
            continue
        backend = (row["device"], row.get("cpu_threads"))
        groups.setdefault(key, {}).setdefault(backend, []).append(row)
    decisions = []
    for key, device_groups in sorted(groups.items()):
        required_seeds = attempted[key] if expected_seeds is None else set(expected_seeds)
        eligible = {backend: group for backend, group in device_groups.items()
                    if len(group) == len(required_seeds)
                    and set(row["seed"] for row in group) == required_seeds}
        backend_timings = {backend: statistics.median(row["timing"]["host_origin_total"]["latency_ms_p50"] for row in group)
                           for backend, group in eligible.items()}
        if not backend_timings:
            continue
        # Compare each device using its best complete, qualified thread option.
        best_per_device = {}
        for backend, value in backend_timings.items():
            if backend[0] not in best_per_device or value < backend_timings[best_per_device[backend[0]]]:
                best_per_device[backend[0]] = backend
        timings = {device: backend_timings[backend] for device, backend in best_per_device.items()}
        fastest = min(timings, key=timings.get)
        selected_backend = best_per_device[fastest]
        decisions.append({"method": key[0], "batch_size": key[1], "receiver_mode": key[2],
                          "source_name": key[3], "preferred_device": fastest,
                          "cpu_threads": selected_backend[1],
                          "best_cpu_threads_by_device": {device: backend[1] for device, backend in best_per_device.items()},
                          "qualified_backend_options": [{"device": backend[0], "cpu_threads": backend[1],
                               "host_origin_p50_ms_median_across_model_seeds": value}
                               for backend, value in sorted(backend_timings.items(), key=lambda item: (item[0][0], item[0][1] or 0))],
                          "input_boundary": eligible[selected_backend][0].get("input_boundary", "received_occupied_ofdm_grid"),
                          "host_origin_p50_ms_median_across_model_seeds": timings,
                          "selection_basis": "completed same-input fp32 host-origin median latency; equivalence gate passed",
                          "workload": "single-frame latency" if key[1] == 1 else "batched throughput",
                          "n_model_seeds": len(eligible[selected_backend])})
    return decisions


def select_inference_device(policy_path, method, batch_size, receiver_mode="common_diagonal",
                            source_name="calibration"):
    """Resolve a verified workload policy for a later inference-only runner.

    The consumer must retain fp32/complex64, the recorded CPU thread count and
    the exact receiver mode. An unavailable GPU falls back to CPU; it cannot be
    silently substituted by a different available accelerator.
    """
    path = Path(policy_path).resolve()
    policy = _json(path)
    report_path = path.parent / "digital_receiver_latency.json"
    if not policy.get("complete") or policy.get("precision") != "fp32":
        raise RuntimeError("Inference device policy is incomplete or has another precision")
    if not report_path.is_file() or _file_hash(report_path) != policy.get("latency_report_sha256"):
        raise RuntimeError("Inference policy does not match its verified latency report")
    report = _json(report_path)
    _validate_policy_report(policy, report, implementation_hashes())
    matches = [row for row in policy["recommendations"]
               if row["method"] == method and row["batch_size"] == batch_size
               and row["receiver_mode"] == receiver_mode and row["source_name"] == source_name]
    if len(matches) != 1:
        raise RuntimeError(f"No unique measured device decision for {(method, batch_size, receiver_mode, source_name)}")
    recommendation = dict(matches[0])
    torch = _torch()
    hardware = report["hardware"]
    if hardware.get("torch") != torch.__version__ or hardware.get("python") != platform.python_version() or hardware.get("logical_cpu_count") != os.cpu_count():
        raise RuntimeError("Server runtime or CPU geometry differs from timing; rerun CPU/GPU timing here")
    device = torch.device(recommendation["preferred_device"])
    if device.type == "cuda":
        if not torch.cuda.is_available() or device.index is not None and device.index >= torch.cuda.device_count():
            if "cpu" not in recommendation["host_origin_p50_ms_median_across_model_seeds"]:
                raise RuntimeError("CUDA unavailable and this workload has no fully qualified CPU fallback; remeasure it")
            recommendation["preferred_device"] = "cpu"
            recommendation["cpu_threads"] = recommendation["best_cpu_threads_by_device"]["cpu"]
            recommendation["runtime_fallback"] = "measured CUDA device currently unavailable"
        else:
            recorded = next((row["name"] for row in report["hardware"]["gpus"] if row["device"] == str(device)), None)
            if recorded != torch.cuda.get_device_name(device):
                raise RuntimeError("GPU hardware differs from the benchmark; rerun CPU/GPU timing on this server")
    if recommendation.get("cpu_threads") is None:
        recommendation["cpu_threads"] = report["protocol"]["cpu_threads"]
    recommendation["precision"] = policy["precision"]
    recommendation["experiment_signature"] = policy["experiment_signature"]
    return recommendation


def resolve_device(policy_path, method, batch_size, mode="common_diagonal", source="calibration"):
    """Short adapter API; returns the qualified device name and policy context."""
    return select_inference_device(policy_path, method, batch_size, mode, source)


def infer_observed_batch(arrays, cfg, state_or_model, *, device="cpu", method=None,
                         mode="common_diagonal", microbatch_size=None, cpu_threads=None):
    """Inference only on already exported observations, with transparent OOM splits.

    Returns complex64 data-symbol CFR/symbols, hard labels/bits and the complete
    original input digest. A split is recorded as effective_microbatch_size and
    never presented as an unsplit B=32 timing measurement.
    """
    _runtime_import_path()
    torch = _torch()
    from models.receiver import SharedUserChannelEstimator, select_data_csi
    if mode not in ("common_diagonal", "native_own_ici"):
        raise ValueError("Unsupported receiver mode")
    processing = torch.device(device)
    if processing.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; resolve a CPU fallback policy first")
    if processing.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU/CUDA devices are supported")
    if cpu_threads is not None:
        if int(cpu_threads) < 1:
            raise ValueError("cpu_threads must be positive")
        torch.set_num_threads(int(cpu_threads))
    if "condition" not in arrays:
        arrays = dict(arrays)
        metadata = json.loads(str(np.asarray(arrays["metadata"]).item()))
        arrays["condition"] = metadata["condition"]
    geometry = {key: cfg["system"][key] for key in ("n_ant", "n_users", "n_subcarriers", "frame_num_symbols")}
    _validate_arrays(arrays, geometry)
    b = arrays["x_pilot"].shape[0]
    size = b if microbatch_size is None else min(b, int(microbatch_size))
    if size < 1:
        raise ValueError("microbatch_size must be positive")
    host_raw, input_digest, _ = _batch_raw({"arrays": arrays}, b)
    if isinstance(state_or_model, torch.nn.Module):
        model = state_or_model.to(processing).eval()
    else:
        if "cfg" in state_or_model and state_or_model["cfg"] != cfg:
            raise RuntimeError("Checkpoint configuration differs from inference frontend configuration")
        model = SharedUserChannelEstimator(cfg).to(processing).eval()
        model.load_state_dict(state_or_model.get("model", state_or_model), strict=True)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("Inference adapter requires the benchmark's fp32 model precision")
    if mode == "native_own_ici" and not (model.estimator_name == "dopnet" and model.base.use_add_branch):
        raise ValueError("Model does not supply native ADD-to-ICI coefficients")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    constellation, shifts = _constellation(arrays["condition"]["modulation"], processing)
    outputs, infos, offset, reductions = [], [], 0, []
    with torch.inference_mode():
        while offset < b:
            stop = min(b, offset + size)
            try:
                raw = _to_device({key: value[offset:stop] for key, value in host_raw.items()}, processing)
                checks = _check_time_demodulation(raw, cfg,
                    {key: value[offset:stop] if key in RAW_KEYS or key == "rx_time" else value for key, value in arrays.items()}, stop - offset)
                diagnostic = _chain(model, cfg, raw, arrays, mode, constellation, shifts, diagnostic=True)
                cfr_full = torch.from_numpy(diagnostic["cfr"])
                data = select_data_csi(cfr_full, cfg).numpy()
                outputs.append({"cfr_data": (data[:, :, 0] + 1j * data[:, :, 1]).astype(np.complex64),
                                "symbols": diagnostic["symbols"],
                                "predicted_labels": diagnostic["labels"],
                                "predicted_bits": diagnostic["bits"]})
                infos.append({"offset": offset, "samples": stop - offset,
                              "equalizer": diagnostic["equalizer"],
                              "time_demodulation_check": checks})
                offset = stop
                del raw, diagnostic, cfr_full, data
            except torch.cuda.OutOfMemoryError:
                if processing.type != "cuda" or size == 1:
                    raise
                raw = None
                diagnostic = None
                size = max(1, size // 2)
                reductions.append({"offset": offset, "new_microbatch_size": size})
                gc.collect()
                torch.cuda.empty_cache()
    return {**{key: np.concatenate([output[key] for output in outputs], axis=0) for key in outputs[0]},
            "input_digest": input_digest, "device": str(processing), "precision": "fp32/complex64",
            "method": method, "receiver_mode": mode, "requested_batch_size": b,
            "effective_microbatch_size": max(info["samples"] for info in infos),
            "effective_microbatch_sizes": sorted(set(info["samples"] for info in infos)),
            "final_microbatch_limit": size, "oom_microbatch_reductions": reductions,
            "microbatch_info": infos}


def benchmark(observations, results, output, *, seeds=SEEDS, batch_sizes=(1, 8, 32),
              devices=("cpu", "cuda:0"), modes=("common_diagonal", "native_own_ici"),
              warmups=5, repetitions=30, cpu_threads=4, cell_ids=None, latency_capture=None,
              cpu_thread_candidates=None):
    """Compare servers' CPU/GPU using existing independent calibration inputs.

    CPU-origin includes pageable host tensor to GPU copies and decoded-bit copy
    back to CPU. Its total is timed directly, not reconstructed by adding stages.
    Extra --cell-id tests are workload timing only; no model/calibration tuning.
    """
    if warmups < 1 or repetitions < 10 or cpu_threads < 1:
        raise ValueError("Use warmups>=1, repetitions>=10 and cpu_threads>=1")
    thread_options = (int(cpu_threads),) if cpu_thread_candidates is None else tuple(cpu_thread_candidates)
    if not thread_options or len(set(thread_options)) != len(thread_options) or any(int(value) != value or value < 1 for value in thread_options):
        raise ValueError("Use unique positive integer CPU thread candidates")
    if not batch_sizes or len(set(batch_sizes)) != len(batch_sizes) or not set(batch_sizes) <= {1, 8, 32}:
        raise ValueError("Use unique batch sizes from {1,8,32}")
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(SEEDS):
        raise ValueError(f"Use unique completed model seeds from {SEEDS}")
    if len(set(devices)) != len(devices) or "cpu" not in devices:
        raise ValueError("Include CPU once as the numerical reference")
    if not set(modes) <= {"common_diagonal", "native_own_ici"}:
        raise ValueError("Unsupported receiver mode")
    _runtime_import_path()
    torch = _torch()
    from models.receiver import SharedUserChannelEstimator
    from utils.checkpoints import load_checkpoint
    export, experiment, sources = _load_sources(observations, results, cell_ids, latency_capture)
    results, output = Path(results).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = [f"{method}_seed{seed}" for method in METHODS for seed in seeds]
    missing = [name for name in names if name not in experiment["configs"] or not (results / "checkpoints" / name / "model.pt").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing completed model checkpoint/config: {missing}")
    reference_cfg = experiment["configs"][names[0]]
    for name in names:
        if _physics_identity(experiment["configs"][name]) != _physics_identity(reference_cfg):
            raise RuntimeError(f"Observed-frame physical system differs for {name}")
    if _physics_identity({"system": export["cfg_system"]}) != _physics_identity(reference_cfg):
        raise RuntimeError("Frozen observed-frame physical system differs from checkpoints")
    available, skipped_devices = [], []
    for name in ("cpu", *(value for value in devices if value != "cpu")):
        device = torch.device(name)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU/CUDA support the declared benchmark protocol")
        if device.type == "cuda" and (not torch.cuda.is_available() or device.index is not None and device.index >= torch.cuda.device_count()):
            skipped_devices.append({"device": name, "reason": "CUDA device unavailable; CPU remains usable"})
        else:
            available.append(device)
    prior_threads = torch.get_num_threads()
    execution_options = [(device, int(threads)) for device in available
                         for threads in (thread_options if device.type == "cpu" else (cpu_threads,))]
    torch.set_num_threads(cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    report = {"complete": False, "schema_version": 1,
              "protocol": {"scope": "received OFDM grid or separately verified real time-domain samples to decoded hard bits; each row states its boundary",
                           "host_origin_total": "H2D + necessary pilot/ADD frontend + neural estimation + MMSE + demapping + bits D2H; directly timed",
                           "device_resident_total": "same digital chain with input/output tensors resident on processing device",
                           "excluded": ["RF/acquisition", "ADC conversion", "channel simulation", "training", "calibration fitting", "BER/NMSE scoring", "disk input/output"],
                           "ofdm_grid_rows_exclude": ["CP removal", "full-band OFDM FFT"],
                           "time_capture_rows_include": ["CP removal", "full-band OFDM FFT", "Sionna phase compensation if applicable", "active-bin selection"],
                           "precision": "float32/complex64; TF32 disabled; eager inference_mode",
                           "timing": "perf_counter wall clock with CUDA synchronization before and after every repetition; stage times measured separately",
                           "methods": list(METHODS), "seeds": list(seeds), "batch_sizes": list(batch_sizes),
                           "warmups": warmups, "repetitions": repetitions, "cpu_threads": cpu_threads,
                           "cpu_thread_candidates": list(thread_options),
                           "receiver_modes": list(modes), "input_export_signature": export["export_signature"],
                           "experiment_signature": experiment["signature"],
                           "script_sha256": _file_hash(__file__),
                           "implementation_sha256": implementation_hashes(),
                           "sources": [{key: source[key] for key in ("name", "role", "path", "sha256", "source_samples", "condition", "realization_sha256")} for source in sources],
                           "timing_references": ["https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution", "https://docs.pytorch.org/docs/stable/benchmark_utils.html"]},
              "hardware": {"platform": platform.platform(), "cpu": platform.processor(),
                           "logical_cpu_count": os.cpu_count(), "python": platform.python_version(),
                           "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                           "gpus": [{"device": str(device), "name": torch.cuda.get_device_name(device),
                                     "total_memory_mb": torch.cuda.get_device_properties(device).total_memory / 2**20}
                                    for device in available if device.type == "cuda"]},
              "skipped_devices": skipped_devices, "rows": [], "device_recommendations": []}
    report_path = output / "digital_receiver_latency.json"
    if report_path.exists():
        old = _json(report_path)
        if old.get("protocol") != report["protocol"]:
            raise RuntimeError("Existing latency output uses another protocol; select a new output folder")
    _write_json(report_path, report)
    try:
        with torch.inference_mode():
            for method in METHODS:
                for seed in seeds:
                    name = f"{method}_seed{seed}"
                    path = results / "checkpoints" / name / "model.pt"
                    checkpoint = load_checkpoint(path, map_location="cpu")
                    cfg = checkpoint["cfg"]
                    if cfg != experiment["configs"][name] or checkpoint.get("selection_metric") != "val_nmse_db":
                        raise RuntimeError(f"Checkpoint provenance differs for {name}")
                    checkpoint_sha = _file_hash(path)
                    cpu_outputs = {}
                    for device, threads in execution_options:
                        torch.set_num_threads(threads)
                        if device.type == "cuda":
                            torch.cuda.set_device(device)
                        model = SharedUserChannelEstimator(cfg).to(device).eval()
                        model.load_state_dict(checkpoint["model"], strict=True)
                        if any(p.dtype != torch.float32 for p in model.parameters()):
                            raise RuntimeError("All models must use identical fp32 precision")
                        for source in sources:
                            arrays = source["arrays"]
                            constellation, shifts = _constellation(arrays["condition"]["modulation"], device)
                            for size in batch_sizes:
                                host_raw, input_hash, unique_count = _batch_raw(source, size)
                                raw = _to_device(host_raw, device)
                                for mode in modes:
                                    identity = (source["name"], size, mode)
                                    base = {"method": method, "seed": seed, "batch_size": size,
                                            "receiver_mode": mode, "device": str(device), "source_name": source["name"],
                                            "cpu_threads": threads,
                                            "checkpoint_sha256": checkpoint_sha, "input_sha256": input_hash,
                                            "input_boundary": "actual_received_time_samples" if "rx_time" in raw else "received_occupied_ofdm_grid",
                                            "unique_observed_frames": unique_count, "timing_frame_repetition_factor": size / unique_count}
                                    if mode == "native_own_ici" and not (model.estimator_name == "dopnet" and model.base.use_add_branch):
                                        report["rows"].append({**base, "status": "not_applicable", "reason": "model does not produce native ADD-to-ICI coefficients"})
                                        continue
                                    print(f"[digital-latency] {name} {device} threads={threads} {source['name']} B={size} {mode}", flush=True)
                                    try:
                                        demodulation_check = _check_time_demodulation(raw, cfg, arrays, size)
                                        diagnostic = _chain(model, cfg, raw, arrays, mode, constellation, shifts, diagnostic=True)
                                        if device.type == "cpu":
                                            if identity not in cpu_outputs:
                                                cpu_outputs[identity] = diagnostic
                                            equivalence = {**compare_outputs(cpu_outputs[identity], diagnostic),
                                                           "reference_device": "cpu", "comparison": "same-input CPU thread/reference comparison"}
                                        else:
                                            equivalence = compare_outputs(cpu_outputs[identity], diagnostic)
                                        # Standalone stage timing uses identical resident inputs.
                                        grid_raw = demodulate_time_capture(raw, cfg, arrays)
                                        batch = preprocess_observed_grid(grid_raw, cfg, arrays)
                                        h, estimate = _estimate(model, cfg, batch, mode)
                                        z, _ = _equalize(estimate, batch, mode)
                                        _, decoded = _demap(z, constellation, shifts)
                                        operations = {
                                            "host_to_device": lambda: _to_device(host_raw, device),
                                            "pilot_and_add_frontend": lambda: preprocess_observed_grid(grid_raw, cfg, arrays),
                                            "channel_estimation_and_native_band_construction": lambda: _estimate(model, cfg, batch, mode),
                                            "equalization": lambda: _equalize(estimate, batch, mode),
                                            "hard_demapping": lambda: _demap(z, constellation, shifts),
                                            "decoded_bits_to_host": lambda: decoded.cpu(),
                                        }
                                        if "rx_time" in raw:
                                            operations["cp_fft_and_subcarrier_selection"] = lambda: demodulate_time_capture(raw, cfg, arrays)
                                        timing = {stage: _time_operation(fn, device, size, warmups, repetitions)
                                                  for stage, fn in operations.items()}
                                        # Release split-stage temporaries before total memory/time measurements.
                                        del operations, batch, h, estimate, z, decoded, diagnostic, grid_raw
                                        timing["device_resident_total"] = _time_operation(
                                            lambda: _chain(model, cfg, raw, arrays, mode, constellation, shifts),
                                            device, size, warmups, repetitions, collect_equalizer=True)
                                        timing["host_origin_total"] = _time_operation(
                                            lambda: _chain(model, cfg, _to_device(host_raw, device), arrays, mode,
                                                           constellation, shifts, host_output=True),
                                            device, size, warmups, repetitions, collect_equalizer=True)
                                        report["rows"].append({**base, "status": "measured", "numerical_equivalence": equivalence,
                                                              "demodulation_reconstruction_check": demodulation_check, "timing": timing})
                                    except torch.cuda.OutOfMemoryError as exc:
                                        report["rows"].append({**base, "status": "oom", "error": str(exc), "numerical_equivalence": {"passed": False}})
                                        gc.collect()
                                        if device.type == "cuda":
                                            torch.cuda.empty_cache()
                                    report["device_recommendations"] = recommend_devices(report["rows"], expected_seeds=seeds)
                                    _write_json(report_path, report)
                                del raw, host_raw
                        del model
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    del checkpoint, cpu_outputs
        report["complete"] = True
        report["numerical_checks_passed"] = all(row.get("numerical_equivalence", {}).get("passed", True)
                                                for row in report["rows"] if row["status"] == "measured")
        _write_json(report_path, report)
        _write_json(output / "inference_device_policy.json", {"complete": True,
                     "qualification": "all_measured_devices_equivalent" if report["numerical_checks_passed"] else "qualified_devices_only",
                     "numerical_checks_passed": report["numerical_checks_passed"],
                     "excluded_measured_devices": [{key: row[key] for key in ("method", "seed", "device", "batch_size", "receiver_mode", "source_name", "status")}
                         for row in report["rows"] if row["status"] == "oom" or
                         row["status"] == "measured" and not row.get("numerical_equivalence", {}).get("passed")],
                     "latency_report_sha256": _file_hash(report_path),
                     "experiment_signature": experiment["signature"],
                     "source_export_signature": export["export_signature"],
                     "precision": "fp32", "recommendations": report["device_recommendations"]})
        return report
    finally:
        torch.set_num_threads(prior_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    parser.add_argument("--modes", nargs="+", choices=["common_diagonal", "native_own_ici"], default=["common_diagonal", "native_own_ici"])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--cpu-thread-candidates", type=int, nargs="+", default=None,
                        help="Optional measured CPU thread search, e.g. 1 4 8; GPU retains --cpu-threads for host operations")
    parser.add_argument("--latency-capture", type=Path, default=None,
                        help="Optional separately attested NPZ containing actual noisy time-domain receive samples")
    parser.add_argument("--cell-id", action="append", default=None)
    args = parser.parse_args(argv)
    benchmark(args.observations, args.results, args.output, seeds=tuple(args.seeds),
              batch_sizes=tuple(args.batch_sizes), devices=tuple(args.devices), modes=tuple(args.modes),
              warmups=args.warmups, repetitions=args.repetitions, cpu_threads=args.cpu_threads,
              cell_ids=args.cell_id, latency_capture=args.latency_capture,
              cpu_thread_candidates=args.cpu_thread_candidates)


if __name__ == "__main__":
    main()
