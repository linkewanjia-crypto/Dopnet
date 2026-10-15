"""Fail-closed loader for the one archived common validation realization.

The cache is an externally supplied, immutable research artifact.  Its content
hash identifies the historical observations; a matching filename alone does
not.  This module never generates replacement observations.
"""

import copy
import hashlib
import inspect
import json
import math
import re
from pathlib import Path

import torch


HISTORICAL_CACHE_NAME = "validation_895338a79e73ad3f.pt"
HISTORICAL_CACHE_BYTES = 4_682_930_065
HISTORICAL_CACHE_SHA256 = (
    "83d0e03524628b617671c42a73d3c458cc3395c31965a4e22295e8389f56e597"
)
PILOT_FEATURE_SOURCE = "direct_regeneration_after_full_historical_core_comparison"
# Derived from the archived full_seed42.yaml after only the frame-mode rename
# dual_domain_v4 -> dopnet.  All physical/system settings and sampling settings
# are included, but estimator architecture, training seed and optimizer are not.
HISTORICAL_IDENTITY_SHA256 = (
    "5e9eafdedb8ec8ca39e899baea705dcc807033065846f592d7f73082e2e0042e"
)
OBSERVATION_KEYS = (
    "add_doppler_bins", "doppler_whitening_reg",
    "add_delay_bins", "add_delay_guard_bins",
)


def validation_identity(cfg):
    """Describe exactly the configuration fields that made the observations."""
    dp = copy.deepcopy(cfg.get("data_pipeline", {}))
    # All model variants are evaluated on the same complete observations.
    dp.update(emit_pilot_ls=True, emit_slow_time_add=True)
    extra = dp.pop("emit_pilot_only_fast_time_add", False)
    if not isinstance(extra, bool):
        raise ValueError("emit_pilot_only_fast_time_add must be a Boolean")
    # The added deterministic feature is formed from regenerated received Y
    # after all archived core observations pass comparison. It does not alter
    # any archived sample or its sampling distribution.
    return {
        "system": copy.deepcopy(cfg["system"]),
        "val_samples": int(cfg["train"]["val_samples"]),
        "seed": int(cfg["train"].get("validation_seed", cfg.get("seed", 42))),
        "generator": dp,
        "observation": {k: cfg.get("model", {}).get(k) for k in OBSERVATION_KEYS},
    }


def identity_sha256(cfg):
    payload = json.dumps(
        validation_identity(cfg), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_historical_identity(cfg):
    actual = identity_sha256(cfg)
    if actual != HISTORICAL_IDENTITY_SHA256:
        raise RuntimeError(
            "Fixed validation cache is incompatible with the current system, "
            "observation-grid, generator settings, validation seed, or sample "
            f"count. Expected identity {HISTORICAL_IDENTITY_SHA256}; got {actual}."
        )


def _expected_cache_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("--fixed-validation-sha256 must be a 64-digit SHA-256 hex string")
    value = value.lower()
    if value != HISTORICAL_CACHE_SHA256:
        raise RuntimeError(
            "The supplied SHA-256 does not identify the archived historical "
            f"validation cache ({HISTORICAL_CACHE_SHA256})."
        )
    return value


def historical_file_metadata(path):
    stat = Path(path).stat()
    return {
        "cache_size_bytes": int(stat.st_size),
        "cache_mtime_ns": int(stat.st_mtime_ns),
        "cache_ctime_ns": int(stat.st_ctime_ns),
        "cache_device": int(stat.st_dev),
        "cache_inode": int(stat.st_ino),
    }


def assert_historical_file_metadata(path, proof):
    current = historical_file_metadata(path)
    for key, value in current.items():
        if proof.get(key) != value:
            raise RuntimeError(
                f"Historical validation cache metadata changed ({key}); "
                "rerun the full SHA-256 server verifier before training."
            )


def verify_historical_cache_file(path, expected_sha256, verified_proof=None):
    """Verify full SHA once, or use a matching server attestation thereafter."""
    expected = _expected_cache_hash(expected_sha256)
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Fixed validation cache is missing: {path}. Extract it explicitly; "
            "training will not regenerate it."
        )
    if path.stat().st_size != HISTORICAL_CACHE_BYTES:
        raise RuntimeError(
            f"Fixed validation cache size mismatch at {path}: "
            f"expected {HISTORICAL_CACHE_BYTES} bytes, got {path.stat().st_size}."
        )
    if verified_proof is not None:
        if (verified_proof.get("status") != "passed"
                or verified_proof.get("cache_sha256") != expected):
            raise RuntimeError("Historical cache attestation is not valid for this SHA-256")
        assert_historical_file_metadata(path, verified_proof)
        return path
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"Fixed validation cache SHA-256 mismatch at {path}: "
            f"expected {expected}, got {actual}."
        )
    return path


def historical_tensor_schema(cfg):
    """Exact schema inspected from the archived tensor-only PyTorch file."""
    n = int(cfg["train"]["val_samples"])
    return {
        "input": (torch.float32, (n, 2, 5, 25, 8, 128)),
        "target_h": (torch.float32, (n, 2, 2, 25, 8, 128)),
        "rx_data": (torch.float32, (n, 2, 18, 8, 128)),
        "tx_symbol": (torch.float32, (n, 2, 2, 18, 128)),
        "tx_label": (torch.int64, (n, 2, 18, 128)),
        "data_mask": (torch.float32, (n, 25)),
        "pilot_symbol_mask": (torch.float32, (n, 25)),
        "symbol_times_s": (torch.float32, (n, 25)),
        "pilot_age_seconds": (torch.float32, (n, 25)),
        "pilot_horizon_seconds": (torch.float32, (n, 25)),
        "time_feature_scale_s": (torch.float32, (n,)),
        "modulation_id": (torch.int64, (n,)),
        "channel_type_id": (torch.int64, (n,)),
        "noise_var": (torch.float32, (n,)),
        "active_re_signal_power": (torch.float32, (n,)),
        "snr_db": (torch.float32, (n,)),
        "pilot_spacing": (torch.int64, (n,)),
        "cp_len": (torch.int64, (n,)),
        "cp_wave_samples": (torch.int64, (n,)),
        "waveform_fft_size": (torch.int64, (n,)),
        "delay_spread_ns": (torch.float32, (n,)),
        "max_doppler_hz": (torch.float32, (n,)),
        "channel_backend_id": (torch.int64, (n,)),
        "snr_reference_id": (torch.int64, (n,)),
        "target_reference_id": (torch.int64, (n,)),
        "add_input": (torch.float32, (n, 2, 2, 15, 8, 48)),
        "target_add": (torch.float32, (n, 2, 2, 15, 8, 48)),
        "add_input_slow": (torch.float32, (n, 2, 2, 15, 8, 48)),
        "ls_grid": (torch.float32, (n, 2, 2, 25, 8, 128)),
    }


def assert_historical_tensor_schema(data, cfg):
    schema = historical_tensor_schema(cfg)
    if not isinstance(data, dict) or set(data) != set(schema):
        keys = set(data) if isinstance(data, dict) else set()
        raise RuntimeError(
            "Fixed validation tensor keys differ from the archived schema: "
            f"missing={sorted(set(schema) - keys)}, extra={sorted(keys - set(schema))}."
        )
    for name, (dtype, shape) in schema.items():
        tensor = data[name]
        if not torch.is_tensor(tensor) or tensor.device.type != "cpu":
            raise RuntimeError(f"Fixed validation {name} is not a CPU tensor.")
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise RuntimeError(
                f"Fixed validation {name} has dtype/shape "
                f"{tensor.dtype}/{tuple(tensor.shape)}; expected {dtype}/{shape}."
            )


def load_historical_core(cfg, path, expected_sha256, verified_proof=None):
    """Load only the exact trusted historical file; never fall back to generation."""
    assert_historical_identity(cfg)
    path = verify_historical_cache_file(path, expected_sha256, verified_proof)
    parameters = inspect.signature(torch.load).parameters
    if "weights_only" not in parameters:
        raise RuntimeError("Fixed validation cache needs PyTorch with weights_only loading")
    kwargs = {"map_location": "cpu", "weights_only": True}
    if "mmap" in parameters:
        kwargs["mmap"] = True
    try:
        data = torch.load(path, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            "The hash-verified fixed validation cache could not be loaded in "
            "weights_only mode. Check server PyTorch compatibility; no unsafe "
            "deserialization or validation-set regeneration is attempted."
        ) from exc
    if verified_proof is not None:
        assert_historical_file_metadata(path, verified_proof)
    assert_historical_tensor_schema(data, cfg)
    print(f"[validation-cache] FIXED {path} sha256={HISTORICAL_CACHE_SHA256}")
    return data


@torch.no_grad()
def derive_pilot_only_fast_time_add(cfg, batch, device="cpu"):
    """Reconstruct pilot-only ADD from archived receiver-only data for diagnostics.

    ``ls_grid`` is sparse Y/x on the transmitted pilot combs and the pilot mask
    in ``input`` records the randomly assigned comb offsets.  The public
    Zadoff-Chu sequence determines x, hence Y on exactly the union of pilot
    bins required by the pilot-only backprojection.  The divide/multiply round
    trip introduces float32 roundoff which can be amplified by the ADD
    backprojection. The accepted validation sidecar uses direct regenerated Y
    after all archived core tensors have passed their independent comparison.
    """
    from data.add_geometry import (
        add_delay_bins, delay_guard_bins, delay_window_indices,
    )
    from data.frame_layout import make_layout
    from data.waveform import _add_backprojection, _zadoff_chu

    device = torch.device(device)
    n = int(batch["input"].shape[0])
    system = cfg["system"]
    users = int(system["n_users"])
    n_ant = int(system["n_ant"])
    n_sub = int(system["n_subcarriers"])
    visible = int(system["frame_num_symbols"])
    layout = make_layout(visible, int(system["pilot_symbol_period"]))
    pilot_visible = torch.as_tensor(layout.pilot_visible, dtype=torch.long)
    pilot_blocks = torch.as_tensor(layout.pilot_blocks, device=device, dtype=torch.long)
    n_pilot = int(pilot_visible.numel())
    q = int(cfg["model"]["add_doppler_bins"])
    n_delay = add_delay_bins(cfg)
    result = torch.empty((n, users, 2, q, n_ant, n_delay), dtype=torch.float32)
    group_values = torch.stack([
        batch["pilot_spacing"].to(torch.int64),
        batch["cp_wave_samples"].to(torch.int64),
    ], dim=1).unique(dim=0)

    for spacing_tensor, cp_wave_tensor in group_values:
        spacing = int(spacing_tensor)
        cp_wave = int(cp_wave_tensor)
        selected = torch.nonzero(
            (batch["pilot_spacing"] == spacing)
            & (batch["cp_wave_samples"] == cp_wave), as_tuple=False,
        ).flatten()
        if selected.numel() == 0:
            continue
        b = int(selected.numel())
        if n_sub % spacing or spacing < users:
            raise RuntimeError("Archived pilot spacing is incompatible with the frame")
        n_per = n_sub // spacing
        raw_input = batch["input"].index_select(0, selected).to(device)
        raw_ls = batch["ls_grid"].index_select(0, selected).to(device)
        # [B,Nu,Np,N] exact binary comb mask.  It is a public receiver input.
        mask = raw_input[:, :, 2, :, 0, :].index_select(
            2, pilot_visible.to(device)
        )
        if not torch.all((mask == 0) | (mask == 1)):
            raise RuntimeError("Archived pilot mask is not binary")
        offset = mask[..., :spacing].argmax(dim=-1)
        k = torch.arange(n_per, device=device, dtype=torch.long)
        indices = offset[..., None] + k * spacing
        check_mask = torch.zeros_like(mask)
        check_mask.scatter_(-1, indices, 1.0)
        if not torch.equal(mask, check_mask):
            raise RuntimeError("Archived pilot mask is not a valid orthogonal comb")
        if torch.any(mask.sum(dim=1) > 1):
            raise RuntimeError("Archived user pilot combs overlap")

        zc = _zadoff_chu(n_per, 1, device)
        sequences = torch.stack([
            torch.roll(zc, shifts=user, dims=0) for user in range(users)
        ], dim=0)
        pilot_frame = torch.zeros(
            (b, users, n_pilot, n_sub), dtype=torch.complex64, device=device,
        )
        pilot_frame.scatter_(
            -1, indices,
            sequences[None, :, None, :].expand(b, users, n_pilot, n_per),
        )
        ls_complex = torch.complex(raw_ls[:, :, 0], raw_ls[:, :, 1])
        pilot_ls = ls_complex.index_select(
            2, pilot_visible.to(device)
        ).permute(0, 1, 3, 2, 4)
        if torch.any((pilot_ls.abs() > 0) & (~mask.bool())[:, :, None]):
            raise RuntimeError("Archived LS grid contains non-pilot observations")
        pilot_y = (pilot_ls * pilot_frame[:, :, None]).sum(dim=1)
        frame = torch.zeros(
            (b, users, layout.num_blocks, n_sub),
            dtype=torch.complex64, device=device,
        )
        frame.index_copy_(2, pilot_blocks, pilot_frame)
        received = torch.zeros(
            (b, n_ant, layout.num_blocks, n_sub),
            dtype=torch.complex64, device=device,
        )
        received.index_copy_(2, pilot_blocks, pilot_y)
        delay_idx = delay_window_indices(
            n_sub, n_delay, delay_guard_bins(cfg), device=device,
        )
        derived = _add_backprojection(
            cfg, received, frame, layout, cp_wave, delay_idx,
            per_user=False, fast_time=True, pilot_only_received=True,
        )
        result.index_copy_(0, selected, derived.cpu())
    return result


def derivation_source_sha256():
    """Bind the server attestation to the generator and reconstruction code."""
    root = Path(__file__).resolve().parent
    paths = [
        Path(__file__), root / "verify_historical_validation_reconstruction.py",
        root / "experiment_protocol.py", root / "train.py",
    ]
    paths += sorted((root / "data").glob("*.py"))
    paths += sorted((root / "utils").glob("*.py"))
    payload = {
        str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def companion_paths(path):
    path = Path(path)
    return (path.with_name(path.stem + "_pilot_only.pt"),
            path.with_name(path.stem + "_verified.json"))


def assert_reconstruction_attestation(cfg, path):
    sidecar, attestation = companion_paths(path)
    if not attestation.is_file():
        raise RuntimeError(
            f"Missing fixed-validation reconstruction attestation: {attestation}. "
            "Run verify_historical_validation_reconstruction.py on the GPU server "
            "before training."
        )
    proof = json.loads(attestation.read_text(encoding="utf-8"))
    expected = {
        "status": "passed",
        "cache_sha256": HISTORICAL_CACHE_SHA256,
        "identity_sha256": HISTORICAL_IDENTITY_SHA256,
        "source_sha256": derivation_source_sha256(),
        "samples_checked": int(cfg["train"]["val_samples"]),
        "core_rtol": 1e-4,
        "core_atol": 1e-5,
        "pilot_feature_source": PILOT_FEATURE_SOURCE,
        "sidecar_name": sidecar.name,
    }
    for key, value in expected.items():
        if proof.get(key) != value:
            raise RuntimeError(
                f"Fixed-validation attestation {key} mismatch; rerun server verifier."
            )
    roundtrip_error = proof.get("archived_ls_roundtrip_max_abs")
    if (not isinstance(roundtrip_error, (int, float))
            or not math.isfinite(roundtrip_error)
            or roundtrip_error < 0):
        raise RuntimeError("Fixed-validation LS roundtrip diagnostic is invalid")
    assert_historical_file_metadata(path, proof)
    if not sidecar.is_file() or sidecar.stat().st_size != proof.get("sidecar_bytes"):
        raise RuntimeError("Fixed-validation derived pilot-only sidecar is missing or truncated")
    return sidecar, proof


def load_historical_validation(cfg, path, expected_sha256):
    """Training entry: original observations plus checked derived feature if needed."""
    need_attestation = bool(cfg.get("data_pipeline", {}).get(
        "emit_pilot_only_fast_time_add", False))
    need_feature = str(cfg.get("model", {}).get(
        "ablation_profile", "full")).lower() == "pilot_only_fast_time"
    if need_feature and not need_attestation:
        raise RuntimeError("Pilot-only profile requires checked pilot-only emission")
    sidecar = proof = None
    if need_attestation:
        sidecar, proof = assert_reconstruction_attestation(cfg, path)
    # The server verifier has already streamed the full SHA-256.  Every child
    # checks file identity/metadata before and after loading, avoiding a second
    # multi-gigabyte hash pass in every epoch/variant process.
    core = load_historical_core(cfg, path, expected_sha256, verified_proof=proof)
    if need_attestation:
        if need_feature:
            if _file_sha256(sidecar) != proof.get("sidecar_sha256"):
                raise RuntimeError("Derived pilot-only sidecar SHA-256 mismatch")
            kwargs = {"map_location": "cpu", "weights_only": True}
            if "mmap" in inspect.signature(torch.load).parameters:
                kwargs["mmap"] = True
            try:
                derived = torch.load(sidecar, **kwargs)
            except Exception as exc:
                raise RuntimeError("Cannot safely load derived pilot-only sidecar") from exc
            expected_shape = (int(cfg["train"]["val_samples"]), 2, 2, 15, 8, 48)
            if (not torch.is_tensor(derived) or derived.device.type != "cpu"
                    or derived.dtype != torch.float32
                    or tuple(derived.shape) != expected_shape):
                raise RuntimeError("Derived pilot-only sidecar has an unexpected schema")
            core["add_input_pilot_only_fast_time"] = derived
    return core


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
