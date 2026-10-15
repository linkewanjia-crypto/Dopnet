"""
Channel registry and train/held-out split.

The five project-proposal channels are always supported. A selected subset of
3GPP TR 38.901 TDL/CDL profiles may be used during training, while other
profiles are held out completely for out-of-distribution/generalization tests.
"""
import math

PROJECT_CHANNEL_TYPES = [
    "awgn",
    "rayleigh",
    "rician",
    "nakagami",
    "doubly_selective",
]

THREEGPP_CHANNEL_TYPES = [
    "tdl_a", "tdl_b", "tdl_c", "tdl_d", "tdl_e",
    "cdl_a", "cdl_b", "cdl_c", "cdl_d", "cdl_e",
]

DEFAULT_SEEN_3GPP_CHANNELS = [
    "tdl_b", "tdl_c", "tdl_d",
    "cdl_b", "cdl_c", "cdl_d",
]

DEFAULT_HELDOUT_3GPP_CHANNELS = [
    "tdl_a", "tdl_e",
    "cdl_a", "cdl_e",
]

SYNTHETIC_CHANNEL_TYPES = PROJECT_CHANNEL_TYPES
CHANNEL_TYPES = PROJECT_CHANNEL_TYPES + THREEGPP_CHANNEL_TYPES
CHANNEL_TO_ID = {name: i for i, name in enumerate(CHANNEL_TYPES)}
ID_TO_CHANNEL = {i: name for name, i in CHANNEL_TO_ID.items()}


def normalize_channel_name(name):
    return str(name).lower().replace("-", "_")


def is_project_channel(name):
    return normalize_channel_name(name) in PROJECT_CHANNEL_TYPES


def is_threegpp_channel(name):
    return normalize_channel_name(name) in THREEGPP_CHANNEL_TYPES


def assert_channel_name(name):
    n = normalize_channel_name(name)
    if n not in CHANNEL_TO_ID:
        raise ValueError(
            f"Unsupported channel type: {name!r}. "
            f"Allowed channels are: {', '.join(CHANNEL_TYPES)}"
        )
    return n


def validate_channel_config(system_cfg):
    """
    Validate the active training split without leaking held-out 3GPP channels.

    The current experiment split supports:
      - all five project channels;
      - selected TDL/CDL profiles in ``channel_types``;
      - disjoint ``heldout_3gpp_channels`` for generalization testing.

    Any train/held-out overlap raises an error instead of being silently fixed.
    """
    channels = [
        assert_channel_name(ch)
        for ch in system_cfg.get("channel_types", PROJECT_CHANNEL_TYPES)
    ]
    weights = system_cfg.get("channel_type_weights")

    if not channels:
        channels = list(PROJECT_CHANNEL_TYPES)

    if weights is None or len(weights) != len(channels):
        weights = [1.0] * len(channels)
    else:
        weights = [float(w) for w in weights]

    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("channel_type_weights must be finite and non-negative")
    z = math.fsum(weights)
    if z <= 0:
        raise ValueError("channel_type_weights must contain positive mass")
    # Preserve already-normalized weights exactly. Re-dividing by a rounded
    # sum on every validation changed the manifest at each call site.
    if not math.isclose(z, 1.0, rel_tol=0.0, abs_tol=1e-12):
        weights = [w / z for w in weights]

    heldout = [
        normalize_channel_name(ch)
        for ch in system_cfg.get("heldout_3gpp_channels", [])
    ]
    bad_heldout = [ch for ch in heldout if ch not in THREEGPP_CHANNEL_TYPES]
    if bad_heldout:
        raise ValueError(
            "heldout_3gpp_channels may contain only TDL/CDL profiles; "
            f"got {bad_heldout}"
        )

    overlap = sorted(set(channels) & set(heldout))
    if overlap:
        raise ValueError(
            "Training/held-out channel leakage detected. These profiles occur "
            f"in both channel_types and heldout_3gpp_channels: {overlap}"
        )

    system_cfg["channel_types"] = channels
    system_cfg["channel_type_weights"] = weights
    system_cfg["heldout_3gpp_channels"] = heldout

    # Shared synthetic channel ranges.
    system_cfg.setdefault("delay_spread_ns_choices", [30.0, 100.0, 300.0])
    system_cfg.setdefault("doppler_hz_choices", [0.0, 70.0, 250.0, 500.0])
    system_cfg.setdefault("jakes_num_sinusoids", 32)
    system_cfg.setdefault("synthetic_pdp_profile", "exponential")

    # Fixed ensemble-reference SNR convention used by the waveform generator.
    if str(system_cfg.get("snr_reference", "")).lower() in {"", "active_received_data_re"}:
        system_cfg["snr_reference"] = "measured_ensemble_active_received_data_re"
    system_cfg.setdefault("snr_noise_policy", "measured_ensemble_reference")
    system_cfg["frame_mode"] = "dopnet"
    system_cfg.setdefault("frame_num_symbols", 13)
    system_cfg.setdefault("pilot_symbol_period", 4)
    system_cfg.setdefault("pilot_mode", "common_time_frequency_orthogonal_comb")
    system_cfg["target_reference"] = "useful_window_average_cfr_diagonal"
    T = int(system_cfg["frame_num_symbols"]); P = int(system_cfg["pilot_symbol_period"])
    if T < 2 or P < 2 or (T - 1) % P != 0:
        raise ValueError("DOP-Net requires frame_num_symbols = k*pilot_symbol_period + 1")
    Nu = int(system_cfg.get("n_users", 1)); N = int(system_cfg.get("n_subcarriers", 64))
    for spacing in system_cfg.get("pilot_spacing_choices", [4]):
        spacing = int(spacing)
        if spacing < Nu:
            raise ValueError("Each pilot_spacing must be >= n_users for disjoint user combs")
        if N % spacing != 0:
            raise ValueError("DOP-Net requires n_subcarriers divisible by each pilot_spacing")

    # 3GPP TR 38.901 link-level settings.
    system_cfg.setdefault("threegpp_delay_spread_ns_choices", [30.0, 100.0, 300.0])
    system_cfg.setdefault("threegpp_doppler_hz_choices", [0.0, 70.0, 250.0, 500.0])
    system_cfg.setdefault("tdl_num_sinusoids", 20)
    system_cfg.setdefault("tdl_rx_corr_rho", 0.0)
    system_cfg.setdefault("normalize_3gpp_channel", False)
    system_cfg.setdefault("threegpp_link_batch_size", 1)
    system_cfg.setdefault("sionna_device", "cpu")
    return system_cfg
