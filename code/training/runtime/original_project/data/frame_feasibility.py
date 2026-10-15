"""Frame feasibility rules for the DOP-Net experiment.

Two physical constraints must hold for every generated sample, otherwise the
per-symbol CFR that the network regresses is not even well defined:

1. **CP >= maximum excess delay.**  Otherwise the received symbol carries
   inter-symbol interference and the "channel frequency response of symbol m"
   is a fiction.
2. **Comb spacing keeps the delay axis unambiguous**, i.e. ``1/(S*df) >= CP``.
   Otherwise the delay-domain backprojection folds and no amount of network
   capacity can undo the fold.

The bound is derived per channel family rather than using one universal
normalized excess delay. A universal bound would be simultaneously

* **too large** for the synthetic families -- the exponential 6-tap PDP built by
  ``data/generator.py::_synthetic_paths`` has a maximum excess delay of only
  ``3.227 * DS``, so 10x forced needlessly long cyclic prefixes and removed
  feasible (CP, comb) pairs from the training distribution; and
* **too small** for TDL-D / CDL-D, whose last TR 38.901 tap sits at
  ``12.525 * DS``.

The implementation derives the bound from the actual TR 38.901 tables
(``data/threegpp_profiles.py``) and from the analytic synthetic PDP.
"""
from __future__ import annotations

import math

from .channel_backend import cp_active_to_waveform_samples
from .channels import normalize_channel_name

# Relative tolerance used when comparing durations, so that an exactly-matched
# pair such as CP = 1/(S*df) is not rejected by float round-off.
_TOL = 1e-9

# Maximum normalized excess delay tau_max / DS of the TR 38.901 profiles, read
# off the tables in data/threegpp_profiles.py.
THREEGPP_MAX_NORMALIZED_EXCESS_DELAY = {
    "tdl_a": 9.6586, "tdl_b": 4.7834, "tdl_c": 8.6523,
    "tdl_d": 12.525, "tdl_e": 20.6519,
    "cdl_a": 9.6586, "cdl_b": 4.7834, "cdl_c": 8.6523,
    "cdl_d": 12.525, "cdl_e": 20.6419,
}

# Analytic maximum excess delay of the synthetic PDPs, in units of DS.  The
# generator places taps at tau_l = (l / rms(PDP)) * DS for l = 0..L-1, so the
# bound is (L-1)/rms(PDP).  Reproduced by validate_physics_numerics.py.
SYNTHETIC_MAX_NORMALIZED_EXCESS_DELAY = {
    "exponential": {4: 2.968, 6: 3.227, 8: 3.363},
    "uniform": {4: 2.683, 6: 2.928, 8: 3.055},
    "two_cluster": {4: 2.412, 6: 2.702, 8: 2.860},
}

# Conservative default when a synthetic (n_taps, profile) pair is not tabulated.
SYNTHETIC_FALLBACK_NORMALIZED_EXCESS_DELAY = 4.0

_SYNTHETIC_FAMILIES = {"awgn", "rayleigh", "rician", "nakagami", "doubly_selective"}


def synthetic_max_normalized_excess_delay(n_taps: int, profile: str) -> float:
    table = SYNTHETIC_MAX_NORMALIZED_EXCESS_DELAY.get(str(profile).lower())
    if table is None:
        return SYNTHETIC_FALLBACK_NORMALIZED_EXCESS_DELAY
    return float(table.get(int(n_taps), SYNTHETIC_FALLBACK_NORMALIZED_EXCESS_DELAY))


def max_normalized_excess_delay(cfg: dict, channel_type=None) -> float:
    """tau_max / DS for one channel family, or the worst case over all of them."""
    s = cfg["system"]
    n_taps = int(s.get("n_taps", 6))
    profile = str(s.get("synthetic_pdp_profile", "exponential"))
    per_channel = dict(s.get("channel_pdp_profiles", {}) or {})

    if channel_type is None:
        active = [normalize_channel_name(c) for c in s.get("channel_types", [])]
        active += [normalize_channel_name(c) for c in s.get("heldout_3gpp_channels", [])]
        if not active:
            active = sorted(_SYNTHETIC_FAMILIES)
        return max(max_normalized_excess_delay(cfg, c) for c in active)

    name = normalize_channel_name(channel_type)
    if name == "awgn":
        return 0.0
    if name in _SYNTHETIC_FAMILIES:
        return synthetic_max_normalized_excess_delay(
            n_taps, per_channel.get(name, profile)
        )
    if name in THREEGPP_MAX_NORMALIZED_EXCESS_DELAY:
        return THREEGPP_MAX_NORMALIZED_EXCESS_DELAY[name]
    raise ValueError(f"Unknown channel type for feasibility analysis: {channel_type!r}")


def cp_duration_s(cp_active: int, cfg: dict) -> float:
    s = cfg["system"]
    nfft = int(s["waveform_fft_size"])
    nactive = int(s["n_subcarriers"])
    scs = float(s["subcarrier_spacing_hz"])
    cpw = cp_active_to_waveform_samples(int(cp_active), nfft, nactive)
    return cpw / float(nfft * scs)


def required_cp_duration_s(cfg: dict, delay_spread_ns: float, channel_type=None) -> float:
    return max_normalized_excess_delay(cfg, channel_type) * float(delay_spread_ns) * 1e-9


def feasible_cp_choices(cfg: dict, delay_spread_ns: float, channel_type=None):
    """CP values whose duration covers the maximum excess delay of the channel."""
    need = required_cp_duration_s(cfg, delay_spread_ns, channel_type)
    ok = [int(c) for c in cfg["system"]["cp_len_choices"]
          if cp_duration_s(c, cfg) >= need * (1.0 - _TOL)]
    if not ok:
        raise ValueError(
            f"No configured CP covers a {delay_spread_ns:.0f} ns delay spread on "
            f"channel {channel_type!r} (needs >= {need*1e6:.2f} us). Extend "
            "system.cp_len_choices, reduce delay_spread_ns_choices, or lower "
            "system.subcarrier_spacing_hz."
        )
    return ok


def feasible_pilot_spacings(cfg: dict, cp_active: int):
    """Comb spacings whose unambiguous delay range covers the CP."""
    s = cfg["system"]
    scs = float(s["subcarrier_spacing_hz"])
    n_users = int(s["n_users"])
    cp_s = cp_duration_s(cp_active, cfg)
    ok = [int(v) for v in s["pilot_spacing_choices"]
          if (1.0 / (int(v) * scs)) >= cp_s * (1.0 - _TOL) and int(v) >= n_users]
    if not ok:
        raise ValueError(
            f"No configured pilot spacing keeps the delay axis unambiguous for "
            f"cp_len_active={cp_active} (CP={cp_s*1e6:.2f} us). Extend "
            "system.pilot_spacing_choices with a smaller value."
        )
    return ok


def feasibility_report(cfg: dict, include_heldout: bool = True):
    """Human-readable audit of every (channel, delay spread, CP, comb) triple."""
    s = cfg["system"]
    scs = float(s["subcarrier_spacing_hz"])
    channels = [normalize_channel_name(c) for c in s["channel_types"]]
    if include_heldout:
        channels += [normalize_channel_name(c) for c in s.get("heldout_3gpp_channels", [])]
    rows = []
    for ch in channels:
        is_3gpp = ch not in _SYNTHETIC_FAMILIES
        spreads = (s["threegpp_delay_spread_ns_choices"] if is_3gpp
                   else s["delay_spread_ns_choices"])
        for ds in spreads:
            cps = feasible_cp_choices(cfg, float(ds), ch)
            for cp in cps:
                spac = feasible_pilot_spacings(cfg, cp)
                rows.append({
                    "channel": ch,
                    "delay_spread_ns": float(ds),
                    "max_normalized_excess_delay": max_normalized_excess_delay(cfg, ch),
                    "required_cp_us": required_cp_duration_s(cfg, float(ds), ch) * 1e6,
                    "cp_len_active": cp,
                    "cp_us": cp_duration_s(cp, cfg) * 1e6,
                    "pilot_spacings": spac,
                    "unambiguous_delay_us": [1e6 / (v * scs) for v in spac],
                })
    return rows


def assert_all_feasible(cfg: dict, include_heldout: bool = True):
    """Raise if any (channel, delay spread) pair in the config has no feasible CP."""
    feasibility_report(cfg, include_heldout=include_heldout)
    return True


def worst_case_ici_to_signal_db(cfg: dict) -> float:
    """Small-eps ICI-to-signal power ratio for a Jakes spectrum, in dB.

    ``P_ICI / P_signal ~= (pi * eps)^2 / 6`` with ``eps = f_d / df``.  This is
    the quantity that decides whether the intra-symbol (fast-time) observation
    that motivates the ADD branch carries information above the noise floor.
    """
    s = cfg["system"]
    scs = float(s["subcarrier_spacing_hz"])
    fd = max([float(x) for x in s["doppler_hz_choices"]]
             + [float(x) for x in s["threegpp_doppler_hz_choices"]])
    eps = fd / scs
    if eps <= 0.0:
        return float("-inf")
    return 10.0 * math.log10((math.pi * eps) ** 2 / 6.0)
