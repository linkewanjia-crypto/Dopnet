"""Minimal waveform-grid and Sionna helpers required by DOP-Net."""
from __future__ import annotations

import numpy as np
import torch
from .threegpp_profiles import parse_3gpp_channel_name

SPEED_OF_LIGHT = 299792458.0
_SIONNA_MODEL_CACHE = {}

def active_signed_subcarriers(n_active: int) -> np.ndarray:
    """Contiguous model subcarriers centered around DC, length exactly n_active."""
    n = int(n_active)
    if n < 1:
        raise ValueError("n_active must be positive")
    start = -(n // 2)
    return np.arange(start, start + n, dtype=np.int64)

def active_fft_bins(n_fft: int, n_active: int) -> np.ndarray:
    """Ordinary NumPy FFT-bin indices (DC at index 0)."""
    if int(n_active) > int(n_fft):
        raise ValueError("n_active cannot exceed n_fft")
    return np.mod(active_signed_subcarriers(n_active), int(n_fft)).astype(np.int64)

def active_sionna_grid_indices(n_fft: int, n_active: int) -> np.ndarray:
    """
    Sionna OFDM resource-grid indices (DC in the middle).
    OFDMModulator applies ifftshift and OFDMDemodulator applies fftshift.
    """
    if int(n_active) > int(n_fft):
        raise ValueError("n_active cannot exceed n_fft")
    center = int(n_fft) // 2
    return (center + active_signed_subcarriers(n_active)).astype(np.int64)

def cp_active_to_waveform_samples(cp_len_active: int, n_fft: int, n_active: int) -> int:
    """
    Convert the project's legacy CP length (measured on the N_active grid)
    to the high-rate waveform grid, preserving CP duration.
    """
    ratio = float(n_fft) / float(n_active)
    cp = int(round(float(cp_len_active) * ratio))
    if cp < 0 or cp > int(n_fft):
        raise ValueError("converted CP must satisfy 0 <= cp <= n_fft")
    return cp

def _require_sionna():
    try:
        import sionna.phy  # noqa: F401
        return
    except Exception as e:
        raise RuntimeError(
            "3GPP TDL/CDL channels require Sionna PHY 2.0.1. "
            "Install the package requirements (`pip install -r requirements.txt`). "
            "The code intentionally does not fall back to a reduced local 3GPP simulator, "
            "because doing so would reintroduce delay quantization and incomplete "
            "TDL/CDL coefficient generation."
        ) from e

def _make_sionna_model_cached(profile_name, n_rx, delay_spread_s,
                              carrier_frequency_hz, speed_mps,
                              antenna_spacing_lambda, tdl_num_sinusoids,
                              tdl_rx_corr_rho, device):
    key = (
        str(profile_name).lower(),
        int(n_rx),
        round(float(delay_spread_s), 15),
        round(float(carrier_frequency_hz), 3),
        round(float(speed_mps), 9),
        round(float(antenna_spacing_lambda), 6),
        int(tdl_num_sinusoids),
        round(float(tdl_rx_corr_rho), 6),
        str(device),
    )
    if key not in _SIONNA_MODEL_CACHE:
        _SIONNA_MODEL_CACHE[key] = _make_sionna_model(
            profile_name, n_rx, delay_spread_s, carrier_frequency_hz,
            speed_mps, antenna_spacing_lambda, tdl_num_sinusoids,
            tdl_rx_corr_rho, device
        )
    return _SIONNA_MODEL_CACHE[key]

def _make_sionna_model(profile_name, n_rx, delay_spread_s, carrier_frequency_hz,
                       speed_mps, antenna_spacing_lambda, tdl_num_sinusoids,
                       tdl_rx_corr_rho, device):
    from sionna.phy.channel.tr38901 import TDL, CDL, Antenna, AntennaArray

    family, profile = parse_3gpp_channel_name(profile_name)
    model_letter = profile.upper()

    if family == "tdl":
        rx_corr = None
        rho = float(tdl_rx_corr_rho)
        if abs(rho) > 1e-12:
            idx = torch.arange(int(n_rx), dtype=torch.float32)
            rx_corr = (rho ** torch.abs(idx[:,None]-idx[None,:])).to(torch.complex64)

        return TDL(
            model=model_letter,
            delay_spread=float(delay_spread_s),
            carrier_frequency=float(carrier_frequency_hz),
            num_sinusoids=int(tdl_num_sinusoids),
            los_angle_of_arrival=float(np.pi/4.0),
            min_speed=float(speed_mps),
            max_speed=float(speed_mps),
            num_rx_ant=int(n_rx),
            num_tx_ant=1,
            rx_corr_mat=rx_corr,
            precision="single",
            device=device,
        )

    ut_array = Antenna(
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=float(carrier_frequency_hz),
        precision="single",
        device=device,
    )
    bs_array = AntennaArray(
        num_rows=1,
        num_cols=int(n_rx),
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=float(carrier_frequency_hz),
        vertical_spacing=float(antenna_spacing_lambda),
        horizontal_spacing=float(antenna_spacing_lambda),
        precision="single",
        device=device,
    )
    return CDL(
        model=model_letter,
        delay_spread=float(delay_spread_s),
        carrier_frequency=float(carrier_frequency_hz),
        ut_array=ut_array,
        bs_array=bs_array,
        direction="uplink",
        min_speed=float(speed_mps),
        max_speed=float(speed_mps),
        normalize_delays=True,
        precision="single",
        device=device,
    )
