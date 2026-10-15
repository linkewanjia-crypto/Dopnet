"""DOP-Net OFDM frame layout and physical time geometry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from .channel_backend import cp_active_to_waveform_samples

@dataclass(frozen=True)
class TimeFrequencyLayout:
    visible_symbols: int
    pilot_period: int
    num_blocks: int
    pre_block: int
    visible_blocks: Tuple[int, ...]
    pilot_visible: Tuple[int, ...]
    pilot_blocks: Tuple[int, ...]
    data_visible: Tuple[int, ...]
    data_blocks: Tuple[int, ...]
    tail_block: int

def make_layout(visible_symbols: int = 13, pilot_period: int = 4) -> TimeFrequencyLayout:
    T = int(visible_symbols)
    P = int(pilot_period)
    if T < 2 or P < 2:
        raise ValueError("frame_num_symbols>=2 and pilot_symbol_period>=2 are required")
    pilot_vis = tuple(range(0, T, P))
    if pilot_vis[-1] != T - 1:
        raise ValueError(
            "The current frame requires the last visible symbol to be a pilot so every data symbol "
            "is bracketed by two pilot occasions. Choose frame_num_symbols = k*period+1."
        )
    data_vis = tuple(i for i in range(T) if i not in set(pilot_vis))
    visible_blocks = tuple(i + 1 for i in range(T))  # hidden pre-history at block 0
    pilot_blocks = tuple(i + 1 for i in pilot_vis)
    data_blocks = tuple(i + 1 for i in data_vis)
    return TimeFrequencyLayout(
        visible_symbols=T,
        pilot_period=P,
        num_blocks=T + 2,
        pre_block=0,
        visible_blocks=visible_blocks,
        pilot_visible=pilot_vis,
        pilot_blocks=pilot_blocks,
        data_visible=data_vis,
        data_blocks=data_blocks,
        tail_block=T + 1,
    )

def physical_symbol_duration_s(n_fft: int, cp_wave: int, scs_hz: float) -> float:
    return (float(n_fft) + float(cp_wave)) / (float(n_fft) * float(scs_hz))

def time_feature_scale_s(cfg: dict) -> float:
    """Training-only public scale used to normalize the two time channels.

    The scale is based only on numerology, configured CP choices and pilot
    period. It does not use any test realization's Doppler/channel truth.
    """
    s = cfg["system"]
    nfft = int(s.get("waveform_fft_size", 1024))
    nactive = int(s["n_subcarriers"])
    scs = float(s.get("subcarrier_spacing_hz", 30000.0))
    period = int(s.get("pilot_symbol_period", 4))
    max_cp = max(int(x) for x in s.get("cp_len_choices", [16]))
    cpw = cp_active_to_waveform_samples(max_cp, nfft, nactive)
    return max(period * physical_symbol_duration_s(nfft, cpw, scs), 1e-9)

def time_geometry(layout: TimeFrequencyLayout, symbol_duration_s: float):
    """Return public physical time coordinates and two causal/acausal distances.

    age_s[t]     = time elapsed since the most recent pilot occasion.
    horizon_s[t] = time remaining until the next pilot occasion.
    At pilot symbols both are zero. For data symbols between pilots their sum is
    exactly the physical pilot-to-pilot interval, including the actual CP.
    """
    T = layout.visible_symbols
    times = np.arange(T, dtype=np.float64) * float(symbol_duration_s)
    pilot = np.asarray(layout.pilot_visible, dtype=np.int64)
    age = np.zeros(T, dtype=np.float64)
    horizon = np.zeros(T, dtype=np.float64)
    pset = set(int(x) for x in pilot)
    for t in range(T):
        if t in pset:
            continue
        prev = pilot[pilot < t].max()
        nxt = pilot[pilot > t].min()
        age[t] = (t - prev) * float(symbol_duration_s)
        horizon[t] = (nxt - t) * float(symbol_duration_s)
    data_mask = np.zeros(T, dtype=np.float32)
    data_mask[list(layout.data_visible)] = 1.0
    pilot_symbol_mask = 1.0 - data_mask
    return times, age, horizon, data_mask, pilot_symbol_mask
