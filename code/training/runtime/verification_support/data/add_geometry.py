"""Shared angle-delay-Doppler (ADD) geometry for DOP-Net.

Both the data generator and the estimator need to agree on

  * how many delay bins the ADD tensor keeps and *which* physical delays they
    correspond to, and
  * the frame-orthogonal Doppler grid.

The implementation uses one explicit delay window

    l = -guard, ..., L - guard - 1        (bin index)
    tau_l = l / (N * df)                  (physical delay)

realised on the circular delay axis as ``(arange(L) - guard) mod N``.  The
window therefore

  * keeps the acausal leakage bins that a truncation ``[:L]`` would drop,
  * covers ``max(cp_len_choices)`` causal bins, which by
    ``data/frame_feasibility.py`` is an upper bound on the excess delay of every
    generated realization, and
  * is used verbatim by the estimator, so ``delay_grid_s`` reported in the aux
    dictionary is the true physical delay of every ADD voxel.
"""
from __future__ import annotations

import torch

DEFAULT_DELAY_GUARD_BINS = 8


def delay_guard_bins(cfg: dict) -> int:
    return int(cfg["model"].get("add_delay_guard_bins", DEFAULT_DELAY_GUARD_BINS))


def add_delay_bins(cfg: dict) -> int:
    """Number of retained delay bins ``L``.

    ``model.add_delay_bins: auto`` derives it from the frame feasibility rule
    ``max excess delay <= CP``:

        L = min(N, max(cp_len_choices) + 2 * guard)

    which is the smallest window that contains every causal delay the sampler
    can produce plus symmetric room for sinc leakage on both sides.
    """
    s, m = cfg["system"], cfg["model"]
    n = int(s["n_subcarriers"])
    raw = m.get("add_delay_bins", "auto")
    guard = delay_guard_bins(cfg)
    if isinstance(raw, str) and str(raw).lower() == "auto":
        max_cp = max(int(v) for v in s["cp_len_choices"])
        value = max_cp + 2 * guard
    else:
        value = int(raw)
    value = int(min(n, max(1, value)))
    if value <= guard:
        raise ValueError(
            f"add_delay_bins={value} must exceed add_delay_guard_bins={guard}"
        )
    max_cp = max(int(v) for v in s["cp_len_choices"])
    if value - guard < max_cp:
        raise ValueError(
            f"add_delay_bins={value} with guard={guard} keeps only {value-guard} "
            f"causal delay bins, which does not cover max(cp_len_choices)={max_cp}. "
            "Widen model.add_delay_bins or shorten the CP choices."
        )
    return value


def delay_window_indices(n_freq: int, n_delay: int, guard: int, device=None) -> torch.Tensor:
    """Circular delay-bin indices of the retained window, length ``n_delay``."""
    idx = torch.arange(int(n_delay), device=device, dtype=torch.long) - int(guard)
    return torch.remainder(idx, int(n_freq))


def delay_window_seconds(n_freq: int, n_delay: int, guard: int, scs_hz: float,
                         device=None) -> torch.Tensor:
    """Signed physical delay of every retained bin, in seconds."""
    l = torch.arange(int(n_delay), device=device, dtype=torch.float32) - float(guard)
    return l / float(int(n_freq) * float(scs_hz))


def select_delay_window(g: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """[..., N] -> [..., L] on the circular delay axis."""
    return torch.index_select(g, -1, idx)


def scatter_delay_window(g: torch.Tensor, idx: torch.Tensor, n_freq: int) -> torch.Tensor:
    """[..., L] -> [..., N], zero outside the retained window."""
    out = g.new_zeros(g.shape[:-1] + (int(n_freq),))
    return out.index_copy(-1, idx, g)


def symbol_duration_s(cfg: dict, cp_wave: int) -> float:
    s = cfg["system"]
    nfft = int(s["waveform_fft_size"])
    scs = float(s["subcarrier_spacing_hz"])
    return (float(nfft) + float(cp_wave)) / (float(nfft) * scs)


def doppler_modes(cfg: dict, device=None) -> torch.Tensor:
    """Integer Doppler modes ``q`` of the frame-orthogonal dictionary."""
    q = int(cfg["model"]["add_doppler_bins"])
    if q % 2 != 1:
        raise ValueError("model.add_doppler_bins must be odd")
    return torch.arange(-(q // 2), q // 2 + 1, device=device, dtype=torch.float32)


def doppler_grid_hz(cfg: dict, cp_wave: int, device=None) -> torch.Tensor:
    """Frame-orthogonal Doppler grid ``nu_q = q / T_frame`` in Hz."""
    t_frame = int(cfg["system"]["frame_num_symbols"]) * symbol_duration_s(cfg, cp_wave)
    return doppler_modes(cfg, device=device) / t_frame


def pilot_symbol_times_s(cfg: dict, layout, cp_wave: int, device=None) -> torch.Tensor:
    """Physical time of every pilot OFDM symbol on the visible-symbol origin."""
    dur = symbol_duration_s(cfg, cp_wave)
    pv = torch.as_tensor(layout.pilot_visible, dtype=torch.float32, device=device)
    return pv * dur


def intra_symbol_offsets_s(cfg: dict, device=None) -> torch.Tensor:
    """Intra-symbol sample times of the active-band IFFT, centred on the symbol.

    The active-band inverse DFT of one OFDM symbol produces ``N`` samples that
    span the useful part of that symbol.  Centring them on the nominal symbol
    instant is what makes the ADD backprojection and the AFT projection share a
    time origin.
    """
    s = cfg["system"]
    n = int(s["n_subcarriers"])
    scs = float(s["subcarrier_spacing_hz"])
    k = torch.arange(n, device=device, dtype=torch.float32)
    return (k - 0.5 * (n - 1)) / float(n * scs)


def pilot_sample_times_s(cfg: dict, layout, cp_wave: int, fast_time: bool = True,
                         device=None) -> torch.Tensor:
    """[Np, N] physical time of every pilot *sample*.

    ``fast_time=False`` collapses every intra-symbol sample onto the nominal
    symbol instant.  That is exactly the classical slow-time CE-BEM aperture and
    is what the ``no_fast_time_add`` ablation observes.
    """
    base = pilot_symbol_times_s(cfg, layout, cp_wave, device=device)[:, None]
    if not fast_time:
        n = int(cfg["system"]["n_subcarriers"])
        return base.expand(base.shape[0], n).contiguous()
    return base + intra_symbol_offsets_s(cfg, device=device)[None, :]
