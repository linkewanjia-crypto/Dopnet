"""GPU-native waveform generation with fast-time ADD observations.

Implementation notes
--------------------
The pilot frame and pilot LS estimate are built without Python loops over
``batch x user x pilot-occasion``.  On a GPU that is thousands of kernel
launches per micro-batch and it dominated the wall-clock cost of data
generation.  Every one of those loops is vectorised here.  The Doppler whitener,
and the Doppler whitener is cached per ``(CP, aperture)``.

* **CFR target.**  For a time-varying channel the diagonal of the OFDM channel
  matrix is the *average* of the tap gain over the useful part of the symbol,
  not its value at the first useful sample.
* **Delay window.**  The ADD tensor now keeps the physically meaningful
  circular window ``[-guard, L-guard)`` (see ``data/add_geometry.py``) instead
  of the first ``N`` bins, which both cuts the ADD cost and stops discarding the
  acausal sinc-leakage bins.
* **SNR reference.**  ``snr_noise_policy: measured_ensemble_reference`` scales
  the noise by the *measured* ensemble received power of the chunk, so the
  realised SNR equals the nominal SNR for every channel family.
* **Slow-time ADD observation.**  A genuine slow-time-only backprojection is
  emitted on request, so the ``no_fast_time_add`` ablation compares two
  *observations* of the same received signal rather than comparing a pilot
  backprojection against the network's own AFT output.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch

from .add_geometry import (
    add_delay_bins, delay_guard_bins, delay_window_indices, delay_window_seconds,
    doppler_grid_hz, pilot_sample_times_s, select_delay_window,
)
from .channel_backend import (
    active_sionna_grid_indices, cp_active_to_waveform_samples,
    _make_sionna_model_cached, SPEED_OF_LIGHT,
)
from .channels import CHANNEL_TO_ID, assert_channel_name
from .frame_layout import make_layout, physical_symbol_duration_s, time_feature_scale_s
from .frame_feasibility import max_normalized_excess_delay
from utils.modem import MOD_TO_ID

from .generator import (
    _random_symbols, _ofdm_modulate_torch, _ofdm_demodulate_torch,
    _synthetic_paths, _embed_active_sionna,
)

SNR_POLICY_TO_ID = {"fixed_ensemble_reference": 0, "measured_ensemble_reference": 1}


@lru_cache(maxsize=32)
def _sionna_ops_cached(nfft: int, cpw: int, ttime: int, lmin: int, lmax: int, device_str: str):
    """Cache stateless Sionna PHY operators for repeated online generation.

    TDL/CDL model objects are cached in ``channel_backend.py``; the waveform
    rebuilt the OFDM modulator/demodulator and ``ApplyTimeChannel`` object for
    every microbatch.  Their configuration depends only on the waveform shape and
    device, so reusing them is exact and removes Python/module construction from
    the hot path.
    """
    from sionna.phy.channel import ApplyTimeChannel
    from sionna.phy.ofdm import OFDMModulator, OFDMDemodulator

    ltot = int(lmax) - int(lmin) + 1
    mod = OFDMModulator(
        cyclic_prefix_length=int(cpw), precision="single", device=str(device_str)
    )
    demod = OFDMDemodulator(
        fft_size=int(nfft), l_min=int(lmin), cyclic_prefix_length=int(cpw),
        precision="single", device=str(device_str),
    )
    apply = ApplyTimeChannel(
        num_time_samples=int(ttime), l_tot=int(ltot),
        precision="single", device=str(device_str),
    )
    return mod, demod, apply


# ---------------------------------------------------------------------------
# Pilot sequence and frame construction
# ---------------------------------------------------------------------------


def _zadoff_chu(length: int, root: int, device):
    """Unit-modulus CAZAC pilot sequence.

    An all-ones comb has an impulse-train inverse DFT, so within one OFDM symbol
    the pilot energy sits on only ``S`` samples and the intra-symbol correlation
    is weighted by a highly non-uniform window.  Here the *occupied comb tones*
    carry a Zadoff-Chu sequence.  Its N/S-point IDFT is constant-envelope; after
    embedding on an S-spaced comb the useful-time waveform is a repeated,
    constant-envelope sequence (up to the comb-offset phase ramp).  The full
    N-tone frequency grid is still sparse, so we deliberately do not call the
    zero-stuffed OFDM symbol itself CAZAC.
    """
    L = int(length)
    m = torch.arange(L, device=device, dtype=torch.float32)
    q = m * (m + 1.0) / 2.0 if L % 2 else m * m / 2.0
    return torch.exp(-1j * 2.0 * math.pi * float(root) * q / float(L)).to(torch.complex64)


def pilot_amplitude(cfg, spacing, n_users):
    """Amplitude of one pilot RE.

    ``none``                   : unit-modulus pilots (classical CE convention).
    ``comb_energy_equalized``  : the pilot OFDM symbol carries the same total
                                 energy as a data symbol, which is what a real
                                 scheduler does when only ``Nu * N / S`` of the
                                 REs are occupied.
    """
    mode = str(cfg["system"].get("pilot_boost_mode", "none")).lower()
    if mode == "none":
        return 1.0
    if mode == "comb_energy_equalized":
        return math.sqrt(float(spacing) / float(n_users))
    raise ValueError(f"Unknown system.pilot_boost_mode={mode!r}")


def _build_frame(B, Nu, N, spacing, modulation, device, gen, visible_symbols,
                 pilot_period, amplitude=1.0):
    layout = make_layout(visible_symbols, pilot_period)
    S = int(spacing)
    if S < Nu:
        raise ValueError(f"pilot_spacing={S} must be >= n_users={Nu}")
    if N % S:
        raise RuntimeError("n_subcarriers must be divisible by pilot_spacing")
    nper = N // S
    Np = len(layout.pilot_visible)

    frame = torch.zeros((B, Nu, layout.num_blocks, N), dtype=torch.complex64, device=device)
    _, qpre = _random_symbols((B, Nu, N), modulation, device, gen)
    _, qtail = _random_symbols((B, Nu, N), modulation, device, gen)
    frame[:, :, layout.pre_block] = qpre
    frame[:, :, layout.tail_block] = qtail

    # One fresh random assignment of comb offsets to users per pilot occasion.
    rnd = torch.rand((B, Np, S), device=device, generator=gen)
    chosen = rnd.argsort(dim=-1)[..., :Nu]                                  # [B,Np,Nu]
    k = torch.arange(nper, device=device, dtype=torch.long)
    pidx = chosen[..., None] + k.view(1, 1, 1, nper) * S                    # [B,Np,Nu,nper]
    pidx = pidx.permute(0, 2, 1, 3).contiguous()                            # [B,Nu,Np,nper]

    zc = _zadoff_chu(nper, 1, device)
    # A distinct cyclic shift per user keeps the two combs separable even when a
    # scheduler assigns them adjacent offsets.
    seq = torch.stack([torch.roll(zc, shifts=int(u), dims=0) for u in range(Nu)], dim=0)
    pval = (float(amplitude) * seq).view(1, Nu, 1, nper).expand(B, Nu, Np, nper).contiguous()

    pilot_grid = torch.zeros((B, Nu, Np, N), dtype=torch.complex64, device=device)
    pilot_grid.scatter_(-1, pidx, pval)
    pb = torch.as_tensor(layout.pilot_blocks, device=device, dtype=torch.long)
    frame.index_copy_(2, pb, pilot_grid)

    pm = torch.zeros((B, Nu, Np, N), dtype=torch.float32, device=device)
    pm.scatter_(-1, pidx, torch.ones((B, Nu, Np, nper), dtype=torch.float32, device=device))
    pmask = torch.zeros((B, Nu, layout.visible_symbols, N), dtype=torch.float32, device=device)
    pv = torch.as_tensor(layout.pilot_visible, device=device, dtype=torch.long)
    pmask.index_copy_(2, pv, pm)

    Td = len(layout.data_visible)
    labels, data_symbols = _random_symbols((B, Nu, Td, N), modulation, device, gen)
    db = torch.as_tensor(layout.data_blocks, device=device, dtype=torch.long)
    frame.index_copy_(2, db, data_symbols)
    return frame, layout, pmask, pidx, pval, labels, data_symbols


# ---------------------------------------------------------------------------
# Pilot LS + interpolation (fully vectorised)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _time_interp_weights_cpu(visible_symbols: int, pilot_period: int):
    layout = make_layout(visible_symbols, pilot_period)
    T = layout.visible_symbols
    pv = list(layout.pilot_visible)
    pos = {t: i for i, t in enumerate(pv)}
    W = torch.zeros((T, len(pv)), dtype=torch.float32)
    for t in range(T):
        if t in pos:
            W[t, pos[t]] = 1.0
            continue
        left = max(x for x in pv if x < t)
        right = min(x for x in pv if x > t)
        a = float(t - left) / float(right - left)
        W[t, pos[left]] = 1.0 - a
        W[t, pos[right]] = a
    return W


def _pilot_ls_and_interp(Y, layout, pidx, pval, Nu, Nr, N, spacing, want_ls_grid=False):
    """``Y`` ``[B,Nr,Bsym,N]`` -> coarse ``[B,Nu,T,Nr,N]``.

    ``pval`` carries the transmitted unit-modulus pilot symbols, so the LS step
    is ``Y/x`` rather than the shortcut ``Y`` (valid only for an all-ones
    comb).  Frequency interpolation is linear on the uniform comb, with linear
    extrapolation outside the first/last pilot subcarrier; time interpolation is
    linear between the two bracketing pilot occasions, which every data symbol
    has by construction of the layout.
    """
    B = Y.shape[0]
    device = Y.device
    Np = len(layout.pilot_visible)
    nper = int(pidx.shape[-1])
    S = int(spacing)

    pb = torch.as_tensor(layout.pilot_blocks, device=device, dtype=torch.long)
    yp = torch.index_select(Y, 2, pb)                                       # [B,Nr,Np,N]
    idx = pidx[:, :, None, :, :].expand(B, Nu, Nr, Np, nper)
    vals = torch.gather(yp[:, None].expand(B, Nu, Nr, Np, N), -1, idx)
    ls = vals / pval[:, :, None, :, :]                                      # [B,Nu,Nr,Np,nper]

    n = torch.arange(N, device=device, dtype=torch.float32)
    off = pidx[..., 0].to(torch.float32)                                    # [B,Nu,Np]
    t = (n.view(1, 1, 1, N) - off[..., None]) / float(S)                    # [B,Nu,Np,N]
    lo = t.floor().clamp(0, nper - 2).long()
    w = (t - lo.to(t.dtype)).to(torch.complex64)
    lo_e = lo[:, :, None, :, :].expand(B, Nu, Nr, Np, N)
    y0 = torch.gather(ls, -1, lo_e)
    y1 = torch.gather(ls, -1, lo_e + 1)
    dense = y0 + (y1 - y0) * w[:, :, None, :, :]                            # [B,Nu,Nr,Np,N]

    W = _time_interp_weights_cpu(layout.visible_symbols, layout.pilot_period).to(device)
    coarse = torch.einsum("tp,burpn->butrn", W.to(torch.complex64), dense)

    ls_grid = None
    if want_ls_grid:
        sparse = torch.zeros((B, Nu, Nr, Np, N), dtype=torch.complex64, device=device)
        sparse.scatter_(-1, idx, ls)
        ls_grid = torch.zeros(
            (B, Nu, layout.visible_symbols, Nr, N), dtype=torch.complex64, device=device
        )
        pv = torch.as_tensor(layout.pilot_visible, device=device, dtype=torch.long)
        ls_grid.index_copy_(2, pv, sparse.permute(0, 1, 3, 2, 4).contiguous())
    return coarse, ls_grid


# ---------------------------------------------------------------------------
# Public time features
# ---------------------------------------------------------------------------


def _time_features(cfg, B, Nu, T, Nr, N, cp_wave, device):
    s = cfg["system"]
    nfft = int(s["waveform_fft_size"])
    scs = float(s["subcarrier_spacing_hz"])
    layout = make_layout(T, int(s["pilot_symbol_period"]))
    dur = physical_symbol_duration_s(nfft, cp_wave, scs)
    scale = time_feature_scale_s(cfg)
    pvis = list(layout.pilot_visible)
    pset = set(pvis)
    age = torch.zeros(T, device=device, dtype=torch.float32)
    horizon = torch.zeros_like(age)
    for t in range(T):
        if t in pset:
            continue
        prev = max(x for x in pvis if x < t)
        nxt = min(x for x in pvis if x > t)
        age[t] = float((t - prev) * dur / scale)
        horizon[t] = float((nxt - t) * dur / scale)
    shape = (B, Nu, T, Nr, N)
    age_grid = age.view(1, 1, T, 1, 1).expand(shape)
    hor_grid = horizon.view(1, 1, T, 1, 1).expand(shape)
    times = torch.arange(T, device=device, dtype=torch.float32) * float(dur)
    raw_age = age * float(scale)
    raw_hor = horizon * float(scale)
    data_mask = torch.zeros(T, device=device, dtype=torch.float32)
    data_mask[list(layout.data_visible)] = 1.0
    return (age_grid, hor_grid, times, raw_age, raw_hor,
            data_mask, 1.0 - data_mask, float(scale))


# ---------------------------------------------------------------------------
# ADD backprojection
# ---------------------------------------------------------------------------


_WHITENER_CACHE = {}


def _doppler_whitener(cfg, layout, cp_wave, fast_time, device):
    """``(B_p^H B_p / M + reg I)^{-1}`` for the actual pilot sampling pattern.

    The raw correlation returns a *matched filter*; the AFT branch produces an
    orthogonal LS projection.  Left-multiplying by this whitener turns the
    matched filter into the regularized-LS estimate on the same grid, so the
    voxel-wise fusion gate combines two quantities in identical coordinates.

    Depends only on the CP and on whether the fast-time samples are used, so it
    is built once in float64 on the host and cached on the device.
    """
    s, m = cfg["system"], cfg["model"]
    key = (
        int(cp_wave), bool(fast_time), int(m["add_doppler_bins"]),
        int(s["frame_num_symbols"]), int(s["pilot_symbol_period"]),
        int(s["n_subcarriers"]), float(s["subcarrier_spacing_hz"]),
        int(s["waveform_fft_size"]), float(m["doppler_whitening_reg"]), str(device),
    )
    cached = _WHITENER_CACHE.get(key)
    if cached is not None:
        return cached
    times = pilot_sample_times_s(cfg, layout, cp_wave, fast_time=fast_time)
    times = times.reshape(-1).to(torch.float64)
    grid = doppler_grid_hz(cfg, cp_wave).to(torch.float64)
    basis = torch.exp(1j * 2.0 * math.pi * times[:, None] * grid[None, :]).to(torch.complex128)
    gram = (basis.conj().transpose(0, 1) @ basis) / float(times.numel())
    eye = torch.eye(grid.numel(), dtype=gram.dtype)
    out = torch.linalg.inv(gram + float(m["doppler_whitening_reg"]) * eye)
    out = out.to(torch.complex64).to(device)
    _WHITENER_CACHE[key] = out
    return out


def _mask_to_joint_pilot_bins(yfreq, xfreq):
    """Retain only pilot bins transmitted by at least one scheduled user.

    The mask is determined from the known transmitted pilot grid ``xfreq``;
    it never depends on a channel realization or on the received amplitudes.
    ``yfreq`` is ``[B,Nr,Np,N]`` and ``xfreq`` is ``[B,Nu,Np,N]``.
    """
    if yfreq.ndim != 4 or xfreq.ndim != 4 or (
        yfreq.shape[0] != xfreq.shape[0]
        or yfreq.shape[2:] != xfreq.shape[2:]
    ):
        raise ValueError("Expected matching [B,Nr,Np,N] and [B,Nu,Np,N] pilot grids")
    joint_mask = xfreq.ne(0).any(dim=1)                             # [B,Np,N]
    return yfreq * joint_mask[:, None]


def _add_backprojection(cfg, Y, frame, layout, cp_wave, delay_idx, *,
                        per_user=False, fast_time=True,
                        pilot_only_received=False):
    """Observable CE-BEM-like ADD backprojection.

    ``per_user=False``: ``Y`` is ``[B,Nr,Bsym,N]`` (the noisy superposition seen
    by the receiver) and the result is the network *input*.
    ``per_user=True``: ``Y`` is ``[B,Nu,Nr,Bsym,N]`` (each user's isolated
    noise-free link) and the result is the training *target*.  Exactly the same
    operator is applied in both cases, so the target contains both inter-symbol
    and intra-symbol Doppler while no channel truth leaks into the input.

    ``pilot_only_received`` additionally masks received frequency bins outside
    the union of transmitted pilot combs *before* the fast-time inverse FFT.
    It is valid only for the noisy, summed receiver observation.  The ordinary
    operator retains all received bins, including Doppler leakage off the comb.
    """
    device = Y.device
    N = int(Y.shape[-1])
    grid = doppler_grid_hz(cfg, cp_wave, device=device)                     # [Q]
    times = pilot_sample_times_s(
        cfg, layout, cp_wave, fast_time=fast_time, device=device
    )                                                                       # [Np,N]
    whitener = _doppler_whitener(cfg, layout, cp_wave, fast_time, device)   # [Q,Q]

    pb = torch.as_tensor(layout.pilot_blocks, device=device, dtype=torch.long)
    xfreq = torch.index_select(frame, 2, pb)                                # [B,Nu,Np,N]
    xt = torch.fft.ifft(xfreq, dim=-1, norm="ortho")
    phase = torch.exp(
        -1j * 2.0 * math.pi * grid[None, :, None] * times[:, None, :]
    ).to(torch.complex64)                                                   # [Np,Q,N]

    if per_user:
        if pilot_only_received:
            raise ValueError("Do not mask the clean per-user supervision target")
        yfreq = torch.index_select(Y, 3, pb)                                # [B,Nu,Nr,Np,N]
        yt = torch.fft.ifft(yfreq, dim=-1, norm="ortho")
        yfft = torch.fft.fft(
            yt[:, :, :, :, None, :] * phase[None, None, None], dim=-1, norm="ortho"
        )                                                                   # [B,Nu,Nr,Np,Q,N]
        cross = torch.einsum("burpqn,bupn->burqn", yfft, xfreq.conj())
    else:
        yfreq = torch.index_select(Y, 2, pb)                                # [B,Nr,Np,N]
        if pilot_only_received:
            if not fast_time:
                raise ValueError("pilot_only_received requires fast_time=True")
            yfreq = _mask_to_joint_pilot_bins(yfreq, xfreq)
        yt = torch.fft.ifft(yfreq, dim=-1, norm="ortho")
        yfft = torch.fft.fft(
            yt[:, :, :, None, :] * phase[None, None], dim=-1, norm="ortho"
        )                                                                   # [B,Nr,Np,Q,N]
        cross = torch.einsum("brpqn,bupn->burqn", yfft, xfreq.conj())

    corr = torch.fft.ifft(cross, dim=-1, norm="ortho")                      # [B,Nu,Nr,Q,N]
    energy = xt.abs().square().sum(dim=(2, 3)).clamp_min(1e-6)              # [B,Nu]
    corr = corr * math.sqrt(float(N)) / energy[:, :, None, None, None]
    corr = corr.permute(0, 1, 3, 2, 4).contiguous()                         # [B,Nu,Q,Nr,N]
    corr = select_delay_window(corr, delay_idx)                             # [B,Nu,Q,Nr,L]
    corr = torch.einsum("qp,bupal->buqal", whitener.to(corr.dtype), corr)
    add = torch.fft.fft(corr, dim=-2, norm="ortho")                         # antenna -> angle
    return torch.stack([add.real, add.imag], dim=2).float()                 # [B,Nu,2,Q,A,L]


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


def _pack_current(cfg, coarse, pmask, H, Y_data, labels, data_symbols, noise_var,
                  active_power, channel_type, modulation, snr_db, pilot_spacing,
                  cp_len, cp_wave, nfft, delay_spread_ns, doppler_hz, backend_id,
                  add_input=None, target_add=None, add_input_slow=None,
                  add_input_pilot_only_fast_time=None, ls_grid=None):
    B, Nu, T, Nr, N = coarse.shape
    mask = pmask[:, :, :, None, :].expand(B, Nu, T, Nr, N)
    age, hor, times, raw_age, raw_hor, data_mask, pilot_symbol_mask, scale = _time_features(
        cfg, B, Nu, T, Nr, N, cp_wave, coarse.device
    )
    ce_input = torch.stack([coarse.real, coarse.imag, mask, age, hor], dim=2).float()
    target_h = torch.stack([H.real, H.imag], dim=2).float()
    yd = Y_data.permute(0, 2, 1, 3).contiguous()
    rx_data = torch.stack([yd.real, yd.imag], dim=1).float()
    tx_symbol = torch.stack([data_symbols.real, data_symbols.imag], dim=2).float()
    dev = coarse.device
    policy = str(cfg["system"].get("snr_noise_policy", "measured_ensemble_reference"))

    def fill(v, dtype):
        return torch.full((B,), v, dtype=dtype, device=dev)

    out = {
        "input": ce_input, "target_h": target_h, "rx_data": rx_data,
        "tx_symbol": tx_symbol, "tx_label": labels.long(),
        "data_mask": data_mask.view(1, T).expand(B, T).clone(),
        "pilot_symbol_mask": pilot_symbol_mask.view(1, T).expand(B, T).clone(),
        "symbol_times_s": times.view(1, T).expand(B, T).clone(),
        "pilot_age_seconds": raw_age.view(1, T).expand(B, T).clone(),
        "pilot_horizon_seconds": raw_hor.view(1, T).expand(B, T).clone(),
        "time_feature_scale_s": fill(scale, torch.float32),
        "modulation_id": fill(MOD_TO_ID[str(modulation).lower()], torch.long),
        "channel_type_id": fill(CHANNEL_TO_ID[assert_channel_name(channel_type)], torch.long),
        "noise_var": noise_var.float(), "active_re_signal_power": active_power.float(),
        "snr_db": snr_db.float(), "pilot_spacing": fill(int(pilot_spacing), torch.long),
        "cp_len": fill(int(cp_len), torch.long), "cp_wave_samples": fill(int(cp_wave), torch.long),
        "waveform_fft_size": fill(int(nfft), torch.long),
        "delay_spread_ns": fill(float(delay_spread_ns), torch.float32),
        "max_doppler_hz": fill(float(doppler_hz), torch.float32),
        "channel_backend_id": fill(int(backend_id), torch.long),
        "snr_reference_id": fill(int(SNR_POLICY_TO_ID.get(policy, 1)), torch.long),
        "target_reference_id": fill(3, torch.long),
    }
    if add_input is not None:
        out["add_input"] = add_input.float()
    if target_add is not None:
        out["target_add"] = target_add.float()
    if add_input_slow is not None:
        out["add_input_slow"] = add_input_slow.float()
    if add_input_pilot_only_fast_time is not None:
        out["add_input_pilot_only_fast_time"] = add_input_pilot_only_fast_time.float()
    if ls_grid is not None:
        out["ls_grid"] = torch.stack([ls_grid.real, ls_grid.imag], dim=2).float()
    return out


def _noise_variance(cfg, snr_db, active_power, n_users, device):
    """Noise variance per received RE for the configured SNR convention."""
    policy = str(cfg["system"].get("snr_noise_policy", "measured_ensemble_reference")).lower()
    B = int(snr_db.shape[0])
    if policy == "fixed_ensemble_reference":
        ref = torch.full((B,), float(n_users), device=device, dtype=torch.float32)
    elif policy == "measured_ensemble_reference":
        # One chunk shares a single channel configuration, so the chunk mean is
        # an ensemble estimate of E[|Y|^2] and not a per-realization quantity.
        ref = active_power.mean().reshape(1).expand(B).to(torch.float32)
    else:
        raise ValueError(f"Unknown system.snr_noise_policy={policy!r}")
    return ref * torch.pow(torch.tensor(10.0, device=device), -snr_db / 10.0)


# ---------------------------------------------------------------------------
# Synthetic families
# ---------------------------------------------------------------------------


def simulate_synthetic_batch(cfg, B, *, channel_type, modulation, pilot_spacing,
                             cp_len, delay_spread_ns, doppler_hz, snr_db,
                             device, gen, pdp_profile="exponential"):
    s = cfg["system"]
    if str(device) == "cuda":
        device = torch.device("cuda:0")
    Nr, Nu, N = int(s["n_ant"]), int(s["n_users"]), int(s["n_subcarriers"])
    nfft = int(s["waveform_fft_size"])
    scs = float(s["subcarrier_spacing_hz"])
    cpw = cp_active_to_waveform_samples(cp_len, nfft, N)
    amp = pilot_amplitude(cfg, pilot_spacing, Nu)

    frame, layout, pmask, pidx, pval, labels, data_symbols = _build_frame(
        B, Nu, N, pilot_spacing, modulation, device, gen,
        int(s["frame_num_symbols"]), int(s["pilot_symbol_period"]), amplitude=amp,
    )
    frame_t = _ofdm_modulate_torch(frame, nfft, cpw)
    Ttime = frame_t.shape[-1]
    fs = nfft * scs

    a_static, a_tv, tau = _synthetic_paths(
        B, Nr, Nu, int(s["n_taps"]), float(delay_spread_ns) * 1e-9, channel_type,
        float(doppler_hz), float(s["antenna_spacing_lambda"]), fs, Ttime,
        device, gen, int(s["jakes_num_sinusoids"]), pdp_profile=pdp_profile,
        rician_k_db_min=float(s["rician_k_db_min"]),
        rician_k_db_max=float(s["rician_k_db_max"]),
        nakagami_m_min=float(s["nakagami_m_min"]),
        nakagami_m_max=float(s["nakagami_m_max"]),
        jakes_time_chunk=int(cfg.get("data_pipeline", {}).get("jakes_time_chunk", 4096)),
    )

    margin = int(s["synthetic_sinc_margin"])
    max_delay = float(tau.max()) if tau.numel() else 0.0
    lmin = -margin
    lmax = int(math.ceil(max_delay * fs)) + margin
    lags = torch.arange(lmin, lmax + 1, device=device, dtype=torch.float32)
    sinc = torch.sinc(lags[None, :] - fs * tau[:, None]).to(torch.complex64)  # [L,Klag]

    y = torch.zeros((B, Nr, Ttime), dtype=torch.complex64, device=device)
    # The clean isolated per-user waveform is only used to build the ADD target.
    y_user = torch.zeros((B, Nu, Nr, Ttime), dtype=torch.complex64, device=device)
    if a_static is not None:
        h = torch.einsum("burl,lk->burk", a_static, sinc)
        for j, lag in enumerate(range(lmin, lmax + 1)):
            coef = h[:, :, :, j]
            if 0 <= lag < Ttime:
                contrib = coef[..., None] * frame_t[:, :, :Ttime - lag].unsqueeze(2)
                y_user[:, :, :, lag:] += contrib
                y[:, :, lag:] += contrib.sum(1)
            elif lag < 0 and -lag < Ttime:
                q = -lag
                contrib = coef[..., None] * frame_t[:, :, q:].unsqueeze(2)
                y_user[:, :, :, :Ttime - q] += contrib
                y[:, :, :Ttime - q] += contrib.sum(1)
    else:
        for j, lag in enumerate(range(lmin, lmax + 1)):
            hj = torch.einsum("burlt,l->burt", a_tv, sinc[:, j])
            if 0 <= lag < Ttime:
                contrib = hj[:, :, :, lag:] * frame_t[:, :, :Ttime - lag].unsqueeze(2)
                y_user[:, :, :, lag:] += contrib
                y[:, :, lag:] += contrib.sum(1)
            elif lag < 0 and -lag < Ttime:
                q = -lag
                contrib = hj[:, :, :, :Ttime - q] * frame_t[:, :, q:].unsqueeze(2)
                y_user[:, :, :, :Ttime - q] += contrib
                y[:, :, :Ttime - q] += contrib.sum(1)

    Yclean = _ofdm_demodulate_torch(y, layout.num_blocks, nfft, cpw, N)
    Yclean_user = _ofdm_demodulate_torch(
        y_user.reshape(B * Nu, Nr, Ttime), layout.num_blocks, nfft, cpw, N
    ).reshape(B, Nu, Nr, layout.num_blocks, N)

    db = torch.as_tensor(layout.data_blocks, device=device, dtype=torch.long)
    active_power = torch.index_select(Yclean, 2, db).abs().square().mean(dim=(1, 2, 3)).real
    noise_var = _noise_variance(cfg, snr_db, active_power, Nu, device)
    noise = (torch.randn(y.shape, device=device, generator=gen)
             + 1j * torch.randn(y.shape, device=device, generator=gen)).to(torch.complex64)
    noise = noise * torch.sqrt(noise_var / 2.0).view(B, 1, 1)
    Y = _ofdm_demodulate_torch(y + noise, layout.num_blocks, nfft, cpw, N)

    # ---- exact per-symbol CFR target ------------------------------------
    signed = torch.arange(-(N // 2), -(N // 2) + N, device=device, dtype=torch.float32)
    phase = torch.exp(
        -1j * 2 * math.pi * signed[:, None] * lags[None, :] / float(nfft)
    ).to(torch.complex64)                                                   # [N,Klag]
    T = layout.visible_symbols
    block_len = nfft + cpw
    if a_static is not None:
        Hfreq = torch.einsum("burk,nk->burn", h, phase)
        H = Hfreq.unsqueeze(2).expand(B, Nu, T, Nr, N).contiguous()
    else:
        # For a time-varying channel the diagonal of the OFDM channel matrix is
        # the time average of the tap gain over the useful part of the symbol.
        seg = a_tv[..., block_len:block_len * (T + 1)].reshape(
            B, Nu, Nr, int(s["n_taps"]), T, block_len
        )[..., cpw:]
        at = seg.mean(dim=-1)                                               # [B,Nu,Nr,L,T]
        href = torch.einsum("burlt,lk->burkt", at, sinc)
        H = torch.einsum("burkt,nk->butrn", href, phase)

    delay_idx = delay_window_indices(
        N, add_delay_bins(cfg), delay_guard_bins(cfg), device=device
    )
    dp = cfg.get("data_pipeline", {})
    coarse, ls_grid = _pilot_ls_and_interp(
        Y, layout, pidx, pval, Nu, Nr, N, pilot_spacing,
        want_ls_grid=bool(dp.get("emit_pilot_ls", True)),
    )
    profile = str(
        cfg.get("model", {}).get("ablation_profile", "full")
    ).lower()
    emit_slow = bool(dp.get("emit_slow_time_add", False))
    emit_pilot_only = (bool(dp.get("emit_pilot_only_fast_time_add", False))
                       or profile == "pilot_only_fast_time")
    use_add_branch = profile != "aft_only"

    if profile == "no_fast_time_add" and not emit_slow:
        raise ValueError(
            "no_fast_time_add requires "
            "data_pipeline.emit_slow_time_add=true"
        )

    add_input = None
    target_add = None
    add_slow = None
    add_pilot_only = None

    if use_add_branch:
        # Full / no-offgrid / no-physics / Q7 all use fast ADD input.
        # no_fast_time_add instead consumes the slow-time observation.
        if profile != "no_fast_time_add":
            add_input = _add_backprojection(
                cfg, Y, frame, layout, cpw, delay_idx,
                per_user=False, fast_time=True
            )

        # Keep the same fast-time ADD supervision target for every
        # ADD-enabled ablation.
        target_add = _add_backprojection(
            cfg, Yclean_user, frame, layout, cpw, delay_idx,
            per_user=True, fast_time=True
        )

        if emit_slow:
            add_slow = _add_backprojection(
                cfg, Y, frame, layout, cpw, delay_idx,
                per_user=False, fast_time=False
            )

        if emit_pilot_only:
            add_pilot_only = _add_backprojection(
                cfg, Y, frame, layout, cpw, delay_idx,
                per_user=False, fast_time=True, pilot_only_received=True,
            )

    Yd = torch.index_select(Y, 2, db)
    return _pack_current(
        cfg, coarse, pmask, H, Yd, labels, data_symbols, noise_var, active_power,
        channel_type, modulation, snr_db, pilot_spacing, cp_len, cpw, nfft,
        delay_spread_ns, doppler_hz, 0, add_input=add_input, target_add=target_add,
        add_input_slow=add_slow,
        add_input_pilot_only_fast_time=add_pilot_only, ls_grid=ls_grid,
    )


def _useful_window_average_time_channel_to_ofdm(h_t, nblocks, nfft, cpw, l_min):
    """Average the LTV impulse response over each useful OFDM window, then FFT.

    This is mathematically identical to gathering all useful-window time indices,
    but uses a contiguous slice + reshape instead of allocating a large index
    tensor and ``index_select`` copy.  The target convention is unchanged.
    """
    ltot = int(h_t.shape[-1])
    block_len = int(nfft) + int(cpw)
    required = int(nblocks) * block_len
    if int(h_t.shape[-2]) < required:
        raise RuntimeError("time-channel tensor is too short for useful-window averaging")

    # The first ``required`` samples are exactly nblocks contiguous OFDM blocks.
    # Reshaping preserves all leading Sionna link/antenna dimensions.
    blocks = h_t[..., :required, :].reshape(
        *h_t.shape[:-2], int(nblocks), block_len, ltot
    )
    h_avg = blocks[..., int(cpw):int(cpw) + int(nfft), :].mean(dim=-2)

    if ltot > int(nfft):
        raise ValueError(f"time-channel lag support {ltot} exceeds fft_size={nfft}")
    if ltot < int(nfft):
        h_avg = torch.cat([
            h_avg, h_avg.new_zeros(h_avg.shape[:-1] + (int(nfft) - ltot,))
        ], dim=-1)
    h_avg = torch.roll(h_avg, shifts=int(l_min), dims=-1)
    return torch.fft.fftshift(torch.fft.fft(h_avg, dim=-1), dim=-1)


# ---------------------------------------------------------------------------
# 3GPP TR 38.901 families via Sionna PHY
# ---------------------------------------------------------------------------


def simulate_sionna_batch(cfg, B, *, channel_type, modulation, pilot_spacing, cp_len,
                          delay_spread_ns, doppler_hz, snr_db, device, gen, seed):
    try:
        import sionna.phy
        from sionna.phy.channel import (
            cir_to_time_channel, time_lag_discrete_time_channel,
        )
    except Exception as e:
        raise RuntimeError("3GPP training requires Sionna PHY 2.0.1") from e

    s = cfg["system"]
    if str(device) == "cuda":
        device = torch.device("cuda:0")
    Nr, Nu, N = int(s["n_ant"]), int(s["n_users"]), int(s["n_subcarriers"])
    nfft = int(s["waveform_fft_size"])
    scs = float(s["subcarrier_spacing_hz"])
    fc = float(s["carrier_frequency_hz"])
    cpw = cp_active_to_waveform_samples(cp_len, nfft, N)
    amp = pilot_amplitude(cfg, pilot_spacing, Nu)

    frame, layout, pmask, pidx, pval, labels, data_symbols = _build_frame(
        B, Nu, N, pilot_spacing, modulation, device, gen,
        int(s["frame_num_symbols"]), int(s["pilot_symbol_period"]), amplitude=amp,
    )
    full = _embed_active_sionna(frame, nfft)

    nblocks = layout.num_blocks
    block_len = nfft + cpw
    Ttime = nblocks * block_len
    fs = nfft * scs
    speed = float(doppler_hz) * SPEED_OF_LIGHT / max(fc, 1.0)

    # A universal 25*DS bound here would be physically conservative
    # but made the Sionna discrete-time channel unnecessarily long (especially
    # TDL/CDL-B/C), which dominates memory and runtime.  Use the exact maximum
    # normalized excess delay of the selected TR 38.901 profile, with a tiny
    # configurable guard.  ``time_lag_discrete_time_channel`` itself adds the
    # sinc interpolation support, so this does not truncate a physical path.
    support_margin = float(s.get("threegpp_delay_support_margin", 1.01))
    if support_margin < 1.0:
        raise ValueError("system.threegpp_delay_support_margin must be >= 1.0")
    max_excess_s = (
        support_margin
        * max_normalized_excess_delay(cfg, channel_type)
        * float(delay_spread_ns) * 1e-9
    )
    lmin, lmax = time_lag_discrete_time_channel(
        fs, maximum_delay_spread=float(max_excess_s)
    )
    lmin, lmax = int(lmin), int(lmax)
    ltot = lmax - lmin + 1

    mod, demod, apply = _sionna_ops_cached(
        nfft, cpw, Ttime, lmin, lmax, str(device)
    )
    x_time = mod(full)
    sionna.phy.config.seed = int(seed)
    sionna.phy.config.device = str(device)
    sionna.phy.config.precision = "single"
    bins = torch.as_tensor(active_sionna_grid_indices(nfft, N), device=device, dtype=torch.long)
    model = _make_sionna_model_cached(
        channel_type, Nr, float(delay_spread_ns) * 1e-9, fc, speed,
        float(s["antenna_spacing_lambda"]), int(s["tdl_num_sinusoids"]),
        float(s["tdl_rx_corr_rho"]), str(device),
    )

    nlinks = B * Nu
    link_bs = max(1, min(int(s.get("threegpp_link_batch_size", nlinks)), nlinks))
    xlinks = x_time.reshape(nlinks, 1, 1, Ttime)
    vis = torch.as_tensor(layout.visible_blocks, device=device, dtype=torch.long)

    yc_all = torch.empty((nlinks, Nr, Ttime + ltot - 1), dtype=torch.complex64, device=device)
    H = torch.empty((nlinks, layout.visible_symbols, Nr, N), dtype=torch.complex64, device=device)
    Yclean_user = torch.empty((nlinks, Nr, nblocks, N), dtype=torch.complex64, device=device)
    for start in range(0, nlinks, link_bs):
        stop = min(start + link_bs, nlinks)
        lb = stop - start
        a, tau = model(batch_size=lb, num_time_steps=Ttime + ltot - 1, sampling_frequency=fs)
        ht = cir_to_time_channel(fs, a, tau, l_min=lmin, l_max=lmax,
                                 normalize=bool(s.get("normalize_3gpp_channel", False)))
        yc = apply(xlinks[start:stop], ht, no=None)                # [lb,1,Nr,Ttime+ltot-1]
        hf = _useful_window_average_time_channel_to_ofdm(
            ht, nblocks, nfft, cpw, lmin
        )[:, 0, :, 0, 0]                                               # [lb,Nr,Bsym,Nfft]
        ha = torch.index_select(hf, -1, bins)
        ha = torch.index_select(ha, 2, vis).permute(0, 2, 1, 3)    # [lb,T,Nr,N]
        yc_all[start:stop] = yc[:, 0]
        H[start:stop] = ha
        Yclean_user[start:stop] = torch.index_select(demod(yc)[:, 0], -1, bins)
        del a, tau, ht, yc, hf, ha

    ysum = yc_all.reshape(B, Nu, Nr, -1).sum(dim=1).unsqueeze(1)   # [B,1,Nr,Ttime+ltot-1]
    H = H.reshape(B, Nu, layout.visible_symbols, Nr, N)
    Yclean_user = Yclean_user.reshape(B, Nu, Nr, nblocks, N)

    Yc = torch.index_select(demod(ysum)[:, 0], -1, bins)
    db = torch.as_tensor(layout.data_blocks, device=device, dtype=torch.long)
    active_power = torch.index_select(Yc, 2, db).abs().square().mean(dim=(1, 2, 3)).real
    noise_var = _noise_variance(cfg, snr_db, active_power, Nu, device)
    noise = (torch.randn(ysum.shape, device=device, generator=gen)
             + 1j * torch.randn(ysum.shape, device=device, generator=gen)).to(torch.complex64)
    noise = noise * torch.sqrt(noise_var / 2.0).view(B, 1, 1, 1)
    Y = torch.index_select(demod(ysum + noise)[:, 0], -1, bins)

    delay_idx = delay_window_indices(
        N, add_delay_bins(cfg), delay_guard_bins(cfg), device=device
    )
    dp = cfg.get("data_pipeline", {})
    coarse, ls_grid = _pilot_ls_and_interp(
        Y, layout, pidx, pval, Nu, Nr, N, pilot_spacing,
        want_ls_grid=bool(dp.get("emit_pilot_ls", True)),
    )
    profile = str(
        cfg.get("model", {}).get("ablation_profile", "full")
    ).lower()
    emit_slow = bool(dp.get("emit_slow_time_add", False))
    emit_pilot_only = (bool(dp.get("emit_pilot_only_fast_time_add", False))
                       or profile == "pilot_only_fast_time")
    use_add_branch = profile != "aft_only"

    if profile == "no_fast_time_add" and not emit_slow:
        raise ValueError(
            "no_fast_time_add requires "
            "data_pipeline.emit_slow_time_add=true"
        )

    add_input = None
    target_add = None
    add_slow = None
    add_pilot_only = None

    if use_add_branch:
        # Full / no-offgrid / no-physics / Q7 all use fast ADD input.
        # no_fast_time_add instead consumes the slow-time observation.
        if profile != "no_fast_time_add":
            add_input = _add_backprojection(
                cfg, Y, frame, layout, cpw, delay_idx,
                per_user=False, fast_time=True
            )

        # Keep the same fast-time ADD supervision target for every
        # ADD-enabled ablation.
        target_add = _add_backprojection(
            cfg, Yclean_user, frame, layout, cpw, delay_idx,
            per_user=True, fast_time=True
        )

        if emit_slow:
            add_slow = _add_backprojection(
                cfg, Y, frame, layout, cpw, delay_idx,
                per_user=False, fast_time=False
            )

        if emit_pilot_only:
            add_pilot_only = _add_backprojection(
                cfg, Y, frame, layout, cpw, delay_idx,
                per_user=False, fast_time=True, pilot_only_received=True,
            )

    Yd = torch.index_select(Y, 2, db)
    return _pack_current(
        cfg, coarse, pmask, H, Yd, labels, data_symbols, noise_var, active_power,
        channel_type, modulation, snr_db, pilot_spacing, cp_len, cpw, nfft,
        delay_spread_ns, doppler_hz, 1, add_input=add_input, target_add=target_add,
        add_input_slow=add_slow,
        add_input_pilot_only_fast_time=add_pilot_only, ls_grid=ls_grid,
    )
