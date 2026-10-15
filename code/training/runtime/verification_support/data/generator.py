"""GPU-native batch generation primitives and online sampler for DOP-Net.

Implementation notes
--------------------
* **Everything is generated on the training device.**  The only host-side work
  left is the categorical draw of the *per-chunk* configuration (channel family,
  modulation, comb spacing, CP, delay spread, Doppler), which is a handful of
  Python ints per micro-batch.
* **The Jakes sum-of-sinusoids is a single batched GEMM** instead of a Python
  loop over the ``M`` sinusoids, avoiding repeated full passes over a
  ``[B,Nu,L,Nr,T_wave]`` accumulator, which was the dominant cost of the whole
  pipeline and was bandwidth-, not FLOP-, bound.
* **Doppler labels are now truthful.**  ``awgn`` / ``rayleigh`` / ``rician`` /
  ``nakagami`` are retained as explicit *block-fading controls* and therefore
  force ``f_d = 0``.  Genuine synthetic fast-time variation lives only in
  ``doubly_selective`` (Jakes/Clarke); 3GPP TDL/CDL profiles use the Sionna time
  channel.  This removes the old semantic overlap between a channel called
  ``rayleigh`` and the dedicated doubly-selective family.
"""
from __future__ import annotations

import math
import copy
import random
from typing import Dict, List

import torch
import numpy as np

from .channels import CHANNEL_TO_ID, assert_channel_name, validate_channel_config
from .threegpp_profiles import is_3gpp_channel
from .frame_feasibility import feasible_cp_choices, feasible_pilot_spacings
from .channel_backend import active_fft_bins, active_sionna_grid_indices
from utils.modem import MOD_TO_ID

SPEED_OF_LIGHT = 299792458.0

# Families that are time-invariant by construction.  The sampler forces their
# Doppler label to zero so that Doppler-stratified evaluation is honest.
STATIC_SYNTHETIC_CHANNELS = {"awgn", "rayleigh", "rician", "nakagami"}
TIME_VARYING_SYNTHETIC_CHANNELS = {"doubly_selective"}


def _gray_to_binary_torch(g: torch.Tensor) -> torch.Tensor:
    b = g.clone()
    b ^= (b >> 1)
    b ^= (b >> 2)
    b ^= (b >> 4)
    return b


def _random_symbols(shape, modulation: str, device, gen: torch.Generator):
    modulation = str(modulation).lower()
    bps = {"bpsk": 1, "qpsk": 2, "16qam": 4, "64qam": 6}[modulation]
    M = 1 << bps
    labels = torch.randint(0, M, shape, device=device, generator=gen, dtype=torch.long)
    if modulation == "bpsk":
        symbols = (2.0 * labels.to(torch.float32) - 1.0).to(torch.complex64)
        return labels, symbols
    side = int(math.sqrt(M))
    bits_axis = bps // 2
    qmask = (1 << bits_axis) - 1
    gq = labels & qmask
    gi = labels >> bits_axis
    bi = _gray_to_binary_torch(gi)
    bq = _gray_to_binary_torch(gq)
    ii = 2.0 * bi.to(torch.float32) - float(side - 1)
    qq = 2.0 * bq.to(torch.float32) - float(side - 1)
    norm = math.sqrt((2.0 / 3.0) * (M - 1))
    return labels, torch.complex(ii / norm, qq / norm).to(torch.complex64)


def _embed_active_fft(frame, n_fft):
    bins = torch.as_tensor(active_fft_bins(int(n_fft), frame.shape[-1]), device=frame.device)
    out = torch.zeros(frame.shape[:-1] + (int(n_fft),), dtype=frame.dtype, device=frame.device)
    out.index_copy_(-1, bins, frame)
    return out


def _embed_active_sionna(frame, n_fft):
    bins = torch.as_tensor(
        active_sionna_grid_indices(int(n_fft), frame.shape[-1]), device=frame.device
    )
    out = torch.zeros(
        frame.shape[:-1] + (int(n_fft),), dtype=frame.dtype, device=frame.device
    )
    out.index_copy_(-1, bins, frame)
    return out


def _ofdm_modulate_torch(frame, n_fft, cp_wave):
    full = _embed_active_fft(frame, n_fft)
    td = torch.fft.ifft(full, n=int(n_fft), dim=-1, norm="ortho")
    if int(cp_wave) > 0:
        td = torch.cat([td[..., -int(cp_wave):], td], dim=-1)
    return td.reshape(td.shape[:-2] + (-1,))


def _ofdm_demodulate_torch(y, n_blocks, n_fft, cp_wave, n_active):
    block_len = int(n_fft) + int(cp_wave)
    useful = y[..., :int(n_blocks) * block_len].reshape(
        y.shape[:-1] + (int(n_blocks), block_len)
    )[..., int(cp_wave):int(cp_wave) + int(n_fft)]
    Y = torch.fft.fft(useful, n=int(n_fft), dim=-1, norm="ortho")
    bins = torch.as_tensor(active_fft_bins(int(n_fft), int(n_active)), device=y.device)
    return torch.index_select(Y, -1, bins)


def _synthetic_pdp_torch(n_taps, profile, device):
    """Torch counterpart of the synthetic PDP definitions used in evaluation."""
    L = int(n_taps)
    profile = str(profile).lower()
    x = torch.arange(L, device=device, dtype=torch.float32)
    if profile == "exponential":
        p = torch.exp(-x / max(L / 2.0, 1.0))
    elif profile == "uniform":
        p = torch.ones(L, device=device, dtype=torch.float32)
    elif profile == "two_cluster":
        c1 = max(0.0, 0.15 * (L - 1))
        c2 = max(0.0, 0.70 * (L - 1))
        s1 = max(0.7, 0.10 * L)
        s2 = max(0.7, 0.13 * L)
        p = (
            torch.exp(-0.5 * ((x - c1) / s1).square())
            + 0.75 * torch.exp(-0.5 * ((x - c2) / s2).square())
        )
    else:
        raise ValueError(f"Unknown PDP profile: {profile}")
    return p / p.sum()


def synthetic_tap_delays_s(n_taps, profile, delay_spread_s, device=None):
    """Tap delays whose discrete PDP has RMS delay spread exactly ``delay_spread_s``."""
    p = _synthetic_pdp_torch(n_taps, profile, device)
    base = torch.arange(int(n_taps), device=p.device, dtype=torch.float32)
    mean = (p * base).sum()
    rms = torch.sqrt((p * (base - mean).square()).sum()).clamp_min(1e-12)
    return (base / rms) * float(delay_spread_s), p


def _jakes_process(B, Nu, L, Nr, M, fd_hz, spacing_lambda, fs, T, device, gen,
                   time_chunk=4096):
    """Clarke/Jakes isotropic scattering, vectorised as a batched GEMM.

    Returns ``[B,Nu,L,Nr,T]`` with unit power per (user, tap, antenna) link,
    i.e. ``E|acc|^2 = 1``.  The angular offset is shared by the ``M`` sinusoids
    of one tap and the phases are independent, which is the standard
    finite-sinusoid construction with reduced PSD bias.

    The antenna steering angle equals the Doppler angle, so the spatial and the
    temporal statistics come from the same physical scatterer -- that coupling
    is exactly what the ADD branch is meant to exploit.
    """
    M = max(8, int(M))
    base_angles = 2.0 * math.pi * (
        torch.arange(M, device=device, dtype=torch.float32) + 0.5
    ) / float(M)
    angle_offset = 2.0 * math.pi * torch.rand((B, Nu, L), device=device, generator=gen)
    phases = 2.0 * math.pi * torch.rand((B, Nu, L, M), device=device, generator=gen)
    theta = base_angles.view(1, 1, 1, M) + angle_offset[..., None]        # [B,Nu,L,M]
    fd = float(fd_hz) * torch.cos(theta)                                  # [B,Nu,L,M]

    rr = torch.arange(Nr, device=device, dtype=torch.float32)
    steer = torch.exp(
        -1j * 2.0 * math.pi * float(spacing_lambda)
        * rr.view(1, 1, 1, 1, Nr) * torch.sin(theta)[..., None]
    ).to(torch.complex64)                                                 # [B,Nu,L,M,Nr]
    steer = steer.permute(0, 1, 2, 4, 3).reshape(B * Nu * L, Nr, M)       # [.,Nr,M]

    acc = torch.empty((B * Nu * L, Nr, int(T)), dtype=torch.complex64, device=device)
    fd_flat = fd.reshape(B * Nu * L, M)
    ph_flat = phases.reshape(B * Nu * L, M)
    step = max(1, int(time_chunk))
    for start in range(0, int(T), step):
        stop = min(start + step, int(T))
        t = torch.arange(start, stop, device=device, dtype=torch.float32) / float(fs)
        temporal = torch.exp(
            1j * (2.0 * math.pi * fd_flat[:, :, None] * t.view(1, 1, -1)
                  + ph_flat[:, :, None])
        ).to(torch.complex64)                                             # [.,M,chunk]
        acc[:, :, start:stop] = torch.bmm(steer, temporal)
    acc = acc.reshape(B, Nu, L, Nr, int(T)) / math.sqrt(float(M))
    return acc


def _specular_ray(B, Nu, Nr, fd_hz, spacing_lambda, fs, T, device, gen,
                  time_chunk=4096):
    """Single deterministic LOS ray with its own AoA and Doppler shift."""
    theta0 = (torch.rand((B, Nu), device=device, generator=gen) - 0.5) * math.pi
    phi0 = 2.0 * math.pi * torch.rand((B, Nu), device=device, generator=gen)
    rr = torch.arange(Nr, device=device, dtype=torch.float32)
    steer = torch.exp(
        -1j * 2.0 * math.pi * float(spacing_lambda)
        * rr.view(1, 1, Nr) * torch.sin(theta0)[..., None]
    ).to(torch.complex64)                                                 # [B,Nu,Nr]
    fd0 = float(fd_hz) * torch.cos(theta0)                                # [B,Nu]
    out = torch.empty((B, Nu, Nr, int(T)), dtype=torch.complex64, device=device)
    step = max(1, int(time_chunk))
    for start in range(0, int(T), step):
        stop = min(start + step, int(T))
        t = torch.arange(start, stop, device=device, dtype=torch.float32) / float(fs)
        temporal = torch.exp(
            1j * (2.0 * math.pi * fd0[..., None] * t.view(1, 1, -1) + phi0[..., None])
        ).to(torch.complex64)                                             # [B,Nu,chunk]
        out[:, :, :, start:stop] = steer[..., None] * temporal[:, :, None, :]
    return out


def _synthetic_paths(B, n_rx, n_users, n_taps, ds_s, channel_type,
                     max_doppler_hz, antenna_spacing_lambda, fs, T,
                     device, gen, num_sinusoids=32, pdp_profile="exponential",
                     rician_k_db_min=0.0, rician_k_db_max=10.0,
                     nakagami_m_min=0.7, nakagami_m_max=2.5,
                     jakes_time_chunk=4096):
    """Generate project-channel path coefficients.

    Normalization convention
    ------------------------
    ``E[sum_l |h_{r,u,l}|^2] = 1`` for every receive-antenna/user link.  We do
    NOT normalize each random realization, because doing so would distort the
    Rayleigh/Rician/Nakagami amplitude distributions.

    Returns ``(a_static, a_tv, tau)`` where exactly one of ``a_static``
    ``[B,Nu,Nr,L]`` and ``a_tv`` ``[B,Nu,Nr,L,T]`` is not ``None``.
    """
    Nr, Nu, L = int(n_rx), int(n_users), int(n_taps)
    channel_type = assert_channel_name(channel_type)

    if channel_type == "awgn":
        # Canonical AWGN test: no fading, no delay spread. A deterministic DFT
        # spatial matrix gives orthogonal user signatures when Nu <= Nr.
        if Nu > Nr:
            raise ValueError("AWGN orthogonal multi-user baseline requires n_users <= n_ant.")
        tau = torch.zeros(1, dtype=torch.float32, device=device)
        r = torch.arange(Nr, device=device, dtype=torch.float32).view(1, 1, Nr, 1)
        u = torch.arange(Nu, device=device, dtype=torch.float32).view(1, Nu, 1, 1)
        a = torch.exp(1j * 2 * math.pi * r * u / max(Nr, 1)).expand(B, Nu, Nr, 1)
        return a.to(torch.complex64), None, tau

    tau, p = synthetic_tap_delays_s(L, pdp_profile, ds_s, device=device)

    if channel_type in {"rayleigh", "rician", "nakagami"}:
        # Explicit block-fading controls.  These families are intentionally
        # time-invariant over one frame and sample_condition() therefore forces
        # their Doppler label to exactly zero.  All fast-time synthetic Doppler
        # is reserved for the dedicated ``doubly_selective`` family.
        theta = (torch.rand((B, Nu, L), device=device, generator=gen) - 0.5) * math.pi
        rr = torch.arange(Nr, device=device, dtype=torch.float32).view(1, 1, 1, Nr)
        steering = torch.exp(
            -1j * 2 * math.pi * float(antenna_spacing_lambda)
            * rr * torch.sin(theta).unsqueeze(-1)
        ).to(torch.complex64)

        if channel_type == "nakagami":
            m_min, m_max = float(nakagami_m_min), float(nakagami_m_max)
            if m_min < 0.5 or m_max < m_min:
                raise ValueError("Nakagami parameters must satisfy m_max >= m_min >= 0.5.")
            m = m_min + (m_max - m_min) * torch.rand((B, Nu, L), device=device, generator=gen)
            try:
                gamma_unit = torch._standard_gamma(m, generator=gen)
            except TypeError:  # PyTorch builds without a generator argument
                gamma_unit = torch.distributions.Gamma(m, torch.ones_like(m)).sample()
            power = gamma_unit * p.view(1, 1, L) / m
            phase = (torch.rand((B, Nu, L), device=device, generator=gen) * 2 - 1) * math.pi
            alpha = torch.sqrt(power).to(torch.complex64) * torch.exp(1j * phase)
        else:
            # CN(0,p_l) diffuse taps.
            re = torch.randn((B, Nu, L), device=device, generator=gen)
            im = torch.randn((B, Nu, L), device=device, generator=gen)
            alpha = torch.complex(re, im) * torch.sqrt(p.view(1, 1, L) / 2.0)
            if channel_type == "rician":
                # K-factor applies to the first tap; preserve its expected power
                # while adding an independent global LOS phase per sample/user.
                k_min, k_max = float(rician_k_db_min), float(rician_k_db_max)
                if k_max < k_min:
                    raise ValueError("rician_k_db_max must be >= rician_k_db_min.")
                Kdb = k_min + (k_max - k_min) * torch.rand((B, Nu), device=device, generator=gen)
                K = torch.pow(10.0, Kdb / 10.0)
                phi0 = 2.0 * math.pi * torch.rand((B, Nu), device=device, generator=gen)
                los = torch.sqrt(p[0]).to(torch.complex64) * torch.exp(1j * phi0)
                alpha[:, :, 0] = (
                    torch.sqrt(1.0 / (K + 1.0)) * alpha[:, :, 0]
                    + torch.sqrt(K / (K + 1.0)) * los
                )

        a = (steering * alpha.unsqueeze(-1)).permute(0, 1, 3, 2).contiguous()
        return a, None, tau

    if channel_type not in TIME_VARYING_SYNTHETIC_CHANNELS:
        raise ValueError(f"Unhandled synthetic channel type {channel_type!r}")

    # ---- time-varying families -------------------------------------------
    acc = _jakes_process(
        B, Nu, L, Nr, num_sinusoids, max_doppler_hz, antenna_spacing_lambda,
        fs, T, device, gen, time_chunk=jakes_time_chunk,
    )                                                          # [B,Nu,L,Nr,T], unit power

    acc = acc * torch.sqrt(p).view(1, 1, L, 1, 1)
    a_tv = acc.permute(0, 1, 3, 2, 4).contiguous()             # [B,Nu,Nr,L,T]
    return None, a_tv, tau


def _concat_batches(chunks: List[Dict[str, torch.Tensor]]):
    keys = chunks[0].keys()
    return {k: torch.cat([c[k] for c in chunks], dim=0) for k in keys}


class BatchChannelGenerator:
    """Online GPU generator.

    One *chunk* shares a single (channel family, modulation, comb, CP, delay
    spread, Doppler) configuration; SNR is drawn per sample inside the chunk.
    Chunks are concatenated into a buffer and shuffled before being sliced into
    optimizer batches, so a batch mixes several configurations.
    """

    def __init__(self, cfg, device, seed=42):
        # Validation adds defaults; never mutate the caller's experiment or
        # checkpoint configuration through a shared nested dictionary.
        self.cfg = copy.deepcopy(cfg)
        cfg = self.cfg
        self.device = torch.device(device)
        self.seed = int(seed)
        s = cfg["system"]
        validate_channel_config(s)
        self.channels = [assert_channel_name(c) for c in s["channel_types"]]
        w = s.get("channel_type_weights")
        if w is None:
            self.weights = [1.0 / len(self.channels)] * len(self.channels)
        else:
            z = float(sum(w))
            self.weights = [float(x) / z for x in w]
        self.modulations = list(s["modulation_choices"])
        self.pilots = list(map(int, s["pilot_spacing_choices"]))
        pw = s.get("pilot_spacing_weights")
        self.pilot_weights = None if pw is None else [float(x) / sum(pw) for x in pw]
        self.cps = list(map(int, s["cp_len_choices"]))
        self.ds_synthetic = list(map(float, s.get("delay_spread_ns_choices", [100, 300, 1000])))
        self.doppler_synthetic = list(map(float, s.get("doppler_hz_choices", [0, 400, 800])))
        self.ds_3gpp = list(map(float, s.get("threegpp_delay_spread_ns_choices", [100, 300, 1000])))
        self.doppler_3gpp = list(map(float, s.get("threegpp_doppler_hz_choices", [0, 400, 800])))
        dp = cfg.get("data_pipeline", {})
        self.generation_batch_size = int(dp.get("generation_batch_size", 256))
        self.microbatch_size = int(dp.get("microbatch_size", 16))
        # Optional split knobs.  Defaults remain identical, preserving the exact
        # Default condition-mixture behavior unless explicitly enabled.
        self.synthetic_microbatch_size = int(
            dp.get("synthetic_microbatch_size", self.microbatch_size)
        )
        self.threegpp_microbatch_size = int(
            dp.get("threegpp_microbatch_size", self.microbatch_size)
        )
        if self.synthetic_microbatch_size < 1 or self.threegpp_microbatch_size < 1:
            raise ValueError("microbatch sizes must be >= 1")

        # If synthetic and 3GPP chunks use different sizes, sampling channel
        # families with the raw per-sample weights would bias the *sample*
        # distribution toward the larger chunk.  Sample chunks with p_i/b_i
        # instead, so q_i*b_i normalizes back to the configured p_i exactly in
        # expectation. With equal chunk sizes this reduces to the original
        # weights bit-for-bit.
        raw_chunk_weights = []
        for ch, w in zip(self.channels, self.weights):
            bs = self.threegpp_microbatch_size if is_3gpp_channel(ch) else self.synthetic_microbatch_size
            raw_chunk_weights.append(float(w) / float(bs))
        zc = float(sum(raw_chunk_weights))
        self.chunk_sampling_weights = [x / zc for x in raw_chunk_weights]

        self.shuffle_buffer = bool(dp.get("shuffle_buffer", True))
        self.require_cuda = bool(dp.get("require_cuda", False))
        if self.require_cuda and self.device.type != "cuda":
            raise RuntimeError(
                "data_pipeline.require_cuda=true but the generator device is not CUDA. "
                "Use --device cuda (or device: cuda) on the training machine."
            )

    def _chunk_weights_for_remaining(self, remaining):
        """Chunk-family probabilities corrected for the actual remaining buffer.

        The final chunk can be smaller than a family's preferred chunk size.
        Recomputing q_i ∝ p_i / min(b_i, remaining) keeps the desired sample-level
        family distribution unbiased at the buffer tail as well.
        """
        remaining = max(1, int(remaining))
        raw = []
        for ch, w in zip(self.channels, self.weights):
            pref = self.threegpp_microbatch_size if is_3gpp_channel(ch) else self.synthetic_microbatch_size
            actual = min(int(pref), remaining)
            raw.append(float(w) / float(actual))
        z = float(sum(raw))
        return [x / z for x in raw]

    # ---- deterministic streams -------------------------------------------

    def _python_rng(self, epoch, buffer_index):
        return random.Random(self.seed + 1000003 * int(epoch) + 7919 * int(buffer_index))

    def _torch_gen(self, epoch, buffer_index, chunk_index):
        # Explicit device generator keeps data RNG separate from model/dropout RNG.
        g = torch.Generator(device=self.device)
        g.manual_seed(
            self.seed + 1000003 * int(epoch) + 7919 * int(buffer_index)
            + 104729 * int(chunk_index)
        )
        return g

    def _choice(self, py, seq, weights=None):
        if weights is None:
            return seq[py.randrange(len(seq))]
        return py.choices(seq, weights=weights, k=1)[0]

    def sample_condition(self, epoch, buffer_index, chunk_index, channel_weights=None):
        """Draw one chunk configuration.

        ``channel_weights`` defaults to the configured per-sample distribution.
        ``iter_batches`` passes the chunk-size-corrected distribution when
        synthetic and 3GPP microbatch sizes differ.
        """
        py = self._python_rng(epoch, buffer_index * 1000 + chunk_index)
        ch = self._choice(
            py, self.channels,
            self.weights if channel_weights is None else channel_weights,
        )
        mod = self._choice(py, self.modulations)
        if is_3gpp_channel(ch):
            ds = float(self._choice(py, self.ds_3gpp))
            dop = float(self._choice(py, self.doppler_3gpp))
        else:
            ds = float(self._choice(py, self.ds_synthetic))
            dop = float(self._choice(py, self.doppler_synthetic))
        # Static families must not advertise a non-zero Doppler label, otherwise
        # Doppler-stratified reporting silently pools static samples into the
        # high-mobility bins.
        if ch in STATIC_SYNTHETIC_CHANNELS:
            dop = 0.0
        # CP must cover the channel and the comb must keep the delay axis
        # unambiguous, so both are sampled inside the feasible set implied by the
        # delay spread and the channel family rather than independently of them.
        cps = [c for c in self.cps if c in feasible_cp_choices(self.cfg, ds, ch)]
        cp = int(self._choice(py, cps))
        allowed = feasible_pilot_spacings(self.cfg, cp)
        pilots = [v for v in self.pilots if v in allowed]
        weights = None
        if self.pilot_weights is not None:
            weights = [self.pilot_weights[self.pilots.index(v)] for v in pilots]
        pilot = int(self._choice(py, pilots, weights))
        return {"channel_type": ch, "modulation": mod, "delay_spread_ns": ds,
                "doppler_hz": dop, "cp_len": cp, "pilot_spacing": pilot}

    @torch.no_grad()
    def _chunk(self, B, epoch, buffer_index, chunk_index, cond=None):
        cond = self.sample_condition(epoch, buffer_index, chunk_index) if cond is None else cond
        gen = self._torch_gen(epoch, buffer_index, chunk_index)
        s = self.cfg["system"]
        snr = (
            float(s["snr_db_min"])
            + (float(s["snr_db_max"]) - float(s["snr_db_min"]))
            * torch.rand((B,), device=self.device, generator=gen)
        )
        chunk_seed = (
            self.seed + 15485863 * int(epoch)
            + 32452843 * int(buffer_index)
            + 49979687 * int(chunk_index)
        )
        out = self.simulate(B, cond, snr, gen, chunk_seed)
        if self.require_cuda:
            wrong = {k: str(v.device) for k, v in out.items()
                     if torch.is_tensor(v) and v.device.type != "cuda"}
            if wrong:
                raise RuntimeError(f"GPU-native data invariant violated: {wrong}")
        return out

    @torch.no_grad()
    def simulate(self, B, cond, snr_db, gen, seed):
        from .waveform import simulate_synthetic_batch, simulate_sionna_batch
        s = self.cfg["system"]
        ch = cond["channel_type"]
        if is_3gpp_channel(ch):
            # Sionna config.seed can reset default Torch generators as well as
            # its private streams. Preserve model/dropout RNG around simulation
            # so validation cache hits and simulator calls do not reseed learning.
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            python_state=random.getstate(); numpy_state=np.random.get_state()
            try:
                with torch.random.fork_rng(devices=devices):
                    return simulate_sionna_batch(
                        self.cfg, B, channel_type=ch, modulation=cond["modulation"],
                        pilot_spacing=cond["pilot_spacing"], cp_len=cond["cp_len"],
                        delay_spread_ns=cond["delay_spread_ns"], doppler_hz=cond["doppler_hz"],
                        snr_db=snr_db, device=self.device, gen=gen, seed=seed,
                    )
            finally:
                random.setstate(python_state); np.random.set_state(numpy_state)
        profiles = dict(s.get("channel_pdp_profiles", {}) or {})
        return simulate_synthetic_batch(
            self.cfg, B, channel_type=ch, modulation=cond["modulation"],
            pilot_spacing=cond["pilot_spacing"], cp_len=cond["cp_len"],
            delay_spread_ns=cond["delay_spread_ns"], doppler_hz=cond["doppler_hz"],
            snr_db=snr_db, device=self.device, gen=gen,
            pdp_profile=str(profiles.get(ch, s.get("synthetic_pdp_profile", "exponential"))),
        )

    # ---- iteration --------------------------------------------------------

    @torch.no_grad()
    def iter_batches(self, n_samples, epoch, train_batch_size):
        n_samples = int(n_samples)
        train_batch_size = int(train_batch_size)
        produced = 0
        buffer_index = 0
        while produced < n_samples:
            target = min(self.generation_batch_size, n_samples - produced)
            chunks = []
            made = 0
            ci = 0
            while made < target:
                remaining = target - made
                cond = self.sample_condition(
                    epoch, buffer_index, ci,
                    channel_weights=self._chunk_weights_for_remaining(remaining),
                )
                preferred = (
                    self.threegpp_microbatch_size
                    if is_3gpp_channel(cond["channel_type"])
                    else self.synthetic_microbatch_size
                )
                b = min(max(1, int(preferred)), target - made)
                chunks.append(self._chunk(b, epoch, buffer_index, ci, cond=cond))
                made += b
                ci += 1
            buf = _concat_batches(chunks)
            if self.shuffle_buffer:
                sg = self._torch_gen(epoch, buffer_index, 99991)
                perm = torch.randperm(target, device=self.device, generator=sg)
                buf = {k: v.index_select(0, perm) for k, v in buf.items()}
            for start in range(0, target, train_batch_size):
                stop = min(start + train_batch_size, target)
                yield {k: v[start:stop] for k, v in buf.items()}
            produced += target
            buffer_index += 1

    @torch.no_grad()
    def materialize_cpu(self, n_samples, epoch=0, batch_size=32):
        chunks = []
        for b in self.iter_batches(n_samples, epoch, batch_size):
            chunks.append({k: v.detach().cpu() for k, v in b.items()})
        return _concat_batches(chunks)
