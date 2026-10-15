"""DOP-Net dual-3D physics-aware AFT/ADD channel estimator.

The estimator has two parallel branches:
  * AFT: antenna-frequency-time, directly anchored by pilot-LS interpolation.
  * ADD: angle-delay-Doppler, fed by a CE-BEM-like pilot backprojection.

Both branches use full-resolution 3-D convolutions and sparse axial graph
attention.  Fusion is performed only after a fixed physical transform maps the
AFT estimate to the ADD coordinates.  A differentiable physics synthesis layer
maps the fused ADD coefficients back to symbol-level CFR.

Implementation notes
--------------------
* **The delay axis is a physical window, not the whole subcarrier axis.**
  ``L = add_delay_bins(cfg)`` keeps ``[-guard, L-guard)`` circular delay bins,
  which covers every delay the frame-feasibility rule allows plus room for sinc
  leakage on both sides.  At the shipped configuration that is 48 instead of 128
  bins, so the ADD branch is substantially cheaper while retaining acausal
  leakage bins rather than discarding them.
* **The slow-time ablation observes the signal, not the network.**  The
  ``no_fast_time_add`` profile consumes ``add_input_slow``, a genuine
  slow-time-only backprojection of the same received pilots.
* **Mixed precision safety.**  Every tensor that enters ``torch.complex`` or a
  complex product is explicitly cast to float32, so the model is safe under
  ``torch.autocast(bfloat16)``.
* **Optional activation checkpointing** on the axial attention blocks, which is
  what makes a useful batch size fit on a single 24-48 GB GPU.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_utils

from data.add_geometry import (
    add_delay_bins, delay_guard_bins, delay_window_indices, delay_window_seconds,
)
from .graph_attention import AxialSparseGraphBuilder, SparseMultiHeadGAT


ABLATION_PROFILES = {
    "dopnet_lite": {
        "use_aft_branch": True, "use_add_branch": True,
        "use_aft_attention": False, "use_add_attention": False,
        "use_fast_time_add_input": True, "fusion_mode": "physics_gated",
        "fractional_refinement": False,
    },
    # Complete proposed architecture.
    "full": {
        "use_aft_branch": True, "use_add_branch": True,
        "use_aft_attention": True, "use_add_attention": True,
        "use_fast_time_add_input": True, "fusion_mode": "physics_gated",
        "fractional_refinement": True,
    },
    # Branch ablations.
    "aft_only": {
        "use_aft_branch": True, "use_add_branch": False,
        "use_aft_attention": True, "use_add_attention": False,
        "use_fast_time_add_input": False, "fusion_mode": "aft_only",
        "fractional_refinement": False,
    },
    "add_only": {
        "use_aft_branch": False, "use_add_branch": True,
        "use_aft_attention": False, "use_add_attention": True,
        "use_fast_time_add_input": True, "fusion_mode": "add_only",
        "fractional_refinement": True,
    },
    # Attention ablations.
    "dual_cnn_no_attention": {
        "use_aft_branch": True, "use_add_branch": True,
        "use_aft_attention": False, "use_add_attention": False,
        "use_fast_time_add_input": True, "fusion_mode": "physics_gated",
        "fractional_refinement": True,
    },
    "no_aft_attention": {
        "use_aft_branch": True, "use_add_branch": True,
        "use_aft_attention": False, "use_add_attention": True,
        "use_fast_time_add_input": True, "fusion_mode": "physics_gated",
        "fractional_refinement": True,
    },
    "no_add_attention": {
        "use_aft_branch": True, "use_add_branch": True,
        "use_aft_attention": True, "use_add_attention": False,
        "use_fast_time_add_input": True, "fusion_mode": "physics_gated",
        "fractional_refinement": True,
    },
    # Fusion / physics-observation ablations.
    "no_physics_aware_fusion": {
        "use_aft_branch": True, "use_add_branch": True,
        "use_aft_attention": True, "use_add_attention": True,
        "use_fast_time_add_input": True, "fusion_mode": "blackbox_cfr",
        "fractional_refinement": True,
    },
    "no_fast_time_add": {
        "use_aft_branch": True, "use_add_branch": True,
        "use_aft_attention": True, "use_add_attention": True,
        "use_fast_time_add_input": False, "fusion_mode": "physics_gated",
        "fractional_refinement": True,
    },
    # Same backbone, target, and loss as Full.  Only the received pilot-symbol
    # frequency bins fed to the fast-time PWB are restricted to the union of
    # transmitted user pilot bins.  This removes off-comb observations, which
    # may carry Doppler-induced leakage, without changing pilot transmissions.
    "pilot_only_fast_time": {
        "use_aft_branch": True, "use_add_branch": True,
        "use_aft_attention": True, "use_add_attention": True,
        "use_fast_time_add_input": True, "fusion_mode": "physics_gated",
        "fractional_refinement": True,
        "add_observation_source": "pilot_only_fast_time",
    },
    # Same architecture / supervision as Full, but the ADD observation is not
    # independently reconstructed from the received fast-time pilot waveform.
    # Instead it is derived from the LS + frequency/time interpolated CFR that
    # is already present in the AFT input.  This is the key same-backbone
    # control for testing whether waveform-derived ADD adds information beyond
    # a transform/projection of the conventional interpolated CFR.
    "cfr_derived_add": {
        "use_aft_branch": True, "use_add_branch": True,
        "use_aft_attention": True, "use_add_attention": True,
        "use_fast_time_add_input": False, "fusion_mode": "physics_gated",
        "fractional_refinement": True,
        "add_observation_source": "cfr_coarse",
    },
    "no_offgrid_refinement": {
        "use_aft_branch": True, "use_add_branch": True,
        "use_aft_attention": True, "use_add_attention": True,
        "use_fast_time_add_input": True, "fusion_mode": "physics_gated",
        "fractional_refinement": False,
    },
}


def available_ablation_profiles():
    return tuple(ABLATION_PROFILES.keys())


# -----------------------------------------------------------------------------
# Local 3-D feature extractors
# -----------------------------------------------------------------------------


def _group_count(channels: int, requested: int = 8) -> int:
    g = min(int(requested), int(channels))
    while g > 1 and channels % g:
        g -= 1
    return max(1, g)


class ResidualConv3DBlock(nn.Module):
    """Full-resolution residual Conv3D block using GroupNorm.

    GroupNorm is intentionally used instead of BatchNorm because the model is
    commonly trained with small GPU microbatches.
    """

    def __init__(self, channels: int, dropout: float = 0.0, groups: int = 8):
        super().__init__()
        g = _group_count(channels, groups)
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(g, channels),
            nn.GELU(),
            nn.Dropout3d(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(g, channels),
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


class ResidualConv3DEncoder(nn.Module):
    def __init__(self, cin: int, channels: int, depth: int = 3, dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(cin, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualConv3DBlock(channels, dropout=dropout)
              for _ in range(max(1, int(depth) - 1))]
        )

    def forward(self, x):
        return self.blocks(self.stem(x))


class TensorTokenizer3D(nn.Module):
    """[B,C,D1,D2,D3] -> [B,D1,D2,D3,D]."""

    def __init__(self, cin: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(cin, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        z = x.permute(0, 2, 3, 4, 1).contiguous()
        return F.gelu(self.norm(self.proj(z)))


class AxialMixGate3D(nn.Module):
    """Sample-adaptive mixing between physical-distance and learned graph priors."""

    def __init__(self, channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )

    def forward(self, fmap):
        z = fmap.mean(dim=(-3, -2, -1))
        lam = torch.sigmoid(self.net(z))
        return lam[:, 0], lam[:, 1], lam[:, 2]


class AxialGraphBlock3D(nn.Module):
    """Sparse graph attention along D1 -> D3 -> D2.

    For AFT the dimensions are [time, antenna, frequency], so the order is
    time -> frequency -> antenna.  For ADD they are [Doppler, angle, delay], so
    the order is Doppler -> delay -> angle.
    """

    def __init__(self, d1, d2, d3, din, dh, heads, dropout, use_graph_bias,
                 top_k_1, top_k_3, top_k_2, sigma_1, sigma_3, sigma_2):
        super().__init__()
        self.d1, self.d2, self.d3 = int(d1), int(d2), int(d3)
        self.g1 = AxialSparseGraphBuilder(self.d1, float(sigma_1), radius=None, top_k=int(top_k_1))
        self.g3 = AxialSparseGraphBuilder(self.d3, float(sigma_3), radius=None, top_k=int(top_k_3))
        self.g2 = AxialSparseGraphBuilder(self.d2, float(sigma_2), radius=None, top_k=int(top_k_2))
        self.a1 = SparseMultiHeadGAT(din, dh, heads, dropout, use_graph_bias)
        self.a3 = SparseMultiHeadGAT(dh, dh, heads, dropout, use_graph_bias)
        self.a2 = SparseMultiHeadGAT(dh, dh, heads, dropout, use_graph_bias)

    def sparsity_report(self, names=("axis1", "axis3", "axis2")):
        out = {}
        for name, g in zip(names, (self.g1, self.g3, self.g2)):
            out[name] = {"axis_size": g.axis_size, "top_k": g.top_k,
                         "density": g.edge_density}
        return out

    def forward(self, x, l1, l3, l2, return_aux=False):
        B, D1, D2, D3, Din = x.shape
        if (D1, D2, D3) != (self.d1, self.d2, self.d3):
            raise ValueError(
                f"AxialGraphBlock3D expected {(self.d1, self.d2, self.d3)}, got {(D1, D2, D3)}"
            )

        # Axis D1 for every D2-D3 coordinate.
        z = x.permute(0, 2, 3, 1, 4).reshape(B * D2 * D3, D1, Din)
        lam = l1[:, None, None].expand(B, D2, D3).reshape(-1)
        i1, p1, v1, _, _ = self.g1(z, lam)
        z = self.a1(z, i1, p1, v1)
        x = z.reshape(B, D2, D3, D1, -1).permute(0, 3, 1, 2, 4).contiguous()

        # Axis D3 for every D1-D2 coordinate.
        D = x.shape[-1]
        z = x.reshape(B * D1 * D2, D3, D)
        lam = l3[:, None, None].expand(B, D1, D2).reshape(-1)
        i3, p3, v3, _, _ = self.g3(z, lam)
        z = self.a3(z, i3, p3, v3)
        x = z.reshape(B, D1, D2, D3, -1)

        # Axis D2 for every D1-D3 coordinate.
        D = x.shape[-1]
        z = x.permute(0, 1, 3, 2, 4).reshape(B * D1 * D3, D2, D)
        lam = l2[:, None, None].expand(B, D1, D3).reshape(-1)
        i2, p2, v2, _, _ = self.g2(z, lam)
        z = self.a2(z, i2, p2, v2)
        x = z.reshape(B, D1, D3, D2, -1).permute(0, 1, 3, 2, 4).contiguous()

        if not return_aux:
            return x, None
        return x, {"axis1_graph_prior": p1, "axis3_graph_prior": p3, "axis2_graph_prior": p2}


# -----------------------------------------------------------------------------
# Fixed physics operators
# -----------------------------------------------------------------------------


def angle_delay_transform(h: torch.Tensor) -> torch.Tensor:
    """Antenna-frequency -> angle-delay, preserving leading dimensions."""
    return torch.fft.ifft(torch.fft.fft(h, dim=-2, norm="ortho"), dim=-1, norm="ortho")


def from_angle_delay(g: torch.Tensor) -> torch.Tensor:
    """Angle-delay -> antenna-frequency, preserving leading dimensions."""
    return torch.fft.ifft(torch.fft.fft(g, dim=-1, norm="ortho"), dim=-2, norm="ortho")


def dft_beam_coordinates(n_ant: int, spacing_lambda: float, device=None):
    """Return DFT-beam spatial frequency, sin(theta), and angle in degrees.

    The ULA uses ``a_r(theta) = exp(-j 2 pi r d/lambda sin(theta))``.  With the
    unshifted antenna FFT used here, beam k corresponds to
    ``sin(theta) ~= -f_k / (d/lambda)`` with ``f_k = fftfreq(n_ant)``.
    """
    spatial = torch.fft.fftfreq(int(n_ant), d=1.0, device=device)
    spacing = max(float(spacing_lambda), 1e-12)
    sin_theta = (-spatial / spacing).clamp(-1.0, 1.0)
    angle_deg = torch.asin(sin_theta) * (180.0 / math.pi)
    return spatial.float(), sin_theta.float(), angle_deg.float()


class DopplerPhysics(nn.Module):
    """Differentiable Doppler projection/synthesis on a frame-orthogonal grid.

    The dictionary is defined in normalized units of cycles-per-frame,

        nu_q(b) = q / T_frame(b),   q = -(Q-1)/2 ... +(Q-1)/2,

    so with the ``T`` uniformly spaced OFDM symbol times of the frame the basis
    is exactly a truncated DFT matrix, ``B^H B = T I``: cond 1, coherence 0, for
    every allowed CP, and the AFT projection and the ADD matched filter live in
    identical coordinates.

    Identifiability is checked at construction on the *actual pilot sample
    times*, which is the aperture the ADD backprojection really has.  The model
    and generator use the same CP-aware, centred definition imported from
    ``data/add_geometry.py``.
    """

    def __init__(self, cfg: dict, q: int):
        super().__init__()
        s, m = cfg["system"], cfg["model"]
        self.T = int(s["frame_num_symbols"])
        self.Q = int(q)
        if self.Q % 2 != 1:
            raise ValueError("add_doppler_bins must be odd so the grid is symmetric about DC")
        if self.Q > self.T:
            raise ValueError("add_doppler_bins must be <= frame_num_symbols")
        modes = torch.arange(-(self.Q // 2), self.Q // 2 + 1, dtype=torch.float32)
        self.register_buffer("doppler_modes_cycles_per_frame", modes)
        self.base_reg = float(m["doppler_projection_reg"])
        self.max_config_basis_condition = float(m["doppler_max_config_basis_condition"])
        self.max_config_basis_coherence = float(m["doppler_max_config_basis_coherence"])
        self.max_pilot_basis_condition = float(m["doppler_max_pilot_basis_condition"])

        nfft = int(s["waveform_fft_size"])
        nactive = int(s["n_subcarriers"])
        scs = float(s["subcarrier_spacing_hz"])
        period = int(s["pilot_symbol_period"])
        self._nfft, self._nactive, self._scs, self._period = nfft, nactive, scs, period
        max_cp_active = max(int(v) for v in s["cp_len_choices"])
        cp_wave = int(round(max_cp_active * nfft / float(nactive)))
        self.time_feature_scale_s = max(period * (nfft + cp_wave) / float(nfft * scs), 1e-9)
        self.nominal_symbol_duration_s = (nfft + cp_wave) / float(nfft * scs)
        self._validate_configured_dictionary(s)

    # ---- grid -------------------------------------------------------------

    def frame_duration_from_times(self, times_s: torch.Tensor) -> torch.Tensor:
        step = (times_s[:, 1] - times_s[:, 0]).abs().clamp_min(1e-12)
        return step * float(self.T)

    def grid_hz(self, times_s: torch.Tensor) -> torch.Tensor:
        frame = self.frame_duration_from_times(times_s)
        modes = self.doppler_modes_cycles_per_frame.to(times_s.device)
        return modes[None, :] / frame[:, None]

    def bin_spacing_hz(self, times_s: torch.Tensor) -> torch.Tensor:
        return 1.0 / self.frame_duration_from_times(times_s)

    @staticmethod
    def basis(times_s: torch.Tensor, grid_hz: torch.Tensor) -> torch.Tensor:
        if grid_hz.ndim == 1:
            phase = times_s[:, :, None] * grid_hz[None, None, :]
        elif grid_hz.ndim == 2:
            phase = times_s[:, :, None] * grid_hz[:, None, :]
        else:
            raise ValueError("basis() grid_hz must be [Q] or [B,Q]")
        return torch.exp(1j * (2.0 * math.pi) * phase.float()).to(torch.complex64)

    @staticmethod
    def _basis_metrics(basis: torch.Tensor):
        gram = basis.conj().transpose(1, 2) @ basis
        diag = gram.diagonal(dim1=-2, dim2=-1).real.clamp_min(1e-12)
        denom = torch.sqrt(diag[:, :, None] * diag[:, None, :])
        corr = gram.abs() / denom
        eye = torch.eye(corr.shape[-1], device=corr.device, dtype=torch.bool)[None]
        coherence = (
            corr.masked_fill(eye, 0.0).amax(dim=(-2, -1))
            if corr.shape[-1] > 1 else corr.new_zeros(corr.shape[0])
        )
        sv = torch.linalg.svdvals(basis)
        condition = sv[:, 0] / sv[:, -1].clamp_min(1e-12)
        return condition.real, coherence.real, gram

    def pilot_sample_times(self, duration_s: float, fast_time: bool = True) -> torch.Tensor:
        """Physical time of every pilot sample, matching ``data/add_geometry.py``.

        The intra-symbol axis is *centred* on the nominal symbol instant and no
        CP offset is applied, exactly as the generator does.
        """
        pil = torch.arange(0, self.T, self._period, dtype=torch.float32)
        block = pil * float(duration_s)
        if not fast_time:
            return block.repeat_interleave(self._nactive)
        within = (torch.arange(self._nactive, dtype=torch.float32)
                  - 0.5 * (self._nactive - 1)) / float(self._nactive * self._scs)
        return (block[:, None] + within[None, :]).reshape(-1)

    def _validate_configured_dictionary(self, s: dict):
        nfft, nactive, scs = self._nfft, self._nactive, self._scs
        worst_cond = 0.0
        worst_coh = 0.0
        worst_pilot_cond = 0.0
        worst_pilot_coh = 0.0
        worst_slow_rank = self.Q
        for cp_active in s["cp_len_choices"]:
            cp_wave = int(round(int(cp_active) * nfft / float(nactive)))
            dur = (nfft + cp_wave) / float(nfft * scs)
            grid = self.doppler_modes_cycles_per_frame / (self.T * dur)

            times = torch.arange(self.T, dtype=torch.float32)[None] * dur
            cond, coh, _ = self._basis_metrics(self.basis(times, grid))
            worst_cond = max(worst_cond, float(cond.item()))
            worst_coh = max(worst_coh, float(coh.item()))

            tp = self.pilot_sample_times(dur, fast_time=True)[None]
            pcond, pcoh, _ = self._basis_metrics(self.basis(tp, grid))
            worst_pilot_cond = max(worst_pilot_cond, float(pcond.item()))
            worst_pilot_coh = max(worst_pilot_coh, float(pcoh.item()))

            # Slow-time-only aperture: one sample per pilot OFDM symbol.
            ts = torch.arange(0, self.T, self._period, dtype=torch.float32)[None] * dur
            sv = torch.linalg.svdvals(self.basis(ts, grid))[0]
            worst_slow_rank = min(worst_slow_rank, int((sv > 1e-6 * sv[0]).sum().item()))

        self.configured_max_basis_condition = worst_cond
        self.configured_max_basis_coherence = worst_coh
        self.configured_max_pilot_basis_condition = worst_pilot_cond
        self.configured_max_pilot_basis_coherence = worst_pilot_coh
        self.configured_slow_time_rank = worst_slow_rank

        if worst_cond > self.max_config_basis_condition or worst_coh > self.max_config_basis_coherence:
            raise ValueError(
                "Frame Doppler dictionary is not orthogonal: "
                f"worst cond(B)={worst_cond:.4f}, coherence={worst_coh:.2e}. "
                "The grid must be q/T_frame with integer q."
            )
        if worst_pilot_cond > self.max_pilot_basis_condition:
            raise ValueError(
                "Doppler dictionary is not identifiable from the pilot aperture: "
                f"worst cond(B_pilot)={worst_pilot_cond:.2f} exceeds "
                f"model.doppler_max_pilot_basis_condition="
                f"{self.max_pilot_basis_condition:.2f}. Reduce add_doppler_bins, "
                "shorten pilot_symbol_period, or shorten the frame."
            )

    # ---- time axis --------------------------------------------------------

    def symbol_times_from_input(self, x: torch.Tensor) -> torch.Tensor:
        """Recover the per-sample OFDM symbol duration from the public time channels."""
        B, _, T, _, _ = x.shape
        age = x[:, 3, :, 0, 0].detach().float()
        hor = x[:, 4, :, 0, 0].detach().float()
        values = torch.cat([age, hor], dim=1)
        inf = torch.full_like(values, float("inf"))
        pos = torch.where(values > 1e-8, values, inf)
        step = pos.min(dim=1).values
        fallback = torch.full_like(
            step, self.nominal_symbol_duration_s / self.time_feature_scale_s
        )
        step = torch.where(torch.isfinite(step), step, fallback)
        duration = step * float(self.time_feature_scale_s)
        return torch.arange(T, device=x.device, dtype=torch.float32)[None, :] * duration[:, None]

    def diagnostics(self, times_s: torch.Tensor):
        grid = self.grid_hz(times_s)
        basis = self.basis(times_s, grid)
        condition, coherence, _ = self._basis_metrics(basis)
        ridge = torch.full_like(condition, self.base_reg)
        return {"condition": condition, "coherence": coherence, "ridge": ridge,
                "bin_spacing_hz": self.bin_spacing_hz(times_s)}

    # ---- operators --------------------------------------------------------

    def project(self, h_t: torch.Tensor, times_s: torch.Tensor) -> torch.Tensor:
        """Orthogonal projection [B,T,A,L] -> [B,Q,A,L].

        Because ``B^H B = T I`` exactly this is ``B^H / T``.  The ridge solve is
        kept so the operator stays correct if a future grid is not exactly
        orthogonal, but on the shipped configuration it is the identity path.
        """
        B, T, A, L = h_t.shape
        basis = self.basis(times_s, self.grid_hz(times_s))
        bh = basis.conj().transpose(1, 2)
        gram = bh @ basis
        eye = torch.eye(self.Q, device=h_t.device, dtype=gram.dtype)[None]
        gram = gram + float(self.base_reg) * eye
        rhs = torch.einsum("bqt,btal->bqal", bh, h_t)
        coeff = torch.linalg.solve(gram, rhs.reshape(B, self.Q, A * L))
        return coeff.reshape(B, self.Q, A, L)

    def synthesize(self, coeff, times_s, effective_grid_hz: Optional[torch.Tensor] = None):
        """[B,Q,A,L] -> [B,T,A,L], supporting per-voxel off-grid frequencies."""
        grid = self.grid_hz(times_s) if effective_grid_hz is None else effective_grid_hz
        if grid.ndim in (1, 2):
            basis = self.basis(times_s, grid)
            return torch.einsum("btq,bqal->btal", basis, coeff)
        if grid.ndim == 4:
            if grid.shape != coeff.shape:
                raise ValueError(
                    f"voxel Doppler grid must match coeff shape {tuple(coeff.shape)}, "
                    f"got {tuple(grid.shape)}"
                )
            phase = times_s[:, :, None, None, None].float() * grid[:, None, :, :, :].float()
            basis = torch.exp(1j * (2.0 * math.pi) * phase).to(torch.complex64)
            return torch.einsum("btqal,bqal->btal", basis, coeff)
        raise ValueError("effective_grid_hz must be [Q], [B,Q], or [B,Q,A,L]")


# -----------------------------------------------------------------------------
# Physics-aware fusion
# -----------------------------------------------------------------------------


class VoxelADDGate(nn.Module):
    """Per-(Doppler, angle, delay) fusion weight, not a single sample scalar."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(6, hidden, 1), nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv3d(hidden, 1, 1),
        )

    def forward(self, aft_add: torch.Tensor, add_est: torch.Tensor) -> torch.Tensor:
        feat = torch.stack(
            [aft_add.real, aft_add.imag, add_est.real, add_est.imag,
             torch.log1p(aft_add.abs()), torch.log1p(add_est.abs())], dim=1,
        ).float()
        return torch.sigmoid(self.net(feat)).float()


class CFRFusionGate(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(8, hidden, 1), nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv3d(hidden, 1, 1),
        )

    def forward(self, aft, phys, pilot_mask):
        diff = (aft - phys).abs()
        feat = torch.stack(
            [aft.real, aft.imag, phys.real, phys.imag,
             torch.log1p(aft.abs()), torch.log1p(phys.abs()),
             pilot_mask.float(), torch.log1p(diff)], dim=1,
        ).float()
        # Disagreement is evidence, not a hard-coded vote for either branch.
        return torch.sigmoid(self.net(feat)).float()


class BlackBoxCFRFusion(nn.Module):
    """Learned late fusion baseline without physics-aware voxel/CFR gates."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(6, hidden, 1), nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv3d(hidden, 2, 1),
        )

    def forward(self, aft, phys):
        feat = torch.stack(
            [aft.real, aft.imag, phys.real, phys.imag,
             torch.log1p(aft.abs()), torch.log1p(phys.abs())], dim=1,
        ).float()
        delta = self.net(feat).float()
        base = 0.5 * (aft + phys)
        return base + torch.complex(delta[:, 0], delta[:, 1])


class FractionalDopplerRefiner(nn.Module):
    """Bounded per-(Doppler, beam, delay)-voxel frequency corrections.

    The correction is emitted in normalized units of one Doppler bin; the caller
    multiplies by the per-sample bin spacing ``1/T_frame``, so the same head is
    valid for every CP length and the offset regularizer is scale free.
    """

    def __init__(self, channels: int, fraction: float = 0.5):
        super().__init__()
        self.max_offset_bins = float(fraction)
        hidden = max(8, channels // 2)
        self.head = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1), nn.GELU(),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv3d(hidden, 1, kernel_size=1),
        )

    def forward(self, fmap):
        return (torch.tanh(self.head(fmap).squeeze(1)) * self.max_offset_bins).float()


# -----------------------------------------------------------------------------
# Main network
# -----------------------------------------------------------------------------


class DOPNetEstimator(nn.Module):
    """Dual 3-D CNN + dual tri-axial attention + physics-aware fusion.

    Parameters
    ----------
    x : [B,5,T,A,F]
        LS/interpolation input (real, imag, pilot mask, pilot age, pilot horizon).
    add_input : [B,2,Q,A,L]
        Fast-time CE-BEM-like angle-delay-Doppler backprojection.
    add_input_slow : [B,2,Q,A,L]
        Slow-time-only backprojection of the same pilots; required only by the
        ``no_fast_time_add`` ablation.
    add_input_pilot_only_fast_time : [B,2,Q,A,L]
        Fast-time backprojection after masking received pilot symbols to the
        jointly transmitted pilot bins; required by ``pilot_only_fast_time``.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        s, m = cfg["system"], cfg["model"]
        legacy_model_keys = {
            "architecture", "add_doppler_max_hz", "use_aft_branch", "use_add_branch",
            "use_aft_attention", "use_add_attention", "use_fast_time_add_input",
            "fusion_mode", "enable_fractional_doppler_refinement", "cnn_channels",
            "cnn_depth", "node_token_dim", "gat_hidden", "gat_heads", "axial_blocks",
            "graph_context_channels", "doppler_grid_hz", "doppler_target_gram_condition",
        }
        legacy_present = sorted(legacy_model_keys.intersection(m))
        if legacy_present:
            raise ValueError(
                "Legacy model configuration keys are unsupported: " + ", ".join(legacy_present)
            )
        self.cfg = cfg
        self.T = int(s["frame_num_symbols"])
        self.A = int(s["n_ant"])
        self.Fq = int(s["n_subcarriers"])
        self.Q = int(m["add_doppler_bins"])
        self.L = add_delay_bins(cfg)
        self.delay_guard = delay_guard_bins(cfg)
        self.ici_half_band = int(m["ici_half_band"])
        scs = float(s["subcarrier_spacing_hz"])
        self.delay_resolution_s = 1.0 / float(self.Fq * scs)
        self.register_buffer(
            "delay_window_idx",
            delay_window_indices(self.Fq, self.L, self.delay_guard), persistent=False,
        )
        self.register_buffer(
            "delay_grid_s",
            delay_window_seconds(self.Fq, self.L, self.delay_guard, scs), persistent=False,
        )
        spf, sint, adeg = dft_beam_coordinates(self.A, float(s["antenna_spacing_lambda"]))
        self.register_buffer("beam_spatial_frequency", spf)
        self.register_buffer("beam_sin_theta", sint)
        self.register_buffer("beam_angle_deg", adeg)

        self.ablation_profile = str(m["ablation_profile"]).lower()
        if self.ablation_profile not in ABLATION_PROFILES:
            raise ValueError(
                f"Unknown model.ablation_profile={self.ablation_profile!r}; "
                f"choose from {available_ablation_profiles()}"
            )
        profile = dict(ABLATION_PROFILES[self.ablation_profile])
        self.use_aft_branch = profile["use_aft_branch"]
        self.use_add_branch = profile["use_add_branch"]
        self.use_aft_attention = profile["use_aft_attention"]
        self.use_add_attention = profile["use_add_attention"]
        self.use_fast_time_add_input = profile["use_fast_time_add_input"]
        self.add_observation_source = profile.get(
            "add_observation_source",
            "fast_time" if self.use_fast_time_add_input else "slow_time",
        )
        self.fusion_mode = profile["fusion_mode"]
        self.grad_checkpoint = bool(m.get("grad_checkpoint", False))

        dropout = float(m.get("dropout", 0.1))
        use_graph_bias = bool(m.get("use_graph_bias", True))

        # ---------------- AFT branch ----------------
        ca = int(m["aft_channels"])
        da = int(m["aft_token_dim"])
        gha = int(m["aft_gat_hidden"])
        heads_a = int(m["aft_gat_heads"])
        self.aft_encoder = ResidualConv3DEncoder(int(m["in_channels"]), ca, int(m["aft_depth"]), dropout)
        self.aft_tokenizer = TensorTokenizer3D(ca, da)
        self.aft_gate = AxialMixGate3D(ca, int(m.get("gate_hidden", 32)))
        self.aft_blocks = nn.ModuleList()
        din = da
        for _ in range(int(m["aft_axial_blocks"])):
            self.aft_blocks.append(
                AxialGraphBlock3D(
                    self.T, self.A, self.Fq, din, gha, heads_a, dropout, use_graph_bias,
                    int(m["time_top_k"]), int(m["freq_top_k"]), int(m["ant_top_k"]),
                    float(m["sigma_time"]), float(m["sigma_freq"]), float(m["sigma_ant"]),
                )
            )
            din = gha
        aft_ctx = int(m["aft_context_channels"])
        self.aft_context = nn.Sequential(nn.Linear(gha, aft_ctx), nn.GELU())
        self.aft_decoder = nn.Sequential(
            nn.Conv3d(ca + aft_ctx, ca, 1), nn.GELU(),
            ResidualConv3DBlock(ca, dropout=dropout), nn.Conv3d(ca, 2, 3, padding=1),
        )

        # ---------------- ADD branch ----------------
        cd = int(m["add_channels"])
        dd = int(m["add_token_dim"])
        ghd = int(m["add_gat_hidden"])
        heads_d = int(m["add_gat_heads"])
        self.add_encoder = ResidualConv3DEncoder(2, cd, int(m["add_depth"]), dropout)
        self.add_tokenizer = TensorTokenizer3D(cd, dd)
        self.add_gate = AxialMixGate3D(cd, int(m["gate_hidden"]))
        self.add_blocks = nn.ModuleList()
        din = dd
        for _ in range(int(m["add_axial_blocks"])):
            self.add_blocks.append(
                AxialGraphBlock3D(
                    self.Q, self.A, self.L, din, ghd, heads_d, dropout, use_graph_bias,
                    int(m["add_doppler_top_k"]), int(m["add_delay_top_k"]),
                    int(m["add_angle_top_k"]), float(m["add_sigma_doppler"]),
                    float(m["add_sigma_delay"]), float(m["add_sigma_angle"]),
                )
            )
            din = ghd
        add_ctx = int(m["add_context_channels"])
        self.add_context = nn.Sequential(nn.Linear(ghd, add_ctx), nn.GELU())
        self.add_decoder = nn.Sequential(
            nn.Conv3d(cd + add_ctx, cd, 1), nn.GELU(),
            ResidualConv3DBlock(cd, dropout=dropout), nn.Conv3d(cd, 2, 3, padding=1),
        )

        self.physics = DopplerPhysics(cfg, self.Q)
        self.add_fusion_gate = VoxelADDGate(int(m["fusion_hidden"]))
        self.cfr_fusion_gate = CFRFusionGate(int(m["fusion_hidden"]))
        self.enable_fractional_refinement = profile["fractional_refinement"]
        self.doppler_refiner = FractionalDopplerRefiner(
            cd + add_ctx, float(m["max_doppler_offset_fraction"])
        ) if self.enable_fractional_refinement and self.use_add_branch else None
        self.blackbox_cfr_fusion = (
            BlackBoxCFRFusion(int(m["fusion_hidden"]))
            if self.fusion_mode == "blackbox_cfr" else None
        )

        # Disabled ablation components are frozen so reported trainable parameter
        # counts correspond to the model that is actually optimized.
        if not self.use_aft_branch:
            self._freeze_modules(self.aft_encoder, self.aft_tokenizer, self.aft_gate,
                                 self.aft_blocks, self.aft_context, self.aft_decoder)
        elif not self.use_aft_attention:
            self._freeze_modules(self.aft_tokenizer, self.aft_gate, self.aft_blocks, self.aft_context)
        if not self.use_add_branch:
            self._freeze_modules(self.add_encoder, self.add_tokenizer, self.add_gate,
                                 self.add_blocks, self.add_context, self.add_decoder,
                                 self.add_fusion_gate, self.cfr_fusion_gate, self.doppler_refiner)
        elif not self.use_add_attention:
            self._freeze_modules(self.add_tokenizer, self.add_gate, self.add_blocks, self.add_context)
        if self.fusion_mode != "physics_gated":
            self._freeze_modules(self.add_fusion_gate, self.cfr_fusion_gate)

    @staticmethod
    def _freeze_modules(*mods):
        for mod in mods:
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad_(False)

    def graph_sparsity_report(self):
        """Axis size / top_k / density for every axial graph, for the paper."""
        out = {}
        for i, blk in enumerate(self.aft_blocks):
            out[f"aft_block{i}"] = blk.sparsity_report(("time", "frequency", "antenna"))
        for i, blk in enumerate(self.add_blocks):
            out[f"add_block{i}"] = blk.sparsity_report(("doppler", "delay", "angle"))
        return out

    # ---- branch helpers -----------------------------------------------------

    def _to_delay_window(self, g: torch.Tensor) -> torch.Tensor:
        return torch.index_select(g, -1, self.delay_window_idx)

    def _from_delay_window(self, g: torch.Tensor) -> torch.Tensor:
        full = g.new_zeros(g.shape[:-1] + (self.Fq,))
        return full.index_copy(-1, self.delay_window_idx, g)

    def _aft_to_add(self, h: torch.Tensor, times_s: torch.Tensor) -> torch.Tensor:
        g = self._to_delay_window(angle_delay_transform(h))
        return self.physics.project(g, times_s)

    def _add_to_cfr(self, coeff, times_s, effective_grid_hz=None):
        g_t = self.physics.synthesize(coeff, times_s, effective_grid_hz)
        return from_angle_delay(self._from_delay_window(g_t)), g_t

    def _apply_block(self, block, tok, l1, l3, l2, return_block_aux):
        if (self.grad_checkpoint and self.training and torch.is_grad_enabled()
                and not return_block_aux):
            out = ckpt_utils.checkpoint(
                lambda a, b, c, d: block(a, b, c, d, return_aux=False)[0],
                tok, l1, l3, l2, use_reentrant=False,
            )
            return out, None
        return block(tok, l1, l3, l2, return_aux=return_block_aux)

    def _run_axial(self, fmap, tokenizer, gate, blocks, context, return_block_aux, enabled=True):
        B, _, D1, D2, D3 = fmap.shape
        if not enabled:
            cout = int(context[0].out_features)
            ctx = fmap.new_zeros((B, cout, D1, D2, D3))
            half = fmap.new_full((B,), 0.5)
            return ctx, (half, half, half), []
        l1, l3, l2 = gate(fmap)
        tok = tokenizer(fmap)
        aux = []
        for block in blocks:
            tok, a = self._apply_block(block, tok, l1, l3, l2, return_block_aux)
            aux.append(a)
        ctx = context(tok).permute(0, 4, 1, 2, 3).contiguous()
        return ctx, (l1, l3, l2), aux

    # ---- public physics diagnostic ----------------------------------------

    def ici_bands_from_add(self, coeff, times_s, effective_grid_hz=None, half_band=None):
        """Narrow-band ICI operator from ADD coefficients.

        Returns complex ``[B,T,Antenna,F,D]`` with ``D = 2*half_band+1``.
        Band index ``d+half_band`` uses ``d = k_out - k_in``; hence the
        coefficient at output subcarrier ``k`` multiplies input ``x[k-d]``.
        """
        P = self.ici_half_band if half_band is None else int(half_band)
        B, Q, A, L = coeff.shape
        if effective_grid_hz is None:
            grid_eff = self.physics.grid_hz(times_s)[:, :, None, None].expand(B, Q, A, L)
        elif effective_grid_hz.ndim == 1:
            grid_eff = effective_grid_hz.view(1, Q, 1, 1).expand(B, Q, A, L)
        elif effective_grid_hz.ndim == 2:
            grid_eff = effective_grid_hz[:, :, None, None].expand(B, Q, A, L)
        elif effective_grid_hz.ndim == 4:
            grid_eff = effective_grid_hz
        else:
            raise ValueError("effective_grid_hz must be [Q], [B,Q], or [B,Q,A,L]")

        times_s = times_s.float()
        grid_eff = grid_eff.float()
        phase_symbol = times_s[:, :, None, None, None] * grid_eff[:, None, :, :, :]
        basis_symbol = torch.exp(1j * 2.0 * math.pi * phase_symbol).to(torch.complex64)

        n = torch.arange(self.Fq, device=coeff.device, dtype=torch.float32)
        scs = float(self.cfg["system"]["subcarrier_spacing_hz"])
        # Centred on the symbol time, matching data/add_geometry.py.
        tn = (n - 0.5 * (self.Fq - 1)) / float(self.Fq * scs)
        offsets = torch.arange(-P, P + 1, device=coeff.device, dtype=torch.float32)
        phase = (
            grid_eff[..., None, None] * tn[None, None, None, None, None, :]
            - offsets[None, None, None, None, :, None]
            * n[None, None, None, None, None, :] / float(self.Fq)
        )
        leakage = torch.exp(1j * 2.0 * math.pi * phase).mean(dim=-1).to(torch.complex64)
        g = torch.einsum("bqal,btqal,bqald->btald", coeff, basis_symbol, leakage)
        # Scatter the delay window back onto the full subcarrier axis.
        g = g.permute(0, 1, 2, 4, 3)                               # [B,T,A,D,L]
        g = self._from_delay_window(g)                             # [B,T,A,D,F]
        g = g.permute(0, 1, 2, 4, 3)                               # [B,T,A,F,D]
        g = torch.fft.ifft(g, dim=2, norm="ortho")                 # DFT beam -> antenna
        return torch.fft.fft(g, dim=3, norm="ortho")               # delay -> frequency

    # ---- forward -----------------------------------------------------------

    def forward(self, x, add_input=None, add_input_slow=None,
                add_input_pilot_only_fast_time=None,
                return_aux=False, return_block_aux=False):
        if x.ndim != 5:
            raise ValueError(f"DOP-Net expects [B,5,T,A,F], got {tuple(x.shape)}")
        B, C, T, A, Fq = x.shape
        if (T, A, Fq) != (self.T, self.A, self.Fq):
            raise ValueError(f"DOP-Net configured for {(self.T, self.A, self.Fq)}, got {(T, A, Fq)}")

        x = x.float()
        coarse = torch.complex(x[:, 0], x[:, 1])
        times_s = self.physics.symbol_times_from_input(x)
        doppler_diag = self.physics.diagnostics(times_s) if return_aux else None
        half = x.new_full((B,), 0.5)
        aft_lam = (half, half, half)
        add_lam = (half, half, half)
        aft_aux, add_aux = [], []
        aft_est = coarse

        # Branch A: antenna-frequency-time.
        if self.use_aft_branch:
            aft_f = self.aft_encoder(x)
            aft_ctx, aft_lam, aft_aux = self._run_axial(
                aft_f, self.aft_tokenizer, self.aft_gate, self.aft_blocks,
                self.aft_context, return_block_aux, enabled=self.use_aft_attention,
            )
            aft_res_ri = self.aft_decoder(torch.cat([aft_f, aft_ctx], dim=1)).float()
            aft_est = coarse + torch.complex(aft_res_ri[:, 0], aft_res_ri[:, 1])

        # AFT-only is a clean branch ablation: no ADD computation, no fusion.
        if not self.use_add_branch:
            out = aft_est
            hhat = torch.stack([out.real, out.imag], dim=1).float()
            if not return_aux:
                return hhat
            return hhat, {
                "architecture": "dual3d_physics_fusion",
                "ablation_profile": self.ablation_profile,
                "lambda_time": aft_lam[0], "lambda_freq": aft_lam[1], "lambda_ant": aft_lam[2],
                "fast_observation_available": False,
                "aft_block_aux": aft_aux, "add_block_aux": [],
            }

        # Select the ADD observation source.  The proposed Full model uses an
        # independently reconstructed fast-time pilot observation.  The
        # cfr_derived_add control instead projects only the conventional
        # LS/interpolated CFR already contained in x onto the exact same ADD
        # coordinates used by PCDF.  All trainable modules, targets, and losses
        # are otherwise unchanged.
        if self.add_observation_source == "cfr_coarse":
            # coarse: [B,T,A,F] complex LS + frequency/time interpolated CFR
            # _aft_to_add: angle/delay transform + common delay window + the
            # same regularized Doppler projection used by the physics fusion.
            add_coarse = self._aft_to_add(coarse, times_s)
            add_in = torch.stack(
                [add_coarse.real, add_coarse.imag], dim=1
            ).float()
            source = "ls_interpolated_cfr_projection"
        else:
            if self.add_observation_source == "fast_time":
                observation = add_input
                required = "add_input"
                source = "fast_time_pilot_backprojection"
            elif self.add_observation_source == "slow_time":
                # Honest slow-time control: the same received pilots observed
                # without the intra-symbol phase ramp.
                observation = add_input_slow
                required = "add_input_slow"
                source = "slow_time_pilot_backprojection"
            elif self.add_observation_source == "pilot_only_fast_time":
                observation = add_input_pilot_only_fast_time
                required = "add_input_pilot_only_fast_time"
                source = "pilot_only_fast_time_backprojection"
            else:
                raise ValueError(
                    f"Unsupported ADD observation source "
                    f"{self.add_observation_source!r}"
                )

            if observation is None:
                raise ValueError(
                    f"profile {self.ablation_profile!r} requires {required}; "
                    "enable its corresponding data_pipeline.emit_* observation "
                    "when generating the shared epoch or validation cache."
                )
            if observation.ndim != 5 or tuple(observation.shape[2:]) != (self.Q, self.A, self.L):
                raise ValueError(
                    f"ADD observation must be [B,2,{self.Q},{self.A},{self.L}], "
                    f"got {tuple(observation.shape)}"
                )
            add_in = observation.float()
            add_coarse = torch.complex(add_in[:, 0], add_in[:, 1])

        add_f = self.add_encoder(add_in)
        add_ctx, add_lam, add_aux = self._run_axial(
            add_f, self.add_tokenizer, self.add_gate, self.add_blocks,
            self.add_context, return_block_aux, enabled=self.use_add_attention,
        )
        add_latent = torch.cat([add_f, add_ctx], dim=1)
        add_res_ri = self.add_decoder(add_latent).float()
        add_est = add_coarse + torch.complex(add_res_ri[:, 0], add_res_ri[:, 1])

        if self.fusion_mode == "physics_gated" and self.use_aft_branch:
            aft_add = self._aft_to_add(aft_est, times_s)
            beta_add = self.add_fusion_gate(aft_add, add_est)[:, 0]
            fused_add = beta_add * aft_add + (1.0 - beta_add) * add_est
        else:
            beta_add = x.new_zeros((B, self.Q, self.A, self.L))
            fused_add = add_est

        base_grid = self.physics.grid_hz(times_s)[:, :, None, None]
        bin_hz = self.physics.bin_spacing_hz(times_s)[:, None, None, None]
        if self.doppler_refiner is not None:
            offset_bins = self.doppler_refiner(add_latent)
            doppler_offset = offset_bins * bin_hz
            effective_grid = base_grid + doppler_offset
        else:
            offset_bins = torch.zeros(B, self.Q, self.A, self.L, device=x.device, dtype=torch.float32)
            doppler_offset = offset_bins
            effective_grid = base_grid.expand(B, self.Q, self.A, self.L)

        physics_h, _ = self._add_to_cfr(fused_add, times_s, effective_grid)

        if self.fusion_mode == "add_only" or not self.use_aft_branch:
            gamma_aft = x.new_zeros((B, T, A, Fq))
            out = physics_h
        elif self.fusion_mode == "blackbox_cfr":
            gamma_aft = x.new_full((B, T, A, Fq), 0.5)
            out = self.blackbox_cfr_fusion(aft_est, physics_h)
        elif self.fusion_mode == "physics_gated":
            gamma_aft = self.cfr_fusion_gate(aft_est, physics_h, x[:, 2])[:, 0]
            out = gamma_aft * aft_est + (1.0 - gamma_aft) * physics_h
        else:
            raise ValueError(f"Unsupported fusion_mode={self.fusion_mode!r}")

        hhat = torch.stack([out.real, out.imag], dim=1).float()
        if not return_aux:
            return hhat

        return hhat, {
            "architecture": "dual3d_physics_fusion",
            "ablation_profile": self.ablation_profile,
            "add_observation_source": source,
            "lambda_time": aft_lam[0], "lambda_freq": aft_lam[1], "lambda_ant": aft_lam[2],
            "lambda_doppler": add_lam[0], "lambda_delay": add_lam[1], "lambda_angle": add_lam[2],
            "domain_beta_add": beta_add, "domain_gamma_aft": gamma_aft,
            "doppler_offset_hz": doppler_offset,
            "doppler_offset_bins": offset_bins,
            "doppler_bin_spacing_hz": self.physics.bin_spacing_hz(times_s),
            "add_branch_pred": torch.stack([add_est.real, add_est.imag], dim=1).float(),
            "add_fused_pred": torch.stack([fused_add.real, fused_add.imag], dim=1).float(),
            "physics_h": torch.stack([physics_h.real, physics_h.imag], dim=1).float(),
            "fast_observation_available": self.add_observation_source in {
                "fast_time", "pilot_only_fast_time"},
            "doppler_basis_condition": doppler_diag["condition"],
            "doppler_basis_coherence": doppler_diag["coherence"],
            "doppler_projection_ridge": doppler_diag["ridge"],
            "doppler_pilot_basis_condition": self.physics.configured_max_pilot_basis_condition,
            "doppler_slow_time_rank": self.physics.configured_slow_time_rank,
            "beam_spatial_frequency": self.beam_spatial_frequency,
            "beam_sin_theta": self.beam_sin_theta,
            "beam_angle_deg": self.beam_angle_deg,
            "delay_grid_s": self.delay_grid_s,
            "delay_resolution_s": self.delay_resolution_s,
            "aft_block_aux": aft_aux, "add_block_aux": add_aux,
        }
