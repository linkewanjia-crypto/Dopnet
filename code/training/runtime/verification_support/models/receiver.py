import torch
import torch.nn as nn

from .dopnet import DOPNetEstimator
from .baselines import build_baseline, available_baselines


PROPOSED_ESTIMATOR = "dopnet"


def build_core_estimator(cfg):
    """Build the per-user core estimator named by ``model.estimator``."""
    name = str(cfg.get("model", {}).get("estimator", PROPOSED_ESTIMATOR)).lower()
    if name == PROPOSED_ESTIMATOR:
        return DOPNetEstimator(cfg)
    if name in available_baselines():
        return build_baseline(name, cfg)
    raise ValueError(
        f"Unknown model.estimator={name!r}; choose {PROPOSED_ESTIMATOR!r} or one of "
        f"{available_baselines()}"
    )


class SharedUserChannelEstimator(nn.Module):
    """Parameter-shared CE network applied independently to each user.

    The user axis is folded into the batch, so every estimator -- proposed or
    baseline -- sees exactly the same per-user tensors and the same amount of
    information.  No estimator receives channel truth, the noise-free per-user
    link, or the other user's transmitted data.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.base = build_core_estimator(cfg)
        self.estimator_name = str(cfg.get("model", {}).get("estimator", PROPOSED_ESTIMATOR)).lower()
        self.requires_training = bool(getattr(self.base, "requires_training", True))

    def forward(self, x, add_input=None, add_input_slow=None,
                add_input_pilot_only_fast_time=None, noise_var=None,
                ls_grid=None, return_aux=False, return_block_aux=False):
        if x.ndim != 6:
            raise ValueError(f"Expected [B,Nu,5,T,Nr,Nf], got {tuple(x.shape)}")
        B, Nu, C, T, Nr, Nf = x.shape
        flat = x.reshape(B * Nu, C, T, Nr, Nf)

        def fold(v):
            if v is None:
                return None
            if v.ndim < 2:
                raise ValueError("expected a batched tensor")
            return v.reshape(B * Nu, *v.shape[2:])

        add_flat = fold(add_input)
        add_slow_flat = fold(add_input_slow)
        add_pilot_only_flat = fold(add_input_pilot_only_fast_time)
        ls_flat = fold(ls_grid)
        nv = None if noise_var is None else noise_var.reshape(B).repeat_interleave(Nu)

        kwargs = dict(add_input=add_flat, add_input_slow=add_slow_flat,
                      noise_var=nv, ls_grid=ls_flat)
        if isinstance(self.base, DOPNetEstimator):
            kwargs = dict(
                add_input=add_flat, add_input_slow=add_slow_flat,
                add_input_pilot_only_fast_time=add_pilot_only_flat,
            )
            kwargs["return_block_aux"] = return_block_aux

        if return_aux:
            hflat, raw = self.base(flat, return_aux=True, **kwargs)
        else:
            hflat = self.base(flat, **kwargs)
            raw = None
        h = hflat.reshape(B, Nu, 2, T, Nr, Nf)
        if not return_aux:
            return h

        aux = {"architecture": raw.get("architecture") if raw else None,
               "ablation_profile": raw.get("ablation_profile") if raw else None}
        if raw is not None and "add_observation_source" in raw:
            aux["add_observation_source"] = raw["add_observation_source"]
        scalar_keys = ("lambda_time", "lambda_freq", "lambda_ant",
                       "lambda_doppler", "lambda_delay", "lambda_angle")
        for key in scalar_keys:
            if raw is not None and key in raw:
                val = raw[key]
                aux[key] = val.reshape(B, Nu) if torch.is_tensor(val) and val.numel() == B * Nu else val
        for key in ("domain_beta_add", "domain_gamma_aft", "doppler_offset_hz",
                    "doppler_offset_bins", "doppler_bin_spacing_hz", "add_branch_pred",
                    "add_fused_pred", "physics_h", "doppler_basis_condition",
                    "doppler_basis_coherence", "doppler_projection_ridge"):
            if raw is not None and key in raw:
                v = raw[key]
                aux[key] = v.reshape(B, Nu, *v.shape[1:])
        if raw is not None:
            for key in ("beam_spatial_frequency", "beam_sin_theta", "beam_angle_deg",
                        "delay_grid_s", "delay_resolution_s",
                        "doppler_pilot_basis_condition", "doppler_slow_time_rank"):
                if key in raw:
                    aux[key] = raw[key]
            if "fast_observation_available" in raw:
                aux["fast_observation_available"] = raw["fast_observation_available"]
        return h, aux


def data_visible_indices(cfg, device=None):
    from data.frame_layout import make_layout
    s = cfg["system"]
    lay = make_layout(int(s["frame_num_symbols"]), int(s["pilot_symbol_period"]))
    return torch.as_tensor(lay.data_visible, dtype=torch.long, device=device)


def select_data_csi(h_ri, cfg):
    """Score only the data symbols.

    Pilot symbols are context/measurements, not easy target REs.  Every loss and
    every reported NMSE in this project is computed on the data symbols only, for
    the proposed model and for every baseline alike.
    """
    if h_ri.ndim != 6:
        raise ValueError("Expected CSI [B,Nu,2,T,Nr,Nf]")
    return torch.index_select(h_ri, 3, data_visible_indices(cfg, h_ri.device))


def ri_channel_to_complex(h_ri):
    return torch.complex(h_ri[:, :, 0], h_ri[:, :, 1])


def ri_rx_to_complex(y_ri):
    return torch.complex(y_ri[:, 0], y_ri[:, 1])


def complex_symbols_to_ri(z):
    return torch.stack([z.real, z.imag], dim=2)


def linear_multiuser_equalize(h_ri, rx_ri, noise_var, method="mmse"):
    if h_ri.ndim != 6 or rx_ri.ndim != 5:
        raise ValueError("Expected CSI [B,Nu,2,Td,Nr,Nf] and rx [B,2,Td,Nr,Nf]")
    B, Nu, _, Td, Nr, Nf = h_ri.shape
    if rx_ri.shape[2] != Td:
        raise ValueError("CSI/data time dimensions differ")
    h = torch.complex(h_ri[:, :, 0].float(), h_ri[:, :, 1].float())
    y = torch.complex(rx_ri[:, 0].float(), rx_ri[:, 1].float())
    H = h.permute(0, 2, 4, 3, 1).contiguous()
    Y = y.permute(0, 1, 3, 2).unsqueeze(-1)
    Hh = H.conj().transpose(-2, -1)
    gram = Hh @ H
    rhs = Hh @ Y
    eye = torch.eye(Nu, device=gram.device, dtype=gram.dtype).view(1, 1, 1, Nu, Nu)
    method = str(method).lower()
    if method == "mmse":
        nv = torch.as_tensor(noise_var, device=gram.device, dtype=gram.real.dtype)
        if nv.ndim == 0:
            nv = nv.expand(B)
        mat = gram + nv.reshape(B, 1, 1, 1, 1).to(gram.dtype) * eye
    elif method == "zf":
        mat = gram + 1e-6 * eye
    else:
        raise ValueError("equalizer method must be mmse or zf")
    xhat = torch.linalg.solve(mat, rhs).squeeze(-1)
    z = xhat.permute(0, 3, 1, 2).contiguous()
    sig = gram.diagonal(dim1=-2, dim2=-1).real.clamp_min(1e-8)
    if method == "mmse":
        nv2 = torch.as_tensor(noise_var, device=gram.device, dtype=sig.dtype)
        if nv2.ndim == 0:
            nv2 = nv2.expand(B)
        conf = torch.log1p(sig / nv2.reshape(B, 1, 1, 1).clamp_min(1e-8))
    else:
        conf = torch.log1p(sig)
    return z, conf.permute(0, 3, 1, 2).contiguous()


def _shift_frequency_zero(x, offset):
    """Return ``out[..., k] = x[..., k + offset]`` with zero band edges.

    The project's 128 model subcarriers are a contiguous *subset* of the
    1024-point waveform FFT.  They must therefore not be circularly wrapped at
    the active-band edges when an ICI diagonal is applied.
    """
    d = int(offset)
    if d == 0:
        return x
    out = torch.zeros_like(x)
    if d > 0:
        if d < x.shape[-1]:
            out[..., :-d] = x[..., d:]
    else:
        q = -d
        if q < x.shape[-1]:
            out[..., q:] = x[..., :-q]
    return out


def apply_banded_multiuser_channel(h_bands, x):
    """Apply a multi-user ICI channel without materialising a dense matrix.

    Parameters
    ----------
    h_bands : complex tensor, [B,Nu,Td,Nr,Nf,2P+1]
        Band ``d+P`` follows the model physics convention
        ``d = k_out-k_in``; ``h_bands[..., k, d+P]`` is :math:`H[k,k-d]`.
    x : complex tensor, [B,Nu,Td,Nf]
        Transmitted symbols on the active subcarriers.

    Returns
    -------
    complex tensor, [B,Td,Nr,Nf]
    """
    if not torch.is_complex(h_bands) or not torch.is_complex(x):
        raise TypeError("h_bands and x must be complex tensors")
    if h_bands.ndim != 6 or x.ndim != 4:
        raise ValueError(
            "Expected h_bands [B,Nu,Td,Nr,Nf,D] and x [B,Nu,Td,Nf]"
        )
    B, Nu, Td, Nr, Nf, D = h_bands.shape
    if tuple(x.shape) != (B, Nu, Td, Nf):
        raise ValueError(f"Incompatible symbol shape {tuple(x.shape)}")
    if D % 2 != 1:
        raise ValueError("ICI band count D must be odd")

    P = D // 2
    y = h_bands.new_zeros((B, Td, Nr, Nf))
    for j, d in enumerate(range(-P, P + 1)):
        xd = _shift_frequency_zero(x, -d)
        y = y + torch.einsum("butaf,butf->btaf", h_bands[..., j], xd)
    return y


def apply_banded_multiuser_channel_adjoint(h_bands, y):
    """Apply the Hermitian adjoint of :func:`apply_banded_multiuser_channel`."""
    if not torch.is_complex(h_bands) or not torch.is_complex(y):
        raise TypeError("h_bands and y must be complex tensors")
    if h_bands.ndim != 6 or y.ndim != 4:
        raise ValueError(
            "Expected h_bands [B,Nu,Td,Nr,Nf,D] and y [B,Td,Nr,Nf]"
        )
    B, Nu, Td, Nr, Nf, D = h_bands.shape
    if tuple(y.shape) != (B, Td, Nr, Nf):
        raise ValueError(f"Incompatible receive shape {tuple(y.shape)}")
    if D % 2 != 1:
        raise ValueError("ICI band count D must be odd")

    P = D // 2
    x = h_bands.new_zeros((B, Nu, Td, Nf))
    for j, d in enumerate(range(-P, P + 1)):
        at_output_k = torch.einsum(
            "butaf,btaf->butf", h_bands[..., j].conj(), y
        )
        x = x + _shift_frequency_zero(at_output_k, d)
    return x


def _banded_gram_diagonal(h_bands):
    """Diagonal of ``H^H H`` for a Jacobi PCG preconditioner."""
    D = int(h_bands.shape[-1])
    if D % 2 != 1:
        raise ValueError("ICI band count D must be odd")
    P = D // 2
    out = h_bands.real.new_zeros(
        (h_bands.shape[0], h_bands.shape[1], h_bands.shape[2], h_bands.shape[4])
    )
    for j, d in enumerate(range(-P, P + 1)):
        at_output_k = h_bands[..., j].abs().square().sum(dim=3)
        out = out + _shift_frequency_zero(at_output_k, d)
    return out


def diagonal_csi_to_ici_bands(h_ri, half_band):
    """Embed diagonal CFR in an otherwise-zero ICI band tensor."""
    if h_ri.ndim != 6:
        raise ValueError("Expected CSI [B,Nu,2,Td,Nr,Nf]")
    P = int(half_band)
    if P < 0:
        raise ValueError("half_band must be non-negative")
    h = torch.complex(h_ri[:, :, 0].float(), h_ri[:, :, 1].float())
    bands = h.new_zeros(h.shape + (2 * P + 1,))
    bands[..., P] = h
    return bands


def replace_ici_center(h_bands, h_ri):
    """Use an estimator's final CFR on the centre diagonal of shared bands."""
    if h_bands.ndim != 6 or h_ri.ndim != 6:
        raise ValueError("Expected bands [B,Nu,Td,Nr,Nf,D] and RI CSI")
    h = torch.complex(h_ri[:, :, 0].float(), h_ri[:, :, 1].float())
    if tuple(h_bands.shape[:-1]) != tuple(h.shape):
        raise ValueError(
            f"Band/CSI geometry mismatch: {tuple(h_bands.shape)} vs {tuple(h.shape)}"
        )
    out = h_bands.clone()
    out[..., out.shape[-1] // 2] = h
    return out


def select_data_ici_bands(h_bands, cfg):
    """Select data-symbol times from complex ``[B,Nu,T,Nr,Nf,D]`` bands."""
    if h_bands.ndim != 6:
        raise ValueError("Expected ICI bands [B,Nu,T,Nr,Nf,D]")
    return torch.index_select(h_bands, 2, data_visible_indices(cfg, h_bands.device))


def ici_bands_from_dopnet_aux(estimator, ce_input, hhat_ri, aux, half_band=None,
                            replace_center=True):
    """Call DOP-Net's native ADD-to-ICI operator.

    The non-diagonal bands come from ``add_fused_pred`` and the model's
    fractional-Doppler offsets.  By default, the centre diagonal is replaced
    by the final two-stage fused CFR ``hhat_ri``; this retains the model's best
    diagonal estimate while using its ADD representation for ICI cancellation.
    """
    if not isinstance(estimator, SharedUserChannelEstimator):
        raise TypeError("estimator must be SharedUserChannelEstimator")
    if not isinstance(estimator.base, DOPNetEstimator):
        raise TypeError("Native ICI bands require the DOP-Net estimator")
    if ce_input.ndim != 6 or hhat_ri.ndim != 6:
        raise ValueError("Expected CE input and CSI with explicit user axes")
    if "add_fused_pred" not in aux:
        raise KeyError("return_aux=True did not provide add_fused_pred")

    B, Nu, _, T, Nr, Nf = ce_input.shape
    add_ri = aux["add_fused_pred"]
    if add_ri.shape[:3] != (B, Nu, 2):
        raise ValueError(f"Unexpected fused ADD shape {tuple(add_ri.shape)}")
    coeff = torch.complex(add_ri[:, :, 0].float(), add_ri[:, :, 1].float())
    coeff = coeff.reshape(B * Nu, *coeff.shape[2:])

    flat_input = ce_input.reshape(B * Nu, *ce_input.shape[2:])
    times_s = estimator.base.physics.symbol_times_from_input(flat_input)
    base_grid = estimator.base.physics.grid_hz(times_s)[:, :, None, None]
    offsets = aux.get("doppler_offset_hz")
    if offsets is None:
        effective_grid = base_grid.expand_as(coeff.real)
    else:
        offsets = offsets.reshape(B * Nu, *offsets.shape[2:]).float()
        effective_grid = base_grid + offsets

    bands = estimator.base.ici_bands_from_add(
        coeff,
        times_s,
        effective_grid_hz=effective_grid,
        half_band=half_band,
    )
    bands = bands.reshape(B, Nu, T, Nr, Nf, bands.shape[-1])
    if replace_center:
        bands = replace_ici_center(bands, hhat_ri)
    return bands


def _build_dense_banded_channel(h_bands):
    """Materialise ``H`` for the reference direct solver (small tests only)."""
    B, Nu, Td, Nr, Nf, D = h_bands.shape
    P = D // 2
    dense = h_bands.new_zeros((B, Td, Nr, Nf, Nu, Nf))
    kout = torch.arange(Nf, device=h_bands.device)
    for u in range(Nu):
        block = dense[..., u, :]
        for j, d in enumerate(range(-P, P + 1)):
            kin = kout - d
            valid = (kin >= 0) & (kin < Nf)
            if bool(valid.any()):
                kv = kout[valid]
                block[:, :, :, kv, kin[valid]] = h_bands[:, u, :, :, kv, j]
    return dense.reshape(B, Td, Nr * Nf, Nu * Nf)


def banded_multiuser_equalize(h_bands, rx_ri, noise_var, solver="pcg",
                              max_iter=24, tol=1e-5, return_info=False):
    """ICI-aware multi-user LMMSE over a finite set of channel diagonals.

    Solves ``(H^H H + noise_var I)x = H^H y`` jointly over users and active
    subcarriers.  ``solver='pcg'`` is matrix-free; ``solver='direct'`` exists as
    a numerical reference and should not be used for production-size sweeps.
    """
    if h_bands.ndim != 6 or rx_ri.ndim != 5:
        raise ValueError(
            "Expected bands [B,Nu,Td,Nr,Nf,D] and rx [B,2,Td,Nr,Nf]"
        )
    if not torch.is_complex(h_bands):
        raise TypeError("h_bands must be complex")
    B, Nu, Td, Nr, Nf, D = h_bands.shape
    if tuple(rx_ri.shape) != (B, 2, Td, Nr, Nf):
        raise ValueError(f"Incompatible receive shape {tuple(rx_ri.shape)}")
    if D % 2 != 1:
        raise ValueError("ICI band count D must be odd")

    y = torch.complex(rx_ri[:, 0].float(), rx_ri[:, 1].float())
    nv = torch.as_tensor(noise_var, device=h_bands.device, dtype=h_bands.real.dtype)
    if nv.ndim == 0:
        nv = nv.expand(B)
    if tuple(nv.shape) != (B,):
        nv = nv.reshape(B)
    noise_var_4d = nv.reshape(B, 1, 1, 1).clamp_min(1e-10)
    rhs = apply_banded_multiuser_channel_adjoint(h_bands, y)
    solver = str(solver).lower()

    if solver == "direct":
        dense = _build_dense_banded_channel(h_bands)
        hh = dense.conj().transpose(-2, -1)
        gram = hh @ dense
        rhs_dense = (hh @ y.reshape(B, Td, Nr * Nf, 1)).squeeze(-1)
        eye = torch.eye(Nu * Nf, device=gram.device, dtype=gram.dtype)
        mat = gram + nv.reshape(B, 1, 1, 1).to(gram.dtype) * eye
        x = torch.linalg.solve(mat, rhs_dense.unsqueeze(-1)).squeeze(-1)
        x = x.reshape(B, Td, Nu, Nf).permute(0, 2, 1, 3).contiguous()
        iterations = 1
        relres = torch.zeros((B, Td), device=x.device, dtype=x.real.dtype)
    elif solver == "pcg":
        max_iter = int(max_iter)
        tol = float(tol)
        if max_iter < 1 or tol <= 0:
            raise ValueError("PCG requires max_iter >= 1 and tol > 0")

        gram_diag = _banded_gram_diagonal(h_bands)
        precond = (gram_diag + noise_var_4d).clamp_min(1e-10)

        def matvec(v):
            return (
                apply_banded_multiuser_channel_adjoint(
                    h_bands, apply_banded_multiuser_channel(h_bands, v)
                )
                + noise_var_4d.to(v.dtype) * v
            )

        def inner(a, b):
            return (a.conj() * b).sum(dim=(1, 3)).real

        x = torch.zeros_like(rhs)
        r = rhs.clone()
        z = r / precond.to(r.dtype)
        p = z.clone()
        rz = inner(r, z).clamp_min(1e-30)
        bnorm = inner(rhs, rhs).clamp_min(1e-30)
        relres = torch.sqrt(inner(r, r) / bnorm)
        iterations = 0
        for it in range(max_iter):
            ap = matvec(p)
            pap = inner(p, ap).clamp_min(1e-30)
            alpha = (rz / pap).reshape(B, 1, Td, 1).to(p.dtype)
            x = x + alpha * p
            r = r - alpha * ap
            iterations = it + 1
            relres = torch.sqrt(inner(r, r).clamp_min(0.0) / bnorm)
            if bool(torch.all(relres <= tol)):
                break
            z = r / precond.to(r.dtype)
            rz_new = inner(r, z).clamp_min(1e-30)
            beta = (rz_new / rz).reshape(B, 1, Td, 1).to(p.dtype)
            p = z + beta * p
            rz = rz_new
    else:
        raise ValueError("solver must be 'pcg' or 'direct'")

    # Verify the actual normal-equation residual for both solvers. A recursive
    # PCG residual can drift in fp32; a direct solve does not imply zero error.
    actual_r = rhs - (
        apply_banded_multiuser_channel_adjoint(
            h_bands, apply_banded_multiuser_channel(h_bands, x))
        + noise_var_4d.to(x.dtype) * x)
    relres = torch.sqrt(actual_r.abs().square().sum(dim=(1,3)) /
                        rhs.abs().square().sum(dim=(1,3)).clamp_min(1e-30))
    diag = _banded_gram_diagonal(h_bands).clamp_min(1e-8)
    confidence = torch.log1p(diag / noise_var_4d)
    info = {
        "solver": solver,
        "iterations": int(iterations),
        "relative_residual_max": float(relres.max().detach().cpu()),
        "relative_residual_mean": float(relres.mean().detach().cpu()),
        "half_band": D // 2,
        "boundary": "zero",
    }
    if return_info:
        return x, confidence, info
    return x, confidence


class NeuralSymbolRefiner(nn.Module):
    def __init__(self, hidden=96, modulation_embedding_dim=8, n_modulations=4):
        super().__init__()
        self.embedding = nn.Embedding(n_modulations, modulation_embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(3 + modulation_embedding_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2),
        )

    def forward(self, z, confidence, modulation_id):
        B, Nu, Td, Nf = z.shape
        zri = torch.stack([z.real, z.imag], dim=2)
        emb = self.embedding(modulation_id.long())[:, None, None, None, :].expand(B, Nu, Td, Nf, -1)
        feat = torch.cat([zri.permute(0, 1, 3, 4, 2), confidence.unsqueeze(-1), emb], dim=-1)
        delta = self.net(feat).permute(0, 1, 4, 2, 3)
        return zri + delta


class ProjectMultiUserReceiver(nn.Module):
    def __init__(self, cfg, use_neural_detector=None):
        super().__init__()
        self.cfg = cfg
        self.estimator = SharedUserChannelEstimator(cfg)
        r = cfg.get("receiver", {})
        self.equalizer = str(r.get("equalizer", "mmse")).lower()
        self.ici_half_band = int(r.get(
            "ici_half_band", cfg.get("model", {}).get("ici_half_band", 2)
        ))
        self.banded_solver = str(r.get("banded_solver", "pcg")).lower()
        self.banded_max_iter = int(r.get("banded_max_iter", 24))
        self.banded_tol = float(r.get("banded_tol", 1e-5))
        if use_neural_detector is None:
            use_neural_detector = bool(r.get("neural_detector", True))
        self.detector = NeuralSymbolRefiner(
            int(r.get("detector_hidden", 96)), int(r.get("modulation_embedding_dim", 8))
        ) if use_neural_detector else None

    def forward(self, ce_input, rx_data, noise_var, modulation_id, add_input=None,
                add_input_slow=None, add_input_pilot_only_fast_time=None,
                ls_grid=None, return_aux=False):
        hhat, aux = self.estimator(
            ce_input, add_input=add_input, add_input_slow=add_input_slow,
            add_input_pilot_only_fast_time=add_input_pilot_only_fast_time,
            noise_var=noise_var, ls_grid=ls_grid, return_aux=True,
        )
        hd = select_data_csi(hhat, self.cfg)
        if self.equalizer in {"banded_mmse", "ici_mmse"}:
            bands = ici_bands_from_dopnet_aux(
                self.estimator,
                ce_input,
                hhat,
                aux,
                half_band=self.ici_half_band,
                replace_center=True,
            )
            bands = select_data_ici_bands(bands, self.cfg)
            z, confidence = banded_multiuser_equalize(
                bands,
                rx_data,
                noise_var,
                solver=self.banded_solver,
                max_iter=self.banded_max_iter,
                tol=self.banded_tol,
            )
            if return_aux:
                aux["ici_bands"] = bands
        else:
            z, confidence = linear_multiuser_equalize(
                hd, rx_data, noise_var, method=self.equalizer
            )
        zri = self.detector(z, confidence, modulation_id) if self.detector is not None \
            else complex_symbols_to_ri(z)
        return (hhat, zri, aux) if return_aux else (hhat, zri)
