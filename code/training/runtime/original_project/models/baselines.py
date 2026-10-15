"""Reference channel estimators used as comparison baselines.

Every baseline implements the same interface as the proposed estimator's core
module: it consumes the ``[B,5,T,A,F]`` public input (LS/interpolation real and
imaginary parts, pilot mask, pilot age, pilot horizon) and returns
``[B,2,T,A,F]``.  Nothing here ever sees ``target_h``, ``target_add`` or the
noise-free per-user link, so the comparison is fair by construction.

Classical references
--------------------
``ls_interp``     Lower bound: the LS estimate at the pilot resource elements,
                  linearly interpolated in frequency and time.  This is exactly
                  channels 0/1 of the shared input, so every learned method is
                  measured against the same starting point.
``lmmse_wiener``  2x1D Wiener smoother on the LS-interpolated grid with lag
                  correlations calibrated on the *training* distribution and the
                  true noise variance.  This is an optimistic classical reference,
                  not a mathematical upper bound on learned estimators.
``cebem_ls``      Ridge-regularized complex-exponential CE-BEM least squares on
                  the pilot occasions, with delay-domain truncation.  This is
                  the proposed ADD branch with the network removed, so it
                  isolates how much of the gain is model-driven and how much is
                  learned.
``dpsbem_ls``     Same estimator on a Slepian / discrete prolate spheroidal
                  basis (Zemen & Mecklenbraeuker, 2005), which is the standard
                  answer to the Gibbs-type edge error of a CE-BEM on a
                  non-periodic observation window.

Deep-learning references
------------------------
``channelnet``    Architecture-faithful PyTorch port of the authors' released
                  SRCNN + DnCNN model.  Real and imaginary channel images are
                  processed as independent one-channel samples with shared weights.
``reesnet``       Paper-faithful residual CNN with learned transposed-convolution
                  upsampling.  No official original-author source release was found;
                  this is an independent PyTorch reimplementation of the published
                  topology, cross-checked against a public reference implementation.
``ha02``          PyTorch port guided by the HA02 authors' released MATLAB code:
                  Transformer encoder followed by a residual CNN decoder.
``dualcnn``       Paper-faithful low-complexity sequential SF-CNN -> DFT ->
                  angle-delay CNN -> inverse DFT topology.

All learned baselines are called through the same public ``[B,5,T,A,F]`` DOP-Net
interface and are trained/evaluated on the same data stream and targets.  To
preserve the published architectures, models consume only the public channels
their native topology needs (normally interpolated LS real/imaginary values);
metadata channels are not injected merely to equalize input width.  No baseline
receives target CSI or hidden simulator state.  These are architecture-faithful
adapters to the DOP-Net system dimensions, not claims of bit-for-bit reproduction of
original framework versions, datasets, pretrained weights, or training recipes.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.add_geometry import add_delay_bins, delay_guard_bins, delay_window_indices
from data.frame_layout import make_layout, time_feature_scale_s


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _complex(x_ri):
    return torch.complex(x_ri[:, 0].float(), x_ri[:, 1].float())


def _ri(z):
    return torch.stack([z.real, z.imag], dim=1).float()


def _extract_pilot_lattice(x, pilot_visible_idx, n_freq):
    """Extract a fixed-size lattice using only true pilot-position LS values.

    ``x`` keeps the public DOP-Net layout ``[B,5,T,A,F]``. Channels 0/1
    contain the common LS/interpolation estimate and channel 2 is the pilot mask.
    Selecting mask==1 therefore recovers the actual observed pilot LS samples
    without injecting non-pilot interpolated values or target CSI.

    Returns ``[B*A,2,Npilot_symbol,Npilot_freq]``.
    """
    x = x.float()
    B, C, T, A, Fq = x.shape
    if C < 3:
        raise ValueError("Pilot-faithful baselines require public pilot-mask channel x[:,2].")
    idx_t = pilot_visible_idx.to(x.device)
    nt = int(idx_t.numel())
    if nt < 1:
        raise ValueError("At least one visible pilot OFDM symbol is required.")
    n_freq = max(1, min(int(n_freq), Fq))

    ri = x[:, :2].permute(0, 3, 1, 2, 4).reshape(B * A, 2, T, Fq)
    mask = x[:, 2:3].permute(0, 3, 1, 2, 4).reshape(B * A, 1, T, Fq)
    ri = torch.index_select(ri, 2, idx_t)
    mask = torch.index_select(mask, 2, idx_t).squeeze(1) > 0.5

    pos = torch.arange(Fq, device=x.device).view(1, 1, Fq)
    sort_key = torch.where(mask, pos, pos + Fq)
    gather_idx = torch.argsort(sort_key, dim=-1)[..., :n_freq]
    valid = torch.gather(mask, -1, gather_idx)
    gather_idx = gather_idx.unsqueeze(1).expand(-1, 2, -1, -1)
    pilot = torch.gather(ri, -1, gather_idx)
    return pilot * valid.unsqueeze(1).to(pilot.dtype)


def _center_crop_or_pad_2d(x, target_hw):
    """Match learned 2-D upsampling by symmetric crop/zero-pad, never interpolation."""
    th, tw = (int(target_hw[0]), int(target_hw[1]))
    h, w = x.shape[-2:]
    if h > th:
        top = (h - th) // 2
        x = x[..., top:top + th, :]
    if w > tw:
        left = (w - tw) // 2
        x = x[..., :, left:left + tw]
    h, w = x.shape[-2:]
    ph, pw = max(0, th - h), max(0, tw - w)
    if ph or pw:
        top, bottom = ph // 2, ph - ph // 2
        left, right = pw // 2, pw - pw // 2
        x = F.pad(x, (left, right, top, bottom))
    return x


class BaselineBase(nn.Module):
    """Common plumbing so baselines can be dropped into the training script."""

    name = "baseline"
    requires_training = True

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        s, m = cfg["system"], cfg["model"]
        self.T = int(s["frame_num_symbols"])
        self.A = int(s["n_ant"])
        self.Fq = int(s["n_subcarriers"])
        self.L = add_delay_bins(cfg)
        self.guard = delay_guard_bins(cfg)
        self.layout = make_layout(self.T, int(s["pilot_symbol_period"]))
        self.register_buffer(
            "delay_window_idx",
            delay_window_indices(self.Fq, self.L, self.guard), persistent=False,
        )
        self.register_buffer(
            "pilot_visible_idx",
            torch.as_tensor(self.layout.pilot_visible, dtype=torch.long), persistent=False,
        )

    # -- to be implemented by subclasses ------------------------------------
    def estimate(self, x, **kw):
        raise NotImplementedError

    def forward(self, x, add_input=None, add_input_slow=None, noise_var=None,
                ls_grid=None, return_aux=False, return_block_aux=False):
        h = self.estimate(x, add_input=add_input, add_input_slow=add_input_slow,
                          noise_var=noise_var, ls_grid=ls_grid)
        if not return_aux:
            return h
        return h, {"architecture": self.name, "ablation_profile": self.name,
                   "fast_observation_available": False,
                   "aft_block_aux": [], "add_block_aux": []}

    def graph_sparsity_report(self):
        return {}


# ---------------------------------------------------------------------------
# classical baselines
# ---------------------------------------------------------------------------


class LSInterpEstimator(BaselineBase):
    """Least squares at the pilots, linear interpolation elsewhere."""

    name = "ls_interp"
    requires_training = False

    def estimate(self, x, **kw):
        return x[:, :2].float().contiguous()



class DFTLSEstimator(BaselineBase):
    """DFT-domain denoising of the common LS/interpolation estimate.

    The interpolated CFR is transformed to the delay domain and projected onto
    the same physically supported delay window used by the project. No target
    CSI, instantaneous delay spread, or test-channel truth is consumed.
    """

    name = "dft_ls"
    requires_training = False

    def estimate(self, x, **kw):
        h = _complex(x)
        g = torch.fft.ifft(h, dim=-1, norm="ortho")
        gd = torch.index_select(g, -1, self.delay_window_idx.to(g.device))
        full = g.new_zeros(g.shape)
        full.index_copy_(-1, self.delay_window_idx.to(g.device), gd)
        return _ri(torch.fft.fft(full, dim=-1, norm="ortho"))


class LMMSEWienerEstimator(BaselineBase):
    """Separable frequency-then-time Wiener smoother on the LS grid.

    The lag correlations ``r_f`` and ``r_t`` are estimated once from a
    calibration set drawn from the *training* distribution with its own seed;
    no test realization is used.  The noise term uses the true per-sample noise
    variance, divided by the number of pilot REs that support one interpolated
    sample, which is the standard bound-style assumption for an LMMSE reference.
    """

    name = "lmmse_wiener"
    requires_training = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self.register_buffer("r_f", torch.zeros(self.Fq, dtype=torch.complex64))
        self.register_buffer("r_t", torch.zeros(self.T, dtype=torch.complex64))
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))
        self.noise_scale = float(cfg["model"].get("lmmse_noise_scale", 1.0))

    @torch.no_grad()
    def calibrate(self, target_h_batches):
        """``target_h_batches``: iterable of ``[B,2,T,A,F]`` true CSI tensors."""
        acc_f = None
        acc_t = None
        power = 0.0
        count = 0
        for th in target_h_batches:
            h = _complex(th.to(self.r_f.device))                    # [B,T,A,F]
            hf = torch.fft.fft(h, dim=-1)
            rf = torch.fft.ifft(hf * hf.conj(), dim=-1).mean(dim=(0, 1, 2))
            ht = torch.fft.fft(h, dim=1)
            rt = torch.fft.ifft(ht * ht.conj(), dim=1).mean(dim=(0, 2, 3))
            acc_f = rf if acc_f is None else acc_f + rf
            acc_t = rt if acc_t is None else acc_t + rt
            power += float(h.abs().square().mean())
            count += 1
        self.r_f.copy_((acc_f / max(count, 1)).to(torch.complex64))
        self.r_t.copy_((acc_t / max(count, 1)).to(torch.complex64))
        self.calibrated.fill_(True)
        return {"calibration_batches": count, "mean_channel_power": power / max(count, 1)}

    @staticmethod
    def _toeplitz(r):
        n = r.numel()
        idx = torch.arange(n, device=r.device)
        lag = (idx[:, None] - idx[None, :]) % n
        return r[lag]

    def _wiener(self, h, r, sigma2, dim):
        R = self._toeplitz(r)
        eye = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
        h = h.movedim(dim, -1)
        shape = h.shape
        flat = h.reshape(-1, shape[-1])
        B = int(sigma2.numel())
        rep = flat.shape[0] // B
        s2 = sigma2.repeat_interleave(rep).to(R.real.dtype)
        mats = R[None] + s2[:, None, None].to(R.dtype) * eye[None]
        sol = torch.linalg.solve(mats, flat.unsqueeze(-1)).squeeze(-1)
        out = (R[None] @ sol.unsqueeze(-1)).squeeze(-1)
        return out.reshape(shape).movedim(-1, dim)

    def estimate(self, x, noise_var=None, **kw):
        if not bool(self.calibrated):
            raise RuntimeError(
                "LMMSEWienerEstimator must be calibrated on training-distribution "
                "samples before use; call calibrate()."
            )
        h = _complex(x)                                             # [B,T,A,F]
        B = h.shape[0]
        if noise_var is None:
            sigma2 = torch.full((B,), 1e-3, device=h.device)
        else:
            sigma2 = noise_var.reshape(-1)[:B].to(h.device).float() * self.noise_scale
        # frequency axis is last, time axis is 1
        h = self._wiener(h, self.r_f, sigma2, dim=3)
        h = self._wiener(h, self.r_t, sigma2, dim=1)
        return _ri(h)


@lru_cache(maxsize=16)
def _dpss_basis(n: int, q: int, half_bandwidth: float):
    """First ``q`` discrete prolate spheroidal sequences of length ``n``."""
    idx = torch.arange(n, dtype=torch.float64)
    d = idx[:, None] - idx[None, :]
    with torch.no_grad():
        s = torch.where(
            d.abs() < 1e-12,
            torch.full_like(d, 2.0 * float(half_bandwidth)),
            torch.sin(2.0 * math.pi * float(half_bandwidth) * d) / (math.pi * d.clamp(min=-1e30)),
        )
    s = 0.5 * (s + s.transpose(0, 1))
    evals, evecs = torch.linalg.eigh(s)
    order = torch.argsort(evals, descending=True)[:int(q)]
    return evecs[:, order].contiguous()


class BEMEstimator(BaselineBase):
    """Ridge-regularized slow-time BEM on pilot occasions.

    Supported bases:
      * ``complex_exp`` : CE-BEM
      * ``dpss``        : Slepian/DPSS BEM with physically correct normalized
                          Doppler bandwidth f_D*T_sym
      * ``poly``        : orthonormal polynomial BEM

    The current DPSS implementation avoids the earlier one-second symbol
    interval assumption (``dur=1``), which clipped practical
    Doppler bandwidths to 0.45 cycles/sample. The symbol duration is recovered
    from the public pilot-age feature, so no hidden channel truth is exposed.
    """

    requires_training = False

    def __init__(self, cfg, basis="complex_exp"):
        super().__init__(cfg)
        self.basis_kind = str(basis).lower()
        names = {
            "complex_exp": "cebem_ls",
            "dpss": "dpsbem_ls",
            "poly": "polybem_ls",
        }
        if self.basis_kind not in names:
            raise ValueError(f"unknown BEM basis {basis!r}")
        self.name = names[self.basis_kind]

        m = cfg["model"]
        npilots = len(self.layout.pilot_visible)
        qkey = {
            "complex_exp": "cebem_q",
            "dpss": "dpsbem_q",
            "poly": "polybem_q",
        }[self.basis_kind]
        configured_q = int(m.get(qkey, m["add_doppler_bins"]))
        self.Q = max(1, min(configured_q, npilots))
        if self.basis_kind == "complex_exp" and self.Q % 2 == 0:
            self.Q = max(1, self.Q - 1)

        rkey = {
            "complex_exp": "cebem_ridge",
            "dpss": "dpsbem_ridge",
            "poly": "polybem_ridge",
        }[self.basis_kind]
        self.ridge = float(m.get(rkey, m.get("bem_ridge", 1e-2)))

        s = cfg["system"]
        self.max_doppler_hz = max(
            [float(v) for v in s["doppler_hz_choices"]]
            + [float(v) for v in s["threegpp_doppler_hz_choices"]]
        )
        self.time_feature_scale = float(time_feature_scale_s(cfg))

    def _symbol_durations(self, x):
        """Recover per-sample OFDM symbol duration from public age features."""
        age = x[:, 3, :, 0, 0].float().abs() * self.time_feature_scale
        s = self.cfg["system"]
        fallback = 1.0 / float(s["subcarrier_spacing_hz"])
        vals = []
        for b in range(age.shape[0]):
            pos = age[b][age[b] > 0]
            vals.append(pos.min() if pos.numel() else age.new_tensor(fallback))
        return torch.stack(vals)

    def _temporal_basis(self, duration_s, device):
        T = self.T
        m = torch.arange(T, device=device, dtype=torch.float32)

        if self.basis_kind == "complex_exp":
            modes = torch.arange(
                -(self.Q // 2), self.Q // 2 + 1,
                device=device, dtype=torch.float32,
            )
            return torch.exp(
                2j * math.pi * m[:, None] * modes[None, :] / float(T)
            ).to(torch.complex64)

        if self.basis_kind == "dpss":
            # Physical normalized half-bandwidth [cycles / OFDM symbol].
            w = min(
                0.499,
                max(1e-6, self.max_doppler_hz * float(duration_s)),
            )
            return _dpss_basis(T, self.Q, float(w)).to(
                device=device, dtype=torch.complex64
            )

        # Orthonormal polynomial basis on [-1,1].
        z = torch.linspace(-1.0, 1.0, T, device=device, dtype=torch.float64)
        V = torch.stack([z.pow(k) for k in range(self.Q)], dim=1)
        Qm, _ = torch.linalg.qr(V, mode="reduced")
        return Qm.to(torch.complex64)

    def _fit_one_group(self, gp, U, device):
        Up = torch.index_select(U, 0, self.pilot_visible_idx.to(device))
        gram = Up.conj().transpose(0, 1) @ Up
        gram = gram + self.ridge * torch.eye(
            self.Q, device=device, dtype=gram.dtype
        )
        rhs = torch.einsum(
            "qp,bpal->bqal", Up.conj().transpose(0, 1), gp
        )
        B = gp.shape[0]
        coeff = torch.linalg.solve(
            gram[None].expand(B, self.Q, self.Q),
            rhs.reshape(B, self.Q, -1),
        ).reshape(rhs.shape)
        return torch.einsum("tq,bqal->btal", U, coeff)

    def estimate(self, x, **kw):
        h = _complex(x)
        device = h.device
        B = h.shape[0]

        g = torch.fft.ifft(h, dim=-1, norm="ortho")
        g = torch.index_select(g, -1, self.delay_window_idx.to(device))
        gp = torch.index_select(g, 1, self.pilot_visible_idx.to(device))

        if self.basis_kind == "dpss":
            durations = self._symbol_durations(x)
            # Supports mixed-CP batches without exposing CP/channel truth.
            keys = torch.round(durations * 1e9).to(torch.long)
            g_hat = gp.new_empty((B, self.T, self.A, self.L))
            for key in torch.unique(keys):
                idx = torch.nonzero(keys == key, as_tuple=False).reshape(-1)
                dur = float(durations[idx[0]])
                U = self._temporal_basis(dur, device)
                g_hat[idx] = self._fit_one_group(
                    torch.index_select(gp, 0, idx), U, device
                )
        else:
            U = self._temporal_basis(0.0, device)
            g_hat = self._fit_one_group(gp, U, device)

        full = g_hat.new_zeros(g_hat.shape[:-1] + (self.Fq,))
        full.index_copy_(-1, self.delay_window_idx.to(device), g_hat)
        return _ri(torch.fft.fft(full, dim=-1, norm="ortho"))


# ---------------------------------------------------------------------------
# deep-learning baselines
# ---------------------------------------------------------------------------


class _PerAntenna2D(BaselineBase):
    """Shared-weight 2-D processing of the (time, frequency) grid per antenna."""

    def _fold(self, x):
        B, C, T, A, Fq = x.shape
        return x.permute(0, 3, 1, 2, 4).reshape(B * A, C, T, Fq), (B, A)

    def _unfold(self, y, meta):
        B, A = meta
        C = y.shape[1]
        return y.reshape(B, A, C, self.T, self.Fq).permute(0, 2, 3, 1, 4).contiguous()


class ChannelNetEstimator(BaselineBase):
    """PyTorch architecture port of the authors' released ChannelNet model.

    The released Keras model treats the real/imaginary channel images as
    one-channel samples, applies SRCNN (64@9x9, 32@1x1, 1@5x5), and then a
    20-convolution DnCNN (first + 18 BN middle + last) that predicts noise.
    We preserve that topology and share the same weights between real and
    imaginary parts. Only the surrounding DOP-Net tensor adapter is project-specific.
    """

    name = "channelnet"

    def __init__(self, cfg):
        super().__init__(cfg)
        mcfg = cfg["model"]
        s1 = int(mcfg.get("channelnet_srcnn_width1", 64))
        s2 = int(mcfg.get("channelnet_srcnn_width2", 32))
        dw = int(mcfg.get("channelnet_dncnn_width", 64))
        middle = int(mcfg.get("channelnet_dncnn_middle_layers", 18))

        self.srcnn = nn.Sequential(
            nn.Conv2d(1, s1, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(s1, s2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(s2, 1, kernel_size=5, padding=2),
        )
        self.dn_first = nn.Conv2d(1, dw, kernel_size=3, padding=1)
        self.dn_middle = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dw, dw, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(dw, eps=1.0e-3),
                nn.ReLU(inplace=True),
            )
            for _ in range(middle)
        ])
        self.dn_last = nn.Conv2d(dw, 1, kernel_size=3, padding=1)
        self._reset_parameters()

    def _reset_parameters(self):
        # Released SRCNN uses He-normal; the DnCNN Keras Conv2D layers use the
        # default Glorot initializer.  Match those choices as closely as PyTorch
        # permits without importing or copying external source code.
        for mod in self.srcnn:
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_in", nonlinearity="relu")
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
        for mod in [self.dn_first, self.dn_last]:
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        for block in self.dn_middle:
            conv, bn = block[0], block[1]
            nn.init.xavier_uniform_(conv.weight)
            nn.init.ones_(bn.weight)
            nn.init.zeros_(bn.bias)

    def estimate(self, x, **kw):
        x = x.float()
        B, _, T, A, Fq = x.shape
        # Official architecture is single-channel.  Fold real/imaginary and
        # antennas into the batch so both components use exactly shared weights.
        z = x[:, :2].permute(0, 3, 1, 2, 4).reshape(B * A * 2, 1, T, Fq)
        sr = self.srcnn(z)
        noise = F.relu(self.dn_first(sr), inplace=False)
        for block in self.dn_middle:
            noise = block(noise)
        noise = self.dn_last(noise)
        out = sr - noise
        return out.reshape(B, A, 2, T, Fq).permute(0, 2, 3, 1, 4).contiguous()


class _ReEsResidualBlock(nn.Module):
    """Two-convolution residual unit used by the published ReEsNet topology."""

    def __init__(self, width):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x), inplace=False)
        y = self.conv2(y)
        return x + y


def _convtranspose_axis(in_size, out_size, stride, kernel):
    """Find symmetric padding/output_padding for an exact ConvTranspose axis."""
    for output_padding in range(stride):
        num = (in_size - 1) * stride + kernel + output_padding - out_size
        if num >= 0 and num % 2 == 0:
            return num // 2, output_padding, True
    # Fallback is only for unusual user configs; forward() resizes to exact size.
    pad = max(0, ((in_size - 1) * stride + kernel - out_size) // 2)
    return pad, 0, False


class ReEsNetEstimator(BaselineBase):
    """ReEsNet residual CNN with learned transposed-convolution upsampling.

    The external constructor/forward/output interface is unchanged. Internally,
    the low-resolution tensor is now formed only from true pilot-position LS
    samples and arranged as [pilot-frequency, pilot-symbol], matching the
    published/reproduced ReEsNet topology.
    """

    name = "reesnet"

    def __init__(self, cfg):
        super().__init__(cfg)
        mcfg = cfg["model"]
        width = int(mcfg.get("reesnet_width", 16))
        n_blocks = int(mcfg.get("reesnet_blocks", 4))
        self.freq_stride = int(mcfg.get("reesnet_coarse_freq_stride", 2))
        kernel = int(mcfg.get("reesnet_deconv_kernel", 10))
        if self.freq_stride < 1:
            raise ValueError("reesnet_coarse_freq_stride must be >= 1")

        self.n_pilot_freq = len(range(0, self.Fq, self.freq_stride))
        self.n_pilot_time = int(self.pilot_visible_idx.numel())
        self.stem = nn.Conv2d(2, width, 3, padding=1)
        self.blocks = nn.ModuleList([_ReEsResidualBlock(width) for _ in range(n_blocks)])
        self.post = nn.Conv2d(width, width, 3, padding=1)

        time_stride = max(1, int(round(self.T / max(1, self.n_pilot_time))))
        pf, opf, exact_f = _convtranspose_axis(
            self.n_pilot_freq, self.Fq, self.freq_stride, kernel)
        pt, opt, exact_t = _convtranspose_axis(
            self.n_pilot_time, self.T, time_stride, kernel)
        self._exact_deconv_geometry = bool(exact_f and exact_t)
        self.deconv = nn.ConvTranspose2d(
            width, width, kernel_size=(kernel, kernel),
            stride=(self.freq_stride, time_stride),
            padding=(pf, pt), output_padding=(opf, opt),
        )
        self.head = nn.Conv2d(width, 2, 3, padding=1)

    def estimate(self, x, **kw):
        x = x.float()
        B, _, _, A, _ = x.shape
        coarse = _extract_pilot_lattice(x, self.pilot_visible_idx, self.n_pilot_freq)
        coarse = coarse.permute(0, 1, 3, 2).contiguous()  # [BA,2,nf,nt]

        stem = self.stem(coarse)
        y = stem
        for block in self.blocks:
            y = block(y)
        y = self.post(y) + stem
        y = self.deconv(y)
        y = _center_crop_or_pad_2d(y, (self.Fq, self.T))
        y = self.head(y).permute(0, 1, 3, 2).contiguous()  # [BA,2,T,F]
        return y.reshape(B, A, 2, self.T, self.Fq).permute(0, 2, 3, 1, 4).contiguous()


class _HA02EncoderBlock(nn.Module):
    """Author-code-guided HA02 transformer encoder block (post-norm)."""

    def __init__(self, feature_size, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(feature_size, heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_size, eps=1.0e-5)
        self.ff1 = nn.Linear(feature_size, feature_size)
        self.ff2 = nn.Linear(feature_size, feature_size)
        self.norm2 = nn.LayerNorm(feature_size, eps=1.0e-5)

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + a)
        y = self.ff2(F.gelu(self.ff1(x)))
        return self.norm2(x + y)


class _SamePadConv2d(nn.Module):
    """TensorFlow/MATLAB-style SAME convolution, including even kernels."""

    def __init__(self, c_in, c_out, kernel_size, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kh, self.kw = kernel_size
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        ph = self.kh - 1
        pw = self.kw - 1
        top = ph // 2
        bottom = ph - top
        left = pw // 2
        right = pw - left
        return self.conv(F.pad(x, (left, right, top, bottom)))


class _HA02FeatureNorm(nn.Module):
    """Normalization along the HA02 feature dimension, as in the released code."""

    def __init__(self, feature_size, eps=1.0e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(feature_size)))
        self.bias = nn.Parameter(torch.zeros(int(feature_size)))
        self.eps = float(eps)

    def forward(self, x):
        mean = x.mean(dim=2, keepdim=True)
        var = (x - mean).square().mean(dim=2, keepdim=True)
        y = (x - mean) * torch.rsqrt(var + self.eps)
        return y * self.weight.view(1, 1, -1, 1) + self.bias.view(1, 1, -1, 1)


class _HA02DecoderBlock(nn.Module):
    def __init__(self, width, feature_size, kernel=2):
        super().__init__()
        self.conv1 = _SamePadConv2d(width, width, kernel)
        self.conv2 = _SamePadConv2d(width, width, kernel)
        self.norm = _HA02FeatureNorm(feature_size)

    def forward(self, x):
        y = F.relu(self.conv1(x), inplace=False)
        y = self.conv2(y)
        return self.norm(x + y)


class HA02Estimator(BaselineBase):
    """HA02 port preserving the authors' feature layout and learned FC upsampler.

    The public interface remains exactly the same. For each antenna, true pilot
    LS samples are concatenated into one feature vector; real/imaginary parts
    form the length-2 Transformer sequence. The decoder uses the released 2x2
    residual CNN followed by the large learned feature->full-slot FC upsampler.
    """

    name = "ha02"

    def __init__(self, cfg):
        super().__init__(cfg)
        mcfg = cfg["model"]
        self.pilot_freq_stride = int(mcfg.get("ha02_pilot_freq_stride", 2))
        if self.pilot_freq_stride < 1:
            raise ValueError("ha02_pilot_freq_stride must be >= 1")
        self.n_pilot_time = int(self.pilot_visible_idx.numel())
        self.n_pilot_freq = len(range(0, self.Fq, self.pilot_freq_stride))
        self.feature_size = self.n_pilot_time * self.n_pilot_freq
        self.output_feature_size = self.T * self.Fq

        heads = int(mcfg.get("ha02_heads", self.n_pilot_time))
        enc_layers = int(mcfg.get("ha02_encoder_layers", 1))
        dec_layers = int(mcfg.get("ha02_decoder_layers", 1))
        dec_width = int(mcfg.get("ha02_decoder_filters", 2))
        if self.feature_size % heads != 0:
            raise ValueError(
                f"HA02 pilot feature size {self.feature_size} must be divisible by heads={heads}")

        self.encoder = nn.ModuleList([
            _HA02EncoderBlock(self.feature_size, heads) for _ in range(enc_layers)
        ])
        self.decoder_in = _SamePadConv2d(1, dec_width, 2)
        self.decoder_blocks = nn.ModuleList([
            _HA02DecoderBlock(dec_width, self.feature_size, 2)
            for _ in range(dec_layers)
        ])
        self.decoder_fc = nn.Linear(self.feature_size, self.output_feature_size)
        self.decoder_out = _SamePadConv2d(dec_width, 1, 2)

    def estimate(self, x, **kw):
        x = x.float()
        B, _, _, A, _ = x.shape
        pilot = _extract_pilot_lattice(x, self.pilot_visible_idx, self.n_pilot_freq)
        z = pilot.permute(0, 2, 3, 1).reshape(B * A, self.feature_size, 2)

        z = z.transpose(1, 2).contiguous()  # [BA,2,feature]
        for block in self.encoder:
            z = block(z)
        z = z.transpose(1, 2).unsqueeze(1).contiguous()  # [BA,1,feature,2]

        z = self.decoder_in(z)
        for block in self.decoder_blocks:
            z = block(z)
        z = self.decoder_fc(z.transpose(2, 3)).transpose(2, 3)  # feature -> T*F
        z = self.decoder_out(z).squeeze(1)  # [BA,T*F,2]

        z = z.reshape(B, A, self.T, self.Fq, 2)
        return z.permute(0, 4, 2, 1, 3).contiguous()


class _DualCNNBlock(nn.Module):
    """Low-complexity three-convolution residual CNN from DualCNN Table I."""

    def __init__(self, channels=2, filters=8, negative_slope=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, filters, 3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.conv3 = nn.Conv2d(filters, channels, 3, padding=1)
        self.negative_slope = float(negative_slope)

    def forward(self, x):
        y = F.leaky_relu(self.conv1(x), negative_slope=self.negative_slope)
        y = F.leaky_relu(self.conv2(y), negative_slope=self.negative_slope)
        y = self.conv3(y)
        return x + y


class DualCNNEstimator(BaselineBase):
    """Paper-faithful low-complexity DualCNN: SF-CNN -> DFT -> AD-CNN -> IDFT.

    The original method applies a residual CNN in the spatial-frequency domain,
    transforms that estimate into the angle-delay domain, applies a second
    residual CNN there, and transforms back.  This implementation follows that
    sequential topology instead of the previous parallel-branch adaptation.
    Each OFDM symbol is processed independently, exactly as a 2-D SF/AD model;
    time is not silently added as a third convolution dimension.
    """

    name = "dualcnn"

    def __init__(self, cfg):
        super().__init__(cfg)
        mcfg = cfg["model"]
        filters = int(mcfg.get("dualcnn_filters", 8))
        slope = float(mcfg.get("dualcnn_negative_slope", 0.2))
        self.sf_cnn = _DualCNNBlock(2, filters, slope)
        self.ad_cnn = _DualCNNBlock(2, filters, slope)

    @staticmethod
    def _sf_to_ad(z):
        z = torch.fft.fft(z, dim=-2, norm="ortho")
        return torch.fft.ifft(z, dim=-1, norm="ortho")

    @staticmethod
    def _ad_to_sf(z):
        z = torch.fft.fft(z, dim=-1, norm="ortho")
        return torch.fft.ifft(z, dim=-2, norm="ortho")

    def estimate(self, x, **kw):
        x = x.float()
        B, _, T, A, Fq = x.shape
        sf = x[:, :2].permute(0, 2, 1, 3, 4).reshape(B * T, 2, A, Fq)
        sf = self.sf_cnn(sf)

        sf_c = torch.complex(sf[:, 0].float(), sf[:, 1].float())
        ad_c = self._sf_to_ad(sf_c)
        ad = torch.stack([ad_c.real, ad_c.imag], dim=1).float()
        ad = self.ad_cnn(ad)

        ad_c = torch.complex(ad[:, 0].float(), ad[:, 1].float())
        out_c = self._ad_to_sf(ad_c)
        out = torch.stack([out_c.real, out_c.imag], dim=1).float()
        return out.reshape(B, T, 2, A, Fq).permute(0, 2, 1, 3, 4).contiguous()

# ---------------------------------------------------------------------------
# 2025 open-source DL baselines: source-guided adapters for the DOP-Net interface
# ---------------------------------------------------------------------------


class _FortiTranConvEnhancer(nn.Module):
    """FortiTran-style ConvEnhancer: 1 -> 8 -> 32 -> 8 -> 1."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class _FortiTranPatchTransformer(nn.Module):
    """FortiTran Transformer encoder topology."""

    def __init__(
        self,
        patch_dim,
        seq_len,
        model_dim=32,
        heads=4,
        layers=6,
        dropout=0.1,
    ):
        super().__init__()

        self.in_proj = nn.Linear(int(patch_dim), int(model_dim))

        self.pos = nn.Parameter(
            torch.zeros(1, int(seq_len), int(model_dim))
        )

        layer = nn.TransformerEncoderLayer(
            d_model=int(model_dim),
            nhead=int(heads),
            dim_feedforward=2 * int(model_dim),
            activation="gelu",
            dropout=float(dropout),
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=int(layers),
        )

        self.out_proj = nn.Linear(
            int(model_dim),
            int(patch_dim),
        )

        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        n = x.shape[1]

        x = self.in_proj(x)
        x = x + self.pos[:, :n]

        x = self.transformer(x)
        x = self.out_proj(x)

        return x


class FortiTranEstimator(BaselineBase):
    """ICC-2025 FortiTran adapted to the DOP-Net channel-estimation input.

    Preserved from the authors' released implementation:
      - learned pilot -> full-grid Linear upsampler
      - ConvEnhancer
      - non-overlapping 2-D patch processing
      - learnable positional embedding
      - 6-layer Transformer
      - d_model = 32
      - 4 attention heads
      - FFN width = 2*d_model
      - residual reconstruction
      - final ConvEnhancer
      - shared processing of real/imaginary components

    DOP-Net adaptation:
      The original model assumes a fixed 40 x 2 pilot lattice.
      The DOP-Net dataset has varying pilot spacing and seven visible
      pilot occasions. Actual observed pilot REs are mask-normalized
      pooled to the configured size (64 x 7 in the submission experiment).

      No target CSI, true Doppler, SNR, delay spread, or hidden
      simulator state is used.
    """

    name = "fortitran"

    def __init__(self, cfg):
        super().__init__(cfg)

        mcfg = cfg["model"]

        # Original FortiTran dimensions.
        self.pilot_pool_f = int(
            mcfg.get("fortitran_pilot_pool_f", 40)
        )
        self.pilot_pool_t = int(
            mcfg.get("fortitran_pilot_pool_t", 2)
        )

        model_dim = int(
            mcfg.get("fortitran_model_dim", 32)
        )
        heads = int(
            mcfg.get("fortitran_heads", 4)
        )
        layers = int(
            mcfg.get("fortitran_layers", 6)
        )
        dropout = float(
            mcfg.get("fortitran_dropout", 0.1)
        )

        # Original patch=(3,2) cannot exactly tile 128 x 25.
        # (4,5) is used for the DOP-Net grid.
        patch_f = int(
            mcfg.get("fortitran_patch_f", 4)
        )
        patch_t = int(
            mcfg.get("fortitran_patch_t", 5)
        )

        if self.Fq % patch_f != 0 or self.T % patch_t != 0:
            raise ValueError(
                f"FortiTran patch ({patch_f},{patch_t}) "
                f"must tile (F,T)=({self.Fq},{self.T})."
            )

        if model_dim % heads != 0:
            raise ValueError(
                "fortitran_model_dim must be divisible "
                "by fortitran_heads"
            )

        patch_dim = patch_f * patch_t

        seq_len = (
            (self.Fq // patch_f)
            * (self.T // patch_t)
        )

        pilot_features = (
            self.pilot_pool_f
            * self.pilot_pool_t
        )

        # Learned pilot -> complete channel grid.
        self.pilot_upsampler = nn.Linear(
            pilot_features,
            self.Fq * self.T,
        )

        self.initial_enhancer = _FortiTranConvEnhancer()

        self.patch_unfold = nn.Unfold(
            kernel_size=(patch_f, patch_t),
            stride=(patch_f, patch_t),
        )

        self.patch_fold = nn.Fold(
            output_size=(self.Fq, self.T),
            kernel_size=(patch_f, patch_t),
            stride=(patch_f, patch_t),
        )

        self.transformer = _FortiTranPatchTransformer(
            patch_dim=patch_dim,
            seq_len=seq_len,
            model_dim=model_dim,
            heads=heads,
            layers=layers,
            dropout=dropout,
        )

        self.final_refiner = _FortiTranConvEnhancer()

    def _fixed_pilot_vector(self, x):
        """Convert actual pilot observations to the configured fixed lattice."""

        B, _, _, A, _ = x.shape

        # [B,2,T,A,F]
        # -> [B,A,2,F,T]
        # -> [B*A,2,F,T]
        ri = (
            x[:, :2]
            .permute(0, 3, 1, 4, 2)
            .reshape(
                B * A,
                2,
                self.Fq,
                self.T,
            )
        )

        # Public pilot mask.
        mask = (
            x[:, 2:3]
            .permute(0, 3, 1, 4, 2)
            .reshape(
                B * A,
                1,
                self.Fq,
                self.T,
            )
        )

        idx = self.pilot_visible_idx.to(x.device)

        ri = torch.index_select(
            ri,
            -1,
            idx,
        )

        mask = torch.index_select(
            mask,
            -1,
            idx,
        )

        n = B * A

        # Real and imaginary components use the same network,
        # matching the original FortiTran processing strategy.
        val = (
            (ri * mask)
            .reshape(
                n * 2,
                1,
                self.Fq,
                -1,
            )
        )

        den = (
            mask.expand(-1, 2, -1, -1)
            .reshape(
                n * 2,
                1,
                self.Fq,
                -1,
            )
        )

        out_hw = (
            self.pilot_pool_f,
            self.pilot_pool_t,
        )

        num = F.adaptive_avg_pool2d(
            val,
            out_hw,
        )

        den = F.adaptive_avg_pool2d(
            den,
            out_hw,
        )

        # Normalize only over actual pilot REs.
        pooled = torch.where(
            den > 1.0e-6,
            num / den.clamp_min(1.0e-6),
            torch.zeros_like(num),
        )

        return pooled.flatten(1)

    def estimate(self, x, **kw):
        x = x.float()

        B, _, T, A, Fq = x.shape

        if T != self.T or Fq != self.Fq:
            raise ValueError(
                f"Expected T,F=({self.T},{self.Fq}), "
                f"got ({T},{Fq})"
            )

        # [B*A*2, pilot_pool_f * pilot_pool_t]
        p = self._fixed_pilot_vector(x)

        # Learned pilot -> full channel surface.
        y = self.pilot_upsampler(p)

        y = y.view(
            B * A * 2,
            1,
            self.Fq,
            self.T,
        )

        # Initial convolutional refinement.
        conv = (
            self.initial_enhancer(y)
            .squeeze(1)
        )

        # Patch -> Transformer -> inverse patch.
        patches = (
            self.patch_unfold(
                conv.unsqueeze(1)
            )
            .transpose(1, 2)
        )

        patches = self.transformer(
            patches
        )

        rec = (
            self.patch_fold(
                patches.transpose(1, 2)
            )
            .squeeze(1)
        )

        # Residual + final convolutional refinement.
        out = (
            self.final_refiner(
                (conv + rec).unsqueeze(1)
            )
            .squeeze(1)
        )

        # [B,A,2,F,T]
        # -> [B,2,T,A,F]
        out = (
            out.view(
                B,
                A,
                2,
                self.Fq,
                self.T,
            )
            .permute(
                0,
                2,
                4,
                1,
                3,
            )
            .contiguous()
        )

        return out


class WiMAECEEstimator(BaselineBase):
    """2025 WiMAE adapted to supervised DOP-Net channel estimation.

    Preserved from the authors' released WiMAE implementation:
      - complex matrix patching
      - real/imaginary patch sequences
      - Linear patch embedding
      - learnable position embedding
      - masked Transformer encoder
      - learnable decoder mask token
      - Transformer reconstruction decoder
      - Linear patch reconstruction

    DOP-Net adaptation:
      - d_model remains 64
      - encoder/decoder depths become 8/2 to match model scale
      - patch size becomes (32,1)
      - physical OFDM pilot-time positions replace random MAE masking
      - only public LS/interpolation information is used
      - no target CSI or simulator metadata is given to the network
    """

    name = "wimae_ce"

    def __init__(self, cfg):
        super().__init__(cfg)

        mcfg = cfg["model"]

        self.patch_f = int(
            mcfg.get("wimae_patch_f", 32)
        )

        self.patch_t = int(
            mcfg.get("wimae_patch_t", 1)
        )

        d_model = int(
            mcfg.get("wimae_dim", 64)
        )

        enc_layers = int(
            mcfg.get("wimae_encoder_layers", 8)
        )

        dec_layers = int(
            mcfg.get("wimae_decoder_layers", 2)
        )

        enc_heads = int(
            mcfg.get("wimae_encoder_heads", 16)
        )

        dec_heads = int(
            mcfg.get("wimae_decoder_heads", 8)
        )

        dropout = float(
            mcfg.get("wimae_dropout", 0.1)
        )

        if (
            self.Fq % self.patch_f != 0
            or self.T % self.patch_t != 0
        ):
            raise ValueError(
                f"WiMAE patch "
                f"({self.patch_f},{self.patch_t}) "
                f"must tile "
                f"(F,T)=({self.Fq},{self.T})."
            )

        if (
            d_model % enc_heads != 0
            or d_model % dec_heads != 0
        ):
            raise ValueError(
                "wimae_dim must be divisible "
                "by encoder and decoder head counts"
            )

        self.n_f_blocks = (
            self.Fq // self.patch_f
        )

        self.n_t_blocks = (
            self.T // self.patch_t
        )

        self.patch_dim = (
            self.patch_f
            * self.patch_t
        )

        self.n_complex_patches = (
            self.n_f_blocks
            * self.n_t_blocks
        )

        # Real patches followed by imaginary patches,
        # matching the released WiMAE patch representation.
        self.seq_len = (
            2 * self.n_complex_patches
        )

        # Pilot-time masking below assumes one OFDM symbol per patch.
        if self.patch_t != 1:
            raise ValueError(
                "DOP-Net WiMAE pilot-time adapter requires "
                "wimae_patch_t=1"
            )

        # Build indices of physically visible pilot-time tokens.
        visible = []

        pilot_t = [
            int(t)
            for t
            in self.pilot_visible_idx.tolist()
        ]

        for comp in range(2):
            comp_base = (
                comp
                * self.n_complex_patches
            )

            for fb in range(
                self.n_f_blocks
            ):
                fb_base = (
                    comp_base
                    + fb * self.T
                )

                visible.extend(
                    fb_base + t
                    for t in pilot_t
                )

        self.register_buffer(
            "wimae_visible_idx",
            torch.as_tensor(
                visible,
                dtype=torch.long,
            ),
            persistent=False,
        )

        # -----------------------------
        # Encoder
        # -----------------------------

        self.encoder_in = nn.Linear(
            self.patch_dim,
            d_model,
        )

        self.encoder_pos = nn.Parameter(
            torch.zeros(
                1,
                self.seq_len,
                d_model,
            )
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=enc_heads,
            dim_feedforward=2 * d_model,
            activation="gelu",
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=enc_layers,
        )

        # -----------------------------
        # Decoder
        # -----------------------------

        self.mask_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                d_model,
            )
        )

        self.decoder_pos = nn.Parameter(
            torch.zeros(
                1,
                self.seq_len,
                d_model,
            )
        )

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=dec_heads,
            dim_feedforward=2 * d_model,
            activation="gelu",
            dropout=dropout,
            batch_first=True,
        )

        self.decoder = nn.TransformerEncoder(
            dec_layer,
            num_layers=dec_layers,
        )

        self.decoder_out = nn.Linear(
            d_model,
            self.patch_dim,
        )

        nn.init.trunc_normal_(
            self.encoder_pos,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.decoder_pos,
            std=0.02,
        )

        nn.init.normal_(
            self.mask_token,
            std=0.02,
        )

    def _patch_complex(self, z):
        """Complex matrix -> real patches + imaginary patches."""

        N = z.shape[0]

        real = (
            z.real
            .reshape(
                N,
                self.n_f_blocks,
                self.patch_f,
                self.n_t_blocks,
                self.patch_t,
            )
            .permute(
                0,
                1,
                3,
                2,
                4,
            )
            .reshape(
                N,
                self.n_complex_patches,
                self.patch_dim,
            )
        )

        imag = (
            z.imag
            .reshape(
                N,
                self.n_f_blocks,
                self.patch_f,
                self.n_t_blocks,
                self.patch_t,
            )
            .permute(
                0,
                1,
                3,
                2,
                4,
            )
            .reshape(
                N,
                self.n_complex_patches,
                self.patch_dim,
            )
        )

        return torch.cat(
            [real, imag],
            dim=1,
        )

    def _unpatch_complex(self, patches):
        """Patch sequence -> complex F x T channel matrix."""

        N = patches.shape[0]

        p = self.n_complex_patches

        real = patches[:, :p]
        imag = patches[:, p:]

        real = (
            real
            .reshape(
                N,
                self.n_f_blocks,
                self.n_t_blocks,
                self.patch_f,
                self.patch_t,
            )
            .permute(
                0,
                1,
                3,
                2,
                4,
            )
            .reshape(
                N,
                self.Fq,
                self.T,
            )
        )

        imag = (
            imag
            .reshape(
                N,
                self.n_f_blocks,
                self.n_t_blocks,
                self.patch_f,
                self.patch_t,
            )
            .permute(
                0,
                1,
                3,
                2,
                4,
            )
            .reshape(
                N,
                self.Fq,
                self.T,
            )
        )

        return torch.complex(
            real.float(),
            imag.float(),
        )

    def estimate(self, x, **kw):
        x = x.float()

        B, _, T, A, Fq = x.shape

        if T != self.T or Fq != self.Fq:
            raise ValueError(
                f"Expected T,F=({self.T},{self.Fq}), "
                f"got ({T},{Fq})"
            )

        # Public LS/interpolated estimate only.
        #
        # [B,T,A,F]
        # -> [B,A,F,T]
        # -> [B*A,F,T]
        z = torch.complex(
            x[:, 0].float(),
            x[:, 1].float(),
        )

        z = (
            z.permute(
                0,
                2,
                3,
                1,
            )
            .reshape(
                B * A,
                Fq,
                T,
            )
        )

        # Complex channel matrix -> WiMAE patch sequence.
        patches = self._patch_complex(
            z
        )

        tokens = (
            self.encoder_in(patches)
            + self.encoder_pos[
                :,
                :self.seq_len,
            ]
        )

        idx = self.wimae_visible_idx.to(
            tokens.device
        )

        # Only physical pilot-time patches enter the encoder.
        visible = torch.index_select(
            tokens,
            1,
            idx,
        )

        encoded = self.encoder(
            visible
        )

        # Fill data-symbol locations with learnable MAE mask tokens.
        full = (
            self.mask_token
            .expand(
                B * A,
                self.seq_len,
                -1,
            )
            .clone()
        )

        full.index_copy_(
            1,
            idx,
            encoded,
        )

        decoded = self.decoder(
            full
            + self.decoder_pos[
                :,
                :self.seq_len,
            ]
        )

        rec_patches = self.decoder_out(
            decoded
        )

        rec = self._unpatch_complex(
            rec_patches
        )

        ri = torch.stack(
            [
                rec.real,
                rec.imag,
            ],
            dim=1,
        ).float()

        # [B*A,2,F,T]
        # -> [B,2,T,A,F]
        ri = (
            ri.view(
                B,
                A,
                2,
                Fq,
                T,
            )
            .permute(
                0,
                2,
                4,
                1,
                3,
            )
            .contiguous()
        )

        return ri
# ---------------------------------------------------------------------------
# Lee et al., IEEE WCL 2025:
# "Alternative Meta-Learning With 3D Dual-CNN for MIMO Channel Estimation"
#
# Fair protocol-aligned reproduction of Sec. III-A: 3D Dual CNN.
# ---------------------------------------------------------------------------


class _Lee3DFSRCNN(nn.Module):
    """3D FSRCNN block used by Lee et al. (IEEE WCL 2025).

    Paper / cited-architecture topology:

        feature extraction : Conv3d 5x5x5
        shrinking          : Conv3d 1x1x1
        mapping            : m x Conv3d 3x3x3
        expanding          : Conv3d 1x1x1
        reconstruction     : ConvTranspose3d 9x9x9

    ReLU is used after the convolutional stages.

    The cited 3D-FSRCNN study selects:
        d = 56
        s = 12
        m = 4

    This project's channel is complex, so real / imaginary parts are represented
    by two feature channels.

    Since Lee et al. feed LS + linear-interpolated full-grid CSI into the network,
    the input and output spatial sizes are identical here. Therefore the final
    9x9x9 transpose convolution uses stride=1 and padding=4.
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        d=56,
        s=12,
        m=4,
    ):
        super().__init__()

        d = int(d)
        s = int(s)
        m = int(m)

        if d < 1 or s < 1 or m < 1:
            raise ValueError(
                "Lee3D-FSRCNN requires positive d, s and m."
            )

        # Feature extraction: 5 x 5 x 5
        self.feature = nn.Conv3d(
            int(in_channels),
            d,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )

        # Shrinking: 1 x 1 x 1
        self.shrink = nn.Conv3d(
            d,
            s,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Nonlinear mapping: m x (3 x 3 x 3)
        self.mapping = nn.ModuleList([
            nn.Conv3d(
                s,
                s,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
            for _ in range(m)
        ])

        # Expanding: 1 x 1 x 1
        self.expand = nn.Conv3d(
            s,
            d,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Reconstruction / deconvolution: 9 x 9 x 9.
        #
        # Lee et al. perform channel-to-channel estimation:
        # input H_LS+Interp and final H_hat have the same S/F/T dimensions.
        #
        # kernel=9, stride=1, padding=4 therefore preserves size.
        self.deconv = nn.ConvTranspose3d(
            d,
            int(out_channels),
            kernel_size=9,
            stride=1,
            padding=4,
            output_padding=0,
            bias=True,
        )

    def forward(self, x):
        y = F.relu(
            self.feature(x),
            inplace=False,
        )

        y = F.relu(
            self.shrink(y),
            inplace=False,
        )

        for layer in self.mapping:
            y = F.relu(
                layer(y),
                inplace=False,
            )

        y = F.relu(
            self.expand(y),
            inplace=False,
        )

        # No ReLU on final channel reconstruction:
        # real / imaginary CSI coefficients must be signed.
        return self.deconv(y)


class Lee3DDualCNNEstimator(BaselineBase):
    """Protocol-aligned reproduction of Lee et al. 2025 3D Dual CNN.

    Paper Sec. III-A / Eq. (5), (7), (9):

        H_LS+Interp
             |
             v
        3D FSRCNN in SFT
             |
             v
           D(.)
         3D FFT
             |
             v
        3D FSRCNN in ADD
             |
             v
          D^{-1}(.)
         3D IFFT
             |
             v
           H_hat

    Adaptation to DDPAGNet project
    ------------------------------
    Public DOP-Net input:

        x : [B,5,T,A,F]

    Only:

        x[:,0] = Re{H_LS+Interp}
        x[:,1] = Im{H_LS+Interp}

    are used.

    The following are deliberately NOT exposed:
        - pilot mask
        - pilot age / horizon
        - waveform-derived ADD observation
        - target CSI
        - target ADD
        - true SNR
        - true Doppler
        - delay spread
        - hidden simulator state

    Thus this remains a fair architecture baseline under the shared protocol.
    """

    name = "lee3ddual"
    requires_training = True

    def __init__(self, cfg):
        super().__init__(cfg)

        mcfg = cfg["model"]

        # Values selected in the cited 3D-FSRCNN architecture.
        d = int(
            mcfg.get(
                "lee3d_fsrcnn_d",
                56,
            )
        )

        s = int(
            mcfg.get(
                "lee3d_fsrcnn_s",
                12,
            )
        )

        mapping_layers = int(
            mcfg.get(
                "lee3d_fsrcnn_mapping_layers",
                4,
            )
        )

        self.sft_net = _Lee3DFSRCNN(
            in_channels=2,
            out_channels=2,
            d=d,
            s=s,
            m=mapping_layers,
        )

        self.add_net = _Lee3DFSRCNN(
            in_channels=2,
            out_channels=2,
            d=d,
            s=s,
            m=mapping_layers,
        )

    # ------------------------------------------------------------------
    # Complex helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_complex(x_ri):
        """[B,2,T,A,F] -> [B,T,A,F] complex."""
        return torch.complex(
            x_ri[:, 0].float(),
            x_ri[:, 1].float(),
        )

    @staticmethod
    def _to_ri(z):
        """[B,T,A,F] complex -> [B,2,T,A,F]."""
        return torch.stack(
            [
                z.real,
                z.imag,
            ],
            dim=1,
        ).float()

    # ------------------------------------------------------------------
    # D(.) and D^{-1}(.) in Lee et al. Eq. (7)/(9)
    # ------------------------------------------------------------------

    @staticmethod
    def _d3(z):
        """Ordinary 3-D forward DFT.

        z shape:
            [B,T,A,F]

        Paper notation is S x F x T.
        Since the transform is separable and is applied to all three axes,
        tensor axis ordering does not alter the mathematical operation.

        norm="backward":
            forward FFT is unnormalised,
            matching the ordinary DFT definition in Eq. (7).
        """
        return torch.fft.fftn(
            z,
            dim=(-3, -2, -1),
            norm="backward",
        )

    @staticmethod
    def _id3(z):
        """Inverse of Lee et al.'s full 3-D DFT."""
        return torch.fft.ifftn(
            z,
            dim=(-3, -2, -1),
            norm="backward",
        )

    # ------------------------------------------------------------------

    def estimate(self, x, **kw):
        x = x.float()

        if x.ndim != 5:
            raise ValueError(
                "Lee3DDualCNN expects "
                f"[B,C,T,A,F], got {tuple(x.shape)}"
            )

        if x.shape[1] < 2:
            raise ValueError(
                "Lee3DDualCNN needs the real/imaginary "
                "LS+interpolation channels."
            )

        # --------------------------------------------------------------
        # Paper Eq. (5)
        #
        # H_SFT = f_SFT(H_LS+Interp ; theta_SFT)
        # --------------------------------------------------------------

        h_ls_interp = x[:, :2].contiguous()

        h_sft_ri = self.sft_net(
            h_ls_interp
        )

        # --------------------------------------------------------------
        # Paper Eq. (7)
        #
        # transform SFT estimate to ADD coordinates.
        # --------------------------------------------------------------

        h_sft_complex = self._to_complex(
            h_sft_ri
        )

        h_add_input_complex = self._d3(
            h_sft_complex
        )

        h_add_input_ri = self._to_ri(
            h_add_input_complex
        )

        # --------------------------------------------------------------
        # Paper Eq. (9)
        #
        # H_3Ddual =
        #     D^{-1}(
        #         f_ADD(
        #             D(H_SFT);
        #             theta_ADD
        #         )
        #     )
        # --------------------------------------------------------------

        h_add_hat_ri = self.add_net(
            h_add_input_ri
        )

        h_add_hat_complex = self._to_complex(
            h_add_hat_ri
        )

        h_out_complex = self._id3(
            h_add_hat_complex
        )

        return self._to_ri(
            h_out_complex
        )

# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


BASELINE_REGISTRY = {
    "ls_interp": lambda cfg: LSInterpEstimator(cfg),
    "dft_ls": lambda cfg: DFTLSEstimator(cfg),
    "lmmse_wiener": lambda cfg: LMMSEWienerEstimator(cfg),
    "cebem_ls": lambda cfg: BEMEstimator(cfg, basis="complex_exp"),
    "dpsbem_ls": lambda cfg: BEMEstimator(cfg, basis="dpss"),
    "polybem_ls": lambda cfg: BEMEstimator(cfg, basis="poly"),

    "channelnet": lambda cfg: ChannelNetEstimator(cfg),
    "reesnet": lambda cfg: ReEsNetEstimator(cfg),
    "ha02": lambda cfg: HA02Estimator(cfg),
    "dualcnn": lambda cfg: DualCNNEstimator(cfg),

    "fortitran": lambda cfg: FortiTranEstimator(cfg),
    "wimae_ce": lambda cfg: WiMAECEEstimator(cfg),
    "lee3ddual": lambda cfg: Lee3DDualCNNEstimator(cfg),
    
}

TRAINABLE_BASELINES = (
    "channelnet",
    "reesnet",
    "ha02",
    "dualcnn",
    "fortitran",
    "wimae_ce",
    "lee3ddual",
)

CLASSICAL_BASELINES = ("ls_interp", "dft_ls", "lmmse_wiener", "cebem_ls", "dpsbem_ls", "polybem_ls")


def available_baselines():
    return tuple(BASELINE_REGISTRY.keys())


def build_baseline(name, cfg):
    key = str(name).lower()
    if key not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline {name!r}; choose from {available_baselines()}"
        )
    return BASELINE_REGISTRY[key](cfg)
