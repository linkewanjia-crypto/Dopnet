"""Corrected pilot-domain statistical and joint BEM references.

This module never loads or trains a neural model.  Calibration channel targets
are used only to learn ensemble statistics; the estimate APIs accept received
pilots, public transmitted pilots, noise variance and public geometry only.
CPU and CUDA execute the same Torch complex arithmetic and solver equations.

The empirical LMMSE is a *separable covariance approximation*, not an oracle
LMMSE.  It uses the actual per-frame comb positions, before interpolation.
The finite cyclic BEM retains physical-Nfft frequency mixing, including the
occupied-band edges; it is not the simulator's full linear-convolution model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
import hashlib
import json
from functools import reduce
from operator import mul
from typing import Iterable

import numpy as np
import torch


BEM_METHODS = ("oce_bem_joint_cholesky", "dpss_bem_joint_cholesky")
STATISTICAL_METHOD = "pilot_covariance_lmmse"
METHODS = (STATISTICAL_METHOD,) + BEM_METHODS


def _product(values):
    return reduce(mul, values, 1)


class SolverConvergenceError(RuntimeError):
    """No estimator score may be accepted after this exception."""


def complex_dtype(dtype):
    if dtype in ("complex64", torch.complex64):
        return torch.complex64
    if dtype in ("complex128", torch.complex128):
        return torch.complex128
    raise ValueError("Precision must be complex64 or complex128")


def _tensor(value, device, dtype):
    return torch.as_tensor(value, device=device, dtype=complex_dtype(dtype))


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Nonfinite {name}")


def _noise(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Strictly positive finite receiver noise variance required")
    return value


@dataclass(frozen=True)
class WaveformGeometry:
    nfft: int = 1024
    n_active: int = 128
    cp_wave_samples: int = 128
    subcarrier_spacing_hz: float = 7500.0
    num_symbols: int = 25
    pilot_visible: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24)
    data_visible: tuple[int, ...] = tuple(i for i in range(25) if i % 4)
    active_signed_bins: tuple[int, ...] = tuple(range(-64, 64))

    def __post_init__(self):
        if not 1 <= self.n_active <= self.nfft or len(self.active_signed_bins) != self.n_active:
            raise ValueError("Invalid active-band geometry")
        if len(set(k % self.nfft for k in self.active_signed_bins)) != self.n_active:
            raise ValueError("Repeated active FFT bins")
        if not 0 <= self.cp_wave_samples <= self.nfft or self.subcarrier_spacing_hz <= 0:
            raise ValueError("Invalid CP or subcarrier spacing")
        p, d = set(self.pilot_visible), set(self.data_visible)
        if p & d or p | d != set(range(self.num_symbols)):
            raise ValueError("Pilot/data indices must partition the visible frame")

    @property
    def sample_rate_hz(self):
        return self.nfft * self.subcarrier_spacing_hz

    @property
    def symbol_duration_s(self):
        return (self.nfft + self.cp_wave_samples) / self.sample_rate_hz

    @property
    def useful_span_samples(self):
        return (self.num_symbols - 1) * (self.nfft + self.cp_wave_samples) + self.nfft


def geometry_from_export(manifest, arrays, frame=0):
    """Read only numerology and layout, never channel/condition oracle metadata."""
    b, _, _, n = arrays["x_pilot"].shape

    def scalar(name):
        value = np.asarray(arrays[name])
        if value.size == 1:
            return value.reshape(-1)[0].item()
        if value.shape == (b,):
            return value[frame].item()
        raise ValueError(f"Invalid per-frame numerology {name}")

    g = manifest["geometry"]
    return WaveformGeometry(
        nfft=int(scalar("waveform_fft_size")), n_active=n,
        cp_wave_samples=int(scalar("cp_wave_samples")),
        subcarrier_spacing_hz=float(scalar("subcarrier_spacing_hz")),
        num_symbols=int(g["frame_num_symbols"]),
        pilot_visible=tuple(int(t) for t in arrays["pilot_visible"]),
        data_visible=tuple(int(t) for t in arrays["data_visible"]),
        active_signed_bins=tuple(g["active_signed_bins"]))


def geometry_key(geometry):
    """Stable public-numerology key, independent of channel/test condition.

Pilot frequency spacing is intentionally absent: the same continuous-grid
prior is sampled at each frame's *actual* positions in estimate(). CP, physical
sample rate, visible pilot occasions and occupied FFT bins remain part of the
key because they determine the time/frequency coordinates of that prior.
"""
    payload = json.dumps(asdict(geometry), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def group_calibration_batches(arrays_list, manifest):
    """Group whole source-generation chunks by their public numerology.

Original independently generated calibration chunks must remain intact. A
mixed-geometry array is rejected rather than splitting its correlated frames
between prior fitting and hyperparameter evaluation.
"""
    groups = {}
    for arrays in arrays_list:
        count = int(arrays["x_pilot"].shape[0])
        if count < 1:
            raise ValueError("Empty calibration source chunk")
        g = geometry_from_export(manifest, arrays, 0)
        if any(geometry_from_export(manifest, arrays, bi) != g for bi in range(count)):
            raise ValueError("A source calibration chunk mixes public numerologies; export separate original chunks without splitting frames")
        key = geometry_key(g)
        group = groups.setdefault(key, {"geometry": g, "batches": [], "frames": 0})
        group["batches"].append(arrays)
        group["frames"] += count
    return groups


def split_calibration_groups(arrays_list, manifest, *, holdout_fraction=.25,
                             min_fit_frames=16, min_holdout_frames=8):
    """Deterministic complete-chunk holdout per CP/numerology, never test data.

Last original chunks form the holdout. The split uses only geometry and source
order, before inspecting NMSE. Insufficient groups require additional fresh
independent calibration, not changing the split after seeing test scores.
"""
    if not 0 < holdout_fraction < 1 or min(min_fit_frames, min_holdout_frames) < 1:
        raise ValueError("Invalid calibration holdout policy")
    groups = group_calibration_batches(arrays_list, manifest)
    result = {}
    for key, group in groups.items():
        goal = max(int(math.ceil(group["frames"] * holdout_fraction)), int(min_holdout_frames))
        cursor, held = len(group["batches"]), 0
        while cursor > 0 and held < goal:
            cursor -= 1
            held += group["batches"][cursor]["x_pilot"].shape[0]
        fit = group["frames"] - held
        if fit < min_fit_frames or held < min_holdout_frames:
            raise ValueError(f"Independent calibration group {key}, CP={group['geometry'].cp_wave_samples}: "
                             f"fit={fit}, holdout={held}; minimum={min_fit_frames}/{min_holdout_frames}. "
                             "Export additional independent calibration with a new declared seed; never use test truth.")
        result[key] = {"geometry": group["geometry"],
                       "fit_batches": group["batches"][:cursor],
                       "holdout_batches": group["batches"][cursor:],
                       "all_batches": group["batches"],
                       "fit_frames": int(fit), "holdout_frames": int(held),
                       "split_rule": "last complete original generation chunks per public numerology"}
    return result


@dataclass(frozen=True)
class BEMSettings:
    method: str = BEM_METHODS[0]
    q: int = 15
    delay_start: int = -8
    delay_stop: int = 40
    training_doppler_envelope_hz: float = 1620.0
    relative_ridge: float = 1.0
    fit_tolerance: float = 1e-6
    detector_tolerance: float = 1e-6

    def __post_init__(self):
        if self.method not in BEM_METHODS or self.q < 1:
            raise ValueError("Unknown BEM method or invalid order")
        if self.method == BEM_METHODS[0] and self.q % 2 != 1:
            raise ValueError("OCE basis order must be odd")
        if self.delay_stop <= self.delay_start or self.relative_ridge <= 0:
            raise ValueError("Positive ridge and nonempty delay support required")
        if self.training_doppler_envelope_hz <= 0:
            raise ValueError("Positive declared training Doppler envelope required")
        if min(self.fit_tolerance, self.detector_tolerance) <= 0:
            raise ValueError("Positive explicit residual thresholds required")


@lru_cache(maxsize=16)
def _temporal_basis_cpu(g, method, q, envelope):
    """Canonical float64 basis constructed once, identically for CPU/CUDA.

DPSS uses a physical-sample lattice including intervening cyclic prefixes;
its arbitrary eigenvector sign is frozen by this shared CPU construction.
"""
    if method == BEM_METHODS[0]:
        modes = np.arange(-(q // 2), q // 2 + 1, dtype=np.float64)
        frequencies = modes / (g.num_symbols * g.symbol_duration_s)
        times = (np.arange(g.num_symbols)[:, None] * g.symbol_duration_s
                 + (np.arange(g.nfft)[None] - (g.nfft - 1) / 2) / g.sample_rate_hz)
        basis = np.exp(2j * np.pi * times[:, None, :] * frequencies[None, :, None])
    elif method == BEM_METHODS[1]:
        from scipy.signal.windows import dpss
        span = g.useful_span_samples
        nw = span * float(envelope) / g.sample_rate_hz
        if not 0 < nw < span / 2 or q > span:
            raise ValueError("DPSS time-bandwidth product/order out of range")
        full = dpss(span, nw, Kmax=q, sym=True, norm=2) * np.sqrt(span)
        indices = (np.arange(g.num_symbols)[:, None] * (g.nfft + g.cp_wave_samples)
                   + np.arange(g.nfft)[None, :])
        basis = full[:, indices].transpose(1, 0, 2).astype(np.complex128)
    else:
        raise ValueError("Unknown BEM basis")
    k = np.asarray(g.active_signed_bins, dtype=np.int64)
    differences = (k[:, None] - k[None, :]) % g.nfft
    mixing = (np.fft.fft(basis, axis=-1) / g.nfft)[..., differences]
    return np.ascontiguousarray(mixing), np.ascontiguousarray(basis.mean(axis=-1))


def strict_cholesky_solve(matrix, rhs, *, tolerance=1e-6, max_refinement=3):
    """Hermitian-positive-definite direct solve with explicit true residual.

There is no jitter, failed-candidate filtering or silent PCG fallback.  The
caller must supply its declared noise/ridge regularization.  Same-precision
iterative refinement improves a direct solve; unmet tolerance aborts scoring.
"""
    if matrix.shape[-1] != matrix.shape[-2] or rhs.shape[-2] != matrix.shape[-1]:
        raise ValueError("Incompatible Cholesky matrix/RHS")
    if matrix.dtype not in (torch.complex64, torch.complex128) or rhs.dtype != matrix.dtype:
        raise ValueError("Matching Torch complex precision required")
    _finite(matrix, "normal matrix")
    _finite(rhs, "normal RHS")
    # Roundoff symmetry is enforced without altering any regularizer.
    matrix = (matrix + matrix.mH) * 0.5
    factor, status = torch.linalg.cholesky_ex(matrix, check_errors=False)
    if bool((status != 0).any()):
        raise SolverConvergenceError(
            f"Declared regularized matrix is not positive definite; cholesky_info={status.detach().cpu().tolist()}")
    answer = torch.cholesky_solve(rhs, factor)
    scale = torch.linalg.vector_norm(rhs, dim=-2).clamp_min(torch.finfo(matrix.real.dtype).tiny)
    refinement = 0
    for refinement in range(max_refinement + 1):
        residual = rhs - matrix @ answer
        relative = torch.linalg.vector_norm(residual, dim=-2) / scale
        if bool(torch.isfinite(relative).all()) and bool((relative <= tolerance).all()):
            break
        if refinement == max_refinement:
            raise SolverConvergenceError(
                f"Cholesky true residual {float(relative.max()):.6g} exceeds {tolerance:g}; "
                f"precision={matrix.dtype}, refinements={refinement}. Use complex128, not a failed score.")
        answer = answer + torch.cholesky_solve(residual, factor)
    return answer, {"solver": "hermitian_positive_definite_cholesky",
                    "relative_residual_max": float(relative.max()),
                    "relative_residual_mean": float(relative.mean()),
                    "tolerance": float(tolerance), "iterations": 0,
                    "refinement_steps": int(refinement),
                    "matrix_dimension": int(matrix.shape[-1]),
                    "precision": str(matrix.dtype).split(".")[-1]}


class TorchJointBEM:
    """Physical pilot design; one frame, all receive branches solved jointly."""

    def __init__(self, geometry, settings, n_users=2, *, device="cpu", dtype="complex128"):
        self.geometry, self.settings = geometry, settings
        self.n_users, self.device, self.dtype = int(n_users), torch.device(device), complex_dtype(dtype)
        if self.n_users < 1 or settings.delay_stop - settings.delay_start > geometry.n_active:
            raise ValueError("Invalid users or repeated delay atoms")
        mixing, mean = _temporal_basis_cpu(geometry, settings.method, settings.q,
                                           settings.training_doppler_envelope_hz)
        self.mixing, self.mean_basis = (_tensor(a, self.device, self.dtype) for a in (mixing, mean))
        delays = torch.arange(settings.delay_start, settings.delay_stop, device=self.device,
                              dtype=torch.float64)
        bins = torch.as_tensor(geometry.active_signed_bins, device=self.device, dtype=torch.float64)
        self.delay_phase = torch.exp(-2j * math.pi * bins[:, None] * delays[None] / geometry.n_active).to(self.dtype)
        self.coefficient_shape = (self.n_users, settings.q, delays.numel())
        self.n_coefficients = _product(self.coefficient_shape)

    def design_matrix(self, x_pilot):
        g = self.geometry
        x = _tensor(x_pilot, self.device, self.dtype)
        if x.shape != (self.n_users, len(g.pilot_visible), g.n_active):
            raise ValueError("Expected public pilots [U,P,N]")
        _finite(x, "public pilots")
        weighted = x[..., None] * self.delay_phase[None, None]
        mixing = self.mixing[list(g.pilot_visible)]
        return torch.einsum("pqnk,upkl->pnuql", mixing, weighted).reshape(
            len(g.pilot_visible) * g.n_active, self.n_coefficients).contiguous()

    def fit(self, x_pilot, y_pilot, noise_variance, *, design=None, relative_ridge=None):
        g, s = self.geometry, self.settings
        y = _tensor(y_pilot, self.device, self.dtype)
        if y.ndim != 3 or y.shape[0] != len(g.pilot_visible) or y.shape[-1] != g.n_active:
            raise ValueError("Expected received pilots [P,R,N]")
        _finite(y, "received pilots")
        noise = _noise(noise_variance)
        d = self.design_matrix(x_pilot) if design is None else _tensor(design, self.device, self.dtype)
        if d.shape != (len(g.pilot_visible) * g.n_active, self.n_coefficients):
            raise ValueError("Invalid BEM pilot design matrix")
        rho = s.relative_ridge if relative_ridge is None else float(relative_ridge)
        if rho <= 0 or not math.isfinite(rho):
            raise ValueError("Positive finite ridge candidate required")
        y_flat = y.permute(0, 2, 1).reshape(d.shape[0], y.shape[1])
        diagonal = d.abs().square().sum(dim=0)
        scale = float(diagonal.mean())
        ridge = rho * noise * scale
        # Underdetermined dictionaries are solved in the smaller observation
        # space: c = D^H (D D^H + lambda I)^-1 y. No PCG budget chooses rho.
        dual = d.shape[0] < d.shape[1]
        gram = d @ d.mH if dual else d.mH @ d
        rhs = y_flat if dual else d.mH @ y_flat
        normal = gram + ridge * torch.eye(gram.shape[0], device=self.device, dtype=self.dtype)
        solved, info = strict_cholesky_solve(normal, rhs, tolerance=s.fit_tolerance)
        c = d.mH @ solved if dual else solved
        # Also check the coefficient-space normal equation; reporting only a
        # successful smaller-space solve would miss a bad recovered estimate.
        primal_rhs = d.mH @ y_flat
        actual = primal_rhs - (d.mH @ (d @ c) + ridge * c)
        rel = torch.linalg.vector_norm(actual, dim=0) / torch.linalg.vector_norm(
            primal_rhs, dim=0).clamp_min(torch.finfo(d.real.dtype).tiny)
        if not bool(torch.isfinite(rel).all()) or bool((rel > s.fit_tolerance).any()):
            raise SolverConvergenceError(f"Recovered BEM normal residual {float(rel.max()):.6g} exceeds {s.fit_tolerance:g}")
        info.update(normal_space="observation" if dual else "coefficient",
                    recovered_normal_relative_residual_max=float(rel.max()),
                    recovered_normal_relative_residual_mean=float(rel.mean()),
                    ridge_lambda=float(ridge), relative_ridge=float(rho),
                    mean_gram_diagonal=scale, observations_per_receive_branch=int(d.shape[0]),
                    coefficients_per_receive_branch=int(d.shape[1]))
        return c.reshape(*self.coefficient_shape, y.shape[1]), info

    def coefficients_to_frequency(self, coefficients):
        c = _tensor(coefficients, self.device, self.dtype)
        if c.ndim != 4 or tuple(c.shape[:3]) != self.coefficient_shape:
            raise ValueError("Expected BEM coefficients [U,Q,L,R]")
        return torch.einsum("uqla,kl->uqak", c, self.delay_phase)

    def diagonal_cfr(self, coefficients, symbol_indices=None):
        indices = self.geometry.data_visible if symbol_indices is None else tuple(symbol_indices)
        return torch.einsum("tq,uqak->utak", self.mean_basis[list(indices)],
                            self.coefficients_to_frequency(coefficients))

    def channel_matrix(self, coefficients, symbol, frequency_coefficients=None):
        frequency = (self.coefficients_to_frequency(coefficients) if frequency_coefficients is None
                     else frequency_coefficients)
        return torch.einsum("qnk,uqak->anuk", self.mixing[int(symbol)], frequency)

    def native_mmse(self, coefficients, rx_data, noise_variance, *, symbol_batch_size=4):
        """Full occupied-band joint ICI-MMSE with batched exact HPD solves.

Symbol chunking changes working memory only. A CUDA OOM reduces this resource
chunk without changing matrix entries, ridge, precision or residual threshold.
"""
        g, s = self.geometry, self.settings
        y = _tensor(rx_data, self.device, self.dtype)
        frequency = self.coefficients_to_frequency(coefficients)
        nr = frequency.shape[2]
        if y.shape != (len(g.data_visible), nr, g.n_active) or symbol_batch_size < 1:
            raise ValueError("Invalid received payload or symbol chunk")
        _finite(y, "received payload")
        noise = _noise(noise_variance)
        out = torch.empty((self.n_users, len(g.data_visible), g.n_active), device=self.device, dtype=self.dtype)
        infos, start, chunk = [], 0, min(int(symbol_batch_size), len(g.data_visible))
        while start < len(g.data_visible):
            stop = min(len(g.data_visible), start + chunk)
            try:
                symbols = list(g.data_visible[start:stop])
                h = torch.einsum("tqnk,uqak->tanuk", self.mixing[symbols], frequency).reshape(
                    stop - start, nr * g.n_active, self.n_users * g.n_active)
                hh = h.mH
                normal = hh @ h + noise * torch.eye(h.shape[-1], device=self.device, dtype=self.dtype)
                rhs = hh @ y[start:stop].reshape(stop - start, nr * g.n_active, 1)
                z, info = strict_cholesky_solve(normal, rhs, tolerance=s.detector_tolerance)
                out[:, start:stop] = z[..., 0].reshape(stop - start, self.n_users, g.n_active).permute(1, 0, 2)
                infos.append(info)
                del h, hh, normal, rhs, z
                start = stop
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda" or chunk == 1:
                    raise
                # References from the failed iteration must not keep the
                # allocator occupied before retrying the identical symbol.
                h = hh = normal = rhs = z = None
                torch.cuda.empty_cache()
                chunk = max(1, chunk // 2)
        return out, {"detector": "full_occupied_band_joint_mmse_cholesky",
                     "relative_residual_max": max(i["relative_residual_max"] for i in infos),
                     "relative_residual_mean": float(np.mean([i["relative_residual_mean"] for i in infos])),
                     "iterations": 0, "symbol_batch_size": chunk,
                     "precision": str(self.dtype).split(".")[-1]}


def diagonal_mmse(cfr, rx_data, noise_variance, *, tolerance=1e-6, return_info=False):
    """Common unit-energy joint diagonal MMSE, independent of estimator family."""
    h = torch.as_tensor(cfr)
    if h.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("Complex CFR required")
    y = torch.as_tensor(rx_data, device=h.device, dtype=h.dtype)
    if h.ndim != 4 or tuple(y.shape) != tuple(h.shape[1:]):
        raise ValueError("Expected [U,D,R,N] CFR and [D,R,N] received data")
    _finite(h, "estimated CFR")
    _finite(y, "received data")
    ht, yt = h.permute(1, 3, 2, 0), y.permute(0, 2, 1)
    normal = ht.mH @ ht + _noise(noise_variance) * torch.eye(h.shape[0], device=h.device, dtype=h.dtype)
    rhs = ht.mH @ yt[..., None]
    z, info = strict_cholesky_solve(normal, rhs, tolerance=tolerance)
    result = z[..., 0].permute(2, 0, 1)
    return (result, info) if return_info else result


def _psd_project(matrix):
    """Nearest PSD matrix under the Frobenius norm; no diagonal jitter."""
    matrix = (matrix + matrix.mH) * 0.5
    values, vectors = torch.linalg.eigh(matrix)
    result = (vectors * values.clamp_min(0)[None]) @ vectors.mH
    return (result + result.mH) * 0.5, {"negative_eigenvalues": int((values < 0).sum()),
                                      "negative_eigenvalue_mass": float((-values.clamp_max(0)).sum())}


class PilotCovarianceLMMSE:
    """Properly scaled sample-covariance reference at the *actual* pilot grid.

    Frequency second moment is E[h(k) h(l)*], normalized by the number of sampled
frame/user/receive/time values. Temporal covariance averages only pairs having
the requested signed time lag, then projects the full-frame Toeplitz matrix to
PSD. Unsupported endpoint lags are explicitly zero before projection. Temporal
correlation is normalized to unit average diagonal; frequency covariance keeps
the measured channel power. No FFT circular-correlation sums are used.

Received user pilots are divided by their public complex values. Accordingly
    the AWGN covariance is diagonal noise_var/|x|^2 on those positions. The
    uncentered second moments specify a homogeneous (zero-intercept) linear
    estimator, rather than claiming an affine estimator with known Rician mean.
    A global
noise_scale is selected on independent calibration, accounting approximately
for pilot contamination/ICI. This is not a modeled full ICI covariance prior.
"""

    def __init__(self, frequency_covariance, temporal_correlation, geometry, *,
                 device="cpu", dtype="complex128", calibration_info=None):
        self.device, self.dtype = torch.device(device), complex_dtype(dtype)
        self.geometry = geometry
        self.rf = _tensor(frequency_covariance, self.device, self.dtype)
        self.rt = _tensor(temporal_correlation, self.device, self.dtype)
        self.n_users = self.rf.shape[0]
        if self.rf.shape != (self.n_users, geometry.n_active, geometry.n_active):
            raise ValueError("Invalid sample frequency covariance")
        if self.rt.shape != (self.n_users, geometry.num_symbols, geometry.num_symbols):
            raise ValueError("Invalid temporal correlation")
        self.calibration_info = dict(calibration_info or {})

    @classmethod
    def from_calibration(cls, arrays_list: Iterable[dict], manifest, *,
                         device="cpu", dtype="complex128"):
        """Targets are permissible here only for the independent ensemble.

The caller verifies calibration COMPLETE.json/seed and partitions frames
before invoking this API. Calibration holds out entire frames, never antennas.
"""
        batches = list(arrays_list)
        if not batches:
            raise ValueError("Empty independent calibration ensemble")
        g = geometry_from_export(manifest, batches[0], 0)
        dtype = complex_dtype(dtype)
        dev = torch.device(device)
        users = batches[0]["target_h"].shape[1]
        rf_sum = torch.zeros((users, g.n_active, g.n_active), device=dev, dtype=dtype)
        rt_sum = torch.zeros((users, g.num_symbols), device=dev, dtype=dtype)
        rt_count = torch.zeros(g.num_symbols, device=dev, dtype=torch.int64)
        freq_count, frames = 0, 0
        data = np.asarray(g.data_visible)
        lag_pairs = [(np.where(data[:, None] - data[None, :] == lag)) for lag in range(g.num_symbols)]
        for arrays in batches:
            h = _tensor(arrays["target_h"], dev, dtype)
            if h.ndim != 5 or h.shape[1] != users or h.shape[2] != len(g.data_visible) or h.shape[-1] != g.n_active:
                raise ValueError("Calibration target must be [B,U,D,R,N]")
            for bi in range(h.shape[0]):
                if geometry_from_export(manifest, arrays, bi) != g:
                    raise ValueError("Calibrate one public numerology at a time; mixed geometry covariance is not silently reused")
            _finite(h, "independent calibration channel")
            # h indexed [sample,U,D,R,k]; exactly averaged products, no
            # circular sum and no division by interpolation support counts.
            rf_sum += torch.einsum("budrk,budrl->ukl", h, h.conj())
            freq_count += h.shape[0] * h.shape[2] * h.shape[3]
            for lag, (left, right) in enumerate(lag_pairs):
                if len(left):
                    a, b = h[:, :, left], h[:, :, right]
                    rt_sum[:, lag] += (a * b.conj()).sum(dim=(0, 2, 3, 4))
                    rt_count[lag] += h.shape[0] * len(left) * h.shape[3] * h.shape[4]
            frames += h.shape[0]
        rf = rf_sum / freq_count
        rt_lag = rt_sum / rt_count.clamp_min(1)[None]
        indices = torch.arange(g.num_symbols, device=dev)
        diff = indices[:, None] - indices[None, :]
        rt = torch.empty((users, g.num_symbols, g.num_symbols), device=dev, dtype=dtype)
        projections = []
        for user in range(users):
            freq, f_info = _psd_project(rf[user])
            toeplitz = rt_lag[user, diff.abs()]
            toeplitz = torch.where(diff >= 0, toeplitz, toeplitz.conj())
            time, t_info = _psd_project(toeplitz)
            power = float(time.diagonal().real.mean())
            if power <= 0:
                raise ValueError("Degenerate calibration channel power")
            rf[user], rt[user] = freq, time / power
            projections.append({"user": user, "frequency": f_info, "time": t_info})
        info = {"calibration_frames": int(frames), "normalization": "mean sample products, not circular sums",
                "mean_convention": "uncentered second moments for a zero-intercept linear estimator",
                "frequency_covariance_samples_per_user": int(freq_count),
                "temporal_lag_product_counts": rt_count.detach().cpu().tolist(),
                "unsupported_temporal_lags_before_psd_projection": torch.where(rt_count == 0)[0].detach().cpu().tolist(),
                "covariance_model": "per-user separable frequency covariance times unit-power temporal correlation",
                "psd_projection": projections,
                "pilot_noise_model": "noise_var / abs(public pilot)^2, multiplied by independent-calibration global noise_scale"}
        return cls(rf, rt, g, device=dev, dtype=dtype, calibration_info=info)

    def state_arrays(self):
        return {"frequency_covariance": self.rf.detach().cpu().numpy(),
                "temporal_correlation": self.rt.detach().cpu().numpy()}

    def estimate(self, x_pilot, y_pilot, noise_variance, geometry=None, *,
                 noise_scale=1.0, tolerance=1e-6):
        g = self.geometry if geometry is None else geometry
        if g != self.geometry:
            raise ValueError("Statistical prior belongs to a different public numerology")
        x, y = (_tensor(a, self.device, self.dtype) for a in (x_pilot, y_pilot))
        if x.shape != (self.n_users, len(g.pilot_visible), g.n_active):
            raise ValueError("Expected public pilots [U,P,N]")
        if y.ndim != 3 or y.shape[0] != x.shape[1] or y.shape[-1] != g.n_active:
            raise ValueError("Expected received pilots [P,R,N]")
        _finite(x, "public pilots")
        _finite(y, "received pilots")
        noise = _noise(noise_variance)
        if noise_scale <= 0 or not math.isfinite(float(noise_scale)):
            raise ValueError("Positive finite global calibration noise_scale required")
        nr = y.shape[1]
        out = torch.empty((self.n_users, len(g.data_visible), nr, g.n_active), device=self.device, dtype=self.dtype)
        t_data = torch.as_tensor(g.data_visible, device=self.device).repeat_interleave(g.n_active)
        k_data = torch.arange(g.n_active, device=self.device).repeat(len(g.data_visible))
        p_times = torch.as_tensor(g.pilot_visible, device=self.device)
        infos, pilot_counts = [], []
        for user in range(self.n_users):
            # torch.nonzero fixes [pilot occasion,frequency] order. No values
            # from interpolated LS grids ever enter the observation vector.
            locations = torch.nonzero(x[user].abs() > 0, as_tuple=False)
            if not locations.numel():
                raise ValueError("User has no transmitted pilot REs")
            pi, ki = locations[:, 0], locations[:, 1]
            ti = p_times[pi]
            xp = x[user, pi, ki]
            ls = y[pi, :, ki] / xp[:, None]
            rpp = self.rt[user][ti[:, None], ti[None, :]] * self.rf[user][ki[:, None], ki[None, :]]
            rhp = self.rt[user][t_data[:, None], ti[None, :]] * self.rf[user][k_data[:, None], ki[None, :]]
            noise_diagonal = float(noise_scale) * noise / xp.abs().square()
            normal = rpp + torch.diag(noise_diagonal).to(self.dtype)
            solved, info = strict_cholesky_solve(normal, ls, tolerance=tolerance)
            estimate = rhp @ solved
            out[user] = estimate.reshape(len(g.data_visible), g.n_active, nr).permute(0, 2, 1)
            infos.append(info)
            pilot_counts.append(locations.shape[0])
        return out, {"solver": "pilot_covariance_hpd_cholesky",
                     "relative_residual_max": max(i["relative_residual_max"] for i in infos),
                     "relative_residual_mean": float(np.mean([i["relative_residual_mean"] for i in infos])),
                     "iterations": 0, "noise_scale": float(noise_scale),
                     "pilot_re_counts_per_user": pilot_counts,
                     "matrix_dimensions_per_user": pilot_counts,
                     "precision": str(self.dtype).split(".")[-1]}


def nmse_per_frame(estimate, target):
    h = torch.as_tensor(estimate)
    truth = torch.as_tensor(target, device=h.device, dtype=h.dtype)
    if h.shape != truth.shape or h.ndim != 5:
        raise ValueError("Expected matching [B,U,D,R,N] CFR tensors")
    _finite(h, "estimated CFR")
    _finite(truth, "scoring target")
    dims = tuple(range(1, h.ndim))
    # Magnitudes promote before squaring to retain the original score scale.
    error = (h - truth).abs().to(torch.float64).square().sum(dim=dims)
    power = truth.abs().to(torch.float64).square().sum(dim=dims)
    return error / power.clamp_min(1e-12)


def frame_bit_counts(symbols, labels, modulation):
    """Original unit-energy Gray constellation, score-only label interface."""
    z = torch.as_tensor(symbols)
    truth = torch.as_tensor(labels, device=z.device, dtype=torch.int64)
    if z.shape != truth.shape or z.ndim != 4:
        raise ValueError("Expected matching [B,U,D,N] symbols and labels")
    _finite(z, "detected symbols")
    bps = {"bpsk": 1, "qpsk": 2, "16qam": 4, "64qam": 6}[modulation.lower()]
    if bool(((truth < 0) | (truth >= 1 << bps)).any()):
        raise ValueError("Invalid true constellation label")
    labels_all = torch.arange(1 << bps, device=z.device, dtype=torch.int64)
    if bps == 1:
        constellation = (2 * labels_all - 1).to(torch.complex64)
    else:
        side = 1 << (bps // 2)
        gi, gq = labels_all >> (bps // 2), labels_all & (side - 1)
        def gray_binary(gray):
            binary = gray.clone()
            for shift in (1, 2, 4):
                binary = binary ^ (binary >> shift)
            return binary
        i, q = (2 * gray_binary(a) - (side - 1) for a in (gi, gq))
        constellation = ((i.to(torch.float64) + 1j * q.to(torch.float64)) /
                         math.sqrt((2 / 3) * ((1 << bps) - 1))).to(torch.complex64)
    predicted = (z.to(torch.complex128)[..., None] - constellation.to(torch.complex128)).abs().argmin(dim=-1)
    xor = predicted ^ truth
    dims = tuple(range(1, z.ndim))
    errors = sum(((xor >> bit) & 1).sum(dim=dims) for bit in range(bps))
    bits = torch.full((z.shape[0],), _product(z.shape[1:]) * bps, device=z.device, dtype=torch.int64)
    return errors, bits
