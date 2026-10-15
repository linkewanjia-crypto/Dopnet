"""Joint pilot-only OCE/DPSS BEM channel estimation and ICI-aware detection.

The implementation adapts conventional BEM least squares to the *existing*
two-user waveform. It is not a reproduction of a different paper's pilot design.
Only received pilots, public pilot values/numerology and noise variance enter
the estimator. Ground-truth CFR and payload labels are scoring-only inputs.

Both bases represent useful-sample variation across the complete frame. The
cyclic-delay BEM forward operator uses the physical waveform FFT (1024), then
crops the occupied 128 bins. It never approximates that frequency mixing with
a 128-point FFT. This is a finite cyclic BEM, not the simulator's full multipath
or intersymbol-interference model; basis/delay mismatch remains in the scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.signal.windows import dpss


METHODS = ("oce_bem_joint_ridge", "dpss_bem_joint_ridge")


class SolverConvergenceError(RuntimeError):
    """A failed solve cannot contribute a performance result."""


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
            raise ValueError("Pilot/data visible indices must partition the frame")

    @property
    def sample_rate_hz(self) -> float:
        return self.nfft * self.subcarrier_spacing_hz

    @property
    def symbol_duration_s(self) -> float:
        return (self.nfft + self.cp_wave_samples) / self.sample_rate_hz

    @property
    def useful_span_samples(self) -> int:
        return (self.num_symbols - 1) * (self.nfft + self.cp_wave_samples) + self.nfft


@dataclass(frozen=True)
class BEMSettings:
    method: str = METHODS[0]
    q: int = 15
    # Delay indices use occupied-band sample units (one is Nfft/Nactive waveform
    # samples), matching the paper's lattice. They are not physical FFT samples.
    delay_start: int = -8
    delay_stop: int = 40
    # One envelope from the declared training distribution; never per-test fd.
    training_doppler_envelope_hz: float = 1620.0
    relative_ridge: float = 1.0
    fit_max_iter: int = 256
    fit_tolerance: float = 1e-6
    detector_max_iter: int = 128
    detector_tolerance: float = 1e-6

    def __post_init__(self):
        if self.method not in METHODS or self.q < 1:
            raise ValueError("Unknown method or invalid BEM order")
        if self.method == METHODS[0] and self.q % 2 != 1:
            raise ValueError("OCE basis needs an odd order")
        if self.delay_stop <= self.delay_start or self.relative_ridge < 0:
            raise ValueError("Invalid delay support or ridge")
        if self.training_doppler_envelope_hz <= 0:
            raise ValueError("Training Doppler envelope must be positive")
        if min(self.fit_max_iter, self.detector_max_iter) < 1:
            raise ValueError("Positive solver iteration budgets required")
        if min(self.fit_tolerance, self.detector_tolerance) <= 0:
            raise ValueError("Positive solver tolerances required")


@lru_cache(maxsize=12)
def temporal_basis(geometry: WaveformGeometry, method: str, q: int,
                   training_doppler_envelope_hz: float) -> np.ndarray:
    """[T,Q,Nfft] useful-window basis, unit RMS scale over the physical frame.

OCE modes are q/Tframe cycles/s, as in the specified DOP-Net dictionary.
The arbitrary sample-origin shift changes the coefficients, not fitted channels.
DPSS uses a uniform physical-sample time grid including the intervening CP gaps.
"""
    n = np.arange(geometry.nfft, dtype=np.float64)
    if method == METHODS[0]:
        if q % 2 != 1:
            raise ValueError("OCE q must be odd")
        frequencies = np.arange(-(q // 2), q // 2 + 1) / (
            geometry.num_symbols * geometry.symbol_duration_s)
        times = (np.arange(geometry.num_symbols)[:, None] * geometry.symbol_duration_s
                 + (n[None] - (geometry.nfft - 1) / 2) / geometry.sample_rate_hz)
        result = np.exp(2j * np.pi * times[:, None, :] * frequencies[None, :, None])
    elif method == METHODS[1]:
        span = geometry.useful_span_samples
        bandwidth = training_doppler_envelope_hz / geometry.sample_rate_hz
        nw = span * bandwidth
        if not 0 < nw < span / 2 or q > span:
            raise ValueError("DPSS time-bandwidth or order out of range")
        continuous = dpss(span, nw, Kmax=q, sym=True, norm=2) * np.sqrt(span)
        sample_indices = (np.arange(geometry.num_symbols)[:, None]
                          * (geometry.nfft + geometry.cp_wave_samples)
                          + np.arange(geometry.nfft)[None, :])
        result = continuous[:, sample_indices].transpose(1, 0, 2).astype(np.complex128)
    else:
        raise ValueError("Unknown BEM method")
    result.setflags(write=False)
    return result


class JointBEMOperator:
    """One frame's physical pilot operator, shared by all receive branches."""

    def __init__(self, geometry: WaveformGeometry, settings: BEMSettings, n_users: int = 2):
        self.geometry, self.settings, self.n_users = geometry, settings, int(n_users)
        if self.n_users < 1:
            raise ValueError("Invalid number of users")
        self.delays = np.arange(settings.delay_start, settings.delay_stop, dtype=np.float64)
        if len(self.delays) > geometry.n_active:
            raise ValueError("Repeated delay atoms are not allowed")
        self.basis = temporal_basis(geometry, settings.method, settings.q,
                                    settings.training_doppler_envelope_hz)
        k = np.asarray(geometry.active_signed_bins, dtype=np.float64)
        self.delay_phase = np.exp(-2j * np.pi * k[:, None] * self.delays[None] / geometry.n_active)
        # F diag(b) F^H at physical FFT resolution. Cropping keeps noncyclic
        # occupied-band edges: shifts wrap modulo Nfft, never modulo Nactive.
        differences = (k.astype(np.int64)[:, None] - k.astype(np.int64)[None]) % geometry.nfft
        fourier = np.fft.fft(self.basis, axis=-1) / geometry.nfft
        self.mixing = fourier[..., differences]  # [T,Q,Nout,Nin]
        self.mean_basis = self.basis.mean(axis=-1)  # exact useful-window average
        self.coefficient_shape = (self.n_users, settings.q, len(self.delays))
        self.n_coefficients = int(np.prod(self.coefficient_shape))

    def design_matrix(self, x_pilot: np.ndarray) -> np.ndarray:
        """[P*N,U*Q*L], including all occupied received pilot-block bins."""
        g, q = self.geometry, self.settings.q
        x = np.asarray(x_pilot, dtype=np.complex128)
        if x.shape != (self.n_users, len(g.pilot_visible), g.n_active):
            raise ValueError("Expected public pilot tensor [U,P,N]")
        if not np.all(np.isfinite(x)) or np.any(np.sum(np.abs(x) ** 2, axis=(1, 2)) <= 0):
            raise ValueError("Invalid pilot values")
        result = np.empty((len(g.pilot_visible), g.n_active, self.n_users, q, len(self.delays)),
                          dtype=np.complex128)
        for pi, symbol in enumerate(g.pilot_visible):
            weighted = x[:, pi, :, None] * self.delay_phase[None]
            result[pi] = np.einsum("qnk,ukl->nuql", self.mixing[symbol], weighted, optimize=True)
        return result.reshape(len(g.pilot_visible) * g.n_active, self.n_coefficients)

    def coefficients_to_frequency(self, coefficients: np.ndarray) -> np.ndarray:
        """[U,Q,L,R] -> [U,Q,R,N], excluding temporal multiplication."""
        c = np.asarray(coefficients, dtype=np.complex128)
        if c.ndim != 4 or c.shape[:3] != self.coefficient_shape:
            raise ValueError("Expected joint BEM coefficients [U,Q,L,R]")
        return np.einsum("uqla,kl->uqak", c, self.delay_phase, optimize=True)

    def diagonal_cfr(self, coefficients: np.ndarray, symbol_indices=None) -> np.ndarray:
        """Return [U,D,R,N], the diagonal averaged over the useful samples."""
        indices = self.geometry.data_visible if symbol_indices is None else tuple(symbol_indices)
        frequency = self.coefficients_to_frequency(coefficients)
        return np.einsum("tq,uqak->utak", self.mean_basis[list(indices)], frequency, optimize=True)

    def channel_matrix(self, coefficients: np.ndarray, symbol: int,
                       frequency_coefficients=None) -> np.ndarray:
        """[R,Nout,U,Nin], with all physical FFT-derived ICI entries retained."""
        frequency = (self.coefficients_to_frequency(coefficients) if frequency_coefficients is None
                     else frequency_coefficients)
        return np.einsum("qnk,uqak->anuk", self.mixing[symbol], frequency, optimize=True)

    def forward_pilots(self, coefficients: np.ndarray, x_pilot: np.ndarray) -> np.ndarray:
        """[U,Q,L,R] -> [P,R,N]."""
        design = self.design_matrix(x_pilot)
        r = coefficients.shape[-1]
        y = design @ np.asarray(coefficients).reshape(self.n_coefficients, r)
        return y.reshape(len(self.geometry.pilot_visible), self.geometry.n_active, r).transpose(0, 2, 1)


def pcg(matvec: Callable[[np.ndarray], np.ndarray], rhs: np.ndarray, diagonal: np.ndarray,
        *, max_iter: int, tolerance: float) -> tuple[np.ndarray, dict]:
    """Independent complex PCG solves for columns of rhs, checked afterwards.

No true-channel fallback and no acceptance of nonconverged results. All normal
equation arithmetic is complex128, including the final explicit residual.
"""
    b = np.asarray(rhs, dtype=np.complex128)
    vector = b.ndim == 1
    if vector:
        b = b[:, None]
    if b.ndim != 2 or not np.all(np.isfinite(b)):
        raise ValueError("PCG rhs must be finite [dimension,rhs_count]")
    diagonal = np.asarray(diagonal, dtype=np.float64)
    if diagonal.shape != (b.shape[0],) or np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
        raise ValueError("PCG needs a finite positive diagonal preconditioner")
    if max_iter < 1 or tolerance <= 0:
        raise ValueError("Invalid PCG settings")
    norm_b = np.linalg.norm(b, axis=0)
    nonzero = norm_b > 0
    scale = np.where(nonzero, norm_b, 1.0)
    x, residual = np.zeros_like(b), b.copy()
    z = residual / diagonal[:, None]
    direction = z.copy()
    rz = np.sum(residual.conj() * z, axis=0).real
    active = nonzero.copy()
    iterations = np.zeros(b.shape[1], dtype=np.int64)
    for it in range(max_iter):
        if not np.any(active):
            break
        ad = np.asarray(matvec(direction), dtype=np.complex128)
        denom = np.sum(direction.conj() * ad, axis=0).real
        if np.any(denom[active] <= 0) or not np.all(np.isfinite(ad)):
            raise SolverConvergenceError("PCG encountered a nonpositive normal-equation direction")
        alpha = np.zeros_like(rz)
        alpha[active] = rz[active] / denom[active]
        x += direction * alpha[None]
        residual -= ad * alpha[None]
        iterations[active] = it + 1
        relative = np.linalg.norm(residual, axis=0) / scale
        active &= relative > tolerance
        z = residual / diagonal[:, None]
        rz_new = np.sum(residual.conj() * z, axis=0).real
        beta = np.zeros_like(rz)
        beta[active] = rz_new[active] / rz[active]
        direction = z + direction * beta[None]
        direction[:, ~active] = 0
        rz = rz_new
    actual = b - np.asarray(matvec(x), dtype=np.complex128)
    relative = np.linalg.norm(actual, axis=0) / scale
    relative[~nonzero] = 0
    info = {"iterations": int(iterations.max(initial=0)),
            "relative_residual_max": float(relative.max(initial=0)),
            "relative_residual_mean": float(relative.mean()),
            "tolerance": float(tolerance), "max_iter": int(max_iter)}
    if not np.all(np.isfinite(relative)) or np.any(relative > tolerance):
        raise SolverConvergenceError(f"PCG did not converge: {info}")
    return (x[:, 0] if vector else x), info


def fit_joint_bem(operator: JointBEMOperator, x_pilot: np.ndarray,
                  y_pilot: np.ndarray, noise_variance: float,
                  *, design=None, relative_ridge=None) -> tuple[np.ndarray, dict]:
    """Estimate [U,Q,L,R] without any channel/payload targets.

lambda = relative_ridge * noise_variance * mean(diag(D^H D)).
Noise variance follows the frozen receiver's knowledge convention. Unit-energy
payload symbols fix its scale; no per-condition Doppler/delay metadata is used.
"""
    g, s = operator.geometry, operator.settings
    y = np.asarray(y_pilot, dtype=np.complex128)
    if y.ndim != 3 or y.shape[0] != len(g.pilot_visible) or y.shape[2] != g.n_active:
        raise ValueError("Expected received pilots [P,R,N]")
    if not np.all(np.isfinite(y)) or noise_variance < 0 or not np.isfinite(noise_variance):
        raise ValueError("Invalid received pilots/noise variance")
    d = operator.design_matrix(x_pilot) if design is None else np.asarray(design, dtype=np.complex128)
    if d.shape != (len(g.pilot_visible) * g.n_active, operator.n_coefficients):
        raise ValueError("Invalid pilot design matrix")
    y_flat = y.transpose(0, 2, 1).reshape(d.shape[0], y.shape[1])
    dh = d.conj().T.copy()
    diag = np.sum(np.abs(d) ** 2, axis=0)
    scale = float(diag.mean())
    chosen_relative = s.relative_ridge if relative_ridge is None else float(relative_ridge)
    if chosen_relative < 0 or not np.isfinite(chosen_relative):
        raise ValueError("Invalid relative ridge")
    ridge = float(chosen_relative) * float(noise_variance) * scale
    rhs = dh @ y_flat

    def matvec(c):
        return dh @ (d @ c) + ridge * c

    c, info = pcg(matvec, rhs, diag + ridge,
                  max_iter=s.fit_max_iter, tolerance=s.fit_tolerance)
    info.update({"ridge_lambda": ridge, "mean_gram_diagonal": scale,
                 "observations_per_receive_branch": int(d.shape[0]),
                 "coefficients_per_receive_branch": int(d.shape[1]),
                 "relative_ridge": float(chosen_relative)})
    return c.reshape(*operator.coefficient_shape, y.shape[1]), info


def diagonal_mmse(cfr: np.ndarray, rx_data: np.ndarray,
                  noise_variance: float) -> np.ndarray:
    """Common unit-energy joint two-user diagonal MMSE, [U,D,R,N] -> [U,D,N]."""
    h, y = np.asarray(cfr, dtype=np.complex128), np.asarray(rx_data, dtype=np.complex128)
    if h.ndim != 4 or y.shape != h.shape[1:]:
        raise ValueError("Expected CFR [U,D,R,N] and received payload [D,R,N]")
    if noise_variance < 0 or not np.isfinite(noise_variance):
        raise ValueError("Invalid noise variance")
    ht = h.transpose(1, 3, 2, 0)  # [D,N,R,U]
    yt = y.transpose(0, 2, 1)     # [D,N,R]
    hh = ht.conj().swapaxes(-1, -2)
    normal = hh @ ht + float(noise_variance) * np.eye(h.shape[0])
    rhs = (hh @ yt[..., None])[..., 0]
    return np.linalg.solve(normal, rhs[..., None])[..., 0].transpose(2, 0, 1)


def native_mmse(operator: JointBEMOperator, coefficients: np.ndarray,
                rx_data: np.ndarray, noise_variance: float) -> tuple[np.ndarray, dict]:
    """Full occupied-band joint ICI-MMSE, matrix-free normal-equation PCG.

One payload-symbol channel is held at a time, with no band-center replacement.
The fitted BEM supplies both diagonal and off-diagonal entries. It uses no
ground-truth CFR, oracle ICI matrix, known payload symbols or hidden path stats.
"""
    g, s = operator.geometry, operator.settings
    y = np.asarray(rx_data, dtype=np.complex128)
    nr = coefficients.shape[-1]
    if y.shape != (len(g.data_visible), nr, g.n_active):
        raise ValueError("Invalid received payload tensor")
    if not np.all(np.isfinite(y)) or noise_variance < 0 or not np.isfinite(noise_variance):
        raise ValueError("Invalid payload/noise variance")
    frequency = operator.coefficients_to_frequency(coefficients)
    out = np.empty((operator.n_users, len(g.data_visible), g.n_active), dtype=np.complex128)
    infos = []
    for di, symbol in enumerate(g.data_visible):
        h = operator.channel_matrix(coefficients, symbol, frequency).reshape(
            nr * g.n_active, operator.n_users * g.n_active)
        hh = h.conj().T.copy()
        rhs = hh @ y[di].reshape(-1)
        diag = np.sum(np.abs(h) ** 2, axis=0) + float(noise_variance)

        def matvec(x):
            return hh @ (h @ x) + float(noise_variance) * x

        x, info = pcg(matvec, rhs, diag, max_iter=s.detector_max_iter,
                      tolerance=s.detector_tolerance)
        out[:, di] = x.reshape(operator.n_users, g.n_active)
        infos.append(info)
    return out, {"relative_residual_max": max(i["relative_residual_max"] for i in infos),
                 "relative_residual_mean": float(np.mean([i["relative_residual_mean"] for i in infos])),
                 "iterations": max(i["iterations"] for i in infos),
                 "detector": "full_occupied_band_joint_mmse_pcg"}
