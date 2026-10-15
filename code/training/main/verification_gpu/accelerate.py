"""Bounded public-design caching for the unchanged complex128 receivers.

The frozen generator, estimator equations, regularizers and residual gates are
unchanged. Cache entries contain only public-pilot design blocks. Calibration
can also reuse a frame's Gram matrix across its declared ridge candidates.
"""
from collections import OrderedDict, Counter
from contextlib import contextmanager
import itertools
import math
import time

import numpy as np
import torch

from verification_extension import classical as C

OriginalJointBEM = C.TorchJointBEM
_tokens = itertools.count()
_blocks = OrderedDict()
_block_bytes = 0
_limit_bytes = 1024 * 1024 * 1024
_native_chunks = {}
COUNTS = Counter()


def configure(cache_mib=1024, native_chunks=None):
    global _limit_bytes, _native_chunks
    if int(cache_mib) < 0:
        raise ValueError('Cache budget must be nonnegative')
    clear()
    _limit_bytes = int(cache_mib) * 1024 * 1024
    _native_chunks = dict(native_chunks or {})


def clear():
    global _block_bytes
    _blocks.clear()
    _block_bytes = 0
    COUNTS.clear()


def _pilot_keys(token, x):
    # Byte equality includes both users, every transmitted value and every zero
    # in an occasion. No assumption of a specific pilot alphabet is necessary.
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return [(token, pi, x.dtype.str, tuple(x[:, pi].shape),
             np.ascontiguousarray(x[:, pi]).tobytes()) for pi in range(x.shape[1])]


class CachedJointBEM(OriginalJointBEM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._token = next(_tokens)
        self._last_design = None
        self._last_version = None
        self._last_gram = None
        self._last_scale = None
        self._last_snapshot = None

    def design_matrix(self, x_pilot):
        global _block_bytes
        if not _limit_bytes:
            return super().design_matrix(x_pilot)
        keys = _pilot_keys(self._token, x_pilot)
        if all(key in _blocks for key in keys):
            # Stored on the host: this cache cannot consume the GPU headroom
            # needed by the original unmodified 3GPP waveform generator.
            chunks = [_blocks[key] for key in keys]
            for key in keys:
                _blocks.move_to_end(key)
            COUNTS['design_hit_frames'] += 1
            return torch.cat(chunks, dim=0).to(device=self.device, dtype=self.dtype)
        result = super().design_matrix(x_pilot)
        COUNTS['design_miss_frames'] += 1
        rows = self.geometry.n_active
        for pi, key in enumerate(keys):
            if key in _blocks:
                _blocks.move_to_end(key)
                continue
            block = result[pi * rows:(pi + 1) * rows].detach().cpu().clone()
            size = block.numel() * block.element_size()
            if size > _limit_bytes:
                continue
            while _blocks and _block_bytes + size > _limit_bytes:
                _, old = _blocks.popitem(last=False)
                _block_bytes -= old.numel() * old.element_size()
                COUNTS['design_evictions'] += 1
            _blocks[key] = block
            _block_bytes += size
        return result

    def fit(self, x_pilot, y_pilot, noise_variance, *, design=None, relative_ridge=None):
        if design is None:
            return super().fit(x_pilot, y_pilot, noise_variance,
                               relative_ridge=relative_ridge)
        g, s = self.geometry, self.settings
        y = C._tensor(y_pilot, self.device, self.dtype)
        if y.ndim != 3 or y.shape[0] != len(g.pilot_visible) or y.shape[-1] != g.n_active:
            raise ValueError('Expected received pilots [P,R,N]')
        C._finite(y, 'received pilots')
        noise = C._noise(noise_variance)
        d = C._tensor(design, self.device, self.dtype)
        if d.shape != (len(g.pilot_visible) * g.n_active, self.n_coefficients):
            raise ValueError('Invalid BEM pilot design matrix')
        rho = s.relative_ridge if relative_ridge is None else float(relative_ridge)
        if rho <= 0 or not math.isfinite(rho):
            raise ValueError('Positive finite ridge candidate required')
        y_flat = y.permute(0, 2, 1).reshape(d.shape[0], y.shape[1])
        dual = d.shape[0] < d.shape[1]
        # Identity plus Torch's mutation counter, never a pointer-only cache.
        # Different observations/noise/ridge still receive their own solve.
        try:
            version = d._version
        except RuntimeError:
            # Inference-mode tensors intentionally have no mutation counter.
            # Compare to a private snapshot instead of assuming immutability.
            version = None
        unchanged = (self._last_design is d and
                     ((version is not None and self._last_version == version) or
                      (version is None and self._last_snapshot is not None and
                       torch.equal(d, self._last_snapshot))))
        if unchanged:
            gram, scale = self._last_gram, self._last_scale
            COUNTS['ridge_gram_hits'] += 1
        else:
            scale = float(d.abs().square().sum(dim=0).mean())
            gram = d @ d.mH if dual else d.mH @ d
            self._last_design, self._last_version = d, version
            self._last_gram, self._last_scale = gram, scale
            self._last_snapshot = d.clone() if version is None else None
            COUNTS['ridge_gram_misses'] += 1
        ridge = rho * noise * scale
        rhs = y_flat if dual else d.mH @ y_flat
        normal = gram + ridge * torch.eye(gram.shape[0], device=self.device, dtype=self.dtype)
        solved, info = C.strict_cholesky_solve(normal, rhs, tolerance=s.fit_tolerance)
        c = d.mH @ solved if dual else solved
        primal_rhs = d.mH @ y_flat
        actual = primal_rhs - (d.mH @ (d @ c) + ridge * c)
        rel = torch.linalg.vector_norm(actual, dim=0) / torch.linalg.vector_norm(
            primal_rhs, dim=0).clamp_min(torch.finfo(d.real.dtype).tiny)
        if not bool(torch.isfinite(rel).all()) or bool((rel > s.fit_tolerance).any()):
            raise C.SolverConvergenceError(
                'Recovered BEM normal residual %g exceeds %g' % (float(rel.max()), s.fit_tolerance))
        info.update(normal_space='observation' if dual else 'coefficient',
                    recovered_normal_relative_residual_max=float(rel.max()),
                    recovered_normal_relative_residual_mean=float(rel.mean()),
                    ridge_lambda=float(ridge), relative_ridge=float(rho),
                    mean_gram_diagonal=scale, observations_per_receive_branch=int(d.shape[0]),
                    coefficients_per_receive_branch=int(d.shape[1]))
        return c.reshape(*self.coefficient_shape, y.shape[1]), info

    def native_mmse(self, coefficients, rx_data, noise_variance, *, symbol_batch_size=4):
        key = str(self.device) + '/' + self.settings.method
        chunk = int(_native_chunks.get(key, symbol_batch_size))
        return super().native_mmse(coefficients, rx_data, noise_variance,
                                   symbol_batch_size=chunk)


@contextmanager
def installed(cache_mib=1024, native_chunks=None):
    configure(cache_mib, native_chunks)
    previous = C.TorchJointBEM
    C.TorchJointBEM = CachedJointBEM
    try:
        yield
    finally:
        C.TorchJointBEM = previous


def statistics():
    return {**dict(COUNTS), 'host_cache_bytes': _block_bytes,
            'host_cache_limit_bytes': _limit_bytes, 'gpu_design_cache_bytes': 0}
