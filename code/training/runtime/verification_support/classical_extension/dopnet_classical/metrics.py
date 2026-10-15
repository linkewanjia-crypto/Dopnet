"""The original unit-energy Gray constellation and per-frame score convention.

Channel estimates and symbols are scored here, never passed back to an estimator.
"""
from __future__ import annotations

import numpy as np


def bits_per_symbol(modulation: str) -> int:
    return {"bpsk": 1, "qpsk": 2, "16qam": 4, "64qam": 6}[modulation.lower()]


def labels_to_symbols(labels, modulation: str) -> np.ndarray:
    """Exactly the generator's label-to-constellation convention."""
    modulation = modulation.lower()
    labels = np.asarray(labels, dtype=np.int64)
    bps = bits_per_symbol(modulation)
    if np.any((labels < 0) | (labels >= 1 << bps)):
        raise ValueError("Constellation label out of range")
    if modulation == "bpsk":
        return (2.0 * labels - 1.0).astype(np.complex128)
    side = 1 << (bps // 2)
    mask = side - 1
    gi, gq = labels >> (bps // 2), labels & mask

    def gray_to_binary(g):
        b = g.copy()
        shift = 1
        while shift < 8:
            b ^= b >> shift
            shift <<= 1
        return b

    i = 2.0 * gray_to_binary(gi) - (side - 1)
    q = 2.0 * gray_to_binary(gq) - (side - 1)
    # The waveform uses complex64 constellation values. Preserve those rounded
    # points when comparing decisions, while doing arithmetic in complex128.
    return ((i + 1j * q) / np.sqrt((2.0 / 3.0) * ((1 << bps) - 1))).astype(
        np.complex64).astype(np.complex128)


def demodulate_to_labels(symbols, modulation: str) -> np.ndarray:
    constellation = labels_to_symbols(np.arange(1 << bits_per_symbol(modulation)), modulation)
    z = np.asarray(symbols, dtype=np.complex128)
    if not np.all(np.isfinite(z)):
        raise ValueError("Nonfinite detector output")
    return np.abs(z[..., None] - constellation).argmin(axis=-1).astype(np.int64)


def frame_bit_counts(symbols, labels, modulation: str) -> tuple[np.ndarray, np.ndarray]:
    """Inputs [B,U,D,N]; users and all payload REs stay within each frame."""
    symbols, labels = np.asarray(symbols), np.asarray(labels, dtype=np.int64)
    if symbols.shape != labels.shape or symbols.ndim != 4:
        raise ValueError("Expected matching [B,U,D,N] symbols and labels")
    bps = bits_per_symbol(modulation)
    if np.any((labels < 0) | (labels >= 1 << bps)):
        raise ValueError("True labels out of range")
    xor = demodulate_to_labels(symbols, modulation) ^ labels
    dims = tuple(range(1, xor.ndim))
    errors = sum(((xor >> bit) & 1).sum(axis=dims) for bit in range(bps))
    bits = np.full(symbols.shape[0], np.prod(symbols.shape[1:]) * bps, dtype=np.int64)
    return np.asarray(errors, dtype=np.int64), bits


def nmse_per_frame(estimate, target) -> np.ndarray:
    """Linear sample NMSE, pooled over two users and payload blocks per frame."""
    estimate, target = np.asarray(estimate), np.asarray(target)
    if estimate.shape != target.shape or estimate.ndim != 5:
        raise ValueError("Expected matching [B,U,D,R,N] CFR tensors")
    if not np.all(np.isfinite(estimate)) or not np.all(np.isfinite(target)):
        raise ValueError("Nonfinite CFR")
    dims = tuple(range(1, estimate.ndim))
    numerator = np.abs(estimate - target).astype(np.float64) ** 2
    denominator = np.abs(target).astype(np.float64) ** 2
    return numerator.sum(axis=dims) / np.maximum(denominator.sum(axis=dims), 1e-12)
