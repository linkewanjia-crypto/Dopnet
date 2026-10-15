#!/usr/bin/env python3
"""Numerical regression tests for the banded ICI-aware MMSE implementation."""
import torch

from models.receiver import (
    apply_banded_multiuser_channel,
    apply_banded_multiuser_channel_adjoint,
    banded_multiuser_equalize,
    diagonal_csi_to_ici_bands,
    linear_multiuser_equalize,
)


def _ri_channel(h):
    return torch.stack([h.real, h.imag], dim=2).float()


def _ri_rx(y):
    return torch.stack([y.real, y.imag], dim=1).float()


def main():
    torch.manual_seed(20260911)

    # Active-band edges must be zero-padded, never circularly wrapped.
    bands = torch.zeros((1, 1, 1, 1, 5, 3), dtype=torch.complex64)
    bands[..., 2] = 1.0  # d=+1=k_out-k_in: y[k] = x[k-1]
    x = torch.zeros((1, 1, 1, 5), dtype=torch.complex64)
    x[..., 0] = 1.0
    y = apply_banded_multiuser_channel(bands, x)
    assert torch.allclose(y[0, 0, 0], torch.tensor([0, 1, 0, 0, 0], dtype=y.dtype))

    # Adjoint identity <Hx,y> = <x,H^H y>.
    B, Nu, Td, Nr, Nf, P = 2, 2, 3, 4, 9, 2
    h = torch.randn(B, Nu, Td, Nr, Nf, 2 * P + 1, dtype=torch.complex64)
    x = torch.randn(B, Nu, Td, Nf, dtype=torch.complex64)
    y = torch.randn(B, Td, Nr, Nf, dtype=torch.complex64)
    lhs = (apply_banded_multiuser_channel(h, x).conj() * y).sum()
    rhs = (x.conj() * apply_banded_multiuser_channel_adjoint(h, y)).sum()
    adjoint_rel = float((lhs - rhs).abs() / lhs.abs().clamp_min(1e-8))
    assert adjoint_rel < 2e-5, adjoint_rel

    # Matrix-free PCG must agree with the direct banded LMMSE reference.
    tx = torch.randn(B, Nu, Td, Nf, dtype=torch.complex64)
    yrx = apply_banded_multiuser_channel(h, tx)
    noise_var = torch.tensor([0.05, 0.08])
    zd, _ = banded_multiuser_equalize(h, _ri_rx(yrx), noise_var, solver="direct")
    zp, _, info = banded_multiuser_equalize(
        h, _ri_rx(yrx), noise_var, solver="pcg", max_iter=120,
        tol=1e-7, return_info=True,
    )
    direct_rel = float((zp - zd).norm() / zd.norm().clamp_min(1e-8))
    assert direct_rel < 2e-4, (direct_rel, info)

    # W=0 direct banded MMSE is exactly the existing per-subcarrier MMSE.
    hdiag = torch.randn(B, Nu, Td, Nr, Nf, dtype=torch.complex64)
    tx = torch.randn(B, Nu, Td, Nf, dtype=torch.complex64)
    ydiag = torch.einsum("butaf,butf->btaf", hdiag, tx)
    hri = _ri_channel(hdiag)
    zold, _ = linear_multiuser_equalize(hri, _ri_rx(ydiag), noise_var, method="mmse")
    znew, _ = banded_multiuser_equalize(
        diagonal_csi_to_ici_bands(hri, 0), _ri_rx(ydiag), noise_var,
        solver="direct",
    )
    w0_rel = float((znew - zold).norm() / zold.norm().clamp_min(1e-8))
    assert w0_rel < 2e-5, w0_rel

    print("banded MMSE validation: PASS")
    print(f"  adjoint relative error: {adjoint_rel:.3e}")
    print(f"  PCG/direct relative error: {direct_rel:.3e}")
    print(f"  W=0/legacy-MMSE relative error: {w0_rel:.3e}")
    print(f"  PCG iterations: {info['iterations']}")
    print(f"  PCG max residual: {info['relative_residual_max']:.3e}")


if __name__ == "__main__":
    main()
