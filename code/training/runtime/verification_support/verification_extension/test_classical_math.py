"""Small algebra tests only: no actual frames, model inference or simulation."""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np
import torch

try:
    from .classical import (BEMSettings, BEM_METHODS, PilotCovarianceLMMSE,
        SolverConvergenceError, TorchJointBEM, WaveformGeometry,
        diagonal_mmse, frame_bit_counts, nmse_per_frame, strict_cholesky_solve,
        split_calibration_groups)
except ImportError:
    from classical import (BEMSettings, BEM_METHODS, PilotCovarianceLMMSE,
        SolverConvergenceError, TorchJointBEM, WaveformGeometry,
        diagonal_mmse, frame_bit_counts, nmse_per_frame, strict_cholesky_solve,
        split_calibration_groups)


def geometry():
    return WaveformGeometry(nfft=16, n_active=8, cp_wave_samples=2,
        subcarrier_spacing_hz=100., num_symbols=5, pilot_visible=(0, 2, 4),
        data_visible=(1, 3), active_signed_bins=tuple(range(-4, 4)))


def pilots():
    x = torch.zeros((2, 3, 8), dtype=torch.complex128)
    for p in range(3):
        x[0, p, p % 2::2] = 1 + 1j
        x[1, p, 1 - p % 2::2] = 1 - 1j
    return x


class ClassicalAlgebra(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_dual_matches_regularized_primal(self):
        for method in BEM_METHODS:
            s = BEMSettings(method=method, q=3, delay_start=-2,
                delay_stop=3, training_doppler_envelope_hz=80.)
            op = TorchJointBEM(geometry(), s)
            d = op.design_matrix(pilots())
            y = torch.randn((3, 2, 8), dtype=torch.complex128)
            c, info = op.fit(pilots(), y, .07, relative_ridge=.1)
            lam = info['ridge_lambda']
            yflat = y.permute(0, 2, 1).reshape(24, 2)
            expected = torch.linalg.solve(d.mH @ d + lam * torch.eye(d.shape[1], dtype=d.dtype), d.mH @ yflat)
            torch.testing.assert_close(c.reshape(-1, 2), expected, rtol=1e-10, atol=1e-11)
            self.assertEqual(info['normal_space'], 'observation')
            self.assertLessEqual(info['recovered_normal_relative_residual_max'], 1e-6)

    def test_native_detector_equals_dense_normal_solve_and_chunking(self):
        op = TorchJointBEM(geometry(), BEMSettings(q=3, delay_start=-1,
            delay_stop=2, training_doppler_envelope_hz=80.))
        c = torch.randn((*op.coefficient_shape, 2), dtype=torch.complex128)
        y = torch.randn((2, 2, 8), dtype=torch.complex128)
        a, info = op.native_mmse(c, y, .13, symbol_batch_size=1)
        b, _ = op.native_mmse(c, y, .13, symbol_batch_size=2)
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        for ti, t in enumerate(op.geometry.data_visible):
            h = op.channel_matrix(c, t).reshape(16, 16)
            truth = torch.linalg.solve(h.mH @ h + .13 * torch.eye(16, dtype=h.dtype), h.mH @ y[ti].reshape(-1))
            torch.testing.assert_close(a[:, ti].reshape(-1), truth, rtol=1e-10, atol=1e-11)
        self.assertLessEqual(info['relative_residual_max'], 1e-6)

    def test_occupied_edges_use_physical_fft_instead_of_active_band_wrap(self):
        g = geometry()
        settings = BEMSettings(q=3, delay_start=-1, delay_stop=2,
                               training_doppler_envelope_hz=80.)
        op = TorchJointBEM(g, settings)
        c = torch.randn((*op.coefficient_shape, 2), dtype=torch.complex128)
        frequency = op.coefficients_to_frequency(c)
        x = torch.randn((2, 8), dtype=torch.complex128)
        indices = torch.tensor(g.active_signed_bins) % g.nfft
        time = (2 * g.symbol_duration_s +
                (torch.arange(g.nfft, dtype=torch.float64) - (g.nfft - 1) / 2) / g.sample_rate_hz)
        mode_hz = torch.arange(-1, 2, dtype=torch.float64) / (g.num_symbols * g.symbol_duration_s)
        basis = torch.exp(2j * math.pi * mode_hz[:, None] * time[None])
        expected = torch.zeros((2, 8), dtype=torch.complex128)
        for user in range(2):
            for q in range(3):
                full = torch.zeros((2, 16), dtype=torch.complex128)
                full[:, indices] = frequency[user, q] * x[user]
                received = torch.fft.fft(torch.fft.ifft(full, dim=-1) * basis[q], dim=-1)
                expected += received[:, indices]
        actual = torch.einsum('anuk,uk->an', op.channel_matrix(c, 2), x)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)

    def test_common_detector_joint_two_users(self):
        h = torch.randn((2, 2, 3, 8), dtype=torch.complex128)
        y = torch.randn((2, 3, 8), dtype=torch.complex128)
        output = diagonal_mmse(h, y, .1)
        for t in range(2):
            for k in range(8):
                local = h[:, t, :, k].T
                expected = torch.linalg.solve(local.mH @ local + .1 * torch.eye(2, dtype=h.dtype), local.mH @ y[t, :, k])
                torch.testing.assert_close(output[:, t, k], expected)

    def test_covariance_power_has_no_fft_length_multiplier(self):
        g = geometry()
        amplitudes = np.asarray([1., 2., 1j, -1j], dtype=np.complex128)
        h = np.ones((4, 2, 2, 2, 8), dtype=np.complex128) * amplitudes[:, None, None, None, None]
        batch = dict(target_h=h, x_pilot=np.broadcast_to(pilots().numpy(), (4, 2, 3, 8)),
                     waveform_fft_size=np.asarray(16), cp_wave_samples=np.asarray(2),
                     subcarrier_spacing_hz=np.asarray(100.), pilot_visible=np.asarray([0, 2, 4]),
                     data_visible=np.asarray([1, 3]))
        manifest = {'geometry': {'frame_num_symbols': 5, 'active_signed_bins': list(range(-4, 4))}}
        estimator = PilotCovarianceLMMSE.from_calibration([batch], manifest)
        torch.testing.assert_close(estimator.rf.diagonal(dim1=-2, dim2=-1).real,
            torch.full((2, 8), 1.75, dtype=torch.float64), rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(estimator.rt.diagonal(dim1=-2, dim2=-1).real.mean(-1),
            torch.ones(2, dtype=torch.float64), rtol=1e-10, atol=1e-10)
        self.assertEqual(estimator.calibration_info['calibration_frames'], 4)

    def test_statistical_estimate_uses_actual_random_pilot_locations(self):
        g = geometry()
        rf = torch.stack([torch.eye(8, dtype=torch.complex128)] * 2)
        rt = torch.ones((2, 5, 5), dtype=torch.complex128)
        estimator = PilotCovarianceLMMSE(rf, rt, g)
        x = pilots()
        y = torch.randn((3, 2, 8), dtype=torch.complex128)
        h, info = estimator.estimate(x, y, .17)
        for user in range(2):
            loc = torch.nonzero(x[user].abs() > 0, as_tuple=False)
            pi, ki = loc[:, 0], loc[:, 1]
            obs = y[pi, :, ki] / x[user, pi, ki, None]
            rpp = rf[user][ki[:, None], ki[None]]
            solved = torch.linalg.solve(rpp + torch.diag(.17 / x[user, pi, ki].abs().square()).to(torch.complex128), obs)
            expected = rf[user][:, ki] @ solved
            torch.testing.assert_close(h[user, 0].T, expected)
            torch.testing.assert_close(h[user, 1].T, expected)
        self.assertEqual(info['pilot_re_counts_per_user'], [12, 12])

    def test_constellation_counts_agree_with_original_gray_mapping(self):
        for modulation, bps in [('bpsk', 1), ('qpsk', 2), ('16qam', 4), ('64qam', 6)]:
            labels = torch.arange(1 << bps).reshape(1, 1, 1, -1)
            if bps == 1:
                symbols = (2 * labels - 1).to(torch.complex128)
            else:
                side = 1 << (bps // 2)
                gi, gq = labels >> (bps // 2), labels & (side - 1)
                def binary(a):
                    v = a.clone()
                    for shift in (1, 2, 4):
                        v ^= v >> shift
                    return v
                symbols = ((2 * binary(gi) - (side - 1)).double() +
                           1j * (2 * binary(gq) - (side - 1)).double()) / np.sqrt((2 / 3) * ((1 << bps) - 1))
                symbols = symbols.to(torch.complex64).to(torch.complex128)
            errors, bits = frame_bit_counts(symbols, labels, modulation)
            self.assertEqual(int(errors[0]), 0)
            self.assertEqual(int(bits[0]), (1 << bps) * bps)

    def test_inadequate_precision_cannot_be_silently_scored(self):
        a = torch.tensor([[1., 1.], [1., 1.]], dtype=torch.complex128)
        with self.assertRaises(SolverConvergenceError):
            strict_cholesky_solve(a, torch.ones((2, 1), dtype=torch.complex128))
        b = torch.randn((10, 10), dtype=torch.complex64)
        matrix = b.mH @ b + .003 * torch.eye(10, dtype=torch.complex64)
        with self.assertRaises(SolverConvergenceError):
            strict_cholesky_solve(matrix, torch.randn((10, 1), dtype=torch.complex64), tolerance=1e-12)

    def test_calibration_split_keeps_complete_source_chunks_and_cp_groups(self):
        manifest = {'geometry': {'frame_num_symbols': 5, 'active_signed_bins': list(range(-4, 4))}}
        chunks = []
        for cp in (2, 4):
            for chunk in range(8):
                chunks.append(dict(x_pilot=np.zeros((4, 2, 3, 8)), waveform_fft_size=np.asarray(16),
                    cp_wave_samples=np.asarray(cp), subcarrier_spacing_hz=np.asarray(100.),
                    pilot_visible=np.asarray([0, 2, 4]), data_visible=np.asarray([1, 3]),
                    source_identity=f'{cp}:{chunk}'))
        groups = split_calibration_groups(chunks, manifest)
        self.assertEqual(len(groups), 2)
        for group in groups.values():
            self.assertEqual(group['fit_frames'], 24)
            self.assertEqual(group['holdout_frames'], 8)
            self.assertEqual(len(group['fit_batches']), 6)
            self.assertEqual(len(group['holdout_batches']), 2)
            fit = {a['source_identity'] for a in group['fit_batches']}
            held = {a['source_identity'] for a in group['holdout_batches']}
            self.assertFalse(fit & held)
        with self.assertRaises(ValueError):
            split_calibration_groups(chunks[:3], manifest)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA server-only parity test')
    def test_same_math_cpu_cuda(self):
        g, x = geometry(), pilots()
        s = BEMSettings(q=3, delay_start=-2, delay_stop=3, training_doppler_envelope_hz=80.)
        y = torch.randn((3, 2, 8), dtype=torch.complex128)
        a, _ = TorchJointBEM(g, s).fit(x, y, .09)
        b, _ = TorchJointBEM(g, s, device='cuda').fit(x, y, .09)
        torch.testing.assert_close(a, b.cpu(), rtol=1e-9, atol=1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
