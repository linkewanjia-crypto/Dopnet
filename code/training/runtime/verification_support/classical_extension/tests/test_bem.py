"""Small deterministic mathematical fixtures; no waveform dataset generation."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dopnet_classical.bem import (BEMSettings, JointBEMOperator, METHODS,
    SolverConvergenceError, WaveformGeometry, diagonal_mmse, fit_joint_bem,
    native_mmse, pcg)
from dopnet_classical.metrics import (bits_per_symbol, demodulate_to_labels,
    frame_bit_counts, labels_to_symbols, nmse_per_frame)


def geometry():
    return WaveformGeometry(nfft=64, n_active=16, cp_wave_samples=16,
        subcarrier_spacing_hz=1000.0, num_symbols=5, pilot_visible=(0, 2, 4),
        data_visible=(1, 3), active_signed_bins=tuple(range(-8, 8)))


class BEMTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(473)
        self.g = geometry()

    def operator(self, method=METHODS[0], q=3):
        return JointBEMOperator(self.g, BEMSettings(method=method, q=q,
            delay_start=-1, delay_stop=4, training_doppler_envelope_hz=150.0,
            fit_max_iter=200, fit_tolerance=1e-10,
            detector_max_iter=200, detector_tolerance=1e-10), n_users=2)

    def complex_array(self, shape):
        return self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape)

    def test_zero_doppler_static_recovery_and_average_cfr(self):
        settings = BEMSettings(q=1, delay_start=0, delay_stop=4,
            relative_ridge=0, fit_max_iter=100, fit_tolerance=1e-10)
        op = JointBEMOperator(self.g, settings, n_users=2)
        c = self.complex_array((2, 1, 4, 3)) / 5
        x = np.zeros((2, 3, 16), dtype=np.complex128)
        x[0, :, ::2] = np.exp(1j * self.rng.uniform(-3, 3, (3, 8)))
        x[1, :, 1::2] = np.exp(1j * self.rng.uniform(-3, 3, (3, 8)))
        y = op.forward_pilots(c, x)
        estimated, info = fit_joint_bem(op, x, y, 0.0)
        np.testing.assert_allclose(estimated, c, atol=1e-9, rtol=1e-9)
        expected = np.einsum('ula,kl->uak', c[:, 0], op.delay_phase)
        np.testing.assert_allclose(op.diagonal_cfr(estimated)[:, 0], expected, atol=1e-9)
        np.testing.assert_allclose(op.diagonal_cfr(estimated)[:, 1], expected, atol=1e-9)
        self.assertLessEqual(info['relative_residual_max'], settings.fit_tolerance)

    def test_forward_adjoint_inner_product(self):
        op = self.operator()
        x = self.complex_array((2, 3, 16))
        c = self.complex_array((op.n_coefficients, 3))
        y = self.complex_array((3 * 16, 3))
        d = op.design_matrix(x)
        lhs, rhs = np.vdot(d @ c, y), np.vdot(c, d.conj().T @ y)
        np.testing.assert_allclose(lhs, rhs, atol=1e-11, rtol=1e-11)
        forward = op.forward_pilots(c.reshape(*op.coefficient_shape, 3), x)
        np.testing.assert_allclose(forward.transpose(0, 2, 1).reshape(48, 3), d @ c,
                                   atol=1e-11, rtol=1e-11)

    def test_ici_operator_matches_independent_full_fft_for_both_bases(self):
        for method in METHODS:
            op = self.operator(method)
            c = self.complex_array((2, 3, 5, 3)) / 20
            x = self.complex_array((2, 16))
            signed = np.asarray(self.g.active_signed_bins)
            for symbol in (0, 1, 4):
                expected_td = np.zeros((3, self.g.nfft), dtype=np.complex128)
                for u in range(2):
                    full = np.zeros(self.g.nfft, dtype=np.complex128)
                    full[signed % self.g.nfft] = x[u]
                    xt = np.fft.ifft(full, norm='ortho')
                    for qi in range(3):
                        for li, ell in enumerate(op.delays):
                            # Integer delay here is ell*(64/16) physical samples.
                            shifted = np.roll(xt, int(ell * self.g.nfft / self.g.n_active))
                            expected_td += c[u, qi, li, :, None] * op.basis[symbol, qi][None] * shifted[None]
                expected = np.fft.fft(expected_td, norm='ortho')[:, signed % self.g.nfft]
                matrix = op.channel_matrix(c, symbol)
                actual = np.einsum('anuk,uk->an', matrix, x)
                np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-11)
                diag = np.stack([np.diagonal(matrix[:, :, u, :], axis1=1, axis2=2) for u in range(2)])
                np.testing.assert_allclose(op.diagonal_cfr(c, (symbol,))[:, 0], diag,
                                           atol=2e-11, rtol=2e-11)

    def test_native_detector_matches_direct_joint_mmse(self):
        op = self.operator()
        c = self.complex_array((2, 3, 5, 3)) / 10
        y = self.complex_array((2, 3, 16))
        nv = 0.07
        actual, info = native_mmse(op, c, y, nv)
        for di, symbol in enumerate(self.g.data_visible):
            h = op.channel_matrix(c, symbol).reshape(48, 32)
            direct = np.linalg.solve(h.conj().T @ h + nv * np.eye(32),
                                     h.conj().T @ y[di].reshape(-1))
            np.testing.assert_allclose(actual[:, di].reshape(-1), direct, atol=2e-8, rtol=2e-8)
        self.assertLessEqual(info['relative_residual_max'], op.settings.detector_tolerance)

    def test_static_common_and_native_detectors_and_known_bits(self):
        op = self.operator(q=1)
        c = self.complex_array((2, 1, 5, 3)) / 4
        labels = self.rng.integers(0, 16, (2, 2, 16))
        symbols = labels_to_symbols(labels, '16qam')
        y = np.stack([np.einsum('anuk,uk->an', op.channel_matrix(c, m), symbols[:, di])
                      for di, m in enumerate(self.g.data_visible)])
        native, _ = native_mmse(op, c, y, 1e-6)
        common = diagonal_mmse(op.diagonal_cfr(c), y, 1e-6)
        np.testing.assert_allclose(native, common, atol=2e-8, rtol=2e-8)
        errors, bits = frame_bit_counts(common[None], labels[None], '16qam')
        self.assertEqual(int(errors[0]), 0)
        self.assertEqual(int(bits[0]), labels.size * 4)

    def test_gray_constellations_and_per_frame_nmse(self):
        for modulation in ('bpsk', 'qpsk', '16qam', '64qam'):
            labels = np.arange(1 << bits_per_symbol(modulation))
            symbols = labels_to_symbols(labels, modulation)
            np.testing.assert_array_equal(demodulate_to_labels(symbols, modulation), labels)
            self.assertAlmostEqual(float(np.mean(np.abs(symbols) ** 2)), 1.0, places=6)
        target = np.ones((2, 2, 2, 3, 16), dtype=np.complex128)
        np.testing.assert_allclose(nmse_per_frame(2 * target, target), [1, 1])
        np.testing.assert_allclose(nmse_per_frame(target, target), [0, 0])

    def test_nonconvergent_solver_is_rejected(self):
        a = self.complex_array((10, 10))
        normal = a.conj().T @ a + 0.01 * np.eye(10)
        rhs = self.complex_array((10, 2))
        with self.assertRaises(SolverConvergenceError):
            pcg(lambda x: normal @ x, rhs, np.diag(normal).real, max_iter=1, tolerance=1e-12)


if __name__ == '__main__':
    unittest.main()
