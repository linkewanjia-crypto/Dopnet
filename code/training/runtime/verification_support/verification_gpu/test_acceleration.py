"""Small algebra checks for public-only caching, without actual data/models."""
from dataclasses import replace
import unittest

import torch

from verification_extension.classical import BEMSettings, BEM_METHODS, frame_bit_counts
from verification_extension.test_classical_math import geometry, pilots
from verification_gpu.accelerate import (CachedJointBEM, OriginalJointBEM,
                                        configure, statistics)


class AccelerationAlgebra(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(1)
        configure(8)

    def test_cached_public_blocks_equal_original_and_do_not_alias(self):
        for method in BEM_METHODS:
            settings = BEMSettings(method=method, q=3, delay_start=-1,
                                   delay_stop=2, training_doppler_envelope_hz=80.)
            a, b = OriginalJointBEM(geometry(), settings), CachedJointBEM(geometry(), settings)
            x = pilots()
            first = b.design_matrix(x)
            torch.testing.assert_close(first, a.design_matrix(x), rtol=0, atol=0)
            first.zero_()
            torch.testing.assert_close(b.design_matrix(x), a.design_matrix(x), rtol=0, atol=0)
            # Recombine occasions seen in separate full public frames.
            other = x.roll(1, dims=-1)
            b.design_matrix(other)
            combined = x.clone(); combined[:, 1] = other[:, 1]
            torch.testing.assert_close(b.design_matrix(combined), a.design_matrix(combined), rtol=0, atol=0)
            self.assertGreater(statistics().get('design_hit_frames', 0), 0)

    def test_all_ridges_equal_original_with_changed_received_pilots(self):
        for method in BEM_METHODS:
            settings = BEMSettings(method=method, q=3, delay_start=-2,
                                   delay_stop=3, training_doppler_envelope_hz=80.)
            a, b = OriginalJointBEM(geometry(), settings), CachedJointBEM(geometry(), settings)
            x, y = pilots(), torch.randn((3, 2, 8), dtype=torch.complex128)
            design = b.design_matrix(x)
            for ridge in (.01, .1, 1., 10., 100.):
                expected, ei = a.fit(x, y, .07, design=design, relative_ridge=ridge)
                actual, ai = b.fit(x, y, .07, design=design, relative_ridge=ridge)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(ai, ei)
            y = y + .1j
            expected, _ = a.fit(x, y, .3, design=design)
            actual, _ = b.fit(x, y, .3, design=design)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertGreaterEqual(statistics()['ridge_gram_hits'], 4)

    def test_mutated_design_invalidates_gram_and_host_cache_is_bounded(self):
        b = CachedJointBEM(geometry(), BEMSettings(q=3, delay_start=-1,
                                delay_stop=2, training_doppler_envelope_hz=80.))
        x, y = pilots(), torch.randn((3, 2, 8), dtype=torch.complex128)
        d = b.design_matrix(x)
        b.fit(x, y, .1, design=d)
        misses = statistics()['ridge_gram_misses']
        d.add_(.01)
        b.fit(x, y, .1, design=d)
        self.assertEqual(statistics()['ridge_gram_misses'], misses + 1)
        self.assertLessEqual(statistics()['host_cache_bytes'], statistics()['host_cache_limit_bytes'])

    def test_inference_tensors_reuse_gram_but_detect_mutation(self):
        with torch.inference_mode():
            b = CachedJointBEM(geometry(), BEMSettings(q=3, delay_start=-1,
                                    delay_stop=2, training_doppler_envelope_hz=80.))
            x, y = pilots(), torch.randn((3, 2, 8), dtype=torch.complex128)
            d = b.design_matrix(x)
            b.fit(x, y, .1, design=d)
            b.fit(x, y, .1, design=d, relative_ridge=2.)
            self.assertEqual(statistics()['ridge_gram_hits'], 1)
            d.add_(.01)
            b.fit(x, y, .1, design=d)
            self.assertEqual(statistics()['ridge_gram_misses'], 2)

    def test_native_chunk_policy_preserves_symbols_and_bits(self):
        settings = BEMSettings(q=3, delay_start=-1, delay_stop=2,
                               training_doppler_envelope_hz=80.)
        a, b = OriginalJointBEM(geometry(), settings), CachedJointBEM(geometry(), settings)
        configure(8, {'cpu/' + settings.method: 2})
        coeff = torch.randn((*a.coefficient_shape, 2), dtype=torch.complex128)
        y = torch.randn((2, 2, 8), dtype=torch.complex128)
        expected, _ = a.native_mmse(coeff, y, .13, symbol_batch_size=1)
        actual, _ = b.native_mmse(coeff, y, .13, symbol_batch_size=1)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        labels = torch.randint(0, 16, (1, 2, 2, 8))
        for x1, x2 in zip(frame_bit_counts(actual[None], labels, '16qam'),
                          frame_bit_counts(expected[None], labels, '16qam')):
            self.assertTrue(torch.equal(x1, x2))


if __name__ == '__main__':
    unittest.main()
