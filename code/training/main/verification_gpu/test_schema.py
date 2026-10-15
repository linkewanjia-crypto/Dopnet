"""Tiny metadata-only tests; never run a receiver or generate observations."""
import unittest
from verification_gpu.schema import guarded_function


def toy(cell, original):
    old_bits = original['bits'].copy()
    return old_bits


def changed(cell, original):
    other = original['bits'].copy()
    return other


class SchemaTests(unittest.TestCase):
    def test_nmse_only_does_not_access_bits(self):
        self.assertIsNone(guarded_function(toy)({'ber': False}, {}))

    def test_ber_keeps_original_copy(self):
        values = [16, 16]
        result = guarded_function(toy)({'ber': True}, {'bits': values})
        self.assertEqual(result, values)
        self.assertIsNot(result, values)

    def test_ber_missing_bits_is_still_an_error(self):
        with self.assertRaises(KeyError):
            guarded_function(toy)({'ber': True}, {})

    def test_unknown_evaluator_is_rejected(self):
        with self.assertRaises(RuntimeError):
            guarded_function(changed)
