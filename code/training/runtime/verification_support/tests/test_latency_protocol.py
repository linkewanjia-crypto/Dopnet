"""Small mathematical/protocol tests; no checkpoint, channel or timing run."""
from pathlib import Path
import copy
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verification_extension.latency import (compare_outputs, recommend_devices,
                                           summarize_times, _physics_identity,
                                           _validate_arrays, _validate_policy_report,
                                           implementation_hashes)


class LatencyProtocolTests(unittest.TestCase):
    def test_quantiles_and_throughput_do_not_confuse_frame_and_batch(self):
        stats = summarize_times([1.0, 2.0, 3.0, 4.0], 8)
        self.assertEqual(stats["latency_ms_p50"], 2.5)
        self.assertAlmostEqual(stats["latency_ms_p95"], 3.85)
        self.assertEqual(stats["frames_per_second"], 3200.0)
        self.assertEqual(stats["amortized_ms_per_frame"], 0.3125)

    def test_physics_identity_does_not_depend_on_emission_or_simulator_batch(self):
        a = {"system": {"n_ant": 8, "n_subcarriers": 128, "threegpp_link_batch_size": 8},
             "data_pipeline": {"emit_pilot_only_fast_time_add": True}}
        b = {"system": {"n_ant": 8, "n_subcarriers": 128, "threegpp_link_batch_size": 2},
             "data_pipeline": {"emit_pilot_only_fast_time_add": False}}
        self.assertEqual(_physics_identity(a), _physics_identity(b))
        b["system"]["n_ant"] = 4
        self.assertNotEqual(_physics_identity(a), _physics_identity(b))

    def test_failed_or_missing_seed_cannot_win_device_policy(self):
        rows = []
        for seed in (42, 43, 44):
            for device in ("cpu", "cuda:0"):
                rows.append({"method": "dopnet", "batch_size": 1,
                             "receiver_mode": "common_diagonal", "source_name": "calibration",
                             "seed": seed, "status": "measured", "device": device,
                             "numerical_equivalence": {"passed": not (device == "cuda:0" and seed == 44)},
                             "timing": {"host_origin_total": {"latency_ms_p50": 10.0 if device == "cpu" else 1.0}}})
        choices = recommend_devices(rows)
        self.assertEqual(choices[0]["preferred_device"], "cpu")
        self.assertEqual(choices[0]["workload"], "single-frame latency")
        rows[-1]["numerical_equivalence"]["passed"] = True
        self.assertEqual(recommend_devices(rows)[0]["preferred_device"], "cuda:0")
        for row in rows:
            if row["seed"] == 44:
                row["status"] = "oom"
                row["numerical_equivalence"]["passed"] = False
        self.assertEqual(recommend_devices(rows), [])
        self.assertEqual(recommend_devices(rows, expected_seeds=(42, 43, 44)), [])

    def test_per_frame_agreement_reports_decoded_bits_and_nonfinite_outputs(self):
        ref = {"cfr": np.ones((2, 1, 2), np.float32),
               "symbols": np.ones((2, 1, 2), np.complex64),
               "bits": np.zeros((2, 1, 2, 2), np.uint8)}
        out = {key: value.copy() for key, value in ref.items()}
        self.assertTrue(compare_outputs(ref, out)["passed"])
        out["bits"][1, 0, 0, 0] = 1
        report = compare_outputs(ref, out)
        self.assertTrue(report["per_frame"][0]["passed"])
        self.assertFalse(report["per_frame"][1]["passed"])

    def test_cpu_thread_choice_uses_complete_seed_options_before_device_choice(self):
        rows = []
        for seed in (42, 43, 44):
            for device, threads, latency in (("cpu", 1, 9.0), ("cpu", 4, 2.0),
                                             ("cpu", 8, 6.0), ("cuda:0", 8, 3.0)):
                rows.append({"method": "dopnet", "batch_size": 1,
                             "receiver_mode": "common_diagonal", "source_name": "calibration",
                             "seed": seed, "status": "measured", "device": device,
                             "cpu_threads": threads, "numerical_equivalence": {"passed": True},
                             "timing": {"host_origin_total": {"latency_ms_p50": latency}}})
        choice = recommend_devices(rows, expected_seeds=(42, 43, 44))[0]
        self.assertEqual(choice["preferred_device"], "cpu")
        self.assertEqual(choice["cpu_threads"], 4)
        self.assertEqual(choice["n_model_seeds"], 3)
        self.assertEqual(choice["best_cpu_threads_by_device"], {"cpu": 4, "cuda:0": 8})
        self.assertEqual(len(choice["qualified_backend_options"]), 4)
        for row in rows:
            if row["device"] == "cpu" and row["cpu_threads"] == 4 and row["seed"] == 44:
                row["numerical_equivalence"]["passed"] = False
        choice = recommend_devices(rows, expected_seeds=(42, 43, 44))[0]
        self.assertEqual(choice["preferred_device"], "cuda:0")
        self.assertEqual(choice["best_cpu_threads_by_device"]["cpu"], 8)
        self.assertEqual(len(choice["qualified_backend_options"]), 3)

    def test_automatic_policy_requires_exact_report_and_implementation_binding(self):
        hashes = {"verification_extension/latency.py": "abc", "models/receiver.py": "def"}
        choices = [{"method": "dopnet", "cpu_threads": 4, "preferred_device": "cpu"}]
        report = {"complete": True, "protocol": {"experiment_signature": "experiment",
                  "input_export_signature": "export", "implementation_sha256": hashes},
                  "device_recommendations": choices}
        policy = {"experiment_signature": "experiment", "source_export_signature": "export",
                  "recommendations": choices}
        _validate_policy_report(policy, report, hashes)
        for field in ("experiment_signature", "source_export_signature", "recommendations"):
            changed = copy.deepcopy(policy)
            changed[field] = "changed"
            with self.assertRaises(RuntimeError):
                _validate_policy_report(changed, report, hashes)
        with self.assertRaises(RuntimeError):
            _validate_policy_report(policy, report, {**hashes, "models/receiver.py": "changed"})
        historical = copy.deepcopy(report)
        del historical["protocol"]["implementation_sha256"]
        with self.assertRaisesRegex(RuntimeError, "remeasure"):
            _validate_policy_report(policy, historical, hashes)
        inventory = implementation_hashes()
        self.assertIn("verification_extension/latency.py", inventory)
        self.assertIn("models/receiver.py", inventory)
        self.assertTrue(all(len(value) == 64 for value in inventory.values()))

    def test_uniform_comb_validation_catches_wrong_declared_density(self):
        pilot = np.zeros((1, 2, 2, 8), dtype=np.complex64)
        pilot[:, 0, :, 0::4] = 1
        pilot[:, 1, :, 1::4] = 1
        arrays = {"x_pilot": pilot, "y_pilot": np.zeros((1, 2, 2, 8), np.complex64),
                  "rx_data": np.zeros((1, 1, 2, 8), np.complex64),
                  "noise_var": np.ones((1,), np.float32),
                  "pilot_visible": np.array([0, 2]), "data_visible": np.array([1]),
                  "condition": {"pilot_spacing": 4}}
        geometry = {"n_users": 2, "n_ant": 2, "n_subcarriers": 8, "frame_num_symbols": 3}
        _validate_arrays(arrays, geometry)
        arrays["condition"]["pilot_spacing"] = 2
        with self.assertRaises(ValueError):
            _validate_arrays(arrays, geometry)

    def test_sionna_phase_compensation_toy_round_trip(self):
        # An 8-point artificial Fourier example verifies the sign of the
        # documented l_min phase compensation; this is not measurement data.
        n, cp, l_min = 8, 2, -2
        grid = np.array([1+2j, 2-1j, 3+0j, 0+1j, 4-2j, 1+0j, 2+2j, -1j])
        time = np.fft.ifft(grid, norm="ortho")
        shifted = np.roll(time, -l_min)
        with_cp = np.concatenate([shifted[-cp:], shifted])
        unshifted_fft = np.fft.fft(with_cp[cp:], norm="ortho")
        corrected = unshifted_fft * np.exp(-2j*np.pi*l_min*np.arange(n)/n)
        np.testing.assert_allclose(corrected, grid, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
