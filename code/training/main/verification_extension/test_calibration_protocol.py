"""Pure NumPy metadata/statistics toys; no receiver/data-generation run."""
from pathlib import Path
import copy
import json
import math
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verification_extension.calibration import (choose_backend_rows,
    compare_classical_outputs, _validated_complete_candidate,
    _calibration_identity, _statistical_identity, _qualified_cpu_fallback)
from verification_extension.protocol import digest


def toy_chunk(count, seed=810000001, role="calibration"):
    return {"x_pilot": np.ones((count, 1, 1, 2), np.complex64),
            "y_pilot": np.ones((count, 1, 1, 2), np.complex64),
            "target_h": np.ones((count, 1, 1, 1, 2), np.complex64),
            "rx_data": np.ones((count, 1, 1, 2), np.complex64),
            "tx_label": np.zeros((count, 1, 1, 2), np.int64),
            "noise_var": np.full(count, .1, np.float32),
            "pilot_visible": np.array([0]), "data_visible": np.array([1]),
            "cp_wave_samples": np.asarray(2), "waveform_fft_size": np.asarray(8),
            "subcarrier_spacing_hz": np.asarray(100),
            "metadata": np.asarray(json.dumps({"role": role, "calibration_seed": seed}))}


class CalibrationProtocolTests(unittest.TestCase):
    def test_oom_fallback_uses_signed_qualified_cpu_without_changing_precision(self):
        policy = {"qualified_backends": [
            {"method": "toy", "device": "cpu", "cpu_threads": 1, "latency_ms_p50": 9},
            {"method": "toy", "device": "cpu", "cpu_threads": 4, "latency_ms_p50": 4},
            {"method": "toy", "device": "cuda:0", "cpu_threads": 1, "latency_ms_p50": 1}]}
        policy["signature"] = digest(policy)
        choice = _qualified_cpu_fallback(policy, "toy", ".")
        self.assertEqual(choice, {"preferred_device": "cpu", "cpu_threads": 4, "precision": "complex128"})
        policy["qualified_backends"][1]["cpu_threads"] = 8
        with self.assertRaises(RuntimeError):
            _qualified_cpu_fallback(policy, "toy", ".")

    def test_backend_choice_cannot_use_failed_or_numerically_unequal_candidate(self):
        rows = []
        for device, threads, speed, passed, status in [("cpu", 1, 10., True, "measured"),
            ("cpu", 4, 4., True, "measured"), ("cuda:0", 1, 1., False, "measured"),
            ("cuda:1", 1, .1, False, "failed")]:
            rows.append({"method": "toy", "device": device, "cpu_threads": threads,
                         "latency_ms_p50": speed, "frames_per_second": 1000 / speed,
                         "numerical_equivalence": {"passed": passed}, "status": status})
        choice = choose_backend_rows(rows, ["toy"])[0]
        self.assertEqual(choice["preferred_device"], "cpu")
        self.assertEqual(choice["cpu_threads"], 4)
        rows[2]["numerical_equivalence"]["passed"] = True
        self.assertEqual(choose_backend_rows(rows, ["toy"])[0]["preferred_device"], "cuda:0")

    def test_partial_candidate_cannot_be_selected(self):
        contract = {"code": "toy"}
        record = {"status": "complete", "contract": contract, "frames": 3,
                  "completed_frames": 3, "frame_nmse": [.1, .2, .3],
                  "mean_linear_nmse": .2, "solver_residual_max": 1e-9}
        _validated_complete_candidate(record, contract, 3)
        for field, value in [("completed_frames", 2), ("mean_linear_nmse", .1),
                             ("solver_residual_max", math.nan), ("status", "failed")]:
            altered = {**record, field: value}
            with self.assertRaises(RuntimeError):
                _validated_complete_candidate(altered, contract, 3)

    def test_diagnostics_gate_cfr_native_symbols_and_exact_ber_counts(self):
        ref = {"cfr": np.ones((1, 2), np.complex128),
               "common_symbols": np.ones((1, 2), np.complex128),
               "native_symbols": np.ones((1, 2), np.complex128),
               "scores": np.array([.1, 1, 8, 2, 8])}
        out = {key: value.copy() for key, value in ref.items()}
        self.assertTrue(compare_classical_outputs(ref, out)["passed"])
        out["scores"][3] += 1
        self.assertFalse(compare_classical_outputs(ref, out)["passed"])
        out["scores"] = ref["scores"].copy()
        out["cfr"][0, 0] += .1
        self.assertFalse(compare_classical_outputs(ref, out)["passed"])

    def test_prior_extra_frames_do_not_change_the_original_bem128_fingerprint(self):
        base = [toy_chunk(64), toy_chunk(64)]
        manifest = {"calibration_seed": 810000001, "calibration_samples": 128,
                    "export_signature": "toy", "statistical_extra_seed": 810000002,
                    "statistical_extra_count": 16}
        identity = _calibration_identity(base, manifest)
        extended = base + [toy_chunk(16, 810000002, "statistical_calibration")]
        stat = _statistical_identity(extended, manifest, identity)
        self.assertEqual(stat["frames"], 144)
        self.assertEqual(stat["seed_frame_counts"], {"810000001": 128, "810000002": 16})
        changed = copy.deepcopy(extended)
        changed[0]["target_h"][0, 0, 0, 0, 0] += .1
        with self.assertRaises(ValueError):
            _statistical_identity(changed, manifest, identity)

    def test_test_truth_is_rejected_before_runtime_import(self):
        bad = toy_chunk(128, 810000001, "test")
        manifest = {"calibration_seed": 810000001, "calibration_samples": 128,
                    "export_signature": "toy"}
        with self.assertRaises(ValueError):
            _calibration_identity([bad], manifest)


if __name__ == "__main__":
    unittest.main()
