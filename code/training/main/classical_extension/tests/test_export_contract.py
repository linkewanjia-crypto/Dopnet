"""Exporter axis/release checks using small synthetic arrays, not radio generation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "export_frozen_observations.py"
SPEC = importlib.util.spec_from_file_location("frozen_export", SCRIPT)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class ArrayTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class ExportContractTests(unittest.TestCase):
    def test_received_and_target_axes_and_wrapper_restore(self):
        b, u, r, n, t = 2, 2, 3, 4, 5
        layout = types.SimpleNamespace(pilot_blocks=(1, 3, 5),
                                       pilot_visible=(0, 2, 4), data_visible=(1, 3))
        frame = (np.arange(b * u * 7 * n).reshape(b, u, 7, n) + 1j).astype(np.complex64)
        received = (np.arange(b * r * 7 * n).reshape(b, r, 7, n) + 2j).astype(np.complex64)
        original_build = lambda: (ArrayTensor(frame), layout)
        original_pilot = lambda y, observed_layout: ("original coarse", "original ls")
        waveform = types.SimpleNamespace(_build_frame=original_build, _pilot_ls_and_interp=original_pilot)
        h = np.arange(b * u * t * r * n).reshape(b, u, t, r, n).astype(np.float32)
        batch = {
            "target_h": ArrayTensor(np.stack((h, -h), axis=2)),
            "rx_data": ArrayTensor(np.zeros((b, 2, 2, r, n), np.float32)),
            "tx_label": ArrayTensor(np.zeros((b, u, 2, n), np.int64)),
            "noise_var": ArrayTensor(np.ones(b, np.float32)),
            "cp_wave_samples": ArrayTensor(np.full(b, 16, np.int64)),
            "waveform_fft_size": ArrayTensor(np.full(b, 32, np.int64)),
        }
        cfg = {"system": {"n_users": u, "n_ant": r, "n_subcarriers": n,
                          "subcarrier_spacing_hz": 7500.0}}
        with EXPORT.CapturePilotObservations(waveform) as capture:
            waveform._build_frame()
            self.assertEqual(waveform._pilot_ls_and_interp(ArrayTensor(received), layout),
                             ("original coarse", "original ls"))
            arrays = capture.arrays(batch, cfg, {"role": "test", "offset": 0})
            np.testing.assert_array_equal(arrays["y_pilot"], received[:, :, [1, 3, 5]].transpose(0, 2, 1, 3))
            np.testing.assert_array_equal(arrays["x_pilot"], frame[:, :, [1, 3, 5]])
            np.testing.assert_array_equal(arrays["target_h"], (h - 1j * h)[:, :, [1, 3]])
            np.testing.assert_array_equal(frame, np.arange(b * u * 7 * n).reshape(b, u, 7, n) + 1j)
            np.testing.assert_array_equal(received, np.arange(b * r * 7 * n).reshape(b, r, 7, n) + 2j)
        self.assertIs(waveform._build_frame, original_build)
        self.assertIs(waveform._pilot_ls_and_interp, original_pilot)

    def test_release_marker_checks_hash_and_contiguous_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = "cells/example/batch_0000.npz"
            record = EXPORT.write_batch(root, relative, {"toy": np.arange(8)}, 0, 8)
            marker = root / "cells/example/COMPLETE.json"
            self.assertFalse(EXPORT.verify_complete(root, marker, "signature", "test", 8))
            EXPORT.atomic_json(marker, {"schema_version": 1, "export_signature": "signature",
                                        "role": "test", "samples": 8, "batches": [record]})
            self.assertTrue(EXPORT.verify_complete(root, marker, "signature", "test", 8))
            with (root / relative).open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(RuntimeError, "corrupt"):
                EXPORT.verify_complete(root, marker, "signature", "test", 8)

    def test_completion_path_cannot_escape_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                EXPORT.safe_child(temporary, "../outside.npz")

    def test_predefined_suite_and_custom_labels(self):
        cells = []
        for channel in EXPORT.CORE_CHANNELS:
            for doppler in (800.0, 1200.0, 1600.0):
                for modulation, snr, study in (("qpsk", 15.0, "nmse"),
                                                ("qpsk", 25.0, "nmse"),
                                                ("16qam", 20.0, "receiver")):
                    cells.append(dict(channel=channel, doppler_hz=doppler,
                                      modulation=modulation, snr_db=snr, cp_len=16,
                                      pilot_spacing=4, delay_spread_ns=300.0,
                                      studies=[study], cell_id=f"cell{len(cells)}"))
        suite, chosen = EXPORT.select_cells(cells, "core", [])
        self.assertEqual((suite, len(chosen)), ("core", 99))
        self.assertEqual(EXPORT.select_cells(cells, "core", ["cell0"])[0], "custom")
        with self.assertRaises(ValueError):
            EXPORT.select_cells(cells, "core", ["cell0", "cell0"])


if __name__ == "__main__":
    unittest.main()
