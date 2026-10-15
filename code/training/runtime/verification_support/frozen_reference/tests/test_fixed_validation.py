"""CPU-only contracts for the archived fixed validation realization."""

import copy
import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_protocol import VARIANTS, make_config
from extract_historical_validation_cache import extract_checked_member
from verify_historical_validation_reconstruction import _select_direct_feature
from fixed_validation import (
    HISTORICAL_CACHE_SHA256, HISTORICAL_IDENTITY_SHA256,
    _expected_cache_hash, assert_historical_identity, historical_tensor_schema,
    identity_sha256, derive_pilot_only_fast_time_add,
    historical_file_metadata, assert_historical_file_metadata,
)


class HistoricalValidationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = yaml.safe_load((ROOT / "configs" / "dopnet.yaml").read_text())

    def test_all_model_variants_share_the_archived_validation_identity(self):
        for variant in VARIANTS:
            for seed in (42, 43, 44):
                with self.subTest(variant=variant, seed=seed):
                    cfg = make_config(self.base, variant, seed)
                    self.assertEqual(identity_sha256(cfg), HISTORICAL_IDENTITY_SHA256)
                    assert_historical_identity(cfg)

    def test_changed_validation_distribution_or_grid_is_rejected(self):
        cfg = make_config(self.base, "dopnet", 42)
        for section, key, value in (
            ("system", "snr_db_min", 6.0),
            ("system", "threegpp_link_batch_size", 2),
            ("model", "add_doppler_bins", 13),
            ("train", "validation_seed", 610000002),
            ("train", "val_samples", 512),
            ("data_pipeline", "microbatch_size", 2),
        ):
            changed = copy.deepcopy(cfg)
            changed[section][key] = value
            with self.subTest(section=section, key=key):
                with self.assertRaises(RuntimeError):
                    assert_historical_identity(changed)

    def test_archived_hash_cannot_be_replaced_by_an_arbitrary_hash(self):
        self.assertEqual(_expected_cache_hash(HISTORICAL_CACHE_SHA256.upper()),
                         HISTORICAL_CACHE_SHA256)
        with self.assertRaises(RuntimeError):
            _expected_cache_hash("0" * 64)
        with self.assertRaises(ValueError):
            _expected_cache_hash("bad")

    def test_schema_is_the_inspected_archived_shape(self):
        schema = historical_tensor_schema(make_config(self.base, "dopnet", 42))
        self.assertEqual(len(schema), 29)
        self.assertEqual(schema["input"][1], (2000, 2, 5, 25, 8, 128))
        self.assertEqual(schema["add_input_slow"][1], (2000, 2, 2, 15, 8, 48))
        self.assertEqual(schema["ls_grid"][1], (2000, 2, 2, 25, 8, 128))

    def test_streaming_extractor_checks_hash_and_refuses_changed_output(self):
        payload = b"small deterministic fixture" * 31
        checksum = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            archive = folder / "source.zip"
            output = folder / "cache.pt"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("cache/cache.pt", payload)
            extract_checked_member(
                archive, "cache/cache.pt", output, len(payload), checksum,
            )
            self.assertEqual(output.read_bytes(), payload)
            # Repeated extraction verifies the existing output and is a no-op.
            extract_checked_member(
                archive, "cache/cache.pt", output, len(payload), checksum,
            )
            output.write_bytes(payload[:-1] + b"!")
            with self.assertRaises(RuntimeError):
                extract_checked_member(
                    archive, "cache/cache.pt", output, len(payload), checksum,
                )

    def test_attested_fast_path_rejects_changed_file_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.pt"
            path.write_bytes(b"unchanged")
            proof = historical_file_metadata(path)
            assert_historical_file_metadata(path, proof)
            path.write_bytes(b"changed payload")
            with self.assertRaises(RuntimeError):
                assert_historical_file_metadata(path, proof)

    def test_pilot_only_derivation_matches_direct_operator_on_small_cpu_fixture(self):
        # Algebra check only: random received pilot spectra, no channel simulation.
        from data.add_geometry import (
            add_delay_bins, delay_guard_bins, delay_window_indices,
        )
        from data.waveform import (
            _build_frame, _pilot_ls_and_interp, _add_backprojection,
        )
        cfg = make_config(self.base, "dopnet", 42)
        b, users, antennas, visible, subcarriers = 2, 2, 8, 25, 128
        mixed_batches = []
        mixed_direct = []
        for spacing in (2, 4, 8):
            gen = torch.Generator(device="cpu").manual_seed(100 + spacing)
            frame, layout, pmask, pidx, pval, _, _ = _build_frame(
                b, users, subcarriers, spacing, "qpsk", "cpu", gen,
                visible, 4,
            )
            received = torch.complex(
                torch.randn(b, antennas, layout.num_blocks, subcarriers,
                            generator=gen),
                torch.randn(b, antennas, layout.num_blocks, subcarriers,
                            generator=gen),
            )
            _, ls_grid = _pilot_ls_and_interp(
                received, layout, pidx, pval, users, antennas,
                subcarriers, spacing, want_ls_grid=True,
            )
            model_input = torch.zeros(
                (b, users, 5, visible, antennas, subcarriers),
                dtype=torch.float32,
            )
            model_input[:, :, 2] = pmask[:, :, :, None, :].expand(
                b, users, visible, antennas, subcarriers,
            )
            for cp_wave in (64, 128, 256):
                batch = {
                    "input": model_input,
                    "ls_grid": torch.stack((ls_grid.real, ls_grid.imag), dim=2),
                    "pilot_spacing": torch.full((b,), spacing, dtype=torch.int64),
                    "cp_wave_samples": torch.full((b,), cp_wave, dtype=torch.int64),
                }
                derived = derive_pilot_only_fast_time_add(cfg, batch, "cpu")
                delay_idx = delay_window_indices(
                    subcarriers, add_delay_bins(cfg), delay_guard_bins(cfg),
                    device="cpu",
                )
                direct = _add_backprojection(
                    cfg, received, frame, layout, cp_wave, delay_idx,
                    fast_time=True, pilot_only_received=True,
                )
                with self.subTest(spacing=spacing, cp_wave=cp_wave):
                    self.assertTrue(torch.allclose(
                        derived, direct, rtol=1e-4, atol=5e-5,
                    ), msg=f"max_abs={(derived - direct).abs().max().item():.9g}")
                mixed_batches.append(batch)
                mixed_direct.append(direct)
        combined = {
            key: torch.cat([part[key] for part in mixed_batches], dim=0)
            for key in mixed_batches[0]
        }
        combined_derived = derive_pilot_only_fast_time_add(cfg, combined, "cpu")
        self.assertTrue(torch.allclose(
            combined_derived, torch.cat(mixed_direct, dim=0),
            rtol=1e-4, atol=5e-5,
        ))

    def test_direct_pilot_feature_survives_float32_ls_roundtrip_error(self):
        direct=torch.tensor([[[1.0, 0.0]]],dtype=torch.float32)
        reconstructed=direct+torch.tensor([[[6.32405e-5, 6.32405e-5]]])
        selected, diagnostic=_select_direct_feature(direct,reconstructed,1384)
        self.assertTrue(torch.equal(selected,direct))
        self.assertGreater(diagnostic,5e-5)
        with self.assertRaises(RuntimeError):
            _select_direct_feature(torch.full_like(direct,float('nan')),
                                   reconstructed,1384)


if __name__ == "__main__":
    unittest.main()
