"""CPU-only structural checks for the matched pilot-observation ablation.

These checks do not simulate a channel or run a neural estimator.
"""
import unittest
from pathlib import Path

import torch
import yaml

from data.add_geometry import add_delay_bins, delay_guard_bins, delay_window_indices
from data.waveform import (
    _add_backprojection, _build_frame, _mask_to_joint_pilot_bins,
)
from models.dopnet import ABLATION_PROFILES


ROOT = Path(__file__).resolve().parents[1]


class PilotOnlyFastTimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = yaml.safe_load((ROOT / 'configs' / 'dopnet.yaml').read_text())

    def _pilot_frame(self, spacing):
        cfg = self.cfg
        system = cfg['system']
        generator = torch.Generator(device='cpu').manual_seed(91038)
        frame, layout, _, _, _, _, _ = _build_frame(
            1, int(system['n_users']), int(system['n_subcarriers']), spacing,
            'qpsk', torch.device('cpu'), generator,
            int(system['frame_num_symbols']), int(system['pilot_symbol_period']),
        )
        return frame, layout

    def test_architecture_switch_is_only_the_observation_source(self):
        full = ABLATION_PROFILES['full']
        control = dict(ABLATION_PROFILES['pilot_only_fast_time'])
        self.assertEqual(control.pop('add_observation_source'), 'pilot_only_fast_time')
        self.assertEqual(control, full)

    def test_joint_pilot_mask_keeps_all_bins_at_spacing_two(self):
        frame, layout = self._pilot_frame(spacing=2)
        pilot_blocks = torch.tensor(layout.pilot_blocks)
        xfreq = frame.index_select(2, pilot_blocks)
        support = xfreq.ne(0).any(dim=1)
        self.assertTrue(bool(support.all()))
        yfreq = torch.complex(
            torch.arange(xfreq.shape[-1], dtype=torch.float32).view(1, 1, 1, -1)
            .expand(1, 8, xfreq.shape[2], -1),
            torch.ones(1, 8, xfreq.shape[2], xfreq.shape[-1]),
        )
        self.assertTrue(torch.equal(_mask_to_joint_pilot_bins(yfreq, xfreq), yfreq))

        # Equal pre-IFFT observations must yield the exact same fast-time PWB.
        all_blocks = int(layout.num_blocks)
        y = torch.zeros(1, 8, all_blocks, xfreq.shape[-1], dtype=torch.complex64)
        y.index_copy_(2, pilot_blocks, yfreq)
        delay_idx = delay_window_indices(
            xfreq.shape[-1], add_delay_bins(self.cfg), delay_guard_bins(self.cfg),
            device=torch.device('cpu'),
        )
        cp_wave = 64
        full = _add_backprojection(
            self.cfg, y, frame, layout, cp_wave, delay_idx,
            per_user=False, fast_time=True,
        )
        pilot_only = _add_backprojection(
            self.cfg, y, frame, layout, cp_wave, delay_idx,
            per_user=False, fast_time=True, pilot_only_received=True,
        )
        self.assertTrue(torch.equal(full, pilot_only))

    def test_joint_mask_retains_both_users_and_removes_off_comb_bins(self):
        frame, layout = self._pilot_frame(spacing=4)
        xfreq = frame.index_select(2, torch.tensor(layout.pilot_blocks))
        support = xfreq.ne(0).any(dim=1)
        self.assertTrue(bool(support.any()))
        self.assertTrue(bool((~support).any()))
        self.assertTrue(bool((xfreq.ne(0).sum(dim=1) <= 1).all()))
        # Every transmitted pilot is retained; the other half of the received
        # bins are removed before inverse FFT.
        self.assertTrue(bool((xfreq.ne(0) <= support[:, None]).all()))
        yfreq = torch.ones(1, 8, xfreq.shape[2], xfreq.shape[-1], dtype=torch.complex64)
        masked = _mask_to_joint_pilot_bins(yfreq, xfreq)
        self.assertTrue(torch.equal(masked[:, 0], support.to(torch.complex64)))
        self.assertEqual(int(support.sum()), support.numel() // 2)

        # A received waveform carried only on off-comb bins reaches the Full
        # PWB through fast-time phase mixing, but the control removes it.
        off_comb = (~support)[:, None].expand(1, 8, -1, -1).to(torch.complex64)
        y = torch.zeros(1, 8, layout.num_blocks, xfreq.shape[-1], dtype=torch.complex64)
        y.index_copy_(2, torch.tensor(layout.pilot_blocks), off_comb)
        delay_idx = delay_window_indices(
            xfreq.shape[-1], add_delay_bins(self.cfg), delay_guard_bins(self.cfg),
            device=torch.device('cpu'),
        )
        full = _add_backprojection(
            self.cfg, y, frame, layout, 64, delay_idx,
            per_user=False, fast_time=True,
        )
        pilot_only = _add_backprojection(
            self.cfg, y, frame, layout, 64, delay_idx,
            per_user=False, fast_time=True, pilot_only_received=True,
        )
        self.assertGreater(float(full.abs().max()), 1e-3)
        self.assertEqual(float(pilot_only.abs().max()), 0.0)


if __name__ == '__main__':
    unittest.main()
