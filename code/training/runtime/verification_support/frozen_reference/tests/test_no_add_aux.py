"""The target-free control must have no coefficient-label gradient path."""

import unittest

import torch

from train import dopnet_auxiliary_loss


class NoAddAuxContract(unittest.TestCase):
    def _inputs(self, target_value):
        branch = torch.ones((1, 2, 2, 3, 2, 2), requires_grad=True)
        fused = torch.full((1, 2, 2, 3, 2, 2), 2.0, requires_grad=True)
        batch = {
            'input': torch.zeros(1),
            'target_add': torch.full_like(branch, target_value),
        }
        aux = {
            'architecture': 'dual3d_physics_fusion',
            'add_branch_pred': branch,
            'add_fused_pred': fused,
        }
        return batch, aux, branch, fused

    def test_coefficient_target_is_unused_when_weight_zero(self):
        cfg = {'train': {'add_branch_loss_weight': 0.0,
                         'add_sparsity_weight': 0.0001}}
        batch_a, aux_a, branch_a, fused_a = self._inputs(0.0)
        batch_b, aux_b, _, _ = self._inputs(123.0)
        loss_a, stats_a = dopnet_auxiliary_loss(aux_a, batch_a, cfg)
        loss_b, stats_b = dopnet_auxiliary_loss(aux_b, batch_b, cfg)
        self.assertTrue(torch.equal(loss_a, loss_b))
        self.assertNotIn('add_branch_mse', stats_a)
        self.assertNotIn('add_fused_mse_monitor', stats_a)
        self.assertEqual(stats_a.keys(), stats_b.keys())
        loss_a.backward()
        self.assertIsNone(branch_a.grad)
        self.assertIsNotNone(fused_a.grad)


if __name__ == '__main__':
    unittest.main()
