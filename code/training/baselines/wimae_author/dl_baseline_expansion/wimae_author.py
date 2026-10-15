"""Official WiMAE modules with an observable-input multiuser CE adapter.

Only training SSL receives clean CSI. CE forward accepts the public LS grid.
Time slices share an antenna-frequency model; no temporal attention is claimed.
"""
from __future__ import annotations

import torch
from torch import nn
from .vendor.wimae_upstream.models.wimae import WiMAE
from .vendor.wimae_upstream.models.modules.patching import InversePatcher


class WiMAESliceCE(nn.Module):
    name = "wimae_author_slice_ce"
    estimator_name = name
    requires_training = True

    def __init__(self, cfg, device=None):
        super().__init__()
        self.cfg = cfg
        s, spec = cfg["system"], cfg["wimae_reproduction"]
        geometry = (s["n_users"], s["frame_num_symbols"], s["n_ant"], s["n_subcarriers"])
        if geometry != (2, 25, 8, 128) or spec["patch_size"] != [1, 16]:
            raise ValueError("This signed adapter is fixed to2users/25symbols/8antennas/128tones")
        self.device = torch.device(device or "cpu")
        self.core = WiMAE(patch_size=(1, 16), encoder_dim=64, encoder_layers=12,
                          encoder_nhead=16, decoder_layers=4, decoder_nhead=8,
                          mask_ratio=0.6, device=self.device)
        self.inverse = InversePatcher(original_shape=(8, 128), patch_size=(1, 16))
        self.core.encoder.masking.generator.manual_seed(int(spec["train_mask_seed"]))
        self.scale_epsilon = float(spec["normalization"]["epsilon"])
        self.scale_floor = float(spec["normalization"]["minimum_scale"])

    @staticmethod
    def _canonical(tensor, channels):
        shared = tensor.ndim == 5
        if shared:
            tensor = tensor.unsqueeze(1)
        if (tensor.ndim != 6 or tensor.shape[2] != channels
                or tuple(tensor.shape[3:]) != (25, 8, 128)
                or (not shared and tensor.shape[1] != 2)):
            raise ValueError("Unexpected public/target tensor layout")
        return tensor, shared

    def observed_scale(self, public):
        public, _ = self._canonical(public, 5)
        ri = public[:, :, :2].float()
        mask = public[:, :, 2].float() > 0.5
        count = mask.sum(dim=(2, 3, 4), keepdim=True)
        if bool((count == 0).any()):
            raise ValueError("CE normalization requires actual pilot observations")
        energy = (ri.square().sum(dim=2) * mask).sum(dim=(2, 3, 4), keepdim=True)
        scale = (energy / count + self.scale_epsilon).sqrt().clamp_min(self.scale_floor)
        if not bool(torch.isfinite(scale).all()):
            raise ValueError("Nonfinite observed pilot scale")
        return scale

    @staticmethod
    def _fold(ri, scale):
        z = torch.complex(ri[:, :, 0].float(), ri[:, :, 1].float()) / scale
        b, u, t, a, f = z.shape
        return z.reshape(b * u * t, a, f), (b, u, t)

    def mask_rng_state(self):
        return self.core.encoder.masking.generator.get_state()

    def load_mask_rng_state(self, state):
        self.core.encoder.masking.generator.set_state(state.cpu())

    def seed_mask_rng(self, seed):
        self.core.encoder.masking.generator.manual_seed(int(seed))

    def ssl_loss(self, target_h, public_input):
        """Training-label MAE; paired RI mask and masked-token MSE only."""
        target, _ = self._canonical(target_h, 2)
        public, _ = self._canonical(public_input, 5)
        if target.shape[:2] != public.shape[:2]:
            raise ValueError("Training target/public frames differ")
        matrices, _ = self._fold(target, self.observed_scale(public))
        patches = self.core.patcher(matrices)
        output = self.core(matrices)
        indices = output["ids_mask"]
        if indices.shape[1] != 78 or output["ids_keep"].shape[1] != 50:
            raise RuntimeError("Official paired-RI nominal0.6 mask contract differs")
        gather = indices.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        prediction = torch.gather(output["reconstructed_patches"].float(), 1, gather)
        truth = torch.gather(patches.float(), 1, gather)
        return torch.nn.functional.mse_loss(prediction, truth)

    def forward(self, public_input, return_aux=False, **unused):
        public, shared = self._canonical(public_input, 5)
        scale = self.observed_scale(public)
        matrices, (b, u, t) = self._fold(public[:, :, :2], scale)
        patches = self.core.patcher(matrices)
        # Calling WiMAE.encode()/decode() would disable training gradients.
        encoded = self.core.encoder(patches, apply_mask=False)
        keep = torch.arange(128, device=patches.device).expand(patches.shape[0], -1)
        reconstructed = self.core.decoder(encoded, keep, 128).float()
        # torch.complex does not accept BF16, so inverse patching is FP32.
        z = self.inverse(reconstructed).reshape(b, u, t, 8, 128) * scale
        result = torch.stack((z.real, z.imag), dim=2).float()
        if shared:
            result = result.squeeze(1)
        if return_aux:
            return result, {"architecture": self.name, "fast_observation_available": False,
                            "aft_block_aux": [], "add_block_aux": [],
                            "normalization": "observed_pilot_RMS_only"}
        return result
