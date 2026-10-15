"""Lee Sec. III-C / Eq. 14-15 parallel A-ML estimator, server-only candidate.

Widths 56/12/4 come from cited Mane et al. (2020), not a disclosed Lee
implementation. Shared per-user weights, RI channels, linear reconstruction
and same-shape padding are explicit adaptations. No old supervised weights.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call


class FSRCNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Conv3d(2, 56, 5, stride=1, padding=2)
        self.shrink = nn.Conv3d(56, 12, 1, stride=1)
        self.mapping = nn.ModuleList([nn.Conv3d(12, 12, 3, stride=1, padding=1)
                                      for _ in range(4)])
        self.expand = nn.Conv3d(12, 56, 1, stride=1)
        self.reconstruct = nn.ConvTranspose3d(56, 2, 9, stride=1, padding=4)

    def forward(self, x):
        z = F.relu(self.feature(x), inplace=False)
        z = F.relu(self.shrink(z), inplace=False)
        for layer in self.mapping:
            z = F.relu(layer(z), inplace=False)
        z = F.relu(self.expand(z), inplace=False)
        return self.reconstruct(z)


class LeeAlternativeMeta(nn.Module):
    def __init__(self):
        super().__init__()
        self.sft = FSRCNN3D()
        self.add = FSRCNN3D()

    @staticmethod
    def _public_ls(x):
        if x.ndim != 6 or tuple(x.shape[1:]) != (2, 5, 25, 8, 128):
            raise ValueError("Expected public input [B,2,5,25,8,128]")
        return x[:, :, :2].float().reshape(-1, 2, 25, 8, 128)

    @staticmethod
    def _complex(ri):
        return torch.complex(ri[:, 0].float(), ri[:, 1].float())

    @staticmethod
    def _ri(z):
        return torch.stack((z.real, z.imag), dim=1).float()

    def _add_from_ls(self, ls, parameters=None):
        # Eq. 15 transforms the original LS input, not the SFT prediction.
        spectrum = torch.fft.fftn(self._complex(ls), dim=(-3, -2, -1), norm="backward")
        add_input = self._ri(spectrum)
        add_output = (self.add(add_input) if parameters is None else
                      functional_call(self.add, parameters, (add_input,), strict=True))
        prediction = torch.fft.ifftn(self._complex(add_output), dim=(-3, -2, -1),
                                    norm="backward")
        return self._ri(prediction)

    def add_prediction(self, x, parameters=None):
        # Eq. 17 offline loss uses only ADD. SFT is not evaluated here.
        out = self._add_from_ls(self._public_ls(x), parameters)
        return out.reshape(x.shape[0], 2, 2, 25, 8, 128)

    def forward(self, x):
        ls = self._public_ls(x)
        # Eq. 14: an unweighted sum, no averaging or learned fusion.
        out = self.sft(ls) + self._add_from_ls(ls)
        return out.reshape(x.shape[0], 2, 2, 25, 8, 128)


def complex_mse(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 6 or prediction.shape[2] != 2:
        raise ValueError("MSE requires matching full-grid RI CFR tensors")
    # Mean |complex error|^2: sum RI first, not a half-scaled RI mean.
    return (prediction.float() - target.float()).square().sum(dim=2).mean()
