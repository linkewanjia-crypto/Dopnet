"""Shared metric definitions.

All evaluation uses the same primary NMSE statistic:

    10 log10( mean_i [ ||Hhat_i - H_i||_F^2 / ||H_i||_F^2 ] ).

Confidence intervals are formed on the *linear* per-sample NMSE values and only
then converted to dB.  This avoids mixing global-energy NMSE, mean sample NMSE,
and averages of per-batch dB values across different scripts.

The stratified reporting helpers avoid averaging NMSE over the whole SNR
and Doppler range hides the effect the ADD branch exists for: its value grows
with ``eps = f_d / df`` and it only pays off once the intra-symbol interference
sits above the noise floor.  A single scalar therefore dilutes it into
invisibility, which is why every headline number in this project is reported on
a Doppler x SNR grid.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch


def nmse_per_sample(pred_ri: torch.Tensor, target_ri: torch.Tensor,
                    eps: float = 1e-12) -> torch.Tensor:
    dims = tuple(range(1, pred_ri.ndim))
    err = (pred_ri - target_ri).pow(2).sum(dim=dims)
    pwr = target_ri.pow(2).sum(dim=dims).clamp_min(eps)
    return err / pwr


def nmse_linear(pred_ri, target_ri, eps=1e-12):
    return nmse_per_sample(pred_ri, target_ri, eps).mean()


def complex_nmse_db(pred_ri, target_ri, eps=1e-12):
    nmse = nmse_linear(pred_ri, target_ri, eps)
    return 10.0 * torch.log10(nmse.clamp_min(eps))


def nmse_summary_db(values, z: float = 1.96, eps: float = 1e-15):
    """Mean-NMSE dB and a normal 95% CI, formed on the linear sample ratios."""
    if torch.is_tensor(values):
        x = values.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    else:
        x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else 0.0
    lo = max(mean - float(z) * se, eps)
    hi = max(mean + float(z) * se, eps)
    return (10.0 * math.log10(max(mean, eps)),
            10.0 * math.log10(lo),
            10.0 * math.log10(hi))


def snr_bin_edges(snr_min, snr_max, width=5.0):
    lo = math.floor(float(snr_min) / width) * width
    hi = math.ceil(float(snr_max) / width) * width
    n = max(1, int(round((hi - lo) / width)))
    return [lo + i * width for i in range(n + 1)]


def assign_snr_bin(snr_db, edges):
    """Return the label of the bin containing ``snr_db``."""
    for i in range(len(edges) - 1):
        if snr_db < edges[i + 1] or i == len(edges) - 2:
            return f"[{edges[i]:.0f},{edges[i+1]:.0f})"
    return f"[{edges[-2]:.0f},{edges[-1]:.0f})"


class StratifiedNMSE:
    """Accumulates linear per-sample NMSE into named cells.

    Cells are keyed by an arbitrary tuple; the canonical use in this project is
    ``(doppler_hz, snr_bin)`` and, for the generalization study, additionally the
    channel family.
    """

    def __init__(self):
        self._cells = defaultdict(list)

    def update(self, key, values):
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy().reshape(-1)
        self._cells[tuple(key)].extend(float(v) for v in np.asarray(values).reshape(-1))

    def keys(self):
        return sorted(self._cells.keys(), key=lambda k: tuple(str(x) for x in k))

    def count(self, key):
        return len(self._cells[tuple(key)])

    def summary(self, key):
        return nmse_summary_db(self._cells[tuple(key)])

    def marginal(self, axis):
        """Pool cells that share the value on ``axis`` and summarise each group."""
        pooled = defaultdict(list)
        for k, v in self._cells.items():
            pooled[k[axis]].extend(v)
        return {k: nmse_summary_db(v) for k, v in pooled.items()}

    def overall(self):
        allv = [v for vals in self._cells.values() for v in vals]
        return nmse_summary_db(allv)

    def rows(self, names):
        out = []
        for k in self.keys():
            mean, lo, hi = self.summary(k)
            row = {n: v for n, v in zip(names, k)}
            row.update({"nmse_db": mean, "ci_lo_db": lo, "ci_hi_db": hi,
                        "samples": self.count(k)})
            out.append(row)
        return out


def hard_decision_bit_errors(z_ri, labels, modulation_ids):
    """Uncoded bit errors of the detector output.

    ``z_ri``  [B,Nu,2,Td,F] estimated symbols.
    ``labels``[B,Nu,Td,F] transmitted constellation labels.
    ``modulation_ids`` [B] modulation index per sample (constant within a chunk).
    """
    from utils.modem import ID_TO_MOD, bits_per_symbol, labels_to_symbols

    errors = 0
    total = 0
    z = torch.complex(z_ri[:, :, 0], z_ri[:, :, 1]).detach().cpu()
    labels = labels.detach().cpu()
    ids = modulation_ids.detach().cpu().reshape(-1)
    for mod_id in sorted(set(int(v) for v in ids)):
        sel = (ids == mod_id)
        if not bool(sel.any()):
            continue
        mod = ID_TO_MOD[int(mod_id)]
        bps = bits_per_symbol(mod)
        M = 1 << bps
        const = torch.as_tensor(
            labels_to_symbols(np.arange(M), mod), dtype=torch.complex64
        )
        zs = z[sel].reshape(-1)
        dist = (zs[:, None] - const[None, :]).abs()
        pred = dist.argmin(dim=-1)
        true = labels[sel].reshape(-1)
        shifts = torch.arange(bps - 1, -1, -1)
        pb = (pred[:, None] >> shifts) & 1
        tb = (true[:, None] >> shifts) & 1
        errors += int((pb != tb).sum())
        total += int(tb.numel())
    return errors, total
