#!/usr/bin/env python3
"""Render two CE summary plots from saved scores; original manuscript assets are separate.

This preserves the original plot_results.py statistics/CI plotting expressions,
but selects NMSE and common diagonal uncoded BER only. It does not claim byte
equivalence to the manuscript's separately preserved golden figure layouts.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = args.output.resolve()
    with (out / "summary/macro_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    dest = out / "tmlcn_summary_figures"
    dest.mkdir(exist_ok=True)
    methods = ("dopnet", "cfr_derived", "slow_time", "lee3ddual", "wimae_ce", "dopnet_lite")
    labels = {"dopnet": "DOP-Net", "cfr_derived": "CFR-derived control", "slow_time": "Slow-time control",
              "lee3ddual": "Lee (supervised)", "wimae_ce": "WiMAE-derived supervised CE", "dopnet_lite": "DOP-Net-Lite"}
    tasks = [("nmse", "gpp_all", "qpsk", "15.0", "nmse", "mean_db", "seed_ci95_lo_db", "seed_ci95_hi_db", "NMSE (dB)"),
             ("receiver", "dynamic_all", "16qam", "20.0", "ber_diagonal", "mean", "seed_ci95_lo", "seed_ci95_hi", "BER, common diagonal detector")]
    for study, scope, mod, snr, metric, mean, lo, hi, label in tasks:
        fig, ax = plt.subplots(figsize=(6.6, 4.2), layout="constrained")
        for method in methods:
            selected = [r for r in rows if (r["study"], r["scope"], r["modulation"], r["snr_db"], r["metric"], r["method"]) == (study, scope, mod, snr, metric, method) and r["doppler_hz"] != "all"]
            selected.sort(key=lambda r: float(r["doppler_hz"]))
            if not selected:
                raise RuntimeError(f"Expected result missing: {metric}, {method}")
            x = np.array([float(r["doppler_hz"]) for r in selected])
            y = np.array([float(r[mean]) for r in selected])
            lower = np.array([float(r[lo]) if r[lo] else np.nan for r in selected])
            upper = np.array([float(r[hi]) if r[hi] else np.nan for r in selected])
            line, = ax.plot(x, y, "o-", label=labels[method])
            ax.fill_between(x, lower, upper, alpha=.12, color=line.get_color())
        ax.set(xlabel="Maximum Doppler (Hz)", ylabel=label,
               title=f"{scope}; {mod}; SNR={float(snr):g} dB\n3 training seeds, pointwise 95% t intervals")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        for ext in ("png", "pdf"):
            fig.savefig(dest / f"{metric}.{ext}", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
