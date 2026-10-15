"""Eight-method assets from complete score CSVs; server CPU only.

Explicit port of reviewed seven-method report, with Lee adaptation caption and
all8 curves/three actual high-mobility BER points required in each BER panel.
"""
import csv
from .protocol import LABELS,require

def render_assets(output):
    with (output / "summary" / "macro_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    dest = output / "paper_assets"
    dest.mkdir(exist_ok=True)
    selected = [r for r in rows if r["doppler_hz"] == "all" and r["snr_db"] == "all"
                and ((r["study"] == "nmse" and r["metric"] == "nmse" and r["modulation"] == "qpsk"
                      and r["scope"] in ("gpp_seen", "gpp_heldout", "gpp_high_mobility"))
                     or (r["study"] == "receiver" and r["metric"] == "ber_diagonal"
                         and r["scope"] in ("gpp_seen", "gpp_heldout")))]
    require(selected, "No headline groups in the complete macro summary")
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\caption{Comparison on identical frozen test frames. Means and 95\% training-seed intervals use seeds 42, 43, and 44. The Lee A-ML task adaptation uses one ADD-only full second-order meta epoch followed by 39 joint SFT+ADD online epochs. Its support/query counts and undivulged hyperparameters are declared task adaptations, not original author values. Task-adapted WiMAE uses 20 epochs of masked reconstruction followed by 20 epochs of supervised channel estimation; each epoch contains 12,000 fresh frames. Equal-budget FortiTran uses 40 epochs. The common-recipe WiMAE and Dual3D entries retain their supervised architecture adaptations.}",
             r"\label{tab:equal_budget_dl_comparison_lee_meta}", r"\begin{tabular}{llllr}",
             r"\toprule", r"Metric & Modulation & Channel scope & Method & Mean [95\% CI] \\", r"\midrule"]
    for row in sorted(selected, key=lambda r: (r["metric"], r["modulation"], r["scope"], list(LABELS).index(r["method"]))):
        keys = ("mean_db", "seed_ci95_lo_db", "seed_ci95_hi_db") if row["metric"] == "nmse" else ("mean", "seed_ci95_lo", "seed_ci95_hi")
        fmt = ".2f" if row["metric"] == "nmse" else ".3g"
        values = [format(float(row[k]), fmt) if row.get(k) else "--" for k in keys]
        lines.append(" & ".join(("NMSE (dB)" if row["metric"] == "nmse" else "BER", row["modulation"].upper(),
                                  row["scope"].replace("_", r"\_"), LABELS[row["method"]],
                                  f"{values[0]} [{values[1]}, {values[2]}]")) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (dest / "equal_budget_dl_comparison.tex").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    for column, scope in enumerate(("gpp_seen", "gpp_heldout")):
        for row_index, (study, metric) in enumerate((("nmse", "nmse"), ("receiver", "ber_diagonal"))):
            ax = axes[row_index, column]
            modulation, snr = ("qpsk", "25.0") if metric == "nmse" else ("16qam", "20.0")
            panel_points = 0
            for method, label in LABELS.items():
                points = sorted([r for r in rows if r["method"] == method and r["scope"] == scope
                                 and r["study"] == study and r["metric"] == metric and r["modulation"] == modulation
                                 and r["snr_db"] == snr and r["doppler_hz"] != "all"],
                                key=lambda r: float(r["doppler_hz"]))
                require(points, "Missing method curve for a declared panel: " + method)
                if metric == "ber_diagonal":
                    require({800.0, 1200.0, 1600.0}.issubset({float(p["doppler_hz"]) for p in points}), "Missing actual high-mobility BER point")
                panel_points += len(points)
                x = [float(r["doppler_hz"]) for r in points]
                key, lo, hi = ("mean_db", "seed_ci95_lo_db", "seed_ci95_hi_db") if metric == "nmse" else ("mean", "seed_ci95_lo", "seed_ci95_hi")
                ax.plot(x, [float(r[key]) for r in points], marker="o", markersize=2.5, linewidth=1, label=label)
                if all(r.get(lo) and r.get(hi) for r in points):
                    ax.fill_between(x, [float(r[lo]) for r in points], [float(r[hi]) for r in points], alpha=.08)
            ax.set_xlabel("Maximum Doppler shift (Hz)")
            ax.set_ylabel("NMSE (dB)" if metric == "nmse" else "BER")
            require(panel_points > 0, "No curves for a declared figure panel")
            ax.set_title(("Seen 3GPP" if scope == "gpp_seen" else "Held-out 3GPP")
                         + f", {'QPSK' if modulation == 'qpsk' else '16-QAM'}, SNR {float(snr):g} dB")
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.1), ncol=2, fontsize=7)
    fig.savefig(dest / "equal_budget_dl_comparison.pdf", bbox_inches="tight")
    fig.savefig(dest / "equal_budget_dl_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
