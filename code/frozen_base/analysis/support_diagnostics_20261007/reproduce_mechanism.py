"""Portable scalar replay for the archived TWC mechanism supplement.

The five small input files contain already calculated statistics. This CLI
does not import Torch, load raw waveforms/checkpoints, resample confidence
intervals, train a model, perform inference, or measure latency.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

INPUTS = {
    "scores": "score_analysis/descriptive_paired_groups.csv",
    "raw": "raw_support_analysis/grouped_energy.csv",
    "pwb": "raw_pwb_energy/grouped_pwb_energy.csv",
    "operators": "operator_analysis/operator_results.json",
    "representations": "representation_analysis/analysis_summary.json",
}
RANK_THRESHOLDS = (1e-8, 1e-6, 1e-4)
GROUP_ORDER = (("static", 4), ("static", 8), ("dynamic", 4), ("dynamic", 8))
STATE_LABELS = {"static": "Zero Doppler", "dynamic": "Nonzero Doppler"}
PALETTE = {"dopnet": "#0072B2", "slow_control": "#D55E00", "pilot_support_control": "#CC79A7"}
SEED_T_CRITICAL = 4.302652729749462  # Original archived df=2 constant.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    if value is None or value == "":
        return None
    result = float(value)
    require(math.isfinite(result), "Non-finite scalar in input")
    return result


def close(a, b, label, scale=None):
    a, b = number(a), number(b)
    if a is None or b is None:
        require(a is None and b is None, label + ": nullable value differs")
        return 0.0
    denom = max(1.0, abs(a), abs(b)) if scale is None else max(1.0, scale)
    error = abs(a - b) / denom
    require(error <= 1e-12, label + ": scalar equation disagrees")
    return error


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grouped_lookup(rows, axis):
    selected = {}
    for row in rows:
        if row["group_axis"] == axis:
            key = json.loads(row["group_key"])
            identity = (key["doppler_state"], int(key["pilot_spacing"]))
            require(identity not in selected, "Duplicate Doppler/spacing group")
            selected[identity] = row
    return selected


def effective_rank(record, view, threshold):
    values = record["spectra"][view]["effective_ranks"]
    matches = [value for key, value in values.items() if float(key) == threshold]
    require(len(matches) == 1, "Missing/duplicate effective-rank threshold")
    return int(matches[0])


def load_and_verify(data_root, bindings):
    paths = {name: data_root / relative for name, relative in INPUTS.items()}
    for name, path in paths.items():
        require(path.is_file(), "Missing input: " + INPUTS[name])
    hashes = {relative: sha256(data_root / relative) for relative in INPUTS.values()}
    expected = read_json(bindings)
    require(set(expected["files"]) == set(hashes), "Bindings do not describe exactly the five scalar inputs")
    require(hashes == expected["files"], "Input SHA256 differs from the declared archived scalar snapshot")
    scores = read_csv(paths["scores"])
    raw = read_csv(paths["raw"])
    pwb = read_csv(paths["pwb"])
    operators = read_json(paths["operators"])
    representations = read_json(paths["representations"])
    gap = sorted((row for row in scores if row["study"] == "nmse" and row["modulation"] == "qpsk"
                  and row["metric"] == "nmse" and row["dimension"] == "doppler_hz_dynamic"
                  and row["comparator"] == "pilot_only_fast_time" and row["reference"] == "dopnet"),
                 key=lambda row: number(row["value"]))
    require(len(gap) == 5, "Expected five archived QPSK support-gap Doppler groups")
    require([number(row["value"]) for row in gap] == [0., 400., 800., 1200., 1600.], "Unexpected Doppler grid")
    maximum_seed_interval_error = 0.0
    for row in gap:
        require(int(row["channel_count"]) == 11 and int(row["cell_count"]) == 44, "Test group composition differs")
        require(row["aggregation"] == "equal cells within channel, then equal channels", "Test aggregation differs")
        seeds = [number(row[f"seed{seed}_gap_db"]) for seed in (42, 43, 44)]
        mean = statistics.mean(seeds)
        delta = SEED_T_CRITICAL * statistics.stdev(seeds) / math.sqrt(3)
        for field, value in (("mean_paired_gap_db", mean), ("paired_gap_db_seed_ci95_lo", mean - delta),
                             ("paired_gap_db_seed_ci95_hi", mean + delta)):
            maximum_seed_interval_error = max(maximum_seed_interval_error, close(row[field], value, field))
    raw_groups, pwb_groups = grouped_lookup(raw, "doppler_state_spacing"), grouped_lookup(pwb, "doppler_state_spacing")
    require(set(raw_groups) == set(pwb_groups) == {(state, s) for state in ("static", "dynamic") for s in (2, 4, 8)},
            "Calibration Doppler/spacing groups differ")
    raw_all = next(row for row in raw if row["group_axis"] == "all")
    pwb_all = next(row for row in pwb if row["group_axis"] == "all")
    require(int(raw_all["frames"]) == int(pwb_all["frames"]) == 384, "Expected 384 saved calibration frames")
    require(int(raw_all["batches"]) == int(pwb_all["batches"]) == 96, "Expected 96 original four-frame batches")
    max_raw_error = max_pwb_error = 0.0
    for row in raw:
        null = number(row["null_energy_observed"]) - number(row["null_energy_noise_expected"])
        total = number(row["total_energy_observed"]) - number(row["total_energy_noise_expected"])
        max_raw_error = max(max_raw_error, close(row["null_energy_excess"], null, "Raw null excess"),
                            close(row["total_energy_excess"], total, "Raw total excess"))
        expected_fraction = null / total if total > 0 and int(row["null_re_count"]) else None
        max_raw_error = max(max_raw_error, close(row["null_excess_fraction_of_total_excess"], expected_fraction, "Pooled raw ratio"))
    for row in pwb:
        energies = [number(row[label + "_feature_energy"]) for label in ("full", "masked", "extra")]
        noises = [number(row[label + "_noise_expected_energy"]) for label in ("full", "masked", "extra")]
        cross = number(row["feature_cross_term"])
        max_pwb_error = max(max_pwb_error, close(energies[0], math.fsum(energies[1:] + [cross]), "PWB cross-term partition"),
                            close(noises[0], math.fsum(noises[1:]), "PWB noise partition"))
        for label, energy, noise in zip(("full", "masked", "extra"), energies, noises):
            max_pwb_error = max(max_pwb_error, close(row[label + "_feature_excess_energy"], energy - noise,
                                                    "PWB excess definition", abs(energy) + abs(noise)))
        expected = number(row["extra_feature_excess_energy"]) / noises[2] if noises[2] else None
        max_pwb_error = max(max_pwb_error, close(row["extra_excess_over_extra_noise"], expected, "PWB excess/noise ratio"),
                            close(row["extra_noise_fraction"], noises[2] / noises[0], "PWB noise share"))
    expected_counts = {("static", 4): (88, 22), ("static", 8): (20, 5),
                       ("dynamic", 4): (84, 21), ("dynamic", 8): (44, 11)}
    for key, counts in expected_counts.items():
        require((int(raw_groups[key]["frames"]), int(raw_groups[key]["batches"])) == counts, "Raw group frame/batch counts differ")
        require((int(pwb_groups[key]["frames"]), int(pwb_groups[key]["batches"])) == counts, "PWB group frame/batch counts differ")
        for group, metric in ((raw_groups[key], "null_excess_fraction_of_total_excess"),
                              (pwb_groups[key], "extra_excess_over_extra_noise")):
            lo, value, hi = (number(group[metric + "_ci95_low"]), number(group[metric]), number(group[metric + "_ci95_high"]))
            require(lo <= value <= hi, "Stored confidence interval does not bracket displayed estimate")
            require(int(group[metric + "_valid_bootstrap_draws"]) == 10000, "Expected 10,000 saved batch-bootstrap draws")
    require(sum(int(raw_groups[(state, 2)]["frames"]) for state in ("static", "dynamic")) == 148, "S=2 raw identity population differs")
    for state in ("static", "dynamic"):
        require(number(raw_groups[(state, 2)]["null_energy_observed"]) == 0, "S=2 raw null energy is not zero")
        require(number(pwb_groups[(state, 2)]["extra_feature_energy"]) == 0, "S=2 PWB extra energy is not zero")
    operators = sorted(operators, key=lambda row: (int(row["cp_active_samples"]), int(row["spacing"])))
    keys = [(int(row["cp_active_samples"]), int(row["spacing"])) for row in operators]
    require(keys == [(8, 2), (8, 4), (8, 8), (16, 2), (16, 4), (16, 8), (32, 2), (32, 4)],
            "Expected eight archived available comb/CP cases; no CP32/S8 case is added")
    max_operator_error = 0.0
    for row in operators:
        require(int(row["raw_observations"]) == 896, "Operator input dimension differs")
        require(int(row["masked_observations"]) == 896 * 2 // int(row["spacing"]), "Joint-support count differs")
        nf, nm, ne = (number(row["noise"][view]["trace_sigma2_one"]) for view in ("full", "masked", "extra"))
        max_operator_error = max(max_operator_error, close(nf, math.fsum([nm, ne]), "Operator noise-trace partition"),
                                 close(row["extra_noise_fraction"], ne / nf, "Operator extra-noise fraction"))
        for threshold in RANK_THRESHOLDS:
            rf, rm = (effective_rank(row, view, threshold) for view in ("full", "masked"))
            require(0 <= rf <= 896 and 0 <= rm <= int(row["masked_observations"]), "Effective rank exceeds input dimension")
            if int(row["spacing"]) == 2:
                require(rf == rm, "S=2 full/masked ranks differ")
    require(int(representations["frames"]) == 2000 and int(representations["spacing2"]["frames"]) == 680, "Representation populations differ")
    require(representations["spacing2"]["allclose_all"] and number(representations["spacing2"]["max_absolute_difference"]) == 0,
            "Saved S=2 representation identity differs")
    require(int(representations["spacing2"]["exact_equal_rows"]) == 1360, "S=2 user tensor count differs")
    overall = representations["overall"]
    close(overall["extra_to_full_energy_ratio"], overall["extra_support_energy_sum"] / overall["full_energy_sum"], "Representation pooled ratio")
    report = {
        "status": "PASS", "mode": "archived scalar replay", "input_sha256": hashes,
        "checks": {"five_input_hashes_match_declared_snapshot": True, "selected_qpsk_doppler_groups": len(gap),
                   "calibration_frames": 384, "calibration_batches": 96, "operator_cases": len(operators),
                   "validation_frames": 2000, "s2_validation_identity_frames": 680,
                   "max_seed_interval_scaled_error": maximum_seed_interval_error,
                   "max_raw_scalar_scaled_error": max_raw_error, "max_pwb_scalar_scaled_error": max_pwb_error,
                   "max_operator_scalar_scaled_error": max_operator_error},
        "no_training": True, "no_inference": True, "no_new_observations": True,
        "no_bootstrap_resampling": True, "no_raw_archive_or_checkpoint_access": True,
        "statistical_scope": [
            "Left figure: archived QPSK test scores, 11 channel cases and 44 cells per Doppler; paired three-seed t intervals.",
            "Right figure: separate archived calibration observations; ratios of pooled noise-corrected energies and original four-frame batch-bootstrap intervals.",
            "The two panels are not frame-paired. All negative saved interval limits are preserved.",
            "Zero-Doppler S=8 has five batches / 20 frames and is a small descriptive group.",
            "PWB feature energy differs from raw physical received energy; its cross term is retained.",
            "Finite map ranks do not establish general channel identifiability or neural NMSE guarantees.",
            "Validation residuals include other-user contributions and noise; they are not pure noise or test NMSE.",
        ],
    }
    return {"gap": gap, "raw_groups": raw_groups, "pwb_groups": pwb_groups, "operators": operators,
            "representations": representations, "report": report, "input_paths": paths}


def interval_tex(mean, lo, hi, scale=1.0):
    def bound(value):
        value *= scale
        return f"${value:.3f}$" if value < 0 else f"{value:.3f}"
    return f"{mean * scale:.3f} [{bound(lo)},{bound(hi)}]"


def table_texts(data):
    operator = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
                r"$C$ & $S$ & Masked inputs & \multicolumn{3}{c}{Rank: full / masked} & Extra noise (\%) \\",
                r"\cmidrule(lr){4-6}", r"& & & $10^{-8}$ & $10^{-6}$ & $10^{-4}$ & \\", r"\midrule"]
    for row in data["operators"]:
        cells = [str(int(row["cp_active_samples"])), str(int(row["spacing"])), str(int(row["masked_observations"]))]
        cells += [f"{effective_rank(row, 'full', threshold)}/{effective_rank(row, 'masked', threshold)}" for threshold in RANK_THRESHOLDS]
        cells.append(f"{100 * number(row['extra_noise_fraction']):.2f}")
        operator.append(" & ".join(cells) + r" \\")
    operator += [r"\bottomrule", r"\end{tabular}"]
    energy = [r"\begin{tabular}{lrrrrl}", r"\toprule",
              r"State & $S$ & \shortstack{Frames /\\batches} & \shortstack{Raw fraction (\%)\\{[95\% CI]}} & \shortstack{Extra-noise\\share (\%)} & \shortstack{Feature excess/noise\\{[95\% CI]}} \\", r"\midrule"]
    for key in GROUP_ORDER:
        raw, pwb = data["raw_groups"][key], data["pwb_groups"][key]
        raw_ci = interval_tex(number(raw["null_excess_fraction_of_total_excess"]),
                              number(raw["null_excess_fraction_of_total_excess_ci95_low"]),
                              number(raw["null_excess_fraction_of_total_excess_ci95_high"]), 100)
        feature_ci = interval_tex(number(pwb["extra_excess_over_extra_noise"]),
                                  number(pwb["extra_excess_over_extra_noise_ci95_low"]),
                                  number(pwb["extra_excess_over_extra_noise_ci95_high"]))
        cells = [STATE_LABELS[key[0]], str(key[1]), f"{int(raw['frames'])}/{int(raw['batches'])}", raw_ci,
                 f"{100 * number(pwb['extra_noise_fraction']):.2f}", feature_ci]
        energy.append(" & ".join(cells) + r" \\")
    energy += [r"\bottomrule", r"\end{tabular}"]
    return {"table_support_operator.tex": "\n".join(operator) + "\n",
            "table_support_energy.tex": "\n".join(energy) + "\n"}


def plot_figure(data, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib import font_manager
    except ImportError as error:
        raise RuntimeError("Plotting requires matplotlib; install requirements-plotting.txt or use --verify-only/--skip-plot") from error
    try:
        font_manager.findfont(font_manager.FontProperties(family="Times New Roman"), fallback_to_default=False)
        family, fallback = "Times New Roman", False
    except ValueError:
        family, fallback = "DejaVu Serif", True
    plt.rcParams.update({"font.family": family, "font.size": 8.5, "axes.labelsize": 9.5,
                         "axes.titlesize": 10., "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "legend.fontsize": 8.5, "axes.linewidth": .65, "lines.linewidth": 1.6,
                         "xtick.major.width": .65, "ytick.major.width": .65,
                         "xtick.major.size": 3, "ytick.major.size": 3,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "grid.color": "#D8D8D8", "grid.linestyle": ":", "grid.linewidth": .55,
                         "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(left=.090, right=.986, bottom=.335, top=.905, wspace=.34)
    gap = data["gap"]
    x = [number(row["value"]) for row in gap]
    y = [number(row["mean_paired_gap_db"]) for row in gap]
    lo = [number(row["paired_gap_db_seed_ci95_lo"]) for row in gap]
    hi = [number(row["paired_gap_db_seed_ci95_hi"]) for row in gap]
    line, = axes[0].plot(x, y, color=PALETTE["pilot_support_control"], linestyle=":", marker="D",
                        markersize=4.2, markerfacecolor="white", markeredgewidth=.9, zorder=3)
    axes[0].fill_between(x, lo, hi, color=PALETTE["pilot_support_control"], alpha=.16, linewidth=0, zorder=1)
    axes[0].set(xlim=(-40, 1640), ylim=(-1.65, 1.7), xticks=x, yticks=[-1.5, -1., -.5, 0., .5, 1., 1.5],
                xlabel="Maximum Doppler shift (Hz)", ylabel="Paired NMSE gap (dB)", title="(a) Paired estimation value")
    for state, color, marker, offset in (("static", PALETTE["slow_control"], "^", -.06),
                                         ("dynamic", PALETTE["dopnet"], "o", .06)):
        groups = [data["raw_groups"][(state, spacing)] for spacing in (4, 8)]
        values = [100 * number(g["null_excess_fraction_of_total_excess"]) for g in groups]
        lows = [100 * number(g["null_excess_fraction_of_total_excess_ci95_low"]) for g in groups]
        highs = [100 * number(g["null_excess_fraction_of_total_excess_ci95_high"]) for g in groups]
        axes[1].errorbar([offset, 1. + offset], values,
                         yerr=[[v - l for v, l in zip(values, lows)], [h - v for v, h in zip(values, highs)]],
                         color=color, fmt=marker, markersize=4.6, markerfacecolor="white", markeredgewidth=.9,
                         elinewidth=.9, capsize=2.1, capthick=.9, zorder=3)
    axes[1].set(xlim=(-.27, 1.27), ylim=(-.3, 4.45), xticks=[0, 1], xticklabels=["4", "8"],
                yticks=[0, 1, 2, 3, 4], xlabel="Pilot spacing", ylabel="Off-support excess fraction (%)",
                title="(b) Off-support received energy")
    for ax in axes:
        ax.grid(True, axis="both", zorder=0)
        ax.axhline(0, color="#606060", linewidth=.8, linestyle="--", zorder=2)
        ax.tick_params(direction="out", pad=3)
        ax.xaxis.labelpad, ax.yaxis.labelpad = 3.5, 4
        ax.title.set_y(1.02)
    axes[0].legend([line], ["Pilot-support control - DOP-Net"], frameon=False, loc="upper center",
                    bbox_to_anchor=(.5, -.39), handlelength=2.8, handletextpad=.6, borderaxespad=0)
    handles = [Line2D([], [], color=color, marker=marker, markerfacecolor="white", markeredgewidth=.9,
                       markersize=4.6, linestyle="none") for color, marker in ((PALETTE["slow_control"], "^"), (PALETTE["dopnet"], "o"))]
    axes[1].legend(handles, ["Zero Doppler", "Nonzero Doppler"], frameon=False, ncol=2, loc="upper center",
                    bbox_to_anchor=(.5, -.39), handlelength=1.4, handletextpad=.4, columnspacing=1.2, borderaxespad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    labels = [ax.title for ax in axes] + [axis.label for ax in axes for axis in (ax.xaxis, ax.yaxis)] + [ax.get_legend() for ax in axes]
    bbox_ok = all(b.x0 >= 0 and b.y0 >= 0 and b.x1 <= width and b.y1 <= height
                  for b in (label.get_window_extent(renderer) for label in labels))
    require(bbox_ok, "Figure labels would be clipped; check installed serif font metrics")
    files = ["fig08_support_tradeoff.pdf", "fig08_support_tradeoff.png"]
    fig.savefig(output_dir / files[0], metadata={"Title": "Archived support diagnostic evidence"})
    fig.savefig(output_dir / files[1])
    plt.close(fig)
    return {"files": files, "font_family": family, "serif_fallback_used": fallback,
            "matplotlib": matplotlib.__version__, "labels_within_canvas": bbox_ok,
            "note": "Rendering bytes can vary with font/matplotlib versions; scalar values and uncertainty definitions remain unchanged."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_data_root = (Path(__file__).resolve().parent / ".." / ".." / ".." / ".." /
                         "数据" / "开源数据" / "results" / "support_diagnostics_20261007").resolve()
    parser.add_argument("--data-root", type=Path, default=default_data_root,
                        help="Supplement result directory; defaults to the sibling integrated data release")
    parser.add_argument("--output-dir", type=Path, required=True, help="Separate output directory; input files are read-only")
    parser.add_argument("--verify-only", action="store_true", help="Standard-library validation and JSON report only; no tables/plots")
    parser.add_argument("--skip-plot", action="store_true", help="Produce both table bodies and report without importing matplotlib")
    args = parser.parse_args(argv)
    try:
        data = load_and_verify(args.data_root, Path(__file__).with_name("expected_data_hashes.json"))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        protected = {path.resolve() for path in data["input_paths"].values()}
        require((args.output_dir / "reproduction_report.json").resolve() not in protected, "Output report would overwrite an input")
        report = data["report"]
        report["verify_only"] = args.verify_only
        report["generated_tables"] = []
        if not args.verify_only:
            tables_dir = args.output_dir / "tables"
            tables_dir.mkdir(exist_ok=True)
            for name, text in table_texts(data).items():
                path = tables_dir / name
                require(path.resolve() not in protected, "Table would overwrite an input")
                path.write_text(text, encoding="utf-8", newline="\n")
                report["generated_tables"].append("tables/" + name)
            report["figure"] = None if args.skip_plot else plot_figure(data, args.output_dir)
        else:
            report["figure"] = None
        require(report["input_sha256"] == {relative: sha256(args.data_root / relative) for relative in INPUTS.values()}, "Inputs changed during scalar replay")
        report["all_input_bytes_unchanged"] = True
        report["output_sha256"] = {relative: sha256(args.output_dir / relative) for relative in report["generated_tables"]}
        if report["figure"]:
            report["output_sha256"].update({relative: sha256(args.output_dir / relative) for relative in report["figure"]["files"]})
        (args.output_dir / "reproduction_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": "PASS", "verify_only": args.verify_only, "tables": report["generated_tables"],
                          "figure": report["figure"], "inputs_unchanged": True}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as error:
        print("Scalar replay failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
