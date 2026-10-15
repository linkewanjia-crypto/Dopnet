"""Seven-method tables/figures and a static, verified WiMAE evidence bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from .prepare_wimae import HERE, digest, file_hash, signed_metadata, write_json
from .train_wimae import DeadlineReached, check_deadline, deadline_from
from .wimae_protocol import LABELS, check_evaluation_metadata, require
from .verify_wimae import ASSET_NAMES, verify_archive, verify_directory


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
             r"\caption{Comparison on identical frozen test frames. Means and 95\% training-seed intervals use seeds 42, 43, and 44. The task-adapted WiMAE uses 20 epochs of masked reconstruction followed by 20 epochs of supervised channel estimation; each epoch contains 12,000 fresh frames. Equal-budget FortiTran uses 40 epochs. The common-recipe WiMAE and Dual3D entries retain their supervised architecture adaptations.}",
             r"\label{tab:equal_budget_dl_comparison}", r"\begin{tabular}{llllr}",
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
                if not points:
                    continue
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "reference-results", "training-output", "forti-evaluation", "forti-code-root", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--deadline-utc", required=True)
    args = ap.parse_args()
    deadline = deadline_from(args.deadline_utc)
    check_deadline(deadline)
    output, training = args.output.resolve(), args.training_output.resolve()
    proof = verify_directory(output, training, args.reference_results, args.forti_evaluation, args.forti_code_root,
                             require_assets=False)
    check_deadline(deadline)
    if (output / "PAPER_READY.json").exists():
        verify_directory(output, training, args.reference_results, args.forti_evaluation, args.forti_code_root)
    else:
        render_assets(output)
        ev, statistics = check_evaluation_metadata(output)
        readiness = dict(status="complete", paper_ready=True, pending=[], evaluation_signature=ev["signature"],
                         statistics_signature=statistics["signature"],
                         assets_sha256={n: file_hash(output / "paper_assets" / n) for n in ASSET_NAMES})
        readiness["signature"] = digest(readiness)
        write_json(output / "PAPER_READY.json", readiness)
    proof = verify_directory(output, training, args.reference_results, args.forti_evaluation, args.forti_code_root)
    recipe = signed_metadata(training / "wimae_manifest.json")
    files = {}
    for path in output.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".csv", ".npz", ".tex", ".pdf", ".png"):
            files["evaluation/" + path.relative_to(output).as_posix()] = path
    for path in training.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".yaml", ".csv", ".pt"):
            files["training/" + path.relative_to(training).as_posix()] = path
    source_names = set(recipe["implementation_sha256"]) | {"wimae_protocol.py", "evaluate_wimae.py", "report_wimae.py",
                                                              "verify_wimae.py", "queue_wimae.py", "WIMAE_REPRODUCTION_zh.md"}
    for name in source_names:
        files["source/wimae/" + name] = HERE / name
    for name in recipe["source_files_sha256"]:
        files["source/archive_runtime/" + name] = args.project / name
    files["evidence/fortitran_budget_matched_results.zip"] = args.forti_evaluation / "budget_matched_results.zip"
    # Explicit delivery-only patch provenance; parent science bytes are preserved.
    for name in ("delivery_patch.json", "parent_verify_wimae.txt", "PARENT_SOURCE_RELEASE_MANIFEST.json"):
        files["source/wimae/" + name] = HERE / name
    # The independent controller stdout is outside these scientific directories.
    # Take one byte snapshot per file; no actively appended log is bundled.
    snapshots = {}
    for name, path in sorted(files.items()):
        check_deadline(deadline)
        snapshots[name] = path.read_bytes()
    manifest = {"schema": "dopnet_wimae_delivery_v1", "status": "complete", "paper_ready": True, "pending": [],
                "evaluation_signature": proof["evaluation_signature"], "training_recipe_signature": recipe["signature"],
                "claim": "task_adapted_equal_frame_budget", "original_CE_recipe_reproduction": False,
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in snapshots.items()}}
    manifest["signature"] = digest(manifest)
    package = output / "wimae_results_for_paper.zip"
    temporary = package.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as z:
        for name, data in snapshots.items():
            check_deadline(deadline)
            z.writestr(name, data)
        z.writestr("DELIVERY_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    verify_archive(temporary, args.reference_results, args.forti_code_root)
    check_deadline(deadline)
    temporary.replace(package)
    package.with_suffix(".zip.sha256").write_text(file_hash(package) + "  " + package.name + "\n", encoding="utf-8")
    print("WIMAE_DELIVERY_COMPLETE seven_methods=7 routes=21 cells=269", flush=True)


if __name__ == "__main__":
    try:
        main()
    except DeadlineReached as error:
        print("WIMAE_REPORT_INCOMPLETE_TIME_LIMIT " + str(error), file=sys.stderr, flush=True)
        raise SystemExit(75)
