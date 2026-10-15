# SPDX-License-Identifier: MIT
"""Read-only diagnostics on the 384 archived calibration pilot frames.

Requires NumPy; Matplotlib is optional and imported only with --plots.
Inputs are original-byte NPZ files and a relative-path inventory under
--data-root. No training, inference, or new waveform generation is performed.
Energy statistics deliberately retain negative noise-subtracted estimates.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
import csv
import hashlib
import io
import json
import platform
from collections import defaultdict
from pathlib import Path, PurePosixPath
import sys
import time
import numpy as np

EXPECTED_POPULATIONS = {"calibration": 32, "statistical_calibration": 64}
BOOTSTRAP_SEED = 20261007


class SourceDirectory:
    """Read only declared ordinary NPZ files; no private archive dependency."""
    def __init__(self, root):
        self.root = root.resolve(strict=True)

    def path(self, relative):
        if not isinstance(relative, str) or "\\" in relative:
            raise ValueError("Inventory paths must use portable forward slashes")
        parts = PurePosixPath(relative)
        if parts.is_absolute() or not parts.parts or ".." in parts.parts:
            raise ValueError("Inventory path must stay inside data-root")
        path = self.root.joinpath(*parts.parts).resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Inventory input is not a file inside data-root")
        return path

    def read(self, relative):
        return self.path(relative).read_bytes()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def validate_inventory(info, source):
    entries = info.get("batches")
    if not isinstance(entries, list) or len(entries) != 96:
        raise ValueError("Expected the original 96 archived four-frame batches")
    by_population = defaultdict(list)
    paths, identifiers = set(), set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid inventory batch")
        required = ("relative_path", "population", "batch_id", "sha256", "bytes",
                    "original_path", "source_archive", "archive_member")
        if any(key not in entry for key in required):
            raise ValueError("Incomplete input identity/lineage in inventory")
        relative, population = entry["relative_path"], entry["population"]
        source.path(relative)
        if population not in EXPECTED_POPULATIONS:
            raise ValueError("This entry supports only the two calibration populations")
        if relative in paths or entry["batch_id"] in identifiers:
            raise ValueError("Duplicate batch path or cluster identifier")
        if entry["batch_id"] != f"{population}:{entry['original_path']}":
            raise ValueError("Historical four-frame cluster identifier differs")
        sha = entry["sha256"]
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Invalid SHA-256 identity")
        if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
            raise ValueError("Invalid original-byte length")
        paths.add(relative)
        identifiers.add(entry["batch_id"])
        by_population[population].append(entry)
    if {key: len(value) for key, value in by_population.items()} != EXPECTED_POPULATIONS:
        raise ValueError("Expected 32 calibration and 64 statistical-calibration batches")
    # Original analysis used calibration, then statistical_calibration. Keep
    # within-population inventory order as well: finite bootstrap draws depend
    # on this ordering even though the population statistic is order-invariant.
    return {population: by_population[population] for population in EXPECTED_POPULATIONS}


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def archive_stat(path):
    st = Path(path).stat()
    return {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def finite(value):
    return float(value) if np.isfinite(value) else None


def aggregate(records):
    sums = {field: sum(row[field] for row in records) for field in (
        "null_re_count", "support_re_count", "total_re_count", "null_energy_observed",
        "null_energy_noise_expected", "null_energy_excess", "total_energy_observed",
        "total_energy_noise_expected", "total_energy_excess")}
    nn = sums["null_re_count"]
    en = sums["null_energy_noise_expected"]
    et = sums["total_energy_excess"]
    return {
        "frames": len(records), "batches": len({row["batch_id"] for row in records}),
        **sums,
        "null_energy_observed_per_re": sums["null_energy_observed"] / nn if nn else None,
        "null_energy_noise_expected_per_re": en / nn if nn else None,
        "null_energy_excess_per_re": sums["null_energy_excess"] / nn if nn else None,
        "null_excess_over_expected_noise": sums["null_energy_excess"] / en if en else None,
        "null_observed_over_expected_noise": sums["null_energy_observed"] / en if en else None,
        "null_excess_fraction_of_total_excess": sums["null_energy_excess"] / et if et > 0 and nn else None,
        "null_excess_fraction_of_total_observed": sums["null_energy_excess"] / sums["total_energy_observed"] if nn else None,
        "negative_null_excess_frames": sum(row["null_energy_excess"] < 0 for row in records),
        "negative_total_excess_frames": sum(row["total_energy_excess"] < 0 for row in records),
    }


def cluster_ci(records, rng, resamples):
    """Batch bootstrap; descriptive uncertainty, not a new blind-test inference."""
    batches = defaultdict(list)
    for row in records:
        batches[row["batch_id"]].append(row)
    matrix = np.array([[sum(r[f] for r in part) for f in (
        "null_energy_excess", "null_energy_noise_expected", "total_energy_excess", "null_re_count")]
        for part in batches.values()], dtype=np.float64)
    result = {}
    for key in ("null_excess_over_expected_noise", "null_excess_fraction_of_total_excess", "null_energy_excess_per_re"):
        result[key + "_ci95_low"] = None
        result[key + "_ci95_high"] = None
        result[key + "_valid_bootstrap_draws"] = 0
    if len(matrix) < 2 or matrix[:, 3].sum() == 0:
        return result
    sample = matrix[rng.integers(0, len(matrix), size=(resamples, len(matrix)))].sum(axis=1)
    for key, denominator in (("null_excess_over_expected_noise", 1), ("null_excess_fraction_of_total_excess", 2), ("null_energy_excess_per_re", 3)):
        good = sample[:, denominator] > 0
        values = sample[good, 0] / sample[good, denominator]
        result[key + "_valid_bootstrap_draws"] = len(values)
        if len(values):
            lo, hi = np.percentile(values, [2.5, 97.5])
            result[key + "_ci95_low"] = float(lo)
            result[key + "_ci95_high"] = float(hi)
    return result


def make_groups(rows, resamples):
    axes = [("all", ()), ("role", ("role",)), ("profile", ("profile",)),
            ("doppler", ("doppler_hz",)), ("spacing", ("pilot_spacing",)),
            ("snr", ("snr_bin",)), ("doppler_state_spacing", ("doppler_state", "pilot_spacing")),
            ("role_profile", ("role", "profile")),
            ("role_profile_doppler_spacing_snr", ("role", "profile", "doppler_hz", "pilot_spacing", "snr_bin"))]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    result = []
    for axis, fields in axes:
        parts = defaultdict(list)
        for row in rows:
            parts[tuple(row[f] for f in fields)].append(row)
        for key, part in sorted(parts.items(), key=lambda item: str(item[0])):
            labels = dict(zip(fields, key))
            result.append({"group_axis": axis, "group_key": json.dumps(labels, ensure_ascii=False, sort_keys=True),
                **aggregate(part), **cluster_ci(part, rng, resamples)})
    return result


def plots(rows, groups, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.6, 4.1), constrained_layout=True)
    for spacing, color in ((4, "#0072B2"), (8, "#D55E00")):
        part = [r for r in rows if r["pilot_spacing"] == spacing and r["null_re_count"]]
        ax.scatter([r["doppler_hz"] for r in part], [r["null_excess_over_expected_noise"] for r in part],
                   label=f"S={spacing} (n={len(part)})", s=17, alpha=.55, color=color)
    ax.axhline(0, color="black", linestyle="--", linewidth=.8)
    ax.set_yscale("symlog", linthresh=.2)
    ax.margins(y=.10)
    ax.set(xlabel="Configured maximum Doppler (Hz)", ylabel="Null-bin excess / expected noise energy",
           title="Archived calibration observations; negative estimates retained")
    ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"null_excess_vs_doppler.{ext}", dpi=240)
    plt.close(fig)

    part = [g for g in groups if g["group_axis"] == "doppler_state_spacing" and g["null_re_count"]]
    fig, ax = plt.subplots(figsize=(6.6, 3.7), constrained_layout=True)
    labels = []
    for i, group in enumerate(part):
        key = json.loads(group["group_key"])
        value = group["null_excess_fraction_of_total_excess"]
        lo = group["null_excess_fraction_of_total_excess_ci95_low"]
        hi = group["null_excess_fraction_of_total_excess_ci95_high"]
        ax.bar(i, value, color="#0072B2" if key["doppler_state"] == "static" else "#D55E00", width=.65)
        if lo is not None:
            ax.errorbar(i, value, yerr=[[max(0, value - lo)], [max(0, hi - value)]], color="black", capsize=4, fmt="none")
        labels.append(f"{key['doppler_state']}, S={key['pilot_spacing']}\nn={group['frames']}, batches={group['batches']}")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(range(len(part)), labels)
    ax.set(ylabel="Null-bin excess / all-bin excess (pooled)",
           title="Descriptive batch-bootstrap 95% intervals")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"null_fraction_static_vs_dynamic.{ext}", dpi=240)
    plt.close(fig)

    part = [g for g in groups if g["group_axis"] == "profile" and g["null_re_count"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
    for i, g in enumerate(part):
        value = g["null_excess_fraction_of_total_excess"]
        lo = g["null_excess_fraction_of_total_excess_ci95_low"]
        hi = g["null_excess_fraction_of_total_excess_ci95_high"]
        ax.plot(value, i, "o", color="#0072B2")
        if lo is not None:
            ax.plot([lo, hi], [i, i], color="#0072B2", linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=.8)
    ax.set_yticks(range(len(part)), [f"{json.loads(g['group_key'])['profile']} (n={g['frames']})" for g in part])
    ax.set(xlabel="Null-bin excess / all-bin excess (pooled)",
           title="Profile diagnostic; mixtures of pilot spacings and Dopplers")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"null_fraction_by_profile.{ext}", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory containing inventory.json and the 96 original NPZ files")
    parser.add_argument("--inventory", type=Path,
                        help="Override inventory path; default: DATA_ROOT/inventory.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Absent or empty output directory outside data-root")
    parser.add_argument("--bootstrap", type=int, default=10000,
                        help="Four-frame cluster bootstrap draws (paper setting: 10000)")
    figures = parser.add_mutually_exclusive_group()
    figures.add_argument("--plots", dest="plots", action="store_true",
                         help="Generate the optional diagnostic plots; requires Matplotlib")
    figures.add_argument("--no-plots", dest="plots", action="store_false",
                         help="Generate CSV/JSON only (default)")
    parser.set_defaults(plots=False)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")
    if not args.data_root.is_dir():
        parser.error("--data-root must be an existing directory")
    args.data_root = args.data_root.resolve(strict=True)
    args.inventory = (args.inventory or args.data_root / "inventory.json").resolve(strict=True)
    args.out = args.output_dir.resolve()
    if args.out.is_relative_to(args.data_root):
        parser.error("--output-dir must be outside the immutable data-root")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error("--output-dir must be absent or empty")
    start = time.perf_counter()
    raw_inventory = args.inventory.read_bytes()
    info = json.loads(raw_inventory)
    rows, bindings = [], []
    max_partition_error = max_double_crosscheck_error = max_float32_reduction_relative_error = 0.
    arrays_read = ("y_pilot", "x_pilot", "noise_var", "metadata", "pilot_visible", "data_visible",
                   "cp_wave_samples", "waveform_fft_size", "subcarrier_spacing_hz")
    with SourceDirectory(args.data_root) as snapshot:
        indices = validate_inventory(info, snapshot)
        before = {e["relative_path"]: archive_stat(snapshot.path(e["relative_path"]))
                  for entries in indices.values() for e in entries}
        for population, entries in indices.items():
            for entry in entries:
                raw = snapshot.read(entry["relative_path"])
                digest = digest_bytes(raw)
                if digest != entry["sha256"] or len(raw) != entry["bytes"]:
                    raise RuntimeError(f"Source hash failed: {entry['original_path']}")
                with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                    arrays = {key: data[key] for key in arrays_read}
                metadata = json.loads(str(arrays["metadata"]))
                if "metadata" in entry and metadata != entry["metadata"]:
                    raise RuntimeError("Inventory metadata differs from actual source metadata")
                if metadata["role"] != population:
                    raise RuntimeError("Actual NPZ role differs from declared calibration population")
                y, x, noise = arrays["y_pilot"], arrays["x_pilot"], arrays["noise_var"]
                if y.ndim != 4 or x.ndim != 4 or y.shape[0] != x.shape[0] or y.shape[1] != x.shape[2] or y.shape[-1] != x.shape[-1]:
                    raise RuntimeError("Unsupported observation shape")
                if y.shape[0] != 4 or noise.shape != (y.shape[0],) or not np.isfinite(y).all() or not np.isfinite(x).all() or not np.isfinite(noise).all() or not (noise > 0).all():
                    raise RuntimeError("Batch size or finite/noise check failed")
                masks = x != 0
                if (masks.sum(axis=1) > 1).any():
                    raise RuntimeError("Pilot supports are not disjoint")
                joint = masks.any(axis=1)  # [B,P,N]
                energies = y.real.astype(np.float64)**2 + y.imag.astype(np.float64)**2
                check_energy = np.abs(y.astype(np.complex128))**2
                batch_id = entry["batch_id"]
                binding = {"population": population, "batch_id": batch_id, "original_path": entry["original_path"],
                    "relative_path": entry["relative_path"], "source_archive": entry["source_archive"],
                    "archive_member": entry["archive_member"], "bytes": len(raw), "expected_sha256": entry["sha256"],
                    "actual_sha256": digest, "sha256_matches": True, "arrays_read": list(arrays_read),
                    "frames": len(y), "y_pilot_shape": list(y.shape), "x_pilot_shape": list(x.shape),
                    "y_pilot_dtype": str(y.dtype), "x_pilot_dtype": str(x.dtype), "noise_var_dtype": str(noise.dtype)}
                bindings.append(binding)
                c = metadata["condition"]
                for i in range(len(y)):
                    P, R, N = y.shape[1:]
                    null = ~joint[i]
                    nnull = int(null.sum()) * R
                    ntotal = P * R * N
                    es = float((energies[i] * joint[i, :, None, :]).sum(dtype=np.float64))
                    en = float((energies[i] * null[:, None, :]).sum(dtype=np.float64))
                    et = float(energies[i].sum(dtype=np.float64))
                    max_partition_error = max(max_partition_error, abs(et-es-en)/max(et, 1e-300))
                    alt = float(check_energy[i].sum(dtype=np.float64))
                    max_double_crosscheck_error = max(max_double_crosscheck_error, abs(et-alt)/max(et, 1e-300))
                    e32 = float((np.abs(y[i])**2).sum(dtype=np.float32))
                    max_float32_reduction_relative_error = max(max_float32_reduction_relative_error, abs(et-e32)/max(et,1e-300))
                    nv = float(noise[i])
                    expected = nv*nnull
                    excess = en-expected
                    total_expected = nv*ntotal
                    snr = float(metadata["snr_db"][i])
                    binlo = int(np.floor(snr/10))*10
                    row = {"population": population, "role": metadata["role"], "batch_id": batch_id,
                        "frame_offset": int(metadata["offset"])+i, "source_sha256": digest,
                        "profile": c["channel_type"], "doppler_hz": float(c["doppler_hz"]),
                        "doppler_state": "static" if float(c["doppler_hz"]) == 0 else "dynamic",
                        "pilot_spacing": int(c["pilot_spacing"]), "snr_db": snr, "snr_bin": f"[{binlo},{binlo+10})",
                        "modulation": c["modulation"], "cp_len_active": int(c["cp_len"]),
                        "delay_spread_ns": float(c["delay_spread_ns"]),
                        "waveform_fft_size": int(arrays["waveform_fft_size"]), "cp_wave_samples": int(arrays["cp_wave_samples"]),
                        "subcarrier_spacing_hz": float(arrays["subcarrier_spacing_hz"]),
                        "pilot_symbols": P, "receivers": R, "occupied_bins": N, "users": x.shape[1],
                        "null_frequency_bins_per_symbol": int(null[0].sum()), "support_re_count": ntotal-nnull,
                        "null_re_count": nnull, "total_re_count": ntotal, "noise_var": nv,
                        "support_energy_observed": es, "support_energy_noise_expected": nv*(ntotal-nnull),
                        "support_energy_excess": es-nv*(ntotal-nnull),
                        "null_energy_observed": en, "null_energy_noise_expected": expected, "null_energy_excess": excess,
                        "total_energy_observed": et, "total_energy_noise_expected": total_expected,
                        "total_energy_excess": et-total_expected,
                        "null_excess_over_expected_noise": excess/expected if nnull else None,
                        "null_noise_only_energy_std": nv*np.sqrt(nnull) if nnull else 0.,
                        "null_noise_only_z_diagnostic": excess/(nv*np.sqrt(nnull)) if nnull else None,
                        "null_excess_fraction_of_total_excess": excess/(et-total_expected) if nnull and et>total_expected else None,
                    }
                    expected_null = P*R*N*(1-x.shape[1]/int(c["pilot_spacing"]))
                    if nnull != expected_null or not np.all(null.sum(axis=-1) == null[0].sum()):
                        raise RuntimeError("Joint pilot support differs from declared comb geometry")
                    if row["pilot_spacing"] == x.shape[1] and (nnull or en or excess):
                        raise RuntimeError("S=U joint-full-support identity failed")
                    rows.append(row)
                del raw, arrays, y, x, noise, masks, joint, energies, check_energy
    after = {e["relative_path"]: archive_stat(snapshot.path(e["relative_path"]))
             for entries in indices.values() for e in entries}
    if before != after or digest_bytes(args.inventory.read_bytes()) != digest_bytes(raw_inventory):
        raise RuntimeError("Source file metadata changed during read-only analysis")
    if max_partition_error > 1e-12 or max_double_crosscheck_error > 1e-12:
        raise RuntimeError("Independent float64 energy checks failed")
    if len(rows) != 384 or len(bindings) != 96:
        raise RuntimeError("Original calibration population changed")
    groups = make_groups(rows, args.bootstrap)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "per_frame_energy.csv", rows)
    write_csv(args.out / "grouped_energy.csv", groups)
    dump_json(args.out / "source_binding.json", {"inventory": str(args.inventory.resolve()),
        "inventory_sha256": digest_bytes(raw_inventory), "data_root": str(args.data_root),
        "source_files_before": before, "source_files_after": after,
        "all_source_files_metadata_unchanged": before==after,
        "batch_sources": bindings})
    dump_json(args.out / "grouped_energy.json", groups)
    if args.plots:
        plots(rows, groups, args.out)
    summary = {"analysis": "archived_raw_pilot_joint_null_energy", "role": "calibration mechanism diagnosis; no new blind-test claim",
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                        "numpy": np.__version__, "matplotlib": sys.modules["matplotlib"].__version__ if args.plots else None,
                        "thread_limit": 4},
        "source_inventory_sha256": digest_bytes(raw_inventory), "source_batches": len(bindings), "source_frames": len(rows),
        "frames_by_role": {role: sum(r["role"]==role for r in rows) for role in sorted({r["role"] for r in rows})},
        "formula": {"joint_support": "M[p,k]=any_u(x_pilot[u,p,k] != 0)",
            "null_observed_energy": "sum_(p,r,k:M[p,k]=0) |y_pilot[p,r,k]|^2",
            "null_expected_noise_energy": "noise_var * N_null_RE", "null_excess_energy": "observed - expected; no clipping",
            "noise_only_energy_std": "noise_var * sqrt(N_null_RE); diagnostic scale only, no p-value",
            "pooled_null_fraction": "sum null_excess_energy / sum all-bin_excess_energy (when denominator > 0)"},
        "interpretation_limits": [
            "Noise subtraction estimates aggregate received signal energy, not target-user-only energy.",
            "All users contribute; no per-user clean receive waveform is reconstructed.",
            "A negative excess energy is a valid finite-sample estimate and is retained.",
            "Off-joint-support energy is not by itself proof of target channel identifiability or lower network NMSE.",
            "Noise-only z is not a significance test when unknown channel signal or colored noise is present.",
            "Profile and Doppler mixtures are not balanced factorial experiments; group comparisons are descriptive.",
            "Bootstrap resamples existing 4-frame batches; it quantifies descriptive dataset variation, not unseen-test generalization.",
            "These float64 statistics use identical saved arrays irrespective of CPU/GPU; no GPU inference or latency is reproduced.",
        ],
        "checks": {"all_96_member_sha256_matched": all(b["sha256_matches"] for b in bindings),
            "source_files_metadata_unchanged": before==after,
            "max_energy_partition_relative_error": max_partition_error,
            "max_independent_float64_energy_relative_error": max_double_crosscheck_error,
            "max_float32_vs_float64_energy_relative_error": max_float32_reduction_relative_error,
            "s2_identity_frames": sum(r["pilot_spacing"]==2 for r in rows), "s2_null_energy_exactly_zero": all(r["null_energy_observed"]==0 for r in rows if r["pilot_spacing"]==2)},
        "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": args.bootstrap, "cluster_unit": "archived 4-frame observation batch"},
        "plots_generated": args.plots,
        "analysis_code_sha256": digest_bytes(Path(__file__).read_bytes()),
        "all_frames": aggregate(rows), "key_groups": [g for g in groups if g["group_axis"] in ("role", "spacing", "doppler_state_spacing")],
        "elapsed_seconds": time.perf_counter()-start,
    }
    dump_json(args.out / "summary.json", summary)
    print(json.dumps({"out": str(args.out.resolve()), "frames": len(rows), "batches": len(bindings), "checks": summary["checks"],
                      "elapsed_seconds": summary["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
