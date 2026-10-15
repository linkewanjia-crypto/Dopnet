"""Analyze historical 4090-saved representations without generation/inference.

The main 4.68 GB cache is verified by full SHA, descriptor hash,
29-field schema and ZIP layout before read-only mapping.
Only selected small samples are copied from mode='r' mappings. The original
saved masked sidecar is read from its published dependency and SHA-checked; it is never reconstructed from LS or simulated anew.
"""
from __future__ import annotations

import csv
from collections import defaultdict
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE
import read_only_tensor_cache as ro
DATA = ROOT / "data"
CACHE = DATA / "reproduction_inputs/validation_895338a79e73ad3f.pt"
MASKED = DATA / "reproduction_inputs/validation_895338a79e73ad3f_pilot_only.pt"

EXPECTED_MASKED_SHA = "ef5460dab53bb81b8a978b6c0f60c513609630ca147583d693163fbf8e85ed99"
NEEDED = ["add_input", "target_add", "add_input_slow", "pilot_spacing", "cp_len",
          "max_doppler_hz", "snr_db", "active_re_signal_power", "noise_var",
          "channel_type_id", "channel_backend_id", "modulation_id", "delay_spread_ns"]


def small_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def bind_main_cache():
    cache = ro.load_read_only_cache(CACHE)
    proof = cache.receipt
    for key in list(cache):
        if key not in NEEDED:
            cache.pop(key).close()
    return cache, {"path": str(CACHE), "historical_full_sha256": proof['source_sha256'],
                   "full_sha_recomputed": True, "source_metadata_before": proof['source_metadata'],
                   "descriptor_sha256": proof['metadata_pickle_sha256'],
                   "mapped_fields": {k:v for k,v in proof['fields'].items() if k in NEEDED},
                   "all_29_field_descriptors_verified": True, "mapping": "NumPy memmap mode=r"}

def bind_saved_masked():
    source = MASKED
    before = ro._metadata(source)
    if before['size_bytes'] != 184321999 or ro._hash_file(source) != EXPECTED_MASKED_SHA:
        raise ValueError("Original saved masked sidecar identity differs")
    with zipfile.ZipFile(source) as z:
        names = [s for s in z.namelist() if s.endswith('/data.pkl')]
        if len(names) != 1:
            raise ValueError("Ambiguous masked-sidecar descriptor")
        prefix = names[0][:-len('data.pkl')]
        raw = z.read(names[0])
        stream = io.BytesIO(raw)
        d = ro.DescriptorUnpickler(stream).load()
        shape = (2000,2,2,15,8,48)
        if (stream.tell() != len(raw) or type(d) is not ro.TensorDescriptor or d.shape != shape
                or d.storage.kind != ro.FLOAT_KIND or d.storage_offset != 0
                or d.stride != ro._contiguous_stride(shape) or d.storage.numel != math.prod(shape)
                or z.read(prefix+'byteorder') != b'little'):
            raise ValueError("Masked-sidecar restricted schema differs")
        info = z.getinfo(prefix+'data/'+d.storage.key)
        if info.file_size != math.prod(shape)*4:
            raise ValueError("Masked storage size differs")
        offset = ro._stored_payload_offset(source, info, z.start_dir)
    if ro._metadata(source) != before:
        raise ValueError("Masked source changed while binding")
    return ro.ReadOnlyTensorAccessor(source,d,offset), {
        "snapshot": str(source), "saved_gpu_sidecar_sha256": EXPECTED_MASKED_SHA,
        "whole_sha_checked": True, "source_snapshot_metadata_before": before,
        "shape": list(shape), "dtype": 'float32', "storage_payload_offset": offset,
        "descriptor_sha256": hashlib.sha256(raw).hexdigest(), "mapping_writeable": False,
        "LS_roundtrip_reconstruction": False, "new_waveform_generation": False}


def sample(accessor, i):
    # This task needs NumPy arithmetic only. Avoid loading Torch's separate
    # OpenMP runtime just to turn an already mapped NumPy sample into Torch
    # and back. Index strictly one first-axis sample and own its small copy.
    if not isinstance(i, int) or not 0 <= i < accessor.shape[0]:
        raise IndexError("Expected one valid first-axis frame index")
    result = np.array(accessor._map[i], copy=True, order="C")
    if not result.flags.owndata or not result.flags.writeable:
        raise RuntimeError("Sample must own its copy, independent of read-only source")
    return result


def squared(x):
    return float(np.square(x, dtype=np.float64).sum(dtype=np.float64))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in fields)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        total = lambda name: sum(r[name] for r in group)
        et = total("target_energy")
        ef = total("full_energy")
        rf, rm = total("full_target_residual_energy"), total("masked_target_residual_energy")
        output.append({**dict(zip(fields, key)), "frames": len({r["frame"] for r in group}),
                       "frame_user_rows": len(group), "target_energy_sum": et,
                       "full_energy_sum": ef, "extra_support_energy_sum": total("extra_support_energy"),
                       "extra_to_full_energy_ratio": total("extra_support_energy") / max(ef, 1e-30),
                       "full_residual_to_target_ratio": rf / max(et, 1e-30),
                       "masked_residual_to_target_ratio": rm / max(et, 1e-30),
                       "masked_minus_full_residual_to_target": (rm-rf) / max(et, 1e-30),
                       "fraction_full_closer_to_representation_target": sum(r["full_closer"] for r in group)/len(group),
                       "slow_residual_to_target_ratio": total("slow_target_residual_energy") / max(et, 1e-30),
                       "max_full_masked_abs_difference": max(r["max_full_masked_abs_difference"] for r in group),
                       "allclose_full_masked_fraction": sum(r["full_masked_allclose"] for r in group)/len(group),
                       "mean_snr_db": total("snr_db")/len(group),
                       "mean_noise_var": total("noise_var")/len(group),
                       "mean_active_re_signal_power": total("active_re_signal_power")/len(group)})
    return output


def make_plots(grouped_main, power_groups, out_summary):
    if importlib.util.find_spec("matplotlib") is None:
        out_summary["plots"] = "matplotlib unavailable; CSV and JSON remain complete"
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    for spacing in (2, 4, 8):
        points = sorted([r for r in grouped_main if r["pilot_spacing"] == spacing], key=lambda r:r["max_doppler_hz"])
        axes[0].plot([p["max_doppler_hz"] for p in points], [p["extra_to_full_energy_ratio"] for p in points], "o-", label=f"S={spacing}")
        axes[1].plot([p["max_doppler_hz"] for p in points], [p["masked_minus_full_residual_to_target"] for p in points], "o-", label=f"S={spacing}")
    axes[0].set_ylabel("Sum extra-support energy / sum full ADD energy")
    axes[1].set_ylabel("(Masked residual - full residual) / target energy")
    for ax in axes:
        ax.set_xlabel("Declared maximum Doppler (Hz)")
        ax.grid(alpha=.25)
        ax.legend()
    axes[1].axhline(0, color="k", linewidth=.7)
    fig.suptitle("Saved 2,000-frame validation representations; no model execution")
    fig.savefig(OUT / "saved_representation_support_and_residual.png")
    fig.savefig(OUT / "saved_representation_support_and_residual.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    for spacing in (2, 4, 8):
        points = sorted([r for r in power_groups if r["pilot_spacing"] == spacing], key=lambda r:r["power_quartile"])
        ax.plot([p["power_quartile"] for p in points], [p["full_residual_to_target_ratio"] for p in points], "o-", label=f"S={spacing}")
    ax.set(xlabel="Empirical active received-power quartile", ylabel="Sum full-target residual energy / sum target energy", title="Other-user + noise residual; descriptive mixture groups")
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(OUT / "representation_residual_by_received_power.png")
    plt.close(fig)
    out_summary["plots"] = ["saved_representation_support_and_residual.png", "saved_representation_support_and_residual.pdf", "representation_residual_by_received_power.png"]


def main():
    import argparse
    global OUT, DATA, CACHE, MASKED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    DATA = args.data_root.resolve()
    CACHE = DATA / 'reproduction_inputs/validation_895338a79e73ad3f.pt'
    MASKED = DATA / 'reproduction_inputs/validation_895338a79e73ad3f_pilot_only.pt'
    OUT = args.output_dir.resolve()
    if OUT == DATA or OUT.is_relative_to(DATA) or DATA.is_relative_to(OUT):
        parser.error('Output and data trees must be separate.')
    if OUT.exists() and any(OUT.iterdir()):
        parser.error('Output must be new or empty.')
    OUT.mkdir(parents=True, exist_ok=True)
    cache, main_binding = bind_main_cache()
    masked, masked_binding = bind_saved_masked()
    provenance = {"analysis_source_sha256": small_sha(__file__), "main_cache": main_binding,
                  "masked_sidecar": masked_binding, "data_layout": "public support diagnostics 2026.10.07",
                  "restricted_reader_sha256": small_sha(ro.__file__), "python": platform.python_version(),
                  "numpy": np.__version__, "device": "CPU", "neural_network_inference": False,
                  "training": False, "new_observations": False, "scientific_timing_claim": False}
    write_json(OUT / "source_binding.json", provenance)
    metadata = {}
    # active_re_signal_power is each frame's noise-free data-RE power
    # (waveform.py means over receive/data-symbol/subcarrier axes). It is
    # NOT the chunk-averaged P_ref used by _noise_variance to set noise.
    for field in NEEDED:
        if field not in ("add_input", "target_add", "add_input_slow"):
            metadata[field] = np.array([sample(cache[field], i).item() for i in range(2000)])
    power_cutoffs = np.quantile(metadata["active_re_signal_power"], [.25, .5, .75])
    rows, qrows = [], []
    formula_error_max = 0.0
    extra_path = OUT / "extra_support_full_minus_saved_masked.npy"
    # Save the actual derived tensor as well as scalar summaries. Float64
    # retains the same subtraction precision used by the energy analysis.
    extra_map = np.lib.format.open_memmap(extra_path, mode="w+", dtype="<f8",
                                          shape=(2000, 2, 2, 15, 8, 48))
    extra_sha = hashlib.sha256()
    with extra_path.open("rb") as stream:
        extra_sha.update(stream.read(extra_map.offset))
    for i in range(2000):
        full = sample(cache["add_input"], i)
        target = sample(cache["target_add"], i)
        slow = sample(cache["add_input_slow"], i)
        mask = sample(masked, i)
        if not all(np.isfinite(a).all() for a in (full, target, slow, mask)):
            raise RuntimeError(f"Nonfinite original representation at frame {i}")
        for user in range(2):
            f, t, m, s = (a[user].astype(np.float64) for a in (full, target, mask, slow))
            extra, rf, rm = f-m, f-t, m-t
            extra_map[i, user] = extra
            extra_sha.update(extra.astype("<f8", copy=False).tobytes(order="C"))
            ex, erf, erm = squared(extra), squared(rf), squared(rm)
            cross = float(np.multiply(rm, extra).sum(dtype=np.float64))
            formula_error = abs(erf-(erm+ex+2*cross))/max(erf+erm+ex, 1e-30)
            formula_error_max = max(formula_error_max, formula_error)
            base = {"frame": i, "user": user, **{k:float(v[i]) for k,v in metadata.items()},
                    "power_quartile": int(np.searchsorted(power_cutoffs, metadata["active_re_signal_power"][i], side="right"))+1,
                    "snr_band": "5_to_15" if metadata["snr_db"][i] < 15 else "15_to_25" if metadata["snr_db"][i] < 25 else "25_to_35",
                    "full_energy": squared(f), "target_energy": squared(t), "masked_energy": squared(m),
                    "extra_support_energy": ex, "full_target_residual_energy": erf,
                    "masked_target_residual_energy": erm, "slow_target_residual_energy": squared(s-t),
                    "masked_minus_full_residual_energy": erm-erf, "masked_residual_extra_inner_product_real": cross,
                    "full_closer": int(erf < erm), "max_full_masked_abs_difference": float(np.abs(extra).max()),
                    "relative_full_masked_l2": math.sqrt(ex/max(squared(f),1e-30)),
                    "full_masked_exact_equal": int(np.array_equal(full[user],mask[user])),
                    "full_masked_allclose": int(np.allclose(full[user],mask[user],rtol=1e-4,atol=1e-5)),
                    "residual_energy_identity_relative_error": formula_error}
            rows.append(base)
            for qi in range(15):
                qrows.append({"frame":i,"user":user,"q":qi-7,"pilot_spacing":int(metadata["pilot_spacing"][i]),
                              "max_doppler_hz":float(metadata["max_doppler_hz"][i]),
                              "full_energy":squared(f[:,qi]), "extra_support_energy":squared(extra[:,qi]),
                              "target_energy":squared(t[:,qi]), "full_target_residual_energy":squared(rf[:,qi]),
                              "masked_target_residual_energy":squared(rm[:,qi])})
        if i % 250 == 0:
            print(f"processed {i}/2000 saved frames", flush=True)
    extra_map.flush()
    extra_map._mmap.close()
    provenance["derived_extra_support"] = {
        "path":str(extra_path), "dtype":"float64", "shape":[2000,2,2,15,8,48],
        "formula":"float64(saved add_input) - float64(saved GPU masked sidecar)",
        "sha256_npy_bytes_computed_in_write_order":extra_sha.hexdigest(),
        "size_bytes":extra_path.stat().st_size, "new_receiver_observations":False,
    }
    write_csv(OUT / "per_frame_user_representation_metrics.csv", rows)
    write_csv(OUT / "per_frame_user_q_metrics.csv", qrows)
    specs = {"by_spacing_doppler":["pilot_spacing","max_doppler_hz"],
             "by_spacing_cp":["pilot_spacing","cp_len"],
             "by_spacing_power":["pilot_spacing","power_quartile"],
             "by_spacing_snr":["pilot_spacing","snr_band"],
             "by_spacing_channel":["pilot_spacing","channel_type_id"],
             "by_spacing_doppler_cp":["pilot_spacing","max_doppler_hz","cp_len"]}
    tables = {}
    for filename, fields in specs.items():
        tables[filename] = grouped(rows, fields)
        write_csv(OUT / (filename+".csv"), tables[filename])
    s2 = [r for r in rows if r["pilot_spacing"] == 2]
    s2_receipt = {"mathematical_identity":"M=I at S=2, hence T_f Y = T_f M Y for the same waveform",
                  "cached_artifact_check":"Original full ADD versus saved direct-regeneration masked sidecar",
                  "frames":len(s2)//2, "frame_user_rows":len(s2), "core_rtol":1e-4,"core_atol":1e-5,
                  "exact_equal_rows":sum(r["full_masked_exact_equal"] for r in s2),
                  "allclose_rows":sum(r["full_masked_allclose"] for r in s2),
                  "allclose_all":all(r["full_masked_allclose"] for r in s2),
                  "max_absolute_difference":max(r["max_full_masked_abs_difference"] for r in s2),
                  "relative_l2_quantiles":{str(q):float(np.quantile([r["relative_full_masked_l2"] for r in s2],q)) for q in [0,.5,.95,1]},
                  "scope":"Front-end identity only; no assertion of equal independently trained checkpoint outputs"}
    write_json(OUT / "spacing2_identity_check.json", s2_receipt)
    overall = grouped(rows, [])[0]
    summary = {"frames":2000,"frame_user_rows":4000,"q_rows":len(qrows),
               "population":"Historical mixed validation set; not nominal test grid or independent training runs",
               "overall":overall,"spacing2":s2_receipt,"power_quartile_cutoffs":power_cutoffs.tolist(),
               "power_variable_definition":"Per-frame noise-free received data-RE power; not the chunk-averaged noise-normalization reference",
               "residual_energy_identity_max_relative_error":formula_error_max,
               "statistics":"Descriptive ratios of energy sums; no confidence intervals or causal comparison of mixture subgroups",
               "interpretation":[
                   "full-target residual includes other-user pilot contribution plus noise, not pure noise",
                   "full-minus-masked is the saved front-end support difference, not a model prediction or clean signal decomposition",
                   "Closer to isolated-user representation target does not establish downstream CFR NMSE or BER improvement",
                   "Power/spacing/Doppler groups have different mixtures; their descriptive associations are not controlled causal effects",
                   "No inference, training, new generation, receiver reconfiguration or performance timing was performed"],
               "channel_id_names":dict(enumerate(["awgn","rayleigh","rician","nakagami","doubly_selective","tdl_a","tdl_b","tdl_c","tdl_d","tdl_e","cdl_a","cdl_b","cdl_c","cdl_d","cdl_e"]))}
    if args.plots:
        make_plots(tables["by_spacing_doppler"], tables["by_spacing_power"], summary)
    after = ro._metadata(CACHE)
    snapshot_after = ro._metadata(Path(masked_binding["snapshot"]))
    if after != main_binding["source_metadata_before"] or snapshot_after != masked_binding["source_snapshot_metadata_before"]:
        raise RuntimeError("Original scientific source metadata changed")
    summary["source_metadata_unchanged"] = True
    provenance["main_cache"]["source_metadata_after"] = after
    provenance["masked_sidecar"]["source_snapshot_metadata_after"] = snapshot_after
    write_json(OUT / "source_binding.json", provenance)
    write_json(OUT / "analysis_summary.json", summary)
    masked.close()
    for accessor in cache.values():
        accessor.close()
    print(json.dumps({"status":"complete","frames":2000,"spacing2_allclose":s2_receipt["allclose_all"],"spacing2_max_abs":s2_receipt["max_absolute_difference"],"source_metadata_unchanged":True,"output":str(OUT)},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
