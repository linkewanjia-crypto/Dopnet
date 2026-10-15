"""PWB feature-energy diagnosis on exactly 384 archived pilot frames.

No training, neural network inference, new frames or slow-time implementation.
The nonorthogonal PWB decomposition retains its masked/extra cross term.
"""
from __future__ import annotations

from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OPERATOR_SOURCE = HERE / 'operator_diagnostics.py'
import operator_diagnostics as ops
import archived_inputs as inputs

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
import time
import zipfile
import numpy as np



def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


SUM_FIELDS = ("full_feature_energy", "masked_feature_energy", "extra_feature_energy", "feature_cross_term",
              "full_noise_expected_energy", "masked_noise_expected_energy", "extra_noise_expected_energy",
              "full_feature_excess_energy", "masked_feature_excess_energy", "extra_feature_excess_energy")


def stats(rows):
    sums = {name: sum(r[name] for r in rows) for name in SUM_FIELDS}
    nf, ne = sums["full_noise_expected_energy"], sums["extra_noise_expected_energy"]
    ef, ee = sums["full_feature_excess_energy"], sums["extra_feature_excess_energy"]
    return {"frames": len(rows), "batches": len({r["batch_id"] for r in rows}), **sums,
        "extra_noise_fraction": ne/nf if nf else None,
        "extra_excess_over_extra_noise": ee/ne if ne else None,
        "full_excess_over_full_noise": ef/nf if nf else None,
        "extra_excess_over_full_excess": ee/ef if ef > 0 else None,
        "cross_over_full_excess": sums["feature_cross_term"]/ef if ef > 0 else None,
        "negative_full_excess_frames": sum(r["full_feature_excess_energy"] < 0 for r in rows),
        "negative_masked_excess_frames": sum(r["masked_feature_excess_energy"] < 0 for r in rows),
        "negative_extra_excess_frames": sum(r["extra_feature_excess_energy"] < 0 for r in rows),
    }


def ci(rows, rng, samples):
    batches = defaultdict(list)
    for r in rows:
        batches[r["batch_id"]].append(r)
    metrics = (("extra_excess_over_extra_noise", "extra_feature_excess_energy", "extra_noise_expected_energy"),
               ("extra_excess_over_full_excess", "extra_feature_excess_energy", "full_feature_excess_energy"),
               ("cross_over_full_excess", "feature_cross_term", "full_feature_excess_energy"))
    result = {}
    fields = tuple(sorted({f for _, a, b in metrics for f in (a, b)}))
    matrix = np.array([[sum(r[f] for r in part) for f in fields] for part in batches.values()])
    sample = matrix[rng.integers(0, len(matrix), size=(samples, len(matrix)))].sum(axis=1) if len(matrix) >= 2 else None
    for key, num, den in metrics:
        result[key+"_ci95_low"] = result[key+"_ci95_high"] = None
        result[key+"_valid_bootstrap_draws"] = 0
        if sample is None:
            continue
        denominator = sample[:, fields.index(den)]
        good = denominator > 0
        values = sample[good, fields.index(num)]/denominator[good]
        result[key+"_valid_bootstrap_draws"] = len(values)
        if len(values):
            a, b = np.percentile(values, [2.5, 97.5])
            result[key+"_ci95_low"], result[key+"_ci95_high"] = float(a), float(b)
    return result


def group_stats(rows, samples):
    rng = np.random.default_rng(20261008)
    all_groups = []
    for label, fields in (("all", ()), ("role", ("role",)), ("spacing", ("pilot_spacing",)),
                         ("doppler_state_spacing", ("doppler_state", "pilot_spacing")),
                         ("profile", ("profile",)), ("doppler", ("doppler_hz",)),
                         ("snr", ("snr_bin",))):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[f] for f in fields)].append(row)
        for key, part in sorted(groups.items(), key=lambda item: str(item[0])):
            all_groups.append({"group_axis": label, "group_key": json.dumps(dict(zip(fields,key)), sort_keys=True),
                               **stats(part), **ci(part,rng,samples)})
    return all_groups


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output-dir", dest="out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=inputs.DATA_ROOT)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    inputs.configure(args.data_root)
    args.out = inputs.new_output(args.out, inputs.DATA_ROOT)
    start = time.perf_counter()
    cfg_path = ops.PROJECT / "configs/dopnet.yaml"
    cfg = ops.yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    inventory_path = inputs.DATA_ROOT / "inventory.json"
    info_raw = inventory_path.read_bytes()
    info = json.loads(info_raw)
    source_paths = [(inputs.DATA_ROOT / b['relative_path']).resolve() for b in info['batches']]
    def source_stats():
        return {p.relative_to(inputs.DATA_ROOT).as_posix(): {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in source_paths}
    source_before = source_stats()
    code_bindings = {str(p.resolve()): sha(p.read_bytes()) for p in (
        Path(__file__), OPERATOR_SOURCE, cfg_path, ops.PROJECT/"data/waveform.py", ops.PROJECT/"data/add_geometry.py")}
    rows, source_bindings, reference_checks = [], [], []
    max_feature_partition_error = max_noise_partition_error = max_excess_partition_error = max_extra_direct_error = 0.
    reference_batch = None
    for population in dict.fromkeys(b['population'] for b in info['batches']):
        for entry in (b for b in info['batches'] if b['population'] == population):
            path = (inputs.DATA_ROOT / entry['relative_path']).resolve()
            if not path.is_relative_to(inputs.DATA_ROOT):
                raise ValueError("Observation path escapes input tree")
            archive_path, member = entry['source_archive'], entry['archive_member']
            entry = dict(entry, expected_sha256=entry['sha256'])
            raw = path.read_bytes()
            digest = sha(raw)
            if digest != entry["expected_sha256"] or digest != entry["sha256"]:
                raise RuntimeError("Archived observation digest mismatch")
            with np.load(io.BytesIO(raw), allow_pickle=False) as a:
                x = a["x_pilot"]
                y = a["y_pilot"]
                noise = a["noise_var"]
                cp_wave = int(a["cp_wave_samples"])
                metadata = json.loads(str(a["metadata"]))
            batch_id = f"{population}:{entry['original_path']}"
            if reference_batch is None:
                reference_batch = batch_id
            source_bindings.append({"batch_id": batch_id, "archive": archive_path, "archive_member": member,
                "expected_sha256": entry["expected_sha256"], "actual_sha256": digest, "sha256_matches": True,
                "arrays_read": ["x_pilot", "y_pilot", "noise_var", "cp_wave_samples", "metadata"], "frames": len(y)})
            c = metadata["condition"]
            for i in range(len(y)):
                t0 = time.perf_counter()
                t, pre, r = ops.fixed_map(cfg, x[i], cp_wave, True)
                mask = (x[i] != 0).any(axis=0).reshape(-1)
                y64 = y[i].astype(np.complex128)
                y_masked = y64 * mask.reshape(7,1,128)
                y_extra = y64 - y_masked
                zf = ops.apply_dense(t, y64)
                zm = ops.apply_dense(t, y_masked)
                ze = zf - zm
                if batch_id == reference_batch:
                    direct_extra = ops.apply_dense(t, y_extra)
                    rel = float(np.linalg.norm(ze-direct_extra)/max(np.linalg.norm(zf),1e-300))
                    max_extra_direct_error = max(max_extra_direct_error, rel)
                energies = {"full": float(np.vdot(zf,zf).real), "masked": float(np.vdot(zm,zm).real),
                            "extra": float(np.vdot(ze,ze).real)}
                cross = float(2*np.vdot(zm,ze).real)
                coeff = np.abs(t)**2
                nf = float(coeff.sum(dtype=np.float64))
                nm = float(coeff[:,mask].sum(dtype=np.float64))
                ne = float(coeff[:,~mask].sum(dtype=np.float64))
                expected = {key: value*float(noise[i])*y.shape[2] for key,value in (("full",nf),("masked",nm),("extra",ne))}
                excess = {key: energies[key]-expected[key] for key in energies}
                energy_err = abs(energies["full"]-energies["masked"]-energies["extra"]-cross)/max(energies["full"],1e-300)
                noise_err = abs(expected["full"]-expected["masked"]-expected["extra"])/max(expected["full"],1e-300)
                excess_err = abs(excess["full"]-excess["masked"]-excess["extra"]-cross)/max(energies["full"]+expected["full"],1e-300)
                max_feature_partition_error = max(max_feature_partition_error,energy_err)
                max_noise_partition_error = max(max_noise_partition_error,noise_err)
                max_excess_partition_error = max(max_excess_partition_error,excess_err)
                if batch_id == reference_batch:
                    for label, z, masked_flag in (("full",zf,False),("masked",zm,True)):
                        ref = ops.fft_reference(cfg,x[i],y[i],cp_wave,True,masked_flag)
                        error = float(np.linalg.norm(z-ref)/max(np.linalg.norm(ref),1e-300))
                        reference_checks.append({"batch_id":batch_id,"frame_offset":int(metadata["offset"])+i,
                            "label":label,"relative_error":error,"maximum_absolute_error":float(np.abs(z-ref).max()),
                            "relative_tolerance":ops.REFERENCE_RELATIVE_TOLERANCE,"pass":error<=ops.REFERENCE_RELATIVE_TOLERANCE})
                snr = float(metadata["snr_db"][i])
                binlo = int(np.floor(snr/10))*10
                row = {"population":population,"role":metadata["role"],"batch_id":batch_id,
                    "frame_offset":int(metadata["offset"])+i,"source_sha256":digest,"profile":c["channel_type"],
                    "doppler_hz":float(c["doppler_hz"]),"doppler_state":"static" if float(c["doppler_hz"])==0 else "dynamic",
                    "pilot_spacing":int(c["pilot_spacing"]),"snr_db":snr,"snr_bin":f"[{binlo},{binlo+10})",
                    "cp_wave_samples":cp_wave,"noise_var":float(noise[i]),"receivers":y.shape[2],
                    "full_map_frobenius_squared":nf,"masked_map_frobenius_squared":nm,"extra_map_frobenius_squared":ne,
                    **{f"{label}_feature_energy":value for label,value in energies.items()},"feature_cross_term":cross,
                    **{f"{label}_noise_expected_energy":value for label,value in expected.items()},
                    "cross_noise_expected_energy":0.,
                    **{f"{label}_feature_excess_energy":value for label,value in excess.items()},
                    "extra_noise_fraction":ne/nf,"extra_excess_over_extra_noise":excess["extra"]/expected["extra"] if expected["extra"] else None,
                    "feature_partition_relative_error":energy_err,"noise_partition_relative_error":noise_err,
                    "excess_partition_relative_error":excess_err,
                }
                if not np.isfinite(list(energies.values())+list(expected.values())+list(excess.values())+[cross]).all():
                    raise RuntimeError("Non-finite diagnostic result")
                if int(c["pilot_spacing"]) == 2 and (energies["extra"] != 0 or ne != 0 or cross != 0):
                    raise RuntimeError("S=2 PWB full/masked identity failed")
                rows.append(row)
                del t, pre, r, coeff, y64, y_masked, y_extra, zf, zm, ze
            del raw, x, y, noise
            if len(rows) % 32 == 0:
                print(f"Processed {len(rows)}/384 frames in {time.perf_counter()-start:.1f}s",flush=True)
    source_after = source_stats()
    if source_before != source_after or sha(inventory_path.read_bytes()) != sha(info_raw):
        raise RuntimeError("Read-only source integrity changed")
    if any(value>1e-12 for value in (max_feature_partition_error,max_noise_partition_error,max_excess_partition_error,max_extra_direct_error)):
        raise RuntimeError("Energy partition check failed")
    if not all(c["pass"] for c in reference_checks):
        raise RuntimeError("Original full/masked FFT implementation agreement failed")
    groups = group_stats(rows,args.bootstrap)
    write_csv(args.out/"per_frame_pwb_energy.csv",rows)
    write_csv(args.out/"grouped_pwb_energy.csv",groups)
    write_json(args.out/"grouped_pwb_energy.json",groups)
    write_json(args.out/"source_binding.json",{"inventory_sha256":sha(info_raw),"source_code_sha256":code_bindings,
        "source_archives_before":source_before,"source_archives_after":source_after,"batch_sources":source_bindings})
    write_json(args.out/"fft_reference_checks.json",reference_checks)
    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
        selected = [g for g in groups if g["group_axis"]=="doppler_state_spacing" and json.loads(g["group_key"])["pilot_spacing"]>2]
        fig,axes = plt.subplots(1,2,figsize=(9,3.8),constrained_layout=True)
        labels = []
        for i,g in enumerate(selected):
            key=json.loads(g["group_key"])
            labels.append(f"{key['doppler_state']}\nS={key['pilot_spacing']}\nn={g['frames']}")
            color="#D55E00" if key["doppler_state"]=="dynamic" else "#0072B2"
            value=g["extra_excess_over_extra_noise"]
            low,high=g["extra_excess_over_extra_noise_ci95_low"],g["extra_excess_over_extra_noise_ci95_high"]
            axes[0].bar(i,value,color=color,width=.65)
            if low is not None:
                axes[0].errorbar(i,value,yerr=[[max(0,value-low)],[max(0,high-value)]],color="black",capsize=3,fmt="none")
            axes[1].bar(i,100*g["extra_noise_fraction"],color=color,width=.65)
        for ax in axes:
            ax.set_xticks(range(len(selected)),labels)
            ax.axhline(0,color="black",linewidth=.8)
        axes[0].set(ylabel="Extra PWB excess / its expected noise",title="Feature energy; batch-bootstrap 95% CI")
        axes[1].set(ylabel="Extra / full expected PWB noise (%)",title="Conditional noise trace; fixed saved pilots")
        fig.savefig(args.out/"pwb_extra_signal_noise.png",dpi=240)
        fig.savefig(args.out/"pwb_extra_signal_noise.pdf")
        plt.close(fig)
    summary={"analysis":"archived PWB feature-energy decomposition","source_frames":len(rows),"source_batches":len(source_bindings),
        "no_training":True,"no_neural_inference":True,"no_new_frames":True,
        "precision":"complex128 matrix/FFT; original phase and preconditioner parameters quantized to complex64",
        "formula":{"pwb":"z_full=T y; z_mask=T M y; z_extra=T(I-M)y",
            "feature_energy":"||z||_2^2; sum over users,Doppler,angle,delay",
            "expected_noise_energy":"Nr * noise_var * ||T_support||_F^2; spatial FFT unitary",
            "cross_term":"2 Re <z_mask,z_extra>","noise_cross_expectation":0,
            "feature_partition":"E_full=E_mask+E_extra+cross",
            "excess_partition":"excess_full=excess_mask+excess_extra+cross; negatives retained"},
        "checks":{"all_member_sha256_match":True,"source_archives_metadata_unchanged":source_before==source_after,
            "max_feature_partition_relative_error":max_feature_partition_error,"max_noise_partition_relative_error":max_noise_partition_error,
            "max_excess_partition_relative_error":max_excess_partition_error,"max_extra_direct_relative_error":max_extra_direct_error,
            "reference_batch_frames":4,"reference_comparisons":len(reference_checks),
            "all_full_masked_fft_reference_checks_pass":all(c["pass"] for c in reference_checks),
            "maximum_reference_relative_error":max(c["relative_error"] for c in reference_checks),
            "reference_relative_tolerance":ops.REFERENCE_RELATIVE_TOLERANCE,
            "s2_exact_full_masked_identity_frames":sum(r["pilot_spacing"]==2 for r in rows)},
        "interpretation_limits":[
            "PWB feature norm is not raw physical receive energy; preconditioning alters its scale and correlations.",
            "Extra feature energy is not E_full-E_mask; the cross term is required and is retained.",
            "Noise expectations are conditional on the saved noise_var and white received-bin noise model.",
            "Excess feature energy describes aggregate received signals from all users, not a target-user clean input.",
            "No SVD or whitening rank truncation is used in this empirical feature-energy diagnosis.",
            "Calibration/statistical-calibration diagnostics are not a new blind-test performance result.",
            "Bootstrap varies existing 4-frame batches and does not establish unseen-channel generalization.",
            "Agreement is checked against original CPU complex64 FFT at 1e-4 relative tolerance; no GPU execution is claimed."],
        "bootstrap":{"resamples":args.bootstrap,"seed":20261008,"cluster":"original 4-frame archived batch"},
        "key_groups":[g for g in groups if g["group_axis"] in ("role","spacing","doppler_state_spacing")],
        "elapsed_seconds":time.perf_counter()-start}
    write_json(args.out/"summary.json",summary)
    print(json.dumps({"out":str(args.out.resolve()),"checks":summary["checks"],"elapsed_seconds":summary["elapsed_seconds"]},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
