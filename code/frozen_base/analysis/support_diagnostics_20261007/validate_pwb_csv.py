"""Portable scalar and bootstrap verifier for saved raw PWB energy CSV.

Usage:
    python validate_from_csv.py --data-root DATA/raw_pwb_energy --output-dir OUTPUT

Runtime dependencies: Python 3.10+ and NumPy 1.22+.
Reads three saved scalar files, with no waveform/operator construction or inference.
The original fsum aggregation and multiplicity bootstrap equations are retained.
"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from collections import OrderedDict
import numpy as np

INPUT_NAMES = ("per_frame_pwb_energy.csv", "grouped_pwb_energy.json", "summary.json")
CATEGORICAL_FIELDS = {"population", "role", "batch_id", "source_sha256", "profile", "doppler_state", "snr_bin"}
REQUIRED_FIELDS = ("population", "role", "batch_id", "frame_offset", "source_sha256", "profile",
                   "doppler_hz", "doppler_state", "pilot_spacing", "snr_db", "snr_bin", "cp_wave_samples",
                   "noise_var", "receivers", "full_map_frobenius_squared", "masked_map_frobenius_squared",
                   "extra_map_frobenius_squared", "full_feature_energy", "masked_feature_energy",
                   "extra_feature_energy", "feature_cross_term", "full_noise_expected_energy",
                   "masked_noise_expected_energy", "extra_noise_expected_energy", "cross_noise_expected_energy",
                   "full_feature_excess_energy", "masked_feature_excess_energy", "extra_feature_excess_energy",
                   "extra_noise_fraction", "extra_excess_over_extra_noise", "feature_partition_relative_error",
                   "noise_partition_relative_error", "excess_partition_relative_error")
SUM_FIELDS = ("full_feature_energy", "masked_feature_energy", "extra_feature_energy", "feature_cross_term",
              "full_noise_expected_energy", "masked_noise_expected_energy", "extra_noise_expected_energy",
              "full_feature_excess_energy", "masked_feature_excess_energy", "extra_feature_excess_energy")
METRICS = (("extra_excess_over_extra_noise", "extra_feature_excess_energy", "extra_noise_expected_energy"),
           ("extra_excess_over_full_excess", "extra_feature_excess_energy", "full_feature_excess_energy"),
           ("cross_over_full_excess", "feature_cross_term", "full_feature_excess_energy"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scaled_error(a,b):
    return abs(a-b)/max(1.,abs(a),abs(b))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path, help="Directory containing the three saved raw PWB scalar input files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Separate output directory; must not overlap the input tree.")
    return parser.parse_args()


def resolved_paths(args):
    data_root = args.data_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if not data_root.is_dir():
        raise ValueError("--data-root must be an existing directory")
    if output_dir == data_root or data_root in output_dir.parents or output_dir in data_root.parents:
        raise ValueError("Input and output directory trees must be disjoint; refusing output that could overwrite input")
    inputs = tuple(data_root / name for name in INPUT_NAMES)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required scalar input: {path}")
        resolved_input = path.resolve(strict=True)
        if resolved_input == output_dir or output_dir in resolved_input.parents:
            raise ValueError("An input resolves into the output tree; refusing output that could overwrite input")
    result_path = output_dir / "validation_from_csv.json"
    resolved_result = result_path.resolve()
    if resolved_result == data_root or data_root in resolved_result.parents:
        raise ValueError("The result path resolves into the input tree; refusing input overwrite")
    for path in inputs:
        if resolved_result == path.resolve(strict=True) or (result_path.exists() and result_path.samefile(path)):
            raise ValueError("The result path aliases an input file; refusing input overwrite")
    return data_root, output_dir, inputs


def main():
    data_root, output_dir, inputs = resolved_paths(parse_args())
    CSV, GROUPS, SUMMARY = inputs
    input_hashes_before = {p.name: digest(p) for p in inputs}
    with CSV.open(newline="",encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
            raise ValueError("CSV schema differs from the recorded 33-field raw PWB scalar schema")
        rawrows = list(reader)
    rows=[]
    for row in rawrows:
        rows.append({k:v if k in CATEGORICAL_FIELDS else (float(v) if v else None) for k,v in row.items()})
    groups=json.loads(GROUPS.read_text(encoding="utf-8"))
    summary=json.loads(SUMMARY.read_text(encoding="utf-8"))
    rng=np.random.default_rng(summary["bootstrap"]["seed"])
    resamples=summary["bootstrap"]["resamples"]
    max_noise_scale_error=max_feature_partition_error=max_noise_partition_error=max_excess_definition_error=max_excess_partition_error=0.
    s2_identity=[]
    for row in rows:
        for label in ("full","masked","extra"):
            expected=float(row["receivers"])*row["noise_var"]*row[label+"_map_frobenius_squared"]
            observed=row[label+"_noise_expected_energy"]
            max_noise_scale_error=max(max_noise_scale_error,scaled_error(expected,observed))
            expected=row[label+"_feature_energy"]-row[label+"_noise_expected_energy"]
            observed=row[label+"_feature_excess_energy"]
            max_excess_definition_error=max(max_excess_definition_error,scaled_error(expected,observed))
        ef,em,ee=[row[x+"_feature_energy"] for x in ("full","masked","extra")]
        nf,nm,ne=[row[x+"_noise_expected_energy"] for x in ("full","masked","extra")]
        xf,xm,xe=[row[x+"_feature_excess_energy"] for x in ("full","masked","extra")]
        cross=row["feature_cross_term"]
        max_feature_partition_error=max(max_feature_partition_error,abs(ef-math.fsum([em,ee,cross]))/max(1.,abs(ef)))
        max_noise_partition_error=max(max_noise_partition_error,abs(nf-math.fsum([nm,ne]))/max(1.,abs(nf)))
        max_excess_partition_error=max(max_excess_partition_error,abs(xf-math.fsum([xm,xe,cross]))/max(1.,abs(ef)+abs(nf)))
        if row["cross_noise_expected_energy"]!=0:
            raise RuntimeError("Disjoint input-noise cross expectation is not zero")
        if row["pilot_spacing"]==2:
            s2_identity.append(ef==em and ee==0 and ne==0 and cross==0)
    max_sum_error=max_ratio_error=max_ci_error=max_ci_abs_error=0.
    matching_counts=True
    per_group=[]
    bootstrap_fields=tuple(sorted({x for _,a,b in METRICS for x in (a,b)}))
    for group in groups:
        criteria=json.loads(group["group_key"])
        part=[r for r in rows if all(r[k]==v for k,v in criteria.items())]
        batches=OrderedDict()
        for row in part:
            batches.setdefault(row["batch_id"],[]).append(row)
        sums={field:math.fsum(r[field] for r in part) for field in SUM_FIELDS}
        for field,total in sums.items():
            max_sum_error=max(max_sum_error,scaled_error(total,group[field]))
        matching_counts &= len(part)==group["frames"] and len(batches)==group["batches"]
        for label in ("full","masked","extra"):
            matching_counts &= sum(r[label+"_feature_excess_energy"]<0 for r in part)==group["negative_"+label+"_excess_frames"]
        ratios={"extra_noise_fraction":sums["extra_noise_expected_energy"]/sums["full_noise_expected_energy"],
                "full_excess_over_full_noise":sums["full_feature_excess_energy"]/sums["full_noise_expected_energy"]}
        for metric,num,den in METRICS:
            ratios[metric]=sums[num]/sums[den] if sums[den]>0 else None
        for metric,value in ratios.items():
            if value is None or group[metric] is None:
                if value is not None or group[metric] is not None:
                    raise RuntimeError("Nullable aggregate ratio differs")
            else:
                max_ratio_error=max(max_ratio_error,scaled_error(value,group[metric]))
        # Reproduce draws, but compute their sums using multiplicity dot products.
        contributions=np.array([[math.fsum(r[f] for r in br) for f in bootstrap_fields] for br in batches.values()],dtype=np.float64)
        if len(batches)>=2:
            indices=rng.integers(0,len(batches),size=(resamples,len(batches)))
            counts=np.stack([np.bincount(draw,minlength=len(batches)) for draw in indices])
            bootstrap=counts@contributions
        else:
            bootstrap=None
        comparison={"group_axis":group["group_axis"],"group_key":group["group_key"],"frames":len(part),"batches":len(batches)}
        for metric,num,den in METRICS:
            low=high=None
            valid=0
            if bootstrap is not None:
                denominator=bootstrap[:,bootstrap_fields.index(den)]
                good=denominator>0
                values=bootstrap[good,bootstrap_fields.index(num)]/denominator[good]
                valid=len(values)
                if valid:
                    # The independent multiplicity reduction feeds quantiles.
                    low,high=[float(v) for v in np.quantile(values,[.025,.975],method="linear")]
            matching_counts &= valid==group[metric+"_valid_bootstrap_draws"]
            for suffix,value in (("_ci95_low",low),("_ci95_high",high)):
                stored=group[metric+suffix]
                if value is None or stored is None:
                    if value is not None or stored is not None:
                        raise RuntimeError("Nullable bootstrap quantile differs")
                else:
                    max_ci_error=max(max_ci_error,scaled_error(value,stored))
                    max_ci_abs_error=max(max_ci_abs_error,abs(value-stored))
            comparison[metric]={"ratio_recomputed":ratios[metric],"ci95_recomputed":[low,high],"valid_draws":valid}
        per_group.append(comparison)
    thresholds={"scalar_relative_or_scaled":1e-12,"bootstrap_relative_or_scaled":1e-12}
    result={"verification":"independent scalar CSV and multiplicity bootstrap verification",
        "no_source_npz_reread":True,"no_operator_construction":True,"no_inference":True,
        "inputs_sha256":input_hashes_before,
        "runtime":{"python":platform.python_version(),"numpy":np.__version__},
        "verifier_sha256":digest(Path(__file__)),
        "portable_cli":{"data_root":str(data_root),"output_dir":str(output_dir),"disjoint_input_output_trees":True},
        "arithmetic":"math.fsum scalar aggregation; identical RNG draws reduced by independent per-batch bootstrap multiplicities",
        "checks":{"frames":len(rows),"groups":len(groups),"receivers_all_eight":all(r["receivers"]==8 for r in rows),
            "noise_scale_formula":"Nr * saved noise_var * recorded per-branch ||T_support||_F^2",
            "max_noise_scale_relative_error":max_noise_scale_error,
            "max_excess_definition_scaled_error":max_excess_definition_error,
            "max_full_mask_extra_cross_partition_relative_error":max_feature_partition_error,
            "max_expected_noise_partition_relative_error":max_noise_partition_error,
            "max_excess_partition_scaled_error":max_excess_partition_error,
            "max_group_sum_scaled_error":max_sum_error,"max_group_ratio_scaled_error":max_ratio_error,
            "max_bootstrap_ci_scaled_error":max_ci_error,"max_bootstrap_ci_absolute_error":max_ci_abs_error,
            "all_frame_batch_negative_count_bootstrap_draw_counts_match":bool(matching_counts),
            "s2_full_mask_extra_identity_frames":len(s2_identity),"all_s2_identities_exact":all(s2_identity)},
        "thresholds":thresholds,"bootstrap":{"seed":summary["bootstrap"]["seed"],"resamples":resamples,"cluster":"archived 4-frame batch"},
        "interpretation_limits":[
            "This checks scalar tables and recorded operator coefficients, not independent reconstruction of the PWB matrix.",
            "Expected noise has Nr scale exactly once; the spatial FFT is unitary.",
            "Feature cross terms are required; E_extra is not the difference E_full-E_mask.",
            "PWB excess energy is a transform-dependent aggregate signal diagnosis; it is not raw physical energy or target-user information.",
            "Grouped ratios are ratios of pooled sums, not unweighted averages of per-frame ratios.",
            "Negative excess estimates are retained; bootstrap denominators must be positive and valid draw counts are preserved.",
            "Static S=8 uses just 5 archived batches / 20 frames; its post-hoc CI is descriptive, not confirmatory generalization evidence.",
            "The same existing calibration arrays and fixed equations are used; this scalar check is not a GPU inference or latency comparison."],
        "doppler_state_spacing_validation":[g for g in per_group if g["group_axis"]=="doppler_state_spacing"]}
    errors=(max_noise_scale_error,max_excess_definition_error,max_feature_partition_error,max_noise_partition_error,
            max_excess_partition_error,max_sum_error,max_ratio_error,max_ci_error)
    result["all_checks_pass"]=bool(max(errors)<1e-12 and matching_counts and all(s2_identity) and len(rows)==384)
    input_hashes_after = {p.name: digest(p) for p in inputs}
    result["input_files_unchanged"] = input_hashes_before == input_hashes_after
    result["all_checks_pass"] &= result["input_files_unchanged"]
    output_dir.mkdir(parents=True, exist_ok=True)
    write=output_dir/"validation_from_csv.json"
    write.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"all_checks_pass":result["all_checks_pass"],"checks":result["checks"]},indent=2))
    if not result["all_checks_pass"]:
        raise RuntimeError("Independent scalar/CI verification failed")


if __name__=="__main__":
    main()
