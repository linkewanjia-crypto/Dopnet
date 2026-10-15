"""Scalar CI, complete eight-method pairing and signed assets.

Ported from the reviewed WiMAE delivery1.0.1 scalar checker. The explicit new
method universe is8 and the comparator universe7; df2 arithmetic is unchanged.
No tensor framework, models, scientific data or local bootstrap is executed.
"""
import csv,math
from pathlib import Path
from .protocol import LABELS,SEEDS,ASSET_NAMES,require,digest,file_hash,signed as signed_metadata

def check_seed_intervals(path, t_critical=4.302652729749464):
    """Scalar arithmetic; preserves the original linear-NMSE aggregation."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if "mean" not in row or not row.get("mean"):
            continue
        require(row.get("n_training_seeds") == "3", "Each neural method needs three training seeds")
        values = [float(row["seed" + str(s)]) for s in SEEDS]
        require(all(math.isfinite(v) and v >= 0 for v in values), "Nonfinite seed point estimate")
        mean = sum(values) / 3
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / 2)
        half = t_critical * sd / math.sqrt(3)
        for key, expected in (("mean", mean), ("seed_sd", sd),
                              ("seed_ci95_lo_unbounded", mean - half), ("seed_ci95_hi_unbounded", mean + half)):
            require(math.isclose(float(row[key]), expected, rel_tol=1e-10, abs_tol=1e-14), "Three-seed interval differs: " + key)
        if row["metric"] == "nmse" and mean > 0:
            require(math.isclose(float(row["mean_db"]), 10 * math.log10(mean), rel_tol=1e-10, abs_tol=1e-12),
                    "NMSE dB was averaged before linear NMSE")
    require(rows, "Empty result summary")

def check_paired_intervals(summary_path, paired_path, macro=False, t_critical=4.302652729749464):
    with Path(summary_path).open(encoding="utf-8-sig", newline="") as stream:
        summaries = list(csv.DictReader(stream))
    with Path(paired_path).open(encoding="utf-8-sig", newline="") as stream:
        differences = list(csv.DictReader(stream))
    context = ("study", "scope", "modulation", "doppler_hz", "snr_db", "metric") if macro else ("cell_id", "metric")
    index = {tuple(row[k] for k in context) + (row["method"],): row for row in summaries}
    require(len(index) == len(summaries), "Duplicate summary groups")
    groups = {}
    for identity in index:
        groups.setdefault(identity[:-1], set()).add(identity[-1])
    require(groups and all(methods == set(LABELS) for methods in groups.values()),
            "Summary group lacks exactly the eight expected methods")
    expected_pairs = {identity for identity in index if identity[-1] != "dopnet"}
    require(len(differences) == len(summaries) * 7 // 8, "Paired summary lacks a method/group")
    seen = set()
    for row in differences:
        key = tuple(row[k] for k in context)
        identity = key + (row["comparator"],)
        require(identity in expected_pairs and identity not in seen and row["reference"] == "dopnet",
                "Duplicate/missing/self paired comparison")
        seen.add(identity)
        comparator, reference = index[identity], index[key + ("dopnet",)]
        deltas = [float(comparator["seed" + str(s)]) - float(reference["seed" + str(s)]) for s in SEEDS]
        def check_interval(values, mean_key, lo_key, hi_key):
            mean = sum(values) / 3
            sd = math.sqrt(sum((v - mean) ** 2 for v in values) / 2)
            half = t_critical * sd / math.sqrt(3)
            for field, expected in ((mean_key, mean), (lo_key, mean - half), (hi_key, mean + half)):
                require(math.isclose(float(row[field]), expected, rel_tol=1e-10, abs_tol=1e-14), "Paired seed interval differs")
        check_interval(deltas, "paired_difference", "paired_seed_ci95_lo", "paired_seed_ci95_hi")
        lo, hi = float(row["paired_test_cluster_ci95_lo"]), float(row["paired_test_cluster_ci95_hi"])
        require(math.isfinite(lo) and math.isfinite(hi) and lo <= hi, "Invalid paired cluster-bootstrap interval")
        if row["metric"] == "nmse" and all(float(x["seed" + str(s)]) > 0 for x in (comparator, reference) for s in SEEDS):
            gaps = [10 * math.log10(float(comparator["seed" + str(s)]) / float(reference["seed" + str(s)])) for s in SEEDS]
            check_interval(gaps, "mean_paired_gap_db", "paired_gap_db_seed_ci95_lo", "paired_gap_db_seed_ci95_hi")
    require(seen == expected_pairs, "Paired comparisons lack an expected comparator/group")

def check_assets(output, ev, statistics):
    ready = signed_metadata(output / "PAPER_READY.json")
    require(ready.get("status") == "complete" and ready.get("paper_ready") is True and ready.get("pending") == []
            and ready.get("evaluation_signature") == ev["signature"]
            and ready.get("statistics_signature") == statistics["signature"]
            and set(ready["assets_sha256"]) == set(ASSET_NAMES), "Paper asset completion binding differs")
    for name in ASSET_NAMES:
        path = output / "paper_assets" / name
        require(path.is_file() and path.stat().st_size > 100 and file_hash(path) == ready["assets_sha256"][name],
                "Paper asset absent/empty/changed: " + name)
        head = path.read_bytes()[:8]
        if name.endswith(".pdf"):
            require(head.startswith(b"%PDF-"), "Invalid PDF asset")
        elif name.endswith(".png"):
            require(head == b"\x89PNG\r\n\x1a\n", "Invalid PNG asset")
        else:
            require(b"\\begin{table" in path.read_bytes(), "Invalid TeX table asset")
