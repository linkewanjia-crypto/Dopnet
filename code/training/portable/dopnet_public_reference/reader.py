"""Read the retained 38-row reference view without impersonating its 53-row source.

Metadata, recipe preparation and view inspection use the standard library.
Numerical NPZ decoding is lazy and retains the original fixed-waveform checks.
The archived experiment/evaluation signatures name the provenance source;
REFERENCE_VIEW.json and PROJECTION.json name the actual public representation.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import re

EXPERIMENT_SHA = "1c91a812720ab807d6664bf66de089fa442ba120ab91bdc67957bc95b9145ae2"
EVALUATION_SHA = "92d43ad877aca8b206836e5a0a78e03795fd06805dfbc6963db56034f2624946"
RUNTIME_SHA = "0af0222908c8e1bd8543e29a2a78f8dc53ede503750464e22445f773d56cb181"
VALIDATION_SHA = "83d0e03524628b617671c42a73d3c458cc3395c31965a4e22295e8389f56e597"
MAIN = ("dopnet", "cfr_derived", "slow_time", "pilot_only_fast_time", "aft_only",
        "no_fusion", "no_offgrid", "lee3ddual", "wimae_ce", "dopnet_lite", "fortitran")
CLASSICAL = ("ls_interp", "lmmse_wiener", "dpsbem_ls", "pwb_linear", "perfect_diagonal_csi")
REFERENCE_WAVEFORM_KEYS = ("input", "add_input", "add_input_slow", "rx_data", "target_h", "tx_label")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signed_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if value.get("signature") != digest({k: v for k, v in value.items() if k != "signature"}):
        raise RuntimeError("Invalid reference metadata signature: " + Path(path).name)
    return value


def _local(root, relative):
    p = (root / relative).resolve()
    if not p.is_relative_to(root):
        raise RuntimeError("Reference member escapes its root")
    return p


def _cell_id(cell):
    value = cell["cell_id"] if isinstance(cell, dict) else cell
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{16}", value):
        raise RuntimeError("Invalid reference cell ID")
    return value


def _metadata(location):
    root = Path(location).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError("Pass the new portable reference directory, not a historical ZIP")
    view = signed_json(root / "REFERENCE_VIEW.json")
    if view.get("schema") != "dopnet_tmlcn_public_reference_view_v1":
        raise RuntimeError("Unknown public reference namespace")
    if file_hash(root / "experiment.json") != EXPERIMENT_SHA or file_hash(root / "evaluation.json") != EVALUATION_SHA:
        raise RuntimeError("Original scientific metadata bytes differ")
    experiment = signed_json(root / "experiment.json")
    evaluation = signed_json(root / "evaluation.json")
    if evaluation.get("experiment_signature") != experiment["signature"]:
        raise RuntimeError("Original experiment/evaluation signatures disagree")
    if experiment.get("code_sha256") != RUNTIME_SHA or experiment.get("validation_cache_sha256") != VALIDATION_SHA:
        raise RuntimeError("Original runtime/validation provenance differs")
    if experiment.get("samples_per_cell") != 512 or experiment.get("batch_size") != 8:
        raise RuntimeError("Expected original 512/8 fixed-test protocol")
    for name in ("PROJECTION.json", "INDEX.json"):
        if file_hash(root / name) != view["files_sha256"][name]:
            raise RuntimeError("Projected metadata bytes differ: " + name)
    projection = signed_json(root / "PROJECTION.json")
    # The retained index is a byte manifest, not an archived signed receipt.
    # Its complete bytes are bound by the independently signed view manifest.
    index = json.loads((root / "INDEX.json").read_text(encoding="utf-8-sig"))
    if projection.get("source_evaluation_signature") != evaluation["signature"] or projection.get("source_evaluation_sha256") != EVALUATION_SHA:
        raise RuntimeError("Projected source is not the archived corrected evaluation")
    if projection.get("source_route_count") != 53 or index.get("projection_signature") != projection["signature"]:
        raise RuntimeError("Projected and original source identities disagree")
    active = [r for r in evaluation["routes"] if r["method"] in MAIN + CLASSICAL]
    if len(active) != 38 or projection.get("public_routes") != active:
        raise RuntimeError("Public view must contain exactly 33 active main and 5 classical routes")
    wanted = [i for i, r in enumerate(evaluation["routes"]) if r["method"] in MAIN + CLASSICAL]
    if projection.get("source_row_indices") != wanted:
        raise RuntimeError("Public row mapping differs from the original 53-route source")
    if view.get("projected_route_count") != 38 or view.get("source_route_count") != 53 or view.get("projected_score_signature") != projection["signature"]:
        raise RuntimeError("Public view claims an incorrect identity or route count")
    if len(evaluation.get("cells", [])) != 636 or index.get("condition_count") != 636 or len(index.get("cells", [])) != 636:
        raise RuntimeError("The reference must retain exactly 636 test cells")
    by_id = {r["cell_id"]: r for r in index["cells"]}
    if len(by_id) != 636 or list(by_id) != [c["cell_id"] for c in evaluation["cells"]]:
        raise RuntimeError("Reference cell order or identifiers differ")
    for cell in evaluation["cells"]:
        cid = _cell_id(cell)
        row = by_id[cid]
        if row["npz"] != "cells/" + cid + ".npz" or not _local(root, row["npz"]).is_file():
            raise RuntimeError("Missing or misbound projected score cell: " + cid)
        if row.get("candidate38_sha256", row["sha256"]) != row["sha256"]:
            raise RuntimeError("Conflicting candidate score SHA: " + cid)
        if not HEX64.fullmatch(row.get("source_score_sha256", "")) or not HEX64.fullmatch(row["sha256"]):
            raise RuntimeError("Missing original or projected score SHA: " + cid)
    return root, experiment, evaluation, view, projection, index, by_id


def check_reference(location):
    """Return real original signed metadata and the distinct public-view root."""
    root, experiment, evaluation, *_ = _metadata(location)
    return root, experiment, evaluation


class ReferenceResults:
    def __init__(self, location):
        (self.location, self.experiment, self.evaluation, self.view,
         self.projection, self.index, self._index) = _metadata(location)
        self.public_routes = self.projection["public_routes"]
        self.score_signature = self.projection["signature"]
        self.cell_fingerprints = {}
        self._cells = {c["cell_id"]: c for c in self.evaluation["cells"]}

    def project_indices(self, original_indices):
        """Map original row positions explicitly; no omitted route is fabricated."""
        mapping = {old: new for new, old in enumerate(self.projection["source_row_indices"])}
        try:
            return [mapping[i] for i in original_indices]
        except KeyError as error:
            raise RuntimeError("Requested route is outside the retained public scope") from error

    def source_score_sha256(self, cell):
        """SHA of original 53-row NPZ; it is not the current file's SHA."""
        return self._index[_cell_id(cell)]["source_score_sha256"]

    def actual_score_sha256(self, cell):
        """Verify and return the SHA of the actual retained 38-row file."""
        cid = _cell_id(cell)
        row = self._index[cid]
        path = _local(self.location, row["npz"])
        if path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]:
            raise RuntimeError("Projected score bytes differ: " + cid)
        return row["sha256"]

    def validate_cell(self, cell):
        """Check one genuine NPZ without loading weights or a waveform cache."""
        import numpy as np
        cid = _cell_id(cell)
        self.actual_score_sha256(cid)
        row = self._index[cid]
        proof = _local(self.location, row["receipt"])
        if file_hash(proof) != self.view["files_sha256"][row["receipt"]]:
            raise RuntimeError("Projected row-byte proof differs: " + cid)
        evidence = json.loads(proof.read_text(encoding="utf-8-sig"))
        if (evidence["source_score_sha256"] != row["source_score_sha256"]
                or evidence["candidate38_sha256"] != row["sha256"]
                or evidence["projection_signature"] != self.score_signature):
            raise RuntimeError("Projected and source score byte proofs disagree: " + cid)
        with np.load(_local(self.location, row["npz"]), allow_pickle=False) as z:
            def scalar(name):
                if z[name].shape != ():
                    raise RuntimeError("Reference field must be a scalar: " + name)
                return str(z[name].item())
            if scalar("signature") != self.score_signature or json.loads(scalar("meta")) != self._cells[cid]:
                raise RuntimeError("Projected cell identity or metadata differs: " + cid)
            if z["nmse"].shape != (38, 512) or not np.array_equal(z["cluster_id"], np.arange(512) // 8):
                raise RuntimeError("Projected cell layout differs: " + cid)
            if z["native_available"].shape != (38,):
                raise RuntimeError("Projected route flags differ: " + cid)
            if self._cells[cid]["ber"]:
                for key in ("bits", "errors_diagonal", "errors_native", "native_residual_max", "native_direct_fallback"):
                    if z[key].shape != (38, 512):
                        raise RuntimeError("Required receiver array differs: " + key)
            fingerprint = scalar("realization_sha256")
            if not HEX64.fullmatch(fingerprint):
                raise RuntimeError("Invalid original waveform fingerprint: " + cid)
            self.cell_fingerprints[cid] = fingerprint
        return dict(cell_id=cid, projected_routes=38, samples=512,
                    source_score_sha256=self.source_score_sha256(cid),
                    candidate_score_sha256=row["sha256"], realization_sha256=fingerprint)

    def validate_protocol(self, manifest, cfg, cells, skipped):
        if manifest.get("samples_per_cell") != 512 or manifest.get("batch_size") != 8:
            raise RuntimeError("Current test protocol must use 512 samples per cell and batch size 8")
        if cells != self.evaluation["cells"] or skipped != self.evaluation["skipped_infeasible"]:
            raise RuntimeError("Current test grid differs from the original 636-cell reference")
        source_cfg = self.experiment["configs"]["dopnet_seed42"]
        if cfg["system"] != source_cfg["system"]:
            raise RuntimeError("Current system differs from fixed-test reference")
        current_pipeline = copy.deepcopy(cfg["data_pipeline"])
        current_pipeline.pop("emit_pilot_only_fast_time_add", None)
        if current_pipeline != source_cfg["data_pipeline"]:
            raise RuntimeError("Current data pipeline differs from fixed-test reference")

    def assert_fingerprint(self, cell, observed):
        cid = _cell_id(cell)
        if cid not in self.cell_fingerprints:
            self.validate_cell(cid)
        if observed != self.cell_fingerprints[cid]:
            raise RuntimeError("Fixed-test waveform mismatch in cell " + cid)

    def view_provenance(self):
        return dict(schema="dopnet_tmlcn_public_reference_view_v1",
                    reference_view_signature=self.view["signature"],
                    projected_score_signature=self.score_signature,
                    source_experiment_sha256=EXPERIMENT_SHA,
                    source_evaluation_sha256=EVALUATION_SHA,
                    source_route_count=53, projected_route_count=38,
                    candidate_score_index_sha256=self.view["files_sha256"]["INDEX.json"],
                    projected_values_recomputed=False, scientific_execution_claim=False)

    def provenance(self):
        # Preserve the strict original per-cell waveform contract. This lazy
        # operation validates every real cell when scientific evaluation calls it.
        for cid in self._cells:
            if cid not in self.cell_fingerprints:
                self.validate_cell(cid)
        fingerprint = digest(dict(evaluation_signature=self.evaluation["signature"],
                                  experiment_signature=self.experiment["signature"],
                                  waveform_keys=REFERENCE_WAVEFORM_KEYS,
                                  realization_sha256=self.cell_fingerprints))
        return dict(reference_evaluation_signature=self.evaluation["signature"],
                    reference_experiment_signature=self.experiment["signature"],
                    reference_realizations_sha256=fingerprint, reference_cells=636,
                    reference_samples_per_cell=512, reference_batch_size=8,
                    reference_waveform_keys=list(REFERENCE_WAVEFORM_KEYS),
                    public_reference_view=self.view_provenance())


def copy_reference(location, destination):
    """Copy the actual public view after checking every declared input SHA.

    This is an explicit future file operation, not a generator or verifier run.
    Original metadata, representation identity, row proofs and cells stay bound.
    """
    import shutil
    reference = ReferenceResults(location)
    target = Path(destination).expanduser().resolve()
    if target.exists() or target.is_relative_to(reference.location):
        raise RuntimeError("Reference copy requires a new independent destination")
    expected = dict(reference.view["files_sha256"])
    expected["REFERENCE_VIEW.json"] = file_hash(reference.location / "REFERENCE_VIEW.json")
    for row in reference.index["cells"]:
        expected[row["npz"]] = row["sha256"]
    for name, value in expected.items():
        if file_hash(_local(reference.location, name)) != value:
            raise RuntimeError("Reference input SHA differs: " + name)
    target.parent.mkdir(parents=True, exist_ok=True)
    needed = sum(_local(reference.location, name).stat().st_size for name in expected)
    if shutil.disk_usage(target.parent).free < needed + 16 * 1024 * 1024:
        raise RuntimeError("Insufficient space for independent reference copy")
    target.mkdir()
    for name, value in expected.items():
        source = _local(reference.location, name)
        if file_hash(source) != value:
            raise RuntimeError("Reference changed during copy: " + name)
        output = _local(target, name)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        if file_hash(output) != value:
            raise RuntimeError("Independent reference copy SHA differs: " + name)
    copied = ReferenceResults(target)
    return dict(reference_dir=str(target), files=len(expected), bytes=needed,
                public_reference_view=copied.view_provenance(),
                scientific_execution_performed=False)
