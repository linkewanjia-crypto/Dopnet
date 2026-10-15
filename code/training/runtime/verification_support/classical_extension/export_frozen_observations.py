"""Export receiver observations by replaying the completed frozen test protocol.

This is an external adapter: it neither edits the training project nor loads a
model. Every test cell is released only after its original six-tensor waveform
fingerprint has matched. --list is read-only and does not import Torch/Sionna.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import uuid

import numpy as np

SCHEMA_VERSION = 1
CORE_CHANNELS = (
    "doubly_selective", "tdl_b", "tdl_c", "tdl_d", "cdl_b", "cdl_c",
    "cdl_d", "tdl_a", "tdl_e", "cdl_a", "cdl_e",
)


def digest(obj):
    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def export_lock(output):
    """Avoid concurrently writing the same export and its completion markers."""
    path = Path(output) / ".export.lock"
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            f"Export lock exists: {path}. If the previous process has ended, "
            "remove this stale lock before resuming.") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "host": socket.gethostname()}, stream)
        yield
    finally:
        path.unlink(missing_ok=True)


def select_cells(cells, suite, cell_ids):
    by_id = {cell["cell_id"]: cell for cell in cells}
    if len(by_id) != len(cells):
        raise RuntimeError("Duplicate reference cell IDs")
    if cell_ids:
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("Do not repeat the same --cell-id")
        unknown = set(cell_ids) - set(by_id)
        if unknown:
            raise ValueError(f"Unknown reference cell IDs: {sorted(unknown)}")
        wanted = set(cell_ids)
        return "custom", [cell for cell in cells if cell["cell_id"] in wanted]
    if suite == "all":
        return "all", list(cells)
    selected = [cell for cell in cells
                if cell["channel"] in CORE_CHANNELS
                and cell["doppler_hz"] in (800.0, 1200.0, 1600.0)
                and cell["cp_len"] == 16 and cell["pilot_spacing"] == 4
                and cell["delay_spread_ns"] == 300.0
                and ((cell["modulation"] == "qpsk"
                      and cell["snr_db"] in (15.0, 25.0)
                      and "nmse" in cell["studies"])
                     or (cell["modulation"] == "16qam"
                         and cell["snr_db"] == 20.0
                         and "receiver" in cell["studies"]))]
    if len(selected) != 99:
        raise RuntimeError(f"Expected 99 predefined core cells, got {len(selected)}")
    return "core", selected


def public_geometry(system):
    n = int(system["n_subcarriers"])
    return {
        "n_users": int(system["n_users"]), "n_ant": int(system["n_ant"]),
        "n_subcarriers": n,
        "waveform_fft_size": int(system["waveform_fft_size"]),
        "subcarrier_spacing_hz": float(system["subcarrier_spacing_hz"]),
        "sample_rate_hz": int(system["waveform_fft_size"])
                          * float(system["subcarrier_spacing_hz"]),
        "active_signed_bins": list(range(-(n // 2), -(n // 2) + n)),
        "frame_num_symbols": int(system["frame_num_symbols"]),
        "pilot_symbol_period": int(system["pilot_symbol_period"]),
        "symbol_time_origin": "first visible useful sample",
        "useful_sample_indices": "0 through waveform_fft_size - 1",
        "target_reference": "diagonal CFR averaged across the useful window",
        "array_axes": {
            "y_pilot": ["batch", "pilot_symbol", "receive", "occupied_bin"],
            "x_pilot": ["batch", "user", "pilot_symbol", "occupied_bin"],
            "target_h": ["batch", "user", "data_symbol", "receive", "occupied_bin"],
            "rx_data": ["batch", "data_symbol", "receive", "occupied_bin"],
            "tx_label": ["batch", "user", "data_symbol", "occupied_bin"],
        },
    }


def estimated_bytes_per_frame(geometry):
    t = geometry["frame_num_symbols"]
    p = len(range(0, t, geometry["pilot_symbol_period"]))
    d = t - p
    u, r, n = (geometry[key] for key in ("n_users", "n_ant", "n_subcarriers"))
    # complex64 observations/CFR, int64 labels, float32 noise; plus headers.
    return 8 * n * (p * r + u * p + u * d * r + d * r + u * d) + 4096


def runtime_physics_hashes(project):
    names = ["experiment_protocol.py", "reference_replay.py", "utils/modem.py"]
    names += [str(path.relative_to(project)).replace("\\", "/")
              for path in sorted((project / "data").glob("*.py"))]
    return {name: file_sha256(project / name) for name in names}


def safe_child(root, relative):
    child = (Path(root) / relative).resolve()
    try:
        child.relative_to(Path(root).resolve())
    except ValueError as error:
        raise RuntimeError(f"Unsafe batch path in completion marker: {relative}") from error
    return child


def verify_complete(output, marker_path, signature, role, expected_samples,
                    expected_meta=None, expected_waveform_sha=None):
    marker_path = Path(marker_path)
    if not marker_path.exists():
        return False
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (marker.get("schema_version") != SCHEMA_VERSION
            or marker.get("export_signature") != signature
            or marker.get("role") != role
            or marker.get("samples") != expected_samples):
        raise RuntimeError(f"Completion marker contract differs: {marker_path}")
    if expected_meta is not None and marker.get("meta") != expected_meta:
        raise RuntimeError(f"Completion marker cell metadata differs: {marker_path}")
    if expected_waveform_sha is not None and marker.get("realization_sha256") != expected_waveform_sha:
        raise RuntimeError(f"Completion marker waveform hash differs: {marker_path}")
    offset = 0
    batches = marker.get("batches", [])
    if not batches:
        raise RuntimeError(f"Empty completion marker: {marker_path}")
    for record in batches:
        size = record.get("samples")
        if record.get("offset") != offset or not isinstance(size, int) or size <= 0:
            raise RuntimeError(f"Noncontiguous completion marker batches: {marker_path}")
        if role == "test" and size != 8:
            raise RuntimeError(f"Original test batch size must remain 8: {marker_path}")
        batch_path = safe_child(output, record["path"])
        if not batch_path.is_file() or file_sha256(batch_path) != record.get("sha256"):
            raise RuntimeError(f"Completed batch missing or corrupt: {batch_path}")
        offset += size
    if offset != expected_samples:
        raise RuntimeError(f"Completion marker frame count differs: {marker_path}")
    return True


class CapturePilotObservations:
    """Observe function results without random draws or modifying any tensor.

    Captures detached references only. Selection/copy to CPU happens after the
    simulator has produced all original output features for the batch.
    """
    def __init__(self, waveform):
        self.waveform = waveform
        self.frame = self.layout = self.received = None
        self.build_calls = self.pilot_calls = 0

    def reset(self):
        self.frame = self.layout = self.received = None
        self.build_calls = self.pilot_calls = 0

    def __enter__(self):
        self.original_build = self.waveform._build_frame
        self.original_pilot = self.waveform._pilot_ls_and_interp

        def build(*args, **kwargs):
            result = self.original_build(*args, **kwargs)
            self.frame, self.layout = result[0].detach(), result[1]
            self.build_calls += 1
            return result

        def pilot(*args, **kwargs):
            result = self.original_pilot(*args, **kwargs)
            received = args[0] if args else kwargs["Y"]
            layout = args[1] if len(args) > 1 else kwargs["layout"]
            if self.layout is None or layout != self.layout:
                raise RuntimeError("Captured frame and receiver layouts differ")
            self.received = received.detach()
            self.pilot_calls += 1
            return result

        self.waveform._build_frame = build
        self.waveform._pilot_ls_and_interp = pilot
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.waveform._build_frame = self.original_build
        self.waveform._pilot_ls_and_interp = self.original_pilot
        self.reset()

    def arrays(self, batch, cfg, metadata):
        if self.build_calls != 1 or self.pilot_calls != 1:
            raise RuntimeError("Expected exactly one frame and one pilot observation per simulation")
        frame = self.frame.cpu().numpy()
        received = self.received.cpu().numpy()
        pilot_blocks = np.asarray(self.layout.pilot_blocks, dtype=np.int64)
        data_visible = np.asarray(self.layout.data_visible, dtype=np.int64)
        h_ri = batch["target_h"].detach().cpu().numpy()
        rx_ri = batch["rx_data"].detach().cpu().numpy()
        arrays = {
            # Original Y [B,R,block,N] -> exported [B,P,R,N].
            "y_pilot": received[:, :, pilot_blocks, :].transpose(0, 2, 1, 3).copy(),
            "x_pilot": frame[:, :, pilot_blocks, :].copy(),
            "target_h": (h_ri[:, :, 0] + 1j * h_ri[:, :, 1])[:, :, data_visible].astype(np.complex64),
            "rx_data": (rx_ri[:, 0] + 1j * rx_ri[:, 1]).astype(np.complex64),
            "tx_label": batch["tx_label"].detach().cpu().numpy().astype(np.int64),
            "noise_var": batch["noise_var"].detach().cpu().numpy().astype(np.float32),
            "pilot_visible": np.asarray(self.layout.pilot_visible, dtype=np.int64),
            "data_visible": data_visible,
            "cp_wave_samples": self.scalar(batch["cp_wave_samples"], "cp_wave_samples", np.int64),
            "waveform_fft_size": self.scalar(batch["waveform_fft_size"], "waveform_fft_size", np.int64),
            "subcarrier_spacing_hz": np.asarray(cfg["system"]["subcarrier_spacing_hz"], dtype=np.float64),
            "metadata": np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
        }
        s = cfg["system"]
        b, u, r, n = (frame.shape[0], int(s["n_users"]), int(s["n_ant"]), int(s["n_subcarriers"]))
        p, d = len(pilot_blocks), len(data_visible)
        shapes = {"y_pilot": (b, p, r, n), "x_pilot": (b, u, p, n),
                  "target_h": (b, u, d, r, n), "rx_data": (b, d, r, n),
                  "tx_label": (b, u, d, n), "noise_var": (b,)}
        for key, shape in shapes.items():
            if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise RuntimeError(f"Invalid exported {key}: expected finite {shape}")
        if any(arrays[key].dtype != np.complex64 for key in ("y_pilot", "x_pilot", "target_h", "rx_data")):
            raise RuntimeError("Frozen complex observations must remain complex64")
        if np.any(arrays["noise_var"] < 0):
            raise RuntimeError("Negative noise variance")
        return arrays

    @staticmethod
    def scalar(tensor, name, dtype):
        values = tensor.detach().cpu().numpy().reshape(-1)
        if not len(values) or not np.all(values == values[0]):
            raise RuntimeError(f"Public geometry must be shared within one exported chunk: {name}")
        return np.asarray(values[0], dtype=dtype)


def write_batch(output, relative, arrays, offset, size):
    path = safe_child(output, relative)
    atomic_npz(path, arrays)
    return {"path": relative, "offset": offset, "samples": int(size),
            "sha256": file_sha256(path)}


def prepare_manifest(output, base, cells, suite_name):
    path = output / "export_manifest.json"
    ids = [cell["cell_id"] for cell in cells]
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("export_signature") != base["export_signature"]:
            raise RuntimeError("Existing export uses a different reference/code/config contract. Use a new output directory.")
        merged = {cell["cell_id"]: cell for cell in previous.get("cells", [])}
        for cell in cells:
            if cell["cell_id"] in merged and merged[cell["cell_id"]] != cell:
                raise RuntimeError("Existing export cell metadata differs")
            merged[cell["cell_id"]] = cell
        suites = previous.get("suites", {})
        if suite_name in suites and suite_name != "custom" and suites[suite_name] != ids:
            raise RuntimeError("Existing predefined suite specification differs")
        suites[suite_name] = (list(dict.fromkeys(suites.get("custom", []) + ids))
                              if suite_name == "custom" else ids)
        base = {**base, "cells": list(merged.values()), "suites": suites}
    else:
        base = {**base, "cells": list(cells), "suites": {suite_name: ids}}
    atomic_json(path, base)
    return base


def check_current_protocol(project, source_cfg, reference):
    """Validate current public generator configuration before using frozen cfg."""
    from train import load_cfg
    from experiment_protocol import make_config, experiment_cells, code_hash
    if code_hash(project) != reference.experiment.get("code_sha256"):
        raise RuntimeError(
            "The simulation/runtime code differs from the completed reference. "
            "Export with the original completed training project; the latency-only hotfix is compatible.")
    config_path = project / "configs" / "dopnet.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing training project configuration: {config_path}")
    current = make_config(load_cfg(str(config_path)), "dopnet", 42)
    if current["system"] != source_cfg["system"]:
        changed = sorted(key for key in set(current["system"]) | set(source_cfg["system"])
                         if current["system"].get(key) != source_cfg["system"].get(key))
        raise RuntimeError(f"Current project differs from frozen generation system: {changed}")
    # This optional extra observation does not enter the original six-key hash.
    def pipeline(cfg):
        values = copy.deepcopy(cfg["data_pipeline"])
        values.pop("emit_pilot_only_fast_time_add", None)
        return values
    if pipeline(current) != pipeline(source_cfg):
        raise RuntimeError("Current project data pipeline differs from frozen reference")
    cells, skipped = experiment_cells(source_cfg)
    if cells != reference.evaluation["cells"] or skipped != reference.evaluation["skipped_infeasible"]:
        raise RuntimeError("Current test grid/feasibility differs from the frozen 636-cell reference")
    if source_cfg["model"].get("ablation_profile") != "full":
        raise RuntimeError("Frozen generator source configuration must be the full DOP-Net configuration")
    if not (source_cfg["data_pipeline"].get("emit_pilot_ls")
            and source_cfg["data_pipeline"].get("emit_slow_time_add")):
        raise RuntimeError("Frozen reference is missing the required receiver features")


def export_test_cell(output, cfg, reference, cell, signature, generator, capture, torch):
    from experiment_protocol import TEST_SEED, digest as protocol_digest
    from reference_replay import update_waveform_digest
    marker_path = output / "cells" / cell["cell_id"] / "COMPLETE.json"
    expected_sha = reference.cell_fingerprints[cell["cell_id"]]
    if verify_complete(output, marker_path, signature, "test", 512, cell, expected_sha):
        print(f"Verified existing cell {cell['cell_id']}; no generation", flush=True)
        return
    records = []
    hasher = hashlib.sha256()
    condition = dict(channel_type=cell["channel"], modulation=cell["modulation"],
                     delay_spread_ns=cell["delay_spread_ns"], doppler_hz=cell["doppler_hz"],
                     cp_len=cell["cp_len"], pilot_spacing=cell["pilot_spacing"])
    for offset in range(0, 512, 8):
        seed = TEST_SEED + int(protocol_digest(dict(cell=cell["cell_id"], offset=offset))[:15], 16)
        random_generator = torch.Generator(device=generator.device).manual_seed(seed)
        capture.reset()
        batch = generator.simulate(8, condition,
                                  torch.full((8,), cell["snr_db"], device=generator.device),
                                  random_generator, seed + 1000000001)
        update_waveform_digest(hasher, batch)
        metadata = {"role": "test", "cell": cell, "condition": condition,
                    "offset": offset, "samples": 8,
                    "reference_realization_sha256": expected_sha}
        arrays = capture.arrays(batch, cfg, metadata)
        relative = f"cells/{cell['cell_id']}/batch_{offset:04d}.npz"
        records.append(write_batch(output, relative, arrays, offset, 8))
        del arrays, batch
        capture.reset()
        if offset % 128 == 0:
            print(f"  {cell['cell_id']} frames={offset + 8}/512", flush=True)
    observed_sha = hasher.hexdigest()
    reference.assert_fingerprint(cell, observed_sha)
    marker = {"schema_version": SCHEMA_VERSION, "export_signature": signature,
              "role": "test", "cell_id": cell["cell_id"], "meta": cell,
              "samples": 512, "batch_size": 8,
              "realization_sha256": observed_sha, "batches": records}
    # A mismatch above leaves no completion marker. The CPU runner must ignore
    # unfinished cells even when individual temporary batch files exist.
    atomic_json(marker_path, marker)
    print(f"Frozen waveform SHA-256 PASS: {cell['cell_id']}", flush=True)


def export_calibration(output, cfg, signature, capture, torch, count=128):
    from data.generator import BatchChannelGenerator
    from experiment_protocol import CALIBRATION_SEED
    marker_path = output / "calibration" / "COMPLETE.json"
    if verify_complete(output, marker_path, signature, "calibration", count):
        print("Verified existing independent calibration cache", flush=True)
        return
    generator = BatchChannelGenerator(cfg, torch.device("cuda:0"), CALIBRATION_SEED)
    records = []
    produced = 0
    buffer_index = 0
    while produced < count:
        target = min(generator.generation_batch_size, count - produced)
        made = 0
        chunk_index = 0
        while made < target:
            remaining = target - made
            condition = generator.sample_condition(
                0, buffer_index, chunk_index,
                channel_weights=generator._chunk_weights_for_remaining(remaining))
            from data.threegpp_profiles import is_3gpp_channel
            preferred = (generator.threegpp_microbatch_size
                         if is_3gpp_channel(condition["channel_type"])
                         else generator.synthetic_microbatch_size)
            size = min(max(1, int(preferred)), remaining)
            capture.reset()
            batch = generator._chunk(size, 0, buffer_index, chunk_index, cond=condition)
            offset = produced + made
            metadata = {"role": "calibration", "condition": condition,
                        "offset": offset, "samples": size, "calibration_seed": CALIBRATION_SEED,
                        "snr_db": batch["snr_db"].detach().cpu().tolist()}
            arrays = capture.arrays(batch, cfg, metadata)
            relative = f"calibration/batch_{offset:04d}.npz"
            records.append(write_batch(output, relative, arrays, offset, size))
            del arrays, batch
            capture.reset()
            made += size
            chunk_index += 1
        produced += target
        buffer_index += 1
        print(f"Independent calibration frames={produced}/{count}", flush=True)
    marker = {"schema_version": SCHEMA_VERSION, "export_signature": signature,
              "role": "calibration", "samples": count, "seed": CALIBRATION_SEED,
              "distribution": "frozen training configuration, epoch 0",
              "batches": records}
    atomic_json(marker_path, marker)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path,
                        help="Original GPU training project; left unmodified")
    parser.add_argument("--reference-results", type=Path,
                        help="Completed results directory or ZIP (default: PROJECT/results/dopnet_additions)")
    parser.add_argument("--output", type=Path, default=Path("frozen_observations"))
    parser.add_argument("--suite", choices=("core", "all"), default="core")
    parser.add_argument("--cell-id", action="append", default=[],
                        help="Export a specified frozen cell; repeat for several cells")
    parser.add_argument("--list", action="store_true", help="Inspect metadata and disk estimate only; no Torch/CUDA")
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument("--calibration-only", action="store_true")
    calibration.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args(argv)
    project = args.project.expanduser().resolve()
    if not project.is_dir() or not (project / "reference_replay.py").is_file():
        parser.error("--project must name the completed DOPNet_Submission_Experiments project")
    sys.path.insert(0, str(project))
    from reference_replay import ReferenceResults
    from experiment_protocol import CALIBRATION_SEED, TEST_SEED
    reference_path = (args.reference_results or project / "results" / "dopnet_additions").expanduser().resolve()
    reference = ReferenceResults(reference_path)
    source_cfg = copy.deepcopy(reference.experiment["configs"]["dopnet_seed42"])
    suite_name, cells = select_cells(reference.evaluation["cells"], args.suite, args.cell_id)
    if args.calibration_only:
        cells = []
        suite_name = "custom"
    geometry = public_geometry(source_cfg["system"])
    estimated_bytes = (512 * len(cells) + (0 if args.skip_calibration else 128)) * estimated_bytes_per_frame(geometry)
    output = args.output.expanduser().resolve()
    if args.list:
        print(json.dumps({"operation": "metadata only; no simulation", "suite": suite_name,
                          "test_cells": len(cells), "test_frames": len(cells) * 512,
                          "calibration_frames": 0 if args.skip_calibration else 128,
                          "estimated_uncompressed_bytes": estimated_bytes,
                          "estimated_uncompressed_GiB": round(estimated_bytes / 1024**3, 2),
                          "test_seed": TEST_SEED, "test_batch_size": 8,
                          "reference_provenance": reference.provenance(),
                          "cell_ids": [cell["cell_id"] for cell in cells]}, indent=2))
        return
    # Keep the project's allocator policy and link chunk sizes unchanged.
    # This module has no Torch dependency, so allocation is configured first.
    from utils.cuda_memory import configure_cuda_allocator
    configure_cuda_allocator()
    # All GPU/runtime imports are deliberately below the read-only --list path.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen test regeneration requires the original CUDA/Sionna server. Use --list locally.")
    from data.generator import BatchChannelGenerator
    from data import waveform
    check_current_protocol(project, source_cfg, reference)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    base = {"schema_version": SCHEMA_VERSION,
            "reference_provenance": reference.provenance(), "cfg_system": source_cfg["system"],
            "generation_config": source_cfg, "geometry": geometry,
            "exporter_sha256": file_sha256(Path(__file__)),
            "runtime_physics_sha256": runtime_physics_hashes(project),
            "samples_per_cell": 512, "batch_size": 8, "test_seed": TEST_SEED,
            "calibration_seed": CALIBRATION_SEED, "calibration_samples": 128,
            "target_role": "fixed-test extension, separate independent calibration"}
    base["export_signature"] = digest(base)
    output.mkdir(parents=True, exist_ok=True)
    with export_lock(output):
        prepare_manifest(output, base, cells, suite_name)
        pending = [cell for cell in cells
                   if not verify_complete(output, output / "cells" / cell["cell_id"] / "COMPLETE.json",
                                          base["export_signature"], "test", 512, cell,
                                          reference.cell_fingerprints[cell["cell_id"]])]
        pending_calibration = (not args.skip_calibration and not verify_complete(
            output, output / "calibration" / "COMPLETE.json", base["export_signature"], "calibration", 128))
        required = (len(pending) * 512 + (128 if pending_calibration else 0)) * estimated_bytes_per_frame(geometry)
        free = shutil.disk_usage(output).free
        if free < required + 512 * 1024**2:
            raise RuntimeError(f"Insufficient conservative export disk budget: need {required / 1024**3:.2f} GiB "
                               f"plus 0.5 GiB reserve, have {free / 1024**3:.2f} GiB. Export fewer --cell-id values "
                               "or select a larger destination. Actual NPZ compression may be smaller.")
        generator = BatchChannelGenerator(source_cfg, torch.device("cuda:0"), TEST_SEED)
        with torch.inference_mode(), CapturePilotObservations(waveform) as capture:
            if not args.skip_calibration:
                export_calibration(output, source_cfg, base["export_signature"], capture, torch)
            for index, cell in enumerate(cells, 1):
                print(f"[{index}/{len(cells)}] {cell}", flush=True)
                export_test_cell(output, source_cfg, reference, cell, base["export_signature"], generator, capture, torch)
    print(f"Export complete: {output}; no model training or inference was performed", flush=True)


if __name__ == "__main__":
    main()
