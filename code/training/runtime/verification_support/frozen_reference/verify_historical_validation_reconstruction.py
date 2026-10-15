"""Server-only check and pilot-only derivation for the archived validation set.

This streams freshly generated samples for comparison.  It does not train a
model or create another copy of the 4.68 GB cache.  On success it writes one
small derived-feature sidecar and a provenance attestation next to the original
cache.  Any disagreement aborts before the attestation is written.
"""

import argparse
import json
import os
import uuid
from pathlib import Path

from utils.cuda_memory import configure_cuda_allocator

configure_cuda_allocator()

import torch
import yaml

from data.generator import BatchChannelGenerator
from experiment_protocol import make_config
from fixed_validation import (
    HISTORICAL_CACHE_SHA256, HISTORICAL_IDENTITY_SHA256,
    PILOT_FEATURE_SOURCE,
    assert_historical_identity, assert_reconstruction_attestation,
    companion_paths, derivation_source_sha256,
    derive_pilot_only_fast_time_add, load_historical_core, _file_sha256,
    historical_file_metadata,
)
from train import setup_backend

CORE_RTOL = 1e-4
CORE_ATOL = 1e-5
DERIVED_KEY = "add_input_pilot_only_fast_time"


def _compare(actual, reference, name, start, *, rtol, atol):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise RuntimeError(
            f"Regenerated validation {name} schema differs at offset {start}: "
            f"{actual.dtype}/{tuple(actual.shape)} vs "
            f"{reference.dtype}/{tuple(reference.shape)}"
        )
    if actual.is_floating_point() or actual.is_complex():
        if not torch.allclose(actual, reference, rtol=rtol, atol=atol,
                              equal_nan=False):
            difference = (actual - reference).abs()
            raise RuntimeError(
                f"Regenerated validation {name} differs at offset {start}; "
                f"maximum absolute discrepancy={difference.max().item():.6g}. "
                "Historical validation cannot be used with this generator."
            )
        return float((actual - reference).abs().max().item())
    if not torch.equal(actual, reference):
        raise RuntimeError(
            f"Regenerated validation {name} differs at offset {start}; "
            "historical validation cannot be used with this generator."
        )
    return 0.0


def _select_direct_feature(direct, ls_reconstruction, start):
    """Use direct Y-derived pilot features; LS divide/multiply is diagnostic.

    The historical cache contains Y/x only. Multiplying by x in complex64
    cannot exactly recover Y, and the ADD backprojection amplifies its last-bit
    roundoff. The freshly regenerated Y is accepted only after every archived
    core tensor for the same frame has passed the separate comparison above.
    """
    if (direct.shape != ls_reconstruction.shape
            or direct.dtype != ls_reconstruction.dtype
            or direct.dtype != torch.float32
            or not torch.isfinite(direct).all()
            or not torch.isfinite(ls_reconstruction).all()):
        raise RuntimeError(f"Pilot-only validation feature schema/nonfinite values at {start}")
    error = (direct - ls_reconstruction).abs()
    return direct.detach().cpu(), float(error.max().item())


def _atomic_save_tensor(tensor, path):
    temporary = path.with_name(path.name + ".partial." + uuid.uuid4().hex)
    try:
        torch.save(tensor, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(obj, path):
    temporary = path.with_name(path.name + ".partial." + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(obj, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_and_derive(cache, sha256, batch_size):
    if not torch.cuda.is_available():
        raise RuntimeError("This check requires the CUDA server; do not run it on a laptop")
    root = Path(__file__).resolve().parent
    base = yaml.safe_load((root / "configs" / "dopnet.yaml").read_text(encoding="utf-8"))
    cfg = make_config(base, "dopnet", 42)
    cfg["data_pipeline"]["emit_pilot_only_fast_time_add"] = True
    assert_historical_identity(cfg)
    setup_backend(cfg, torch.device("cuda"))
    old = load_historical_core(cfg, cache, sha256)
    if DERIVED_KEY in old:
        raise RuntimeError("The archived cache unexpectedly contains the derived key")
    sidecar, attestation = companion_paths(cache)
    # A failed or interrupted recheck must not leave an older "passed" proof
    # usable for subsequent training.
    _atomic_write_json({
        "status": "in_progress", "cache_sha256": HISTORICAL_CACHE_SHA256,
        "source_sha256": derivation_source_sha256(),
    }, attestation)
    generator = BatchChannelGenerator(
        cfg, device="cuda", seed=int(cfg["train"]["validation_seed"]) + 900001,
    )
    n_samples = int(cfg["train"]["val_samples"])
    old_keys = set(old)
    pilot_chunks = []
    max_abs = {key: 0.0 for key in old_keys}
    ls_roundtrip_max_abs = 0.0
    start = 0
    with torch.no_grad():
        for fresh_gpu in generator.iter_batches(n_samples, epoch=0,
                                                train_batch_size=batch_size):
            n = int(fresh_gpu["input"].shape[0])
            if set(fresh_gpu) != old_keys | {DERIVED_KEY}:
                raise RuntimeError(
                    "Regenerated validation has a different tensor-key schema: "
                    f"missing={sorted((old_keys | {DERIVED_KEY}) - set(fresh_gpu))}, "
                    f"extra={sorted(set(fresh_gpu) - old_keys - {DERIVED_KEY})}"
                )
            archived_batch = {key: value[start:start + n]
                              for key, value in old.items()}
            for key in sorted(old_keys):
                fresh_cpu = fresh_gpu[key].detach().cpu()
                error = _compare(fresh_cpu, archived_batch[key], key, start,
                                 rtol=CORE_RTOL, atol=CORE_ATOL)
                max_abs[key] = max(max_abs[key], error)
            ls_reconstruction = derive_pilot_only_fast_time_add(
                cfg, archived_batch, device="cuda",
            )
            direct = fresh_gpu[DERIVED_KEY].detach().cpu()
            chosen, error = _select_direct_feature(direct, ls_reconstruction, start)
            ls_roundtrip_max_abs = max(ls_roundtrip_max_abs, error)
            pilot_chunks.append(chosen)
            start += n
            print(f"verified historical validation {start}/{n_samples}", flush=True)
    if start != n_samples:
        raise RuntimeError(f"Regenerated {start} samples; expected {n_samples}")
    pilot_all = torch.cat(pilot_chunks, dim=0)
    if tuple(pilot_all.shape) != (n_samples, 2, 2, 15, 8, 48):
        raise RuntimeError("Derived pilot-only tensor shape differs from expectation")
    _atomic_save_tensor(pilot_all, sidecar)
    proof = {
        "status": "passed",
        "cache_sha256": HISTORICAL_CACHE_SHA256,
        "identity_sha256": HISTORICAL_IDENTITY_SHA256,
        "source_sha256": derivation_source_sha256(),
        "samples_checked": n_samples,
        "core_rtol": CORE_RTOL,
        "core_atol": CORE_ATOL,
        "core_max_abs_by_key": max_abs,
        "pilot_feature_source": PILOT_FEATURE_SOURCE,
        "archived_ls_roundtrip_max_abs": ls_roundtrip_max_abs,
        "sidecar_name": sidecar.name,
        "sidecar_bytes": sidecar.stat().st_size,
        "sidecar_sha256": _file_sha256(sidecar),
        "derivation": (
            "Direct pilot-only observation from the regenerated received waveform "
            "after all 29 archived core tensors passed the fixed validation "
            "comparison. Reconstruction from archived sparse LS is retained "
            "only as a float32 roundtrip diagnostic."
        ),
    }
    proof.update(historical_file_metadata(cache))
    _atomic_write_json(proof, attestation)
    assert_reconstruction_attestation(cfg, cache)
    print(f"Passed; derived sidecar: {sidecar}")
    print(f"Attestation: {attestation}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sha256", default=HISTORICAL_CACHE_SHA256)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Streaming comparison batch; must divide the 32-sample generator buffer")
    args = parser.parse_args()
    if args.batch_size < 1 or 32 % args.batch_size:
        raise ValueError("--batch-size must be a positive divisor of 32")
    verify_and_derive(args.cache, args.sha256, args.batch_size)


if __name__ == "__main__":
    main()
