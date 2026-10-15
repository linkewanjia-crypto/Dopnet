"""Build the independent static source release, excluding old plans and code."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import zlib

from .prepare_wimae import HERE, UPSTREAM_COMMIT, digest, file_hash, implementation_sources, verify_upstream

MODULES = ("prepare_wimae.py", "wimae_author.py", "train_wimae.py", "wimae_protocol.py",
           "evaluate_wimae.py", "queue_wimae.py", "verify_wimae.py", "report_wimae.py", "pack_wimae.py",
           "test_wimae_metadata.py", "test_wimae_pipeline_metadata.py", "WIMAE_REPRODUCTION_zh.md")


def build(output):
    verify_upstream()
    paths = [HERE / name for name in MODULES]
    paths += [p for p in (HERE / "vendor").rglob("*") if p.is_file() and
              (p.suffix in (".py", ".json") or p.name == "LICENSE")]
    snapshots = {}
    for path in sorted(paths):
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        snapshots["dl_baseline_expansion/" + path.relative_to(HERE).as_posix()] = path.read_bytes()
    records = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                      "crc32": format(zlib.crc32(data) & 0xffffffff, "08x")}
               for name, data in snapshots.items()}
    manifest = {"schema": "wimae_static_source_release_v1", "upstream_commit": UPSTREAM_COMMIT,
                "original_source_license": "MIT", "claim": "task_adapted_equal_frame_budget",
                "epochs": "20MAE+20CE", "frames_per_epoch": 12000, "seeds": [42, 43, 44],
                "physical_geometry": {"Nu": 2, "T": 25, "Nr": 8, "Nf": 128},
                "effective_frame_batch": 32, "generation_frame_batch": 32, "threegpp_link_chunk": 8,
                "implementation_sha256": implementation_sources(), "files": records,
                "models_executed_by_packaging": False, "runtime_GPU_qualification_required": True}
    manifest["signature"] = digest(manifest)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as z:
        for name, data in snapshots.items():
            z.writestr(name, data)
        z.writestr("SOURCE_RELEASE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    with ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise RuntimeError("Source ZIP CRC failed")
        for name, record in records.items():
            if hashlib.sha256(z.read(name)).hexdigest() != record["sha256"]:
                raise RuntimeError("Source ZIP SHA failed")
    temporary.replace(output)
    sha = file_hash(output)
    output.with_suffix(output.suffix + ".sha256").write_text(sha + "  " + output.name + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"package": str(output), "sha256": sha, "files": len(records),
                      "manifest_signature": manifest["signature"], "GPU_run": False}, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
