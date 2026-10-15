"""Extract the immutable historical validation cache from the old project ZIP.

Streaming extraction needs disk space for the 4.68 GB output but never holds
the tensor payload in RAM or invokes PyTorch/CUDA.  The cache is intentionally
not bundled with this source release.
"""

import argparse
import hashlib
import os
import uuid
import zipfile
from pathlib import Path

from fixed_validation import (
    HISTORICAL_CACHE_BYTES, HISTORICAL_CACHE_NAME, HISTORICAL_CACHE_SHA256,
)


ARCHIVE_MEMBER_SUFFIX = "/cache/" + HISTORICAL_CACHE_NAME


def find_historical_member(archive):
    with zipfile.ZipFile(archive, "r") as bundle:
        matches = [name for name in bundle.namelist()
                   if name.endswith(ARCHIVE_MEMBER_SUFFIX)]
    if len(matches) != 1:
        raise RuntimeError("Expected one archived historical validation cache")
    return matches[0]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_checked_member(archive, member, output, expected_size, expected_sha256):
    """Atomically extract one known member after size, CRC and SHA checks."""
    archive = Path(archive)
    output = Path(output)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output.exists():
        if not output.is_file() or output.stat().st_size != expected_size:
            raise RuntimeError(f"Existing output is not the expected cache: {output}")
        actual = _sha256(output)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Existing output SHA-256 mismatch: {actual}; expected {expected_sha256}"
            )
        print(f"Already verified: {output} sha256={actual}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial." + uuid.uuid4().hex)
    try:
        with zipfile.ZipFile(archive, "r") as bundle:
            try:
                info = bundle.getinfo(member)
            except KeyError as exc:
                raise RuntimeError(f"Archive lacks required member: {member}") from exc
            if info.file_size != expected_size:
                raise RuntimeError(
                    f"Archive member size {info.file_size} != {expected_size} bytes"
                )
            digest = hashlib.sha256()
            copied = 0
            with bundle.open(info, "r") as source, temporary.open("xb") as target:
                while True:
                    block = source.read(16 * 1024 * 1024)
                    if not block:
                        break
                    target.write(block)
                    digest.update(block)
                    copied += len(block)
                target.flush()
                os.fsync(target.fileno())
            # Reading the ZIP member to EOF also checks its recorded CRC.
            actual = digest.hexdigest()
            if copied != expected_size or actual != expected_sha256:
                raise RuntimeError(
                    f"Extracted cache mismatch: bytes={copied}, sha256={actual}; "
                    f"expected bytes={expected_size}, sha256={expected_sha256}"
                )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Extracted and verified: {output} sha256={expected_sha256}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path,
                        help="Path to the historical DOPNet.zip on the server")
    parser.add_argument("--output", type=Path,
                        default=Path("cache") / HISTORICAL_CACHE_NAME)
    args = parser.parse_args()
    extract_checked_member(
        args.archive, find_historical_member(args.archive), args.output,
        HISTORICAL_CACHE_BYTES, HISTORICAL_CACHE_SHA256,
    )


if __name__ == "__main__":
    main()
