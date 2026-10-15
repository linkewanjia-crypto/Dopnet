"""Restricted descriptor reader and read-only shared mapping of ONE fixed cache.

The accepted file is bound to the public fixed_validation.py full SHA, size,
29-field dtype/shape schema and contiguous layout. Pickle never constructs
Torch tensors or imports arbitrary globals: only descriptor-only rebuild,
FloatStorage/LongStorage sentinels and empty OrderedDict are allowed.
ZIP_STORED payloads are mapped with NumPy mode='r', not Torch copy-on-write.
An accessor exposes shape and ONLY a single first-axis sample. That sample is
copied into a small writable NumPy array before torch.from_numpy is called.
No whole-cache Torch mapping, unsafe torch.load, or new observations are used.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import struct
import sys
import zipfile

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PUBLIC_SCHEMA_SOURCE = ROOT / "code/training/main/fixed_validation.py"
EXPECTED_BYTES = 4_682_930_065
EXPECTED_SHA256 = "83d0e03524628b617671c42a73d3c458cc3395c31965a4e22295e8389f56e597"
DEFAULT_CACHE = ROOT / "data/reproduction_inputs/validation_895338a79e73ad3f.pt"


def _expected_schema():
    """Exactly fixed_validation.historical_tensor_schema, n=2000, lines 142--174."""
    n = 2000
    float_shapes = {
        "input": (n,2,5,25,8,128), "target_h": (n,2,2,25,8,128),
        "rx_data": (n,2,18,8,128), "tx_symbol": (n,2,2,18,128),
        "data_mask": (n,25), "pilot_symbol_mask": (n,25), "symbol_times_s": (n,25),
        "pilot_age_seconds": (n,25), "pilot_horizon_seconds": (n,25),
        "time_feature_scale_s": (n,), "noise_var": (n,), "active_re_signal_power": (n,),
        "snr_db": (n,), "delay_spread_ns": (n,), "max_doppler_hz": (n,),
        "add_input": (n,2,2,15,8,48), "target_add": (n,2,2,15,8,48),
        "add_input_slow": (n,2,2,15,8,48), "ls_grid": (n,2,2,25,8,128),
    }
    long_shapes = {"tx_label": (n,2,18,128)}
    for name in ("modulation_id", "channel_type_id", "pilot_spacing", "cp_len",
                 "cp_wave_samples", "waveform_fft_size", "channel_backend_id",
                 "snr_reference_id", "target_reference_id"):
        long_shapes[name] = (n,)
    return {**{k:(np.dtype("<f4"),v) for k,v in float_shapes.items()},
            **{k:(np.dtype("<i8"),v) for k,v in long_shapes.items()}}


@dataclass(frozen=True)
class _StorageKind:
    name: str
    dtype: str


FLOAT_KIND = _StorageKind("FloatStorage", "<f4")
LONG_KIND = _StorageKind("LongStorage", "<i8")


@dataclass(frozen=True)
class StorageDescriptor:
    key: str
    kind: _StorageKind
    numel: int


@dataclass(frozen=True)
class TensorDescriptor:
    storage: StorageDescriptor
    storage_offset: int
    shape: tuple
    stride: tuple


def _descriptor_rebuild(storage, offset, shape, stride, requires_grad, backward_hooks):
    if (type(storage) is not StorageDescriptor or type(offset) is not int or offset < 0
            or type(shape) is not tuple or type(stride) is not tuple
            or not shape or len(shape) != len(stride)
            or any(type(v) is not int or v < 1 for v in shape)
            or any(type(v) is not int or v < 1 for v in stride)
            or requires_grad is not False or type(backward_hooks) is not OrderedDict
            or backward_hooks):
        raise pickle.UnpicklingError("Unexpected tensor rebuild descriptor")
    return TensorDescriptor(storage, offset, shape, stride)


class DescriptorUnpickler(pickle.Unpickler):
    """No arbitrary GLOBAL, import, tensor construction or user callback execution."""
    def __init__(self, stream):
        super().__init__(stream)
        self.globals_seen, self.storages = [], {}

    def find_class(self, module, name):
        allowed = {
            ("torch._utils", "_rebuild_tensor_v2"): _descriptor_rebuild,
            ("torch", "FloatStorage"): FLOAT_KIND,
            ("torch", "LongStorage"): LONG_KIND,
            ("collections", "OrderedDict"): OrderedDict,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Forbidden pickle global: {module}.{name}")
        self.globals_seen.append([module, name])
        return allowed[(module, name)]

    def persistent_load(self, pid):
        if (type(pid) is not tuple or len(pid) != 5 or pid[0] != "storage"
                or pid[1] not in (FLOAT_KIND, LONG_KIND) or type(pid[2]) is not str
                or not pid[2].isdigit() or pid[3] != "cpu"
                or type(pid[4]) is not int or pid[4] < 1):
            raise pickle.UnpicklingError("Unexpected persistent storage descriptor")
        descriptor = StorageDescriptor(pid[2], pid[1], pid[4])
        if pid[2] in self.storages and self.storages[pid[2]] != descriptor:
            raise pickle.UnpicklingError("Inconsistent repeated storage descriptor")
        self.storages[pid[2]] = descriptor
        return descriptor


def _contiguous_stride(shape):
    result, running = [], 1
    for size in reversed(shape):
        result.append(running)
        running *= size
    return tuple(reversed(result))


def _metadata(path):
    s = path.stat()
    return {"size_bytes": s.st_size, "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns,
            "device": s.st_dev, "inode": s.st_ino}


def _hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stored_payload_offset(path, info, central_directory_start):
    if (info.compress_type != zipfile.ZIP_STORED or info.compress_size != info.file_size
            or info.flag_bits & 1):
        raise ValueError("Storage must be unencrypted ZIP_STORED with exact byte size")
    with path.open("rb") as stream:
        stream.seek(info.header_offset)
        header = stream.read(30)
        if len(header) != 30:
            raise ValueError("Truncated ZIP local header")
        signature, _, flags, compression, _, _, _, _, _, name_len, extra_len = struct.unpack(
            "<IHHHHHIIIHH", header)
        if signature != 0x04034B50 or compression != zipfile.ZIP_STORED or flags & 1:
            raise ValueError("Unexpected ZIP local header")
        name = stream.read(name_len).decode("utf-8" if flags & 0x800 else "cp437")
        if name != info.filename:
            raise ValueError("ZIP central/local member names disagree")
        offset = info.header_offset+30+name_len+extra_len
    if offset < 0 or offset+info.file_size > central_directory_start:
        raise ValueError("Storage payload range escapes ZIP data area")
    return offset


class ReadOnlyTensorAccessor:
    """A sample-copy accessor; no API materializes the entire mapped tensor."""
    def __init__(self, path, descriptor, payload_offset):
        self.descriptor, self.shape = descriptor, descriptor.shape
        self.dtype = np.dtype(descriptor.storage.kind.dtype)
        self.payload_offset = int(payload_offset)
        self.tensor_byte_offset = self.payload_offset+descriptor.storage_offset*self.dtype.itemsize
        self._map = np.memmap(path, mode="r", dtype=self.dtype,
                              offset=self.tensor_byte_offset, shape=self.shape, order="C")
        if self._map.flags.writeable or self._map.mode != "r":
            raise ValueError("Cache mapping is not read-only")

    def __getitem__(self, index):
        # Existing LS code also uses field[i,:,0]. The first axis is still
        # strictly ONE sample; all remaining indexing is on the small copy.
        if isinstance(index, tuple):
            if not index:
                raise TypeError("A single first-axis sample index is required")
            frame, remaining = index[0], index[1:]
        else:
            frame, remaining = index, ()
        if not isinstance(frame, (int, np.integer)) or isinstance(frame, (bool, np.bool_)):
            raise TypeError("Only a single integer sample index is supported")
        frame = int(frame)
        if not 0 <= frame < self.shape[0]:
            raise IndexError("Sample index outside historical cache")
        import torch
        sample = np.array(self._map[frame], copy=True, order="C")
        if not sample.flags.writeable or not sample.flags.owndata:
            raise ValueError("Per-sample copy must own writable memory")
        tensor = torch.from_numpy(sample)
        return tensor[remaining] if remaining else tensor

    def close(self):
        self._map._mmap.close()


class ReadOnlyTensorCache(dict):
    def __init__(self, values, receipt):
        super().__init__(values)
        self.receipt = receipt

    def close(self):
        for value in self.values():
            value.close()


def load_read_only_cache(path=DEFAULT_CACHE, *, expected_sha256=EXPECTED_SHA256):
    """Verify full source SHA before restricted pickle and shared read-only maps."""
    path = Path(path).resolve()
    before = _metadata(path)
    if before["size_bytes"] != EXPECTED_BYTES or expected_sha256 != EXPECTED_SHA256:
        raise ValueError("Unsupported cache size or identity SHA")
    actual_sha = _hash_file(path)
    if actual_sha != expected_sha256 or _metadata(path) != before:
        raise ValueError("Fixed cache SHA mismatch or source changed during hashing")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_members = [name for name in names if name.endswith("/data.pkl")]
        if len(metadata_members) != 1 or len(names) != len(set(names)):
            raise ValueError("Ambiguous ZIP metadata or duplicate member name")
        pkl_name = metadata_members[0]
        prefix = pkl_name[:-len("data.pkl")]
        if archive.read(prefix+"byteorder") != b"little":
            raise ValueError("Only the fixed little-endian cache is supported")
        raw = archive.read(pkl_name)
        if len(raw) > 64*1024:
            raise ValueError("Unexpectedly large tensor descriptor metadata")
        stream = io.BytesIO(raw)
        decoder = DescriptorUnpickler(stream)
        descriptors = decoder.load()
        if stream.tell() != len(raw):
            raise ValueError("Trailing pickle bytes are not accepted")
        schema = _expected_schema()
        if type(descriptors) is not dict or set(descriptors) != set(schema) or len(schema) != 29:
            raise ValueError("Historical 29-field schema mismatch")
        members = {name:archive.getinfo(name) for name in names}
        values, descriptions, ranges = {}, {}, []
        for name, descriptor in descriptors.items():
            if type(descriptor) is not TensorDescriptor:
                raise ValueError("Cache field is not a restricted tensor descriptor")
            dtype, shape = schema[name]
            if (np.dtype(descriptor.storage.kind.dtype) != dtype or descriptor.shape != shape
                    or descriptor.stride != _contiguous_stride(shape)
                    or descriptor.storage_offset != 0 or descriptor.storage.numel != math.prod(shape)):
                raise ValueError(f"Historical contiguous descriptor mismatch: {name}")
            member = prefix+"data/"+descriptor.storage.key
            info = members.get(member)
            if info is None or info.file_size != descriptor.storage.numel*dtype.itemsize:
                raise ValueError(f"Storage member size mismatch: {name}")
            payload = _stored_payload_offset(path, info, archive.start_dir)
            if payload % dtype.itemsize:
                raise ValueError("Storage payload is not dtype aligned")
            ranges.append((payload,payload+info.file_size))
            values[name] = ReadOnlyTensorAccessor(path,descriptor,payload)
            descriptions[name] = {"dtype":dtype.str,"shape":list(shape),"stride":list(descriptor.stride),
                "storage_key":descriptor.storage.key,"storage_offset_elements":descriptor.storage_offset,
                "zip_member":member,"zip_compression":"ZIP_STORED","payload_file_offset":payload,
                "storage_bytes":info.file_size,"memmap_mode":"r","mapping_writeable":False}
        if len({d.storage.key for d in descriptors.values()}) != 29:
            raise ValueError("Unexpected shared storage fields")
        ranges.sort()
        if any(ranges[i][1] > ranges[i+1][0] for i in range(len(ranges)-1)):
            raise ValueError("Overlapping storage payloads")
        if _metadata(path) != before:
            raise ValueError("Source cache metadata changed during descriptor mapping")
    receipt = {"source_path":str(path),"source_sha256":actual_sha,"source_metadata":before,
        "metadata_pickle_sha256":hashlib.sha256(raw).hexdigest(),"allowed_globals_seen":decoder.globals_seen,
        "public_schema_source":str(PUBLIC_SCHEMA_SOURCE),"public_schema_source_sha256":_hash_file(PUBLIC_SCHEMA_SOURCE),
        "mapping_mode":"NumPy memmap(mode='r'); read-only file mapping; per-sample copy only",
        "whole_cache_torch_copy_on_write_mapping":False,"fields":descriptions,
        "loader_sha256":_hash_file(Path(__file__))}
    return ReadOnlyTensorCache(values,receipt)

