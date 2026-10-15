"""Portable read-only access to original archived pilot observations."""
from pathlib import Path
import hashlib
import io
import json
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
PROJECT = HERE.parents[3] / "code/training/main"
DATA_ROOT = HERE.parents[3] / "data/reproduction_inputs/support_diagnostics_20261007/raw_observations"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "classical_extension"))
from dopnet_classical.bem import WaveformGeometry

def configure(path):
    global DATA_ROOT
    DATA_ROOT = Path(path).resolve()

def inventory():
    return json.loads((DATA_ROOT / "inventory.json").read_text(encoding="utf-8"))

def raw_batches():
    info = inventory()
    for item in info["batches"]:
        path = (DATA_ROOT / item["relative_path"]).resolve()
        if not path.is_relative_to(DATA_ROOT):
            raise ValueError("Observation path escapes data root")
        raw = path.read_bytes()
        if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("Original observation identity differs")
        entry = dict(item, expected_sha256=item["sha256"])
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            yield item["population"], entry, {key: arrays[key] for key in arrays.files}

def geometry(nfft, occupied, cp, scs=7500., symbols=25, pv=(0,4,8,12,16,20,24)):
    return WaveformGeometry(nfft=int(nfft), n_active=int(occupied), cp_wave_samples=int(cp),
        subcarrier_spacing_hz=float(scs), num_symbols=int(symbols), pilot_visible=tuple(pv),
        data_visible=tuple(t for t in range(symbols) if t not in pv),
        active_signed_bins=tuple(range(-occupied//2, occupied//2)))

def new_output(path, source):
    path, source = Path(path).resolve(), Path(source).resolve()
    if path == source or path.is_relative_to(source) or source.is_relative_to(path):
        raise ValueError("Output and input trees must be separate")
    if path.exists() and any(path.iterdir()):
        raise ValueError("Output directory must be new or empty")
    path.mkdir(parents=True, exist_ok=True)
    return path
