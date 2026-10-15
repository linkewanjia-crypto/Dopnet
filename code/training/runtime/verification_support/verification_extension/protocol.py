"""Metadata and file contracts for corrected receiver evaluations."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import numpy as np

CORE_CHANNELS = ('doubly_selective', 'tdl_b', 'tdl_c', 'tdl_d', 'cdl_b',
                 'cdl_c', 'cdl_d', 'tdl_a', 'tdl_e', 'cdl_a', 'cdl_e')
METHODS = ('pilot_covariance_lmmse', 'oce_bem_joint_cholesky', 'dpss_bem_joint_cholesky')
METHOD_LABELS = {'pilot_covariance_lmmse': 'Pilot covariance LMMSE',
                 'oce_bem_joint_cholesky': 'Joint OCE-BEM',
                 'dpss_bem_joint_cholesky': 'Joint DPSS-BEM'}
BASE_METHOD = {'oce_bem_joint_cholesky': 'oce_bem_joint_cholesky',
               'dpss_bem_joint_cholesky': 'dpss_bem_joint_cholesky'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_hashes():
    code = Path(__file__).resolve().parents[1]
    files = list((code / 'verification_extension').glob('*.py'))
    files += [code / 'run_verification.py']
    return {p.relative_to(code).as_posix(): file_hash(p) for p in sorted(files)}


def result_root(path):
    path = Path(path).expanduser().resolve()
    if path.name == 'cells' and (path.parent / 'evaluation.json').is_file():
        path = path.parent
    if not (path / 'evaluation.json').is_file():
        raise FileNotFoundError('Pass the completed dopnet_additions directory (or its cells directory): ' + str(path))
    return path


def signed_json(path):
    value = read_json(path)
    if value.get('signature') != digest({k: v for k, v in value.items() if k != 'signature'}):
        raise RuntimeError('Invalid metadata signature: ' + str(path))
    return value


def select_suite(cells, suite):
    def nominal(m):
        return (m['cp_len'] == 16 and m['pilot_spacing'] == 4
                and m['delay_spread_ns'] == 300.0)

    def main_nmse(m):
        return (nominal(m) and m['modulation'] == 'qpsk' and 'nmse' in m['studies'])

    def core(m):
        return (nominal(m) and m['channel'] in CORE_CHANNELS
                and m['doppler_hz'] in (800.0, 1200.0, 1600.0)
                and ((m['modulation'] == 'qpsk' and m['snr_db'] in (15.0, 25.0)
                      and 'nmse' in m['studies'])
                     or (m['modulation'] == '16qam' and m['snr_db'] == 20.0
                         and 'receiver' in m['studies'])))

    selected = [m for m in cells if suite == 'all' or
                (suite == 'core' and core(m)) or
                (suite == 'nominal' and main_nmse(m)) or
                (suite == 'manuscript' and (main_nmse(m) or core(m)))]
    expected = {'core': 99, 'nominal': 236, 'manuscript': 269, 'all': 636}[suite]
    if len(selected) != expected:
        raise RuntimeError('Frozen %s suite needs %d cells, found %d' % (suite, expected, len(selected)))
    return selected


def check_reference(path):
    path = result_root(path)
    experiment = signed_json(path / 'experiment.json')
    evaluation = signed_json(path / 'evaluation.json')
    if evaluation.get('experiment_signature') != experiment['signature']:
        raise RuntimeError('Experiment/evaluation identities differ')
    if len(evaluation['cells']) != 636:
        raise RuntimeError('Expected the original full 636-cell score archive')
    for cell in evaluation['cells']:
        if not (path / 'cells' / (cell['cell_id'] + '.npz')).is_file():
            raise FileNotFoundError('Missing frozen reference cell ' + cell['cell_id'])
    return path, experiment, evaluation


def frame_slice(arrays, start, stop):
    """Slice frame arrays while leaving scalar/index/public geometry intact."""
    count = len(arrays['noise_var'])
    batched = {'y_pilot', 'x_pilot', 'target_h', 'rx_data', 'tx_label', 'noise_var', 'rx_time'}
    return {key: value[start:stop] if key in batched and np.asarray(value).ndim
            and np.asarray(value).shape[0] == count else value
            for key, value in arrays.items()}


@contextmanager
def writer_lock(output, filename='.verification.lock'):
    """OS-released lock; no stale lock file needs deletion after interruption."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if Path(filename).name != filename:
        raise ValueError('Lock filename must stay in output')
    handle = (output / filename).open('a+b')
    if handle.seek(0, os.SEEK_END) == 0:
        handle.write(b'0'); handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        handle.close()
