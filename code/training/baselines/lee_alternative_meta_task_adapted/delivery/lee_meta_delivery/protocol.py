"""Metadata-only gates for eight methods /24 routes /269 frozen cells.

This downstream version does not edit or replace training0.3.1 or parent science.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

DEADLINE = '2026-10-02T13:46:32.725968Z'
WIMAE_ARCHIVE_SHA = 'b4220c699f92d82b147365444e59684fc2a61139d5056295a496a6e04fddea13'
TRAINING_MANIFEST_SHA = 'c90ac58274b8ccb249acf6b20e3f33d354dfceb46096591d0d5753441639cdc1'
TRAINING_SOURCE_SIGNATURE = '209af087fe72ed5411d93f3d5219930064cce2893b096f34352f268076e13b24'
READER_SHA = '562f9130f1e8a2dbd632c05bfaf57ba4de785a161d9c6531f8e5f80e30b9c714'
SEEDS = (42, 43, 44)
NEW_METHOD = 'lee_alternative_meta_task_adapted'
LABELS = {
    'dopnet': 'DOP-Net', 'dopnet_lite': 'DOP-Net-Lite',
    'lee3ddual': 'Dual3D (supervised adapter)',
    'wimae_ce': 'WiMAE (supervised adapter)',
    'fortitran': 'FortiTran (common recipe)',
    'fortitran_budget_matched': 'FortiTran (equal budget)',
    'wimae_author': 'WiMAE (20 SSL +20 CE)',
    NEW_METHOD: 'Lee A-ML (1 meta +39 joint)'}
CLAIM = 'fulltext_guided_original_algorithm_equal_frame_budget_task_adaptation'
SUMMARY_NAMES = {'cell_summary.csv', 'cell_paired_differences.csv',
                 'macro_summary.csv', 'macro_paired_differences.csv', 'receiver_diagnostics.csv'}
ASSET_NAMES = ('equal_budget_dl_comparison.tex', 'equal_budget_dl_comparison.pdf',
               'equal_budget_dl_comparison.png')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16*2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def signed(path):
    result = read(path)
    require(result.get('signature') == digest({k:v for k,v in result.items() if k != 'signature'}), 'Invalid metadata signature')
    return result


def sign(value):
    require('signature' not in value, 'Refuse to re-sign another identity')
    return dict(value, signature=digest(value))


def write(path, value):
    import os, uuid
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def safe_name(name):
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name and p.as_posix() == name


def check_deadline():
    """Compatibility call site: signed continuous policy and explicit STOP only."""
    from .execution import check_execution
    check_execution()


class DeadlineReached(RuntimeError):
    """Compatibility exception for preserving completed work on explicit STOP."""


def verify_source(root):
    root = Path(root).resolve(); m = signed(root/'SOURCE_MANIFEST.json')
    require(m.get('version') == '0.4.2' and m.get('claim') == CLAIM, 'Wrong downstream version/claim')
    for name, record in m['members'].items():
        require(safe_name(name) and file_hash(root/name) == record['sha256'] and (root/name).stat().st_size == record['bytes'], 'Downstream source changed: '+name)
    return m


def bind_training_source(root):
    root = Path(root).resolve()
    require(file_hash(root/'SOURCE_MANIFEST.json') == TRAINING_MANIFEST_SHA, 'Use reviewed immutable training0.3.1')
    m = signed(root/'SOURCE_MANIFEST.json')
    require(m['signature'] == TRAINING_SOURCE_SIGNATURE, 'Training release signature differs')
    for name, record in m['members'].items():
        require(safe_name(name) and file_hash(root/name) == record['sha256'], 'Training source changed: '+name)
    return m


def require_training_receipt(proof, seed, manifest, cfg, execution):
    require(type(seed) is int and seed in SEEDS, 'Wrong training seed')
    from lee_meta_equal40.train import binding_for
    require(proof.get('status') == 'single_seed_training_metadata_pass' and type(proof.get('exit_code')) is int and proof['exit_code'] == 0
            and proof.get('training_delivery_qualified') is True and type(proof.get('committed_epochs')) is int and proof['committed_epochs'] == 40
            and proof.get('binding') == binding_for(manifest, cfg, seed)
            and proof.get('actual_restricted_reader_sha256') == READER_SHA
            and proof.get('rng_presence_verified') is True and proof.get('rng_boundary_bitwise_verified') is False
            and proof.get('all_experiments_complete') is False, 'No actual exact40 training metadata receipt')
    selected = proof.get('selected_epoch')
    require(type(selected) is int and 2 <= selected <= 40, 'Selected model must be an online-stage best, not forced last40')
    from .execution import require_fields
    require_fields(proof, execution)


def require_routes(routes, inherited):
    require(len(inherited) == 21 and len(routes) == 24 and routes[:21] == inherited, 'All21 inherited routes must be reused exactly')
    require(routes[21:] == [dict(method=NEW_METHOD, training_seed=s, trained=True) for s in SEEDS], 'Lee routes/seeds differ')
    require(list(dict.fromkeys(r['method'] for r in routes)) == list(LABELS), 'Expected eight method order differs')
    for method in LABELS:
        rr = [r for r in routes if r['method'] == method]
        require([r['training_seed'] for r in rr] == list(SEEDS) and all(r['trained'] is True for r in rr), 'Incomplete three-seed route')


def check_statistics(statistics, ev, output):
    require(statistics.get('signature') == digest({k:v for k,v in statistics.items() if k != 'signature'}), 'Statistics signature differs')
    require(statistics.get('evaluation_signature') == ev['signature'], 'Statistics belongs to another evaluation')
    require(type(statistics.get('bootstrap')) is int and statistics['bootstrap'] == 2000, 'Exactly2000 cluster bootstrap replicates required')
    require(math.isclose(statistics.get('t_critical_df2', 0), math.sqrt(2*.95**2/(1-.95**2)), rel_tol=1e-10), 'Incorrect df2 critical value')
    require(set(statistics.get('summary_csv_sha256', {})) == SUMMARY_NAMES, 'Exactly five summary CSVs required')
    for name, h in statistics['summary_csv_sha256'].items():
        require(file_hash(Path(output)/'summary'/name) == h, 'Summary CSV changed: '+name)


def read_parent_archive(archive):
    """Reuse the actual accepted seven-method delivery, with all CRC/member SHA.

The upstream full semantic checker was already executed on this exact SHA.
Final uniform local delivery must also execute its original full checker.
"""
    archive = Path(archive)
    require(file_hash(archive) == WIMAE_ARCHIVE_SHA, 'Parent is not the actual accepted WiMAE delivery')
    with ZipFile(archive) as z:
        names = [x.filename for x in z.infolist() if not x.is_dir()]
        require(len(names) == len(set(names)) and all(safe_name(n) for n in names), 'Unsafe/duplicate parent members')
        require(z.testzip() is None, 'Parent ZIP CRC failed')
        m = json.loads(z.read('DELIVERY_MANIFEST.json'))
        require(m.get('signature') == digest({k:v for k,v in m.items() if k != 'signature'}) and m.get('paper_ready') is True and m.get('pending') == [], 'Parent full delivery is incomplete')
        require(set(names) == set(m['files'])|{'DELIVERY_MANIFEST.json'}, 'Parent membership differs')
        for name, expected in m['files'].items():
            require(hashlib.sha256(z.read(name)).hexdigest() == expected, 'Parent member SHA differs: '+name)
        ev = json.loads(z.read('evaluation/evaluation.json'))
        require(ev.get('signature') == digest({k:v for k,v in ev.items() if k != 'signature'}) and len(ev['cells']) == 269 and len(ev['routes']) == 21 and ev['samples_per_cell'] == 512 and ev['batch_size'] == 8, 'Parent frozen suite differs')
    return ev, m


def no_live_training():
    import os
    if os.name != 'posix':
        return
    forbidden = {b'lee_meta_equal40.queue', b'lee_meta_equal40.train', b'lee_meta_equal40.probe',
                 b'lee_execution_extension.controller', b'lee_execution_extension.worker'}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if int(path.parent.name) == os.getpid():
            continue
        try:
            raw=path.with_name('stat').read_text()
            if raw[raw.rfind(')')+2:].split()[0] == 'Z':
                continue
            words = path.read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        except (OSError, IndexError):
            raise ValueError('Cannot determine training/probe process liveness')
        require(not forbidden.intersection(words), 'Training/probe/queue must naturally exit before downstream work')
