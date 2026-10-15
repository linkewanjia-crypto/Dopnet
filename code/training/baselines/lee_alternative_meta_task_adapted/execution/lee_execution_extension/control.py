"""Standard-library source, authorization, migration and commit-control gates."""
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

LEGACY_SOURCE_SHA = 'c90ac58274b8ccb249acf6b20e3f33d354dfceb46096591d0d5753441639cdc1'
LEGACY_DEADLINE = '2026-10-02T13:46:32.725968Z'
READER_SHA = '562f9130f1e8a2dbd632c05bfaf57ba4de785a161d9c6531f8e5f80e30b9c714'
SEEDS = (42, 43, 44)
TASKS = tuple(k+str(s) for s in SEEDS for k in ('train_seed', 'verify_seed'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16*2**20), b''):
            h.update(block)
    return h.hexdigest()


def sign(value):
    require('signature' not in value, 'Never re-sign an already signed identity')
    return dict(value, signature=digest(value))


def signed(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    require(value.get('signature') == digest({k:v for k,v in value.items() if k != 'signature'}), 'Invalid signed metadata')
    return value


def write(path, value):
    import uuid
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0, 'Timestamp must explicitly use UTC')
    return parsed


def source(root, expected_sha=None):
    root = Path(root).resolve(); path = root/'SOURCE_MANIFEST.json'
    if expected_sha is not None:
        require(file_hash(path) == expected_sha, 'Source manifest bytes differ')
    m = signed(path)
    for name, row in m['members'].items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts and ':' not in name and '\\' not in name, 'Unsafe source member')
        p = root/name
        require(p.is_file() and not p.is_symlink() and file_hash(p) == row['sha256'] and p.stat().st_size == row['bytes'], 'Source member changed: '+name)
    return m


def verify_source(root):
    m = source(root)
    require(m.get('version') == '1.0.0' and m.get('schema') == 'lee_execution_extension_source_v1', 'Unknown execution extension')
    return m


def check_control(c):
    require(c.get('schema') == 'lee_execution_control_v1' and c.get('mode') == 'until_complete_or_explicit_user_stop' and c.get('deadline_utc') is None, 'Unknown execution authorization mode')
    require(c.get('legacy_deadline_utc') == LEGACY_DEADLINE and c.get('legacy_source_manifest_sha256') == LEGACY_SOURCE_SHA, 'Legacy source/deadline binding differs')
    require(c.get('authorization_time_basis') == 'agent_recorded_after_actual_user_message', 'Authorization timestamp is an agent record, not an invented message send time')
    require(isinstance(c.get('authorization_text'), str) and c['authorization_text'].strip() and timestamp(c['authorization_utc']) <= timestamp(c['created_utc']), 'Missing actual user authorization')
    require(c.get('all_experiments_complete') is False and c.get('purchase_or_renew_authorized') is False, 'Authorization must not imply purchases/completion')
    for key in ('legacy_code_root', 'legacy_results', 'legacy_launch', 'new_output', 'project', 'parent_receipt', 'forti_code_root'):
        require(isinstance(c.get(key), str) and PurePosixPath(c[key]).is_absolute() and '..' not in PurePosixPath(c[key]).parts, 'Missing safe absolute server execution path: '+key)
    return c


def check_migration(m, c):
    require(m.get('schema') == 'lee_execution_migration_v1' and m.get('status') == 'immutable_prefix_copied_and_verified' and m.get('execution_control_signature') == c['signature'], 'Migration/control identity differs')
    require(m.get('legacy_queue_status') == 'incomplete_deadline' and type(m.get('legacy_queue_exit_code')) is int and m['legacy_queue_exit_code'] == 75, 'Legacy queue must naturally exit75')
    require(set(m.get('prefix_commits', {})) == {'42', '43', '44'}, 'Missing seed prefix ledger')
    for seed in SEEDS:
        rows = m['prefix_commits'][str(seed)]
        require(len(rows) <= 40 and [x['epoch'] for x in rows] == list(range(1, len(rows)+1)), 'Migration prefix must be contiguous within40')
        for row in rows:
            require(type(row['epoch']) is int and timestamp(row['submitted_utc']) < timestamp(LEGACY_DEADLINE), 'Legacy prefix submitted after old deadline')
            require(all(isinstance(row[k],str) and len(row[k]) == 64 for k in ('state_sha256','commit_sha256')), 'Missing immutable prefix SHA')
    require(0 < len(m['prefix_commits']['42']) <= 40 and any(len(m['prefix_commits'][str(s)]) < 40 for s in SEEDS), 'No interrupted same-budget training remains')
    return m


def stop_requested(control_path, control):
    path = Path(control_path).parent/'USER_STOP.json'
    if not path.exists():
        return False
    stop = signed(path)
    require(stop.get('schema') == 'lee_execution_user_stop_v1' and stop.get('status') == 'user_requested_stop' and stop.get('execution_control_signature') == control['signature'], 'Invalid/mismatched user stop sentinel')
    timestamp(stop['requested_utc'])
    return True


def process_words():
    if os.name != 'posix':
        return []
    result = []
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name) == os.getpid():
            continue
        try:
            words = [x.decode(errors='replace') for x in p.read_bytes().split(b'\0') if x]
        except OSError:
            continue
        result.append((int(p.parent.name), words))
    return result


def no_live_training(allowed_pid=None):
    tokens = ('lee_meta_equal40.queue', 'lee_meta_equal40.train', 'lee_meta_equal40.probe', 'lee_execution_extension.controller', 'lee_execution_extension.worker')
    for pid, words in process_words():
        if pid != allowed_pid:
            require(not any(t in ' '.join(words) for t in tokens), 'Live Lee controller/worker/probe; retain serial execution')


class Context:
    def __init__(self, control_path, migration_path, legacy_code_root, evidence_root=None):
        self.control_path = Path(control_path); self.migration_path = Path(migration_path)
        self.control = check_control(signed(self.control_path)); self.migration = check_migration(signed(self.migration_path), self.control)
        own = verify_source(Path(__file__).resolve().parent.parent)
        require(own['signature'] == self.control['extension_source_signature'], 'Execution extension source changed')
        self.legacy_root = Path(legacy_code_root).resolve(); source(self.legacy_root, LEGACY_SOURCE_SHA)
        self.evidence_root = Path(evidence_root) if evidence_root is not None else self.migration_path.parent
        self.migration_sha = file_hash(self.migration_path)
        qpath = self.evidence_root/'legacy_queue_state.json'; lpath = self.evidence_root/'legacy_launch_receipt.json'
        require(file_hash(qpath) == self.migration['legacy_queue_sha256'] and file_hash(lpath) == self.migration['legacy_launch_sha256'], 'Legacy terminal evidence SHA differs')
        q = signed(qpath); launch = signed(lpath)
        require(q.get('schema') == 'lee_serial_training_queue_v1' and q.get('status') == 'incomplete_deadline' and type(q.get('exit_code')) is int and q['exit_code'] == 75 and type(launch.get('pid')) is int, 'Invalid actual legacy terminal evidence')
        allowed = {'probe_micro'+str(x) for x in (1,2,4,8)}|set(TASKS)
        require(set(q.get('tasks',{})) <= allowed and all(q['tasks'].get('probe_micro'+str(x),{}).get('status') == 'complete' and type(q['tasks']['probe_micro'+str(x)].get('returncode')) is int and q['tasks']['probe_micro'+str(x)]['returncode'] == 0 for x in (1,2,4,8)), 'Legacy actual probe/task provenance differs')

    def check(self):
        require(signed(self.control_path) == self.control and file_hash(self.migration_path) == self.migration_sha, 'Execution control/migration bytes changed')
        if stop_requested(self.control_path, self.control):
            from lee_meta_equal40.train import DeadlineReached
            raise DeadlineReached('Explicit user stop requested under execution extension')

    def receipt_path(self, seed, epoch):
        return self.evidence_root/'execution_commits'/f'lee_meta_seed{seed}'/f'epoch{epoch:02d}.json'

    def check_commit_control(self, folder, record):
        seed = record['binding']['seed']; epoch = record['epoch']
        require(type(seed) is int and seed in SEEDS and type(epoch) is int and 1 <= epoch <= 40, 'Invalid controlled commit identity')
        prefix = self.migration['prefix_commits'][str(seed)]
        path = Path(folder)/'commits'/f'epoch{epoch:02d}'/'commit.json'
        if epoch <= len(prefix):
            r = prefix[epoch-1]
            require(record['state_sha256'] == r['state_sha256'] and record['state_bytes'] == r['state_bytes'] and file_hash(path) == r['commit_sha256'] and record['submitted_utc'] == r['submitted_utc'], 'Migrated prefix bytes changed')
            require(timestamp(record['submitted_utc']) < timestamp(LEGACY_DEADLINE), 'Invalid legacy submission time')
        else:
            r = signed(self.receipt_path(seed, epoch))
            require(r.get('schema') == 'lee_execution_commit_control_v1' and r.get('status') == 'committed' and r.get('execution_control_signature') == self.control['signature'] and r.get('extension_source_signature') == self.control['extension_source_signature'] and r.get('migration_receipt_sha256') == self.migration_sha, 'Missing exact execution commit receipt')
            require(type(r.get('epoch')) is int and type(r.get('seed')) is int and r['epoch'] == epoch and r['seed'] == seed and r.get('commit_sha256') == file_hash(path) and r.get('state_sha256') == record['state_sha256'] and type(r.get('state_bytes')) is int and r['state_bytes'] == record['state_bytes'] and r.get('submitted_utc') == record['submitted_utc'], 'Extension commit SHA/bytes/time differ')
            require(timestamp(record['submitted_utc']) >= timestamp(self.migration['copied_utc']) and timestamp(r['committed_utc']) >= timestamp(record['submitted_utc']), 'Extension submission precedes its authorized migration')


def activate_for_read(control_path, migration_path, legacy_code_root, evidence_root=None):
    """Install explicit metadata overlay, returning Context; no Torch/NumPy import.

    evidence_root is the live new_output, or archived relocated extension evidence.
    The original scientific source files, recipe, binding and state bytes are unchanged.
    """
    from .overlay import install
    context = Context(control_path, migration_path, legacy_code_root, evidence_root)
    sys.path.insert(0, str(context.legacy_root))
    prepare = importlib.import_module('lee_meta_equal40.prepare')
    require(Path(prepare.__file__).resolve() == context.legacy_root/'lee_meta_equal40/prepare.py', 'Wrong legacy scientific module origin')
    install(context)
    require('torch' not in sys.modules, 'Metadata overlay imported Torch')
    return context
