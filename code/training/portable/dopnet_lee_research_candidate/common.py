"""New-run metadata and immutable commits; signatures are unkeyed content hashes."""
import hashlib
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def sign(value):
    if 'signature' in value:
        raise ValueError('Refuse to sign an already signed object')
    return dict(value, signature=digest(value))

def read_signed(path):
    result = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if result.get('signature') != digest({k: v for k, v in result.items() if k != 'signature'}):
        raise ValueError('Invalid signature: ' + str(path))
    return result

def check_row(row, epoch):
    from .protocol import phase_for_epoch
    expected = {'global_epoch': epoch, 'data_epoch': epoch,
                'phase': phase_for_epoch(epoch), 'fresh_frames': 12000,
                'optimizer_updates': 375, 'support_frames': 6000 if epoch == 1 else 0,
                'query_frames': 6000 if epoch == 1 else 0,
                'meta_inner_steps': 375 if epoch == 1 else 0,
                'online_lr': None if epoch == 1 else 0.001}
    for key, value in expected.items():
        if row.get(key) != value or (type(value) is int and type(row.get(key)) is not int):
            raise ValueError(f'Epoch {epoch}: invalid {key}')
    for key in ('train_loss', 'val_mse', 'val_nmse_db', 'train_seconds'):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Epoch {epoch}: nonfinite {key}')
    if min(row['train_loss'], row['val_mse'], row['train_seconds']) < 0:
        raise ValueError('Negative loss or duration')
    if epoch == 1:
        count = row.get('homogeneous_meta_episodes')
        if type(count) is not int or not 0 <= count <= 375:
            raise ValueError('Missing actual episode distribution ledger')

def checkpoint_path(folder, epoch):
    if type(epoch) is not int or not 1 <= epoch <= 40:
        raise ValueError('Invalid committed epoch')
    return Path(folder) / 'commits' / f'epoch{epoch:02d}' / 'state.pt'

def derive_summary(records):
    rows = [r['row'] for r in records]
    online = [r for r in records if r['epoch'] >= 2]
    selected = min(online, key=lambda r: r['row']['val_mse']) if online else None
    return {'committed_epochs': len(records), 'history': rows,
            'selected_epoch': selected['epoch'] if selected else None,
            'selected_state_sha256': selected['state_sha256'] if selected else None,
            'terminal_state_sha256': records[-1]['state_sha256'] if records else None,
            'meta_parent_sha256': records[0]['state_sha256'] if records else None,
            'all_experiments_complete': False}


def protect_output(output, *inputs):
    output = Path(output).resolve()
    for item in inputs:
        if item is None:
            continue
        protected = Path(item).resolve()
        require(output != protected and protected not in output.parents and output not in protected.parents,
                'Use independent output outside all input/source paths')
    return output

def scan_commits(folder, binding):
    root = Path(folder) / 'commits'
    if not root.exists():
        return []
    names = []
    for p in root.iterdir():
        if p.name.startswith('.') and p.name.endswith('.staging'):
            continue
        require(p.is_dir() and not p.is_symlink() and re.fullmatch(r'epoch(?:0[1-9]|[1-3][0-9]|40)', p.name),
                'Unexpected entry in independent immutable commit store')
        names.append(p.name)
    names.sort()
    require(names == [f'epoch{i:02d}' for i in range(1, len(names)+1)], 'Noncontiguous commit store')
    records = []
    for epoch, name in enumerate(names, 1):
        path = root / name
        require(set(p.name for p in path.iterdir()) == {'state.pt', 'commit.json'}, 'Incomplete published assets')
        record = read_signed(path / 'commit.json')
        require(record.get('schema') == 'dopnet_lee_research_commit_v1' and record.get('binding') == binding
                and record.get('epoch') == epoch, 'Wrong independent training identity')
        check_row(record['row'], epoch)
        submitted = datetime.fromisoformat(record['submitted_utc'])
        require(submitted.tzinfo is not None, 'Missing actual submission timezone')
        state = path / 'state.pt'
        require(not state.is_symlink() and state.stat().st_size == record['state_bytes']
                and file_hash(state) == record['state_sha256'], 'Submitted checkpoint bytes changed')
        require(record['meta_parent_sha256'] == (records[0]['state_sha256'] if epoch > 1 else None),
                'Offline parent chain changed')
        require(record['history_sha256'] == digest([r['row'] for r in records] + [record['row']]),
                'Submitted history hash differs')
        records.append(record)
    return records

def publish_commit(folder, state, binding, save_state, deadline_check):
    """Save full state into a new immutable directory; never replace old state."""
    epoch = state['epoch']
    check_row(state['history'][-1], epoch)
    root = Path(folder) / 'commits'
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f'epoch{epoch:02d}'
    if destination.exists():
        raise FileExistsError('Refuse to overwrite a submitted epoch')
    staging = root / f'.epoch{epoch:02d}.{uuid.uuid4().hex}.staging'
    staging.mkdir()
    path = staging / 'state.pt'
    save_state(state, path)
    with path.open('r+b') as stream:
        os.fsync(stream.fileno())
    record = sign({'schema': 'dopnet_lee_research_commit_v1', 'epoch': epoch,
                   'binding': binding, 'row': state['history'][-1],
                   'history_sha256': digest(state['history']),
                   'state_sha256': file_hash(path), 'state_bytes': path.stat().st_size,
                   'meta_parent_sha256': state['meta_parent_sha256'],
                   'submitted_utc': datetime.now(timezone.utc).isoformat()})
    write_json(staging / 'commit.json', record)
    deadline_check()
    # The directory publication is the transaction boundary. Derived pointers
    # may be reconstructed from these bytes after a crash.
    staging.rename(destination)
    if os.name == 'posix':
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return record

