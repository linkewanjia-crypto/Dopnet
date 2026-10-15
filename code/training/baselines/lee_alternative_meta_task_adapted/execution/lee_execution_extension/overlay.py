"""Explicit in-process administrative overlay; frozen scientific bodies retained."""
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import re
import uuid
from .control import require, sign, signed, write, file_hash


def scan_commits(context, folder, binding):
    a = importlib.import_module('lee_meta_equal40.artifacts')
    root = Path(folder)/'commits'
    if not root.exists():
        return []
    names = []
    for p in root.iterdir():
        if p.name.startswith('.') and p.name.endswith('.staging'):
            continue
        require(p.is_dir() and not p.is_symlink() and re.fullmatch(r'epoch(?:0[1-9]|[1-3][0-9]|40)',p.name), 'Unexpected commit-store entry')
        names.append(p.name)
    names.sort(); require(names == [f'epoch{i:02d}' for i in range(1,len(names)+1)], 'Noncontiguous commits')
    records = []
    for epoch,name in enumerate(names,1):
        p = root/name; require(set(x.name for x in p.iterdir()) == {'state.pt','commit.json'}, 'Unexpected published commit assets')
        require(not any(x.is_symlink() for x in p.iterdir()), 'Symlinked commit assets forbidden')
        r = a.read_signed(p/'commit.json')
        require(r.get('binding') == binding and r.get('epoch') == epoch, 'Commit binding/epoch differs')
        a.check_row(r['row'],epoch)
        state = p/'state.pt'; require(state.stat().st_size == r['state_bytes'] and a.file_hash(state) == r['state_sha256'], 'Submitted state bytes changed')
        require(r.get('meta_parent_sha256') == (records[0]['state_sha256'] if epoch>1 else None), 'Meta parent differs')
        context.check_commit_control(folder,r); records.append(r)
    return records


def publish_commit(context, folder, state, binding, save_state, deadline_check):
    """Original immutable store plus durable external execution-control transaction."""
    a = importlib.import_module('lee_meta_equal40.artifacts')
    context.check(); epoch = state['epoch']; a.check_row(state['history'][-1],epoch)
    root = Path(folder)/'commits'; root.mkdir(parents=True,exist_ok=True)
    destination = root/f'epoch{epoch:02d}'; require(not destination.exists(), 'Never overwrite submitted state')
    staging = root/f'.epoch{epoch:02d}.{uuid.uuid4().hex}.staging'; staging.mkdir()
    path = staging/'state.pt'; save_state(state,path)
    with path.open('r+b') as stream:
        os.fsync(stream.fileno())
    record = a.sign(dict(schema='lee_full_epoch_commit_v1',epoch=epoch,binding=binding,row=state['history'][-1],history_sha256=a.digest(state['history']),state_sha256=a.file_hash(path),state_bytes=path.stat().st_size,meta_parent_sha256=state['meta_parent_sha256'],submitted_utc=datetime.now(timezone.utc).isoformat()))
    a.write_json(staging/'commit.json',record); context.check()
    receipt_path = context.receipt_path(binding['seed'],epoch)
    require(not receipt_path.exists(), 'Existing extension receipt requires explicit recovery; never overwrite blindly')
    intent = dict(schema='lee_execution_commit_control_v1',status='publication_intent',execution_control_signature=context.control['signature'],extension_source_signature=context.control['extension_source_signature'],migration_receipt_sha256=context.migration_sha,seed=binding['seed'],epoch=epoch,state_sha256=record['state_sha256'],state_bytes=record['state_bytes'],commit_sha256=file_hash(staging/'commit.json'),submitted_utc=record['submitted_utc'])
    write(receipt_path,sign(intent)); context.check(); staging.rename(destination)
    if os.name == 'posix':
        fd = os.open(root,os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    write(receipt_path,sign(dict(intent,status='committed',committed_utc=datetime.now(timezone.utc).isoformat())))
    return record


def recover_intents(context, training_root):
    """Only reconcile exact already-published bytes; no state deserialization."""
    for path in (context.evidence_root/'execution_commits').glob('lee_meta_seed*/epoch*.json'):
        r = signed(path)
        if r.get('status') != 'publication_intent':
            continue
        require(r.get('execution_control_signature') == context.control['signature'] and r.get('migration_receipt_sha256') == context.migration_sha, 'Foreign publication intent')
        folder = Path(training_root)/'checkpoints'/f'lee_meta_seed{r["seed"]}'/'commits'/f'epoch{r["epoch"]:02d}'
        require(folder.is_dir() and set(x.name for x in folder.iterdir()) == {'state.pt','commit.json'}, 'Unpublished intent remains incomplete; retain staging for diagnosis')
        record = signed(folder/'commit.json')
        require(file_hash(folder/'commit.json') == r['commit_sha256'] and file_hash(folder/'state.pt') == r['state_sha256'] and (folder/'state.pt').stat().st_size == r['state_bytes'] and record['binding']['seed'] == r['seed'] and record['epoch'] == r['epoch'], 'Publication recovery bytes differ')
        write(path,sign(dict({k:v for k,v in r.items() if k!='signature'},status='committed',committed_utc=datetime.now(timezone.utc).isoformat(),recovered_published_intent=True)))


def install(context):
    a = importlib.import_module('lee_meta_equal40.artifacts'); t = importlib.import_module('lee_meta_equal40.train'); v = importlib.import_module('lee_meta_equal40.verify_training')
    if getattr(t,'_extension_context',None) is not None:
        require(t._extension_context.control == context.control and t._extension_context.migration_sha == context.migration_sha, 'Cannot swap execution contexts in one process')
        t._extension_context = context
        return
    t._extension_context = context
    original_online = t.online_step; original_verify = v.verify; original_write = t.write_json
    def online(*args, **kwargs):
        kwargs['deadline_check'] = t._extension_context.check
        return original_online(*args,**kwargs)
    def controlled_scan(folder,binding):
        return scan_commits(t._extension_context,folder,binding)
    def controlled_publish(folder,state,binding,save_state,deadline_check):
        active = t._extension_context
        return publish_commit(active,folder,state,binding,save_state,active.check)
    def verify(folder,manifest_path,project,seed,reader):
        context = t._extension_context
        result = original_verify(folder,manifest_path,project,seed,reader)
        return sign(dict({k:x for k,x in result.items() if k!='signature'},execution_control_signature=context.control['signature'],extension_source_signature=context.control['extension_source_signature'],migration_receipt_sha256=context.migration_sha,execution_mode=context.control['mode'],legacy_deadline_superseded=True))
    def diagnostic_write(path,value):
        context = t._extension_context
        if Path(path).name == 'incomplete.json' and value.get('reason') == 'original_global_deadline':
            value = sign(dict({k:x for k,x in value.items() if k!='signature'},reason='explicit_user_stop_under_execution_extension',execution_control_signature=context.control['signature']))
        original_write(path,value)
    t.check_deadline = lambda: t._extension_context.check(); t.online_step = online
    t.scan_commits = a.scan_commits = v.scan_commits = controlled_scan
    t.publish_commit = a.publish_commit = controlled_publish
    t.write_json = diagnostic_write; v.verify = verify
