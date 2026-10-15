"""Byte-preserving migration after legacy controller/worker naturally exited75."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from .control import (LEGACY_SOURCE_SHA, SEEDS, check_control, file_hash, no_live_training,
                      require, sign, signed, source, write)


def copy_exact(src, dst, files, relative):
    src,dst = Path(src),Path(dst)
    require(src.is_file() and not src.is_symlink() and not dst.exists(), 'Unsafe/duplicate migration file')
    before = dict(bytes=src.stat().st_size,sha256=file_hash(src))
    dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dst)
    require(dst.stat().st_size == before['bytes'] and file_hash(dst) == before['sha256'] and src.stat().st_size == before['bytes'] and file_hash(src) == before['sha256'], 'Source/destination bytes changed during copy')
    files[relative] = dict(before,source=str(src.resolve()))


def legacy_terminal(control):
    qpath = Path(control['legacy_results'])/'queue_state.json'; q = signed(qpath)
    launch = signed(control['legacy_launch'])
    require(q.get('status') == 'incomplete_deadline' and type(q.get('exit_code')) is int and q['exit_code'] == 75, 'Legacy queue has not naturally exited75')
    for pid in [launch['pid']]+[t['pid'] for t in q.get('tasks',{}).values() if 'pid' in t]:
        require(type(pid) is int and not Path(f'/proc/{pid}').exists(), 'Legacy process still exists; wait without signalling it')
    no_live_training()
    selection = q['selection']; require(selection.get('microbatch') == 8, 'Only the actual selected micro8 candidate is eligible')
    manifest = Path(selection['manifest']); qualification = Path(selection['qualification'])
    expected = Path(control['legacy_results']).resolve()/'candidate_micro8'
    require(manifest.resolve() == expected/'lee_manifest.json' and qualification.resolve() == expected/'actual_double_backward_probe.json', 'Legacy selected paths escaped expected root')
    require(file_hash(qualification) == selection['qualification_sha256'], 'Actual qualification SHA differs')
    from .control import TASKS
    require(set(q['tasks']) <= set(TASKS)|{'probe_micro'+str(x) for x in (1,2,4,8)} and 'train_seed42' in q['tasks'], 'Unexpected legacy training schedule; preserve and review')
    return q,launch,manifest,qualification


def migrate(control_path):
    c = check_control(signed(control_path)); source(c['legacy_code_root'],LEGACY_SOURCE_SHA)
    q,launch,manifest,qualification = legacy_terminal(c)
    output = Path(c['new_output']).resolve(); require(not output.exists(), 'Never overwrite an existing migration')
    sys.path.insert(0,c['legacy_code_root'])
    from lee_meta_equal40.prepare import validate_manifest
    from lee_meta_equal40.train import binding_for
    from lee_meta_equal40.artifacts import scan_commits
    m,_,_ = validate_manifest(manifest,c['project'],42)
    records = {}; files = {}; original = manifest.parent; target = output/'candidate_micro8'
    for seed in SEEDS:
        _,cfg,name = validate_manifest(manifest,c['project'],seed)
        records[str(seed)] = scan_commits(original/'checkpoints'/name,binding_for(m,cfg,seed))
    require(0 < len(records['42']) <= 40 and any(len(records[str(s)]) < 40 for s in SEEDS), 'No incomplete same-budget training remains')
    output.mkdir(); copy_exact(manifest,target/'lee_manifest.json',files,'candidate_micro8/lee_manifest.json')
    for row in m['configs'].values():
        name = row['file']; require(name.startswith('configs/') and '..' not in Path(name).parts, 'Unsafe config path')
        copy_exact(original/name,target/name,files,'candidate_micro8/'+name)
    copy_exact(qualification,target/'actual_double_backward_probe.json',files,'candidate_micro8/actual_double_backward_probe.json')
    prefixes = {str(s):[] for s in SEEDS}
    for seed in SEEDS:
        name = f'lee_meta_seed{seed}'
        for record in records[str(seed)]:
            relative = f'candidate_micro8/checkpoints/{name}/commits/epoch{record["epoch"]:02d}'
            for leaf in ('state.pt','commit.json'):
                copy_exact(original/'checkpoints'/name/'commits'/f'epoch{record["epoch"]:02d}'/leaf,output/relative/leaf,files,relative+'/'+leaf)
            prefixes[str(seed)].append(dict(epoch=record['epoch'],state_sha256=record['state_sha256'],state_bytes=record['state_bytes'],commit_sha256=file_hash(output/relative/'commit.json'),submitted_utc=record['submitted_utc']))
        # Preserve complete legacy diagnostics in evidence without polluting a fresh seed output.
        oldfolder = original/'checkpoints'/name
        for leaf in ('progress.json','incomplete.json','failure.json','historical_validation_attestation.json','TRAINING_SUBMITTED40.json','TRAINING_METADATA_VERIFIED.json'):
            if (oldfolder/leaf).exists():
                relative = f'legacy_diagnostics/{name}/{leaf}'
                copy_exact(oldfolder/leaf,output/relative,files,relative)
        if len(records[str(seed)]) == 40:
            require(q['tasks'].get('train_seed'+str(seed),{}).get('status') == 'complete' and type(q['tasks']['train_seed'+str(seed)].get('returncode')) is int and q['tasks']['train_seed'+str(seed)]['returncode'] == 0,'Full40 prefix lacks natural train exit0')
            leaf = 'TRAINING_SUBMITTED40.json'; require((oldfolder/leaf).is_file(),'Missing full40 submission marker')
            relative = f'candidate_micro8/checkpoints/{name}/{leaf}'
            copy_exact(oldfolder/leaf,output/relative,files,relative)
    copy_exact(Path(c['legacy_results'])/'queue_state.json',output/'legacy_queue_state.json',files,'legacy_queue_state.json')
    copy_exact(c['legacy_launch'],output/'legacy_launch_receipt.json',files,'legacy_launch_receipt.json')
    for p in sorted((Path(c['legacy_results'])/'logs').glob('*.log')):
        copy_exact(p,output/'legacy_logs'/p.name,files,'legacy_logs/'+p.name)
    require(signed(Path(c['legacy_results'])/'queue_state.json') == q and signed(c['legacy_launch']) == launch, 'Legacy terminal evidence changed during migration')
    receipt = sign(dict(schema='lee_execution_migration_v1',status='immutable_prefix_copied_and_verified',execution_control_signature=c['signature'],extension_source_signature=c['extension_source_signature'],legacy_queue_status=q['status'],legacy_queue_exit_code=q['exit_code'],legacy_queue_sha256=files['legacy_queue_state.json']['sha256'],legacy_launch_sha256=files['legacy_launch_receipt.json']['sha256'],training_manifest_sha256=files['candidate_micro8/lee_manifest.json']['sha256'],prefix_commits=prefixes,copied_files=files,copied_utc=datetime.now(timezone.utc).isoformat(),legacy_bytes_rewritten=False,scientific_binding_changed=False,all_experiments_complete=False))
    write(output/'MIGRATION_RECEIPT.json',receipt)
    return receipt
