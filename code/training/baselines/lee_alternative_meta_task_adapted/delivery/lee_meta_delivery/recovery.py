# HISTORICAL_SANITIZED_NON_RUNNABLE: original bytes retained in source archive.
"""Stopped-score reuse and the original fixed receiver-noise diagnostic scope.

All236 existing NMSE arrays are preserved, with original bytes in recovery
evidence. Only the copied score signature changes to the new evaluation. No
training, model, waveform or inference is executed by this module.
"""
import hashlib
import io
import json
import os
from pathlib import Path
from .protocol import digest,file_hash,require,safe_name,sign,signed,write

OLD_SOURCE_SHA='121383754bc727286652a50e1d8cb5f1edf8fe1410c62661fbbfc2bed5270675'
OLD_SOURCE_SIG='3b02e2725ab4b26060ffa445cbf4b4a19c0e358921d67c411efab0a37b33f794'
REC_ADMIN_SHA='52bcf55f522312026f5de94a06ab7e50b23114c5f2eb0271a646907a25f0a4a4'
PARENT=Path('__HISTORICAL_MACHINE_PATH_REDACTED__')
OLD_CODE=Path('__HISTORICAL_MACHINE_PATH_REDACTED__')
CONTROL=Path('__HISTORICAL_MACHINE_PATH_REDACTED__')
LAUNCH=Path('__HISTORICAL_MACHINE_PATH_REDACTED__')
CORE=('errors_diagonal','errors_native','bits','native_residual_max','native_direct_fallback')
NOISE=('errors_noise_minus3db','errors_noise_plus3db')

def recovery_launch(receipt,request,request_sha):
    require(receipt.get('schema')=='lee_scoring_environment_recovery_launch_receipt_v1' and type(receipt.get('pid')) is int and receipt['pid']==169795 and receipt.get('administrative_source_sha256')==REC_ADMIN_SHA and receipt.get('launched_utc')=='2026-10-02T23:27:30.778973+00:00','Wrong recovery launch parent')
    require(request.get('schema')=='lee_scoring_environment_recovery_request_v1' and request.get('administrative_source_sha256')==REC_ADMIN_SHA and receipt.get('launch_request_sha256')==request_sha and receipt.get('preconditions_signature')==request['preconditions']['signature'],'REC request/receipt lineage differs')
    require(request['preconditions']['failure_parent']==receipt['failure_parent'],'REC failed parent declaration differs')
    require(receipt.get('expected_output')==str(PARENT) and receipt.get('expected_queue_control')==str(CONTROL),'Wrong REC output/control identity')

def receiver_keys(cell,available):
    """Exact original receiver scope, including all33 BER denominators."""
    if not cell['ber']:
        require(not set(CORE+NOISE).intersection(available),'NMSE-only cell has receiver data')
        return ()
    require(set(CORE)<=set(available),'Mandatory BER/native fields absent')
    sweep=cell['modulation']=='16qam' and cell['snr_db']==20 and cell['doppler_hz']==1200
    require(set(NOISE).intersection(available)==(set(NOISE) if sweep else set()),'Parent receiver noise fields differ from frozen original scope')
    return CORE+(NOISE if sweep else ())

def old_source(root):
    root=Path(root);require(root.is_dir() and not root.is_symlink(),'Regular stopped source root required');m=signed(root/'SOURCE_MANIFEST.json')
    require(file_hash(root/'SOURCE_MANIFEST.json')==OLD_SOURCE_SHA and m.get('version')=='0.4.1' and m['signature']==OLD_SOURCE_SIG and len(m['members'])==12,'Stopped scientific source differs')
    for name,row in m['members'].items():
        p=root/name
        require(safe_name(name) and p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and file_hash(p)==row['sha256'],'Stopped source member differs')
    return m

def require_absent(pid):
    require(type(pid) is int and pid>0 and not (Path('/proc')/str(pid)).exists(),'Recorded failed process must remain naturally absent')

def failed_queue(q):
    require(q.get('schema')=='lee_extension_scoring_queue_v1' and q.get('source_signature')==OLD_SOURCE_SIG and q.get('status')=='failed_or_incomplete' and q.get('current')=='evaluate' and type(q.get('exit_code')) is int and q['exit_code']==1,'Only the diagnosed stopped scoring parent')
    require(set(q.get('tasks',{}))=={'evaluate'} and q.get('reason')=='evaluate failed; retain all evidence','Wrong stopped task/reason')
    task=q['tasks']['evaluate']
    require(task.get('status')=='incomplete' and type(task.get('returncode')) is int and task['returncode']==1 and type(task.get('pid')) is int and task['pid']==169804,'Wrong failed evaluate identity')
    require(q.get('all_experiments_complete') is False,'Failure overclaims completion')

def failed_log(path):
    require(path.is_file() and not path.is_symlink(),'Regular stopped log required')
    with path.open('rb') as f:
        f.seek(0,2);size=f.tell();f.seek(max(0,size-4096));tail=f.read(4096).decode('utf-8')
    require(tail.rstrip().endswith('ValueError: Parent lacks required actual BER/noise diagnostics') and 'lee_meta_delivery/evaluate.py' in tail and 'Traceback (most recent call last):' in tail,'Another child failure must be independently diagnosed')

def reuse_binding(old_ev,records,evidence):
    cells={c['cell_id']:c for c in old_ev['cells']}
    expected={c['cell_id'] for c in old_ev['cells'] if c['ber'] is False}
    require(len(old_ev['cells'])==269 and len(old_ev['routes'])==24 and len(expected)==236 and set(records)==expected,'Exactly236 complete NMSE-only scores required')
    require(old_ev.get('implementation_signature')==OLD_SOURCE_SIG,'Stopped evaluation source differs')
    for cid,row in records.items():
        require(cid in cells and cells[cid]['ber'] is False and set(row)=={'score_sha256','marker_sha256'},'Unknown reuse cell/fields')
        require(all(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in row.values()),'Invalid reuse digest')
    require(type(evidence) is dict and {'evaluation.json','training_queue_state.json','source/SOURCE_MANIFEST.json','control/queue_state.json','control/evaluate.log','launch/launch_receipt.json','launch/launch_request.json'}<=set(evidence),'Incomplete stopped source/evidence set')
    require(evidence['source/SOURCE_MANIFEST.json']==OLD_SOURCE_SHA,'Wrong stopped source evidence manifest')
    for name,h in evidence.items():require(safe_name(name) and type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h),'Unsafe reuse evidence digest')
    for cid,row in records.items():
        require(evidence.get('cells/'+cid+'.npz')==row['score_sha256'] and evidence.get('cells/'+cid+'.COMPLETE.json')==row['marker_sha256'],'Evidence does not bind complete cell bytes')
    return dict(schema='lee_stopped_score_reuse_binding_v1',parent_evaluation_signature=old_ev['signature'],parent_source_manifest_sha256=OLD_SOURCE_SHA,reused_NMSEonly_cells=236,reused_BER_cells=0,
        original_noise_scope=dict(modulation='16qam',snr_db=20,doppler_hz=1200,BER_cells=33,noise_cells=11),records=records,evidence_files_sha256=evidence)

def match_evaluations(old_ev,new_ev):
    old={k:v for k,v in old_ev.items() if k not in ('signature','implementation_signature')}
    new={k:v for k,v in new_ev.items() if k not in ('signature','implementation_signature','scoring_recovery')}
    require(old==new,'Recovery changed model/test/receiver/core scientific configuration')
    require(old_ev['signature']==new_ev['scoring_recovery']['parent_evaluation_signature'],'Wrong stopped evaluation parent')

def inspect_stopped(parent):
    """Source/administrative JSON and completed score byte hashes, no framework."""
    require(os.name=='posix' and Path(parent).resolve()==PARENT and not PARENT.is_symlink(),'Use exact diagnosed stopped output on authorized server')
    own=old_source(OLD_CODE);q=signed(CONTROL/'queue_state.json');failed_queue(q)
    require_absent(169795);require_absent(169804);failed_log(CONTROL/'evaluate.log')
    receipt=signed(LAUNCH/'launch_receipt.json')
    request=signed(LAUNCH/'launch_request.json');recovery_launch(receipt,request,file_hash(LAUNCH/'launch_request.json'))
    require(q.get('execution_control_signature')==receipt.get('execution_control_signature') and q.get('migration_receipt_sha256')==receipt.get('migration_receipt_sha256'),'Failed execution lineage differs')
    files={}
    def add(name,path):
        require(safe_name(name) and path.is_file() and not path.is_symlink(),'Unsafe stopped evidence')
        files[name]=dict(path=str(path),sha256=file_hash(path),bytes=path.stat().st_size)
    for name in own['members']:add('source/'+name,OLD_CODE/name)
    add('source/SOURCE_MANIFEST.json',OLD_CODE/'SOURCE_MANIFEST.json')
    for root,label in ((CONTROL,'control'),(LAUNCH,'launch')):
        for path in root.rglob('*'):
            require(not path.is_symlink(),'No symlinked failure evidence')
            if path.is_file():add(label+'/'+path.relative_to(root).as_posix(),path)
    expected_original=[Path('__HISTORICAL_MACHINE_PATH_REDACTED__')/n for n in ('launch_request.json','launch_receipt.json','queue_controller.log','queue_controller.pid')]
    expected_original += [Path('__HISTORICAL_MACHINE_PATH_REDACTED__')/n for n in ('queue_state.json','evaluate.log')]
    require(set(receipt['failure_parent'])==set(map(str,expected_original)),'Original missing-module failure file set differs')
    for i,(name,row) in enumerate(receipt['failure_parent'].items()):
        original=Path(name);snap=LAUNCH/'failed_parent_snapshot'/('%02d_'%i+original.name)
        require(file_hash(original)==file_hash(snap)==row['sha256'] and original.stat().st_size==snap.stat().st_size==row['bytes'],'Original failed source/snapshot changed')
        add('original_failure/%02d_'%i+original.name,original)
    for name in ('COMPLETE.json','evaluation_complete.json','PAPER_READY.json','COMPLETE_DELIVERY.json','FINAL_VERIFICATION.json'):
        require(not (PARENT/name).exists(),'Partial parent already has a completion record')
    old_ev=signed(PARENT/'evaluation.json');add('evaluation.json',PARENT/'evaluation.json')
    add('training_queue_state.json',PARENT/'training_queue_state.json')
    records={}
    for marker in (PARENT/'cells').glob('*.COMPLETE.json'):
        cid=marker.name[:-len('.COMPLETE.json')];score=marker.with_name(cid+'.npz')
        require(marker.is_file() and not marker.is_symlink() and score.is_file() and not score.is_symlink(),'Regular complete score files required')
        m=signed(marker)
        require(m['evaluation_signature']==old_ev['signature'] and file_hash(score)==m['score_sha256'],'Stopped score marker/source changed')
        records[cid]=dict(score_sha256=m['score_sha256'],marker_sha256=file_hash(marker))
        add('cells/'+score.name,score);add('cells/'+marker.name,marker)
    binding=reuse_binding(old_ev,records,{n:r['sha256'] for n,r in files.items()})
    require({p.name for p in (PARENT/'cells').iterdir()}=={cid+ext for cid in records for ext in ('.npz','.COMPLETE.json')},'Stopped parent has partial/extra score files')
    require(signed(CONTROL/'queue_state.json')==q and old_source(OLD_CODE)==own,'Stopped source/queue changed')
    require_absent(169795);require_absent(169804)
    return binding,files,old_ev

def reuse_completed(output,binding,files,old_ev,new_ev):
    """Server CPU array-preserving copies; original score files remain unchanged."""
    import numpy as np
    from verification_extension import protocol as P
    out=Path(output);match_evaluations(old_ev,new_ev)
    require(new_ev['scoring_recovery']==binding and not (out/'recovery').exists(),'Refuse duplicate reuse or another evaluation')
    backup=out/'recovery';backup.mkdir()
    hashes={}
    for name,row in files.items():
        p=Path(row['path']);data=p.read_bytes()
        require(len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],'Stopped evidence changed before copy')
        dst=backup/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data);hashes[name]=row['sha256']
    by_id={c['cell_id']:c for c in old_ev['cells']}
    for cid,row in binding['records'].items():
        src=backup/'cells'/f'{cid}.npz'
        with np.load(src,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        require(set(arrays)=={'nmse','native_available','cluster_id','signature','meta','realization_sha256','source_score_sha256','parent_score_sha256'},'Saved NMSE fields differ')
        require(arrays['signature'].item()==old_ev['signature'] and json.loads(arrays['meta'].item())==by_id[cid],'Old completed score identity differs')
        require(arrays['nmse'].shape==(24,512) and arrays['nmse'].dtype==np.dtype('float64') and np.isfinite(arrays['nmse']).all() and np.all(arrays['nmse']>=0),'Invalid saved Lee NMSE')
        receiver_keys(by_id[cid],arrays)
        require(np.array_equal(arrays['cluster_id'],np.arange(512)//8),'Saved cluster changed')
        arrays['signature']=np.array(new_ev['signature'])
        dst=out/'cells'/f'{cid}.npz';P.atomic_npz(dst,arrays)
        marker=signed(backup/'cells'/f'{cid}.COMPLETE.json')
        payload={k:v for k,v in marker.items() if k!='signature'}
        payload.update(evaluation_signature=new_ev['signature'],score_sha256=file_hash(dst),reused_from_score_sha256=row['score_sha256'])
        write(dst.with_suffix('.COMPLETE.json'),sign(payload))
    write(backup/'REUSE_RECEIPT.json',sign(dict(schema='lee_array_preserving_reuse_receipt_v1',evaluation_signature=new_ev['signature'],binding=binding,evidence_files_sha256=hashes,source_files_rewritten=False,numeric_score_arrays_changed=False,metadata_only_score_signature_changed=True,training_or_inference_or_waveform_repeated=False,all_experiments_complete=False)))
    verify_reuse(out,new_ev)

def verify_reuse(output,ev):
    """Final relocated archive/directory gate: all original files and arrays."""
    import numpy as np
    out=Path(output);backup=out/'recovery';receipt=signed(backup/'REUSE_RECEIPT.json')
    require(receipt.get('schema')=='lee_array_preserving_reuse_receipt_v1' and receipt.get('evaluation_signature')==ev['signature'] and receipt.get('binding')==ev['scoring_recovery'],'Reuse receipt belongs to another evaluation')
    require(receipt.get('source_files_rewritten') is False and receipt.get('numeric_score_arrays_changed') is False and receipt.get('metadata_only_score_signature_changed') is True and receipt.get('training_or_inference_or_waveform_repeated') is False and receipt.get('all_experiments_complete') is False,'Reuse claims differ')
    expected=receipt['evidence_files_sha256']
    require(expected==ev['scoring_recovery']['evidence_files_sha256'],'Recovery evidence is not bound to new evaluation')
    require({p.relative_to(backup).as_posix() for p in backup.rglob('*') if p.is_file()}==set(expected)|{'REUSE_RECEIPT.json'},'Reuse evidence membership differs')
    for name,h in expected.items():require(safe_name(name) and not (backup/name).is_symlink() and file_hash(backup/name)==h,'Raw stopped source/score/evidence SHA differs')
    old_source(backup/'source');old_ev=signed(backup/'evaluation.json');match_evaluations(old_ev,ev)
    records=ev['scoring_recovery']['records'];require(reuse_binding(old_ev,records,expected)==ev['scoring_recovery'],'Reuse scientific/cell set differs')
    q=signed(backup/'control/queue_state.json');failed_queue(q);failed_log(backup/'control/evaluate.log')
    launch=signed(backup/'launch/launch_receipt.json');request=signed(backup/'launch/launch_request.json');recovery_launch(launch,request,file_hash(backup/'launch/launch_request.json'))
    for i,(name,row) in enumerate(launch['failure_parent'].items()):
        label='%02d_'%i+Path(name).name
        for path in (backup/'original_failure'/label,backup/'launch/failed_parent_snapshot'/label):
            require(path.stat().st_size==row['bytes'] and file_hash(path)==row['sha256'],'Final failed parent/snapshot chain differs')
    for cid,row in records.items():
        raw=backup/'cells'/f'{cid}.npz';old_marker=signed(raw.with_suffix('.COMPLETE.json'))
        require(file_hash(raw)==row['score_sha256'] and file_hash(raw.with_suffix('.COMPLETE.json'))==row['marker_sha256'] and old_marker['evaluation_signature']==old_ev['signature'] and old_marker['score_sha256']==row['score_sha256'],'Saved original cell provenance differs')
        path=out/'cells'/f'{cid}.npz';new_marker=signed(path.with_suffix('.COMPLETE.json'))
        require(new_marker['reused_from_score_sha256']==row['score_sha256'],'New marker does not bind original prediction bytes')
        with np.load(raw,allow_pickle=False) as old,np.load(path,allow_pickle=False) as new:
            require(set(new.files)==set(old.files),'Reused score fields changed')
            for key in old.files:
                if key=='signature':
                    require(new[key].item()==ev['signature'] and old[key].item()==old_ev['signature'],'Copied metadata signature differs')
                else:require(new[key].dtype==old[key].dtype and np.array_equal(new[key],old[key]),'Reused array/value/dtype changed: '+key)
    return records
