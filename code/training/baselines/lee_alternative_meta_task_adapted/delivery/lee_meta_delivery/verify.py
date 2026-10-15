"""Final Lee directory/ZIP checker: SHA/CRC, restricted metadata and score arrays.

No Torch, model, waveform generation, training, inference or latency is executed.
The accepted parent21 columns are compared bit-for-bit; new predictions cannot
inherit an old supervised Lee identity. All three40-state chains are checked.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from zipfile import ZipFile
from .protocol import (ASSET_NAMES, CLAIM, DEADLINE, NEW_METHOD, READER_SHA, SEEDS, TRAINING_SOURCE_SIGNATURE,
    WIMAE_ARCHIVE_SHA, bind_training_source, check_statistics, digest, file_hash,
    read, read_parent_archive, require, require_routes, require_training_receipt,
    safe_name, signed, verify_source)
from .statistics_checks import check_seed_intervals, check_paired_intervals, check_assets
from . import execution as X
from . import recovery as Recovery


def check_score_arrays(arrays, inherited, cell, ev, parent_sha, old_sha):
    import numpy as np
    require(arrays['signature'].item() == ev['signature'] and json.loads(arrays['meta'].item()) == cell
            and arrays['parent_score_sha256'].item() == parent_sha and arrays['source_score_sha256'].item() == old_sha
            and arrays['realization_sha256'].item() == inherited['realization_sha256'].item(), 'Cell score/reference/waveform identity differs')
    require(np.array_equal(arrays['cluster_id'],np.arange(512)//8) and np.array_equal(inherited['cluster_id'],arrays['cluster_id']), 'Entire generation8 cluster changed')
    nmse=arrays['nmse']
    require(nmse.shape==(24,512) and np.isfinite(nmse).all() and np.all(nmse>=0) and np.array_equal(nmse[:21],inherited['nmse']), 'Finite new NMSE and exact inherited21 required')
    require(arrays['native_available'].shape==(24,) and np.array_equal(arrays['native_available'][:21],inherited['native_available']) and np.all(arrays['native_available'][21:]==0), 'Native detector identity differs')
    fields=('errors_diagonal','errors_native','bits','native_residual_max','native_direct_fallback','errors_noise_minus3db','errors_noise_plus3db')
    if cell['ber']:
        bits=arrays['bits'];require(bits.shape==(24,512) and np.issubdtype(bits.dtype,np.integer) and np.all(bits>0) and np.all(bits==inherited['bits'][0]), 'Actual BER denominators differ')
        expected_receiver=Recovery.receiver_keys(cell,inherited.files)
        require(set(k for k in fields if k in arrays)==set(expected_receiver),'Copied receiver/noise scope differs')
        for key in expected_receiver:
            values=arrays[key]
            require(values.shape==(24,512) and np.isfinite(values).all() and np.array_equal(values[:21],inherited[key]), 'Inherited receiver field differs: '+key)
            if key.startswith('errors_'):
                require(np.issubdtype(values.dtype,np.integer) and np.all(values>=0) and np.all(values<=bits), 'Invalid BER numerator')
        require(np.array_equal(arrays['errors_native'][21:],arrays['errors_diagonal'][21:]) and np.all(arrays['native_residual_max'][21:]==0) and np.all(arrays['native_direct_fallback'][21:]==0), 'Lee diagonal/native-NA diagnostics differ')
    else:
        require(not any(k in arrays for k in fields), 'NMSE-only cell contains fictitious bits/receiver fields')


def verify_directory(output, training, project, reference, parent_archive, training_code, forti_code, source_root, require_assets=True,
                     execution_control=None, migration_receipt=None, extension_code=None, evidence_root=None):
    import numpy as np
    require('torch' not in sys.modules,'Torch must not enter the delivery checker')
    own=verify_source(source_root);training_source=bind_training_source(training_code)
    if execution_control is not None or migration_receipt is not None or extension_code is not None:
        require(all(p is not None for p in (execution_control,migration_receipt,extension_code)), 'Supply the complete execution provenance context')
        X.activate(execution_control,migration_receipt,training_code,extension_code,evidence_root)
    execution_binding=X.identity()
    pkg=Path(forti_code).resolve();require(file_hash(pkg/'verify_author_delivery.py')==READER_SHA,'Unreviewed restricted reader')
    sys.path.insert(0,str(pkg.parent));sys.path.insert(0,str(Path(training_code).resolve()))
    from budget_matched_baselines.verify_author_delivery import checkpoint_metadata
    from lee_meta_equal40.verify_training import verify as verify_training
    from lee_meta_equal40.prepare import build_config, validate_manifest
    output,training,project,reference=map(Path,(output,training,project,reference))
    ev=signed(output/'evaluation.json');complete=signed(output/'COMPLETE.json')
    parent_ev,parent_delivery=read_parent_archive(parent_archive)
    require(ev.get('implementation_signature')==own['signature'] and ev.get('training_source_signature')==training_source['signature']==TRAINING_SOURCE_SIGNATURE
            and ev.get('claim')==CLAIM and ev.get('original_author_dataset_recipe_reproduction') is False
            and ev.get('test_adaptation_steps')==0,'Unknown downstream/train source or overclaimed reproduction')
    require(ev.get('parent_archive_sha256')==WIMAE_ARCHIVE_SHA and ev.get('parent_evaluation_signature')==parent_ev['signature'] and ev.get('inherited_routes')==21,'Parent delivery provenance differs')
    require_routes(ev['routes'],parent_ev['routes'])
    require(ev['cells']==parent_ev['cells'] and len(ev['cells'])==269 and ev['samples_per_cell']==512 and ev['batch_size']==8,'Frozen269/512/8 identity differs')
    require(ev.get('task_adaptation')==dict(meta_epochs=1,online_epochs=39,fresh_frames_per_epoch=12000,updates_per_epoch=375,alpha=.001,beta=.001,K=1,support=16,query=16), 'Equal40 task adaptation claim differs')
    X.require_fields(ev,execution_binding);X.require_fields(complete,execution_binding)
    Recovery.verify_reuse(output,ev)
    require(complete.get('status')=='complete' and complete.get('paper_ready') is False and complete.get('pending')==['paper_assets'] and complete.get('cells')==269 and complete.get('evaluation_signature')==ev['signature'] and complete.get('deadline_utc') is None and complete.get('legacy_deadline_utc')==DEADLINE,'Score completion and final paper readiness must stay separate')
    finished=__import__('datetime').datetime.fromisoformat(complete['finished_utc'])
    authorized=__import__('datetime').datetime.fromisoformat(X.CONTEXT.control['created_utc'].replace('Z','+00:00'))
    require(finished.tzinfo is not None and finished>=authorized,'Scoring completion precedes its explicit execution authorization')
    require(read(output/'evaluation_complete.json')==dict(signature=ev['signature'],cells=269),'Incomplete269 scoring marker')
    experiment=signed(reference/'experiment.json');old_ev=signed(reference/'evaluation.json')
    require(len(old_ev['cells'])==636 and old_ev['experiment_signature']==experiment['signature'] and ev['reference_experiment_signature']==experiment['signature']==parent_ev['reference_experiment_signature'] and ev['reference_evaluation_signature']==old_ev['signature']==parent_ev['reference_evaluation_signature'],'Original636 source binding differs')
    manifest_path=training/'lee_manifest.json';manifest=signed(manifest_path)
    require(ev['training_manifest_sha256']==file_hash(manifest_path) and ev['training_manifest_signature']==manifest['signature'],'Lee training manifest differs')
    queue=signed(output/'training_queue_state.json')
    require(file_hash(output/'training_queue_state.json')==ev['training_queue_sha256'] and queue.get('status')=='three_seed_training_metadata_complete_scoring_pending' and queue.get('current') is None,'Actual naturally completed training queue differs')
    X.check_queue_ready(queue,execution_binding)
    X.check_receipt_members(X.CONTEXT)
    X.check_migration_copies(X.CONTEXT,training)
    for seed in SEEDS:
        m,cfg,name=validate_manifest(manifest_path,project,seed)
        require(cfg==build_config(experiment['configs']['lee3ddual_seed'+str(seed)],seed,cfg['lee_meta_reproduction']['microbatch']),'Actual config is not the reviewed adaptation')
        folder=training/'checkpoints'/name
        proof=signed(folder/'TRAINING_METADATA_VERIFIED.json')
        for key in ('train_seed'+str(seed),'verify_seed'+str(seed)):
            task=queue['tasks'][key]
            require(task.get('status')=='complete' and type(task.get('returncode')) is int and task['returncode']==0,'A required training/metadata task is incomplete')
        require(file_hash(folder/'TRAINING_METADATA_VERIFIED.json')==queue['tasks']['verify_seed'+str(seed)]['proof_sha256'],'Actual queue training metadata receipt SHA differs')
        require_training_receipt(proof,seed,m,cfg,execution_binding)
        actual=verify_training(folder,manifest_path,project,seed,checkpoint_metadata)
        require(actual==proof==ev['training_receipts'][str(seed)] and actual['selected_state_sha256']==ev['checkpoint_sha256'][str(seed)],'Actual40 metadata/earliest selected weight/parent/RNG provenance differs')
    statistics=signed(output/'summary/statistics.json');check_statistics(statistics,ev,output)
    require(set(statistics['cell_file_sha256'])=={c['cell_id'] for c in ev['cells']},'Statistics lacks a frozen cell')
    diagnostic_expected={};cell_points={}
    with ZipFile(parent_archive) as parent:
        for cell in ev['cells']:
            cid=cell['cell_id'];path=output/'cells'/f'{cid}.npz';marker=signed(path.with_suffix('.COMPLETE.json'))
            parent_name=f'evaluation/cells/{cid}.npz';parent_data=parent.read(parent_name);parent_sha=hashlib.sha256(parent_data).hexdigest()
            require(parent_delivery['files'][parent_name]==parent_sha,'Parent score bytes changed')
            old_sha=file_hash(reference/'cells'/f'{cid}.npz')
            require(file_hash(path)==statistics['cell_file_sha256'][cid]==marker['score_sha256'] and marker['evaluation_signature']==ev['signature'] and marker['parent_score_sha256']==parent_sha and marker['source_score_sha256']==old_sha,'Actual per-cell SHA/marker differs')
            with np.load(path,allow_pickle=False) as values,np.load(io.BytesIO(parent_data),allow_pickle=False) as inherited:
                check_score_arrays(values,inherited,cell,ev,parent_sha,old_sha)
                require(marker['realization_sha256']==values['realization_sha256'].item(),'Marker full waveform differs')
                metrics={'nmse':values['nmse']}
                if cell['ber']:
                    for suffix in ('diagonal','native','noise_minus3db','noise_plus3db'):
                        if 'errors_'+suffix not in values:continue
                        metrics['ber_'+suffix]=values['errors_'+suffix]/values['bits']
                    for index,route in enumerate(ev['routes']):
                        diagnostic_expected[(cid,route['method'],str(route['training_seed']))]=dict(native_ici_available=int(values['native_available'][index]),residual_max=float(values['native_residual_max'][index].max()),direct_fallback_frames=int(values['native_direct_fallback'][index].sum()),bit_errors_native=int(values['errors_native'][index].sum()),bit_total=int(values['bits'][index].sum()))
                for metric,data in metrics.items():
                    for index,route in enumerate(ev['routes']):
                        cell_points[(cid,metric,route['method'],str(route['training_seed']))]=float(data[index].mean())
    for name in ('cell_summary.csv','macro_summary.csv'):
        check_seed_intervals(output/'summary'/name,statistics['t_critical_df2'])
    check_paired_intervals(output/'summary/cell_summary.csv',output/'summary/cell_paired_differences.csv',t_critical=statistics['t_critical_df2'])
    check_paired_intervals(output/'summary/macro_summary.csv',output/'summary/macro_paired_differences.csv',True,statistics['t_critical_df2'])
    with (output/'summary/cell_summary.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    seen_points=set()
    for row in rows:
        for seed in SEEDS:
            key=(row['cell_id'],row['metric'],row['method'],str(seed))
            require(key in cell_points and key not in seen_points and math.isclose(float(row['seed'+str(seed)]),cell_points[key],rel_tol=1e-10,abs_tol=1e-14),'Cell point estimate differs from actual score array')
            seen_points.add(key)
    require(seen_points==set(cell_points),'Cell summaries omit a score/method/seed/metric')
    with (output/'summary/receiver_diagnostics.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    seen=set()
    for row in rows:
        key=(row['cell_id'],row['method'],row['training_seed'])
        require(key in diagnostic_expected and key not in seen,'Unknown/duplicate receiver diagnostics');seen.add(key)
        for field,value in diagnostic_expected[key].items():
            require(math.isclose(float(row[field]),value,rel_tol=1e-12,abs_tol=1e-14),'Receiver CSV differs from actual counts')
    require(seen==set(diagnostic_expected),'Receiver diagnostics missing rows')
    if require_assets:
        check_assets(output,ev,statistics)
    require('torch' not in sys.modules,'Torch imported by metadata checker')
    return dict(status='passed',paper_ready=bool(require_assets),pending=[] if require_assets else ['paper_assets'],cells=269,methods=8,routes=24,
                training_epochs_per_seed=40,selected_epochs={s:ev['training_receipts'][s]['selected_epoch'] for s in ('42','43','44')},
                evaluation_signature=ev['signature'],training_manifest_signature=manifest['signature'],implementation_signature=own['signature'],
                parent_archive_sha256=WIMAE_ARCHIVE_SHA,model_execution=False,real_data_generated=False,all_experiments_complete=False,
                **execution_binding)


def verify_archive(archive,reference,training_code,forti_code,source_root,extension_code=None):
    own=verify_source(source_root);bind_training_source(training_code)
    with ZipFile(archive) as z,tempfile.TemporaryDirectory(prefix='lee_equal40_delivery_') as temporary:
        names=[x.filename for x in z.infolist() if not x.is_dir()]
        require(len(names)==len(set(names)) and all(safe_name(n) for n in names),'Unsafe/duplicate archive members')
        require(z.testzip() is None,'ZIP CRC failed')
        m=json.loads(z.read('DELIVERY_MANIFEST.json'))
        require(m.get('signature')==digest({k:v for k,v in m.items() if k!='signature'}) and m.get('status')=='complete' and m.get('paper_ready') is True and m.get('pending')==[] and m.get('claim')==CLAIM and m.get('all_experiments_complete') is False,'Final Lee package identity/qualification differs')
        require(set(names)==set(m['files'])|{'DELIVERY_MANIFEST.json'},'Archive membership differs')
        root=Path(temporary)
        for name,h in m['files'].items():
            data=z.read(name);require(hashlib.sha256(data).hexdigest()==h,'Archive internal SHA differs: '+name)
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        bundled=verify_source(root/'source/downstream')
        require(bundled==own,'Archived downstream source differs from reviewed bytes')
        for name,record in own['members'].items():
            require(file_hash(root/'source/downstream'/name)==file_hash(Path(source_root)/name)==record['sha256'],'Local/archived downstream mismatch')
        train=bind_training_source(root/'source/training')
        require(train==bind_training_source(training_code),'Archived training source differs from reviewed source')
        if extension_code is None:
            extension_code=X.EXTENSION_ROOT
        if extension_code is None:
            import importlib
            extension_module=importlib.import_module('lee_execution_extension.control')
            extension_code=Path(extension_module.__file__).resolve().parent.parent
        require(extension_code is not None, 'Supply the reviewed execution extension source root')
        bundled_extension=root/'source/execution_extension'
        require(file_hash(bundled_extension/'SOURCE_MANIFEST.json')==X.EXTENSION_MANIFEST_SHA
                and file_hash(Path(extension_code)/'SOURCE_MANIFEST.json')==X.EXTENSION_MANIFEST_SHA,
                'Actual/local extension manifest differs')
        extension=signed(bundled_extension/'SOURCE_MANIFEST.json')
        require(extension['signature']==X.EXTENSION_SOURCE_SIGNATURE, 'Archived extension signature differs')
        for name,record in extension['members'].items():
            require(safe_name(name) and file_hash(bundled_extension/name)==file_hash(Path(extension_code)/name)==record['sha256']
                    and (bundled_extension/name).stat().st_size==record['bytes'], 'Archived/local extension source mismatch')
        evidence=root/'evidence/execution'
        previous_context,previous_extension=X.CONTEXT,X.EXTENSION_ROOT
        try:
            proof=verify_directory(root/'evaluation',root/'training',root/'source/archive_runtime',reference,root/'evidence/wimae_results_for_paper.zip',training_code,forti_code,source_root,
                                   execution_control=evidence/'EXECUTION_CONTROL.json',migration_receipt=evidence/'MIGRATION_RECEIPT.json',
                                   extension_code=extension_code,evidence_root=evidence)
            X.require_fields(m,X.identity())
            require(proof['evaluation_signature']==m['evaluation_signature'] and proof['training_manifest_signature']==m['training_manifest_signature'],'Archive result identity differs')
        finally:
            X.restore(previous_context,previous_extension)
    proof.update(zip_crc='passed',archive_sha256=file_hash(archive));return proof


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group(required=True);group.add_argument('--archive',type=Path);group.add_argument('--directory',type=Path)
    for name in ('reference-results','training-code-root','forti-code-root','implementation-root'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--extension-code-root',type=Path,required=True)
    ap.add_argument('--execution-control',type=Path)
    ap.add_argument('--migration-receipt',type=Path)
    for name in ('training-output','project','parent-archive'):
        ap.add_argument('--'+name,type=Path)
    args=ap.parse_args()
    if args.archive:
        proof=verify_archive(args.archive,args.reference_results,args.training_code_root,args.forti_code_root,args.implementation_root,args.extension_code_root)
    else:
        require(args.training_output and args.project and args.parent_archive and args.execution_control and args.migration_receipt,'Directory verification requires training/project/parent archive/execution control/migration')
        proof=verify_directory(args.directory,args.training_output,args.project,args.reference_results,args.parent_archive,args.training_code_root,args.forti_code_root,args.implementation_root,
                               execution_control=args.execution_control,migration_receipt=args.migration_receipt,extension_code=args.extension_code_root)
    print(json.dumps(proof,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
