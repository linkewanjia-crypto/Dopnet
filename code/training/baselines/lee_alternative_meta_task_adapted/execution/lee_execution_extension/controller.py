"""CPU waiter then unique serial continuation; no tensor runtime in controller."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .control import (LEGACY_DEADLINE, LEGACY_SOURCE_SHA, TASKS, check_control,
    file_hash, no_live_training, require, sign, signed, source, stop_requested,
    verify_source, write)
from .migrate import legacy_terminal, migrate


def run(args):
    require(os.name == 'posix','Authenticated server only')
    require('torch' not in sys.modules and 'numpy' not in sys.modules,'CPU administrative controller only')
    own = verify_source(Path(__file__).resolve().parent.parent); source(args.legacy_code_root,LEGACY_SOURCE_SHA)
    control = args.control.resolve(); output = args.output.resolve()
    require(not control.exists() and not output.exists(),'Exclusive new control/output required; refuse duplicate launch')
    control.mkdir()
    c = check_control(sign(dict(schema='lee_execution_control_v1',mode='until_complete_or_explicit_user_stop',deadline_utc=None,legacy_deadline_utc=LEGACY_DEADLINE,legacy_source_manifest_sha256=LEGACY_SOURCE_SHA,extension_source_signature=own['signature'],authorization_text=args.authorization_text,authorization_utc=args.authorization_utc,authorization_time_basis='agent_recorded_after_actual_user_message',created_utc=datetime.now(timezone.utc).isoformat(),legacy_code_root=str(args.legacy_code_root.resolve()),legacy_results=str(args.legacy_results.resolve()),legacy_launch=str(args.legacy_launch.resolve()),new_output=str(output),project=str(args.project.resolve()),parent_receipt=str(args.parent_receipt.resolve()),forti_code_root=str(args.forti_code_root.resolve()),purchase_or_renew_authorized=False,all_experiments_complete=False)))
    cp = control/'EXECUTION_CONTROL.json'; write(cp,c)
    state = dict(schema='lee_execution_extension_queue_v1',status='waiting_for_legacy_natural_exit75',current=None,tasks={},started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),execution_control_signature=c['signature'],extension_source_signature=own['signature'],execution_mode=c['mode'],legacy_deadline_superseded=True,all_experiments_complete=False)
    write(control/'queue_state.json',sign(state)); write(control/'launch_receipt.json',sign(dict(status='CPU_waiter_started_no_GPU',pid=os.getpid(),execution_control_signature=c['signature'],extension_source_signature=own['signature'],started_utc=state['started_utc'],all_experiments_complete=False)))
    try:
        while True:
            require(signed(cp) == c,'Execution control bytes changed')
            if stop_requested(cp,c):
                raise RuntimeError('USER_STOP')
            oldpath = args.legacy_results/'queue_state.json'
            q = signed(oldpath) if oldpath.exists() else {}
            if q.get('status') == 'incomplete_deadline':
                launch = signed(c['legacy_launch'])
                oldpids = [launch['pid']]+[t['pid'] for t in q.get('tasks',{}).values() if 'pid' in t]
                require(all(type(pid) is int for pid in oldpids),'Invalid legacy process ledger')
                if any(Path(f'/proc/{pid}').exists() for pid in oldpids):
                    time.sleep(30); continue
                legacy_terminal(c)
                gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
                if gpu.stdout.strip():
                    time.sleep(30); continue
                break
            require(q.get('status') in (None,'running'),'Unexpected legacy terminal status; retain for review')
            time.sleep(30)
        migrated = migrate(cp); migration = output/'MIGRATION_RECEIPT.json'
        selected = output/'candidate_micro8'
        state.update(status='running',migration_receipt_sha256=file_hash(migration),training_output=str(selected),legacy_queue_sha256=migrated['legacy_queue_sha256'],selection=dict(microbatch=8,manifest=str(selected/'lee_manifest.json'),qualification=str(selected/'actual_double_backward_probe.json'),qualification_sha256=file_hash(selected/'actual_double_backward_probe.json'),criterion='reuse_actual_legacy_micro8_qualification_same_scientific_source_no_reprobe',fullpath_qualification=False))
        write(output/'queue_state.json',sign(state)); write(control/'queue_state.json',sign(state))
        env = os.environ.copy(); env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        own_root = str(Path(__file__).resolve().parent.parent)
        env['PYTHONPATH'] = os.pathsep.join([own_root,str(args.legacy_code_root.resolve()),env.get('PYTHONPATH','')])
        for key in TASKS:
            if stop_requested(cp,c):
                raise RuntimeError('USER_STOP')
            stage = 'train' if key.startswith('train') else 'verify'; seed = int(key[-2:])
            no_live_training()
            if stage=='train' and len(migrated['prefix_commits'][str(seed)]) == 40:
                oldq = signed(output/'legacy_queue_state.json'); oldtask = oldq['tasks'][key]
                require(oldtask.get('status') == 'complete' and type(oldtask.get('returncode')) is int and oldtask['returncode'] == 0,'Reuse requires actual legacy natural40 exit0')
                state['tasks'][key] = dict(status='complete',returncode=0,pid=oldtask['pid'],command=oldtask['command'],started_utc=oldtask['started_utc'],exited_utc=oldtask['exited_utc'],reused_full40_submission=True,legacy_task=oldtask,legacy_queue_sha256=migrated['legacy_queue_sha256'],execution_control_signature=c['signature'])
                write(output/'queue_state.json',sign(state)); write(control/'queue_state.json',sign(state)); continue
            words = [sys.executable,'-u','-B','-m','lee_execution_extension.worker','--execution-control',str(cp),'--migration-receipt',str(migration),'--legacy-code-root',str(args.legacy_code_root.resolve()),'--stage',stage,'--seed',str(seed),'--controller-pid',str(os.getpid()),'--reference-token','lee_meta_equal40.train' if stage=='train' else 'lee_meta_equal40.verify_training']
            logpath = output/'logs'/(key+'.log'); logpath.parent.mkdir(exist_ok=True)
            with logpath.open('xb') as log:
                p = subprocess.Popen(words,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
                task = dict(status='running',pid=p.pid,command=words,started_utc=datetime.now(timezone.utc).isoformat(),log=str(logpath)); state['tasks'][key] = task; state['current'] = key
                write(output/'queue_state.json',sign(state)); write(control/'queue_state.json',sign(state)); code = p.wait()
            task.update(status='complete' if code==0 else 'incomplete',returncode=code,exited_utc=datetime.now(timezone.utc).isoformat())
            if stage=='verify' and code==0:
                proofpath = output/'candidate_micro8/checkpoints'/f'lee_meta_seed{seed}'/'TRAINING_METADATA_VERIFIED.json'
                proof = signed(proofpath)
                require(proof.get('status')=='single_seed_training_metadata_pass' and type(proof.get('exit_code')) is int and proof['exit_code']==0 and proof.get('execution_control_signature')==c['signature'] and proof.get('committed_epochs')==40,'Actual training proof is incomplete')
                task['proof_sha256'] = file_hash(proofpath)
            write(output/'queue_state.json',sign(state)); write(control/'queue_state.json',sign(state))
            require(code==0,'USER_STOP' if code==75 else 'Child failed; keep all evidence')
        state.update(status='three_seed_training_metadata_complete_scoring_pending',current=None,finished_utc=datetime.now(timezone.utc).isoformat())
        write(output/'queue_state.json',sign(state)); write(control/'queue_state.json',sign(state)); return 0
    except Exception as error:
        code = 75 if str(error)=='USER_STOP' else 1
        state.update(status='incomplete_user_stop' if code==75 else 'failed_or_incomplete',exit_code=code,reason=str(error),traceback=traceback.format_exc(),observed_utc=datetime.now(timezone.utc).isoformat())
        write(control/'queue_state.json',sign(state)); write(control/'INCOMPLETE.json',sign(state))
        if output.exists():
            write(output/'queue_state.json',sign(state)); write(output/'INCOMPLETE.json',sign(state))
        return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('legacy-code-root','legacy-results','legacy-launch','output','control','project','parent-receipt','forti-code-root'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--authorization-text',required=True); p.add_argument('--authorization-utc',required=True)
    sys.exit(run(p.parse_args()))


if __name__ == '__main__':
    main()
