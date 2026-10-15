"""Serial training-only controller; completion never qualifies scoring/delivery.

No tensor framework is imported in this controller. All CUDA work lives in
separate, sequential probe/train subprocesses with the original deadline.
"""
import argparse
import ast
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .artifacts import file_hash, read_signed, sign, write_json
from .prepare import prepare
from .train import check_deadline, DeadlineReached, utc_now


def select_candidate(records):
    safe = []
    for micro, manifest, qualification in records:
        if type(micro) is not int or micro not in (1, 2, 4, 8):
            raise ValueError('Wrong resource candidate')
        proof = read_signed(qualification)
        if proof.get('status') != 'passed':
            continue
        if (type(proof.get('microbatch')) is not int or proof.get('microbatch') != micro or
                proof.get('synthetic_hardware_only') is not True or
                proof.get('scientific_data_generated') is not False or
                proof.get('training_started') is not False or
                proof.get('offline_sft_unchanged') is not True or
                type(proof.get('torch_peak_reserved_bytes')) is not int or
                not 0 < proof['torch_peak_reserved_bytes'] <= 6*2**30):
            raise ValueError('Invalid passed hardware proof')
        phases = proof.get('expected_exclusive_seconds_per12000_frames', {})
        if set(phases) != {'add_meta', 'joint_online'}:
            raise ValueError('Missing actual phase throughput')
        values = list(phases.values())
        if any(type(x) not in (float, int) or not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError('Invalid resource throughput')
        # Equal40 workload contains one meta and39 online epochs.
        cost = phases['add_meta'] + 39*phases['joint_online']
        safe.append((cost, micro, manifest, qualification))
    if not safe:
        raise ValueError('No safe actual second-order candidate; preserve failures')
    return min(safe, key=lambda x: (x[0], x[1]))


@contextmanager
def controller_lock(output):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output/'queue.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0); stream.truncate(); stream.write(str(os.getpid())+'\n'); stream.flush()
        yield


def run_child(module, words, output, state, key, allow_failed_probe=False):
    check_deadline()
    path = output/'logs'/f'{key}.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', '-B', '-m', module, *map(str, words)]
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    with path.open('xb') as log:
        worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, env=env)
        state['current'] = key
        task = dict(status='running', pid=worker.pid, command=command,
                    started_utc=utc_now().isoformat(), log=str(path))
        state['tasks'][key] = task
        write_json(output/'queue_state.json', sign(state))
        print('LEE_QUEUE',key,'pid',worker.pid,flush=True)
        # A child checks the same harddeadline before every microstep and commit.
        # Wait for it to preserve published commits; never kill another task.
        while worker.poll() is None:
            time.sleep(15)
        code = worker.returncode
    task.update(status='complete' if code == 0 else 'incomplete', returncode=code,
                exited_utc=utc_now().isoformat())
    write_json(output/'queue_state.json', sign(state))
    if code == 75:
        raise DeadlineReached('Child preserved committed state at original deadline')
    if code != 0 and not allow_failed_probe:
        raise RuntimeError(f'{key} failed with returncode={code}; inspect retained log')
    check_deadline()
    return code, path


def run(args):
    if os.name != 'posix':
        raise ValueError('Use authenticated server terminal only')
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refuse duplicate/nonempty training queue output')
    check_deadline()
    state = dict(schema='lee_serial_training_queue_v1',status='running',current=None,
                 tasks={},started_utc=utc_now().isoformat(),all_experiments_complete=False,
                 scoring_delivery_qualified=False,parent_receipt_sha256=file_hash(args.parent_receipt))
    with controller_lock(output):
        try:
            candidates=[]
            for micro in (1,2,4,8):
                check_deadline()
                folder=output/f'candidate_micro{micro}'
                manifest=prepare(args.project,args.reference_results,args.validation_cache,folder,micro)
                qualification=folder/'actual_double_backward_probe.json'
                code,_=run_child('lee_meta_equal40.probe',
                    ['--manifest',manifest,'--project',args.project,'--parent-receipt',args.parent_receipt,
                     '--output',qualification,'--microbatch',micro],output,state,f'probe_micro{micro}',True)
                if qualification.exists():
                    proof=read_signed(qualification)
                    if (code==0) != (proof.get('status')=='passed'):
                        raise ValueError('Probe exit status differs from signed qualification')
                    candidates.append((micro,manifest,qualification))
            cost,micro,manifest,qualification=select_candidate(candidates)
            state['selection']=dict(microbatch=micro,manifest=str(manifest),qualification=str(qualification),
                qualification_sha256=file_hash(qualification),
                criterion='one_meta_plus39_online_actual_exclusive_hardware_seconds',
                estimated_exclusive_seconds=cost,fullpath_qualification=False)
            write_json(output/'microbatch_selection.json',sign(state['selection']))
            write_json(output/'queue_state.json',sign(state))
            for seed in (42,43,44):
                run_child('lee_meta_equal40.train',
                    ['--manifest',manifest,'--project',args.project,'--parent-receipt',args.parent_receipt,
                     '--qualification',qualification,'--seed',seed],output,state,f'train_seed{seed}')
                folder=manifest.parent/'checkpoints'/f'lee_meta_seed{seed}'
                _,log=run_child('lee_meta_equal40.verify_training',
                    ['--manifest',manifest,'--project',args.project,'--folder',folder,
                     '--forti-code-root',args.forti_code_root,'--seed',seed],output,state,f'verify_seed{seed}')
                text=log.read_text(encoding='utf-8').strip()
                proof=ast.literal_eval(text)
                if (proof.get('status')!='single_seed_training_metadata_pass' or
                        proof.get('training_delivery_qualified') is not True or
                        proof.get('all_experiments_complete') is not False):
                    raise ValueError('Single-seed actual metadata not qualified')
                verification=folder/'TRAINING_METADATA_VERIFIED.json'
                write_json(verification,proof)
                if read_signed(verification)!=proof:
                    raise ValueError('Training metadata signature differs')
                state['tasks'][f'verify_seed{seed}']['proof_sha256']=file_hash(verification)
            state.update(status='three_seed_training_metadata_complete_scoring_pending',current=None,
                         finished_utc=utc_now().isoformat())
            write_json(output/'queue_state.json',sign(state))
            print('LEE_THREE_SEED_TRAINING_METADATA_ONLY_COMPLETE',flush=True)
            return 0
        except Exception as error:
            code=75 if isinstance(error,DeadlineReached) else 1
            state.update(status='incomplete_deadline' if code==75 else 'failed_or_incomplete',
                         exit_code=code,reason=str(error),traceback=traceback.format_exc(),
                         observed_utc=utc_now().isoformat())
            write_json(output/'queue_state.json',sign(state))
            write_json(output/'INCOMPLETE.json',sign(state))
            return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output','project','reference-results','validation-cache','parent-receipt','forti-code-root'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    check_deadline()
    if 'torch' in sys.modules or 'numpy' in sys.modules:
        raise ValueError('Controller imported a tensor runtime')
    sys.exit(run(args))


if __name__=='__main__':
    main()
