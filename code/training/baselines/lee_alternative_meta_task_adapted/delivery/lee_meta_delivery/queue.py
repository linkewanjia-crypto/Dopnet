"""Unique serial scoring queue under explicit continuous execution control.

All three40-epoch/metadata tasks must naturally exit first. This administrative
controller never imports a tensor runtime or starts training or buys/renews rent.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import traceback
from .protocol import (DeadlineReached, bind_training_source, check_deadline,
                       no_live_training, require, sign, verify_source, write)
from . import execution as X
from . import recovery as Recovery


@contextmanager
def scoring_lease(path):
    import fcntl
    with Path(path).open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stream.seek(0);stream.truncate();stream.write(str(os.getpid())+'\n');stream.flush()
        yield


def gpu_empty():
    value=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                         check=True,capture_output=True,text=True)
    require(not value.stdout.strip(),'Another actual GPU compute process is active; preserve serial execution')


def run(args):
    require(os.name=='posix','Authorized server only')
    own=verify_source(Path(__file__).resolve().parent.parent)
    expected,queue=X.bind_args(args);check_deadline();no_live_training();gpu_empty()
    Recovery.inspect_stopped(args.recovery_parent)
    bind_training_source(args.training_code_root)
    control=args.control.resolve();out=args.output.resolve()
    require(not control.exists() and (not out.exists() or not any(out.iterdir())),
            'Refuse duplicate scoring launch/nonempty independent output')
    with scoring_lease(X.CONTEXT.control_path.parent/'.lee_scoring_queue.lock'):
        check_deadline();X.no_live_tasks(queue);no_live_training();gpu_empty()
        control.mkdir()
        state=dict(schema='lee_extension_scoring_queue_v1',status='running',tasks={},current=None,
                   started_utc=datetime.now(timezone.utc).isoformat(),source_signature=own['signature'],
                   all_experiments_complete=False,**expected)
        write(control/'queue_state.json',sign(state))
        common=[]
        for name in ('project','reference-results','training-output','training-code-root','training-queue',
                     'parent-archive','output','execution-control','migration-receipt','extension-code-root'):
            common+=['--'+name,str(getattr(args,name.replace('-','_')))]
        try:
            for stage in ('evaluate','report'):
                check_deadline();X.no_live_tasks(queue);no_live_training();gpu_empty()
                words=[sys.executable,'-u','-B','-m','lee_meta_delivery.'+stage,*common]
                if stage=='evaluate':words+=['--recovery-parent',str(args.recovery_parent)]
                if stage=='report':words+=['--forti-code-root',str(args.forti_code_root)]
                with (control/(stage+'.log')).open('xb') as log:
                    process=subprocess.Popen(words,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    state['current']=stage;state['tasks'][stage]=dict(status='running',pid=process.pid,command=words)
                    write(control/'queue_state.json',sign(state));code=process.wait()
                state['tasks'][stage].update(status='complete' if code==0 else 'incomplete',returncode=code,
                                           exited_utc=datetime.now(timezone.utc).isoformat())
                write(control/'queue_state.json',sign(state))
                if code==75:
                    raise DeadlineReached('Child preserved completed artifacts on explicit user stop')
                require(type(code) is int and code==0,stage+' failed; retain all evidence')
            check_deadline()
            state.update(status='Lee_scoring_report_delivery_complete_latency_and_local_handoff_pending',
                         current=None,finished_utc=datetime.now(timezone.utc).isoformat())
            write(control/'queue_state.json',sign(state));return 0
        except Exception as error:
            code=75 if isinstance(error,DeadlineReached) else 1
            state.update(status='incomplete_explicit_stop' if code==75 else 'failed_or_incomplete',
                         exit_code=code,reason=str(error),traceback=traceback.format_exc(),
                         observed_utc=datetime.now(timezone.utc).isoformat())
            write(control/'queue_state.json',sign(state));return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','reference-results','training-output','training-code-root','training-queue',
                 'parent-archive','output','forti-code-root','control'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--recovery-parent',type=Path,required=True)
    X.cli_arguments(parser);args=parser.parse_args()
    require('torch' not in sys.modules and 'numpy' not in sys.modules,'Administrative queue must not import tensor runtimes')
    try:
        code=run(args)
    except DeadlineReached:
        code=75
    sys.exit(code)


if __name__=='__main__':
    main()
