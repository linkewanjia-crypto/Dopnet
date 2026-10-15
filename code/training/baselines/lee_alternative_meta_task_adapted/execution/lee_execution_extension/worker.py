"""One explicitly authorized continuation or metadata task, never a probe."""
import argparse
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from .control import (READER_SHA, activate_for_read, file_hash, no_live_training,
                      require, sign, write)
from .overlay import recover_intents


def run(args):
    require(os.name == 'posix','Authenticated CUDA server only')
    context = activate_for_read(args.execution_control,args.migration_receipt,args.legacy_code_root)
    c = context.control; require(args.controller_pid > 0,'Missing actual administrative controller PID')
    from .control import signed
    require(signed(Path(args.execution_control).parent/'launch_receipt.json').get('pid') == args.controller_pid,'Controller PID differs from signed launch')
    no_live_training(args.controller_pid)
    import subprocess
    gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
    require(not gpu.stdout.strip(),'GPU compute process active; do not resume concurrently')
    root = Path(c['new_output'])/'candidate_micro8'; recover_intents(context,root)
    context.check()
    manifest = root/'lee_manifest.json'; folder = root/'checkpoints'/f'lee_meta_seed{args.seed}'
    if args.stage == 'train':
        require(args.reference_token == 'lee_meta_equal40.train','Public legacy process token required')
        t = importlib.import_module('lee_meta_equal40.train')
        result = t.train(SimpleNamespace(manifest=str(manifest),project=c['project'],qualification=str(root/'actual_double_backward_probe.json'),parent_receipt=c['parent_receipt'],seed=args.seed,resume=True))
        from .control import source, verify_source, LEGACY_SOURCE_SHA
        source(context.legacy_root,LEGACY_SOURCE_SHA); verify_source(Path(__file__).resolve().parent.parent)
        write(Path(c['new_output'])/'execution_tasks'/f'train_seed{args.seed}.json',sign(dict(schema='lee_execution_task_v1',stage='train',seed=args.seed,exit_code=result,execution_control_signature=c['signature'],migration_receipt_sha256=context.migration_sha,all_experiments_complete=False)))
        return result
    require(args.reference_token == 'lee_meta_equal40.verify_training','Public metadata process token required')
    pkg = Path(c['forti_code_root']); require(file_hash(pkg/'verify_author_delivery.py') == READER_SHA,'Restricted reader source changed')
    sys.path.insert(0,str(pkg.parent)); reader = importlib.import_module('budget_matched_baselines.verify_author_delivery')
    require(Path(reader.__file__).resolve() == pkg.resolve()/'verify_author_delivery.py','Wrong reader module origin')
    require('torch' not in sys.modules,'Metadata process imported Torch')
    verifier = importlib.import_module('lee_meta_equal40.verify_training')
    proof = verifier.verify(folder,manifest,c['project'],args.seed,reader.checkpoint_metadata)
    require('torch' not in sys.modules,'Metadata verification imported Torch')
    write(folder/'TRAINING_METADATA_VERIFIED.json',proof); print(proof,flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('execution-control','migration-receipt','legacy-code-root','reference-token'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--stage',choices=('train','verify'),required=True); p.add_argument('--seed',type=int,choices=(42,43,44),required=True)
    p.add_argument('--controller-pid',type=int,required=True); args = p.parse_args()
    try:
        result = run(args)
    except Exception as error:
        # The scientific trainer preserves published commits on user stop; no other process is signalled.
        if type(error).__name__ == 'DeadlineReached':
            result = 75
        else:
            raise
    sys.exit(result)


if __name__ == '__main__':
    main()
