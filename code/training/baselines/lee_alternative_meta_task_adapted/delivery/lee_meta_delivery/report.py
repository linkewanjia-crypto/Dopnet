"""Server CPU report and frozen Lee delivery; full directory and real ZIP gates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED
from .protocol import (ASSET_NAMES, CLAIM, DEADLINE, DeadlineReached, bind_training_source,
    check_deadline, digest, file_hash, no_live_training, require, sign, signed, verify_source, write)
from .render import render_assets
from .verify import verify_archive, verify_directory
from . import execution as X


def run(args):
    execution_binding,_=X.bind_args(args)
    check_deadline();no_live_training()
    source=Path(__file__).resolve().parent.parent;own=verify_source(source)
    training_source=bind_training_source(args.training_code_root)
    out=args.output.resolve();training=args.training_output.resolve()
    def directory(assets):
        return verify_directory(out,training,args.project,args.reference_results,args.parent_archive,args.training_code_root,args.forti_code_root,source,assets)
    if (out/'COMPLETE_DELIVERY.json').exists():
        complete=signed(out/'COMPLETE_DELIVERY.json');package=out/'lee_meta_results_for_paper.zip'
        proof=verify_archive(package,args.reference_results,args.training_code_root,args.forti_code_root,source,args.extension_code_root)
        X.require_fields(complete,execution_binding)
        require(complete.get('status')=='complete' and complete.get('paper_ready') is True and complete.get('pending')==[]
                and complete.get('all_experiments_complete') is False and complete.get('deadline_utc') is None
                and complete.get('legacy_deadline_utc')==DEADLINE
                and proof['archive_sha256']==complete['archive_sha256'] and package.stat().st_size==complete['archive_bytes'],
                'Existing final archive/explicit execution commit differs')
        print(json.dumps(proof));return
    directory(False);check_deadline()
    if not (out/'PAPER_READY.json').exists():
        render_assets(out);check_deadline()
        ev=signed(out/'evaluation.json');statistics=signed(out/'summary/statistics.json')
        write(out/'PAPER_READY.json',sign(dict(status='complete',paper_ready=True,pending=[],evaluation_signature=ev['signature'],statistics_signature=statistics['signature'],assets_sha256={n:file_hash(out/'paper_assets'/n) for n in ASSET_NAMES})))
    proof=directory(True);check_deadline()
    manifest=signed(training/'lee_manifest.json');files={}
    for path in out.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.csv','.npz','.tex','.pdf','.png') and path.name not in ('COMPLETE_DELIVERY.json','FINAL_VERIFICATION.json'):
            files['evaluation/'+path.relative_to(out).as_posix()]=path
    for path in training.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.pt'):
            require(not any(part.endswith('.staging') for part in path.relative_to(training).parts),'Unresolved checkpoint staging directory')
            files['training/'+path.relative_to(training).as_posix()]=path
    for name in training_source['members']:
        files['source/training/'+name]=args.training_code_root/name
    files['source/training/SOURCE_MANIFEST.json']=args.training_code_root/'SOURCE_MANIFEST.json'
    for name in own['members']:
        files['source/downstream/'+name]=source/name
    files['source/downstream/SOURCE_MANIFEST.json']=source/'SOURCE_MANIFEST.json'
    for name in manifest['runtime_sources_sha256']:
        files['source/archive_runtime/'+name]=args.project/name
    files['evidence/wimae_results_for_paper.zip']=args.parent_archive
    # Preserve the original policy/queue/launch and every new immutable-commit
    # control receipt. Offline verification uses this relocated evidence root.
    context=X.CONTEXT
    files['evidence/execution/EXECUTION_CONTROL.json']=context.control_path
    files['evidence/execution/MIGRATION_RECEIPT.json']=context.migration_path
    for name in ('legacy_queue_state.json','legacy_launch_receipt.json'):
        files['evidence/execution/'+name]=context.evidence_root/name
    X.check_receipt_members(context)
    X.check_migration_copies(context,training)
    for name in context.migration['copied_files']:
        if not name.startswith('candidate_micro8/'):
            files['evidence/execution/'+name]=context.evidence_root/name
    for path in (context.evidence_root/'execution_commits').rglob('*.json'):
        files['evidence/execution/execution_commits/'+path.relative_to(context.evidence_root/'execution_commits').as_posix()]=path
    extension=signed(args.extension_code_root/'SOURCE_MANIFEST.json')
    for name in extension['members']:
        files['source/execution_extension/'+name]=args.extension_code_root/name
    files['source/execution_extension/SOURCE_MANIFEST.json']=args.extension_code_root/'SOURCE_MANIFEST.json'
    # Snapshots are taken after all training workers and the controller exit.
    # Logs are auxiliary; exact scientific/selected state provenance is in commits.
    for path in args.training_queue.parent.joinpath('logs').glob('*.log'):
        files['logs/training/'+path.name]=path
    for path in (out/'recovery').rglob('*'):
        require(not path.is_symlink(),'No symlinked recovery archive members')
        if path.is_file():files['evaluation/recovery/'+path.relative_to(out/'recovery').as_posix()]=path
    package=out/'lee_meta_results_for_paper.zip';temporary=out/'lee_meta_results_for_paper.zip.tmp'
    hashes={}
    with ZipFile(temporary,'w',ZIP_DEFLATED) as z:
        for name,path in sorted(files.items()):
            check_deadline();data=path.read_bytes();hashes[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
        delivery=sign(dict(schema='lee_equal40_delivery_v1',status='complete',paper_ready=True,pending=[],
                          claim=CLAIM,original_author_dataset_recipe_reproduction=False,all_experiments_complete=False,
                          evaluation_signature=proof['evaluation_signature'],training_manifest_signature=manifest['signature'],files=hashes,
                          **execution_binding))
        z.writestr('DELIVERY_MANIFEST.json',json.dumps(delivery,indent=2))
    check_deadline();actual=verify_archive(temporary,args.reference_results,args.training_code_root,args.forti_code_root,source,args.extension_code_root)
    check_deadline();temporary.replace(package)
    require(file_hash(package)==actual['archive_sha256'],'Published ZIP differs from checked bytes')
    final=sign(dict(status='complete',paper_ready=True,pending=[],archive_sha256=actual['archive_sha256'],archive_bytes=package.stat().st_size,
                    evaluation_signature=proof['evaluation_signature'],finished_utc=datetime.now(timezone.utc).isoformat(),deadline_utc=None,legacy_deadline_utc=DEADLINE,
                    Lee_latency_complete=False,uniform_local_delivery_complete=False,all_experiments_complete=False,**execution_binding))
    check_deadline();write(out/'COMPLETE_DELIVERY.json',final);write(out/'FINAL_VERIFICATION.json',sign(actual))
    print('LEE_FULL_DELIVERY_VERIFIED eight_methods=8 routes=24 cells=269 latency_and_local_handoff_pending',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('project','reference-results','training-output','training-code-root','training-queue','parent-archive','forti-code-root','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    X.cli_arguments(ap)
    args=ap.parse_args()
    try:
        run(args)
    except DeadlineReached as error:
        print('LEE_REPORT_INCOMPLETE_EXPLICIT_STOP '+str(error),file=sys.stderr,flush=True)
        raise SystemExit(75)


if __name__=='__main__':
    main()
