"""Explicit continuous-execution provenance; standard-library gates only.

Scientific training source19, recipe, model states and score protocols remain
immutable. No arbitrary future deadline or rewritten historical time is used.
"""
import importlib
import os
from pathlib import Path
import sys
from .protocol import (DeadlineReached, TRAINING_MANIFEST_SHA, digest, file_hash, require,
                       signed, bind_training_source)

MODE = 'until_complete_or_explicit_user_stop'
LEGACY_DEADLINE = '2026-10-02T13:46:32.725968Z'
EXTENSION_MANIFEST_SHA = '0a4c37934113bb69d06ea39a3aee67ee51cc19c8785565dcb0cbb423f0e49b4f'
EXTENSION_SOURCE_SIGNATURE = '2ac3df34ba9daa97f80deec837979b0835216281be1c63ffd661963dc2357efa'
TASKS = {kind+str(seed) for seed in (42,43,44) for kind in ('train_seed','verify_seed')}
LEGACY_TASKS = TASKS | {'probe_micro'+str(n) for n in (1,2,4,8)}
CONTEXT = None
EXTENSION_ROOT = None


class ExecutionStopped(DeadlineReached):
    pass


def control_fields(control, migration_sha):
    require(control.get('schema') == 'lee_execution_control_v1'
            and control.get('mode') == MODE and control.get('deadline_utc') is None
            and control.get('legacy_deadline_utc') == LEGACY_DEADLINE
            and control.get('legacy_source_manifest_sha256') == TRAINING_MANIFEST_SHA,
            'Bind explicit continuous-execution authorization, not a future date')
    return dict(execution_mode=MODE, execution_control_signature=control['signature'],
                extension_source_signature=control['extension_source_signature'],
                migration_receipt_sha256=migration_sha, legacy_deadline_superseded=True)


def identity():
    require(CONTEXT is not None, 'Bind reviewed extension control before downstream work')
    return control_fields(CONTEXT.control, CONTEXT.migration_sha)


def check_execution():
    require(CONTEXT is not None, 'No bound execution context')
    require(signed(CONTEXT.control_path) == CONTEXT.control
            and file_hash(CONTEXT.migration_path) == CONTEXT.migration_sha,
            'Execution control/migration changed during this operation')
    from lee_execution_extension.control import stop_requested
    if stop_requested(CONTEXT.control_path, CONTEXT.control):
        raise ExecutionStopped('Preserve completed work: explicit user stop')


def check_legacy_evidence(context):
    """Retain the actual old exit75/10-task queue and exclusive launch receipt."""
    root=context.evidence_root; migration=context.migration
    queue_path=root/'legacy_queue_state.json'; launch_path=root/'legacy_launch_receipt.json'
    require(file_hash(queue_path) == migration['legacy_queue_sha256']
            and file_hash(launch_path) == migration['legacy_launch_sha256'],
            'Legacy queue/launch snapshot SHA differs')
    queue=signed(queue_path)
    require(queue.get('status') == 'incomplete_deadline'
            and type(queue.get('exit_code')) is int and queue['exit_code'] == 75
            and {'probe_micro'+str(n) for n in (1,2,4,8)} <= set(queue.get('tasks', {})) <= LEGACY_TASKS,
            'Retain the actual interrupted subset of the legacy10-task plan')
    signed(launch_path)


def activate(control_path, migration_path, legacy_code, extension_code, evidence_root=None):
    global CONTEXT, EXTENSION_ROOT
    legacy_code=Path(legacy_code).resolve(); extension_code=Path(extension_code).resolve()
    bind_training_source(legacy_code)
    require(type(EXTENSION_MANIFEST_SHA) is str and len(EXTENSION_MANIFEST_SHA) == 64
            and type(EXTENSION_SOURCE_SIGNATURE) is str and len(EXTENSION_SOURCE_SIGNATURE) == 64,
            'Independent extension source pins are not frozen')
    require(file_hash(extension_code/'SOURCE_MANIFEST.json') == EXTENSION_MANIFEST_SHA,
            'Use the independently reviewed execution extension')
    sys.path.insert(0,str(extension_code))
    module=importlib.import_module('lee_execution_extension.control')
    require(Path(module.__file__).resolve() == extension_code/'lee_execution_extension/control.py',
            'Wrong execution extension module origin')
    own=module.verify_source(extension_code)
    require(own['signature'] == EXTENSION_SOURCE_SIGNATURE, 'Execution extension signature differs')
    context=module.activate_for_read(control_path,migration_path,legacy_code,evidence_root=evidence_root)
    require(context.control['extension_source_signature'] == own['signature'], 'Control binds another extension')
    if CONTEXT is not None:
        require(CONTEXT.control == context.control and CONTEXT.migration_sha == context.migration_sha,
                'Cannot switch execution identity in one downstream process')
    check_legacy_evidence(context)
    CONTEXT=context; EXTENSION_ROOT=extension_code
    require('torch' not in sys.modules, 'Metadata activation must not import Torch')
    return identity()


def require_fields(record, expected):
    require(all(record.get(k) == v for k,v in expected.items()), 'Continuous execution provenance differs')


def check_queue_ready(queue, expected):
    require(queue.get('schema') == 'lee_execution_extension_queue_v1'
            and queue.get('status') == 'three_seed_training_metadata_complete_scoring_pending'
            and queue.get('current') is None and set(queue.get('tasks', {})) == TASKS,
            'Only the public extension6-task complete queue qualifies')
    require_fields(queue, expected)
    for name in TASKS:
        task=queue['tasks'][name]
        require(task.get('status') == 'complete' and type(task.get('returncode')) is int
                and task['returncode'] == 0 and type(task.get('pid')) is int and task['pid'] > 0,
                'Every extension training/metadata task must actually exit integer0')


def live_pid(pid):
    if os.name != 'posix':
        return False
    path=Path('/proc')/str(pid)/'stat'
    try:
        raw=path.read_text(); state=raw[raw.rfind(')')+2:].split()[0]
    except FileNotFoundError:
        return False
    except (OSError, IndexError):
        raise ValueError('Cannot determine recorded worker liveness')
    return state != 'Z'


def no_live_tasks(queue):
    for task in queue['tasks'].values():
        require(not live_pid(task['pid']), 'Recorded worker is live or PID has been reused')


def check_receipt_members(context):
    expected={f'lee_meta_seed{seed}/epoch{epoch:02d}.json' for seed in (42,43,44)
              for epoch in range(len(context.migration['prefix_commits'][str(seed)])+1,41)}
    root=context.evidence_root/'execution_commits'
    actual=set()
    if root.exists():
        for path in root.rglob('*'):
            require(not path.is_symlink(), 'No symlinked execution receipts')
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
    require(actual==expected, 'The complete120-state ledger needs exactly its new-commit control receipts')


def check_migration_copies(context, training):
    from .protocol import safe_name
    migration=context.migration
    require(migration.get('extension_source_signature')==context.control['extension_source_signature']
            and migration.get('legacy_bytes_rewritten') is False
            and migration.get('scientific_binding_changed') is False
            and migration.get('all_experiments_complete') is False
            and migration.get('training_manifest_sha256')==file_hash(Path(training)/'lee_manifest.json'),
            'The extension must preserve actual legacy scientific bytes/binding')
    require(type(migration.get('copied_files')) is dict and bool(migration['copied_files']), 'Missing complete migration copy ledger')
    for relative,record in migration['copied_files'].items():
        require(safe_name(relative), 'Unsafe migration member path')
        if relative.startswith('candidate_micro8/'):
            path=Path(training)/relative[len('candidate_micro8/'):]
        else:
            require(relative in ('legacy_queue_state.json','legacy_launch_receipt.json')
                    or relative.startswith(('legacy_diagnostics/','legacy_logs/')), 'Unknown migration copy namespace')
            path=context.evidence_root/relative
        require(path.is_file() and not path.is_symlink() and path.stat().st_size==record['bytes']
                and type(record['bytes']) is int and record['bytes']>0 and file_hash(path)==record['sha256'],
                'Actual archived/live migration copy SHA/bytes differs: '+relative)


def restore(context, extension_root):
    global CONTEXT, EXTENSION_ROOT
    if context is not None:
        activate(context.control_path,context.migration_path,context.legacy_root,
                 extension_root,evidence_root=context.evidence_root)
    else:
        # A standalone ZIP check must not retain a pointer into a deleted tempdir.
        CONTEXT=None;EXTENSION_ROOT=extension_root


def bind_args(args, read_only=False):
    expected=activate(args.execution_control,args.migration_receipt,args.training_code_root,
                      args.extension_code_root)
    control=CONTEXT.control
    require(Path(control['project']).resolve() == args.project.resolve()
            and Path(control['legacy_code_root']).resolve() == args.training_code_root.resolve()
            and args.training_queue.resolve() == Path(control['new_output']).resolve()/'queue_state.json',
            'Use the authorized project, legacy source and extension queue')
    queue=signed(args.training_queue)
    check_queue_ready(queue,expected); no_live_tasks(queue)
    require(Path(queue['selection']['manifest']).resolve() == (args.training_output/'lee_manifest.json').resolve()
            and (Path(control['new_output']).resolve()==args.training_output.resolve()
                 or Path(control['new_output']).resolve() in args.training_output.resolve().parents),
            'Use the selected independent extension training root')
    if not read_only:
        check_execution()
    return expected, queue


def cli_arguments(parser):
    for name in ('execution-control','migration-receipt','extension-code-root'):
        parser.add_argument('--'+name,type=Path,required=True)
