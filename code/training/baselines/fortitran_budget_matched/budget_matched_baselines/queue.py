"""Resume-safe serial training queue, gated by completed frozen verification.

This queue never stops the verification process and never trains from its test
or calibration arrays.  A compute-only resource probe cannot open the gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
if not __package__:
    _package_dir = Path(__file__).resolve().parent
    sys.path[:] = [v for v in sys.path if Path(v or '.').resolve() != _package_dir]
    sys.path.insert(0, str(_package_dir.parent))
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from .resource_probe import digest, pid_from_file, process_alive, read_json, utc_now, write_json
except ImportError:
    from budget_matched_baselines.resource_probe import digest, pid_from_file, process_alive, read_json, utc_now, write_json

_REBUNDLED_OUTPUTS = set()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verification_workers_alive(code_root, controller_pid):
    if process_alive(controller_pid):
        return True
    if os.name != 'posix':
        return False
    tokens = ('run_fast_verification.py', 'run_verification.py', 'resume_scoring_controller.py',
              'resume_gpu_verification.py')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')
        except OSError:
            continue
        if str(code_root) in command and any(token in command for token in tokens) and process_alive(int(proc.name)):
            return True
    return False


def completion_gate(args):
    output = args.verification_output
    pid = pid_from_file(args.verification_pid_file)
    if verification_workers_alive(args.verification_code_root, pid):
        return False, 'verification controller or worker is still running', None
    try:
        statistics = read_json(output / 'corrected_summary/statistics.json')
        if statistics.get('paper_ready') is not True or statistics.get('pending') != []:
            return False, 'verification statistics are not paper ready', None
        if statistics.get('completed_verified_cells') != args.expected_cells:
            return False, 'the declared corrected-cell suite is incomplete', None
        # The worker/controller must have exited before rebuilding.  The initial
        # report ZIP may have captured a log which was still being appended.
        # Bundle a static snapshot once per queue process; no score, calibration,
        # timing or statistical calculation is repeated.
        code_root = str(args.verification_code_root.resolve())
        if code_root not in sys.path:
            sys.path.insert(0, code_root)
        output_key = str(output.resolve())
        if output_key not in _REBUNDLED_OUTPUTS:
            from verification_gpu.reporting import bundle_gpu_results
            bundle_gpu_results(output, statistics)
            _REBUNDLED_OUTPUTS.add(output_key)
        # Reuses the delivery verifier: full 269 score SHA binding, 512 frames,
        # independent calibration lock, exact GPU 144-row matrix and ZIP CRC.
        from verification_gpu.verify_delivery import verify
        archive = output / 'verification_results_for_paper.zip'
        delivery = verify(archive, args.reference_results, args.verification_code_root)
        if delivery.get('manuscript_cells_verified') != args.expected_cells:
            return False, 'delivery count differs', None
        return True, 'full frozen verification delivery passed', delivery
    except (OSError, ValueError, KeyError, ImportError) as error:
        return False, str(error), None


def load_tasks(path):
    value = read_json(path)
    tasks = value.get('tasks') if isinstance(value, dict) else value
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('Tasks JSON requires a nonempty list of task objects')
    names = set()
    for task in tasks:
        name, command = task.get('name'), task.get('command')
        if not isinstance(name, str) or not name or name in names or '/' in name or '\\' in name:
            raise ValueError('Task names must be unique path-free labels')
        if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
            raise ValueError('Each task requires an argv list, not a shell string')
        names.add(name)
    return tasks


def parallel_gate(args, tasks):
    """Full-path evidence only; the default 6-GiB compute probe never passes."""
    if not args.allow_parallel or not args.parallel_probe:
        return False
    probe = read_json(args.parallel_probe)
    if (probe.get('status') != 'complete' or probe.get('eligible_full_path') is not True
            or probe.get('probe_kind') != 'full-path' or probe.get('real_training_data') is not True):
        return False
    # Tasks must explicitly bind their full-pipeline resource eligibility to the
    # measured probe; a different recipe or generator is not silently approved.
    sha = file_hash(args.parallel_probe)
    return all(task.get('full_path_probe_sha256') == sha for task in tasks)


def expected_outputs(task):
    records = {}
    for item in task.get('expected_json', []):
        value = read_json(item['file'])
        for key, expected in item['fields'].items():
            if value.get(key) != expected:
                raise RuntimeError('Task did not meet its completion contract: ' + str(item['file']) + ': ' + key)
    for item in task.get('expected_outputs', []):
        path = Path(item)
        if not path.is_file():
            raise RuntimeError('Task reported success but an expected output is absent: ' + str(path))
        records[str(path.resolve())] = file_hash(path)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--verification-output', type=Path, required=True)
    p.add_argument('--verification-pid-file', type=Path, required=True)
    p.add_argument('--verification-code-root', type=Path, default=ROOT)
    p.add_argument('--reference-results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--parallel-probe', type=Path)
    p.add_argument('--allow-parallel', action='store_true')
    p.add_argument('--expected-cells', type=int, default=269)
    p.add_argument('--poll-seconds', type=float, default=60)
    p.add_argument('--deadline-utc')
    args = p.parse_args()
    if args.expected_cells != 269 or args.poll_seconds <= 0:
        p.error('Current manuscript gate requires 269 cells and a positive polling interval')
    args.output = args.output.resolve()
    protected = (args.verification_output.resolve(), args.reference_results.resolve())
    if any(args.output == path or path in args.output.parents for path in protected):
        p.error('Queue output must be separate from frozen verification/reference folders')
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.tasks)
    task_digest = digest(tasks)
    lock = args.output / 'queue.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        old = read_json(lock)
        if process_alive(old.get('pid')):
            raise SystemExit('The existing training queue is still running; no duplicate started')
        lock.unlink()
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump({'pid': os.getpid(), 'started_utc': utc_now(), 'task_manifest_sha256': task_digest}, stream)
    state_path = args.output / 'queue_state.json'
    state = read_json(state_path) if state_path.is_file() else {
        'schema_version': 1, 'started_utc': utc_now(), 'tasks': {}, 'task_manifest_sha256': task_digest}
    if state['task_manifest_sha256'] != task_digest:
        lock.unlink()
        raise SystemExit('Task list differs from the existing queue; choose a new queue output')
    for name, record in state.get('tasks', {}).items():
        if record.get('status') == 'running' and process_alive(record.get('pid')):
            lock.unlink()
            raise SystemExit('Previously launched worker is still running: ' + name + '; no duplicate started')
    deadline = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00')) if args.deadline_utc else None
    if deadline and deadline.tzinfo is None:
        lock.unlink()
        p.error('--deadline-utc must include a UTC offset')
    try:
        parallel = parallel_gate(args, tasks)
        while not parallel:
            passed, reason, delivery = completion_gate(args)
            state.update(status='ready' if passed else 'waiting_for_verification',
                         checked_utc=utc_now(), verification_gate_reason=reason)
            if delivery:
                state['verification_delivery'] = delivery
            write_json(state_path, state)
            print('[training-queue] ' + reason, flush=True)
            if passed:
                break
            if deadline and datetime.now(timezone.utc) >= deadline:
                state.update(status='deferred_deadline_before_training', completed_utc=utc_now())
                write_json(state_path, state)
                return
            time.sleep(args.poll_seconds)
        for task in tasks:
            previous = state['tasks'].get(task['name'], {})
            if previous.get('status') == 'complete':
                if expected_outputs(task) != previous.get('output_sha256', {}):
                    raise RuntimeError('Completed task output changed: ' + task['name'])
                continue
            if deadline and datetime.now(timezone.utc) >= deadline:
                state.update(status='deferred_deadline_between_tasks', completed_utc=utc_now())
                write_json(state_path, state)
                return
            env = os.environ.copy()
            env.update({str(k): str(v) for k, v in task.get('env', {}).items()})
            if deadline:
                env['DOPNET_TRAINING_DEADLINE_UTC'] = deadline.isoformat()
            log = args.output / (task['name'] + '.log')
            record = {'status': 'running', 'started_utc': utc_now(), 'command_sha256': digest(task),
                      'log': str(log), 'parallel_with_verification': parallel}
            state['tasks'][task['name']] = record
            state['status'] = 'training'
            write_json(state_path, state)
            print('[training-queue] starting ' + task['name'], flush=True)
            with log.open('ab') as stream:
                worker = subprocess.Popen(task['command'], cwd=task.get('cwd'), env=env,
                                          stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                record['pid'] = worker.pid
                write_json(state_path, state)
                code = worker.wait()
            record.update(returncode=code, completed_utc=utc_now())
            if code == 75:
                record['status'] = 'incomplete_time_limit'
                state['status'] = 'deferred_deadline_incomplete_training'
                write_json(state_path, state)
                print('[training-queue] checkpoint saved; time budget reached; scoring not started', flush=True)
                return
            if code:
                record['status'] = 'failed'
                state['status'] = 'failed'
                write_json(state_path, state)
                raise RuntimeError('Training task failed; verification was untouched; see ' + str(log))
            record.update(status='complete', output_sha256=expected_outputs(task))
            write_json(state_path, state)
        state.update(status='complete', completed_utc=utc_now())
        write_json(state_path, state)
        print('[training-queue] all declared training tasks complete', flush=True)
    finally:
        # This lock belongs to this queue only; never touch verification locks.
        if lock.is_file() and read_json(lock).get('pid') == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    main()
