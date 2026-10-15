"""Short resource and verification-throughput probe; never trains on test frames.

The compute-only worker uses synthetic shape-matched tensors.  Its timings are
hardware information, not scientific training or proof of full-pipeline safety.
Only a separately attested full-path probe can qualify concurrent real training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import sys
# Direct script execution must not expose this package's queue.py as stdlib
# queue (Torch and concurrent.futures import the standard module by that name).
if not __package__:
    _package_dir = Path(__file__).resolve().parent
    sys.path[:] = [v for v in sys.path if Path(v or '.').resolve() != _package_dir]
    sys.path.insert(0, str(_package_dir.parent))
import subprocess
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def gpu_snapshot(index=0):
    command = ['nvidia-smi', '-i', str(index),
               '--query-gpu=uuid,name,memory.total,memory.used,memory.free,utilization.gpu',
               '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError('The requested GPU must resolve to exactly one device')
    parts = [v.strip() for v in lines[0].split(',')]
    if len(parts) != 6:
        raise RuntimeError('Unexpected nvidia-smi resource schema')
    return {'uuid': parts[0], 'name': parts[1],
            'total_gib': float(parts[2]) / 1024,
            'used_gib': float(parts[3]) / 1024,
            'free_gib': float(parts[4]) / 1024,
            'utilization_percent': float(parts[5])}


def process_alive(pid):
    if not pid or int(pid) <= 1:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # An unreaped child has finished work and must not block the queue forever.
    path = Path('/proc') / str(pid) / 'stat'
    if path.is_file():
        try:
            if path.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                return False
        except OSError:
            pass
    return True


def pid_from_file(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


FRAME_LINE = re.compile(r'\[(\d+)/(\d+)\]\s+(\w+)\s+(\d+)/512 frames;\s+([\d.]+)s')


class ProgressReader:
    def __init__(self, output, log=None):
        self.output = Path(output)
        self.log = Path(log) if log else self._find_log()
        self.offset = max(0, self.log.stat().st_size - 300000) if self.log else 0
        self.last = None
        self.latest_by_cell = {}

    def _find_log(self):
        candidates = list((self.output / 'logs').glob('*evaluate*.log'))
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    def sample(self):
        new = []
        if self.log and self.log.is_file():
            if self.log.stat().st_size < self.offset:
                self.offset = 0
            with self.log.open('rb') as stream:
                stream.seek(self.offset)
                data = stream.read()
                self.offset = stream.tell()
            for match in FRAME_LINE.finditer(data.decode('utf-8', errors='replace')):
                row = {'number': int(match[1]), 'total': int(match[2]),
                       'cell_id': match[3], 'frames': int(match[4]),
                       'cell_elapsed_seconds': float(match[5])}
                previous = self.latest_by_cell.get(row['cell_id'])
                if previous and row['frames'] > previous['frames']:
                    dt = row['cell_elapsed_seconds'] - previous['cell_elapsed_seconds']
                    if dt > 0:
                        row['interval_frames_per_second'] = (row['frames'] - previous['frames']) / dt
                self.latest_by_cell[row['cell_id']] = row
                self.last = row
                new.append(row)
        complete = 0
        for marker in (self.output / 'cells').glob('*.COMPLETE.json'):
            try:
                m = read_json(marker)
                complete += m.get('status') == 'complete' and m.get('frozen_waveform_sha256_pass') is True
            except (OSError, ValueError):
                continue
        cursor = complete * 512
        if self.last:
            cursor = max(cursor, (self.last['number'] - 1) * 512 + self.last['frames'])
        return {'completed_cells': complete, 'frame_cursor': cursor,
                'last_logged_progress': self.last, 'new_progress': new}


def summarize(samples):
    if not samples:
        return {}
    seconds = samples[-1]['monotonic'] - samples[0]['monotonic']
    frames = samples[-1]['verification']['frame_cursor'] - samples[0]['verification']['frame_cursor']
    intervals = [r for s in samples for r in s['verification']['new_progress']
                 if 'interval_frames_per_second' in r]
    by_cell = {}
    for row in intervals:
        by_cell.setdefault(row['cell_id'], []).append(row['interval_frames_per_second'])
    return {'observed_seconds': seconds, 'completed_cells_start': samples[0]['verification']['completed_cells'],
            'completed_cells_end': samples[-1]['verification']['completed_cells'],
            'logged_frame_progress': max(0, frames),
            'logged_frames_per_second': max(0, frames) / seconds if seconds else None,
            'per_cell_interval_fps': {k: statistics.median(v) for k, v in by_cell.items()},
            'gpu_utilization_percent_median': statistics.median(s['gpu']['utilization_percent'] for s in samples),
            'gpu_memory_used_gib_max': max(s['gpu']['used_gib'] for s in samples),
            'gpu_memory_free_gib_min': min(s['gpu']['free_gib'] for s in samples)}


def sample_resources(reader, index):
    return {'utc': utc_now(), 'monotonic': time.monotonic(),
            'gpu': gpu_snapshot(index), 'verification': reader.sample()}


def stop_own_worker(process):
    """Only signal the newly created probe process group, never verification."""
    if process.poll() is not None:
        return
    if os.name == 'posix':
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=15)


def load_command(path):
    value = read_json(path)
    spec = {'command': value} if isinstance(value, list) else value
    command = spec.get('command')
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError('Probe command must be a nonempty argv array (no shell string)')
    return spec


def run_probe(args):
    reader = ProgressReader(args.verification_output, args.verification_log)
    # Prime the cursor from existing log text, then discard historical intervals.
    # Throughput evidence must be observed inside the declared baseline window.
    reader.sample()
    reader.latest_by_cell.clear()
    spec = load_command(args.probe_command_json)
    result = {'schema_version': 1, 'started_utc': utc_now(), 'probe_kind': args.probe_kind,
              'real_training_data': False, 'eligible_full_path': False,
              'command_sha256': digest(spec), 'worker_result': None,
              'verification_pid': pid_from_file(args.verification_pid_file),
              'verification_pid_file': str(args.verification_pid_file),
              'gpu_memory_cap_gib': args.gpu_memory_cap_gib,
              'reserve_gib': args.reserve_gib,
              'verification_protected_peak_gib': args.verification_peak_gib,
              'full_path_generation_peak_gib': args.full_path_generation_peak_gib,
              'scientific_generation_contract': {'generation_batch_size': 32, 'microbatch_size': 4,
                                                  'threegpp_link_batch_size': 8},
              'samples': {'baseline': [], 'probe': []}}
    started = time.monotonic()
    while time.monotonic() - started < args.baseline_seconds:
        result['samples']['baseline'].append(sample_resources(reader, args.gpu_index))
        time.sleep(min(args.sample_seconds, max(0, args.baseline_seconds - (time.monotonic() - started))))
    baseline = result['samples']['baseline']
    worst_probe_headroom = (min(s['gpu']['total_gib'] for s in baseline)
                           - args.verification_peak_gib - args.reserve_gib)
    reasons = []
    if args.gpu_memory_cap_gib > worst_probe_headroom:
        reasons.append('Probe cap plus protected verification peak and reserve exceeds GPU capacity')
    if min(s['gpu']['free_gib'] for s in baseline) < args.gpu_memory_cap_gib + args.reserve_gib:
        reasons.append('Observed GPU free memory cannot cover the capped worker and reserve')
    full_path_safe = (min(s['gpu']['free_gib'] for s in baseline)
                      >= args.full_path_generation_peak_gib + args.reserve_gib
                      and worst_probe_headroom >= args.full_path_generation_peak_gib)
    result['full_path_memory_precheck_passed'] = full_path_safe
    if args.probe_kind == 'full-path' and not full_path_safe:
        reasons.append('Full-path 3GPP generation peak is unsafe while verification is running')
    if reasons:
        result.update(status='deferred_resource_precheck', reasons=reasons)
    else:
        worker_result_path = Path(args.probe_result)
        if worker_result_path.exists():
            raise ValueError('Use a fresh --probe-result path; stale attestation must not be reused')
        log = Path(args.output).with_suffix('.worker.log')
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in spec.get('env', {}).items()})
        env['DOPNET_PROBE_GPU_CAP_GIB'] = str(args.gpu_memory_cap_gib)
        env['DOPNET_PROBE_RESULT'] = str(worker_result_path.resolve())
        env['OMP_NUM_THREADS'] = str(args.cpu_threads)
        env['MKL_NUM_THREADS'] = str(args.cpu_threads)
        with log.open('wb') as stream:
            process = subprocess.Popen(spec['command'], cwd=spec.get('cwd'), env=env,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            result['own_probe_pid'] = process.pid
            probe_started = time.monotonic()
            # Drop the interval crossing from baseline into the probe window.
            reader.latest_by_cell.clear()
            aborted = None
            try:
                while process.poll() is None:
                    sample = sample_resources(reader, args.gpu_index)
                    result['samples']['probe'].append(sample)
                    if sample['gpu']['free_gib'] < args.reserve_gib:
                        aborted = 'GPU reserve crossed; only the probe worker was stopped'
                        stop_own_worker(process)
                        break
                    if time.monotonic() - probe_started > args.worker_timeout_seconds:
                        aborted = 'Probe time budget exceeded; only the probe worker was stopped'
                        stop_own_worker(process)
                        break
                    time.sleep(args.sample_seconds)
            finally:
                stop_own_worker(process)
            result['worker_returncode'] = process.returncode
            result['worker_log'] = str(log)
            if worker_result_path.is_file():
                result['worker_result'] = read_json(worker_result_path)
            attestation = result['worker_result'] or {}
            cap_pass = attestation.get('cuda_memory_cap_enforced') is True
            complete = process.returncode == 0 and not aborted and cap_pass
            result['status'] = 'complete' if complete else 'probe_not_qualified'
            result['reasons'] = [aborted] if aborted else ([] if complete else ['Worker did not attest successful memory-capped completion'])
            contract = attestation.get('generation_contract', {})
            result['real_training_data'] = args.probe_kind == 'full-path' and attestation.get('real_training_data') is True
            result['eligible_full_path'] = bool(complete and args.probe_kind == 'full-path' and full_path_safe
                and result['real_training_data'] and attestation.get('optimizer_steps', 0) > 0
                and contract == result['scientific_generation_contract'])
    result['baseline'] = summarize(result['samples']['baseline'])
    result['during_probe'] = summarize(result['samples']['probe'])
    b = result['baseline'].get('per_cell_interval_fps', {})
    p = result['during_probe'].get('per_cell_interval_fps', {})
    common = sorted(set(b) & set(p))
    ratios = {cell: p[cell] / b[cell] for cell in common if b[cell] > 0}
    result['verification_throughput_comparison'] = {
        'matched_cell_ids': common, 'within_same_cell_fps_ratios': ratios,
        'matched_median_ratio': statistics.median(ratios.values()) if ratios else None,
        'scientific_scope': 'Logged 64-frame verification intervals; different conditions are not treated as matched timing samples'}
    if result['eligible_full_path']:
        ratio = result['verification_throughput_comparison']['matched_median_ratio']
        if ratio is None or ratio < args.minimum_verification_fps_ratio:
            result['eligible_full_path'] = False
            result.setdefault('reasons', []).append('Full-path probe lacks matched verification throughput or slows it beyond the limit')
    result['completed_utc'] = utc_now()
    result['decision'] = ('eligible_for_guarded_full_path' if result['eligible_full_path']
                          else 'wait_for_verification_completion_before_real_training')
    result['synthetic_compute_is_not_full_path_evidence'] = args.probe_kind == 'compute-only'
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'probe_kind', 'real_training_data',
        'eligible_full_path', 'decision', 'baseline', 'during_probe', 'verification_throughput_comparison')}, ensure_ascii=False, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verification-output', type=Path, required=True)
    p.add_argument('--verification-pid-file', type=Path, required=True)
    p.add_argument('--verification-log', type=Path)
    p.add_argument('--probe-command-json', type=Path, required=True)
    p.add_argument('--probe-result', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--probe-kind', choices=('compute-only', 'full-path'), default='compute-only')
    p.add_argument('--gpu-index', type=int, default=0)
    p.add_argument('--gpu-memory-cap-gib', type=float, default=6.0)
    p.add_argument('--verification-peak-gib', type=float, default=36.0)
    p.add_argument('--full-path-generation-peak-gib', type=float, default=32.2)
    p.add_argument('--reserve-gib', type=float, default=4.0)
    p.add_argument('--baseline-seconds', type=float, default=60.0)
    p.add_argument('--sample-seconds', type=float, default=5.0)
    p.add_argument('--worker-timeout-seconds', type=float, default=240.0)
    p.add_argument('--minimum-verification-fps-ratio', type=float, default=.9)
    p.add_argument('--cpu-threads', type=int, default=2)
    args = p.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in (args.gpu_memory_cap_gib, args.verification_peak_gib,
        args.full_path_generation_peak_gib, args.reserve_gib, args.baseline_seconds,
        args.sample_seconds, args.worker_timeout_seconds, args.minimum_verification_fps_ratio)):
        p.error('Resource and duration bounds must be finite and positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_probe(args)


if __name__ == '__main__':
    main()
