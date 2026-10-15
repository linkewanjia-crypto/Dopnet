"""Prepare and run an isolated source-guided baseline for exactly 40 epochs.

Training is fixed-budget, with validation-MSE checkpoint selection and no early
stopping. The original paper's 1000-epoch cap remains source metadata only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

if not __package__:
    _package_dir = Path(__file__).resolve().parent
    sys.path[:] = [value for value in sys.path if Path(value or '.').resolve() != _package_dir]
    sys.path.insert(0, str(_package_dir.parent))


def validate_reused_probe(evidence, parent, child, parent_configs, child_configs):
    """Read-only scope/cap/geometry gate; never grants parallel eligibility."""
    from budget_matched_baselines.migrate_prefix import assert_config_transition, PARENT_RECIPE, PARENT_CLAIM
    worker = evidence.get('worker_result') or {}
    if (evidence.get('status') != 'complete' or evidence.get('probe_kind') != 'compute-only'
            or evidence.get('real_training_data') is not False or evidence.get('eligible_full_path') is not False
            or evidence.get('synthetic_compute_is_not_full_path_evidence') is not True
            or worker.get('scope') != 'synthetic_shape_only' or worker.get('cuda_memory_cap_enforced') is not True
            or worker.get('validation_cache_opened') is not False or worker.get('real_waveforms_generated') is not False):
        raise RuntimeError('Historical probe lacks synthetic compute-only/cap evidence; use --skip-resource-probe')
    caps = [worker.get('gpu_cap_gib'), evidence.get('gpu_memory_cap_gib')]
    if (not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0 for value in caps)
            or not math.isclose(caps[0], caps[1], rel_tol=1e-12)):
        raise RuntimeError('Historical worker and outer GPU caps differ; use --skip-resource-probe')
    if (parent.get('recipe') != PARENT_RECIPE or parent.get('claim') != PARENT_CLAIM
            or parent.get('seeds') != [42, 43, 44]
            or parent.get('source_files_sha256') != child.get('source_files_sha256')
            or parent.get('reference_experiment_signature') != child.get('reference_experiment_signature')
            or parent.get('validation_cache_sha256') != child.get('validation_cache_sha256')):
        raise RuntimeError('Historical probe parent differs from the new frozen source plan; use --skip-resource-probe')
    for seed in (42, 43, 44):
        cfg = parent_configs[seed]
        assert_config_transition(cfg, child_configs[seed])
        generation = dict(generation_batch_size=cfg['data_pipeline']['generation_batch_size'],
                          microbatch_size=cfg['data_pipeline']['microbatch_size'],
                          threegpp_link_batch_size=cfg['system']['threegpp_link_batch_size'])
        if (generation != dict(generation_batch_size=32, microbatch_size=4, threegpp_link_batch_size=8)
                or evidence.get('scientific_generation_contract') != generation):
            raise RuntimeError('Historical probe generation geometry differs; use --skip-resource-probe')
    return caps[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--reference-results', type=Path, required=True)
    p.add_argument('--validation-cache', type=Path, required=True)
    p.add_argument('--verification-code-root', type=Path, required=True)
    p.add_argument('--verification-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline-utc', required=True,
                   help='Explicit authorized deadline with UTC offset; this launcher never extends it')
    probe_options = p.add_mutually_exclusive_group()
    probe_options.add_argument('--reuse-probe', type=Path,
                               help='Read-only historical synthetic/compute-only evidence; never qualifies parallel training')
    probe_options.add_argument('--skip-resource-probe', action='store_true',
                               help='Use the serial queue without another resource probe (default)')
    a = p.parse_args()
    try:
        authorized_deadline = datetime.fromisoformat(a.deadline_utc.replace('Z', '+00:00'))
    except ValueError:
        p.error('--deadline-utc must be an ISO-8601 timestamp with UTC offset')
    if authorized_deadline.tzinfo is None:
        p.error('--deadline-utc must include a UTC offset')
    deadline = authorized_deadline.astimezone(timezone.utc).isoformat()
    root = Path(__file__).resolve().parents[1]
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join((str(root), str(a.verification_code_root.resolve())))
    env['OMP_NUM_THREADS'] = '4'
    env['MKL_NUM_THREADS'] = '4'
    def execute(module, arguments):
        command = [sys.executable, '-u', '-m', 'budget_matched_baselines.' + module, *map(str, arguments)]
        subprocess.run(command, cwd=root, env=env, check=True)
    plan = output / 'recipe_manifest.json'
    execute('prepare', ['--project', a.project, '--reference-results', a.reference_results,
                        '--validation-cache', a.validation_cache, '--output', output])
    budget_path = output / 'execution_budget.json'
    if budget_path.exists():
        recorded = datetime.fromisoformat(json.loads(budget_path.read_text())['deadline_utc'].replace('Z', '+00:00'))
        if recorded.tzinfo is None or recorded != authorized_deadline:
            raise RuntimeError('Explicit deadline differs from the recorded budget; no automatic budget extension')
    else:
        budget_path.write_text(json.dumps({'deadline_utc': deadline, 'authorization': 'explicit_command_deadline',
                                          'includes': 'verification wait, three new training runs, scoring and reports'}, indent=2))
    if a.reuse_probe:
        from budget_matched_baselines.recipe import signed_metadata, digest, file_hash
        import yaml
        source = a.reuse_probe.resolve()
        evidence_bytes = source.read_bytes()
        evidence = json.loads(evidence_bytes)
        parent_plan = source.parent / 'recipe_manifest.json'
        parent = signed_metadata(parent_plan)
        child = signed_metadata(plan)
        parent_configs, child_configs, parent_config_hashes = {}, {}, {}
        snapshots = {'parent_recipe_manifest.json': parent_plan.read_bytes()}
        for seed in (42, 43, 44):
            for manifest, folder, label, configs in (
                    (parent, source.parent, f'fortitran_author_recipe_seed{seed}', parent_configs),
                    (child, output, f'fortitran_budget_matched_seed{seed}', child_configs)):
                config_record = manifest['configs'][label]
                relative = Path(config_record['file'])
                if relative.is_absolute() or '..' in relative.parts:
                    raise RuntimeError('Unsafe historical configuration path; use --skip-resource-probe')
                path = folder / relative
                data = path.read_bytes()
                cfg = yaml.safe_load(data)
                if hashlib.sha256(data).hexdigest() != config_record['sha256'] or digest(cfg) != config_record['config_sha256']:
                    raise RuntimeError('Historical/new configuration SHA differs; use --skip-resource-probe')
                configs[seed] = cfg
                if manifest is parent:
                    parent_config_hashes[label] = config_record['sha256']
                    snapshots[f'parent_configs/{label}.yaml'] = data
        cap = validate_reused_probe(evidence, parent, child, parent_configs, child_configs)
        record = {'source_path': str(source), 'source_sha256': hashlib.sha256(evidence_bytes).hexdigest(),
                  'probe_kind': 'compute-only', 'eligible_full_path': False,
                  'source_manifest_signature': parent['signature'], 'source_manifest_sha256': file_hash(parent_plan),
                  'source_config_sha256': parent_config_hashes, 'child_manifest_signature': child['signature'],
                  'gpu_memory_cap_gib': cap, 'scientific_generation_contract': evidence['scientific_generation_contract'],
                  'use': 'historical_resource_evidence_only; serial_queue_required'}
        record['signature'] = digest(record)
        record_path = output / 'reused_resource_probe.json'
        if record_path.exists() and json.loads(record_path.read_text()) != record:
            raise RuntimeError('Historical resource evidence changed; its existing reference is immutable')
        snapshot_path = output / 'reused_resource_probe_snapshot.json'
        if snapshot_path.exists() and snapshot_path.read_bytes() != evidence_bytes:
            raise RuntimeError('Historical resource snapshot changed')
        record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        snapshot_path.write_bytes(evidence_bytes)
        for relative, data in snapshots.items():
            snapshot = output / 'reused_resource_probe_evidence' / relative
            if snapshot.exists() and snapshot.read_bytes() != data:
                raise RuntimeError('Historical probe source snapshot changed')
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(data)
    # No additional resource worker is launched. Synthetic historical evidence
    # is retained for audit only; real training always waits for verification.
    tasks = []
    for seed in (42, 43, 44):
        folder = output / 'checkpoints' / f'fortitran_budget_matched_seed{seed}'
        command = [sys.executable, '-u', '-m', 'budget_matched_baselines.train_author',
                   '--project', str(a.project), '--manifest', str(plan), '--seed', str(seed),
                   '--resume', 'auto', '--deadline-utc', deadline]
        tasks.append({'name': f'fortitran_budget_matched_seed{seed}', 'command': command, 'cwd': str(root),
                      'env': {'PYTHONPATH': env['PYTHONPATH'], 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'},
                      'expected_outputs': [str(folder / 'complete.json'), str(folder / 'model.pt')],
                      'expected_json': [{'file': str(folder / 'complete.json'), 'fields': {
                          'status': 'complete', 'training_complete': True,
                          'completed_epoch': 40, 'max_epochs': 40,
                          'termination': 'fixed_budget_40_epochs', 'claim': 'source_guided_equal_budget'}}]})
    scores = output / 'evaluation'
    tasks.append({'name': 'frozen_budget_matched_scoring', 'cwd': str(root), 'env': {'PYTHONPATH': env['PYTHONPATH']},
                  'command': [sys.executable, '-u', '-m', 'budget_matched_baselines.evaluate_author',
                              '--project', str(a.project), '--reference-results', str(a.reference_results),
                              '--training-output', str(output), '--output', str(scores)],
                  'expected_outputs': [str(scores / 'COMPLETE.json'), str(scores / 'summary/statistics.json')],
                  'expected_json': [{'file': str(scores / 'COMPLETE.json'), 'fields': {'status': 'complete', 'cells': 269}}]})
    tasks.append({'name': 'budget_matched_comparison_report', 'cwd': str(root), 'env': {'PYTHONPATH': env['PYTHONPATH']},
                  'command': [sys.executable, '-u', '-m', 'budget_matched_baselines.report_author',
                              '--output', str(scores), '--training-output', str(output)],
                  'expected_outputs': [str(scores / 'budget_matched_results.zip'), str(scores / 'budget_matched_results.zip.sha256')]})
    task_path = output / 'training_tasks.json'
    if task_path.exists() and json.loads(task_path.read_text()) != tasks:
        raise RuntimeError('Existing queue differs; do not overwrite its tasks')
    task_path.write_text(json.dumps(tasks, indent=2))
    execute('queue', ['--tasks', task_path, '--verification-output', a.verification_output,
                      '--verification-pid-file', a.verification_output / 'fast_controller.pid',
                      '--verification-code-root', a.verification_code_root, '--reference-results', a.reference_results,
                      '--output', output / 'queue', '--deadline-utc', deadline])


if __name__ == '__main__':
    main()
