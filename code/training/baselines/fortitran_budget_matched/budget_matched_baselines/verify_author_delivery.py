"""Read-only integrity gate for the fixed 40-epoch equal-budget result ZIP.

No simulator, model, torch import, tensor deserialization, or statistical
recomputation is performed. Checkpoint scalar metadata is read by a restricted
unpickler whose tensor/storage objects are replaced by inert placeholders.
Unknown pickle globals fail closed. Run only after the full result ZIP arrives.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import pickle
import re
import sys
from zipfile import ZipFile, BadZipFile

if not __package__:
    _package_dir = Path(__file__).resolve().parent
    sys.path[:] = [value for value in sys.path if Path(value or '.').resolve() != _package_dir]
    sys.path.insert(0, str(_package_dir.parent))

import numpy as np

SEEDS = (42, 43, 44)
METHODS = ('dopnet', 'fortitran', 'fortitran_budget_matched')
VALIDATION_SHA = '83d0e03524628b617671c42a73d3c458cc3395c31965a4e22295e8389f56e597'
RECIPE = 'fortitran_source_guided_equal_budget_v1'
CLAIM = 'source_guided_equal_budget'
MAX_EPOCHS = 40
TERMINATION = 'fixed_budget_40_epochs'
TRACKER_PATIENCE = 41
HEX = re.compile(r'^[0-9a-f]{64}$')
SUMMARY_FILES = ('cell_summary.csv', 'cell_paired_differences.csv',
                 'macro_summary.csv', 'macro_paired_differences.csv',
                 'receiver_diagnostics.csv')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signed(obj, label):
    require(obj.get('signature') == digest({k: v for k, v in obj.items() if k != 'signature'}),
            'Invalid signed metadata: ' + label)
    return obj


def safe_name(name):
    p = PurePosixPath(name)
    return (bool(name) and not p.is_absolute() and '..' not in p.parts
            and '\\' not in name and ':' not in name and p.as_posix() == name)


class _Opaque:
    """Inert stand-in; never opens storage or materializes a tensor/ndarray."""
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


def _opaque(*args, **kwargs):
    return _Opaque()


def _safe_latin1_encode(value, encoding='latin1', errors='strict'):
    """Decode protocol-2 pickle byte literals without opening arbitrary codecs."""
    require(type(value) is str and len(value) <= 32 * 1024 * 1024,
            'Checkpoint byte literal exceeds bound or is not a string')
    require(type(encoding) is str and encoding.lower() in ('latin1', 'latin-1', 'latin_1', 'iso-8859-1'),
            'Checkpoint byte literal requires Latin-1')
    require(errors == 'strict', 'Checkpoint byte literal requires strict encoding')
    return value.encode('latin-1', 'strict')


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('_codecs', 'encode'):
            return _safe_latin1_encode
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        if (module, name) in (('builtins', 'set'), ('__builtin__', 'set')):
            return set
        if module == 'torch' and name in {
                'FloatStorage', 'DoubleStorage', 'HalfStorage', 'BFloat16Storage',
                'ByteStorage', 'CharStorage', 'ShortStorage', 'IntStorage',
                'LongStorage', 'BoolStorage', 'ComplexFloatStorage',
                'ComplexDoubleStorage', 'UntypedStorage'}:
            return _Opaque
        if module == 'torch._utils' and name in {
                '_rebuild_tensor', '_rebuild_tensor_v2', '_rebuild_tensor_v3',
                '_rebuild_parameter', '_rebuild_parameter_with_state'}:
            return _opaque
        if module in ('numpy.core.multiarray', 'numpy._core.multiarray') and name in ('_reconstruct', 'scalar'):
            return _opaque
        if module == 'numpy' and name in ('dtype', 'ndarray'):
            return _Opaque
        raise ValueError('Blocked checkpoint pickle global: ' + module + '.' + name)

    def persistent_load(self, pid):
        require(isinstance(pid, tuple) and len(pid) == 5 and pid[0] == 'storage',
                'Unknown checkpoint persistent object')
        return _Opaque()


def checkpoint_metadata(data, label):
    with ZipFile(io.BytesIO(data)) as archive:
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
        require(len(names) == len(set(names)) and all(safe_name(n) for n in names),
                'Unsafe/duplicate checkpoint ZIP entries: ' + label)
        require(archive.testzip() is None, 'Checkpoint ZIP CRC failed: ' + label)
        candidates = [n for n in names if n.endswith('/data.pkl') or n == 'data.pkl']
        require(len(candidates) == 1, 'Checkpoint needs one data.pkl: ' + label)
        info = archive.getinfo(candidates[0])
        require(info.file_size <= 32 * 1024 * 1024, 'Checkpoint metadata exceeds bound: ' + label)
        obj = MetadataUnpickler(io.BytesIO(archive.read(info))).load()
    require(isinstance(obj, dict), 'Checkpoint metadata must be a dictionary: ' + label)
    return obj


def scalar(z, key):
    value = z[key]
    require(value.shape == (), 'NPZ metadata must be scalar: ' + key)
    return str(value.item())


def manuscript_cells(cells):
    dynamic = {'doubly_selective', 'tdl_b', 'tdl_c', 'tdl_d', 'cdl_b', 'cdl_c',
               'cdl_d', 'tdl_a', 'tdl_e', 'cdl_a', 'cdl_e'}
    result = []
    for m in cells:
        nominal = m['cp_len'] == 16 and m['pilot_spacing'] == 4 and m['delay_spread_ns'] == 300.
        main = nominal and m['modulation'] == 'qpsk' and 'nmse' in m['studies']
        core = (nominal and m['channel'] in dynamic and m['doppler_hz'] in (800., 1200., 1600.)
                and ((m['modulation'] == 'qpsk' and m['snr_db'] in (15., 25.) and 'nmse' in m['studies'])
                     or (m['modulation'] == '16qam' and m['snr_db'] == 20. and 'receiver' in m['studies'])))
        if main or core:
            result.append(m)
    require(len(result) == 269, 'Reference manuscript suite must contain all 269 cells')
    return result


def number(value, label, nonnegative=False):
    value = float(value)
    require(math.isfinite(value) and (not nonnegative or value >= 0), 'Invalid numeric metadata: ' + label)
    return value


def check_budget_marker(marker, seed, recipe_signature=None):
    """A former early-stop marker cannot attest fixed-budget completion."""
    require(marker.get('status') == 'complete' and marker.get('training_complete') is True
            and marker.get('seed') == seed and marker.get('claim') == CLAIM
            and type(marker.get('completed_epoch')) is int and marker['completed_epoch'] == MAX_EPOCHS
            and marker.get('max_epochs') == MAX_EPOCHS and marker.get('termination') == TERMINATION
            and marker.get('early_stopping_enabled') is False
            and marker.get('reason') in (TERMINATION, 'authenticated_terminal_checkpoint')
            and marker.get('exact_original_dataset_reproduction') is False,
            'Training requires exactly 40 complete epochs with fixed-budget termination: seed' + str(seed))
    if recipe_signature is not None:
        require(marker.get('recipe_signature') == recipe_signature, 'Completion belongs to another budget plan')


def check_config(cfg, baseline, seed):
    require(cfg.get('seed') == seed and cfg.get('device') == 'cuda', 'Equal-budget config seed/device differs')
    keys = set(cfg) - {'train', 'seed', 'device', 'author_recipe'}
    require(keys == set(baseline) - {'train', 'seed', 'device', 'author_recipe'}
            and all(cfg[k] == baseline[k] for k in keys), 'Physical/model/receiver sampling config changed')
    expected = copy.deepcopy(baseline['train'])
    micro = cfg['train'].get('gradient_microbatch_size')
    require(type(micro) is int and 1 <= micro <= 32, 'Gradient microbatch differs')
    expected.update(task='ce', batch_size=512, gradient_microbatch_size=micro,
                    validation_microbatch_size=micro, epochs=MAX_EPOCHS, optimizer='adam',
                    lr=.001, weight_decay=0., betas=[.9, .999], eps=1e-8, amsgrad=False,
                    scheduler='exponential', scheduler_gamma=.995, warmup_epochs=0,
                    min_lr=0., baseline_loss_mode='paper', nmse_weight=0.,
                    symbol_loss_weight=0., grad_clip=None, amp='none', tf32=False,
                    cudnn_benchmark=False, fused_optimizer=False, selection_metric='val_mse',
                    early_stopping_patience=TRACKER_PATIENCE, early_stopping_min_delta=0.)
    require(cfg['train'] == expected, 'Equal-budget training configuration differs from declared recipe')
    claim = cfg['author_recipe']
    require(claim.get('identifier') == RECIPE and claim.get('claim') == CLAIM
            and claim.get('exact_original_dataset_reproduction') is False
            and claim.get('optimizer_batch_frames') == 512
            and claim.get('precision') == 'float32; TF32 disabled'
            and claim.get('early_stopping_enabled') is False
            and claim.get('termination') == TERMINATION and claim.get('max_epochs') == MAX_EPOCHS
            and claim.get('original_max_epochs') == 1000
            and claim.get('original_early_stopping') is True
            and claim.get('original_repository_patience') == 3, 'Unsupported equal-budget claim')


def check_history(rows, model, last, marker, label):
    require(len(rows) == MAX_EPOCHS, 'Equal-budget history must contain exactly 40 completed epochs: ' + label)
    best = None
    stale = 0
    selected = None
    for epoch, row in enumerate(rows, 1):
        require(int(row['epoch']) == epoch and row['selection_metric'] == 'val_mse'
                and int(row['train_frames']) == 12000 and int(row['optimizer_steps']) == 24,
                'Incomplete epoch or wrong selection metric: ' + label)
        lr = number(row['lr'], label + '.lr')
        next_lr = number(row['next_lr'], label + '.next_lr')
        require(math.isclose(lr, .001 * .995 ** (epoch - 1), rel_tol=1e-10, abs_tol=1e-15)
                and math.isclose(next_lr, lr * .995, rel_tol=1e-10, abs_tol=1e-15),
                'Learning-rate trajectory differs: ' + label)
        val = number(row['val_loss'], label + '.val_loss', True)
        number(row['train_loss'], label + '.train_loss', True)
        number(row['val_nmse_db'], label + '.val_nmse_db')
        if best is None or val < best:
            best, stale, selected = val, 0, epoch
        else:
            stale += 1
        require(int(row['stale_epochs']) == stale, 'MSE-selection tracker history differs: ' + label)
    require(marker['completed_epoch'] == len(rows) and marker['selected_epoch'] == selected
            and marker['best_val_mse'] == best and marker['max_epochs'] == MAX_EPOCHS
            and marker.get('termination') == TERMINATION,
            'Completion marker disagrees with training history: ' + label)
    require(model['epoch'] == selected and last['epoch'] == len(rows)
            and model['best_val_loss'] == best and last['best_val_loss'] == best
            and model['early_stopping'] == {'patience': TRACKER_PATIENCE, 'best': best, 'stale': 0}
            and last['training_complete'] is True
            and last.get('termination') == TERMINATION
            and last['early_stopping'] == {'patience': TRACKER_PATIENCE, 'best': best, 'stale': stale},
            'Checkpoint MSE selection/terminal state differs: ' + label)
    for state, expected in ((model, rows[:selected]), (last, rows)):
        stored = state['history']
        require(len(stored) == len(expected), 'Checkpoint history length differs: ' + label)
        for obj, csv_row in zip(stored, expected):
            require(set(obj) == set(csv_row), 'Checkpoint history columns differ: ' + label)
            for key, value in obj.items():
                require((value == csv_row[key]) if isinstance(value, str)
                        else (float(value) == float(csv_row[key])), 'Checkpoint history differs: ' + label + '.' + key)


def verify_prefix_provenance(archive, hashes, recipe, cfg, states, rows, seed):
    """Verify immutable parent/migrated snapshots without loading any tensors."""
    from budget_matched_baselines.migrate_prefix import (
        PARENT_RECIPE, PARENT_CLAIM, PARENT_IMPLEMENTATION, PRESERVED_STATE,
        assert_config_transition, check_committed_prefix,
    )
    import yaml
    relative = f'prefix_provenance/seed{seed}/provenance.json'
    base = f'training/prefix_provenance/seed{seed}/'
    references = [state.get('prefix_provenance') for state in states]
    if not any(reference is not None for reference in references):
        require(not any(name.startswith(base) for name in hashes), 'Orphan prefix provenance: seed' + str(seed))
        return False
    require(all(isinstance(reference, dict) for reference in references), 'Partial checkpoint prefix binding')
    read_json = lambda name: json.loads(archive.read(base + name).decode('utf-8-sig'))
    proof = signed(read_json('provenance.json'), 'prefix provenance')
    receipt = signed(read_json('migration_receipt.json'), 'migration receipt')
    parent = signed(read_json('parent_recipe_manifest.json'), 'parent training plan')
    require(proof.get('schema') == 'fortitran_equal_budget_prefix_provenance_v1'
            and proof.get('status') == 'authenticated_prefix' and proof.get('seed') == seed
            and proof.get('claim') == CLAIM and proof.get('termination') == TERMINATION
            and proof.get('child_manifest_signature') == recipe['signature']
            and proof.get('child_manifest_sha256') == hashes['training/recipe_manifest.json']
            and proof.get('parent_manifest_signature') == parent['signature']
            and proof.get('parent_manifest_sha256') == hashes[base + 'parent_recipe_manifest.json']
            and proof.get('state_preservation') == list(PRESERVED_STATE), 'Prefix plan/source binding differs')
    require(parent.get('recipe') == PARENT_RECIPE and parent.get('claim') == PARENT_CLAIM
            and parent.get('seeds') == list(SEEDS)
            and parent.get('reference_experiment_signature') == recipe['reference_experiment_signature']
            and parent.get('reference_evaluation_signature') == recipe['reference_evaluation_signature']
            and parent.get('validation_cache_sha256') == VALIDATION_SHA
            and all(parent.get('validation', {}).get(key) == recipe['validation'].get(key)
                    for key in ('sha256', 'bytes', 'frames', 'generation_seed', 'usage'))
            and parent.get('source_files_sha256') == recipe['source_files_sha256']
            and parent.get('reference_runtime_code_sha256') == recipe['reference_runtime_code_sha256'],
            'Parent is not the same frozen source/validation experiment')
    engines = proof['parent_implementation_sha256']
    runtime = proof['parent_runtime_files_sha256']
    require(engines == parent['implementation_sha256'] and set(engines) == set(PARENT_IMPLEMENTATION)
            and runtime == parent['source_files_sha256'], 'Parent implementation/runtime declarations differ')
    for folder, mapping in (('parent_engine', engines), ('parent_runtime', runtime)):
        expected_names = {base + folder + '/' + name for name in mapping}
        require({name for name in hashes if name.startswith(base + folder + '/')} == expected_names,
                'Parent source snapshot file coverage differs: ' + folder)
        for name, sha in mapping.items():
            require(safe_name(name) and hashes[base + folder + '/' + name] == sha,
                    'Parent frozen source SHA differs: ' + name)
    parent_record = parent['configs'][f'fortitran_author_recipe_seed{seed}']
    require(proof['parent_config_sha256'] == parent_record['sha256'] == hashes[base + 'parent_config.yaml'],
            'Parent YAML SHA differs')
    parent_cfg = yaml.safe_load(archive.read(base + 'parent_config.yaml'))
    require(digest(parent_cfg) == parent_record['config_sha256'], 'Parent YAML digest differs')
    assert_config_transition(parent_cfg, cfg)
    parent_files = proof['parent_checkpoint_files_sha256']
    required = {'model.pt', 'last.pt', 'history.csv', 'model.pt.sha256.json', 'last.pt.sha256.json'}
    require(set(parent_files) == required, 'Parent checkpoint snapshot is incomplete')
    for name, sha in parent_files.items():
        require(hashes[base + 'parent_checkpoints/' + name] == sha, 'Parent checkpoint snapshot SHA differs: ' + name)
    parent_states = []
    for basename in ('model.pt', 'last.pt'):
        parent_data = archive.read(base + 'parent_checkpoints/' + basename)
        state = checkpoint_metadata(parent_data, 'parent.' + basename)
        sidecar = read_json('parent_checkpoints/' + basename + '.sha256.json')
        require(sidecar['sha256'] == parent_files[basename] and sidecar['bytes'] == len(parent_data)
                and sidecar['recipe_signature'] == parent['signature'] and sidecar['epoch'] == state['epoch'],
                'Parent checkpoint sidecar differs: ' + basename)
        require(state['cfg'] == parent_cfg and state['recipe_signature'] == parent['signature']
                and state.get('recipe_claim') == PARENT_CLAIM
                and state.get('source_files_sha256') == runtime and state.get('implementation_sha256') == engines
                and state.get('selection_metric') == 'val_mse' and state.get('validation_cache_sha256') == VALIDATION_SHA
                and state.get('optimizer_name') == 'adam' and state.get('scheduler_name') == 'exponential'
                and state.get('effective_batch_frames') == 512
                and state.get('physical_gradient_microbatch_frames') == cfg['train']['gradient_microbatch_size']
                and all(key in state for key in PRESERVED_STATE)
                and isinstance(state.get('model'), dict) and bool(state['model'])
                and isinstance(state.get('optimizer'), dict) and isinstance(state.get('scheduler'), dict),
                'Parent checkpoint resume contract differs: ' + basename)
        require(state['scheduler'].get('gamma') == .995 and state['scheduler'].get('last_epoch') == state['epoch'],
                'Parent scheduler state differs')
        groups = state['optimizer'].get('param_groups', [])
        require(bool(groups) and all(math.isclose(group['lr'], .001 * .995 ** state['epoch'],
                    rel_tol=1e-10, abs_tol=1e-15) for group in groups), 'Parent optimizer next LR differs')
        parent_states.append(state)
        data = archive.read(base + 'migrated_checkpoints/' + basename)
        migrated = checkpoint_metadata(data, 'migration.' + basename)
        new_sidecar = read_json('migrated_checkpoints/' + basename + '.sha256.json')
        require(new_sidecar['sha256'] == hashes[base + 'migrated_checkpoints/' + basename]
                and new_sidecar['bytes'] == len(data) and new_sidecar['recipe_signature'] == recipe['signature']
                and new_sidecar['epoch'] == state['epoch'], 'Migrated snapshot sidecar differs')
        require(migrated['cfg'] == cfg and migrated['recipe_signature'] == recipe['signature']
                and migrated.get('recipe_claim') == CLAIM and migrated.get('termination') == TERMINATION
                and migrated.get('implementation_sha256') == recipe['implementation_sha256']
                and migrated.get('source_files_sha256') == runtime and migrated['epoch'] == state['epoch']
                and migrated['best_val_loss'] == state['best_val_loss'] and migrated['history'] == state['history']
                and migrated['early_stopping'] == {**state['early_stopping'], 'patience': TRACKER_PATIENCE}
                and migrated.get('training_complete') is (state['epoch'] == MAX_EPOCHS)
                and migrated.get('prefix_provenance') == dict(signature=proof['signature'],
                    parent_checkpoint_sha256=parent_files[basename], relative_path=relative),
                'Migrated metadata transition differs: ' + basename)
    parent_rows = list(csv.DictReader(io.StringIO(archive.read(base + 'parent_checkpoints/history.csv').decode('utf-8-sig'))))
    prefix = check_committed_prefix(parent_rows, parent_states[0], parent_states[1])
    require(all(proof.get(key) == value for key, value in prefix.items()), 'Proof history/selection summary differs')
    require(receipt.get('status') == 'migrated' and receipt.get('seed') == seed
            and receipt.get('run') == f'fortitran_budget_matched_seed{seed}'
            and receipt.get('prefix_provenance_signature') == proof['signature']
            and receipt.get('child_manifest_signature') == recipe['signature']
            and receipt.get('parent_checkpoint_files_sha256') == parent_files
            and receipt.get('completed_epoch') == prefix['completed_epoch']
            and receipt.get('remaining_epochs') == MAX_EPOCHS - prefix['completed_epoch']
            and receipt.get('old_run_unchanged') is True and receipt.get('state_equality_checked') == list(PRESERVED_STATE),
            'Signed migration receipt differs')
    migrated_files = receipt['new_checkpoint_files_sha256']
    require(set(migrated_files) == {'model.pt', 'last.pt', 'history.csv'}, 'Migrated receipt file coverage differs')
    for name, sha in migrated_files.items():
        require(hashes[base + 'migrated_checkpoints/' + name] == sha, 'Initial migrated snapshot SHA differs: ' + name)
    require(archive.read(base + 'migrated_checkpoints/history.csv') == archive.read(base + 'parent_checkpoints/history.csv'),
            'Initial migrated history bytes changed')
    for obj, row in zip(parent_states[1]['history'], rows[:prefix['completed_epoch']]):
        require(set(obj) == set(row), 'Final history prefix columns differ')
        for key, value in obj.items():
            require(value == row[key] if isinstance(value, str) else float(value) == float(row[key]),
                    'Final history differs from its authenticated prefix: ' + key)
    for index, (state, reference) in enumerate(zip(states, references)):
        parent_key = 'model.pt' if index == 0 and state['epoch'] <= prefix['completed_epoch'] else 'last.pt'
        require(reference == dict(signature=proof['signature'], parent_checkpoint_sha256=parent_files[parent_key],
                                  relative_path=relative), 'Final checkpoint lost its inherited prefix reference')
    return True


def verify(archive, reference_results, implementation_root=None):
    archive, reference = Path(archive), Path(reference_results)
    if reference.name == 'cells':
        reference = reference.parent
    implementation = Path(implementation_root or Path(__file__).parent)
    read_local = lambda n: json.loads((reference / n).read_text(encoding='utf-8-sig'))
    experiment = signed(read_local('experiment.json'), 'local experiment')
    old_ev = signed(read_local('evaluation.json'), 'local evaluation')
    require(old_ev.get('experiment_signature') == experiment['signature'], 'Reference signatures differ')
    require(len(old_ev['cells']) == 636 and len({m['cell_id'] for m in old_ev['cells']}) == 636,
            'Reference requires all 636 original cells')
    expected = manuscript_cells(old_ev['cells'])
    old_indices = [i for i, r in enumerate(old_ev['routes']) if r['method'] in ('dopnet', 'fortitran')]
    old_routes = [old_ev['routes'][i] for i in old_indices]
    for method in METHODS[:2]:
        require([r['training_seed'] for r in old_routes if r['method'] == method] == list(SEEDS),
                'Historical routes have missing/duplicate seeds')
    expected_routes = old_routes + [dict(method='fortitran_budget_matched', training_seed=s, trained=True) for s in SEEDS]
    fingerprints = {}
    # These are existing score/identity reads, never waveform generation.
    for cell in old_ev['cells']:
        cid = cell['cell_id']
        with np.load(reference / 'cells' / (cid + '.npz'), allow_pickle=False) as old:
            require(scalar(old, 'signature') == old_ev['signature']
                    and json.loads(scalar(old, 'meta')) == cell, 'Original cell identity differs: ' + cid)
            require(old['nmse'].shape == (len(old_ev['routes']), 512)
                    and np.array_equal(old['cluster_id'], np.arange(512) // 8), 'Original score/cluster layout differs: ' + cid)
            fingerprint = scalar(old, 'realization_sha256')
            require(bool(HEX.fullmatch(fingerprint)), 'Invalid waveform SHA: ' + cid)
            fingerprints[cid] = fingerprint
    with ZipFile(archive) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        require(len(names) == len(set(names)) and all(safe_name(n) for n in names), 'ZIP has unsafe/duplicate entries')
        require(z.testzip() is None, 'Delivery ZIP CRC failed')
        read_json = lambda n: json.loads(z.read(n).decode('utf-8-sig'))
        manifest = read_json('DELIVERY_MANIFEST.json')
        hashes = manifest['files']
        require(manifest.get('schema') == 1 and manifest.get('status') == 'complete'
                and manifest.get('training_claim') == CLAIM
                and manifest.get('training_termination') == TERMINATION
                and manifest.get('training_max_epochs') == MAX_EPOCHS
                and set(hashes) == set(names) - {'DELIVERY_MANIFEST.json'}, 'Incomplete delivery manifest file coverage')
        for name, sha in hashes.items():
            require(HEX.fullmatch(sha) is not None and hashlib.sha256(z.read(name)).hexdigest() == sha,
                    'Internal SHA mismatch: ' + name)
        ev = signed(read_json('evaluation/evaluation.json'), 'equal-budget evaluation')
        stats = signed(read_json('evaluation/summary/statistics.json'), 'equal-budget statistics')
        recipe = signed(read_json('training/recipe_manifest.json'), 'training recipe')
        require(ev['suite'] == 'manuscript' and ev['cells'] == expected
                and ev['routes'] == expected_routes and len(ev['routes']) == 9,
                'Evaluation must contain the unchanged 269-cell suite and all nine routes')
        require(ev['samples_per_cell'] == 512 and ev['batch_size'] == 8
                and ev['test_seed'] == old_ev['test_seed']
                and ev['receiver_protocols'] == ['common_diagonal', 'native_own_ici_else_diagonal'], 'Frozen test protocol differs')
        require(ev['reference_evaluation_signature'] == old_ev['signature']
                and ev['reference_experiment_signature'] == experiment['signature']
                and recipe['reference_evaluation_signature'] == old_ev['signature']
                and recipe['reference_experiment_signature'] == experiment['signature'], 'Local/server reference binding differs')
        require(recipe['recipe'] == RECIPE and recipe['seeds'] == list(SEEDS)
                and recipe['claim'] == CLAIM
                and recipe['validation_cache_sha256'] == recipe['validation']['sha256'] == VALIDATION_SHA
                and recipe['validation']['frames'] == 2000 and recipe['validation']['generation_seed'] == 610000001
                and recipe['reference_runtime_code_sha256'] == experiment['code_sha256']
                and digest(recipe['source_files_sha256']) == experiment['code_sha256'], 'Training recipe provenance differs')
        require(recipe.get('paper_settings', {}).get('max_epochs') == 1000
                and recipe['paper_settings'].get('early_stopping') is True,
                'Original paper settings must remain source metadata, distinct from the 40-epoch budget')
        require(recipe.get('termination') == TERMINATION and recipe.get('max_epochs') == MAX_EPOCHS
                and recipe.get('early_stopping_enabled') is False
                and recipe.get('comparison_budget') == dict(fresh_frames_per_epoch=12000, epochs=MAX_EPOCHS,
                                                            total_fresh_training_frames=480000),
                'Fixed-budget plan termination/observation budget differs')
        require(ev.get('training_claim') == CLAIM and ev.get('training_termination') == TERMINATION
                and ev.get('training_max_epochs') == MAX_EPOCHS, 'Evaluation budget claim differs')
        validation = read_json('training/historical_validation_attestation.json')
        require(recipe['validation']['bytes'] == 4682930065 and validation.get('status') == 'passed'
                and validation.get('cache_sha256') == VALIDATION_SHA
                and validation.get('cache_size_bytes') == 4682930065
                and validation.get('use') == 'checkpoint_selection_only; never training optimizer input',
                'Historical validation attestation differs')
        require(ev['training_recipe_sha256'] == hashes['training/recipe_manifest.json'], 'Evaluation recipe SHA differs')
        for name, sha in recipe['implementation_sha256'].items():
            require(name in ('recipe.py', 'prepare.py', 'train_author.py', 'migrate_prefix.py') and file_hash(implementation / name) == sha,
                    'Local equal-budget engine differs from training provenance: ' + name)
        require(set(recipe['implementation_sha256']) == {'recipe.py', 'prepare.py', 'train_author.py', 'migrate_prefix.py'}, 'Training engine hashes missing')
        require(file_hash(implementation / 'evaluate_author.py') == ev['evaluator_sha256'], 'Local evaluator differs')
        waveform_keys = ['input', 'add_input', 'add_input_slow', 'rx_data', 'target_h', 'tx_label']
        provenance = dict(reference_evaluation_signature=old_ev['signature'], reference_experiment_signature=experiment['signature'],
                          reference_realizations_sha256=digest(dict(evaluation_signature=old_ev['signature'],
                              experiment_signature=experiment['signature'], waveform_keys=tuple(waveform_keys), realization_sha256=fingerprints)),
                          reference_cells=636, reference_samples_per_cell=512, reference_batch_size=8, reference_waveform_keys=waveform_keys)
        require(ev['fixed_test_reference'] == provenance, 'Frozen waveform provenance differs')
        complete = read_json('evaluation/COMPLETE.json')
        require(complete == dict(status='complete', signature=ev['signature'], cells=269, checkpoint_sha256=ev['checkpoint_sha256'])
                and read_json('evaluation/evaluation_complete.json') == dict(signature=ev['signature'], cells=269)
                and manifest['evaluation_signature'] == ev['signature'], 'Evaluation completion binding differs')
        require(stats['evaluation_signature'] == ev['signature'] and stats['training_seeds'] == list(SEEDS)
                and stats['bootstrap'] >= 2000 and stats['data_binding_version'] == 1
                and math.isclose(stats['t_critical_df2'], 4.302652729696142, rel_tol=1e-10), 'Statistics protocol/signature binding differs')
        ids = {c['cell_id'] for c in expected}
        delivered_ids = {n.removeprefix('evaluation/cells/').removesuffix('.npz') for n in hashes
                         if n.startswith('evaluation/cells/') and n.endswith('.npz')}
        marker_ids = {n.removeprefix('evaluation/cells/').removesuffix('.COMPLETE.json') for n in hashes
                      if n.startswith('evaluation/cells/') and n.endswith('.COMPLETE.json')}
        require(ids == delivered_ids == marker_ids == set(stats['cell_file_sha256']), 'Score/marker/statistics cell set differs')
        for cell in expected:
            cid = cell['cell_id']
            name = 'evaluation/cells/' + cid + '.npz'
            source_sha = file_hash(reference / 'cells' / (cid + '.npz'))
            done = read_json('evaluation/cells/' + cid + '.COMPLETE.json')
            require(done == dict(signature=ev['signature'], source_score_sha256=source_sha,
                                 score_sha256=hashes[name], realization_sha256=fingerprints[cid])
                    and stats['cell_file_sha256'][cid] == hashes[name], 'Cell completion/source/SHA binding differs: ' + cid)
            with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as new, np.load(reference / 'cells' / (cid + '.npz'), allow_pickle=False) as old:
                require(scalar(new, 'signature') == ev['signature'] and json.loads(scalar(new, 'meta')) == cell
                        and scalar(new, 'source_score_sha256') == source_sha and scalar(new, 'realization_sha256') == fingerprints[cid]
                        and np.array_equal(new['cluster_id'], np.arange(512) // 8), 'Cell frame metadata differs: ' + cid)
                require(new['nmse'].shape == (9, 512) and np.isfinite(new['nmse']).all() and (new['nmse'] >= 0).all()
                        and np.array_equal(new['nmse'][:6], old['nmse'][old_indices]), 'Invalid/unpaired NMSE scores: ' + cid)
                require(new['native_available'].shape == (9,) and np.array_equal(new['native_available'][:6], old['native_available'][old_indices])
                        and np.array_equal(new['native_available'][6:], np.zeros(3, dtype=np.int8)), 'Native capability differs: ' + cid)
                if cell['ber']:
                    keys = ['bits', 'errors_diagonal', 'errors_native', 'native_residual_max', 'native_direct_fallback']
                    keys += [k for k in ('errors_noise_minus3db', 'errors_noise_plus3db') if k in old]
                    for key in keys:
                        values = new[key]
                        require(values.shape == (9, 512) and np.isfinite(values).all()
                                and np.array_equal(values[:6], old[key][old_indices]), 'Unpaired receiver field: ' + cid + '.' + key)
                        if key == 'bits' or key.startswith('errors_'):
                            require(np.issubdtype(values.dtype, np.integer) and (values >= 0).all(), 'Invalid count dtype/range: ' + cid)
                        if key.startswith('errors_'):
                            require((values <= new['bits']).all(), 'BER errors exceed bits: ' + cid)
                    require((new['bits'] > 0).all() and np.array_equal(new['bits'][6:], np.broadcast_to(old['bits'][old_indices[0]], (3, 512)))
                            and np.array_equal(new['errors_native'][6:], new['errors_diagonal'][6:])
                            and not new['native_residual_max'][6:].any() and not new['native_direct_fallback'][6:].any(), 'Author receiver denominator/native identity differs: ' + cid)
                else:
                    require(not any(k in new for k in ('bits', 'errors_diagonal', 'errors_native')), 'NMSE-only cell contains invented BER fields: ' + cid)
        import yaml
        require(set(recipe['configs']) == {f'fortitran_budget_matched_seed{s}' for s in SEEDS}
                and set(ev['checkpoint_sha256']) == {str(s) for s in SEEDS}, 'Three-seed config/checkpoint set differs')
        migrated_seeds = []
        for seed in SEEDS:
            label = f'fortitran_budget_matched_seed{seed}'
            record = recipe['configs'][label]
            cfgname = 'training/' + record['file']
            require(safe_name(record['file']) and hashes[cfgname] == record['sha256'], 'Prepared YAML SHA differs: ' + label)
            cfg = yaml.safe_load(z.read(cfgname))
            require(digest(cfg) == record['config_sha256'], 'Prepared config digest differs: ' + label)
            check_config(cfg, experiment['configs'][f'fortitran_seed{seed}'], seed)
            folder = 'training/checkpoints/' + label + '/'
            marker = read_json(folder + 'complete.json')
            check_budget_marker(marker, seed, recipe['signature'])
            require(marker['selection_metric'] == 'val_mse' and marker['validation_cache_sha256'] == VALIDATION_SHA
                    and marker['effective_batch_frames'] == 512 and marker['gradient_microbatch_frames'] == cfg['train']['gradient_microbatch_size']
                    and marker['exact_original_dataset_reproduction'] is False, 'Training metadata differs: ' + label)
            require(marker['selected_checkpoint_sha256'] == marker['model_sha256'] == hashes[folder + 'model.pt'] == ev['checkpoint_sha256'][str(seed)]
                    and marker['last_checkpoint_sha256'] == hashes[folder + 'last.pt'], 'Selected/final checkpoint hashes differ: ' + label)
            states = []
            for basename in ('model.pt', 'last.pt'):
                data = z.read(folder + basename)
                sidecar = read_json(folder + basename + '.sha256.json')
                state = checkpoint_metadata(data, label + '.' + basename)
                require(sidecar['sha256'] == hashes[folder + basename] and sidecar['bytes'] == len(data)
                        and sidecar['recipe_signature'] == recipe['signature'] and sidecar['epoch'] == state['epoch'], 'Checkpoint sidecar differs: ' + label)
                require(state['cfg'] == cfg and state['recipe_signature'] == recipe['signature']
                        and state.get('recipe_claim') == CLAIM
                        and state['selection_metric'] == 'val_mse' and state['validation_cache_sha256'] == VALIDATION_SHA
                        and state['source_files_sha256'] == recipe['source_files_sha256']
                        and state['implementation_sha256'] == recipe['implementation_sha256']
                        and state['optimizer_name'] == 'adam' and state['scheduler_name'] == 'exponential'
                        and state['effective_batch_frames'] == 512 and state['physical_gradient_microbatch_frames'] == cfg['train']['gradient_microbatch_size']
                        and isinstance(state['model'], dict) and bool(state['model']), 'Checkpoint recipe/selection metadata differs: ' + label)
                states.append(state)
            rows = list(csv.DictReader(io.StringIO(z.read(folder + 'history.csv').decode('utf-8-sig'))))
            check_history(rows, states[0], states[1], marker, label)
            if verify_prefix_provenance(z, hashes, recipe, cfg, states, rows, seed):
                migrated_seeds.append(seed)
        require({name.split('/')[2] for name in hashes if name.startswith('training/prefix_provenance/')}
                == {f'seed{seed}' for seed in migrated_seeds}, 'Unchecked prefix provenance folders')
        require(set(stats['summary_csv_sha256']) == set(SUMMARY_FILES), 'Expected CI/paired summary files missing')
        csvs = {}
        for name, sha in stats['summary_csv_sha256'].items():
            full = 'evaluation/summary/' + name
            require(hashes[full] == sha, 'CI summary binding differs: ' + name)
            csvs[name] = list(csv.DictReader(io.StringIO(z.read(full).decode('utf-8-sig'))))
            require(bool(csvs[name]), 'Empty summary CSV: ' + name)
        macro = csvs['macro_summary.csv']
        require({r['method'] for r in macro} == set(METHODS), 'Summary methods missing')
        for row in macro:
            require(int(row['n_training_seeds']) == 3, 'Missing three-seed uncertainty')
            for key in ('mean', 'seed_ci95_lo', 'seed_ci95_hi', 'test_cluster_ci95_lo', 'test_cluster_ci95_hi'):
                number(row[key], 'CI.' + key, True)
            require(float(row['seed_ci95_lo']) <= float(row['mean']) <= float(row['seed_ci95_hi'])
                    and float(row['test_cluster_ci95_lo']) <= float(row['test_cluster_ci95_hi']), 'Invalid CI order')
        for method in METHODS:
            for scope in ('gpp_seen', 'gpp_heldout', 'gpp_high_mobility'):
                require(any(r['method'] == method and r['study'] == 'nmse' and r['metric'] == 'nmse' and r['scope'] == scope
                            and r['modulation'] == 'qpsk' and r['doppler_hz'] == r['snr_db'] == 'all' for r in macro), 'Missing manuscript NMSE summary')
            for scope in ('gpp_seen', 'gpp_heldout'):
                require(any(r['method'] == method and r['study'] == 'receiver' and r['metric'] == 'ber_diagonal' and r['scope'] == scope
                            and r['modulation'] == '16qam' and r['doppler_hz'] == r['snr_db'] == 'all' for r in macro), 'Missing manuscript BER summary')
                require(any(r['method'] == method and r['study'] == 'nmse' and r['scope'] == scope and r['modulation'] == 'qpsk'
                            and r['snr_db'] == '25.0' and r['doppler_hz'] != 'all' for r in macro), 'Missing plotted Doppler data')
        for name in ('cell_paired_differences.csv', 'macro_paired_differences.csv'):
            require({r['comparator'] for r in csvs[name]} == set(METHODS[1:]), 'Paired comparator missing: ' + name)
            for row in csvs[name]:
                require(row['reference'] == 'dopnet', 'Wrong paired reference')
                for key in ('paired_difference', 'paired_seed_ci95_lo', 'paired_seed_ci95_hi',
                            'paired_test_cluster_ci95_lo', 'paired_test_cluster_ci95_hi'):
                    number(row[key], 'paired CI.' + key)
        assets = 'evaluation/paper_assets/fortitran_budget_matched'
        table = z.read(assets + '_table.tex').decode('utf-8')
        require('DOP-Net' in table and 'FortiTran (common recipe)' in table and 'FortiTran (equal budget)' in table
                and '40 epochs' in table and 'without early stopping' in table
                and 'NMSE (dB)' in table and 'BER' in table and '95\\%' in table, 'Comparison table is incomplete')
        require(z.read(assets + '.pdf').startswith(b'%PDF-')
                and z.read(assets + '.png').startswith(b'\x89PNG\r\n\x1a\n'), 'Comparison figures missing/invalid format')
    return dict(status='pass', archive=str(archive.resolve()), archive_sha256=file_hash(archive),
                zip_crc_pass=True, manifest_sha256_files_verified=len(hashes), manuscript_cells_verified=269,
                local_original_cells_verified=636, local_original_score_bindings_verified=269,
                routes_verified=9, frames_per_cell=512, completed_training_seeds=list(SEEDS),
                checkpoint_hashes_and_metadata_verified=6, selection_metric='val_mse',
                training_claim=CLAIM, training_termination=TERMINATION, training_max_epochs=MAX_EPOCHS,
                migrated_training_prefixes_verified=migrated_seeds,
                summary_tables_and_figure_files_verified=True,
                scope='Read-only provenance/completeness checks; no model, tensor materialization, or statistics recomputation',
                limitations=['File hashes establish identity, not model accuracy.',
                             'CI values are checked for binding/layout, not independently recomputed.',
                             'Figure file format/coverage is checked; visual layout requires separate inspection.'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--reference-results', type=Path, required=True)
    ap.add_argument('--implementation-root', type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    try:
        result = verify(args.archive, args.reference_results, args.implementation_root)
    except (ValueError, RuntimeError, KeyError, OSError, BadZipFile, pickle.UnpicklingError, EOFError) as error:
        print(json.dumps(dict(status='fail', error=str(error)), ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
