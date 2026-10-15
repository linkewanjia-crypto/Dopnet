"""Read-only final delivery checks; no generation, inference or statistics.

Run after all server stages finish and the return ZIP has been downloaded.
Only archive contents, JSON/NPZ metadata and existing file SHA-256 are read.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import sys
from zipfile import ZipFile

import numpy as np

METHODS = ('pilot_covariance_lmmse', 'oce_bem_joint_cholesky', 'dpss_bem_joint_cholesky')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signed(value, label):
    require(value.get('signature') == digest({k: v for k, v in value.items() if k != 'signature'}),
            'Invalid signed metadata: ' + label)
    return value


def scalar(z, key):
    return str(z[key].item())


def selected_cells(cells):
    result = []
    dynamic = {'doubly_selective', 'tdl_b', 'tdl_c', 'tdl_d', 'cdl_b', 'cdl_c',
               'cdl_d', 'tdl_a', 'tdl_e', 'cdl_a', 'cdl_e'}
    for m in cells:
        nominal = m['cp_len'] == 16 and m['pilot_spacing'] == 4 and m['delay_spread_ns'] == 300.
        main = nominal and m['modulation'] == 'qpsk' and 'nmse' in m['studies']
        core = (nominal and m['channel'] in dynamic and m['doppler_hz'] in (800., 1200., 1600.)
                and ((m['modulation'] == 'qpsk' and m['snr_db'] in (15., 25.) and 'nmse' in m['studies'])
                     or (m['modulation'] == '16qam' and m['snr_db'] == 20. and 'receiver' in m['studies'])))
        if main or core:
            result.append(m)
    require(len(result) == 269, 'Local reference does not contain the complete 269-cell manuscript suite')
    return result


def native_profiles(code_root):
    path = Path(code_root) / 'models/dopnet.py'
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ABLATION_PROFILES' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Cannot establish native-interface eligibility from model source')


def verify(archive, reference_results, code_root):
    archive, reference, code = Path(archive), Path(reference_results), Path(code_root)
    if reference.name == 'cells':
        reference = reference.parent
    local = lambda name: json.loads((reference / name).read_text(encoding='utf-8-sig'))
    experiment = signed(local('experiment.json'), 'local reference experiment')
    reference_eval = signed(local('evaluation.json'), 'local reference evaluation')
    require(reference_eval.get('experiment_signature') == experiment['signature'], 'Local reference provenance differs')
    require(len(reference_eval['cells']) == 636, 'All 636 original score cells are required locally')
    require(all((reference / 'cells' / (m['cell_id'] + '.npz')).is_file() for m in reference_eval['cells']),
            'Local original score archive is missing cells')
    expected = {m['cell_id']: m for m in selected_cells(reference_eval['cells'])}
    with ZipFile(archive) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        require(len(names) == len(set(names)), 'ZIP contains duplicate file names')
        require(z.testzip() is None, 'ZIP CRC validation failed')
        suffix = '/return_package_manifest.json'
        manifests = [name for name in names if name.endswith(suffix)]
        require(len(manifests) == 1, 'Need exactly one return package manifest')
        prefix = manifests[0][:-len('return_package_manifest.json')]
        require(all(name.startswith(prefix) for name in names), 'Unexpected files outside return package root')
        read_json = lambda name: json.loads(z.read(prefix + name).decode('utf-8-sig'))
        manifest = read_json('return_package_manifest.json')
        require(manifest.get('paper_ready') is True and manifest.get('pending') == [], 'Return manifest is not paper ready')
        require(manifest.get('suite') == 'manuscript' and manifest.get('training_performed') is False,
                'Return package suite/training scope differs')
        actual = {name[len(prefix):] for name in names} - {'return_package_manifest.json'}
        require(set(manifest['files']) == actual, 'Manifest must bind every returned file exactly once')
        for name, sha in manifest['files'].items():
            p = PurePosixPath(name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in name, 'Unsafe manifest path')
            require(hashlib.sha256(z.read(prefix + name)).hexdigest() == sha, 'ZIP SHA-256 mismatch: ' + name)
        statistics = read_json('corrected_summary/statistics.json')
        require(statistics.get('paper_ready') is True and statistics.get('pending') == [], 'Statistics are not paper ready')
        require(statistics.get('suite') == 'manuscript' and statistics.get('expected_cells') == 269
                and statistics.get('completed_verified_cells') == 269, 'Statistics must verify all 269 cells')
        for name in ('score_assets_ready', 'single_frame_latency_ready', 'throughput_assets_ready',
                     'digital_latency_ready', 'classical_verification_cost_ready'):
            require(statistics.get(name) is True, 'Missing paper readiness asset: ' + name)
        for name, sha in statistics['report_asset_sha256'].items():
            require(manifest['files'].get('corrected_summary/' + name) == sha, 'Report asset binding differs: ' + name)
        evaluation = signed(read_json('evaluation.json'), 'corrected evaluation')
        lock = signed(read_json('calibration_lock.json'), 'calibration lock')
        complete = read_json('COMPLETE.json')
        require(evaluation['suite'] == 'manuscript' and evaluation['methods'] == list(METHODS), 'Incorrect corrected methods/suite')
        require(evaluation['reference_evaluation_signature'] == reference_eval['signature']
                and evaluation['reference_experiment_signature'] == experiment['signature'], 'Local/server reference provenance differs')
        require(statistics['reference_evaluation_signature'] == reference_eval['signature']
                and statistics['evaluation_signature'] == evaluation['signature'], 'Statistics reference binding differs')
        lock_sha = manifest['files']['calibration_lock.json']
        require(evaluation['calibration_lock_sha256'] == lock_sha and statistics['calibration_lock_sha256'] == lock_sha,
                'Calibration lock SHA differs')
        require(lock.get('status') == 'locked' and lock.get('methods') == list(METHODS), 'Calibration is incomplete')
        require(lock['code_sha256'] == evaluation['code_sha256'], 'Calibration/evaluation source differs')
        require(complete.get('status') == 'complete' and complete.get('signature') == evaluation['signature']
                and complete.get('methods') == list(METHODS) and complete.get('suite') == 'manuscript'
                and complete.get('calibration_lock_sha256') == lock_sha
                and complete.get('source_fingerprints_verified') is True, 'Root completion contract differs')
        require({m['cell_id']: m for m in evaluation['cells']} == expected and len(evaluation['cells']) == 269,
                'Corrected evaluation cells differ from the complete local suite')
        require(set(complete['cell_files']) == set(expected) == set(statistics['cell_files']), 'Completed cell set differs')
        new_scores = {name.removeprefix('cells/').removesuffix('.npz') for name in actual
                      if name.startswith('cells/') and name.endswith('.npz')}
        require(new_scores == set(expected), 'Return package score cells differ from the declared suite')
        for cid, meta in expected.items():
            name = 'cells/' + cid + '.npz'
            sha = manifest['files'][name]
            require(complete['cell_files'][cid] == sha == statistics['cell_files'][cid], 'Cell SHA binding differs: ' + cid)
            source_sha = file_hash(reference / name)
            marker = read_json('cells/' + cid + '.COMPLETE.json')
            require(marker.get('status') == 'complete' and marker.get('samples') == 512
                    and marker.get('methods') == list(METHODS) and marker.get('meta') == meta
                    and marker.get('frozen_waveform_sha256_pass') is True, 'Incomplete/incorrect cell marker: ' + cid)
            bindings = {'signature': evaluation['signature'], 'lock_sha256': lock_sha,
                        'source_score_sha256': source_sha, 'npz_sha256': sha}
            for key, value in bindings.items():
                require(marker.get(key) == value, 'Cell marker binding differs: ' + cid + '.' + key)
            with np.load(reference / name, allow_pickle=False) as old, np.load(io.BytesIO(z.read(prefix + name)), allow_pickle=False) as new:
                require(scalar(old, 'signature') == reference_eval['signature'] and json.loads(scalar(old, 'meta')) == meta,
                        'Local original score metadata differs: ' + cid)
                require(scalar(new, 'realization_sha256') == scalar(old, 'realization_sha256') == marker['realization_sha256'],
                        'Frozen realization binding differs: ' + cid)
                require(json.loads(scalar(new, 'meta')) == meta and scalar(new, 'cell_id') == cid,
                        'Corrected NPZ cell identity differs: ' + cid)
                for key in ('signature', 'lock_sha256', 'source_score_sha256'):
                    require(scalar(new, key) == bindings[key], 'Corrected NPZ binding differs: ' + cid + '.' + key)
                require(list(new['methods'].astype(str)) == list(METHODS) and new['nmse'].shape == (3, 512),
                        'Corrected NPZ method/sample dimensions differ: ' + cid)
                require(np.array_equal(new['frame_index'], np.arange(512))
                        and np.array_equal(new['cluster_ids'], np.arange(512) // 8)
                        and np.array_equal(new['cluster_ids'], old['cluster_id']), 'Frame/cluster metadata differs: ' + cid)
        latency = read_json('latency_gpu/digital_receiver_latency.json')
        protocol = latency['protocol']
        require(latency.get('complete') is True and latency.get('numerical_checks_passed') is True, 'GPU latency is incomplete')
        require(protocol['devices'] == ['cuda:0'] and set(protocol['methods']) == {'dopnet', 'dopnet_lite', 'slow_time', 'fortitran'}
                and protocol['seeds'] == [42, 43, 44] and set(protocol['batch_sizes']) == {1, 8, 32}
                and set(protocol['receiver_modes']) == {'common_diagonal', 'native_own_ici'}, 'GPU latency protocol differs')
        require(protocol['experiment_signature'] == experiment['signature'], 'GPU latency reference provenance differs')
        sources = {s['name']: s for s in protocol['sources']}
        require(len(protocol['sources']) == len(sources) == 2 and set(sources) == {'calibration', 'time_capture'},
                'Both original grid and actual time-capture timing sources are required')
        require(protocol['warmups'] >= 1 and protocol['repetitions'] >= 30, 'GPU timing repetitions are incomplete')
        for name, sha in protocol['implementation_sha256'].items():
            require(file_hash(code / name) == sha, 'Local receiver source differs from timed source: ' + name)
        require(file_hash(code / 'verification_gpu/latency.py') == protocol['gpu_adapter_sha256'], 'GPU timing adapter differs')
        profiles = native_profiles(code)
        keys = set(itertools.product(protocol['methods'], protocol['seeds'], sources,
                                     protocol['batch_sizes'], protocol['receiver_modes']))
        rows = {}
        checkpoint_hashes = {}
        for row in latency['rows']:
            key = tuple(row[k] for k in ('method', 'seed', 'source_name', 'batch_size', 'receiver_mode'))
            require(key in keys and key not in rows, 'Missing/duplicate/undeclared GPU workload row: ' + str(key))
            rows[key] = row
            require(row['device'] == 'cuda:0', 'Unexpected non-GPU receiver timing')
            boundary = 'actual_received_time_samples' if row['source_name'] == 'time_capture' else 'received_occupied_ofdm_grid'
            require(row['input_boundary'] == boundary, 'Timing input boundary differs')
            ckname = f'{row["method"]}_seed{row["seed"]}'
            if ckname not in checkpoint_hashes:
                checkpoint_hashes[ckname] = file_hash(reference / 'checkpoints' / ckname / 'model.pt')
            require(row['checkpoint_sha256'] == checkpoint_hashes[ckname], 'Timed checkpoint differs from local saved weights: ' + ckname)
            cfg = experiment['configs'][ckname]['model']
            native = False
            if cfg.get('estimator') == 'dopnet':
                require(cfg['ablation_profile'] in profiles, 'Unknown native-interface model profile')
                native = bool(profiles[cfg['ablation_profile']]['use_add_branch'])
            applicable = row['receiver_mode'] == 'common_diagonal' or native
            if not applicable:
                require(row['status'] == 'not_applicable' and bool(row.get('reason')), 'Inapplicable native interface needs an explicit NA row')
                continue
            require(row['status'] == 'measured' and row.get('numerical_equivalence', {}).get('passed') is True,
                    'GPU workload not fully measured (including OOM): ' + str(key))
            for stage in ('device_resident_total', 'host_origin_total'):
                timing = row['timing'][stage]
                values = timing.get('repetition_latency_ms', [])
                require(len(values) == protocol['repetitions'] and all(math.isfinite(v) and v > 0 for v in values),
                        'Missing/invalid timing repetitions: ' + str(key))
                require(all(math.isfinite(float(timing[k])) and float(timing[k]) > 0
                            for k in ('latency_ms_p50', 'latency_ms_p95', 'frames_per_second')), 'Invalid timing summary')
        require(set(rows) == keys and len(rows) == 144, 'The complete GPU timing Cartesian matrix is missing rows')
        latency_sha = manifest['files']['latency_gpu/digital_receiver_latency.json']
        policy = read_json('latency_gpu/inference_device_policy.json')
        require(policy.get('complete') is True and policy.get('numerical_checks_passed') is True
                and policy.get('latency_report_sha256') == latency_sha
                and policy.get('experiment_signature') == experiment['signature']
                and policy.get('source_export_signature') == protocol['input_export_signature']
                and policy.get('recommendations') == latency['device_recommendations'], 'GPU inference policy is unbound/incomplete')
        require(statistics['latency']['latency_json_sha256'] == latency_sha, 'Statistics GPU latency binding differs')
        return {'status': 'pass', 'archive': str(archive.resolve()), 'archive_sha256': file_hash(archive),
                'zip_crc_pass': True, 'manifest_sha256_files_verified': len(manifest['files']),
                'manuscript_cells_verified': 269, 'frames_per_cell': 512,
                'local_original_score_bindings_verified': 269, 'local_original_cells_present': 636,
                'gpu_workload_rows_verified': len(rows), 'gpu_status_counts': dict(Counter(r['status'] for r in rows.values())),
                'local_timed_checkpoint_hashes_verified': len(checkpoint_hashes),
                'paper_ready': True, 'pending': [], 'waveform_arrays_included': manifest.get('waveform_arrays_included'),
                'scope': 'Read-only file/provenance/completeness checks; no model execution or statistical recomputation'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--reference-results', type=Path, default=root.parent / '实验结果及数据/dopnet_additions')
    p.add_argument('--code-root', type=Path, default=root)
    args = p.parse_args()
    try:
        result = verify(args.archive, args.reference_results, args.code_root)
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({'status': 'fail', 'error': str(error)}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
