"""Restricted checkpoint metadata verification; no Torch/model execution.

This qualifies one seed's submitted training artifacts only. It does not
qualify frozen scoring, latency, bitwise RNG continuity or final delivery.
"""
import argparse
import importlib
from pathlib import Path
import sys
import os
from .artifacts import (digest, file_hash, read_signed, sign, write_json,
                        scan_commits, checkpoint_path, derive_summary)
from .prepare import validate_manifest
from .protocol import check_terminal_ledger
from .train import binding_for

READER_SHA = '562f9130f1e8a2dbd632c05bfaf57ba4de785a161d9c6531f8e5f80e30b9c714'


def require(value, message):
    if not value:
        raise ValueError(message)


def check_state(state, record, history, binding, cfg, meta_parent, terminal=False):
    epoch = record['epoch']
    require(state.get('schema') == 'lee_equal40_full_state_v1', 'State schema differs')
    require(state.get('binding') == binding and state.get('cfg') == cfg, 'State config/source binding differs')
    require(state.get('epoch') == epoch and state.get('data_epoch') == epoch and
            state.get('phase') == history[-1]['phase'], 'State global phase/data epoch differs')
    require(state.get('history') == history and digest(history) == record['history_sha256'], 'State ledger differs')
    require(state.get('meta_parent_sha256') == (None if epoch == 1 else meta_parent), 'State parent differs')
    expected = set()
    for branch in ('sft', 'add'):
        for layer in ('feature', 'shrink', 'mapping.0', 'mapping.1', 'mapping.2', 'mapping.3', 'expand', 'reconstruct'):
            expected.update(f'{branch}.{layer}.{kind}' for kind in ('weight', 'bias'))
    require(set(state.get('model', {})) == expected, 'Missing/extra parallel branch weights')
    for key in ('rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda'):
        require(state.get(key) is not None, 'Missing RNG metadata: ' + key)
    require(isinstance(state['rng_python'], tuple) and len(state['rng_python']) == 3 and
            isinstance(state['rng_numpy'], tuple) and len(state['rng_numpy']) == 5 and
            isinstance(state['rng_cuda'], list) and len(state['rng_cuda']) > 0, 'Incomplete RNG containers')
    require(state.get('generator_identity') == {'seed': binding['seed']+10001,
            'last_global_epoch': epoch, 'next_global_epoch': epoch+1,
            'generation_batch': 32, 'link_batch': 8, 'stateless_per_epoch': True}, 'Generator epoch identity differs')
    require(isinstance(state.get('initial_sft_sha256'), str) and len(state['initial_sft_sha256']) == 64,
            'Missing SFT offline isolation attestation')
    if epoch == 1:
        require(state.get('optimizer') is None and state.get('optimizer_name') == 'explicit_GD' and
                state.get('adam_step_counts') == [], 'Offline optimizer differs')
    else:
        optimizer = state.get('optimizer', {})
        require(state.get('optimizer_name') == 'Adam' and state.get('adam_step_counts') == [375*(epoch-1)]*32,
                'Joint Adam update ledger differs')
        require(len(optimizer.get('param_groups', [])) == 1 and len(optimizer.get('state', {})) == 32,
                'Incomplete joint Adam state')
        group = optimizer['param_groups'][0]
        require(group.get('lr') == 0.001 and group.get('betas') == (0.9, 0.999) and
                group.get('eps') == 1e-8 and group.get('weight_decay') == 0 and
                len(group.get('params', [])) == 32, 'Adam hyperparameters differ')
        require(all(set(s) == {'step', 'exp_avg', 'exp_avg_sq'} for s in optimizer['state'].values()),
                'Missing Adam moments')
    if terminal:
        check_terminal_ledger(history)


def verify(folder, manifest_path, project, seed, reader):
    manifest, cfg, name = validate_manifest(manifest_path, project, seed)
    folder = Path(folder).resolve()
    require(folder == Path(manifest_path).resolve().parent / 'checkpoints' / name, 'Wrong seed output root')
    binding = binding_for(manifest, cfg, seed)
    if os.name == 'posix':
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            if int(path.parent.name) == os.getpid():
                continue
            try:
                words = path.read_bytes().split(b'\0')
            except OSError:
                continue
            if b'lee_meta_equal40.train' in words and b'--seed' in words:
                index = words.index(b'--seed')
                if words[index+1:index+2] == [str(seed).encode()]:
                    raise ValueError('Seed worker is still alive; do not read active checkpoint')
    records = scan_commits(folder, binding)
    require(len(records) == 40, 'Training has not naturally submitted40 full epochs')
    marker = read_signed(folder / 'TRAINING_SUBMITTED40.json')
    summary = derive_summary(records)
    require(marker.get('binding') == binding and marker.get('status') == 'submitted40_pending_actual_metadata_verifier',
            'Missing submitted40 marker')
    require(all(marker.get(k) == v for k, v in summary.items()), 'Marker differs from immutable commits')
    require(not (folder / 'failure.json').exists() and not (folder / 'incomplete.json').exists(),
            'Unresolved failure/incomplete artifact')
    parent = records[0]['state_sha256']
    env = initial_sft = None
    for i, record in enumerate(records):
        state = reader(checkpoint_path(folder, i+1).read_bytes(), f'lee_seed{seed}_epoch{i+1}')
        check_state(state, record, summary['history'][:i+1], binding, cfg, parent, terminal=i == 39)
        if i == 0:
            env, initial_sft = state['environment'], state['initial_sft_sha256']
        require(state['environment'] == env and state['initial_sft_sha256'] == initial_sft,
                'Resume environment or offline SFT identity differs')
    require(scan_commits(folder, binding) == records, 'Submitted bytes changed during verification')
    again, cfg_again, _ = validate_manifest(manifest_path, project, seed)
    require(again == manifest and cfg_again == cfg, 'Sources changed during verification')
    return sign(dict(summary, status='single_seed_training_metadata_pass', exit_code=0,
                     training_delivery_qualified=True, rng_presence_verified=True,
                     rng_boundary_bitwise_verified=False, all_experiments_complete=False,
                     binding=binding, actual_restricted_reader_sha256=READER_SHA))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('folder', 'manifest', 'project', 'forti-code-root'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), required=True)
    args = parser.parse_args()
    pkg = Path(args.forti_code_root).resolve()
    require(file_hash(pkg / 'verify_author_delivery.py') == READER_SHA, 'Unreviewed restricted reader')
    sys.path.insert(0, str(pkg.parent))
    module = importlib.import_module('budget_matched_baselines.verify_author_delivery')
    require(Path(module.__file__).resolve() == pkg / 'verify_author_delivery.py', 'Wrong reader import')
    require('torch' not in sys.modules, 'Torch imported in metadata process')
    result = verify(args.folder, args.manifest, args.project, args.seed, module.checkpoint_metadata)
    require('torch' not in sys.modules, 'Metadata verification imported Torch')
    print(result)


if __name__ == '__main__':
    main()
