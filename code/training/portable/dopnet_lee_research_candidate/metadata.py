"""Restricted, non-tensor metadata inspection of newly created checkpoints."""
import importlib.util
import sys
from pathlib import Path
from .common import require, file_hash, scan_commits, derive_summary, checkpoint_path, read_signed, sign, digest
from .protocol import check_terminal_ledger
READER_SHA = '562f9130f1e8a2dbd632c05bfaf57ba4de785a161d9c6531f8e5f80e30b9c714'

def load_reader(path):
    path = Path(path).resolve()
    require(file_hash(path) == READER_SHA, 'Use the original restricted metadata reader')
    name = 'dopnet_lee_candidate_restricted_reader'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = 'budget_matched_baselines'
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.checkpoint_metadata

def check_state(state, record, history, binding, cfg, meta_parent, terminal=False):
    epoch = record['epoch']
    require(state.get('schema') == 'dopnet_lee_research_full_state_v1', 'State schema differs')
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


def verify(folder, binding, cfg, reader_path):
    records = scan_commits(folder, binding)
    require(len(records) == 40, 'Forty complete, immutable epochs required')
    marker = read_signed(Path(folder)/'TRAINING_SUBMITTED40.json')
    summary = derive_summary(records)
    require(marker.get('binding') == binding and marker.get('status') == 'submitted40_pending_actual_metadata_verifier'
            and all(marker.get(k) == v for k,v in summary.items()), 'Submitted marker differs')
    require(not (Path(folder)/'failure.json').exists() and not (Path(folder)/'incomplete.json').exists(), 'Incomplete attempt')
    reader = load_reader(reader_path)
    parent = records[0]['state_sha256']
    env = initial_sft = None
    for i, record in enumerate(records):
        state = reader(checkpoint_path(folder,i+1).read_bytes(), f'new_lee_seed{binding["seed"]}_epoch{i+1}')
        check_state(state, record, summary['history'][:i+1], binding, cfg, parent, terminal=i==39)
        if i == 0:
            env, initial_sft = state['environment'], state['initial_sft_sha256']
        require(state['environment'] == env and state['initial_sft_sha256'] == initial_sft, 'Resume identity changed')
    require(scan_commits(folder,binding) == records, 'Committed bytes changed during verification')
    return sign(dict(summary, schema='dopnet_lee_research_training_metadata_v1',
                     status='single_seed_training_metadata_pass', exit_code=0, training_delivery_qualified=True,
                     rng_presence_verified=True, rng_boundary_bitwise_verified=False, all_experiments_complete=False,
                     binding=binding, actual_restricted_reader_sha256=READER_SHA,
                     historical_training_delivery=False))
