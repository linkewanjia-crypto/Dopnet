#!/usr/bin/env python3
"""Inspect twelve reduced-data recipes without starting science processes.

list shows saved configurations/checkpoints; verify checks identities and the
fixed-update budget; plan inspects original commands and their prerequisites.
This additive standard-library entry never executes scientific processes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

METHODS = ('dopnet_25pct', 'dopnet_50pct', 'slow_time_25pct', 'slow_time_50pct')
SEEDS = (42, 43, 44)
VALIDATION_REL = 'reproduction_inputs/validation_895338a79e73ad3f.pt'
VALIDATION_SHA = '83d0e03524628b617671c42a73d3c458cc3395c31965a4e22295e8389f56e597'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('An identity escapes the declared root')
    return path


def verify_identity(root, row):
    path = checked(root, row['relative'])
    if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
        raise ValueError('Saved identity differs: ' + row['relative'])
    return path


def budget(cfg, method):
    train = cfg['train']
    unique = 3000 if method.endswith('_25pct') else 6000
    expected = {'train_samples': 12000, 'shared_epoch_samples': 12000,
                'unique_epoch_samples': unique, 'batch_size': 32, 'epochs': 40}
    if any(train.get(key) != value for key, value in expected.items()):
        raise ValueError('A retained reduced-data recipe has a different scientific budget')
    profile = 'full' if method.startswith('dopnet_') else 'no_fast_time_add'
    if cfg['model']['estimator'] != 'dopnet' or cfg['model']['ablation_profile'] != profile:
        raise ValueError('A retained reduced-data recipe has the wrong model profile')
    return dict(expected, optimizer_updates_per_epoch=math.ceil(12000 / 32),
                repeated_prefix_cycles=12000 // unique)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path, default=None,
                        help='Then DOPNET_DATA_ROOT; otherwise sibling ../data')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List the twelve additional saved recipes; no framework import')
    sub.add_parser('verify', help='SHA-check twelve configurations, twelve weights and original engine sources')
    plan = sub.add_parser('plan', help='Inspect original engine commands; does not execute or create files')
    plan.add_argument('--method', choices=METHODS, required=True)
    plan.add_argument('--seed', choices=SEEDS, type=int, required=True)
    plan.add_argument('--stage', choices=('generate', 'train'), default='train')
    plan.add_argument('--epoch', choices=range(1, 41), type=int, default=1)
    plan.add_argument('--output', type=Path, required=True)
    plan.add_argument('--validation-cache', type=Path)
    args = parser.parse_args(argv)
    code_root = args.code_root.expanduser().resolve()
    data_root = (args.data_root or Path(os.environ.get('DOPNET_DATA_ROOT', str(code_root.parent / 'data')))).expanduser().resolve()
    manifest = json.loads(Path(__file__).with_name('data_fraction_inputs.json').read_text(encoding='utf8'))
    routes = manifest['recipes']
    if args.command == 'list':
        print(json.dumps({'route_count': len(routes), 'new_science_execution_supported_by_this_entry': False,
                          'original_engine_support_retained': True, 'recipes': [dict(
                              name=r['name'], config=r['config']['relative'], weight=r['weight']['relative'],
                              config_present=checked(code_root, r['config']['relative']).is_file(),
                              weight_present=checked(data_root, r['weight']['relative']).is_file())
                              for r in routes]}, indent=2))
        return 0
    if args.command == 'verify':
        for row in manifest['original_engine_sources']:
            verify_identity(code_root, row)
        verified = []
        for route in routes:
            config = verify_identity(code_root, route['config'])
            verify_identity(data_root, route['weight'])
            cfg = json.loads(config.read_text(encoding='utf-8-sig'))
            if cfg['seed'] != route['seed']:
                raise ValueError('Saved seed differs')
            verified.append(dict(name=route['name'], budget=budget(cfg, route['method'])))
        print(json.dumps({'status': 'PASS_12_RECIPE_AND_WEIGHT_IDENTITIES_AND_FIXED_UPDATE_BUDGET',
                          'routes': verified, 'checkpoint_deserialization': False,
                          'training_or_evaluation_executed': False,
                          'full_historical_reproduction_proved': False}, indent=2))
        return 0
    route = next(r for r in routes if r['method'] == args.method and r['seed'] == args.seed)
    cfg_path = verify_identity(code_root, route['config'])
    for row in manifest['original_engine_sources']:
        verify_identity(code_root, row)
    cfg = json.loads(cfg_path.read_text(encoding='utf-8-sig'))
    cfg_budget = budget(cfg, args.method)
    output = args.output.expanduser().resolve()
    if output.exists() or any(output == root or output.is_relative_to(root) for root in (code_root, data_root)):
        raise ValueError('Plan a new output directory outside code and data')
    source = code_root / 'training/main'
    cache = output / 'epoch_cache' / f'seed{args.seed}_epoch{args.epoch:03d}'
    validation = args.validation_cache.expanduser().resolve() if args.validation_cache else data_root / VALIDATION_REL
    proof = validation.with_name(validation.stem + '_verified.json')
    command = [sys.executable, '-B', str(source / 'epoch_cache.py'), '--config', str(cfg_path),
               '--epoch', str(args.epoch), '--output', str(cache)]
    if args.stage == 'train':
        command = [sys.executable, '-B', str(source / 'train.py'), '--config', str(cfg_path),
                   '--run-name', str(output / 'checkpoints' / route['name']), '--seed', str(args.seed),
                   '--device', 'cuda', '--epoch-cache', str(cache), '--stop-after-epoch', str(args.epoch),
                   '--fixed-validation-cache', str(validation), '--fixed-validation-sha256', VALIDATION_SHA]
        if args.epoch > 1:
            command += ['--resume', str(output / 'checkpoints' / route['name'] / 'last.pt')]
    print(json.dumps({
        'execute': False, 'route': route['name'], 'stage': args.stage, 'epoch': args.epoch,
        'budget': cfg_budget, 'original_engine_command_for_inspection_only': command, 'cwd': str(source),
        'pythonpath': [str(source)], 'new_process_or_output_created': False,
        'prerequisites': {
            'compatible_Linux_CUDA_and_research_dependencies_required': True,
            'fixed_validation_cache_present': validation.is_file(),
            'genuine_fixed_validation_attestation_present': proof.is_file(),
            'validation_identity_and_attestation_verified_by_this_plan': False,
            'matching_epoch_cache_required_for_train': True,
            'genuine_epoch_tensor_proofs_required_for_exact_historical_replay': True,
            'previous_epoch_checkpoint_required_if_epoch_gt_one': args.epoch > 1},
        'evaluation_limit': 'The retained 38-row public reference excludes these twelve routes. This entry does not provide a full-grid evaluation shortcut or manufacture old 53-route completion/signature evidence.',
        'scope': 'Existing original engine supports unique-frame replay; saved recipes and weights do not prove missing full historical training or evaluation inputs.'}, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print('DATA_FRACTION_ENTRY_ERROR: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
