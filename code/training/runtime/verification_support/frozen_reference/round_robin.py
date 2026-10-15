"""Epoch barriers with one shared data realization per seed and epoch."""
import json
import shutil
import sys

from experiment_protocol import SEEDS, VARIANTS, write_json
from utils.checkpoints import load_checkpoint


def train_rounds(out, configs, manifest, run,
                 fixed_validation_cache=None, fixed_validation_sha256=None,
                 train_variants=None, cache_config_prefix='dopnet',
                 historical_training_reference=None):
    if bool(fixed_validation_cache) != bool(fixed_validation_sha256):
        raise ValueError('Fixed validation cache path and SHA-256 must be supplied together')
    total = {int(cfg['train']['epochs']) for cfg in configs.values()}
    if len(total) != 1:
        raise ValueError('All learners require the same epoch budget')
    epochs = total.pop()
    variants = tuple(VARIANTS) if train_variants is None else tuple(train_variants)
    if not variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError('Invalid training variant subset')
    cache_root = (out / 'epoch_cache').resolve()

    def progress(name):
        path = out / 'checkpoints' / name / 'last.pt'
        if not path.exists():
            return 0
        checkpoint = load_checkpoint(path, 'cpu')
        if checkpoint['cfg'] != configs[name]:
            raise RuntimeError(f'Config mismatch: {name}')
        return int(checkpoint['epoch'])

    for epoch in range(1, epochs + 1):
        for seed in SEEDS:
            names = [f'{variant}_seed{seed}' for variant in variants]
            pending = [name for name in names if progress(name) < epoch]
            if not pending:
                continue
            cache = cache_root / f'seed{seed}_epoch{epoch:03d}'
            run([
                sys.executable, 'epoch_cache.py',
                '--config', out / 'configs' / f'{cache_config_prefix}_seed{seed}.yaml',
                '--epoch', epoch,
                '--output', cache,
            ], f'cache_{seed}_{epoch}')
            data_manifest = json.loads((cache / 'complete.json').read_text())
            if historical_training_reference is not None:
                proof=historical_training_reference.check_epoch(data_manifest,seed,epoch)
                write_json(out/'historical_data_match'/f'seed{seed}_epoch{epoch:03d}.json',proof)
            write_json(
                out / 'data_manifests' / f'seed{seed}_epoch{epoch:03d}.json',
                data_manifest,
            )
            for name in pending:
                if progress(name) != epoch - 1:
                    raise RuntimeError(f'Epoch barrier broken: {name}')
                folder = out / 'checkpoints' / name
                command = [
                    sys.executable, 'train.py',
                    '--config', out / 'configs' / f'{name}.yaml',
                    '--run-name', folder,
                    '--seed', seed,
                    '--device', 'cuda',
                    '--epoch-cache', cache,
                    '--stop-after-epoch', epoch,
                ]
                if fixed_validation_cache:
                    command += [
                        '--fixed-validation-cache', fixed_validation_cache,
                        '--fixed-validation-sha256', fixed_validation_sha256,
                    ]
                if epoch > 1:
                    command += ['--resume', folder / 'last.pt']
                run(command, name)
                if progress(name) != epoch:
                    raise RuntimeError('Epoch checkpoint not committed')
                if epoch == epochs:
                    write_json(
                        folder / 'complete.json',
                        dict(signature=manifest['signature'], config=configs[name]),
                    )
            if cache.resolve().parent != cache_root:
                raise RuntimeError('Unsafe cache path')
            shutil.rmtree(cache)
        write_json(
            out / 'training_progress.json',
            dict(completed_epoch=epoch, total_epochs=epochs),
        )

    for seed in SEEDS:
        for variant in variants:
            name = f'{variant}_seed{seed}'
            folder = out / 'checkpoints' / name
            if progress(name) != epochs or not (folder / 'model.pt').exists():
                raise RuntimeError(f'Incomplete training: {name}')
            write_json(
                folder / 'complete.json',
                dict(signature=manifest['signature'], config=configs[name]),
            )
