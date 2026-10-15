"""CUDA-server-only exact40 trainer; full second-order ADD then joint online.

The published commit directory is the only authoritative submitted epoch.
Partial epochs never qualify and are not resumed halfway through a stream.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from .artifacts import (digest, file_hash, read_signed, sign, write_json,
                        scan_commits, publish_commit, checkpoint_path, derive_summary)
from .prepare import validate_manifest, VALIDATION_SHA256
from .protocol import DEADLINE_UTC, episode_partition, phase_for_epoch, check_terminal_ledger


class DeadlineReached(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc)


def check_deadline():
    if utc_now() >= datetime.fromisoformat(DEADLINE_UTC.replace('Z', '+00:00')):
        raise DeadlineReached('Original global deadline reached; preserve committed states')


@contextmanager
def seed_lock(folder):
    # flock cannot be inherited as a misleading stale PID lock after a crash.
    import fcntl
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'worker.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()) + '\n')
        stream.flush()
        yield


def process_memory_limit(total, hardware_only=False):
    if type(total) is not int or total < 47000*2**20 or type(hardware_only) is not bool:
        raise ValueError('Original GPU capacity or resource mode differs')
    # Sionna and the estimator share the Torch allocator in this runtime.
    # The6GiB model budget must not cap original3GPP waveform generation.
    return 6*2**30 if hardware_only else total-3*2**30


def load_runtime(project, *, hardware_only=False):
    if os.name != 'posix':
        raise RuntimeError('CUDA server only; never execute on local Windows')
    project = Path(project).resolve()
    sys.path.insert(0, str(project))
    importlib.import_module('utils.cuda_memory').configure_cuda_allocator()
    torch = importlib.import_module('torch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required')
    total=int(torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.set_per_process_memory_fraction(process_memory_limit(total,hardware_only)/total)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    trainer = importlib.import_module('train')
    receiver = importlib.import_module('models.receiver')
    if (Path(trainer.__file__).resolve().parent != project or
            Path(receiver.__file__).resolve().parent.parent != project):
        raise RuntimeError('Wrong archived runtime imported')
    model_module = importlib.import_module('lee_meta_equal40.model')
    updates = importlib.import_module('lee_meta_equal40.meta_updates')
    return torch, trainer, receiver, model_module, updates


def environment(torch):
    import numpy as np
    return {'torch': str(torch.__version__), 'cuda': str(torch.version.cuda),
            'cudnn': torch.backends.cudnn.version(), 'numpy': np.__version__,
            'python': sys.version, 'gpu': torch.cuda.get_device_name(0)}


def seed_runtime(torch, seed):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state(torch):
    import numpy as np
    return {'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
            'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all()}


def restore_rng(torch, state):
    import numpy as np
    random.setstate(state['rng_python'])
    np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch'].cpu())
    torch.cuda.set_rng_state_all([r.cpu() for r in state['rng_cuda']])


def branch_tensor_hash(model):
    import hashlib
    h = hashlib.sha256()
    for name, value in model.sft.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def adam(torch, model):
    return torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0, foreach=False)


def adam_step_counts(optimizer):
    return [int(optimizer.state[p]['step'].item())
            for group in optimizer.param_groups for p in group['params']]


def online_step(torch, module, model, optimizer, inputs, truth, micro, deadline_check=check_deadline):
    if inputs.shape[0] != 32 or truth.shape[0] != 32:
        raise ValueError('Online updates consume32 fresh frames')
    optimizer.zero_grad(set_to_none=True)
    weighted = 0.0
    from torch.utils.checkpoint import checkpoint
    for start in range(0, 32, micro):
        deadline_check()
        x, h = inputs[start:start + micro], truth[start:start + micro]
        prediction = checkpoint(model, x, use_reentrant=False, preserve_rng_state=True)
        loss = module.complex_mse(prediction, h)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite online loss')
        (loss * (x.shape[0] / 32)).backward()
        weighted += float(loss.detach()) * (x.shape[0] / 32)
        del prediction, loss
    gradients = [p.grad for p in model.parameters()]
    if any(g is None or not bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError('Missing/nonfinite joint SFT+ADD gradients')
    deadline_check()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return weighted


def train_epoch(torch, module, updates, model, optimizer, generator, cfg, epoch):
    phase, micro = phase_for_epoch(epoch), cfg['lee_meta_reproduction']['microbatch']
    frames = steps = homogeneous = 0
    weighted = 0.0
    fullpath_allocated_peak = fullpath_reserved_peak = model_incremental_peak = 0
    model.train()
    started = time.perf_counter()
    for batch in generator.iter_batches(12000, epoch, 32):
        check_deadline()
        if batch['input'].shape[0] != 32 or batch['target_h'].shape[0] != 32:
            raise ValueError('Original batch32 episode boundary changed')
        fullpath_allocated_peak=max(fullpath_allocated_peak,int(torch.cuda.max_memory_allocated()))
        fullpath_reserved_peak=max(fullpath_reserved_peak,int(torch.cuda.max_memory_reserved()))
        model_baseline=int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        if phase == 'add_meta':
            fd = batch['max_doppler_hz'].detach().cpu().reshape(-1).tolist()
            support, query = episode_partition(fd)
            homogeneous += int(max(fd) == min(fd))
            with torch.enable_grad():
                result = updates.add_meta_step(model, batch['input'][support], batch['target_h'][support],
                                               batch['input'][query], batch['target_h'][query],
                                               microbatch=micro, deadline_check=check_deadline)
            loss = result['query_loss']
            if result['second_order'] is not True or result['outer_updates'] != 1:
                raise ValueError('Meta update qualification differs')
        else:
            with torch.enable_grad():
                loss = online_step(torch, module, model, optimizer, batch['input'], batch['target_h'], micro)
        model_peak=int(torch.cuda.max_memory_allocated())
        increment=max(0,model_peak-model_baseline)
        if increment > 6*2**30:
            raise ValueError('Actual estimator update exceeded incremental6GiB allocation budget')
        model_incremental_peak=max(model_incremental_peak,increment)
        fullpath_allocated_peak=max(fullpath_allocated_peak,model_peak)
        fullpath_reserved_peak=max(fullpath_reserved_peak,int(torch.cuda.max_memory_reserved()))
        frames += 32
        steps += 1
        weighted += loss * 32
        del batch
        if steps == 1 or steps % 50 == 0:
            print(f'lee seed={cfg["seed"]} global={epoch} phase={phase} frames={frames}/12000 updates={steps}', flush=True)
    torch.cuda.synchronize()
    if frames != 12000 or steps != 375:
        raise ValueError('Epoch did not consume exactly12000 fresh frames /375 updates')
    return {'global_epoch': epoch, 'data_epoch': epoch, 'phase': phase,
            'fresh_frames': frames, 'optimizer_updates': steps,
            'support_frames': 6000 if epoch == 1 else 0, 'query_frames': 6000 if epoch == 1 else 0,
            'meta_inner_steps': steps if epoch == 1 else 0,
            'online_lr': None if epoch == 1 else float(optimizer.param_groups[0]['lr']),
            'homogeneous_meta_episodes': homogeneous if epoch == 1 else 0,
            'train_loss': weighted / frames, 'train_seconds': time.perf_counter() - started,
            'fullpath_torch_allocated_peak_bytes':fullpath_allocated_peak,
            'fullpath_torch_reserved_peak_bytes':fullpath_reserved_peak,
            'model_incremental_peak_allocated_bytes':model_incremental_peak}


def load_validation(cfg, manifest, folder):
    fixed = importlib.import_module('fixed_validation')
    path = Path(manifest['validation']['path'])
    fixed.assert_historical_identity(cfg)
    before = fixed.historical_file_metadata(path)
    fixed.verify_historical_cache_file(path, VALIDATION_SHA256)
    if fixed.historical_file_metadata(path) != before:
        raise ValueError('Historical cache changed during verification')
    proof = dict(before, status='passed', cache_sha256=VALIDATION_SHA256,
                 use='checkpoint_selection_only_never_optimizer_input')
    data = fixed.load_historical_core(cfg, path, VALIDATION_SHA256, verified_proof=proof)
    write_json(folder / 'historical_validation_attestation.json', sign(proof))
    return data


def validate(torch, receiver, model, validation, cfg, epoch):
    model.eval()
    saved_rng = rng_state(torch)
    total = err = power = 0.0
    micro = cfg['lee_meta_reproduction']['microbatch']
    try:
        with torch.no_grad():
            for start in range(0, 2000, micro):
                check_deadline()
                stop = min(start + micro, 2000)
                x = validation['input'][start:stop].to('cuda:0')
                h = validation['target_h'][start:stop].to('cuda:0')
                prediction = model.add_prediction(x) if epoch == 1 else model(x)
                prediction, h = receiver.select_data_csi(prediction, cfg), receiver.select_data_csi(h, cfg)
                # This is the same common val_mse scale used by prior methods.
                loss = torch.nn.functional.mse_loss(prediction.float(), h.float())
                per_error = (prediction - h).square().reshape(stop-start, -1).sum(dim=1)
                per_power = h.square().reshape(stop-start, -1).sum(dim=1).clamp_min(1e-12)
                err += float((per_error / per_power).sum())
                power += stop - start
                total += float(loss) * (stop - start)
                del x, h, prediction, loss, per_error, per_power
    finally:
        restore_rng(torch, saved_rng)
    result = {'val_mse': total / 2000, 'val_nmse_db': 10 * math.log10(max(err / power, 1e-15))}
    if any(not math.isfinite(v) for v in result.values()):
        raise FloatingPointError('Nonfinite historical validation')
    return result


def binding_for(manifest, cfg, seed):
    return {'manifest_signature': manifest['signature'], 'config_sha256': digest(cfg),
            'recipe_sha256': cfg['lee_meta_reproduction']['recipe_sha256'], 'seed': seed,
            'validation_cache_sha256': VALIDATION_SHA256}


def validate_resume(state, record, rows, binding, cfg, env):
    if (state.get('binding') != binding or state.get('cfg') != cfg or state.get('environment') != env or
            state.get('epoch') != len(rows) or state.get('history') != rows or
            digest(rows) != record['history_sha256'] or state.get('model') is None):
        raise ValueError('Full-state resume provenance/history/environment differs')
    expected_parent = record['meta_parent_sha256']
    if state.get('meta_parent_sha256') != expected_parent:
        raise ValueError('Resume parent changed')
    for key in ('rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda'):
        if state.get(key) is None:
            raise ValueError('Missing resume RNG: ' + key)
    if (state['epoch'] == 1) != (state.get('optimizer') is None):
        raise ValueError('Offline/joint Adam state differs')


def train(args):
    check_deadline()
    manifest, cfg, name = validate_manifest(args.manifest, args.project, args.seed)
    qualification = read_signed(args.qualification)
    if (qualification.get('status') != 'passed' or
            qualification.get('microbatch') != cfg['lee_meta_reproduction']['microbatch'] or
            qualification.get('implementation_sha256') != manifest['implementation_sha256'] or
            qualification.get('recipe_sha256') != cfg['lee_meta_reproduction']['recipe_sha256'] or
            qualification.get('scalar_full_chain_rule_gradient') is None or
            qualification.get('offline_sft_unchanged') is not True or
            any(qualification.get('phases', {}).get(p, {}).get('actual_updates') != 2
                for p in ('add_meta', 'joint_online'))):
        raise ValueError('Missing actual exact-source CUDA double-backward qualification')
    from .probe import gate_parents
    parent, _ = gate_parents(args.parent_receipt)
    if qualification.get('parent_receipt_sha256') != file_hash(args.parent_receipt):
        raise ValueError('Qualification belongs to another parent receipt')
    folder = Path(args.manifest).resolve().parent / 'checkpoints' / name
    binding = binding_for(manifest, cfg, args.seed)
    with seed_lock(folder):
        records = scan_commits(folder, binding)
        if records and not args.resume:
            raise ValueError('Existing full commits require explicit --resume')
        if len(records) == 40:
            raise ValueError('Training already submitted40; use the metadata verifier')
        if args.resume:
            # Keep prior diagnostics intact while making this explicit new
            # attempt's terminal status unambiguous. All destinations stay here.
            import uuid
            for name in ('failure.json', 'incomplete.json'):
                path = folder / name
                if path.exists():
                    old = folder / 'prior_attempts' / (uuid.uuid4().hex + '_' + name)
                    old.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(old)
        torch, trainer, receiver, module, updates = load_runtime(args.project)
        seed_runtime(torch, args.seed)
        model = module.LeeAlternativeMeta().to('cuda:0')
        env = environment(torch)
        initial_sft = branch_tensor_hash(model)
        optimizer = None
        summary = derive_summary(records)
        history = summary['history']
        state = None
        if records:
            state = torch.load(checkpoint_path(folder, len(records)), map_location='cuda:0', weights_only=False)
            validate_resume(state, records[-1], history, binding, cfg, env)
            model.load_state_dict(state['model'], strict=True)
            initial_sft = state['initial_sft_sha256']
            if len(records) >= 2:
                optimizer = adam(torch, model)
                optimizer.load_state_dict(state['optimizer'])
        generator = trainer.BatchChannelGenerator(cfg, device='cuda:0', seed=args.seed + 10001)
        validation = load_validation(cfg, manifest, folder)
        if state is not None:
            restore_rng(torch, state)
        del state
        try:
            for epoch in range(len(records) + 1, 41):
                check_deadline()
                if epoch == 2:
                    if optimizer is not None:
                        raise ValueError('Joint Adam must initialize exactly once at global2')
                    optimizer = adam(torch, model)
                    print('LEE_META_TO_JOINT global_epoch=2 data_epoch=2 preserve_global_RNG=true', flush=True)
                torch.cuda.reset_peak_memory_stats()
                row = train_epoch(torch, module, updates, model, optimizer, generator, cfg, epoch)
                row.update(validate(torch, receiver, model, validation, cfg, epoch))
                row['torch_peak_allocated_bytes'] = max(row['fullpath_torch_allocated_peak_bytes'],int(torch.cuda.max_memory_allocated()))
                row['torch_peak_reserved_bytes'] = max(row['fullpath_torch_reserved_peak_bytes'],int(torch.cuda.max_memory_reserved()))
                if epoch == 1 and branch_tensor_hash(model) != initial_sft:
                    raise ValueError('SFT changed during ADD-only offline training')
                history = history + [row]
                # Recheck frozen runtime/recipe at the full-epoch commit boundary.
                again, cfg_again, _ = validate_manifest(args.manifest, args.project, args.seed)
                if again != manifest or cfg_again != cfg:
                    raise ValueError('Provenance changed during an epoch')
                committed = {'schema': 'lee_equal40_full_state_v1', 'binding': binding, 'cfg': cfg,
                             'epoch': epoch, 'data_epoch': epoch, 'phase': phase_for_epoch(epoch),
                             'history': history, 'model': model.state_dict(),
                             'optimizer': optimizer.state_dict() if optimizer is not None else None,
                             'optimizer_name': 'explicit_GD' if epoch == 1 else 'Adam',
                             'adam_step_counts': adam_step_counts(optimizer) if optimizer is not None else [],
                             'meta_parent_sha256': records[0]['state_sha256'] if records else None,
                             'initial_sft_sha256': initial_sft, 'environment': env,
                             'generator_identity': {'seed': args.seed+10001, 'last_global_epoch': epoch,
                                                    'next_global_epoch': epoch+1, 'generation_batch': 32,
                                                    'link_batch': 8, 'stateless_per_epoch': True},
                             'state_created_utc': utc_now().isoformat(), **rng_state(torch)}
                record = publish_commit(folder, committed, binding, torch.save, check_deadline)
                records.append(record)
                summary = derive_summary(records)
                write_json(folder / 'progress.json', sign(summary))
                print(f'LEE_COMMIT seed={args.seed} epoch={epoch}/40 val_mse={row["val_mse"]:.9g} sha={record["state_sha256"]}', flush=True)
                del committed
            check_deadline()
            check_terminal_ledger(history)
            write_json(folder / 'TRAINING_SUBMITTED40.json', sign(dict(summary,
                       status='submitted40_pending_actual_metadata_verifier',
                       training_delivery_qualified=False, all_experiments_complete=False,
                       finished_utc=utc_now().isoformat(), binding=binding)))
            return 0
        except DeadlineReached:
            write_json(folder / 'incomplete.json', sign(dict(derive_summary(records),
                       status='incomplete', exit_code=75, reason='original_global_deadline',
                       current_partial_epoch_discarded=True, observed_utc=utc_now().isoformat())))
            return 75
        except Exception:
            write_json(folder / 'failure.json', sign(dict(derive_summary(records),
                       status='incomplete', exit_code=1, traceback=traceback.format_exc(),
                       observed_utc=utc_now().isoformat())))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--qualification', required=True)
    parser.add_argument('--parent-receipt', required=True)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        result = train(args)
    except DeadlineReached:
        print('incomplete: original global deadline; no new CUDA work', flush=True)
        result = 75
    sys.exit(result)


if __name__ == '__main__':
    main()
