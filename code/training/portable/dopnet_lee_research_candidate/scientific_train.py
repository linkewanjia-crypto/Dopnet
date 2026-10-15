"""Original numerical helpers in a new policy namespace; no legacy gate import."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import importlib
import math
import os
from pathlib import Path
import random
import sys
import time
from .common import digest, file_hash, read_signed, sign, write_json
from .preparation import VALIDATION_SHA256
from .protocol import episode_partition, phase_for_epoch, check_terminal_ledger
_RUN_POLICY = ContextVar('dopnet_lee_research_run_policy', default=None)

def bind_policy(policy):
    _RUN_POLICY.set(policy)

def check_deadline():
    policy = _RUN_POLICY.get()
    if policy is None:
        raise RuntimeError('Explicit new-run policy must be bound before scientific execution')
    policy.check()

def utc_now():
    return datetime.now(timezone.utc)

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
    model_module = importlib.import_module(__package__ + '.model')
    updates = importlib.import_module(__package__ + '.meta_updates')
    return torch, trainer, receiver, model_module, updates

def chain_rule_scalar_gate(torch):
    from torch.utils.checkpoint import checkpoint
    theta = torch.tensor(0.7, device='cuda:0', requires_grad=True)
    support = checkpoint(lambda t: (t-1.2).square()/2, theta, use_reentrant=False)
    inner = torch.autograd.grad(support, theta, create_graph=True)[0]
    phi = theta-0.2*inner
    query = checkpoint(lambda p: (p+0.5).square()/2, phi, use_reentrant=False)
    full = float(torch.autograd.grad(query, theta)[0])
    if abs(full-1.04) > 2e-6 or abs(full-1.3) < 0.1:
        raise ValueError('Actual CUDA checkpointed full chain rule failed')
    return full

