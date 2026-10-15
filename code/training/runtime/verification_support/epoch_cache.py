"""Disk-backed, lossless shared epoch; never materialize a full epoch in RAM/GPU."""
import argparse
import copy
import json
import math
import hashlib
from pathlib import Path
import shutil
from utils.cuda_memory import configure_cuda_allocator

configure_cuda_allocator()

import torch
from data.generator import BatchChannelGenerator
from experiment_protocol import code_hash, digest, file_hash, write_json
from utils.checkpoints import save_checkpoint

KEYS = ('input','add_input','add_input_slow','add_input_pilot_only_fast_time',
        'noise_var','ls_grid','target_h','target_add')

def _json_value(value):
    """Make YAML tuples/NumPy scalars and JSON lists compare identically."""
    return json.loads(json.dumps(value, sort_keys=True, default=str,
                                 allow_nan=False))

def tensor_hash(batch):
    h=hashlib.sha256()
    for key,value in sorted(batch.items()):
        h.update(digest([key,str(value.dtype),list(value.shape)]).encode())
        h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()

def identity(cfg, epoch):
    # DOP-Net-Lite and ablations differ only in the estimator; physical coordinates
    # and simulator settings must be identical for reuse.
    # Canonicalize BEFORE recording or comparing the identity, independently
    # of whether the caller has already validated its config.
    from data.channels import validate_channel_config
    cfg=copy.deepcopy(cfg)
    validate_channel_config(cfg['system'])
    return _json_value(dict(
        seed=cfg['seed'], epoch=epoch, system=cfg['system'],
        data_pipeline=cfg['data_pipeline'],
        observation={k:cfg['model'].get(k) for k in ('add_doppler_bins',
            'add_delay_bins','add_delay_guard_bins','doppler_projection_reg',
            'doppler_whitening_reg')},
        samples=cfg['train'].get('shared_epoch_samples', cfg['train']['train_samples']),
        batch_size=cfg['train']['batch_size'],
        code_sha256=code_hash(Path(__file__).resolve().parent)))

def identity_differences(recorded, expected, path=''):
    if isinstance(recorded,dict) and isinstance(expected,dict):
        rows=[]
        for key in sorted(set(recorded)|set(expected)):
            rows.extend(identity_differences(recorded.get(key),expected.get(key),
                        f'{path}.{key}' if path else key))
        return rows
    return [] if recorded==expected else [f'{path}: cached={recorded!r}, current={expected!r}']

def generate(cfg, epoch, destination):
    path=Path(destination); expected=identity(cfg,epoch)
    if (path/'complete.json').exists():
        old=json.loads((path/'complete.json').read_text())
        if old['identity']!=expected: raise RuntimeError('Epoch cache identity differs')
        for row in old['batches']:
            if file_hash(path/row['file'])!=row['sha256']: raise RuntimeError('Damaged epoch cache')
        return
    path.mkdir(parents=True,exist_ok=True)
    canonical=copy.deepcopy(cfg)
    canonical['model']['ablation_profile']='full'
    generator=BatchChannelGenerator(canonical,torch.device('cuda'),int(cfg['seed'])+10001)
    rows=[]; count=0
    n=int(cfg['train'].get('shared_epoch_samples', cfg['train']['train_samples']))
    bs=int(cfg['train']['batch_size'])
    for index,batch in enumerate(generator.iter_batches(n,epoch,bs)):
        saved={k:batch[k].detach().cpu() for k in KEYS if k in batch}
        if index==0:
            estimate=sum(v.numel()*v.element_size() for v in saved.values())*math.ceil(n/bs)
            if shutil.disk_usage(path).free < estimate*1.15:
                raise RuntimeError(f'Need approximately {estimate/2**30:.1f} GiB free for one shared epoch')
        file=path/f'batch_{index:06d}.pt'
        save_checkpoint(saved,file)
        count+=int(saved['input'].shape[0])
        legacy={k:v for k,v in saved.items() if k!='add_input_pilot_only_fast_time'}
        rows.append(dict(file=file.name,sha256=file_hash(file),
                         tensor_sha256=tensor_hash(saved),
                         legacy_tensor_sha256=tensor_hash(legacy)))
        if index%100==0: print(f'cache epoch={epoch} frames={count}/{n}',flush=True)
    if count!=n: raise RuntimeError('Incomplete generated epoch')
    signature=digest(dict(identity=expected,tensors=[row['tensor_sha256'] for row in rows]))
    write_json(path/'complete.json',dict(identity=expected,batches=rows,frames=count,data_signature=signature))

def check_identity(destination,cfg,epoch):
    path=Path(destination)
    manifest=json.loads((path/'complete.json').read_text())
    expected=identity(cfg,epoch)
    if manifest['identity']!=expected:
        differing=identity_differences(manifest['identity'],expected)
        raise RuntimeError('Wrong shared epoch cache; '+ '; '.join(differing))
    return manifest

def read_batches(destination,cfg,epoch,device):
    path=Path(destination)
    manifest=check_identity(destination,cfg,epoch)
    requested=int(cfg['train']['train_samples'])
    unique=int(cfg['train'].get('unique_epoch_samples',requested))
    batch_size=int(cfg['train']['batch_size'])
    if not 0 < unique <= manifest['frames'] or requested < unique or requested % unique:
        raise RuntimeError('Invalid unique/shared training-sample budget')
    if batch_size <= 0:
        raise RuntimeError('Training batch size must be positive')

    # Reduced-data controls replay one prefix of the *same* epoch cache. Rebatch
    # across replay boundaries so all routes take ceil(requested/batch_size)
    # optimizer steps; a short tail must not add an extra optimizer step.
    pending=None
    emitted=0
    for _ in range(requested//unique):
        consumed=0
        for row in manifest['batches']:
            if consumed >= unique:
                break
            file=path/row['file']
            if file_hash(file)!=row['sha256']:
                raise RuntimeError(f'Corrupt batch: {file}')
            batch=torch.load(file,map_location='cpu',weights_only=True)
            available=int(batch['input'].shape[0])
            take=min(available,unique-consumed)
            if take < available:
                batch={k:v[:take] for k,v in batch.items()}
            if pending is not None:
                if batch.keys()!=pending.keys():
                    raise RuntimeError('Shared cache batch keys differ')
                batch={k:torch.cat((pending[k],v),dim=0) for k,v in batch.items()}
            consumed+=take
            while int(batch['input'].shape[0]) >= batch_size:
                ready={k:v[:batch_size] for k,v in batch.items()}
                emitted+=batch_size
                yield {k:v.to(device) for k,v in ready.items()}
                batch={k:v[batch_size:] for k,v in batch.items()}
            pending=batch if int(batch['input'].shape[0]) else None
        if consumed!=unique:
            raise RuntimeError('Cached unique sample count differs')
    if pending is not None:
        remaining=int(pending['input'].shape[0])
        emitted+=remaining
        yield {k:v.to(device) for k,v in pending.items()}
    if emitted!=requested:
        raise RuntimeError('Cached optimizer sample count differs')

if __name__=='__main__':
    from train import load_cfg, setup_backend
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True)
    ap.add_argument('--epoch',type=int,required=True); ap.add_argument('--output',required=True)
    args=ap.parse_args()
    if not torch.cuda.is_available(): raise SystemExit('Generate on CUDA server only')
    cfg=load_cfg(args.config); setup_backend(cfg,torch.device('cuda'))
    generate(cfg,args.epoch,args.output)
