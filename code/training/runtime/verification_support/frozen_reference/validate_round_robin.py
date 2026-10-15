"""REMOTE ONLY: continuous online training versus cached, interrupted epochs.

Small synthetic regression, NOT paper results or full-run convergence evidence.
Tests optimizer/scheduler/scaler and Torch RNG continuation.
"""
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
import numpy as np
import yaml
from train import load_cfg
from data.generator import BatchChannelGenerator
from epoch_cache import KEYS, tensor_hash
from experiment_protocol import make_config, VARIANTS
from utils.checkpoints import load_checkpoint

def run(*args):
    env=os.environ.copy()
    env['DOPNET_DETERMINISTIC_TEST']='1'
    env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    subprocess.run([sys.executable,*map(str,args)],check=True,env=env)

def compare(a,b,path='state'):
    if isinstance(a,np.ndarray):
        if not np.array_equal(a,b): raise AssertionError(path)
    elif torch.is_tensor(a):
        if a.is_floating_point():
            try:
                torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7,msg=path)
            except AssertionError as error:
                delta=(a.float()-b.float()).abs()
                scale=torch.maximum(a.float().abs(),b.float().abs()).clamp_min(1e-30)
                raise AssertionError(
                    f'{path}: max_abs={delta.max().item():.9g}, '
                    f'max_rel={(delta/scale).max().item():.9g}\n{error}') from error
        elif not torch.equal(a,b): raise AssertionError(path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys(): raise AssertionError(path)
        for k in a: compare(a[k],b[k],f'{path}.{k}')
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b): raise AssertionError(path)
        for i,(x,y) in enumerate(zip(a,b)): compare(x,y,f'{path}.{i}')
    elif a!=b: raise AssertionError(f'{path}: {a} != {b}')

def main():
    if not torch.cuda.is_available(): raise SystemExit('Run only on CUDA server')
    root=Path(__file__).resolve().parent; os.chdir(root)
    base=load_cfg(root/'configs/dopnet.yaml')
    # Same physical grid but only 8 frames/epoch and 8 validation frames.
    base['system'].update(channel_types=['doubly_selective'],channel_type_weights=[1.])
    base['train'].update(epochs=2,train_samples=8,val_samples=8,validation_cache=False)
    with tempfile.TemporaryDirectory(prefix='dopnet_roundtrip_') as temporary:
        tmp=Path(temporary); dopnet=make_config(base,'dopnet',42)
        config_path=tmp/'dopnet.yaml'; config_path.write_text(yaml.safe_dump(dopnet),encoding='utf-8')
        for epoch in (1,2):
            run('epoch_cache.py','--config',config_path,'--epoch',epoch,'--output',tmp/f'cache{epoch}')
            import json
            manifest=json.loads((tmp/f'cache{epoch}'/'complete.json').read_text())
            generator=BatchChannelGenerator(dopnet,torch.device('cuda'),dopnet['seed']+10001)
            observed=[]
            for batch in generator.iter_batches(dopnet['train']['train_samples'],epoch,
                                                  dopnet['train']['batch_size']):
                observed.append(tensor_hash({k:batch[k].detach().cpu()
                                             for k in KEYS if k in batch}))
            expected=[row['tensor_sha256'] for row in manifest['batches']]
            if observed!=expected:
                raise AssertionError(f'epoch {epoch}: online/cache data differ')
            print(f'PASS epoch {epoch}: online/cache tensors are bit-identical',flush=True)
        for variant in VARIANTS:
            cfg=make_config(base,variant,42)
            config=tmp/f'{variant}.yaml'; config.write_text(yaml.safe_dump(cfg),encoding='utf-8')
            online=tmp/f'{variant}_online'; rotated=tmp/f'{variant}_rotated'
            common=['train.py','--config',config,'--seed',42,'--device','cuda']
            subset = int(cfg['train']['unique_epoch_samples']) < int(cfg['train']['shared_epoch_samples'])
            if not subset:
                run(*common,'--run-name',online)
            for epoch in (1,2):
                resume=['--resume',rotated/'last.pt'] if epoch>1 else []
                run(*common,'--run-name',rotated,'--epoch-cache',tmp/f'cache{epoch}',
                    '--stop-after-epoch',epoch,*resume)
            b=load_checkpoint(rotated/'last.pt')
            if subset:
                # Reduced-data routes replay the first N unique frames until
                # reaching the full optimizer-step budget. A separately
                # generated N-frame online batch would consume RNG differently.
                from epoch_cache import read_batches
                for epoch in (1,2):
                    batches=list(read_batches(tmp/f'cache{epoch}',cfg,epoch,torch.device('cpu')))
                    if sum(int(batch['input'].shape[0]) for batch in batches) != cfg['train']['train_samples']:
                        raise AssertionError(f'{variant}: replayed sample count differs')
                    expected_steps=(int(cfg['train']['train_samples'])+
                                    int(cfg['train']['batch_size'])-1)//int(cfg['train']['batch_size'])
                    if len(batches)!=expected_steps:
                        raise AssertionError(f'{variant}: optimizer step count differs')
                if b['epoch'] != 2:
                    raise AssertionError(f'{variant}: interrupted training did not resume')
                print(f'PASS {variant}: prefix replay, equal step count and two-epoch resume',flush=True)
                continue
            a=load_checkpoint(online/'last.pt')
            for key in ('model','optimizer','scheduler','scaler','rng_python','rng_numpy',
                        'rng_torch','rng_cuda','epoch'):
                compare(a[key],b[key],f'{variant}.{key}')
            print(f'PASS {variant}: online continuous versus cached + resume',flush=True)
        print('ROUND-ROBIN REGRESSION PASSED; tolerance rtol=1e-5, atol=1e-7')

if __name__=='__main__': main()
