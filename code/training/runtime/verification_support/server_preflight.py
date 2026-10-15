"""Fail-fast scientific checks on the remote CUDA server. Not run during delivery."""
import copy
import importlib.metadata
import os
import subprocess
import sys
import random
import numpy as np
import torch
from train import load_cfg, model_inputs, compute_loss
from data.generator import BatchChannelGenerator
from models.receiver import SharedUserChannelEstimator
from models.pwb_linear import PWBLinear
from experiment_protocol import VARIANTS, make_config

def main():
    if not torch.cuda.is_available(): raise SystemExit('Server CUDA GPU required')
    if sys.version_info<(3,11): raise SystemExit('Sionna 2.0.1 requires Python >=3.11')
    if importlib.metadata.version('sionna')!='2.0.1': raise SystemExit('Use sionna==2.0.1')
    print('torch',torch.__version__,'CUDA',torch.version.cuda,'GPU',torch.cuda.get_device_name())
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],check=True)
    for script in ('validate_banded_mmse.py','validate_physics_numerics.py','validate_waveform_consistency.py'):
        subprocess.run([sys.executable,script],check=True)
    device=torch.device('cuda:0')
    base=load_cfg('configs/dopnet.yaml')
    cfg=make_config(base,'dopnet',42)
    gen=BatchChannelGenerator(cfg,device,310001)
    def generate(channel):
        g=torch.Generator(device=device).manual_seed(410001)
        return gen.simulate(2,dict(channel_type=channel,modulation='qpsk',delay_spread_ns=300.,
            doppler_hz=1200.,cp_len=16,pilot_spacing=4),torch.full((2,),20.,device=device),g,510001)
    for channel in ('doubly_selective','tdl_b','cdl_c'):
        rng_python=random.getstate(); rng_numpy=np.random.get_state()
        rng_cpu=torch.get_rng_state().clone()
        rng_cuda=torch.cuda.get_rng_state().clone()
        a=generate(channel); b=generate(channel)
        if not torch.equal(rng_cpu,torch.get_rng_state()) or not torch.equal(rng_cuda,torch.cuda.get_rng_state()):
            raise RuntimeError(f'Simulator changed model RNG: {channel}')
        if rng_python!=random.getstate(): raise RuntimeError('Simulator changed Python RNG')
        current=np.random.get_state()
        if rng_numpy[0]!=current[0] or not np.array_equal(rng_numpy[1],current[1]) or rng_numpy[2:]!=current[2:]:
            raise RuntimeError('Simulator changed NumPy RNG')
        for key in ('input','add_input','add_input_slow','target_h','rx_data','tx_label'):
            if not torch.equal(a[key],b[key]):
                raise RuntimeError(f'Replay failed for {channel}:{key}; do not proceed to training')
        del b
        if channel=='doubly_selective':
            linear=PWBLinear(cfg).to(device)
            with torch.inference_mode():
                lp=linear(a['input'],**model_inputs(a))
                if lp.shape!=a['target_h'].shape or not torch.isfinite(lp).all():
                    raise RuntimeError('Invalid PWB-linear control')
            del linear,lp
            for variant in VARIANTS:
                mcfg=make_config(base,variant,42)
                model=SharedUserChannelEstimator(mcfg).to(device).train()
                loss,_,_,_,_=compute_loss(model,a,mcfg,'ce')
                if not torch.isfinite(loss): raise RuntimeError(f'Invalid {variant} loss')
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise RuntimeError(f'Invalid {variant} gradient')
                model.eval()
                with torch.inference_mode():
                    pred=model(a['input'],**model_inputs(a))
                    changed=dict(a,target_h=torch.randn_like(a['target_h']),target_add=torch.randn_like(a['target_add']))
                    again=model(changed['input'],**model_inputs(changed))
                    if not torch.equal(pred,again): raise RuntimeError(f'Target leakage: {variant}')
                del model,loss
        del a
    print('SERVER PREFLIGHT PASSED (numerics, replay, all variants, target-input separation)')

if __name__=='__main__': main()
