"""Remote fp32 eager model-stage measurements; excludes pilot preprocessing.

No real-time/end-to-end receiver claim is inferred from these measurements.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from train import model_inputs
from data.generator import BatchChannelGenerator
from models.receiver import SharedUserChannelEstimator
from utils.checkpoints import load_checkpoint
from experiment_protocol import VARIANTS, write_json

@torch.inference_mode()
def profile(out):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA server required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0'); rows=[]
    for name in VARIANTS:
        ck=load_checkpoint(out/'checkpoints'/f'{name}_seed42'/'model.pt')
        cfg=ck['cfg']; model=SharedUserChannelEstimator(cfg).to(device).eval()
        model.load_state_dict(ck['model'],strict=True)
        for B in (1,8):
            gen=BatchChannelGenerator(cfg,device,920001)
            g=torch.Generator(device=device).manual_seed(920001)
            b=gen.simulate(B,dict(channel_type='doubly_selective',modulation='qpsk',
                delay_spread_ns=300.,doppler_hz=1200.,cp_len=16,pilot_spacing=4),
                torch.full((B,),20.,device=device),g,920002)
            kw=model_inputs(b)
            for _ in range(10): model(b['input'],**kw)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            base_memory=torch.cuda.memory_allocated(); times=[]
            for _ in range(100):
                torch.cuda.synchronize(); start=time.perf_counter()
                pred=model(b['input'],**kw)
                torch.cuda.synchronize(); times.append((time.perf_counter()-start)*1000)
                del pred
            rows.append(dict(method=name,seed=42,batch_size=B,precision='fp32',execution='eager',
                warmups=10,repetitions=100,latency_ms_mean=float(np.mean(times)),
                latency_ms_p50=float(np.median(times)),latency_ms_p95=float(np.quantile(times,.95)),
                frames_per_second=1000*B/float(np.mean(times)),
                allocated_peak_mb=torch.cuda.max_memory_allocated()/2**20,
                incremental_peak_mb=(torch.cuda.max_memory_allocated()-base_memory)/2**20,
                parameters_total=sum(p.numel() for p in model.parameters()),
                parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
            del b,kw,gen
        del model,ck; gc.collect(); torch.cuda.empty_cache()
    write_json(out/'profile.json',dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        excludes=['waveform acquisition','LS/interpolation','ADD front end','equalizer','host-device transfer'],
        note='Model-stage timing only. No full receiver latency or real-time claim.',rows=rows))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='results/dopnet')
    profile(Path(ap.parse_args().output).resolve())
