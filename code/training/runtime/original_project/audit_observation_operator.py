"""Actual implemented PWB observation-map diagnostics, executed on server only.

This is an input-representation map, NOT the physical joint channel forward
operator or a proof of channel identifiability. It includes comb correlation,
occupied-band IFFT, whitening and delay cropping from the production function.
"""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from train import load_cfg
from data.waveform import _build_frame, _add_backprojection, pilot_amplitude
from data.add_geometry import add_delay_bins, delay_guard_bins, delay_window_indices
from data.channel_backend import cp_active_to_waveform_samples
from data.frame_feasibility import feasible_pilot_spacings
from experiment_protocol import write_json

@torch.inference_mode()
def audit(output):
    cfg=copy.deepcopy(load_cfg('configs/dopnet.yaml'))
    cfg['system']['n_ant']=1
    device=torch.device('cuda:0'); s=cfg['system']; N=s['n_subcarriers']; U=s['n_users']
    delay=delay_window_indices(N,add_delay_bins(cfg),delay_guard_bins(cfg),device)
    rows=[]
    for cp in s['cp_len_choices']:
        for spacing in feasible_pilot_spacings(cfg,cp):
            g=torch.Generator(device=device).manual_seed(17001)
            frame,layout,_,_,_,_,_=_build_frame(1,U,N,spacing,'qpsk',device,g,
                s['frame_num_symbols'],s['pilot_symbol_period'],amplitude=pilot_amplitude(cfg,spacing,U))
            pb=torch.tensor(layout.pilot_blocks,device=device)
            pilot=frame[:,:,pb,:]
            own=pilot[0,0].abs().reshape(-1)>0
            joint=pilot[0].abs().sum(0).reshape(-1)>0
            M=len(layout.pilot_blocks)*N
            cpw=cp_active_to_waveform_samples(cp,s['waveform_fft_size'],N)
            for fast in (True,False):
                pieces=[]
                for start in range(0,M,16):
                    n=min(16,M-start)
                    impulses=torch.zeros((n,M),dtype=torch.complex64,device=device)
                    impulses[torch.arange(n,device=device),torch.arange(start,start+n,device=device)]=1
                    Y=torch.zeros((n,1,layout.num_blocks,N),dtype=torch.complex64,device=device)
                    Y[:,:,pb,:]=impulses.reshape(n,1,len(pb),N)
                    obs=_add_backprojection(cfg,Y,frame.expand(n,-1,-1,-1),layout,cpw,delay,fast_time=fast)
                    z=torch.complex(obs[:,0,0],obs[:,0,1]).reshape(n,-1)
                    pieces.append(z.T)
                A=torch.cat(pieces,dim=1).to(torch.complex128)
                sv=torch.linalg.svdvals(A)
                rank=int((sv>sv[0]*1e-6).sum())
                energy=A.abs().square().sum().clamp_min(1e-30)
                rows.append(dict(cp=cp,pilot_spacing=spacing,fast_time=fast,user=0,
                    input_complex_dimension=M,output_complex_dimension=A.shape[0],
                    effective_rank_rtol_1e6=rank,
                    retained_singular_condition=float(sv[0]/sv[rank-1]),
                    energy_fraction_outside_own_pilot_tones=float(A[:,~own].abs().square().sum()/energy),
                    energy_fraction_outside_joint_pilot_tones=float(A[:,~joint].abs().square().sum()/energy),
                    singular_values=sv.cpu().tolist()))
                del A,pieces
    write_json(output,dict(interpretation='Implemented linear map from received occupied pilot-symbol bins to user-0 ADD. Not the channel forward model.',
        warning='Both representations are deterministic functions of the same received frame. At fully occupied joint pilots, no extra raw measurements are created.',rows=rows))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='results/operator_audit.json')
    audit(Path(ap.parse_args().output))
