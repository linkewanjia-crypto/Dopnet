"""No-network backprojection/synthesis control using the same observable ADD.

This is not a joint LS solution of the complete channel operator. Its purpose
is to separate the fixed waveform front end from learned denoising and fusion.
"""
import torch
from torch import nn
from .dopnet import DopplerPhysics, from_angle_delay
from data.add_geometry import add_delay_bins, delay_guard_bins, delay_window_indices

class PWBLinear(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg; self.N=cfg['system']['n_subcarriers']
        self.physics=DopplerPhysics(cfg,cfg['model']['add_doppler_bins'])
        self.register_buffer('delay_idx',delay_window_indices(self.N,add_delay_bins(cfg),delay_guard_bins(cfg)))
    def forward(self,x,add_input=None,return_aux=False,**kwargs):
        if add_input is None: raise ValueError('PWB-linear requires observable ADD')
        B,U,C,T,A,N=x.shape
        times=self.physics.symbol_times_from_input(x.reshape(B*U,C,T,A,N))
        obs=add_input.reshape(B*U,*add_input.shape[2:])
        coeff=torch.complex(obs[:,0],obs[:,1])
        window=self.physics.synthesize(coeff,times)
        full=window.new_zeros((*window.shape[:-1],self.N))
        full.index_copy_(-1,self.delay_idx,window)
        h=from_angle_delay(full)
        result=torch.stack([h.real,h.imag],dim=1).reshape(B,U,2,T,A,N).float()
        return (result,{}) if return_aux else result
