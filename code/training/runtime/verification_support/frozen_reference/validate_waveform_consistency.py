"""Waveform/target consistency test for the DOP-Net generator."""
import copy
import torch
from train import load_cfg, assert_current_config
from data.waveform import simulate_synthetic_batch
from data.add_geometry import add_delay_bins

@torch.no_grad()
def main():
    cfg=assert_current_config(load_cfg("configs/dopnet.yaml"))
    # Avoid Sionna dependency here; test the five synthetic families.
    tests=[("awgn",0.0),("rayleigh",0.0),("rician",0.0),("nakagami",0.0),("doubly_selective",1200.0)]
    for i,(ch,dop) in enumerate(tests):
        g=torch.Generator(device="cpu"); g.manual_seed(1234+i)
        b=simulate_synthetic_batch(cfg,2,channel_type=ch,modulation="qpsk",pilot_spacing=4,cp_len=32,delay_spread_ns=300.0,doppler_hz=dop,snr_db=torch.full((2,),120.0),device=torch.device("cpu"),gen=g)
        assert torch.isfinite(b["input"]).all() and torch.isfinite(b["target_h"]).all()
        assert b["input"].shape[2]==5 and b["target_h"].ndim==6
        assert "add_input" in b and "target_add" in b
        q=cfg["model"]["add_doppler_bins"]
        assert tuple(b["add_input"].shape[2:])==(2,q,cfg["system"]["n_ant"],add_delay_bins(cfg)), b["add_input"].shape
        assert b["add_input"].shape==b["target_add"].shape
        print(f"{ch:18s} PASS input={tuple(b['input'].shape)} add={tuple(b['add_input'].shape)}")
    print("DOP-Net waveform/target consistency: PASS")
if __name__=="__main__": main()
