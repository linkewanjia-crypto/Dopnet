"""Generate each realization once, evaluate all three training seeds on it.

Stores per-frame scores, bit counts and generation-cluster IDs, not huge waveforms.
Every completed cell is atomic and bound to code/config/checkpoint fingerprints.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from train import model_inputs
from data.generator import BatchChannelGenerator
from models.receiver import (SharedUserChannelEstimator, select_data_csi,
    linear_multiuser_equalize, banded_multiuser_equalize, ici_bands_from_dopnet_aux,
    select_data_ici_bands)
from utils.metrics import nmse_per_sample
from utils.modem import labels_to_symbols, bits_per_symbol
from utils.checkpoints import load_checkpoint
from models.pwb_linear import PWBLinear
from experiment_protocol import (SEEDS,VARIANTS,CLASSICAL,TEST_SEED,CALIBRATION_SEED,
    experiment_cells,file_hash,digest,write_json)
from reference_replay import update_waveform_digest
from dopnet_public_reference import ReferenceResults

def frame_bit_counts(z, labels, modulation):
    """Users/symbols/subcarriers stay within one independent received frame."""
    B=z.shape[0]
    bps=bits_per_symbol(modulation)
    constellation=torch.as_tensor(labels_to_symbols(np.arange(1<<bps),modulation),
                                  dtype=z.dtype,device=z.device)
    pred=(z.reshape(B,-1,1)-constellation).abs().argmin(-1)
    xor=pred ^ labels.reshape(B,-1).long()
    errors=sum(((xor>>bit)&1).sum(-1) for bit in range(bps))
    total=torch.full_like(errors,labels[0].numel()*bps)
    return errors.cpu().numpy(),total.cpu().numpy()

def calibrated_lmmse(model, cfg, device):
    calibration=copy.deepcopy(cfg)
    calibration['model']['ablation_profile']='aft_only'
    calibration['data_pipeline'].update(emit_slow_time_add=False,emit_pilot_ls=False,
                                        emit_pilot_only_fast_time_add=False)
    gen=BatchChannelGenerator(calibration,device,CALIBRATION_SEED)
    targets=[]
    for batch in gen.iter_batches(512,0,8):
        h=batch['target_h']
        targets.append(h.reshape(h.shape[0]*h.shape[1],*h.shape[2:]))
    info=model.base.calibrate(targets)
    del targets
    return str(info)

def load_routes(out, manifest, device):
    routes=[]; models=[]; cfgs=[]; hashes={}
    for variant in VARIANTS:
        for seed in SEEDS:
            name=f'{variant}_seed{seed}'
            directory=out/'checkpoints'/name
            marker=json.loads((directory/'complete.json').read_text(encoding='utf-8'))
            if marker['signature'] != manifest['signature']:
                raise RuntimeError('Checkpoint completion signature mismatch')
            path=directory/'model.pt'
            ck=load_checkpoint(path)
            cfg=ck['cfg']
            if cfg != manifest['configs'][name] or ck['selection_metric']!='val_nmse_db':
                raise RuntimeError(f'Checkpoint provenance mismatch: {name}')
            model=SharedUserChannelEstimator(cfg).to(device)
            model.load_state_dict(ck['model'],strict=True)
            model.eval()
            routes.append(dict(method=variant,training_seed=seed,trained=True))
            models.append(model); cfgs.append(cfg); hashes[name]=file_hash(path)
    base=copy.deepcopy(cfgs[0])
    for name in CLASSICAL:
        cfg=copy.deepcopy(base); cfg['model']['estimator']=name
        model=(PWBLinear(cfg) if name=='pwb_linear' else SharedUserChannelEstimator(cfg)).to(device).eval()
        if name=='lmmse_wiener': calibrated_lmmse(model,cfg,device)
        routes.append(dict(method=name,training_seed=None,trained=False))
        models.append(model); cfgs.append(cfg)
    routes.append(dict(method='perfect_diagonal_csi',training_seed=None,trained=False))
    models.append(None); cfgs.append(base)
    return routes,models,cfgs,hashes

def run(out, reference_results=None):
    device=torch.device('cuda:0')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA server required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    manifest=json.loads((out/'experiment.json').read_text(encoding='utf-8'))
    reference=ReferenceResults(reference_results) if reference_results is not None else None
    routes,models,cfgs,hashes=load_routes(out,manifest,device)
    cfg=copy.deepcopy(cfgs[0])
    cfg['model']['ablation_profile']='full'
    cfg['data_pipeline'].update(emit_pilot_ls=True,emit_slow_time_add=True)
    if any(str(model_cfg['model'].get('ablation_profile', '')).lower()
           == 'pilot_only_fast_time' for model_cfg in cfgs):
        cfg['data_pipeline']['emit_pilot_only_fast_time_add']=True
    cells,skipped=experiment_cells(cfg)
    if reference is not None:
        reference.validate_protocol(manifest,cfg,cells,skipped)
    payload=dict(experiment_signature=manifest['signature'],checkpoint_sha256=hashes,
                 routes=routes,cells=cells,skipped_infeasible=skipped,
                 test_seed=TEST_SEED,calibration_seed=CALIBRATION_SEED,
                 confidence_unit='simulation batch: shared ensemble noise reference',
                 receiver_protocols=['common_diagonal','native_own_ici_else_diagonal'],
                 note='Perfect diagonal CSI is a diagonal reference, not full-ICI oracle.')
    if reference is not None:
        payload['fixed_test_reference']=reference.provenance()
    signature=digest(payload)
    evaluation=dict(**payload,signature=signature)
    path=out/'evaluation.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=evaluation:
        raise RuntimeError('Evaluation signature changed; choose a new output directory')
    write_json(path,evaluation)
    N=int(manifest['samples_per_cell']); bs=int(manifest['batch_size']); R=len(routes)
    gen=BatchChannelGenerator(cfg,device,TEST_SEED)
    cell_dir=out/'cells'; cell_dir.mkdir(exist_ok=True)
    with torch.inference_mode():
        for number,meta in enumerate(cells):
            path=cell_dir/f"{meta['cell_id']}.npz"
            if path.exists():
                with np.load(path,allow_pickle=False) as z:
                    if z['signature'].item()!=signature or z['nmse'].shape!=(R,N):
                        raise RuntimeError(f'Invalid cached cell {path}')
                    if reference is not None:
                        reference.assert_fingerprint(meta,z['realization_sha256'].item())
                continue
            started=time.perf_counter()
            print(f"[{number+1}/{len(cells)}] {meta}",flush=True)
            arrays=dict(nmse=np.empty((R,N),dtype=np.float64))
            if meta['ber']:
                for name in ('errors_diagonal','errors_native','bits'):
                    arrays[name]=np.zeros((R,N),dtype=np.int64)
                arrays['native_residual_max']=np.zeros((R,N),dtype=np.float64)
                arrays['native_direct_fallback']=np.zeros((R,N),dtype=np.int8)
                # Receiver noise-variance knowledge sensitivity, with identical CSI/waveforms.
                noise_sweep=(meta['modulation']=='16qam' and meta['snr_db']==20 and meta['doppler_hz']==1200)
                if noise_sweep:
                    for name in ('errors_noise_minus3db','errors_noise_plus3db'):
                        arrays[name]=np.zeros((R,N),dtype=np.int64)
            else: noise_sweep=False
            native_available=np.zeros(R,dtype=np.int8)
            sample_digest=hashlib.sha256()
            for start in range(0,N,bs):
                # Semantic seed keys make resumption/order independent. Separate data RNG.
                seed=TEST_SEED+int(digest(dict(cell=meta['cell_id'],offset=start))[:15],16)
                tg=torch.Generator(device=device).manual_seed(seed)
                cond=dict(channel_type=meta['channel'],modulation=meta['modulation'],
                          delay_spread_ns=meta['delay_spread_ns'],doppler_hz=meta['doppler_hz'],
                          cp_len=meta['cp_len'],pilot_spacing=meta['pilot_spacing'])
                batch=gen.simulate(bs,cond,torch.full((bs,),meta['snr_db'],device=device),tg,seed+1000000001)
                target=select_data_csi(batch['target_h'],cfg)
                if reference is not None:
                    update_waveform_digest(sample_digest,batch)
                else:
                    digest_keys=['input','add_input','add_input_slow','rx_data',
                                 'target_h','tx_label']
                    if 'add_input_pilot_only_fast_time' in batch:
                        digest_keys.append('add_input_pilot_only_fast_time')
                    for name in digest_keys:
                        sample_digest.update(batch[name].contiguous().cpu().numpy().tobytes())
                inputs=model_inputs(batch)
                for r,(route,model,mcfg) in enumerate(zip(routes,models,cfgs)):
                    if model is None:
                        pred=batch['target_h']; aux={}
                    else:
                        pred,aux=model(batch['input'],return_aux=True,**inputs)
                    score=select_data_csi(pred,mcfg)
                    vals=nmse_per_sample(score,target).cpu().numpy()
                    if not np.isfinite(vals).all(): raise RuntimeError(f'Nonfinite NMSE: {route}')
                    arrays['nmse'][r,start:start+bs]=vals
                    if not meta['ber']: continue
                    z,_=linear_multiuser_equalize(score,batch['rx_data'],batch['noise_var'],method='mmse')
                    if not torch.isfinite(z).all(): raise RuntimeError('Nonfinite detector output')
                    errors,bits=frame_bit_counts(z,batch['tx_label'],meta['modulation'])
                    arrays['errors_diagonal'][r,start:start+bs]=errors
                    arrays['bits'][r,start:start+bs]=bits
                    bands=None
                    if model is not None and 'add_fused_pred' in aux:
                        bands=select_data_ici_bands(ici_bands_from_dopnet_aux(model,batch['input'],pred,aux,
                                                     half_band=2,replace_center=True),mcfg)
                        native_available[r]=1
                        zn,_,info=banded_multiuser_equalize(bands,batch['rx_data'],batch['noise_var'],
                                                solver='pcg',max_iter=96,tol=1e-6,return_info=True)
                        fallback=not np.isfinite(info['relative_residual_max']) or info['relative_residual_max']>1e-4
                        if fallback:
                            zn,_,info=banded_multiuser_equalize(bands,batch['rx_data'],batch['noise_var'],
                                                              solver='direct',return_info=True)
                        if not np.isfinite(info['relative_residual_max']) or info['relative_residual_max']>1e-4:
                            raise RuntimeError(f'Unresolved equalizer solve: {info}')
                        if not torch.isfinite(zn).all(): raise RuntimeError('Nonfinite native receiver')
                        native_errors,_=frame_bit_counts(zn,batch['tx_label'],meta['modulation'])
                        arrays['native_residual_max'][r,start:start+bs]=info['relative_residual_max']
                        arrays['native_direct_fallback'][r,start:start+bs]=int(fallback)
                    else: native_errors=errors
                    arrays['errors_native'][r,start:start+bs]=native_errors
                    if noise_sweep:
                        for db,name in [(-3,'errors_noise_minus3db'),(3,'errors_noise_plus3db')]:
                            # Fixed common diagonal detector isolates noise knowledge sensitivity.
                            zm,_=linear_multiuser_equalize(score,batch['rx_data'],batch['noise_var']*10**(db/10),method='mmse')
                            if not torch.isfinite(zm).all(): raise RuntimeError('Nonfinite mismatch detector')
                            arrays[name][r,start:start+bs]=frame_bit_counts(zm,batch['tx_label'],meta['modulation'])[0]
                del batch,target
            if reference is not None:
                reference.assert_fingerprint(meta,sample_digest.hexdigest())
            arrays.update(cluster_id=np.arange(N)//bs,native_available=native_available,
                          signature=np.array(signature),meta=np.array(json.dumps(meta)),
                          realization_sha256=np.array(sample_digest.hexdigest()))
            temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as f: np.savez_compressed(f,**arrays)
            temporary.replace(path)
            print(f"Saved {path.name}, {time.perf_counter()-started:.1f}s",flush=True)
    write_json(out/'evaluation_complete.json',dict(signature=signature,cells=len(cells)))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='results/dopnet')
    ap.add_argument('--reference-results',default=None,
                    help='DOPNet_Results.zip or its extracted directory for strict 636-cell replay')
    args=ap.parse_args()
    run(Path(args.output).resolve(),args.reference_results)
