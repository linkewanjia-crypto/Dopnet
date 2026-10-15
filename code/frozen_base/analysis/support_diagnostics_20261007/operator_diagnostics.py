"""Fixed-pilot operator diagnostics; no training, model inference or new frames.

All pilot sequences are read from the archived 4090 observation NPZ files.
The physical finite dictionary is the immutable joint BEM implementation.
Noise whitening uses a thin SVD, with all three preregistered rank thresholds.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import torch  # initialize its OpenMP runtime first, as in the accepted local replay
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
import numpy as np
import scipy.linalg as la

import csv, hashlib, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PROJECT = ROOT / "code/training/main"
OUT = HERE
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'classical_extension'))
import archived_inputs as inputs
from archived_inputs import raw_batches, geometry
from data.waveform import _add_backprojection, _doppler_whitener
from data.add_geometry import doppler_grid_hz, pilot_sample_times_s, delay_window_indices
from data.frame_layout import make_layout
from dopnet_classical.bem import JointBEMOperator, BEMSettings, METHODS
import yaml

RANK_RCONDS = (1e-8, 1e-6, 1e-4)
REFERENCE_RELATIVE_TOLERANCE = 1e-4

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False), encoding='utf-8')

def sv_summary(s):
    largest = float(s[0]) if len(s) else 0.
    return {'largest': largest,
            'frobenius_squared': float(np.sum(s*s)),
            'effective_ranks': {str(r): int(np.sum(s > largest*r)) for r in RANK_RCONDS},
            'effective_condition_numbers': {
                str(r): float(largest/s[np.flatnonzero(s > largest*r)[-1]])
                if largest and np.any(s > largest*r) else None for r in RANK_RCONDS}}

def fixed_map(cfg, x, cp_wave, fast=True):
    """Dense per-receive-branch PWB map using the original rounded parameters.

    Input order P,N; output order U,Q,L. Spatial FFT is applied separately.
    Matrix algebra uses complex128. grid/times/phase/R are the original
    implementation's CPU parameters; R and phase are quantized to complex64.
    """
    users, p, n = x.shape
    q = int(cfg['model']['add_doppler_bins'])
    l = 48
    layout = make_layout(25, 4)
    times = pilot_sample_times_s(cfg, layout, cp_wave, fast_time=fast)
    grid = doppler_grid_hz(cfg, cp_wave)
    phase = torch.exp(-1j * 2. * np.pi * grid[None, :, None] * times[:, None, :])
    phase = phase.to(torch.complex64).numpy().astype(np.complex128)
    r = _doppler_whitener(cfg, layout, cp_wave, fast, torch.device('cpu'))
    r = r.numpy().astype(np.complex128)
    f = np.fft.fft(np.eye(n), axis=0, norm='ortho')
    d = f.conj().T[(np.arange(l)-8) % n]
    energy = np.sum(np.abs(x.astype(np.complex128))**2, axis=(1,2))
    a = np.empty((users,q,l,p,n), dtype=np.complex128)
    for pi in range(p):
        for qi in range(q):
            if fast:
                uq = (f * phase[pi,qi][None,:]) @ f.conj().T
            else:
                uq = np.eye(n, dtype=np.complex128) * phase[pi,qi,0]
            for ui in range(users):
                a[ui,qi,:,pi,:] = (d * x[ui,pi].conj()[None,:]) @ uq * np.sqrt(n)/energy[ui]
    unconditioned = a.reshape(users,q,l,p*n)
    conditioned = np.einsum('aq,uqlk->ualk', r, unconditioned, optimize=True)
    return conditioned.reshape(users*q*l,p*n), unconditioned.reshape(users*q*l,p*n), r

def fft_reference(cfg, x, y, cp_wave, fast=True, masked=False):
    """Execute only the immutable deterministic PWB preprocessing, no network."""
    layout = make_layout(25,4)
    yt = torch.zeros((1,y.shape[1],27,128),dtype=torch.complex64)
    xt = torch.zeros((1,2,27,128),dtype=torch.complex64)
    yt[:,:,list(layout.pilot_blocks)] = torch.from_numpy(y.transpose(1,0,2).copy())[None]
    xt[:,:,list(layout.pilot_blocks)] = torch.from_numpy(x.copy())[None]
    with torch.no_grad():
        z = _add_backprojection(cfg,yt,xt,layout,cp_wave,
                delay_window_indices(128,48,8),fast_time=fast,
                pilot_only_received=masked)
    z = z.numpy()[0]
    return z[:,0].astype(np.float64)+1j*z[:,1].astype(np.float64)

def apply_dense(t, y):
    r = y.shape[1]
    z = (t @ y.transpose(0,2,1).reshape(896,r)).reshape(2,15,48,r)
    return np.fft.fft(z.transpose(0,1,3,2),axis=2,norm='ortho')

def fft_high_precision_reference(cfg, x, y, cp_wave, fast=True, masked=False):
    """Independent FFT path for the same fixed rounded linear operator.

    Retains the implementation's complex64 phase and preconditioner values,
    but performs subsequent arithmetic in complex128. This validates the
    dense formula; it is not an RTX4090 inference-equivalence check.
    """
    layout = make_layout(25,4)
    times = pilot_sample_times_s(cfg,layout,cp_wave,fast_time=fast)
    grid = doppler_grid_hz(cfg,cp_wave)
    phase = torch.exp(-1j*2.*np.pi*grid[None,:,None]*times[:,None,:])
    phase = phase.to(torch.complex64).numpy().astype(np.complex128)
    r = _doppler_whitener(cfg,layout,cp_wave,fast,torch.device('cpu'))
    r = r.numpy().astype(np.complex128)
    xf = x.astype(np.complex128)
    yf = y.astype(np.complex128)
    if masked:
        yf = yf * np.any(x != 0,axis=0)[:,None,:]
    yt = np.fft.ifft(yf,axis=-1,norm='ortho')
    yf = np.fft.fft(yt[:,None,:,:]*phase[:,:,None,:],axis=-1,norm='ortho')
    cross = np.einsum('pqrn,upn->uqrn',yf,xf.conj(),optimize=True)
    corr = np.fft.ifft(cross,axis=-1,norm='ortho')
    energy = np.sum(np.abs(np.fft.ifft(xf,axis=-1,norm='ortho'))**2,axis=(1,2))
    corr = corr * np.sqrt(128) / energy[:,None,None,None]
    corr = corr[:,:,:,(np.arange(48)-8)%128]
    corr = np.einsum('aq,uqrn->uarn',r,corr,optimize=True)
    return np.fft.fft(corr,axis=2,norm='ortho')

def noise_stats(t):
    c = t @ t.conj().T
    diag = np.maximum(np.diag(c).real,0)
    total = float(np.sum(diag))
    diagonal_energy = float(np.sum(diag*diag))
    full_energy = float(np.sum(np.abs(c)**2))
    return {'trace_sigma2_one': total,
            'maximum_variance_sigma2_one': float(diag.max()),
            'minimum_variance_sigma2_one': float(diag.min()),
            'off_diagonal_covariance_energy_fraction':
                (full_energy-diagonal_energy)/full_energy if full_energy else 0.,
            'trace_frobenius_relative_error':
                abs(total-float(np.sum(np.abs(t)**2)))/max(total,1e-30),
            'hermitian_relative_error':
                float(la.norm(c-c.conj().T)/max(la.norm(c),1e-30))}

def main():
    import argparse
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=inputs.DATA_ROOT)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--plots', action='store_true', help='Optional diagnostic figures; manuscript Fig. S-3 has a separate generator.')
    args = parser.parse_args()
    inputs.configure(args.data_root)
    OUT = inputs.new_output(args.output_dir, inputs.DATA_ROOT)
    started = time.perf_counter()
    cfg_path = PROJECT / 'configs/dopnet.yaml'
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    write_json('protocol.json', {
        'scope':'fixed saved pilots and deterministic operators, not new performance evaluation',
        'analysis_script_sha256':digest(Path(__file__)),
        'no_training':True,'no_neural_network_inference':True,'no_new_frames':True,
        'matrix_precision':'complex128; original complex64 phase/preconditioner parameters',
        'rank_relative_thresholds':RANK_RCONDS,
        'reference_relative_tolerance':REFERENCE_RELATIVE_TOLERANCE,
        'representative_selection':'first observed frame per (CP,pilot_spacing), independent of scores',
        'physical_dictionary':'existing JointBEMOperator OCE Q15, signed delay [-8,39], physical FFT1024',
        'source_files':{(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else 'data-root/'+p.name):digest(p) for p in [cfg_path,
             PROJECT/'data/waveform.py',PROJECT/'data/add_geometry.py',
             PROJECT/'classical_extension/dopnet_classical/bem.py',
             inputs.DATA_ROOT/'inventory.json']},
        'interpretation':'rank thresholds describe finite dictionary / retained subspaces; no general identifiability or NMSE guarantee'})
    cases = {}
    for role, entry, a in raw_batches():
        meta = json.loads(str(a['metadata'].item()))
        spacing = int(meta['condition']['pilot_spacing'])
        cp_wave = int(a['cp_wave_samples'])
        key = (cp_wave//8,spacing)
        if key not in cases:
            cases[key] = {'role':role,'entry':entry,'metadata':meta,
                          'x':a['x_pilot'][0].copy(),'y':a['y_pilot'][0].copy()}
    all_results, rows = [], []
    spectra = {}
    for ci, (key, case) in enumerate(sorted(cases.items())):
        cp, spacing = key
        name = f'cp{cp}_spacing{spacing}'
        t0 = time.perf_counter()
        print(f'START {name} {ci+1}/{len(cases)}',flush=True)
        x,y = case['x'],case['y']
        mask = np.any(x != 0,axis=0).reshape(-1)
        full, pre, r = fixed_map(cfg,x,cp*8,True)
        slow, slow_pre, slow_r = fixed_map(cfg,x,cp*8,False)
        extra = full[:,~mask]
        masked = full[:,mask]
        g = geometry(1024,128,cp*8)
        physical = JointBEMOperator(g,BEMSettings(method=METHODS[0],q=15,
                                    delay_start=-8,delay_stop=40),2)
        d = physical.design_matrix(x)
        result = {'case':name,'cp_active_samples':cp,'spacing':spacing,
           'source_role':case['role'],'source_sha256':case['entry']['sha256'],
           'selection_metadata':case['metadata'],
           'raw_observations':896,'masked_observations':int(mask.sum()),
           'dictionary_coefficients':1440,
           's2_map_max_difference':float(np.max(np.abs(full[:,~mask]))) if np.any(~mask) else 0.,
           'slow_extra_max_absolute':float(np.max(np.abs(slow[:,~mask]))) if np.any(~mask) else 0.,
           'q0_preconditioner_extra_max_absolute':
                float(np.max(np.abs(pre.reshape(2,15,48,896)[:,7,:,~mask]))) if np.any(~mask) else 0.,
           'preconditioner_condition_number':float(np.linalg.cond(r)),
           'noise':{},'spectra':{},'whitened_dictionary':{},'reference_checks':{},
           'high_precision_fft_checks':{}}
        for label,t in [('full',full),('masked',masked),('extra',extra),('slow',slow[:,mask])]:
            if t.shape[1]==0:
                result['noise'][label] = {'trace_sigma2_one':0.}
                result['spectra'][label] = sv_summary(np.array([]))
                continue
            result['noise'][label] = noise_stats(t)
            _,s,vh = la.svd(t,full_matrices=False,check_finite=False,lapack_driver='gesdd')
            result['spectra'][label]=sv_summary(s)
            if label in ('full','masked'):
                corresponding_d = d if label=='full' else d[mask]
                for rc in RANK_RCONDS:
                    rank=int(np.sum(s>s[0]*rc))
                    effective=vh[:rank] @ corresponding_d
                    ds=la.svdvals(effective,check_finite=False)
                    item=sv_summary(ds)
                    item['retained_map_rank']=rank
                    item['dictionary_energy_capture_fraction']=float(np.sum(np.abs(effective)**2)/np.sum(np.abs(corresponding_d)**2))
                    result['whitened_dictionary'][f'{label}_{rc}']=item
                    if rc==1e-6:
                        spectra[(name,label+'_whitened')]=ds
                        np.save(OUT/f'{name}_{label}_whitened_singular_values.npy',ds)
            spectra[(name,label)]=s
            np.save(OUT/f'{name}_{label}_singular_values.npy',s)
            del vh
        for label,dt in [('raw_full',d),('raw_masked',d[mask])]:
            ds=la.svdvals(dt,check_finite=False)
            result['spectra'][label]=sv_summary(ds)
            spectra[(name,label)]=ds
        noise_full=result['noise']['full']['trace_sigma2_one']
        noise_mask=result['noise']['masked']['trace_sigma2_one']
        noise_extra=result['noise']['extra']['trace_sigma2_one']
        result['noise_partition_relative_error']=abs(noise_full-noise_mask-noise_extra)/max(noise_full,1e-30)
        result['extra_noise_fraction']=noise_extra/noise_full
        for label,t,fast,masked_flag in [('full',full,True,False),
                ('masked',full*mask[None,:],True,True),('slow',slow,False,False)]:
            z=apply_dense(t,y)
            ref=fft_reference(cfg,x,y,cp*8,fast,masked_flag)
            rel=float(la.norm(z-ref)/max(la.norm(ref),1e-30))
            result['reference_checks'][label]={'relative_error':rel,
                'maximum_absolute_error':float(np.max(np.abs(z-ref))),
                'pass':rel<=REFERENCE_RELATIVE_TOLERANCE,
                'reference':'unchanged original CPU complex64 PWB function; not a new GPU execution'}
            hp = fft_high_precision_reference(cfg,x,y,cp*8,fast,masked_flag)
            hp_rel = float(la.norm(z-hp)/max(la.norm(hp),1e-30))
            result['high_precision_fft_checks'][label] = {
                'relative_error':hp_rel,'pass':hp_rel < 1e-10,
                'reference':'independent complex128 FFT path with identical rounded phase and R'}
        result['elapsed_seconds']=time.perf_counter()-t0
        all_results.append(result)
        row={'case':name,'cp':cp,'spacing':spacing,'raw_rows':896,
             'mask_rows':int(mask.sum()),'full_map_rank_1e6':result['spectra']['full']['effective_ranks']['1e-06'],
             'masked_map_rank_1e6':result['spectra']['masked']['effective_ranks']['1e-06'],
             'extra_noise_fraction':result['extra_noise_fraction'],
             'noise_correlation_fraction_full':result['noise']['full']['off_diagonal_covariance_energy_fraction'],
             'whitened_full_rank_1e6':result['whitened_dictionary']['full_1e-06']['effective_ranks']['1e-06'],
             'whitened_mask_rank_1e6':result['whitened_dictionary']['masked_1e-06']['effective_ranks']['1e-06'],
             'noise_partition_error':result['noise_partition_relative_error'],
             'maximum_fast_time_reference_relative_error':max(result['reference_checks'][v]['relative_error'] for v in ('full','masked')),
             'slow_reference_relative_error':result['reference_checks']['slow']['relative_error']}
        rows.append(row)
        write_json('operator_results.json',all_results)
        print(f'DONE {name}: {result["elapsed_seconds"]:.2f}s, ranks {row["full_map_rank_1e6"]}/{row["masked_map_rank_1e6"]}, extra_noise {row["extra_noise_fraction"]:.5f}',flush=True)
        del full,masked,extra,slow,pre,slow_pre,d,physical
    with (OUT/'operator_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
        for ax,spacing in zip(axs,[2,4,8]):
            name=f'cp16_spacing{spacing}'
            for label,color,style in [('raw_full','#1565c0','-'),('raw_masked','#e65100','--')]:
                s=spectra[(name,label)]
                ax.semilogy(np.arange(1,len(s)+1),np.maximum(s/s[0],1e-14),style,color=color,label=label.replace('_',' '))
            ax.set(title=f'CP16, spacing {spacing}',xlabel='Singular-value index',ylabel='Normalized singular value',ylim=(1e-12,2))
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.savefig(OUT/'physical_dictionary_spectra.png',dpi=180)
        fig.savefig(OUT/'physical_dictionary_spectra.pdf');plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
        for cp in [8,16,32]:
            selected=[r for r in rows if r['cp']==cp]
            axs[0].plot([r['spacing'] for r in selected],[100*r['extra_noise_fraction'] for r in selected],marker='o',label=f'CP {cp}')
            axs[1].plot([r['spacing'] for r in selected],[100*r['noise_correlation_fraction_full'] for r in selected],marker='o',label=f'CP {cp}')
        axs[0].set(xlabel='Pilot spacing',ylabel='Extra-support noise / full PWB noise (%)')
        axs[1].set(xlabel='Pilot spacing',ylabel='Off-diagonal covariance energy (%)')
        for ax in axs:ax.grid(alpha=.2);ax.legend()
        fig.savefig(OUT/'pwb_noise_geometry.png',dpi=180)
        fig.savefig(OUT/'pwb_noise_geometry.pdf');plt.close(fig)
    checks={'no_training':True,'no_neural_inference':True,'no_new_frames':True,
       'all_original_cpu_reference_checks_pass':all(v['pass'] for r in all_results for v in r['reference_checks'].values()),
       'all_primary_fast_time_cpu_reference_checks_pass':all(r['reference_checks'][v]['pass'] for r in all_results for v in ('full','masked')),
       'all_dense_formula_high_precision_checks_pass':all(v['pass'] for r in all_results for v in r['high_precision_fft_checks'].values()),
       'excluded_from_primary_numerical_conclusions':['slow operator spectra and covariance'],
       'slow_reference_exclusion_reason':'Slow-time complex64 FFT path differs from the dense formula above the declared 1e-4 tolerance; the measured increase in pre/post-R error is consistent with rounding amplification by the ill-conditioned preconditioner, without uniquely isolating all rounding contributions. Original failure values are retained. Only structural zero extra-support is used.',
       'all_s2_maps_identical':all(r['s2_map_max_difference']==0 for r in all_results if r['spacing']==2),
       'all_noise_partition_checks_pass':all(r['noise_partition_relative_error']<1e-12 for r in all_results),
       'slow_extra_max_absolute':max(r['slow_extra_max_absolute'] for r in all_results),
       'q0_pre_extra_max_absolute':max(r['q0_preconditioner_extra_max_absolute'] for r in all_results),
       'case_count':len(all_results),'elapsed_seconds':time.perf_counter()-started,
       'hardware_equivalence_scope':'saved pilot values unchanged; dense diagnostics are new CPU calculations, not claimed bitwise GPU outputs',
       'interpretation_limits':[
           'Finite OCE dictionary includes signed guard delay atoms; it is not a physical path identification theorem.',
           'Effective rank depends on the declared numerical threshold; all three thresholds are retained.',
           'Noise covariance is exact for the fixed rounded linear map conditional on supplied white-bin noise variance.',
           'Preconditioning cannot create observations or guarantee learned NMSE improvements.',
           'One first observed pilot pattern per geometry is a deterministic representative, not a distributional average.']}
    write_json('verification_summary.json',checks)
    print(json.dumps(checks,ensure_ascii=False),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
