"""Seed-t and paired cluster-bootstrap uncertainty; no inference/training here.

The seed-t interval is conditional on the shared test realization. The cluster
bootstrap interval is conditional on the three trained checkpoints. These are
different uncertainty sources and must never be labelled as interchangeable.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t
from experiment_protocol import SEEDS, STATIC, TRAIN_GPP, HELDOUT, digest, file_hash, write_json

def seed_interval(values, confidence=.95):
    x=np.asarray(values,dtype=float)
    if x.shape!=(3,) or not np.isfinite(x).all():
        raise ValueError('Exactly three finite training-seed estimates required')
    mean=float(x.mean()); sd=float(x.std(ddof=1))
    half=float(t.ppf((1+confidence)/2,2))*sd/math.sqrt(3)
    return mean,sd,mean-half,mean+half

def db(value):
    return 10*math.log10(value) if value is not None and value>0 else None

def csv_write(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def scopes(meta):
    ch=meta['channel']; out=['all']
    if ch in STATIC: out+=['static']
    else: out+=['dynamic_all']
    if ch=='doubly_selective': out+=['dynamic_synthetic']
    if ch in TRAIN_GPP+HELDOUT:
        out+=['gpp_all','gpp_seen' if ch in TRAIN_GPP else 'gpp_heldout']
        if meta['doppler_hz']>=800: out+=['gpp_high_mobility']
    return out

def summarize(points, boots, routes, context, metric):
    by_method=defaultdict(list)
    for i,r in enumerate(routes): by_method[r['method']].append(i)
    result=[]; differences=[]
    fi=by_method['dopnet']
    if [routes[i]['training_seed'] for i in fi]!=list(SEEDS): raise ValueError('DOP-Net seeds mismatch')
    for method,idx in by_method.items():
        seeds=[routes[i]['training_seed'] for i in idx]
        trained=routes[idx[0]]['trained']
        values=points[idx]
        if trained:
            if seeds!=list(SEEDS): raise ValueError(f'Seeds missing/duplicated for {method}')
            mean,sd,lo,hi=seed_interval(values)
        else: mean=float(values[0]); sd=lo=hi=None
        test_boot=boots[:,idx].mean(axis=1)
        test_lo,test_hi=np.quantile(test_boot,[.025,.975])
        row=dict(**context,metric=metric,method=method,n_training_seeds=3 if trained else 0,
                 mean=mean,seed_sd=sd,seed_ci95_lo_unbounded=lo,seed_ci95_hi_unbounded=hi,
                 seed_ci95_lo=max(0,lo) if lo is not None else None,
                 seed_ci95_hi=min(1,hi) if hi is not None and metric.startswith('ber') else hi,
                 test_cluster_ci95_lo=float(test_lo),test_cluster_ci95_hi=float(test_hi),
                 ci_note='seed_t_df2 conditional on test set; bootstrap conditional on checkpoints',
                 zero_error_warning=bool(mean==0 and metric.startswith('ber')))
        for s,v in zip(seeds,values):
            if s is not None: row[f'seed{s}']=float(v)
        if metric=='nmse':
            row.update(mean_db=db(mean),seed_ci95_lo_db=db(max(0,lo)) if lo is not None else None,
                       seed_ci95_hi_db=db(hi),test_cluster_ci95_lo_db=db(test_lo),
                       test_cluster_ci95_hi_db=db(test_hi))
        if mean==0 and metric.startswith('ber') and 'n_test_clusters' in context:
            # One-sided bound for the probability of any error in a whole IID
            # generation cluster. This conservatively bounds its mean BER.
            row['zero_error_cluster_event_upper95']=1-.05**(1/context['n_test_clusters'])
        result.append(row)
        if method in ('dopnet','perfect_diagonal_csi'): continue
        # Positive paired difference: comparator worse than DOP-Net. Pair seed s with s.
        delta=values-points[fi] if trained else values[0]-points[fi]
        dm,ds,dl,dh=seed_interval(delta)
        bdelta=boots[:,idx].mean(axis=1)-boots[:,fi].mean(axis=1)
        blo,bhi=np.quantile(bdelta,[.025,.975])
        drow=dict(**context,metric=metric,comparator=method,reference='dopnet',
                  paired_difference=dm,paired_seed_ci95_lo=dl,paired_seed_ci95_hi=dh,
                  paired_test_cluster_ci95_lo=float(blo),paired_test_cluster_ci95_hi=float(bhi),
                  interpretation='positive = comparator worse; exploratory pointwise intervals')
        if metric=='nmse' and mean>0 and np.all(points[fi]>0) and np.all(values>0):
            gaps=10*np.log10(values/points[fi]) if trained else 10*np.log10(values[0]/points[fi])
            gm,gs,gl,gh=seed_interval(gaps)
            drow.update(mean_paired_gap_db=gm,paired_gap_db_seed_ci95_lo=gl,paired_gap_db_seed_ci95_hi=gh)
        differences.append(drow)
    return result,differences

def aggregate(out, repetitions):
    if repetitions<1000: raise ValueError('Use at least 1000 bootstrap replicates')
    ev=json.loads((out/'evaluation.json').read_text(encoding='utf-8'))
    done=json.loads((out/'evaluation_complete.json').read_text(encoding='utf-8'))
    if done!={'signature':ev['signature'],'cells':len(ev['cells'])}:
        raise RuntimeError('Evaluation is incomplete; no partial headline tables allowed')
    routes=ev['routes']; R=len(routes)
    cell_rows=[]; paired_rows=[]; diagnostics=[]
    cell_file_sha256={}
    buckets={}
    def add_bucket(key,channel,point,boot):
        b=buckets.setdefault(key,{})
        if channel not in b: b[channel]=[point.copy(),boot.copy(),1]
        else:
            b[channel][0]+=point; b[channel][1]+=boot; b[channel][2]+=1
    for meta in ev['cells']:
        path=out/'cells'/f"{meta['cell_id']}.npz"
        cell_file_sha256[meta['cell_id']]=file_hash(path)
        with np.load(path,allow_pickle=False) as z:
            if z['signature'].item()!=ev['signature']: raise RuntimeError(f'Stale cell: {path}')
            if json.loads(z['meta'].item())!=meta: raise RuntimeError('Cell metadata differs')
            cluster=z['cluster_id']; ids=np.unique(cluster); C=len(ids)
            if C<32: raise RuntimeError('Too few independent generation clusters')
            rng=np.random.default_rng(int(digest(meta)[:16],16))
            # The exact same resampling weights are used for every method/seed.
            weights=rng.multinomial(C,np.full(C,1/C),size=repetitions)/C
            metrics={'nmse':z['nmse']}
            if meta['ber']:
                for key in ('diagonal','native','noise_minus3db','noise_plus3db'):
                    if 'errors_'+key in z: metrics['ber_'+key]=z['errors_'+key]/z['bits']
                for r,route in enumerate(routes):
                    diagnostics.append(dict(cell_id=meta['cell_id'],**route,
                        native_ici_available=int(z['native_available'][r]),
                        residual_max=float(z['native_residual_max'][r].max()),
                        direct_fallback_frames=int(z['native_direct_fallback'][r].sum()),
                        bit_errors_native=int(z['errors_native'][r].sum()),
                        bit_total=int(z['bits'][r].sum())))
            for metric,data in metrics.items():
                if data.shape[0]!=R or not np.isfinite(data).all(): raise RuntimeError('Invalid metric tensor')
                cluster_values=np.stack([data[:,cluster==c].mean(axis=1) for c in ids],axis=1)
                # All generation clusters have equal size; enforce to avoid weighting drift.
                if len(set(int((cluster==c).sum()) for c in ids))!=1: raise ValueError('Unequal cluster sizes')
                point=cluster_values.mean(axis=1)
                boot=weights@cluster_values.T
                context={k:v for k,v in meta.items() if k not in ('studies','ber')}
                context['n_test_clusters']=C
                rows,diffs=summarize(point,boot,routes,context,metric)
                cell_rows.extend(rows); paired_rows.extend(diffs)
                for study in meta['studies']:
                    for scope in scopes(meta):
                        key=(study,scope,meta['modulation'],'all','all',metric)
                        add_bucket(key,meta['channel'],point,boot)
                        if study in ('nmse','receiver','receiver_snr'):
                            key=(study,scope,meta['modulation'],str(meta['doppler_hz']),str(meta['snr_db']),metric)
                            add_bucket(key,meta['channel'],point,boot)
    macro=[]; macro_diffs=[]
    for key,channels in buckets.items():
        point=np.mean([p/n for p,b,n in channels.values()],axis=0)
        boot=np.mean([b/n for p,b,n in channels.values()],axis=0)
        study,scope,mod,fd,snr,metric=key
        context=dict(study=study,scope=scope,modulation=mod,doppler_hz=fd,snr_db=snr,
                     channel_count=len(channels),aggregation='equal cells within channel, then equal channels')
        rows,diffs=summarize(point,boot,routes,context,metric)
        macro.extend(rows); macro_diffs.extend(diffs)
    dest=out/'summary'
    csv_write(dest/'cell_summary.csv',cell_rows)
    csv_write(dest/'cell_paired_differences.csv',paired_rows)
    csv_write(dest/'macro_summary.csv',macro)
    csv_write(dest/'macro_paired_differences.csv',macro_diffs)
    csv_write(dest/'receiver_diagnostics.csv',diagnostics)
    summary_names=('cell_summary.csv','cell_paired_differences.csv','macro_summary.csv',
                   'macro_paired_differences.csv','receiver_diagnostics.csv')
    statistics=dict(evaluation_signature=ev['signature'],bootstrap=repetitions,
        training_seeds=list(SEEDS),t_critical_df2=float(t.ppf(.975,2)),
        test_cluster='entire simulation batch (noise normalization is shared)',
        data_binding_version=1,
        cell_file_sha256=cell_file_sha256,
        summary_csv_sha256={name:file_hash(dest/name) for name in summary_names},
        warnings=['Three seeds give imprecise t intervals; do not claim guaranteed normality.',
                  'Intervals are pointwise, not simultaneous or multiplicity corrected.',
                  'Zero-error empirical bootstrap is degenerate, not evidence of zero BER.',
                  'No unconditional joint training-and-test population interval is claimed.',
                  'The fixed channel profiles are not random samples of all real-world channels.'])
    statistics['signature']=digest(statistics)
    write_json(dest/'statistics.json',statistics)
    print(f'Saved seed means, 95% CIs and paired differences to {dest}')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='results/dopnet')
    ap.add_argument('--bootstrap',type=int,default=2000)
    a=ap.parse_args(); aggregate(Path(a.output).resolve(),a.bootstrap)
