"""Reaggregate archived server scores; no torch, inference, or waveform generation.

One NPZ (at most 0.36 MB compressed) is read at a time. Source files are never
modified. New groups are descriptive, condition-grid-conditional summaries.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

DATA = OUT = SCORES = RESULTS = None
EXPECTED_AGGREGATE_SHA256 = 'edde972f7def733af6bf7ac4e9b7f123959b0e6654e1ac37302a33da58ec351a'
METHODS = ('dopnet', 'pilot_only_fast_time', 'slow_time', 'cfr_derived')
SEEDS = (42, 43, 44)
STATIC = ('awgn', 'rayleigh', 'rician', 'nakagami')
SEEN = ('tdl_b', 'tdl_c', 'tdl_d', 'cdl_b', 'cdl_c', 'cdl_d')
HELD = ('tdl_a', 'tdl_e', 'cdl_a', 'cdl_e')
# Historical CSV df=2 critical value (also the exact analytic quantile).
# Local SciPy 1.16 ppf differs by 5.3e-11; using it would alter tiny CI bounds.
TCRIT = 4.302652729749462

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def readjson(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def writejson(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def writecsv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def readcsv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def seed_stats(x):
    x = np.asarray(x, dtype=np.float64)
    assert x.shape == (3,) and np.isfinite(x).all()
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    half = TCRIT * sd / math.sqrt(3)
    return mean, sd, mean-half, mean+half

def db(x):
    return 10 * math.log10(x) if x > 0 else None

def summaries(point, context, metric):
    rows, pairs = [], []
    for m, method in enumerate(METHODS):
        values = point[m]
        mean, sd, lo, hi = seed_stats(values)
        row = dict(context, metric=metric, method=method, mean=mean, seed_sd=sd,
                   seed_ci95_lo_unbounded=lo, seed_ci95_hi_unbounded=hi,
                   n_training_seeds=3, seed42=float(values[0]), seed43=float(values[1]), seed44=float(values[2]))
        if metric == 'nmse':
            row.update(mean_db=db(mean), seed_ci95_lo_db=db(max(0, lo)), seed_ci95_hi_db=db(hi))
        rows.append(row)
        if m:
            delta = values-point[0]
            dm, ds, dl, dh = seed_stats(delta)
            p = dict(context, metric=metric, comparator=method, reference='dopnet',
                     paired_difference=dm, paired_seed_ci95_lo=dl, paired_seed_ci95_hi=dh,
                     seed42_difference=float(delta[0]), seed43_difference=float(delta[1]), seed44_difference=float(delta[2]),
                     interpretation='Positive favors DOP-Net; pointwise descriptive seed-t interval conditional on archived frames')
            if metric == 'nmse':
                gap = 10*np.log10(values/point[0])
                gm, gs, gl, gh = seed_stats(gap)
                p.update(mean_paired_gap_db=gm, paired_gap_db_seed_ci95_lo=gl, paired_gap_db_seed_ci95_hi=gh,
                         seed42_gap_db=float(gap[0]), seed43_gap_db=float(gap[1]), seed44_gap_db=float(gap[2]))
            elif metric.startswith('ber'):
                p.update(gap_percentage_points=100*dm, gap_pp_lo=100*dl, gap_pp_hi=100*dh)
            pairs.append(p)
    return rows, pairs

def scopes(meta):
    ch = meta['channel']
    result = ['all', 'static' if ch in STATIC else 'dynamic_all']
    if ch == 'doubly_selective': result.append('dynamic_synthetic')
    if ch in SEEN+HELD:
        result += ['gpp_all', 'gpp_seen' if ch in SEEN else 'gpp_heldout']
        if meta['doppler_hz'] >= 800: result.append('gpp_high_mobility')
    return result

def pool(cells, metric):
    by_channel = defaultdict(list)
    for cell in cells:
        by_channel[cell['meta']['channel']].append(cell['points'][metric])
    point = np.mean([np.mean(points, axis=0) for points in by_channel.values()], axis=0)
    return point, by_channel

def flatten(obj, prefix=''):
    result = {}
    for k, v in obj.items():
        path = prefix+'.'+k if prefix else k
        if isinstance(v, dict): result.update(flatten(v, path))
        else: result[path] = v
    return result

def normalized_cfg(cfg):
    cfg = json.loads(json.dumps(cfg))
    train = cfg['train']
    train.setdefault('shared_epoch_samples', train['train_samples'])
    train.setdefault('unique_epoch_samples', train['train_samples'])
    cfg['data_pipeline'].setdefault('emit_pilot_only_fast_time_add', False)
    return cfg


def input_child(root, relative):
    """Resolve an index entry inside its declared input root."""
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError(f'Input index path must be relative: {relative}')
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f'Input index path escapes its root: {relative}')
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved

def aggregate_source_binding(code_root):
    if code_root is None:
        return dict(status='NOT_REQUESTED', expected_sha256=EXPECTED_AGGREGATE_SHA256,
                    note='Pass --code-root to byte-check the retained aggregate_results.py; it is never imported or executed.')
    code_root = code_root.resolve()
    candidates = [code_root/'src/current_project/aggregate_results.py',
                  code_root/'current_project/aggregate_results.py',
                  code_root/'aggregate_results.py']
    matches = [p for p in candidates if p.is_file()]
    if len(matches) != 1:
        raise ValueError('--code-root must identify exactly one aggregate_results.py at src/current_project/, current_project/, or its root')
    path = matches[0]
    actual = sha(path)
    if actual != EXPECTED_AGGREGATE_SHA256:
        raise ValueError('aggregate_results.py differs from the retained source SHA256')
    return dict(status='PASS_BYTE_IDENTITY', source_relative=path.relative_to(code_root).as_posix(),
                sha256=actual, expected_sha256=EXPECTED_AGGREGATE_SHA256,
                note='Byte identity only; the source is not imported or executed.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Reaggregate all 636 archived score cells using only NumPy and the standard library.')
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Public data root containing scores/original_retained, results/original_retained, weights, and metadata.')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New or empty output directory, disjoint from the input data tree.')
    parser.add_argument('--code-root', type=Path,
                        help='Optional public code root for a retained aggregate_results.py SHA256 check.')
    parser.add_argument('--no-plots', action='store_true', default=True,
                        help='Compatibility flag; plotting is disabled by default and this NumPy-only candidate never creates figures or PDFs.')
    args = parser.parse_args(argv)
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.data_root.is_dir():
        parser.error('--data-root does not exist')
    if args.output_dir.is_relative_to(args.data_root) or args.data_root.is_relative_to(args.output_dir):
        parser.error('--output-dir and --data-root must be disjoint trees')
    if args.code_root is not None:
        args.code_root = args.code_root.resolve()
        if args.output_dir.is_relative_to(args.code_root) or args.code_root.is_relative_to(args.output_dir):
            parser.error('--output-dir and --code-root must be disjoint trees')
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error('--output-dir must be a new or empty directory')
    return args

def main(argv=None):
    global DATA, OUT, SCORES, RESULTS
    args = parse_args(argv)
    DATA, OUT = args.data_root, args.output_dir
    SCORES, RESULTS = DATA/'scores/original_retained', DATA/'results/original_retained'
    source_binding = aggregate_source_binding(args.code_root)
    started = time.perf_counter()
    for required in (SCORES/'PROJECTION.json', SCORES/'INDEX.json', DATA/'weights/SELECTED_MODEL_INDEX.json'):
        if not required.is_file(): raise FileNotFoundError(required)
    OUT.mkdir(parents=True, exist_ok=True)
    projection, index = readjson(SCORES/'PROJECTION.json'), readjson(SCORES/'INDEX.json')
    assert len(index['cells']) == 636 and index['condition_count'] == 636
    assert len({item['cell_id'] for item in index['cells']}) == 636
    routes = projection['public_routes']
    selected = np.array([[next(i for i, r in enumerate(routes) if r['method']==m and r['training_seed']==s)
                          for s in SEEDS] for m in METHODS])
    assert len(routes)==50 and len(set(selected.ravel()))==12
    cells, cellrows, cellpairs, file_receipts = [], [], [], []
    metrics_count, studies_count = Counter(), Counter()
    for item in index['cells']:
        path = input_child(SCORES, item['npz'])
        actual_hash = sha(path)
        assert actual_hash == item['sha256'], path
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(z['meta'].item())
            assert meta['cell_id'] == item['cell_id']
            # Byte projection intentionally assigns a new derivative signature;
            # source evaluation identity remains separately in PROJECTION.json.
            assert z['signature'].item() == projection['signature']
            assert np.array_equal(z['cluster_id'], np.arange(512)//8)
            data = z['nmse']
            assert data.shape==(50,512) and data.dtype==np.float64 and np.isfinite(data).all() and (data>=0).all()
            metrics = {'nmse': data}
            if meta['ber']:
                bits = z['bits']
                assert bits.shape==(50,512) and (bits>0).all()
                # Equal frame bit denominators preserve the source mean(errors/bits) metric.
                assert np.all(bits==bits[0,0]), (path, 'unequal bit denominators')
                for k in ('diagonal','native','noise_minus3db','noise_plus3db'):
                    if 'errors_'+k in z.files:
                        errors = z['errors_'+k]
                        assert errors.shape==bits.shape and ((errors>=0)&(errors<=bits)).all()
                        metrics['ber_'+k] = errors/bits
            # Match the original boolean-index cluster reductions exactly, even
            # when a near-zero CI lower bound magnifies floating-point roundoff.
            cluster=z['cluster_id']; ids=np.unique(cluster)
            points = {metric: np.stack([values[:,cluster==c].mean(axis=1) for c in ids],axis=1).mean(axis=1)[selected]
                      for metric, values in metrics.items()}
            context = {k:v for k,v in meta.items() if k not in ('studies','ber')}
            context['n_test_clusters'] = 64
            for metric, point in points.items():
                r, p = summaries(point, context, metric)
                cellrows.extend(r); cellpairs.extend(p); metrics_count[metric]+=1
            cells.append({'meta':meta, 'points':points})
            studies_count.update(meta['studies'])
            file_receipts.append(dict(cell_id=meta['cell_id'], source_relative=path.relative_to(DATA).as_posix(), bytes=path.stat().st_size,
                                      sha256=actual_hash, realization_sha256=str(z['realization_sha256'].item()), studies='|'.join(meta['studies'])))
        if len(cells)%100==0: print(f'reaggregated {len(cells)}/{len(index["cells"])} score cells', flush=True)
    writecsv('source_score_receipts.csv', file_receipts)
    writecsv('cell_recomputed.csv', cellrows)
    writecsv('cell_paired_recomputed.csv', cellpairs)
    # Recreate exact original macro membership and channel-balanced point statistics.
    buckets = defaultdict(list)
    for c in cells:
        m = c['meta']
        for metric in c['points']:
            for study in m['studies']:
                for scope in scopes(m):
                    buckets[(study,scope,m['modulation'],'all','all',metric)].append(c)
                    if study in ('nmse','receiver','receiver_snr'):
                        buckets[(study,scope,m['modulation'],str(m['doppler_hz']),str(m['snr_db']),metric)].append(c)
    macrorows, macropairs = [], []
    for key, population in buckets.items():
        study, scope, mod, fd, snr, metric = key
        point, by_ch = pool(population, metric)
        context = dict(study=study,scope=scope,modulation=mod,doppler_hz=fd,snr_db=snr,channel_count=len(by_ch),
                       cell_count=len(population), aggregation='equal cells within channel, then equal channels')
        r,p = summaries(point,context,metric)
        macrorows.extend(r);macropairs.extend(p)
    writecsv('macro_recomputed.csv',macrorows)
    writecsv('macro_paired_recomputed.csv',macropairs)
    # Every comparison below is against retained source CSV, not recalculated inference.
    checks = []
    for filename,newrows,keys,fields in (
        ('cell_summary.csv',cellrows,('cell_id','metric','method'),('mean','seed_sd','seed42','seed43','seed44','mean_db','seed_ci95_lo_db','seed_ci95_hi_db')),
        ('cell_paired_differences.csv',cellpairs,('cell_id','metric','comparator'),('paired_difference','paired_seed_ci95_lo','paired_seed_ci95_hi','mean_paired_gap_db','paired_gap_db_seed_ci95_lo','paired_gap_db_seed_ci95_hi')),
        ('macro_summary.csv',macrorows,('study','scope','modulation','doppler_hz','snr_db','metric','method'),('mean','seed_sd','seed42','seed43','seed44','mean_db','seed_ci95_lo_db','seed_ci95_hi_db')),
        ('macro_paired_differences.csv',macropairs,('study','scope','modulation','doppler_hz','snr_db','metric','comparator'),('paired_difference','paired_seed_ci95_lo','paired_seed_ci95_hi','mean_paired_gap_db','paired_gap_db_seed_ci95_lo','paired_gap_db_seed_ci95_hi'))):
        saved={tuple(row[k] for k in keys):row for row in readcsv(RESULTS/filename) if row.get('method',row.get('comparator')) in METHODS}
        field_max=defaultdict(float); compared=0
        for row in newrows:
            key=tuple(str(row[k]) for k in keys)
            assert key in saved,(filename,key)
            old=saved[key]
            for field in fields:
                if field not in row:continue
                if row[field] is None: assert not old[field],(filename,key,field);continue
                delta=abs(float(row[field])-float(old[field]))
                field_max[field]=max(field_max[field],delta)
                assert delta<=1e-10,(filename,key,field,delta)
                compared+=1
        assert len(newrows)==len(saved),(filename,len(newrows),len(saved))
        checks.append(dict(source=filename,source_sha256=sha(RESULTS/filename),rows=len(newrows),numeric_fields_compared=compared,
                           max_absolute_error=dict(field_max),status='PASS_at_absolute_tolerance_1e-10'))
    # New descriptive groups: separate studies/modulations to avoid mixing different grids.
    group_rows,group_pairs,weights=[],[],[]
    group_specs = [('nmse','qpsk','nmse',('doppler_hz','doppler_hz_dynamic','snr_db','channel')),
                   ('receiver','16qam','ber_diagonal',('doppler_hz','channel')),
                   ('receiver_snr','16qam','ber_diagonal',('snr_db',)),
                   ('pilot_cp','qpsk','nmse',('pilot_spacing','cp_len','pilot_spacing_cp_len')),
                   ('unseen_doppler','qpsk','nmse',('doppler_hz',)),
                   ('delay_shift','qpsk','nmse',('delay_spread_ns',))]
    for study,mod,metric,dimensions in group_specs:
        population=[c for c in cells if study in c['meta']['studies'] and c['meta']['modulation']==mod]
        for dimension in dimensions:
            groups=defaultdict(list)
            for c in population:
                if dimension=='doppler_hz_dynamic' and c['meta']['channel'] in STATIC:
                    continue
                value=((c['meta']['pilot_spacing'],c['meta']['cp_len']) if dimension=='pilot_spacing_cp_len'
                       else c['meta']['doppler_hz'] if dimension=='doppler_hz_dynamic' else c['meta'][dimension])
                groups[str(value)].append(c)
            for value, subset in groups.items():
                point,by_ch=pool(subset,metric)
                gid=f'{study}:{metric}:{dimension}:{value}'
                context=dict(group_id=gid,study=study,modulation=mod,dimension=dimension,value=value,
                             channel_count=len(by_ch),cell_count=len(subset),unique_received_frames=512*len(subset),
                             aggregation='equal cells within channel, then equal channels',scope='descriptive_new_group')
                r,p=summaries(point,context,metric)
                group_rows.extend(r)
                for pair in p:
                    comparator=METHODS.index(pair['comparator'])
                    diffs=[float(c['points'][metric][comparator].mean()-c['points'][metric][0].mean()) for c in subset]
                    pair.update(cells_favoring_dopnet=sum(d>0 for d in diffs),cells_favoring_comparator=sum(d<0 for d in diffs),
                                all_three_seeds_favor_dopnet_cells=sum(bool(np.all(c['points'][metric][comparator]>c['points'][metric][0])) for c in subset))
                group_pairs.extend(p)
                for c in subset:
                    ch=c['meta']['channel'];n_ch=len(by_ch[ch])
                    weights.append(dict(group_id=gid,cell_id=c['meta']['cell_id'],channel=ch,cell_weight=1/(len(by_ch)*n_ch),
                                        frame_weight=1/(len(by_ch)*n_ch*512),frames=512,clusters=64))
    writecsv('descriptive_groups.csv',group_rows)
    writecsv('descriptive_paired_groups.csv',group_pairs)
    writecsv('descriptive_group_weights.csv',weights)
    # Verify portable checkpoint bytes; no deserialization or inference.
    model_index=readjson(DATA/'weights/SELECTED_MODEL_INDEX.json')
    bindings=[];configs={};config_diffs=[]
    for entry in model_index['models']:
        if entry['method'] not in METHODS:continue
        path=input_child(DATA, entry['local_relative']); actual=sha(path);assert actual==entry['sha256']
        cfg_path=input_child(DATA, entry['source_completion_metadata']);completion=readjson(cfg_path)
        assert sha(cfg_path) == entry['source_completion_metadata_sha256']
        configs[(entry['method'],entry['seed'])]=normalized_cfg(completion['config'])
        bindings.append(dict(method=entry['method'],seed=entry['seed'],checkpoint_sha256=actual,selected_epoch=entry['selected_epoch'],
                             checkpoint_relative=entry['local_relative'],config_relative=entry['source_completion_metadata'],config_sha256=sha(cfg_path)))
    for seed in SEEDS:
        ref=flatten(configs[('dopnet',seed)])
        for method in METHODS[1:]:
            alt=flatten(configs[(method,seed)])
            diffs=[dict(path=k,dopnet=ref.get(k),control=alt.get(k)) for k in sorted(ref.keys()|alt.keys()) if ref.get(k)!=alt.get(k)]
            assert all(d['path'] in ('model.ablation_profile','data_pipeline.emit_pilot_only_fast_time_add') for d in diffs)
            config_diffs.append(dict(method=method,seed=seed,differences=diffs))
    assert len(bindings)==12
    writejson('checkpoint_and_fairness_bindings.json',dict(checkpoints=bindings,normalized_configuration_differences=config_diffs,
        defaults_applied=['shared_epoch_samples=train_samples','unique_epoch_samples=train_samples','emit_pilot_only_fast_time_add=False'],
        validation='Checkpoint bytes only; no weights deserialized; architecture/supervision matches also rely on source implementation.',
        source_input_controls='waveform.py preserves full isolated-user PWB target for ADD controls; DOP.forward selects only ADD observation source.'))
    # Direct comparison with printed main-paper three-decimal mechanism entries.
    rounded_targets={('slow_time','all'):.315,('slow_time','gpp_high_mobility'):.609,('slow_time','gpp_heldout'):1.043,
        ('pilot_only_fast_time','all'):1.161,('pilot_only_fast_time','gpp_high_mobility'):1.470,('pilot_only_fast_time','gpp_heldout'):1.270,
        ('cfr_derived','all'):.666,('cfr_derived','gpp_high_mobility'):.943,('cfr_derived','gpp_heldout'):1.269}
    table_checks=[]
    for (method,scope),target in rounded_targets.items():
        row=next(r for r in macropairs if r['study']=='nmse' and r['scope']==scope and r['metric']=='nmse' and r['doppler_hz']=='all' and r['snr_db']=='all' and r['comparator']==method)
        rounded=round(row['mean_paired_gap_db'],3);assert rounded==target
        table_checks.append(dict(method=method,scope=scope,recomputed_gap=row['mean_paired_gap_db'],printed_3dp=target,status='PASS'))
    plot_files=[]  # Disabled: only NumPy/stdlib are required; no figures or PDFs.
    writejson('verification_summary.json',dict(status='COMPLETE',source='Archived server per-frame scores; no new inference, generation, training, or calibration performance estimate',
        methods=METHODS,training_seeds=SEEDS,cells=len(cells),unique_original_frames=512*len(cells),source_compressed_bytes=sum(r['bytes'] for r in file_receipts),
        study_memberships=dict(studies_count),metric_cell_counts=dict(metrics_count),source_csv_checks=checks,main_mechanism_table_checks=table_checks,
        grouping_files=['descriptive_groups.csv','descriptive_paired_groups.csv','descriptive_group_weights.csv'],plots=plot_files,
        limits=['No inference reproduced on CPU; this is score reaggregation equivalence to archived server values.',
                'Three seeds and selected simulator grid; intervals pointwise and conditional, no multiplicity or universal dominance claim.',
                'Cell win counts are descriptive and unweighted; reported macro point estimates use explicit channel-balanced weights.',
                'Separate studies and modulations are not pooled; geometry groups may have unequal eligible cell counts.',
                'No validation/calibration cache is used as a fresh blind performance test.'],
        historical_seed_t_df2=TCRIT,numpy_version=np.__version__,cpu_max_threads=4,
        source_aggregate_script_sha256=source_binding.get('sha256'),source_aggregate_script_check=source_binding,
        plotting_policy='Disabled by default and always disabled in this NumPy-only candidate',
        source_projection_sha256=sha(SCORES/'PROJECTION.json'),source_index_sha256=sha(SCORES/'INDEX.json'),
        runtime_seconds=time.perf_counter()-started,script_sha256=sha(Path(__file__))))
    print(json.dumps({'status':'COMPLETE','cells':len(cells),'csv_checks':len(checks),'plot_files':plot_files,'seconds':round(time.perf_counter()-started,2)},ensure_ascii=False))

if __name__=='__main__':main()
