"""Reuse paired-score statistics and emit GPU-only timing tables."""
from collections import defaultdict
from pathlib import Path
import os
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np

from verification_extension import protocol as P
from verification_extension import report as R


def gpu_latency_assets(dest, latency_output, reference_signature):
    path = Path(latency_output) / 'digital_receiver_latency.json'
    if not path.exists():
        return [], {'status': 'not_available'}
    report = P.read_json(path)
    protocol = report['protocol']
    if not report.get('complete'):
        return [], {'status': 'incomplete'}
    if protocol['experiment_signature'] != reference_signature or protocol.get('devices') != ['cuda:0']:
        raise ValueError('Not the declared GPU-only timing experiment')
    measured = [row for row in report['rows'] if row['status'] == 'measured'
                and row.get('numerical_equivalence', {}).get('passed')]
    if any(row['device'] != 'cuda:0' for row in measured):
        raise ValueError('GPU-only paper timing cannot include CPU rows')
    flat, groups = [], defaultdict(list)
    for row in measured:
        for stage, values in row['timing'].items():
            context = {key: row[key] for key in ('method', 'seed', 'batch_size', 'receiver_mode',
                                                'device', 'source_name', 'input_boundary')}
            flat.append({**context, 'stage': stage, **{key: value for key, value in values.items()
                         if value is None or isinstance(value, (str, float, int))}})
        key = tuple(row[key] for key in ('source_name', 'input_boundary', 'method', 'receiver_mode', 'batch_size'))
        groups[key].append(row)
    R._old.write_csv(dest / 'corrected_digital_receiver_latency.csv', flat)
    R._old.write_csv(dest / 'corrected_inference_device_choices.csv', report['device_recommendations'])
    required = set(protocol['seeds'])
    qualified = {key: rows for key, rows in groups.items() if len(rows) == len(required)
                 and {row['seed'] for row in rows} == required}
    files = ['corrected_digital_receiver_latency.csv', 'corrected_inference_device_choices.csv']
    for single, name in ((True, 'corrected_digital_receiver_latency'), (False, 'corrected_digital_receiver_throughput')):
        selected = [(key, rows) for key, rows in qualified.items() if (key[-1] == 1) == single]
        if not selected:
            continue
        heading = 'Method & Receiver & Median/P95 (ms)' if single else 'Method & Receiver & Batch & Frames/s'
        caption = ('GPU single-frame digital receiver latency. Median and 95th-percentile wall-clock times are averaged over three model seeds. '
                   if single else 'GPU batched digital receiver throughput, averaged over three model seeds. ')
        lines = ['\\begin{table*}[t]', '\\centering', '\\scriptsize',
                 '\\caption{' + caption + 'Host-origin totals include input/output transfers. Each block states its input boundary.}',
                 '\\label{tab:' + ('corrected_digital_latency' if single else 'corrected_digital_throughput') + '}',
                 '\\begin{tabular}{' + ('lll' if single else 'llrr') + '}', '\\toprule', heading + ' \\\\', '\\midrule']
        last = None
        for key, rows in sorted(selected):
            source, boundary, method, mode, size = key
            if last != (source, boundary):
                label = 'Actual noisy time samples' if boundary == 'actual_received_time_samples' else 'Received occupied OFDM grid'
                lines.append('\\multicolumn{' + ('3' if single else '4') + '}{l}{' + source.replace('_', '\\_') + ': ' + label + '} \\\\')
                last = (source, boundary)
            total = [row['timing']['host_origin_total'] for row in rows]
            value = (f'{np.mean([v["latency_ms_p50"] for v in total]):.3f} / {np.mean([v["latency_ms_p95"] for v in total]):.3f}'
                     if single else f'{np.mean([v["frames_per_second"] for v in total]):.1f}')
            entries = [R.LABELS.get(method, method), 'Diagonal' if mode == 'common_diagonal' else 'Own ICI']
            if not single:
                entries.append(str(size))
            lines.append(' & '.join(entries + [value]) + ' \\\\')
        lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
        filename = name + '.tex'
        (dest / filename).write_text('\n'.join(lines), encoding='utf-8')
        files.append(filename)
    P.write_json(dest / 'corrected_latency_scope.json', {'protocol': protocol, 'hardware': report['hardware'],
                 'latency_json_sha256': P.file_hash(path), 'mode': 'GPU-only; CPU receiver timing omitted'})
    files.append('corrected_latency_scope.json')
    sources = {row['source_name'] for row in report['rows']}
    single_ready = all((source, boundary, method, 'common_diagonal', 1) in qualified
                       for source in sources for boundary in {row['input_boundary'] for row in report['rows'] if row['source_name'] == source}
                       for method in protocol['methods'])
    batch_ready = all(any(key[0] == source and key[2] == method and key[3] == 'common_diagonal' and key[4] > 1
                         for key in qualified) for source in sources for method in protocol['methods'])
    return files, {'status': 'complete' if single_ready and batch_ready else 'incomplete_gpu_workloads',
                   'latency_json_sha256': P.file_hash(path), 'mode': 'GPU-only',
                   'single_frame_all_models_ready': single_ready, 'batched_all_models_ready': batch_ready}


def report(args):
    # Override reporting presentation in memory, preserving the calibrated files.
    previous = R.latency_assets
    R.latency_assets = gpu_latency_assets
    try:
        statistics = R.report(args.output, args.reference_results, bootstrap=args.bootstrap,
                              latency_output=Path(args.output) / 'latency_gpu')
    finally:
        R.latency_assets = previous
    statistics['gpu_report_adapter_sha256'] = P.file_hash(__file__)
    statistics['data_generation_mode'] = 'original GPU frozen replay; no CPU/GPU generation benchmark'
    statistics['pending'] = [item.replace('CPU/CUDA', 'GPU') for item in statistics['pending']]
    insertion = Path(args.output) / 'corrected_summary/corrected_paper_insertion.tex'
    text = insertion.read_text(encoding='utf-8').replace(
        'the corresponding tables report measured CPU/CUDA workloads; '
        'the fastest eligible device is selected separately for each receiver and batch size.',
        'the corresponding tables report GPU workloads for each receiver and batch size.')
    # Match the capitalized sentence emitted by the original reporter as well.
    text = text.replace('The corresponding tables report measured CPU/CUDA workloads; '
                        'the fastest eligible device is selected separately for each receiver and batch size.',
                        'The corresponding tables report GPU workloads for each receiver and batch size.')
    insertion.write_text(text, encoding='utf-8')
    statistics['report_asset_sha256']['corrected_paper_insertion.tex'] = P.file_hash(insertion)
    P.write_json(Path(args.output) / 'corrected_summary/statistics.json', statistics)
    bundle_gpu_results(args.output, statistics)
    return statistics


def bundle_gpu_results(output, statistics):
    """Keep old CPU-model timing on the server, outside the returned package."""
    root = Path(output).resolve()
    files = [p for p in root.rglob('*') if p.is_file()
             and not any(part.startswith('.') for part in p.relative_to(root).parts)
             and 'latency' not in p.relative_to(root).parts
             and p.relative_to(root).as_posix() != 'logs/latency.log'
             and p.suffix not in ('.zip', '.tmp')
             and not ('observation_cache' in p.relative_to(root).parts and p.suffix == '.npz')
             and p.name != 'return_package_manifest.json']
    manifest = root / 'return_package_manifest.json'
    P.write_json(manifest, {'schema_version': 2, 'suite': statistics['suite'],
        'paper_ready': statistics['paper_ready'], 'pending': statistics['pending'],
        'receiver_timing': 'GPU only; old CPU-model timing omitted', 'training_performed': False,
        'waveform_arrays_included': False,
        'files': {p.relative_to(root).as_posix(): P.file_hash(p) for p in sorted(files)}})
    files.append(manifest)
    archive = root / 'verification_results_for_paper.zip'
    temporary = archive.with_suffix('.zip.tmp')
    with ZipFile(temporary, 'w', compression=ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(files):
            z.write(p, 'dopnet_verification_results/' + p.relative_to(root).as_posix())
    os.replace(temporary, archive)
    print('GPU verification return package: ' + str(archive), flush=True)
    return archive
