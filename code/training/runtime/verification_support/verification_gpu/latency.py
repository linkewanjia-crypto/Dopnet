"""Measure receiver latency only on the GPU; never run a CPU model forward."""
import gc
from pathlib import Path
import platform
import os

from verification_extension import latency as L


def benchmark(args):
    if args.latency_repetitions < 10 or args.latency_warmups < 1:
        raise ValueError('GPU timing needs warmups>=1 and repetitions>=10')
    L._runtime_import_path()
    torch = L._torch()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU-only latency requires CUDA; no CPU timing is substituted')
    from models.receiver import SharedUserChannelEstimator
    from utils.checkpoints import load_checkpoint
    results = Path(args.reference_results)
    output = Path(args.output) / 'latency_gpu'
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.output) / 'observation_cache'
    export, experiment, sources = L._load_sources(cache, results, None, cache / 'latency_capture.npz')
    names = [f'{method}_seed{seed}' for method in L.METHODS for seed in L.SEEDS]
    reference_cfg = experiment['configs'][names[0]]
    for name in names:
        if L._physics_identity(experiment['configs'][name]) != L._physics_identity(reference_cfg):
            raise RuntimeError('Checkpoint physical geometry differs: ' + name)
        if not (results / 'checkpoints' / name / 'model.pt').is_file():
            raise FileNotFoundError('Missing existing checkpoint: ' + name)
    if L._physics_identity({'system': export['cfg_system']}) != L._physics_identity(reference_cfg):
        raise RuntimeError('Calibration observations and checkpoints have different physical geometry')
    device = torch.device('cuda:0')
    threads = max(args.cpu_threads)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('highest')
    report = {'complete': False, 'schema_version': 1,
        'protocol': {
            'scope': 'GPU-only digital receiver latency; no CPU model timing or CPU/GPU comparison',
            'devices': ['cuda:0'], 'methods': list(L.METHODS), 'seeds': list(L.SEEDS),
            'batch_sizes': list(args.batch_sizes), 'receiver_modes': ['common_diagonal', 'native_own_ici'],
            'cpu_threads': threads, 'cpu_thread_candidates': [],
            'warmups': args.latency_warmups, 'repetitions': args.latency_repetitions,
            'precision': 'float32/complex64; TF32 disabled; eager inference_mode',
            'timing': 'synchronized CUDA wall clock; host-origin total directly timed',
            'host_origin_total': 'H2D + CP/FFT if time input + pilot/ADD frontend + model + MMSE + demapping + bits D2H',
            'device_resident_total': 'same receiver with GPU-resident input/output',
            'excluded': ['RF/acquisition', 'ADC conversion', 'channel simulation', 'training', 'calibration fitting',
                         'BER/NMSE scoring', 'disk input/output'],
            'ofdm_grid_rows_exclude': ['CP removal', 'full-band OFDM FFT'],
            'time_capture_rows_include': ['CP removal', 'full-band OFDM FFT', 'Sionna phase compensation', 'active-bin selection'],
            'experiment_signature': experiment['signature'], 'input_export_signature': export['export_signature'],
            'implementation_sha256': L.implementation_hashes(), 'gpu_adapter_sha256': L._file_hash(__file__),
            'numerical_check': 'finite outputs and repeated same-input GPU inference; no cross-device claim',
            'sources': [{key: source[key] for key in ('name', 'role', 'path', 'sha256', 'source_samples',
                                                     'condition', 'realization_sha256')} for source in sources]},
        'hardware': {'platform': platform.platform(), 'cpu': platform.processor(), 'logical_cpu_count': os.cpu_count(),
            'python': platform.python_version(), 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'gpus': [{'device': str(device), 'name': torch.cuda.get_device_name(device),
                      'total_memory_mb': torch.cuda.get_device_properties(device).total_memory / 2**20}]},
        'rows': [], 'device_recommendations': [], 'skipped_devices': []}
    path = output / 'digital_receiver_latency.json'
    if path.exists() and L._json(path).get('protocol') != report['protocol']:
        raise RuntimeError('GPU latency folder belongs to another protocol; choose a new output directory')
    L._write_json(path, report)
    try:
        with torch.inference_mode():
            for method in L.METHODS:
                for seed in L.SEEDS:
                    name = f'{method}_seed{seed}'
                    checkpoint_path = results / 'checkpoints' / name / 'model.pt'
                    checkpoint = load_checkpoint(checkpoint_path, map_location='cpu')
                    cfg = checkpoint['cfg']
                    if cfg != experiment['configs'][name] or checkpoint.get('selection_metric') != 'val_nmse_db':
                        raise RuntimeError('Existing checkpoint provenance differs: ' + name)
                    model = SharedUserChannelEstimator(cfg).to(device).eval()
                    model.load_state_dict(checkpoint['model'], strict=True)
                    if any(p.dtype != torch.float32 for p in model.parameters()):
                        raise RuntimeError('GPU receiver timing requires fp32 model weights')
                    for source in sources:
                        arrays = source['arrays']
                        constellation, shifts = L._constellation(arrays['condition']['modulation'], device)
                        for size in args.batch_sizes:
                            host_raw, input_hash, unique_count = L._batch_raw(source, size)
                            for mode in ('common_diagonal', 'native_own_ici'):
                                base = {'method': method, 'seed': seed, 'batch_size': size, 'device': str(device),
                                    'cpu_threads': threads, 'receiver_mode': mode, 'source_name': source['name'],
                                    'checkpoint_sha256': L._file_hash(checkpoint_path), 'input_sha256': input_hash,
                                    'input_boundary': 'actual_received_time_samples' if 'rx_time' in host_raw else 'received_occupied_ofdm_grid',
                                    'unique_observed_frames': unique_count, 'timing_frame_repetition_factor': size / unique_count}
                                if mode == 'native_own_ici' and not (model.estimator_name == 'dopnet' and model.base.use_add_branch):
                                    report['rows'].append({**base, 'status': 'not_applicable', 'reason': 'no native ICI coefficients'})
                                    continue
                                print(f'[gpu-latency] {name} cuda:0 {source["name"]} B={size} {mode}', flush=True)
                                raw = grid_raw = batch = h = estimate = symbols = decoded = operations = diagnostic = repeat = None
                                try:
                                    raw = L._to_device(host_raw, device)
                                    demodulation = L._check_time_demodulation(raw, cfg, arrays, size)
                                    diagnostic = L._chain(model, cfg, raw, arrays, mode, constellation, shifts, diagnostic=True)
                                    repeat = L._chain(model, cfg, raw, arrays, mode, constellation, shifts, diagnostic=True)
                                    equivalence = {**L.compare_outputs(diagnostic, repeat), 'reference_device': 'cuda:0',
                                                   'comparison': 'same-input GPU repeatability; no CPU inference'}
                                    if not equivalence['passed']:
                                        raise RuntimeError('GPU same-input repeatability check failed: ' + name)
                                    grid_raw = L.demodulate_time_capture(raw, cfg, arrays)
                                    batch = L.preprocess_observed_grid(grid_raw, cfg, arrays)
                                    h, estimate = L._estimate(model, cfg, batch, mode)
                                    symbols, _ = L._equalize(estimate, batch, mode)
                                    _, decoded = L._demap(symbols, constellation, shifts)
                                    operations = {
                                        'host_to_device': lambda: L._to_device(host_raw, device),
                                        'pilot_and_add_frontend': lambda: L.preprocess_observed_grid(grid_raw, cfg, arrays),
                                        'channel_estimation_and_native_band_construction': lambda: L._estimate(model, cfg, batch, mode),
                                        'equalization': lambda: L._equalize(estimate, batch, mode),
                                        'hard_demapping': lambda: L._demap(symbols, constellation, shifts),
                                        'decoded_bits_to_host': lambda: decoded.cpu()}
                                    if 'rx_time' in raw:
                                        operations['cp_fft_and_subcarrier_selection'] = lambda: L.demodulate_time_capture(raw, cfg, arrays)
                                    timing = {stage: L._time_operation(fn, device, size, args.latency_warmups,
                                                                     args.latency_repetitions) for stage, fn in operations.items()}
                                    operations = batch = h = estimate = symbols = decoded = diagnostic = repeat = grid_raw = None
                                    timing['device_resident_total'] = L._time_operation(
                                        lambda: L._chain(model, cfg, raw, arrays, mode, constellation, shifts), device,
                                        size, args.latency_warmups, args.latency_repetitions, collect_equalizer=True)
                                    timing['host_origin_total'] = L._time_operation(
                                        lambda: L._chain(model, cfg, L._to_device(host_raw, device), arrays, mode,
                                                         constellation, shifts, host_output=True), device, size,
                                        args.latency_warmups, args.latency_repetitions, collect_equalizer=True)
                                    report['rows'].append({**base, 'status': 'measured', 'timing': timing,
                                                          'numerical_equivalence': equivalence,
                                                          'demodulation_reconstruction_check': demodulation})
                                except torch.cuda.OutOfMemoryError as error:
                                    report['rows'].append({**base, 'status': 'oom', 'error': str(error),
                                                          'numerical_equivalence': {'passed': False}})
                                finally:
                                    raw = grid_raw = batch = h = estimate = symbols = decoded = operations = diagnostic = repeat = None
                                    gc.collect(); torch.cuda.empty_cache()
                                report['device_recommendations'] = L.recommend_devices(report['rows'], expected_seeds=L.SEEDS)
                                L._write_json(path, report)
                    del checkpoint, model
                    gc.collect(); torch.cuda.empty_cache()
        report['complete'] = True
        report['numerical_checks_passed'] = all(row.get('numerical_equivalence', {}).get('passed', False)
                                              for row in report['rows'] if row['status'] == 'measured')
        L._write_json(path, report)
        L._write_json(output / 'inference_device_policy.json', {
            'complete': True, 'precision': 'fp32', 'qualification': 'GPU-only; same-input repeatability, no cross-device comparison',
            'numerical_checks_passed': report['numerical_checks_passed'], 'latency_report_sha256': L._file_hash(path),
            'experiment_signature': experiment['signature'], 'source_export_signature': export['export_signature'],
            'recommendations': report['device_recommendations']})
        return report
    finally:
        torch.set_num_threads(old_threads)
