"""Replay frozen receiver observations; keep only a small calibration cache."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .protocol import atomic_npz, digest, file_hash, read_json, write_json


class CaptureReceivedTime:
    """Read-only capture of the final noisy demodulator input, without RNG draws."""
    def __init__(self, waveform):
        self.waveform = waveform
        self.reset()

    def reset(self):
        self.received = None
        self.backend = self.l_min = self.num_blocks = None

    def __enter__(self):
        self.original_synthetic = self.waveform._ofdm_demodulate_torch
        self.original_sionna = self.waveform._sionna_ops_cached

        def synthetic(samples, nblocks, nfft, cpw, nactive):
            result = self.original_synthetic(samples, nblocks, nfft, cpw, nactive)
            self.received = samples.detach()
            self.backend, self.l_min, self.num_blocks = 'synthetic', 0, int(nblocks)
            return result

        def sionna(nfft, cpw, ttime, lmin, lmax, device_str):
            mod, demod, apply = self.original_sionna(nfft, cpw, ttime, lmin, lmax, device_str)

            def observed_demod(samples):
                result = demod(samples)
                self.received = samples[:, 0].detach()
                self.backend, self.l_min = 'sionna', int(lmin)
                self.num_blocks = int(ttime) // (int(nfft) + int(cpw))
                return result
            return mod, observed_demod, apply

        self.waveform._ofdm_demodulate_torch = synthetic
        self.waveform._sionna_ops_cached = sionna
        return self

    def __exit__(self, *_):
        self.waveform._ofdm_demodulate_torch = self.original_synthetic
        self.waveform._sionna_ops_cached = self.original_sionna
        self.reset()

    def arrays(self, expected_frames):
        if self.received is None or self.received.ndim != 3 or self.received.shape[0] != expected_frames:
            raise RuntimeError('Final noisy receiver samples were not captured')
        return {'rx_time': self.received.cpu().numpy().astype(np.complex64),
                'rx_time_backend': np.asarray(self.backend),
                'rx_time_l_min': np.asarray(self.l_min, dtype=np.int64),
                'rx_time_num_blocks': np.asarray(self.num_blocks, dtype=np.int64)}


class FrozenReplay:
    """Use the exact original runtime in an isolated server worker process."""
    def __init__(self, project, reference_results, cells, suite):
        project = Path(project).resolve()
        if not (project / 'reference_replay.py').is_file():
            raise FileNotFoundError('Missing frozen reference project: ' + str(project))
        # Generation must not accidentally reuse the public edition's modules.
        for module_name in ('data.generator', 'experiment_protocol', 'reference_replay', 'train'):
            loaded = sys.modules.get(module_name)
            if loaded and not Path(loaded.__file__).resolve().is_relative_to(project):
                raise RuntimeError('Run frozen replay in a fresh process; a different runtime is already loaded: ' + module_name)
        sys.path.insert(0, str(project))
        from utils.cuda_memory import configure_cuda_allocator
        configure_cuda_allocator()
        import torch
        from data.generator import BatchChannelGenerator
        from data import waveform
        from experiment_protocol import TEST_SEED, CALIBRATION_SEED, code_hash
        from reference_replay import ReferenceResults, update_waveform_digest
        from classical_extension.export_frozen_observations import (
            CapturePilotObservations, check_current_protocol, public_geometry,
            runtime_physics_hashes,
        )
        if not torch.cuda.is_available():
            raise RuntimeError('Frozen waveform replay needs the original CUDA/Sionna server; classical scoring can use CPU')
        self.reference = ReferenceResults(reference_results)
        if code_hash(project) != self.reference.experiment['code_sha256']:
            raise RuntimeError('Frozen source fingerprint differs from the completed training experiment')
        self.cfg = copy.deepcopy(self.reference.experiment['configs']['dopnet_seed42'])
        check_current_protocol(project, self.cfg, self.reference)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.torch, self.waveform = torch, waveform
        self.generator_class = BatchChannelGenerator
        self.capture_class = CapturePilotObservations
        self.update_waveform_digest = update_waveform_digest
        self.test_seed, self.calibration_seed = TEST_SEED, CALIBRATION_SEED
        self.generator = BatchChannelGenerator(self.cfg, torch.device('cuda:0'), TEST_SEED)
        base = {'schema_version': 1, 'reference_provenance': self.reference.provenance(),
                'cfg_system': self.cfg['system'], 'generation_config': self.cfg,
                'geometry': public_geometry(self.cfg['system']),
                'runtime_physics_sha256': runtime_physics_hashes(project),
                'samples_per_cell': 512, 'batch_size': 8, 'test_seed': TEST_SEED,
                'calibration_seed': CALIBRATION_SEED, 'calibration_samples': 128,
                'exporter_sha256': file_hash(Path(__file__)),
                'cache_mode': 'independent calibration retained; test observations streamed'}
        base['export_signature'] = digest(base)
        self.manifest = {**base, 'cells': list(cells), 'suites': {suite: [m['cell_id'] for m in cells]}}

    def prepare_statistical_extra(self, output, count=256):
        """Extra independent training-mixture calibration, separate from BEM128."""
        output = Path(output)
        marker_path = output / 'calibration_extra/COMPLETE.json'
        seed = self.calibration_seed + 1
        if count < 0 or count % self.generator.generation_batch_size:
            raise ValueError('Extra calibration count must be a nonnegative multiple of the original generation batch size')
        if not count:
            return []
        if marker_path.exists():
            marker = read_json(marker_path)
            if (marker['samples'] != count or marker['seed'] != seed
                or marker['export_signature'] != self.manifest['export_signature']):
                raise RuntimeError('Existing additional calibration differs; use a new output directory')
            arrays = []
            for record in marker['batches']:
                path = (output / record['path']).resolve()
                path.relative_to(output.resolve())
                if file_hash(path) != record['sha256']:
                    raise RuntimeError('Additional calibration cache checksum differs')
                with np.load(path, allow_pickle=False) as z:
                    arrays.append({k: z[k] for k in z.files})
            if sum(len(a['noise_var']) for a in arrays) != count:
                raise RuntimeError('Additional calibration frame count differs')
            return arrays
        torch = self.torch
        generator = self.generator_class(self.cfg, torch.device('cuda:0'), seed)
        records, arrays = [], []
        produced = buffer_index = 0
        with torch.inference_mode(), self.capture_class(self.waveform) as capture:
            while produced < count:
                target = min(generator.generation_batch_size, count - produced)
                made = chunk_index = 0
                while made < target:
                    remaining = target - made
                    condition = generator.sample_condition(0, buffer_index, chunk_index,
                                   channel_weights=generator._chunk_weights_for_remaining(remaining))
                    from data.threegpp_profiles import is_3gpp_channel
                    preferred = (generator.threegpp_microbatch_size if is_3gpp_channel(condition['channel_type'])
                                 else generator.synthetic_microbatch_size)
                    size = min(max(1, int(preferred)), remaining)
                    offset = produced + made
                    capture.reset()
                    batch = generator._chunk(size, 0, buffer_index, chunk_index, cond=condition)
                    metadata = {'role': 'statistical_calibration', 'condition': condition,
                                'offset': offset, 'samples': size, 'calibration_seed': seed,
                                'snr_db': batch['snr_db'].detach().cpu().tolist()}
                    item = capture.arrays(batch, self.cfg, metadata)
                    relative = 'calibration_extra/batch_%04d.npz' % offset
                    atomic_npz(output / relative, item)
                    records.append({'path': relative, 'offset': offset, 'samples': size,
                                    'sha256': file_hash(output / relative)})
                    arrays.append(item)
                    del batch
                    capture.reset()
                    made += size; chunk_index += 1
                produced += target; buffer_index += 1
                print('Extra independent covariance calibration %d/%d' % (produced, count), flush=True)
        write_json(marker_path, {'schema_version': 2, 'role': 'statistical_calibration',
                                 'seed': seed, 'samples': count, 'export_signature': self.manifest['export_signature'],
                                 'distribution': 'same frozen training mixture; independent seed', 'batches': records})
        return arrays

    def prepare_calibration(self, output):
        """Original independent 128-frame ensemble; never use test truth for fitting."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / 'export_manifest.json'
        marker_path = output / 'calibration/COMPLETE.json'
        if manifest_path.exists():
            previous = read_json(manifest_path)
            if previous['export_signature'] != self.manifest['export_signature']:
                raise RuntimeError('Calibration cache belongs to another frozen runtime')
        write_json(manifest_path, self.manifest)
        if marker_path.exists():
            from classical_extension.run_classical_cpu import read_complete, load_batch
            marker, _ = read_complete(output, self.manifest, 'calibration')
            arrays = [load_batch(output, item) for item in marker['batches']]
            capture = output / 'latency_capture.npz'
            if not capture.is_file():
                raise RuntimeError('Calibration is present but latency capture is missing; use a fresh cache directory')
            sidecar = read_json(capture.with_suffix('.json'))
            if sidecar.get('capture_sha256') != file_hash(capture):
                raise RuntimeError('Saved receiver-time capture is corrupt')
            print('Verified independent calibration and receiver-time cache reused', flush=True)
            return arrays
        torch = self.torch
        generator = self.generator_class(self.cfg, torch.device('cuda:0'), self.calibration_seed)
        batches, arrays_all = [], []
        produced = buffer_index = 0
        with torch.inference_mode(), self.capture_class(self.waveform) as capture, CaptureReceivedTime(self.waveform) as time_capture:
            while produced < 128:
                target = min(generator.generation_batch_size, 128 - produced)
                made = chunk_index = 0
                while made < target:
                    remaining = target - made
                    cond = generator.sample_condition(0, buffer_index, chunk_index,
                              channel_weights=generator._chunk_weights_for_remaining(remaining))
                    from data.threegpp_profiles import is_3gpp_channel
                    preferred = (generator.threegpp_microbatch_size if is_3gpp_channel(cond['channel_type'])
                                 else generator.synthetic_microbatch_size)
                    size = min(max(1, int(preferred)), remaining)
                    offset = produced + made
                    capture.reset(); time_capture.reset()
                    batch = generator._chunk(size, 0, buffer_index, chunk_index, cond=cond)
                    meta = {'role': 'calibration', 'condition': cond, 'offset': offset,
                            'samples': size, 'calibration_seed': self.calibration_seed,
                            'snr_db': batch['snr_db'].detach().cpu().tolist()}
                    arrays = capture.arrays(batch, self.cfg, meta)
                    relative = 'calibration/batch_%04d.npz' % offset
                    atomic_npz(output / relative, arrays)
                    batches.append({'path': relative, 'offset': offset, 'samples': size,
                                    'sha256': file_hash(output / relative)})
                    arrays_all.append(arrays)
                    if offset == 0:
                        timing_arrays = dict(arrays)
                        timing_meta = {**meta, 'reference_experiment_signature': self.reference.experiment['signature']}
                        timing_arrays['metadata'] = np.asarray(json.dumps(timing_meta, sort_keys=True))
                        timing_arrays.update(time_capture.arrays(size))
                        path = output / 'latency_capture.npz'
                        atomic_npz(path, timing_arrays)
                        write_json(path.with_suffix('.json'), {
                            'capture_sha256': file_hash(path),
                            'reference_experiment_signature': self.reference.experiment['signature'],
                            'calibration_seed': self.calibration_seed,
                            'role': 'independent calibration receiver samples',
                            'input_boundary': 'real simulated noisy time samples before original OFDM demodulator',
                            'source_calibration_batch_sha256': batches[-1]['sha256']})
                    del batch
                    capture.reset(); time_capture.reset()
                    made += size; chunk_index += 1
                produced += target; buffer_index += 1
                print('Independent calibration replay %d/128; no neural model training' % produced, flush=True)
        write_json(marker_path, {'schema_version': 1, 'export_signature': self.manifest['export_signature'],
                                 'role': 'calibration', 'samples': 128, 'seed': self.calibration_seed,
                                 'distribution': 'frozen training configuration, epoch 0', 'batches': batches})
        return arrays_all

    def test_batches(self, cell):
        """Yield public observations; commit scores only after the generator ends."""
        torch = self.torch
        condition = dict(channel_type=cell['channel'], modulation=cell['modulation'],
                         delay_spread_ns=cell['delay_spread_ns'], doppler_hz=cell['doppler_hz'],
                         cp_len=cell['cp_len'], pilot_spacing=cell['pilot_spacing'])
        hasher = hashlib.sha256()
        with torch.inference_mode(), self.capture_class(self.waveform) as capture:
            for offset in range(0, 512, 8):
                seed = self.test_seed + int(digest({'cell': cell['cell_id'], 'offset': offset})[:15], 16)
                random_generator = torch.Generator(device=self.generator.device).manual_seed(seed)
                capture.reset()
                batch = self.generator.simulate(8, condition,
                    torch.full((8,), cell['snr_db'], device=self.generator.device), random_generator, seed + 1000000001)
                self.update_waveform_digest(hasher, batch)
                arrays = capture.arrays(batch, self.cfg, {'role': 'test', 'cell': cell,
                              'condition': condition, 'offset': offset, 'samples': 8})
                del batch
                capture.reset()
                yield offset, arrays
        observed = hasher.hexdigest()
        self.reference.assert_fingerprint(cell, observed)
        print('Frozen 512-frame waveform SHA-256 PASS: ' + cell['cell_id'], flush=True)
